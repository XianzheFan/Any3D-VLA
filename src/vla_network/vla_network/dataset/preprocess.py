from typing import Dict, List, Tuple, Callable, Optional, Any
import random
import numpy as np
from PIL import Image
import torch
from transformers import PreTrainedTokenizerBase
from transforms3d.euler import mat2euler, euler2mat
import open3d as o3d
import torch_scatter
from datetime import datetime
import os
import transforms3d
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from mapanything.models import MapAnything  # type: ignore
from mapanything.utils.image import preprocess_inputs  # type: ignore
from concerto.transform import Compose as ConcertoCompose

from model_utils.transform.pose import to_pose
from model_utils.robot import RobotModel, IKSolver, get_robot_cfg
from vla_network.type import BatchVLAData, RawVLAData
from vla_network.config import VLADataConfig, ImageTransform

from .tokenizer import RobotTokenizer
from .token_pattern import TokenResult, get_token_pattern


def get_k(values: list, k: int) -> list:
    if len(values) < k:
        values = list(values) + [values[-1]] * (k - len(values))
    else:
        values = values[:k]
    return values

def create_custom_concerto_transform():
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.01, hash_type="fnv",
             mode="train", return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord","grid_coord","color","inverse"),
             feat_keys=("coord","color")),
    ]
    return ConcertoCompose(config)

def make_SE3(position_xyz, quat, quat_format):
    q = np.array(quat, dtype=np.float64).copy()
    if quat_format == "xyzw":
        q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # [w,x,y,z]
    elif quat_format == "wxyz":
        pass
    else:
        raise ValueError("quat_format must be 'xyzw' or 'wxyz'")
    q = q / (np.linalg.norm(q) + 1e-12)
    R = transforms3d.quaternions.quat2mat(q)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return T

def build_robot_T_image(
    cam_extrinsics: Dict[str, Any], robot_base_extrinsics: Optional[Dict[str, Any]] = None,
    quat_format_cam: str = "wxyz", quat_format_robot: str = "xyzw",
    T_cam_correction: Optional[np.ndarray] = None):
    if T_cam_correction is None:
        T_cam_correction = np.array([[0,0,1,0], [-1,0,0,0], [0,-1,0,0], [0,0,0,1]], dtype=np.float64)
    T_cam = make_SE3(cam_extrinsics["position"], cam_extrinsics["orientation"], quat_format_cam)
    if robot_base_extrinsics is not None:
        T_world_robot = make_SE3(robot_base_extrinsics["position"], robot_base_extrinsics["orientation"], quat_format_robot)
        T_robot_world = np.linalg.inv(T_world_robot)
        T_robot_cam = T_robot_world @ T_cam
    else:
        T_robot_cam = T_cam
    T_robot_image = T_robot_cam @ T_cam_correction
    return T_robot_image, T_robot_cam

def camera_position_in_robot(cam_extrinsics: Dict[str, Any], robot_base_extrinsics: Optional[Dict[str, Any]] = None,
                             quat_format_cam: str = "wxyz", quat_format_robot: str = "xyzw"):
    _, T_robot_cam = build_robot_T_image(cam_extrinsics, robot_base_extrinsics,
                                         quat_format_cam=quat_format_cam, quat_format_robot=quat_format_robot)
    return T_robot_cam[:3, 3].copy()

def depth_rgb_to_point_cloud(depth_map: np.ndarray, rgb_image: np.ndarray, 
    intrinsics: Dict[str, Any], extrinsics: Dict[str, Any],
    cam_crop_cfg: Dict[str, Any], robot_base_extrinsics: Optional[Dict[str, Any]] = None, 
    quat_format_cam: str = "wxyz", quat_format_robot: str = "xyzw",
    T_cam_correction: np.ndarray = np.array([[0,0,1,0], [-1,0,0,0], [0,-1,0,0], [0,0,0,1]], dtype=np.float64),
    *, apply_extrinsics: bool = False, robot_crop_cfg: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]

    H, W = depth_map.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    v, u = np.mgrid[0:H, 0:W]
    z_cam = depth_map.astype(np.float32, copy=False)
    valid_mask_2d = np.isfinite(z_cam) & (z_cam > 0)

    if not np.any(valid_mask_2d):
        return np.zeros((0, 6), np.float32), np.zeros((0, 2), np.float32)
    u_valid = u[valid_mask_2d].astype(np.float32)
    v_valid = v[valid_mask_2d].astype(np.float32)
    z_valid = z_cam[valid_mask_2d]  # float32, shape (N,)
    colors_valid = rgb_image[valid_mask_2d]  # (N,3), uint8
    if not apply_extrinsics:
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        y_min, y_max = cam_crop_cfg.get("y_range", (-1.0, 1.0))
        r_xz_max = cam_crop_cfg.get("radius_xz", 1.8)
        z_range = cam_crop_cfg.get("z_range", (0.05, 1.3))
        keep = (y_cam > y_min) & (y_cam < y_max)
        keep &= (np.sqrt(x_cam**2 + z_valid**2) < r_xz_max)

        if z_range is not None:
            z_min, z_max = z_range
            keep &= (z_valid > z_min) & (z_valid < z_max)

        if not np.any(keep):
            return np.zeros((0, 6), np.float32), np.zeros((0, 2), np.float32)

        u_valid = u_valid[keep]
        v_valid = v_valid[keep]
        z_valid = z_valid[keep]
        x_cam = x_cam[keep]
        y_cam = y_cam[keep]
        colors = colors_valid[keep]
        pts = np.stack((x_cam, y_cam, z_valid), axis=-1)  # (N',3)
    else:
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        points_cam = np.stack((x_cam, y_cam, z_valid), axis=-1).astype(np.float64)
        colors = colors_valid
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
        points_h = np.hstack([points_cam, ones]).T  # (4,N)

        T_robot_img, _ = build_robot_T_image(
            extrinsics,
            robot_base_extrinsics,
            quat_format_cam=quat_format_cam,
            quat_format_robot=quat_format_robot,
            T_cam_correction=T_cam_correction,
        )
        pts_robot = (T_robot_img @ points_h)[:3].T
        pts = pts_robot.astype(np.float32)

        if robot_crop_cfg is None:
            robot_crop_cfg = {}
        Y_MIN, Y_MAX = robot_crop_cfg.get("y_range", (-1.0, 1.0))
        R_MAX = robot_crop_cfg.get("radius_xz", 1.8)

        mask_y = (pts[:, 1] > Y_MIN) & (pts[:, 1] < Y_MAX)
        mask_r = np.linalg.norm(pts[:, [0, 2]], axis=1) < R_MAX
        keep_robot = mask_y & mask_r
        if not np.any(keep_robot):
            return np.zeros((0, 6), np.float32), np.zeros((0, 2), np.float32)
        pts = pts[keep_robot]
        colors = colors[keep_robot]
        u_valid = u_valid[keep_robot]
        v_valid = v_valid[keep_robot]

    pc_rgb = np.hstack([pts.astype(np.float32), colors.astype(np.float32)])
    uv_coords_valid = np.stack([u_valid, v_valid], axis=-1).astype(np.float32)
    return pc_rgb, uv_coords_valid

def calculate_normals(points_xyz: np.ndarray, camera_position: np.ndarray = np.array([0,0,0])) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return np.zeros((0,3), np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=camera_position.astype(np.float64).reshape((3,1)))
    return np.asarray(pcd.normals).astype(np.float32)

def ensure_pc_ctx(pc_ctx: Optional[dict]) -> dict:
    if pc_ctx is None:
        pc_ctx = {}
    cam_info = pc_ctx.get("camera_info") or {}
    cam_info.setdefault("front_rand_small", {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]})
    rbe = pc_ctx.get("robot_base_extrinsics", None)
    rbe = rbe if rbe is not None else {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    return {"camera_info": cam_info, "robot_base_extrinsics": rbe}

def resize_with_bbox(  # rgb0 (256,256,3) depth0 (256,256,1)
    image: Image.Image,
    bbox: Optional[np.ndarray],
    target_size: Tuple[int, int],
    random_padding: bool = True,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Resize the image to target size. Pad if necessary.
    Also computes the bbox on the resized & padded image.
    """
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGB", target_size)
    if random_padding:
        paste_x = random.randint(0, target_size[0] - new_size[0])
        paste_y = random.randint(0, target_size[1] - new_size[1])
    else:
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))

    if bbox is not None:
        new_bbox = bbox * ratio
        new_bbox[0] += paste_x
        new_bbox[1] += paste_y
        new_bbox[2] += paste_x
        new_bbox[3] += paste_y
        new_bbox = np.array([int(t) for t in new_bbox])
    else:
        new_bbox = None

    return new_image, new_bbox

def to_open3d_pcd(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.size == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return pcd

def save_point_cloud_as_ply(filename: str, points: np.ndarray, colors: np.ndarray):
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    pcd = to_open3d_pcd(points, colors)
    ok = o3d.io.write_point_cloud(filename, pcd)
    if ok:
        print(f"Point cloud successfully saved to: {os.path.abspath(filename)}")
    else:
        cwd = os.getcwd()
        print(f"Failed to write: {filename}\n Current working directory: {cwd}\n Directory exists: {os.path.isdir(out_dir) if out_dir else True}")

def _disc_gripper_to_pm1(x: np.ndarray) -> np.ndarray:
    # x: (...,) float
    return np.where(x >= 0, 1, -1).astype(np.int8)

def bool_to_pm1(b: np.ndarray, dtype=np.int8) -> np.ndarray:
    # b: (...,) bool
    return np.where(b, 1, -1).astype(dtype)

class DataPreprocessor:
    config: VLADataConfig
    robot_tokenizer: RobotTokenizer
    tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform

    def __init__(self, config: VLADataConfig):
        self.config = config
        self.robot_cfg = get_robot_cfg(config.robot)
        self.robot_model = RobotModel.init("pin", self.robot_cfg.path_urdf)
        self.tokenizer = config.tokenizer
        config.tokenizer = None
        self.robot_tokenizer = RobotTokenizer.init(config, self.tokenizer.vocab_size)
        config.tokenizer = self.tokenizer
        self.image_transform = config.image_transform
        self.ik = IKSolver.init(self.robot_cfg, ik_type="cvx")
        self.use_depth = config.use_depth
        self.use_depth_pro = config.use_depth_pro
        self.use_unidepthv2 = config.use_unidepthv2
        self.use_da3 = getattr(config, 'use_da3', 0)
        self.use_calvin = 1
        self.use_map_anything = getattr(config, 'use_map_anything', 0)
        self.pc_transform = create_custom_concerto_transform() if (
            self.use_depth or self.use_depth_pro or self.use_unidepthv2 or self.use_da3 or self.use_map_anything
        ) else None
        self._depth_model = None
        self._unidepth_model = None
        self._da3_model = None
        self._map_anything_model = None
        self._saved_first_ply = False
        if config.pred == 'token_pred':
            self.pattern = get_token_pattern(config, 'graspvla')
            self.bbox_pattern = get_token_pattern(config, 'graspvla_bbox')
        elif config.pred == 'flow_matching':
            self.pattern = get_token_pattern(config, 'pi0')
            self.bbox_pattern = None
        elif config.pred == 'cot_flow_matching':
            self.pattern = get_token_pattern(config, 'pi0_cot_action')
            self.bbox_pattern = get_token_pattern(config, 'pi0_cot_grounding')
        elif config.pred == 'cot_bbox_flow_matching':
            self.pattern = get_token_pattern(config, 'pi0_bbox_cot')
            self.bbox_pattern = get_token_pattern(config, 'pi0_cot_grounding')
        elif config.pred == 'cotrain_flow_matching':
            self.pattern = get_token_pattern(config, 'pi0_goal_cot')
            self.bbox_pattern = get_token_pattern(config, 'pi0_cot_grounding')

    def setup(self, load: Callable[[], RawVLAData]):
        def get_transform_input():
            raw_data = load()
            return self.transform_input(raw_data, aug=True)

        self.robot_tokenizer.setup(get_transform_input)

    def save(self) -> dict:
        return self.robot_tokenizer.save()

    def load(self, data: dict):
        self.robot_tokenizer.load(data)

    def transform_img_bbox(self, raw_images: Dict[str, np.ndarray], raw_bboxs: Optional[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        pixel_values: List[Dict[str, torch.Tensor]] = []
        bboxs: List[np.ndarray] = []

        img_key = self.config.img_key
        assert all(len(raw_images[k]) == self.config.img_steps for k in img_key)
        for i in range(self.config.img_steps):
            for img_k in img_key:
                img, bbox = resize_with_bbox(
                    Image.fromarray(raw_images[img_k][i]),
                    raw_bboxs[img_k][i] if raw_bboxs is not None else None,
                    (self.config.image_size, self.config.image_size),
                    # random_padding=random_padding,
                )
                pixel_value = self.image_transform(img)
                if bbox is not None:
                    bbox = bbox / self.config.image_size * 2 - 1
                pixel_values.append(pixel_value)
                bboxs.append(bbox)
        pixel_values = torch.stack(pixel_values)[None]
        bboxs = np.stack(bboxs) if bboxs[0] is not None else None
        return pixel_values, bboxs
    
    def build_transformed_pc_from_singleview(
        self, raw_images: Dict[str, np.ndarray],
        raw_depths: Dict[str, Optional[np.ndarray]], pc_ctx: Optional[dict],
        vit_image_size: int, patch_size: int = 14,
    ) -> Optional[torch.Tensor]:
        use_extrinsics = False
        if not raw_depths:
            return None

        all_camera_info = {}
        robot_base_extrinsics = None
        if pc_ctx and use_extrinsics:
            all_camera_info = pc_ctx.get("camera_info", {}) or {}
            robot_base_extrinsics = pc_ctx.get("robot_base_extrinsics", None)

        all_pcs = []
        cam_pos_dict = {}

        depth_np = raw_depths.get("front", None)
        imgs = raw_images.get("front", None)
        img = np.array(imgs[0])
        
        # depth_np.shape (1, 256, 256, 1)
        depth_np = np.squeeze(depth_np, axis=0) if depth_np.shape[0] == 1 else depth_np
        depth_np = np.squeeze(depth_np, axis=-1)
        
        cam_info = all_camera_info.get(f"front_rand_small", None)
        H_orig, W_orig = img.shape[:2]
        # print(f"[build_transformed_pc_from_singleview] H_orig: {H_orig}, W_orig: {W_orig}", flush=True)
        intr = {
            'fx': 322.6666666666667 * (W_orig / 256.0), 'fy': 322.6666666666667 * (H_orig / 256.0),
            'cx': W_orig / 2.0, 'cy': H_orig / 2.0,
        }
        
        if self.use_unidepthv2:
            pc_with_rgb, uv_coords = depth_rgb_to_point_cloud(
                depth_map=depth_np, rgb_image=img, 
                intrinsics=intr,
                extrinsics=({'position': [0,0,0], 'orientation': [1,0,0,0]} if not use_extrinsics
                            else {'position': cam_info['position'], 'orientation': cam_info['orientation']}),
                cam_crop_cfg={"y_range": (-0.5, 0.8), "radius_xz": 1.2, "z_range": (0.05, 1.2)},
                robot_base_extrinsics=(None if not use_extrinsics else robot_base_extrinsics),
                quat_format_cam="wxyz", quat_format_robot="xyzw",
                apply_extrinsics=False
            )
        elif self.use_da3:
            pc_with_rgb, uv_coords = depth_rgb_to_point_cloud(
                depth_map=depth_np, rgb_image=img, intrinsics=intr,
                extrinsics=({'position': [0,0,0], 'orientation': [1,0,0,0]} if not use_extrinsics
                            else {'position': cam_info['position'], 'orientation': cam_info['orientation']}),
                cam_crop_cfg={"y_range": (-0.75, 0.75), "radius_xz": 1.7, "z_range": (0.2, 1.7)},
                robot_base_extrinsics=(None if not use_extrinsics else robot_base_extrinsics),
                quat_format_cam="wxyz", quat_format_robot="xyzw",
                apply_extrinsics=False
            )
        elif self.use_map_anything:
            pc_with_rgb, uv_coords = depth_rgb_to_point_cloud(
                depth_map=depth_np, rgb_image=img, intrinsics=intr,
                extrinsics=({'position': [0,0,0], 'orientation': [1,0,0,0]} if not use_extrinsics
                            else {'position': cam_info['position'], 'orientation': cam_info['orientation']}),
                cam_crop_cfg={"y_range": (-0.25, 0.3), "radius_xz": 1.3, "z_range": (0.5, 1.3)},
                robot_base_extrinsics=(None if not use_extrinsics else robot_base_extrinsics),
                quat_format_cam="wxyz", quat_format_robot="xyzw",
                apply_extrinsics=False
            )
        elif self.use_calvin:
            pc_with_rgb, uv_coords = depth_rgb_to_point_cloud(
                depth_map=depth_np, rgb_image=img,
                intrinsics={
                    'fx': 1463.046694753452 * (W_orig / 256.0), 'fy': 1463.046694753452 * (H_orig / 256.0),
                    'cx': W_orig / 2.0, 'cy': H_orig / 2.0,
                },
                extrinsics=({'position': [0,0,0], 'orientation': [1,0,0,0]} if not use_extrinsics
                            else {'position': cam_info['position'], 'orientation': cam_info['orientation']}),
                cam_crop_cfg={"y_range": (-2.0, 2.0), "radius_xz": 2.0, "z_range": (0.05, 2.0)},
                robot_base_extrinsics=(None if not use_extrinsics else robot_base_extrinsics),
                quat_format_cam="wxyz", quat_format_robot="xyzw",
                apply_extrinsics=False
            )
        else:
            pc_with_rgb, uv_coords = depth_rgb_to_point_cloud(
                depth_map=depth_np, rgb_image=img,
                intrinsics=intr,
                extrinsics=({'position': [0,0,0], 'orientation': [1,0,0,0]} if not use_extrinsics
                            else {'position': cam_info['position'], 'orientation': cam_info['orientation']}),
                cam_crop_cfg={"y_range": (-1.0, 1.0), "radius_xz": 1.4, "z_range": (0.05, 1.4)},
                robot_base_extrinsics=(None if not use_extrinsics else robot_base_extrinsics),
                quat_format_cam="wxyz", quat_format_robot="xyzw",
                apply_extrinsics=False
            )
        if pc_with_rgb.shape[0] == 0:
            return None

        H_orig, W_orig = img.shape[0], img.shape[1]
        grid_w, grid_h = vit_image_size // patch_size, vit_image_size // patch_size
        u, v = uv_coords[:, 0], uv_coords[:, 1]
        u_vit = u * (vit_image_size / float(W_orig))
        v_vit = v * (vit_image_size / float(H_orig))
        pu = np.clip((u_vit // patch_size).astype(np.int64), 0, grid_w - 1)
        pv = np.clip((v_vit // patch_size).astype(np.int64), 0, grid_h - 1)
        patch_idx_np = (pv * grid_w + pu).astype(np.int64).reshape(-1, 1)

        cam_idx = 1
        cam_col = np.full((pc_with_rgb.shape[0], 1), cam_idx, dtype=np.int16)

        final_pc_with_cam_patch = np.hstack([
            pc_with_rgb, cam_col.astype(np.float32), patch_idx_np.astype(np.float32)
        ])  # (N, 11)
        all_pcs.append(final_pc_with_cam_patch)

        cam_pos = (np.array([0,0,0], dtype=np.float64) if not use_extrinsics else
                    camera_position_in_robot(
                        cam_extrinsics={'position': cam_info['position'], 'orientation': cam_info['orientation']},
                        robot_base_extrinsics=robot_base_extrinsics, quat_format_cam="wxyz", quat_format_robot="xyzw"
                    ))
        cam_pos_dict["front"] = cam_pos

        if not all_pcs:
            return None
        per_coords, per_colors, per_grids, per_cam_down, per_patch_down = [], [], [], [], []
        
        cam_idx = 1
        pcs_view = [pc for pc in all_pcs if pc.shape[1] >= 8 and int(pc[0, 6]) == cam_idx]
        combined_view = np.concatenate(pcs_view, axis=0)
        pcs_np = combined_view[:, :6]
        cam_id_np = combined_view[:, 6].astype(np.int64)
        patch_np = combined_view[:, 7].astype(np.int64)
        point_data = {"coord": pcs_np[:, :3].astype(np.float32), "color": pcs_np[:, 3:6].astype(np.float32)}
        td = self.pc_transform(point_data)  # per-view grid sampling
        coord_v, color_v = td["coord"], td["color"]
        grid_coord_v, inverse_idx_v = td["grid_coord"], td["inverse"].long()
        N_v = coord_v.shape[0]
        cam_t = torch.from_numpy(cam_id_np).to(torch.long)
        patch_t = torch.from_numpy(patch_np).to(torch.long)
        cam_down_v = torch_scatter.scatter_mean(cam_t.to(torch.float32), inverse_idx_v, dim=0, dim_size=N_v).round().clamp(min=0).unsqueeze(-1)
        patch_down_v = torch_scatter.scatter_mean(patch_t.to(torch.float32), inverse_idx_v, dim=0, dim_size=N_v).round().clamp(min=0).unsqueeze(-1)
        per_coords.append(coord_v); per_colors.append(color_v); per_grids.append(grid_coord_v)
        per_cam_down.append(cam_down_v); per_patch_down.append(patch_down_v)
        if not per_coords:
            return None
        coord = torch.cat(per_coords, dim=0)
        color = torch.cat(per_colors, dim=0)
        grid_coord = torch.cat(per_grids, dim=0)
        cam_down = torch.cat(per_cam_down, dim=0)
        patch_down = torch.cat(per_patch_down, dim=0)
        
        if coord.shape[0] == 0:
            return None

        coord_np = coord.detach().cpu().numpy()
        normals_np = np.zeros_like(coord_np, dtype=np.float32)

        mask_front = (cam_down.squeeze(-1) == 1).cpu().numpy()
        cam_pos_front = cam_pos_dict.get("front", np.array([0,0,0], dtype=np.float64))
        if mask_front.any():
            normals_np[mask_front] = calculate_normals(coord_np[mask_front], camera_position=cam_pos_front)
        normal = torch.from_numpy(normals_np).to(coord.dtype).to(coord.device)

        transformed_pc = torch.cat([coord, color, normal, grid_coord, cam_down, patch_down], dim=-1)
        return transformed_pc

    def save_transformed_pc_as_ply(self, transformed_pc: torch.Tensor,
        base_name: Optional[str] = None, out_dir: str = "plys", save_once: bool = True):
        try:
            save_every = os.environ.get("SAVE_PLY_EVERY", "0") == "1"
            if save_once and (self._saved_first_ply and not save_every):
                return

            if isinstance(transformed_pc, torch.Tensor):
                pc_np = transformed_pc.detach().cpu().numpy()
            else:
                pc_np = np.asarray(transformed_pc)

            pts = pc_np[:, 0:3].astype(np.float32)  # xyz
            cols = pc_np[:, 3:6].astype(np.float32)  # rgb

            cmax = float(cols.max()) if cols.size else 0.0
            if cmax <= 1.5:
                cols = np.clip(cols * 255.0, 0, 255)
            cols_uint8 = cols.astype(np.uint8)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if base_name is None:
                base = f"pc_after_transform_{ts}"
            else:
                base = base_name

            out_dir = os.path.abspath(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}.ply")
            save_point_cloud_as_ply(out_path, pts, cols_uint8)
            print(f"[save_transformed_pc_as_ply] Saved {pts.shape[0]} pts -> {out_path}", flush=True)

            self._saved_first_ply = True
        except Exception as e:
            print(f"[save_transformed_pc_as_ply][WARN] {e}", flush=True)

    # Predict depth for a single RGB frame (np.uint8,H,W,3), and return a float32 depth map (H,W,1)
    def _predict_depth_one(self, rgb_np: np.ndarray) -> np.ndarray:
        model, transform = self._ensure_depth_model()
        H, W = rgb_np.shape[:2]
        # print(f"[_predict_depth_one] H: {H}, W: {W}", flush=True)
        fx = 322.6666666666667 * (W / 256.0)
        pil_img = Image.fromarray(rgb_np)
        inp = transform(pil_img)
        with torch.inference_mode():
            pred = model.infer(inp, f_px=torch.as_tensor([fx], dtype=inp.dtype))
        depth_m = pred["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        depth_m = np.array(Image.fromarray(depth_m).resize((W, H), Image.BILINEAR), dtype=np.float32)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_m[..., None]
    
    def _build_predicted_depths_tensor_from_images(self, raw_images: Dict[str, np.ndarray], t_idx: int) -> torch.Tensor:
        max_idx = len(raw_images["front"]) - 1
        t_idx = max(0, min(t_idx, max_idx))
        depth_front = self._predict_depth_one(raw_images["front"][t_idx])  # (H,W,1)
        temp_tensor = torch.from_numpy(depth_front[..., 0]).float().unsqueeze(0)
        return temp_tensor.unsqueeze(0)
    
    def _ensure_unidepthv2_model(self):
        if self._unidepth_model is None:
            from unidepth.models import UniDepthV2 # type: ignore
            model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval()
            model.resolution_level = 2
            self._unidepth_model = model
        return self._unidepth_model
    
    def _predict_unidepth_one(self, rgb_np: np.ndarray) -> np.ndarray:
        model = self._ensure_unidepthv2_model()
        H, W = rgb_np.shape[:2]
        # print(f"[_predict_unidepth_one] H: {H}, W: {W}", flush=True)
        fx = 322.6666666666667 * (W / 256.0)
        fy = 322.6666666666667 * (H / 256.0)
        cx, cy = W / 2.0, H / 2.0

        rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous()
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        with torch.inference_mode():
            preds = model.infer(rgb_t, K)
        depth_m = preds["depth"].squeeze()
        if depth_m.dim() != 2:
            depth_m = depth_m.reshape(depth_m.shape[-2], depth_m.shape[-1])
        depth_m = depth_m.detach().float().cpu().numpy()
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        return depth_m[..., None]

    def _build_predicted_depths_tensor_from_images_unidepth(self, raw_images: Dict[str, np.ndarray], t_idx: int) -> torch.Tensor:
        max_idx = len(raw_images["front"]) - 1
        t_idx = max(0, min(t_idx, max_idx))
        depth_front = self._predict_unidepth_one(raw_images["front"][t_idx])  # (H,W,1)
        temp_tensor = torch.from_numpy(depth_front[..., 0]).float().unsqueeze(0)
        return temp_tensor.unsqueeze(0)

    def _ensure_da3_model(self):
        if self._da3_model is None:
            print("Loading Depth Anything 3 model...")
            model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").eval()
            if torch.cuda.is_available():
                model = model.cuda()
            self._da3_model = model
        return self._da3_model

    def _predict_da3_one(self, rgb_np: np.ndarray) -> np.ndarray:
        model = self._ensure_da3_model()
        H, W = rgb_np.shape[:2]
        pil_img = Image.fromarray(rgb_np)
        with torch.no_grad():
            prediction = model.inference([pil_img])
        pred_depth = prediction.depth[0]
        # Resize depth map
        if pred_depth.shape != (H, W):
            device = next(model.parameters()).device
            depth_t = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0).to(device) # (1,1,H_pred,W_pred)
            depth_t = F.interpolate(depth_t, size=(H, W), mode='bilinear', align_corners=False)
            depth_m = depth_t.squeeze().cpu().numpy()
        else:
            depth_m = pred_depth
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = depth_m * 3.0
        return depth_m[..., None]

    def _build_predicted_depths_tensor_from_images_da3(self, raw_images: Dict[str, np.ndarray], t_idx: int) -> torch.Tensor:
        max_idx = len(raw_images["front"]) - 1
        t_idx = max(0, min(t_idx, max_idx))
        depth_front = self._predict_da3_one(raw_images["front"][t_idx])  # (H,W,1)
        temp_tensor = torch.from_numpy(depth_front[..., 0]).float().unsqueeze(0)  # (1,2,H,W)
        return temp_tensor.unsqueeze(0)  # (1,1,2,H,W)
    
    def _ensure_map_anything_model(self):
        if self._map_anything_model is None:
            self._map_anything_model = MapAnything.from_pretrained("facebook/map-anything").to("cuda").eval()
        return self._map_anything_model
    
    def _build_predicted_depths_tensor_from_images_map_anything(self, raw_images: Dict[str, np.ndarray], t_idx: int) -> torch.Tensor:
        max_idx = len(raw_images["front"]) - 1
        t_idx = max(0, min(t_idx, max_idx))
        img_front = raw_images["front"][t_idx]
        views = [{"img": img_front}]
        processed_views = preprocess_inputs(views)
        model = self._ensure_map_anything_model()
        
        with torch.inference_mode():
            predictions = model.infer(
                processed_views, 
                memory_efficient_inference=False,
                use_amp=True, 
                amp_dtype="bf16",
                apply_mask=True, 
                mask_edges=True, 
                apply_confidence_mask=False,
            )
        processed_depths = []
        target_size = (self.config.image_size, self.config.image_size) # usually (256, 256)
        
        for pred in predictions:
            depth_tensor = pred["depth_z"].permute(2, 0, 1).unsqueeze(0) # (1, 1, H, W)
            depth_resized = F.interpolate(depth_tensor, size=target_size, mode='bilinear', align_corners=False)
            depth_resized = depth_resized / 3.0
            depth_m = depth_resized.squeeze().float().cpu().numpy()
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            processed_depths.append(depth_m)
        depth_front = processed_depths[0]
        temp_tensor = torch.from_numpy(depth_front).float().unsqueeze(0)  # (1,2,H,W)
        return temp_tensor.unsqueeze(0)  # (1,1,2,H,W)

    def transform_input(
        self, raw_input: RawVLAData, aug: bool = False
    ) -> Dict[str, np.ndarray]:

        proprio, action, for_rel_proprio, goal = (
            raw_input.proprio,
            raw_input.action,
            raw_input.for_rel_proprio,
            raw_input.goal,
        )

        if self.config.robot_rep == "joint_angle":
            ret_proprio = proprio[:, :8].copy()
            ret_proprio[:, -1] = ret_proprio[:, -1] > 0.85 # TODO:
            if action is not None:
                ret_action = action[:, :8].copy()
                ret_action[:, -1] = ret_action[:, -1] > 0.85 # TODO:
                ret_action[:, :-1] -= for_rel_proprio[:, :7]
            else:
                ret_action = None
        elif self.config.robot_rep in ["xyz_rpy", "xyz_rpy_rot"]:
            proprio_trans, proprio_rot = self.robot_model.batch_fk_link(
                proprio, self.robot_cfg.link_eef
            )
            if action is not None:
                action_trans, action_rot = self.robot_model.batch_fk_link(
                    action, self.robot_cfg.link_eef
                )
                for_rel_trans, for_rel_rot = self.robot_model.batch_fk_link(
                    for_rel_proprio, self.robot_cfg.link_eef
                )


            if aug:
                proprio_trans += np.random.uniform(
                    -self.config.trans_noise,
                    self.config.trans_noise,
                    size=(len(proprio_trans), 3),
                )
                proprio_rpy_noise = np.random.uniform(
                    -self.config.rot_noise,
                    self.config.rot_noise,
                    size=(len(proprio_rot), 3),
                )
                proprio_noise = np.array([euler2mat(*xx) for xx in proprio_rpy_noise])
                proprio_rot = np.einsum("nab,nbc->nac", proprio_rot, proprio_noise)

            if raw_input.proprio_grippers is None:
                g = raw_input.proprio[:, -1]
                proprio_gripper = np.where(g > 0.5, 1, -1).astype(np.int8)[:, None]
            else:
                proprio_gripper = raw_input.proprio_grippers[:, None]
            if self.config.robot_rep == 'xyz_rpy':
                proprio_rot = np.array([mat2euler(xx) for xx in proprio_rot])
            elif self.config.robot_rep == 'xyz_rpy_rot':
                proprio_rot = proprio_rot.reshape(-1, 9)

            ret_proprio = np.concatenate(
                [proprio_trans, proprio_rot, proprio_gripper], axis=-1
            )
            if action is not None:
                delta_trans = action_trans - for_rel_trans
                action_gripper = raw_input.action_grippers[:, None]
                delta_rot = np.einsum("nab,ncb->nac", action_rot, for_rel_rot)
                delta_rpy = np.array([mat2euler(xx) for xx in delta_rot])
                ret_action = np.concatenate(
                    [delta_trans, delta_rpy, action_gripper], axis=-1
                )
            else:
                ret_action = None
        elif self.config.robot_rep == "identity":
            ret_proprio = proprio
            ret_action = action

        ret_goal = None
        if goal is not None:
            if self.config.goal_rep == "joint_angle":
                ret_goal = goal[:7]
            elif self.config.goal_rep in ["xyz_rpy", "xyz_rot"]:
                goal_trans, goal_rot = self.robot_model.fk_link(
                    goal, self.robot_cfg.link_eef
                )
                if self.config.goal_rep == 'xyz_rpy':
                    goal_rot = mat2euler(goal_rot)
                elif self.config.goal_rep == 'xyz_rot':
                    goal_rot = goal_rot.flatten()
                ret_goal = np.concatenate([goal_trans, goal_rot])
            elif self.config.goal_rep == "identity":
                ret_goal = goal
        return dict(proprio=ret_proprio, action=ret_action, goal=ret_goal)

    def transform_output(
        self,
        raw_data: RawVLAData,
        ret_goal: Optional[np.ndarray] = None,
        ret_action: Optional[np.ndarray] = None,
        idx: int = 0,
    ) -> dict:
        if self.config.goal_rep in ["xyz_rpy", "xyz_rot"]:
            goal_trans, goal_rot = ret_goal[:3], ret_goal[3:]
            if self.config.goal_rep == "xyz_rpy":
                goal_rot = euler2mat(*goal_rot)
            elif self.config.goal_rep == "xyz_rot":
                goal_rot = goal_rot.reshape(3, 3)
            goal = to_pose(goal_trans, goal_rot)
        elif self.config.goal_rep == "joint_angle":
            goal = ret_goal

        if self.config.robot_rep in ["xyz_rpy", "xyz_rpy_rot"]:
            for_rel_proprio = raw_data.for_rel_proprio[idx]
            for_rel_trans, for_rel_rot = self.robot_model.fk_link(
                for_rel_proprio, self.robot_cfg.link_eef
            )

            cur_action = ret_action.reshape(-1, 7)[idx]
            delta_trans, delta_rpy, disc_gripper = np.split(cur_action, [3, 6])
            delta_rot = euler2mat(*delta_rpy)
            action_gripper = disc_gripper # TODO:
            action_trans = for_rel_trans + delta_trans
            action_rot = np.einsum("ab,bc->ac", delta_rot, for_rel_rot)

            succ, action_qpos = self.ik.ik(
                action_trans, action_rot, raw_data.proprio[-1]
            )
            if not succ:
                action_qpos = raw_data.proprio[-1]
            action_qpos[7:] = action_gripper # TODO:
        elif self.config.robot_rep == "joint_angle":
            action_qpos = ret_action.reshape(-1, 8)[idx].copy()
            action_qpos[:-1] += raw_data.for_rel_proprio[idx][:7]
            action_qpos[-1] = (1 - action_qpos[-1]) * 0.8  # TODO:

        return dict(
            goal=goal,
            action=action_qpos,
        )
    


    def transform(self, raw_data: RawVLAData, inference: bool = False) -> BatchVLAData:

        pixel_values, bboxs = self.transform_img_bbox(raw_data.images, raw_data.bboxs)

        trans_dic = self.transform_input(raw_data, aug=(not inference))
        assert len(trans_dic["proprio"]) == self.config.proprio_len

        text_ids = self.tokenizer(raw_data.instruction, add_special_tokens=True).input_ids
        depths_tensor = None
        transformed_pc = None
        
        if self.use_depth and raw_data.depths and any(v is not None for v in raw_data.depths.values()):
            view_depth = raw_data.depths["front"]
            if isinstance(view_depth, list):
                view_depth = np.array(view_depth)

            temp_tensor = torch.from_numpy(view_depth)
            if temp_tensor.dim() == 5:
                processed_tensor = temp_tensor.squeeze(0).permute(0, 3, 1, 2).float()
            elif temp_tensor.dim() == 4:
                processed_tensor = temp_tensor.permute(0, 3, 1, 2).float()
            else:
                raise ValueError(f"error")

            depths_tensor = processed_tensor.unsqueeze(0)

            pc_ctx_raw = getattr(raw_data, "pc_ctx", None)
            pc_ctx = ensure_pc_ctx(pc_ctx_raw)
            try:
                transformed_pc = self.build_transformed_pc_from_singleview(
                    raw_images=raw_data.images,
                    raw_depths={k: (v[0] if isinstance(v, list) else v) for k,v in raw_data.depths.items()},
                    pc_ctx=pc_ctx,
                    vit_image_size=self.config.image_size,
                )
                
                # base_name = f"{getattr(raw_data, 'dataset_name', 'ds')}_{getattr(raw_data, 'data_id', 'id')}_f{getattr(raw_data, 'frame', 0)}"
                # self.save_transformed_pc_as_ply(
                #     transformed_pc=transformed_pc,
                #     base_name=base_name,
                #     out_dir="plys",
                #     save_once=True
                # )
            except Exception:
                transformed_pc = torch.zeros((0, 14), dtype=torch.float32)
        elif not self.use_depth and self.use_unidepthv2 == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_unidepth(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)
            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}  # (1,H,W,1)
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )
            base_name = f"{getattr(raw_data, 'dataset_name', 'ds')}_{getattr(raw_data, 'data_id', 'id')}_f{getattr(raw_data, 'frame', 0)}"
            self.save_transformed_pc_as_ply(
                transformed_pc=transformed_pc,
                base_name=base_name,
                out_dir="unidepthv2_plys",
                save_once=True
            )
        elif not self.use_depth and self.use_da3 == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_da3(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)

            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )
            base_name = f"{getattr(raw_data, 'dataset_name', 'ds')}_{getattr(raw_data, 'data_id', 'id')}_f{getattr(raw_data, 'frame', 0)}"
            self.save_transformed_pc_as_ply(
                transformed_pc=transformed_pc,
                base_name=base_name,
                out_dir="da3_plys",
                save_once=True
            )
        elif not self.use_depth and self.use_map_anything == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_map_anything(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)

            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )
            base_name = f"{getattr(raw_data, 'dataset_name', 'ds')}_{getattr(raw_data, 'data_id', 'id')}_f{getattr(raw_data, 'frame', 0)}"
            self.save_transformed_pc_as_ply(
                 transformed_pc=transformed_pc,
                 base_name=base_name,
                 out_dir="mapanything_plys",
                 save_once=True
            )

        debug_dict = None
        if not inference:
            assert len(trans_dic["action"]) == self.config.action_len

            input_ids, labels = self.pattern.get_input_id_label(
                text_ids=text_ids,
                bbox=self.robot_tokenizer.bbox(bboxs) if bboxs is not None else None,
                goal=self.robot_tokenizer.goal(trans_dic['goal']) if 'goal' in trans_dic else None,
                hist_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][:-1]) if 'proprio' in trans_dic else None,
                cur_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][-1]) if 'proprio' in trans_dic else None,
                eos=[self.tokenizer.eos_token_id],
            )

            robot_input_ids, robot_labels = self.pattern.get_robot_input_id_label(
                hist_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][:-1]),
                cur_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][-1]),
                goal=self.robot_tokenizer.goal(trans_dic['goal']) if trans_dic['goal'] is not None else None,
                action=self.robot_tokenizer.action(trans_dic['action']),
                eos=[self.tokenizer.eos_token_id],
            )
            inference_kwargs = None
        else:
            inference_kwargs = [dict(
                text_ids=text_ids,
                hist_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][:-1]),
                cur_proprio=self.robot_tokenizer.proprio(trans_dic['proprio'][-1]),
            )]
            token_result = self.pattern.update_tokens(
                output=[], 
                **inference_kwargs[0]
            )
            input_ids = token_result.input_ids
            robot_input_ids = token_result.robot_input_ids
            if 'action' in trans_dic:
                debug_dict = dict(
                    action=trans_dic['action'],
                    goal=trans_dic['goal'] if trans_dic.get('goal') is not None else None
                )

        return BatchVLAData(
            debug=[debug_dict],
            input_ids=torch.tensor(input_ids)[None],
            labels=torch.tensor(labels)[None] if not inference else None,
            attention_mask=torch.ones(len(input_ids))[None].bool(),
            robot_input_ids=torch.tensor(robot_input_ids)[None],
            robot_attention_mask=torch.ones(len(robot_input_ids))[None].bool(),
            robot_labels=torch.tensor(robot_labels)[None] if not inference else None,
            images=pixel_values,
            action=torch.from_numpy(self.robot_tokenizer.norm_action(trans_dic['action'])).float()[None] if trans_dic['action'] is not None else None,
            proprio=torch.from_numpy(self.robot_tokenizer.norm_proprio(trans_dic['proprio'])).float()[None],
            goal=torch.from_numpy(self.robot_tokenizer.norm_goal(trans_dic['goal'])).float()[None] if trans_dic['goal'] is not None else None,
            is_action=torch.ones(1).bool(),
            inference_kwargs=inference_kwargs,
            depths=depths_tensor,
            transformed_pc=transformed_pc,
        )

    def parse_output(
        self, raw_data: RawVLAData, token_result: TokenResult, idx: int = 0
    ) -> dict:
        action_tokens = np.array(token_result.action).reshape(-1, 7)
        action = self.robot_tokenizer.inv_action(action_tokens)
        if hasattr(token_result, 'goal'):
            goal_tokens = np.array(token_result.goal)
            goal = self.robot_tokenizer.inv_goal(goal_tokens)
        else:
            goal = None

        # TODO:
        return dict(
            action=action,
            goal=goal,
        )

    def transform_bbox(self, raw_data: RawVLAData, inference: bool = False) -> BatchVLAData:
        pixel_values, bboxs = self.transform_img_bbox(raw_data.images, raw_data.bboxs)
        text_ids = self.tokenizer(raw_data.instruction, add_special_tokens=True).input_ids

        depths_tensor = None
        transformed_pc = None
        
        if self.use_depth and raw_data.depths and any(v is not None for v in raw_data.depths.values()):
            view_depth = raw_data.depths["front"]
            if isinstance(view_depth, list):
                view_depth = np.array(view_depth)

            temp_tensor = torch.from_numpy(view_depth)
            if temp_tensor.dim() == 5:
                processed_tensor = temp_tensor.squeeze(0).permute(0, 3, 1, 2).float()
            elif temp_tensor.dim() == 4:
                processed_tensor = temp_tensor.permute(0, 3, 1, 2).float()
            else:
                raise ValueError(f"error")

            depths_tensor = processed_tensor.unsqueeze(0)
        elif not self.use_depth and self.use_unidepthv2 == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_unidepth(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)
            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}  # (1,H,W,1)
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )
        elif not self.use_depth and self.use_da3 == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_da3(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)

            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )
        elif not self.use_depth and self.use_map_anything == 1:
            depths_tensor = self._build_predicted_depths_tensor_from_images_map_anything(raw_data.images, raw_data.frame)
            transformed_pc = None
            depth_front = depths_tensor[0, 0].detach().cpu().numpy()[..., None]  # (H,W,1)

            raw_depths_pred_for_pc = {"front": depth_front[None, ...]}
            transformed_pc = self.build_transformed_pc_from_singleview(
                raw_images=raw_data.images,
                raw_depths=raw_depths_pred_for_pc,
                pc_ctx=None,
                vit_image_size=self.config.image_size,
            )

        input_ids, labels = self.bbox_pattern.get_input_id_label(
            text_ids=text_ids,
            bbox=self.robot_tokenizer.bbox(bboxs),
        )
        robot_input_ids, robot_labels = self.bbox_pattern.get_robot_input_id_label(
            eos=[self.tokenizer.eos_token_id],
        )

        return BatchVLAData(
            debug=[],
            input_ids=torch.tensor(input_ids)[None],
            labels=torch.tensor(labels)[None] if not inference else None,
            attention_mask=torch.ones(len(input_ids))[None].bool(),
            robot_input_ids=torch.tensor(robot_input_ids)[None],
            robot_attention_mask=torch.ones(len(robot_input_ids))[None].bool(),
            robot_labels=torch.tensor(robot_labels)[None] if not inference else None,
            images=pixel_values,
            action=torch.zeros((1, self.config.action_len, self.config.action_dim)).float() if not inference else None,
            proprio=torch.zeros((1, self.config.proprio_len, self.config.proprio_dim)).float(),
            goal=torch.zeros((1, self.config.goal_dim)).float(),
            is_action=torch.zeros(1).bool(),
            depths=depths_tensor,
            transformed_pc=transformed_pc,
        )
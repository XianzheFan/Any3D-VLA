import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from dataclasses import dataclass
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
import torch_scatter
import spconv.pytorch as spconv

from concerto.structure import Point
import concerto

from . import Backbone2D, Backbone2DConfig
from vla_network.config import ImageTransform

try:
    import flash_attn
except ImportError:
    flash_attn = None

class PCEncoder32(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod.float()

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

    def _apply(self, fn):
        super()._apply(fn)
        for p in self.parameters(recurse=True):
            with torch.no_grad():
                p.data = p.data.to(dtype=torch.float32)
        for b in self.buffers(recurse=True):
            if isinstance(b, torch.Tensor):
                with torch.no_grad():
                    b.data = b.data.to(dtype=torch.float32)
        return self

@dataclass
class CombineImageTransform:
    transforms: List[ImageTransform]
    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        return torch.stack([t(img, **kwargs) for t in self.transforms], dim=0)

class ViT(nn.Module):
    model: VisionTransformer
    def __init__(self, model: VisionTransformer) -> None:
        super().__init__()
        self.model = model
        self.n = len(self.model.blocks) - 2
    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_intermediate_layers(x, n={self.n})[0]

class ConcertoViTMonoBackbone(Backbone2D):
    config: Backbone2DConfig
    pc_encoder: nn.Module
    pc_feature_dim: int
    empty_pc_token: nn.Parameter

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__(config)

        self.device = torch.device("cuda")
        self.compute_dtype = torch.float32

        print("Initializing Point Cloud Encoder (Concerto, image encoders removed)...")
        raw_pc_encoder = concerto.load("concerto_large", repo_id="Pointcept/Concerto").float()
        self.pc_encoder = PCEncoder32(raw_pc_encoder)
        self.pc_encoder.requires_grad_(False)
        self.pc_encoder.to(self.device)

        self._spconv_layers = []
        for m in self.pc_encoder.modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d)):
                self._spconv_layers.append(m)
        self.set_partial_freeze(unfreeze_last_spconvs=4)

        self.pc_feature_dim = 1728
        self.empty_pc_token = nn.Parameter(
            torch.randn(1, self.pc_feature_dim, device=self.device, dtype=self.compute_dtype)
        )
        
        base_transform = Compose([
            Resize((config.image_size, config.image_size)),
            ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image_transform = CombineImageTransform([base_transform, base_transform])

    def set_partial_freeze(self, unfreeze_last_spconvs: int = 0):
        self.pc_encoder.requires_grad_(False)
        if unfreeze_last_spconvs > 0:
            for m in self._spconv_layers[-unfreeze_last_spconvs:]:
                m.requires_grad_(True)

    @property
    def feature_dim(self) -> int:
        return self.pc_feature_dim

    def forward(self, images: torch.Tensor, depths: Optional[torch.Tensor] = None,
                transformed_pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_img, n_cams = images.shape[:2]

        if transformed_pc is None or transformed_pc.numel() == 0:
            print("transformed_pc is None", flush=True)
            B = B_img
            L = 1
            feats_out = self.empty_pc_token.view(1, 1, 1, -1).expand(B, 1, L, -1)
            feats_out = feats_out.reshape(B, L, self.pc_feature_dim)
            return feats_out

        # xyz(3) rgb(3) normal(3) grid(3) cam_id(1) patch_idx(1) batch(1)
        with torch.autocast(device_type="cuda", enabled=False):
            point_coord = transformed_pc[:, 0:3].to(dtype=torch.float32)
            point_color = transformed_pc[:, 3:6].to(dtype=torch.float32)
            point_normal = transformed_pc[:, 6:9].to(dtype=torch.float32)
            point_grid = transformed_pc[:, 9:12].long()
            cam_id = transformed_pc[:, 12].long()
            patch_idx = transformed_pc[:, 13].long()
            batch_idx_raw = transformed_pc[:, 14].long()
            
            view_mask = cam_id.eq(1)  # front view
            point_coord = point_coord[view_mask]
            point_color = point_color[view_mask]
            point_normal = point_normal[view_mask]
            point_grid = point_grid[view_mask]
            patch_idx = patch_idx[view_mask]
            batch_idx_raw = batch_idx_raw[view_mask]
            
            B_pc = int(batch_idx_raw.max().item() + 1)
            B = max(B_img, B_pc)

            valid_patch_mask = patch_idx.ge(0)
            if valid_patch_mask.any():
                L = int(patch_idx[valid_patch_mask].max().item() + 1)
            else:
                L = 1

            batch_idx = (batch_idx_raw + cam_id).long()

            point_input = Point({
                "coord": point_coord, "color": point_color, "normal": point_normal,
                "feat": torch.cat([point_coord, point_color, point_normal], dim=1),
                "grid_coord": point_grid, "batch": batch_idx,
            })
            if "grid_size" not in point_input:
                point_input["grid_size"] = 0.01

            point_out = self.pc_encoder(point_input)
            point_processed = point_out
            while "pooling_parent" in point_processed.keys():
                parent = point_processed.pop("pooling_parent")
                inverse = point_processed.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_processed.feat[inverse]], dim=-1)
                point_processed = parent
            final_point_feats = point_processed.feat  # [N_points, D_pc] float32

            valid_mask = patch_idx.ge(0)
            valid_feats = final_point_feats[valid_mask]  # float32
            valid_batch_idx = batch_idx[valid_mask]
            valid_patch_idx = patch_idx[valid_mask]
            global_idx = (valid_batch_idx * L + valid_patch_idx).long()
            total_dim_size = B * L

            sum_flat = torch_scatter.scatter(
                src=valid_feats, index=global_idx, dim=0, dim_size=total_dim_size, reduce="sum"
            )  # [B*L, D_pc]
            counts_flat = torch_scatter.scatter(
                src=torch.ones_like(global_idx, dtype=torch.float32),
                index=global_idx, dim=0,
                dim_size=total_dim_size, reduce="sum",
            )  # [B*L]
            mean_flat = sum_flat / counts_flat.clamp_min(1.0).unsqueeze(-1)

            pc_feats_per_view = mean_flat.view(B, 1, L, self.pc_feature_dim)  # float32
            counts_per_view = counts_flat.view(B, 1, L)

            pc_feats_per_view = torch.where(
                counts_per_view.eq(0).unsqueeze(-1),
                self.empty_pc_token.view(1, 1, 1, -1),
                pc_feats_per_view
            )

        feats_out = pc_feats_per_view.reshape(B, L, self.pc_feature_dim)
        return feats_out
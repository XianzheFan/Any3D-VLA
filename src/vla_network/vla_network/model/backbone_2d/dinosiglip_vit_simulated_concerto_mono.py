import timm
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
from torchvision.transforms import Compose, Resize
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

# Registry =>> Supported DinoSigLIP Pairs (as TIMM identifiers)
DINOSigLIP_NAMES = {
    224: {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_224",
    },
    384: {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
    },
}

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

class ConcertoDinoSigLIPViTMonoBackbone(Backbone2D):
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    dino: ViT
    siglip: ViT

    pc_encoder: nn.Module
    pc_feature_dim: int
    image_feature_dim: int
    empty_pc_token: nn.Parameter

    fuse_ln: nn.LayerNorm
    fuse_mlp: nn.Sequential
    fuse_gate: nn.Parameter

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__(config)

        self.device = torch.device("cuda")
        self.compute_dtype = torch.float32

        self.models = ["dino", "siglip"]
        transforms = []
        for model_type in self.models:
            name = DINOSigLIP_NAMES[config.image_size][model_type]
            model: ViT = ViT(
                timm.create_model(name, pretrained=True, num_classes=0, img_size=config.image_size)
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            model.to(self.device)
            model.model = model.model.to(dtype=self.compute_dtype)
            setattr(self, model_type, model)

            model_cfg = timm.data.resolve_model_data_config(model.model)
            model_cfg["input_size"] = (3, config.image_size, config.image_size)
            transform = timm.data.create_transform(**model_cfg, is_training=False)

            target_size = (config.image_size, config.image_size)
            resize_transform = Compose(
                [Resize(target_size, interpolation=transform.transforms[0].interpolation),
                    *transform.transforms[1:]]
            )
            transforms.append(resize_transform)
        self.image_transform = CombineImageTransform(transforms)
        self.image_feature_dim = self.dino.embed_dim + self.siglip.embed_dim

        print("Initializing Point Cloud Encoder (Concerto)...")
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

        # Gated Residual Fusion
        self.fuse_ln = nn.LayerNorm(self.image_feature_dim).to(self.device, dtype=self.compute_dtype)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.pc_feature_dim, self.image_feature_dim),
            nn.GELU(),
            nn.Linear(self.image_feature_dim, self.image_feature_dim),
        ).to(self.device, dtype=self.compute_dtype)
        # A learnable gate that takes small, slow steps
        self.fuse_gate = nn.Parameter(torch.full((1,), -2.1972, dtype=self.compute_dtype, device=self.device))

    def set_partial_freeze(self, unfreeze_last_spconvs: int = 0):
        self.pc_encoder.requires_grad_(False)
        if unfreeze_last_spconvs > 0:
            for m in self._spconv_layers[-unfreeze_last_spconvs:]:
                m.requires_grad_(True)

    @property
    def feature_dim(self) -> int:
        return self.image_feature_dim

    def forward(self, images: torch.Tensor, depths: Optional[torch.Tensor] = None,
                transformed_pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, n_cams, _, C, H, W = images.shape  # torch.Size([8, 1, 2, 3, 224, 224])
        
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            imgs_dino = images[:, 0, 0].reshape(B, C, H, W)  # front view
            imgs_siglip = images[:, 0, 1].reshape(B, C, H, W)
            feat_dino = self.dino(imgs_dino)[:, 1:, :]  # remove CLS
            feat_siglip = self.siglip(imgs_siglip)[:, 1:, :]  # remove CLS
            
            image_feats_flat = torch.cat([feat_dino, feat_siglip], dim=-1)  # [B, L, D_img]
            L = image_feats_flat.shape[1]
            image_feats_per_view = image_feats_flat.view(B, 1, L, -1)  # [B, 1, L, D_img]
        
        if transformed_pc is None or transformed_pc.numel() == 0:
            print("transformed_pc is None", flush=True)
            pc_feats_per_view = self.empty_pc_token.view(1, 1, 1, -1).expand(B, 1, L, -1)
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                fused_in = torch.cat([pc_feats_per_view.to(dtype=self.compute_dtype), image_feats_per_view], dim=-1)
                delta = self.fuse_mlp(fused_in.reshape(B * L, -1)).reshape(B, 1, L, self.image_feature_dim)
                gate = torch.sigmoid(self.fuse_gate).view(1, 1, 1, 1)
                fused = image_feats_per_view + gate * self.fuse_ln(delta)
                feats_out = fused.reshape(B, L, self.image_feature_dim)
            return feats_out

        # xyz(3) rgb(3) normal(3) grid(3) cam_id(1) patch_idx(1) batch(1)
        with torch.autocast(device_type="cuda", enabled=False):
            point_coord = transformed_pc[:, 0:3].to(dtype=torch.float32)
            point_color = transformed_pc[:, 3:6].to(dtype=torch.float32)
            point_normal = transformed_pc[:, 6:9].to(dtype=torch.float32)
            point_grid = transformed_pc[:, 9:12].long()
            cam_id = transformed_pc[:, 12].long()
            patch_idx = transformed_pc[:, 13].long()
            batch_idx = transformed_pc[:, 14].long()
            
            view_mask = cam_id.eq(1)  # front view
            point_coord = point_coord[view_mask]
            point_color = point_color[view_mask]
            point_normal = point_normal[view_mask]
            point_grid = point_grid[view_mask]
            patch_idx = patch_idx[view_mask]
            batch_idx = batch_idx[view_mask]

            invalid = (patch_idx < 0) | (patch_idx >= L)
            patch_idx = torch.where(invalid, torch.full_like(patch_idx, -1), patch_idx)

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
            
            # print('zero-count patches ratio =', (counts_per_view == 0).float().mean().item(), flush=True)

            pc_feats_per_view = torch.where(
                counts_per_view.eq(0).unsqueeze(-1),
                self.empty_pc_token.view(1, 1, 1, -1).to(dtype=self.compute_dtype),
                pc_feats_per_view.to(dtype=self.compute_dtype)
            )

        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            fused_in = torch.cat([pc_feats_per_view, image_feats_per_view], dim=-1)  # [B,1,L,D_pc+D_img]
            delta = self.fuse_mlp(fused_in.reshape(B * L, -1)).reshape(B, 1, L, self.image_feature_dim)
            gate = torch.sigmoid(self.fuse_gate).view(1, 1, 1, 1)
            fused = image_feats_per_view + gate * self.fuse_ln(delta)
            feats_out = fused.reshape(B, L, self.image_feature_dim)
        return feats_out
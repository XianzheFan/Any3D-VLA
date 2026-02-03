import timm
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
from torchvision import transforms as TF
from torchvision.transforms import Compose, Resize

from . import Backbone2D, Backbone2DConfig
from vla_network.config import ImageTransform
from vggt.models.vggt import VGGT # type: ignore
from vggt.layers.vision_transformer import DinoVisionTransformer # type: ignore

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


class VGGTPreprocessSingle:
    """
    Preprocess a single PIL RGB image following the logic of
    `load_and_preprocess_images(image_path_list, mode="crop")` in VGGT:
    - Convert to RGB.
    - Resize width to target_size while keeping aspect ratio.
      Height is rounded to be divisible by patch_size.
    - If height > target_size, center-crop vertically to target_size.
    - No mean/std normalization: output is in [0, 1].
    """
    def __init__(self, target_size: int = 518, patch_size: int = 14) -> None:
        self.target_size = target_size
        self.patch_size = patch_size
        self.to_tensor = TF.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        w, h = img.size
        ts = self.target_size
        ps = self.patch_size

        # Mode "crop": match width to target_size, adjust height with aspect ratio
        new_w = ts
        new_h = round(h * (new_w / w) / ps) * ps  # make divisible by patch_size
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        x = self.to_tensor(img)  # [3, H, W], in [0, 1]

        # Center-crop height to target_size if needed
        if new_h > ts:
            start_y = (new_h - ts) // 2
            x = x[:, start_y:start_y + ts, :]
        # If you want to replicate "pad" mode, you can add padding here.
        # For "crop" mode, we keep width == target_size and height <= target_size.
        return x


class VGGTImageBranch(nn.Module):
    """
    VGGT image branch that only uses the aggregator to produce image tokens.
    All explicit geometry heads (camera/depth/point/track) are disabled.
    This branch provides implicit 3D priors through image-plane tokens.
    """
    def __init__(self, img_size: int = 518, patch_size: int = 14, embed_dim: int = 1024) -> None:
        super().__init__()
        self.vggt = VGGT(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, enable_camera=False,
                         enable_point=False, enable_depth=False, enable_track=False)
        self.vggt.eval()
        self._token_dim = 2 * embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

    @property
    def embed_dim(self) -> int:
        return self._token_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(1)  # [B, 1, 3, H, W]
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(x_seq)
        tokens = aggregated_tokens_list[-1]  # [B, 1, T, C]
        tokens = tokens[:, 0]  # [B, T, C], since S = 1
        return tokens

class DinoSigLIPVGGTMonoBackbone(Backbone2D):
    # from parent class
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    dino: ViT
    siglip: ViT
    vggt_branch: VGGTImageBranch
    
    image_feature_dim: int
    vggt_feature_dim: int

    # Gated residual fusion
    fuse_ln: nn.LayerNorm
    fuse_mlp: nn.Sequential
    fuse_gate: nn.Parameter

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__(config)
        self.models = ["dino", "siglip"]

        transforms: List[ImageTransform] = []
        for model_type in self.models:
            name = DINOSigLIP_NAMES[config.image_size][model_type]
            vit_model: ViT = ViT(
                timm.create_model(
                    name, pretrained=True, num_classes=0, img_size=config.image_size
                )
            )
            vit_model.eval()
            for p in vit_model.parameters():
                p.requires_grad = False
                
            setattr(self, model_type, vit_model)
            model_cfg = timm.data.resolve_model_data_config(vit_model.model)
            model_cfg["input_size"] = (3, config.image_size, config.image_size)
            transform = timm.data.create_transform(**model_cfg, is_training=False)

            # Replace the resize transform with the target size
            target_size = (config.image_size, config.image_size)
            resize_transform = Compose(
                [
                    Resize(
                        target_size, interpolation=transform.transforms[0].interpolation
                    ),
                    *transform.transforms[1:],
                ]
            ) 
            transforms.append(resize_transform)
        self.image_feature_dim = self.dino.embed_dim + self.siglip.embed_dim

        # For best compatibility, config.image_size is recommended to be 518 when using VGGT (the default VGGT target size)
        self.vggt_branch = VGGTImageBranch(img_size=config.image_size, patch_size=14, embed_dim=1024)
        self.vggt_branch.eval()
        self.vggt_feature_dim = self.vggt_branch.embed_dim
        self.set_partial_freeze_vggt(unfreeze_last_blocks=4)
        
        vggt_transform = VGGTPreprocessSingle(target_size=config.image_size, patch_size=14)
        transforms.append(vggt_transform)
        self.image_transform = CombineImageTransform(transforms)
        
        self.fuse_ln = nn.LayerNorm(self.image_feature_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.image_feature_dim + self.vggt_feature_dim, self.image_feature_dim),
            nn.GELU(),
            nn.Linear(self.image_feature_dim, self.image_feature_dim),
        )
        self.fuse_gate = nn.Parameter(torch.full((1,), -2.1972))

    @property
    def feature_dim(self) -> int:
        return self.image_feature_dim
    
    def _get_vggt_vit(self) -> DinoVisionTransformer:
        v = self.vggt_branch.vggt
        if isinstance(v, DinoVisionTransformer):
            return v
        for m in v.modules():
            if isinstance(m, DinoVisionTransformer):
                return m
        raise RuntimeError("Could not locate DinoVisionTransformer inside VGGT. Please check VGGT implementation and adjust _get_vggt_vit().")
    
    def set_partial_freeze_vggt(self, unfreeze_last_blocks: int = 4) -> None:
        vit = self._get_vggt_vit()
        vit.requires_grad_(False)
        all_blocks = []
        if vit.chunked_blocks:
            for chunk in vit.blocks:
                for b in chunk:
                    if b.__class__.__name__ != "Identity":
                        all_blocks.append(b)
        else:
            for b in vit.blocks:
                all_blocks.append(b)

        total = len(all_blocks)
        n_unfreeze = min(unfreeze_last_blocks, total)
        for b in all_blocks[-n_unfreeze:]:
            b.requires_grad_(True)
        print(f"Unfrozen last {n_unfreeze} VGGT transformer blocks (total blocks = {total}).", flush=True)
        # Unfrozen last 4 VGGT transformer blocks (total blocks = 24)

    def forward(self, images: torch.Tensor, depths: Optional[torch.Tensor] = None, transformed_pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, n_cams, _, C, H, W = images.shape
        imgs_dino = images[:, 0, 0].reshape(B, C, H, W)
        imgs_siglip = images[:, 0, 1].reshape(B, C, H, W)

        feat_dino = self.dino(imgs_dino)[:, 1:, :]  # remove CLS token: [B*n_cams, L, D_dino]
        feat_siglip = self.siglip(imgs_siglip)[:, 1:, :]  # [B*n_cams, L, D_siglip]

        image_feats_flat = torch.cat([feat_dino, feat_siglip], dim=-1)  # [B*n_cams, L, D_img]
        L = image_feats_flat.shape[1]
        image_feats_per_view = image_feats_flat.view(B, 1, L, self.image_feature_dim)

        vggt_imgs = images[:, :, 2].reshape(B, C, H, W)
        vggt_feat_flat = self.vggt_branch(vggt_imgs)  # [B*n_cams, L_vggt, D_vggt]
        
        L_vggt = vggt_feat_flat.shape[1]
        if L_vggt != L:
            if L_vggt > L:
                vggt_feat_flat = vggt_feat_flat[:, :L, :]
            else:
                pad_len = L - L_vggt
                pad = vggt_feat_flat.new_zeros(B, pad_len, self.vggt_feature_dim)
                vggt_feat_flat = torch.cat([vggt_feat_flat, pad], dim=1)
        
        vggt_feats_per_view = vggt_feat_flat.view(B, 1, L, self.vggt_feature_dim)
        fused_in = torch.cat([vggt_feats_per_view, image_feats_per_view], dim=-1)

        delta = self.fuse_mlp(fused_in.view(B * L, -1))
        delta = delta.view(B, 1, L, self.image_feature_dim)

        gate = torch.sigmoid(self.fuse_gate).view(1, 1, 1, 1)
        fused = image_feats_per_view + gate * self.fuse_ln(delta)
        feats_out = fused.view(B, L, self.image_feature_dim)
        return feats_out
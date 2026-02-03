import timm
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
from torchvision.transforms import Compose, Resize

from . import Backbone2D, Backbone2DConfig
from vla_network.config import ImageTransform

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


class DinoSigLIPViTSimulatedDepthMonoBackbone(Backbone2D):
    # from parent class
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    rgb_dino: ViT
    rgb_siglip: ViT
    depth_dino: ViT
    depth_siglip: ViT

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__(config)
        self.model_keys = ["dino", "siglip"]
        transforms = []
        for model_type in self.model_keys:
            name = DINOSigLIP_NAMES[config.image_size][model_type]
            raw_rgb_model = timm.create_model(
                name, pretrained=True, num_classes=0, img_size=config.image_size
            )
            raw_rgb_model.eval()
            for param in raw_rgb_model.parameters():
                param.requires_grad = False
            
            rgb_vit = ViT(raw_rgb_model)
            setattr(self, f"rgb_{model_type}", rgb_vit)
            raw_depth_model = timm.create_model(
                name, pretrained=True, num_classes=0, img_size=config.image_size
            )
            raw_depth_model.train()
            for param in raw_depth_model.parameters():
                param.requires_grad = False

            for block in raw_depth_model.blocks[-4:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            if hasattr(raw_depth_model, 'norm'):
                for param in raw_depth_model.norm.parameters():
                    param.requires_grad = True

            depth_vit = ViT(raw_depth_model)
            setattr(self, f"depth_{model_type}", depth_vit)

            model_cfg = timm.data.resolve_model_data_config(raw_rgb_model)
            model_cfg["input_size"] = (3, config.image_size, config.image_size)
            transform = timm.data.create_transform(**model_cfg, is_training=False)

            target_size = (config.image_size, config.image_size)
            resize_transform = Compose(
                [Resize(target_size, interpolation=transform.transforms[0].interpolation),
                    *transform.transforms[1:]]
            )
            transforms.append(resize_transform)
        self.image_transform = CombineImageTransform(transforms)
        self.fusion_proj = nn.ModuleDict()
        for k in self.model_keys:
            m: ViT = getattr(self, f"rgb_{k}")
            self.fusion_proj[k] = nn.Linear(2 * m.embed_dim, m.embed_dim)

    @property
    def feature_dim(self) -> int:
        return self.rgb_dino.embed_dim + self.rgb_siglip.embed_dim

    def forward(self, images: torch.Tensor, depths: Optional[torch.Tensor] = None, transformed_pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, S, C, H, W = images.shape
        
        if depths is not None:
            depths = torch.nan_to_num(depths, nan=0.0, posinf=2.0, neginf=0.0)
            depths = torch.clamp(depths, min=0.0, max=2.0)
        if depths.shape[-2:] != (H, W):
            _b, _n, _s, _h, _w = depths.shape
            depths_reshaped = depths.view(_b * _n * _s, 1, _h, _w)
            depths_resized = torch.nn.functional.interpolate(depths_reshaped, size=(H, W), mode='bilinear', align_corners=False)
            depths = depths_resized.view(_b, _n, _s, H, W)

        feats = []
        for i, k in enumerate(self.model_keys):
            with torch.no_grad():
                rgb_input = images[:, 0, i]
                rgb_feat = getattr(self, f"rgb_{k}")(rgb_input)
            d_slice = depths[:, 0, i]
            depth_input = d_slice.unsqueeze(1).repeat(1, 3, 1, 1)
            depth_feat = getattr(self, f"depth_{k}")(depth_input)
            fused_feat = self.fusion_proj[k](torch.cat([rgb_feat, depth_feat], dim=-1))
            feats.append(fused_feat.unsqueeze(1))
        return torch.cat(feats, dim=-1)
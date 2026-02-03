import timm
import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Optional
from timm.models.vision_transformer import VisionTransformer
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize
import cv2

from . import Backbone2D, Backbone2DConfig
from vla_network.config import ImageTransform
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize as DepthResize, NormalizeImage, PrepareForNet

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

DEPTH_ANYTHING_V2_LOCAL_MODELS = {
    "small": "depth_anything_v2_models/depth_anything_v2_vits.pth",
    "base": "depth_anything_v2_models/depth_anything_v2_vitb.pth", 
    "large": "depth_anything_v2_models/depth_anything_v2_vitl.pth"
}

DEPTH_ANYTHING_V2_CONFIGS = {
    "small": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "patch_size": 14,
        "embed_dim": 384  # DINOv2 vits embedding dimension
    },
    "base": {
        "encoder": "vitb", 
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "patch_size": 14,
        "embed_dim": 768  # DINOv2 vitb embedding dimension
    },
    "large": {
        "encoder": "vitl",
        "features": 256, 
        "out_channels": [256, 512, 1024, 1024],
        "patch_size": 14,
        "embed_dim": 1024  # DINOv2 vitl embedding dimension
    }
}

@dataclass
class CombineImageTransform:
    transforms: List[ImageTransform]

    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        return torch.stack([t(img, **kwargs) for t in self.transforms], dim=0)

class DepthImageTransform:
    """Custom transform for depth processing that matches DPT preprocessing"""
    
    def __init__(self, input_size: int = 518):
        self.input_size = input_size
        self.transform = Compose([
            DepthResize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        # Convert PIL Image to numpy array (BGR format like cv2)
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to RGB and normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) / 255.0
        
        # Apply the depth transform
        transformed = self.transform({'image': img_rgb})['image']
        
        return torch.from_numpy(transformed)

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

class DepthFeatureExtractor(nn.Module):
    """Extract intermediate layer features from DINOv2 backbone in Depth Anything V2"""
    def __init__(self, model_size: str = "large"):
        super().__init__()
        self.model_size = model_size
        config = DEPTH_ANYTHING_V2_CONFIGS[model_size]
        
        self.depth_model = DepthAnythingV2(
            encoder=config["encoder"],
            features=config["features"],
            out_channels=config["out_channels"]
        )
        
        model_path = DEPTH_ANYTHING_V2_LOCAL_MODELS[model_size]
        checkpoint = torch.load(model_path, map_location='cpu')
        self.depth_model.load_state_dict(checkpoint)
        for p in self.depth_model.parameters():
            p.requires_grad = False
        
        encoder = self.depth_model.pretrained
        blocks = encoder.blocks  # list of transformer blocks
        n_unfreeze = 4
        
        for block in blocks[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True
        
        # Set feature dimension to DINOv2 embedding dimension
        self.feature_dim = config["embed_dim"]
        self.intermediate_layer_idx = self.depth_model.intermediate_layer_idx[config["encoder"]]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate layer features from DINOv2 backbone
        Args:
            x: Input image tensor [B, C, H, W]
        """
        features = self.depth_model.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx, return_class_token=True
        )  # 4 layers
        
        # features is a list of tuples: [(patch_features, class_token), ...]
        last_layer_features = features[-1][0]  # Get patch features from last layer
        return last_layer_features

class DinoSigLIPViTDepthMonoBackbone(Backbone2D):
    """
    Backbone combining DinoV2, SigLIP, and Depth Anything V2 features
    Ensures all modalities produce 16x16 patches for consistent fusion
    """
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    dino: ViT
    siglip: ViT
    depth_extractor: DepthFeatureExtractor
    fusion_layer: nn.Module

    def __init__(self, config: Backbone2DConfig, depth_model_size: str = "large") -> None:
        super().__init__(config)
        self.models = ["dino", "siglip"]
        self.depth_model_size = depth_model_size

        transforms = []
        for model_type in self.models:
            name = DINOSigLIP_NAMES[config.image_size][model_type]
            model: ViT = ViT(
                timm.create_model(name, pretrained=True, num_classes=0, img_size=config.image_size)
            )
            model.eval()

            model_cfg = timm.data.resolve_model_data_config(model.model)
            model_cfg["input_size"] = (3, config.image_size, config.image_size)
            transform = timm.data.create_transform(**model_cfg, is_training=False)

            target_size = (config.image_size, config.image_size)
            resize_transform = Compose(
                [Resize(target_size, interpolation=transform.transforms[0].interpolation),
                    *transform.transforms[1:]]
            )

            setattr(self, model_type, model)
            transforms.append(resize_transform)

        self.depth_extractor = DepthFeatureExtractor(depth_model_size)
        
        depth_transform = DepthImageTransform(input_size=config.image_size)
        transforms.append(depth_transform)

        self.image_transform = CombineImageTransform(transforms)
        depth_dim = self.depth_extractor.feature_dim
        target_dim = self.dino.embed_dim + self.siglip.embed_dim
        
        self.depth_align = nn.Linear(depth_dim, target_dim)
        
        self.total_feature_dim = target_dim * 2  # dinosiglip + aligned_depth
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.total_feature_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU()
        )

    @property
    def feature_dim(self) -> int:
        return self.dino.embed_dim + self.siglip.embed_dim

    def forward(self, images: torch.Tensor, depths: Optional[torch.Tensor] = None,
                transformed_pc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with multi-modal feature fusion
        Args:
            images: Input images [B, N_views, N_transforms, H, W] 
                   where N_transforms=3 (dino, siglip, depth)
        Returns:
            Fused features [B, N_patches, feature_dim] with consistent patch count
        """
        b, n, _, *chw = images.shape
        depth_feats = self.depth_extractor(images[:, 0, 2].reshape(b, *chw))  # [12, 256, 1024]

        feats = []
        for i, k in enumerate(self.models):
            front_view_input = images[:, 0, i].reshape(b, *chw)
            feat = getattr(self, k)(front_view_input)
            feats.append(feat.reshape(b, -1, feat.shape[-1]))
        
        dinosiglip_feats = torch.cat(feats, dim=-1)  # [12, 256, 2176]
        aligned_depth_feats = self.depth_align(depth_feats)
        all_feats = torch.cat([dinosiglip_feats, aligned_depth_feats], dim=-1)
        fused_feats = self.fusion_layer(all_feats)
        return fused_feats
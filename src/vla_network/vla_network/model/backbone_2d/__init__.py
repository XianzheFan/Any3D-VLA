from abc import ABC, abstractmethod
import torch
from torch import nn

from vla_network.config import Backbone2DConfig, ImageTransform


class Backbone2D(nn.Module, ABC):
    config: Backbone2DConfig
    image_transform: ImageTransform

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init(config: Backbone2DConfig) -> "Backbone2D":
        if config.name == "dinosiglipdepthmono":
            from .dinosiglip_vit_depth_mono import DinoSigLIPViTDepthMonoBackbone
            return DinoSigLIPViTDepthMonoBackbone(config)
        if config.name == "dinosiglipmono":
            from .dinosiglip_vit_mono import DinoSigLIPViTMonoBackbone
            return DinoSigLIPViTMonoBackbone(config)
        if config.name == "dinosiglipvggtMono":
            from .dinosiglip_vit_vggt_mono import DinoSigLIPVGGTMonoBackbone
            return DinoSigLIPVGGTMonoBackbone(config)
        if config.name == "dinosiglipsimulateddepthMono":
            from .dinosiglip_vit_simulated_depth_mono import DinoSigLIPViTSimulatedDepthMonoBackbone
            return DinoSigLIPViTSimulatedDepthMonoBackbone(config)
        if config.name == "dinosiglipsimulatedConcertoMono":
            from .dinosiglip_vit_simulated_concerto_mono import ConcertoDinoSigLIPViTMonoBackbone
            return ConcertoDinoSigLIPViTMonoBackbone(config)
        if config.name == "smalldinosiglipsimulatedConcertoMono":
            from .smalldino_siglip_vit_simulated_concerto_mono import ConcertoSmallDinoSigLIPViTMonoBackbone
            return ConcertoSmallDinoSigLIPViTMonoBackbone(config)
        if config.name == "ConcertoMono":
            from .vit_simulated_concerto_mono import ConcertoViTMonoBackbone
            return ConcertoViTMonoBackbone(config)
        else:
            raise NotImplementedError

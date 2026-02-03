from dataclasses import dataclass, field
from typing import Optional, List, Any
import torch

from gx_utils.dtype import RawVLAData


@dataclass
class BatchVLAData:
    debug: List[Any]

    # tokens
    input_ids: torch.Tensor  # (B, N_token)
    robot_input_ids: torch.Tensor # (B, N_robot_token)
    labels: Optional[torch.Tensor]  # (B, N_token)
    robot_labels: Optional[torch.Tensor] # (B, N_robot_token)
    attention_mask: torch.Tensor  # (B, N_token)
    robot_attention_mask: torch.Tensor # (B, N_robot_token)

    # robot
    action: torch.Tensor # (B, T_action, D_action)
    proprio: torch.Tensor # (B, T_proprio, D_proprio)
    goal: Optional[torch.Tensor] # (B, D_goal)

    # Images
    images: torch.Tensor  # (B, T_image, N_backbone, C, H, W)
    depths: Optional[torch.Tensor]
    
    # type
    is_action: torch.Tensor  # (B,)
    transformed_pc: Optional[torch.Tensor]

    # inference
    inference_kwargs: Optional[list] = None


@dataclass
class BatchVAData:
    pcs: torch.Tensor = field(default=None)  # (B, T_pc, N_pc, 3)
    proprio: torch.Tensor = field(default=None)  # (B, T_proprio, D_proprio)
    proprio_trans: torch.Tensor = field(default=None)  # (B, T_proprio, 3)
    proprio_rot: torch.Tensor = field(default=None)  # (B, T_proprio, 3, 3)
    action: torch.Tensor = field(default=None)  # (B, T_action, D_action)
    action_trans: torch.Tensor = field(default=None)  # (B, T_action, 3)
    action_rot: torch.Tensor = field(default=None)  # (B, T_action, 3, 3)
    goal: torch.Tensor = field(default=None)  # (B, D_goal)
    goal_trans: torch.Tensor = field(default=None)  # (B, 3)
    goal_rot: torch.Tensor = field(default=None)  # (B, 3, 3)
    robot_pcs: torch.Tensor = field(default=None)  # (B, T_pc2, N_pc2, 3)
    goal_pcs: torch.Tensor = field(default=None)  # (B, T_pc2, N_pc2, 3)
    depths: torch.Tensor = field(default=None)
    transformed_pc: torch.Tensor = field(default=None)
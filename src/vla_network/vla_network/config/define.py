import os
from os.path import join, dirname, abspath
from typing import Optional, List, Type, Union, Dict
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
import torch
import importlib
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    AutoTokenizer
)

from gx_utils.logger import log
from gx_utils.file_manager import get_path_exp
from gx_utils.robot import get_robot_cfg


def optional_str(x: Union[str, None]) -> Union[str, None]:
    if x is None or x == "none" or x == "None":
        return None
    else:
        return x


class ImageTransform:
    def __call__(
        self, img: Image, **kwargs: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

class MixDatasetConfig(BaseModel):
    datasets: Optional[List[str]] = Field(default=None)
    datasets_type: Optional[List[str]] = Field(default=None)
    datasets_weight: Optional[List[float]] = Field(default=None)

    @staticmethod
    def from_str(string: str) -> "MixDatasetConfig":
        datasets = []
        datasets_type = []
        datasets_weight = []
        for item in string.split(";"):
            d, dt, dw = item.split(',')
            datasets.append(d)
            datasets_type.append(dt)
            datasets_weight.append(float(dw))
        return MixDatasetConfig(
            datasets=datasets,
            datasets_type=datasets_type,
            datasets_weight=datasets_weight,
        )


class BasicDataConfig(BaseModel):
    exp_name: Optional[str] = Field(default=None)
    train_datasets: Union[str, MixDatasetConfig] = Field(default=MixDatasetConfig())
    val_datasets: Optional[Union[str, MixDatasetConfig]] = Field(default=None)
    robot: str
    proprio_len: int
    action_len: int
    action_dim: int = Field(default=None)
    goal_dim: Optional[int] = Field(default=None)
    action_rel_len: int
    dt_steps: int
    
    def setup(self):
        if isinstance(self.train_datasets, str):
            self.train_datasets = MixDatasetConfig.from_str(self.train_datasets)
        if isinstance(self.val_datasets, str):
            self.val_datasets = MixDatasetConfig.from_str(self.val_datasets)


class VADataConfig(BasicDataConfig):
    pass


class AvoidEverythingDataConfig(VADataConfig):
    vis_links: List[str]


class AugImgConfig(BaseModel):
    brightness: float
    contrast: float
    saturation: float
    hue: float


class VLADataConfig(BasicDataConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Optional[PreTrainedTokenizerBase] = Field(init=False, default=None)
    image_transform: Optional[ImageTransform] = Field(init=False, default=None)
    action_token_num: int
    img_steps: int
    img_key: Optional[List[str]]
    image_size: Optional[int]
    anything_prob: float
    robot_rep: str
    goal_rep: Optional[str]
    tokenizer_type: str
    tokenizer_ratio_limit: float
    count_num: int
    trans_noise: float
    rot_noise: float
    aug_img_config: AugImgConfig
    brightness_img: str
    brightness_threshold: float
    crop_mode: Dict[str, str]
    proprio_dim: Optional[int] = Field(default=None)
    use_bbox: int
    use_depth: int # 1 true 0 false
    use_depth_pro: Optional[int] = Field(default=0)
    use_unidepthv2: Optional[int] = Field(default=0)
    pred: Optional[str] = Field(init=False, default=None)

    def setup(self):
        super().setup()

        if self.action_dim is None:
            if self.robot_rep in ['xyz_rpy', 'xyz_rpy_rot', 'identity']:
                self.action_dim = 7 # xyz rpy gripper
        if self.goal_dim is None:
            if self.goal_rep in ['xyz_rpy', 'identity']:
                self.goal_dim = 6 # xyz rpy
            elif self.goal_rep == 'xyz_rot':
                self.goal_dim = 12 # xyz rotmat
        if self.proprio_dim is None:
            if self.robot_rep in ['xyz_rpy', 'identity']:
                self.proprio_dim = 7 # xyz rpy gripper
            elif self.robot_rep == 'xyz_rpy_rot':
                self.proprio_dim = 13 # xyz rotmat gripper

        if self.img_key is None:
            robot_cfg = get_robot_cfg(self.robot)
            self.img_key = robot_cfg.camera_names
    
    @property
    def img_num(self) -> int:
        return len(self.img_key) * self.img_steps


LLM_CONFIG = {
    "meta-llama/Llama-2-7b-hf": {
        "family": "llama2",
        "model_cls": ("transformers", "LlamaForCausalLM"),
        "token_cls": ("transformers", "AutoTokenizer"),
    },
    "internlm/internlm2-1_8b": {
        "family": "internlm",
        "model_cls": (
            "vla_network.model.backbone_llm.internlm.modeling_internlm2",
            "InternLM2ForCausalLM",
        ),
        "token_cls": (
            "vla_network.model.backbone_llm.internlm.tokenization_internlm2_fast",
            "InternLM2TokenizerFast",
        ),
    },
    "Qwen/Qwen2-1_5B": {
        "family": "qwen2",
        "model_cls": (
            "vla_network.model.backbone_llm.qwen2.modeling_qwen2",
            "Qwen2ForCausalLM",
        ),
        "token_cls": (
            "vla_network.model.backbone_llm.qwen2.tokenization_qwen2_fast",
            "Qwen2TokenizerFast",
        ),
    },
}


class BasicModelConfig(BaseModel):
    pass


class VAModelConfig(BasicModelConfig):
    name: str
    proprio_dim: int
    action_dim: int
    qpos_scale: float
    delta_goal: int
    input_eef_pose: int


class LLMConfig(BaseModel):
    name: str
    max_len: int = Field(default=2048)
    special_tokens: List[str] = Field(default_factory=lambda: [])
    pad_multiple_of: int = Field(default=64)
    attn_implementation: str

    @property
    def family(self) -> str:
        return LLM_CONFIG[self.name]["family"]

    @staticmethod
    def get_cls(package: str, name: str):
        module = importlib.import_module(package)
        return getattr(module, name)

    @property
    def model_cls(self) -> Type[PreTrainedModel]:
        cls_package, cls_name = LLM_CONFIG[self.name]["model_cls"]
        return self.get_cls(cls_package, cls_name)

    @property
    def token_cls(self) -> Type[PreTrainedTokenizerFast]:
        cls_package, cls_name = LLM_CONFIG[self.name]["token_cls"]
        return self.get_cls(cls_package, cls_name)

class Backbone2DConfig(BaseModel):
    name: str
    image_size: int

class ActionExpertConfig(BaseModel):
    hidden_size_scale: Optional[int] = Field(default=None)
    intermediate_size_scale: Optional[int] = Field(default=None)
    hidden_size: Optional[int] = Field(init=False, default=None)
    intermediate_size: Optional[int] = Field(init=False, default=None)
    hidden_act: Optional[str] = Field(init=False, default=None)
            
class AvoidEverythingModelConfig(VAModelConfig):
    num_target_points: int
    num_robot_points: int
    num_obstacle_points: int
    point_match_loss_weight: float
    collision_loss_weight: float
    collision_loss_margin: float
    pc_bounds: List[List[float]]

class FlowMatchingConfig(BaseModel):
    beta_alpha: float
    beta_beta: float
    time_min: float
    time_max: float

class VLAModelConfig(BasicModelConfig):
    backbone_2d: Backbone2DConfig
    llm: LLMConfig
    ckpt: str
    pred: str # flow_matching or token_pred
    action_len: int = Field(init=False, default=None)
    action_dim: int = Field(init=False, default=None)
    proprio_dim: int = Field(init=False, default=None)
    action_expert: int
    action_expert_cfg: Optional[ActionExpertConfig] = None
    flow_matching_cfg: Optional[FlowMatchingConfig] = None

    def to_dict(self):
        return self.model_dump()

class BasicTrainConfig(BaseModel):
    exp_name: str
    args: Optional[TrainingArguments] = Field(init=False, default=None)
    max_steps: int
    global_batch_size: int
    device_batch_size: int
    lr: float
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    log_step: int
    save_step: int
    save_total_limit: int
    eval_step: int
    eval_each: int
    num_workers: int
    deepspeed: Optional[str]
    fsdp: Optional[str]
    cache_path: Optional[str]
    profiler: bool
    bf16: bool
    full_bf16: bool
    gradient_checkpointing: bool
    resume_from: Optional[str] = Field(default=None)

    def setup(self):
        # huggingface cache
        if self.cache_path is not None:
            os.system(f"rm -rf ~/.cache && ln -s {self.cache_path} ~/.cache")

        # set batch size
        device_count = int(os.getenv("WORLD_SIZE", 1))
        assert self.global_batch_size % (self.device_batch_size * device_count) == 0, f'global_bs {self.global_batch_size} % (device_bs {self.device_batch_size} * device_count {device_count}) != 0'
        grad_accum = self.global_batch_size // (self.device_batch_size * device_count)
        log.info(
            f"Batch size {self.global_batch_size} = "
            + f"gpu num {device_count} * "
            + f"device batch size {self.device_batch_size} * "
            + f"grad_accum {grad_accum}"
        )

        # wandb
        os.environ["WANDB_PROJECT"] = "any3d_vla"

        # deepspeed
        deepspeed = optional_str(self.deepspeed)
        if deepspeed is not None:
            deepspeed = join(dirname(abspath(__file__)), deepspeed)

        self.args = TrainingArguments(
            output_dir=get_path_exp(self.exp_name),
            run_name=self.exp_name,
            deepspeed=deepspeed,
            max_steps=self.max_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            bf16=self.bf16,
            learning_rate=self.lr,
            max_grad_norm=self.max_grad_norm,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            weight_decay=self.weight_decay,
            per_device_train_batch_size=self.device_batch_size,
            per_device_eval_batch_size=self.device_batch_size,
            gradient_accumulation_steps=grad_accum,
            report_to="wandb" if self.exp_name != "debug" else "none",
            logging_strategy="steps",
            logging_steps=self.log_step,
            eval_strategy="steps",
            eval_steps=self.eval_step,
            eval_delay=self.eval_step,
            save_strategy="steps",
            save_steps=self.save_step,
            save_total_limit=self.save_total_limit,
            dataloader_drop_last=True,
            dataloader_num_workers=self.num_workers,
            dataloader_persistent_workers=False,
            # split_batches=True,
            # dispatch_batches=False,
            dataloader_prefetch_factor=10 if (self.num_workers > 1) else None,
            fsdp=self.fsdp,
            fsdp_config=dict(
                limit_all_gathers=True,
                use_orig_params=True,
            ),
        )

class VLATrainConfig(BasicTrainConfig):
    backbone_2d_mode: str
    projector_mode: str
    llm_mode: str


class BasicConfig(BaseModel):
    data: BasicDataConfig
    model: BasicModelConfig
    train: BasicTrainConfig
    dummy: str = None  # For unneeded arguments


class VLAConfig(BasicConfig):
    data: VLADataConfig
    model: VLAModelConfig
    train: VLATrainConfig
    dummy: str = None  # For unneeded arguments

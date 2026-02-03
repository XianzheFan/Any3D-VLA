import os
from vla_network.utils.path import get_path_pretrained

from .args import Arg, Args, get_config  # type: ignore
from .define import optional_str, ImageTransform, Backbone2DConfig, BasicDataConfig, VADataConfig, VLADataConfig, BasicModelConfig, VAModelConfig, LLMConfig, VLAModelConfig, VATrainConfig, VLATrainConfig, BasicConfig, VAConfig, VLAConfig, AvoidEverythingModelConfig, AvoidEverythingDataConfig, AugImgConfig, MixDatasetConfig, ActionExpertConfig, FlowMatchingConfig  # type: ignore

ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "franka")

aug_img_config = AugImgConfig(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05,
)

VLADataDefault = VLADataConfig(
    train_datasets=MixDatasetConfig(
        datasets=["dummy"], datasets_type=["dummy"], datasets_weight=None
    ),
    robot='franka',
    action_token_num=256,
    img_steps=1,
    img_key=["front"] if ROBOT_TYPE == 'franka' else None,
    # img_key=["front", "side"] if ROBOT_TYPE == 'franka' else None,
    image_size=None,
    anything_prob=0.2,
    proprio_len=2,
    action_len=4,
    action_rel_len=0,
    dt_steps=3,
    robot_rep="xyz_rpy",
    goal_rep="xyz_rpy",
    tokenizer_type="ratio_min_max_uniform",
    tokenizer_ratio_limit=0.01,
    count_num=25000,
    trans_noise=0.01,
    rot_noise=0.05,
    aug_img_config=aug_img_config,
    brightness_img="front" if ROBOT_TYPE == 'franka' else "front_camera",
    brightness_threshold=50,
    crop_mode=dict() if ROBOT_TYPE == 'franka' else dict(front_camera="left", left_camera="center"),
    use_bbox=0,
    use_depth=1,
    use_depth_pro=0,
    use_unidepthv2=0,
)

VAModelDefault = VAModelConfig(
    name="example",
    proprio_dim=7,
    action_dim=7,
    qpos_scale=0.025,
    delta_goal=1,
    input_eef_pose=0,
)

AvoidEverythingModel = AvoidEverythingModelConfig(
    name="avoid_everything",
    proprio_dim=7,
    action_dim=7,
    qpos_scale=0.025,
    delta_goal=1,
    input_eef_pose=1,
    num_target_points=128,
    num_robot_points=2048,
    num_obstacle_points=4096,
    point_match_loss_weight=1,
    collision_loss_weight=5,
    collision_loss_margin=0.03,
    pc_bounds=[[-1.5, -1.5, -0.1], [1.5, 1.5, 1.5]],
)

VLAModelDefault = VLAModelConfig(
    backbone_2d=Backbone2DConfig(name="dinosiglip", image_size=224),
    llm=LLMConfig(
        name="internlm/internlm2-1_8b",
        attn_implementation="flash_attention_2",
    ),
    ckpt=get_path_pretrained("prism-dinosiglip-224px+1_8b/checkpoints/latest-checkpoint.pt"),
    pred="token_pred",
    action_expert=0,
    action_expert_cfg=ActionExpertConfig(
        hidden_size_scale=2,
        intermediate_size_scale=4,
    ),
    flow_matching_cfg=FlowMatchingConfig(
        beta_alpha=1.5,
        beta_beta=1.0,
        time_min=0.001,
        time_max=1.0,
    )
)

VATrainDefault = VATrainConfig(
    exp_name="temp",
    max_steps=10000,
    global_batch_size=64,
    device_batch_size=64,
    lr=1e-3,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    max_grad_norm=10.0,
    warmup_ratio=0.0,
    log_step=100,
    eval_step=500,
    eval_each=25,
    save_total_limit=1,
    save_step=10000,
    num_workers=32,
    deepspeed=None,
    fsdp='',
    cache_path=None,
    profiler=False,
    bf16=False,
    full_bf16=False,
    gradient_checkpointing=False,
)

VLATrainDefault = VLATrainConfig(
    exp_name="temp",
    backbone_2d_mode="freeze",
    llm_mode="train",
    projector_mode="train",
    max_steps=25000,
    global_batch_size=512,
    device_batch_size=64,
    lr=1.6e-4,
    lr_scheduler_type="constant" if ROBOT_TYPE == 'franka' else "cosine",
    weight_decay=0.0,
    max_grad_norm=1.0,
    warmup_ratio=0.0 if ROBOT_TYPE == 'franka' else 0.03,
    log_step=10,
    save_step=2500,
    save_total_limit=1000000000,
    eval_step=2500,
    eval_each=25,
    num_workers=64,
    deepspeed="stage1.json",
    fsdp="",
    cache_path=None,
    profiler=False,
    # bf16=False,
    bf16=True,
    # full_bf16=False,
    full_bf16=True if ROBOT_TYPE == 'franka' else False,
    gradient_checkpointing=True,
)

VADefault = VAConfig(
    data=VADataDefault,
    model=VAModelDefault,
    train=VATrainDefault,
)

AvoidEverythingConfig = VAConfig(
    data=AvoidEverythingData,
    model=AvoidEverythingModel,
    train=VATrainDefault,
)

VLADefault = VLAConfig(
    data=VLADataDefault,
    model=VLAModelDefault,
    train=VLATrainDefault,
)


def get_config_from_args(args: Args, default: BasicConfig) -> BasicConfig:
    config: BasicConfig = get_config(args, default)
    config.train.setup()
    config.data.setup()
    return config

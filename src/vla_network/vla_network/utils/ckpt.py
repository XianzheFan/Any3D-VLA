import re
from os import listdir
from os.path import join, exists
from typing import Optional
from safetensors.torch import load_file
from torch import nn

from model_utils.logger import log
from model_utils.file_manager import get_path_exp


def get_ckpt_path(root: str, it: Optional[int] = None) -> str:
    if it is None:
        ckpt_iters = [
            int(re.match(r"checkpoint-(\d+)", d).group(1))
            for d in listdir(root)
            if re.match(r"checkpoint-\d+", d)
        ]
        it = max(ckpt_iters)
    ckpt_path = get_path_ckpt(exp_path=root, it=it)
    assert exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    log.info(f"Loading checkpoint from {ckpt_path}")
    return ckpt_path


def load_safetensors(path: str) -> dict:
    return load_file(path)


def convert_old_action_expert_ckpt(ckpt: dict, model: nn.Module):
    """
    Our previous implementation of action expert is within LLM backbone,
        which causes challenges in inference speedup.
    This function provides compatibility for loading previous checkpoints into new ones.
    """
    modified_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("llm.llm.") and "action_expert_" in k:
            # Extract the layer number and parameter name
            match = re.search(r"llm\.llm\.(.+)\.action_expert_(.*)", k)
            if match:
                layer_num, param_name = match.groups()
                new_key = f"action_expert.{layer_num}.{param_name}"
                modified_ckpt[new_key] = v
        else:
            modified_ckpt[k] = v
    # These parameters only work in language models, simply copy it to avoid missing keys warnings of load_state_dict().
    for k in ['action_expert.model.tok_embeddings.weight', 'action_expert.output.weight']:
        modified_ckpt[k] = dict(model.named_parameters())[k]
    return modified_ckpt


def load_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = load_safetensors(ckpt_path)

    # For compatibility with old checkpoints which implement action expert within LLM backbone.
    if any(k.startswith("llm.llm.") and "action_expert_" in k for k in ckpt.keys()):
        ckpt = convert_old_action_expert_ckpt(ckpt, model)

    model.load_state_dict(ckpt)
    return model

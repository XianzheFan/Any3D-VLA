from transformers import TrainerCallback
from transformers.trainer import TrainingArguments

from model_utils.logger import log


class FixTrainerStateCallback(TrainerCallback):
    """
    Huggingface trainer ignores training arguments on resume. Fix it with this class.
    """
    def __init__(self, args: TrainingArguments):
        self.args = args
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        to_fix = ["save_steps", "max_steps", "logging_steps", "eval_steps"]
        for name in to_fix:
            args_val = getattr(self.args, name)
            state_val = getattr(state, name)
            if state_val != args_val:
                log.info(f"fix trainer.state.{name} from {state_val} to {args_val}")
                setattr(state, name, args_val)
        return control

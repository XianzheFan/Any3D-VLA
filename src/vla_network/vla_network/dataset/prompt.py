from model_utils.constant import ANYTHING_NAME # type: ignore

PICK_UP_INST = lambda obj: f"pick up {obj}"
COT_PROMPT = lambda prompt: f"In: What action should the robot take to {prompt}?\nOut: "
PICK_UP_COT_PROMPT = lambda obj: COT_PROMPT(PICK_UP_INST(obj))
GROUNDING_COT_PROMPT = lambda obj: f"In: What is the bounding box of {obj} in the images?\nOut: "

from typing import Any
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, ForwardPass, ModEvent

from max.driver import Tensor

@mod
def adjust_logits(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod makes the model not produce an em dash.

    Simple usage of adjust_logits action
    """
    if isinstance(event, ForwardPass):
        # Convert logits from a tensor to numpy array - In the future this wont be necessary
        logits = event.logits.to_numpy()
        em_dash_id = tokenizer.encode("—", add_special_tokens=False)
        logits[em_dash_id] = -1e9
        return action.adjust_logits(Tensor.from_numpy(logits))
    return action.noop()

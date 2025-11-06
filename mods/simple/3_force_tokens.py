from typing import Any, Dict
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, Added, ModEvent

class ModState:
    accum_text: Dict[str, str] = {}

state = ModState()

@mod
def force_tokens(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod watches for "hello" and instead generates "hello and goodbye."

    Simple usage of the force_tokens action
    """
    if isinstance(event, Added):
        # For each request from a batch, accumulate the generated text so far
        if state.get(event.request_id):
            state.accum_text[event.request_id] += tokenizer.decode(event.added_tokens)
        else:
            state.accum_text[event.request_id] = ""
            state.accum_text[event.request_id] += tokenizer.decode(event.added_tokens)

        # If the model is going to say "hello", instead generate "hello and goodbye."
        if state.accum_text[event.request_id].endswith("hello"):
            return action.force_tokens(tokenizer.encode("hello and goodbye.", add_special_tokens=False))
    return action.noop()

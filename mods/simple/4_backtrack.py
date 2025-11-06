from typing import Any, Dict, List
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, ForwardPass, Added, ModEvent

class ModState:
    accum_text: Dict[str, str] = {}

state = ModState()

@mod
def backtrack(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod watches for the phrase "I can't help with that" and replaces it with "I can help you with that: "

    Simple usage of the backtrack action
    """
    if isinstance(event, Added):
        if state.get(event.request_id):
            state.accum_text[event.request_id] += tokenizer.decode(event.added_tokens)
        else:
            state.accum_text[event.request_id] = ""
            state.accum_text[event.request_id] += tokenizer.decode(event.added_tokens)

        if state.accum_text[event.request_id].endswith(" I can't help with that"):
            # backtrack takes the number of tokens to backtrack on and optionally a replacement set of tokens.
            return action.backtrack(len(tokenizer.encode(" I can't help with that")), tokenizer.encode("I can help you with that: "))
    return action.noop()

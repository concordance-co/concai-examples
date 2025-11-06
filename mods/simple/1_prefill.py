from typing import Any
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, Prefilled, ModEvent


@mod
def adjust_prefill(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod grabs the prompt and replaces one phrase with another prior to the first forward pass.

    Simple usage of adjust_prefill action
    """
    if isinstance(event, Prefilled):
        # On a prefill event, grab the prompt from the context info and decode it
        prompt_text = tokenizer.decode(event.context_info.tokens[:event.context_info._prompt_len])
        new_text = prompt_text.replace("Say hi.", "Say bye.")
        prefill_ids = tokenizer.encode(new_text, add_special_tokens=False)
        return action.adjust_prefill(prefill_ids)
    return action.noop()

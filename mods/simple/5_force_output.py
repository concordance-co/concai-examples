from typing import Any
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, Prefilled, ModEvent, get_conversation

@mod
def force_output(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod looks for simple conversational patterns like "hi" and "thanks" and skips any forward passes for responding to them.
    """
    if isinstance(event, Prefilled):
        convo = get_conversation()
        if len(convo) == 1:
            content = convo[0]["content"].lower()
            is_hello = content == "hi" or content == "hello" or content == "hello there"
            if is_hello:
                return action.force_output(tokenizer.encode("Hi! How can I help you today?"))
        else:
            content = convo[len(convo) - 1]["content"].lower()
            if convo[len(convo) - 1]["role"] == "user":
                is_thanks = content == "thank you" or content == "thanks"
                if is_thanks:
                    return action.force_output(tokenizer.encode("You're welcome."))
    return action.noop()

from typing import Any
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, Prefilled, ModEvent, get_conversation

@mod
def tool_calls(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod takes a simple trigger from the user and produces a tool call
    """
    if isinstance(event, Prefilled):
        convo = get_conversation()
        content = convo[len(convo) - 1]["content"].lower()
        if convo[len(convo) - 1]["role"] == "user":
            triggered = content == "<tool_call>call_search()</tool_call>"
            if triggered:
                payload = {
                    "id": f"call_{event.request_id.split('-')[0]}",
                    "type": "function",
                    "function": {"name": "call_search" },
                }
                return action.tool_calls(payload)
    return action.noop()

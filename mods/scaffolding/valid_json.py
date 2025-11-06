from typing import Any, Dict, List, Tuple, Optional
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, ForwardPass, Added, ModEvent

import json
from json import JSONDecodeError
import re

pattern = re.compile(r"```json[^\n]*\n([\s\S]*?)```", re.IGNORECASE)

class PathState:
    accum_text: str = ""
    accum_token_ids: list[int] = []
    accum_token_strs: list[str] = []
    reject_id: int | None =  None

state: Dict[str, PathState] = {}

@mod
def valid_json_only(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod watches for a json codeblock and validates it. If the LLM produced invalid json, backtrack to the error point and regenerate
    from that point forward.

    Note: in general, just-in-time constrained generation is probably better here, but if the schema is unknown this may be useful.
    """
    if not state.get(event.request_id):
        state[event.request_id] = PathState()

    if isinstance(event, ForwardPass):
        # If we backtracked, and are trying again, choose a different path by masking off the logit for the chosen token
        if state[event.request_id].reject_id:
            logits = event.logits.to_numpy()
            logits[state[event.request_id].reject_id] = -1e9
            state[event.request_id].reject_id = None
            return action.adjust_logits(logits)
    if isinstance(event, Added):
        state[event.request_id].accum_token_ids.extend(event.added_tokens)
        for i in event.added_tokens:
            tok_as_str = tokenizer.decode(i)
            state[event.request_id].accum_token_strs.append(tok_as_str)
            state[event.request_id].accum_text += tok_as_str

        # find all complete json blocks
        m = re.search(pattern, state[event.request_id].accum_text)
        if m:
            # get start and end of block and pass that to the locate error function
            block_start, block_end = m.span(0)
            (relative_tok_idx, res) = locate_json_error_token(state[event.request_id].accum_token_strs[block_start, block_end])
            if relative_tok_idx:
                err_idx = block_start + relative_tok_idx
                n_backtrack = len(state[event.request_id].accum_token_strs) - err_idx
                # set the reject id to be the error generating token.
                state[event.request_id].reject_id = state[event.request_id].accum_token_ids[err_idx]
                state[event.request_id].accum_token_ids[:err_idx]
                state[event.request_id].accum_token_strs[:err_idx]
                state[event.request_id].accum_text[:err_idx]
                # remove all tokens from the err_idx onward
                return action.backtrack(n_backtrack)
    return action.noop()


def locate_json_error_token(token_strs: List[str]) -> Tuple[Optional[int], dict]:
    """
    tokens: list of decoded text fragments (each str). If you have byte tokens,
            decode each with UTF-8 BEFORE passing in.
    Returns:
      (token_index, info)
      where token_index is the index of the token that contains the error position (or None)
      and info has parser-provided details: {"pos", "lineno", "colno", "message"}.
    """
    text = "".join(token_strs)
    try:
        json.loads(text)
        return None, {"pos": None, "lineno": None, "colno": None, "message": "valid JSON"}
    except JSONDecodeError as e:
        pos = e.pos        # 0-based character index in `text`
        lineno = e.lineno  # 1-based line number
        colno = e.colno    # 1-based column number

        # Map char offset -> token index
        cum = 0
        token_index = None
        in_token_char_offset = None
        for i, t in enumerate(token_strs):
            next_cum = cum + len(t)  # character length of this decoded piece
            if pos < next_cum:
                token_index = i
                in_token_char_offset = pos - cum
                break
            cum = next_cum

        return token_index, {
            "pos": pos,
            "lineno": lineno,
            "colno": colno,
            "in_token_char_offset": in_token_char_offset,
            "message": e.msg,
        }

from typing import Any, Dict, Iterable, Optional
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, Prefilled, ForwardPass, Added, ModEvent
from quote_mod_sdk.self_prompt import SelfPrompt
from quote_mod_sdk.strategies.strategy_constructor import UntilStrat, UntilEndType

import numpy as np
import math


clarify_prompt = SelfPrompt(
    prompt={"text", " I need more information from the user. I should only ask about that. I should wrap my question in XML (starting with <question_to_user>) so the client-side chat can process it so I should say (I must close the question tag with </question_to_user> when done):"},
    strategy=UntilStrat("<question_to_user>", UntilEndType.TAG, "</question_to_user>"),
)

class State:
    curr_logits: np.darray | None = None
    sequence_probabilities: list[float] = []
    clarify: SelfPrompt | None = None

state: Dict[str, State] = {}

@mod
def human_in_loop(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    """
    This mod calculates a measure of confidence over a sequence of tokens. If that measure of confidence
    drops too low, that kicks off a SelfPrompt that asks the user a clarify question.
    """
    if state.get(event.request_id):
        req_state = state[event.request_id]
    else:
        state[event.request_id] = State()
        req_state = state[event.request_id]

    # if we have a self prompt defined, step it forward until answered
    if req_state.clarify:
        answer_tokens = req_state.clarify.answer_tokens(event.request_id)
        if answer_tokens:
            # strip out the tags
            answer_tokens = answer_tokens[len("<question_to_user>"):-1*len("</question_to_user>")]
            return action.force_output(answer_tokens)
        elif isinstance(event, Prefilled):
            return req_state.clarify.handle_prefilled(event)
        elif isinstance(event, ForwardPass):
            return req_state.clarify.handle_forward_pass(event)
        elif isinstance(event, Added):
            return req_state.clarify.handle_added(event)
        else:
            return action.noop()

    if isinstance(event, ForwardPass):
        logits = event.logits.to_numpy()
        req_state.curr_logits = logits
    if isinstance(event, Added):
        if event.forced:
            req_state.sequence_probabilities.append(1.0)
        elif len(event.added_tokens) > 1:
            for i in event.added_tokens:
                req_state.sequence_probabilities.append(1.0)
        else:
            assert req_state.curr_logits, "No logits"
            req_state.sequence_probabilities.append(selected_token_prob(req_state.curr_logits, event.added_tokens[0]))

        if len(req_state.sequence) > 100:
            # any time after 100 tokens, calculate the sequence confidence by checking the geometric mean
            conf = sequence_confidence(req_state.sequence)
            if conf and conf < 0.5:
                req_state.clarify = clarify_prompt
                action.force_tokens(tokenizer.encode(" - Wait, I am uncertain about something."))
    return action.noop()

def sequence_confidence(token_probs: Iterable[float]) -> Optional[float]:
    """
    Geometric mean over provided token probabilities.
    Returns None if the iterator is empty.
    """
    vals = [float(x) for x in token_probs if x is not None]
    if not vals:
        return None
    logs = [math.log(max(min(v, 1.0), 1e-32)) for v in vals]
    return float(math.exp(sum(logs) / len(logs)))


def logsumexp(x: np.ndarray) -> float:
    """
    Stable log-sum-exp for 1D arrays.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return float("-inf")
    m = float(np.max(arr))
    # Avoid overflow in exp; if m is -inf handle gracefully
    if not math.isfinite(m):
        return m
    s = float(np.sum(np.exp(arr - m)))
    return float(m + math.log(s))


def selected_token_prob(logits_row: np.ndarray, token_id: int) -> float:
    """
    Compute p(token_id) from the raw model logits row via softmax.

    Returns a Python float in [0,1]. Raises ValueError on invalid input.
    """
    row = np.asarray(logits_row, dtype=np.float64)
    if row.ndim != 1:
        row = row.reshape(-1)
    if token_id < 0 or token_id >= row.size:
        raise ValueError(f"token_id {token_id} out of bounds for logits row of size {row.size}")
    lse = logsumexp(row)
    if not math.isfinite(lse):  # all -inf case
        return 0.0
    val = float(math.exp(float(row[int(token_id)]) - lse))
    # Numerical guard
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val

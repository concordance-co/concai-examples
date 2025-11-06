from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, ForwardPass, Added
from enum import Enum, auto
from typing import Any

import math
import random
import numpy as np

def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64, copy=False)
    m = x.max()
    shifted = x - m
    lse = m + np.log(np.exp(shifted).sum())
    return x - lse

class Phase(Enum):
    OLD = auto()    # initial collection of the block (base logits, old tokens)
    NEW = auto()    # propose a new suffix from pivot m with sharpened sampler
    REV = auto()    # reverse-walk: score old suffix under NEW prefix at tau
    DECIDE = auto()
    DONE = auto()

class ReasoningWithSamplingState:
    def __init__(self, alpha: float = 4.0, block_size: int = 192, nmcmc: int = 6):
        self.block_size: int = block_size
        self.alpha: float = alpha
        self.tau: float = 1.0 / alpha
        self.nmcmc: int = nmcmc
        self.phase: Phase = Phase.OLD

        self.base_logits_old: list = []      # list[Tensor], length B
        self.old_tokens: list[int] = []      # list[int], length B

        self.iter_idx: int = 0               # 0..nmcmc-1
        self.pivot_m: int = 0                # random pivot in [0, B-1]
        self.suf_len: int = 0                # suffix length = B - m

        self.base_logits_new: list = []      # base logits along NEW prefix (len = suf_len)
        self.new_tokens: list[int] = []      # proposed tokens for suffix (len = suf_len)
        self.logp_new_suf: float = 0.0       # Σ log p_base(new_t | new_prefix), t∈suffix
        self.logq_fwd: float = 0.0           # Σ log q(new_t | old_prefix+new_so_far), t∈suffix

        self.logq_rev: float = 0.0           # Σ log q(old_t | new_prefix+old_so_far), t∈suffix
        self.base_logits_rev_last = None     # last base logits observed in REV step
        self._rev_pos: int = 0               # 0..suf_len-1 (position inside suffix)

        self._logp_old_suf: float = 0.0

    def start_first_iteration(self):
        self.iter_idx = 0
        self._start_iteration()

    def _start_iteration(self):
        self.pivot_m = random.randint(0, self.block_size - 1)  # uniform pivot
        self.suf_len = self.block_size - self.pivot_m
        self.base_logits_new.clear()
        self.new_tokens.clear()
        self.logp_new_suf = 0.0
        self.logq_fwd = 0.0
        self.logq_rev = 0.0
        self.base_logits_rev_last = None
        self._rev_pos = 0
        self._logp_old_suf = 0.0
        self.phase = Phase.NEW

    def _finish_iteration_and_maybe_continue(self, actions):
        self.iter_idx += 1
        if self.iter_idx < self.nmcmc:
            self._start_iteration()
            return actions.backtrack(self.suf_len)
        else:
            self.phase = Phase.DONE
            return actions.noop()

    def decide_and_apply(self, actions):
        m = self.pivot_m
        self._logp_old_suf = 0.0
        for t in range(m, self.block_size):
            tok = self.old_tokens[t]
            self._logp_old_suf += log_softmax(self.base_logits_old[t])[tok]

        logA = self.alpha * (self.logp_new_suf - self._logp_old_suf) + (self.logq_rev - self.logq_fwd)
        self.phase = Phase.DECIDE
        if np.random.binomial(1, min(1.0, math.exp(min(0.0, logA)))):
            self.old_tokens[m:] = list(self.new_tokens)
            self.base_logits_old[m:] = list(self.base_logits_new)
            return actions.backtrack(self.suf_len, self.new_tokens)
        else:
            return actions.noop()

RWS: dict[str, ReasoningWithSamplingState] = {}

@mod
def reasoning_with_sampling(event, actions: ActionBuilder, tokenizer: Any):
    """
    This mod implements the algorithm from "Reasoning with Sampling: Your Base Model is Smarter Than You Think". Effectively,
    it generates a sequence, backtracks, generates a new sequence with a different temperature, then decides which to keep.
    """
    state = RWS.get(event.request_id)
    if state is None:
        RWS[event.request_id] = ReasoningWithSamplingState()
        state = RWS[event.request_id]

    if isinstance(event, ForwardPass):
        logits = event.logits
        if state.phase == Phase.OLD:
            state.base_logits_old.append(logits.to_numpy())
        if state.phase == Phase.NEW:
            state.base_logits_new.append(logits.to_numpy())
            return actions.adjusted_logits(logits / state.tau)
        if state.phase == Phase.REV:
            state.base_logits_rev_last = logits.to_numpy()
        return actions.noop()

    if isinstance(event, Added):
        if len(event.tokens) > 1 and event.forced and state.phase != Phase.DECIDE:
            return actions.noop()
        if state.phase == Phase.OLD:
            tok = event.added_tokens[0]
            state.old_tokens.append(tok)
            if len(state.old_tokens) == state.block_size:
                state.start_first_iteration()
                return actions.backtrack(state.suf_len)
            return actions.noop()

        if state.phase == Phase.NEW:
            tok = event.added_tokens[0]
            state.new_tokens.append(tok)
            i = len(state.new_tokens) - 1
            state.logp_new_suf += log_softmax(state.base_logits_new[i])[tok]
            state.logq_fwd += log_softmax(state.base_logits_new[i] / state.tau)[tok]
            if len(state.new_tokens) == state.suf_len:
                state.phase = Phase.REV
                state._rev_pos = 0
                return actions.backtrack(state.suf_len)
            return actions.noop()

        if state.phase == Phase.REV:
            i = state._rev_pos
            if i < state.suf_len:
                old_tok = state.old_tokens[state.pivot_m + i]
                logits_here = state.base_logits_rev_last
                state.logq_rev += log_softmax(logits_here / state.tau)[old_tok]
                state._rev_pos += 1
                if state._rev_pos == state.suf_len:
                    state.phase = Phase.DECIDE
                    return state.decide_and_apply(actions)
                else:
                    return actions.force_tokens([old_tok])
            return actions.noop()

        if state.phase == Phase.DECIDE:
            return state._finish_iteration_and_maybe_continue(actions)
    return actions.noop()

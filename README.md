# Concordance ConcAI Mod Examples

This repository contains a set of small, focused examples that demonstrate how to write and reason about “mods” against the Quote Mods SDK. A mod is a Python function decorated with `@mod` that observes generation events (e.g., prefill, logits, token additions) and returns actions that steer the model in real time (e.g., edit the prefill, adjust logits, force tokens, backtrack, or short-circuit output).

The goal is to give you a library of patterns you can reuse:
- Simple one-off behaviors (mask a token, replace a phrase, short-circuit trivial chat turns)
- Just-in-time editing (force or backtrack tokens)
- Tool triggering
- Scaffolding for more advanced control loops (self prompts, structured output guards, “reasoning with sampling”)

These examples are intentionally small and didactic. They emphasize the control pattern over production hardening.

---

## Table of contents

- What is a “mod”?
- Events, actions, and execution model
- Repository layout
- Simple examples
- Scaffolding examples
- Requirements and compatibility
- Using these mods in your host
- Extending and composing mods
- Debugging tips
- Caveats and notes
- License

---

## What is a “mod”?

A mod is a small Python function that:
- Is decorated with `@mod` from `quote_mod_sdk`.
- Receives a `ModEvent` (one of several event types fired during generation).
- Returns an action via an `ActionBuilder` to influence the generation process.

You’ll see the same shape in every example:

```python
from typing import Any
from quote_mod_sdk.mod import ActionBuilder
from quote_mod_sdk import mod, ModEvent

@mod
def my_mod(event: ModEvent, action: ActionBuilder, tokenizer: Any):
    # inspect event and return an action
    return action.noop()
```

A host environment (not included here) collects these functions and runs them during generation.

---

## Events, actions, and execution model

The examples use the following key concepts:

Events (observations you receive):
- `Prefilled`: Fired before the first forward pass. You have access to the initial prompt (input tokens).
- `ForwardPass`: Fired with the model’s raw logits before sampling a next token.
- `Added`: Fired after a token (or tokens) has been added to the sequence (either sampled or forced).

Actions (responses you can return):
- `adjust_prefill(new_token_ids)`: Replace the input prefill tokens before the first pass.
- `adjust_logits(new_logits)`: Modify next-token probabilities. Common uses include masking or biasing.
- `force_tokens(token_ids)`: Directly force specific next tokens, skipping sampling for them.
- `backtrack(n_to_remove, replacement_token_ids=None)`: Remove the most recent tokens and (optionally) replace them with a new sequence.
- `force_output(token_ids)`: Skip all forward passes and finalize output immediately.
- `tool_calls(payload)`: Emit a tool call payload (structured data) instead of textual output.
- `noop()`: Do nothing for the current event.

Other useful utilities visible in examples:
- `get_conversation()`: Read the current chat turns (role/content pairs).
- `tokenizer`: Encode/decode tokens to and from strings.

Notes:
- The host decides ordering and combination when multiple mods are active. Your mod should be side-effect free except for its own minimal state.
- Each `event.request_id` identifies the active request. If you keep per-request state, key on this ID.

---

## Repository layout

- `mods/`
  - `simple/`
    - `1_prefill.py` — Replace a phrase in the prefill before any forward pass.
    - `2_logits.py` — Mask a specific token (the em dash) via `adjust_logits`.
    - `3_force_tokens.py` — Watch for a trigger and replace it with a longer phrase via `force_tokens`.
    - `4_backtrack.py` — Detect an undesirable phrase and replace it via `backtrack`.
    - `5_force_output.py` — Short-circuit trivial chat turns (e.g., “hi”, “thanks”) using `force_output`.
    - `6_tool_calls.py` — Emit a tool call when a simple trigger is detected.
  - `scaffolding/`
    - `human_in_loop.py` — Track token confidence and, when confidence drops, initiate a clarifying self-prompt.
    - `reasoning_with_sampling.py` — Demonstrate a “reasoning with sampling” control loop that proposes and scores suffixes at different temperatures, optionally replacing previous output.
    - `valid_json.py` — Guardrails for JSON-in-codeblock output: detect invalid JSON, backtrack to the error, and try again.
  - `agent/` — Placeholder folder for agent-style multi-step scaffolding.

Top-level:
- `LICENSE` — MIT License.
- `README.md` — You are here.

---

## Simple examples

All simple examples share the same mod shape and differ only by which event they watch and which action they return.

- `1_prefill.py`
  - Watches: `Prefilled`
  - Pattern: Modify the textual prefill (e.g., replace “Say hi.” with “Say bye.”) and `adjust_prefill` with re-encoded tokens.

- `2_logits.py`
  - Watches: `ForwardPass`
  - Pattern: Convert logits to an array, set an undesired token’s logit very low (effectively masking), then `adjust_logits`.

- `3_force_tokens.py`
  - Watches: `Added`
  - Pattern: Accumulate decoded text and, on a suffix trigger (`...hello`), `force_tokens` with “hello and goodbye.”.

- `4_backtrack.py`
  - Watches: `Added`
  - Pattern: If the last tokens form an undesirable phrase (e.g., “I can’t help with that”), compute how many tokens to remove and `backtrack`, optionally replacing with a friendlier phrase.

- `5_force_output.py`
  - Watches: `Prefilled`
  - Pattern: Inspect the conversation summary and, for trivial user turns (“hi”, “thanks”), `force_output` with canned replies.

- `6_tool_calls.py`
  - Watches: `Prefilled`
  - Pattern: Trigger a tool call payload when the user sends a special tokenized sequence. Emits `tool_calls({...})` instead of text.

---

## Scaffolding examples

These examples show multi-step scaffolding. They are “skeletons” and may require small edits for your version of the SDK/runtime.

- `human_in_loop.py`
  - Idea: Maintain a running confidence over generated tokens (e.g., via probabilities from logits). If confidence drops below a threshold after some length, trigger a clarifying self-prompt. The self-prompt emits a tagged question (e.g., `<question_to_user>...</question_to_user>`) that your client can detect and route back to the user.
  - Key pieces:
    - Keep current logits from `ForwardPass`.
    - Convert logits to probabilities for the selected token (softmax on a 1D row).
    - Maintain a per-request state object (sequence probabilities, optional `SelfPrompt` instance).
    - When triggered, handle the self-prompt life cycle across events until an answer is available, then `force_output` the result.

- `reasoning_with_sampling.py`
  - Idea: Implement the high-level loop from “Reasoning with Sampling: Your Base Model is Smarter Than You Think”. Roughly:
    - Generate a block (OLD).
    - Pick a pivot m and propose a NEW suffix with a sharpened sampler (temperature τ).
    - Reverse-walk to compute proposal probabilities.
    - Decide (Metropolis–Hastings-style) whether to keep the NEW or revert to OLD.
    - Backtrack to apply the accepted suffix; repeat for several iterations.
  - This example wires phases and per-request state to orchestrate `ForwardPass` and `Added` events, switching temperatures and invoking `backtrack` or `force_tokens` as needed.

- `valid_json.py`
  - Idea: When the model writes a fenced JSON code block (```json ... ```), stream-validate it. If invalid, locate the token position of the error, backtrack to just before it, and sample a different continuation (optionally masking the previous wrong token).
  - Key pieces:
    - Regex to extract fenced JSON blocks.
    - Incremental accumulation of token IDs and their decoded fragments.
    - Mapping JSON parse errors (character offset) back to a token index.
    - `backtrack(n)` to remove the bad tail and try again; `adjust_logits` to avoid the previously chosen token on retry.

Important: These scaffolding files are illustrative. Expect to adapt signatures and fix small type/attribute mismatches to align with your SDK/runtime version.


---

### Warning: Scaffolding examples are conceptual and may not work.

---

## License

MIT © 2025 Concordance. See `LICENSE` for details.

"""Format trajectory JSONL into per-step chat-template samples for SFT.

Each step in a trajectory becomes one training sample:
  system  -> task instructions
  user    -> situation + prior_state + prior_calls
  assistant -> <think>{thinking}</think>{json.dumps(tool_call)}

The assistant content is what the model is trained to produce. We use the
tokenizer's chat template at training time to wrap these.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_SYSTEM = (
    "You are an agent that operates a support-ticket system through its JSON-RPC API."
    " For each prompt, produce <think>…</think> reasoning grounded in the prior"
    " state, then emit a single JSON-RPC request object on the line immediately"
    " after </think>. The request must validate against the method's params schema."
)


def _user_content(step: dict[str, Any]) -> str:
    ctx = step["prompt_context"]
    parts = [f"Prior state: {ctx['prior_state']}"]
    if ctx.get("situation"):
        parts.append(f"Situation: {ctx['situation']}")
    if ctx.get("prior_calls"):
        parts.append(f"Prior calls: {json.dumps(ctx['prior_calls'])}")
    return "\n".join(parts)


def _assistant_content(step: dict[str, Any]) -> str:
    return f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"


def format_step_as_chat_messages(
    step: dict[str, Any],
) -> list[dict[str, str]]:
    """Return [{system}, {user}, {assistant}] for one trajectory step."""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_content(step)},
        {"role": "assistant", "content": _assistant_content(step)},
    ]


def iter_step_samples(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Yield per-step samples from a trajectory JSONL file.

    Streams line-by-line — does not materialize the whole file in memory.
    """
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            for step in traj["steps"]:
                yield {
                    "trajectory_id": traj["trajectory_id"],
                    "messages": format_step_as_chat_messages(step),
                }

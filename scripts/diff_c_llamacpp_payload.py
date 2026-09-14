"""Differential: C `llamacpp-payload` request shape vs the Python eval prompt.

The two-stage llama.cpp eval sends the model a prompt built from
`toyforge.train.data.format_step_as_chat_messages` (system `_SYSTEM` + user
`_user_content`) plus two fixed cues; stage 2 also carries an assistant
`<think>…</think>` and the GBNF grammar. That prompt is *what drives model
behavior*, yet no gate pinned the C eval payload's message content to Python —
the verifier/seeds gates check the assistant OUTPUT, and the live-HTTP mock
check distinguishes stage/grammar/temperature but not the prompt text.

This gate renders the C `llamacpp-payload` command over several synthetic
1-step trajectories — including the `prior_calls` path (rendered via
`json.dumps`, which no real seed exercises) and a Unicode situation — and
asserts, for both stage payloads, that the messages (role + content), model,
max_tokens, temperature, and embedded grammar match the Python oracle.

Byte-formatting of the payload JSON is intentionally NOT compared (the C
command pretty-prints; Python's wire payload is `json.dumps` — a llama.cpp
server parses either). Only the semantic request content is pinned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.train.data import _SYSTEM, _user_content

SCHEMAS_DIR = Path("schemas")
GRAMMAR_PATH = SCHEMAS_DIR / "jsonrpc.gbnf"

# Cues mirror toyforge.llamacpp.render_step_outputs_llamacpp (inline literals).
THINK_CUE = "Write only the reasoning text that belongs inside <think>."
CALL_CUE = "Now emit only the JSON-RPC request object."

MODEL = "qwen"
MAX_TOKENS = 64
TEMPERATURE = 0.0

# Synthetic 1-step trajectories spanning the user-prompt branches:
# minimal, no-situation, prior_calls (json.dumps path), and Unicode.
_TRAJECTORIES: list[dict] = [
    {
        "trajectory_id": "synth_minimal",
        "initial_state": "NEW",
        "difficulty": "easy",
        "source": "hand_seed",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW", "situation": "fixture"},
                "thinking": "Opening from NEW emits ticket_open.accepted.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_open",
                    "params": {
                        "requester_id": "user:c",
                        "body_text": "Yw==",
                        "lifecycle_profile": "std",
                    },
                    "id": 1,
                },
                "expected_trigger": "ticket_open.accepted",
                "expected_state_after": "TRIAGED",
            }
        ],
        "final_state": "TRIAGED",
    },
    {
        "trajectory_id": "synth_no_situation",
        "initial_state": "NEW",
        "difficulty": "easy",
        "source": "hand_seed",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW"},
                "thinking": "No situation provided.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_open",
                    "params": {
                        "requester_id": "user:c",
                        "body_text": "Yw==",
                        "lifecycle_profile": "std",
                    },
                    "id": 1,
                },
                "expected_trigger": "ticket_open.accepted",
                "expected_state_after": "TRIAGED",
            }
        ],
        "final_state": "TRIAGED",
    },
    {
        "trajectory_id": "synth_prior_calls_unicode",
        "initial_state": "TRIAGED",
        "difficulty": "medium",
        "source": "hand_seed",
        "steps": [
            {
                "prompt_context": {
                    "prior_state": "TRIAGED",
                    "situation": "two prior calls — détails ünïcode",
                    "prior_calls": [
                        {
                            "method": "ticket_open",
                            "params": {"requester_id": "user:a", "lifecycle_profile": "std"},
                        },
                        {"method": "ticket_status", "params": {}},
                    ],
                },
                "thinking": "Status check after opening.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_status",
                    "params": {"ticket_id": "T-1"},
                    "id": 2,
                },
                "expected_trigger": "",
                "expected_state_after": "TRIAGED",
            }
        ],
        "final_state": "TRIAGED",
    },
]


def _expected(step: dict, grammar: str) -> tuple[list[dict], list[dict]]:
    prompt = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_content(step)},
    ]
    stage1 = [*prompt, {"role": "user", "content": THINK_CUE}]
    stage2 = [
        *prompt,
        {"role": "assistant", "content": f"<think>{step['thinking']}</think>"},
        {"role": "user", "content": CALL_CUE},
    ]
    return stage1, stage2


def _check_payload(
    label: str, got: dict, want_messages: list[dict], want_grammar: str | None
) -> str:
    if got.get("model") != MODEL:
        return f"{label}: model {got.get('model')!r} != {MODEL!r}"
    if int(got.get("max_tokens", -1)) != MAX_TOKENS:
        return f"{label}: max_tokens {got.get('max_tokens')} != {MAX_TOKENS}"
    if float(got.get("temperature", -1)) != TEMPERATURE:
        return f"{label}: temperature {got.get('temperature')} != {TEMPERATURE}"
    if got.get("messages") != want_messages:
        return f"{label}: messages mismatch\n  C  ={got.get('messages')}\n  want={want_messages}"
    if want_grammar is None:
        if "grammar" in got:
            return f"{label}: unexpected grammar field on stage-1 payload"
    else:
        if got.get("grammar") != want_grammar:
            return f"{label}: grammar field differs from {GRAMMAR_PATH}"
    return ""


def run_one(toyforge_c: Path, traj: dict, grammar: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "train.jsonl"
        data_path.write_text(json.dumps(traj) + "\n")
        prefix = Path(tmp) / "pl"
        subprocess.run(
            [
                str(toyforge_c), "llamacpp-payload",
                "--data-path", str(data_path),
                "--grammar-path", str(GRAMMAR_PATH),
                "--out-prefix", str(prefix),
                "--model", MODEL,
                "--max-tokens", str(MAX_TOKENS),
                "--temperature", str(TEMPERATURE),
            ],
            check=True, text=True, capture_output=True,
        )  # fmt: skip
        think = json.loads(Path(f"{prefix}.think.json").read_text())
        call = json.loads(Path(f"{prefix}.call.json").read_text())

    stage1, stage2 = _expected(traj["steps"][0], grammar)
    tid = traj["trajectory_id"]
    err = _check_payload(f"{tid} stage-1", think, stage1, None)
    if err:
        return err
    return _check_payload(f"{tid} stage-2", call, stage2, grammar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    grammar = GRAMMAR_PATH.read_text()
    for traj in _TRAJECTORIES:
        err = run_one(args.toyforge_c, traj, grammar)
        if err:
            print(err)
            return 1
    print(
        f"llamacpp-payload parity: {len(_TRAJECTORIES)} trajectories (incl. prior_calls + "
        "unicode) — stage-1/stage-2 messages, fields, and grammar match the Python eval prompt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

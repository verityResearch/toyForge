"""Differential: C `teacher-request` vs Python's teacher request construction.

The C command must build the exact OpenAI-compatible chat request the Python
teacher would send for a seed expansion. The critical parity anchor is the
system prompt: `provenance.system_prompt_sha256` fingerprints it, so the C
constant must be byte-identical to `expand._SYSTEM_PROMPT`. The user prompt is
also compared byte-for-byte (the seed file is written as `json.dumps(indent=2)`,
matching `_build_user_prompt`), along with the model / max_tokens / temperature
fields.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.scenario_gen.expand import _SYSTEM_PROMPT, _build_user_prompt

# A representative seed trajectory (shape matches scenarios/seeds.yaml entries).
SEED = {
    "trajectory_id": "seed_diff_request",
    "initial_state": "ESCALATED",
    "steps": [
        {
            "prompt_context": {
                "prior_state": "ESCALATED",
                "situation": "T-7 degraded on agent-4",
            },
            "thinking": "ESCALATED -> admin_escalate emits escalation.start.",
            "tool_call": {
                "jsonrpc": "2.0",
                "method": "admin_escalate",
                "params": {"agent_id": "agent-4", "ticket_id": "T-7"},
                "id": 1,
            },
            "expected_trigger": "escalation.start",
            "expected_state_after": "ENGINEERING",
        }
    ],
    "final_state": "ENGINEERING",
}

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
TEMPERATURE = 0.3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        seed_path = tmp / "seed.json"
        req_path = tmp / "request.json"
        # Write the seed exactly as Python's _build_user_prompt serialises it.
        seed_path.write_text(json.dumps(SEED, indent=2))

        proc = subprocess.run(
            [
                str(args.toyforge_c),
                "teacher-request",
                "--seed-path",
                str(seed_path),
                "--model",
                MODEL,
                "--max-tokens",
                str(MAX_TOKENS),
                "--temperature",
                str(TEMPERATURE),
                "--out-path",
                str(req_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"C teacher-request failed: {proc.stderr or proc.stdout}")
            return 1

        req = json.loads(req_path.read_text())

        if req.get("model") != MODEL:
            failures.append(f"model: C={req.get('model')!r} expected={MODEL!r}")
        if req.get("max_tokens") != MAX_TOKENS:
            failures.append(f"max_tokens: C={req.get('max_tokens')!r} expected={MAX_TOKENS}")
        if abs(float(req.get("temperature", -1)) - TEMPERATURE) > 1e-9:
            failures.append(f"temperature: C={req.get('temperature')!r} expected={TEMPERATURE}")

        messages = req.get("messages", [])
        if len(messages) != 2:
            failures.append(f"messages: expected 2, got {len(messages)}")
        else:
            system_msg = {"role": "system", "content": _SYSTEM_PROMPT}
            user_msg = {"role": "user", "content": _build_user_prompt(SEED)}
            if messages[0] != system_msg:
                # Pinpoint the system-prompt divergence (the provenance anchor).
                c_sys = messages[0].get("content", "")
                if messages[0].get("role") != "system":
                    failures.append(f"messages[0].role: {messages[0].get('role')!r}")
                if c_sys != _SYSTEM_PROMPT:
                    lens = f"(len C={len(c_sys)} py={len(_SYSTEM_PROMPT)})"
                    for idx, (a, b) in enumerate(zip(c_sys, _SYSTEM_PROMPT, strict=False)):
                        if a != b:
                            failures.append(
                                f"system prompt diverges at byte {idx}: C={a!r} Python={b!r} {lens}"
                            )
                            break
                    else:
                        failures.append(f"system prompt length differs {lens}")
            if messages[1] != user_msg:
                failures.append("user message does not match _build_user_prompt(seed)")

    if failures:
        print("\n".join(failures))
        return 1
    print("teacher-request parity: system prompt + user prompt + fields match Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

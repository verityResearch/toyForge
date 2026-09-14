"""Differential: C `verify-step --rubric-preset` reward vs Python compute_reward.

The GRPO reward is `sum(weight_i * subscore_i)` over the rubric preset's weights
(toyforge.verifier.reward.compute_reward). Per AGENTS.md the verifier is
"simultaneously the training reward and the eval grade", so the reward — and the
preset weights it sums, loaded independently from schemas/reward-rubric.yaml by
both the Python and C schema loaders — is a converted runtime surface.

The C subscores are already pinned by diff_c_verifier; this gate adds the two
factors that were only covered by hand-computed unit constants: the rubric
preset WEIGHT loading (C schemas.c vs Python rubric_from_preset) and the
weighted-sum arithmetic (tf_compute_reward vs compute_reward). It runs over
subscore-vector-diverse inputs × every preset and asserts the rewards match.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step
from toyforge.verifier.reward import compute_reward, rubric_from_preset

SCHEMAS_DIR = Path("schemas")

_VALID_OPEN = (
    '<think>t</think>{"jsonrpc":"2.0","method":"ticket_open","params":'
    '{"requester_id":"user:c","body_text":"Yw==","lifecycle_profile":"std"},"id":1}'
)
_SCHEMA_FAIL = '<think>t</think>{"jsonrpc":"2.0","method":"ticket_open","params":{},"id":1}'
_UNKNOWN_METHOD = '<think>t</think>{"jsonrpc":"2.0","method":"no_such_method","params":{},"id":1}'
_PARSE_FAIL = "no think tag, not json at all"
_QUERY = (
    '<think>t</think>{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"o1"},"id":2}'
)

# (label, prior_state, expected_trigger, expected_state_after, infer_trigger, output)
_CASES = [
    ("valid_no_trigger", "NEW", None, None, False, _VALID_OPEN),
    ("valid_infer", "NEW", None, None, True, _VALID_OPEN),
    ("valid_gold_trigger", "NEW", "ticket_open.accepted", "TRIAGED", False, _VALID_OPEN),
    ("schema_fail", "NEW", None, None, False, _SCHEMA_FAIL),
    ("unknown_method", "NEW", None, None, False, _UNKNOWN_METHOD),
    ("parse_fail", "NEW", None, None, False, _PARSE_FAIL),
    ("query_infer", "TRIAGED", None, None, True, _QUERY),
]


def _py_reward(schemas, case, preset: str) -> float:
    _label, prior, trig, after, infer, output = case
    ctx: dict = {"prior_state": prior}
    if trig is not None:
        ctx["expected_trigger"] = trig
    if after is not None:
        ctx["expected_state_after"] = after
    sr = verify_step(ctx, output, schemas, infer_trigger=infer)
    return compute_reward(sr.subscores, rubric_from_preset(schemas, preset))


def _c_reward(toyforge_c: Path, case, preset: str) -> float:
    _label, prior, trig, after, infer, output = case
    cmd = [
        str(toyforge_c), "verify-step",
        "--prior-state", prior,
        "--schemas-dir", str(SCHEMAS_DIR),
        "--rubric-preset", preset,
        "--output", output,
    ]  # fmt: skip
    if trig is not None:
        cmd += ["--expected-trigger", trig]
    if after is not None:
        cmd += ["--expected-state-after", after]
    if infer:
        cmd += ["--infer-trigger"]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    m = re.search(r"^reward=([-0-9.]+)$", proc.stdout, re.MULTILINE)
    if not m:
        raise RuntimeError(f"no reward in C output:\n{proc.stdout}\n{proc.stderr}")
    return float(m.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    schemas = load_schemas(SCHEMAS_DIR)
    presets = sorted(schemas.rubric_presets)
    checked = 0
    for case in _CASES:
        for preset in presets:
            py = _py_reward(schemas, case, preset)
            c = _c_reward(args.toyforge_c, case, preset)
            if abs(c - py) > 1e-6:
                print(f"reward mismatch case={case[0]} preset={preset}: C={c} Python={py}")
                return 1
            checked += 1

    print(
        f"reward parity: {checked} (case × preset) combinations match compute_reward "
        f"across presets {presets}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

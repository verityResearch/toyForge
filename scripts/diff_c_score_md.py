"""Differential: C `score-data` .md failed-trajectory section vs the Python oracle.

The compare renderer has its own gate (diff_c_compare); the single-run score-data
.md report did not have an ongoing C-vs-Python differential. Its most intricate
section is "## Failed trajectories" (failures grouped by initial_state, sorted,
with per-state tables and pipe-escaped errors). This runs C `score-data` on a
fixed mixed pass/fail corpus and asserts that section byte-matches
`toyforge.eval.runner._render_failed_trajectories` fed C's own per_trajectory
data — isolating the renderer with a shared input (the rest of the .md carries a
non-deterministic provenance block, so only this section is compared).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from toyforge.eval.runner import _render_failed_trajectories

ING = {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"}


def _call(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}


def _step(prior: str, method: str, params: dict[str, Any], trig=None, after=None) -> dict[str, Any]:
    s: dict[str, Any] = {
        "prompt_context": {"prior_state": prior, "situation": "s"},
        "thinking": "t",
        "tool_call": _call(method, params),
    }
    if trig is not None:
        s["expected_trigger"] = trig
    if after is not None:
        s["expected_state_after"] = after
    return s


# Mixed pass/fail across initial_states, a pipe in a field, several failure kinds.
CORPUS: list[dict[str, Any]] = [
    {"trajectory_id": "pass1", "initial_state": "NEW",
     "steps": [_step("NEW", "ticket_open", ING, "ticket_open.accepted", "TRIAGED")],
     "final_state": "TRIAGED"},
    {"trajectory_id": "schemafail_draft", "initial_state": "NEW",
     "steps": [_step("NEW", "ticket_open", {}, "ticket_open.accepted", "TRIAGED")],
     "final_state": "TRIAGED"},
    {"trajectory_id": "transfail_canon", "initial_state": "RESOLVED",
     "steps": [_step("RESOLVED", "ticket_status", {"ticket_id": "o|pipe"})],
     "final_state": "NEW"},
    {"trajectory_id": "badmethod_degraded", "initial_state": "ESCALATED",
     "steps": [_step("ESCALATED", "frobnicate", {}, "x", "Y")], "final_state": "Y"},
    {"trajectory_id": "pass2_canon", "initial_state": "RESOLVED",
     "steps": [_step("RESOLVED", "ticket_status", {"ticket_id": "o2"})],
     "final_state": "RESOLVED"},
]  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(t) + "\n" for t in CORPUS))
        prefix = Path(td) / "out"
        proc = subprocess.run(
            [
                str(args.toyforge_c), "score-data",
                "--data-path", str(corpus),
                "--schemas-dir", "schemas",
                "--grammar-path", "schemas/jsonrpc.gbnf",
                "--out-prefix", str(prefix),
                "--run-name", "score-md-diff",
            ],
            check=False, text=True, capture_output=True,
        )  # fmt: skip
        if proc.returncode not in (0, 1):
            print(f"C score-data failed: {proc.stderr or proc.stdout}")
            return 1
        scorecard = json.loads(Path(f"{prefix}.json").read_text())
        c_md = Path(f"{prefix}.md").read_text(encoding="utf-8")

    py_section = _render_failed_trajectories(scorecard["per_trajectory"]).strip()
    idx = c_md.find("## Failed trajectories")
    c_section = c_md[idx:].strip() if idx >= 0 else "<MISSING>"

    if c_section != py_section:
        cl, pl = c_section.splitlines(), py_section.splitlines()
        for i, (a, b) in enumerate(zip(cl, pl, strict=False)):
            if a != b:
                print(f"failed-trajectories section diverges at line {i}:\n  C ={a!r}\n  py={b!r}")
                return 1
        print(f"section length differs: C={len(cl)} py={len(pl)} lines")
        return 1
    print("score-data .md failed-trajectory section matches the Python renderer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

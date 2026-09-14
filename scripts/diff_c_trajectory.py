"""Trajectory-level differential: C `score-data` vs the Python verify_trajectory oracle.

The step-level harness (scripts/diff_c_verifier.py) covers single steps; this one
exercises multi-step state threading and the trajectory failure paths (mid-step
failure, final-state mismatch) that the all-passing real data never hits.

It builds a corpus of schema-valid trajectories — the real test trajectories plus
mutations that fail at the schema / transition / final-state level — runs them
through the C `score-data` command and the Python `verify_trajectory`, and asserts
the per-trajectory passed flag, first-failing step, and first-failing subscore agree.

Mutations keep the trajectory *schema* valid (real states, intact step shape) so
the comparison isolates verifier behavior; the trajectory schema does not validate
tool_call params against the per-method schemas, so a bad params object is still a
schema-valid trajectory whose step the verifier rejects.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from toyforge.scenario_gen.seeds import load_seeds
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

SUBSCORE_NAMES = [
    "parse",
    "schema",
    "method_known",
    "precondition_met",
    "transition_valid",
    "sequence_optimal",
]


def _outputs(traj: dict) -> list[str]:
    return [f"<think>{s['thinking']}</think>{json.dumps(s['tool_call'])}" for s in traj["steps"]]


def _py_first_failing(result) -> tuple[int | None, str | None]:
    """Mirror eval/score.py: first step with a failing subscore, by priority order."""
    for idx, step in enumerate(result.steps):
        sub = step.subscores.as_dict()
        failing = next((n for n in SUBSCORE_NAMES if sub[n] < 1.0), None)
        if failing is not None:
            return idx, failing
    return None, None


def build_corpus(base_rows: list[dict]) -> list[dict]:
    """Real trajectories (pass) plus per-state failure mutations, all schema-valid."""
    corpus: list[dict] = []

    # Pass-through: the real trajectories should all pass in both engines.
    for r in base_rows:
        t = copy.deepcopy(r)
        t["trajectory_id"] = f"pass__{r['trajectory_id']}"
        corpus.append(t)

    # One representative per initial_state for targeted mutations.
    by_state: dict[str, dict] = {}
    for r in base_rows:
        by_state.setdefault(r["initial_state"], r)

    for state, r in by_state.items():
        # Schema fail on step 0: empty params object (missing required fields).
        t = copy.deepcopy(r)
        t["trajectory_id"] = f"schemafail__{state}"
        t["steps"][0]["tool_call"]["params"] = {}
        corpus.append(t)

        # Transition fail: point step 0's expected_state_after at a wrong real state.
        if r["steps"][0].get("expected_trigger"):
            t = copy.deepcopy(r)
            t["trajectory_id"] = f"transitionfail__{state}"
            wrong = "ARCHIVED" if state != "ARCHIVED" else "NEW"
            t["steps"][0]["expected_state_after"] = wrong
            t["final_state"] = wrong
            corpus.append(t)

        # Final-state mismatch: steps unchanged but declared final_state is wrong.
        if r.get("final_state"):
            t = copy.deepcopy(r)
            t["trajectory_id"] = f"finalmismatch__{state}"
            t["final_state"] = "NEW" if r["final_state"] != "NEW" else "ARCHIVED"
            corpus.append(t)

    # Multi-step: fail a non-first step (schema fail on the last step) to exercise
    # state threading up to the failure point.
    for r in base_rows:
        if len(r["steps"]) > 1:
            t = copy.deepcopy(r)
            t["trajectory_id"] = f"midfail__{r['trajectory_id']}"
            t["steps"][-1]["tool_call"]["params"] = {}
            corpus.append(t)

    return corpus


def _run_c(toyforge_c: Path, corpus_path: Path) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "c-traj"
        cmd = [
            str(toyforge_c),
            "score-data",
            "--data-path",
            str(corpus_path),
            "--schemas-dir",
            "schemas",
            "--grammar-path",
            "schemas/jsonrpc.gbnf",
            "--out-prefix",
            str(prefix),
            "--run-name",
            "c-traj",
        ]
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode not in (0, 1):  # 1 == some trajectories failed (expected)
            raise RuntimeError(f"C score-data error: {proc.stderr or proc.stdout}")
        data = json.loads(Path(f"{prefix}.json").read_text())
    return {e["trajectory_id"]: e for e in data["per_trajectory"]}


def _edge_trajectories() -> list[dict[str, Any]]:
    """Hand-built composition edge cases (state threading, final-state checks,
    mid-step failure, query-only steps) confirmed against verify_trajectory by a
    differential fuzz. Empty-step trajectories are intentionally excluded: they
    violate the trajectory schema's `steps: minItems: 1`, so C's schema-first
    validator reports a `schema` failure while Python's verify_trajectory reports
    a generic empty-trajectory failure — both reject, only the diagnostic differs,
    and real data never has empty steps."""
    ing = {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"}

    def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}

    def step(prior, method, params, trig=None, after=None):
        s = {
            "prompt_context": {"prior_state": prior, "situation": "s"},
            "thinking": "t",
            "tool_call": call(method, params),
        }
        if trig is not None:
            s["expected_trigger"] = trig
        if after is not None:
            s["expected_state_after"] = after
        return s

    def traj(tid, init, steps, final=None):
        t = {"trajectory_id": tid, "initial_state": init, "steps": steps}
        if final is not None:
            t["final_state"] = final
        return t

    window = {"ticket_id": "o1", "window": {"from_utc": "2026-05-29T12:00:00Z"}}
    return [
        traj(
            "edge_single_query",
            "RESOLVED",
            [step("RESOLVED", "ticket_status", {"ticket_id": "o1"})],
            "RESOLVED",
        ),
        traj(
            "edge_two_query_thread",
            "RESOLVED",
            [
                step("RESOLVED", "ticket_status", {"ticket_id": "o1"}),
                step("RESOLVED", "ticket_history", window),
            ],
            "RESOLVED",
        ),
        traj(
            "edge_query_final_mismatch",
            "RESOLVED",
            [step("RESOLVED", "ticket_status", {"ticket_id": "o1"})],
            "NEW",
        ),
        traj(
            "edge_open_then_query",
            "NEW",
            [
                step("NEW", "ticket_open", ing, "ticket_open.accepted", "TRIAGED"),
                step("TRIAGED", "ticket_status", {"ticket_id": "o1"}),
            ],
            "TRIAGED",
        ),
        traj(
            "edge_open_final_mismatch",
            "NEW",
            [step("NEW", "ticket_open", ing, "ticket_open.accepted", "TRIAGED")],
            "RESOLVED",
        ),
        traj(
            "edge_mid_step_fail",
            "NEW",
            [
                step("NEW", "ticket_open", ing, "ticket_open.accepted", "TRIAGED"),
                step(
                    "TRIAGED",
                    "ticket_open",
                    {**ing, "requester_id": "bad"},
                    "ticket_open.accepted",
                    "TRIAGED",
                ),
            ],
            "TRIAGED",
        ),
        traj(
            "edge_no_final",
            "NEW",
            [step("NEW", "ticket_open", ing, "ticket_open.accepted", "TRIAGED")],
        ),
        traj(
            "edge_wrong_trigger_step0",
            "NEW",
            [step("NEW", "ticket_open", ing, "resolution_confirm.passed", "TRIAGED")],
            "TRIAGED",
        ),
        traj(
            "edge_escalation_arc",
            "ESCALATED",
            [
                step(
                    "ESCALATED",
                    "admin_escalate",
                    {"agent_id": "n1", "ticket_id": "o1"},
                    "escalation.start",
                    "ENGINEERING",
                )
            ],
            "ENGINEERING",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    # Base trajectories come from the checked-in hand seeds (which are exactly
    # what test.jsonl contains) so the corpus is reproducible without generated
    # data. Pass --data-path to source base rows from a JSONL file instead.
    parser.add_argument("--seeds-path", type=Path, default=Path("scenarios/seeds.yaml"))
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()

    schemas = load_schemas(Path("schemas"))
    if args.data_path is not None:
        base_rows = [
            json.loads(line) for line in args.data_path.read_text().splitlines() if line.strip()
        ]
    else:
        base_rows = load_seeds(args.seeds_path)
    corpus = build_corpus(base_rows) + _edge_trajectories()

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.jsonl"
        corpus_path.write_text("".join(json.dumps(t) + "\n" for t in corpus))
        c_by_id = _run_c(args.toyforge_c, corpus_path)

    failures: list[str] = []
    for traj in corpus:
        tid = traj["trajectory_id"]
        c = c_by_id.get(tid)
        if c is None:
            failures.append(f"{tid}: missing from C score-data output")
            continue
        py = verify_trajectory(traj, _outputs(traj), schemas)
        py_step, py_sub = _py_first_failing(py)
        c_step = c.get("first_failing_step")
        c_sub = c.get("first_failing_subscore") or None  # C uses "" / null for None

        if py.passed != c["passed"]:
            failures.append(f"{tid}: passed mismatch Python={py.passed} C={c['passed']}")
        if py_step != c_step:
            failures.append(f"{tid}: first_failing_step Python={py_step} C={c_step}")
        if py_sub != c_sub:
            failures.append(f"{tid}: first_failing_subscore Python={py_sub} C={c_sub}")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"{len(corpus)} C trajectory parity cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for run_eval resume support (Task 6.4) and adapter_dir=None (Task 7.5).

Covers:
- TrajectoryResult serialization round-trip.
- run_eval accepts adapter_dir=None (base-only eval, B0 baseline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# runner.py imports torch/peft at module-level; skip if not installed.
pytest.importorskip("torch")
pytest.importorskip("peft")

from toyforge.eval.runner import (
    _trajectory_result_from_dict,
    _trajectory_result_to_dict,
)
from toyforge.verifier.types import StepResult, Subscores, TrajectoryResult


def _make_traj(passed: bool = True) -> TrajectoryResult:
    step = StepResult(
        passed=passed,
        subscores=Subscores(
            parse=1.0,
            schema=1.0,
            method_known=1.0,
            precondition_met=1.0 if passed else 0.0,
            transition_valid=1.0 if passed else 0.0,
            sequence_optimal=1.0 if passed else 0.0,
        ),
        parsed_thinking="thinking",
        parsed_call={
            "jsonrpc": "2.0",
            "method": "ticket_status",
            "params": {"ticket_id": "x"},
            "id": 1,
        },
        method="ticket_status",
        new_state="TRIAGED" if passed else None,
        error_message=None if passed else "test failure",
    )
    return TrajectoryResult(
        passed=passed,
        steps=[step],
        final_state="TRIAGED" if passed else None,
        error_message=None if passed else "step 0 failed",
    )


def test_trajectory_result_round_trip():
    """to_dict/from_dict must produce an equivalent TrajectoryResult."""
    original = _make_traj(passed=True)
    d = _trajectory_result_to_dict(original)
    json_str = json.dumps(d)  # must be JSON-serializable
    reloaded_d = json.loads(json_str)
    reloaded = _trajectory_result_from_dict(reloaded_d)
    assert reloaded == original


def test_trajectory_result_round_trip_failed():
    """Round-trip preserves failure state."""
    original = _make_traj(passed=False)
    d = _trajectory_result_to_dict(original)
    reloaded = _trajectory_result_from_dict(d)
    assert reloaded == original
    assert reloaded.passed is False
    assert reloaded.error_message == "step 0 failed"
    assert reloaded.steps[0].error_message == "test failure"


def test_run_eval_accepts_adapter_dir_none(tmp_path):
    """When adapter_dir is None, run_eval skips PeftModel.from_pretrained
    and evaluates the base model directly.

    An empty test set means need_gpu=False, so no model is loaded —
    we exercise the signature + provenance path cheaply.
    """
    from toyforge.eval.runner import run_eval

    # Empty test set → _load_test returns [] → need_gpu=False → no GPU work
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("")

    out_dir = tmp_path / "reports"
    schemas_dir = Path(__file__).resolve().parents[2] / "schemas"

    sc = run_eval(
        adapter_dir=None,
        test_path=test_path,
        schemas_dir=schemas_dir,
        out_dir=out_dir,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        run_name="B0-test",
    )

    # Base-only path encodes adapter_dir as "(base-only)" in the scorecard
    assert sc.adapter_dir == "(base-only)"
    # Empty test set → zero-stats scorecard
    assert sc.n_examples == 0

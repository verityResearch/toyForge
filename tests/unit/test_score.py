"""Tests for eval.score.aggregate — pass@k math, failure-mode breakdown."""

from __future__ import annotations

from toyforge.eval.score import aggregate
from toyforge.verifier.types import StepResult, Subscores, TrajectoryResult


def _step(passed: bool, **overrides) -> StepResult:
    defaults = {
        "parse": 1.0,
        "schema": 1.0,
        "method_known": 1.0,
        "precondition_met": 1.0,
        "transition_valid": 1.0,
        "sequence_optimal": 1.0,
    }
    defaults.update(overrides)
    return StepResult(
        passed=passed,
        subscores=Subscores(**defaults),
        parsed_thinking=None,
        parsed_call=None,
        method=None,
        new_state=None,
        error_message=None,
    )


def _traj(steps: list[StepResult], passed: bool) -> TrajectoryResult:
    return TrajectoryResult(
        passed=passed,
        steps=steps,
        final_state=None,
        error_message=None,
    )


def test_pass_at_maj_votes_on_canonical_call_not_raw_string():
    """Regression: pass@maj used to vote on raw <think>...</think>{json} strings.

    When the correct call wins the call-vote (3/5) but a wrong call shares
    a common <think> text (2 identical raw strings), the buggy raw-string vote
    picks WRONG by majority. The fixed canonical-call vote picks the correct call.
    """
    correct = '{"jsonrpc":"2.0","method":"ticket_get","params":{"ticket_id":"ok"},"id":1}'
    wrong = '{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"x"},"id":1}'
    sampled_finals = [
        [
            f"<think>A unique</think>{correct}",
            f"<think>B unique</think>{correct}",
            f"<think>C unique</think>{correct}",
            f"<think>SHARED</think>{wrong}",
            f"<think>SHARED</think>{wrong}",
        ]
    ]
    sampled_results = [
        [
            _traj([_step(passed=True)], passed=True),
            _traj([_step(passed=True)], passed=True),
            _traj([_step(passed=True)], passed=True),
            _traj([_step(passed=False)], passed=False),
            _traj([_step(passed=False)], passed=False),
        ]
    ]
    greedy = [_traj([_step(passed=True)], passed=True)]
    sc = aggregate(
        "t",
        greedy_results=greedy,
        sampled_results=sampled_results,
        sampled_final_calls=sampled_finals,
        k=5,
    )
    # Correct call has 3 votes (canonical), wrong call has 2.
    # Majority winner is the correct call, which passed verification → pass_at_maj=1.0
    assert sc.pass_at_maj == 1.0, (
        f"expected pass_at_maj=1.0 (3/5 emit correct call), got {sc.pass_at_maj}"
    )


def test_failure_mode_finds_first_failing_step_not_just_step_0():
    """Regression: score.py used to break after step 0 unconditionally,
    silently dropping multi-step failures."""
    t = _traj(
        steps=[
            _step(passed=True),
            _step(passed=False, transition_valid=0.0, sequence_optimal=0.0),
        ],
        passed=False,
    )
    sc = aggregate("test", greedy_results=[t], sampled_results=[[]], sampled_final_calls=[[]], k=0)
    assert sc.failure_modes == {"transition_valid": 1}, (
        f"expected transition_valid=1, got {sc.failure_modes}"
    )


def test_aggregate_empty_results():
    """Zero greedy results → Scorecard with n_examples=0 and default scores."""
    from toyforge.eval.score import Scorecard

    sc = aggregate(
        "empty",
        greedy_results=[],
        sampled_results=[],
        sampled_final_calls=[],
        k=8,
    )
    assert isinstance(sc, Scorecard)
    assert sc.n_examples == 0
    assert sc.pass_at_1 == 0.0
    assert sc.failure_modes == {}


def test_aggregate_all_passing_no_failure_modes():
    """When every greedy trajectory passes, failure_modes is empty."""
    greedy = [_traj([_step(passed=True)], passed=True)] * 3
    sampled = [[_traj([_step(passed=True)], passed=True)] * 2] * 3
    finals = [
        [
            '<think>x</think>{"jsonrpc":"2.0","method":"ticket_get","params":{"ticket_id":"o"},"id":1}'
        ]
        * 2
    ] * 3
    sc = aggregate(
        "ok",
        greedy_results=greedy,
        sampled_results=sampled,
        sampled_final_calls=finals,
        k=2,
    )
    assert sc.pass_at_1 == 1.0
    assert sc.pass_at_k == 1.0
    assert sc.pass_at_maj == 1.0
    assert sc.failure_modes == {}


def test_aggregate_k_zero_no_crash():
    """k=0 with empty sampled lists yields pass_at_maj=0.0, no crash."""
    greedy = [_traj([_step(passed=True)], passed=True)]
    sc = aggregate(
        "k0",
        greedy_results=greedy,
        sampled_results=[[]],
        sampled_final_calls=[[]],
        k=0,
    )
    assert sc.pass_at_maj == 0.0
    assert sc.pass_at_1 == 1.0


def test_pass_at_maj_uses_any_passing_winner_not_first_occurrence():
    """Regression: pass@maj used to inspect only the first trajectory emitting
    the winning canonical call. If that specific trajectory failed verification
    but later emitters of the same call passed, the consensus answer is still
    verifiable and pass@maj should be 1.0.
    """
    call = '{"jsonrpc":"2.0","method":"ticket_get","params":{"ticket_id":"o"},"id":1}'
    # 3/5 sampled trajectories emit the winning call.
    # Sample 0 (first emitter of the winning call) fails verification.
    # Samples 2 and 3 (also winning call) pass verification.
    # Samples 1 and 4 emit a losing call (so they don't dominate the vote).
    losing = '{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"o"},"id":1}'
    sampled_finals = [
        [
            f"<think>A</think>{call}",  # winning, failed
            f"<think>B</think>{losing}",  # losing
            f"<think>C</think>{call}",  # winning, passed
            f"<think>D</think>{call}",  # winning, passed
            f"<think>E</think>{losing}",  # losing
        ]
    ]
    sampled_results = [
        [
            _traj([_step(passed=False)], passed=False),
            _traj([_step(passed=False)], passed=False),
            _traj([_step(passed=True)], passed=True),
            _traj([_step(passed=True)], passed=True),
            _traj([_step(passed=False)], passed=False),
        ]
    ]
    greedy = [_traj([_step(passed=True)], passed=True)]
    sc = aggregate(
        "regression",
        greedy_results=greedy,
        sampled_results=sampled_results,
        sampled_final_calls=sampled_finals,
        k=5,
    )
    # Winning call has 3 votes; 2 of the 3 emitters passed verification.
    # Consensus answer is reachable → pass_at_maj=1.0
    assert sc.pass_at_maj == 1.0, (
        f"expected pass_at_maj=1.0 (winning call has passing emitters), got {sc.pass_at_maj}"
    )


# ---------------------------------------------------------------------------
# Provenance field tests
# ---------------------------------------------------------------------------


def test_scorecard_has_provenance_fields_with_defaults():
    """All provenance fields default to empty/sensible values so existing test
    fixtures that only set `run_name` still construct cleanly."""
    from toyforge.eval.score import Scorecard

    sc = Scorecard(run_name="x")
    assert sc.eval_timestamp_utc == ""
    assert sc.git_sha == ""
    assert sc.adapter_dir == ""
    assert sc.base_model == ""
    assert sc.test_set_path == ""
    assert sc.test_set_sha256 == ""
    assert sc.eval_mode == "teacher_forced"


def test_scorecard_provenance_round_trips_through_to_dict():
    """Scorecard.to_dict must preserve all provenance fields."""
    from toyforge.eval.score import Scorecard

    sc = Scorecard(
        run_name="x",
        eval_timestamp_utc="2026-05-28T00:00:00+00:00",
        git_sha="abc1234",
        adapter_dir="/tmp/adapter",
        base_model="Qwen/Qwen3-8B-Instruct",
        test_set_path="/tmp/test.jsonl",
        test_set_sha256="deadbeef",
        eval_mode="teacher_forced",
    )
    d = sc.to_dict()
    assert d["git_sha"] == "abc1234"
    assert d["test_set_sha256"] == "deadbeef"
    assert d["adapter_dir"] == "/tmp/adapter"


def test_aggregate_works_without_provenance():
    """aggregate() doesn't take provenance args — those are populated by run_eval
    via dataclasses.replace post-construction."""
    from toyforge.eval.score import aggregate

    sc = aggregate(run_name="x", greedy_results=[], sampled_results=[], sampled_final_calls=[], k=8)
    assert sc.run_name == "x"
    assert sc.eval_timestamp_utc == ""  # not populated by aggregate


# ---------------------------------------------------------------------------
# Per-trajectory failure breakdown tests
# ---------------------------------------------------------------------------


def test_per_trajectory_populated_for_passing_and_failing(monkeypatch):
    """Aggregate produces per_trajectory entries for each input, with first-failure
    metadata only for failed trajectories."""
    from toyforge.eval.score import aggregate
    from toyforge.verifier.types import StepResult, Subscores, TrajectoryResult

    # Build minimal fixtures.
    passing_step = StepResult(
        passed=True,
        subscores=Subscores(
            parse=1,
            schema=1,
            method_known=1,
            precondition_met=1,
            transition_valid=1,
            sequence_optimal=1,
        ),
        parsed_thinking="x",
        parsed_call={"method": "ticket_status"},
        method="ticket_status",
        new_state="TRIAGED",
        error_message=None,
    )
    failing_step = StepResult(
        passed=False,
        subscores=Subscores(
            parse=1,
            schema=1,
            method_known=1,
            precondition_met=0,
            transition_valid=0,
            sequence_optimal=0,
        ),
        parsed_thinking="y",
        parsed_call={"method": "ticket_open"},
        method="ticket_open",
        new_state=None,
        error_message="precondition_met=0",
    )
    greedy = [
        TrajectoryResult(
            passed=True, steps=[passing_step], final_state="TRIAGED", error_message=None
        ),
        TrajectoryResult(
            passed=False,
            steps=[failing_step],
            final_state=None,
            error_message="step 0 precondition",
        ),
    ]
    trajectories = [
        {"trajectory_id": "t_pass", "initial_state": "NEW"},
        {"trajectory_id": "t_fail", "initial_state": "NEW"},
    ]
    sc = aggregate(
        run_name="x",
        greedy_results=greedy,
        sampled_results=[[], []],
        sampled_final_calls=[[], []],
        k=8,
        trajectories=trajectories,
    )
    assert len(sc.per_trajectory) == 2
    pt_pass = sc.per_trajectory[0]
    assert pt_pass == {
        "trajectory_id": "t_pass",
        "initial_state": "NEW",
        "passed": True,
        "first_failing_step": None,
        "first_failing_subscore": None,
        "error_message": None,
    }
    pt_fail = sc.per_trajectory[1]
    assert pt_fail["trajectory_id"] == "t_fail"
    assert pt_fail["passed"] is False
    assert pt_fail["first_failing_step"] == 0
    assert pt_fail["first_failing_subscore"] == "precondition_met"
    assert pt_fail["error_message"] == "precondition_met=0"


def test_scorecard_max_new_tokens_and_temperature_round_trip():
    from toyforge.eval.score import Scorecard

    sc = Scorecard(run_name="x", max_new_tokens=1024, temperature=0.7)
    d = sc.to_dict()
    assert d["max_new_tokens"] == 1024
    assert d["temperature"] == 0.7


def test_per_trajectory_defaults_to_empty_when_trajectories_absent():
    """When trajectories arg is None, per_trajectory entries have blank ids."""
    from toyforge.eval.score import aggregate
    from toyforge.verifier.types import StepResult, Subscores, TrajectoryResult

    failing_step = StepResult(
        passed=False,
        subscores=Subscores(parse=0),
        parsed_thinking=None,
        parsed_call=None,
        method=None,
        new_state=None,
        error_message="parse=0",
    )
    greedy = [
        TrajectoryResult(passed=False, steps=[failing_step], final_state=None, error_message=None)
    ]
    sc = aggregate("x", greedy, [[]], [[]], k=8)  # no trajectories arg
    assert len(sc.per_trajectory) == 1
    assert sc.per_trajectory[0]["trajectory_id"] == ""
    assert sc.per_trajectory[0]["initial_state"] == ""
    assert sc.per_trajectory[0]["first_failing_subscore"] == "parse"

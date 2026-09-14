"""verify_trajectory must agree with sequential verify_step calls."""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from toyforge.verifier.core import verify_step
from toyforge.verifier.trajectory import verify_trajectory

from .strategies import schemas, state_names


def _build_traj(prior, n_steps):
    """Build a minimal trajectory dict: all query-only ticket_get steps with
    no state change. final_state == initial_state == prior."""
    steps = [
        {
            "prompt_context": {"prior_state": prior},
            "thinking": "x",
            "tool_call": {
                "jsonrpc": "2.0",
                "method": "ticket_get",
                "params": {"ticket_id": f"obj-{i}"},
                "id": i,
            },
        }
        for i in range(n_steps)
    ]
    return {
        "trajectory_id": "synthetic",
        "initial_state": prior,
        "final_state": prior,
        "steps": steps,
    }


@given(prior=state_names(), n=st.integers(min_value=1, max_value=4))
def test_trajectory_passed_iff_every_step_passed_and_final_state_matches(prior, n):
    s = schemas()
    traj = _build_traj(prior, n)
    outputs = [f"<think>x</think>{json.dumps(step['tool_call'])}" for step in traj["steps"]]

    # Compute step results manually with state threading.
    state = traj["initial_state"]
    step_results = []
    for step, out in zip(traj["steps"], outputs, strict=True):
        ctx = dict(step["prompt_context"])
        ctx["prior_state"] = state
        r = verify_step(ctx, out, s)
        step_results.append(r)
        # Mirror verify_trajectory's advance rule exactly: only thread state
        # forward on passing steps, so this manual computation stays in lockstep
        # with the trajectory verifier under any future refactor.
        if r.passed and r.new_state is not None:
            state = r.new_state

    every_step_passed = all(r.passed for r in step_results)
    final_state_matches = state == traj["final_state"]
    expected_passed = every_step_passed and final_state_matches

    actual = verify_trajectory(traj, outputs, s)
    assert actual.passed == expected_passed, (
        f"trajectory.passed={actual.passed} but expected {expected_passed}; "
        f"step_passed={[r.passed for r in step_results]}, "
        f"final_state={state}, expected_final={traj['final_state']}"
    )


@given(prior=state_names())
def test_mismatched_final_state_fails_trajectory(prior):
    """If we claim a wrong final_state, the trajectory must fail even when all
    steps pass."""
    s = schemas()
    traj = _build_traj(prior, n_steps=1)
    # Force a wrong final_state (one different from prior).
    other_states = [st_ for st_ in s.states if st_ != prior]
    if not other_states:
        return  # only one state, can't test
    traj["final_state"] = other_states[0]

    outputs = [f"<think>x</think>{json.dumps(step['tool_call'])}" for step in traj["steps"]]
    actual = verify_trajectory(traj, outputs, s)
    assert actual.passed is False

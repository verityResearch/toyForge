"""Edge cases in verify_trajectory."""

from __future__ import annotations

from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory


def test_empty_trajectory_fails(schemas_dir):
    """Zero steps should fail (was vacuously True via all([])==True)."""
    schemas = load_schemas(schemas_dir)
    traj = {
        "trajectory_id": "empty",
        "initial_state": "NEW",
        "final_state": "NEW",
        "steps": [],
    }
    r = verify_trajectory(traj, [], schemas)
    assert r.passed is False
    assert r.error_message and "empty" in r.error_message.lower()


def test_step_failure_populates_error_message(schemas_dir):
    """A step that fails to parse must surface its error in TrajectoryResult.error_message."""
    schemas = load_schemas(schemas_dir)
    traj = {
        "trajectory_id": "bad",
        "initial_state": "NEW",
        "final_state": "NEW",
        "steps": [{"prompt_context": {"prior_state": "NEW"}}],
    }
    r = verify_trajectory(traj, ["not a think block"], schemas)
    assert r.passed is False
    assert r.error_message, "expected error_message on step failure, got None"

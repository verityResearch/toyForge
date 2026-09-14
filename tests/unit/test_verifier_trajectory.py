"""Tests for the trajectory-level verifier."""

from __future__ import annotations

import json

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory


@pytest.fixture(scope="module")
def schemas(schemas_dir):
    return load_schemas(schemas_dir)


def _render_step(step: dict) -> str:
    """Render a fixture step as a model output string."""
    return f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"


def _model_outputs_from_fixture(path):
    data = json.loads(path.read_text())
    return data, [_render_step(s) for s in data["steps"]]


def test_valid_minimal_trajectory(fixtures_dir, schemas):
    data, outputs = _model_outputs_from_fixture(
        fixtures_dir / "trajectories" / "valid_minimal.json"
    )
    r = verify_trajectory(data, outputs, schemas)
    assert r.passed is True
    assert r.final_state == "TRIAGED"
    assert all(s.passed for s in r.steps)


def test_invalid_transition_trajectory_fails(fixtures_dir, schemas):
    data, outputs = _model_outputs_from_fixture(
        fixtures_dir / "trajectories" / "invalid_transition.json"
    )
    r = verify_trajectory(data, outputs, schemas)
    assert r.passed is False
    assert any(not s.passed for s in r.steps)


def test_state_threads_through_steps(fixtures_dir, schemas):
    # Multi-step fixture has deliberate mismatch in step 2.
    data, outputs = _model_outputs_from_fixture(
        fixtures_dir / "trajectories" / "valid_multi_step.json"
    )
    r = verify_trajectory(data, outputs, schemas)
    assert r.steps[0].passed is True
    assert r.steps[0].new_state == "TRIAGED"
    assert r.steps[1].passed is False
    assert r.passed is False

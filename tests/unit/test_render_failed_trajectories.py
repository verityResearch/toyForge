"""Unit test for the failed-trajectories markdown renderer in eval/runner.py."""

from __future__ import annotations

import pytest

# runner.py imports torch/peft at module level — skip if not installed.
pytest.importorskip("torch")
pytest.importorskip("peft")

from toyforge.eval.runner import _render_failed_trajectories


def test_render_no_failures_returns_empty_section():
    """All-passing entries render the 'None' fallback."""
    entries = [
        {
            "trajectory_id": "t1",
            "initial_state": "NEW",
            "passed": True,
            "first_failing_step": None,
            "first_failing_subscore": None,
            "error_message": None,
        },
    ]
    out = _render_failed_trajectories(entries)
    assert "## Failed trajectories" in out
    assert "None (all greedy passes)" in out


def test_render_groups_by_initial_state_and_escapes_pipes():
    """Failed entries get grouped by initial_state; '|' in errors is escaped."""
    entries = [
        {
            "trajectory_id": "t_a",
            "initial_state": "NEW",
            "passed": False,
            "first_failing_step": 0,
            "first_failing_subscore": "parse",
            "error_message": "msg|with|pipes",
        },
        {
            "trajectory_id": "t_b",
            "initial_state": "TRIAGED",
            "passed": False,
            "first_failing_step": 1,
            "first_failing_subscore": "schema",
            "error_message": "short",
        },
        {
            "trajectory_id": "t_c",
            "initial_state": "NEW",
            "passed": False,
            "first_failing_step": 0,
            "first_failing_subscore": "parse",
            "error_message": "x" * 100,
        },  # truncated to 80 chars
    ]
    out = _render_failed_trajectories(entries)
    assert "### NEW" in out
    assert "### TRIAGED" in out
    assert "msg\\|with\\|pipes" in out  # escaped
    assert "x" * 80 in out
    assert "x" * 100 not in out  # truncated


def test_render_handles_unknown_initial_state():
    """Entries with empty initial_state get grouped under '(unknown)'."""
    entries = [
        {
            "trajectory_id": "t1",
            "initial_state": "",
            "passed": False,
            "first_failing_step": 0,
            "first_failing_subscore": "parse",
            "error_message": "err",
        },
    ]
    out = _render_failed_trajectories(entries)
    assert "### (unknown)" in out

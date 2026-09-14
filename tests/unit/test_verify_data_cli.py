"""Tests for the `toyforge verify-data` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from toyforge.cli import app

runner = CliRunner()


def _write_passing_trajectory(p: Path) -> None:
    """A minimal trajectory that passes the verifier."""
    traj = {
        "trajectory_id": "test_t1",
        "initial_state": "NEW",
        "final_state": "TRIAGED",
        "source": "hand_seed",
        "description": "test",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW", "situation": "x"},
                "thinking": "From NEW, ticket_open moves to TRIAGED.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_open",
                    "params": {
                        "requester_id": "user:t",
                        "body_text": "dGVzdA==",
                        "lifecycle_profile": "std",
                    },
                    "id": 1,
                },
                "expected_trigger": "ticket_open.accepted",
                "expected_state_after": "TRIAGED",
            }
        ],
    }
    p.write_text(json.dumps(traj) + "\n")


def test_verify_data_passes_when_files_valid(tmp_path):
    """All files exist and all trajectories pass."""
    repo = Path(__file__).resolve().parents[2]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ["train", "dev", "test"]:
        _write_passing_trajectory(data_dir / f"{name}.jsonl")
    result = runner.invoke(
        app,
        ["verify-data", "--data-dir", str(data_dir), "--schemas-dir", str(repo / "schemas")],
    )
    assert result.exit_code == 0, result.output
    assert "1/1 pass" in result.output


def test_verify_data_skips_missing_files(tmp_path):
    """Missing data files are reported as skip, not failure."""
    repo = Path(__file__).resolve().parents[2]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Only train exists; dev and test are missing
    _write_passing_trajectory(data_dir / "train.jsonl")
    result = runner.invoke(
        app,
        ["verify-data", "--data-dir", str(data_dir), "--schemas-dir", str(repo / "schemas")],
    )
    assert result.exit_code == 0, result.output
    assert "skip" in result.output


def test_verify_data_exits_nonzero_when_trajectory_fails(tmp_path):
    """A failing trajectory makes the CLI exit nonzero."""
    repo = Path(__file__).resolve().parents[2]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Write a trajectory with a deliberately wrong expected_state_after
    bad_traj = {
        "trajectory_id": "bad",
        "initial_state": "NEW",
        "final_state": "WRONG",  # mismatch with what step produces
        "source": "hand_seed",
        "description": "bad",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW", "situation": "x"},
                "thinking": "Wrong claim.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_open",
                    "params": {
                        "requester_id": "user:t",
                        "body_text": "dGVzdA==",
                        "lifecycle_profile": "std",
                    },
                    "id": 1,
                },
                "expected_trigger": "ticket_open.accepted",
                "expected_state_after": "WRONG_STATE",  # doesn't match transition
            }
        ],
    }
    (data_dir / "train.jsonl").write_text(json.dumps(bad_traj) + "\n")
    result = runner.invoke(
        app,
        ["verify-data", "--data-dir", str(data_dir), "--schemas-dir", str(repo / "schemas")],
    )
    assert result.exit_code != 0

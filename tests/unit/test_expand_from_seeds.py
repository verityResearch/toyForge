"""Tests for the expand-from-seeds smoke-tiny helper CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from toyforge.cli import app

runner = CliRunner()

_REPO = Path(__file__).resolve().parents[2]
_SCHEMAS_DIR = _REPO / "schemas"


def test_expand_from_seeds_writes_train_dev_test(tmp_path):
    """expand-from-seeds writes each split as JSONL containing the seeds."""
    seeds_path = _REPO / "scenarios" / "smoke-tiny-seeds.yaml"
    out_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "expand-from-seeds",
            "--seeds-path",
            str(seeds_path),
            "--schemas-dir",
            str(_SCHEMAS_DIR),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    for name in ["train", "dev", "test"]:
        p = out_dir / f"{name}.jsonl"
        assert p.exists(), f"{name}.jsonl was not created"
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 5, f"{name}.jsonl: expected 5 rows, got {len(lines)}"
        for line in lines[:3]:
            traj = json.loads(line)
            assert "trajectory_id" in traj
            assert traj.get("source") == "hand_seed"
            assert "provenance" in traj
            prov = traj["provenance"]
            assert prov["teacher_provider"] == "hand_seed"
            assert prov["seed_trajectory_id"] == traj["trajectory_id"]


def test_expand_from_seeds_all_splits_identical(tmp_path):
    """All three splits contain the same rows (smoke-tiny is pipeline-only)."""
    seeds_path = _REPO / "scenarios" / "smoke-tiny-seeds.yaml"
    out_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "expand-from-seeds",
            "--seeds-path",
            str(seeds_path),
            "--schemas-dir",
            str(_SCHEMAS_DIR),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    def _ids(name: str) -> list[str]:
        p = out_dir / f"{name}.jsonl"
        return [json.loads(ln)["trajectory_id"] for ln in p.read_text().splitlines() if ln.strip()]

    train_ids = _ids("train")
    assert _ids("dev") == train_ids
    assert _ids("test") == train_ids


def test_expand_from_seeds_works_with_full_seeds(tmp_path):
    """expand-from-seeds also works with the full seeds.yaml (not just smoke-tiny)."""
    seeds_path = _REPO / "scenarios" / "seeds.yaml"
    out_dir = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "expand-from-seeds",
            "--seeds-path",
            str(seeds_path),
            "--schemas-dir",
            str(_SCHEMAS_DIR),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    for name in ["train", "dev", "test"]:
        p = out_dir / f"{name}.jsonl"
        assert p.exists()
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) > 0


def test_expand_from_seeds_creates_output_dir(tmp_path):
    """expand-from-seeds creates the output directory if it does not exist."""
    seeds_path = _REPO / "scenarios" / "smoke-tiny-seeds.yaml"
    out_dir = tmp_path / "deeply" / "nested" / "dir"

    result = runner.invoke(
        app,
        [
            "expand-from-seeds",
            "--seeds-path",
            str(seeds_path),
            "--schemas-dir",
            str(_SCHEMAS_DIR),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "train.jsonl").exists()

"""Tests for scripts.audit_train_dev_independence."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_train_dev_independence import audit_independence


def _write_split(p: Path, trajs: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(t) for t in trajs) + "\n")


def _t(traj_id: str, seed_id: str | None, situation: str) -> dict:
    """Build a minimal trajectory dict."""
    return {
        "trajectory_id": traj_id,
        "initial_state": "NEW",
        "final_state": "TRIAGED",
        "source": "teacher_expansion" if seed_id else "hand_seed",
        "seed_trajectory_id": seed_id,
        "steps": [{"prompt_context": {"prior_state": "NEW", "situation": situation}}],
    }


def test_audit_flags_exact_trajectory_overlap(tmp_path):
    train_p = tmp_path / "train.jsonl"
    dev_p = tmp_path / "dev.jsonl"
    test_p = tmp_path / "test.jsonl"

    shared = _t("t1", "seed_a", "T-7 still failing after the fix on agent-4")
    _write_split(train_p, [shared, _t("t2", "seed_b", "T-9 ready")])
    _write_split(dev_p, [shared])
    _write_split(test_p, [_t("seed_c", None, "ESCALATED state observed")])

    report = audit_independence(train_p, dev_p, test_p)
    assert report["train_vs_dev"]["exact_trajectory_overlap"] == 1
    assert report["train_vs_dev"]["overlap_examples"][0]["trajectory_id"] == "t1"


def test_audit_test_independent_of_train(tmp_path):
    train_p = tmp_path / "train.jsonl"
    dev_p = tmp_path / "dev.jsonl"
    test_p = tmp_path / "test.jsonl"
    _write_split(train_p, [_t("t1", "seed_a", "situation A")])
    _write_split(dev_p, [_t("t2", "seed_a", "situation B")])
    _write_split(test_p, [_t("seed_a", None, "situation A original")])

    report = audit_independence(train_p, dev_p, test_p)
    # Trajectories should differ (different trajectory_id)
    assert report["train_vs_test"]["exact_trajectory_overlap"] == 0
    # But shared seed_id should be visible
    assert report["train_vs_test"]["shared_seed_ids"] == ["seed_a"]


def test_audit_prefix_similarity_high(tmp_path):
    train_p = tmp_path / "train.jsonl"
    dev_p = tmp_path / "dev.jsonl"
    test_p = tmp_path / "test.jsonl"
    sim_text = "T-9 still failing after agent-4 reset the customer's password twice"
    _write_split(train_p, [_t("t1", "seed_a", sim_text)])
    _write_split(dev_p, [_t("t2", "seed_a", sim_text)])  # same first-step text
    _write_split(test_p, [_t("seed_c", None, "different content entirely")])

    report = audit_independence(train_p, dev_p, test_p)
    assert report["train_vs_dev"]["high_prefix_similarity_pairs"] >= 1


def test_audit_handles_missing_files(tmp_path):
    train_p = tmp_path / "absent_train.jsonl"
    dev_p = tmp_path / "absent_dev.jsonl"
    test_p = tmp_path / "absent_test.jsonl"
    report = audit_independence(train_p, dev_p, test_p)
    assert report["train_vs_dev"]["exact_trajectory_overlap"] == 0
    assert "missing" in report["files"]["train"] or report["files"]["train"]["count"] == 0


def test_audit_report_is_json_serializable(tmp_path):
    """Report dict must be JSON-serializable (no frozenset, no Path)."""
    import json

    train_p = tmp_path / "train.jsonl"
    dev_p = tmp_path / "dev.jsonl"
    test_p = tmp_path / "test.jsonl"
    _write_split(train_p, [_t("t1", "seed_a", "x")])
    _write_split(dev_p, [_t("t2", "seed_a", "y")])
    _write_split(test_p, [_t("seed_b", None, "z")])
    report = audit_independence(train_p, dev_p, test_p)
    # Should not raise:
    json.dumps(report)

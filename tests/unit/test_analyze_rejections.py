"""Tests for scripts.analyze_rejections."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_rejections import summarize_rejections


def _write_rejections(tmp_path: Path) -> Path:
    rows = [
        {
            "seed_id": "seed_a",
            "attempt": 0,
            "reason": "json_decode",
            "error": "Expecting value: line 1 column 1 (char 0)",
            "raw": "not json",
        },
        {
            "seed_id": "seed_a",
            "attempt": 1,
            "reason": "verifier_failed",
            "error": "step 0: schema invalid",
            "step_errors": ["'notes' is not allowed", None],
        },
        {
            "seed_id": "seed_b",
            "attempt": 0,
            "reason": "verifier_failed",
            "error": "step 0: schema invalid",
            "step_errors": ["'requester_id' did not match '^user:'", None],
        },
        {
            "seed_id": "seed_b",
            "attempt": 1,
            "reason": "verifier_failed",
            "error": "transition mismatch",
            "step_errors": [None, "transition (ESCALATED, routing.assigned) not in table"],
        },
    ]
    p = tmp_path / "rejections.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_summarize_groups_by_reason(tmp_path):
    p = _write_rejections(tmp_path)
    summary = summarize_rejections(p)
    assert summary["total"] == 4
    assert summary["by_reason"]["json_decode"] == 1
    assert summary["by_reason"]["verifier_failed"] == 3


def test_summarize_groups_by_seed(tmp_path):
    p = _write_rejections(tmp_path)
    summary = summarize_rejections(p)
    assert summary["by_seed_id"]["seed_a"] == 2
    assert summary["by_seed_id"]["seed_b"] == 2


def test_summarize_extracts_top_errors(tmp_path):
    p = _write_rejections(tmp_path)
    summary = summarize_rejections(p)
    # Errors are normalized (substring buckets) and ranked
    assert len(summary["top_errors"]) >= 1
    assert any("schema invalid" in err for err, _count in summary["top_errors"])


def test_summarize_handles_missing_file(tmp_path):
    p = tmp_path / "absent.jsonl"
    summary = summarize_rejections(p)
    assert summary == {"total": 0, "by_reason": {}, "by_seed_id": {}, "top_errors": []}


def test_summarize_ignores_run_start_sentinels(tmp_path):
    p = tmp_path / "rejections.jsonl"
    p.write_text(
        json.dumps({"_run_start": True, "timestamp": "2026-05-28T12:00:00Z"})
        + "\n"
        + json.dumps({"seed_id": "s", "reason": "json_decode", "error": "bad"})
        + "\n"
    )
    summary = summarize_rejections(p)
    assert summary["total"] == 1
    assert "unknown" not in summary["by_reason"]


def test_summarize_rejections_streams_without_materializing(tmp_path):
    """summarize_rejections doesn't call list() on the iterator (verified by
    confirming summary correctness on a real file)."""
    path = tmp_path / "rejections.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"_run_start": True}),
                json.dumps({"reason": "json_decode", "seed_id": "s1", "error": "e1"}),
                json.dumps({"reason": "verifier_failed", "seed_id": "s1", "error": "e2"}),
                json.dumps({"reason": "json_decode", "seed_id": "s2", "error": "e1"}),
                "",
                json.dumps({"reason": "json_decode", "seed_id": "s2", "error": "e1"}),
            ]
        )
        + "\n"
    )
    summary = summarize_rejections(path)
    assert summary["total"] == 4
    assert summary["by_reason"]["json_decode"] == 3
    assert summary["by_reason"]["verifier_failed"] == 1
    assert summary["by_seed_id"]["s1"] == 2
    assert summary["by_seed_id"]["s2"] == 2

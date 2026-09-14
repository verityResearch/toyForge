"""Differential testing: verifier vs hand-graded labels in diff_fixtures.jsonl.

Each fixture entry asserts both `passed` and per-subscore expected values.
Drift on either is a FAIL with a structured diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step

_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "verifier_diff" / "diff_fixtures.jsonl"
)


def _load_fixtures():
    rows = []
    for line in _FIXTURES.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


_CASES = _load_fixtures()


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["name"])
def test_verifier_matches_hand_grade(case, schemas_dir):
    schemas = load_schemas(schemas_dir)
    ctx = {"prior_state": case["prior_state"]}
    ctx.update(case.get("extra_prompt_context", {}))

    r = verify_step(ctx, case["output"], schemas)
    actual_subscores = r.subscores.as_dict()

    errors = []
    if r.passed != case["expected_passed"]:
        errors.append(f"passed: got {r.passed}, expected {case['expected_passed']}")
    for key, expected_val in case["expected_subscores"].items():
        if abs(actual_subscores[key] - expected_val) > 1e-9:
            errors.append(f"subscore {key!r}: got {actual_subscores[key]}, expected {expected_val}")
    expected_substr = case.get("expected_error_substring")
    if expected_substr and (not r.error_message or expected_substr not in r.error_message):
        errors.append(
            f"error_message expected to contain {expected_substr!r}, got {r.error_message!r}"
        )
    assert not errors, (
        f"\n=== fixture: {case['name']} ===\nnote: {case.get('note', '')}\n" + "\n".join(errors)
    )

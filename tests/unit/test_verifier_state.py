"""Tests for state-transition stage of verify_step."""

from __future__ import annotations

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step


@pytest.fixture(scope="module")
def schemas(schemas_dir):
    return load_schemas(schemas_dir)


def _step(
    method: str,
    params: dict,
    prior_state: str,
    expected_trigger: str | None = None,
    expected_state_after: str | None = None,
):
    import json

    call = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    out = f"<think>x</think>{json.dumps(call)}"
    ctx = {"prior_state": prior_state}
    if expected_trigger is not None:
        ctx["expected_trigger"] = expected_trigger
    if expected_state_after is not None:
        ctx["expected_state_after"] = expected_state_after
    return out, ctx


def test_valid_path_new_to_triaged(schemas):
    out, ctx = _step(
        "ticket_open",
        {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"},
        prior_state="NEW",
        expected_trigger="ticket_open.accepted",
        expected_state_after="TRIAGED",
    )
    r = verify_step(ctx, out, schemas)
    assert r.subscores.precondition_met == 1.0
    assert r.subscores.transition_valid == 1.0
    assert r.passed is True
    assert r.new_state == "TRIAGED"


def test_precondition_fail_wrong_prior_state(schemas):
    out, ctx = _step(
        "ticket_open",
        {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"},
        prior_state="RESOLVED",
        expected_trigger="ticket_open.accepted",
        expected_state_after="TRIAGED",
    )
    r = verify_step(ctx, out, schemas)
    assert r.subscores.precondition_met == 0.0


def test_query_method_no_transition_required(schemas):
    out, ctx = _step(
        "ticket_status",
        {"ticket_id": "T-123"},
        prior_state="RESOLVED",
    )
    r = verify_step(ctx, out, schemas)
    assert r.subscores.precondition_met == 1.0
    assert r.subscores.transition_valid == 1.0
    assert r.passed is True
    assert r.new_state == "RESOLVED"


def test_transition_fail_wrong_expected_state(schemas):
    out, ctx = _step(
        "ticket_open",
        {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"},
        prior_state="NEW",
        expected_trigger="ticket_open.accepted",
        expected_state_after="RESOLVED",
    )
    r = verify_step(ctx, out, schemas)
    assert r.subscores.transition_valid == 0.0
    assert r.new_state is None, (
        f"new_state should be None when transition_valid=0, got {r.new_state!r}"
    )


def test_sequence_optimal_perfect_for_direct_call(schemas):
    out, ctx = _step(
        "ticket_open",
        {"requester_id": "user:1", "body_text": "YWJj", "lifecycle_profile": "std"},
        prior_state="NEW",
        expected_trigger="ticket_open.accepted",
        expected_state_after="TRIAGED",
    )
    r = verify_step(ctx, out, schemas)
    assert r.subscores.sequence_optimal == 1.0

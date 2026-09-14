"""Verifier must reject claims that a method emits a trigger it cannot emit."""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step


def test_method_cannot_emit_unrelated_trigger():
    """admin_escalate emits {escalation.start, escalation.resolved}; it cannot emit
    ticket_open.accepted. The verifier should set transition_valid=0 for any
    expected_trigger not in method_triggers[method] — independently of whether
    that trigger has an arc from prior_state in the transitions table."""
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")

    # admin_escalate emits {escalation.start, escalation.resolved}; not ticket_open.accepted.
    # (NEW, ticket_open.accepted) IS a valid arc, but only ticket_open can emit it.
    call = {
        "jsonrpc": "2.0",
        "method": "admin_escalate",
        "params": {"agent_id": "agent-1", "ticket_id": "T-1"},
        "id": 1,
    }
    ctx = {
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
    }
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas)
    assert r.subscores.transition_valid == 0.0, (
        f"admin_escalate cannot emit ticket_open.accepted; "
        f"got transition_valid={r.subscores.transition_valid}"
    )

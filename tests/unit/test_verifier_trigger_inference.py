"""GRPO mode: verifier infers expected_trigger from model's call when not given."""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step
from toyforge.verifier.reward import compute_grpo_reward
from toyforge.verifier.types import Rubric


def test_grpo_mode_infers_ticket_open_trigger():
    """ticket_open from NEW — only emittable trigger is ticket_open.accepted.
    Without expected_trigger in context, GRPO mode infers it and awards 1.0."""
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")
    call = {
        "jsonrpc": "2.0",
        "method": "ticket_open",
        "params": {
            "requester_id": "user:alice",
            "body_text": "YWFhYQ==",
            "lifecycle_profile": "std",
        },
        "id": 1,
    }
    ctx = {"prior_state": "NEW"}  # NO expected_trigger or expected_state_after
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas, infer_trigger=True)
    assert r.subscores.transition_valid == 1.0, r.subscores.as_dict()
    assert r.new_state == "TRIAGED"


def test_grpo_mode_multi_trigger_method_picks_consistent():
    """admin_escalate declares two triggers (escalation.start, escalation.resolved)
    but only escalation.start has an arc from ESCALATED. Inference picks the single
    candidate that has a valid arc."""
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")
    call = {
        "jsonrpc": "2.0",
        "method": "admin_escalate",
        "params": {"agent_id": "agent-1", "ticket_id": "T-1"},
        "id": 1,
    }
    ctx = {"prior_state": "ESCALATED"}
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas, infer_trigger=True)
    assert r.subscores.transition_valid == 1.0
    # new_state can be ENGINEERING (from escalation.start) — inferred
    assert r.new_state == "ENGINEERING"


def test_compute_grpo_reward_ignores_pinned_expected_trigger():
    """A seed-derived prompt context typically carries `expected_trigger`. The
    GRPO reward must IGNORE that and grade against what the model called —
    otherwise GRPO rollouts that diverge from the gold trigger get reward=0
    even when they're valid."""
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")

    # Seed says expected_trigger=ticket_open.accepted, but the prompt context
    # is passed to the reward function as-is. The model emits a valid call;
    # the reward should still award transition_valid=1.
    call = {
        "jsonrpc": "2.0",
        "method": "ticket_open",
        "params": {
            "requester_id": "user:bob",
            "body_text": "YmJiYg==",
            "lifecycle_profile": "std",
        },
        "id": 1,
    }
    ctx = {
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",  # gold-trigger pin
    }
    output = f"<think>x</think>{json.dumps(call)}"
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation="mean")
    reward, sub = compute_grpo_reward(ctx, output, schemas, rubric)
    assert sub.transition_valid == 1.0
    assert reward == 1.0


def test_grpo_mode_method_with_no_valid_trigger_from_state():
    """admin_escalate from RESOLVED has no plausible trigger (no arc exists).
    GRPO mode still gives transition_valid=0."""
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")
    call = {
        "jsonrpc": "2.0",
        "method": "admin_escalate",
        "params": {"agent_id": "agent-1", "ticket_id": "T-1"},
        "id": 1,
    }
    ctx = {"prior_state": "RESOLVED"}
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas, infer_trigger=True)
    assert r.subscores.transition_valid == 0.0

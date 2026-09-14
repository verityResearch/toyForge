"""Tests for the train.data module."""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.train.data import (
    format_step_as_chat_messages,
    iter_step_samples,
)


def _example_trajectory():
    return {
        "trajectory_id": "t1",
        "initial_state": "NEW",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW", "situation": "open"},
                "thinking": "I should call ticket_open.",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_open",
                    "params": {
                        "requester_id": "user:a",
                        "body_text": "YQ==",
                        "lifecycle_profile": "p",
                    },
                    "id": 1,
                },
                "expected_state_after": "TRIAGED",
                "expected_trigger": "ticket_open.accepted",
            }
        ],
        "final_state": "TRIAGED",
    }


def test_format_step_chat_messages_have_system_user_assistant_roles():
    traj = _example_trajectory()
    msgs = format_step_as_chat_messages(traj["steps"][0])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant"]


def test_assistant_content_contains_thinking_and_call():
    traj = _example_trajectory()
    msgs = format_step_as_chat_messages(traj["steps"][0])
    assistant = msgs[-1]["content"]
    assert "<think>" in assistant
    assert "</think>" in assistant
    assert '"method": "ticket_open"' in assistant or '"method":"ticket_open"' in assistant


def test_iter_step_samples_emits_one_per_step(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_example_trajectory()) + "\n")
    samples = list(iter_step_samples(path))
    assert len(samples) == 1
    assert samples[0]["trajectory_id"] == "t1"
    assert "messages" in samples[0]

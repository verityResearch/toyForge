"""Tests for schema loading."""

from __future__ import annotations

import pytest

from toyforge.schemas import Schemas, load_schemas


def test_load_schemas_returns_populated_object(schemas_dir):
    s = load_schemas(schemas_dir)
    assert isinstance(s, Schemas)
    assert len(s.method_names) == 9
    assert "ticket_open" in s.method_names
    assert "admin_lifecycle_apply" in s.method_names


def test_state_machine_has_expected_transitions(schemas_dir):
    s = load_schemas(schemas_dir)
    assert ("NEW", "ticket_open.accepted") in s.transitions
    assert s.transitions[("NEW", "ticket_open.accepted")] == "TRIAGED"
    assert ("RESOLVED", "ticket_reopen.issued") in s.transitions
    assert len(s.transitions) == 11


def test_states_set_matches_state_machine(schemas_dir):
    s = load_schemas(schemas_dir)
    expected = {
        "NEW",
        "TRIAGED",
        "ASSIGNED",
        "IN_PROGRESS",
        "RESOLVED",
        "REOPENED",
        "ESCALATED",
        "ENGINEERING",
        "CLOSED",
        "ARCHIVED",
    }
    assert s.states == expected


def test_method_params_schema_validates(schemas_dir):
    s = load_schemas(schemas_dir)
    ticket_open_schema = s.method_params_schema("ticket_open")
    assert ticket_open_schema["type"] == "object"
    assert "requester_id" in ticket_open_schema["properties"]


def test_method_params_schema_unknown_method_raises(schemas_dir):
    s = load_schemas(schemas_dir)
    with pytest.raises(KeyError):
        s.method_params_schema("not_a_real_method")


def test_rubric_presets_load(schemas_dir):
    s = load_schemas(schemas_dir)
    assert "shaped" in s.rubric_presets
    assert "binary" in s.rubric_presets
    assert s.rubric_presets["binary"] == {"transition_valid": 1.0}

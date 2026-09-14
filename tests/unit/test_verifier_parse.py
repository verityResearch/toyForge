"""Tests for parse stage of verify_step."""

from __future__ import annotations

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step


@pytest.fixture(scope="module")
def schemas(schemas_dir):
    return load_schemas(schemas_dir)


def _ctx(prior_state: str = "NEW") -> dict:
    return {"prior_state": prior_state}


def test_parses_canonical_format(schemas):
    out = (
        "<think>I should open the ticket.</think>"
        '{"jsonrpc":"2.0","method":"ticket_open",'
        '"params":{"requester_id":"user:1","body_text":"YWJj","lifecycle_profile":"std"},'
        '"id":1}'
    )
    r = verify_step(_ctx(), out, schemas)
    assert r.subscores.parse == 1.0
    assert r.parsed_thinking is not None
    assert r.parsed_call is not None
    assert r.parsed_call["method"] == "ticket_open"


def test_missing_think_tags_fails_parse(schemas):
    out = '{"jsonrpc":"2.0","method":"ticket_open","params":{},"id":1}'
    r = verify_step(_ctx(), out, schemas)
    assert r.subscores.parse == 0.0
    assert r.passed is False
    assert "think" in (r.error_message or "").lower()


def test_unclosed_think_tag_fails_parse(schemas):
    out = '<think>thinking but never closed {"jsonrpc":"2.0","method":"ticket_open"}'
    r = verify_step(_ctx(), out, schemas)
    assert r.subscores.parse == 0.0


def test_malformed_json_after_think_fails_parse(schemas):
    out = "<think>plan</think>not json at all"
    r = verify_step(_ctx(), out, schemas)
    assert r.subscores.parse == 0.0


def test_empty_thinking_still_parses(schemas):
    out = (
        "<think></think>"
        '{"jsonrpc":"2.0","method":"ticket_open",'
        '"params":{"requester_id":"user:1","body_text":"YWJj","lifecycle_profile":"std"},'
        '"id":1}'
    )
    r = verify_step(_ctx(), out, schemas)
    assert r.subscores.parse == 1.0
    assert r.parsed_thinking == ""

"""Tests for schema stage of verify_step."""

from __future__ import annotations

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step


@pytest.fixture(scope="module")
def schemas(schemas_dir):
    return load_schemas(schemas_dir)


def _good_open() -> str:
    return (
        "<think>plan</think>"
        '{"jsonrpc":"2.0","method":"ticket_open",'
        '"params":{"requester_id":"user:1","body_text":"YWJj","lifecycle_profile":"std"},'
        '"id":1}'
    )


def test_valid_params_score_1(schemas):
    r = verify_step({"prior_state": "NEW"}, _good_open(), schemas)
    assert r.subscores.schema == 1.0
    assert r.subscores.method_known == 1.0


def test_missing_required_param_fails_schema(schemas):
    out = (
        "<think>plan</think>"
        '{"jsonrpc":"2.0","method":"ticket_open",'
        '"params":{"requester_id":"user:1"},'
        '"id":1}'
    )
    r = verify_step({"prior_state": "NEW"}, out, schemas)
    assert r.subscores.parse == 1.0
    assert r.subscores.schema == 0.0


def test_unknown_method_fails_method_known(schemas):
    out = '<think>plan</think>{"jsonrpc":"2.0","method":"fake_method","params":{},"id":1}'
    r = verify_step({"prior_state": "NEW"}, out, schemas)
    assert r.subscores.method_known == 0.0


def test_extra_param_fails_schema(schemas):
    out = (
        "<think>plan</think>"
        '{"jsonrpc":"2.0","method":"ticket_open",'
        '"params":{"requester_id":"user:1","body_text":"YWJj","lifecycle_profile":"std","extra":"nope"},'
        '"id":1}'
    )
    r = verify_step({"prior_state": "NEW"}, out, schemas)
    assert r.subscores.schema == 0.0


def test_wrong_jsonrpc_version_fails_parse(schemas):
    out = (
        "<think>plan</think>"
        '{"jsonrpc":"1.0","method":"ticket_open","params":{"requester_id":"user:1","body_text":"YWJj","lifecycle_profile":"std"},"id":1}'
    )
    r = verify_step({"prior_state": "NEW"}, out, schemas)
    assert not r.passed

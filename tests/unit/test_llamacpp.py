"""Tests for llama.cpp helper surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from toyforge.llamacpp import (
    LlamaCppClient,
    LlamaCppConfig,
    build_jsonrpc_gbnf,
    grammar_sha256,
    render_step_outputs_llamacpp,
)


def test_build_jsonrpc_gbnf_mentions_every_method(schemas_dir: Path):
    grammar = build_jsonrpc_gbnf(schemas_dir)
    methods = json.loads((schemas_dir / "jsonrpc-methods.json").read_text())["methods"]
    for method in methods:
        assert method in grammar
    assert "jsonrpc" in grammar
    assert "params" in grammar


def test_versioned_grammar_matches_builder(schemas_dir: Path):
    assert (schemas_dir / "jsonrpc.gbnf").read_text() == build_jsonrpc_gbnf(schemas_dir)


def test_grammar_sha256_empty_for_missing(tmp_path: Path):
    assert grammar_sha256(tmp_path / "missing.gbnf") == ""


def test_llamacpp_client_posts_openai_compatible_payload(monkeypatch):
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = LlamaCppClient(LlamaCppConfig(base_url="http://localhost:8080/v1", model="qwen"))
    out = client.complete(
        [{"role": "user", "content": "hi"}],
        max_tokens=32,
        temperature=0.1,
        grammar='root ::= "x"',
        extra_body={"cache_prompt": True},
    )

    assert out == "ok"
    assert seen["url"] == "http://localhost:8080/v1/chat/completions"
    assert seen["body"]["model"] == "qwen"
    assert seen["body"]["grammar"] == 'root ::= "x"'
    assert seen["body"]["cache_prompt"] is True


def test_render_step_outputs_llamacpp_two_stage_constrained(tmp_path: Path):
    grammar = tmp_path / "g.gbnf"
    grammar.write_text('root ::= "x"\n')
    client = MagicMock()
    client.config = LlamaCppConfig(grammar_path=grammar)
    client.complete.side_effect = [
        "reason",
        '{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"o"},"id":1}',
    ]
    traj = {
        "steps": [
            {
                "prompt_context": {"prior_state": "TRIAGED", "situation": "check o"},
                "thinking": "gold",
                "tool_call": {
                    "jsonrpc": "2.0",
                    "method": "ticket_status",
                    "params": {"ticket_id": "o"},
                    "id": 1,
                },
            }
        ]
    }

    outs = render_step_outputs_llamacpp(
        client,
        traj,
        k=0,
        max_new_tokens=64,
        constrained=True,
    )

    assert outs == [
        [
            '<think>reason</think>{"jsonrpc":"2.0","method":"ticket_status","params":{"ticket_id":"o"},"id":1}'
        ]
    ]
    assert client.complete.call_args_list[1].kwargs["grammar"] == 'root ::= "x"\n'

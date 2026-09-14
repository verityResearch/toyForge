"""Tests for the shared rollout module (gen, gen_batched, render_step_outputs)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")  # rollout.py doesn't need peft, but eval-side does

from unittest.mock import MagicMock

import torch

from toyforge.rollout import gen_batched  # Other helpers tested indirectly via eval integration


def test_gen_batched_returns_k_completions():
    """The batched generate call decodes k completions from a single forward."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.decode.side_effect = lambda ids, **kw: f"completion_{int(ids[0])}"

    model = MagicMock()
    model.device = "cpu"
    prompt_len = 3
    gen_len = 2
    k = 4
    out_tensor = torch.zeros(k, prompt_len + gen_len, dtype=torch.long)
    for i in range(k):
        out_tensor[i, prompt_len] = i
    model.generate.return_value = out_tensor

    prompt_ids = torch.zeros(1, prompt_len, dtype=torch.long)

    results = gen_batched(model, tokenizer, prompt_ids, max_new_tokens=gen_len, k=k)
    assert len(results) == k
    for i in range(k):
        assert f"completion_{i}" in results[i]


def test_gen_batched_k_zero_returns_empty():
    """k=0 returns an empty list without calling model.generate."""
    tokenizer = MagicMock()
    model = MagicMock()
    prompt_ids = torch.zeros(1, 3, dtype=torch.long)
    results = gen_batched(model, tokenizer, prompt_ids, max_new_tokens=2, k=0)
    assert results == []
    model.generate.assert_not_called()


def test_render_step_outputs_returns_one_plus_k_per_step():
    """Each step in a trajectory produces 1 greedy + k sampled outputs."""
    from toyforge.rollout import render_step_outputs

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.return_value = torch.zeros(1, 5, dtype=torch.long)
    tokenizer.decode.side_effect = lambda ids, **kw: "x"

    model = MagicMock()
    model.device = "cpu"

    # generate returns a batch — for greedy (k=1, do_sample=False) and batched (k=k)
    def fake_generate(prompt_ids, **kwargs):
        batch_size = prompt_ids.shape[0]
        seq_len = prompt_ids.shape[1] + 2  # prompt + 2 generated tokens
        return torch.zeros(batch_size, seq_len, dtype=torch.long)

    model.generate.side_effect = fake_generate

    trajectory = {
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW"},
                "thinking": "t1",
                "tool_call": {"jsonrpc": "2.0", "method": "ticket_status", "params": {}, "id": 1},
            },
            {
                "prompt_context": {"prior_state": "TRIAGED"},
                "thinking": "t2",
                "tool_call": {"jsonrpc": "2.0", "method": "ticket_status", "params": {}, "id": 2},
            },
        ]
    }
    per_step = render_step_outputs(model, tokenizer, trajectory, k=4, max_new_tokens=10)
    assert len(per_step) == 2  # 2 steps
    for step_outs in per_step:
        assert len(step_outs) == 5  # 1 greedy + 4 sampled


def test_render_step_outputs_strips_gold_assistant_turn():
    """Chat template is built with [system, user, assistant] but assistant turn
    is stripped before calling .apply_chat_template — the model must generate
    its own assistant content."""
    from toyforge.rollout import render_step_outputs

    captured = []

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.decode.side_effect = lambda ids, **kw: "x"

    def fake_apply(messages, **kwargs):
        captured.append(messages)
        return torch.zeros(1, 5, dtype=torch.long)

    tokenizer.apply_chat_template.side_effect = fake_apply

    model = MagicMock()
    model.device = "cpu"

    def fake_generate(prompt_ids, **kwargs):
        return torch.zeros(prompt_ids.shape[0], prompt_ids.shape[1] + 1, dtype=torch.long)

    model.generate.side_effect = fake_generate

    trajectory = {
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW"},
                "thinking": "x",
                "tool_call": {"jsonrpc": "2.0", "method": "ticket_status", "params": {}, "id": 1},
            }
        ]
    }
    render_step_outputs(model, tokenizer, trajectory, k=1, max_new_tokens=5)
    # Chat template was called with [system, user] only (assistant stripped)
    msgs = captured[0]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_render_step_outputs_passes_temperature_to_sampled_generate():
    """A non-default temperature must thread all the way to gen_batched's
    model.generate(...) call. Greedy generation ignores temperature; sampled does not."""
    from toyforge.rollout import render_step_outputs

    captured_kwargs: list[dict] = []
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template.return_value = torch.zeros(1, 4, dtype=torch.long)
    tokenizer.decode.side_effect = lambda ids, **kw: "x"

    model = MagicMock()
    model.device = "cpu"

    def fake_generate(prompt_ids, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return torch.zeros(prompt_ids.shape[0], prompt_ids.shape[1] + 1, dtype=torch.long)

    model.generate.side_effect = fake_generate

    trajectory = {
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW"},
                "thinking": "x",
                "tool_call": {"jsonrpc": "2.0", "method": "ticket_status", "params": {}, "id": 1},
            }
        ]
    }
    render_step_outputs(model, tokenizer, trajectory, k=2, max_new_tokens=5, temperature=0.3)
    # 2 generate calls: one greedy (no temperature kwarg) + one batched-sampled (with temperature)
    assert len(captured_kwargs) == 2
    sampled_kwargs = next(kw for kw in captured_kwargs if kw.get("do_sample") is True)
    assert sampled_kwargs["temperature"] == 0.3

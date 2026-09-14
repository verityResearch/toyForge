"""Unit tests for the GRPO skeleton — validation only, no GPU/TRL required."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from toyforge.train.config import GRPOConfig, LoRAConfig, TrainConfig
from toyforge.train.grpo import run_grpo


def _base_cfg(**overrides) -> TrainConfig:
    base = TrainConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        method="grpo",
        output_dir=Path("out/run"),
        train_path=Path("data/train.jsonl"),
        dev_path=Path("data/dev.jsonl"),
        lora=LoRAConfig(r=8, alpha=16, target_modules=["q_proj"], dropout=0.05),
    )
    return replace(base, **overrides)


def test_run_grpo_raises_when_grpo_block_missing():
    cfg = _base_cfg(grpo=None, adapter_init=Path("out/sft/adapter"))
    with pytest.raises(ValueError, match=r"grpo"):
        run_grpo(cfg)


def test_run_grpo_raises_when_adapter_init_missing():
    cfg = _base_cfg(grpo=GRPOConfig(), adapter_init=None)
    with pytest.raises(ValueError, match=r"adapter_init"):
        run_grpo(cfg)


def test_run_grpo_raises_not_implemented_when_fully_configured():
    """With both grpo and adapter_init set, the skeleton prints the banner then
    raises NotImplementedError — the visible signal that Phase 2 implementation
    is pending."""
    cfg = _base_cfg(grpo=GRPOConfig(), adapter_init=Path("out/sft/adapter"))
    with pytest.raises(NotImplementedError, match=r"skeleton"):
        run_grpo(cfg)


def test_run_grpo_dispatches_llamacpp_runtime_to_feasibility_gate(tmp_path):
    from toyforge.train.config import LlamaCppTrainConfig

    grammar = tmp_path / "jsonrpc.gbnf"
    grammar.write_text('root ::= "x"\n')
    cfg = _base_cfg(
        grpo=GRPOConfig(),
        adapter_init=Path("out/sft/adapter"),
        runtime="llamacpp",
        llamacpp=LlamaCppTrainConfig(grammar_path=grammar),
    )
    with pytest.raises(RuntimeError, match=r"llama\.cpp.*binary"):
        run_grpo(cfg)

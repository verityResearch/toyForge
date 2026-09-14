"""Tests for TrainConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from toyforge.train.config import load_train_config


def test_load_train_config_yaml(tmp_path: Path):
    path = tmp_path / "train.yaml"
    path.write_text("""
base_model: Qwen/Qwen3-8B-Instruct
method: vanilla
output_dir: out/phase1-sft
data:
  train: data/train.jsonl
  dev:   data/dev.jsonl
  test:  data/test.jsonl
lora:
  r: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj]
context:
  train: 8192
  eval: 16384
optimizer: paged_adamw_8bit
batch_size: 1
grad_accum: 16
lr: 2.0e-5
num_epochs: 3
""")
    cfg = load_train_config(path)
    assert cfg.base_model == "Qwen/Qwen3-8B-Instruct"
    assert cfg.method == "vanilla"
    assert cfg.lora.r == 32
    assert cfg.lora.alpha == 64
    assert "q_proj" in cfg.lora.target_modules
    assert cfg.context_train == 8192
    assert cfg.lr == 2e-5


def test_load_train_config_missing_required_raises(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("method: vanilla\n")
    with pytest.raises(KeyError):
        load_train_config(path)


def test_train_config_handles_null_lora_block(tmp_path):
    """`lora: null` in YAML must not crash with AttributeError."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n"  # explicit null block — common authoring mistake
    )
    cfg = load_train_config(p)  # Should not raise
    assert cfg.lora.r == 32  # default


def test_train_config_handles_null_data_block(tmp_path):
    """`data:` with no body in YAML must not crash."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "data:\n"
    )
    cfg = load_train_config(p)  # Should not raise
    assert str(cfg.train_path) == "data/train.jsonl"


def test_train_config_has_warmup_and_scheduler_defaults(tmp_path):
    """warmup_ratio and lr_scheduler_type are first-class TrainConfig fields."""
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
    )
    cfg = load_train_config(p)
    assert cfg.warmup_ratio == 0.05
    assert cfg.lr_scheduler_type == "cosine"


def test_train_config_yaml_overrides_warmup(tmp_path):
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "warmup_ratio: 0.1\n"
        "lr_scheduler_type: linear\n"
    )
    cfg = load_train_config(p)
    assert cfg.warmup_ratio == 0.1
    assert cfg.lr_scheduler_type == "linear"


def test_train_config_loads_grpo_block(tmp_path):
    from toyforge.train.config import load_train_config

    p = tmp_path / "grpo.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: grpo\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "adapter_init: out/phase1-sft/adapter\n"
        "grpo:\n  num_generations: 4\n  beta: 0.02\n  rubric_preset: shaped\n"
        "  rollout_temperature: 0.3\n"
    )
    cfg = load_train_config(p)
    assert cfg.method == "grpo"
    assert cfg.grpo is not None
    assert cfg.grpo.num_generations == 4
    assert cfg.grpo.beta == 0.02
    assert str(cfg.adapter_init) == "out/phase1-sft/adapter"
    assert cfg.grpo.rollout_temperature == 0.3


def test_train_config_grpo_defaults_to_none(tmp_path):
    """A vanilla SFT config has no `grpo:` or `adapter_init:` — both fields are None."""
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
    )
    cfg = load_train_config(p)
    assert cfg.grpo is None
    assert cfg.adapter_init is None


def test_load_train_config_warns_on_unknown_keys(tmp_path):
    import warnings

    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "learning_rate: 1e-4\n"  # typo — should be `lr`
        "this_does_not_exist: 99\n"
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_train_config(p)
    msgs = [str(x.message) for x in w]
    assert any("learning_rate" in m or "unknown" in m.lower() for m in msgs), (
        f"expected unknown-keys warning, got: {msgs}"
    )


def test_load_with_inheritance_basic(tmp_path):
    """Child inherits base and overrides specific fields."""
    base = tmp_path / "phase1-sft.yaml"
    base.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 32\n  alpha: 64\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "num_epochs: 3\n"
        "lr: 2e-5\n"
    )
    child = tmp_path / "phase3-lora-r64.yaml"
    child.write_text("extends: phase1-sft.yaml\nlora:\n  r: 64\noutput_dir: out/phase3-lora-r64\n")
    cfg = load_train_config(child)
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"  # from base
    assert cfg.method == "vanilla"  # from base
    assert cfg.lora.r == 64  # overridden
    assert cfg.lora.alpha == 64  # inherited (not in child)
    assert "q_proj" in cfg.lora.target_modules  # inherited list
    assert cfg.num_epochs == 3  # inherited
    assert str(cfg.output_dir) == "out/phase3-lora-r64"  # new in child


def test_load_with_inheritance_three_levels(tmp_path):
    """A → B → C chain merges left-to-right (C wins, then B, then A)."""
    grandparent = tmp_path / "grandparent.yaml"
    grandparent.write_text(
        "base_model: gp-model\n"
        "method: vanilla\n"
        "lr: 1e-4\n"
        "lora:\n  r: 4\n  alpha: 4\n  target_modules: [a]\n  dropout: 0.05\n"
    )
    parent = tmp_path / "parent.yaml"
    parent.write_text("extends: grandparent.yaml\nlr: 2.0e-5\nlora:\n  r: 16\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nlora:\n  alpha: 32\n")
    cfg = load_train_config(child)
    assert cfg.base_model == "gp-model"  # from grandparent
    assert cfg.lr == 2e-5  # from parent (overrides grandparent)
    assert cfg.lora.r == 16  # from parent
    assert cfg.lora.alpha == 32  # from child (overrides grandparent's 4)
    assert cfg.lora.target_modules == ["a"]  # from grandparent


def test_load_with_inheritance_detects_cycle(tmp_path):
    """Circular extends raises ValueError mentioning circular."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("extends: b.yaml\nbase_model: x\nmethod: vanilla\n")
    b.write_text("extends: a.yaml\n")
    import pytest

    with pytest.raises(ValueError, match=r"circular"):
        load_train_config(a)


def test_load_with_inheritance_missing_base(tmp_path):
    """Reference to a nonexistent base config raises FileNotFoundError."""
    child = tmp_path / "child.yaml"
    child.write_text("extends: nonexistent.yaml\nbase_model: x\nmethod: vanilla\n")
    import pytest

    with pytest.raises(FileNotFoundError, match=r"extends target not found"):
        load_train_config(child)


def test_deep_merge_replaces_lists():
    """Lists are replaced wholesale, not element-merged."""
    from toyforge.train.config import _deep_merge

    out = _deep_merge(
        {"lora": {"target_modules": ["a", "b", "c"], "r": 8}},
        {"lora": {"target_modules": ["x"]}},
    )
    assert out["lora"]["target_modules"] == ["x"]
    assert out["lora"]["r"] == 8  # preserved


def test_load_with_inheritance_self_reference_cycle(tmp_path):
    """A file that extends itself raises ValueError."""
    a = tmp_path / "self.yaml"
    a.write_text("extends: self.yaml\nbase_model: x\nmethod: vanilla\n")
    import pytest

    with pytest.raises(ValueError, match=r"circular"):
        load_train_config(a)


def test_train_config_report_to_defaults_to_tensorboard(tmp_path):
    """When no `report_to` key is present, defaults to ['tensorboard']."""
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
    )
    cfg = load_train_config(p)
    assert cfg.report_to == ["tensorboard"]


def test_train_config_report_to_override(tmp_path):
    """Explicit `report_to: []` disables all reporting."""
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "report_to: []\n"
    )
    cfg = load_train_config(p)
    assert cfg.report_to == []


def test_train_config_report_to_multiple(tmp_path):
    """Multiple reporters can be configured."""
    from toyforge.train.config import load_train_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 8\n  alpha: 16\n  target_modules: [q_proj]\n  dropout: 0.05\n"
        "report_to: [tensorboard, wandb]\n"
    )
    cfg = load_train_config(p)
    assert "tensorboard" in cfg.report_to
    assert "wandb" in cfg.report_to


def test_train_config_loads_llamacpp_runtime_block(tmp_path):
    p = tmp_path / "grpo-llamacpp.yaml"
    p.write_text(
        "base_model: Qwen/Qwen3-8B-Instruct\n"
        "method: grpo\n"
        "runtime: llamacpp\n"
        "adapter_init: out/phase1-sft/adapter\n"
        "grpo:\n  num_generations: 4\n"
        "llamacpp:\n"
        "  base_url: http://localhost:8080/v1\n"
        "  model: qwen.gguf\n"
        "  grammar_path: schemas/jsonrpc.gbnf\n"
        "  constrained: true\n"
        "  reference_adapter_scale: 0.0\n"
    )
    cfg = load_train_config(p)
    assert cfg.runtime == "llamacpp"
    assert cfg.llamacpp is not None
    assert cfg.llamacpp.model == "qwen.gguf"
    assert cfg.llamacpp.constrained is True
    assert str(cfg.llamacpp.grammar_path) == "schemas/jsonrpc.gbnf"

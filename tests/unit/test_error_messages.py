"""Tests for actionable error messages (Task 6.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from toyforge.train.config import load_train_config


def test_sft_no_samples_error_mentions_just_expand(tmp_path: Path):
    """Empty training jsonl error must tell the user to run `just expand`.

    Skipped on CPU environments where the `train` extra (torch/trl/datasets/peft)
    is not installed — the module-level imports in sft.py fail before the function
    body is reachable.
    """
    pytest.importorskip("datasets")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from toyforge.train.sft import _build_dataset

    empty = tmp_path / "train.jsonl"
    empty.write_text("")  # zero samples
    with pytest.raises(ValueError, match=r"just expand"):
        _build_dataset(empty)


def test_eval_runner_missing_test_set_mentions_just_expand(tmp_path: Path):
    """Missing test set error must tell the user to run `just expand`.

    Skipped on CPU environments where torch/peft are not installed.
    """
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from toyforge.eval.runner import _load_test

    with pytest.raises(FileNotFoundError, match=r"just expand"):
        _load_test(tmp_path / "does-not-exist.jsonl")


def test_anthropic_teacher_error_includes_actionable_export_hint(monkeypatch):
    """The missing-env-var error should show the user how to set the var."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from toyforge.scenario_gen.teachers.anthropic_teacher import AnthropicTeacher
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    cfg = TeacherConfig(provider="anthropic", model="claude-3-haiku-20240307")
    with pytest.raises(RuntimeError, match=r"export ANTHROPIC_API_KEY"):
        AnthropicTeacher(cfg)


def test_train_config_missing_required_key_includes_path(tmp_path: Path):
    """Missing required key error should name the config file path."""
    p = tmp_path / "bad.yaml"
    p.write_text("method: vanilla\n")  # missing 'base_model'
    with pytest.raises(KeyError, match=r"base_model.*bad\.yaml"):
        load_train_config(p)

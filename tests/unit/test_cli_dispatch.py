"""CLI dispatch tests for the `train` command."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# CLI dispatch tests patch into toyforge.train.sft, which top-level-imports
# trl/torch/transformers/peft. Skip on CPU-only environments without those.
pytest.importorskip("trl")
pytest.importorskip("torch")

from typer.testing import CliRunner

from toyforge.cli import app

runner = CliRunner()


def test_train_dispatches_to_grpo_for_grpo_method(tmp_path, monkeypatch):
    cfg_path = tmp_path / "grpo.yaml"
    cfg_path.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: grpo\n"
        "lora:\n"
        "  r: 8\n"
        "  alpha: 16\n"
        "  target_modules: [q_proj]\n"
    )
    monkeypatch.chdir(tmp_path)
    with patch("toyforge.train.sft.run_sft") as mock_sft:
        result = runner.invoke(app, ["train", "--config-path", str(cfg_path)])
    # Should NOT call run_sft when method=grpo.
    mock_sft.assert_not_called()
    # Should exit non-zero with a clear method-name message.
    assert result.exit_code != 0
    assert (
        "grpo" in (result.output or result.stderr or "").lower()
        or "grpo" in str(result.exception or "").lower()
    )


def test_train_dispatches_to_sft_for_vanilla_method(tmp_path, monkeypatch):
    cfg_path = tmp_path / "sft.yaml"
    cfg_path.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n"
        "  r: 8\n"
        "  alpha: 16\n"
        "  target_modules: [q_proj]\n"
    )
    monkeypatch.chdir(tmp_path)
    # Patch where it's IMPORTED into cli.py (cli imports inside function body)
    with patch("toyforge.train.sft.run_sft") as mock_sft:
        mock_sft.return_value = tmp_path / "out"
        result = runner.invoke(app, ["train", "--config-path", str(cfg_path)])
    mock_sft.assert_called_once()
    assert result.exit_code == 0, result.output

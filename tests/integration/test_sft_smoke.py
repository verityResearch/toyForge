"""Smoke test for run_sft: build configs, mock transformers/peft/trl, verify
that the SFT call assembles correctly without actually running training.

This test pins the invocation contract for SFTTrainer (TRL 1.x kwargs +
assistant_only_loss) so refactors can't silently regress.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Skip if train extra isn't installed.
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("datasets")


def _stub_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<|endoftext|>"
    # apply_chat_template must return a string that contains the response template.
    tok.apply_chat_template.return_value = (
        "<|im_start|>system\nfoo<|im_end|>\n"
        "<|im_start|>user\nbar<|im_end|>\n"
        "<|im_start|>assistant\n<think>x</think>{}<|im_end|>\n"
    )
    return tok


def test_run_sft_invokes_sfttrainer_with_trl1x_kwargs(tmp_path):
    """run_sft must:
    - construct SFTConfig with assistant_only_loss=True and max_length (not max_seq_length)
    - pass processing_class=tokenizer (not tokenizer=) to SFTTrainer
    - call apply_chat_template once for the startup guard (verifies response template is present)
    - call trainer.train() then save the adapter
    """
    from toyforge.train.config import LoRAConfig, TrainConfig

    train_path = tmp_path / "train.jsonl"
    train_path.write_text("")  # iter_step_samples is mocked anyway
    dev_path = tmp_path / "dev.jsonl"
    dev_path.write_text("")
    cfg = TrainConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        method="vanilla",
        lora=LoRAConfig(r=8, alpha=16, target_modules=["q_proj"], dropout=0.05),
        attn_impl="sdpa",
        context_train=512,
        train_path=train_path,
        dev_path=dev_path,
        output_dir=tmp_path / "out",
    )

    fake_sample = {
        "trajectory_id": "t1",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "<think>x</think>{}"},
        ],
    }

    # _build_dataset calls iter_step_samples then Dataset.from_generator (post-I2).
    # We patch iter_step_samples so both train and dev datasets get the fake sample,
    # and patch Dataset to return a mock that supports indexing for the startup guard.
    fake_dataset = MagicMock()
    fake_dataset.__len__ = MagicMock(return_value=1)
    fake_dataset.__getitem__ = MagicMock(return_value=fake_sample)

    with (
        patch("toyforge.train.sft.AutoTokenizer") as mock_tok_cls,
        patch("toyforge.train.sft.AutoModelForCausalLM") as mock_lm_cls,
        patch("toyforge.train.sft.LoraConfig") as mock_lora_cls,
        patch("toyforge.train.sft.get_peft_model") as mock_get_peft,
        patch("toyforge.train.sft.SFTTrainer") as mock_trainer_cls,
        patch("toyforge.train.sft.SFTConfig") as mock_sftconfig_cls,
        patch(
            "toyforge.train.sft.iter_step_samples",
            side_effect=lambda _path: iter([fake_sample]),
        ),
        patch("toyforge.train.sft.Dataset") as mock_dataset_cls,
    ):
        mock_tok_cls.from_pretrained.return_value = _stub_tokenizer()
        mock_lm_cls.from_pretrained.return_value = MagicMock()
        mock_lora_cls.return_value = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = mock_peft_model
        mock_dataset_cls.from_generator.return_value = fake_dataset
        mock_trainer = MagicMock()
        mock_trainer.model = mock_peft_model
        mock_trainer_cls.return_value = mock_trainer

        from toyforge.train import sft as sft_module

        # Force re-execution in the patched context (module was already imported).
        sft_module.run_sft(cfg)

    # 1) SFTConfig was constructed with TRL 1.x kwargs.
    mock_sftconfig_cls.assert_called_once()
    sftconfig_kwargs = mock_sftconfig_cls.call_args.kwargs
    assert sftconfig_kwargs.get("assistant_only_loss") is True, (
        f"assistant_only_loss must be True; got SFTConfig kwargs={list(sftconfig_kwargs)}"
    )
    assert "max_length" in sftconfig_kwargs, (
        f"TRL 1.x uses max_length, not max_seq_length; got {list(sftconfig_kwargs)}"
    )
    assert "max_seq_length" not in sftconfig_kwargs, (
        "max_seq_length is the TRL 0.x kwarg; do not use it"
    )

    # 2) SFTTrainer was constructed with TRL 1.x kwargs.
    mock_trainer_cls.assert_called_once()
    trainer_kwargs = mock_trainer_cls.call_args.kwargs
    assert "processing_class" in trainer_kwargs, (
        f"TRL 1.x uses processing_class, not tokenizer; got {list(trainer_kwargs)}"
    )
    assert "tokenizer" not in trainer_kwargs, "tokenizer= is the TRL 0.x kwarg; do not use it"

    # 3) Training and save were called.
    mock_trainer.train.assert_called_once()
    mock_peft_model.save_pretrained.assert_called()

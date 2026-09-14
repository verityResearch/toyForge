"""Truncation guard: catch the case where assistant turn falls past max_length."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")


def test_run_sft_raises_when_assistant_turn_starts_past_max_length(tmp_path):
    """If the first sample's assistant turn starts beyond max_length, run_sft
    must raise at startup rather than training silently with zero gradient
    (TRL #3927)."""
    from toyforge.train.config import LoRAConfig, TrainConfig

    cfg = TrainConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        method="vanilla",
        lora=LoRAConfig(r=8, alpha=16, target_modules=["q_proj"], dropout=0.05),
        attn_impl="sdpa",
        context_train=32,  # tiny limit to trigger overflow
        train_path=tmp_path / "train.jsonl",
        dev_path=tmp_path / "dev.jsonl",
        output_dir=tmp_path / "out",
    )
    cfg.train_path.write_text("")
    cfg.dev_path.write_text("")

    long_user = "x " * 200
    long_sample = {
        "trajectory_id": "t1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": long_user},
            {"role": "assistant", "content": "<think>x</think>{}"},
        ],
    }

    # Build the rendered string that apply_chat_template will return.
    # The assistant marker appears after the long user section — well past token 32.
    rendered = (
        "<|im_start|>system\nsystem<|im_end|>\n"
        "<|im_start|>user\n" + long_user + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>x</think>{}<|im_end|>\n"
    )

    with (
        patch("toyforge.train.sft.AutoTokenizer") as mock_tok_cls,
        patch("toyforge.train.sft.AutoModelForCausalLM") as mock_lm_cls,
        patch("toyforge.train.sft.get_peft_model") as mock_get_peft,
        patch("toyforge.train.sft.SFTTrainer"),
        patch("toyforge.train.sft.SFTConfig"),
        patch("toyforge.train.sft.LoraConfig"),
        patch("toyforge.train.sft.Dataset") as mock_dataset_cls,
        patch(
            "toyforge.train.sft.iter_step_samples",
            side_effect=lambda _p: iter([long_sample]),
        ),
    ):
        tok = MagicMock()
        tok.pad_token = "<pad>"
        tok.eos_token = "<eos>"
        tok.apply_chat_template.return_value = rendered

        # Tokenize: one token per char using ord() so each character maps to a
        # unique id. This ensures the assistant-marker subsequence only matches
        # at the correct position within full_ids (no false match at index 0).
        def fake_tokenize(text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

        tok.side_effect = fake_tokenize

        mock_tok_cls.from_pretrained.return_value = tok
        mock_lm_cls.from_pretrained.return_value = MagicMock()
        mock_get_peft.return_value = MagicMock()

        fake_dataset = MagicMock()
        fake_dataset.__len__ = MagicMock(return_value=1)
        fake_dataset.__getitem__ = MagicMock(return_value=long_sample)
        mock_dataset_cls.from_generator.return_value = fake_dataset

        from toyforge.train.sft import run_sft

        with pytest.raises(ValueError, match=r"(max_length|truncat|3927)"):
            run_sft(cfg)

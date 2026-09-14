"""SFT cold-start trainer for Phase 1.

Builds a TRL SFTTrainer with LoRA on Qwen3-8B-Instruct. Reads TrainConfig.
Uses TRL 1.x API (max_length, processing_class).
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from toyforge._console import console as _console
from toyforge.train.config import TrainConfig
from toyforge.train.data import iter_step_samples

# Qwen chat template uses this assistant-turn opener. If swapping bases, confirm
# the new tokenizer's template and update here (or thread through TrainConfig).
# TRL 1.x uses assistant_only_loss=True + get_training_chat_template internally;
# we keep this constant as a startup-guard probe only.
_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def _build_dataset(jsonl_path: Path) -> Dataset:
    """Stream samples lazily from JSONL → Dataset."""

    def _gen():
        yield from iter_step_samples(jsonl_path)

    ds = Dataset.from_generator(_gen)
    if len(ds) == 0:
        raise ValueError(
            f"no samples in {jsonl_path}. Run `just expand` to populate data/ before training."
        )
    return ds


def run_sft(cfg: TrainConfig) -> Path:
    """Train an SFT cold-start adapter. Returns the output dir containing the adapter."""
    _console.print(f"[bold]SFT[/bold] base={cfg.base_model} method={cfg.method}")
    _console.print(f"  train={cfg.train_path}  dev={cfg.dev_path}  out={cfg.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        attn_implementation=cfg.attn_impl,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = _build_dataset(cfg.train_path)
    dev_ds = _build_dataset(cfg.dev_path)

    # Sanity check: confirm the response template appears in the tokenizer's
    # chat output. If not, assistant_only_loss would produce zero training signal
    # (all tokens masked) — catch it at startup instead of silently flat loss.
    sample_messages = train_ds[0]["messages"]
    rendered = tokenizer.apply_chat_template(sample_messages, tokenize=False)
    if _RESPONSE_TEMPLATE not in rendered:
        raise ValueError(
            f"response template {_RESPONSE_TEMPLATE!r} not found in tokenized "
            f"chat output for {cfg.base_model!r}. The assistant_only_loss collator "
            f"would mask ALL tokens, giving zero training signal. First 400 chars "
            f"of rendered chat:\n{rendered[:400]}"
        )

    # TRL #3927 guard: the assistant turn must start within max_length, otherwise
    # the assistant_only_loss mask is empty after truncation → silent zero-loss.
    tokenized_full = tokenizer(rendered, add_special_tokens=False)
    assistant_marker_ids = tokenizer(_RESPONSE_TEMPLATE, add_special_tokens=False)["input_ids"]
    full_ids = tokenized_full["input_ids"]
    # Find the position where the assistant marker ends.
    assistant_end = None
    for i in range(len(full_ids) - len(assistant_marker_ids) + 1):
        if list(full_ids[i : i + len(assistant_marker_ids)]) == list(assistant_marker_ids):
            assistant_end = i + len(assistant_marker_ids)
            break
    if assistant_end is not None and assistant_end >= cfg.context_train:
        raise ValueError(
            f"Sample 0's assistant turn starts at token {assistant_end} but "
            f"max_length={cfg.context_train}. After truncation the assistant "
            f"mask would be empty, producing silent zero-loss training "
            f"(TRL #3927). Increase context_train or shorten the data."
        )

    sft_cfg = SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optimizer,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        bf16=True,
        max_length=cfg.context_train,  # TRL 1.x: renamed from max_seq_length
        # Mask system+user tokens — gradients flow only through assistant turns
        # (<think>…</think>{tool_call}). TRL 1.x handles this natively for
        # conversational datasets by patching the chat template with {% generation %}
        # markers and using return_assistant_tokens_mask=True during tokenization.
        assistant_only_loss=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=cfg.seed,
        report_to=cfg.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,  # TRL 1.x: renamed from tokenizer
    )

    trainer.train()
    adapter_dir = Path(cfg.output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Log which checkpoint was saved. With load_best_model_at_end=True the
    # adapter should reflect the best dev_loss checkpoint; a missing
    # best_model_checkpoint indicates PEFT/load-best didn't fire and we saved
    # the LAST checkpoint instead — silent failure mode worth catching.
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        _console.print(f"[cyan]Saved adapter from best checkpoint:[/cyan] {best_ckpt}")
    else:
        _console.print(
            "[yellow]Warning: no best_model_checkpoint recorded — saved last "
            "checkpoint. If eval was enabled, this may indicate PEFT/load_best "
            "integration didn't fire.[/yellow]"
        )
    _console.print(f"[green]Saved adapter to[/green] {adapter_dir}")
    return adapter_dir

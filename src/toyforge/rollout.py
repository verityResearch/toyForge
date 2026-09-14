"""Shared generation helpers used by both eval (teacher-forced) and GRPO rollouts.

`gen` produces one greedy completion. `gen_batched` produces k sampled
completions in a single batched generate call. `render_step_outputs` orchestrates
both for a single trajectory.
"""

from __future__ import annotations

import torch  # noqa: TCH002 — runtime import for tensor ops

from toyforge.train.data import format_step_as_chat_messages


def gen(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
) -> str:
    """Generate one completion. `prompt_ids` must already be on the model's device.

    `temperature` is ignored when do_sample=False (greedy mode).
    """
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(prompt_ids, **gen_kwargs)
    return tokenizer.decode(out[0, prompt_ids.shape[1] :], skip_special_tokens=True)


def gen_batched(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens: int,
    k: int,
    temperature: float = 0.7,
) -> list[str]:
    """Generate k sampled completions in a single batched call.

    `prompt_ids` must be on the model's device. Memory cost: k × prompt_seq_len ×
    hidden_dim of KV-cache. At k=8 and typical prompts (<4096 tokens), well within
    a 24GB GPU.
    """
    if k <= 0:
        return []
    batch_prompt = prompt_ids.expand(k, -1)
    with torch.no_grad():
        out = model.generate(
            batch_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(out[i, prompt_ids.shape[1] :], skip_special_tokens=True) for i in range(k)
    ]


def render_step_outputs(
    model,
    tokenizer,
    trajectory: dict,
    k: int,
    max_new_tokens: int,
    temperature: float = 0.7,
) -> list[list[str]]:
    """For each step in a trajectory: 1 greedy + k sampled outputs.

    Returns list of length len(steps); each entry is [greedy, sample_0, ..., sample_{k-1}].
    """
    per_step: list[list[str]] = []
    for step in trajectory["steps"]:
        msgs = format_step_as_chat_messages(step)[:-1]  # strip gold assistant turn
        prompt_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_ids_dev = prompt_ids.to(model.device)
        greedy_text = gen(model, tokenizer, prompt_ids_dev, max_new_tokens, do_sample=False)
        sampled_texts = gen_batched(
            model, tokenizer, prompt_ids_dev, max_new_tokens, k=k, temperature=temperature
        )
        per_step.append([greedy_text, *sampled_texts])
    return per_step

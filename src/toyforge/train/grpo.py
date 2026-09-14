"""GRPO trainer for Phase 2.

Reads TrainConfig with `grpo: GRPOConfig` and `adapter_init: Path`. Loads the
Phase 1 SFT adapter as the initial policy, uses LoRA-adapter-zero as the
reference (memory-cheap), and runs TRL's GRPOTrainer with our verifier-as-reward.
"""

from __future__ import annotations

from pathlib import Path

from toyforge._console import console as _console
from toyforge.train.config import TrainConfig


def run_grpo(cfg: TrainConfig) -> Path:
    """Train a GRPO adapter starting from the SFT cold-start. Returns adapter dir."""
    if cfg.runtime == "llamacpp":
        from toyforge.train.llamacpp_grpo import run_llamacpp_grpo

        return run_llamacpp_grpo(cfg)

    if cfg.grpo is None:
        raise ValueError("method=grpo requires a `grpo:` block in the YAML config")
    if cfg.adapter_init is None:
        raise ValueError("method=grpo requires `adapter_init:` pointing to a Phase 1 SFT adapter")

    _console.print(
        f"[bold]GRPO[/bold] base={cfg.base_model} adapter_init={cfg.adapter_init} "
        f"k={cfg.grpo.num_generations} beta={cfg.grpo.beta} preset={cfg.grpo.rubric_preset}"
    )

    # TODO: implement (Phase 2 plan tasks 2-7). This skeleton makes the dispatch
    # surface complete so we can fail loudly with a clear message until then.
    raise NotImplementedError("GRPO trainer skeleton is in place; implementation pending.")

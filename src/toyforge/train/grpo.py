"""GRPO trainer for Phase 2 — NOT IMPLEMENTED YET.

What exists: config validation (`grpo:` block and `adapter_init:`), dispatch to the
llama.cpp-native variant, and the verifier-as-reward function in `toyforge.verifier.reward`.

What does not exist: the training loop. The intended design loads the Phase 1 SFT
adapter as the initial policy, uses a zero-scaled LoRA adapter as the reference, and
optimizes group-relative advantages computed from the verifier reward. Until that lands,
`run_grpo` validates its config and then raises NotImplementedError.
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

    # Not implemented: rollouts, group-relative advantages and the policy update. The
    # skeleton keeps config + dispatch real and fails loudly instead of pretending to train.
    raise NotImplementedError(
        "GRPO is not implemented yet: this is a config-and-dispatch skeleton. The verifier "
        "reward exists (toyforge.verifier.reward); the training loop does not."
    )

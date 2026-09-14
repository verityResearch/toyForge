"""Experimental native llama.cpp GRPO entrypoint.

This module intentionally fails closed until a local llama.cpp build exposes the
training/logprob surfaces needed for native GRPO. It exists so `runtime:
llamacpp` is a real, testable dispatch path with explicit feasibility checks
instead of an implicit fallback to the Transformers trainer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from toyforge._console import console as _console
from toyforge.train.config import TrainConfig


def _find_llama_binary() -> str | None:
    for name in ("llama-finetune", "finetune", "llama-cli"):
        found = shutil.which(name)
        if found:
            return found
    return None


def run_llamacpp_grpo(cfg: TrainConfig) -> Path:
    """Run the llama.cpp-native GRPO feasibility gate.

    Returns only after a real trainer is implemented. Today it validates config
    and local binary availability, then raises NotImplementedError with the
    missing native surfaces. This is deliberate: native llama.cpp training cannot
    be honestly emulated by the existing TRL skeleton.
    """
    if cfg.grpo is None:
        raise ValueError("runtime=llamacpp method=grpo requires a `grpo:` block")
    if cfg.llamacpp is None:
        raise ValueError("runtime=llamacpp requires a `llamacpp:` config block")
    if cfg.adapter_init is None:
        raise ValueError("runtime=llamacpp method=grpo requires `adapter_init:`")
    if not cfg.llamacpp.grammar_path.exists():
        raise FileNotFoundError(f"llama.cpp grammar not found: {cfg.llamacpp.grammar_path}")

    api_key = os.environ.get(cfg.llamacpp.api_key_env or "LLAMA_CPP_API_KEY", "no-key")
    binary = _find_llama_binary()
    if binary is None:
        raise RuntimeError(
            "runtime=llamacpp requires a local llama.cpp training binary "
            "(`llama-finetune`/`finetune`) on PATH"
        )

    try:
        version = subprocess.check_output(
            [binary, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError):
        version = "(version unavailable)"

    _console.print(
        "[bold]llama.cpp GRPO feasibility[/bold] "
        f"server={cfg.llamacpp.base_url} model={cfg.llamacpp.model} "
        f"grammar={cfg.llamacpp.grammar_path} api_key={'set' if api_key else 'unset'}"
    )
    _console.print(f"[dim]{binary}: {version.strip()}[/dim]")

    raise NotImplementedError(
        "llama.cpp-native GRPO needs policy/reference logprobs, group-relative "
        "advantage updates, and LoRA adapter update/export wired to llama.cpp. "
        "The config and dispatch path are in place; implement the native trainer "
        "once those local binary surfaces are confirmed."
    )

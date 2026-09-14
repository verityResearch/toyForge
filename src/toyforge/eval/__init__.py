"""Evaluation — load adapter, sample, verify, scorecard.

Lazy-import contract: `run_eval` is intentionally lazy and is ONLY exposed via
`__getattr__`. It pulls in torch/transformers/peft on first access. To keep
CPU-only tests fast (no `train` extra required), `from toyforge.eval import
run_eval` MUST be done inside a function body, not at module top-level. If you
write top-level `from toyforge.eval import run_eval`, the heavy import chain
runs at module-import time, breaking CPU-only test runs.
"""

from __future__ import annotations

from toyforge.eval.score import Scorecard

__all__ = ["Scorecard", "run_eval"]


def __getattr__(name: str):
    if name == "run_eval":
        from toyforge.eval.runner import run_eval  # noqa: PLC0415

        return run_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

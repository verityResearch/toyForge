"""TRL/Transformers training callbacks for toyForge.

The GRPO trainer (Phase 2) will produce a `Subscores` value per rollout. The
`SubscoreLoggerCallback` buffers a rolling window of subscores and emits the
per-subscore window mean into TRL's `logs` dict on each log step. From there
they propagate to TensorBoard (Task 7.7) and stdout.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from toyforge.verifier.types import Subscores


_SUBSCORE_NAMES = (
    "parse",
    "schema",
    "method_known",
    "precondition_met",
    "transition_valid",
    "sequence_optimal",
)


class SubscoreLoggerCallback(TrainerCallback):
    """Buffer per-rollout subscores; emit per-subscore window-mean on each on_log."""

    def __init__(self, window_size: int = 100) -> None:
        self._buffer: dict[str, deque[float]] = {
            name: deque(maxlen=window_size) for name in _SUBSCORE_NAMES
        }

    def record(self, subscores_batch: list[Subscores]) -> None:
        """Push a batch of Subscores into the rolling window. Called by the
        GRPO reward function alongside the scalar reward."""
        for sub in subscores_batch:
            for name, value in sub.as_dict().items():
                if name in self._buffer:
                    self._buffer[name].append(value)

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # noqa: ARG002 — TRL signature
        """TRL/HF hook. Inject window-mean for each subscore into the logs dict
        so TensorBoard and stdout see them under `subscore_mean/{name}`."""
        if logs is None:
            return
        for name, buf in self._buffer.items():
            if buf:
                logs[f"subscore_mean/{name}"] = sum(buf) / len(buf)

"""Types shared across verifier components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Subscores:
    """Subscore set for a single step, all normalized to [0, 1]."""

    parse: float = 0.0
    schema: float = 0.0
    method_known: float = 0.0
    precondition_met: float = 0.0
    transition_valid: float = 0.0
    sequence_optimal: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StepResult:
    """Verifier output for one step."""

    passed: bool
    subscores: Subscores
    parsed_thinking: str | None
    parsed_call: dict[str, Any] | None
    method: str | None
    new_state: str | None
    error_message: str | None


@dataclass(frozen=True)
class TrajectoryResult:
    """Verifier output for a full trajectory (composite)."""

    passed: bool  # every step.passed AND final state matches expectation
    steps: list[StepResult]
    final_state: str | None
    error_message: str | None


@dataclass(frozen=True)
class Rubric:
    """Reward rubric — subscore weights + aggregation."""

    weights: dict[str, float]
    aggregation: str = "mean"  # "mean" | "min" | "product"

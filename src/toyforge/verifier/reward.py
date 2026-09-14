"""Reward function — rubric-driven aggregation of Subscores into a scalar."""

from __future__ import annotations

from typing import Any

from toyforge.schemas import Schemas
from toyforge.verifier.core import verify_step
from toyforge.verifier.types import Rubric, Subscores


def compute_reward(subscores: Subscores, rubric: Rubric) -> float:
    """Linear combination: sum(weight_i * subscore_i).

    Subscore keys not present in `rubric.weights` are ignored. We do not
    renormalize — total reward depends on the rubric's weight magnitudes.
    """
    sub = subscores.as_dict()
    return sum(weight * sub.get(key, 0.0) for key, weight in rubric.weights.items())


def compute_trajectory_reward(
    step_subscores: list[Subscores],
    rubric: Rubric,
) -> float:
    """Aggregate per-step rewards into a trajectory-level scalar.

    Mode is governed by `rubric.aggregation`:
      - "mean":    average per-step reward
      - "min":     worst step (chain-integrity proxy)
      - "product": multiplicative; one bad step zeroes the whole trajectory
    """
    if not step_subscores:
        raise ValueError("empty step_subscores; trajectory reward undefined for zero steps")
    step_rewards = [compute_reward(s, rubric) for s in step_subscores]
    if rubric.aggregation == "mean":
        return sum(step_rewards) / len(step_rewards)
    if rubric.aggregation == "min":
        return min(step_rewards)
    if rubric.aggregation == "product":
        out = 1.0
        for r in step_rewards:
            out *= r
        return out
    raise ValueError(f"unknown rubric.aggregation: {rubric.aggregation!r}")


def compute_grpo_reward(
    prompt_context: dict[str, Any],
    model_output: str,
    schemas: Schemas,
    rubric: Rubric,
) -> tuple[float, Subscores]:
    """Reward for a single GRPO rollout step. Uses `infer_trigger=True` so the
    verifier evaluates against what the model actually called, not against a
    pinned gold trigger that may differ from the model's choice.

    `expected_trigger` is stripped from the prompt context before verification —
    seed-derived data carries gold triggers, but GRPO is graded against the
    model's actual choice, not the seed's expectation.
    """
    # Strip expected_trigger so the infer_trigger path is always taken, even
    # when the caller passes a seed-derived context that contains a gold trigger.
    # Without this, `infer_trigger=True` is silently bypassed (the gate is
    # `if infer_trigger and not expected_trigger`), causing valid rollouts that
    # diverge from the gold trigger to receive transition_valid=0.
    ctx = {k: v for k, v in prompt_context.items() if k != "expected_trigger"}
    step_result = verify_step(ctx, model_output, schemas, infer_trigger=True)
    return compute_reward(step_result.subscores, rubric), step_result.subscores


def rubric_from_preset(schemas: Schemas, preset_name: str) -> Rubric:
    """Build a Rubric from a named preset in schemas/reward-rubric.yaml."""
    if preset_name not in schemas.rubric_presets:
        raise KeyError(f"unknown rubric preset: {preset_name!r}")
    return Rubric(
        weights=dict(schemas.rubric_presets[preset_name]),
        aggregation=schemas.rubric_aggregation,
    )

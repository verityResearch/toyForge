"""Tests for trajectory-level reward aggregation honoring rubric.aggregation."""

from __future__ import annotations

import pytest

from toyforge.verifier.reward import compute_reward, compute_trajectory_reward
from toyforge.verifier.types import Rubric, Subscores


def _sub(transition_valid: float, parse: float = 1.0, schema: float = 1.0) -> Subscores:
    return Subscores(
        parse=parse,
        schema=schema,
        method_known=1.0,
        precondition_met=1.0,
        transition_valid=transition_valid,
        sequence_optimal=transition_valid,
    )


def test_mean_aggregation():
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation="mean")
    steps = [_sub(1.0), _sub(0.0), _sub(1.0)]
    assert compute_trajectory_reward(steps, rubric) == pytest.approx(2 / 3)


def test_min_aggregation():
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation="min")
    steps = [_sub(1.0), _sub(0.0), _sub(1.0)]
    assert compute_trajectory_reward(steps, rubric) == 0.0


def test_product_aggregation():
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation="product")
    steps = [_sub(1.0), _sub(0.5), _sub(0.5)]
    assert compute_trajectory_reward(steps, rubric) == pytest.approx(0.25)


@pytest.mark.parametrize("agg", ["mean", "min", "product"])
def test_empty_trajectory_raises(agg):
    """Empty step_subscores raises ValueError under every aggregation mode —
    safer than returning a per-mode identity (0.0 for mean/min, 1.0 for product)
    because callers must explicitly handle the empty case."""
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation=agg)
    with pytest.raises(ValueError, match="empty"):
        compute_trajectory_reward([], rubric)


def test_unknown_aggregation_raises():
    rubric = Rubric(weights={"transition_valid": 1.0}, aggregation="median")
    with pytest.raises(ValueError, match="median"):
        compute_trajectory_reward([_sub(1.0)], rubric)


def test_single_step_trajectory_matches_compute_reward():
    """For a 1-step trajectory under mean aggregation, the result equals compute_reward."""
    rubric = Rubric(
        weights={"parse": 0.5, "transition_valid": 0.5},
        aggregation="mean",
    )
    sub = _sub(0.8)
    assert compute_trajectory_reward([sub], rubric) == pytest.approx(compute_reward(sub, rubric))

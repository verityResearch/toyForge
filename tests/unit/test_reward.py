"""Tests for compute_reward."""

from __future__ import annotations

import pytest

from toyforge.schemas import load_schemas
from toyforge.verifier.reward import compute_reward, rubric_from_preset
from toyforge.verifier.types import Rubric, Subscores


@pytest.fixture(scope="module")
def schemas(schemas_dir):
    return load_schemas(schemas_dir)


def test_default_rubric_perfect_subscores_score_1(schemas):
    r = rubric_from_preset(schemas, "shaped")
    s = Subscores(
        parse=1.0,
        schema=1.0,
        method_known=1.0,
        precondition_met=1.0,
        transition_valid=1.0,
        sequence_optimal=1.0,
    )
    assert compute_reward(s, r) == pytest.approx(1.0)


def test_binary_preset_uses_only_transition(schemas):
    r = rubric_from_preset(schemas, "binary")
    perfect_but_no_transition = Subscores(
        parse=1.0,
        schema=1.0,
        method_known=1.0,
        precondition_met=1.0,
        transition_valid=0.0,
        sequence_optimal=1.0,
    )
    assert compute_reward(perfect_but_no_transition, r) == 0.0

    only_transition = Subscores(transition_valid=1.0)
    assert compute_reward(only_transition, r) == pytest.approx(1.0)


def test_unknown_preset_raises(schemas):
    with pytest.raises(KeyError):
        rubric_from_preset(schemas, "nonexistent")


def test_custom_rubric_weights():
    custom = Rubric(weights={"parse": 0.5, "schema": 0.5}, aggregation="mean")
    s = Subscores(parse=1.0, schema=0.0)
    assert compute_reward(s, custom) == pytest.approx(0.5)


def test_weights_not_summing_to_one_still_work():
    r = Rubric(weights={"parse": 0.3, "schema": 0.3}, aggregation="mean")
    s = Subscores(parse=1.0, schema=1.0)
    assert compute_reward(s, r) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "preset_name,subscores_kwargs,expected_reward",
    [
        # shaped: default weighted sum across all six subscores → 1.0 when perfect
        (
            "shaped",
            {
                "parse": 1.0,
                "schema": 1.0,
                "method_known": 1.0,
                "precondition_met": 1.0,
                "transition_valid": 1.0,
                "sequence_optimal": 1.0,
            },
            1.0,
        ),
        # binary: only transition_valid counts (weight=1.0); partial credit allowed
        (
            "binary",
            {
                "parse": 1.0,
                "schema": 1.0,
                "method_known": 1.0,
                "precondition_met": 1.0,
                "transition_valid": 0.5,
                "sequence_optimal": 1.0,
            },
            0.5,
        ),
        # schema_only: parse (0.5) + schema (0.5); schema=0 zeroes half
        (
            "schema_only",
            {
                "parse": 1.0,
                "schema": 0.0,
                "method_known": 1.0,
                "precondition_met": 1.0,
                "transition_valid": 1.0,
                "sequence_optimal": 1.0,
            },
            0.5,
        ),
        # schema_only: parse=0 + schema=0 → 0
        (
            "schema_only",
            {
                "parse": 0.0,
                "schema": 0.0,
                "method_known": 1.0,
                "precondition_met": 1.0,
                "transition_valid": 1.0,
                "sequence_optimal": 1.0,
            },
            0.0,
        ),
    ],
)
def test_compute_reward_preset(preset_name, subscores_kwargs, expected_reward, schemas):
    """Each rubric preset weights subscores per its declared spec in
    schemas/reward-rubric.yaml — catches drift between YAML and consumer."""
    rubric = rubric_from_preset(schemas, preset_name)
    sub = Subscores(**subscores_kwargs)
    assert compute_reward(sub, rubric) == pytest.approx(expected_reward)

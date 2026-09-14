"""Cascade invariants over Subscores."""

from __future__ import annotations

from hypothesis import given

from toyforge.verifier.core import verify_step

from .strategies import (
    malformed_outputs,
    model_outputs,
    prompt_contexts,
    schemas,
)


@given(ctx=prompt_contexts(), output=malformed_outputs())
def test_parse_failure_zeros_everything(ctx, output):
    """If parse == 0, every other subscore must be 0."""
    r = verify_step(ctx, output, schemas())
    s = r.subscores
    if s.parse == 0.0:
        assert s.schema == 0.0
        assert s.method_known == 0.0
        assert s.precondition_met == 0.0
        assert s.transition_valid == 0.0
        assert s.sequence_optimal == 0.0


@given(ctx=prompt_contexts(), output=model_outputs())
def test_method_unknown_zeros_semantic_subscores(ctx, output):
    """If method_known == 0, schema/precondition/transition/sequence must be 0."""
    r = verify_step(ctx, output, schemas())
    s = r.subscores
    if s.parse == 1.0 and s.method_known == 0.0:
        assert s.schema == 0.0
        assert s.precondition_met == 0.0
        assert s.transition_valid == 0.0
        assert s.sequence_optimal == 0.0


@given(ctx=prompt_contexts(), output=model_outputs())
def test_passed_iff_all_six_subscores_one(ctx, output):
    """passed must be True iff every subscore is 1.0.

    Note: the verifier's `passed` predicate explicitly checks only four
    subscores (parse / schema / precondition_met / transition_valid). The
    biconditional with all six holds because (a) method_known=0 short-circuits
    to early-return passed=False, and (b) sequence_optimal is derived from
    transition_valid. If a future refactor decouples sequence_optimal from
    transition_valid, this test will (correctly) flag the regression.
    """
    r = verify_step(ctx, output, schemas())
    s = r.subscores
    all_one = (
        s.parse == 1.0
        and s.schema == 1.0
        and s.method_known == 1.0
        and s.precondition_met == 1.0
        and s.transition_valid == 1.0
        and s.sequence_optimal == 1.0
    )
    assert r.passed == all_one, (
        f"passed={r.passed} but all_one={all_one} for subscores={s.as_dict()}"
    )

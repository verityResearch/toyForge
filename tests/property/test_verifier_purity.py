"""verify_step is a pure function: same inputs → same output, always."""

from __future__ import annotations

from hypothesis import given

from toyforge.verifier.core import verify_step

from .strategies import model_outputs, prompt_contexts, schemas


@given(ctx=prompt_contexts(), output=model_outputs())
def test_verify_step_is_deterministic(ctx, output):
    r1 = verify_step(ctx, output, schemas())
    r2 = verify_step(ctx, output, schemas())
    assert r1 == r2, f"verifier non-deterministic on ctx={ctx!r}, output={output!r}"


@given(ctx=prompt_contexts(), output=model_outputs())
def test_verify_step_does_not_mutate_inputs(ctx, output):
    before_ctx = dict(ctx)
    before_output = output
    verify_step(ctx, output, schemas())
    assert ctx == before_ctx
    assert output == before_output

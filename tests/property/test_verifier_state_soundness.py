"""When transition_valid == 1, the recorded new_state must match the schema."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from toyforge.verifier.core import verify_step

from .strategies import schemas, state_names


def _all_valid_step_outputs():
    """Enumerate every (prior_state, method, trigger) where the schema allows the
    transition, and yield a synthesized model_output exercising it.

    Uses param payloads that are intentionally too sparse to validate against
    the JSON Schema in most cases — so we focus this test on transition_valid /
    new_state consistency, not on schema correctness.
    """
    s = schemas()
    cases = []
    for prior in s.states:
        for method, triggers in s.method_triggers.items():
            is_query = len(triggers) == 0
            if is_query:
                cases.append((prior, method, None, prior))  # query: new_state == prior
            else:
                for trig in triggers:
                    if (prior, trig) in s.transitions:
                        cases.append((prior, method, trig, s.transitions[(prior, trig)]))
    return cases


_CASES = _all_valid_step_outputs()

# Methods with at least one trigger (i.e., non-query). Used by the negative-case
# test so query methods aren't silently skipped by an early return.
_TRANSITING_METHODS = sorted(
    method for method, triggers in schemas().method_triggers.items() if triggers
)


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[f"{prior}|{method}|{trigger}" for prior, method, trigger, _ in _CASES],
)
def test_valid_transitions_produce_matching_new_state(case):
    """For every (prior, method, expected_trigger) the schema accepts,
    StepResult.new_state must equal what schemas dictate.

    Parametrize (not Hypothesis) here: the case set is small and finite, and
    we want exhaustive coverage of every transition in the table.
    """
    prior, method, trigger, expected_new = case

    # Use empty params; we'll get schema=0 for non-query methods, but
    # transition_valid is computed independently.
    ctx = {"prior_state": prior}
    if trigger is not None:
        ctx["expected_trigger"] = trigger

    call = {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1}
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas())

    if r.subscores.transition_valid == 1.0:
        assert r.new_state == expected_new, (
            f"case={(prior, method, trigger)}: expected new_state={expected_new!r} "
            f"got {r.new_state!r}"
        )


@given(prior=state_names(), method=st.sampled_from(_TRANSITING_METHODS))
def test_mismatched_trigger_drops_transition_valid(prior, method):
    """If we claim a trigger the method cannot emit, transition_valid must not be 1.

    The bogus trigger string is chosen outside the `<method>.<outcome>` namespace
    used by all real triggers, so it's guaranteed never to appear in the table.
    """
    bogus_trigger = "definitely.not.a.real.trigger.string"

    ctx = {"prior_state": prior, "expected_trigger": bogus_trigger}
    call = {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1}
    output = f"<think>x</think>{json.dumps(call)}"
    r = verify_step(ctx, output, schemas())
    assert r.subscores.transition_valid != 1.0

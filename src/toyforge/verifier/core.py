"""Pure-function step verifier — all stages.

Pure function (no I/O, no model). Determinstic, exhaustively unit-testable.
Two consumer modes: eval grader (uses StepResult), GRPO reward (uses subscores).
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema.exceptions import ValidationError

from toyforge.schemas import Schemas
from toyforge.verifier.types import StepResult, Subscores

# Non-greedy match: splits at the FIRST </think>. If the thinking text itself
# contains </think> (e.g., the model is explaining XML syntax), parsing splits at
# the wrong boundary and the JSON tail looks malformed. This is rare in practice;
# if you see "json after </think> did not parse" with model output containing
# valid JSON, check for nested </think> in the thinking content.
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)


def verify_step(
    prompt_context: dict[str, Any],
    model_output: str,
    schemas: Schemas,
    infer_trigger: bool = False,
) -> StepResult:
    """Verify a single step of a trajectory.

    `prompt_context` must contain `prior_state`. Optional: `expected_trigger`,
    `expected_state_after`.

    When `infer_trigger=True` and no `expected_trigger` is provided in
    `prompt_context`, the verifier infers the trigger from the model's method
    call: it finds all triggers the method can emit that have a valid arc from
    `prior_state`. If one or more candidates exist, it awards `transition_valid=1`
    and sets `new_state` accordingly. This is the GRPO mode — it evaluates the
    model's actual call rather than requiring a pinned gold trigger.
    """
    # ---- Parse ----
    m = _THINK_RE.match(model_output)
    if not m:
        return _fail("missing or malformed <think>…</think> wrapper")

    thinking, after = m.group(1), m.group(2).strip()
    try:
        call = json.loads(after)
    except json.JSONDecodeError as e:
        return _fail(f"json after </think> did not parse: {e}")

    if not isinstance(call, dict):
        return _fail("json after </think> is not an object")

    required_top = {"jsonrpc", "method", "params", "id"}
    if not required_top.issubset(call.keys()):
        return _fail(f"missing top-level keys: {required_top - call.keys()}")
    if call.get("jsonrpc") != "2.0":
        return _fail("jsonrpc must be '2.0'")

    parse_score = 1.0
    parsed_thinking = thinking
    parsed_call = call

    # ---- Method known ----
    method = call.get("method")
    if method is None:
        return StepResult(
            passed=False,
            subscores=Subscores(parse=parse_score, method_known=0.0),
            parsed_thinking=parsed_thinking,
            parsed_call=parsed_call,
            method=None,
            new_state=None,
            error_message="method is null in JSON-RPC body",
        )
    if method not in schemas.method_names:
        return StepResult(
            passed=False,
            subscores=Subscores(parse=parse_score, method_known=0.0),
            parsed_thinking=parsed_thinking,
            parsed_call=parsed_call,
            method=method,
            new_state=None,
            error_message=f"unknown method: {method!r}",
        )

    # ---- Schema ----
    params = call.get("params", {})
    schema_err: str | None = None
    try:
        schemas.method_validators[method].validate(params)
        schema_score = 1.0
    except ValidationError as e:
        schema_score = 0.0
        schema_err = e.message

    # ---- State logic ----
    prior_state = prompt_context.get("prior_state")
    if prior_state is None:
        return _fail("prompt_context missing 'prior_state'")

    triggers = schemas.method_triggers.get(method, [])
    is_query = len(triggers) == 0  # query methods don't change state

    expected_trigger = prompt_context.get("expected_trigger")
    expected_state_after = prompt_context.get("expected_state_after")

    precondition_met = 0.0
    transition_valid = 0.0
    new_state: str | None = None

    if is_query:
        precondition_met = 1.0
        transition_valid = 1.0
        new_state = prior_state
    else:
        plausible_triggers = [t for t in triggers if (prior_state, t) in schemas.transitions]
        if plausible_triggers:
            precondition_met = 1.0
        # transition_valid requires three things:
        #  (a) the method can actually emit this trigger (per method_triggers)
        #  (b) the trigger has an arc from prior_state in the transition table
        #  (c) expected_state_after matches the resulting state (if provided)
        if infer_trigger and not expected_trigger:
            # Find any trigger the method can emit that has a valid arc from prior_state.
            candidates = [t for t in triggers if (prior_state, t) in schemas.transitions]
            if candidates:
                # Prefer one whose resulting state matches expected_state_after.
                if expected_state_after:
                    matching = [
                        t
                        for t in candidates
                        if schemas.transitions[(prior_state, t)] == expected_state_after
                    ]
                    if matching:
                        candidates = matching
                inferred = candidates[0]
                resulting = schemas.transitions[(prior_state, inferred)]
                if expected_state_after is None or resulting == expected_state_after:
                    transition_valid = 1.0
                    new_state = resulting
        elif (
            expected_trigger
            and expected_trigger in triggers
            and (prior_state, expected_trigger) in schemas.transitions
        ):
            resulting = schemas.transitions[(prior_state, expected_trigger)]
            if expected_state_after is None or resulting == expected_state_after:
                transition_valid = 1.0
                new_state = resulting

    sequence_optimal = 1.0 if transition_valid == 1.0 or is_query else 0.0

    if transition_valid == 0.0 and not is_query and schema_err is None:
        if infer_trigger and not expected_trigger:
            candidates = [t for t in triggers if (prior_state, t) in schemas.transitions]
            if not candidates:
                schema_err = (
                    f"transition_valid=0: no valid arc inferred for method={method!r} "
                    f"from prior_state={prior_state!r} "
                    f"(method emits {triggers!r}; none have an arc from this state)"
                )
            else:
                resulting_states = [schemas.transitions[(prior_state, t)] for t in candidates]
                schema_err = (
                    f"transition_valid=0: method={method!r} from prior_state={prior_state!r} "
                    f"has arcs via {candidates!r} leading to {resulting_states!r}, "
                    f"but none matches expected_state_after={expected_state_after!r}"
                )
        else:
            schema_err = (
                f"transition_valid=0: expected_trigger={expected_trigger!r} "
                f"from prior_state={prior_state!r} did not validate "
                f"(method={method!r} emits {triggers!r})"
            )

    subscores = Subscores(
        parse=parse_score,
        schema=schema_score,
        method_known=1.0,
        precondition_met=precondition_met,
        transition_valid=transition_valid,
        sequence_optimal=sequence_optimal,
    )

    passed = (
        parse_score == 1.0
        and schema_score == 1.0
        and precondition_met == 1.0
        and transition_valid == 1.0
    )

    return StepResult(
        passed=passed,
        subscores=subscores,
        parsed_thinking=parsed_thinking,
        parsed_call=parsed_call,
        method=method,
        new_state=new_state,
        error_message=schema_err,
    )


def _fail(msg: str) -> StepResult:
    return StepResult(
        passed=False,
        subscores=Subscores(),
        parsed_thinking=None,
        parsed_call=None,
        method=None,
        new_state=None,
        error_message=msg,
    )

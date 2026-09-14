"""Trajectory-level verifier (composite over per-step results)."""

from __future__ import annotations

from typing import Any

from toyforge.schemas import Schemas
from toyforge.verifier.core import verify_step
from toyforge.verifier.types import StepResult, TrajectoryResult


def verify_trajectory(
    trajectory: dict[str, Any],
    model_outputs: list[str],
    schemas: Schemas,
) -> TrajectoryResult:
    """Verify a full multi-step trajectory.

    `trajectory` is a dict matching `schemas/trajectory-schema.yaml` shape
    (used for initial_state, expected_trigger/expected_state_after per step).
    `model_outputs` is the parallel list of model-rendered output strings.
    """
    if len(model_outputs) != len(trajectory["steps"]):
        return TrajectoryResult(
            passed=False,
            steps=[],
            final_state=None,
            error_message=(
                f"length mismatch: {len(model_outputs)} outputs"
                f" vs {len(trajectory['steps'])} expected steps"
            ),
        )

    if not model_outputs and not trajectory.get("steps"):
        return TrajectoryResult(
            passed=False,
            steps=[],
            final_state=trajectory.get("initial_state"),
            error_message="empty trajectory (no steps)",
        )

    state = trajectory["initial_state"]
    results: list[StepResult] = []

    for spec_step, model_output in zip(trajectory["steps"], model_outputs, strict=True):
        ctx: dict[str, Any] = {
            "prior_state": state,
            "expected_trigger": spec_step.get("expected_trigger"),
            "expected_state_after": spec_step.get("expected_state_after"),
        }
        r = verify_step(ctx, model_output, schemas)
        results.append(r)
        if r.passed and r.new_state is not None:
            state = r.new_state

    passed = all(r.passed for r in results)
    expected_final = trajectory.get("final_state")
    if passed and expected_final is not None and state != expected_final:
        passed = False
        err = f"final state {state!r} does not match expected {expected_final!r}"
    else:
        err = None

    if not all(s.passed for s in results) and err is None:
        first_fail = next((i for i, s in enumerate(results) if not s.passed), None)
        if first_fail is not None:
            err = f"step {first_fail} failed: {results[first_fail].error_message or '?'}"

    return TrajectoryResult(
        passed=passed,
        steps=results,
        final_state=state,
        error_message=err,
    )

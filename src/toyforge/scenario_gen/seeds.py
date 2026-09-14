"""Load hand-seeded trajectories from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_seeds(path: Path) -> list[dict[str, Any]]:
    """Load `scenarios/seeds.yaml` (or any compatible file). Returns a list of
    trajectory dicts. Raises ValueError if any seed is structurally invalid.

    When `path`'s parent has a sibling `schemas/` directory, performs additional
    cross-schema validation: initial_state must be a known state, and query-only
    methods must NOT carry expected_trigger (the seed's own trigger claim would
    otherwise drive transition_valid=1, silently inflating test-set scores).
    """
    text = Path(path).read_text()
    doc = yaml.safe_load(text) or {}
    seeds = doc.get("seeds", [])

    # Optionally load the canonical schemas to perform cross-schema validation.
    # Fixture YAMLs in tests typically don't have a schemas/ sibling — skip those.
    schemas_dir = Path(path).parent.parent / "schemas"
    schemas = None
    if schemas_dir.exists() and (schemas_dir / "state-machine.yaml").exists():
        from toyforge.schemas import load_schemas as _load_schemas

        schemas = _load_schemas(schemas_dir)

    for i, s in enumerate(seeds):
        if not isinstance(s, dict):
            raise ValueError(f"seed[{i}]: expected mapping, got {type(s).__name__}")
        if "trajectory_id" not in s:
            raise ValueError(f"seed[{i}]: missing 'trajectory_id'")
        tid = s["trajectory_id"]
        if "initial_state" not in s:
            raise ValueError(f"seed[{tid!r}]: missing 'initial_state'")
        if schemas is not None and s["initial_state"] not in schemas.states:
            raise ValueError(
                f"seed[{tid!r}]: unknown initial_state {s['initial_state']!r}; "
                f"valid: {sorted(schemas.states)}"
            )
        if "steps" not in s:
            raise ValueError(f"seed[{tid!r}]: missing 'steps' list")
        if not isinstance(s["steps"], list):
            raise ValueError(
                f"seed[{tid!r}]: 'steps' must be a list, got {type(s['steps']).__name__}"
            )
        if schemas is not None:
            for j, step in enumerate(s["steps"]):
                tc = step.get("tool_call") or {}
                method = tc.get("method")
                if method and method in schemas.method_triggers:
                    is_query = len(schemas.method_triggers[method]) == 0
                    if is_query and "expected_trigger" in step:
                        raise ValueError(
                            f"seed[{tid!r}].steps[{j}]: query method "
                            f"{method!r} must not have 'expected_trigger'"
                        )
        s.setdefault("source", "hand_seed")
        # Auto-populate prior_calls from preceding steps' tool_calls.
        # Seeds that already declared prior_calls keep them (override-friendly).
        for j, step in enumerate(s["steps"]):
            pc = step.setdefault("prompt_context", {})
            if "prior_calls" not in pc and j > 0:
                pc["prior_calls"] = [
                    prior_step.get("tool_call")
                    for prior_step in s["steps"][:j]
                    if prior_step.get("tool_call")
                ]
    return seeds

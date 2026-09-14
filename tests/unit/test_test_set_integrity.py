"""Test-set integrity: every entry in scenarios/seeds.yaml must pass the verifier
when rendered as a perfect-fidelity trajectory. Acts as an independent gate on
top of load_seeds's structural validation."""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.scenario_gen.seeds import load_seeds
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory


def test_every_seed_passes_perfect_fidelity_verification():
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")
    seeds = load_seeds(repo / "scenarios" / "seeds.yaml")
    failures: list[tuple[str, str]] = []
    for s in seeds:
        outputs = [
            f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"
            for step in s["steps"]
        ]
        r = verify_trajectory(s, outputs, schemas)
        if not r.passed:
            failures.append((s["trajectory_id"], r.error_message or "?"))
    assert not failures, f"{len(failures)} seeds fail perfect-fidelity verification:\n" + "\n".join(
        f"  - {tid}: {err}" for tid, err in failures
    )

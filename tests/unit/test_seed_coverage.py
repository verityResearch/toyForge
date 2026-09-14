"""Coverage assertion: every JSON-RPC method must appear in scenarios/seeds.yaml."""

from __future__ import annotations

from pathlib import Path

from toyforge.scenario_gen.seeds import load_seeds
from toyforge.schemas import load_schemas


def test_every_method_appears_in_at_least_one_seed():
    repo = Path(__file__).resolve().parents[2]
    schemas = load_schemas(repo / "schemas")
    seeds = load_seeds(repo / "scenarios" / "seeds.yaml")
    methods_in_seeds: set[str] = set()
    for s in seeds:
        for step in s.get("steps", []):
            tc = step.get("tool_call")
            if isinstance(tc, dict) and "method" in tc:
                methods_in_seeds.add(tc["method"])
    missing = schemas.method_names - methods_in_seeds
    assert not missing, f"methods missing from seeds.yaml: {sorted(missing)}"


def test_seed_count_at_least_minimal():
    """Sanity: seed file should contain a non-trivial number of seeds."""
    repo = Path(__file__).resolve().parents[2]
    seeds = load_seeds(repo / "scenarios" / "seeds.yaml")
    assert len(seeds) >= 9, f"only {len(seeds)} seeds — expect at least one per method"


def test_at_least_one_multistep_seed_starts_from_draft():
    """Phase 2 needs at least one multi-step seed exercising NEW→TRIAGED
    so per-step GRPO can learn the only reachable forward transition."""
    repo = Path(__file__).resolve().parents[2]
    seeds = load_seeds(repo / "scenarios" / "seeds.yaml")
    multistep_draft = [
        s for s in seeds if s.get("initial_state") == "NEW" and len(s.get("steps", [])) >= 2
    ]
    assert multistep_draft, "no multi-step seed starts from NEW (Phase 2 blocker)"

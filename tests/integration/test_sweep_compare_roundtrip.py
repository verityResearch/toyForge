"""Integration test: sweep generates configs → mock scorecard JSONs → compare_runs
correctly aggregates them. This catches naming-convention coupling between
scripts/sweep.py's slug format and compare.py's -seed{N} pattern."""

from __future__ import annotations

import json

from scripts.sweep import generate_sweep_configs
from toyforge.eval.compare import compare_runs


def _mock_scorecard(name: str, pass_at_1: float) -> dict:
    """Minimal scorecard JSON matching the shape compare_runs expects."""
    return {
        "run_name": name,
        "n_examples": 30,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_1 + 0.1,
        "pass_at_maj": pass_at_1 + 0.05,
        "k": 8,
        "failure_modes": {},
    }


def test_sweep_generates_seed_suffixed_runs_compare_can_aggregate(tmp_path):
    """A sweep with a `seed` axis produces runs named `{spec_stem}-seed{N}`.
    compare_runs should detect these as members of one arm family and produce
    an Arm summary section."""
    # Set up sweep spec
    tc_dir = tmp_path / "train_configs"
    tc_dir.mkdir()
    (tc_dir / "phase1-sft.yaml").write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\nmethod: vanilla\n"
        "lora:\n  r: 32\n  alpha: 64\n  target_modules: [q_proj]\n  dropout: 0.05\n"
    )

    spec_path = tmp_path / "phase3.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  seed: [42, 7, 123]\n")

    gen_dir = tc_dir / ".generated"
    configs = generate_sweep_configs(spec_path, gen_dir)
    assert len(configs) == 3

    # Simulate training run: write mock scorecard JSON per config to reports/
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    # Sweep generates configs as phase3-seed42.yaml, etc.; assume the train job
    # uses the config stem as run_name (matches typical workflow):
    # e.g. ["phase3-seed42", "phase3-seed7", "phase3-seed123"]
    run_names = [c.stem for c in configs]
    for i, name in enumerate(run_names):
        sc = _mock_scorecard(name, pass_at_1=0.5 + 0.05 * i)
        (reports_dir / f"{name}.json").write_text(json.dumps(sc))

    # Now compare
    md = compare_runs(reports_dir)
    assert "## Arm summary" in md, "sweep+compare arm detection broken"
    assert "phase3" in md  # family name (without -seedN)


def test_sweep_run_names_align_with_compare_split_run_name(tmp_path):
    """The slug format `phase3-seed42` produced by sweep must be parseable by
    compare's `_split_run_name` regex."""
    from toyforge.eval.compare import _split_run_name

    # Generate a config to see the actual slug format
    tc_dir = tmp_path / "train_configs"
    tc_dir.mkdir()
    (tc_dir / "phase1-sft.yaml").write_text(
        "base_model: x\nmethod: vanilla\nlora:\n  r: 8\n  alpha: 16\n"
        "  target_modules: [q_proj]\n  dropout: 0.05\n"
    )
    spec_path = tmp_path / "phase3.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  seed: [42]\n")
    configs = generate_sweep_configs(spec_path, tc_dir / ".generated")
    # The stem is the run name; compare must successfully split it
    stem = configs[0].stem  # phase3-seed42
    family, seed = _split_run_name(stem)
    assert seed == 42, f"compare can't extract seed from sweep slug {stem!r}"
    assert family == "phase3", f"unexpected family extracted: {family!r}"

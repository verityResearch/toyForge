"""Tests for eval.compare.compare_runs — defensive read of reports/.

Includes Task 7.4 tests: delta + top-failures columns, regression bold.
"""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.eval.compare import compare_runs

# ---------------------------------------------------------------------------
# Task 7.4 helpers
# ---------------------------------------------------------------------------


def _write_report(reports_dir: Path, run_name: str, **kwargs) -> None:
    """Write a minimal scorecard JSON. Caller supplies pass_at_1, n_examples, failure_modes."""
    data = {
        "run_name": run_name,
        "n_examples": kwargs.get("n_examples", 30),
        "pass_at_1": kwargs.get("pass_at_1", 0.5),
        "pass_at_k": kwargs.get("pass_at_k", 0.7),
        "pass_at_maj": kwargs.get("pass_at_maj", 0.6),
        "k": kwargs.get("k", 8),
        "failure_modes": kwargs.get("failure_modes", {}),
    }
    (reports_dir / f"{run_name}.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Task 7.4 tests
# ---------------------------------------------------------------------------


def test_compare_runs_includes_delta_column_when_baseline_set(tmp_path):
    _write_report(tmp_path, "baseline-run", pass_at_1=0.6)
    _write_report(tmp_path, "improved-run", pass_at_1=0.7)
    md = compare_runs(tmp_path, baseline="baseline-run")
    assert "Δpass@1" in md
    assert "(base)" in md  # baseline row
    assert "+0.100" in md  # improved row delta
    # No bold on the improved row
    assert "**0.700**" not in md


def test_compare_runs_bolds_regression(tmp_path):
    _write_report(tmp_path, "baseline-run", pass_at_1=0.6)
    _write_report(tmp_path, "worse-run", pass_at_1=0.5)
    md = compare_runs(tmp_path, baseline="baseline-run")
    assert "-0.100" in md
    # Regression row is bolded
    assert "**0.500**" in md
    # Baseline row is NOT bolded
    assert "**0.600**" not in md


def test_compare_runs_top_failures_column(tmp_path):
    _write_report(
        tmp_path,
        "run-with-fails",
        pass_at_1=0.4,
        failure_modes={"transition_valid": 8, "schema": 4, "parse": 2, "method_known": 1},
    )
    md = compare_runs(tmp_path)
    # top 3 failures rendered
    assert "transition_valid(8)" in md
    assert "schema(4)" in md
    assert "parse(2)" in md
    # 4th is omitted from the main table top-failures column (but may appear in chart)
    table_section = md.split("## Failure mode chart")[0]
    assert "method_known" not in table_section


def test_compare_runs_empty_failure_modes_renders_dash(tmp_path):
    _write_report(tmp_path, "perfect-run", pass_at_1=1.0, failure_modes={})
    md = compare_runs(tmp_path)
    # Find the row for perfect-run and verify the top failures cell is "-"
    perfect_line = next(line for line in md.split("\n") if "perfect-run" in line)
    # Last cell of the table row is top failures
    cells = [c.strip() for c in perfect_line.split("|") if c.strip()]
    assert cells[-1] == "-"


def test_compare_runs_no_baseline_no_delta_column_filled(tmp_path):
    """Without --baseline, the delta column shows empty strings (or is dropped).
    Either is acceptable as long as the table is still well-formed."""
    _write_report(tmp_path, "solo-run", pass_at_1=0.5)
    md = compare_runs(tmp_path)
    # Just verify the run shows up correctly
    assert "solo-run" in md
    assert "0.500" in md


def test_compare_runs_baseline_not_found_renders_question_marks(tmp_path):
    """If --baseline is specified but no matching report exists, deltas are '?'."""
    _write_report(tmp_path, "run-a", pass_at_1=0.6)
    _write_report(tmp_path, "run-b", pass_at_1=0.7)
    md = compare_runs(tmp_path, baseline="run-nonexistent")
    # No (base) marker since no row matches baseline
    assert "(base)" not in md
    # Deltas should all be ? since baseline is unknown
    assert "?" in md


def test_compare_tolerates_missing_optional_keys(tmp_path):
    """A report missing 'k' or 'failure_modes' must not crash the rollup."""
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    (rpt_dir / "old_format.json").write_text(
        json.dumps(
            {
                "run_name": "old",
                "n_examples": 10,
                "pass_at_1": 0.5,
                "pass_at_k": 0.7,
                "pass_at_maj": 0.6,
                # missing "k" and "failure_modes"
            }
        )
    )
    md = compare_runs(rpt_dir)
    assert "old" in md


def test_compare_skips_unparseable_json(tmp_path):
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    (rpt_dir / "junk.json").write_text("not valid json {{{")
    (rpt_dir / "ok.json").write_text(
        json.dumps(
            {
                "run_name": "ok",
                "n_examples": 5,
                "pass_at_1": 1.0,
                "pass_at_k": 1.0,
                "pass_at_maj": 1.0,
                "k": 4,
                "failure_modes": {},
            }
        )
    )
    md = compare_runs(rpt_dir)
    assert "ok" in md


def test_compare_empty_dir_returns_no_reports_message(tmp_path):
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    md = compare_runs(rpt_dir)
    assert "no reports" in md.lower()


def test_compare_handles_explicit_null_run_name(tmp_path):
    """A report with `run_name: null` should not render the string 'None'."""
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    (rpt_dir / "weird.json").write_text(
        json.dumps(
            {
                "run_name": None,
                "n_examples": 1,
                "pass_at_1": 1.0,
                "pass_at_k": 1.0,
                "pass_at_maj": 1.0,
            }
        )
    )
    md = compare_runs(rpt_dir)
    assert "None" not in md, f"got: {md}"
    assert "?" in md  # the placeholder for missing/null


def test_compare_tolerates_non_numeric_pass_at_1_with_baseline(tmp_path):
    """A corrupt report whose pass_at_1 is a non-number must not crash the delta
    computation when a baseline is set — the delta cell falls back to '?'."""
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    _write_report(rpt_dir, "baseline-run", pass_at_1=0.6)
    (rpt_dir / "corrupt.json").write_text(
        json.dumps(
            {
                "run_name": "corrupt",
                "n_examples": 10,
                "pass_at_1": "oops",  # non-numeric — used to raise TypeError on subtraction
            }
        )
    )
    md = compare_runs(rpt_dir, baseline="baseline-run")
    assert "corrupt" in md
    assert "(base)" in md  # baseline row still rendered
    assert "?" in md  # corrupt row's delta cell


def test_compare_skips_non_object_json(tmp_path):
    """A file with valid JSON that isn't a dict (e.g. `[]` or `"string"`) must
    not crash the rollup with AttributeError on `.get()`."""
    rpt_dir = tmp_path / "reports"
    rpt_dir.mkdir()
    (rpt_dir / "list.json").write_text("[]")
    (rpt_dir / "string.json").write_text('"just a string"')
    (rpt_dir / "ok.json").write_text(
        json.dumps(
            {
                "run_name": "ok",
                "n_examples": 1,
                "pass_at_1": 1.0,
                "pass_at_k": 1.0,
                "pass_at_maj": 1.0,
                "k": 1,
                "failure_modes": {},
            }
        )
    )
    md = compare_runs(rpt_dir)
    assert "ok" in md
    # Non-object files should be silently skipped (no crash, no entry in table).


# ---------------------------------------------------------------------------
# Task 7.3 tests: arm summary (multi-seed aggregation)
# ---------------------------------------------------------------------------


def test_arm_summary_for_multi_seed_runs(tmp_path):
    """3-seed runs of the same arm produce a median ± half_IQR row in arm summary."""
    for seed, p1 in [(42, 0.60), (7, 0.65), (123, 0.62)]:
        _write_report(
            tmp_path, f"phase1-sft-seed{seed}", pass_at_1=p1, pass_at_k=0.78, pass_at_maj=0.71
        )
    md = compare_runs(tmp_path)
    assert "## Arm summary" in md
    assert "phase1-sft" in md
    assert "0.620" in md  # median pass@1


def test_arm_summary_skipped_when_no_multi_seed(tmp_path):
    """Single-run arms (no -seed{N} suffix) don't trigger arm summary."""
    _write_report(tmp_path, "run-a", pass_at_1=0.5)
    _write_report(tmp_path, "run-b", pass_at_1=0.6)
    md = compare_runs(tmp_path)
    # Arm summary should NOT appear (would duplicate main table)
    assert "## Arm summary" not in md


def test_arm_summary_with_4plus_seeds_shows_iqr(tmp_path):
    """With n>=4 seeds, the ± half_IQR component is shown."""
    for seed, p1 in [(42, 0.60), (7, 0.65), (123, 0.62), (1, 0.70)]:
        _write_report(tmp_path, f"arm-seed{seed}", pass_at_1=p1)
    md = compare_runs(tmp_path)
    assert "## Arm summary" in md
    assert "±" in md


def test_arm_summary_with_3_seeds_no_iqr(tmp_path):
    """With n<4 seeds, the median is shown without ± (IQR undefined for small n)."""
    for seed, p1 in [(42, 0.60), (7, 0.65), (123, 0.62)]:
        _write_report(tmp_path, f"arm-seed{seed}", pass_at_1=p1)
    md = compare_runs(tmp_path)
    assert "## Arm summary" in md
    # Find the arm data rows (skip header lines containing 'n_seeds' or '---')
    arm_section = md.split("## Arm summary")[1]
    data_rows = [
        line
        for line in arm_section.split("\n")
        if line.startswith("|") and "n_seeds" not in line and "---" not in line and line.strip()
    ]
    assert data_rows, "Expected at least one data row in arm summary"
    for row in data_rows:
        assert "±" not in row, f"Unexpected ± in data row: {row}"


def test_split_run_name_with_seed_suffix():
    from toyforge.eval.compare import _split_run_name

    assert _split_run_name("phase1-sft-seed42") == ("phase1-sft", 42)


def test_split_run_name_without_seed_suffix():
    from toyforge.eval.compare import _split_run_name

    assert _split_run_name("phase1-sft") == ("phase1-sft", None)


def test_split_run_name_complex_family_name():
    """Family names can contain dashes; only the trailing -seed{N} is stripped."""
    from toyforge.eval.compare import _split_run_name

    assert _split_run_name("phase3-lora-r64-seed7") == ("phase3-lora-r64", 7)


# ---------------------------------------------------------------------------
# Phase 9 tests: --last N + --pattern GLOB filtering
# ---------------------------------------------------------------------------


def test_compare_runs_last_n_filters_recent_reports(tmp_path):
    """`last=2` selects only the 2 most recently modified reports."""
    import time

    # Write 4 reports with sleep between to ensure distinct mtimes
    for i, p1 in enumerate([0.5, 0.6, 0.7, 0.8]):
        _write_report(tmp_path, f"run-{i}", pass_at_1=p1)
        time.sleep(0.02)
    md = compare_runs(tmp_path, last=2)
    # Only the 2 newest (run-2, run-3) should appear
    assert "run-3" in md
    assert "run-2" in md
    assert "run-1" not in md
    assert "run-0" not in md


def test_compare_runs_pattern_filters_by_glob(tmp_path):
    """`pattern` filters report filenames via Path.glob."""
    _write_report(tmp_path, "phase1-sft", pass_at_1=0.5)
    _write_report(tmp_path, "phase3-lora", pass_at_1=0.6)
    _write_report(tmp_path, "phase3-dpo", pass_at_1=0.7)
    md = compare_runs(tmp_path, pattern="phase3-*.json")
    assert "phase3-lora" in md
    assert "phase3-dpo" in md
    assert "phase1-sft" not in md


def test_compare_runs_last_and_pattern_combined(tmp_path):
    """When both --last and --pattern are set, pattern filters first then last."""
    import time

    _write_report(tmp_path, "phase3-a", pass_at_1=0.5)
    time.sleep(0.02)
    _write_report(tmp_path, "phase3-b", pass_at_1=0.6)
    time.sleep(0.02)
    _write_report(tmp_path, "phase1-x", pass_at_1=0.7)
    time.sleep(0.02)
    _write_report(tmp_path, "phase3-c", pass_at_1=0.8)
    md = compare_runs(tmp_path, pattern="phase3-*.json", last=2)
    # phase3-* matches a, b, c. last=2 takes most recently modified: c, b
    assert "phase3-c" in md
    assert "phase3-b" in md
    assert "phase3-a" not in md
    assert "phase1-x" not in md


def test_compare_runs_no_filters_keeps_existing_behavior(tmp_path):
    """Without --last or --pattern, all *.json reports are included (backward compat)."""
    _write_report(tmp_path, "run-a", pass_at_1=0.5)
    _write_report(tmp_path, "run-b", pass_at_1=0.6)
    md = compare_runs(tmp_path)
    assert "run-a" in md
    assert "run-b" in md


# ---------------------------------------------------------------------------
# Phase 9 tests: per-subscore failure-rate ASCII bar chart
# ---------------------------------------------------------------------------


def test_failure_chart_renders_when_failures_present(tmp_path):
    """A run with non-empty failure_modes produces an ASCII chart section."""
    _write_report(
        tmp_path,
        "run-with-fails",
        pass_at_1=0.4,
        failure_modes={"transition_valid": 8, "schema": 4, "parse": 2},
    )
    md = compare_runs(tmp_path)
    assert "## Failure mode chart" in md
    # Bar for transition_valid (max=8) should be longer than for parse (2)
    # Use the count appearance as a proxy
    assert "transition_valid" in md
    assert "schema" in md
    assert "parse" in md
    # The 8-failure subscore gets a full 20-char bar; the 2-failure one gets ~5
    transition_line = next(
        line for line in md.split("\n") if "transition_valid" in line and "█" in line
    )
    schema_line = next(line for line in md.split("\n") if "schema" in line and "█" in line)
    assert transition_line.count("█") > schema_line.count("█")


def test_failure_chart_handles_zero_failures(tmp_path):
    """All-passing runs produce a 'No greedy failures' message in the chart section."""
    _write_report(tmp_path, "all-pass", pass_at_1=1.0, failure_modes={})
    md = compare_runs(tmp_path)
    assert "## Failure mode chart" in md
    assert "No greedy failures across runs" in md


def test_failure_chart_shows_zero_for_unrepresented_subscores(tmp_path):
    """Subscores with 0 failures across all runs render `▏ (0)`."""
    _write_report(
        tmp_path,
        "partial-failures",
        pass_at_1=0.5,
        failure_modes={"parse": 5},  # only parse failures, others 0
    )
    md = compare_runs(tmp_path)
    assert "## Failure mode chart" in md
    # schema, method_known, precondition_met, transition_valid, sequence_optimal all zero
    chart_section = md.split("## Failure mode chart")[1]
    assert "schema" in chart_section
    assert "(0)" in chart_section  # at least one zero rendering


def test_arm_summary_no_iqr_when_few_non_null_pass1(tmp_path):
    """If a family has 4+ runs but <4 have non-null pass_at_1, the IQR is undefined
    and the cell should show bare median (no ± 0.000)."""
    # 4 runs, but 2 have null pass_at_1 (older format / corrupted)
    _write_report(tmp_path, "arm-seed1", pass_at_1=0.60)
    _write_report(tmp_path, "arm-seed2", pass_at_1=0.65)
    # Two with null pass_at_1
    p3 = tmp_path / "arm-seed3.json"
    p3.write_text('{"run_name": "arm-seed3", "n_examples": 30, "pass_at_1": null, "k": 8}')
    p4 = tmp_path / "arm-seed4.json"
    p4.write_text('{"run_name": "arm-seed4", "n_examples": 30, "pass_at_1": null, "k": 8}')

    md = compare_runs(tmp_path)
    arm_section = md.split("## Arm summary")[1]
    # Should NOT have "± 0.000" since only 2 non-null values (n<4 for IQR)
    assert "± 0.000" not in arm_section


def test_failure_chart_aggregates_across_runs(tmp_path):
    """Multiple runs with overlapping failure modes sum correctly."""
    _write_report(tmp_path, "run-a", failure_modes={"parse": 5})
    _write_report(tmp_path, "run-b", failure_modes={"parse": 3, "schema": 2})
    md = compare_runs(tmp_path)
    chart_section = md.split("## Failure mode chart")[1]
    # parse should be 8 (5+3), schema should be 2
    parse_line = next(line for line in chart_section.split("\n") if "parse" in line and "█" in line)
    assert "8" in parse_line

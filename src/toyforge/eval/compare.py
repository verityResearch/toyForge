"""Multi-run scorecard rollup -> reports/compare.md."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from rich.table import Table

from toyforge._console import console as _console

# ---------------------------------------------------------------------------
# Arm-summary helpers (Task 7.3)
# ---------------------------------------------------------------------------

_SEED_RE = re.compile(r"^(.+)-seed(\d+)$")


def _split_run_name(run_name: str) -> tuple[str, int | None]:
    """Return (family, seed) — seed is None if the name has no -seed{N} suffix."""
    m = _SEED_RE.match(run_name)
    if not m:
        return run_name, None
    return m.group(1), int(m.group(2))


def _median_iqr_half(values: list[float]) -> tuple[float, float]:
    """Returns (median, half_IQR).

    half_IQR = (Q3 - Q1) / 2, the symmetric error bar around the median.
    For n < 4, returns (median, 0.0) since IQR is degenerate.
    """
    if not values:
        return 0.0, 0.0
    median = statistics.median(values)
    if len(values) < 4:
        return median, 0.0
    q1, _, q3 = statistics.quantiles(values, n=4)
    return median, (q3 - q1) / 2.0


def _fmt_med_iqr(median: float, half_iqr: float, n: int) -> str:
    """Render '0.620 ± 0.030' for n>=4, '0.620' for n<4."""
    if n < 4:
        return f"{median:.3f}"
    return f"{median:.3f} ± {half_iqr:.3f}"


def _fmt(val: object, fmt: str = ".3f") -> str:
    """Format a numeric value, returning '?' for missing/None."""
    if val is None:
        return "?"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def _str_or_q(val: object) -> str:
    """Stringify val, returning '?' for None (vs. the literal 'None')."""
    return "?" if val is None else str(val)


def _fmt_delta(val: float | None) -> str:
    """Format a signed delta, returning '?' for missing/None."""
    if val is None:
        return "?"
    return f"{val:+.3f}"


def _is_number(val: object) -> bool:
    """True for a real numeric value (int/float, excluding bool)."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _top_failures(failure_modes: dict | None, top_n: int = 3) -> str:
    """Render top-N failure modes sorted by count descending.

    Returns '-' when failure_modes is empty or missing.
    """
    if not failure_modes:
        return "-"
    items = sorted(failure_modes.items(), key=lambda kv: -kv[1])[:top_n]
    return ", ".join(f"{name}({count})" for name, count in items)


def _build_row(
    r: dict, baseline_pass1: float | None, baseline: str | None
) -> tuple[str, str, str, str, str, str, str, str]:
    """Build a single row's columns for both the markdown and Rich table renderings.

    Returns: (run_name, n, pass1_str, delta_cell, passk, passmaj, k, top_failures)
    """
    run_name = _str_or_q(r.get("run_name"))
    n = _str_or_q(r.get("n_examples"))
    pass1 = r.get("pass_at_1")
    passk = _fmt(r.get("pass_at_k"))
    passmaj = _fmt(r.get("pass_at_maj"))
    k = _str_or_q(r.get("k"))
    top_fails = _top_failures(r.get("failure_modes"))

    pass1_str = _fmt(pass1)
    delta_cell = ""
    if baseline is not None:
        if run_name == baseline:
            delta_cell = "(base)"
        elif _is_number(pass1) and _is_number(baseline_pass1):
            delta = pass1 - baseline_pass1
            delta_cell = _fmt_delta(delta)
            if delta < 0:
                pass1_str = f"**{pass1_str}**"
        else:
            delta_cell = "?"
    return run_name, n, pass1_str, delta_cell, passk, passmaj, k, top_fails


def _render_failure_charts(runs: list[dict]) -> str:
    """Aggregate failure_modes across runs into a per-subscore ASCII bar chart."""
    aggregated: Counter[str] = Counter()
    for r in runs:
        fm = r.get("failure_modes") or {}
        for name, count in fm.items():
            aggregated[name] += count

    subscore_names = [
        "parse",
        "schema",
        "method_known",
        "precondition_met",
        "transition_valid",
        "sequence_optimal",
    ]

    if not any(aggregated.values()):
        return (
            "\n\n## Failure mode chart (aggregate across runs)\n\n"
            "_No greedy failures across runs._\n"
        )

    max_count = max(aggregated.values())
    lines = ["", "", "## Failure mode chart (aggregate across runs)", ""]
    for name in subscore_names:
        count = aggregated.get(name, 0)
        if count == 0:
            lines.append(f"    {name:<17} ▏ (0)")
        else:
            bar_count = max(1, round(20 * count / max_count))
            bar = "█" * bar_count
            lines.append(f"    {name:<17} {bar} {count}")
    lines.append("")
    return "\n".join(lines)


def compare_runs(
    reports_dir: Path,
    out_path: Path | None = None,
    baseline: str | None = None,
    last: int | None = None,
    pattern: str | None = None,
) -> str:
    """Read all reports/{name}.json and produce a comparison markdown table.

    Tolerates missing optional keys (`k`, `failure_modes`) and skips files that
    fail to parse as JSON (logging a warning).

    When *baseline* is provided (a run_name string), a Δpass@1 column is added.
    Regression rows (pass@1 < baseline pass@1) have their pass@1 value bolded in
    the markdown output.

    *last* — if set, keep only the N most recently modified reports (by mtime)
    before applying alphabetical ordering for the table.

    *pattern* — glob filter on report filenames (e.g., ``"phase3-*.json"``).
    The caller is responsible for including the ``.json`` extension in the pattern.
    Defaults to ``"*.json"`` when not specified.
    """
    reports_dir = Path(reports_dir)
    glob_pattern = pattern or "*.json"
    candidates = list(reports_dir.glob(glob_pattern))
    # Sort by mtime descending so that `last` takes the newest N.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if last is not None and last > 0:
        candidates = candidates[:last]
    # Re-sort alphabetically for stable table ordering.
    candidates.sort()
    runs = []
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            _console.print(f"[yellow]skip unparseable {p.name}[/yellow]")
            continue
        if not isinstance(data, dict):
            _console.print(f"[yellow]skip non-object report {p.name}[/yellow]")
            continue
        runs.append(data)

    if not runs:
        return "no reports found"

    # Resolve baseline pass@1 value (None if baseline not set or not found).
    baseline_pass1: float | None = None
    if baseline is not None:
        for r in runs:
            if r.get("run_name") == baseline:
                baseline_pass1 = r.get("pass_at_1")
                break

    lines = [
        "# toyForge run comparison",
        "",
        "| run | n | pass@1 | Δpass@1 | pass@k | pass@maj | k | top failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        run_name, n, pass1_str, delta_cell, passk, passmaj, k_str, top_fails = _build_row(
            r, baseline_pass1, baseline
        )
        lines.append(
            f"| {run_name} | {n} | {pass1_str} | {delta_cell} | "
            f"{passk} | {passmaj} | {k_str} | {top_fails} |"
        )
    md = "\n".join(lines) + "\n"

    # --- Arm summary (Task 7.3) ---
    families: dict[str, list[dict]] = {}
    for r in runs:
        fam, _ = _split_run_name(r.get("run_name") or "")
        families.setdefault(fam, []).append(r)

    has_multi_seed = any(len(rs) > 1 for rs in families.values())
    if has_multi_seed:
        arm_lines = [
            "",
            "## Arm summary (multi-seed aggregation)",
            "| arm | n_seeds | pass@1 (median ± half_IQR) | pass@k | pass@maj |",
            "| --- | --- | --- | --- | --- |",
        ]
        for fam, fam_runs in sorted(families.items()):
            seed_count = len(fam_runs)
            p1_vals = [r["pass_at_1"] for r in fam_runs if r.get("pass_at_1") is not None]
            pk_vals = [r["pass_at_k"] for r in fam_runs if r.get("pass_at_k") is not None]
            pmaj_vals = [r["pass_at_maj"] for r in fam_runs if r.get("pass_at_maj") is not None]

            med1, hiqr1 = _median_iqr_half(p1_vals)
            medk, hiqrk = _median_iqr_half(pk_vals)
            medmaj, hiqrmaj = _median_iqr_half(pmaj_vals)

            cell1 = _fmt_med_iqr(med1, hiqr1, len(p1_vals)) if p1_vals else "?"
            cellk = _fmt_med_iqr(medk, hiqrk, len(pk_vals)) if pk_vals else "?"
            cellmaj = _fmt_med_iqr(medmaj, hiqrmaj, len(pmaj_vals)) if pmaj_vals else "?"

            arm_lines.append(f"| {fam} | {seed_count} | {cell1} | {cellk} | {cellmaj} |")

        md = md + "\n".join(arm_lines) + "\n"

    md += _render_failure_charts(runs)

    if out_path:
        Path(out_path).write_text(md)

    # Rich console table mirrors markdown columns.
    tbl = Table(title="toyForge runs")
    for col in ["run", "n", "pass@1", "Δ", "pass@k", "pass@maj", "k", "top failures"]:
        tbl.add_column(col)
    for r in runs:
        cells = _build_row(r, baseline_pass1, baseline)
        tbl.add_row(*cells)
    _console.print(tbl)
    return md

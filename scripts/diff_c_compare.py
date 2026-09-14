"""Differential: C `compare` markdown must byte-match the Python compare oracle.

Runs both renderers on the shared scorecard fixtures under
`c/tests/fixtures/compare` and asserts identical output, so the failure-mode
chart (Unicode block glyphs + round-half-to-even bar lengths), the run table,
baseline deltas, and the arm summary stay in lockstep with
`toyforge.eval.compare`.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from toyforge.eval.compare import compare_runs

FIXTURES = Path("c/tests/fixtures/compare")

# (label, baseline, pattern) — mirror the CTest compare scenarios.
SCENARIOS: list[tuple[str, str | None, str]] = [
    ("baseline_all", "baseline-run", "*.json"),
    ("arm_summary", None, "phase1-sft-*.json"),
    ("no_baseline", None, "*.json"),
    ("no_match", None, "nomatch-does-not-exist-*.json"),  # empty result: "no reports found"
]


def _run_c(toyforge_c: Path, baseline: str | None, pattern: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "c-compare.md"
        cmd = [
            str(toyforge_c),
            "compare",
            "--reports-dir",
            str(FIXTURES),
            "--out-path",
            str(out_path),
            "--pattern",
            pattern,
        ]
        if baseline is not None:
            cmd.extend(["--baseline", baseline])
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"C compare failed: {proc.stderr or proc.stdout}")
        return out_path.read_text(encoding="utf-8")


def _run_py(baseline: str | None, pattern: str) -> str:
    return compare_runs(FIXTURES, out_path=None, baseline=baseline, pattern=pattern)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    failures: list[str] = []
    for label, baseline, pattern in SCENARIOS:
        c_md = _run_c(args.toyforge_c, baseline, pattern).strip()
        py_md = _run_py(baseline, pattern).strip()
        if c_md != py_md:
            failures.append(label)
            c_lines = c_md.splitlines()
            py_lines = py_md.splitlines()
            for i in range(max(len(c_lines), len(py_lines))):
                cl = c_lines[i] if i < len(c_lines) else "<missing>"
                pl = py_lines[i] if i < len(py_lines) else "<missing>"
                if cl != pl:
                    print(f"[{label}] line {i} differs:\n  C : {cl!r}\n  PY: {pl!r}")

    if failures:
        print(f"compare parity FAILED for: {', '.join(failures)}")
        return 1
    print(f"{len(SCENARIOS)} C compare parity scenarios matched the Python oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Differential: C `analyze-rejections` vs the Python analyze_rejections oracle.

The C command (run_analyze_rejections) ports scripts/analyze_rejections.py. This
asserts the rendered markdown report matches Python byte-for-byte (via --output,
the canonical write_text form) across cases exercising: the `_run_start` skip,
count ties (first-seen order), error bucketing (first 80 codepoints + strip,
coalescing near-identical messages), top-20 seed / top-10 error truncation, and
the empty/missing-input report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

# Sibling module in scripts/ (script dir is on sys.path when run as python scripts/X.py).
from analyze_rejections import render_markdown, summarize_rejections


def _rows_full() -> list[dict]:
    rows: list[dict] = [{"_run_start": True, "ts": "x"}]  # must be skipped
    # reasons with a deliberate tie (verifier_failed vs json_decode both 3) to
    # exercise first-seen tie ordering.
    long_err = (
        "transition_valid=0: expected trigger did not validate from prior_state "
        "NEW — this message intentionally exceeds eighty characters so it coalesces"
    )
    for i in range(3):
        rows.append({"reason": "verifier_failed", "seed_id": f"seed_{i % 2}", "error": long_err})
    for _ in range(3):
        rows.append({"reason": "json_decode", "seed_id": "seed_9", "error": "  expecting value  "})
    rows.append({"reason": "malformed_structure", "seed_id": "seed_3"})  # no error field
    # Many distinct seeds to exercise the top-20 truncation in "By seed".
    for i in range(25):
        rows.append({"reason": "verifier_failed", "seed_id": f"bulk_{i:02d}", "error": "schema x"})
    return rows


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _py_md(inp: Path, tmp: Path) -> str:
    # render_markdown(summarize_rejections(...)) is exactly what main() writes to
    # a file via write_text (the canonical report, no print newline).
    return render_markdown(summarize_rejections(inp))


def _c_md(toyforge_c: Path, inp: Path, tmp: Path) -> str:
    out = tmp / "c.md"
    subprocess.run(
        [str(toyforge_c), "analyze-rejections", "--input", str(inp), "--output", str(out)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return out.read_text()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cases = {
            "full": _rows_full(),
            "single": [{"reason": "json_decode", "seed_id": "s1", "error": "x"}],
            "empty_lines": [],  # exists but no rows
        }
        for label, rows in cases.items():
            inp = tmp / f"{label}.jsonl"
            _write_jsonl(inp, rows)
            py = _py_md(inp, tmp)
            c = _c_md(args.toyforge_c, inp, tmp)
            if py != c:
                print(f"[{label}] markdown differs:")
                pyl, cl = py.splitlines(), c.splitlines()
                for i in range(max(len(pyl), len(cl))):
                    a = pyl[i] if i < len(pyl) else "<EOF>"
                    b = cl[i] if i < len(cl) else "<EOF>"
                    if a != b:
                        print(f"  line {i + 1}:\n    py: {a!r}\n    c : {b!r}")
                        break
                return 1

        # Missing input file -> empty report.
        missing = tmp / "does_not_exist.jsonl"
        if _py_md(missing, tmp) != _c_md(args.toyforge_c, missing, tmp):
            print("[missing] markdown differs")
            return 1

    print("analyze-rejections parity: markdown report matches Python (reasons/seeds/errors, empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

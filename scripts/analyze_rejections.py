"""Summarize data/rejected/rejections.jsonl into a markdown report.

Usage:
    uv run python scripts/analyze_rejections.py [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def _iter_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines silently


def summarize_rejections(path: Path, top_n: int = 10) -> dict:
    by_reason: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    err_bucket: Counter[str] = Counter()
    rejection_count = 0

    for r in _iter_rows(path):  # stream directly, no list()
        if r.get("_run_start"):
            continue
        rejection_count += 1
        by_reason[r.get("reason", "unknown")] += 1
        by_seed[r.get("seed_id", "<unknown>")] += 1
        err = r.get("error") or ""
        # Bucket by first 80 chars to coalesce trivially-varying messages.
        err_bucket[err[:80].strip()] += 1

    top_errors = err_bucket.most_common(top_n)
    return {
        "total": rejection_count,
        "by_reason": dict(by_reason),
        "by_seed_id": dict(by_seed),
        "top_errors": top_errors,
    }


def render_markdown(summary: dict) -> str:
    if summary["total"] == 0:
        return "# Rejection summary\n\n_No rejections found._\n"

    out = ["# Rejection summary\n"]
    out.append(f"**Total rejections:** {summary['total']}\n")

    out.append("## By reason\n")
    for reason, count in sorted(summary["by_reason"].items(), key=lambda kv: -kv[1]):
        out.append(f"- `{reason}`: {count}")
    out.append("")

    out.append("## By seed (top 20)\n")
    items = sorted(summary["by_seed_id"].items(), key=lambda kv: -kv[1])[:20]
    for seed, count in items:
        out.append(f"- `{seed}`: {count}")
    out.append("")

    out.append("## Top error patterns\n")
    for err, count in summary["top_errors"]:
        out.append(f"- ({count}×) `{err}`")
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/rejected/rejections.jsonl", type=Path)
    ap.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write markdown report here (default: stdout)",
    )
    args = ap.parse_args()
    summary = summarize_rejections(args.input)
    md = render_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"wrote {args.output}", flush=True)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit overlap between data/train.jsonl, data/dev.jsonl, and data/test.jsonl.

Checks: exact trajectory overlap (by full content), shared seed_trajectory_id,
and high prompt-context prefix-similarity (Jaccard on tokenized first-step text).

Usage:
    uv run python scripts/audit_train_dev_independence.py [--data-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _first_step_situation(t: dict) -> str:
    steps = t.get("steps", [])
    if not steps:
        return ""
    return (steps[0].get("prompt_context") or {}).get("situation", "")


def _token_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _content_hash(t: dict) -> str:
    """Canonical-JSON-style hash of trajectory content (excluding provenance)."""
    keep = {k: v for k, v in t.items() if k not in {"provenance", "source"}}
    return json.dumps(keep, sort_keys=True)


def _pair_report(left: list[dict], right: list[dict], jaccard_threshold: float = 0.8) -> dict:
    left_hashes = {_content_hash(t): t for t in left}
    right_hashes = {_content_hash(t): t for t in right}
    exact = set(left_hashes) & set(right_hashes)

    left_seeds = {t.get("seed_trajectory_id") or t.get("trajectory_id") for t in left}
    right_seeds = {t.get("seed_trajectory_id") or t.get("trajectory_id") for t in right}
    shared_seeds = (left_seeds & right_seeds) - {None}

    high_sim = 0
    high_sim_examples: list[dict] = []
    right_token_sets = [
        (t.get("trajectory_id"), _token_set(_first_step_situation(t))) for t in right
    ]
    for lt in left:
        lt_tokens = _token_set(_first_step_situation(lt))
        if not lt_tokens:
            continue
        for rid, rt_tokens in right_token_sets:
            if (
                rt_tokens
                and _jaccard(lt_tokens, rt_tokens) >= jaccard_threshold
                and lt.get("trajectory_id") != rid
            ):
                high_sim += 1
                if len(high_sim_examples) < 3:
                    high_sim_examples.append({"left": lt.get("trajectory_id"), "right": rid})

    return {
        "exact_trajectory_overlap": len(exact),
        "overlap_examples": [left_hashes[h] for h in list(exact)[:3]],
        # sorted list (not frozenset) so the report dict is JSON-serializable
        "shared_seed_ids": sorted(shared_seeds),
        "high_prefix_similarity_pairs": high_sim,
        "high_prefix_similarity_examples": high_sim_examples,
    }


def audit_independence(train_p: Path, dev_p: Path, test_p: Path) -> dict:
    train = _load(train_p)
    dev = _load(dev_p)
    test = _load(test_p)

    def _f(p: Path, rows: list[dict]) -> dict:
        if p.exists():
            return {"path": str(p), "count": len(rows)}
        return {"path": str(p), "missing": True, "count": 0}

    return {
        "files": {
            "train": _f(train_p, train),
            "dev": _f(dev_p, dev),
            "test": _f(test_p, test),
        },
        "train_vs_dev": _pair_report(train, dev),
        "train_vs_test": _pair_report(train, test),
        "dev_vs_test": _pair_report(dev, test),
    }


def render_markdown(report: dict) -> str:
    out = ["# Train/Dev/Test independence audit\n"]
    out.append("## File counts")
    for k, v in report["files"].items():
        if v.get("missing"):
            out.append(f"- `{k}`: MISSING ({v['path']})")
        else:
            out.append(f"- `{k}`: {v['count']} rows ({v['path']})")
    out.append("")

    for pair in ("train_vs_dev", "train_vs_test", "dev_vs_test"):
        r = report[pair]
        out.append(f"## {pair.replace('_', ' ')}")
        out.append(f"- Exact trajectory overlap: **{r['exact_trajectory_overlap']}**")
        out.append(
            f"- Shared seed-ids: {len(r['shared_seed_ids'])} → {sorted(r['shared_seed_ids'])[:5]}"
        )
        out.append(
            f"- High prefix-similarity pairs (Jaccard ≥ 0.8): "
            f"**{r['high_prefix_similarity_pairs']}**"
        )
        if r["high_prefix_similarity_examples"]:
            out.append("- Examples:")
            for ex in r["high_prefix_similarity_examples"]:
                out.append(f"  - `{ex['left']}` ↔ `{ex['right']}`")
        out.append("")

    out.append("## Policy")
    out.append(
        "- `train_vs_test` and `dev_vs_test` **MUST** report 0 exact overlap and 0 shared_seed_ids."
    )
    out.append(
        "- `train_vs_dev` may share seed-ids (both are teacher expansions of the same seed pool); "
    )
    out.append(
        "  exact overlap must still be 0, and high_prefix_similarity_pairs warrants investigation."
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output", default=None, type=Path)
    args = ap.parse_args()
    report = audit_independence(
        args.data_dir / "train.jsonl",
        args.data_dir / "dev.jsonl",
        args.data_dir / "test.jsonl",
    )
    md = render_markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

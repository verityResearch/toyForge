"""Differential: C `audit-independence` vs the Python audit_independence oracle.

The C command (run_audit_independence) ports scripts/audit_train_dev_independence.py
— a mandatory pre-Phase-1 integrity gate. This asserts the per-pair counts
(exact trajectory overlap, shared seed-ids, high prefix-similarity pairs) and
file counts match Python on a synthetic corpus exercising: content overlap
(modulo provenance/source), duplicate contents (dedup), shared vs distinct
seed-ids, high/low Jaccard situation similarity, Unicode, and a missing split.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from audit_train_dev_independence import audit_independence  # sibling module in scripts/


def _traj(tid, *, seed=None, situation="s", source=None, prov=None, method="ticket_status"):
    t: dict = {
        "trajectory_id": tid,
        "initial_state": "NEW",
        "steps": [
            {
                "prompt_context": {"prior_state": "NEW", "situation": situation},
                "thinking": "t",
                "tool_call": {"jsonrpc": "2.0", "method": method, "params": {}, "id": 1},
            }
        ],
        "final_state": "NEW",
    }
    if seed is not None:
        t["seed_trajectory_id"] = seed
    if source is not None:
        t["source"] = source
    if prov is not None:
        t["provenance"] = prov
    return t


# Shared content used to create an exact overlap modulo provenance/source.
SHARED_SIT = "open a new support ticket from draft now"
TRAIN = [
    _traj("t1", seed="s1", situation=SHARED_SIT, source="teacher_expansion", prov={"a": 1}),
    _traj("t2", seed="s2", situation="retrieve the stored object quickly please"),
    _traj(
        "t2dup", seed="s2", situation="retrieve the stored object quickly please"
    ),  # dup content? no: diff id
    _traj("t3", seed="s3", situation="audit verify the chain état déjà"),
]
DEV = [
    _traj("d1", seed="s2", situation="completely different dev only text here"),
    _traj("d2", seed="s9", situation="retrieve the stored object quickly please"),  # high-sim to t2
]
TEST = [
    # identical CONTENT to t1 (different provenance/source) -> exact overlap.
    _traj("t1", situation=SHARED_SIT),
    _traj("x1", seed="s1", situation="brand new unseen test situation tokens"),
]


def _py_counts(d: Path) -> dict:
    r = audit_independence(d / "train.jsonl", d / "dev.jsonl", d / "test.jsonl")
    out = {"files": {k: (v.get("count", 0), bool(v.get("missing"))) for k, v in r["files"].items()}}
    for pair in ("train_vs_dev", "train_vs_test", "dev_vs_test"):
        pr = r[pair]
        out[pair] = (
            pr["exact_trajectory_overlap"],
            len(pr["shared_seed_ids"]),
            pr["high_prefix_similarity_pairs"],
        )
    return out


def _c_counts(toyforge_c: Path, d: Path) -> dict:
    proc = subprocess.run(
        [str(toyforge_c), "audit-independence", "--data-dir", str(d)],
        check=True, text=True, capture_output=True,
    )  # fmt: skip
    md = proc.stdout
    out: dict = {"files": {}}
    for name in ("train", "dev", "test"):
        m = re.search(rf"- `{name}`: (?:(\d+) rows|MISSING)", md)
        if m and m.group(1) is not None:
            out["files"][name] = (int(m.group(1)), False)
        else:
            out["files"][name] = (0, True)
    for pair, header in (
        ("train_vs_dev", "train vs dev"),
        ("train_vs_test", "train vs test"),
        ("dev_vs_test", "dev vs test"),
    ):
        sect = md.split(f"## {header}", 1)[1]
        exact = int(re.search(r"Exact trajectory overlap: \*\*(\d+)\*\*", sect).group(1))
        shared = int(re.search(r"Shared seed-ids: (\d+)", sect).group(1))
        high = int(re.search(r"Jaccard . 0\.8\): \*\*(\d+)\*\*", sect).group(1))
        out[pair] = (exact, shared, high)
    return out


def _write(d: Path, name: str, rows: list[dict] | None) -> None:
    if rows is None:
        return  # omit -> missing split
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    scenarios = [
        ("full", TRAIN, DEV, TEST),
        ("missing_dev", TRAIN, None, TEST),
        ("empty_test", TRAIN, DEV, []),
    ]
    for label, tr, dv, te in scenarios:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write(d, "train", tr)
            _write(d, "dev", dv)
            _write(d, "test", te)
            py = _py_counts(d)
            c = _c_counts(args.toyforge_c, d)
            if py != c:
                print(f"[{label}] mismatch:\n  Python={py}\n  C     ={c}")
                return 1

    print("audit-independence parity: per-pair overlap/seed/similarity + file counts match Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

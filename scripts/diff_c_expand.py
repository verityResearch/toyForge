"""Differential: C `expand-from-seeds` vs the Python expand-from-seeds oracle.

`hand_seed_provenance` embeds a non-deterministic `expansion_timestamp_utc`, so
byte-parity is impossible; this checks structural parity (all other fields,
including the 9-field provenance block and auto-populated prior_calls) across
all three splits, and confirms the C output round-trips through C verify-data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.cli import expand_from_seeds

SEEDS = Path("scenarios/smoke-tiny-seeds.yaml")
SCHEMAS = Path("schemas")
SPLITS = ["train", "dev", "test"]


def _load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if isinstance(row.get("provenance"), dict):
            row["provenance"].pop("expansion_timestamp_utc", None)  # non-deterministic
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        cout = Path(td) / "c"
        pyout = Path(td) / "py"
        proc = subprocess.run(
            [
                str(args.toyforge_c),
                "expand-from-seeds",
                "--seeds-path",
                str(SEEDS),
                "--schemas-dir",
                str(SCHEMAS),
                "--out-dir",
                str(cout),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"C expand-from-seeds failed: {proc.stderr.strip()}")
            return 1
        expand_from_seeds(seeds_path=SEEDS, schemas_dir=SCHEMAS, out_dir=pyout)

        for split in SPLITS:
            c_rows = _load(cout / f"{split}.jsonl")
            py_rows = _load(pyout / f"{split}.jsonl")
            if c_rows != py_rows:
                failures.append(f"{split}: structural mismatch vs Python")

        # Round-trip: the C output must pass C verify-data.
        vd = subprocess.run(
            [
                str(args.toyforge_c),
                "verify-data",
                "--data-dir",
                str(cout),
                "--schemas-dir",
                str(SCHEMAS),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if vd.returncode != 0:
            failures.append(f"C verify-data rejected the C expand output:\n{vd.stdout}")

    if failures:
        print("\n".join(failures))
        return 1
    print("expand-from-seeds parity: 3 splits match Python (modulo timestamp), round-trip OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Differential: C `verify-seeds` vs the Python verify-seeds oracle.

Runs the C command on scenarios/seeds.yaml and, independently, the Python
load_seeds + verify_trajectory path, and asserts the same per-seed pass/fail set
and the same pass count. This pins the strict-C seed loader + verifier to the
Python behavior on the real seed corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from toyforge.scenario_gen.seeds import load_seeds
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

SEEDS_PATH = Path("scenarios/seeds.yaml")
SCHEMAS_DIR = Path("schemas")


def _python_results() -> tuple[int, set[str]]:
    schemas = load_schemas(SCHEMAS_DIR)
    seeds = load_seeds(SEEDS_PATH)
    passed = 0
    failed_ids: set[str] = set()
    for s in seeds:
        outputs = [
            f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"
            for step in s["steps"]
        ]
        if verify_trajectory(s, outputs, schemas).passed:
            passed += 1
        else:
            failed_ids.add(s["trajectory_id"])
    return passed, failed_ids


def _c_results(toyforge_c: Path) -> tuple[int, int, set[str]]:
    proc = subprocess.run(
        [
            str(toyforge_c),
            "verify-seeds",
            "--seeds-path",
            str(SEEDS_PATH),
            "--schemas-dir",
            str(SCHEMAS_DIR),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    failed_ids = set(re.findall(r"^FAIL (\S+):", proc.stdout, re.MULTILINE))
    m = re.search(r"verify_seeds: (\d+)/(\d+) seeds pass", proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse C output:\n{proc.stdout}\n{proc.stderr}")
    return int(m.group(1)), int(m.group(2)), failed_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    py_pass, py_failed = _python_results()
    c_pass, c_total, c_failed = _c_results(args.toyforge_c)

    failures: list[str] = []
    if c_pass != py_pass:
        failures.append(f"pass count mismatch: C={c_pass} Python={py_pass}")
    if c_failed != py_failed:
        failures.append(
            f"failing-seed set mismatch: C={sorted(c_failed)} Python={sorted(py_failed)}"
        )

    if failures:
        print("\n".join(failures))
        return 1
    print(f"verify-seeds parity: {c_pass}/{c_total} seeds pass, matches Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

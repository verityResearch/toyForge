"""Differential: the C YAML reader must match `yaml.safe_load` on the repo YAML.

Runs `toyforge-c yaml-to-json` on every checked-in YAML file and asserts the
parsed structure equals PyYAML's, so the strict-C reader stays faithful to the
oracle for the constructs the project actually uses (block + flow maps/seqs,
quoted/plain scalars, comments).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

YAML_FILES = [
    "scenarios/seeds.yaml",
    "schemas/state-machine.yaml",
    "schemas/reward-rubric.yaml",
    "schemas/trajectory-schema.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    failures: list[str] = []
    for rel in YAML_FILES:
        path = Path(rel)
        if not path.exists():
            failures.append(f"{rel}: missing")
            continue
        proc = subprocess.run(
            [str(args.toyforge_c), "yaml-to-json", "--path", rel],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            failures.append(f"{rel}: C exited {proc.returncode}: {proc.stderr.strip()[:200]}")
            continue
        try:
            c_value = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            failures.append(f"{rel}: C emitted invalid JSON: {e}")
            continue
        py_value = yaml.safe_load(path.read_text())
        if c_value != py_value:
            failures.append(f"{rel}: parse differs from yaml.safe_load")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"{len(YAML_FILES)} YAML files parsed identically to yaml.safe_load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

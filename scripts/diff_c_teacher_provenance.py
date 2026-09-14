"""Differential: C `teacher-provenance` vs Python's build_provenance.

The provenance block is written into every teacher-expanded row, so the C
emitter must match scenario_gen.provenance.build_provenance field-for-field.
`expansion_timestamp_utc` is non-deterministic (wall clock) and is stripped from
both sides before comparison; everything else — including the
`system_prompt_sha256` computed over the C system-prompt constant — is compared
exactly (temperature with a float tolerance).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.scenario_gen.expand import _SYSTEM_PROMPT
from toyforge.scenario_gen.provenance import build_provenance
from toyforge.scenario_gen.teachers.base import TeacherConfig

PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-6"
EXPANSION_SEED = 7
TEMPERATURE = 0.3
TOYFORGE_VERSION = "9.9.9"
SEED_TRAJECTORY_ID = "seed_diff_provenance"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    cfg = TeacherConfig(
        provider=PROVIDER,
        model=MODEL,
        temperature=TEMPERATURE,
        response_format=None,
        grammar_path=None,
    )
    py = build_provenance(
        teacher_config=cfg,
        system_prompt=_SYSTEM_PROMPT,
        expansion_seed=EXPANSION_SEED,
        seed_trajectory_id=SEED_TRAJECTORY_ID,
        toyforge_version=TOYFORGE_VERSION,
    )

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "prov.json"
        proc = subprocess.run(
            [
                str(args.toyforge_c),
                "teacher-provenance",
                "--provider",
                PROVIDER,
                "--model",
                MODEL,
                "--expansion-seed",
                str(EXPANSION_SEED),
                "--temperature",
                str(TEMPERATURE),
                "--toyforge-version",
                TOYFORGE_VERSION,
                "--seed-trajectory-id",
                SEED_TRAJECTORY_ID,
                "--out-path",
                str(out_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            print(f"C teacher-provenance failed: {proc.stderr or proc.stdout}")
            return 1
        c = json.loads(out_path.read_text())

    failures: list[str] = []

    # Same set of keys, in the same order.
    if list(c.keys()) != list(py.keys()):
        failures.append(f"key order/set differs:\n  C ={list(c.keys())}\n  py={list(py.keys())}")

    # expansion_timestamp_utc is wall-clock; only check both are present + ISO-ish.
    for side, block in (("C", c), ("py", py)):
        ts = block.get("expansion_timestamp_utc")
        if not isinstance(ts, str) or not ts.endswith("Z"):
            failures.append(f"{side} expansion_timestamp_utc not an ISO Z string: {ts!r}")

    c_cmp = {k: v for k, v in c.items() if k != "expansion_timestamp_utc"}
    py_cmp = {k: v for k, v in py.items() if k != "expansion_timestamp_utc"}

    # temperature compared with tolerance; everything else exactly.
    if (
        abs(float(c_cmp.pop("expansion_temperature", -1)) - py_cmp.pop("expansion_temperature"))
        > 1e-9
    ):
        failures.append("expansion_temperature differs")
    for key in sorted(set(c_cmp) | set(py_cmp)):
        if c_cmp.get(key) != py_cmp.get(key):
            failures.append(f"{key}: C={c_cmp.get(key)!r} py={py_cmp.get(key)!r}")

    if failures:
        print("\n".join(failures))
        return 1
    print("teacher-provenance parity: 11 fields match build_provenance (modulo timestamp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

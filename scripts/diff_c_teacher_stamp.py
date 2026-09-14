"""Differential: C `teacher-stamp` vs Python's accepted-row stamping.

On an accepted expansion, expand_seeds does:
    produced["source"] = "teacher_expansion"
    produced["provenance"] = build_provenance(...)
    # trajectory_id deduped to `new_traj_id`
The C `teacher-stamp` must produce a single-line JSONL row that parses to the
same dict (modulo the wall-clock `expansion_timestamp_utc`). Covers: a seed with
no `source` (appended), one that already has `source` (overwritten in place), a
multi-step seed, and a trajectory_id rename (dedup).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.scenario_gen.expand import _SYSTEM_PROMPT
from toyforge.scenario_gen.provenance import build_provenance
from toyforge.scenario_gen.seeds import load_seeds
from toyforge.scenario_gen.teachers.base import TeacherConfig

PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-6"
EXPANSION_SEED = 7
TEMPERATURE = 0.3
TOYFORGE_VERSION = "9.9.9"

CFG = TeacherConfig(
    provider=PROVIDER, model=MODEL, temperature=TEMPERATURE, response_format=None, grammar_path=None
)


def python_row(seed: dict, seed_id: str, new_id: str | None) -> dict:
    produced = copy.deepcopy(seed)
    produced["source"] = "teacher_expansion"
    produced["provenance"] = build_provenance(
        teacher_config=CFG,
        system_prompt=_SYSTEM_PROMPT,
        expansion_seed=EXPANSION_SEED,
        seed_trajectory_id=seed_id,
        toyforge_version=TOYFORGE_VERSION,
    )
    if new_id is not None:
        produced["trajectory_id"] = new_id
    return produced


def c_row(toyforge_c: Path, traj: dict, seed_id: str, new_id: str | None, tmp: Path) -> dict:
    tp = tmp / "traj.json"
    op = tmp / "row.json"
    tp.write_text(json.dumps(traj))
    cmd = [
        str(toyforge_c),
        "teacher-stamp",
        "--trajectory-path",
        str(tp),
        "--seed-trajectory-id",
        seed_id,
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
        "--out-path",
        str(op),
    ]
    if new_id is not None:
        cmd += ["--new-trajectory-id", new_id]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"C teacher-stamp failed: {proc.stderr or proc.stdout}")
    text = op.read_text()
    # Must be a single JSONL line (exactly one trailing newline, none embedded).
    if text.count("\n") != 1 or not text.endswith("\n"):
        raise RuntimeError(f"row is not a single JSONL line: {text!r}")
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    seeds = load_seeds(Path("scenarios/seeds.yaml"))
    seed = seeds[0]
    multi = next((s for s in seeds if len(s.get("steps", [])) > 1), seed)

    with_source = copy.deepcopy(seed)
    with_source["source"] = "hand_seed"  # should be overwritten in place

    # Adversarial strings: the C json_minify_to must preserve string contents
    # (interior whitespace + structural chars + escapes) while stripping only
    # structural whitespace. A minifier bug here would silently corrupt written
    # training rows, so pin it against Python's json.dumps.
    tricky = copy.deepcopy(seed)
    tricky["steps"][0]["thinking"] = 'spaces   {b} [c] : , "q" end\ttab'
    tricky["steps"][0]["prompt_context"]["situation"] = 'esc \\ and \\" and \\t and é'
    tricky["steps"][0]["tool_call"]["params"]["requester_id"] = "user: a {b} ,c"

    cases = [
        ("plain", seed, seed["trajectory_id"], None),
        ("multi_step", multi, multi["trajectory_id"], None),
        ("existing_source", with_source, with_source["trajectory_id"], None),
        ("renamed", seed, seed["trajectory_id"], seed["trajectory_id"] + "_3"),
        ("adversarial_strings", tricky, tricky["trajectory_id"], None),
    ]

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, traj, seed_id, new_id in cases:
            py = python_row(traj, seed_id, new_id)
            c = c_row(args.toyforge_c, traj, seed_id, new_id, tmp)
            py["provenance"].pop("expansion_timestamp_utc", None)
            c.get("provenance", {}).pop("expansion_timestamp_utc", None)
            if c != py:
                diff_keys = [k for k in set(c) | set(py) if c.get(k) != py.get(k)]
                failures.append(f"{name}: differs on keys {diff_keys}")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"teacher-stamp parity: {len(cases)} accepted rows match Python (modulo timestamp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

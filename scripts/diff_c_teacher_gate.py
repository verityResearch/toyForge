"""Differential: C `teacher-gate` vs Python's expand-gate classification.

The C command must classify a teacher completion exactly the way
expand._process_one_expansion does after the network call: strip markdown
fences, json.loads, _render_outputs, verify_trajectory — yielding one of
`json_decode` / `malformed_structure` / `verifier_failed` / `accept`. Python is
the oracle: for each fixture we compute the Python verdict and require the C
`teacher-gate` stdout to match.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

from toyforge.scenario_gen.expand import _render_outputs, _strip_markdown_fences
from toyforge.scenario_gen.seeds import load_seeds
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

SCHEMAS_DIR = Path("schemas")
SEEDS_PATH = Path("scenarios/seeds.yaml")


def python_verdict(raw: str, schemas) -> str:
    try:
        traj = json.loads(_strip_markdown_fences(raw))
    except json.JSONDecodeError:
        return "json_decode"
    try:
        outputs = _render_outputs(traj)
        result = verify_trajectory(traj, outputs, schemas)
    except (KeyError, TypeError, IndexError):
        return "malformed_structure"
    return "accept" if result.passed else "verifier_failed"


def c_verdict(toyforge_c: Path, raw: str, tmp: Path) -> str:
    path = tmp / "response.txt"
    path.write_text(raw)
    proc = subprocess.run(
        [
            str(toyforge_c),
            "teacher-gate",
            "--response-path",
            str(path),
            "--schemas-dir",
            str(SCHEMAS_DIR),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"C teacher-gate error: {proc.stderr or proc.stdout}")
    return proc.stdout.strip()


def build_fixtures() -> dict[str, str]:
    seeds = load_seeds(SEEDS_PATH)
    seed = seeds[0]
    multi = next((s for s in seeds if len(s.get("steps", [])) > 1), seed)

    fixtures: dict[str, str] = {
        "valid_bare": json.dumps(seed),
        "valid_fenced_json": "```json\n" + json.dumps(seed) + "\n```",
        "valid_fenced_plain": "```\n" + json.dumps(seed) + "\n```",
        "valid_multi_step": json.dumps(multi),
        "invalid_json": "{not valid json",
        "not_an_object": "[1, 2, 3]",
        "missing_steps": '{"trajectory_id": "x", "initial_state": "NEW"}',
        "empty_steps": json.dumps({"trajectory_id": "x", "initial_state": "NEW", "steps": []}),
    }

    no_thinking = copy.deepcopy(seed)
    no_thinking["steps"][0].pop("thinking", None)
    fixtures["step_missing_thinking"] = json.dumps(no_thinking)

    no_tool = copy.deepcopy(seed)
    no_tool["steps"][0].pop("tool_call", None)
    fixtures["step_missing_tool_call"] = json.dumps(no_tool)

    bad_trigger = copy.deepcopy(seed)
    if bad_trigger["steps"][0].get("expected_trigger") is not None:
        bad_trigger["steps"][0]["expected_trigger"] = "bogus.trigger"
    else:
        bad_trigger["steps"][0]["expected_trigger"] = "bogus.trigger"
    fixtures["bad_trigger"] = json.dumps(bad_trigger)

    bad_final = copy.deepcopy(seed)
    bad_final["final_state"] = "NOT_A_REAL_FINAL_STATE_XYZ"
    fixtures["bad_final_state"] = json.dumps(bad_final)

    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    schemas = load_schemas(SCHEMAS_DIR)
    fixtures = build_fixtures()

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, raw in fixtures.items():
            pv = python_verdict(raw, schemas)
            cv = c_verdict(args.toyforge_c, raw, tmp)
            if pv != cv:
                failures.append(f"{name}: Python={pv} C={cv}")

    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"teacher-gate parity: {len(fixtures)} fixtures classified identically to the Python gate"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

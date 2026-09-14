"""Phase 0 end-to-end smoke test — runs expand_seeds with a mock teacher
and validates every output through verify_trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.scenario_gen.expand import expand_seeds
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

_VALID = """{
  "trajectory_id": "smoke_001",
  "initial_state": "NEW",
  "difficulty": "easy",
  "source": "teacher_expansion",
  "description": "smoke",
  "steps": [
    {
      "prompt_context": { "prior_state": "NEW", "situation": "smoke" },
      "thinking": "open from NEW",
      "tool_call": {
        "jsonrpc": "2.0", "method": "ticket_open",
        "params": {
          "requester_id": "user:s", "body_text": "YWFh",
          "lifecycle_profile": "std"
        },
        "id": 1
      },
      "expected_trigger": "ticket_open.accepted",
      "expected_state_after": "TRIAGED"
    }
  ],
  "final_state": "TRIAGED"
}"""


class _SmokeTeacher:
    def __init__(self) -> None:
        from toyforge.scenario_gen.teachers.base import TeacherConfig

        self.config = TeacherConfig(provider="mock", model="mock-1", temperature=0.0)

    def complete(self, system: str, user: str) -> str:
        return _VALID


def test_phase0_pipeline_end_to_end(tmp_path: Path, fixtures_dir, schemas_dir):
    stats = expand_seeds(
        seeds_path=fixtures_dir / "seeds" / "seed_sample.yaml",
        schemas_dir=schemas_dir,
        teacher=_SmokeTeacher(),
        expansions_per_seed=4,
        out_dir=tmp_path / "data",
        shuffle_seed=0,
    )
    assert stats["accepted"] >= 3, stats
    schemas = load_schemas(schemas_dir)
    for split in ("train", "dev", "test"):
        path = tmp_path / "data" / f"{split}.jsonl"
        assert path.exists()
        for line in path.read_text().splitlines():
            traj = json.loads(line)
            outs = [
                f"<think>{s['thinking']}</think>{json.dumps(s['tool_call'])}" for s in traj["steps"]
            ]
            r = verify_trajectory(traj, outs, schemas)
            assert r.passed, (split, traj["trajectory_id"], r.error_message)

"""Integration test: expand.py with a mocked teacher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toyforge.scenario_gen.expand import expand_seeds


class _MockTeacher:
    """Returns a hand-crafted valid expansion regardless of prompt."""

    def __init__(self, response_template: str):
        from toyforge.scenario_gen.teachers.base import TeacherConfig

        self.config = TeacherConfig(provider="mock", model="mock-1", temperature=0.0)
        self._response = response_template

    def complete(self, system: str, user: str) -> str:
        return self._response


_VALID_EXPANSION = """{
  "trajectory_id": "mock_expansion_001",
  "initial_state": "NEW",
  "difficulty": "easy",
  "source": "teacher_expansion",
  "description": "Mock expansion: open a new ticket.",
  "steps": [
    {
      "prompt_context": { "prior_state": "NEW", "situation": "open" },
      "thinking": "I should call ticket_open from NEW.",
      "tool_call": {
        "jsonrpc": "2.0",
        "method": "ticket_open",
        "params": {
          "requester_id": "user:bob",
          "body_text": "ZGVm",
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


def test_expand_writes_splits(fixtures_dir, schemas_dir, tmp_path: Path):
    seeds_path = fixtures_dir / "seeds" / "seed_sample.yaml"
    out_dir = tmp_path / "data"
    teacher = _MockTeacher(_VALID_EXPANSION)

    stats = expand_seeds(
        seeds_path=seeds_path,
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=3,
        out_dir=out_dir,
        shuffle_seed=42,
    )

    train = out_dir / "train.jsonl"
    dev = out_dir / "dev.jsonl"
    test = out_dir / "test.jsonl"

    assert train.exists()
    assert dev.exists()
    assert test.exists()
    assert stats["accepted"] >= 3
    assert stats["rejected"] == 0

    for line in train.read_text().splitlines():
        obj = json.loads(line)
        assert "trajectory_id" in obj
        assert "steps" in obj


def test_expand_rejects_invalid_teacher_output(fixtures_dir, schemas_dir, tmp_path: Path):
    seeds_path = fixtures_dir / "seeds" / "seed_sample.yaml"
    teacher = _MockTeacher("not json at all")

    stats = expand_seeds(
        seeds_path=seeds_path,
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=2,
        out_dir=tmp_path / "data",
        shuffle_seed=42,
        max_retries=1,
    )

    assert stats["accepted"] == 0
    assert stats["rejected"] >= 2


def test_expand_writes_provenance_blocks(fixtures_dir, schemas_dir, tmp_path: Path):
    """Every emitted row in train/dev/test carries a provenance block with the
    expected fields. Hand-seeds in test.jsonl get teacher_provider='hand_seed'."""
    seeds_path = fixtures_dir / "seeds" / "seed_sample.yaml"
    out_dir = tmp_path / "data"
    teacher = _MockTeacher(_VALID_EXPANSION)

    expand_seeds(
        seeds_path=seeds_path,
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=2,
        out_dir=out_dir,
        shuffle_seed=42,
    )

    for split in ("train", "dev", "test"):
        p = out_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            row = json.loads(line)
            assert "provenance" in row, f"{split}: row missing provenance block"
            prov = row["provenance"]
            assert "teacher_provider" in prov
            assert "system_prompt_sha256" in prov
            assert "expansion_timestamp_utc" in prov
            if split == "test":
                assert prov["teacher_provider"] == "hand_seed"
            else:
                assert prov["teacher_provider"] == "mock"


def test_expand_renames_duplicate_trajectory_ids(fixtures_dir, schemas_dir, tmp_path: Path):
    """If a teacher echoes the seed's trajectory_id (or returns identical IDs across
    expansions), expand_seeds must dedupe by appending an index suffix."""
    # _VALID_EXPANSION is the canonical fixture with trajectory_id "mock_expansion_001".
    # If the mock teacher returns the same response for every call, we get duplicate IDs.
    teacher = _MockTeacher(_VALID_EXPANSION)

    expand_seeds(
        seeds_path=fixtures_dir / "seeds" / "seed_sample.yaml",
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=3,
        out_dir=tmp_path / "out",
        shuffle_seed=42,
    )

    ids: list[str] = []
    for split in ("train", "dev"):
        p = tmp_path / "out" / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            ids.append(json.loads(line)["trajectory_id"])
    assert len(ids) == len(set(ids)), f"duplicate trajectory_ids: {ids}"


def test_expand_handles_structurally_incomplete_teacher_output(
    fixtures_dir, schemas_dir, tmp_path: Path
):
    """A teacher returning valid JSON but missing required fields (no 'steps',
    no 'thinking', etc.) must be logged as a rejection, not crash the run."""
    # Valid JSON, missing 'steps' key — would raise KeyError when verify_trajectory
    # tries to access trajectory["steps"].
    teacher = _MockTeacher('{"trajectory_id": "broken", "initial_state": "NEW"}')

    # Should NOT raise; should log rejection and produce zero accepted rows.
    stats = expand_seeds(
        seeds_path=fixtures_dir / "seeds" / "seed_sample.yaml",
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=2,
        out_dir=tmp_path / "out",
        shuffle_seed=42,
        max_retries=1,
    )
    assert stats["accepted"] == 0
    assert stats["rejected"] >= 2
    # Confirm the rejection log captured the failure mode.
    rej_log = tmp_path / "out" / "rejected" / "rejections.jsonl"
    assert rej_log.exists()
    contents = rej_log.read_text()
    # Should mention either "malformed_structure" reason or the underlying error.
    low = contents.lower()
    assert "malformed" in low or "keyerror" in low or "steps" in low


class _MockTeacherSequence:
    """Returns a different response each call (cycles through provided list)."""

    def __init__(self, responses: list[str]):
        from toyforge.scenario_gen.teachers.base import TeacherConfig

        self.config = TeacherConfig(provider="mock", model="mock-1", temperature=0.0)
        self._responses = responses
        self._idx = 0

    def complete(self, system: str, user: str) -> str:
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r


def test_expand_handles_dedup_collision_pile_up(fixtures_dir, schemas_dir, tmp_path: Path):
    """If the dedup fallback id ALSO collides, the dedup loop must keep trying.

    Adversarial: teacher returns id=foo_2 (accepted as foo_2), then id=foo
    (accepted as foo), then id=foo again — naive fallback would be foo_2
    (collision with the first), which the robust loop must skip past.
    """
    base_traj = json.loads(_VALID_EXPANSION)

    def with_id(tid: str) -> str:
        traj = dict(base_traj)
        traj["trajectory_id"] = tid
        return json.dumps(traj)

    teacher = _MockTeacherSequence(
        [with_id("foo_2"), with_id("foo"), with_id("foo"), with_id("foo")]
    )
    expand_seeds(
        seeds_path=fixtures_dir / "seeds" / "seed_sample.yaml",
        schemas_dir=schemas_dir,
        teacher=teacher,
        expansions_per_seed=4,
        out_dir=tmp_path / "out",
        shuffle_seed=42,
    )
    ids: list[str] = []
    for split in ("train", "dev"):
        p = tmp_path / "out" / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            ids.append(json.loads(line)["trajectory_id"])
    assert len(ids) == len(set(ids)), f"duplicate IDs after dedup: {ids}"


def test_expand_rejects_invalid_frac_sum(fixtures_dir, schemas_dir, tmp_path: Path):
    """train_frac + dev_frac > 1.0 must raise (silent truncation is misleading)."""
    teacher = _MockTeacher(_VALID_EXPANSION)
    with pytest.raises(ValueError, match=r"(train_frac|dev_frac|> 1\.0|truncate)"):
        expand_seeds(
            seeds_path=fixtures_dir / "seeds" / "seed_sample.yaml",
            schemas_dir=schemas_dir,
            teacher=teacher,
            expansions_per_seed=2,
            out_dir=tmp_path / "out",
            train_frac=0.7,
            dev_frac=0.4,  # 0.7 + 0.4 = 1.1 > 1.0
        )

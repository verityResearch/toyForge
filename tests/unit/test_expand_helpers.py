"""Unit tests for expand.py module-level helpers."""

from __future__ import annotations

import json
from pathlib import Path

from toyforge.scenario_gen.expand import _strip_markdown_fences

# ---------------------------------------------------------------------------
# Shared fixtures used by expand_seeds integration tests
# ---------------------------------------------------------------------------

_TINY_SEED_YAML = """
seeds:
  - trajectory_id: t1
    initial_state: NEW
    final_state: TRIAGED
    source: hand_seed
    description: "Test seed."
    steps:
      - prompt_context:
          prior_state: NEW
          situation: "Test."
        thinking: "Thinking."
        tool_call:
          jsonrpc: "2.0"
          method: ticket_open
          params:
            requester_id: "user:t"
            body_text: "dGVzdA=="
            lifecycle_profile: "std"
          id: 1
        expected_trigger: ticket_open.accepted
        expected_state_after: TRIAGED
"""

_FAKE_VALID_EXPANSION = {
    "trajectory_id": "exp1",
    "initial_state": "NEW",
    "final_state": "TRIAGED",
    "source": "teacher_expansion",
    "description": "Expanded test.",
    "steps": [
        {
            "prompt_context": {
                "prior_state": "NEW",
                "situation": "Different scenario.",
            },
            "thinking": "Different thinking.",
            "tool_call": {
                "jsonrpc": "2.0",
                "method": "ticket_open",
                "params": {
                    "requester_id": "user:diff",
                    "body_text": "ZGlmZg==",
                    "lifecycle_profile": "std",
                },
                "id": 1,
            },
            "expected_trigger": "ticket_open.accepted",
            "expected_state_after": "TRIAGED",
        }
    ],
}

_REPO = Path(__file__).resolve().parents[2]


def _make_seeds_dir(tmp_path: Path) -> Path:
    seeds_dir = tmp_path / "scenarios"
    seeds_dir.mkdir()
    (seeds_dir / "seeds.yaml").write_text(_TINY_SEED_YAML)
    return seeds_dir


# ---------------------------------------------------------------------------
# _strip_markdown_fences tests
# ---------------------------------------------------------------------------


def test_strips_json_fenced_block():
    raw = '```json\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_strips_bare_fenced_block():
    raw = '```\n{"a": 1}\n```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_passes_through_plain_json():
    raw = '{"a": 1}'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_handles_leading_whitespace():
    raw = '   \n```json\n{"a": 1}\n```\n  '
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_handles_single_line_fence_no_newline():
    """Teachers occasionally emit ```json{...}``` without a newline after the opener."""
    raw = '```json{"a": 1}```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


def test_handles_single_line_bare_fence():
    raw = '```{"a": 1}```'
    assert _strip_markdown_fences(raw) == '{"a": 1}'


# ---------------------------------------------------------------------------
# expand_seeds smoke test (Task 6.2)
# ---------------------------------------------------------------------------


def test_expand_seeds_runs_progress_in_silent_mode(tmp_path):
    """expand_seeds with a mock teacher should not hang or crash even when
    rich.progress is auto-disabled (non-tty environment under pytest).

    This is a smoke test — verifies the progress bar code path doesn't
    introduce regressions. Uses a teacher that returns the same valid
    expansion for every call.
    """
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = _make_seeds_dir(tmp_path)

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            return json.dumps(_FAKE_VALID_EXPANSION)

    stats = expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=2,
        out_dir=tmp_path / "data",
        shuffle_seed=42,
    )
    assert stats["accepted"] >= 1
    assert stats["rejected"] == 0


# ---------------------------------------------------------------------------
# Partial file / resume tests (Task 6.3)
# ---------------------------------------------------------------------------


def test_expand_writes_partial_and_deletes_on_success(tmp_path):
    """A successful run streams accepted trajectories to accepted_partial.jsonl
    during the run, then deletes it on clean exit."""
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = _make_seeds_dir(tmp_path)

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            return json.dumps(_FAKE_VALID_EXPANSION)

    out_dir = tmp_path / "data"
    expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=2,
        out_dir=out_dir,
        shuffle_seed=42,
    )
    # Partial file must be deleted after a clean exit.
    assert not (out_dir / "accepted_partial.jsonl").exists()
    # The canonical split files must exist.
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()


def test_expand_resumes_from_partial_file(tmp_path):
    """--resume reads the partial file and skips already-accepted expansions."""
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = _make_seeds_dir(tmp_path)
    out_dir = tmp_path / "data"
    out_dir.mkdir()

    # Pre-populate partial with 2 entries for seed t1 (matches expansions_per_seed).
    partial = out_dir / "accepted_partial.jsonl"
    pre_entry_1 = {
        "trajectory_id": "exp_pre_1",
        "source": "teacher_expansion",
        "provenance": {"seed_trajectory_id": "t1"},
    }
    pre_entry_2 = {
        "trajectory_id": "exp_pre_2",
        "source": "teacher_expansion",
        "provenance": {"seed_trajectory_id": "t1"},
    }
    partial.write_text(json.dumps(pre_entry_1) + "\n" + json.dumps(pre_entry_2) + "\n")

    teacher_calls: dict[str, int] = {"count": 0}

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            teacher_calls["count"] += 1
            return json.dumps(_FAKE_VALID_EXPANSION)

    expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=2,
        out_dir=out_dir,
        shuffle_seed=42,
        resume=True,
    )
    # t1 already has 2/2 expansions in the partial file — no teacher calls needed.
    assert teacher_calls["count"] == 0


def test_expand_stale_partial_deleted_when_not_resuming(tmp_path):
    """When resume=False and a stale partial file exists, it is deleted before
    the new run starts (to avoid mixing results from different runs)."""
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = _make_seeds_dir(tmp_path)
    out_dir = tmp_path / "data"
    out_dir.mkdir()

    # Create a stale partial file from a previous (crashed) run.
    stale = out_dir / "accepted_partial.jsonl"
    stale.write_text(json.dumps({"trajectory_id": "stale"}) + "\n")

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            return json.dumps(_FAKE_VALID_EXPANSION)

    stats = expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=1,
        out_dir=out_dir,
        shuffle_seed=42,
        resume=False,
    )
    # Run should succeed with fresh data only (stale entry not included).
    assert stats["accepted"] >= 1
    # Partial file is cleaned up on success.
    assert not (out_dir / "accepted_partial.jsonl").exists()


# ---------------------------------------------------------------------------
# Parallel expansion tests (Task 8.2)
# ---------------------------------------------------------------------------


def test_expand_seeds_parallel_works(tmp_path):
    """expand_seeds with num_workers > 1 produces the same accepted/rejected
    counts as serial (when teacher is deterministic)."""
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = tmp_path / "scenarios"
    seeds_dir.mkdir()
    (seeds_dir / "seeds.yaml").write_text(_TINY_SEED_YAML)

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            return json.dumps(_FAKE_VALID_EXPANSION)

    out_dir = tmp_path / "data"
    stats = expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=4,
        out_dir=out_dir,
        shuffle_seed=42,
        num_workers=3,  # parallel
    )
    assert stats["accepted"] == 4
    assert stats["rejected"] == 0


def test_expand_warns_on_overwrite(tmp_path, capsys):
    """When data files exist, expand_seeds prints a yellow warning."""
    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = tmp_path / "scenarios"
    seeds_dir.mkdir()
    (seeds_dir / "seeds.yaml").write_text(_TINY_SEED_YAML)

    out_dir = tmp_path / "data"
    out_dir.mkdir()
    # Pre-existing train.jsonl that will be overwritten.
    (out_dir / "train.jsonl").write_text('{"trajectory_id": "old"}\n')

    class FakeTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            return json.dumps(_FAKE_VALID_EXPANSION)

    expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FakeTeacher(),
        expansions_per_seed=2,
        out_dir=out_dir,
        shuffle_seed=42,
    )
    out = capsys.readouterr().out
    assert "Overwriting" in out
    assert "train.jsonl" in out


def test_expand_seeds_handles_teacher_exceptions_in_parallel(tmp_path):
    """When the teacher raises mid-expansion in a worker, the worker logs a
    rejection and continues; the rest of the run completes."""
    import threading

    from toyforge.scenario_gen.expand import expand_seeds
    from toyforge.scenario_gen.teachers.base import TeacherConfig

    seeds_dir = tmp_path / "scenarios"
    seeds_dir.mkdir()
    (seeds_dir / "seeds.yaml").write_text(_TINY_SEED_YAML)

    # Thread-safe counter for concurrent workers.
    call_count_lock = threading.Lock()
    state = {"call_count": 0}

    class FlakeyTeacher:
        config = TeacherConfig(provider="anthropic", model="test-model")

        def complete(self, system: str, user: str) -> str:
            with call_count_lock:
                state["call_count"] += 1
                count = state["call_count"]
            if count <= 2:
                raise RuntimeError("simulated API failure")
            return json.dumps(_FAKE_VALID_EXPANSION)

    out_dir = tmp_path / "data"
    stats = expand_seeds(
        seeds_path=seeds_dir / "seeds.yaml",
        schemas_dir=_REPO / "schemas",
        teacher=FlakeyTeacher(),
        expansions_per_seed=2,
        out_dir=out_dir,
        shuffle_seed=42,
        num_workers=2,
        max_retries=3,
    )
    # The first 2 calls raise; remaining calls succeed — overall accepted should be 2
    # (since max_retries lets each expansion get a fresh attempt)
    assert stats["accepted"] >= 1, stats

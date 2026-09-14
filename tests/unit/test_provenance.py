"""Tests for toyforge.scenario_gen.provenance."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from toyforge.scenario_gen.provenance import (
    build_provenance,
    hand_seed_provenance,
    system_prompt_sha256,
)
from toyforge.scenario_gen.teachers.base import TeacherConfig


def test_system_prompt_sha256_is_stable():
    a = system_prompt_sha256("hello")
    b = system_prompt_sha256("hello")
    assert a == b
    assert re.match(r"^[0-9a-f]{64}$", a)


def test_system_prompt_sha256_is_content_sensitive():
    assert system_prompt_sha256("hello") != system_prompt_sha256("hello!")


def test_build_provenance_includes_required_fields():
    cfg = TeacherConfig(provider="anthropic", model="claude-sonnet-4-6", temperature=0.3)
    prov = build_provenance(
        teacher_config=cfg,
        system_prompt="sys",
        expansion_seed=42,
        seed_trajectory_id="seed_a",
        toyforge_version="0.1.0",
    )
    assert prov["teacher_provider"] == "anthropic"
    assert prov["teacher_model"] == "claude-sonnet-4-6"
    assert prov["system_prompt_sha256"] == system_prompt_sha256("sys")
    assert prov["expansion_seed"] == 42
    assert prov["expansion_temperature"] == 0.3
    assert prov["seed_trajectory_id"] == "seed_a"
    assert prov["toyforge_version"] == "0.1.0"
    # ISO-8601 UTC timestamp, parseable.
    assert prov["expansion_timestamp_utc"].endswith("Z")
    datetime.fromisoformat(prov["expansion_timestamp_utc"].rstrip("Z")).replace(tzinfo=UTC)


def test_hand_seed_provenance_marks_provider():
    prov = hand_seed_provenance(seed_trajectory_id="seed_a", toyforge_version="0.1.0")
    assert prov["teacher_provider"] == "hand_seed"
    assert prov["teacher_model"] is None
    assert prov["seed_trajectory_id"] == "seed_a"


def test_system_prompt_sha256_is_cached():
    """Repeated calls to system_prompt_sha256 should hit the LRU cache."""
    system_prompt_sha256.cache_clear()
    prompt = "x" * 1000
    r1 = system_prompt_sha256(prompt)
    r2 = system_prompt_sha256(prompt)
    assert r1 == r2
    info = system_prompt_sha256.cache_info()
    assert info.hits >= 1
    assert info.misses >= 1

"""Provenance metadata for teacher-expanded and hand-seed training rows."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from toyforge.scenario_gen.teachers.base import TeacherConfig


@lru_cache(maxsize=4)  # 4 is plenty; we never have more than a handful of distinct prompts
def system_prompt_sha256(prompt: str) -> str:
    """Hex-encoded SHA-256 of the UTF-8-encoded prompt. Cached because the
    teacher's system prompt is a module-level constant called for every
    accepted/rejected trajectory."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _now_utc_iso() -> str:
    # Microsecond resolution — second-resolution timestamps collide when many
    # rejections fire in the same second during expansion.
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_provenance(
    *,
    teacher_config: TeacherConfig,
    system_prompt: str,
    expansion_seed: int,
    seed_trajectory_id: str,
    toyforge_version: str,
) -> dict[str, Any]:
    """Provenance block for a teacher-expanded row."""
    return {
        "teacher_provider": teacher_config.provider,
        "teacher_model": teacher_config.model,
        "teacher_model_version": None,
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "expansion_seed": expansion_seed,
        "expansion_temperature": teacher_config.temperature,
        "response_format": teacher_config.response_format,
        "grammar_path": str(teacher_config.grammar_path) if teacher_config.grammar_path else None,
        "expansion_timestamp_utc": _now_utc_iso(),
        "toyforge_version": toyforge_version,
        "seed_trajectory_id": seed_trajectory_id,
    }


def hand_seed_provenance(
    *,
    seed_trajectory_id: str,
    toyforge_version: str,
) -> dict[str, Any]:
    """Provenance block for an un-expanded hand seed (test.jsonl)."""
    return {
        "teacher_provider": "hand_seed",
        "teacher_model": None,
        "teacher_model_version": None,
        "system_prompt_sha256": None,
        "expansion_seed": None,
        "expansion_temperature": None,
        "expansion_timestamp_utc": _now_utc_iso(),
        "toyforge_version": toyforge_version,
        "seed_trajectory_id": seed_trajectory_id,
    }

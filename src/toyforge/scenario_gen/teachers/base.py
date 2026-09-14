"""Teacher protocol + factory.

A `Teacher` takes a prompt + system prompt and returns a single completion string.
We use it to expand hand-seed trajectories into ~2000 trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TeacherConfig:
    provider: str  # "anthropic" | "openai" | "xai" | "local"
    model: str
    base_url: str | None = None  # for openai-compat (xai, local)
    api_key_env: str | None = None  # name of the env var holding the key
    max_tokens: int = 2048
    temperature: float = 0.3
    response_format: dict[str, Any] | None = None  # OpenAI/llama.cpp structured output
    grammar_path: Path | None = None  # llama.cpp GBNF file to pass as `grammar`
    extra_body: dict[str, Any] = field(default_factory=dict)  # provider-specific payload


class Teacher(Protocol):
    """Single-call teacher: produce one expansion given system + user prompt.

    The `config` attribute is the **provenance identity surface**: `scenario_gen.provenance.
    build_provenance` reads `config.provider`, `config.model`, and `config.temperature` from
    every teacher to stamp provenance metadata on each emitted training row. Any new teacher
    implementation must expose `config: TeacherConfig`.
    """

    config: TeacherConfig

    def complete(self, system: str, user: str) -> str: ...


def build_teacher(cfg: TeacherConfig) -> Teacher:
    """Factory — picks the adapter implementation based on cfg.provider."""
    if cfg.provider == "anthropic":
        from toyforge.scenario_gen.teachers.anthropic_teacher import AnthropicTeacher

        return AnthropicTeacher(cfg)
    if cfg.provider in {"openai", "xai", "local"}:
        from toyforge.scenario_gen.teachers.openai_compat_teacher import OpenAICompatTeacher

        return OpenAICompatTeacher(cfg)
    raise ValueError(f"unknown teacher provider: {cfg.provider!r}")

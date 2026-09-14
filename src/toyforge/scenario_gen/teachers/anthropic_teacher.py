"""Anthropic (Claude) teacher adapter."""

from __future__ import annotations

import os

import anthropic

from toyforge.scenario_gen.teachers.base import TeacherConfig


class AnthropicTeacher:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        key_env = config.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"missing env var {key_env!r} — set it to your Anthropic API key"
                f" (e.g. export {key_env}=sk-ant-...)"
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

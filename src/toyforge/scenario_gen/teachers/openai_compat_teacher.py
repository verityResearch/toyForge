"""OpenAI-compatible teacher adapter.

Covers:
  - provider=openai  -> https://api.openai.com (key from OPENAI_API_KEY by default)
  - provider=xai     -> https://api.x.ai/v1 (key from XAI_API_KEY by default)
  - provider=local   -> http://localhost:8080/v1 (key from LLAMA_CPP_API_KEY, often empty)
"""

from __future__ import annotations

import os
from typing import Any

import openai

from toyforge.scenario_gen.teachers.base import TeacherConfig

_PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "local": ("http://localhost:8080/v1", "LLAMA_CPP_API_KEY"),
}


class OpenAICompatTeacher:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        default_url, default_key_env = _PROVIDER_DEFAULTS[config.provider]
        base_url = config.base_url or default_url
        key_env = config.api_key_env or default_key_env
        api_key = os.environ.get(key_env)
        if not api_key:
            if config.provider == "local":
                api_key = "no-key"  # local servers often accept any string
            else:
                raise RuntimeError(
                    f"missing api key: env var {key_env!r} is unset for "
                    f"provider {config.provider!r}"
                )
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        # Newer OpenAI models (gpt-5+) require `max_completion_tokens` and reject
        # the legacy `max_tokens`; xai/local openai-compat endpoints still use
        # `max_tokens`.
        token_kwarg = "max_completion_tokens" if self.config.provider == "openai" else "max_tokens"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            token_kwarg: self.config.max_tokens,
        }
        if self.config.response_format is not None:
            payload["response_format"] = self.config.response_format
        if self.config.extra_body:
            payload["extra_body"] = dict(self.config.extra_body)
        if self.config.grammar_path is not None:
            grammar = self.config.grammar_path.read_text()
            extra_body = dict(payload.get("extra_body") or {})
            extra_body["grammar"] = grammar
            payload["extra_body"] = extra_body

        resp = self._client.chat.completions.create(**payload)
        if not resp.choices:
            return ""
        return resp.choices[0].message.content or ""

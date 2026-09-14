"""Unit tests for teacher adapters: env-var handling, fail-fast symmetry."""

from __future__ import annotations

import pytest

from toyforge.scenario_gen.teachers.base import TeacherConfig


def test_anthropic_teacher_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from toyforge.scenario_gen.teachers.anthropic_teacher import AnthropicTeacher

    cfg = TeacherConfig(provider="anthropic", model="claude-3-haiku-20240307")
    with pytest.raises(RuntimeError, match=r"(ANTHROPIC_API_KEY|api_key)"):
        AnthropicTeacher(cfg)


def test_openai_compat_teacher_raises_when_xai_key_missing(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    from toyforge.scenario_gen.teachers.openai_compat_teacher import OpenAICompatTeacher

    cfg = TeacherConfig(provider="xai", model="grok-2-latest")
    with pytest.raises(RuntimeError, match=r"(XAI_API_KEY|api_key)"):
        OpenAICompatTeacher(cfg)


def test_openai_compat_teacher_handles_empty_choices(monkeypatch):
    """When the API returns an empty choices list (content filter, safety stop),
    complete() must return an empty string rather than raising IndexError."""
    from unittest.mock import MagicMock

    from toyforge.scenario_gen.teachers.openai_compat_teacher import OpenAICompatTeacher

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    teacher = OpenAICompatTeacher(TeacherConfig(provider="xai", model="grok-2-latest"))

    mock_resp = MagicMock()
    mock_resp.choices = []
    teacher._client = MagicMock()
    teacher._client.chat.completions.create.return_value = mock_resp

    assert teacher.complete("sys", "user") == ""


def test_openai_compat_teacher_allows_no_key_for_local_provider(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    from toyforge.scenario_gen.teachers.openai_compat_teacher import OpenAICompatTeacher

    cfg = TeacherConfig(
        provider="local",
        model="any",
        base_url="http://localhost:8000/v1",
    )
    # Should NOT raise; local providers typically don't need keys.
    OpenAICompatTeacher(cfg)


def test_openai_compat_teacher_passes_llamacpp_structured_options(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from toyforge.scenario_gen.teachers.openai_compat_teacher import OpenAICompatTeacher

    grammar = tmp_path / "jsonrpc.gbnf"
    grammar.write_text('root ::= "x"\n')
    cfg = TeacherConfig(
        provider="local",
        model="qwen.gguf",
        base_url="http://localhost:8080/v1",
        response_format={"type": "json_object"},
        grammar_path=grammar,
        extra_body={"cache_prompt": True},
    )
    teacher = OpenAICompatTeacher(cfg)
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "{}"
    teacher._client = MagicMock()
    teacher._client.chat.completions.create.return_value = mock_resp

    assert teacher.complete("sys", "user") == "{}"
    kwargs = teacher._client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"]["cache_prompt"] is True
    assert kwargs["extra_body"]["grammar"] == 'root ::= "x"\n'

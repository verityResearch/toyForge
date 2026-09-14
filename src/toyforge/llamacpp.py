"""llama.cpp helpers for local teacher, constrained eval, and GRPO rollout prototypes."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toyforge.train.data import format_step_as_chat_messages


@dataclass(frozen=True)
class LlamaCppConfig:
    """Connection and decoding options for a llama.cpp OpenAI-compatible server."""

    base_url: str = "http://localhost:8080/v1"
    model: str = "local"
    api_key: str = "no-key"
    grammar_path: Path | None = Path("schemas/jsonrpc.gbnf")
    extra_body: dict[str, Any] = field(default_factory=dict)
    commit: str = ""


def grammar_sha256(grammar_path: Path | None) -> str:
    """Return the SHA-256 digest of a grammar file, or an empty string when absent."""
    if grammar_path is None or not grammar_path.exists():
        return ""
    return hashlib.sha256(grammar_path.read_bytes()).hexdigest()


def build_jsonrpc_gbnf(schemas_dir: Path) -> str:
    """Build a conservative GBNF grammar for the JSON-RPC envelope.

    The grammar constrains the envelope and known method names. Method-specific
    params are still validated by the verifier, which remains the source of truth.
    """
    methods_doc = json.loads((Path(schemas_dir) / "jsonrpc-methods.json").read_text())
    methods = sorted(methods_doc["methods"])
    # Each method VALUE is a JSON string in the call, so the GBNF literal must match the
    # *quoted* text "<name>". json.dumps(m) yields the JSON string `"<name>"`; a second
    # json.dumps escapes that into the GBNF string literal `"\"<name>\""`. One json.dumps
    # alone would emit `"<name>"`, a GBNF literal that matches the BARE token <name> and
    # rejects the JSON-valid `"<name>"` — the bug surfaced via a strict GBNF parser.
    quoted_methods = " | ".join(json.dumps(json.dumps(m)) for m in methods)
    return (
        "\n".join(
            [
                'root ::= "{" ws jsonrpc "," ws method "," ws params "," ws id ws "}"',
                'jsonrpc ::= "\\"jsonrpc\\"" ws ":" ws "\\"2.0\\""',
                f'method ::= "\\"method\\"" ws ":" ws ({quoted_methods})',
                'params ::= "\\"params\\"" ws ":" ws object',
                'id ::= "\\"id\\"" ws ":" ws (number | string)',
                'object ::= "{" ws (member ("," ws member)*)? ws "}"',
                'member ::= string ws ":" ws value',
                'array ::= "[" ws (value ("," ws value)*)? ws "]"',
                'value ::= object | array | string | number | "true" | "false" | "null"',
                'string ::= "\\"" char* "\\""',
                (
                    'char ::= [^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" '
                    "[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])"
                ),
                'number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
                "ws ::= [ \\t\\n\\r]*",
            ]
        )
        + "\n"
    )


class LlamaCppClient:
    """Small OpenAI-compatible chat client for llama.cpp server."""

    def __init__(self, config: LlamaCppConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        grammar: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        merged_extra = dict(self.config.extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        if grammar is not None:
            merged_extra["grammar"] = grammar
        payload.update(merged_extra)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"llama.cpp request failed: {e}") from e

        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")


def render_step_outputs_llamacpp(
    client: LlamaCppClient,
    trajectory: dict[str, Any],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    constrained: bool = True,
) -> list[list[str]]:
    """Render 1 greedy + k sampled outputs for each step using llama.cpp.

    Constrained mode uses a two-stage decode: free-form thinking followed by a
    JSON-RPC decode under the configured grammar. The verifier sees the same
    `<think>...</think>{json}` surface as the Transformers path.
    """
    grammar = None
    if constrained and client.config.grammar_path is not None:
        grammar = client.config.grammar_path.read_text()

    per_step: list[list[str]] = []
    for step in trajectory["steps"]:
        prompt_messages = format_step_as_chat_messages(step)[:-1]

        def _one(
            sample_temperature: float,
            prompt_messages: list[dict[str, str]] = prompt_messages,
        ) -> str:
            think = client.complete(
                [
                    *prompt_messages,
                    {
                        "role": "user",
                        "content": "Write only the reasoning text that belongs inside <think>.",
                    },
                ],
                max_tokens=max_new_tokens,
                temperature=sample_temperature,
            ).strip()
            call = client.complete(
                [
                    *prompt_messages,
                    {
                        "role": "assistant",
                        "content": f"<think>{think}</think>",
                    },
                    {"role": "user", "content": "Now emit only the JSON-RPC request object."},
                ],
                max_tokens=max_new_tokens,
                temperature=sample_temperature,
                grammar=grammar,
            ).strip()
            return f"<think>{think}</think>{call}"

        greedy = _one(0.0)
        samples = [_one(temperature) for _ in range(max(0, k))]
        per_step.append([greedy, *samples])
    return per_step

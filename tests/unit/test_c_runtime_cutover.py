"""Regression checks for the strict-C runtime cutover in the justfile."""

from __future__ import annotations

from pathlib import Path

EVAL_STRICT_C_RECIPE = (
    'eval-strict-c run_name="phase1-strict-c" '
    'model="toyforge-export" '
    'base_url="http://localhost:8080/v1" '
    'cstrict_repo=env_var_or_default("CSTRICT_REPO", ""):'
)


def _recipe_body(justfile: str, recipe: str) -> str:
    lines = justfile.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line == recipe or line.startswith(f"{recipe} ")
    )
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith((" ", "\t")) and ":" in line:
            break
        body.append(line)
    return "\n".join(body)


def test_cpu_runtime_recipes_use_strict_c_binary() -> None:
    justfile = Path("justfile").read_text()
    for recipe in (
        "seeds:",
        'expand provider="anthropic" model="claude-sonnet-4-6" expansions="25":',
        "verify-data:",
        'analyze-rejections out="reports/rejections.md":',
        'audit-independence out="reports/independence.md":',
        "compare:",
        'eval-b0-llamacpp model="Qwen3-8B-Instruct-GGUF" run="B0-llamacpp-constrained":',
        EVAL_STRICT_C_RECIPE,
    ):
        body = _recipe_body(justfile, recipe)
        assert "c/build/toyforge-c" in body, recipe
        assert "uv run toyforge" not in body, recipe


def test_training_and_transformers_recipes_remain_python_boundary() -> None:
    justfile = Path("justfile").read_text()
    for recipe in (
        'train config="train_configs/phase1-sft.yaml":',
        'eval-phase1 run_name="phase1-sft":',
        'eval-b0 model="Qwen/Qwen3-8B-Instruct" run="B0-zero-shot":',
        'tokenizer-audit model="Qwen/Qwen3-8B-Instruct" out="reports/tokenizer-audit.md":',
    ):
        body = _recipe_body(justfile, recipe)
        assert "uv run toyforge" in body or "uv run python" in body, recipe


def test_eval_strict_c_stamps_cstrict_commit() -> None:
    justfile = Path("justfile").read_text()
    body = _recipe_body(justfile, EVAL_STRICT_C_RECIPE)

    assert "--llamacpp-commit" in body
    assert "git -C {{cstrict_repo}} rev-parse --short HEAD" in body


def test_smoke_tiny_uses_strict_c_data_setup() -> None:
    justfile = Path("justfile").read_text()
    body = _recipe_body(justfile, "smoke-tiny:")

    assert "c/build/toyforge-c expand-from-seeds" in body
    assert "uv run toyforge expand-from-seeds" not in body
    assert "uv run toyforge train" in body
    assert "uv run toyforge eval" in body

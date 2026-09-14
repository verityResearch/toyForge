# Repository Guidelines

## Project Structure & Module Organization

`toyForge` is a Python 3.11 package under `src/toyforge`. The Typer CLI entry point is
`src/toyforge/cli.py`. Core areas are `verifier/` for deterministic JSON-RPC grading,
`scenario_gen/` for seed expansion, `train/` for SFT/GRPO configuration and data loading,
and `eval/` for scoring and comparisons. Schemas in `schemas/` are the source of truth for
methods, rewards, trajectories, and the state machine. Tests live in `tests/unit`,
`tests/integration`, and `tests/property`; reusable fixtures are in `tests/fixtures`.
Operational inputs are in `scenarios/`, `train_configs/`, and `sweeps/`. Generated outputs
such as `data/`, `out/`, and `reports/` should be treated as artifacts.

## Build, Test, and Development Commands

Prefer `just` targets over raw commands:

```bash
just install          # uv sync --extra dev
just install-train    # add torch/trl/peft training dependencies
just lint             # ruff check plus format check
just fmt              # ruff format src tests scripts
just test-unit        # fast CPU unit tests
just test             # full pytest suite
just smoke-tiny       # 5-seed train/eval pipeline without teacher API calls
```

Install `just` with `uv tool install just` if it is missing. Use
`uv run toyforge <command>` only when no `just` target exists. Local llama.cpp
paths use the OpenAI-compatible server at `http://localhost:8080/v1`; see
`just eval-b0-llamacpp` for the constrained baseline command.

## Coding Style & Naming Conventions

Use Ruff with line length 100 and Python 3.11 syntax. The configured lint set is
`E,F,I,B,UP,SIM`; run `just lint` before handing off changes. Keep functions and modules
snake_case, classes PascalCase, and tests named `test_*.py`. Keep verifier logic pure and
deterministic: no I/O, model calls, or hidden global state in `src/toyforge/verifier`.

## Testing Guidelines

Use pytest for all tests and Hypothesis for property checks in `tests/property`. Run the
smallest relevant target during development, then broaden when touching shared behavior:
`just test-unit` for local changes, `just test-integration` for CLI/pipeline changes, and
`just verifier-property-tests` for verifier or schema-transition changes. Add fixtures
under `tests/fixtures` when examples are reused across tests.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commits, often with scopes: `fix(sweep): ...`,
`feat(cli): ...`, `test(config): ...`, `refactor(compare): ...`. Keep commits focused and
describe the behavior change, not just the edited file. Pull requests should include a
short problem statement, implementation summary, commands run, and any data/model
implications. Link issues or plans when relevant, especially for verifier, schema, or
training-pipeline changes.

## Security & Configuration Tips

Teacher expansion requires provider credentials such as `ANTHROPIC_API_KEY`; do not commit
secrets or local `.env` files. GPU training requires `just install-train` and appropriate
hardware. Before Phase 1 runs, use `just audit-independence` to check train/dev/test
separation.

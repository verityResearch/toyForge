# toyForge

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

toyForge is a training pipeline for small reasoning models that must make **verifiable tool calls**. It fine-tunes a model to emit a `<think>…</think>` trace followed by a JSON-RPC request that is valid against a schema *and* legal in a state machine, and it grades those outputs with one pure, deterministic verifier.

The task domain is a deliberately small, invented **support-ticket API**: 9 methods, 10 ticket states, 11 transitions. It is a test bed, not a product. The point is the method: a verifier that is simultaneously the eval grader and the reinforcement-learning reward, with a strict-C port held byte-for-byte to the Python reference.

## The domain

```
NEW --ticket_open--> TRIAGED --(routing)--> ASSIGNED --(agent ack)--> IN_PROGRESS
IN_PROGRESS --resolution_confirm.passed--> RESOLVED
RESOLVED --ticket_reopen--> REOPENED --resolution_confirm.passed--> RESOLVED
                            REOPENED --resolution_confirm.failed--> ESCALATED
ESCALATED --admin_escalate (start)--> ENGINEERING --admin_escalate (resolved)--> RESOLVED
RESOLVED --admin_lifecycle_apply (close)--> CLOSED --(archive window)--> ARCHIVED
```

Query methods (`ticket_get`, `ticket_status`, `ticket_history`, `admin_agent_add`) never change state. Two transitions are background events that no client call can trigger, so the model has to learn when it *cannot* act. The source of truth is [`schemas/`](schemas/): method parameter schemas, the state machine, the trajectory format, the reward rubric, and a generated GBNF grammar.

## How it works

1. **Seeds.** 31 hand-written trajectories in [`scenarios/seeds.yaml`](scenarios/seeds.yaml) cover the reachable transitions, queries, and multi-step recoveries.
2. **Expansion.** A teacher model (Anthropic or any OpenAI-compatible API) expands seeds into variants. Every variant is verified before admission, with provenance stamped on each row.
3. **Verifier.** Each step is scored on `parse`, `schema`, `method_known`, `precondition_met`, `transition_valid`, and `sequence_optimal`. The same function grades evaluation runs and, with weights from [`schemas/reward-rubric.yaml`](schemas/reward-rubric.yaml), produces the GRPO reward.
4. **Training.** SFT cold-start with loss masked to the assistant turn. **GRPO is not implemented yet** — see [Status](#status).
5. **Evaluation.** pass@1, pass@k, and majority-vote scoring against held-out trajectories, through Transformers or a llama.cpp server with grammar-constrained decoding.

## Strict-C port

[`c/`](c/) is a hand-written C11 port of every CPU-side surface: the verifier and reward, a Draft 2020-12 JSON Schema validator, a YAML reader, seed and data validation, scoring and comparison, the llama.cpp HTTP client and GBNF builder, and teacher expansion. Python is the parity oracle. Seventeen `scripts/diff_c_*.py` gates run both implementations on shared inputs and require identical results. See [`c/README.md`](c/README.md).

## Quick start

toyForge uses [uv](https://docs.astral.sh/uv/) and the [`just`](https://github.com/casey/just) command runner. Install `just` with `uv tool install rust-just`. The PyPI package named `just` is unrelated.

```bash
just install          # CPU: data, verifier, tests
just test             # full pytest suite
just test-c           # build and test the C port (CMake, C11 compiler)
just verify-seeds-c   # verify every hand seed against the schemas
```

Pipeline phases:

```bash
just phase0           # seeds -> teacher expansion -> verification (needs ANTHROPIC_API_KEY or equivalent)
just install-train    # GPU training dependencies (torch, transformers, trl, peft)
just phase1           # SFT cold-start + evaluation (GPU)
just smoke-tiny       # 5-seed train/eval smoke run, no teacher API calls
just compare          # roll up evaluation scorecards
```

The `serve-strict-c` / `eval-strict-c` recipes expect an external strict-C llama.cpp server binary. Set `CSTRICT_SERVER` to its path.

## Layout

| Path | Contents |
| --- | --- |
| `schemas/` | Method schemas, state machine, trajectory schema, reward rubric, GBNF grammar |
| `scenarios/` | Hand-written seed trajectories |
| `src/toyforge/` | CLI, verifier, scenario generation, training, evaluation |
| `c/` | Strict C11 port with CTest suite and vendored JSON Schema test suite |
| `scripts/` | C-vs-Python parity gates, audits, sweeps |
| `tests/` | Unit, integration, and Hypothesis property tests |
| `train_configs/`, `sweeps/` | Training and sweep configurations |
| `docs/` | [Design notes](docs/DESIGN.md), [reports](docs/reports/) |

## Status

- Verifier, schemas, seeds, teacher expansion, SFT, evaluation, and the C port: implemented and tested.
- **GRPO: not implemented.** What exists and is tested: the verifier-as-reward function and its rubric presets, the `method: grpo` config (including a llama.cpp-native variant), and CLI dispatch. What doesn't exist is the training loop itself — rollouts, group-relative advantages, and the policy update. Running `method: grpo` or `just phase2-llamacpp` stops with a `NotImplementedError` that says so, rather than training anything.

## Contributing

See [AGENTS.md](AGENTS.md) for repository conventions. Run `just lint` and `just test` before opening a pull request. Keep the verifier pure: no I/O, no model calls, no global state.

## License

Apache 2.0. See [LICENSE](LICENSE).

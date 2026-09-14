# toyForge — strict C11 port

A hand-written **strict C11** port of every CPU-only and llama.cpp-runtime
surface of toyForge. Python remains the **parity oracle** (each C surface is
differentially gated against it); the GPU/training surfaces stay in Python as an
external boundary.

Design notes: [`../docs/DESIGN.md`](../docs/DESIGN.md).

## Layout

```
c/
  CMakeLists.txt   # -Wall -Wextra -Wpedantic -Werror; optional libcurl (TF_HAVE_CURL)
  include/toyforge/*.h
  src/*.c          # json, jsonschema, schemas, verifier, reward, yaml,
                   # regex, sha256, http, io, cli (the command dispatch)
  tests/*.c        # CTest unit tests + fixtures + vendored JSON Schema suite
```

## Build & test

The `justfile` (repo root) is the orchestrator; prefer it over raw CMake.

```bash
just build-c            # configure + build c/build (gcc, Debug)
just test-c             # ctest on c/build
just c-port-cycle       # the canonical full gate (see below)
```

`c-port-cycle` is the single source of truth for "is the port green": it runs
the 25 CTests, all 17 Python-vs-C differential gates, the official JSON Schema
suite, the report renderers, the GBNF/payload/eval fixtures, then rebuilds under the sanitizer and re-runs every differential against it.

### Build matrix

| Config | Command | Why |
|--------|---------|-----|
| gcc Debug | `just build-c` / `test-c` | default |
| clang Debug | `just test-c-clang` | second toolchain; gcc/clang diagnose differently |
| ASan/UBSan | `just test-c-sanitize` + `just diff-c-sanitize` | memory/UB safety |
| no-libcurl | `just build-c-nocurl` | the live/HTTP commands fail closed (exit 3) |
| Release `-O2` | `just build-c-release` | optimizer/`_FORTIFY_SOURCE` warnings |

libcurl is optional: with it, the live `llamacpp-*` and `teacher-expand`
commands work; without it (`-DCMAKE_DISABLE_FIND_PACKAGE_CURL=ON`) they fail
closed with exit code 3. CMake prints which mode it built.

## Parity methodology

Python is the oracle. Every C surface has a `scripts/diff_c_*.py` gate that runs
the C binary and the Python implementation on shared inputs and asserts they
agree (subscores, counts, or byte-for-byte output). The core logic modules are
additionally **randomized-fuzz-verified** at high volume against Python
(verifier, JSON Schema validator, trajectory verifier, regex, JSON parser,
SHA-256, and the data-audit commands). Run `just diff-c` (+ the other
`diff-c-*` targets) or just `just c-port-cycle`.

## Command surface

`toyforge-c <command> [...]` — 22 commands. Highlights:

- **Verify / score:** `verify-step`, `verify-seeds`, `verify-data`, `score-data`,
  `compare`.
- **Data audit:** `audit-independence` (mandatory pre-Phase-1 train/dev/test
  overlap gate), `analyze-rejections`.
- **Schemas / grammar:** `jsonschema-validate`, `build-jsonrpc-gbnf`,
  `grammar-sha256`, `yaml-to-json`.
- **llama.cpp runtime (libcurl):** `llamacpp-complete`, `llamacpp-eval`,
  `llamacpp-payload`, `llamacpp-verify-response`, `llamacpp-score-responses`.
- **Teacher expansion:** `teacher-request`, `teacher-provenance`, `teacher-gate`,
  `teacher-stamp`, `teacher-expand` (OpenAI-compatible **and** Anthropic
  providers).

Run `toyforge-c --help` for the full usage. The verifier (`verify-step` /
`score-data`) is the GRPO reward and the eval grade — one pure implementation,
two callers — so it must stay byte-faithful to Python.

## Conversion scope

**In C (complete, parity-gated):** the entire CPU-only + llama.cpp-runtime
surface — verifier/reward, the Draft 2020-12 JSON Schema validator, YAML reader,
seed/data validation, eval/compare/score, the llama.cpp HTTP client + GBNF
builder, teacher expansion (both providers), and the data-audit diagnostics.

**External boundary:** GPU training (`train`) and Transformers-based eval
(`eval --runtime transformers`) stay in Python, as does the
Transformers-dependent tokenizer-fragmentation audit. Exercising teacher
expansion against a real paid API and `llamacpp-eval` against a real GGUF
llama.cpp server are strict-C commands, but resource-gated. The project must not
simulate native GRPO behavior that llama.cpp does not expose.

## Alignment with strict-C llama.cpp servers

toyForge's llama.cpp client speaks the OpenAI-compatible chat-completions wire
contract (top-level GBNF `grammar`, response `choices[].message.content`), which
is the contract a strict-C llama.cpp-style server (`llamachat`) is expected to
serve. toyForge realigns only if that request/response contract changes.

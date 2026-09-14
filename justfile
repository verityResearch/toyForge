# toyForge orchestration targets

set shell := ["bash", "-cu"]

# --- Setup ---
install:
    uv sync --extra dev

install-train:
    uv sync --extra dev --extra train

# --- Lint / test ---
lint:
    uv run ruff check src tests scripts
    uv run ruff format --check src tests scripts

fmt:
    uv run ruff format src tests scripts

mypy:
    uv run mypy src/toyforge

test:
    uv run pytest -v

test-unit:
    uv run pytest tests/unit -v

test-integration:
    uv run pytest tests/integration -v

verifier-property-tests:
    uv run pytest tests/property -v

verifier-property-tests-ci:
    uv run pytest tests/property --hypothesis-profile=ci -v

# --- Straight C port ---
build-c:
    cmake -S c -B c/build
    cmake --build c/build

test-c: build-c
    ctest --test-dir c/build --output-on-failure

test-c-sanitize:
    cmake -S c -B c/build-sanitize -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
    cmake --build c/build-sanitize
    ctest --test-dir c/build-sanitize --output-on-failure

# Build + test under clang as well as gcc: the strict flags (-Wall -Wextra
# -Wpedantic -Werror) catch different things across compilers, so a second
# toolchain broadens the strict-C portability guarantee.
test-c-clang:
    CC=clang cmake -S c -B c/build-clang -DCMAKE_BUILD_TYPE=Debug
    cmake --build c/build-clang
    ctest --test-dir c/build-clang --output-on-failure

# Build with libcurl forced off (a supported config: the HTTP/live commands
# fail closed with exit 3). CI always has libcurl, so this guards the
# #ifdef TF_HAVE_CURL paths against unused-function/variable breakage under -Werror.
build-c-nocurl:
    cmake -S c -B c/build-nocurl -DCMAKE_DISABLE_FIND_PACKAGE_CURL=ON
    cmake --build c/build-nocurl
    ctest --test-dir c/build-nocurl --output-on-failure
    c/build-nocurl/toyforge-c llamacpp-eval --data-path x --schemas-dir schemas; test $? -eq 3

# Release (-O2) build: the optimizer + _FORTIFY_SOURCE activate warnings Debug
# misses (-Wstringop-overflow, -Wmaybe-uninitialized) and can expose
# optimization-dependent UB. The production build should be -Werror-clean.
build-c-release:
    cmake -S c -B c/build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build c/build-release
    ctest --test-dir c/build-release --output-on-failure

# Run the Python-vs-C differential corpora against the sanitized binary so the
# broad generated input set (not just the CTest fixtures) is checked under
# ASan/UBSan. Assumes `test-c-sanitize` has built c/build-sanitize.
diff-c-sanitize:
    uv run python scripts/diff_c_verifier.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_trajectory.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_compare.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_score_md.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_yaml.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_seeds.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_expand.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_jsonschema.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_audit_independence.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_analyze_rejections.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_gbnf.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_llamacpp_payload.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_reward.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_teacher_request.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_teacher_provenance.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_teacher_gate.py --toyforge-c c/build-sanitize/toyforge-c
    uv run python scripts/diff_c_teacher_stamp.py --toyforge-c c/build-sanitize/toyforge-c

diff-c: build-c
    uv run python scripts/diff_c_verifier.py --toyforge-c c/build/toyforge-c

diff-c-compare: build-c
    uv run python scripts/diff_c_compare.py --toyforge-c c/build/toyforge-c

diff-c-score-md: build-c
    uv run python scripts/diff_c_score_md.py --toyforge-c c/build/toyforge-c

diff-c-traj: build-c
    uv run python scripts/diff_c_trajectory.py --toyforge-c c/build/toyforge-c

diff-c-yaml: build-c
    uv run python scripts/diff_c_yaml.py --toyforge-c c/build/toyforge-c

diff-c-seeds: build-c
    uv run python scripts/diff_c_seeds.py --toyforge-c c/build/toyforge-c

diff-c-expand: build-c
    uv run python scripts/diff_c_expand.py --toyforge-c c/build/toyforge-c

diff-c-jsonschema: build-c
    uv run python scripts/diff_c_jsonschema.py --toyforge-c c/build/toyforge-c

diff-c-audit-independence: build-c
    uv run python scripts/diff_c_audit_independence.py --toyforge-c c/build/toyforge-c

diff-c-analyze-rejections: build-c
    uv run python scripts/diff_c_analyze_rejections.py --toyforge-c c/build/toyforge-c

diff-c-gbnf: build-c
    uv run python scripts/diff_c_gbnf.py --toyforge-c c/build/toyforge-c

diff-c-llamacpp-payload: build-c
    uv run python scripts/diff_c_llamacpp_payload.py --toyforge-c c/build/toyforge-c

diff-c-reward: build-c
    uv run python scripts/diff_c_reward.py --toyforge-c c/build/toyforge-c

diff-c-teacher-request: build-c
    uv run python scripts/diff_c_teacher_request.py --toyforge-c c/build/toyforge-c

diff-c-teacher-provenance: build-c
    uv run python scripts/diff_c_teacher_provenance.py --toyforge-c c/build/toyforge-c

diff-c-teacher-gate: build-c
    uv run python scripts/diff_c_teacher_gate.py --toyforge-c c/build/toyforge-c

diff-c-teacher-stamp: build-c
    uv run python scripts/diff_c_teacher_stamp.py --toyforge-c c/build/toyforge-c

# Official JSON Schema Test Suite (Draft 2020-12) — vendored supported-keyword
# files, with a documented skip list for unsupported vocabularies.
jsonschema-suite-c: build-c
    uv run python scripts/run_jsonschema_suite.py --toyforge-c c/build/toyforge-c

expand-from-seeds-c seeds_path="scenarios/smoke-tiny-seeds.yaml" out_dir="data/smoke-tiny" schemas_dir="schemas": build-c
    c/build/toyforge-c expand-from-seeds --seeds-path {{seeds_path}} --out-dir {{out_dir}} --schemas-dir {{schemas_dir}}

# Live-HTTP smoke tests against a local mock server (bind a loopback socket;
# kept out of c-port-cycle so the core gate stays network-free).
check-llamacpp-http-c: build-c
    uv run python scripts/check_c_llamacpp_http.py --toyforge-c c/build/toyforge-c

check-llamacpp-eval-c: build-c
    uv run python scripts/check_c_llamacpp_eval.py --toyforge-c c/build/toyforge-c

check-teacher-expand-c: build-c
    uv run python scripts/check_c_teacher_expand.py --toyforge-c c/build/toyforge-c

verify-seeds-c seeds_path="scenarios/seeds.yaml" schemas_dir="schemas": build-c
    c/build/toyforge-c verify-seeds --seeds-path {{seeds_path}} --schemas-dir {{schemas_dir}}

verify-data-c data_dir="data" schemas_dir="schemas": build-c
    c/build/toyforge-c verify-data --data-dir {{data_dir}} --schemas-dir {{schemas_dir}}

grammar-sha256-c grammar_path="schemas/jsonrpc.gbnf": build-c
    c/build/toyforge-c grammar-sha256 --grammar-path {{grammar_path}}

build-jsonrpc-gbnf-c schemas_dir="schemas" out_path="reports/c-port/jsonrpc.gbnf": build-c
    mkdir -p "$(dirname {{out_path}})"
    c/build/toyforge-c build-jsonrpc-gbnf --schemas-dir "{{schemas_dir}}" --out-path "{{out_path}}"

llamacpp-payload-c data_path="c/tests/fixtures/verify-data-one-split/train.jsonl" grammar_path="schemas/jsonrpc.gbnf" out_prefix="reports/c-port/llamacpp-payload" model="qwen" max_tokens="64" temperature="0.0": build-c
    mkdir -p "$(dirname {{out_prefix}})"
    c/build/toyforge-c llamacpp-payload --data-path "{{data_path}}" --grammar-path "{{grammar_path}}" --out-prefix "{{out_prefix}}" --model "{{model}}" --max-tokens "{{max_tokens}}" --temperature "{{temperature}}"

llamacpp-verify-response-c data_path="c/tests/fixtures/verify-data-one-split/train.jsonl" think_response="c/tests/fixtures/llamacpp-response/think.response.json" call_response="c/tests/fixtures/llamacpp-response/call.response.json" out_path="reports/c-port/llamacpp-model-output.txt" schemas_dir="schemas": build-c
    mkdir -p "$(dirname {{out_path}})"
    c/build/toyforge-c llamacpp-verify-response --data-path "{{data_path}}" --schemas-dir "{{schemas_dir}}" --think-response "{{think_response}}" --call-response "{{call_response}}" --out-path "{{out_path}}"

llamacpp-score-responses-c data_path="c/tests/fixtures/verify-data-one-split/train.jsonl" responses_dir="c/tests/fixtures/llamacpp-score-responses" schemas_dir="schemas" grammar_path="schemas/jsonrpc.gbnf" out_prefix="reports/c-port/c-llamacpp-responses" run_name="c-llamacpp-responses": build-c
    mkdir -p "$(dirname {{out_prefix}})"
    c/build/toyforge-c llamacpp-score-responses --data-path "{{data_path}}" --responses-dir "{{responses_dir}}" --schemas-dir "{{schemas_dir}}" --grammar-path "{{grammar_path}}" --out-prefix "{{out_prefix}}" --run-name "{{run_name}}"

score-data-c data_path="data/test.jsonl" schemas_dir="schemas" grammar_path="schemas/jsonrpc.gbnf" out_prefix="reports/c-port/c-score-data" run_name="c-score-data": build-c
    mkdir -p "$(dirname {{out_prefix}})"
    c/build/toyforge-c score-data --data-path "{{data_path}}" --schemas-dir "{{schemas_dir}}" --grammar-path "{{grammar_path}}" --out-prefix "{{out_prefix}}" --run-name "{{run_name}}"

compare-c reports_dir="c/tests/fixtures/compare" out_path="reports/c-port/c-compare.md" baseline="baseline-run" pattern="*.json": build-c
    mkdir -p "$(dirname {{out_path}})"
    c/build/toyforge-c compare --reports-dir "{{reports_dir}}" --out-path "{{out_path}}" --baseline "{{baseline}}" --pattern "{{pattern}}"

# Source-quality gate: flag Python-templating escape leaks (e.g. `%%` in a
# comment / non-printf string) that -Wformat can't see. No build needed.
check-format-leaks-c:
    uv run python scripts/check_format_leaks.py

c-port-cycle:
    just test-c
    just check-format-leaks-c
    just diff-c
    just diff-c-compare
    just diff-c-score-md
    just diff-c-traj
    just diff-c-yaml
    just diff-c-seeds
    just diff-c-expand
    just diff-c-jsonschema
    just diff-c-audit-independence
    just diff-c-analyze-rejections
    just diff-c-gbnf
    just diff-c-llamacpp-payload
    just diff-c-reward
    just diff-c-teacher-request
    just diff-c-teacher-provenance
    just diff-c-teacher-gate
    just diff-c-teacher-stamp
    just jsonschema-suite-c
    just score-data-c c/tests/fixtures/verify-data-one-split/train.jsonl
    just build-jsonrpc-gbnf-c
    just llamacpp-payload-c
    just llamacpp-verify-response-c
    just llamacpp-score-responses-c
    just compare-c
    just test-c-sanitize
    just diff-c-sanitize
    git diff --check

clean-c:
    rm -rf c/build c/build-sanitize

# --- Phase 0 (no GPU) ---
seeds: build-c
    c/build/toyforge-c verify-seeds --seeds-path scenarios/seeds.yaml --schemas-dir schemas

expand provider="anthropic" model="claude-sonnet-4-6" expansions="25": build-c
    base_url="${TOYFORGE_TEACHER_BASE_URL:-}"; \
    if [ -z "$base_url" ]; then \
      if [ "{{provider}}" = "anthropic" ]; then base_url="https://api.anthropic.com"; \
      else base_url="http://localhost:8080/v1"; fi; \
    fi; \
    api_key="${TOYFORGE_TEACHER_API_KEY:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-no-key}}}"; \
    c/build/toyforge-c teacher-expand \
      --seeds-path scenarios/seeds.yaml \
      --schemas-dir schemas \
      --out-dir data \
      --provider "{{provider}}" \
      --model "{{model}}" \
      --expansions-per-seed "{{expansions}}" \
      --base-url "$base_url" \
      --api-key "$api_key"

verify-data: build-c
    c/build/toyforge-c verify-data --data-dir data --schemas-dir schemas

phase0: seeds expand verify-data

# --- Audit / diagnostics ---
analyze-rejections out="reports/rejections.md": build-c
    c/build/toyforge-c analyze-rejections --input data/rejected/rejections.jsonl --output {{out}}

audit-independence out="reports/independence.md": build-c
    c/build/toyforge-c audit-independence --data-dir data --output {{out}}

# Strict-C port of the independence audit (same report; parity-gated by diff-c-audit-independence).
audit-independence-c out="reports/independence-c.md": build-c
    c/build/toyforge-c audit-independence --data-dir data --output {{out}}

analyze-rejections-c out="reports/rejections-c.md": build-c
    c/build/toyforge-c analyze-rejections --input data/rejected/rejections.jsonl --output {{out}}

tokenizer-audit model="Qwen/Qwen3-8B-Instruct" out="reports/tokenizer-audit.md":
    uv run python scripts/tokenizer_fragmentation_audit.py --model {{model}} --output {{out}}

# --- Phase 1 (GPU) ---
train config="train_configs/phase1-sft.yaml":
    uv run toyforge train --config-path {{config}}

eval-phase1 run_name="phase1-sft":
    uv run toyforge eval --run-name {{run_name}}

eval-b0 model="Qwen/Qwen3-8B-Instruct" run="B0-zero-shot":
    uv run toyforge eval-baseline --base-model {{model}} --run-name {{run}}

eval-b0-llamacpp model="Qwen3-8B-Instruct-GGUF" run="B0-llamacpp-constrained": build-c
    mkdir -p reports
    c/build/toyforge-c llamacpp-eval \
      --data-path data/test.jsonl \
      --schemas-dir schemas \
      --grammar-path schemas/jsonrpc.gbnf \
      --base-url http://localhost:8080/v1 \
      --model "{{model}}" \
      --run-name "{{run}}" \
      --out-prefix "reports/{{run}}" \
      --k 8 \
      --max-tokens 1024 \
      --temperature 0.0 \
      --sample-temperature 0.7; \
    rc=$?; test "$rc" -eq 0 -o "$rc" -eq 1

compare: build-c
    c/build/toyforge-c compare --reports-dir reports --out-path reports/compare.md

# Deploy seam: merge LoRA into base + export a GGUF for an external strict-C llama.cpp runtime.
merge-export adapter="out/phase1-sft/adapter" base="Qwen/Qwen3-8B-Instruct" out="out/export" quant="none":
    uv run toyforge merge-export --adapter-dir {{adapter}} --base-model {{base}} --out-dir {{out}} --quant {{quant}}

# End-to-end deploy → eval through the strict-C runtime — one documented flow (args are POSITIONAL):
#   1) just merge-export                                # adapter -> out/export/model-f16.gguf
#   2) just serve-strict-c out/export/model-f16.gguf    # start the server on :8080 (own terminal; blocks)
#   3) just eval-strict-c                               # eval through it; stamps the server build's sha if CSTRICT_REPO is set
# OPTIONAL and EXTERNAL: the strict-C llama.cpp server is not part of this repository. Point
# CSTRICT_SERVER at a built binary; optionally set CSTRICT_REPO to its source checkout so the
# scorecard records which build ran. Everything else in toyForge works without it.
cstrict-server := env_var_or_default("CSTRICT_SERVER", "llamachat-server")

# Serve an exported GGUF on the strict-C server (live-eval launch config; n_ctx/prompt sized for toyForge's ~2k-token prompts).
serve-strict-c gguf="out/export/model-f16.gguf" port="8080" n_ctx="5120" max_prompt="4096" max_tokens="256":
    command -v "{{cstrict-server}}" >/dev/null 2>&1 || { echo "strict-C server not found: set CSTRICT_SERVER to a built llamachat-server binary (an external build, not part of this repo)" >&2; exit 2; }
    "{{cstrict-server}}" {{gguf}} {{port}} {{n_ctx}} {{max_prompt}} {{max_tokens}}

# Eval the served GGUF through the strict-C server (constrained JSON-RPC, provider=local). Stamps the server build's sha into the scorecard when CSTRICT_REPO is set, otherwise "unknown".
eval-strict-c run_name="phase1-strict-c" model="toyforge-export" base_url="http://localhost:8080/v1" cstrict_repo=env_var_or_default("CSTRICT_REPO", ""): build-c
    mkdir -p reports
    c/build/toyforge-c llamacpp-eval \
      --data-path data/test.jsonl \
      --schemas-dir schemas \
      --grammar-path schemas/jsonrpc.gbnf \
      --base-url "{{base_url}}" \
      --model "{{model}}" \
      --run-name "{{run_name}}" \
      --out-prefix "reports/{{run_name}}" \
      --k 8 \
      --max-tokens 1024 \
      --temperature 0.0 \
      --sample-temperature 0.7 \
      --llamacpp-commit "$( { [ -n "{{cstrict_repo}}" ] && git -C {{cstrict_repo}} rev-parse --short HEAD 2>/dev/null; } || echo unknown)"; \
    rc=$?; test "$rc" -eq 0 -o "$rc" -eq 1

sweep spec:
    uv run python scripts/sweep.py {{spec}}

phase1: train eval-phase1

# Phase 2 (GRPO) is NOT implemented yet. Config, CLI dispatch and the verifier reward exist, but
# the trainer raises NotImplementedError, so this recipe stops with that message.
phase2-llamacpp config="train_configs/phase2-grpo-llamacpp.yaml":
    uv run toyforge train --config-path {{config}}

tensorboard run="phase1-sft":
    uv run tensorboard --logdir out/{{run}}

# --- Smoke (tiny model, fast iteration) ---
smoke-config := "train_configs/smoke.yaml"
smoke:
    uv run toyforge train --config-path {{smoke-config}}
    uv run toyforge eval --adapter-dir out/smoke/adapter --base-model Qwen/Qwen2.5-Coder-3B-Instruct --run-name smoke

# Pipeline smoke test — 5 seeds, no teacher API needed.
# Verifies the train→eval pipeline end-to-end without burning a full corpus.
smoke-tiny: install-train build-c
    mkdir -p data/smoke-tiny
    c/build/toyforge-c expand-from-seeds \
        --seeds-path scenarios/smoke-tiny-seeds.yaml \
        --schemas-dir schemas \
        --out-dir data/smoke-tiny
    uv run toyforge train --config-path train_configs/smoke-tiny.yaml
    uv run toyforge eval \
        --adapter-dir out/smoke-tiny/adapter \
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
        --test-path data/smoke-tiny/test.jsonl \
        --run-name smoke-tiny

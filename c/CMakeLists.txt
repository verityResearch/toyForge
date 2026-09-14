cmake_minimum_required(VERSION 3.20)

project(toyforge_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(toyforge_c
  src/io.c
  src/json.c
  src/jsonschema.c
  src/sha256.c
  src/schemas.c
  src/verifier.c
  src/reward.c
  src/yaml.c
  src/regex.c
)

target_include_directories(toyforge_c PUBLIC include)

# Optional libcurl-backed HTTP for live llama.cpp calls. Verifier-only builds do
# not require it; when absent, live-HTTP commands fail closed.
find_package(CURL QUIET)
if (CURL_FOUND)
  target_sources(toyforge_c PRIVATE src/http.c)
  target_compile_definitions(toyforge_c PUBLIC TF_HAVE_CURL=1)
  target_link_libraries(toyforge_c PUBLIC CURL::libcurl)
  message(STATUS "toyforge_c: libcurl found, live llama.cpp HTTP enabled")
else()
  message(STATUS "toyforge_c: libcurl not found, live llama.cpp HTTP disabled")
endif()

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set(TOYFORGE_C_WARNINGS -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(toyforge-c src/cli.c)
target_link_libraries(toyforge-c PRIVATE toyforge_c)

enable_testing()

add_executable(test_verifier tests/test_verifier.c)
target_link_libraries(test_verifier PRIVATE toyforge_c)
add_test(NAME verifier COMMAND test_verifier)

add_executable(test_json tests/test_json.c)
target_link_libraries(test_json PRIVATE toyforge_c)
add_test(NAME json COMMAND test_json)

add_executable(test_io tests/test_io.c)
target_link_libraries(test_io PRIVATE toyforge_c)
add_test(NAME io COMMAND test_io)

add_executable(test_jsonschema tests/test_jsonschema.c)
target_link_libraries(test_jsonschema PRIVATE toyforge_c)
add_test(NAME jsonschema COMMAND test_jsonschema)

add_executable(test_schemas tests/test_schemas.c)
target_link_libraries(test_schemas PRIVATE toyforge_c)
add_test(NAME schemas COMMAND test_schemas)

add_executable(test_sha256 tests/test_sha256.c)
target_link_libraries(test_sha256 PRIVATE toyforge_c)
add_test(NAME sha256 COMMAND test_sha256)

add_executable(test_yaml tests/test_yaml.c)
target_link_libraries(test_yaml PRIVATE toyforge_c)
add_test(NAME yaml COMMAND test_yaml)

add_executable(test_regex tests/test_regex.c)
target_link_libraries(test_regex PRIVATE toyforge_c)
add_test(NAME regex COMMAND test_regex)

add_test(
  NAME cli_verify_data
  COMMAND toyforge-c verify-data --data-dir ../tests/fixtures/verify-data-one-split --schemas-dir ../../schemas
)
set_tests_properties(
  cli_verify_data
  PROPERTIES PASS_REGULAR_EXPRESSION "train: 1/1 pass"
)
add_test(NAME cli_grammar_sha256 COMMAND toyforge-c grammar-sha256 --grammar-path ../../schemas/jsonrpc.gbnf)
set_tests_properties(
  cli_grammar_sha256
  PROPERTIES PASS_REGULAR_EXPRESSION "5a64ebe171e49a6e98abd6949ef55de2849eea40423e4bee4fd3f5c20b96126d"
)
add_test(
  NAME cli_build_jsonrpc_gbnf
  COMMAND toyforge-c build-jsonrpc-gbnf
          --schemas-dir ../../schemas
          --out-path c-jsonrpc.gbnf
)
set_tests_properties(
  cli_build_jsonrpc_gbnf
  PROPERTIES PASS_REGULAR_EXPRESSION "5a64ebe171e49a6e98abd6949ef55de2849eea40423e4bee4fd3f5c20b96126d"
)
add_test(
  NAME cli_llamacpp_payload
  COMMAND toyforge-c llamacpp-payload
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --grammar-path ../../schemas/jsonrpc.gbnf
          --out-prefix c-llamacpp-payload
          --model qwen
          --max-tokens 64
          --temperature 0.0
)
set_tests_properties(
  cli_llamacpp_payload
  PROPERTIES PASS_REGULAR_EXPRESSION
    "think_sha256=6ef0bcfbe6c4486749bb4701f7e41e8ddd328345613e859d44bbbef1181caaf6"
)
add_test(
  NAME cli_llamacpp_verify_response
  COMMAND toyforge-c llamacpp-verify-response
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --schemas-dir ../../schemas
          --think-response ../tests/fixtures/llamacpp-response/think.response.json
          --call-response ../tests/fixtures/llamacpp-response/call.response.json
          --out-path c-llamacpp-model-output.txt
)
set_tests_properties(
  cli_llamacpp_verify_response
  PROPERTIES PASS_REGULAR_EXPRESSION "passed=true"
)
add_test(
  NAME cli_llamacpp_verify_response_rejects_bad_call
  COMMAND toyforge-c llamacpp-verify-response
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --schemas-dir ../../schemas
          --think-response ../tests/fixtures/llamacpp-response/think.response.json
          --call-response ../tests/fixtures/llamacpp-response/call-bad.response.json
)
set_tests_properties(
  cli_llamacpp_verify_response_rejects_bad_call
  PROPERTIES WILL_FAIL TRUE
)
add_test(
  NAME cli_llamacpp_score_responses
  COMMAND toyforge-c llamacpp-score-responses
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --responses-dir ../tests/fixtures/llamacpp-score-responses
          --schemas-dir ../../schemas
          --grammar-path ../../schemas/jsonrpc.gbnf
          --out-prefix c-llamacpp-responses
          --run-name c-llamacpp-responses
)
set_tests_properties(
  cli_llamacpp_score_responses
  PROPERTIES PASS_REGULAR_EXPRESSION "llamacpp_score_responses: 1/1 pass"
)
add_test(
  NAME cli_llamacpp_score_responses_bad_fixture
  COMMAND toyforge-c llamacpp-score-responses
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --responses-dir ../tests/fixtures/llamacpp-score-responses-bad
          --schemas-dir ../../schemas
          --grammar-path ../../schemas/jsonrpc.gbnf
          --out-prefix c-llamacpp-responses-bad
          --run-name c-llamacpp-responses-bad
)
set_tests_properties(
  cli_llamacpp_score_responses_bad_fixture
  PROPERTIES PASS_REGULAR_EXPRESSION "llamacpp_score_responses: 0/1 pass"
)
add_test(
  NAME cli_score_data
  COMMAND toyforge-c score-data
          --data-path ../tests/fixtures/verify-data-one-split/train.jsonl
          --schemas-dir ../../schemas
          --grammar-path ../../schemas/jsonrpc.gbnf
          --out-prefix c-score-data
          --run-name c-score-data
)
set_tests_properties(
  cli_score_data
  PROPERTIES PASS_REGULAR_EXPRESSION "score_data: 1/1 pass"
)
add_test(
  NAME cli_score_data_bad_fixture
  COMMAND toyforge-c score-data
          --data-path ../tests/fixtures/verify-data-bad/train.jsonl
          --schemas-dir ../../schemas
          --grammar-path ../../schemas/jsonrpc.gbnf
          --out-prefix c-score-data-bad
          --run-name c-score-data-bad
)
set_tests_properties(
  cli_score_data_bad_fixture
  PROPERTIES PASS_REGULAR_EXPRESSION "score_data: 0/1 pass"
)
add_test(
  NAME cli_compare
  COMMAND toyforge-c compare
          --reports-dir ../tests/fixtures/compare
          --out-path c-compare.md
          --baseline baseline-run
)
set_tests_properties(
  cli_compare
  PROPERTIES PASS_REGULAR_EXPRESSION "\\| worse-run \\| 30 \\| \\*\\*0.500\\*\\* \\| -0.100"
)
add_test(
  NAME cli_compare_pattern_arm_summary
  COMMAND toyforge-c compare
          --reports-dir ../tests/fixtures/compare
          --out-path c-compare-arm.md
          --pattern "phase1-sft-*.json"
)
set_tests_properties(
  cli_compare_pattern_arm_summary
  PROPERTIES PASS_REGULAR_EXPRESSION "\\| phase1-sft \\| 3 \\| 0.620"
)
add_test(
  NAME cli_verify_data_skips_missing_splits
  COMMAND toyforge-c verify-data --data-dir ../tests/fixtures/verify-data-one-split --schemas-dir ../../schemas
)
set_tests_properties(
  cli_verify_data_skips_missing_splits
  PROPERTIES PASS_REGULAR_EXPRESSION "skip .*/dev.jsonl"
)
add_test(
  NAME cli_verify_data_rejects_bad_fixture
  COMMAND toyforge-c verify-data --data-dir ../tests/fixtures/verify-data-bad --schemas-dir ../../schemas
)
set_tests_properties(
  cli_verify_data_rejects_bad_fixture
  PROPERTIES WILL_FAIL TRUE
)
add_test(
  NAME cli_verify_seeds
  COMMAND toyforge-c verify-seeds
          --seeds-path ../tests/fixtures/verify-seeds/ok.yaml
          --schemas-dir ../../schemas
)
set_tests_properties(
  cli_verify_seeds
  PROPERTIES PASS_REGULAR_EXPRESSION "verify_seeds: 1/1 seeds pass"
)
add_test(
  NAME cli_verify_seeds_rejects_bad_fixture
  COMMAND toyforge-c verify-seeds
          --seeds-path ../tests/fixtures/verify-seeds/bad.yaml
          --schemas-dir ../../schemas
)
set_tests_properties(
  cli_verify_seeds_rejects_bad_fixture
  PROPERTIES WILL_FAIL TRUE
)
add_test(
  NAME cli_expand_from_seeds
  COMMAND toyforge-c expand-from-seeds
          --seeds-path ../tests/fixtures/verify-seeds/ok.yaml
          --schemas-dir ../../schemas
          --out-dir expand-out
)
set_tests_properties(
  cli_expand_from_seeds
  PROPERTIES PASS_REGULAR_EXPRESSION "expand_from_seeds: wrote 1 seeds"
)

foreach(target toyforge_c toyforge-c test_verifier test_json test_io test_jsonschema test_schemas test_sha256 test_yaml test_regex)
  if (TOYFORGE_C_WARNINGS)
    target_compile_options(${target} PRIVATE ${TOYFORGE_C_WARNINGS})
  endif()
endforeach()

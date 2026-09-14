#include "toyforge/reward.h"
#include "toyforge/schemas.h"
#include "toyforge/verifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define ASSERT_TRUE(expr)                                                                  \
  do {                                                                                     \
    if (!(expr)) {                                                                         \
      fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #expr);    \
      failures++;                                                                          \
    }                                                                                      \
  } while (0)

#define ASSERT_DOUBLE_EQ(actual, expected)                                                   \
  do {                                                                                       \
    if ((actual) != (expected)) {                                                            \
      fprintf(                                                                               \
        stderr,                                                                               \
        "ASSERT_DOUBLE_EQ failed at %s:%d: %s=%f expected=%f\n",                            \
        __FILE__,                                                                             \
        __LINE__,                                                                             \
        #actual,                                                                              \
        (double)(actual),                                                                     \
        (double)(expected)                                                                    \
      );                                                                                      \
      failures++;                                                                            \
    }                                                                                        \
  } while (0)

#define ASSERT_STR_EQ(actual, expected)                                                       \
  do {                                                                                       \
    if (strcmp((actual), (expected)) != 0) {                                                  \
      fprintf(                                                                               \
        stderr,                                                                               \
        "ASSERT_STR_EQ failed at %s:%d: %s='%s' expected='%s'\n",                           \
        __FILE__,                                                                             \
        __LINE__,                                                                             \
        #actual,                                                                              \
        (actual),                                                                             \
        (expected)                                                                            \
      );                                                                                      \
      failures++;                                                                            \
    }                                                                                        \
  } while (0)

static TfStepResult verify(const char *prior_state, const char *trigger, const char *after, const char *out) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!tf_schemas_load_dir(&schemas, "../../schemas", err, sizeof(err)) &&
      !tf_schemas_load_dir(&schemas, "schemas", err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    failures++;
    tf_schemas_load_builtin(&schemas);
  }
  TfVerifyContext ctx = {
    .prior_state = prior_state,
    .expected_trigger = trigger,
    .expected_state_after = after,
    .infer_trigger = false,
  };
  return tf_verify_step(&ctx, out, &schemas);
}

static void test_valid_ticket_open(void) {
  const char *out =
    "<think>I should open the ticket.</think>"
    "{\"jsonrpc\":\"2.0\",\"method\":\"ticket_open\","
    "\"params\":{\"requester_id\":\"user:1\",\"body_text\":\"YWJj\","
    "\"lifecycle_profile\":\"std\"},\"id\":1}";
  TfStepResult r = verify("NEW", "ticket_open.accepted", "TRIAGED", out);
  ASSERT_TRUE(r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.parse, 1.0);
  ASSERT_DOUBLE_EQ(r.subscores.schema, 1.0);
  ASSERT_DOUBLE_EQ(r.subscores.transition_valid, 1.0);
  ASSERT_STR_EQ(r.method, "ticket_open");
  ASSERT_STR_EQ(r.new_state, "TRIAGED");
}

static void test_missing_think_fails_parse(void) {
  TfStepResult r = verify(
    "NEW",
    "ticket_open.accepted",
    "TRIAGED",
    "{\"jsonrpc\":\"2.0\",\"method\":\"ticket_open\",\"params\":{},\"id\":1}"
  );
  ASSERT_TRUE(!r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.parse, 0.0);
}

static void test_malformed_json_number_fails_parse(void) {
  TfStepResult r = verify(
    "NEW",
    "ticket_open.accepted",
    "TRIAGED",
    "<think>x</think>{\"jsonrpc\":\"2.0\",\"method\":\"ticket_open\","
    "\"params\":{\"requester_id\":\"user:1\",\"body_text\":\"YWJj\","
    "\"lifecycle_profile\":\"std\"},\"id\":1e}"
  );
  ASSERT_TRUE(!r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.parse, 0.0);
}

static void test_wrong_prior_state_fails_precondition(void) {
  const char *out =
    "<think>x</think>{\"jsonrpc\":\"2.0\",\"method\":\"ticket_open\","
    "\"params\":{\"requester_id\":\"user:1\",\"body_text\":\"YWJj\","
    "\"lifecycle_profile\":\"std\"},\"id\":1}";
  TfStepResult r = verify("RESOLVED", "ticket_open.accepted", "TRIAGED", out);
  ASSERT_TRUE(!r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.schema, 1.0);
  ASSERT_DOUBLE_EQ(r.subscores.precondition_met, 0.0);
  ASSERT_DOUBLE_EQ(r.subscores.transition_valid, 0.0);
}

static void test_query_method_passes_without_transition(void) {
  const char *out =
    "<think>x</think>{\"jsonrpc\":\"2.0\",\"method\":\"ticket_status\","
    "\"params\":{\"ticket_id\":\"T-123\"},\"id\":1}";
  TfStepResult r = verify("RESOLVED", 0, 0, out);
  ASSERT_TRUE(r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.precondition_met, 1.0);
  ASSERT_DOUBLE_EQ(r.subscores.transition_valid, 1.0);
  ASSERT_STR_EQ(r.new_state, "RESOLVED");
}

static void test_schema_rejects_extra_property(void) {
  const char *out =
    "<think>x</think>{\"jsonrpc\":\"2.0\",\"method\":\"ticket_status\","
    "\"params\":{\"ticket_id\":\"T-123\",\"extra\":\"bad\"},\"id\":1}";
  TfStepResult r = verify("RESOLVED", 0, 0, out);
  ASSERT_TRUE(!r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.schema, 0.0);
  ASSERT_DOUBLE_EQ(r.subscores.transition_valid, 1.0);
}

static void test_infer_trigger_picks_state_arc(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!tf_schemas_load_dir(&schemas, "../../schemas", err, sizeof(err)) &&
      !tf_schemas_load_dir(&schemas, "schemas", err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    failures++;
    return;
  }
  TfVerifyContext ctx = {
    .prior_state = "ESCALATED",
    .expected_trigger = 0,
    .expected_state_after = 0,
    .infer_trigger = true,
  };
  const char *out =
    "<think>x</think>{\"jsonrpc\":\"2.0\",\"method\":\"admin_escalate\","
    "\"params\":{\"agent_id\":\"n-1\",\"ticket_id\":\"T-1\"},\"id\":1}";
  TfStepResult r = tf_verify_step(&ctx, out, &schemas);
  ASSERT_TRUE(r.passed);
  ASSERT_DOUBLE_EQ(r.subscores.transition_valid, 1.0);
  ASSERT_STR_EQ(r.new_state, "ENGINEERING");
}

static void test_reward_linear_combination(void) {
  TfSubscores sub = {
    .parse = 1.0,
    .schema = 1.0,
    .method_known = 1.0,
    .precondition_met = 0.0,
    .transition_valid = 0.0,
    .sequence_optimal = 0.0,
  };
  TfRubric rubric = {
    .parse = 0.1,
    .schema = 0.2,
    .method_known = 0.1,
    .precondition_met = 0.2,
    .transition_valid = 0.3,
    .sequence_optimal = 0.1,
  };
  ASSERT_DOUBLE_EQ(tf_compute_reward(sub, rubric), 0.4);
}

static void test_oversized_inputs_are_memory_safe(void) {
  // The verifier is the GRPO reward function and consumes untrusted model
  // output into fixed buffers (parsed_thinking[TF_MAX_TEXT], method[96], ...).
  // Oversized fields must be bounded/truncated, never overflow (ASan-checked).

  // A <think> far larger than TF_MAX_TEXT: parsed_thinking stays NUL-terminated
  // and within bounds; the call still parses.
  const size_t big = 8000;
  const size_t cap = big + 256;
  char *out = (char *)malloc(cap);
  ASSERT_TRUE(out != 0);
  int n = snprintf(out, cap, "<think>");
  memset(out + n, 'z', big);
  // Bound must be the ACTUAL remaining space (cap - used), not a constant that
  // overstates it — gcc -O2 fortify (-Wstringop-overflow) flags the mismatch.
  snprintf(
    out + n + big, cap - (size_t)n - big,
    "</think>{\"jsonrpc\":\"2.0\",\"method\":\"ticket_status\","
    "\"params\":{\"ticket_id\":\"o\"},\"id\":1}"
  );
  TfStepResult r = verify("RESOLVED", 0, 0, out);
  ASSERT_TRUE(strlen(r.parsed_thinking) < sizeof(r.parsed_thinking));
  ASSERT_DOUBLE_EQ(r.subscores.parse, 1.0);
  free(out);

  // A 300-char method name: the method buffer stays bounded and the method is
  // classified unknown (method_known=0), not an overflow.
  char m[512];
  int k = snprintf(m, sizeof(m), "<think>t</think>{\"jsonrpc\":\"2.0\",\"method\":\"");
  memset(m + k, 'm', 300);
  snprintf(m + k + 300, sizeof(m) - (size_t)k - 300, "\",\"params\":{},\"id\":1}");
  TfStepResult r2 = verify("NEW", 0, 0, m);
  ASSERT_TRUE(!r2.passed);
  ASSERT_TRUE(strlen(r2.method) < sizeof(r2.method));
  ASSERT_DOUBLE_EQ(r2.subscores.method_known, 0.0);
}

int main(void) {
  test_valid_ticket_open();
  test_missing_think_fails_parse();
  test_malformed_json_number_fails_parse();
  test_wrong_prior_state_fails_precondition();
  test_query_method_passes_without_transition();
  test_schema_rejects_extra_property();
  test_infer_trigger_picks_state_arc();
  test_reward_linear_combination();
  test_oversized_inputs_are_memory_safe();

  if (failures != 0) {
    fprintf(stderr, "%d verifier test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("verifier tests passed");
  return EXIT_SUCCESS;
}

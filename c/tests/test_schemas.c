#include "toyforge/schemas.h"
#include "toyforge/reward.h"
#include "toyforge/verifier.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define ASSERT_TRUE(expr)                                                               \
  do {                                                                                  \
    if (!(expr)) {                                                                      \
      fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      failures++;                                                                       \
    }                                                                                   \
  } while (0)

#define ASSERT_STR_EQ(actual, expected)                                                       \
  do {                                                                                        \
    if (strcmp((actual), (expected)) != 0) {                                                   \
      fprintf(                                                                                \
        stderr,                                                                                \
        "ASSERT_STR_EQ failed at %s:%d: %s='%s' expected='%s'\n",                            \
        __FILE__,                                                                              \
        __LINE__,                                                                              \
        #actual,                                                                               \
        (actual),                                                                              \
        (expected)                                                                             \
      );                                                                                       \
      failures++;                                                                              \
    }                                                                                          \
  } while (0)

static bool load_repo_schemas(TfSchemas *schemas, char *err, size_t err_cap) {
  if (tf_schemas_load_dir(schemas, "../../schemas", err, err_cap)) {
    return true;
  }
  return tf_schemas_load_dir(schemas, "schemas", err, err_cap);
}

static void test_schema_storage_fits_default_stack(void) {
  ASSERT_TRUE(sizeof(TfSchemas) <= 2u * 1024u * 1024u);
}

static void test_loads_method_catalog_and_state_machine(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!load_repo_schemas(&schemas, err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    ASSERT_TRUE(false);
    return;
  }
  if (err[0] != '\0') {
    fprintf(stderr, "schema load error: %s\n", err);
  }
  ASSERT_TRUE(schemas.method_count == 9);
  ASSERT_TRUE(schemas.state_count == 10);
  ASSERT_STR_EQ(schemas.initial_state, "NEW");
  ASSERT_TRUE(schemas.terminal_state_count == 1);
  ASSERT_STR_EQ(schemas.terminal_states[0], "ARCHIVED");
  ASSERT_TRUE(schemas.transition_count == 11);
  ASSERT_TRUE(tf_schemas_state_allowed(&schemas, "RESOLVED"));
  ASSERT_TRUE(!tf_schemas_state_allowed(&schemas, "MISSING"));

  const TfMethodSpec *status = tf_schemas_find_method(&schemas, "ticket_status");
  ASSERT_TRUE(status != 0);
  ASSERT_TRUE(status->query_only);
  ASSERT_TRUE(status->trigger_count == 0);
  ASSERT_TRUE(status->params_schema.loaded);
  ASSERT_TRUE(status->params_schema.property_count == 1);
  ASSERT_STR_EQ(status->params_schema.properties[0].name, "ticket_id");
  ASSERT_TRUE(status->params_schema.properties[0].required);
  ASSERT_TRUE(status->params_schema.properties[0].has_min_length);

  const TfMethodSpec *resolution_confirm = tf_schemas_find_method(&schemas, "resolution_confirm");
  ASSERT_TRUE(resolution_confirm != 0);
  ASSERT_TRUE(!resolution_confirm->query_only);
  ASSERT_TRUE(resolution_confirm->trigger_count == 2);
  ASSERT_STR_EQ(resolution_confirm->triggers[0], "resolution_confirm.passed");
  ASSERT_STR_EQ(resolution_confirm->triggers[1], "resolution_confirm.failed");

  const TfMethodSpec *ticket_history = tf_schemas_find_method(&schemas, "ticket_history");
  ASSERT_TRUE(ticket_history != 0);
  ASSERT_TRUE(ticket_history->params_schema.loaded);
  ASSERT_TRUE(ticket_history->params_schema.property_count == 2);
  ASSERT_STR_EQ(ticket_history->params_schema.properties[1].name, "window");
  ASSERT_TRUE(ticket_history->params_schema.properties[1].child_property_count == 2);

  ASSERT_STR_EQ(tf_schemas_transition_to(&schemas, "ESCALATED", "escalation.start"), "ENGINEERING");
  ASSERT_TRUE(tf_schemas_transition_to(&schemas, "NEW", "escalation.start") == 0);
}

static void test_loads_trajectory_schema_enum_ref(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!load_repo_schemas(&schemas, err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    ASSERT_TRUE(false);
    return;
  }

  ASSERT_TRUE(schemas.trajectory_schema_loaded);
  ASSERT_TRUE(schemas.trajectory_initial_state_enum_ref_resolved);
  ASSERT_STR_EQ(schemas.trajectory_initial_state_enum_ref, "state-machine.states");
  ASSERT_TRUE(schemas.trajectory_initial_state_enum_count == schemas.state_count);
  ASSERT_STR_EQ(schemas.trajectory_initial_state_enum_values[0], "NEW");
  ASSERT_STR_EQ(
    schemas.trajectory_initial_state_enum_values[schemas.trajectory_initial_state_enum_count - 1],
    "ARCHIVED"
  );
}

static void test_runtime_loaded_schemas_drive_verifier_transitions(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!load_repo_schemas(&schemas, err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    ASSERT_TRUE(false);
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
  TfStepResult result = tf_verify_step(&ctx, out, &schemas);
  ASSERT_TRUE(result.passed);
  ASSERT_STR_EQ(result.new_state, "ENGINEERING");
}

static void test_loads_reward_rubric_presets(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  if (!load_repo_schemas(&schemas, err, sizeof(err))) {
    fprintf(stderr, "schema load error: %s\n", err);
    ASSERT_TRUE(false);
    return;
  }

  ASSERT_STR_EQ(schemas.rubric_aggregation, "mean");
  ASSERT_TRUE(schemas.rubric_preset_count == 3);

  const TfRubricPreset *shaped = tf_schemas_find_rubric_preset(&schemas, "shaped");
  ASSERT_TRUE(shaped != 0);
  TfSubscores perfect = {
    .parse = 1.0,
    .schema = 1.0,
    .method_known = 1.0,
    .precondition_met = 1.0,
    .transition_valid = 1.0,
    .sequence_optimal = 1.0,
  };
  ASSERT_TRUE(tf_compute_reward(perfect, shaped->weights) > 0.999);
  ASSERT_TRUE(tf_compute_reward(perfect, shaped->weights) < 1.001);

  const TfRubricPreset *binary = tf_schemas_find_rubric_preset(&schemas, "binary");
  ASSERT_TRUE(binary != 0);
  TfSubscores no_transition = perfect;
  no_transition.transition_valid = 0.0;
  ASSERT_TRUE(tf_compute_reward(no_transition, binary->weights) == 0.0);

  const TfRubricPreset *schema_only = tf_schemas_find_rubric_preset(&schemas, "schema_only");
  ASSERT_TRUE(schema_only != 0);
  TfSubscores parse_only = {.parse = 1.0, .schema = 0.0};
  ASSERT_TRUE(tf_compute_reward(parse_only, schema_only->weights) == 0.5);
  ASSERT_TRUE(tf_schemas_find_rubric_preset(&schemas, "missing") == 0);
}

static void test_missing_schemas_dir_fails_closed(void) {
  TfSchemas schemas;
  char err[TF_MAX_ERROR] = "";
  ASSERT_TRUE(!tf_schemas_load_dir(&schemas, "does-not-exist", err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "failed to open") != 0);
  ASSERT_TRUE(schemas.method_count == 0);
}

int main(void) {
  test_schema_storage_fits_default_stack();
  if (failures != 0) {
    fprintf(stderr, "%d schemas test failures\n", failures);
    return EXIT_FAILURE;
  }

  test_loads_method_catalog_and_state_machine();
  test_loads_trajectory_schema_enum_ref();
  test_runtime_loaded_schemas_drive_verifier_transitions();
  test_loads_reward_rubric_presets();
  test_missing_schemas_dir_fails_closed();

  if (failures != 0) {
    fprintf(stderr, "%d schemas test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("schemas tests passed");
  return EXIT_SUCCESS;
}

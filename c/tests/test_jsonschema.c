#include "toyforge/json.h"
#include "toyforge/jsonschema.h"
#include "toyforge/schemas.h"

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

static TfJsonSpan parse_object(const char *json) {
  const char *start = tf_json_skip_ws(json);
  const char *end = start;
  ASSERT_TRUE(tf_json_skip_value(&end));
  ASSERT_TRUE(*tf_json_skip_ws(end) == '\0');
  return (TfJsonSpan){start, end};
}

static bool load_repo_schemas(TfSchemas *schemas, char *err, size_t err_cap) {
  if (tf_schemas_load_dir(schemas, "../../schemas", err, err_cap)) {
    return true;
  }
  return tf_schemas_load_dir(schemas, "schemas", err, err_cap);
}

static void test_ticket_open_accepts_valid_params(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object(
    "{\"requester_id\":\"user:1\",\"body_text\":\"YWJj\","
    "\"lifecycle_profile\":\"std\"}"
  );
  ASSERT_TRUE(tf_jsonschema_validate_method_params(&schemas, "ticket_open", params, err, sizeof(err)));
  ASSERT_TRUE(err[0] == '\0');
}

static void test_ticket_open_rejects_bad_requester_prefix(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object(
    "{\"requester_id\":\"ex:1\",\"body_text\":\"YWJj\","
    "\"lifecycle_profile\":\"std\"}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_method_params(&schemas, "ticket_open", params, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "requester_id") != 0);
}

static void test_rejects_additional_properties(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object("{\"ticket_id\":\"T-1\",\"extra\":\"bad\"}");
  ASSERT_TRUE(!tf_jsonschema_validate_method_params(&schemas, "ticket_status", params, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "additional property") != 0);
}

static void test_rejects_missing_required_property(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object("{}");
  ASSERT_TRUE(!tf_jsonschema_validate_method_params(&schemas, "ticket_status", params, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "ticket_id") != 0);
}

static void test_rejects_object_property_with_wrong_type(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object("{\"review_id\":\"c-1\",\"resolution_notes\":\"bad\"}");
  ASSERT_TRUE(!tf_jsonschema_validate_method_params(&schemas, "resolution_confirm", params, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "resolution_notes") != 0);
}

static void test_object_lookup_uses_last_duplicate_key_like_python(void) {
  TfJsonSpan obj = parse_object("{\"method\":\"ticket_status\",\"method\":\"ticket_open\"}");
  char value[64] = "";
  ASSERT_TRUE(tf_json_object_get_string(obj, "method", value, sizeof(value)));
  ASSERT_TRUE(strcmp(value, "ticket_open") == 0);
}

static void test_nested_malformed_json_is_rejected(void) {
  const char *json = "{\"resolution_notes\":{\"bad\":1e}}";
  const char *end = tf_json_skip_ws(json);
  ASSERT_TRUE(!tf_json_skip_value(&end));
}

static void test_audit_history_format_is_annotation_only_for_current_oracle(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan params = parse_object(
    "{\"ticket_id\":\"T-1\",\"window\":{\"from_utc\":\"not-a-date\"}}"
  );
  ASSERT_TRUE(tf_jsonschema_validate_method_params(&schemas, "ticket_history", params, err, sizeof(err)));
  ASSERT_TRUE(err[0] == '\0');
}

static void test_trajectory_accepts_valid_minimal_fixture(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\","
    "\"initial_state\":\"NEW\","
    "\"steps\":[{"
    "\"prompt_context\":{\"prior_state\":\"NEW\",\"prior_calls\":[{}]},"
    "\"thinking\":\"choose status\","
    "\"tool_call\":{\"jsonrpc\":\"2.0\",\"method\":\"ticket_status\",\"params\":{\"ticket_id\":\"T-1\"},\"id\":1},"
    "\"expected_state_after\":\"NEW\""
    "}],"
    "\"difficulty\":\"easy\","
    "\"source\":\"hand_seed\"}"
  );
  ASSERT_TRUE(tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(err[0] == '\0');
}

static void test_trajectory_rejects_unknown_initial_state(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\","
    "\"initial_state\":\"UNKNOWN\","
    "\"steps\":[{"
    "\"prompt_context\":{\"prior_state\":\"NEW\"},"
    "\"thinking\":\"x\","
    "\"tool_call\":{}"
    "}]}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "initial_state") != 0);
}

static void test_trajectory_rejects_empty_steps(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\",\"initial_state\":\"NEW\",\"steps\":[]}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "steps") != 0);
}

static void test_trajectory_rejects_missing_step_required_field(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\","
    "\"initial_state\":\"NEW\","
    "\"steps\":[{\"prompt_context\":{\"prior_state\":\"NEW\"},\"tool_call\":{}}]}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "thinking") != 0);
}

static void test_trajectory_rejects_bad_difficulty_enum(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\","
    "\"initial_state\":\"NEW\","
    "\"steps\":[{"
    "\"prompt_context\":{\"prior_state\":\"NEW\"},"
    "\"thinking\":\"x\","
    "\"tool_call\":{}"
    "}],"
    "\"difficulty\":\"extreme\"}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "difficulty") != 0);
}

static void test_trajectory_rejects_bad_prior_calls_type(void) {
  TfSchemas schemas;
  char err[256] = "";
  ASSERT_TRUE(load_repo_schemas(&schemas, err, sizeof(err)));
  TfJsonSpan trajectory = parse_object(
    "{\"trajectory_id\":\"t1\","
    "\"initial_state\":\"NEW\","
    "\"steps\":[{"
    "\"prompt_context\":{\"prior_state\":\"NEW\",\"prior_calls\":{}},"
    "\"thinking\":\"x\","
    "\"tool_call\":{}"
    "}]}"
  );
  ASSERT_TRUE(!tf_jsonschema_validate_trajectory(&schemas, trajectory, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "prior_calls") != 0);
}

static bool validates(const char *schema, const char *instance) {
  char err[256] = "";
  TfJsonSpan s = parse_object(schema);
  TfJsonSpan i = parse_object(instance);
  return tf_jsonschema_validate(s, i, err, sizeof(err));
}

static void test_general_validator(void) {
  struct {
    const char *schema;
    const char *instance;
    bool expect;
  } cases[] = {
    {"true", "5", true},
    {"false", "5", false},
    {"{\"type\":\"integer\"}", "5", true},
    {"{\"type\":\"integer\"}", "5.0", true},
    {"{\"type\":\"integer\"}", "5.5", false},
    {"{\"type\":\"string\"}", "5", false},
    {"{\"type\":[\"string\",\"number\"]}", "\"a\"", true},
    {"{\"type\":[\"string\",\"number\"]}", "true", false},
    {"{\"enum\":[1,2,3]}", "2", true},
    {"{\"enum\":[1,2,3]}", "4", false},
    {"{\"enum\":[\"a\",{\"k\":1}]}", "{\"k\":1}", true},
    {"{\"const\":\"2.0\"}", "\"2.0\"", true},
    {"{\"const\":\"2.0\"}", "\"2.1\"", false},
    {"{\"minimum\":3,\"maximum\":7}", "5", true},
    {"{\"minimum\":3}", "2", false},
    {"{\"exclusiveMinimum\":3}", "3", false},
    {"{\"exclusiveMaximum\":7}", "7", false},
    {"{\"multipleOf\":2}", "8", true},
    {"{\"multipleOf\":2}", "7", false},
    {"{\"minLength\":2,\"maxLength\":4}", "\"abc\"", true},
    {"{\"minLength\":2}", "\"a\"", false},
    {"{\"maxLength\":2}", "\"abc\"", false},
    {"{\"minItems\":1,\"maxItems\":2}", "[1]", true},
    {"{\"minItems\":2}", "[1]", false},
    {"{\"uniqueItems\":true}", "[1,2,1]", false},
    {"{\"uniqueItems\":true}", "[1,2,3]", true},
    {"{\"items\":{\"type\":\"integer\"}}", "[1,2,3]", true},
    {"{\"items\":{\"type\":\"integer\"}}", "[1,\"a\"]", false},
    {"{\"prefixItems\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}", "[\"a\",1]", true},
    {"{\"prefixItems\":[{\"type\":\"string\"}]}", "[1]", false},
    {"{\"required\":[\"a\"]}", "{\"a\":1}", true},
    {"{\"required\":[\"a\"]}", "{\"b\":1}", false},
    {"{\"properties\":{\"a\":{\"type\":\"integer\"}}}", "{\"a\":1}", true},
    {"{\"properties\":{\"a\":{\"type\":\"integer\"}}}", "{\"a\":\"x\"}", false},
    {"{\"additionalProperties\":false}", "{\"a\":1}", false},
    {"{\"properties\":{\"a\":{}},\"additionalProperties\":false}", "{\"a\":1}", true},
    {"{\"additionalProperties\":{\"type\":\"integer\"}}", "{\"a\":\"x\"}", false},
    {"{\"minProperties\":2}", "{\"a\":1}", false},
    {"{\"maxProperties\":1}", "{\"a\":1,\"b\":2}", false},
    {"{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}", "5", true},
    {"{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"boolean\"}]}", "5", false},
    {"{\"oneOf\":[{\"type\":\"integer\"},{\"minimum\":3}]}", "5", false},
    {"{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}", "5", true},
    {"{\"allOf\":[{\"type\":\"integer\"},{\"minimum\":3}]}", "5", true},
    {"{\"allOf\":[{\"type\":\"integer\"},{\"minimum\":7}]}", "5", false},
    {"{\"not\":{\"type\":\"string\"}}", "5", true},
    {"{\"not\":{\"type\":\"integer\"}}", "5", false},
  };
  size_t n = sizeof(cases) / sizeof(cases[0]);
  for (size_t i = 0; i < n; i++) {
    bool got = validates(cases[i].schema, cases[i].instance);
    if (got != cases[i].expect) {
      fprintf(
        stderr, "general validator case %zu: schema=%s instance=%s got=%d expect=%d\n", i,
        cases[i].schema, cases[i].instance, got, cases[i].expect
      );
      failures++;
    }
  }
}

int main(void) {
  test_general_validator();
  test_ticket_open_accepts_valid_params();
  test_ticket_open_rejects_bad_requester_prefix();
  test_rejects_additional_properties();
  test_rejects_missing_required_property();
  test_rejects_object_property_with_wrong_type();
  test_object_lookup_uses_last_duplicate_key_like_python();
  test_nested_malformed_json_is_rejected();
  test_audit_history_format_is_annotation_only_for_current_oracle();
  test_trajectory_accepts_valid_minimal_fixture();
  test_trajectory_rejects_unknown_initial_state();
  test_trajectory_rejects_empty_steps();
  test_trajectory_rejects_missing_step_required_field();
  test_trajectory_rejects_bad_difficulty_enum();
  test_trajectory_rejects_bad_prior_calls_type();

  if (failures != 0) {
    fprintf(stderr, "%d jsonschema test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("jsonschema tests passed");
  return EXIT_SUCCESS;
}

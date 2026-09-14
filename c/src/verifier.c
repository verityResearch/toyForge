#include "toyforge/verifier.h"

#include "toyforge/io.h"
#include "toyforge/json.h"
#include "toyforge/jsonschema.h"
#include "toyforge/schemas.h"

#include <stdio.h>
#include <string.h>

static TfStepResult fail_result(const char *msg) {
  TfStepResult result;
  memset(&result, 0, sizeof(result));
  result.passed = false;
  tf_copy_cstr(result.error_message, sizeof(result.error_message), msg);
  return result;
}

static bool method_can_emit(const TfMethodSpec *method, const char *trigger) {
  for (size_t i = 0; i < method->trigger_count; i++) {
    if (strcmp(method->triggers[i], trigger) == 0) {
      return true;
    }
  }
  return false;
}

TfStepResult tf_verify_step(
  const TfVerifyContext *ctx,
  const char *model_output,
  const TfSchemas *schemas
) {
  const char *p = tf_json_skip_ws(model_output);
  if (strncmp(p, "<think>", 7) != 0) {
    return fail_result("missing or malformed <think>...</think> wrapper");
  }
  const char *thinking_start = p + 7;
  const char *thinking_end = strstr(thinking_start, "</think>");
  if (thinking_end == 0) {
    return fail_result("missing or malformed <think>...</think> wrapper");
  }

  TfStepResult result;
  memset(&result, 0, sizeof(result));
  result.subscores.parse = 1.0;
  tf_copy_span(result.parsed_thinking, sizeof(result.parsed_thinking), thinking_start, thinking_end);

  const char *json_start = tf_json_skip_ws(thinking_end + 8);
  const char *json_end = json_start;
  if (!tf_json_skip_value(&json_end) || *tf_json_skip_ws(json_end) != '\0') {
    return fail_result("json after </think> did not parse");
  }
  if (*tf_json_skip_ws(json_start) != '{') {
    return fail_result("json after </think> is not an object");
  }
  TfJsonSpan call = {tf_json_skip_ws(json_start), json_end};

  if (!tf_json_object_has_key(call, "jsonrpc") || !tf_json_object_has_key(call, "method") ||
      !tf_json_object_has_key(call, "params") || !tf_json_object_has_key(call, "id")) {
    return fail_result("missing top-level keys");
  }

  char jsonrpc[32];
  if (!tf_json_object_get_string(call, "jsonrpc", jsonrpc, sizeof(jsonrpc)) ||
      strcmp(jsonrpc, "2.0") != 0) {
    return fail_result("jsonrpc must be '2.0'");
  }

  if (!tf_json_object_get_string(call, "method", result.method, sizeof(result.method))) {
    result.passed = false;
    result.subscores.method_known = 0.0;
    tf_copy_cstr(result.error_message, sizeof(result.error_message), "method is null in JSON-RPC body");
    return result;
  }

  const TfMethodSpec *method = tf_schemas_find_method(schemas, result.method);
  if (method == 0) {
    result.passed = false;
    result.subscores.method_known = 0.0;
    snprintf(result.error_message, sizeof(result.error_message), "unknown method: '%s'", result.method);
    return result;
  }
  result.subscores.method_known = 1.0;

  // Python parity: a non-object `params` is a schema failure, not a hard parse
  // fail. Python evaluates `schemas.method_validators[method].validate(params)`,
  // which raises ValidationError (schema=0) but leaves parse/method_known intact
  // and still computes the state subscores below. Returning fail_result() here
  // would zero parse and method_known and skip the state logic, diverging the
  // GRPO reward signal for reachable malformed-params outputs.
  char schema_err[TF_MAX_ERROR] = "";
  TfJsonSpan params;
  if (tf_json_object_get_object(call, "params", &params)) {
    if (tf_jsonschema_validate_method_params(
          schemas, result.method, params, schema_err, sizeof(schema_err))) {
      result.subscores.schema = 1.0;
    } else {
      result.subscores.schema = 0.0;
    }
  } else {
    result.subscores.schema = 0.0;
    tf_copy_cstr(schema_err, sizeof(schema_err), "params is not an object");
  }

  if (ctx->prior_state == 0 || ctx->prior_state[0] == '\0') {
    return fail_result("prompt_context missing 'prior_state'");
  }

  if (method->query_only) {
    result.subscores.precondition_met = 1.0;
    result.subscores.transition_valid = 1.0;
    tf_copy_cstr(result.new_state, sizeof(result.new_state), ctx->prior_state);
  } else {
    // One candidate per trigger the method can emit; bound by the trigger cap so
    // adding a method with >3 transition-valid triggers from one state can't
    // overflow this array.
    const char *candidate_states[TF_MAX_METHOD_TRIGGERS] = {0};
    size_t candidate_count = 0;
    for (size_t i = 0; i < method->trigger_count; i++) {
      const char *to = tf_schemas_transition_to(schemas, ctx->prior_state, method->triggers[i]);
      if (to != 0) {
        result.subscores.precondition_met = 1.0;
        candidate_states[candidate_count] = to;
        candidate_count++;
      }
    }

    if (ctx->infer_trigger && (ctx->expected_trigger == 0 || ctx->expected_trigger[0] == '\0')) {
      for (size_t i = 0; i < candidate_count; i++) {
        if (ctx->expected_state_after == 0 || ctx->expected_state_after[0] == '\0' ||
            strcmp(candidate_states[i], ctx->expected_state_after) == 0) {
          result.subscores.transition_valid = 1.0;
          tf_copy_cstr(result.new_state, sizeof(result.new_state), candidate_states[i]);
          break;
        }
      }
    } else if (ctx->expected_trigger != 0 && ctx->expected_trigger[0] != '\0' &&
               method_can_emit(method, ctx->expected_trigger)) {
      const char *to = tf_schemas_transition_to(schemas, ctx->prior_state, ctx->expected_trigger);
      if (to != 0 &&
          (ctx->expected_state_after == 0 || ctx->expected_state_after[0] == '\0' ||
           strcmp(to, ctx->expected_state_after) == 0)) {
        result.subscores.transition_valid = 1.0;
        tf_copy_cstr(result.new_state, sizeof(result.new_state), to);
      }
    }
  }

  result.subscores.sequence_optimal =
    (result.subscores.transition_valid == 1.0 || method->query_only) ? 1.0 : 0.0;

  if (result.subscores.transition_valid == 0.0 && !method->query_only && schema_err[0] == '\0') {
    snprintf(
      schema_err,
      sizeof(schema_err),
      "transition_valid=0: expected_trigger='%s' from prior_state='%s' did not validate",
      ctx->expected_trigger ? ctx->expected_trigger : "",
      ctx->prior_state
    );
  }

  result.passed = result.subscores.parse == 1.0 && result.subscores.schema == 1.0 &&
                  result.subscores.precondition_met == 1.0 &&
                  result.subscores.transition_valid == 1.0;
  if (schema_err[0] != '\0') {
    tf_copy_cstr(result.error_message, sizeof(result.error_message), schema_err);
  }
  return result;
}

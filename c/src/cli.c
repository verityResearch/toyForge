#include "toyforge/json.h"
#include "toyforge/jsonschema.h"
#include "toyforge/io.h"
#include "toyforge/schemas.h"
#include "toyforge/http.h"
#include "toyforge/reward.h"
#include "toyforge/sha256.h"
#include "toyforge/verifier.h"
#include "toyforge/yaml.h"

#include <time.h>

#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define TF_CLI_PATH_CAP 512
#define TF_CLI_NAME_CAP 128
#define TF_CLI_THINKING_CAP 8192
#define TF_CLI_PAYLOAD_TEXT_CAP 8192
#define TF_CLI_COMPARE_MAX_FAILURES 16

typedef struct {
  bool passed;
  char trajectory_id[96];
  char initial_state[96];
  bool has_first_failing_step;
  size_t first_failing_step;
  char first_failing_subscore[32];
  char error_message[TF_MAX_ERROR];
} TfTrajectorySummary;

typedef struct {
  size_t parse;
  size_t schema;
  size_t method_known;
  size_t precondition_met;
  size_t transition_valid;
  size_t sequence_optimal;
} TfFailureModes;

typedef struct {
  char name[64];
  size_t count;
} TfFailureItem;

typedef struct {
  char path[TF_CLI_PATH_CAP];
  char name[TF_CLI_NAME_CAP];
  long mtime;
} TfReportFile;

typedef struct {
  char run_name[TF_CLI_NAME_CAP];
  bool has_run_name;
  size_t n_examples;
  bool has_n_examples;
  double pass_at_1;
  bool has_pass_at_1;
  double pass_at_k;
  bool has_pass_at_k;
  double pass_at_maj;
  bool has_pass_at_maj;
  size_t k;
  bool has_k;
  TfFailureItem failures[TF_CLI_COMPARE_MAX_FAILURES];
  size_t failure_count;
} TfScoreRun;

typedef struct {
  const char *adapter_dir;
  const char *base_model;
  const char *eval_mode;
  size_t max_new_tokens;
  double temperature;
  const char *runtime;
  bool constrained_decoding;
  const char *llamacpp_base_url;
  const char *llamacpp_model;
  const char *llamacpp_commit;
} TfScoreMeta;

static void usage(FILE *stream) {
  fprintf(
    stream,
    "usage:\n"
    "  toyforge-c verify-step --prior-state STATE [--expected-trigger TRIGGER]\n"
    "                         [--expected-state-after STATE] [--infer-trigger]\n"
    "                         [--schemas-dir DIR] [--rubric-preset NAME]\n"
    "                         --output '<think>...</think>{...}'\n"
    "  toyforge-c verify-data [--data-dir DIR] [--schemas-dir DIR]\n"
    "  toyforge-c audit-independence [--data-dir DIR] [--output PATH]\n"
    "  toyforge-c analyze-rejections [--input PATH] [--output PATH]\n"
    "  toyforge-c score-data [--data-path PATH] [--schemas-dir DIR]\n"
    "                         [--grammar-path PATH] [--out-prefix PATH]\n"
    "                         [--run-name NAME]\n"
    "  toyforge-c compare [--reports-dir DIR] [--out-path PATH]\n"
    "                      [--baseline NAME] [--pattern GLOB] [--last N]\n"
    "  toyforge-c build-jsonrpc-gbnf [--schemas-dir DIR] [--out-path PATH]\n"
    "  toyforge-c llamacpp-payload --data-path PATH [--step-index N]\n"
    "                              [--grammar-path PATH] [--out-prefix PATH]\n"
    "                              [--model NAME] [--max-tokens N]\n"
    "                              [--temperature FLOAT] [--thinking TEXT]\n"
    "  toyforge-c llamacpp-verify-response --data-path PATH --think-response PATH\n"
    "                                      --call-response PATH [--step-index N]\n"
    "                                      [--schemas-dir DIR] [--out-path PATH]\n"
    "  toyforge-c llamacpp-score-responses --data-path PATH --responses-dir DIR\n"
    "                                     [--schemas-dir DIR] [--grammar-path PATH]\n"
    "                                     [--out-prefix PATH] [--run-name NAME]\n"
    "  toyforge-c llamacpp-eval --data-path PATH [--schemas-dir DIR]\n"
    "                           [--grammar-path PATH] [--base-url URL]\n"
    "                           [--api-key KEY] [--out-prefix PATH]\n"
    "                           [--run-name NAME] [--model NAME] [--k N]\n"
    "                           [--llamacpp-commit SHA]\n"
    "  toyforge-c grammar-sha256 [--grammar-path PATH]\n"
    "  toyforge-c teacher-request --seed-path PATH [--model NAME]\n"
    "                             [--max-tokens N] [--temperature FLOAT]\n"
    "                             [--out-path PATH]\n"
    "  toyforge-c teacher-provenance --seed-trajectory-id ID [--provider NAME]\n"
    "                                [--model NAME] [--expansion-seed N]\n"
    "                                [--temperature FLOAT] [--grammar-path PATH]\n"
    "                                [--timestamp ISO] [--toyforge-version V]\n"
    "                                [--out-path PATH]\n"
    "  toyforge-c teacher-gate --response-path PATH [--schemas-dir DIR]\n"
    "  toyforge-c teacher-stamp --trajectory-path PATH --seed-trajectory-id ID\n"
    "                           [--new-trajectory-id ID] [--provider NAME] [--model NAME]\n"
    "                           [--expansion-seed N] [--temperature FLOAT]\n"
    "                           [--grammar-path PATH] [--timestamp ISO]\n"
    "                           [--toyforge-version V] [--out-path PATH]\n"
    "  toyforge-c teacher-expand [--seeds-path PATH] [--schemas-dir DIR] [--out-dir DIR]\n"
    "                            [--base-url URL] [--api-key KEY] [--provider NAME]\n"
    "                            [--model NAME] [--expansions-per-seed N] [--max-retries N]\n"
    "                            [--shuffle-seed N] [--train-frac F] [--dev-frac F]\n"
    "                            [--temperature F] [--max-tokens N] [--toyforge-version V]\n"
  );
}

static bool load_cli_schemas(TfSchemas *schemas, const char *schemas_dir) {
  char schema_err[TF_MAX_ERROR] = "";
  if (!tf_schemas_load_dir(schemas, schemas_dir, schema_err, sizeof(schema_err))) {
    fprintf(stderr, "schema load failed: %s\n", schema_err);
    return false;
  }
  return true;
}

static void print_step_result(const TfStepResult *result) {
  printf("passed=%s\n", result->passed ? "true" : "false");
  printf(
    "subscores parse=%.1f schema=%.1f method_known=%.1f precondition_met=%.1f "
    "transition_valid=%.1f sequence_optimal=%.1f\n",
    result->subscores.parse,
    result->subscores.schema,
    result->subscores.method_known,
    result->subscores.precondition_met,
    result->subscores.transition_valid,
    result->subscores.sequence_optimal
  );
  if (result->method[0] != '\0') {
    printf("method=%s\n", result->method);
  }
  if (result->new_state[0] != '\0') {
    printf("new_state=%s\n", result->new_state);
  }
  if (result->error_message[0] != '\0') {
    printf("error=%s\n", result->error_message);
  }
}

static char *trim_line(char *line) {
  while (*line && isspace((unsigned char)*line)) {
    line++;
  }
  char *end = line + strlen(line);
  while (end > line && isspace((unsigned char)end[-1])) {
    end--;
  }
  *end = '\0';
  return line;
}

static bool parse_json_line(const char *line, TfJsonSpan *out) {
  const char *start = tf_json_skip_ws(line);
  if (*start == '\0') {
    return false;
  }
  const char *end = start;
  if (!tf_json_skip_value(&end) || *tf_json_skip_ws(end) != '\0') {
    return false;
  }
  out->start = start;
  out->end = end;
  return true;
}

static bool parse_cli_size(const char *text, size_t *out) {
  if (text == 0 || *text == '\0' || *text == '-' || *text == '+') {
    return false;
  }
  char *end = 0;
  unsigned long parsed = strtoul(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  *out = (size_t)parsed;
  return true;
}

static bool get_optional_string(
  TfJsonSpan object,
  const char *key,
  char *out,
  size_t out_cap,
  bool *present
) {
  *present = false;
  if (!tf_json_object_has_key(object, key)) {
    return true;
  }
  if (!tf_json_object_get_string(object, key, out, out_cap)) {
    return false;
  }
  *present = true;
  return true;
}

static void summary_set_error(
  TfTrajectorySummary *summary,
  const char *subscore,
  bool has_step,
  size_t step,
  const char *message
) {
  summary->passed = false;
  summary->has_first_failing_step = has_step;
  summary->first_failing_step = step;
  tf_copy_cstr(summary->first_failing_subscore, sizeof(summary->first_failing_subscore), subscore);
  tf_copy_cstr(summary->error_message, sizeof(summary->error_message), message);
}

static void format_step_message(
  char *out,
  size_t out_cap,
  const char *prefix,
  size_t step,
  const char *suffix,
  const char *message
) {
  int wrote = snprintf(out, out_cap, "%s%zu%s", prefix, step, suffix);
  if (wrote < 0 || (size_t)wrote >= out_cap) {
    if (out_cap > 0) {
      out[out_cap - 1] = '\0';
    }
    return;
  }
  tf_copy_cstr(out + wrote, out_cap - (size_t)wrote, message);
}

static const char *first_failing_subscore(const TfStepResult *result) {
  if (result->subscores.parse < 1.0) {
    return "parse";
  }
  if (result->subscores.schema < 1.0) {
    return "schema";
  }
  if (result->subscores.method_known < 1.0) {
    return "method_known";
  }
  if (result->subscores.precondition_met < 1.0) {
    return "precondition_met";
  }
  if (result->subscores.transition_valid < 1.0) {
    return "transition_valid";
  }
  if (result->subscores.sequence_optimal < 1.0) {
    return "sequence_optimal";
  }
  return "";
}

static void count_failure_mode(TfFailureModes *modes, const char *subscore) {
  if (strcmp(subscore, "parse") == 0) {
    modes->parse++;
  } else if (strcmp(subscore, "schema") == 0) {
    modes->schema++;
  } else if (strcmp(subscore, "method_known") == 0) {
    modes->method_known++;
  } else if (strcmp(subscore, "precondition_met") == 0) {
    modes->precondition_met++;
  } else if (strcmp(subscore, "transition_valid") == 0) {
    modes->transition_valid++;
  } else if (strcmp(subscore, "sequence_optimal") == 0) {
    modes->sequence_optimal++;
  }
}

static char *build_model_output(TfJsonSpan step, char *err, size_t err_cap) {
  char thinking[TF_CLI_THINKING_CAP];
  if (!tf_json_object_get_string(step, "thinking", thinking, sizeof(thinking))) {
    tf_copy_cstr(err, err_cap, "thinking is too large or invalid");
    return 0;
  }

  TfJsonSpan tool_call;
  if (!tf_json_object_get_object(step, "tool_call", &tool_call)) {
    tf_copy_cstr(err, err_cap, "tool_call must be an object");
    return 0;
  }

  const char *prefix = "<think>";
  const char *suffix = "</think>";
  size_t prefix_len = strlen(prefix);
  size_t suffix_len = strlen(suffix);
  size_t thinking_len = strlen(thinking);
  size_t tool_len = (size_t)(tool_call.end - tool_call.start);
  if (thinking_len > (size_t)-1 - prefix_len - suffix_len - tool_len - 1) {
    tf_copy_cstr(err, err_cap, "model output is too large");
    return 0;
  }
  size_t total = prefix_len + thinking_len + suffix_len + tool_len;
  char *out = (char *)malloc(total + 1);
  if (out == 0) {
    tf_copy_cstr(err, err_cap, "out of memory building model output");
    return 0;
  }

  char *p = out;
  memcpy(p, prefix, prefix_len);
  p += prefix_len;
  memcpy(p, thinking, thinking_len);
  p += thinking_len;
  memcpy(p, suffix, suffix_len);
  p += suffix_len;
  memcpy(p, tool_call.start, tool_len);
  p += tool_len;
  *p = '\0';
  return out;
}

static bool summarize_trajectory_span(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  const char *fallback_id,
  TfTrajectorySummary *summary
) {
  memset(summary, 0, sizeof(*summary));
  tf_copy_cstr(summary->trajectory_id, sizeof(summary->trajectory_id), fallback_id);
  (void)tf_json_object_get_string(
    trajectory,
    "trajectory_id",
    summary->trajectory_id,
    sizeof(summary->trajectory_id)
  );

  char err[TF_MAX_ERROR] = "";
  if (!tf_jsonschema_validate_trajectory(schemas, trajectory, err, sizeof(err))) {
    summary_set_error(summary, "schema", false, 0, err);
    return false;
  }

  char current_state[96];
  if (!tf_json_object_get_string(trajectory, "initial_state", current_state, sizeof(current_state))) {
    summary_set_error(
      summary,
      "schema",
      false,
      0,
      "initial_state is required and must fit in the C state buffer"
    );
    return false;
  }
  tf_copy_cstr(summary->initial_state, sizeof(summary->initial_state), current_state);

  TfJsonSpan steps;
  if (!tf_json_object_get_value(trajectory, "steps", &steps)) {
    summary_set_error(summary, "schema", false, 0, "steps is required");
    return false;
  }
  size_t step_count = 0;
  if (!tf_json_array_count(steps, &step_count)) {
    summary_set_error(summary, "schema", false, 0, "steps must be an array");
    return false;
  }

  for (size_t i = 0; i < step_count; i++) {
    TfJsonSpan step;
    if (!tf_json_array_get(steps, i, &step)) {
      snprintf(err, sizeof(err), "steps[%zu] is malformed", i);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }

    char expected_trigger[96] = "";
    char expected_state_after[96] = "";
    bool has_expected_trigger = false;
    bool has_expected_state_after = false;
    if (!get_optional_string(
          step,
          "expected_trigger",
          expected_trigger,
          sizeof(expected_trigger),
          &has_expected_trigger
        ) ||
        !get_optional_string(
          step,
          "expected_state_after",
          expected_state_after,
          sizeof(expected_state_after),
          &has_expected_state_after
        )) {
      snprintf(err, sizeof(err), "steps[%zu] has an invalid expected state field", i);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }

    char output_err[TF_MAX_ERROR] = "";
    char *model_output = build_model_output(step, output_err, sizeof(output_err));
    if (model_output == 0) {
      format_step_message(err, sizeof(err), "steps[", i, "]: ", output_err);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }

    TfVerifyContext ctx = {
      .prior_state = current_state,
      .expected_trigger = has_expected_trigger ? expected_trigger : 0,
      .expected_state_after = has_expected_state_after ? expected_state_after : 0,
      .infer_trigger = false,
    };
    TfStepResult result = tf_verify_step(&ctx, model_output, schemas);
    free(model_output);
    if (!result.passed) {
      format_step_message(
        err,
        sizeof(err),
        "step ",
        i,
        " failed: ",
        result.error_message[0] != '\0' ? result.error_message : "?"
      );
      summary_set_error(summary, first_failing_subscore(&result), true, i, err);
      return false;
    }
    if (result.new_state[0] != '\0') {
      tf_copy_cstr(current_state, sizeof(current_state), result.new_state);
    }
  }

  char final_state[96] = "";
  bool has_final_state = false;
  if (!get_optional_string(
        trajectory,
        "final_state",
        final_state,
        sizeof(final_state),
        &has_final_state
      )) {
    summary_set_error(summary, "schema", false, 0, "final_state is too large or invalid");
    return false;
  }
  if (has_final_state && strcmp(current_state, final_state) != 0) {
    snprintf(
      err,
      sizeof(err),
      "final state '%s' does not match expected '%s'",
      current_state,
      final_state
    );
    summary_set_error(summary, "", false, 0, err);
    return false;
  }
  summary->passed = true;
  return true;
}

static bool verify_trajectory_span(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  char *err,
  size_t err_cap
) {
  TfTrajectorySummary summary;
  bool ok = summarize_trajectory_span(schemas, trajectory, "trajectory", &summary);
  if (!ok) {
    tf_copy_cstr(err, err_cap, summary.error_message);
  }
  return ok;
}

static int run_verify_step(int argc, char **argv) {
  const char *schemas_dir = "schemas";

  const char *prior_state = 0;
  const char *expected_trigger = 0;
  const char *expected_state_after = 0;
  const char *output = 0;
  const char *rubric_preset = 0;
  bool infer_trigger = false;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--prior-state") == 0 && i + 1 < argc) {
      prior_state = argv[++i];
    } else if (strcmp(argv[i], "--expected-trigger") == 0 && i + 1 < argc) {
      expected_trigger = argv[++i];
    } else if (strcmp(argv[i], "--expected-state-after") == 0 && i + 1 < argc) {
      expected_state_after = argv[++i];
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--rubric-preset") == 0 && i + 1 < argc) {
      rubric_preset = argv[++i];
    } else if (strcmp(argv[i], "--infer-trigger") == 0) {
      infer_trigger = true;
    } else {
      usage(stderr);
      return 2;
    }
  }

  if (prior_state == 0 || output == 0) {
    usage(stderr);
    return 2;
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }
  TfVerifyContext ctx = {
    .prior_state = prior_state,
    .expected_trigger = expected_trigger,
    .expected_state_after = expected_state_after,
    .infer_trigger = infer_trigger,
  };
  TfStepResult result = tf_verify_step(&ctx, output, &schemas);
  print_step_result(&result);
  // With --rubric-preset, also emit the GRPO reward (weighted subscore sum) for
  // that preset, matching toyforge.verifier.reward.compute_reward.
  if (rubric_preset != 0) {
    const TfRubricPreset *preset = tf_schemas_find_rubric_preset(&schemas, rubric_preset);
    if (preset == 0) {
      fprintf(stderr, "unknown rubric preset: %s\n", rubric_preset);
      return 2;
    }
    printf("reward=%.6f\n", tf_compute_reward(result.subscores, preset->weights));
  }
  return result.passed ? 0 : 1;
}

static int run_verify_data(int argc, char **argv) {
  const char *data_dir = "data";
  const char *schemas_dir = "schemas";

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  const char *splits[] = {"train", "dev", "test"};
  int overall_failed = 0;
  for (size_t split_i = 0; split_i < sizeof(splits) / sizeof(splits[0]); split_i++) {
    char filename[32];
    char path[TF_CLI_PATH_CAP];
    snprintf(filename, sizeof(filename), "%s.jsonl", splits[split_i]);
    if (!tf_join_path(path, sizeof(path), data_dir, filename)) {
      fprintf(stderr, "data path is too long for %s\n", filename);
      return 2;
    }

    FILE *f = fopen(path, "rb");
    if (f == 0) {
      printf("skip %s (missing)\n", path);
      continue;
    }

    size_t ok = 0;
    size_t total = 0;
    char *line = 0;
    while ((line = tf_read_line(f)) != 0) {
      char *trimmed = trim_line(line);
      if (*trimmed == '\0') {
        free(line);
        continue;
      }

      total++;
      TfJsonSpan trajectory;
      if (!parse_json_line(trimmed, &trajectory)) {
        printf("malformed line in %s: invalid JSON\n", filename);
        overall_failed++;
        free(line);
        continue;
      }

      char trajectory_id[96];
      if (!tf_json_object_get_string(
            trajectory,
            "trajectory_id",
            trajectory_id,
            sizeof(trajectory_id)
          )) {
        snprintf(trajectory_id, sizeof(trajectory_id), "line-%zu", total);
      }

      char err[TF_MAX_ERROR] = "";
      if (verify_trajectory_span(&schemas, trajectory, err, sizeof(err))) {
        ok++;
      } else {
        overall_failed++;
        printf("FAIL %s: %s\n", trajectory_id, err);
      }
      free(line);
    }
    fclose(f);
    printf("%s: %zu/%zu pass\n", splits[split_i], ok, total);
  }
  return overall_failed == 0 ? 0 : 1;
}

static bool append_trajectory_summary(
  TfTrajectorySummary **entries,
  size_t *count,
  size_t *cap,
  const TfTrajectorySummary *summary
) {
  if (*count == *cap) {
    size_t new_cap = *cap == 0 ? 32 : *cap * 2;
    if (new_cap < *cap) {
      return false;
    }
    TfTrajectorySummary *new_entries =
      (TfTrajectorySummary *)realloc(*entries, new_cap * sizeof(**entries));
    if (new_entries == 0) {
      return false;
    }
    *entries = new_entries;
    *cap = new_cap;
  }
  (*entries)[*count] = *summary;
  (*count)++;
  return true;
}

static bool make_suffixed_path(char *out, size_t out_cap, const char *prefix, const char *suffix) {
  int wrote = snprintf(out, out_cap, "%s%s", prefix, suffix);
  return wrote >= 0 && (size_t)wrote < out_cap;
}

static bool write_json_string_value(FILE *out, const char *value) {
  size_t len = strlen(value);
  if (len > ((size_t)-1 - 3) / 6) {
    return false;
  }
  size_t cap = len * 6 + 3;
  char *encoded = (char *)malloc(cap);
  if (encoded == 0) {
    return false;
  }
  bool ok = tf_json_write_string(value, encoded, cap);
  if (ok && fputs(encoded, out) == EOF) {
    ok = false;
  }
  free(encoded);
  return ok;
}

// Emit a (validated) JSON value span with insignificant whitespace removed, so a
// possibly pretty-printed value can be written as a single JSONL line. String
// contents (including escapes) are copied verbatim; only whitespace between
// tokens is dropped.
static bool json_minify_to(FILE *out, TfJsonSpan v) {
  const char *p = v.start;
  bool in_str = false;
  while (p < v.end) {
    char c = *p;
    if (in_str) {
      if (fputc(c, out) == EOF) {
        return false;
      }
      if (c == '\\') {
        p++;
        if (p < v.end && fputc(*p, out) == EOF) {
          return false;
        }
      } else if (c == '"') {
        in_str = false;
      }
      p++;
      continue;
    }
    if (c == '"') {
      in_str = true;
      if (fputc(c, out) == EOF) {
        return false;
      }
    } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      if (fputc(c, out) == EOF) {
        return false;
      }
    }
    p++;
  }
  return !ferror(out);
}

static bool write_json_string_field(FILE *out, const char *key, const char *value, bool comma) {
  fprintf(out, "  \"%s\": ", key);
  if (!write_json_string_value(out, value)) {
    return false;
  }
  fprintf(out, "%s\n", comma ? "," : "");
  return !ferror(out);
}

static bool write_failure_count(
  FILE *out,
  bool *wrote_any,
  const char *name,
  size_t count
) {
  if (count == 0) {
    return true;
  }
  fprintf(out, "%s\n    \"%s\": %zu", *wrote_any ? "," : "", name, count);
  *wrote_any = true;
  return !ferror(out);
}

static bool write_failure_modes_json(FILE *out, const TfFailureModes *modes) {
  fputs("  \"failure_modes\": {", out);
  bool wrote_any = false;
  if (!write_failure_count(out, &wrote_any, "parse", modes->parse) ||
      !write_failure_count(out, &wrote_any, "schema", modes->schema) ||
      !write_failure_count(out, &wrote_any, "method_known", modes->method_known) ||
      !write_failure_count(out, &wrote_any, "precondition_met", modes->precondition_met) ||
      !write_failure_count(out, &wrote_any, "transition_valid", modes->transition_valid) ||
      !write_failure_count(out, &wrote_any, "sequence_optimal", modes->sequence_optimal)) {
    return false;
  }
  fputs(wrote_any ? "\n  },\n" : "},\n", out);
  return !ferror(out);
}

static bool write_nullable_json_string(FILE *out, const char *value) {
  if (value == 0 || value[0] == '\0') {
    return fputs("null", out) != EOF;
  }
  return write_json_string_value(out, value);
}

static bool write_score_json(
  const char *path,
  const char *run_name,
  const char *data_path,
  const char *test_set_sha256,
  const char *grammar_sha256,
  const TfScoreMeta *meta,
  const TfTrajectorySummary *entries,
  size_t entry_count,
  size_t passed_count,
  double pass_at_1,
  double pass_at_k,
  double pass_at_maj,
  size_t k_value,
  const TfFailureModes *failure_modes
) {
  (void)passed_count;
  FILE *out = fopen(path, "wb");
  if (out == 0) {
    return false;
  }

  fputs("{\n", out);
  if (!write_json_string_field(out, "run_name", run_name, true)) {
    fclose(out);
    return false;
  }
  fprintf(out, "  \"n_examples\": %zu,\n", entry_count);
  fprintf(out, "  \"pass_at_1\": %.6f,\n", pass_at_1);
  fprintf(out, "  \"pass_at_k\": %.6f,\n", pass_at_k);
  fprintf(out, "  \"pass_at_maj\": %.6f,\n", pass_at_maj);
  fprintf(out, "  \"k\": %zu,\n", k_value);
  if (!write_failure_modes_json(out, failure_modes)) {
    fclose(out);
    return false;
  }
  fputs("  \"per_trajectory\": [\n", out);
  for (size_t i = 0; i < entry_count; i++) {
    const TfTrajectorySummary *entry = &entries[i];
    fputs("    {\n", out);
    fputs("      \"trajectory_id\": ", out);
    if (!write_json_string_value(out, entry->trajectory_id)) {
      fclose(out);
      return false;
    }
    fputs(",\n      \"initial_state\": ", out);
    if (!write_json_string_value(out, entry->initial_state)) {
      fclose(out);
      return false;
    }
    fprintf(out, ",\n      \"passed\": %s,\n", entry->passed ? "true" : "false");
    fputs("      \"first_failing_step\": ", out);
    if (entry->has_first_failing_step) {
      fprintf(out, "%zu", entry->first_failing_step);
    } else {
      fputs("null", out);
    }
    fputs(",\n      \"first_failing_subscore\": ", out);
    if (!write_nullable_json_string(out, entry->first_failing_subscore)) {
      fclose(out);
      return false;
    }
    fputs(",\n      \"error_message\": ", out);
    if (!write_nullable_json_string(out, entry->passed ? "" : entry->error_message)) {
      fclose(out);
      return false;
    }
    fprintf(out, "\n    }%s\n", i + 1 == entry_count ? "" : ",");
  }
  fputs("  ],\n", out);
  if (!write_json_string_field(out, "eval_timestamp_utc", "", true) ||
      !write_json_string_field(out, "git_sha", "", true) ||
      !write_json_string_field(out, "adapter_dir", meta->adapter_dir, true) ||
      !write_json_string_field(out, "base_model", meta->base_model, true) ||
      !write_json_string_field(out, "test_set_path", data_path, true) ||
      !write_json_string_field(out, "test_set_sha256", test_set_sha256, true) ||
      !write_json_string_field(out, "eval_mode", meta->eval_mode, true)) {
    fclose(out);
    return false;
  }
  fprintf(out, "  \"max_new_tokens\": %zu,\n", meta->max_new_tokens);
  fprintf(out, "  \"temperature\": %.6f,\n", meta->temperature);
  if (!write_json_string_field(out, "runtime", meta->runtime, true)) {
    fclose(out);
    return false;
  }
  fprintf(out, "  \"constrained_decoding\": %s,\n", meta->constrained_decoding ? "true" : "false");
  if (!write_json_string_field(out, "grammar_sha256", grammar_sha256, true) ||
      !write_json_string_field(out, "llamacpp_base_url", meta->llamacpp_base_url, true) ||
      !write_json_string_field(out, "llamacpp_model", meta->llamacpp_model, true) ||
      !write_json_string_field(out, "llamacpp_commit", meta->llamacpp_commit, false)) {
    fclose(out);
    return false;
  }
  fputs("}\n", out);
  bool ok = !ferror(out);
  fclose(out);
  return ok;
}

// Match Python `(err or "")[:80].replace("|", "\\|")`: take up to 80 input
// characters, then escape pipes. Verifier error messages are single-line, so
// (like the Python renderer) no newline handling is applied.
static void write_error_cell(FILE *out, const char *value) {
  for (size_t i = 0; i < 80 && value[i] != '\0'; i++) {
    if (value[i] == '|') {
      fputs("\\|", out);
    } else {
      fputc(value[i], out);
    }
  }
}

static bool write_score_md(
  const char *path,
  const char *run_name,
  const char *data_path,
  const char *test_set_sha256,
  const char *grammar_sha256,
  const TfScoreMeta *meta,
  const TfTrajectorySummary *entries,
  size_t entry_count,
  size_t passed_count,
  double pass_at_1,
  double pass_at_k,
  double pass_at_maj,
  size_t k_value
) {
  (void)passed_count;
  FILE *out = fopen(path, "wb");
  if (out == 0) {
    return false;
  }
  fprintf(out, "# Eval report \xE2\x80\x94 %s\n\n", run_name);  // em dash, matches Python
  fprintf(out, "- n_examples: %zu\n", entry_count);
  fprintf(out, "- pass@1: %.3f\n", pass_at_1);
  fprintf(out, "- pass@%zu: %.3f\n", k_value, pass_at_k);
  fprintf(out, "- pass@maj: %.3f\n\n", pass_at_maj);
  fputs("## Provenance\n", out);
  fputs("- timestamp_utc: \n", out);
  fputs("- git_sha: (no git context)\n", out);
  fprintf(out, "- adapter_dir: %s\n", meta->adapter_dir);
  fprintf(out, "- base_model: %s\n", meta->base_model);
  fprintf(out, "- test_set_path: %s\n", data_path);
  fprintf(out, "- test_set_sha256: %s\n", test_set_sha256);
  fprintf(out, "- eval_mode: %s\n", meta->eval_mode);
  fprintf(out, "- max_new_tokens: %zu\n", meta->max_new_tokens);
  fprintf(out, "- temperature: %.6f\n\n", meta->temperature);
  fprintf(out, "- runtime: %s\n", meta->runtime);
  fprintf(out, "- constrained_decoding: %s\n", meta->constrained_decoding ? "true" : "false");
  fprintf(out, "- grammar_sha256: %s\n", grammar_sha256);
  fprintf(out, "- llamacpp_base_url: %s\n", meta->llamacpp_base_url);
  fprintf(out, "- llamacpp_model: %s\n", meta->llamacpp_model);
  fprintf(out, "- llamacpp_commit: %s\n\n", meta->llamacpp_commit);
  // Failure-mode counts in first-seen order, derived from each failed
  // trajectory's first_failing_subscore exactly as Python's Counter does, and
  // "\n"-joined with no trailing newline to match `"\n".join(...)`. Final-state
  // mismatch failures carry no failing subscore and are not counted (Python
  // skips them too).
  fputs("## Failure modes (greedy)\n", out);
  {
    char mode_names[6][32];
    size_t mode_counts[6];
    size_t mode_n = 0;
    for (size_t i = 0; i < entry_count; i++) {
      if (entries[i].passed || entries[i].first_failing_subscore[0] == '\0') {
        continue;
      }
      const char *mode = entries[i].first_failing_subscore;
      size_t m = 0;
      for (; m < mode_n; m++) {
        if (strcmp(mode_names[m], mode) == 0) {
          break;
        }
      }
      if (m == mode_n && mode_n < sizeof(mode_counts) / sizeof(mode_counts[0])) {
        tf_copy_cstr(mode_names[mode_n], sizeof(mode_names[mode_n]), mode);
        mode_counts[mode_n] = 0;
        mode_n++;
      }
      if (m < sizeof(mode_counts) / sizeof(mode_counts[0])) {
        mode_counts[m]++;
      }
    }
    for (size_t m = 0; m < mode_n; m++) {
      fprintf(out, m == 0 ? "- %s: %zu" : "\n- %s: %zu", mode_names[m], mode_counts[m]);
    }
  }

  // Failed-trajectory section grouped by initial_state, states sorted
  // alphabetically, matching Python eval/runner._render_failed_trajectories.
  fputs("\n\n## Failed trajectories\n", out);
  if (passed_count == entry_count) {
    fputs("None (all greedy passes).\n", out);
  } else {
    char states[TF_MAX_STATES + 1][96];
    size_t state_n = 0;
    for (size_t i = 0; i < entry_count; i++) {
      if (entries[i].passed) {
        continue;
      }
      const char *st =
        entries[i].initial_state[0] != '\0' ? entries[i].initial_state : "(unknown)";
      bool seen = false;
      for (size_t s = 0; s < state_n; s++) {
        if (strcmp(states[s], st) == 0) {
          seen = true;
          break;
        }
      }
      if (!seen && state_n < sizeof(states) / sizeof(states[0])) {
        tf_copy_cstr(states[state_n], sizeof(states[state_n]), st);
        state_n++;
      }
    }
    for (size_t a = 1; a < state_n; a++) {
      char key[96];
      tf_copy_cstr(key, sizeof(key), states[a]);
      size_t b = a;
      while (b > 0 && strcmp(states[b - 1], key) > 0) {
        tf_copy_cstr(states[b], sizeof(states[b]), states[b - 1]);
        b--;
      }
      tf_copy_cstr(states[b], sizeof(states[b]), key);
    }
    for (size_t s = 0; s < state_n; s++) {
      fprintf(out, "\n### %s\n\n", states[s]);
      fputs("| trajectory_id | first_failing_step | first_failing_subscore | error |\n", out);
      fputs("|---|---|---|---|\n", out);
      for (size_t i = 0; i < entry_count; i++) {
        const TfTrajectorySummary *entry = &entries[i];
        if (entry->passed) {
          continue;
        }
        const char *st = entry->initial_state[0] != '\0' ? entry->initial_state : "(unknown)";
        if (strcmp(st, states[s]) != 0) {
          continue;
        }
        fprintf(out, "| %s | ", entry->trajectory_id);
        if (entry->has_first_failing_step) {
          fprintf(out, "%zu", entry->first_failing_step);
        } else {
          fputs("None", out);
        }
        fputs(" | ", out);
        fputs(entry->first_failing_subscore[0] != '\0' ? entry->first_failing_subscore : "None", out);
        fputs(" | ", out);
        write_error_cell(out, entry->error_message);
        fputs(" |\n", out);
      }
    }
  }
  bool ok = !ferror(out);
  fclose(out);
  return ok;
}

static bool optional_file_sha256(const char *path, char digest[TF_SHA256_HEX_SIZE]) {
  char err[TF_MAX_ERROR] = "";
  digest[0] = '\0';
  if (tf_sha256_file_hex(path, digest, err, sizeof(err))) {
    return true;
  }
  if (strncmp(err, "failed to open ", strlen("failed to open ")) == 0) {
    digest[0] = '\0';
    return true;
  }
  return false;
}

static int run_score_data(int argc, char **argv) {
  const char *data_path = "data/test.jsonl";
  const char *schemas_dir = "schemas";
  const char *grammar_path = "schemas/jsonrpc.gbnf";
  const char *out_prefix = "c-score-data";
  const char *run_name = "c-score-data";

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-path") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (strcmp(argv[i], "--run-name") == 0 && i + 1 < argc) {
      run_name = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  char test_set_sha256[TF_SHA256_HEX_SIZE] = "";
  char hash_err[TF_MAX_ERROR] = "";
  if (!tf_sha256_file_hex(data_path, test_set_sha256, hash_err, sizeof(hash_err))) {
    fprintf(stderr, "test set hash failed: %s\n", hash_err);
    return 2;
  }
  char grammar_digest[TF_SHA256_HEX_SIZE] = "";
  if (!optional_file_sha256(grammar_path, grammar_digest)) {
    fprintf(stderr, "grammar hash failed for %s\n", grammar_path);
    return 2;
  }

  FILE *f = fopen(data_path, "rb");
  if (f == 0) {
    fprintf(stderr, "failed to open %s\n", data_path);
    return 2;
  }

  TfTrajectorySummary *entries = 0;
  size_t entry_count = 0;
  size_t entry_cap = 0;
  size_t passed_count = 0;
  TfFailureModes failure_modes;
  memset(&failure_modes, 0, sizeof(failure_modes));

  size_t line_number = 0;
  char *line = 0;
  while ((line = tf_read_line(f)) != 0) {
    line_number++;
    char *trimmed = trim_line(line);
    if (*trimmed == '\0') {
      free(line);
      continue;
    }

    char fallback_id[96];
    snprintf(fallback_id, sizeof(fallback_id), "line-%zu", line_number);
    TfTrajectorySummary summary;
    TfJsonSpan trajectory;
    if (!parse_json_line(trimmed, &trajectory)) {
      memset(&summary, 0, sizeof(summary));
      tf_copy_cstr(summary.trajectory_id, sizeof(summary.trajectory_id), fallback_id);
      summary_set_error(&summary, "parse", false, 0, "invalid JSON");
    } else {
      (void)summarize_trajectory_span(&schemas, trajectory, fallback_id, &summary);
    }
    if (summary.passed) {
      passed_count++;
    } else if (summary.first_failing_subscore[0] != '\0') {
      count_failure_mode(&failure_modes, summary.first_failing_subscore);
    }
    if (!append_trajectory_summary(&entries, &entry_count, &entry_cap, &summary)) {
      fprintf(stderr, "out of memory while scoring %s\n", data_path);
      free(line);
      fclose(f);
      free(entries);
      return 2;
    }
    free(line);
  }
  fclose(f);

  char json_path[TF_CLI_PATH_CAP];
  char md_path[TF_CLI_PATH_CAP];
  if (!make_suffixed_path(json_path, sizeof(json_path), out_prefix, ".json") ||
      !make_suffixed_path(md_path, sizeof(md_path), out_prefix, ".md")) {
    fprintf(stderr, "out-prefix is too long\n");
    free(entries);
    return 2;
  }

  const TfScoreMeta meta = {
    .adapter_dir = "(gold-data)",
    .base_model = "(gold-data)",
    .eval_mode = "teacher_forced",
    .max_new_tokens = 0,
    .temperature = 0.0,
    .runtime = "c-gold",
    .constrained_decoding = false,
    .llamacpp_base_url = "",
    .llamacpp_model = "",
    .llamacpp_commit = "",
  };
  double sd_rate = entry_count == 0 ? 0.0 : (double)passed_count / (double)entry_count;
  if (!write_score_json(
        json_path,
        run_name,
        data_path,
        test_set_sha256,
        grammar_digest,
        &meta,
        entries,
        entry_count,
        passed_count,
        sd_rate,
        sd_rate,
        sd_rate,
        1,
        &failure_modes
      ) ||
      !write_score_md(
        md_path,
        run_name,
        data_path,
        test_set_sha256,
        grammar_digest,
        &meta,
        entries,
        entry_count,
        passed_count,
        sd_rate,
        sd_rate,
        sd_rate,
        1
      )) {
    fprintf(stderr, "failed to write score report for %s\n", out_prefix);
    free(entries);
    return 2;
  }

  printf("score_data: %zu/%zu pass\n", passed_count, entry_count);
  printf("wrote %s and %s\n", json_path, md_path);
  free(entries);
  return 0;
}

static bool parse_span_double(TfJsonSpan value, double *out) {
  const char *start = tf_json_skip_ws(value.start);
  const char *end = value.end;
  while (end > start && isspace((unsigned char)end[-1])) {
    end--;
  }
  size_t len = (size_t)(end - start);
  if (len == 0 || len >= 64) {
    return false;
  }
  char tmp[64];
  memcpy(tmp, start, len);
  tmp[len] = '\0';
  char *parse_end = 0;
  double parsed = strtod(tmp, &parse_end);
  if (parse_end == tmp || *parse_end != '\0') {
    return false;
  }
  *out = parsed;
  return true;
}

static bool parse_span_size(TfJsonSpan value, size_t *out) {
  const char *start = tf_json_skip_ws(value.start);
  const char *end = value.end;
  while (end > start && isspace((unsigned char)end[-1])) {
    end--;
  }
  size_t len = (size_t)(end - start);
  if (len == 0 || len >= 32) {
    return false;
  }
  if (*start == '-' || *start == '+') {
    return false;
  }
  char tmp[32];
  memcpy(tmp, start, len);
  tmp[len] = '\0';
  char *parse_end = 0;
  unsigned long parsed = strtoul(tmp, &parse_end, 10);
  if (parse_end == tmp || *parse_end != '\0') {
    return false;
  }
  *out = (size_t)parsed;
  return true;
}

static bool json_span_is_null(TfJsonSpan value) {
  const char *start = tf_json_skip_ws(value.start);
  const char *end = value.end;
  while (end > start && isspace((unsigned char)end[-1])) {
    end--;
  }
  return (size_t)(end - start) == 4 && strncmp(start, "null", 4) == 0;
}

static bool get_score_string(TfJsonSpan object, const char *key, char *out, size_t out_cap, bool *has) {
  *has = false;
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value) || json_span_is_null(value)) {
    return true;
  }
  if (!tf_json_object_get_string(object, key, out, out_cap)) {
    return false;
  }
  *has = true;
  return true;
}

static bool get_score_double(TfJsonSpan object, const char *key, double *out, bool *has) {
  *has = false;
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value) || json_span_is_null(value)) {
    return true;
  }
  if (!parse_span_double(value, out)) {
    return false;
  }
  *has = true;
  return true;
}

static bool get_score_size(TfJsonSpan object, const char *key, size_t *out, bool *has) {
  *has = false;
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value) || json_span_is_null(value)) {
    return true;
  }
  if (!parse_span_size(value, out)) {
    return false;
  }
  *has = true;
  return true;
}

static void append_failure_item(TfScoreRun *run, const char *name, size_t count) {
  if (run->failure_count >= TF_CLI_COMPARE_MAX_FAILURES) {
    return;
  }
  tf_copy_cstr(run->failures[run->failure_count].name, sizeof(run->failures[run->failure_count].name), name);
  run->failures[run->failure_count].count = count;
  run->failure_count++;
}

static bool load_failure_modes(TfJsonSpan report, TfScoreRun *run) {
  TfJsonSpan failures;
  if (!tf_json_object_get_object(report, "failure_modes", &failures)) {
    return true;
  }

  char keys[TF_CLI_COMPARE_MAX_FAILURES][96];
  size_t key_count = 0;
  if (!tf_json_collect_object_keys(failures, keys, TF_CLI_COMPARE_MAX_FAILURES, &key_count)) {
    return false;
  }
  for (size_t i = 0; i < key_count; i++) {
    TfJsonSpan value;
    size_t count = 0;
    if (tf_json_object_get_value(failures, keys[i], &value) && parse_span_size(value, &count)) {
      append_failure_item(run, keys[i], count);
    }
  }
  return true;
}

static bool load_score_run(const char *path, TfScoreRun *run) {
  memset(run, 0, sizeof(*run));
  char err[TF_MAX_ERROR] = "";
  char *text = tf_read_file(path, err, sizeof(err));
  if (text == 0) {
    return false;
  }

  TfJsonSpan report;
  bool ok = parse_json_line(text, &report) && *tf_json_skip_ws(report.start) == '{' &&
            get_score_string(report, "run_name", run->run_name, sizeof(run->run_name), &run->has_run_name) &&
            get_score_size(report, "n_examples", &run->n_examples, &run->has_n_examples) &&
            get_score_double(report, "pass_at_1", &run->pass_at_1, &run->has_pass_at_1) &&
            get_score_double(report, "pass_at_k", &run->pass_at_k, &run->has_pass_at_k) &&
            get_score_double(report, "pass_at_maj", &run->pass_at_maj, &run->has_pass_at_maj) &&
            get_score_size(report, "k", &run->k, &run->has_k) && load_failure_modes(report, run);
  free(text);
  return ok;
}

static bool append_report_file(TfReportFile **files, size_t *count, size_t *cap, const TfReportFile *file) {
  if (*count == *cap) {
    size_t new_cap = *cap == 0 ? 16 : *cap * 2;
    if (new_cap < *cap) {
      return false;
    }
    TfReportFile *new_files = (TfReportFile *)realloc(*files, new_cap * sizeof(**files));
    if (new_files == 0) {
      return false;
    }
    *files = new_files;
    *cap = new_cap;
  }
  (*files)[*count] = *file;
  (*count)++;
  return true;
}

static bool append_score_run(TfScoreRun **runs, size_t *count, size_t *cap, const TfScoreRun *run) {
  if (*count == *cap) {
    size_t new_cap = *cap == 0 ? 16 : *cap * 2;
    if (new_cap < *cap) {
      return false;
    }
    TfScoreRun *new_runs = (TfScoreRun *)realloc(*runs, new_cap * sizeof(**runs));
    if (new_runs == 0) {
      return false;
    }
    *runs = new_runs;
    *cap = new_cap;
  }
  (*runs)[*count] = *run;
  (*count)++;
  return true;
}

static int compare_report_mtime_desc(const void *a, const void *b) {
  const TfReportFile *left = (const TfReportFile *)a;
  const TfReportFile *right = (const TfReportFile *)b;
  if (left->mtime > right->mtime) {
    return -1;
  }
  if (left->mtime < right->mtime) {
    return 1;
  }
  return strcmp(left->name, right->name);
}

static int compare_report_name_asc(const void *a, const void *b) {
  const TfReportFile *left = (const TfReportFile *)a;
  const TfReportFile *right = (const TfReportFile *)b;
  return strcmp(left->name, right->name);
}

static bool collect_report_files(
  const char *reports_dir,
  const char *pattern,
  size_t last,
  TfReportFile **out_files,
  size_t *out_count
) {
  DIR *dir = opendir(reports_dir);
  if (dir == 0) {
    return false;
  }
  TfReportFile *files = 0;
  size_t count = 0;
  size_t cap = 0;
  struct dirent *entry = 0;
  while ((entry = readdir(dir)) != 0) {
    if (entry->d_name[0] == '.') {
      continue;
    }
    if (fnmatch(pattern, entry->d_name, 0) != 0) {
      continue;
    }
    TfReportFile file;
    memset(&file, 0, sizeof(file));
    if (!tf_join_path(file.path, sizeof(file.path), reports_dir, entry->d_name)) {
      closedir(dir);
      free(files);
      return false;
    }
    tf_copy_cstr(file.name, sizeof(file.name), entry->d_name);
    struct stat st;
    if (stat(file.path, &st) != 0) {
      continue;
    }
    file.mtime = (long)st.st_mtime;
    if (!append_report_file(&files, &count, &cap, &file)) {
      closedir(dir);
      free(files);
      return false;
    }
  }
  closedir(dir);

  if (last > 0 && count > last) {
    qsort(files, count, sizeof(*files), compare_report_mtime_desc);
    count = last;
  }
  // Guard: with no matching report files, `files` is NULL and `count` 0;
  // qsort(NULL, 0, ...) passes NULL to a nonnull parameter (technically UB).
  if (count > 0) {
    qsort(files, count, sizeof(*files), compare_report_name_asc);
  }
  *out_files = files;
  *out_count = count;
  return true;
}

static const char *score_run_name(const TfScoreRun *run) {
  return run->has_run_name ? run->run_name : "?";
}

static void write_optional_size_cell(FILE *out, bool has, size_t value) {
  if (has) {
    fprintf(out, "%zu", value);
  } else {
    fputc('?', out);
  }
}

static void write_optional_double_cell(FILE *out, bool has, double value, bool bold) {
  if (!has) {
    fputc('?', out);
    return;
  }
  if (bold) {
    fprintf(out, "**%.3f**", value);
  } else {
    fprintf(out, "%.3f", value);
  }
}

static void write_delta_cell(FILE *out, const TfScoreRun *run, const char *baseline, bool baseline_found, double baseline_pass1) {
  if (baseline == 0) {
    return;
  }
  if (strcmp(score_run_name(run), baseline) == 0) {
    fputs("(base)", out);
  } else if (baseline_found && run->has_pass_at_1) {
    fprintf(out, "%+.3f", run->pass_at_1 - baseline_pass1);
  } else {
    fputc('?', out);
  }
}

static void write_top_failures(FILE *out, const TfScoreRun *run) {
  bool used[TF_CLI_COMPARE_MAX_FAILURES];
  memset(used, 0, sizeof(used));
  size_t written = 0;
  for (size_t rank = 0; rank < 3; rank++) {
    size_t best = TF_CLI_COMPARE_MAX_FAILURES;
    for (size_t i = 0; i < run->failure_count; i++) {
      if (used[i] || run->failures[i].count == 0) {
        continue;
      }
      if (best == TF_CLI_COMPARE_MAX_FAILURES ||
          run->failures[i].count > run->failures[best].count) {
        best = i;
      }
    }
    if (best == TF_CLI_COMPARE_MAX_FAILURES) {
      break;
    }
    if (written > 0) {
      fputs(", ", out);
    }
    fprintf(out, "%s(%zu)", run->failures[best].name, run->failures[best].count);
    used[best] = true;
    written++;
  }
  if (written == 0) {
    fputc('-', out);
  }
}

static size_t run_failure_count(const TfScoreRun *run, const char *name) {
  for (size_t i = 0; i < run->failure_count; i++) {
    if (strcmp(run->failures[i].name, name) == 0) {
      return run->failures[i].count;
    }
  }
  return 0;
}

static void write_failure_chart(FILE *out, const TfScoreRun *runs, size_t run_count) {
  const char *names[] = {
    "parse",
    "schema",
    "method_known",
    "precondition_met",
    "transition_valid",
    "sequence_optimal",
  };
  size_t counts[sizeof(names) / sizeof(names[0])];
  memset(counts, 0, sizeof(counts));
  size_t max_count = 0;
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    for (size_t r = 0; r < run_count; r++) {
      counts[i] += run_failure_count(&runs[r], names[i]);
    }
    if (counts[i] > max_count) {
      max_count = counts[i];
    }
  }

  fputs("\n\n## Failure mode chart (aggregate across runs)\n\n", out);
  if (max_count == 0) {
    fputs("_No greedy failures across runs._\n", out);
    return;
  }
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    fprintf(out, "    %-17s ", names[i]);
    if (counts[i] == 0) {
      fputs("\xE2\x96\x8F (0)\n", out);  // U+258F LEFT ONE-EIGHTH BLOCK, matches Python
      continue;
    }
    // Match Python's `max(1, round(20 * count / max_count))`. Python's round()
    // is round-half-to-even; replicate it in exact integer arithmetic so there
    // is no floating-point rounding-mode dependence.
    size_t num = 20 * counts[i];
    size_t bar_count = num / max_count;
    size_t two_rem = 2 * (num % max_count);
    if (two_rem > max_count || (two_rem == max_count && (bar_count % 2) != 0)) {
      bar_count++;
    }
    if (bar_count == 0) {
      bar_count = 1;
    }
    for (size_t b = 0; b < bar_count; b++) {
      fputs("\xE2\x96\x88", out);  // U+2588 FULL BLOCK, matches Python
    }
    fprintf(out, " %zu\n", counts[i]);
  }
}

static bool split_seed_family(const char *run_name, char *family, size_t family_cap) {
  const char *seed = strstr(run_name, "-seed");
  if (seed == 0 || seed == run_name) {
    tf_copy_cstr(family, family_cap, run_name);
    return false;
  }
  const char *digits = seed + 5;
  if (*digits == '\0') {
    tf_copy_cstr(family, family_cap, run_name);
    return false;
  }
  for (const char *p = digits; *p != '\0'; p++) {
    if (!isdigit((unsigned char)*p)) {
      tf_copy_cstr(family, family_cap, run_name);
      return false;
    }
  }
  tf_copy_span(family, family_cap, run_name, seed);
  return true;
}

static int compare_strings_for_qsort(const void *a, const void *b) {
  const char *const *left = (const char *const *)a;
  const char *const *right = (const char *const *)b;
  return strcmp(*left, *right);
}

static int compare_doubles_for_qsort(const void *a, const void *b) {
  double left = *(const double *)a;
  double right = *(const double *)b;
  if (left < right) {
    return -1;
  }
  if (left > right) {
    return 1;
  }
  return 0;
}

static double median_sorted(const double *values, size_t count) {
  if (count == 0) {
    return 0.0;
  }
  if (count % 2 == 1) {
    return values[count / 2];
  }
  return (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

static double exclusive_quartile_sorted(const double *values, size_t count, size_t quartile) {
  double pos = (double)quartile * (double)(count + 1) / 4.0;
  size_t idx = (size_t)pos;
  double frac = pos - (double)idx;
  if (idx == 0) {
    return values[0];
  }
  if (idx >= count) {
    return values[count - 1];
  }
  return values[idx - 1] + frac * (values[idx] - values[idx - 1]);
}

static void write_med_iqr_cell(FILE *out, double *values, size_t count) {
  if (count == 0) {
    fputc('?', out);
    return;
  }
  qsort(values, count, sizeof(*values), compare_doubles_for_qsort);
  double med = median_sorted(values, count);
  if (count < 4) {
    fprintf(out, "%.3f", med);
    return;
  }
  double q1 = exclusive_quartile_sorted(values, count, 1);
  double q3 = exclusive_quartile_sorted(values, count, 3);
  fprintf(out, "%.3f \xC2\xB1 %.3f", med, (q3 - q1) / 2.0);
}

static bool write_arm_summary(FILE *out, const TfScoreRun *runs, size_t run_count) {
  char **families = (char **)calloc(run_count == 0 ? 1 : run_count, sizeof(*families));
  if (families == 0) {
    return false;
  }
  size_t family_count = 0;
  bool has_multi_seed = false;
  for (size_t i = 0; i < run_count; i++) {
    char family[TF_CLI_NAME_CAP];
    (void)split_seed_family(score_run_name(&runs[i]), family, sizeof(family));
    bool exists = false;
    for (size_t f = 0; f < family_count; f++) {
      if (strcmp(families[f], family) == 0) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      families[family_count] = (char *)malloc(strlen(family) + 1);
      if (families[family_count] == 0) {
        for (size_t f = 0; f < family_count; f++) {
          free(families[f]);
        }
        free(families);
        return false;
      }
      strcpy(families[family_count], family);
      family_count++;
    }
  }
  for (size_t f = 0; f < family_count; f++) {
    size_t seeds = 0;
    for (size_t i = 0; i < run_count; i++) {
      char family[TF_CLI_NAME_CAP];
      (void)split_seed_family(score_run_name(&runs[i]), family, sizeof(family));
      if (strcmp(families[f], family) == 0) {
        seeds++;
      }
    }
    if (seeds > 1) {
      has_multi_seed = true;
      break;
    }
  }
  if (!has_multi_seed) {
    for (size_t f = 0; f < family_count; f++) {
      free(families[f]);
    }
    free(families);
    return true;
  }
  qsort(families, family_count, sizeof(*families), compare_strings_for_qsort);
  fputs(
    "\n## Arm summary (multi-seed aggregation)\n"
    "| arm | n_seeds | pass@1 (median \xC2\xB1 half_IQR) | pass@k | pass@maj |\n"
    "| --- | --- | --- | --- | --- |\n",
    out
  );
  for (size_t f = 0; f < family_count; f++) {
    double p1[128];
    double pk[128];
    double pmaj[128];
    size_t p1_count = 0;
    size_t pk_count = 0;
    size_t pmaj_count = 0;
    size_t seed_count = 0;
    for (size_t i = 0; i < run_count; i++) {
      char family[TF_CLI_NAME_CAP];
      (void)split_seed_family(score_run_name(&runs[i]), family, sizeof(family));
      if (strcmp(families[f], family) != 0) {
        continue;
      }
      seed_count++;
      if (runs[i].has_pass_at_1 && p1_count < sizeof(p1) / sizeof(p1[0])) {
        p1[p1_count++] = runs[i].pass_at_1;
      }
      if (runs[i].has_pass_at_k && pk_count < sizeof(pk) / sizeof(pk[0])) {
        pk[pk_count++] = runs[i].pass_at_k;
      }
      if (runs[i].has_pass_at_maj && pmaj_count < sizeof(pmaj) / sizeof(pmaj[0])) {
        pmaj[pmaj_count++] = runs[i].pass_at_maj;
      }
    }
    fprintf(out, "| %s | %zu | ", families[f], seed_count);
    write_med_iqr_cell(out, p1, p1_count);
    fputs(" | ", out);
    write_med_iqr_cell(out, pk, pk_count);
    fputs(" | ", out);
    write_med_iqr_cell(out, pmaj, pmaj_count);
    fputs(" |\n", out);
  }
  for (size_t f = 0; f < family_count; f++) {
    free(families[f]);
  }
  free(families);
  return true;
}

static bool write_compare_markdown(
  FILE *out,
  const TfScoreRun *runs,
  size_t run_count,
  const char *baseline
) {
  bool baseline_found = false;
  double baseline_pass1 = 0.0;
  if (baseline != 0) {
    for (size_t i = 0; i < run_count; i++) {
      if (strcmp(score_run_name(&runs[i]), baseline) == 0 && runs[i].has_pass_at_1) {
        baseline_pass1 = runs[i].pass_at_1;
        baseline_found = true;
        break;
      }
    }
  }

  fputs(
    "# toyForge run comparison\n\n"
    "| run | n | pass@1 | \xCE\x94pass@1 | pass@k | pass@maj | k | top failures |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n",
    out
  );
  for (size_t i = 0; i < run_count; i++) {
    const TfScoreRun *run = &runs[i];
    bool is_regression =
      baseline != 0 && baseline_found && run->has_pass_at_1 && run->pass_at_1 < baseline_pass1 &&
      strcmp(score_run_name(run), baseline) != 0;
    fprintf(out, "| %s | ", score_run_name(run));
    write_optional_size_cell(out, run->has_n_examples, run->n_examples);
    fputs(" | ", out);
    write_optional_double_cell(out, run->has_pass_at_1, run->pass_at_1, is_regression);
    fputs(" | ", out);
    write_delta_cell(out, run, baseline, baseline_found, baseline_pass1);
    fputs(" | ", out);
    write_optional_double_cell(out, run->has_pass_at_k, run->pass_at_k, false);
    fputs(" | ", out);
    write_optional_double_cell(out, run->has_pass_at_maj, run->pass_at_maj, false);
    fputs(" | ", out);
    write_optional_size_cell(out, run->has_k, run->k);
    fputs(" | ", out);
    write_top_failures(out, run);
    fputs(" |\n", out);
  }
  if (!write_arm_summary(out, runs, run_count)) {
    return false;
  }
  write_failure_chart(out, runs, run_count);
  return !ferror(out);
}

static int run_compare(int argc, char **argv) {
  const char *reports_dir = "reports";
  const char *out_path = "reports/compare.md";
  const char *baseline = 0;
  const char *pattern = "*.json";
  size_t last = 0;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--reports-dir") == 0 && i + 1 < argc) {
      reports_dir = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) {
      baseline = argv[++i];
    } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
      pattern = argv[++i];
    } else if (strcmp(argv[i], "--last") == 0 && i + 1 < argc) {
      char *end = 0;
      unsigned long parsed = strtoul(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0') {
        usage(stderr);
        return 2;
      }
      last = (size_t)parsed;
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfReportFile *files = 0;
  size_t file_count = 0;
  if (!collect_report_files(reports_dir, pattern, last, &files, &file_count)) {
    fprintf(stderr, "failed to read reports directory: %s\n", reports_dir);
    return 2;
  }

  TfScoreRun *runs = 0;
  size_t run_count = 0;
  size_t run_cap = 0;
  for (size_t i = 0; i < file_count; i++) {
    TfScoreRun run;
    if (!load_score_run(files[i].path, &run)) {
      printf("skip unparseable %s\n", files[i].name);
      continue;
    }
    if (!append_score_run(&runs, &run_count, &run_cap, &run)) {
      fprintf(stderr, "out of memory while comparing reports\n");
      free(files);
      free(runs);
      return 2;
    }
  }
  free(files);

  if (run_count == 0) {
    FILE *out = fopen(out_path, "wb");
    if (out != 0) {
      // No trailing newline — byte-match Python compare_runs's returned string
      // (its non-empty output has none either; diff_c_compare enforces this).
      fputs("no reports found", out);
      fclose(out);
    }
    puts("no reports found");
    free(runs);
    return 0;
  }

  FILE *out = fopen(out_path, "wb");
  if (out == 0) {
    fprintf(stderr, "failed to write %s\n", out_path);
    free(runs);
    return 2;
  }
  bool ok = write_compare_markdown(out, runs, run_count, baseline);
  fclose(out);
  if (!ok) {
    fprintf(stderr, "failed to render compare report\n");
    free(runs);
    return 2;
  }
  if (!write_compare_markdown(stdout, runs, run_count, baseline)) {
    free(runs);
    return 2;
  }
  printf("\ncompare: %zu reports -> %s\n", run_count, out_path);
  free(runs);
  return 0;
}

static int compare_cstr_ptrs(const void *a, const void *b) {
  const char *const *left = (const char *const *)a;
  const char *const *right = (const char *const *)b;
  return strcmp(*left, *right);
}

static bool write_jsonrpc_gbnf(FILE *out, const TfSchemas *schemas) {
  const char *methods[TF_MAX_METHODS];
  if (schemas->method_count > TF_MAX_METHODS) {
    return false;
  }
  for (size_t i = 0; i < schemas->method_count; i++) {
    methods[i] = schemas->methods[i].name;
  }
  qsort(methods, schemas->method_count, sizeof(methods[0]), compare_cstr_ptrs);

  fputs("root ::= \"{\" ws jsonrpc \",\" ws method \",\" ws params \",\" ws id ws \"}\"\n", out);
  fputs("jsonrpc ::= \"\\\"jsonrpc\\\"\" ws \":\" ws \"\\\"2.0\\\"\"\n", out);
  fputs("method ::= \"\\\"method\\\"\" ws \":\" ws (", out);
  for (size_t i = 0; i < schemas->method_count; i++) {
    /* The method VALUE is a JSON string in the call, so the GBNF literal must match the
     * quoted text "<name>". One json-encode yields "<name>" (a GBNF literal matching the
     * BARE token); a second encode escapes it into the GBNF literal "\"<name>\"" that
     * matches the JSON-valid quoted value. Mirrors json.dumps(json.dumps(m)) in Python. */
    char json_val[192];
    char gbnf_lit[256];
    if (!tf_json_write_string(methods[i], json_val, sizeof(json_val)) ||
        !tf_json_write_string(json_val, gbnf_lit, sizeof(gbnf_lit))) {
      return false;
    }
    if (i > 0) {
      fputs(" | ", out);
    }
    fputs(gbnf_lit, out);
  }
  fputs(")\n", out);
  fputs("params ::= \"\\\"params\\\"\" ws \":\" ws object\n", out);
  fputs("id ::= \"\\\"id\\\"\" ws \":\" ws (number | string)\n", out);
  fputs("object ::= \"{\" ws (member (\",\" ws member)*)? ws \"}\"\n", out);
  fputs("member ::= string ws \":\" ws value\n", out);
  fputs("array ::= \"[\" ws (value (\",\" ws value)*)? ws \"]\"\n", out);
  fputs("value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n", out);
  fputs("string ::= \"\\\"\" char* \"\\\"\"\n", out);
  fputs(
    "char ::= [^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F] [0-9a-fA-F] "
    "[0-9a-fA-F] [0-9a-fA-F])\n",
    out
  );
  fputs("number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?\n", out);
  fputs("ws ::= [ \\t\\n\\r]*\n", out);
  return !ferror(out);
}

static int run_build_jsonrpc_gbnf(int argc, char **argv) {
  const char *schemas_dir = "schemas";
  const char *out_path = 0;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  if (out_path == 0) {
    if (!write_jsonrpc_gbnf(stdout, &schemas)) {
      fprintf(stderr, "failed to write JSON-RPC GBNF\n");
      return 2;
    }
    return 0;
  }

  FILE *out = fopen(out_path, "wb");
  if (out == 0) {
    fprintf(stderr, "failed to write %s\n", out_path);
    return 2;
  }
  bool ok = write_jsonrpc_gbnf(out, &schemas);
  fclose(out);
  if (!ok) {
    fprintf(stderr, "failed to write JSON-RPC GBNF\n");
    return 2;
  }

  char digest[TF_SHA256_HEX_SIZE] = "";
  char err[TF_MAX_ERROR] = "";
  if (!tf_sha256_file_hex(out_path, digest, err, sizeof(err))) {
    fprintf(stderr, "grammar hash failed: %s\n", err);
    return 2;
  }
  printf("build_jsonrpc_gbnf: wrote %s\n", out_path);
  puts(digest);
  return 0;
}

static bool append_text(char *dst, size_t dst_cap, const char *src) {
  size_t used = strlen(dst);
  size_t src_len = strlen(src);
  if (used > dst_cap || src_len >= dst_cap - used) {
    return false;
  }
  memcpy(dst + used, src, src_len + 1);
  return true;
}

static bool append_span_text(char *dst, size_t dst_cap, const char *start, const char *end) {
  size_t used = strlen(dst);
  size_t span_len = (size_t)(end - start);
  if (used > dst_cap || span_len >= dst_cap - used) {
    return false;
  }
  memcpy(dst + used, start, span_len);
  dst[used + span_len] = '\0';
  return true;
}

static bool span_is_empty_array(TfJsonSpan value) {
  const char *p = tf_json_skip_ws(value.start);
  if (p >= value.end || *p != '[') {
    return false;
  }
  p = tf_json_skip_ws(p + 1);
  return p < value.end && *p == ']' && tf_json_skip_ws(p + 1) == value.end;
}

static bool build_user_content(TfJsonSpan step, char *out, size_t out_cap, char *err, size_t err_cap) {
  TfJsonSpan context;
  if (!tf_json_object_get_object(step, "prompt_context", &context)) {
    tf_copy_cstr(err, err_cap, "prompt_context must be an object");
    return false;
  }
  char prior_state[96];
  if (!tf_json_object_get_string(context, "prior_state", prior_state, sizeof(prior_state))) {
    tf_copy_cstr(err, err_cap, "prompt_context.prior_state is missing or too large");
    return false;
  }

  out[0] = '\0';
  if (!append_text(out, out_cap, "Prior state: ") || !append_text(out, out_cap, prior_state)) {
    tf_copy_cstr(err, err_cap, "user prompt is too large");
    return false;
  }

  char situation[TF_CLI_PAYLOAD_TEXT_CAP];
  if (tf_json_object_get_string(context, "situation", situation, sizeof(situation)) &&
      situation[0] != '\0') {
    if (!append_text(out, out_cap, "\nSituation: ") || !append_text(out, out_cap, situation)) {
      tf_copy_cstr(err, err_cap, "user prompt is too large");
      return false;
    }
  }

  TfJsonSpan prior_calls;
  if (tf_json_object_get_value(context, "prior_calls", &prior_calls) &&
      !span_is_empty_array(prior_calls)) {
    if (!append_text(out, out_cap, "\nPrior calls: ") ||
        !append_span_text(out, out_cap, prior_calls.start, prior_calls.end)) {
      tf_copy_cstr(err, err_cap, "user prompt is too large");
      return false;
    }
  }
  return true;
}

static bool write_message(FILE *out, const char *role, const char *content, bool comma) {
  fputs("    {\"role\": ", out);
  if (!write_json_string_value(out, role)) {
    return false;
  }
  fputs(", \"content\": ", out);
  if (!write_json_string_value(out, content)) {
    return false;
  }
  fprintf(out, "}%s\n", comma ? "," : "");
  return !ferror(out);
}

static bool write_llamacpp_payload(
  const char *path,
  const char *model,
  const char *system_content,
  const char *user_content,
  const char *assistant_content,
  const char *final_user_content,
  size_t max_tokens,
  double temperature,
  const char *grammar
) {
  FILE *out = fopen(path, "wb");
  if (out == 0) {
    return false;
  }

  fputs("{\n  \"model\": ", out);
  if (!write_json_string_value(out, model)) {
    fclose(out);
    return false;
  }
  fputs(",\n  \"messages\": [\n", out);
  if (!write_message(out, "system", system_content, true) ||
      !write_message(out, "user", user_content, true)) {
    fclose(out);
    return false;
  }
  if (assistant_content != 0) {
    if (!write_message(out, "assistant", assistant_content, true)) {
      fclose(out);
      return false;
    }
  }
  if (!write_message(out, "user", final_user_content, false)) {
    fclose(out);
    return false;
  }
  fprintf(out, "  ],\n  \"max_tokens\": %zu,\n  \"temperature\": %.6f", max_tokens, temperature);
  if (grammar != 0) {
    fputs(",\n  \"grammar\": ", out);
    if (!write_json_string_value(out, grammar)) {
      fclose(out);
      return false;
    }
  }
  fputs("\n}\n", out);
  bool ok = !ferror(out);
  fclose(out);
  return ok;
}

static bool load_first_trajectory_step(
  const char *data_path,
  size_t step_index,
  char **line_out,
  TfJsonSpan *step_out,
  char *err,
  size_t err_cap
) {
  *line_out = 0;
  FILE *f = fopen(data_path, "rb");
  if (f == 0) {
    tf_copy_cstr(err, err_cap, "failed to open data path");
    return false;
  }
  char *line = 0;
  while ((line = tf_read_line(f)) != 0) {
    char *trimmed = trim_line(line);
    if (*trimmed == '\0') {
      free(line);
      continue;
    }
    TfJsonSpan trajectory;
    if (!parse_json_line(trimmed, &trajectory)) {
      tf_copy_cstr(err, err_cap, "first non-empty data line is invalid JSON");
      free(line);
      fclose(f);
      return false;
    }
    TfJsonSpan steps;
    if (!tf_json_object_get_value(trajectory, "steps", &steps)) {
      tf_copy_cstr(err, err_cap, "trajectory missing steps");
      free(line);
      fclose(f);
      return false;
    }
    if (!tf_json_array_get(steps, step_index, step_out)) {
      tf_copy_cstr(err, err_cap, "step index is out of range or malformed");
      free(line);
      fclose(f);
      return false;
    }
    *line_out = line;
    fclose(f);
    return true;
  }
  fclose(f);
  tf_copy_cstr(err, err_cap, "data file has no trajectories");
  return false;
}

// Extract choices[0].message.content (or choices[0].text) from an
// OpenAI-compatible chat-completion response string.
static bool extract_llamacpp_content(
  const char *text,
  char *out,
  size_t out_cap,
  char *err,
  size_t err_cap
) {
  TfJsonSpan root;
  if (!parse_json_line(text, &root)) {
    tf_copy_cstr(err, err_cap, "response is not valid JSON");
    return false;
  }

  TfJsonSpan choices;
  if (!tf_json_object_get_value(root, "choices", &choices)) {
    tf_copy_cstr(err, err_cap, "response missing choices");
    return false;
  }

  TfJsonSpan choice;
  if (!tf_json_array_get(choices, 0, &choice)) {
    tf_copy_cstr(err, err_cap, "response choices[0] is missing or malformed");
    return false;
  }

  TfJsonSpan message;
  if (tf_json_object_get_object(choice, "message", &message)) {
    if (!tf_json_object_get_string(message, "content", out, out_cap)) {
      tf_copy_cstr(err, err_cap, "response message.content is missing, invalid, or too large");
      return false;
    }
  } else if (!tf_json_object_get_string(choice, "text", out, out_cap)) {
    tf_copy_cstr(err, err_cap, "response choice has no message.content or text");
    return false;
  }
  return true;
}

#ifdef TF_HAVE_CURL  // anthropic response parsing is only used by the live HTTP path
// Extract and concatenate the text of every {"type":"text"} block from an
// Anthropic Messages API response ({"content":[{"type":"text","text":...},...]}).
static bool extract_anthropic_content(
  const char *text,
  char *out,
  size_t out_cap,
  char *err,
  size_t err_cap
) {
  TfJsonSpan root;
  if (!parse_json_line(text, &root)) {
    tf_copy_cstr(err, err_cap, "response is not valid JSON");
    return false;
  }
  TfJsonSpan content;
  if (!tf_json_object_get_value(root, "content", &content) ||
      tf_json_type(content) != TF_JSON_ARRAY) {
    tf_copy_cstr(err, err_cap, "anthropic response missing content array");
    return false;
  }
  size_t n = 0;
  tf_json_array_count(content, &n);
  size_t pos = 0;
  if (out_cap > 0) {
    out[0] = '\0';
  }
  for (size_t i = 0; i < n; i++) {
    TfJsonSpan block;
    if (!tf_json_array_get(content, i, &block) || tf_json_type(block) != TF_JSON_OBJECT) {
      continue;
    }
    char type[32];
    if (!tf_json_object_get_string(block, "type", type, sizeof(type)) ||
        strcmp(type, "text") != 0) {
      continue;
    }
    if (pos >= out_cap) {
      tf_copy_cstr(err, err_cap, "anthropic content exceeds buffer");
      return false;
    }
    if (!tf_json_object_get_string(block, "text", out + pos, out_cap - pos)) {
      tf_copy_cstr(err, err_cap, "anthropic content text is invalid or too large");
      return false;
    }
    pos += strlen(out + pos);
  }
  return true;
}
#endif  // TF_HAVE_CURL

static bool read_llamacpp_response_content(
  const char *path,
  char *out,
  size_t out_cap,
  char *err,
  size_t err_cap
) {
  char *text = tf_read_file(path, err, err_cap);
  if (text == 0) {
    return false;
  }
  bool ok = extract_llamacpp_content(text, out, out_cap, err, err_cap);
  free(text);
  return ok;
}

static char *build_llamacpp_model_output(
  char *thinking_content,
  char *call_content,
  char *err,
  size_t err_cap
) {
  const char *thinking = trim_line(thinking_content);
  const char *call = trim_line(call_content);
  const char *think_open = "<think>";
  const char *think_close = "</think>";
  size_t open_len = strlen(think_open);
  size_t thinking_len = strlen(thinking);
  size_t close_len = strlen(think_close);
  size_t call_len = strlen(call);
  if (thinking_len > (size_t)-1 - open_len - close_len - call_len - 1) {
    tf_copy_cstr(err, err_cap, "combined llama.cpp output is too large");
    return 0;
  }
  size_t total = open_len + thinking_len + close_len + call_len;
  char *out = (char *)malloc(total + 1);
  if (out == 0) {
    tf_copy_cstr(err, err_cap, "out of memory building llama.cpp output");
    return 0;
  }
  char *p = out;
  memcpy(p, think_open, open_len);
  p += open_len;
  memcpy(p, thinking, thinking_len);
  p += thinking_len;
  memcpy(p, think_close, close_len);
  p += close_len;
  memcpy(p, call, call_len);
  p += call_len;
  *p = '\0';
  return out;
}

static bool write_text_file(const char *path, const char *text) {
  FILE *out = fopen(path, "wb");
  if (out == 0) {
    return false;
  }
  bool ok = fputs(text, out) != EOF && fputc('\n', out) != EOF;
  fclose(out);
  return ok;
}

static int run_llamacpp_verify_response(int argc, char **argv) {
  const char *data_path = 0;
  const char *schemas_dir = "schemas";
  const char *think_response_path = 0;
  const char *call_response_path = 0;
  const char *out_path = 0;
  size_t step_index = 0;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-path") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--think-response") == 0 && i + 1 < argc) {
      think_response_path = argv[++i];
    } else if (strcmp(argv[i], "--call-response") == 0 && i + 1 < argc) {
      call_response_path = argv[++i];
    } else if (strcmp(argv[i], "--step-index") == 0 && i + 1 < argc) {
      if (!parse_cli_size(argv[++i], &step_index)) {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (data_path == 0 || think_response_path == 0 || call_response_path == 0) {
    usage(stderr);
    return 2;
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *line = 0;
  TfJsonSpan step;
  if (!load_first_trajectory_step(data_path, step_index, &line, &step, err, sizeof(err))) {
    fprintf(stderr, "llama.cpp response verification failed: %s\n", err);
    return 2;
  }

  TfJsonSpan context;
  char prior_state[96];
  if (!tf_json_object_get_object(step, "prompt_context", &context) ||
      !tf_json_object_get_string(context, "prior_state", prior_state, sizeof(prior_state))) {
    fprintf(stderr, "llama.cpp response verification failed: prompt prior_state is missing\n");
    free(line);
    return 2;
  }

  char expected_trigger[96] = "";
  char expected_state_after[96] = "";
  bool has_expected_trigger = false;
  bool has_expected_state_after = false;
  if (!get_optional_string(
        step,
        "expected_trigger",
        expected_trigger,
        sizeof(expected_trigger),
        &has_expected_trigger
      ) ||
      !get_optional_string(
        step,
        "expected_state_after",
        expected_state_after,
        sizeof(expected_state_after),
        &has_expected_state_after
      )) {
    fprintf(stderr, "llama.cpp response verification failed: expected state fields are invalid\n");
    free(line);
    return 2;
  }

  char thinking_content[TF_CLI_THINKING_CAP];
  char call_content[TF_CLI_PAYLOAD_TEXT_CAP];
  if (!read_llamacpp_response_content(
        think_response_path,
        thinking_content,
        sizeof(thinking_content),
        err,
        sizeof(err)
      ) ||
      !read_llamacpp_response_content(
        call_response_path,
        call_content,
        sizeof(call_content),
        err,
        sizeof(err)
      )) {
    fprintf(stderr, "llama.cpp response verification failed: %s\n", err);
    free(line);
    return 2;
  }

  char *model_output =
    build_llamacpp_model_output(thinking_content, call_content, err, sizeof(err));
  if (model_output == 0) {
    fprintf(stderr, "llama.cpp response verification failed: %s\n", err);
    free(line);
    return 2;
  }

  if (out_path != 0 && !write_text_file(out_path, model_output)) {
    fprintf(stderr, "llama.cpp response verification failed: could not write %s\n", out_path);
    free(model_output);
    free(line);
    return 2;
  }

  TfVerifyContext verify_ctx = {
    .prior_state = prior_state,
    .expected_trigger = has_expected_trigger ? expected_trigger : 0,
    .expected_state_after = has_expected_state_after ? expected_state_after : 0,
    .infer_trigger = false,
  };
  TfStepResult result = tf_verify_step(&verify_ctx, model_output, &schemas);
  printf("llamacpp_verify_response: %s + %s\n", think_response_path, call_response_path);
  if (out_path != 0) {
    printf("model_output=%s\n", out_path);
  }
  print_step_result(&result);
  free(model_output);
  free(line);
  return result.passed ? 0 : 1;
}

static bool safe_path_component(const char *value) {
  if (value == 0 || value[0] == '\0' || strcmp(value, ".") == 0 || strcmp(value, "..") == 0) {
    return false;
  }
  for (const char *p = value; *p != '\0'; p++) {
    unsigned char c = (unsigned char)*p;
    if (!(isalnum(c) || c == '_' || c == '-' || c == '.')) {
      return false;
    }
  }
  return true;
}

static bool make_step_response_path(
  char *out,
  size_t out_cap,
  const char *responses_dir,
  const char *trajectory_id,
  size_t step_index,
  const char *kind
) {
  size_t dir_len = strlen(responses_dir);
  const char *sep = dir_len > 0 && responses_dir[dir_len - 1] == '/' ? "" : "/";
  int wrote = snprintf(
    out,
    out_cap,
    "%s%s%s/step-%zu.%s.response.json",
    responses_dir,
    sep,
    trajectory_id,
    step_index,
    kind
  );
  return wrote >= 0 && (size_t)wrote < out_cap;
}

static bool summarize_trajectory_llamacpp_responses(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  const char *fallback_id,
  const char *responses_dir,
  TfTrajectorySummary *summary
) {
  memset(summary, 0, sizeof(*summary));
  tf_copy_cstr(summary->trajectory_id, sizeof(summary->trajectory_id), fallback_id);
  (void)tf_json_object_get_string(
    trajectory,
    "trajectory_id",
    summary->trajectory_id,
    sizeof(summary->trajectory_id)
  );

  char err[TF_MAX_ERROR] = "";
  if (!tf_jsonschema_validate_trajectory(schemas, trajectory, err, sizeof(err))) {
    summary_set_error(summary, "schema", false, 0, err);
    return false;
  }
  if (!safe_path_component(summary->trajectory_id)) {
    summary_set_error(summary, "schema", false, 0, "trajectory_id is not safe for response paths");
    return false;
  }

  char current_state[96];
  if (!tf_json_object_get_string(trajectory, "initial_state", current_state, sizeof(current_state))) {
    summary_set_error(
      summary,
      "schema",
      false,
      0,
      "initial_state is required and must fit in the C state buffer"
    );
    return false;
  }
  tf_copy_cstr(summary->initial_state, sizeof(summary->initial_state), current_state);

  TfJsonSpan steps;
  if (!tf_json_object_get_value(trajectory, "steps", &steps)) {
    summary_set_error(summary, "schema", false, 0, "steps is required");
    return false;
  }
  size_t step_count = 0;
  if (!tf_json_array_count(steps, &step_count)) {
    summary_set_error(summary, "schema", false, 0, "steps must be an array");
    return false;
  }

  for (size_t i = 0; i < step_count; i++) {
    TfJsonSpan step;
    if (!tf_json_array_get(steps, i, &step)) {
      snprintf(err, sizeof(err), "steps[%zu] is malformed", i);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }

    char expected_trigger[96] = "";
    char expected_state_after[96] = "";
    bool has_expected_trigger = false;
    bool has_expected_state_after = false;
    if (!get_optional_string(
          step,
          "expected_trigger",
          expected_trigger,
          sizeof(expected_trigger),
          &has_expected_trigger
        ) ||
        !get_optional_string(
          step,
          "expected_state_after",
          expected_state_after,
          sizeof(expected_state_after),
          &has_expected_state_after
        )) {
      snprintf(err, sizeof(err), "steps[%zu] has an invalid expected state field", i);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }

    char think_path[TF_CLI_PATH_CAP];
    char call_path[TF_CLI_PATH_CAP];
    if (!make_step_response_path(
          think_path,
          sizeof(think_path),
          responses_dir,
          summary->trajectory_id,
          i,
          "think"
        ) ||
        !make_step_response_path(
          call_path,
          sizeof(call_path),
          responses_dir,
          summary->trajectory_id,
          i,
          "call"
        )) {
      snprintf(err, sizeof(err), "steps[%zu] response path is too long", i);
      summary_set_error(summary, "parse", true, i, err);
      return false;
    }

    char thinking_content[TF_CLI_THINKING_CAP];
    char call_content[TF_CLI_PAYLOAD_TEXT_CAP];
    if (!read_llamacpp_response_content(
          think_path,
          thinking_content,
          sizeof(thinking_content),
          err,
          sizeof(err)
        ) ||
        !read_llamacpp_response_content(
          call_path,
          call_content,
          sizeof(call_content),
          err,
          sizeof(err)
        )) {
      char step_err[TF_MAX_ERROR];
      format_step_message(step_err, sizeof(step_err), "step ", i, " response failed: ", err);
      summary_set_error(summary, "parse", true, i, step_err);
      return false;
    }

    char *model_output =
      build_llamacpp_model_output(thinking_content, call_content, err, sizeof(err));
    if (model_output == 0) {
      char step_err[TF_MAX_ERROR];
      format_step_message(step_err, sizeof(step_err), "step ", i, " response failed: ", err);
      summary_set_error(summary, "parse", true, i, step_err);
      return false;
    }

    TfVerifyContext ctx = {
      .prior_state = current_state,
      .expected_trigger = has_expected_trigger ? expected_trigger : 0,
      .expected_state_after = has_expected_state_after ? expected_state_after : 0,
      .infer_trigger = false,
    };
    TfStepResult result = tf_verify_step(&ctx, model_output, schemas);
    free(model_output);
    if (!result.passed) {
      format_step_message(
        err,
        sizeof(err),
        "step ",
        i,
        " failed: ",
        result.error_message[0] != '\0' ? result.error_message : "?"
      );
      summary_set_error(summary, first_failing_subscore(&result), true, i, err);
      return false;
    }
    if (result.new_state[0] != '\0') {
      tf_copy_cstr(current_state, sizeof(current_state), result.new_state);
    }
  }

  char final_state[96] = "";
  bool has_final_state = false;
  if (!get_optional_string(
        trajectory,
        "final_state",
        final_state,
        sizeof(final_state),
        &has_final_state
      )) {
    summary_set_error(summary, "schema", false, 0, "final_state is too large or invalid");
    return false;
  }
  if (has_final_state && strcmp(current_state, final_state) != 0) {
    snprintf(
      err,
      sizeof(err),
      "final state '%s' does not match expected '%s'",
      current_state,
      final_state
    );
    summary_set_error(summary, "", false, 0, err);
    return false;
  }
  summary->passed = true;
  return true;
}

static int run_llamacpp_score_responses(int argc, char **argv) {
  const char *data_path = "data/test.jsonl";
  const char *responses_dir = 0;
  const char *schemas_dir = "schemas";
  const char *grammar_path = "schemas/jsonrpc.gbnf";
  const char *out_prefix = "c-llamacpp-responses";
  const char *run_name = "c-llamacpp-responses";

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-path") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--responses-dir") == 0 && i + 1 < argc) {
      responses_dir = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (strcmp(argv[i], "--run-name") == 0 && i + 1 < argc) {
      run_name = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (responses_dir == 0) {
    usage(stderr);
    return 2;
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  char test_set_sha256[TF_SHA256_HEX_SIZE] = "";
  char hash_err[TF_MAX_ERROR] = "";
  if (!tf_sha256_file_hex(data_path, test_set_sha256, hash_err, sizeof(hash_err))) {
    fprintf(stderr, "test set hash failed: %s\n", hash_err);
    return 2;
  }
  char grammar_digest[TF_SHA256_HEX_SIZE] = "";
  if (!optional_file_sha256(grammar_path, grammar_digest)) {
    fprintf(stderr, "grammar hash failed for %s\n", grammar_path);
    return 2;
  }

  FILE *f = fopen(data_path, "rb");
  if (f == 0) {
    fprintf(stderr, "failed to open %s\n", data_path);
    return 2;
  }

  TfTrajectorySummary *entries = 0;
  size_t entry_count = 0;
  size_t entry_cap = 0;
  size_t passed_count = 0;
  TfFailureModes failure_modes;
  memset(&failure_modes, 0, sizeof(failure_modes));

  size_t line_number = 0;
  char *line = 0;
  while ((line = tf_read_line(f)) != 0) {
    line_number++;
    char *trimmed = trim_line(line);
    if (*trimmed == '\0') {
      free(line);
      continue;
    }

    char fallback_id[96];
    snprintf(fallback_id, sizeof(fallback_id), "line-%zu", line_number);
    TfTrajectorySummary summary;
    TfJsonSpan trajectory;
    if (!parse_json_line(trimmed, &trajectory)) {
      memset(&summary, 0, sizeof(summary));
      tf_copy_cstr(summary.trajectory_id, sizeof(summary.trajectory_id), fallback_id);
      summary_set_error(&summary, "parse", false, 0, "invalid JSON");
    } else {
      (void)summarize_trajectory_llamacpp_responses(
        &schemas,
        trajectory,
        fallback_id,
        responses_dir,
        &summary
      );
    }
    if (summary.passed) {
      passed_count++;
    } else if (summary.first_failing_subscore[0] != '\0') {
      count_failure_mode(&failure_modes, summary.first_failing_subscore);
    }
    if (!append_trajectory_summary(&entries, &entry_count, &entry_cap, &summary)) {
      fprintf(stderr, "out of memory while scoring saved responses from %s\n", responses_dir);
      free(line);
      fclose(f);
      free(entries);
      return 2;
    }
    free(line);
  }
  fclose(f);

  char json_path[TF_CLI_PATH_CAP];
  char md_path[TF_CLI_PATH_CAP];
  if (!make_suffixed_path(json_path, sizeof(json_path), out_prefix, ".json") ||
      !make_suffixed_path(md_path, sizeof(md_path), out_prefix, ".md")) {
    fprintf(stderr, "out-prefix is too long\n");
    free(entries);
    return 2;
  }

  const TfScoreMeta meta = {
    .adapter_dir = "(saved-llamacpp-responses)",
    .base_model = "(saved-llamacpp-responses)",
    .eval_mode = "auto_regressive",
    .max_new_tokens = 0,
    .temperature = 0.0,
    .runtime = "llamacpp",
    .constrained_decoding = true,
    .llamacpp_base_url = "",
    .llamacpp_model = "(saved-responses)",
    .llamacpp_commit = "",
  };
  double lr_rate = entry_count == 0 ? 0.0 : (double)passed_count / (double)entry_count;
  if (!write_score_json(
        json_path,
        run_name,
        data_path,
        test_set_sha256,
        grammar_digest,
        &meta,
        entries,
        entry_count,
        passed_count,
        lr_rate,
        lr_rate,
        lr_rate,
        1,
        &failure_modes
      ) ||
      !write_score_md(
        md_path,
        run_name,
        data_path,
        test_set_sha256,
        grammar_digest,
        &meta,
        entries,
        entry_count,
        passed_count,
        lr_rate,
        lr_rate,
        lr_rate,
        1
      )) {
    fprintf(stderr, "failed to write saved-response score report for %s\n", out_prefix);
    free(entries);
    return 2;
  }

  printf("llamacpp_score_responses: %zu/%zu pass\n", passed_count, entry_count);
  printf("wrote %s and %s\n", json_path, md_path);
  free(entries);
  return 0;
}

static int run_llamacpp_payload(int argc, char **argv) {
  const char *data_path = 0;
  const char *grammar_path = 0;
  const char *out_prefix = "llamacpp-payload";
  const char *model = "local";
  const char *thinking_override = 0;
  size_t step_index = 0;
  size_t max_tokens = 1024;
  double temperature = 0.7;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-path") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--step-index") == 0 && i + 1 < argc) {
      if (!parse_cli_size(argv[++i], &step_index)) {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      if (!parse_cli_size(argv[++i], &max_tokens)) {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      char *end = 0;
      temperature = strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0') {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--thinking") == 0 && i + 1 < argc) {
      thinking_override = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (data_path == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *line = 0;
  TfJsonSpan step;
  if (!load_first_trajectory_step(data_path, step_index, &line, &step, err, sizeof(err))) {
    fprintf(stderr, "payload build failed: %s\n", err);
    return 2;
  }

  char user_content[TF_CLI_PAYLOAD_TEXT_CAP];
  if (!build_user_content(step, user_content, sizeof(user_content), err, sizeof(err))) {
    fprintf(stderr, "payload build failed: %s\n", err);
    free(line);
    return 2;
  }

  char thinking[TF_CLI_THINKING_CAP];
  if (thinking_override != 0) {
    tf_copy_cstr(thinking, sizeof(thinking), thinking_override);
  } else if (!tf_json_object_get_string(step, "thinking", thinking, sizeof(thinking))) {
    fprintf(stderr, "payload build failed: thinking is missing or too large\n");
    free(line);
    return 2;
  }

  char assistant_content[TF_CLI_THINKING_CAP + 32];
  assistant_content[0] = '\0';
  if (!append_text(assistant_content, sizeof(assistant_content), "<think>") ||
      !append_text(assistant_content, sizeof(assistant_content), thinking) ||
      !append_text(assistant_content, sizeof(assistant_content), "</think>")) {
    fprintf(stderr, "payload build failed: thinking text is too large\n");
    free(line);
    return 2;
  }

  char *grammar = 0;
  if (grammar_path != 0) {
    grammar = tf_read_file(grammar_path, err, sizeof(err));
    if (grammar == 0) {
      fprintf(stderr, "payload build failed: %s\n", err);
      free(line);
      return 2;
    }
  }

  char think_path[TF_CLI_PATH_CAP];
  char call_path[TF_CLI_PATH_CAP];
  if (!make_suffixed_path(think_path, sizeof(think_path), out_prefix, ".think.json") ||
      !make_suffixed_path(call_path, sizeof(call_path), out_prefix, ".call.json")) {
    fprintf(stderr, "payload build failed: out-prefix is too long\n");
    free(grammar);
    free(line);
    return 2;
  }

  const char *system_content =
    "You are an agent that operates a support-ticket system through its JSON-RPC API. For each prompt, "
    "produce <think>\xE2\x80\xA6</think> reasoning grounded in the prior state, then emit a "
    "single JSON-RPC request object on the line immediately after </think>. The request must "
    "validate against the method's params schema.";

  bool ok = write_llamacpp_payload(
              think_path,
              model,
              system_content,
              user_content,
              0,
              "Write only the reasoning text that belongs inside <think>.",
              max_tokens,
              temperature,
              0
            ) &&
            write_llamacpp_payload(
              call_path,
              model,
              system_content,
              user_content,
              assistant_content,
              "Now emit only the JSON-RPC request object.",
              max_tokens,
              temperature,
              grammar
            );
  free(grammar);
  free(line);
  if (!ok) {
    fprintf(stderr, "payload build failed: could not write output files\n");
    return 2;
  }

  char think_digest[TF_SHA256_HEX_SIZE] = "";
  char call_digest[TF_SHA256_HEX_SIZE] = "";
  if (!tf_sha256_file_hex(think_path, think_digest, err, sizeof(err)) ||
      !tf_sha256_file_hex(call_path, call_digest, err, sizeof(err))) {
    fprintf(stderr, "payload hash failed: %s\n", err);
    return 2;
  }
  printf("llamacpp_payload: wrote %s and %s\n", think_path, call_path);
  printf("think_sha256=%s\n", think_digest);
  printf("call_sha256=%s\n", call_digest);
  return 0;
}

static int run_grammar_sha256(int argc, char **argv) {
  const char *grammar_path = "schemas/jsonrpc.gbnf";

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  char digest[TF_SHA256_HEX_SIZE] = "";
  char err[TF_MAX_ERROR] = "";
  if (!tf_sha256_file_hex(grammar_path, digest, err, sizeof(err))) {
    if (strncmp(err, "failed to open ", strlen("failed to open ")) == 0) {
      puts("");
      return 0;
    }
    fprintf(stderr, "grammar hash failed: %s\n", err);
    return 2;
  }
  puts(digest);
  return 0;
}

static int run_llamacpp_complete(int argc, char **argv) {
  const char *request_path = 0;
  const char *base_url = "http://localhost:8080/v1";
  const char *api_key = "no-key";
  const char *out_path = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--request") == 0 && i + 1 < argc) {
      request_path = argv[++i];
    } else if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc) {
      base_url = argv[++i];
    } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
      api_key = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (request_path == 0) {
    usage(stderr);
    return 2;
  }

#ifdef TF_HAVE_CURL
  char err[TF_MAX_ERROR] = "";
  char *body = tf_read_file(request_path, err, sizeof(err));
  if (body == 0) {
    fprintf(stderr, "llamacpp-complete failed: %s\n", err);
    return 2;
  }

  // url = base_url (trailing '/' stripped) + "/chat/completions"
  char base[1024];
  tf_copy_cstr(base, sizeof(base), base_url);
  size_t blen = strlen(base);
  while (blen > 0 && base[blen - 1] == '/') {
    base[--blen] = '\0';
  }
  char url[1100];
  if ((size_t)snprintf(url, sizeof(url), "%s/chat/completions", base) >= sizeof(url)) {
    fprintf(stderr, "llamacpp-complete failed: base-url is too long\n");
    free(body);
    return 2;
  }

  char *response = 0;
  bool ok = tf_http_post_json(url, api_key, body, &response, err, sizeof(err));
  free(body);
  if (!ok) {
    fprintf(stderr, "llamacpp-complete failed: %s\n", err);
    free(response);
    return 2;
  }

  char content[TF_CLI_PAYLOAD_TEXT_CAP];
  bool extracted = extract_llamacpp_content(response, content, sizeof(content), err, sizeof(err));
  free(response);
  if (!extracted) {
    fprintf(stderr, "llamacpp-complete failed: %s\n", err);
    return 2;
  }

  if (out_path != 0 && !write_text_file(out_path, content)) {
    fprintf(stderr, "llamacpp-complete failed: could not write %s\n", out_path);
    return 2;
  }
  puts(content);
  return 0;
#else
  (void)base_url;
  (void)api_key;
  (void)out_path;
  fprintf(stderr, "llamacpp-complete unavailable: built without libcurl\n");
  return 3;
#endif
}

#ifdef TF_HAVE_CURL
// POST the JSON request stored at req_path and extract the reply content.
static bool http_complete_from_file(
  const char *url,
  const char *api_key,
  const char *req_path,
  char *content_out,
  size_t content_cap,
  char *err,
  size_t err_cap
) {
  char *body = tf_read_file(req_path, err, err_cap);
  if (body == 0) {
    return false;
  }
  char *resp = 0;
  bool ok = tf_http_post_json(url, api_key, body, &resp, err, err_cap);
  free(body);
  if (!ok) {
    free(resp);
    return false;
  }
  ok = extract_llamacpp_content(resp, content_out, content_cap, err, err_cap);
  free(resp);
  return ok;
}

// POST an Anthropic Messages request (x-api-key + anthropic-version headers) and
// extract the concatenated text content.
static bool anthropic_complete_from_file(
  const char *url,
  const char *api_key,
  const char *req_path,
  char *content_out,
  size_t content_cap,
  char *err,
  size_t err_cap
) {
  char *body = tf_read_file(req_path, err, err_cap);
  if (body == 0) {
    return false;
  }
  char key_header[1024];
  snprintf(key_header, sizeof(key_header), "x-api-key: %s", api_key != 0 ? api_key : "");
  const char *headers[] = {key_header, "anthropic-version: 2023-06-01"};
  char *resp = 0;
  bool ok = tf_http_post_json_headers(url, headers, 2, body, &resp, err, err_cap);
  free(body);
  if (!ok) {
    free(resp);
    return false;
  }
  ok = extract_anthropic_content(resp, content_out, content_cap, err, err_cap);
  free(resp);
  return ok;
}

// Live counterpart of summarize_trajectory_span: instead of reconstructing the
// model output from gold thinking/tool_call, generate it per step via the
// two-stage llama.cpp decode (free-form <think> text, then a grammar-constrained
// JSON-RPC call), thread state, and verify. Mirrors Python
// render_step_outputs_llamacpp + verify_trajectory for k=1 greedy.
// Generate and verify one full trajectory live. Generates every step (no
// short-circuit) so the last step's call is always available for pass@maj, and
// — like Python verify_trajectory — advances state only on a passed step and
// records the first failing step without stopping. Returns false only on a
// generation/infrastructure error; otherwise summary->passed reports the
// verification outcome and `final_call_out` (when non-NULL) gets the last
// step's JSON-RPC call (the canonical-call surface for pass@maj).
static bool summarize_trajectory_live(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  const char *fallback_id,
  const char *system_content,
  const char *url,
  const char *api_key,
  const char *grammar,
  const char *model,
  size_t max_tokens,
  double temperature,
  const char *think_req_path,
  const char *call_req_path,
  char *final_call_out,
  size_t final_call_cap,
  TfTrajectorySummary *summary
) {
  memset(summary, 0, sizeof(*summary));
  if (final_call_out != 0 && final_call_cap > 0) {
    final_call_out[0] = '\0';
  }
  tf_copy_cstr(summary->trajectory_id, sizeof(summary->trajectory_id), fallback_id);
  (void)tf_json_object_get_string(
    trajectory, "trajectory_id", summary->trajectory_id, sizeof(summary->trajectory_id)
  );

  char current_state[96];
  if (!tf_json_object_get_string(trajectory, "initial_state", current_state, sizeof(current_state))) {
    summary_set_error(summary, "schema", false, 0, "initial_state is required");
    return false;
  }
  tf_copy_cstr(summary->initial_state, sizeof(summary->initial_state), current_state);

  TfJsonSpan steps;
  if (!tf_json_object_get_value(trajectory, "steps", &steps)) {
    summary_set_error(summary, "schema", false, 0, "steps is required");
    return false;
  }
  size_t step_count = 0;
  if (!tf_json_array_count(steps, &step_count)) {
    summary_set_error(summary, "schema", false, 0, "steps must be an array");
    return false;
  }

  bool failed = false;
  char err[TF_MAX_ERROR] = "";
  for (size_t i = 0; i < step_count; i++) {
    TfJsonSpan step;
    if (!tf_json_array_get(steps, i, &step)) {
      snprintf(err, sizeof(err), "steps[%zu] is malformed", i);
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }
    char user_content[TF_CLI_PAYLOAD_TEXT_CAP];
    if (!build_user_content(step, user_content, sizeof(user_content), err, sizeof(err))) {
      summary_set_error(summary, "schema", true, i, err);
      return false;
    }
    char expected_trigger[96] = "";
    char expected_state_after[96] = "";
    bool has_et = false;
    bool has_esa = false;
    (void)get_optional_string(
      step, "expected_trigger", expected_trigger, sizeof(expected_trigger), &has_et
    );
    (void)get_optional_string(
      step, "expected_state_after", expected_state_after, sizeof(expected_state_after), &has_esa
    );

    // Stage 1: free-form thinking (no grammar).
    if (!write_llamacpp_payload(
          think_req_path, model, system_content, user_content, 0,
          "Write only the reasoning text that belongs inside <think>.", max_tokens, temperature, 0
        )) {
      summary_set_error(summary, "parse", true, i, "failed to write think request");
      return false;
    }
    char think[TF_CLI_THINKING_CAP];
    if (!http_complete_from_file(url, api_key, think_req_path, think, sizeof(think), err, sizeof(err))) {
      summary_set_error(summary, "parse", true, i, err);
      return false;
    }

    char assistant[TF_CLI_THINKING_CAP + 32];
    assistant[0] = '\0';
    if (!append_text(assistant, sizeof(assistant), "<think>") ||
        !append_text(assistant, sizeof(assistant), think) ||
        !append_text(assistant, sizeof(assistant), "</think>")) {
      summary_set_error(summary, "parse", true, i, "thinking text is too large");
      return false;
    }

    // Stage 2: grammar-constrained JSON-RPC call.
    if (!write_llamacpp_payload(
          call_req_path, model, system_content, user_content, assistant,
          "Now emit only the JSON-RPC request object.", max_tokens, temperature, grammar
        )) {
      summary_set_error(summary, "parse", true, i, "failed to write call request");
      return false;
    }
    char call[TF_CLI_PAYLOAD_TEXT_CAP];
    if (!http_complete_from_file(url, api_key, call_req_path, call, sizeof(call), err, sizeof(err))) {
      summary_set_error(summary, "parse", true, i, err);
      return false;
    }
    if (final_call_out != 0 && final_call_cap > 0) {
      tf_copy_cstr(final_call_out, final_call_cap, trim_line(call));  // last step wins
    }

    char *model_output = build_llamacpp_model_output(think, call, err, sizeof(err));
    if (model_output == 0) {
      summary_set_error(summary, "parse", true, i, err);
      return false;
    }
    TfVerifyContext ctx = {
      .prior_state = current_state,
      .expected_trigger = has_et ? expected_trigger : 0,
      .expected_state_after = has_esa ? expected_state_after : 0,
      .infer_trigger = false,
    };
    TfStepResult result = tf_verify_step(&ctx, model_output, schemas);
    free(model_output);
    if (!result.passed && !failed) {
      format_step_message(
        err, sizeof(err), "step ", i, " failed: ",
        result.error_message[0] != '\0' ? result.error_message : "?"
      );
      summary_set_error(summary, first_failing_subscore(&result), true, i, err);
      failed = true;
    }
    if (result.passed && result.new_state[0] != '\0') {
      tf_copy_cstr(current_state, sizeof(current_state), result.new_state);  // advance only on pass
    }
  }

  char final_state[96] = "";
  bool has_final = false;
  (void)get_optional_string(trajectory, "final_state", final_state, sizeof(final_state), &has_final);
  if (!failed && has_final && strcmp(current_state, final_state) != 0) {
    snprintf(
      err, sizeof(err), "final state '%s' does not match expected '%s'", current_state, final_state
    );
    summary_set_error(summary, "", false, 0, err);
    failed = true;
  }
  summary->passed = !failed;
  return true;
}
#endif  // TF_HAVE_CURL

static int run_llamacpp_eval(int argc, char **argv) {
  const char *data_path = 0;
  const char *schemas_dir = "schemas";
  const char *grammar_path = "schemas/jsonrpc.gbnf";
  const char *base_url = "http://localhost:8080/v1";
  const char *api_key = "no-key";
  const char *out_prefix = "llamacpp-eval";
  const char *run_name = "llamacpp-eval";
  const char *model = "local";
  const char *llamacpp_commit = "";
  size_t max_tokens = 1024;
  size_t k = 1;
  double temperature = 0.0;
  double sample_temperature = 0.7;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-path") == 0 && i + 1 < argc) {
      data_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc) {
      base_url = argv[++i];
    } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
      api_key = argv[++i];
    } else if (strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
      out_prefix = argv[++i];
    } else if (strcmp(argv[i], "--run-name") == 0 && i + 1 < argc) {
      run_name = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--llamacpp-commit") == 0 && i + 1 < argc) {
      llamacpp_commit = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      if (!parse_cli_size(argv[++i], &k)) {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      if (!parse_cli_size(argv[++i], &max_tokens)) {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      char *end = 0;
      temperature = strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0') {
        usage(stderr);
        return 2;
      }
    } else if (strcmp(argv[i], "--sample-temperature") == 0 && i + 1 < argc) {
      char *end = 0;
      sample_temperature = strtod(argv[++i], &end);
      if (end == argv[i] || *end != '\0') {
        usage(stderr);
        return 2;
      }
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (data_path == 0) {
    usage(stderr);
    return 2;
  }

#ifdef TF_HAVE_CURL
  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  const char *system_content =
    "You are an agent that operates a support-ticket system through its JSON-RPC API. For each prompt, "
    "produce <think>\xE2\x80\xA6</think> reasoning grounded in the prior state, then emit a "
    "single JSON-RPC request object on the line immediately after </think>. The request must "
    "validate against the method's params schema.";

  char *grammar = 0;
  if (grammar_path != 0 && grammar_path[0] != '\0') {
    char gerr[TF_MAX_ERROR] = "";
    grammar = tf_read_file(grammar_path, gerr, sizeof(gerr));  // NULL -> unconstrained
  }

  char base[1024];
  tf_copy_cstr(base, sizeof(base), base_url);
  size_t blen = strlen(base);
  while (blen > 0 && base[blen - 1] == '/') {
    base[--blen] = '\0';
  }
  char url[1100];
  if ((size_t)snprintf(url, sizeof(url), "%s/chat/completions", base) >= sizeof(url)) {
    fprintf(stderr, "llamacpp-eval failed: base-url is too long\n");
    free(grammar);
    return 2;
  }

  char think_req[TF_CLI_PATH_CAP];
  char call_req[TF_CLI_PATH_CAP];
  if (!make_suffixed_path(think_req, sizeof(think_req), out_prefix, ".req.think.json") ||
      !make_suffixed_path(call_req, sizeof(call_req), out_prefix, ".req.call.json")) {
    fprintf(stderr, "llamacpp-eval failed: out-prefix is too long\n");
    free(grammar);
    return 2;
  }

  char test_set_sha256[TF_SHA256_HEX_SIZE] = "";
  char hash_err[TF_MAX_ERROR] = "";
  if (!tf_sha256_file_hex(data_path, test_set_sha256, hash_err, sizeof(hash_err))) {
    fprintf(stderr, "test set hash failed: %s\n", hash_err);
    free(grammar);
    return 2;
  }
  char grammar_digest[TF_SHA256_HEX_SIZE] = "";
  if (!optional_file_sha256(grammar_path, grammar_digest)) {
    fprintf(stderr, "grammar hash failed for %s\n", grammar_path);
    free(grammar);
    return 2;
  }

  FILE *f = fopen(data_path, "rb");
  if (f == 0) {
    fprintf(stderr, "failed to open %s\n", data_path);
    free(grammar);
    return 2;
  }

  TfTrajectorySummary *entries = 0;
  size_t entry_count = 0;
  size_t entry_cap = 0;
  size_t passed_count = 0;   // greedy passes (pass@1)
  size_t passk_count = 0;    // any of k samples passes (pass@k)
  size_t passmaj_count = 0;  // majority-call class has a passing sample (pass@maj)
  TfFailureModes failure_modes;
  memset(&failure_modes, 0, sizeof(failure_modes));

  size_t line_number = 0;
  char *line = 0;
  int rc = 0;
  while ((line = tf_read_line(f)) != 0) {
    line_number++;
    char *trimmed = trim_line(line);
    if (*trimmed == '\0') {
      free(line);
      continue;
    }
    char fallback_id[96];
    snprintf(fallback_id, sizeof(fallback_id), "line-%zu", line_number);
    TfTrajectorySummary summary;
    TfJsonSpan trajectory;
    bool parsed = parse_json_line(trimmed, &trajectory);
    if (!parsed) {
      memset(&summary, 0, sizeof(summary));
      tf_copy_cstr(summary.trajectory_id, sizeof(summary.trajectory_id), fallback_id);
      summary_set_error(&summary, "parse", false, 0, "invalid JSON");
    } else {
      // Greedy decode → pass@1, per_trajectory diagnostics, failure modes.
      (void)summarize_trajectory_live(
        &schemas, trajectory, fallback_id, system_content, url, api_key, grammar, model, max_tokens,
        temperature, think_req, call_req, 0, 0, &summary
      );
    }
    if (summary.passed) {
      passed_count++;
    } else {
      if (summary.first_failing_subscore[0] != '\0') {
        count_failure_mode(&failure_modes, summary.first_failing_subscore);
      }
      printf("FAIL %s: %s\n", summary.trajectory_id, summary.error_message);
      rc = 1;
    }

    // Sampled decodes → pass@k (any passes) and pass@maj (majority canonical
    // call has a passing sample), matching eval/score.aggregate.
    if (parsed && k > 0) {
      // Reject a `k` so large that `k * TF_CLI_PAYLOAD_TEXT_CAP` overflows size_t
      // (which would wrap to a small allocation and then be written past in the
      // sampling loop). Fail closed before allocating.
      if (k > ((size_t)-1) / TF_CLI_PAYLOAD_TEXT_CAP) {
        fprintf(stderr, "llamacpp-eval: --k %zu too large\n", k);
        free(line);
        fclose(f);
        free(entries);
        free(grammar);
        return 2;
      }
      char(*sample_calls)[TF_CLI_PAYLOAD_TEXT_CAP] = malloc(k * sizeof(*sample_calls));
      bool *sample_passed = malloc(k * sizeof(bool));
      if (sample_calls == 0 || sample_passed == 0) {
        fprintf(stderr, "out of memory sampling %s\n", data_path);
        free(sample_calls);
        free(sample_passed);
        free(line);
        fclose(f);
        free(entries);
        free(grammar);
        return 2;
      }
      bool any_passed = false;
      for (size_t j = 0; j < k; j++) {
        TfTrajectorySummary ss;
        (void)summarize_trajectory_live(
          &schemas, trajectory, fallback_id, system_content, url, api_key, grammar, model,
          max_tokens, sample_temperature, think_req, call_req, sample_calls[j],
          sizeof(sample_calls[j]), &ss
        );
        sample_passed[j] = ss.passed;
        any_passed = any_passed || ss.passed;
      }
      if (any_passed) {
        passk_count++;
      }
      // Most common final call (first-seen wins ties), then check if any sample
      // emitting it passed.
      size_t best = 0;
      size_t best_count = 0;
      for (size_t a = 0; a < k; a++) {
        size_t c = 0;
        for (size_t b = 0; b < k; b++) {
          if (strcmp(sample_calls[a], sample_calls[b]) == 0) {
            c++;
          }
        }
        if (c > best_count) {
          best_count = c;
          best = a;
        }
      }
      bool maj_passed = false;
      for (size_t a = 0; a < k; a++) {
        if (strcmp(sample_calls[a], sample_calls[best]) == 0 && sample_passed[a]) {
          maj_passed = true;
          break;
        }
      }
      if (maj_passed) {
        passmaj_count++;
      }
      free(sample_calls);
      free(sample_passed);
    }

    if (!append_trajectory_summary(&entries, &entry_count, &entry_cap, &summary)) {
      fprintf(stderr, "out of memory while evaluating %s\n", data_path);
      free(line);
      fclose(f);
      free(entries);
      free(grammar);
      return 2;
    }
    free(line);
  }
  fclose(f);
  remove(think_req);
  remove(call_req);

  char json_path[TF_CLI_PATH_CAP];
  char md_path[TF_CLI_PATH_CAP];
  if (!make_suffixed_path(json_path, sizeof(json_path), out_prefix, ".json") ||
      !make_suffixed_path(md_path, sizeof(md_path), out_prefix, ".md")) {
    fprintf(stderr, "out-prefix is too long\n");
    free(entries);
    free(grammar);
    return 2;
  }

  const TfScoreMeta meta = {
    .adapter_dir = "(llamacpp)",
    .base_model = model,
    .eval_mode = "llamacpp_live",
    .max_new_tokens = max_tokens,
    .temperature = temperature,
    .runtime = "llamacpp",
    .constrained_decoding = grammar != 0,
    .llamacpp_base_url = base_url,
    .llamacpp_model = model,
    .llamacpp_commit = llamacpp_commit,
  };
  double ev_pass1 = entry_count == 0 ? 0.0 : (double)passed_count / (double)entry_count;
  double ev_passk = entry_count == 0 ? 0.0 : (double)passk_count / (double)entry_count;
  double ev_passmaj = entry_count == 0 ? 0.0 : (double)passmaj_count / (double)entry_count;
  bool wrote =
    write_score_json(
      json_path, run_name, data_path, test_set_sha256, grammar_digest, &meta, entries, entry_count,
      passed_count, ev_pass1, ev_passk, ev_passmaj, k, &failure_modes
    ) &&
    write_score_md(
      md_path, run_name, data_path, test_set_sha256, grammar_digest, &meta, entries, entry_count,
      passed_count, ev_pass1, ev_passk, ev_passmaj, k
    );
  free(entries);
  free(grammar);
  if (!wrote) {
    fprintf(stderr, "failed to write eval report for %s\n", out_prefix);
    return 2;
  }
  printf("llamacpp_eval: %zu/%zu pass\n", passed_count, entry_count);
  printf("wrote %s and %s\n", json_path, md_path);
  return rc;
#else
  (void)schemas_dir;
  (void)grammar_path;
  (void)base_url;
  (void)api_key;
  (void)out_prefix;
  (void)run_name;
  (void)model;
  (void)llamacpp_commit;
  (void)max_tokens;
  (void)temperature;
  (void)sample_temperature;
  fprintf(stderr, "llamacpp-eval unavailable: built without libcurl\n");
  return 3;
#endif
}

// Non-const mapping lookup over the public TfYamlNode layout.
static TfYamlNode *yaml_map_get_mut(TfYamlNode *map, const char *key) {
  if (map == 0 || map->type != TF_YAML_MAPPING) {
    return 0;
  }
  for (size_t i = 0; i < map->as.map.count; i++) {
    if (strcmp(map->as.map.keys[i], key) == 0) {
      return map->as.map.values[i];
    }
  }
  return 0;
}

static void utc_iso_now(char *out, size_t cap) {
  time_t t = time(0);
  struct tm *tmv = gmtime(&t);
  if (tmv == 0 || strftime(out, cap, "%Y-%m-%dT%H:%M:%S.000000Z", tmv) == 0) {
    tf_copy_cstr(out, cap, "1970-01-01T00:00:00.000000Z");
  }
}

// Recursively create `path` (like mkdir -p). Returns true if it exists as a dir.
static bool ensure_dir(const char *path) {
  char buf[TF_CLI_PATH_CAP];
  tf_copy_cstr(buf, sizeof(buf), path);
  for (char *p = buf + 1; *p != '\0'; p++) {
    if (*p == '/') {
      *p = '\0';
      (void)mkdir(buf, 0777);
      *p = '/';
    }
  }
  (void)mkdir(buf, 0777);
  struct stat st;
  return stat(buf, &st) == 0 && S_ISDIR(st.st_mode);
}

// Attach hand-seed provenance + source, matching Python
// scenario_gen/provenance.hand_seed_provenance + expand-from-seeds.
static bool add_seed_provenance(
  TfYamlNode *seed,
  const char *timestamp,
  const char *version,
  const char *trajectory_id
) {
  TfYamlNode *prov = tf_yaml_new_mapping();
  if (prov == 0) {
    return false;
  }
  bool ok = tf_yaml_map_set(prov, "teacher_provider", tf_yaml_new_string("hand_seed")) &&
            tf_yaml_map_set(prov, "teacher_model", tf_yaml_new_null()) &&
            tf_yaml_map_set(prov, "teacher_model_version", tf_yaml_new_null()) &&
            tf_yaml_map_set(prov, "system_prompt_sha256", tf_yaml_new_null()) &&
            tf_yaml_map_set(prov, "expansion_seed", tf_yaml_new_null()) &&
            tf_yaml_map_set(prov, "expansion_temperature", tf_yaml_new_null()) &&
            tf_yaml_map_set(prov, "expansion_timestamp_utc", tf_yaml_new_string(timestamp)) &&
            tf_yaml_map_set(prov, "toyforge_version", tf_yaml_new_string(version)) &&
            tf_yaml_map_set(prov, "seed_trajectory_id", tf_yaml_new_string(trajectory_id));
  if (!ok) {
    tf_yaml_free(prov);
    return false;
  }
  return tf_yaml_map_set(seed, "source", tf_yaml_new_string("hand_seed")) &&
         tf_yaml_map_set(seed, "provenance", prov);
}

// Auto-populate prompt_context.prior_calls for steps after the first from the
// preceding steps' tool_calls (only when not already present), matching Python
// scenario_gen/seeds.load_seeds.
static bool add_prior_calls(TfYamlNode *seed) {
  TfYamlNode *steps = yaml_map_get_mut(seed, "steps");
  if (steps == 0 || steps->type != TF_YAML_SEQUENCE) {
    return true;
  }
  for (size_t j = 0; j < steps->as.seq.count; j++) {
    TfYamlNode *step = steps->as.seq.items[j];
    TfYamlNode *pc = yaml_map_get_mut(step, "prompt_context");
    if (pc == 0 || pc->type != TF_YAML_MAPPING || j == 0) {
      continue;
    }
    if (tf_yaml_map_get(pc, "prior_calls") != 0) {
      continue;
    }
    TfYamlNode *prior = tf_yaml_new_sequence();
    if (prior == 0) {
      return false;
    }
    for (size_t k = 0; k < j; k++) {
      TfYamlNode *tc = yaml_map_get_mut(steps->as.seq.items[k], "tool_call");
      if (tc != 0) {
        TfYamlNode *copy = tf_yaml_clone(tc);
        if (copy == 0 || !tf_yaml_seq_append_node(prior, copy)) {
          tf_yaml_free(prior);
          return false;
        }
      }
    }
    if (!tf_yaml_map_set(pc, "prior_calls", prior)) {
      return false;
    }
  }
  return true;
}

static int run_expand_from_seeds(int argc, char **argv) {
  const char *seeds_path = "scenarios/smoke-tiny-seeds.yaml";
  const char *schemas_dir = "schemas";
  const char *out_dir = "data/smoke-tiny";
  const char *version = "0.1.0";
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seeds-path") == 0 && i + 1 < argc) {
      seeds_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (strcmp(argv[i], "--toyforge-version") == 0 && i + 1 < argc) {
      version = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfSchemas schemas;  // validates schemas/ structure, mirroring Python
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *text = tf_read_file(seeds_path, err, sizeof(err));
  if (text == 0) {
    fprintf(stderr, "expand-from-seeds failed: %s\n", err);
    return 2;
  }
  TfYamlNode *root = tf_yaml_parse(text, err, sizeof(err));
  free(text);
  if (root == 0) {
    fprintf(stderr, "expand-from-seeds parse error: %s\n", err);
    return 2;
  }
  TfYamlNode *seeds = yaml_map_get_mut(root, "seeds");
  if (seeds == 0 || seeds->type != TF_YAML_SEQUENCE) {
    fprintf(stderr, "expand-from-seeds: 'seeds' is missing or not a list\n");
    tf_yaml_free(root);
    return 2;
  }

  if (!ensure_dir(out_dir)) {
    fprintf(stderr, "expand-from-seeds: could not create %s\n", out_dir);
    tf_yaml_free(root);
    return 2;
  }

  char timestamp[64];
  utc_iso_now(timestamp, sizeof(timestamp));

  for (size_t i = 0; i < seeds->as.seq.count; i++) {
    TfYamlNode *seed = seeds->as.seq.items[i];
    if (seed->type != TF_YAML_MAPPING) {
      fprintf(stderr, "expand-from-seeds: seed[%zu] is not a mapping\n", i);
      tf_yaml_free(root);
      return 2;
    }
    const TfYamlNode *tid = tf_yaml_map_get(seed, "trajectory_id");
    const char *tid_str = (tid != 0 && tid->type == TF_YAML_STRING) ? tid->as.string : "";
    if (!add_seed_provenance(seed, timestamp, version, tid_str) || !add_prior_calls(seed)) {
      fprintf(stderr, "expand-from-seeds: out of memory augmenting seed[%zu]\n", i);
      tf_yaml_free(root);
      return 2;
    }
  }

  const char *splits[] = {"train", "dev", "test"};
  for (size_t s = 0; s < sizeof(splits) / sizeof(splits[0]); s++) {
    char filename[32];
    char path[TF_CLI_PATH_CAP];
    snprintf(filename, sizeof(filename), "%s.jsonl", splits[s]);
    if (!tf_join_path(path, sizeof(path), out_dir, filename)) {
      fprintf(stderr, "expand-from-seeds: output path too long\n");
      tf_yaml_free(root);
      return 2;
    }
    FILE *f = fopen(path, "wb");
    if (f == 0) {
      fprintf(stderr, "expand-from-seeds: could not write %s\n", path);
      tf_yaml_free(root);
      return 2;
    }
    bool ok = true;
    for (size_t i = 0; ok && i < seeds->as.seq.count; i++) {
      char *json = tf_yaml_to_json(seeds->as.seq.items[i]);
      if (json == 0) {
        ok = false;
        break;
      }
      ok = fputs(json, f) != EOF && fputc('\n', f) != EOF;
      free(json);
    }
    if (!ok || ferror(f)) {
      fclose(f);
      fprintf(stderr, "expand-from-seeds: write error for %s\n", path);
      tf_yaml_free(root);
      return 2;
    }
    fclose(f);
  }

  printf("expand_from_seeds: wrote %zu seeds to %s/{train,dev,test}.jsonl\n",
         seeds->as.seq.count, out_dir);
  tf_yaml_free(root);
  return 0;
}

static int run_verify_seeds(int argc, char **argv) {
  const char *seeds_path = "scenarios/seeds.yaml";
  const char *schemas_dir = "schemas";
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seeds-path") == 0 && i + 1 < argc) {
      seeds_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *text = tf_read_file(seeds_path, err, sizeof(err));
  if (text == 0) {
    fprintf(stderr, "verify-seeds failed: %s\n", err);
    return 2;
  }
  TfYamlNode *root = tf_yaml_parse(text, err, sizeof(err));
  free(text);
  if (root == 0) {
    fprintf(stderr, "verify-seeds parse error: %s\n", err);
    return 2;
  }

  const TfYamlNode *seeds = tf_yaml_map_get(root, "seeds");
  if (seeds == 0 || seeds->type != TF_YAML_SEQUENCE) {
    fprintf(stderr, "verify-seeds: 'seeds' is missing or not a list\n");
    tf_yaml_free(root);
    return 2;
  }

  size_t total = seeds->as.seq.count;
  size_t passed = 0;
  int rc = 0;
  for (size_t i = 0; i < total; i++) {
    char fallback[32];
    snprintf(fallback, sizeof(fallback), "seed-%zu", i);
    char *json = tf_yaml_to_json(seeds->as.seq.items[i]);
    if (json == 0) {
      fprintf(stderr, "verify-seeds: out of memory serializing %s\n", fallback);
      tf_yaml_free(root);
      return 2;
    }
    TfJsonSpan span;
    TfTrajectorySummary summary;
    if (!parse_json_line(json, &span)) {
      printf("FAIL %s: seed did not serialize to a JSON object\n", fallback);
      rc = 1;
      free(json);
      continue;
    }
    if (summarize_trajectory_span(&schemas, span, fallback, &summary)) {
      passed++;
    } else {
      printf("FAIL %s: %s\n", summary.trajectory_id, summary.error_message);
      rc = 1;
    }
    free(json);
  }
  tf_yaml_free(root);
  printf("verify_seeds: %zu/%zu seeds pass\n", passed, total);
  return rc;
}

#define TF_REGISTRY_MAX 64

// Load every *.json file in `dir` into the registry arrays (heap buffers kept in
// `texts` so the parsed spans stay valid). Returns the count, or (size_t)-1 on
// error (with a message in `err`).
static size_t load_registry_dir(
  const char *dir, char **texts, TfJsonSpan *spans, char *err, size_t err_cap
) {
  DIR *d = opendir(dir);
  if (d == 0) {
    snprintf(err, err_cap, "could not open registry dir %s", dir);
    return (size_t)-1;
  }
  size_t n = 0;
  struct dirent *ent;
  while ((ent = readdir(d)) != 0) {
    size_t namelen = strlen(ent->d_name);
    if (namelen < 6 || strcmp(ent->d_name + namelen - 5, ".json") != 0) {
      continue;
    }
    if (n >= TF_REGISTRY_MAX) {
      // Fail loud rather than silently dropping docs: a $ref to a dropped
      // document would resolve to a pass-through annotation and mis-validate.
      tf_copy_cstr(err, err_cap, "too many registry documents");
      for (size_t i = 0; i < n; i++) {
        free(texts[i]);
      }
      closedir(d);
      return (size_t)-1;
    }
    char path[TF_CLI_PATH_CAP];
    if (!tf_join_path(path, sizeof(path), dir, ent->d_name)) {
      continue;
    }
    char *text = tf_read_file(path, err, err_cap);
    if (text == 0) {
      continue;
    }
    TfJsonSpan span;
    if (!parse_json_line(text, &span)) {
      free(text);
      continue;
    }
    texts[n] = text;
    spans[n] = span;
    n++;
  }
  closedir(d);
  return n;
}

static int run_jsonschema_validate(int argc, char **argv) {
  const char *schema_path = 0;
  const char *instance_path = 0;
  const char *registry_dir = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--schema") == 0 && i + 1 < argc) {
      schema_path = argv[++i];
    } else if (strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
      instance_path = argv[++i];
    } else if (strcmp(argv[i], "--registry-dir") == 0 && i + 1 < argc) {
      registry_dir = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (schema_path == 0 || instance_path == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *schema_text = tf_read_file(schema_path, err, sizeof(err));
  if (schema_text == 0) {
    fprintf(stderr, "jsonschema-validate failed: %s\n", err);
    return 2;
  }
  char *instance_text = tf_read_file(instance_path, err, sizeof(err));
  if (instance_text == 0) {
    fprintf(stderr, "jsonschema-validate failed: %s\n", err);
    free(schema_text);
    return 2;
  }

  TfJsonSpan schema;
  TfJsonSpan instance;
  if (!parse_json_line(schema_text, &schema)) {
    fprintf(stderr, "jsonschema-validate failed: schema is not valid JSON\n");
    free(schema_text);
    free(instance_text);
    return 2;
  }
  if (!parse_json_line(instance_text, &instance)) {
    fprintf(stderr, "jsonschema-validate failed: instance is not valid JSON\n");
    free(schema_text);
    free(instance_text);
    return 2;
  }

  char *reg_texts[TF_REGISTRY_MAX];
  TfJsonSpan reg_spans[TF_REGISTRY_MAX];
  size_t reg_count = 0;
  if (registry_dir != 0) {
    reg_count = load_registry_dir(registry_dir, reg_texts, reg_spans, err, sizeof(err));
    if (reg_count == (size_t)-1) {
      fprintf(stderr, "jsonschema-validate failed: %s\n", err);
      free(schema_text);
      free(instance_text);
      return 2;
    }
  }

  bool ok = tf_jsonschema_validate_ex(schema, instance, reg_spans, reg_count, err, sizeof(err));
  for (size_t i = 0; i < reg_count; i++) {
    free(reg_texts[i]);
  }
  free(schema_text);
  free(instance_text);
  if (ok) {
    puts("valid");
    return 0;
  }
  printf("invalid: %s\n", err);
  return 1;
}

static int run_yaml_to_json(int argc, char **argv) {
  const char *path = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--path") == 0 && i + 1 < argc) {
      path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (path == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *text = tf_read_file(path, err, sizeof(err));
  if (text == 0) {
    fprintf(stderr, "yaml-to-json failed: %s\n", err);
    return 2;
  }
  TfYamlNode *root = tf_yaml_parse(text, err, sizeof(err));
  free(text);
  if (root == 0) {
    fprintf(stderr, "yaml-to-json parse error: %s\n", err);
    return 2;
  }
  char *json = tf_yaml_to_json(root);
  tf_yaml_free(root);
  if (json == 0) {
    fprintf(stderr, "yaml-to-json failed: out of memory serializing\n");
    return 2;
  }
  puts(json);
  free(json);
  return 0;
}

// Byte-identical to toyforge.scenario_gen.expand._SYSTEM_PROMPT (Python). The
// provenance block fingerprints this string with SHA-256, so it MUST match the
// Python constant exactly; scripts/diff_c_teacher_request.py asserts equality.
static const char *const TF_TEACHER_SYSTEM_PROMPT_PARTS[] = {
  "You produce JSON trajectories for a reasoning-model training corpus.\n",
  "A trajectory is a sequence of steps; each step has <think>reasoning</think> followed by a JSON-RPC call against the support-ticket API.\n",
  "\n",
  "## Method -> trigger map (authoritative)\n",
  "Each method can ONLY emit the listed triggers. Never claim another trigger.\n",
  "  ticket_open            -> ticket_open.accepted\n",
  "  ticket_get             -> (query-only, no trigger, no state change)\n",
  "  ticket_status          -> (query-only, no trigger, no state change)\n",
  "  ticket_history         -> (query-only, no trigger, no state change)\n",
  "  admin_agent_add        -> (query-only, no trigger, no state change)\n",
  "  ticket_reopen          -> ticket_reopen.issued\n",
  "  resolution_confirm     -> resolution_confirm.passed | resolution_confirm.failed\n",
  "  admin_escalate         -> escalation.start | escalation.resolved\n",
  "  admin_lifecycle_apply  -> lifecycle.close | lifecycle.archive_window_elapsed\n",
  "\n",
  "## Param format constraints (enforced by verifier)\n",
  "- `requester_id`: must match ^user: (e.g. user:alice, user:acme-it).\n",
  "- `body_text`: the customer's message; must be 4+ characters.\n",
  "- `additionalProperties: false` on every method — do NOT add extra fields (e.g. no `notes`, `priority`, `reason`). Stick to the documented params only.\n",
  "- `review_id`: non-empty string. `ticket_id`: non-empty string.\n",
  "\n",
  "## Query-step structure\n",
  "For query-only methods, omit `expected_trigger` and `expected_state_after` from the step. The `final_state` must equal the `initial_state`.\n",
  "\n",
  "## <think> quality bar\n",
  "Each thinking block must: (a) name the prior_state explicitly, (b) state which trigger will be emitted (or 'query-only, no trigger'), (c) explain WHY this method/transition is correct given the situation (not just paraphrase the prompt). Reasoning must be substantive, not decorative.\n",
  "\n",
  "## Multi-step prior_calls\n",
  "For step N > 0 in a multi-step trajectory, `prompt_context.prior_calls` must list the prior steps' `tool_call` objects in order. This grounds the model's reasoning in what has already been called.\n",
  "\n",
  "## Variation guidance\n",
  "Vary requester_id names, ticket_id values, agent_id values, and scenario descriptions across expansions. Avoid repeating the seed's identifiers (e.g. if seed uses user:alice and T-7, use different values).\n",
  "\n",
  "## Worked example\n",
  "Seed (abbreviated):\n",
  "  {\"trajectory_id\": \"seed_escalation_start\", \"initial_state\": \"ESCALATED\", \"steps\": [{\"prompt_context\": {\"prior_state\": \"ESCALATED\", \"situation\": \"T-7 failed customer confirmation twice\"}, \"thinking\": \"...\", \"tool_call\": {\"jsonrpc\": \"2.0\", \"method\": \"admin_escalate\", \"params\": {\"agent_id\": \"agent-4\", \"ticket_id\": \"T-7\"}, \"id\": 1}, \"expected_trigger\": \"escalation.start\", \"expected_state_after\": \"ENGINEERING\"}], \"final_state\": \"ENGINEERING\"}\n",
  "\n",
  "Good expansion:\n",
  "  {\"trajectory_id\": \"exp_escalation_start_abc123\", \"initial_state\": \"ESCALATED\", \"source\": \"teacher_expansion\", \"difficulty\": \"easy\", \"description\": \"Hand T-88 to engineering after the fix did not hold.\", \"steps\": [{\"prompt_context\": {\"prior_state\": \"ESCALATED\", \"situation\": \"T-88 was reopened and the customer rejected the resolution.\"}, \"thinking\": \"The ticket is in ESCALATED because the customer rejected the resolution on review. admin_escalate from ESCALATED emits escalation.start (not escalation.resolved — that comes only from ENGINEERING). I pass agent_id=agent-12 and ticket_id=T-88 to open the engineering escalation.\", \"tool_call\": {\"jsonrpc\": \"2.0\", \"method\": \"admin_escalate\", \"params\": {\"agent_id\": \"agent-12\", \"ticket_id\": \"T-88\"}, \"id\": 1}, \"expected_trigger\": \"escalation.start\", \"expected_state_after\": \"ENGINEERING\"}], \"final_state\": \"ENGINEERING\"}\n",
  "\n",
  "## Negative example (verifier will REJECT this)\n",
  "BAD: calling admin_escalate but claiming expected_trigger: routing.assigned\n",
  "  {\"method\": \"admin_escalate\", ..., \"expected_trigger\": \"routing.assigned\"}\n",
  "Why it fails: admin_escalate can only emit escalation.start or escalation.resolved. The verifier sets precondition_met=0 and transition_valid=0 for this mismatch.\n",
  "\n",
  "## Output format\n",
  "Return ONLY the JSON trajectory (no prose, no markdown fences), matching the seed's top-level structure exactly. Set source=\"teacher_expansion\" and use a unique trajectory_id.",
};

// Concatenate TF_TEACHER_SYSTEM_PROMPT_PARTS into one heap string (the parts are
// split only to stay under C99's 4095-char string-literal limit). Caller frees.
static char *teacher_system_prompt(void) {
  size_t n = sizeof(TF_TEACHER_SYSTEM_PROMPT_PARTS) / sizeof(TF_TEACHER_SYSTEM_PROMPT_PARTS[0]);
  size_t total = 0;
  for (size_t i = 0; i < n; i++) {
    total += strlen(TF_TEACHER_SYSTEM_PROMPT_PARTS[i]);
  }
  char *buf = (char *)malloc(total + 1);
  if (buf == 0) {
    return 0;
  }
  size_t o = 0;
  for (size_t i = 0; i < n; i++) {
    size_t l = strlen(TF_TEACHER_SYSTEM_PROMPT_PARTS[i]);
    memcpy(buf + o, TF_TEACHER_SYSTEM_PROMPT_PARTS[i], l);
    o += l;
  }
  buf[o] = '\0';
  return buf;
}

// Build the user prompt: a fixed prefix followed by the seed JSON. The prefix
// matches Python's _build_user_prompt; the seed JSON need not be byte-identical
// to json.dumps(indent=2) because the user prompt is not fingerprinted.
static const char *const TF_TEACHER_USER_PREFIX =
  "Seed trajectory (extend with a NEW variant):\n\n";

// Write an OpenAI-compatible teacher chat request:
// {"model","messages":[{system},{user}],"max_tokens","temperature"}.
static bool write_teacher_request(
  FILE *out,
  const char *model,
  const char *system_content,
  const char *user_content,
  size_t max_tokens,
  double temperature
) {
  fputs("{\n  \"model\": ", out);
  if (!write_json_string_value(out, model)) {
    return false;
  }
  fputs(",\n  \"messages\": [\n", out);
  if (!write_message(out, "system", system_content, true) ||
      !write_message(out, "user", user_content, false)) {
    return false;
  }
  fprintf(out, "  ],\n  \"max_tokens\": %zu,\n  \"temperature\": %.6f\n}\n", max_tokens, temperature);
  return !ferror(out);
}

#ifdef TF_HAVE_CURL  // only used by the live teacher-expand anthropic HTTP path
// Write an Anthropic Messages API request: the system prompt is a top-level
// field (not a message), mirroring Python's AnthropicTeacher.complete.
static bool write_anthropic_request(
  FILE *out,
  const char *model,
  const char *system_content,
  const char *user_content,
  size_t max_tokens,
  double temperature
) {
  fputs("{\n  \"model\": ", out);
  if (!write_json_string_value(out, model)) {
    return false;
  }
  fprintf(out, ",\n  \"max_tokens\": %zu,\n  \"temperature\": %.6f,\n  \"system\": ", max_tokens, temperature);
  if (!write_json_string_value(out, system_content)) {
    return false;
  }
  fputs(",\n  \"messages\": [\n", out);
  if (!write_message(out, "user", user_content, false)) {
    return false;
  }
  fputs("  ]\n}\n", out);
  return !ferror(out);
}
#endif  // TF_HAVE_CURL

static int run_teacher_request(int argc, char **argv) {
  const char *seed_path = 0;
  const char *model = "claude-sonnet-4-6";
  const char *out_path = 0;
  size_t max_tokens = 2048;
  double temperature = 0.3;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seed-path") == 0 && i + 1 < argc) {
      seed_path = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      max_tokens = (size_t)strtoul(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      temperature = strtod(argv[++i], 0);
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (seed_path == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *seed_text = tf_read_file(seed_path, err, sizeof(err));
  if (seed_text == 0) {
    fprintf(stderr, "teacher-request failed: %s\n", err);
    return 2;
  }
  // Trim trailing whitespace/newline so the embedded JSON is clean.
  size_t slen = strlen(seed_text);
  while (slen > 0 && (seed_text[slen - 1] == '\n' || seed_text[slen - 1] == '\r' ||
                      seed_text[slen - 1] == ' ' || seed_text[slen - 1] == '\t')) {
    seed_text[--slen] = '\0';
  }
  size_t prefix_len = strlen(TF_TEACHER_USER_PREFIX);
  char *user = (char *)malloc(prefix_len + slen + 1);
  if (user == 0) {
    free(seed_text);
    fprintf(stderr, "teacher-request failed: out of memory\n");
    return 2;
  }
  memcpy(user, TF_TEACHER_USER_PREFIX, prefix_len);
  memcpy(user + prefix_len, seed_text, slen + 1);
  free(seed_text);

  char *system = teacher_system_prompt();
  if (system == 0) {
    free(user);
    fprintf(stderr, "teacher-request failed: out of memory\n");
    return 2;
  }

  FILE *out = stdout;
  if (out_path != 0) {
    out = fopen(out_path, "wb");
    if (out == 0) {
      free(user);
      free(system);
      fprintf(stderr, "teacher-request failed: could not open %s\n", out_path);
      return 2;
    }
  }
  bool ok = write_teacher_request(out, model, system, user, max_tokens, temperature);
  free(user);
  free(system);
  if (out_path != 0) {
    ok = (fclose(out) == 0) && ok;
  }
  if (!ok) {
    fprintf(stderr, "teacher-request failed: write error\n");
    return 2;
  }
  return 0;
}

// Write the 11-field teacher provenance block, byte-faithful to Python
// scenario_gen.provenance.build_provenance (field order + null shape).
// response_format is emitted as null (the default-teacher path does not set a
// structured-output format); grammar_path is a string when provided, else null.
static bool write_teacher_provenance(
  FILE *out,
  const char *provider,
  const char *model,
  const char *system_prompt_sha256,
  long long expansion_seed,
  double temperature,
  const char *grammar_path,
  const char *timestamp,
  const char *toyforge_version,
  const char *seed_trajectory_id
) {
  // Compact single-line JSON so the block can embed in a JSONL row verbatim.
  fputs("{\"teacher_provider\": ", out);
  if (!write_json_string_value(out, provider)) {
    return false;
  }
  fputs(", \"teacher_model\": ", out);
  if (!write_json_string_value(out, model)) {
    return false;
  }
  fputs(", \"teacher_model_version\": null, \"system_prompt_sha256\": ", out);
  if (!write_json_string_value(out, system_prompt_sha256)) {
    return false;
  }
  fprintf(out, ", \"expansion_seed\": %lld", expansion_seed);
  fprintf(out, ", \"expansion_temperature\": %.6f", temperature);
  fputs(", \"response_format\": null, \"grammar_path\": ", out);
  if (grammar_path != 0) {
    if (!write_json_string_value(out, grammar_path)) {
      return false;
    }
  } else {
    fputs("null", out);
  }
  fputs(", \"expansion_timestamp_utc\": ", out);
  if (!write_json_string_value(out, timestamp)) {
    return false;
  }
  fputs(", \"toyforge_version\": ", out);
  if (!write_json_string_value(out, toyforge_version)) {
    return false;
  }
  fputs(", \"seed_trajectory_id\": ", out);
  if (!write_json_string_value(out, seed_trajectory_id)) {
    return false;
  }
  fputs("}", out);
  return !ferror(out);
}

// Stamp an accepted teacher trajectory into a compact JSONL row, mirroring
// Python's produced["source"]=...; produced["provenance"]=build_provenance(...):
// the original object's fields are re-emitted (minified), `source` is set to
// "teacher_expansion" (in place, or appended), `trajectory_id` is replaced with
// `new_traj_id` (the dedup result), and `provenance` is appended.
static bool write_accepted_row(
  FILE *out,
  TfJsonSpan traj,
  const char *new_traj_id,
  const char *provider,
  const char *model,
  const char *sys_sha,
  long long expansion_seed,
  double temperature,
  const char *grammar_path,
  const char *timestamp,
  const char *toyforge_version,
  const char *seed_trajectory_id
) {
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t nk = 0;
  if (!tf_json_collect_object_keys(traj, keys, TF_MAX_SCHEMA_PROPERTIES, &nk)) {
    return false;
  }
  fputs("{", out);
  bool first = true;
  bool wrote_source = false;
  for (size_t i = 0; i < nk; i++) {
    if (strcmp(keys[i], "provenance") == 0) {
      continue;  // appended fresh at the end
    }
    TfJsonSpan val;
    if (!tf_json_object_get_value(traj, keys[i], &val)) {
      return false;
    }
    if (!first) {
      fputs(", ", out);
    }
    first = false;
    if (!write_json_string_value(out, keys[i])) {
      return false;
    }
    fputs(": ", out);
    if (strcmp(keys[i], "source") == 0) {
      wrote_source = true;
      if (!write_json_string_value(out, "teacher_expansion")) {
        return false;
      }
    } else if (strcmp(keys[i], "trajectory_id") == 0) {
      if (!write_json_string_value(out, new_traj_id)) {
        return false;
      }
    } else if (!json_minify_to(out, val)) {
      return false;
    }
  }
  if (!wrote_source) {
    fputs(first ? "\"source\": " : ", \"source\": ", out);
    first = false;
    if (!write_json_string_value(out, "teacher_expansion")) {
      return false;
    }
  }
  fputs(first ? "\"provenance\": " : ", \"provenance\": ", out);
  if (!write_teacher_provenance(
        out, provider, model, sys_sha, expansion_seed, temperature, grammar_path, timestamp,
        toyforge_version, seed_trajectory_id
      )) {
    return false;
  }
  fputs("}", out);
  return !ferror(out);
}

static int run_teacher_provenance(int argc, char **argv) {
  const char *provider = "anthropic";
  const char *model = "claude-sonnet-4-6";
  const char *grammar_path = 0;
  const char *timestamp = "1970-01-01T00:00:00.000000Z";
  const char *toyforge_version = "0.0.0";
  const char *seed_trajectory_id = 0;
  const char *out_path = 0;
  long long expansion_seed = 0;
  double temperature = 0.3;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
      provider = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--expansion-seed") == 0 && i + 1 < argc) {
      expansion_seed = strtoll(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      temperature = strtod(argv[++i], 0);
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--timestamp") == 0 && i + 1 < argc) {
      timestamp = argv[++i];
    } else if (strcmp(argv[i], "--toyforge-version") == 0 && i + 1 < argc) {
      toyforge_version = argv[++i];
    } else if (strcmp(argv[i], "--seed-trajectory-id") == 0 && i + 1 < argc) {
      seed_trajectory_id = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (seed_trajectory_id == 0) {
    usage(stderr);
    return 2;
  }

  // system_prompt_sha256 = SHA-256 of the teacher system prompt (UTF-8 bytes).
  char *system = teacher_system_prompt();
  if (system == 0) {
    fprintf(stderr, "teacher-provenance failed: out of memory\n");
    return 2;
  }
  char sys_sha[TF_SHA256_HEX_SIZE];
  bool hashed = tf_sha256_hex(system, strlen(system), sys_sha);
  free(system);
  if (!hashed) {
    fprintf(stderr, "teacher-provenance failed: could not hash system prompt\n");
    return 2;
  }

  FILE *out = stdout;
  if (out_path != 0) {
    out = fopen(out_path, "wb");
    if (out == 0) {
      fprintf(stderr, "teacher-provenance failed: could not open %s\n", out_path);
      return 2;
    }
  }
  bool ok = write_teacher_provenance(
    out, provider, model, sys_sha, expansion_seed, temperature, grammar_path, timestamp,
    toyforge_version, seed_trajectory_id
  );
  if (out_path != 0) {
    ok = (fclose(out) == 0) && ok;
  }
  if (!ok) {
    fprintf(stderr, "teacher-provenance failed: write error\n");
    return 2;
  }
  return 0;
}

static int run_teacher_stamp(int argc, char **argv) {
  const char *trajectory_path = 0;
  const char *seed_trajectory_id = 0;
  const char *new_traj_id = 0;
  const char *provider = "anthropic";
  const char *model = "claude-sonnet-4-6";
  const char *grammar_path = 0;
  const char *timestamp = "1970-01-01T00:00:00.000000Z";
  const char *toyforge_version = "0.0.0";
  const char *out_path = 0;
  long long expansion_seed = 0;
  double temperature = 0.3;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--trajectory-path") == 0 && i + 1 < argc) {
      trajectory_path = argv[++i];
    } else if (strcmp(argv[i], "--seed-trajectory-id") == 0 && i + 1 < argc) {
      seed_trajectory_id = argv[++i];
    } else if (strcmp(argv[i], "--new-trajectory-id") == 0 && i + 1 < argc) {
      new_traj_id = argv[++i];
    } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
      provider = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--expansion-seed") == 0 && i + 1 < argc) {
      expansion_seed = strtoll(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      temperature = strtod(argv[++i], 0);
    } else if (strcmp(argv[i], "--grammar-path") == 0 && i + 1 < argc) {
      grammar_path = argv[++i];
    } else if (strcmp(argv[i], "--timestamp") == 0 && i + 1 < argc) {
      timestamp = argv[++i];
    } else if (strcmp(argv[i], "--toyforge-version") == 0 && i + 1 < argc) {
      toyforge_version = argv[++i];
    } else if (strcmp(argv[i], "--out-path") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (trajectory_path == 0 || seed_trajectory_id == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *raw = tf_read_file(trajectory_path, err, sizeof(err));
  if (raw == 0) {
    fprintf(stderr, "teacher-stamp failed: %s\n", err);
    return 2;
  }
  TfJsonSpan traj;
  if (!parse_json_line(raw, &traj) || tf_json_type(traj) != TF_JSON_OBJECT) {
    free(raw);
    fprintf(stderr, "teacher-stamp failed: trajectory is not a JSON object\n");
    return 2;
  }
  char idbuf[256];
  if (new_traj_id == 0) {
    // Python: produced.get("trajectory_id", "unknown") — keep the original id.
    if (tf_json_object_get_string(traj, "trajectory_id", idbuf, sizeof(idbuf))) {
      new_traj_id = idbuf;
    } else {
      new_traj_id = "unknown";
    }
  }
  char *system = teacher_system_prompt();
  if (system == 0) {
    free(raw);
    fprintf(stderr, "teacher-stamp failed: out of memory\n");
    return 2;
  }
  char sys_sha[TF_SHA256_HEX_SIZE];
  bool hashed = tf_sha256_hex(system, strlen(system), sys_sha);
  free(system);
  if (!hashed) {
    free(raw);
    fprintf(stderr, "teacher-stamp failed: could not hash system prompt\n");
    return 2;
  }

  FILE *out = stdout;
  if (out_path != 0) {
    out = fopen(out_path, "wb");
    if (out == 0) {
      free(raw);
      fprintf(stderr, "teacher-stamp failed: could not open %s\n", out_path);
      return 2;
    }
  }
  bool ok = write_accepted_row(
    out, traj, new_traj_id, provider, model, sys_sha, expansion_seed, temperature, grammar_path,
    timestamp, toyforge_version, seed_trajectory_id
  );
  ok = (fputc('\n', out) != EOF) && ok;
  free(raw);
  if (out_path != 0) {
    ok = (fclose(out) == 0) && ok;
  }
  if (!ok) {
    fprintf(stderr, "teacher-stamp failed: write error\n");
    return 2;
  }
  return 0;
}

// Strip ```json ... ``` (or ``` ... ```) fences, mirroring Python
// expand._strip_markdown_fences: trim, drop a leading fence + optional language
// tag + whitespace, then drop a trailing fence. Returns a heap string.
static char *strip_markdown_fences(const char *raw) {
  const char *start = raw;
  while (*start != '\0' && isspace((unsigned char)*start)) {
    start++;
  }
  const char *end = raw + strlen(raw);
  while (end > start && isspace((unsigned char)end[-1])) {
    end--;
  }
  size_t len = (size_t)(end - start);
  char *s = (char *)malloc(len + 1);
  if (s == 0) {
    return 0;
  }
  memcpy(s, start, len);
  s[len] = '\0';

  if (len >= 3 && strncmp(s, "```", 3) == 0) {
    char *p = s + 3;
    while (*p != '\0' && isalpha((unsigned char)*p)) {  // ^```[A-Za-z]*
      p++;
    }
    while (*p != '\0' && isspace((unsigned char)*p)) {  // \s*
      p++;
    }
    size_t rlen = strlen(p);
    memmove(s, p, rlen + 1);
    char *e = s + rlen;  // .rstrip()
    while (e > s && isspace((unsigned char)e[-1])) {
      e--;
    }
    *e = '\0';
    size_t cur = strlen(s);
    if (cur >= 3 && strcmp(s + cur - 3, "```") == 0) {  // s[:-3].rstrip()
      e = s + cur - 3;
      while (e > s && isspace((unsigned char)e[-1])) {
        e--;
      }
      *e = '\0';
    }
  }
  return s;
}

// Classify a stripped teacher completion the way expand._process_one_expansion
// does after the network call: json.loads -> _render_outputs -> verify_trajectory.
// Returns "json_decode" / "malformed_structure" / "verifier_failed" / "accept".
// Mirrors verify_trajectory's exact []-vs-.get() boundary (missing steps /
// initial_state or a step missing thinking/tool_call is malformed_structure;
// empty steps, a failing step, or a final-state mismatch is verifier_failed).
// Scoped to realistic teacher output (string thinking/triggers, object tool_call).
static const char *classify_teacher_output(const TfSchemas *schemas, const char *stripped) {
  TfJsonSpan traj;
  if (!parse_json_line(stripped, &traj)) {
    return "json_decode";
  }
  if (tf_json_type(traj) != TF_JSON_OBJECT) {
    return "malformed_structure";  // traj["steps"] on a non-dict -> TypeError
  }
  TfJsonSpan steps;
  if (!tf_json_object_get_value(traj, "steps", &steps) || tf_json_type(steps) != TF_JSON_ARRAY) {
    return "malformed_structure";  // missing/non-iterable steps
  }
  size_t step_count = 0;
  if (!tf_json_array_count(steps, &step_count)) {
    return "malformed_structure";
  }

  // _render_outputs: a step missing thinking/tool_call (or not an object) raises.
  char **outputs = 0;
  if (step_count > 0) {
    outputs = (char **)calloc(step_count, sizeof(char *));
    if (outputs == 0) {
      return "malformed_structure";
    }
    for (size_t i = 0; i < step_count; i++) {
      TfJsonSpan step;
      char oerr[TF_MAX_ERROR];
      if (!tf_json_array_get(steps, i, &step) ||
          (outputs[i] = build_model_output(step, oerr, sizeof(oerr))) == 0) {
        for (size_t j = 0; j < i; j++) {
          free(outputs[j]);
        }
        free(outputs);
        return "malformed_structure";
      }
    }
  }

  if (step_count == 0) {
    return "verifier_failed";  // verify_trajectory: empty trajectory
  }

  char state[96];
  if (!tf_json_object_get_string(traj, "initial_state", state, sizeof(state))) {
    for (size_t j = 0; j < step_count; j++) {
      free(outputs[j]);
    }
    free(outputs);
    return "malformed_structure";  // traj["initial_state"] KeyError
  }

  bool all_passed = true;
  for (size_t i = 0; i < step_count; i++) {
    TfJsonSpan step;
    (void)tf_json_array_get(steps, i, &step);
    char et[96] = "";
    char esa[96] = "";
    bool has_et = false;
    bool has_esa = false;
    (void)get_optional_string(step, "expected_trigger", et, sizeof(et), &has_et);
    (void)get_optional_string(step, "expected_state_after", esa, sizeof(esa), &has_esa);
    TfVerifyContext ctx = {
      .prior_state = state,
      .expected_trigger = has_et ? et : 0,
      .expected_state_after = has_esa ? esa : 0,
      .infer_trigger = false,
    };
    TfStepResult r = tf_verify_step(&ctx, outputs[i], schemas);
    if (!r.passed) {
      all_passed = false;
    } else if (r.new_state[0] != '\0') {
      tf_copy_cstr(state, sizeof(state), r.new_state);
    }
  }
  for (size_t j = 0; j < step_count; j++) {
    free(outputs[j]);
  }
  free(outputs);

  if (all_passed) {
    char final_state[96] = "";
    bool has_final = false;
    if (get_optional_string(traj, "final_state", final_state, sizeof(final_state), &has_final) &&
        has_final && strcmp(state, final_state) != 0) {
      all_passed = false;
    }
  }
  return all_passed ? "accept" : "verifier_failed";
}

static int run_teacher_gate(int argc, char **argv) {
  const char *response_path = 0;
  const char *schemas_dir = "schemas";
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--response-path") == 0 && i + 1 < argc) {
      response_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (response_path == 0) {
    usage(stderr);
    return 2;
  }

  char err[TF_MAX_ERROR] = "";
  char *raw = tf_read_file(response_path, err, sizeof(err));
  if (raw == 0) {
    fprintf(stderr, "teacher-gate failed: %s\n", err);
    return 2;
  }
  char *stripped = strip_markdown_fences(raw);
  free(raw);
  if (stripped == 0) {
    fprintf(stderr, "teacher-gate failed: out of memory\n");
    return 2;
  }
  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    free(stripped);
    return 2;
  }
  const char *verdict = classify_teacher_output(&schemas, stripped);
  free(stripped);
  puts(verdict);
  return strcmp(verdict, "accept") == 0 ? 0 : 1;
}

#ifdef TF_HAVE_CURL
// Deterministic 64-bit PRNG (splitmix64) for the seeded train/dev shuffle. NOT
// CPython-RNG-identical — parity is impossible since Python's accepted order is
// non-deterministic under its thread pool — only reproducible within this tool.
static unsigned long long tf_splitmix64(unsigned long long *state) {
  unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void shuffle_lines(char **items, size_t n, unsigned long long seed) {
  unsigned long long state = seed;
  for (size_t i = n; i > 1; i--) {
    size_t j = (size_t)(tf_splitmix64(&state) % (unsigned long long)i);
    char *tmp = items[i - 1];
    items[i - 1] = items[j];
    items[j] = tmp;
  }
}

static char *dup_cstr(const char *s) {
  size_t n = strlen(s) + 1;
  char *out = (char *)malloc(n);
  if (out != 0) {
    memcpy(out, s, n);
  }
  return out;
}

static bool seen_contains(char **seen, size_t n, const char *id) {
  for (size_t i = 0; i < n; i++) {
    if (strcmp(seen[i], id) == 0) {
      return true;
    }
  }
  return false;
}

// Write one rejection log entry, mirroring expand's rejection dict shape
// (seed_id / attempt / reason / error / provenance).
static bool write_rejection(
  FILE *f,
  const char *seed_id,
  int attempt,
  const char *reason,
  const char *error,
  const char *provider,
  const char *model,
  const char *sys_sha,
  long long expansion_seed,
  double temperature,
  const char *timestamp,
  const char *toyforge_version
) {
  fputs("{\"seed_id\": ", f);
  if (!write_json_string_value(f, seed_id)) {
    return false;
  }
  fprintf(f, ", \"attempt\": %d, \"reason\": ", attempt);
  if (!write_json_string_value(f, reason)) {
    return false;
  }
  fputs(", \"error\": ", f);
  if (!write_json_string_value(f, error)) {
    return false;
  }
  fputs(", \"provenance\": ", f);
  if (!write_teacher_provenance(
        f, provider, model, sys_sha, expansion_seed, temperature, 0, timestamp,
        toyforge_version, seed_id
      )) {
    return false;
  }
  fputs("}\n", f);
  return !ferror(f);
}
#endif  // TF_HAVE_CURL

static int run_teacher_expand(int argc, char **argv) {
  const char *seeds_path = "scenarios/seeds.yaml";
  const char *schemas_dir = "schemas";
  const char *out_dir = "data";
  const char *base_url = "http://localhost:8080/v1";
  const char *api_key = "no-key";
  const char *provider = "anthropic";
  const char *model = "claude-sonnet-4-6";
  const char *toyforge_version = "0.0.0";
  long long expansions_per_seed = 25;
  long long max_retries = 3;
  long long shuffle_seed = 42;
  double train_frac = 0.8;
  double dev_frac = 0.1;
  double temperature = 0.3;
  long long max_tokens = 2048;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--seeds-path") == 0 && i + 1 < argc) {
      seeds_path = argv[++i];
    } else if (strcmp(argv[i], "--schemas-dir") == 0 && i + 1 < argc) {
      schemas_dir = argv[++i];
    } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (strcmp(argv[i], "--base-url") == 0 && i + 1 < argc) {
      base_url = argv[++i];
    } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
      api_key = argv[++i];
    } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
      provider = argv[++i];
    } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--toyforge-version") == 0 && i + 1 < argc) {
      toyforge_version = argv[++i];
    } else if (strcmp(argv[i], "--expansions-per-seed") == 0 && i + 1 < argc) {
      expansions_per_seed = strtoll(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--max-retries") == 0 && i + 1 < argc) {
      max_retries = strtoll(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--shuffle-seed") == 0 && i + 1 < argc) {
      shuffle_seed = strtoll(argv[++i], 0, 10);
    } else if (strcmp(argv[i], "--train-frac") == 0 && i + 1 < argc) {
      train_frac = strtod(argv[++i], 0);
    } else if (strcmp(argv[i], "--dev-frac") == 0 && i + 1 < argc) {
      dev_frac = strtod(argv[++i], 0);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      temperature = strtod(argv[++i], 0);
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      max_tokens = strtoll(argv[++i], 0, 10);
    } else {
      usage(stderr);
      return 2;
    }
  }

#ifndef TF_HAVE_CURL
  (void)seeds_path;
  (void)schemas_dir;
  (void)out_dir;
  (void)base_url;
  (void)api_key;
  (void)provider;
  (void)model;
  (void)toyforge_version;
  (void)expansions_per_seed;
  (void)max_retries;
  (void)shuffle_seed;
  (void)train_frac;
  (void)dev_frac;
  (void)temperature;
  (void)max_tokens;
  fprintf(stderr, "teacher-expand unavailable: built without libcurl\n");
  return 3;
#else
  if (train_frac + dev_frac > 1.0) {
    fprintf(stderr, "teacher-expand: train_frac + dev_frac > 1.0\n");
    return 2;
  }
  TfSchemas schemas;
  if (!load_cli_schemas(&schemas, schemas_dir)) {
    return 2;
  }
  char err[TF_MAX_ERROR] = "";
  char *seeds_text = tf_read_file(seeds_path, err, sizeof(err));
  if (seeds_text == 0) {
    fprintf(stderr, "teacher-expand failed: %s\n", err);
    return 2;
  }
  TfYamlNode *root = tf_yaml_parse(seeds_text, err, sizeof(err));
  free(seeds_text);
  if (root == 0) {
    fprintf(stderr, "teacher-expand parse error: %s\n", err);
    return 2;
  }
  TfYamlNode *seeds = yaml_map_get_mut(root, "seeds");
  if (seeds == 0 || seeds->type != TF_YAML_SEQUENCE) {
    fprintf(stderr, "teacher-expand: 'seeds' is missing or not a list\n");
    tf_yaml_free(root);
    return 2;
  }
  if (!ensure_dir(out_dir)) {
    fprintf(stderr, "teacher-expand: could not create %s\n", out_dir);
    tf_yaml_free(root);
    return 2;
  }
  char rejected_dir[TF_CLI_PATH_CAP];
  if (!tf_join_path(rejected_dir, sizeof(rejected_dir), out_dir, "rejected") ||
      !ensure_dir(rejected_dir)) {
    fprintf(stderr, "teacher-expand: could not create %s/rejected\n", out_dir);
    tf_yaml_free(root);
    return 2;
  }

  // Auto-populate prior_calls on every seed (Python load_seeds) before they are
  // serialized into teacher prompts.
  for (size_t i = 0; i < seeds->as.seq.count; i++) {
    if (!add_prior_calls(seeds->as.seq.items[i])) {
      fprintf(stderr, "teacher-expand: out of memory augmenting seeds\n");
      tf_yaml_free(root);
      return 2;
    }
  }

  char *system = teacher_system_prompt();
  char sys_sha[TF_SHA256_HEX_SIZE];
  if (system == 0 || !tf_sha256_hex(system, strlen(system), sys_sha)) {
    free(system);
    tf_yaml_free(root);
    fprintf(stderr, "teacher-expand: could not build system prompt\n");
    return 2;
  }

  // Anthropic uses its own Messages endpoint + request/response shape + headers;
  // every other provider speaks the OpenAI-compatible chat-completions contract.
  bool is_anthropic = strcmp(provider, "anthropic") == 0;
  char url[1100];
  char base[1024];
  tf_copy_cstr(base, sizeof(base), base_url);
  size_t blen = strlen(base);
  while (blen > 0 && base[blen - 1] == '/') {
    base[--blen] = '\0';
  }
  const char *endpoint = is_anthropic ? "/v1/messages" : "/chat/completions";
  if ((size_t)snprintf(url, sizeof(url), "%s%s", base, endpoint) >= sizeof(url)) {
    free(system);
    tf_yaml_free(root);
    fprintf(stderr, "teacher-expand: base-url too long\n");
    return 2;
  }

  char req_path[TF_CLI_PATH_CAP];
  char partial_path[TF_CLI_PATH_CAP];
  char rej_path[TF_CLI_PATH_CAP];
  if (!tf_join_path(req_path, sizeof(req_path), out_dir, ".teacher-request.json") ||
      !tf_join_path(partial_path, sizeof(partial_path), out_dir, "accepted_partial.jsonl") ||
      !tf_join_path(rej_path, sizeof(rej_path), rejected_dir, "rejections.jsonl")) {
    free(system);
    tf_yaml_free(root);
    fprintf(stderr, "teacher-expand: output path too long\n");
    return 2;
  }

  FILE *rej_log = fopen(rej_path, "a");
  FILE *partial_log = fopen(partial_path, "w");
  if (rej_log == 0 || partial_log == 0) {
    if (rej_log) {
      fclose(rej_log);
    }
    if (partial_log) {
      fclose(partial_log);
    }
    free(system);
    tf_yaml_free(root);
    fprintf(stderr, "teacher-expand: could not open log files\n");
    return 2;
  }
  char ts[64];
  utc_iso_now(ts, sizeof(ts));
  fprintf(rej_log, "{\"_run_start\": true, \"timestamp\": ");
  (void)write_json_string_value(rej_log, ts);
  fprintf(rej_log, ", \"shuffle_seed\": %lld}\n", shuffle_seed);

  char **seen = 0;
  size_t seen_n = 0;
  size_t seen_cap = 0;
  size_t accepted_count = 0;
  size_t rejected_count = 0;
  char *content = (char *)malloc(65536);
  bool fatal = (content == 0);

  for (size_t si = 0; !fatal && si < seeds->as.seq.count; si++) {
    TfYamlNode *seed = seeds->as.seq.items[si];
    const TfYamlNode *tid = tf_yaml_map_get(seed, "trajectory_id");
    const char *seed_id = (tid != 0 && tid->type == TF_YAML_STRING) ? tid->as.string : "unknown";
    char *seed_json = tf_yaml_to_json(seed);
    if (seed_json == 0) {
      fatal = true;
      break;
    }
    // user prompt = prefix + seed JSON
    size_t plen = strlen(TF_TEACHER_USER_PREFIX);
    size_t jlen = strlen(seed_json);
    char *user = (char *)malloc(plen + jlen + 1);
    if (user == 0) {
      free(seed_json);
      fatal = true;
      break;
    }
    memcpy(user, TF_TEACHER_USER_PREFIX, plen);
    memcpy(user + plen, seed_json, jlen + 1);
    free(seed_json);

    for (long long e = 0; !fatal && e < expansions_per_seed; e++) {
      bool accepted = false;
      for (long long attempt = 0; !accepted && attempt < max_retries; attempt++) {
        FILE *rf = fopen(req_path, "wb");
        bool req_ok =
          rf != 0 &&
          (is_anthropic
             ? write_anthropic_request(rf, model, system, user, (size_t)max_tokens, temperature)
             : write_teacher_request(rf, model, system, user, (size_t)max_tokens, temperature));
        if (!req_ok) {
          if (rf) {
            fclose(rf);
          }
          fatal = true;
          break;
        }
        fclose(rf);

        char herr[TF_MAX_ERROR] = "";
        bool got = is_anthropic
                     ? anthropic_complete_from_file(
                         url, api_key, req_path, content, 65536, herr, sizeof(herr)
                       )
                     : http_complete_from_file(url, api_key, req_path, content, 65536, herr, sizeof(herr));
        if (!got) {
          (void)write_rejection(
            rej_log, seed_id, (int)attempt, "teacher_error", herr, provider, model, sys_sha,
            shuffle_seed, temperature, ts, toyforge_version
          );
          continue;
        }
        char *stripped = strip_markdown_fences(content);
        if (stripped == 0) {
          fatal = true;
          break;
        }
        const char *verdict = classify_teacher_output(&schemas, stripped);
        if (strcmp(verdict, "accept") != 0) {
          (void)write_rejection(
            rej_log, seed_id, (int)attempt, verdict, "", provider, model, sys_sha, shuffle_seed,
            temperature, ts, toyforge_version
          );
          free(stripped);
          continue;
        }
        // Accept: dedupe trajectory_id, then stamp the row to the partial file.
        TfJsonSpan traj;
        if (!parse_json_line(stripped, &traj)) {
          // classify_teacher_output already accepted this, so a re-parse failure
          // is unexpected — fail safe rather than read an uninitialized span.
          (void)write_rejection(
            rej_log, seed_id, (int)attempt, "json_decode", "accepted output failed re-parse",
            provider, model, sys_sha, shuffle_seed, temperature, ts, toyforge_version
          );
          free(stripped);
          continue;
        }
        char orig_id[256];
        if (!tf_json_object_get_string(traj, "trajectory_id", orig_id, sizeof(orig_id))) {
          tf_copy_cstr(orig_id, sizeof(orig_id), "unknown");
        }
        char new_id[320];
        tf_copy_cstr(new_id, sizeof(new_id), orig_id);
        if (seen_contains(seen, seen_n, new_id)) {
          long long suffix = (long long)accepted_count;
          do {
            snprintf(new_id, sizeof(new_id), "%s_%lld", orig_id, suffix);
            suffix++;
          } while (seen_contains(seen, seen_n, new_id));
        }
        if (seen_n == seen_cap) {
          size_t ncap = seen_cap == 0 ? 16 : seen_cap * 2;
          char **ns = (char **)realloc(seen, ncap * sizeof(char *));
          if (ns == 0) {
            free(stripped);
            fatal = true;
            break;
          }
          seen = ns;
          seen_cap = ncap;
        }
        seen[seen_n] = dup_cstr(new_id);
        if (seen[seen_n] == 0) {
          free(stripped);
          fatal = true;
          break;
        }
        seen_n++;
        char acc_ts[64];
        utc_iso_now(acc_ts, sizeof(acc_ts));
        bool wrote = write_accepted_row(
          partial_log, traj, new_id, provider, model, sys_sha, shuffle_seed, temperature, 0,
          acc_ts, toyforge_version, seed_id
        );
        wrote = (fputc('\n', partial_log) != EOF) && wrote;
        free(stripped);
        if (!wrote) {
          fatal = true;
          break;
        }
        fflush(partial_log);
        accepted = true;
        accepted_count++;
      }
      if (!accepted && !fatal) {
        rejected_count++;
      }
    }
    free(user);
  }

  fclose(partial_log);
  fclose(rej_log);
  (void)remove(req_path);
  free(system);
  for (size_t i = 0; i < seen_n; i++) {
    free(seen[i]);
  }
  free(seen);
  free(content);

  if (fatal) {
    tf_yaml_free(root);
    fprintf(stderr, "teacher-expand: out of memory or write error\n");
    return 2;
  }

  // Read accepted rows back, shuffle, and split into train/dev.
  char **rows = 0;
  size_t rows_n = 0;
  size_t rows_cap = 0;
  FILE *pf = fopen(partial_path, "r");
  if (pf != 0) {
    char *line = 0;
    while ((line = tf_read_line(pf)) != 0) {
      char *trimmed = trim_line(line);
      if (*trimmed == '\0') {
        free(line);
        continue;
      }
      if (rows_n == rows_cap) {
        size_t ncap = rows_cap == 0 ? 32 : rows_cap * 2;
        char **nr = (char **)realloc(rows, ncap * sizeof(char *));
        if (nr == 0) {
          free(line);
          break;
        }
        rows = nr;
        rows_cap = ncap;
      }
      rows[rows_n] = dup_cstr(trimmed);
      free(line);
      if (rows[rows_n] == 0) {
        break;
      }
      rows_n++;
    }
    fclose(pf);
  }
  shuffle_lines(rows, rows_n, (unsigned long long)shuffle_seed);
  // Guard against negative fractions: a negative product is undefined behavior
  // when cast to size_t and would wrap to a huge value (defeating the clamps
  // below via integer overflow and printing a nonsensical SIZE_MAX count).
  // Treat a negative fraction as 0 — an empty split, matching the effective
  // result of Python's negative-slice semantics (accepted[:int(n*frac)]).
  size_t n_train = train_frac > 0.0 ? (size_t)((double)rows_n * train_frac) : 0;
  size_t n_dev = dev_frac > 0.0 ? (size_t)((double)rows_n * dev_frac) : 0;
  if (n_train > rows_n) {
    n_train = rows_n;
  }
  if (n_dev > rows_n) {
    n_dev = rows_n;
  }
  if (n_train + n_dev > rows_n) {
    n_dev = rows_n - n_train;
  }

  bool wrote_ok = true;
  const struct {
    const char *name;
    size_t lo;
    size_t hi;
  } parts[] = {{"train", 0, n_train}, {"dev", n_train, n_train + n_dev}};
  for (size_t pi = 0; pi < 2 && wrote_ok; pi++) {
    char fn[32];
    char path[TF_CLI_PATH_CAP];
    snprintf(fn, sizeof(fn), "%s.jsonl", parts[pi].name);
    if (!tf_join_path(path, sizeof(path), out_dir, fn)) {
      wrote_ok = false;
      break;
    }
    FILE *f = fopen(path, "wb");
    if (f == 0) {
      wrote_ok = false;
      break;
    }
    for (size_t r = parts[pi].lo; r < parts[pi].hi; r++) {
      if (fputs(rows[r], f) == EOF || fputc('\n', f) == EOF) {
        wrote_ok = false;
        break;
      }
    }
    if (ferror(f)) {
      wrote_ok = false;
    }
    fclose(f);
  }
  for (size_t i = 0; i < rows_n; i++) {
    free(rows[i]);
  }
  free(rows);

  // test.jsonl = hand seeds (source=hand_seed + provenance), never expanded.
  char seed_ts[64];
  utc_iso_now(seed_ts, sizeof(seed_ts));
  for (size_t i = 0; wrote_ok && i < seeds->as.seq.count; i++) {
    TfYamlNode *seed = seeds->as.seq.items[i];
    const TfYamlNode *tid = tf_yaml_map_get(seed, "trajectory_id");
    const char *tid_str = (tid != 0 && tid->type == TF_YAML_STRING) ? tid->as.string : "";
    if (!add_seed_provenance(seed, seed_ts, toyforge_version, tid_str)) {
      wrote_ok = false;
    }
  }
  if (wrote_ok) {
    char path[TF_CLI_PATH_CAP];
    if (tf_join_path(path, sizeof(path), out_dir, "test.jsonl")) {
      FILE *f = fopen(path, "wb");
      if (f != 0) {
        for (size_t i = 0; i < seeds->as.seq.count; i++) {
          char *json = tf_yaml_to_json(seeds->as.seq.items[i]);
          if (json == 0 || fputs(json, f) == EOF || fputc('\n', f) == EOF) {
            wrote_ok = false;
          }
          free(json);
        }
        if (ferror(f)) {
          wrote_ok = false;
        }
        fclose(f);
      } else {
        wrote_ok = false;
      }
    } else {
      wrote_ok = false;
    }
  }

  size_t test_count = seeds->as.seq.count;
  (void)remove(partial_path);
  tf_yaml_free(root);
  if (!wrote_ok) {
    fprintf(stderr, "teacher-expand: write error finalizing splits\n");
    return 2;
  }
  printf(
    "teacher_expand: accepted=%zu rejected=%zu train=%zu dev=%zu test=%zu\n",
    accepted_count,
    rejected_count,
    n_train,
    n_dev,
    test_count
  );
  return 0;
#endif  // TF_HAVE_CURL
}

// ---- audit-independence: train/dev/test overlap audit ---------------------
// Mirrors scripts/audit_train_dev_independence.py: per data-split pair, reports
// exact trajectory overlap (content excluding provenance/source), shared
// seed-ids, and high prompt-context prefix-similarity (Jaccard >= 0.8 on the
// tokenized first-step situation).

#define TF_AUDIT_MAX_TRAJ 8192
#define TF_AUDIT_MAX_TOKENS 512
#define TF_AUDIT_SEED_CAP 128
#define TF_AUDIT_TOKEN_CAP 128

typedef struct {
  char **lines;             // owned line buffers (keep alive for spans)
  TfJsonSpan *spans;        // trajectory spans into `lines`
  size_t count;
} TfTrajSet;

static void trajset_free(TfTrajSet *s) {
  for (size_t i = 0; i < s->count; i++) {
    free(s->lines[i]);
  }
  free(s->lines);
  free(s->spans);
  s->lines = 0;
  s->spans = 0;
  s->count = 0;
}

// Load every non-empty, valid-JSON line of a JSONL file as a trajectory span.
// missing=true when the file does not exist. Returns false only on a hard
// error (OOM or > TF_AUDIT_MAX_TRAJ rows — fail loud, like the other caps).
static bool trajset_load(const char *path, TfTrajSet *out, bool *missing) {
  out->lines = 0;
  out->spans = 0;
  out->count = 0;
  *missing = false;
  FILE *f = fopen(path, "rb");
  if (f == 0) {
    *missing = true;
    return true;
  }
  size_t cap = 0;
  char *line = 0;
  bool ok = true;
  while ((line = tf_read_line(f)) != 0) {
    char *trimmed = trim_line(line);
    if (*trimmed == '\0') {
      free(line);
      continue;
    }
    TfJsonSpan span;
    if (!parse_json_line(trimmed, &span)) {
      free(line);  // skip malformed lines, like the Python _load
      continue;
    }
    if (out->count >= TF_AUDIT_MAX_TRAJ) {
      free(line);
      fprintf(stderr, "audit-independence: %s exceeds %d trajectories\n", path, TF_AUDIT_MAX_TRAJ);
      ok = false;
      break;
    }
    if (out->count >= cap) {
      size_t ncap = cap == 0 ? 64 : cap * 2;
      char **nl = (char **)realloc(out->lines, ncap * sizeof(char *));
      TfJsonSpan *ns = (TfJsonSpan *)realloc(out->spans, ncap * sizeof(TfJsonSpan));
      if (nl != 0) {
        out->lines = nl;
      }
      if (ns != 0) {
        out->spans = ns;
      }
      if (nl == 0 || ns == 0) {
        free(line);
        ok = false;
        break;
      }
      cap = ncap;
    }
    // span points into `trimmed`, which lives inside `line`; keep `line`.
    out->lines[out->count] = line;
    out->spans[out->count] = span;
    out->count++;
  }
  fclose(f);
  if (!ok) {
    trajset_free(out);
  }
  return ok;
}

// Content equality excluding the top-level provenance/source keys (the Python
// _content_hash drops those). Order-independent via tf_json_equal.
static bool audit_filtered_key_count(TfJsonSpan obj, size_t *out) {
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t n = 0;
  if (!tf_json_collect_object_keys(obj, keys, TF_MAX_SCHEMA_PROPERTIES, &n)) {
    return false;
  }
  size_t kept = 0;
  for (size_t i = 0; i < n; i++) {
    if (strcmp(keys[i], "provenance") != 0 && strcmp(keys[i], "source") != 0) {
      kept++;
    }
  }
  *out = kept;
  return true;
}

static bool audit_content_equal(TfJsonSpan a, TfJsonSpan b) {
  if (tf_json_type(a) != TF_JSON_OBJECT || tf_json_type(b) != TF_JSON_OBJECT) {
    return tf_json_equal(a, b);
  }
  size_t ka = 0;
  size_t kb = 0;
  if (!audit_filtered_key_count(a, &ka) || !audit_filtered_key_count(b, &kb) || ka != kb) {
    return false;
  }
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t n = 0;
  if (!tf_json_collect_object_keys(a, keys, TF_MAX_SCHEMA_PROPERTIES, &n)) {
    return false;
  }
  for (size_t i = 0; i < n; i++) {
    if (strcmp(keys[i], "provenance") == 0 || strcmp(keys[i], "source") == 0) {
      continue;
    }
    TfJsonSpan va;
    TfJsonSpan vb;
    if (!tf_json_object_get_value(a, keys[i], &va) ||
        !tf_json_object_get_value(b, keys[i], &vb) || !tf_json_equal(va, vb)) {
      return false;
    }
  }
  return true;
}

// seed_trajectory_id, falling back to trajectory_id (Python: `... or ...`).
static void audit_seed_id(TfJsonSpan traj, char *out, size_t cap) {
  if (tf_json_object_get_string(traj, "seed_trajectory_id", out, cap) && out[0] != '\0') {
    return;
  }
  if (tf_json_object_get_string(traj, "trajectory_id", out, cap)) {
    return;
  }
  out[0] = '\0';
}

static void audit_traj_id(TfJsonSpan traj, char *out, size_t cap) {
  if (!tf_json_object_get_string(traj, "trajectory_id", out, cap)) {
    out[0] = '\0';
  }
}

// First-step situation text (steps[0].prompt_context.situation), or "".
static void audit_first_situation(TfJsonSpan traj, char *out, size_t cap) {
  out[0] = '\0';
  TfJsonSpan steps;
  TfJsonSpan step0;
  TfJsonSpan ctx;
  if (!tf_json_object_get_value(traj, "steps", &steps) ||
      !tf_json_array_get(steps, 0, &step0) ||
      !tf_json_object_get_object(step0, "prompt_context", &ctx)) {
    return;
  }
  (void)tf_json_object_get_string(ctx, "situation", out, cap);
}

// Lowercase whitespace-split into a deduped token set. Returns token count.
static size_t audit_tokenize(const char *text, char tokens[][96], size_t max_tokens) {
  size_t n = 0;
  const char *p = text;
  while (*p != '\0') {
    while (*p != '\0' && isspace((unsigned char)*p)) {
      p++;
    }
    if (*p == '\0') {
      break;
    }
    char tok[96];
    size_t ti = 0;
    while (*p != '\0' && !isspace((unsigned char)*p)) {
      if (ti + 1 < sizeof(tok)) {
        tok[ti++] = (char)tolower((unsigned char)*p);
      }
      p++;
    }
    tok[ti] = '\0';
    bool seen = false;
    for (size_t i = 0; i < n; i++) {
      if (strcmp(tokens[i], tok) == 0) {
        seen = true;
        break;
      }
    }
    if (!seen && n < max_tokens) {
      snprintf(tokens[n], 96, "%s", tok);
      n++;
    }
  }
  return n;
}

static double audit_jaccard(char a[][96], size_t na, char b[][96], size_t nb) {
  if (na == 0 && nb == 0) {
    return 1.0;
  }
  if (na == 0 || nb == 0) {
    return 0.0;
  }
  size_t inter = 0;
  for (size_t i = 0; i < na; i++) {
    for (size_t j = 0; j < nb; j++) {
      if (strcmp(a[i], b[j]) == 0) {
        inter++;
        break;
      }
    }
  }
  size_t uni = na + nb - inter;
  return uni == 0 ? 1.0 : (double)inter / (double)uni;
}

typedef struct {
  size_t exact_overlap;
  size_t shared_seed_count;
  char shared_seeds[TF_AUDIT_SEED_CAP][96];
  size_t shared_seed_listed;
  size_t high_sim_pairs;
} TfPairReport;

static void audit_pair_report(const TfTrajSet *left, const TfTrajSet *right, TfPairReport *r) {
  memset(r, 0, sizeof(*r));

  // Exact overlap: distinct left-contents that match some right-content.
  for (size_t i = 0; i < left->count; i++) {
    bool first = true;
    for (size_t j = 0; j < i; j++) {
      if (audit_content_equal(left->spans[i], left->spans[j])) {
        first = false;
        break;
      }
    }
    if (!first) {
      continue;
    }
    for (size_t k = 0; k < right->count; k++) {
      if (audit_content_equal(left->spans[i], right->spans[k])) {
        r->exact_overlap++;
        break;
      }
    }
  }

  // Shared seed-ids (set intersection, deduped).
  for (size_t i = 0; i < left->count; i++) {
    char lid[96];
    audit_seed_id(left->spans[i], lid, sizeof(lid));
    if (lid[0] == '\0') {
      continue;
    }
    bool in_right = false;
    for (size_t k = 0; k < right->count && !in_right; k++) {
      char rid[96];
      audit_seed_id(right->spans[k], rid, sizeof(rid));
      if (strcmp(lid, rid) == 0) {
        in_right = true;
      }
    }
    if (!in_right) {
      continue;
    }
    bool already = false;
    for (size_t s = 0; s < r->shared_seed_listed; s++) {
      if (strcmp(r->shared_seeds[s], lid) == 0) {
        already = true;
        break;
      }
    }
    if (already) {
      continue;
    }
    r->shared_seed_count++;
    if (r->shared_seed_listed < TF_AUDIT_SEED_CAP) {
      snprintf(r->shared_seeds[r->shared_seed_listed], 96, "%s", lid);
      r->shared_seed_listed++;
    }
  }

  // High prefix-similarity pairs (Jaccard >= 0.8, distinct trajectory_ids).
  for (size_t i = 0; i < left->count; i++) {
    char lsit[TF_MAX_TEXT];
    audit_first_situation(left->spans[i], lsit, sizeof(lsit));
    char ltok[TF_AUDIT_TOKEN_CAP][96];
    size_t lnt = audit_tokenize(lsit, ltok, TF_AUDIT_TOKEN_CAP);
    if (lnt == 0) {
      continue;
    }
    char lid[96];
    audit_traj_id(left->spans[i], lid, sizeof(lid));
    for (size_t k = 0; k < right->count; k++) {
      char rsit[TF_MAX_TEXT];
      audit_first_situation(right->spans[k], rsit, sizeof(rsit));
      char rtok[TF_AUDIT_TOKEN_CAP][96];
      size_t rnt = audit_tokenize(rsit, rtok, TF_AUDIT_TOKEN_CAP);
      if (rnt == 0) {
        continue;
      }
      char rid[96];
      audit_traj_id(right->spans[k], rid, sizeof(rid));
      if (strcmp(lid, rid) != 0 && audit_jaccard(ltok, lnt, rtok, rnt) >= 0.8) {
        r->high_sim_pairs++;
      }
    }
  }
}

static int cmp_cstr_qsort(const void *a, const void *b) {
  return strcmp((const char *)a, (const char *)b);
}

static void audit_print_pair(FILE *out, const char *name, const TfPairReport *r) {
  char label[64];
  snprintf(label, sizeof(label), "%s", name);
  for (char *p = label; *p; p++) {
    if (*p == '_') {
      *p = ' ';
    }
  }
  fprintf(out, "## %s\n", label);
  fprintf(out, "- Exact trajectory overlap: **%zu**\n", r->exact_overlap);
  // Sort the listed shared seed-ids and show up to 5 (Python sorts the set).
  char sorted[TF_AUDIT_SEED_CAP][96];
  for (size_t i = 0; i < r->shared_seed_listed; i++) {
    snprintf(sorted[i], 96, "%s", r->shared_seeds[i]);
  }
  qsort(sorted, r->shared_seed_listed, sizeof(sorted[0]), cmp_cstr_qsort);
  fprintf(out, "- Shared seed-ids: %zu \xE2\x86\x92 [", r->shared_seed_count);
  size_t show = r->shared_seed_listed < 5 ? r->shared_seed_listed : 5;
  for (size_t i = 0; i < show; i++) {
    fprintf(out, "%s'%s'", i ? ", " : "", sorted[i]);
  }
  fputs("]\n", out);
  fprintf(out, "- High prefix-similarity pairs (Jaccard \xE2\x89\xA5 0.8): **%zu**\n",
          r->high_sim_pairs);
  fputs("\n", out);
}

static int run_audit_independence(int argc, char **argv) {
  const char *data_dir = "data";
  const char *out_path = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  const char *names[3] = {"train", "dev", "test"};
  char paths[3][TF_CLI_PATH_CAP];
  TfTrajSet sets[3] = {{0}, {0}, {0}};
  bool missing[3] = {false, false, false};
  for (int i = 0; i < 3; i++) {
    char fn[32];
    snprintf(fn, sizeof(fn), "%s.jsonl", names[i]);
    if (!tf_join_path(paths[i], sizeof(paths[i]), data_dir, fn)) {
      fprintf(stderr, "audit-independence: path too long for %s\n", fn);
      for (int j = 0; j < i; j++) {
        trajset_free(&sets[j]);
      }
      return 2;
    }
    if (!trajset_load(paths[i], &sets[i], &missing[i])) {
      for (int j = 0; j <= i; j++) {
        trajset_free(&sets[j]);
      }
      return 2;
    }
  }

  FILE *out = stdout;
  if (out_path != 0) {
    // Create the output's parent directory (like Python's parent.mkdir).
    const char *slash = strrchr(out_path, '/');
    if (slash != 0 && slash != out_path) {
      char parent[TF_CLI_PATH_CAP];
      size_t plen = (size_t)(slash - out_path);
      if (plen < sizeof(parent)) {
        memcpy(parent, out_path, plen);
        parent[plen] = '\0';
        (void)ensure_dir(parent);
      }
    }
    out = fopen(out_path, "wb");
    if (out == 0) {
      fprintf(stderr, "audit-independence: cannot open %s\n", out_path);
      for (int j = 0; j < 3; j++) {
        trajset_free(&sets[j]);
      }
      return 2;
    }
  }

  fputs("# Train/Dev/Test independence audit\n\n", out);
  fputs("## File counts\n", out);
  for (int i = 0; i < 3; i++) {
    if (missing[i]) {
      fprintf(out, "- `%s`: MISSING (%s)\n", names[i], paths[i]);
    } else {
      fprintf(out, "- `%s`: %zu rows (%s)\n", names[i], sets[i].count, paths[i]);
    }
  }
  fputs("\n", out);

  TfPairReport tvd;
  TfPairReport tvt;
  TfPairReport dvt;
  audit_pair_report(&sets[0], &sets[1], &tvd);
  audit_pair_report(&sets[0], &sets[2], &tvt);
  audit_pair_report(&sets[1], &sets[2], &dvt);
  audit_print_pair(out, "train_vs_dev", &tvd);
  audit_print_pair(out, "train_vs_test", &tvt);
  audit_print_pair(out, "dev_vs_test", &dvt);

  fputs("## Policy\n", out);
  fputs(
    "- `train_vs_test` and `dev_vs_test` **MUST** report 0 exact overlap and 0 shared_seed_ids.\n",
    out
  );
  fputs(
    "- `train_vs_dev` may share seed-ids (both are teacher expansions of the same seed pool); \n",
    out
  );
  fputs(
    "  exact overlap must still be 0, and high_prefix_similarity_pairs warrants investigation.\n",
    out
  );

  bool wrote_ok = !ferror(out);
  if (out_path != 0) {
    fclose(out);
    if (wrote_ok) {
      printf("wrote %s\n", out_path);
    }
  }
  for (int j = 0; j < 3; j++) {
    trajset_free(&sets[j]);
  }
  return wrote_ok ? 0 : 2;
}

// ---- analyze-rejections: bucket data/rejected/rejections.jsonl ------------
// Mirrors scripts/analyze_rejections.py: counts by reason, by seed_id, and by
// error pattern (first 80 codepoints, stripped). Ties break by first-seen
// order (Python Counter.most_common / stable sort semantics).

#define TF_REJ_MAX_REASONS 32
#define TF_REJ_MAX_SEEDS 512
#define TF_REJ_MAX_ERRORS 2048
#define TF_REJ_KEY_CAP 384  // 80 codepoints * up to 4 bytes, plus headroom

typedef struct {
  char key[TF_REJ_KEY_CAP];
  size_t count;
  size_t first_seen;  // insertion order, for stable tie-break
} TfRejCount;

// Find-or-append `key`, incrementing its count. Returns false if the table is
// full with a new key (fail loud, like the other caps).
static bool rej_count_add(TfRejCount *tab, size_t *n, size_t cap, const char *key) {
  for (size_t i = 0; i < *n; i++) {
    if (strcmp(tab[i].key, key) == 0) {
      tab[i].count++;
      return true;
    }
  }
  if (*n >= cap) {
    return false;
  }
  snprintf(tab[*n].key, TF_REJ_KEY_CAP, "%s", key);
  tab[*n].count = 1;
  tab[*n].first_seen = *n;
  (*n)++;
  return true;
}

// qsort comparator: count descending, then first-seen ascending (stable order).
static int rej_cmp(const void *a, const void *b) {
  const TfRejCount *x = (const TfRejCount *)a;
  const TfRejCount *y = (const TfRejCount *)b;
  if (x->count != y->count) {
    return x->count < y->count ? 1 : -1;
  }
  if (x->first_seen != y->first_seen) {
    return x->first_seen < y->first_seen ? -1 : 1;
  }
  return 0;
}

// Bucket key for an error: first 80 codepoints, then strip ASCII whitespace
// (Python err[:80].strip()). `err` is the already-decoded error string.
static void rej_error_bucket(const char *err, char *out, size_t out_cap) {
  // Take the first 80 Unicode codepoints (count UTF-8 lead bytes).
  size_t cps = 0;
  size_t bytes = 0;
  while (err[bytes] != '\0' && cps < 80) {
    unsigned char c = (unsigned char)err[bytes];
    size_t adv = 1;
    if (c >= 0xF0) {
      adv = 4;
    } else if (c >= 0xE0) {
      adv = 3;
    } else if (c >= 0xC0) {
      adv = 2;
    }
    for (size_t k = 0; k < adv && err[bytes] != '\0'; k++) {
      bytes++;
    }
    cps++;
  }
  // Strip leading/trailing ASCII whitespace over [0, bytes).
  size_t start = 0;
  while (start < bytes && isspace((unsigned char)err[start])) {
    start++;
  }
  size_t end = bytes;
  while (end > start && isspace((unsigned char)err[end - 1])) {
    end--;
  }
  size_t len = end - start;
  if (len >= out_cap) {
    len = out_cap - 1;
  }
  memcpy(out, err + start, len);
  out[len] = '\0';
}

static int run_analyze_rejections(int argc, char **argv) {
  const char *input = "data/rejected/rejections.jsonl";
  const char *out_path = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      input = argv[++i];
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }

  TfRejCount *by_reason = (TfRejCount *)calloc(TF_REJ_MAX_REASONS, sizeof(TfRejCount));
  TfRejCount *by_seed = (TfRejCount *)calloc(TF_REJ_MAX_SEEDS, sizeof(TfRejCount));
  TfRejCount *by_err = (TfRejCount *)calloc(TF_REJ_MAX_ERRORS, sizeof(TfRejCount));
  if (by_reason == 0 || by_seed == 0 || by_err == 0) {
    free(by_reason);
    free(by_seed);
    free(by_err);
    fprintf(stderr, "analyze-rejections: out of memory\n");
    return 2;
  }
  size_t n_reason = 0;
  size_t n_seed = 0;
  size_t n_err = 0;
  size_t total = 0;
  bool ok = true;

  FILE *f = fopen(input, "rb");
  if (f != 0) {  // missing file => empty summary (like Python)
    char *line = 0;
    while (ok && (line = tf_read_line(f)) != 0) {
      char *trimmed = trim_line(line);
      if (*trimmed == '\0') {
        free(line);
        continue;
      }
      TfJsonSpan row;
      if (!parse_json_line(trimmed, &row)) {
        free(line);  // skip malformed, like Python
        continue;
      }
      if (tf_json_object_has_key(row, "_run_start")) {
        free(line);
        continue;
      }
      total++;
      char reason[TF_REJ_KEY_CAP];
      if (!tf_json_object_get_string(row, "reason", reason, sizeof(reason))) {
        snprintf(reason, sizeof(reason), "unknown");
      }
      char seed[TF_REJ_KEY_CAP];
      if (!tf_json_object_get_string(row, "seed_id", seed, sizeof(seed))) {
        snprintf(seed, sizeof(seed), "<unknown>");
      }
      char err[TF_MAX_TEXT];
      if (!tf_json_object_get_string(row, "error", err, sizeof(err))) {
        err[0] = '\0';
      }
      char err_key[TF_REJ_KEY_CAP];
      rej_error_bucket(err, err_key, sizeof(err_key));
      if (!rej_count_add(by_reason, &n_reason, TF_REJ_MAX_REASONS, reason) ||
          !rej_count_add(by_seed, &n_seed, TF_REJ_MAX_SEEDS, seed) ||
          !rej_count_add(by_err, &n_err, TF_REJ_MAX_ERRORS, err_key)) {
        fprintf(stderr, "analyze-rejections: too many distinct reasons/seeds/errors\n");
        ok = false;
      }
      free(line);
    }
    fclose(f);
  }

  if (!ok) {
    free(by_reason);
    free(by_seed);
    free(by_err);
    return 2;
  }

  qsort(by_reason, n_reason, sizeof(TfRejCount), rej_cmp);
  qsort(by_seed, n_seed, sizeof(TfRejCount), rej_cmp);
  qsort(by_err, n_err, sizeof(TfRejCount), rej_cmp);

  FILE *out = stdout;
  if (out_path != 0) {
    const char *slash = strrchr(out_path, '/');
    if (slash != 0 && slash != out_path) {
      char parent[TF_CLI_PATH_CAP];
      size_t plen = (size_t)(slash - out_path);
      if (plen < sizeof(parent)) {
        memcpy(parent, out_path, plen);
        parent[plen] = '\0';
        (void)ensure_dir(parent);
      }
    }
    out = fopen(out_path, "wb");
    if (out == 0) {
      free(by_reason);
      free(by_seed);
      free(by_err);
      fprintf(stderr, "analyze-rejections: cannot open %s\n", out_path);
      return 2;
    }
  }

  if (total == 0) {
    fputs("# Rejection summary\n\n_No rejections found._\n", out);
  } else {
    fputs("# Rejection summary\n\n", out);
    fprintf(out, "**Total rejections:** %zu\n\n", total);
    fputs("## By reason\n\n", out);
    for (size_t i = 0; i < n_reason; i++) {
      fprintf(out, "- `%s`: %zu\n", by_reason[i].key, by_reason[i].count);
    }
    fputs("\n## By seed (top 20)\n\n", out);
    for (size_t i = 0; i < n_seed && i < 20; i++) {
      fprintf(out, "- `%s`: %zu\n", by_seed[i].key, by_seed[i].count);
    }
    fputs("\n## Top error patterns\n\n", out);
    for (size_t i = 0; i < n_err && i < 10; i++) {
      fprintf(out, "- (%zu\xC3\x97) `%s`\n", by_err[i].count, by_err[i].key);
    }
  }
  // The report body above equals Python's `md` (write_text to a file). On stdout
  // Python uses print(md), which appends one newline — mirror that.
  if (out_path == 0) {
    fputs("\n", out);
  }

  bool wrote_ok = !ferror(out);
  if (out_path != 0) {
    fclose(out);
    if (wrote_ok) {
      printf("wrote %s\n", out_path);
    }
  }
  free(by_reason);
  free(by_seed);
  free(by_err);
  return wrote_ok ? 0 : 2;
}

int main(int argc, char **argv) {
  if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    usage(stdout);
    return argc < 2 ? 1 : 0;
  }
  if (strcmp(argv[1], "verify-step") == 0) {
    return run_verify_step(argc, argv);
  }
  if (strcmp(argv[1], "verify-data") == 0) {
    return run_verify_data(argc, argv);
  }
  if (strcmp(argv[1], "audit-independence") == 0) {
    return run_audit_independence(argc, argv);
  }
  if (strcmp(argv[1], "analyze-rejections") == 0) {
    return run_analyze_rejections(argc, argv);
  }
  if (strcmp(argv[1], "score-data") == 0) {
    return run_score_data(argc, argv);
  }
  if (strcmp(argv[1], "compare") == 0) {
    return run_compare(argc, argv);
  }
  if (strcmp(argv[1], "build-jsonrpc-gbnf") == 0) {
    return run_build_jsonrpc_gbnf(argc, argv);
  }
  if (strcmp(argv[1], "llamacpp-payload") == 0) {
    return run_llamacpp_payload(argc, argv);
  }
  if (strcmp(argv[1], "llamacpp-verify-response") == 0) {
    return run_llamacpp_verify_response(argc, argv);
  }
  if (strcmp(argv[1], "llamacpp-score-responses") == 0) {
    return run_llamacpp_score_responses(argc, argv);
  }
  if (strcmp(argv[1], "grammar-sha256") == 0) {
    return run_grammar_sha256(argc, argv);
  }
  if (strcmp(argv[1], "yaml-to-json") == 0) {
    return run_yaml_to_json(argc, argv);
  }
  if (strcmp(argv[1], "jsonschema-validate") == 0) {
    return run_jsonschema_validate(argc, argv);
  }
  if (strcmp(argv[1], "verify-seeds") == 0) {
    return run_verify_seeds(argc, argv);
  }
  if (strcmp(argv[1], "expand-from-seeds") == 0) {
    return run_expand_from_seeds(argc, argv);
  }
  if (strcmp(argv[1], "teacher-request") == 0) {
    return run_teacher_request(argc, argv);
  }
  if (strcmp(argv[1], "teacher-provenance") == 0) {
    return run_teacher_provenance(argc, argv);
  }
  if (strcmp(argv[1], "teacher-gate") == 0) {
    return run_teacher_gate(argc, argv);
  }
  if (strcmp(argv[1], "teacher-stamp") == 0) {
    return run_teacher_stamp(argc, argv);
  }
  if (strcmp(argv[1], "teacher-expand") == 0) {
    return run_teacher_expand(argc, argv);
  }
  if (strcmp(argv[1], "llamacpp-complete") == 0) {
    return run_llamacpp_complete(argc, argv);
  }
  if (strcmp(argv[1], "llamacpp-eval") == 0) {
    return run_llamacpp_eval(argc, argv);
  }
  usage(stderr);
  return 2;
}

#include "toyforge/schemas.h"

#include "toyforge/io.h"
#include "toyforge/json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *err, size_t err_cap, const char *msg) {
  tf_copy_cstr(err, err_cap, msg);
}

static char *trim(char *s) {
  while (*s && isspace((unsigned char)*s)) {
    s++;
  }
  char *end = s + strlen(s);
  while (end > s && isspace((unsigned char)end[-1])) {
    end--;
  }
  *end = '\0';
  return s;
}

static bool starts_with(const char *s, const char *prefix) {
  return strncmp(s, prefix, strlen(prefix)) == 0;
}

static TfMethodSpec *find_method_mut(TfSchemas *schemas, const char *method) {
  for (size_t i = 0; i < schemas->method_count; i++) {
    if (strcmp(schemas->methods[i].name, method) == 0) {
      return &schemas->methods[i];
    }
  }
  return 0;
}

static size_t method_index(const TfSchemas *schemas, const TfMethodSpec *method) {
  return (size_t)(method - schemas->methods);
}

static bool span_is_literal(TfJsonSpan span, const char *literal) {
  const char *p = tf_json_skip_ws(span.start);
  size_t len = strlen(literal);
  return (size_t)(span.end - p) == len && strncmp(p, literal, len) == 0;
}

static bool json_object_get_bool_default(TfJsonSpan object, const char *key, bool fallback) {
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value)) {
    return fallback;
  }
  if (span_is_literal(value, "true")) {
    return true;
  }
  if (span_is_literal(value, "false")) {
    return false;
  }
  return fallback;
}

static bool json_object_get_size(TfJsonSpan object, const char *key, size_t *out) {
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value)) {
    return false;
  }
  const char *p = tf_json_skip_ws(value.start);
  size_t parsed = 0;
  if (p >= value.end || !isdigit((unsigned char)*p)) {
    return false;
  }
  while (p < value.end && isdigit((unsigned char)*p)) {
    parsed = parsed * 10 + (size_t)(*p - '0');
    p++;
  }
  if (p != value.end) {
    return false;
  }
  *out = parsed;
  return true;
}

static bool collect_string_array(
  TfJsonSpan array,
  char values[][96],
  size_t max_values,
  size_t *count
) {
  const char *p = tf_json_skip_ws(array.start);
  *count = 0;
  if (p >= array.end || *p != '[') {
    return false;
  }
  p++;
  p = tf_json_skip_ws(p);
  if (*p == ']') {
    return true;
  }
  while (true) {
    if (*count >= max_values) {
      return false;
    }
    if (!tf_json_read_string(&p, values[*count], 96)) {
      return false;
    }
    (*count)++;
    p = tf_json_skip_ws(p);
    if (*p == ',') {
      p++;
      p = tf_json_skip_ws(p);
      continue;
    }
    if (*p == ']') {
      p++;
      return tf_json_skip_ws(p) == array.end;
    }
    return false;
  }
}

static bool name_in_list(const char *name, char values[][96], size_t value_count) {
  for (size_t i = 0; i < value_count; i++) {
    if (strcmp(name, values[i]) == 0) {
      return true;
    }
  }
  return false;
}

static bool compile_string_constraints(
  TfJsonSpan prop_schema,
  bool *has_min_length,
  size_t *min_length,
  bool *has_prefix,
  char *prefix,
  size_t prefix_cap,
  bool *format_annotation,
  char *err,
  size_t err_cap
) {
  *has_min_length = false;
  *min_length = 0;
  *has_prefix = false;
  *format_annotation = false;

  size_t parsed_min = 0;
  if (json_object_get_size(prop_schema, "minLength", &parsed_min)) {
    *has_min_length = true;
    *min_length = parsed_min;
  }
  char pattern[96];
  if (tf_json_object_get_string(prop_schema, "pattern", pattern, sizeof(pattern))) {
    if (pattern[0] != '^') {
      snprintf(err, err_cap, "unsupported JSON Schema pattern %s", pattern);
      return false;
    }
    *has_prefix = true;
    tf_copy_cstr(prefix, prefix_cap, pattern + 1);
  }
  char format[64];
  if (tf_json_object_get_string(prop_schema, "format", format, sizeof(format))) {
    *format_annotation = true;
  }
  return true;
}

static bool compile_child_property(
  TfSchemaChildProperty *out,
  const char *name,
  TfJsonSpan prop_schema,
  bool required,
  char *err,
  size_t err_cap
) {
  memset(out, 0, sizeof(*out));
  tf_copy_cstr(out->name, sizeof(out->name), name);
  out->required = required;

  char type[32];
  if (!tf_json_object_get_string(prop_schema, "type", type, sizeof(type))) {
    snprintf(err, err_cap, "schema property %s missing type", name);
    return false;
  }
  if (strcmp(type, "string") == 0) {
    out->type = TF_SCHEMA_VALUE_STRING;
    return compile_string_constraints(
      prop_schema,
      &out->has_min_length,
      &out->min_length,
      &out->has_prefix,
      out->prefix,
      sizeof(out->prefix),
      &out->format_annotation,
      err,
      err_cap
    );
  }
  if (strcmp(type, "object") == 0) {
    out->type = TF_SCHEMA_VALUE_OBJECT;
    return true;
  }
  snprintf(err, err_cap, "unsupported schema type %s for %s", type, name);
  return false;
}

static bool compile_schema_property(
  TfSchemaProperty *out,
  const char *name,
  TfJsonSpan prop_schema,
  bool required,
  char *err,
  size_t err_cap
) {
  memset(out, 0, sizeof(*out));
  tf_copy_cstr(out->name, sizeof(out->name), name);
  out->required = required;

  char type[32];
  if (!tf_json_object_get_string(prop_schema, "type", type, sizeof(type))) {
    snprintf(err, err_cap, "schema property %s missing type", name);
    return false;
  }
  if (strcmp(type, "string") == 0) {
    bool ignored_format = false;
    out->type = TF_SCHEMA_VALUE_STRING;
    return compile_string_constraints(
      prop_schema,
      &out->has_min_length,
      &out->min_length,
      &out->has_prefix,
      out->prefix,
      sizeof(out->prefix),
      &ignored_format,
      err,
      err_cap
    );
  }
  if (strcmp(type, "object") != 0) {
    snprintf(err, err_cap, "unsupported schema type %s for %s", type, name);
    return false;
  }

  out->type = TF_SCHEMA_VALUE_OBJECT;
  out->additional_properties = json_object_get_bool_default(prop_schema, "additionalProperties", true);
  TfJsonSpan properties;
  if (!tf_json_object_get_object(prop_schema, "properties", &properties)) {
    return true;
  }

  char child_names[TF_MAX_SCHEMA_CHILD_PROPERTIES][96];
  size_t child_count = 0;
  if (!tf_json_collect_object_keys(
        properties,
        child_names,
        TF_MAX_SCHEMA_CHILD_PROPERTIES,
        &child_count
      )) {
    snprintf(err, err_cap, "too many nested properties for %s", name);
    return false;
  }
  for (size_t i = 0; i < child_count; i++) {
    TfJsonSpan child_schema;
    if (!tf_json_object_get_object(properties, child_names[i], &child_schema)) {
      snprintf(err, err_cap, "nested property %s is not a schema object", child_names[i]);
      return false;
    }
    if (!compile_child_property(
          &out->child_properties[i],
          child_names[i],
          child_schema,
          false,
          err,
          err_cap
        )) {
      return false;
    }
  }
  out->child_property_count = child_count;
  return true;
}

static bool compile_param_schema(
  TfParamSchema *out,
  TfJsonSpan params,
  char *err,
  size_t err_cap
) {
  memset(out, 0, sizeof(*out));
  char type[32];
  if (!tf_json_object_get_string(params, "type", type, sizeof(type)) || strcmp(type, "object") != 0) {
    set_error(err, err_cap, "method params schema must be an object schema");
    return false;
  }
  out->additional_properties = json_object_get_bool_default(params, "additionalProperties", true);

  char required[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t required_count = 0;
  TfJsonSpan required_span;
  if (tf_json_object_get_value(params, "required", &required_span) &&
      !collect_string_array(required_span, required, TF_MAX_SCHEMA_PROPERTIES, &required_count)) {
    set_error(err, err_cap, "method params required list is invalid");
    return false;
  }

  TfJsonSpan properties;
  if (!tf_json_object_get_object(params, "properties", &properties)) {
    set_error(err, err_cap, "method params schema missing properties");
    return false;
  }
  char property_names[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t property_count = 0;
  if (!tf_json_collect_object_keys(properties, property_names, TF_MAX_SCHEMA_PROPERTIES, &property_count)) {
    set_error(err, err_cap, "too many method params properties");
    return false;
  }
  for (size_t i = 0; i < property_count; i++) {
    TfJsonSpan prop_schema;
    if (!tf_json_object_get_object(properties, property_names[i], &prop_schema)) {
      snprintf(err, err_cap, "property %s is not a schema object", property_names[i]);
      return false;
    }
    if (!compile_schema_property(
          &out->properties[i],
          property_names[i],
          prop_schema,
          name_in_list(property_names[i], required, required_count),
          err,
          err_cap
        )) {
      return false;
    }
  }
  out->property_count = property_count;
  out->loaded = true;
  return true;
}

static bool load_methods_json(TfSchemas *schemas, const char *schemas_dir, char *err, size_t err_cap) {
  char path[512];
  if (!tf_join_path(path, sizeof(path), schemas_dir, "jsonrpc-methods.json")) {
    set_error(err, err_cap, "schemas_dir path is too long");
    return false;
  }

  char *json = tf_read_file(path, err, err_cap);
  if (json == 0) {
    return false;
  }

  const char *start = tf_json_skip_ws(json);
  const char *end = start;
  bool ok = false;
  if (!tf_json_skip_value(&end) || *tf_json_skip_ws(end) != '\0') {
    set_error(err, err_cap, "jsonrpc-methods.json did not parse as JSON");
    goto done;
  }
  TfJsonSpan doc = {start, end};
  TfJsonSpan methods;
  if (!tf_json_object_get_object(doc, "methods", &methods)) {
    set_error(err, err_cap, "jsonrpc-methods.json missing methods object");
    goto done;
  }

  char keys[TF_MAX_METHODS][96];
  size_t key_count = 0;
  if (!tf_json_collect_object_keys(methods, keys, TF_MAX_METHODS, &key_count)) {
    set_error(err, err_cap, "failed to collect JSON-RPC method names");
    goto done;
  }
  if (key_count == 0) {
    set_error(err, err_cap, "jsonrpc-methods.json has no methods");
    goto done;
  }

  schemas->method_count = key_count;
  for (size_t i = 0; i < key_count; i++) {
    tf_copy_cstr(schemas->method_name_storage[i], sizeof(schemas->method_name_storage[i]), keys[i]);
    schemas->methods[i].name = schemas->method_name_storage[i];
    schemas->methods[i].trigger_count = 0;
    schemas->methods[i].query_only = true;
    for (size_t j = 0; j < TF_MAX_METHOD_TRIGGERS; j++) {
      schemas->methods[i].triggers[j] = 0;
    }
    TfJsonSpan method_schema;
    TfJsonSpan params_schema;
    if (!tf_json_object_get_object(methods, keys[i], &method_schema) ||
        !tf_json_object_get_object(method_schema, "params", &params_schema)) {
      snprintf(err, err_cap, "method %s missing params schema", keys[i]);
      goto done;
    }
    if (!compile_param_schema(&schemas->methods[i].params_schema, params_schema, err, err_cap)) {
      goto done;
    }
  }
  ok = true;

done:
  free(json);
  return ok;
}

static bool add_transition(
  TfSchemas *schemas,
  const char *from,
  const char *trigger,
  const char *to,
  char *err,
  size_t err_cap
) {
  if (schemas->transition_count >= TF_MAX_TRANSITIONS) {
    set_error(err, err_cap, "too many state-machine transitions");
    return false;
  }
  size_t i = schemas->transition_count;
  tf_copy_cstr(schemas->transition_from_storage[i], sizeof(schemas->transition_from_storage[i]), from);
  tf_copy_cstr(
    schemas->transition_trigger_storage[i],
    sizeof(schemas->transition_trigger_storage[i]),
    trigger
  );
  tf_copy_cstr(schemas->transition_to_storage[i], sizeof(schemas->transition_to_storage[i]), to);
  schemas->transitions[i].from = schemas->transition_from_storage[i];
  schemas->transitions[i].trigger = schemas->transition_trigger_storage[i];
  schemas->transitions[i].to = schemas->transition_to_storage[i];
  schemas->transition_count++;
  return true;
}

static bool add_state(TfSchemas *schemas, const char *state, char *err, size_t err_cap) {
  if (schemas->state_count >= TF_MAX_STATES) {
    set_error(err, err_cap, "too many state-machine states");
    return false;
  }
  tf_copy_cstr(schemas->states[schemas->state_count], sizeof(schemas->states[schemas->state_count]), state);
  schemas->state_count++;
  return true;
}

static bool add_terminal_state(TfSchemas *schemas, const char *state, char *err, size_t err_cap) {
  if (schemas->terminal_state_count >= TF_MAX_TERMINAL_STATES) {
    set_error(err, err_cap, "too many terminal states");
    return false;
  }
  tf_copy_cstr(
    schemas->terminal_states[schemas->terminal_state_count],
    sizeof(schemas->terminal_states[schemas->terminal_state_count]),
    state
  );
  schemas->terminal_state_count++;
  return true;
}

static bool parse_yaml_list_item(char *line, char **value) {
  char *s = trim(line);
  if (!starts_with(s, "- ")) {
    return false;
  }
  *value = trim(s + 2);
  return **value != '\0';
}

static char *trim_inline_value(char *value) {
  value = trim(value);
  char *end = value;
  while (*end && *end != ',' && *end != '}') {
    end++;
  }
  *end = '\0';
  return trim(value);
}

static bool parse_trigger_list(
  TfSchemas *schemas,
  const char *method_name,
  char *list,
  char *err,
  size_t err_cap
) {
  TfMethodSpec *method = find_method_mut(schemas, method_name);
  if (method == 0) {
    snprintf(err, err_cap, "state-machine method_triggers contains unknown method %s", method_name);
    return false;
  }
  size_t index = method_index(schemas, method);
  method->trigger_count = 0;
  method->query_only = true;

  char *token = trim(list);
  if (*token == '\0') {
    return true;
  }
  while (*token) {
    char *comma = strchr(token, ',');
    if (comma != 0) {
      *comma = '\0';
    }
    char *value = trim(token);
    if (*value != '\0') {
      if (method->trigger_count >= TF_MAX_METHOD_TRIGGERS) {
        snprintf(err, err_cap, "too many triggers for method %s", method_name);
        return false;
      }
      size_t trigger_i = method->trigger_count;
      tf_copy_cstr(
        schemas->trigger_storage[index][trigger_i],
        sizeof(schemas->trigger_storage[index][trigger_i]),
        value
      );
      method->triggers[trigger_i] = schemas->trigger_storage[index][trigger_i];
      method->trigger_count++;
      method->query_only = false;
    }
    if (comma == 0) {
      break;
    }
    token = comma + 1;
  }
  return true;
}

static bool parse_method_trigger_line(TfSchemas *schemas, char *line, char *err, size_t err_cap) {
  char *colon = strchr(line, ':');
  if (colon == 0) {
    return true;
  }
  *colon = '\0';
  char *method_name = trim(line);
  char *open = strchr(colon + 1, '[');
  char *close = strrchr(colon + 1, ']');
  if (open == 0 || close == 0 || close < open) {
    snprintf(err, err_cap, "invalid method_triggers line for %s", method_name);
    return false;
  }
  *close = '\0';
  return parse_trigger_list(schemas, method_name, open + 1, err, err_cap);
}

static bool set_rubric_weight(TfRubric *rubric, const char *key, double value) {
  if (strcmp(key, "parse") == 0) {
    rubric->parse = value;
  } else if (strcmp(key, "schema") == 0) {
    rubric->schema = value;
  } else if (strcmp(key, "method_known") == 0) {
    rubric->method_known = value;
  } else if (strcmp(key, "precondition_met") == 0) {
    rubric->precondition_met = value;
  } else if (strcmp(key, "transition_valid") == 0) {
    rubric->transition_valid = value;
  } else if (strcmp(key, "sequence_optimal") == 0) {
    rubric->sequence_optimal = value;
  } else {
    return false;
  }
  return true;
}

static bool parse_weight_line(TfRubric *rubric, char *line, char *err, size_t err_cap) {
  char *colon = strchr(line, ':');
  if (colon == 0) {
    return true;
  }
  *colon = '\0';
  char *key = trim(line);
  char *value_text = trim(colon + 1);
  if (*value_text == '\0') {
    return true;
  }
  char *end = 0;
  double value = strtod(value_text, &end);
  if (end == value_text || *trim(end) != '\0') {
    snprintf(err, err_cap, "invalid rubric weight for %s", key);
    return false;
  }
  if (!set_rubric_weight(rubric, key, value)) {
    snprintf(err, err_cap, "unknown rubric subscore %s", key);
    return false;
  }
  return true;
}

static TfRubricPreset *add_rubric_preset(
  TfSchemas *schemas,
  const char *name,
  char *err,
  size_t err_cap
) {
  if (schemas->rubric_preset_count >= TF_MAX_RUBRIC_PRESETS) {
    set_error(err, err_cap, "too many reward rubric presets");
    return 0;
  }
  size_t i = schemas->rubric_preset_count++;
  tf_copy_cstr(
    schemas->rubric_preset_name_storage[i],
    sizeof(schemas->rubric_preset_name_storage[i]),
    name
  );
  schemas->rubric_presets[i].name = schemas->rubric_preset_name_storage[i];
  memset(&schemas->rubric_presets[i].weights, 0, sizeof(schemas->rubric_presets[i].weights));
  return &schemas->rubric_presets[i];
}

static bool load_reward_rubric_yaml(
  TfSchemas *schemas,
  const char *schemas_dir,
  char *err,
  size_t err_cap
) {
  char path[512];
  if (!tf_join_path(path, sizeof(path), schemas_dir, "reward-rubric.yaml")) {
    set_error(err, err_cap, "schemas_dir path is too long");
    return false;
  }

  char *yaml = tf_read_file(path, err, err_cap);
  if (yaml == 0) {
    return false;
  }

  enum { RUBRIC_NONE, RUBRIC_DEFAULT, RUBRIC_PRESETS } section = RUBRIC_NONE;
  TfRubricPreset *current_preset = 0;
  bool ok = true;

  for (char *line = yaml; line != 0 && *line;) {
    char *next = strchr(line, '\n');
    if (next != 0) {
      *next = '\0';
      next++;
    }
    char *comment = strchr(line, '#');
    if (comment != 0) {
      *comment = '\0';
    }
    size_t indent = 0;
    while (line[indent] == ' ') {
      indent++;
    }
    char *s = trim(line);
    if (*s == '\0') {
      line = next;
      continue;
    }

    if (strcmp(s, "default_weights:") == 0) {
      section = RUBRIC_DEFAULT;
      current_preset = 0;
    } else if (strcmp(s, "presets:") == 0) {
      section = RUBRIC_PRESETS;
      current_preset = 0;
    } else if (starts_with(s, "aggregation:")) {
      tf_copy_cstr(
        schemas->rubric_aggregation,
        sizeof(schemas->rubric_aggregation),
        trim(s + strlen("aggregation:"))
      );
    } else if (section == RUBRIC_DEFAULT) {
      ok = parse_weight_line(&schemas->rubric_default, s, err, err_cap);
    } else if (section == RUBRIC_PRESETS) {
      char *colon = strchr(s, ':');
      if (colon != 0 && trim(colon + 1)[0] == '\0' && indent <= 2) {
        *colon = '\0';
        current_preset = add_rubric_preset(schemas, trim(s), err, err_cap);
        ok = current_preset != 0;
      } else if (current_preset != 0) {
        ok = parse_weight_line(&current_preset->weights, s, err, err_cap);
      }
    }
    if (!ok) {
      break;
    }
    line = next;
  }

  if (ok && schemas->rubric_aggregation[0] == '\0') {
    set_error(err, err_cap, "reward-rubric.yaml missing aggregation");
    ok = false;
  }
  if (ok && schemas->rubric_preset_count == 0) {
    set_error(err, err_cap, "reward-rubric.yaml has no presets");
    ok = false;
  }
  free(yaml);
  return ok;
}

static bool load_trajectory_schema_yaml(
  TfSchemas *schemas,
  const char *schemas_dir,
  char *err,
  size_t err_cap
) {
  char path[512];
  if (!tf_join_path(path, sizeof(path), schemas_dir, "trajectory-schema.yaml")) {
    set_error(err, err_cap, "schemas_dir path is too long");
    return false;
  }

  char *yaml = tf_read_file(path, err, err_cap);
  if (yaml == 0) {
    return false;
  }

  bool ok = true;
  schemas->trajectory_schema_loaded = true;

  for (char *line = yaml; line != 0 && *line;) {
    char *next = strchr(line, '\n');
    if (next != 0) {
      *next = '\0';
      next++;
    }
    char *comment = strchr(line, '#');
    if (comment != 0) {
      *comment = '\0';
    }
    char *s = trim(line);
    if (*s == '\0') {
      line = next;
      continue;
    }
    char *enum_ref = strstr(s, "enum_ref:");
    if (enum_ref != 0) {
      tf_copy_cstr(
        schemas->trajectory_initial_state_enum_ref,
        sizeof(schemas->trajectory_initial_state_enum_ref),
        trim_inline_value(enum_ref + strlen("enum_ref:"))
      );
      if (strcmp(schemas->trajectory_initial_state_enum_ref, "state-machine.states") != 0) {
        snprintf(
          err,
          err_cap,
          "unsupported trajectory enum_ref %s",
          schemas->trajectory_initial_state_enum_ref
        );
        ok = false;
        break;
      }
      schemas->trajectory_initial_state_enum_ref_resolved = true;
      schemas->trajectory_initial_state_enum_count = schemas->state_count;
      for (size_t i = 0; i < schemas->state_count; i++) {
        tf_copy_cstr(
          schemas->trajectory_initial_state_enum_values[i],
          sizeof(schemas->trajectory_initial_state_enum_values[i]),
          schemas->states[i]
        );
      }
    }
    line = next;
  }

  if (ok && !schemas->trajectory_initial_state_enum_ref_resolved) {
    set_error(err, err_cap, "trajectory-schema.yaml missing state-machine enum_ref");
    ok = false;
  }
  free(yaml);
  return ok;
}

static bool load_state_machine_yaml(
  TfSchemas *schemas,
  const char *schemas_dir,
  char *err,
  size_t err_cap
) {
  char path[512];
  if (!tf_join_path(path, sizeof(path), schemas_dir, "state-machine.yaml")) {
    set_error(err, err_cap, "schemas_dir path is too long");
    return false;
  }

  char *yaml = tf_read_file(path, err, err_cap);
  if (yaml == 0) {
    return false;
  }

  enum {
    SECTION_NONE,
    SECTION_STATES,
    SECTION_TERMINAL_STATES,
    SECTION_TRANSITIONS,
    SECTION_METHOD_TRIGGERS,
  } section = SECTION_NONE;
  char current_from[96] = "";
  char current_trigger[96] = "";
  bool ok = true;

  for (char *line = yaml; line != 0 && *line;) {
    char *next = strchr(line, '\n');
    if (next != 0) {
      *next = '\0';
      next++;
    }
    char *comment = strchr(line, '#');
    if (comment != 0) {
      *comment = '\0';
    }
    char *s = trim(line);
    if (*s == '\0') {
      line = next;
      continue;
    }

    if (strcmp(s, "states:") == 0) {
      section = SECTION_STATES;
      line = next;
      continue;
    }
    if (starts_with(s, "initial_state:")) {
      tf_copy_cstr(
        schemas->initial_state,
        sizeof(schemas->initial_state),
        trim(s + strlen("initial_state:"))
      );
      section = SECTION_NONE;
      line = next;
      continue;
    }
    if (strcmp(s, "terminal_states:") == 0) {
      section = SECTION_TERMINAL_STATES;
      line = next;
      continue;
    }
    if (strcmp(s, "transitions:") == 0) {
      section = SECTION_TRANSITIONS;
      line = next;
      continue;
    }
    if (strcmp(s, "method_triggers:") == 0) {
      section = SECTION_METHOD_TRIGGERS;
      line = next;
      continue;
    }

    if (section == SECTION_STATES) {
      char *value = 0;
      if (parse_yaml_list_item(s, &value) && !add_state(schemas, value, err, err_cap)) {
        ok = false;
        break;
      }
    } else if (section == SECTION_TERMINAL_STATES) {
      char *value = 0;
      if (parse_yaml_list_item(s, &value) && !add_terminal_state(schemas, value, err, err_cap)) {
        ok = false;
        break;
      }
    } else if (section == SECTION_TRANSITIONS) {
      if (starts_with(s, "- from:")) {
        tf_copy_cstr(current_from, sizeof(current_from), trim(s + strlen("- from:")));
        current_trigger[0] = '\0';
      } else if (starts_with(s, "trigger:")) {
        tf_copy_cstr(current_trigger, sizeof(current_trigger), trim(s + strlen("trigger:")));
      } else if (starts_with(s, "to:")) {
        char *to = trim(s + strlen("to:"));
        if (current_from[0] == '\0' || current_trigger[0] == '\0' || *to == '\0') {
          set_error(err, err_cap, "incomplete transition in state-machine.yaml");
          ok = false;
          break;
        }
        if (!add_transition(schemas, current_from, current_trigger, to, err, err_cap)) {
          ok = false;
          break;
        }
      }
    } else if (section == SECTION_METHOD_TRIGGERS) {
      if (!parse_method_trigger_line(schemas, s, err, err_cap)) {
        ok = false;
        break;
      }
    }

    line = next;
  }

  if (ok && schemas->transition_count == 0) {
    set_error(err, err_cap, "state-machine.yaml has no transitions");
    ok = false;
  }
  if (ok && schemas->state_count == 0) {
    set_error(err, err_cap, "state-machine.yaml has no states");
    ok = false;
  }
  if (ok && schemas->initial_state[0] == '\0') {
    set_error(err, err_cap, "state-machine.yaml missing initial_state");
    ok = false;
  }
  free(yaml);
  return ok;
}

void tf_schemas_load_builtin(TfSchemas *schemas) {
  memset(schemas, 0, sizeof(*schemas));

  schemas->methods[0] = (TfMethodSpec){"ticket_open", {"ticket_open.accepted"}, 1, false, {0}};
  schemas->methods[1] = (TfMethodSpec){"ticket_get", {0}, 0, true, {0}};
  schemas->methods[2] = (TfMethodSpec){"ticket_status", {0}, 0, true, {0}};
  schemas->methods[3] =
    (TfMethodSpec){"ticket_reopen", {"ticket_reopen.issued"}, 1, false, {0}};
  schemas->methods[4] =
    (TfMethodSpec){"resolution_confirm", {"resolution_confirm.passed", "resolution_confirm.failed"}, 2, false, {0}};
  schemas->methods[5] = (TfMethodSpec){"ticket_history", {0}, 0, true, {0}};
  schemas->methods[6] = (TfMethodSpec){"admin_agent_add", {0}, 0, true, {0}};
  schemas->methods[7] =
    (TfMethodSpec){"admin_escalate", {"escalation.start", "escalation.resolved"}, 2, false, {0}};
  schemas->methods[8] = (TfMethodSpec){
    "admin_lifecycle_apply",
    {"lifecycle.close", "lifecycle.archive_window_elapsed"},
    2,
    false,
    {0},
  };
  schemas->method_count = 9;

  const char *states[] = {
    "NEW",
    "TRIAGED",
    "ASSIGNED",
    "IN_PROGRESS",
    "RESOLVED",
    "REOPENED",
    "ESCALATED",
    "ENGINEERING",
    "CLOSED",
    "ARCHIVED",
  };
  schemas->state_count = 10;
  for (size_t i = 0; i < schemas->state_count; i++) {
    tf_copy_cstr(schemas->states[i], sizeof(schemas->states[i]), states[i]);
    tf_copy_cstr(
      schemas->trajectory_initial_state_enum_values[i],
      sizeof(schemas->trajectory_initial_state_enum_values[i]),
      states[i]
    );
  }
  tf_copy_cstr(schemas->initial_state, sizeof(schemas->initial_state), "NEW");
  tf_copy_cstr(schemas->terminal_states[0], sizeof(schemas->terminal_states[0]), "ARCHIVED");
  schemas->terminal_state_count = 1;
  tf_copy_cstr(
    schemas->trajectory_initial_state_enum_ref,
    sizeof(schemas->trajectory_initial_state_enum_ref),
    "state-machine.states"
  );
  schemas->trajectory_schema_loaded = true;
  schemas->trajectory_initial_state_enum_ref_resolved = true;
  schemas->trajectory_initial_state_enum_count = schemas->state_count;

  schemas->transitions[0] = (TfTransition){"NEW", "ticket_open.accepted", "TRIAGED"};
  schemas->transitions[1] = (TfTransition){"TRIAGED", "routing.assigned", "ASSIGNED"};
  schemas->transitions[2] = (TfTransition){"ASSIGNED", "agent.acknowledged", "IN_PROGRESS"};
  schemas->transitions[3] = (TfTransition){"IN_PROGRESS", "resolution_confirm.passed", "RESOLVED"};
  schemas->transitions[4] = (TfTransition){"RESOLVED", "ticket_reopen.issued", "REOPENED"};
  schemas->transitions[5] = (TfTransition){"REOPENED", "resolution_confirm.passed", "RESOLVED"};
  schemas->transitions[6] = (TfTransition){"REOPENED", "resolution_confirm.failed", "ESCALATED"};
  schemas->transitions[7] = (TfTransition){"ESCALATED", "escalation.start", "ENGINEERING"};
  schemas->transitions[8] = (TfTransition){"ENGINEERING", "escalation.resolved", "RESOLVED"};
  schemas->transitions[9] = (TfTransition){"RESOLVED", "lifecycle.close", "CLOSED"};
  schemas->transitions[10] =
    (TfTransition){"CLOSED", "lifecycle.archive_window_elapsed", "ARCHIVED"};
  schemas->transition_count = 11;
  schemas->rubric_default = (TfRubric){
    .parse = 0.10,
    .schema = 0.20,
    .method_known = 0.10,
    .precondition_met = 0.20,
    .transition_valid = 0.30,
    .sequence_optimal = 0.10,
  };
  tf_copy_cstr(schemas->rubric_aggregation, sizeof(schemas->rubric_aggregation), "mean");
  schemas->rubric_presets[0] = (TfRubricPreset){"shaped", schemas->rubric_default};
  schemas->rubric_presets[1] = (TfRubricPreset){"binary", {.transition_valid = 1.0}};
  schemas->rubric_presets[2] = (TfRubricPreset){"schema_only", {.parse = 0.5, .schema = 0.5}};
  schemas->rubric_preset_count = 3;
}

bool tf_schemas_load_dir(TfSchemas *schemas, const char *schemas_dir, char *err, size_t err_cap) {
  memset(schemas, 0, sizeof(*schemas));
  if (err_cap > 0) {
    err[0] = '\0';
  }
  if (schemas_dir == 0 || schemas_dir[0] == '\0') {
    set_error(err, err_cap, "schemas_dir is required");
    return false;
  }
  if (!load_methods_json(schemas, schemas_dir, err, err_cap)) {
    memset(schemas, 0, sizeof(*schemas));
    return false;
  }
  if (!load_state_machine_yaml(schemas, schemas_dir, err, err_cap)) {
    memset(schemas, 0, sizeof(*schemas));
    return false;
  }
  if (!load_reward_rubric_yaml(schemas, schemas_dir, err, err_cap)) {
    memset(schemas, 0, sizeof(*schemas));
    return false;
  }
  if (!load_trajectory_schema_yaml(schemas, schemas_dir, err, err_cap)) {
    memset(schemas, 0, sizeof(*schemas));
    return false;
  }
  return true;
}

const TfMethodSpec *tf_schemas_find_method(const TfSchemas *schemas, const char *method) {
  for (size_t i = 0; i < schemas->method_count; i++) {
    if (strcmp(schemas->methods[i].name, method) == 0) {
      return &schemas->methods[i];
    }
  }
  return 0;
}

const TfRubricPreset *tf_schemas_find_rubric_preset(
  const TfSchemas *schemas,
  const char *preset
) {
  for (size_t i = 0; i < schemas->rubric_preset_count; i++) {
    if (strcmp(schemas->rubric_presets[i].name, preset) == 0) {
      return &schemas->rubric_presets[i];
    }
  }
  return 0;
}

bool tf_schemas_state_allowed(const TfSchemas *schemas, const char *state) {
  for (size_t i = 0; i < schemas->state_count; i++) {
    if (strcmp(schemas->states[i], state) == 0) {
      return true;
    }
  }
  return false;
}

const char *tf_schemas_transition_to(
  const TfSchemas *schemas,
  const char *from,
  const char *trigger
) {
  for (size_t i = 0; i < schemas->transition_count; i++) {
    if (strcmp(schemas->transitions[i].from, from) == 0 &&
        strcmp(schemas->transitions[i].trigger, trigger) == 0) {
      return schemas->transitions[i].to;
    }
  }
  return 0;
}

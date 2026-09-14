#include "toyforge/jsonschema.h"

#include "toyforge/io.h"
#include "toyforge/regex.h"
#include "toyforge/schemas.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const TfSchemaProperty *find_property(const TfParamSchema *schema, const char *name) {
  for (size_t i = 0; i < schema->property_count; i++) {
    if (strcmp(schema->properties[i].name, name) == 0) {
      return &schema->properties[i];
    }
  }
  return 0;
}

static const TfSchemaChildProperty *find_child_property(
  const TfSchemaProperty *schema,
  const char *name
) {
  for (size_t i = 0; i < schema->child_property_count; i++) {
    if (strcmp(schema->child_properties[i].name, name) == 0) {
      return &schema->child_properties[i];
    }
  }
  return 0;
}

static const char *skip_ws_until(const char *p, const char *end) {
  while (p < end && isspace((unsigned char)*p)) {
    p++;
  }
  return p;
}

static bool span_has_json_start(TfJsonSpan span, char start) {
  const char *p = tf_json_skip_ws(span.start);
  if (p >= span.end || *p != start) {
    return false;
  }
  if (!tf_json_skip_value(&p)) {
    return false;
  }
  return p == span.end;
}

static bool span_is_string(TfJsonSpan span) {
  size_t decoded_size = 0;
  return tf_json_string_decoded_size(span, &decoded_size);
}

static bool span_is_nonempty_string(TfJsonSpan span) {
  size_t decoded_size = 0;
  return tf_json_string_decoded_size(span, &decoded_size) && decoded_size > 0;
}

static bool validate_string_constraints(
  const char *name,
  const char *value,
  bool has_min_length,
  size_t min_length,
  bool has_prefix,
  const char *prefix,
  char *err,
  size_t err_cap
) {
  if (has_min_length && strlen(value) < min_length) {
    snprintf(err, err_cap, "%s must be at least %zu characters", name, min_length);
    return false;
  }
  if (has_prefix && strncmp(value, prefix, strlen(prefix)) != 0) {
    snprintf(err, err_cap, "%s must match prefix %s", name, prefix);
    return false;
  }
  return true;
}

static bool require_json_value(
  TfJsonSpan object,
  const char *key,
  TfJsonSpan *value,
  char *err,
  size_t err_cap
) {
  if (!tf_json_object_get_value(object, key, value)) {
    snprintf(err, err_cap, "%s is required", key);
    return false;
  }
  return true;
}

static bool require_string_value(
  TfJsonSpan object,
  const char *key,
  char *err,
  size_t err_cap
) {
  TfJsonSpan value;
  if (!require_json_value(object, key, &value, err, err_cap)) {
    return false;
  }
  if (!span_is_string(value)) {
    snprintf(err, err_cap, "%s must be a string", key);
    return false;
  }
  return true;
}

static bool require_nonempty_string_value(
  TfJsonSpan object,
  const char *key,
  char *err,
  size_t err_cap
) {
  TfJsonSpan value;
  if (!require_json_value(object, key, &value, err, err_cap)) {
    return false;
  }
  if (!span_is_nonempty_string(value)) {
    snprintf(err, err_cap, "%s must be a non-empty string", key);
    return false;
  }
  return true;
}

static bool optional_string_value(
  TfJsonSpan object,
  const char *key,
  char *err,
  size_t err_cap
) {
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value)) {
    return true;
  }
  if (!span_is_string(value)) {
    snprintf(err, err_cap, "%s must be a string", key);
    return false;
  }
  return true;
}

static bool require_object_value(
  TfJsonSpan object,
  const char *key,
  TfJsonSpan *value,
  char *err,
  size_t err_cap
) {
  if (!require_json_value(object, key, value, err, err_cap)) {
    return false;
  }
  if (!span_has_json_start(*value, '{')) {
    snprintf(err, err_cap, "%s must be an object", key);
    return false;
  }
  return true;
}

static bool require_array_value(
  TfJsonSpan object,
  const char *key,
  TfJsonSpan *value,
  char *err,
  size_t err_cap
) {
  if (!require_json_value(object, key, value, err, err_cap)) {
    return false;
  }
  if (!span_has_json_start(*value, '[')) {
    snprintf(err, err_cap, "%s must be an array", key);
    return false;
  }
  return true;
}

static bool string_in_const_list(const char *value, const char *const *values, size_t value_count) {
  for (size_t i = 0; i < value_count; i++) {
    if (strcmp(value, values[i]) == 0) {
      return true;
    }
  }
  return false;
}

static bool string_in_schema_states(const TfSchemas *schemas, const char *value) {
  for (size_t i = 0; i < schemas->trajectory_initial_state_enum_count; i++) {
    if (strcmp(value, schemas->trajectory_initial_state_enum_values[i]) == 0) {
      return true;
    }
  }
  return false;
}

static bool optional_enum_value(
  TfJsonSpan object,
  const char *key,
  const char *const *allowed,
  size_t allowed_count,
  char *err,
  size_t err_cap
) {
  if (!tf_json_object_has_key(object, key)) {
    return true;
  }
  char value[96];
  if (!tf_json_object_get_string(object, key, value, sizeof(value))) {
    snprintf(err, err_cap, "%s must be a string", key);
    return false;
  }
  if (!string_in_const_list(value, allowed, allowed_count)) {
    snprintf(err, err_cap, "%s has unsupported value %s", key, value);
    return false;
  }
  return true;
}

typedef bool (*TfArrayItemValidator)(
  TfJsonSpan item,
  size_t index,
  void *ctx,
  char *err,
  size_t err_cap
);

static bool validate_array_items(
  TfJsonSpan array,
  const char *name,
  size_t min_items,
  TfArrayItemValidator validate_item,
  void *ctx,
  char *err,
  size_t err_cap
) {
  const char *p = skip_ws_until(array.start, array.end);
  if (p >= array.end || *p != '[') {
    snprintf(err, err_cap, "%s must be an array", name);
    return false;
  }
  p++;
  p = skip_ws_until(p, array.end);
  size_t count = 0;
  if (p < array.end && *p == ']') {
    p++;
    if (skip_ws_until(p, array.end) != array.end) {
      snprintf(err, err_cap, "%s is malformed", name);
      return false;
    }
    if (count < min_items) {
      snprintf(err, err_cap, "%s must contain at least %zu item", name, min_items);
      return false;
    }
    return true;
  }

  while (p < array.end) {
    const char *item_start = skip_ws_until(p, array.end);
    const char *item_end = item_start;
    if (!tf_json_skip_value(&item_end) || item_end > array.end) {
      snprintf(err, err_cap, "%s contains invalid JSON", name);
      return false;
    }
    TfJsonSpan item = {item_start, item_end};
    if (validate_item != 0 && !validate_item(item, count, ctx, err, err_cap)) {
      return false;
    }
    count++;
    p = skip_ws_until(item_end, array.end);
    if (p >= array.end) {
      snprintf(err, err_cap, "%s is malformed", name);
      return false;
    }
    if (*p == ',') {
      p++;
      p = skip_ws_until(p, array.end);
      continue;
    }
    if (*p == ']') {
      p++;
      if (skip_ws_until(p, array.end) != array.end) {
        snprintf(err, err_cap, "%s is malformed", name);
        return false;
      }
      if (count < min_items) {
        snprintf(err, err_cap, "%s must contain at least %zu item", name, min_items);
        return false;
      }
      return true;
    }
    snprintf(err, err_cap, "%s is malformed", name);
    return false;
  }

  snprintf(err, err_cap, "%s is malformed", name);
  return false;
}

static bool validate_child_object(
  const TfSchemaProperty *prop,
  TfJsonSpan object,
  char *err,
  size_t err_cap
) {
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t key_count = 0;
  if (!tf_json_collect_object_keys(object, keys, TF_MAX_SCHEMA_PROPERTIES, &key_count)) {
    snprintf(err, err_cap, "%s is not a JSON object", prop->name);
    return false;
  }
  if (!prop->additional_properties) {
    for (size_t i = 0; i < key_count; i++) {
      if (find_child_property(prop, keys[i]) == 0) {
        snprintf(err, err_cap, "additional property is not allowed: %s", keys[i]);
        return false;
      }
    }
  }
  for (size_t i = 0; i < prop->child_property_count; i++) {
    const TfSchemaChildProperty *child = &prop->child_properties[i];
    if (!tf_json_object_has_key(object, child->name)) {
      if (child->required) {
        snprintf(err, err_cap, "%s is required", child->name);
        return false;
      }
      continue;
    }
    if (child->type == TF_SCHEMA_VALUE_STRING) {
      char value[TF_MAX_TEXT];
      if (!tf_json_object_get_string(object, child->name, value, sizeof(value))) {
        snprintf(err, err_cap, "%s must be a string", child->name);
        return false;
      }
      if (!validate_string_constraints(
            child->name,
            value,
            child->has_min_length,
            child->min_length,
            child->has_prefix,
            child->prefix,
            err,
            err_cap
          )) {
        return false;
      }
    } else {
      snprintf(err, err_cap, "unsupported nested schema type for %s", child->name);
      return false;
    }
  }
  return true;
}

static bool validate_object_schema(
  const TfParamSchema *schema,
  TfJsonSpan object,
  char *err,
  size_t err_cap
) {
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t key_count = 0;
  if (!tf_json_collect_object_keys(object, keys, TF_MAX_SCHEMA_PROPERTIES, &key_count)) {
    tf_copy_cstr(err, err_cap, "params is not a JSON object");
    return false;
  }
  if (!schema->additional_properties) {
    for (size_t i = 0; i < key_count; i++) {
      if (find_property(schema, keys[i]) == 0) {
        snprintf(err, err_cap, "additional property is not allowed: %s", keys[i]);
        return false;
      }
    }
  }

  for (size_t i = 0; i < schema->property_count; i++) {
    const TfSchemaProperty *prop = &schema->properties[i];
    if (!tf_json_object_has_key(object, prop->name)) {
      if (prop->required) {
        snprintf(err, err_cap, "%s is required", prop->name);
        return false;
      }
      continue;
    }

    if (prop->type == TF_SCHEMA_VALUE_STRING) {
      char value[TF_MAX_TEXT];
      if (!tf_json_object_get_string(object, prop->name, value, sizeof(value))) {
        snprintf(err, err_cap, "%s is required and must be a string", prop->name);
        return false;
      }
      if (!validate_string_constraints(
            prop->name,
            value,
            prop->has_min_length,
            prop->min_length,
            prop->has_prefix,
            prop->prefix,
            err,
            err_cap
          )) {
        return false;
      }
    } else if (prop->type == TF_SCHEMA_VALUE_OBJECT) {
      TfJsonSpan child;
      if (!tf_json_object_get_object(object, prop->name, &child)) {
        snprintf(err, err_cap, "%s is required and must be an object", prop->name);
        return false;
      }
      if (!validate_child_object(prop, child, err, err_cap)) {
        return false;
      }
    } else {
      snprintf(err, err_cap, "unsupported schema type for %s", prop->name);
      return false;
    }
  }
  return true;
}

bool tf_jsonschema_validate_method_params(
  const TfSchemas *schemas,
  const char *method,
  TfJsonSpan params,
  char *err,
  size_t err_cap
) {
  if (err_cap > 0) {
    err[0] = '\0';
  }
  const TfMethodSpec *spec = tf_schemas_find_method(schemas, method);
  if (spec == 0) {
    tf_copy_cstr(err, err_cap, "unknown method schema");
    return false;
  }
  if (!spec->params_schema.loaded) {
    tf_copy_cstr(err, err_cap, "method params schema was not loaded");
    return false;
  }
  return validate_object_schema(&spec->params_schema, params, err, err_cap);
}

static bool validate_object_array_item(
  TfJsonSpan item,
  size_t index,
  void *ctx,
  char *err,
  size_t err_cap
) {
  (void)ctx;
  if (!span_has_json_start(item, '{')) {
    snprintf(err, err_cap, "array item %zu must be an object", index);
    return false;
  }
  return true;
}

static bool validate_prompt_context(TfJsonSpan prompt_context, char *err, size_t err_cap) {
  if (!require_string_value(prompt_context, "prior_state", err, err_cap)) {
    return false;
  }
  if (!optional_string_value(prompt_context, "situation", err, err_cap)) {
    return false;
  }
  TfJsonSpan prior_calls;
  if (tf_json_object_get_value(prompt_context, "prior_calls", &prior_calls)) {
    if (!span_has_json_start(prior_calls, '[')) {
      tf_copy_cstr(err, err_cap, "prior_calls must be an array");
      return false;
    }
    if (!validate_array_items(
          prior_calls,
          "prior_calls",
          0,
          validate_object_array_item,
          0,
          err,
          err_cap
        )) {
      return false;
    }
  }
  return true;
}

static bool validate_trajectory_step(
  TfJsonSpan step,
  size_t index,
  void *ctx,
  char *err,
  size_t err_cap
) {
  (void)ctx;
  if (!span_has_json_start(step, '{')) {
    snprintf(err, err_cap, "steps[%zu] must be an object", index);
    return false;
  }

  TfJsonSpan prompt_context;
  if (!require_object_value(step, "prompt_context", &prompt_context, err, err_cap)) {
    return false;
  }
  if (!validate_prompt_context(prompt_context, err, err_cap)) {
    return false;
  }
  if (!require_nonempty_string_value(step, "thinking", err, err_cap)) {
    return false;
  }

  TfJsonSpan tool_call;
  if (!require_object_value(step, "tool_call", &tool_call, err, err_cap)) {
    return false;
  }
  (void)tool_call;
  if (!optional_string_value(step, "expected_state_after", err, err_cap)) {
    return false;
  }
  if (!optional_string_value(step, "expected_trigger", err, err_cap)) {
    return false;
  }
  return true;
}

bool tf_jsonschema_validate_trajectory(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  char *err,
  size_t err_cap
) {
  static const char *difficulty_values[] = {"easy", "medium", "hard"};
  static const char *source_values[] = {"hand_seed", "teacher_expansion"};

  if (err_cap > 0) {
    err[0] = '\0';
  }
  if (schemas == 0 || !schemas->trajectory_schema_loaded) {
    tf_copy_cstr(err, err_cap, "trajectory schema was not loaded");
    return false;
  }
  if (!schemas->trajectory_initial_state_enum_ref_resolved ||
      schemas->trajectory_initial_state_enum_count == 0) {
    tf_copy_cstr(err, err_cap, "trajectory initial_state enum_ref was not resolved");
    return false;
  }
  if (!span_has_json_start(trajectory, '{')) {
    tf_copy_cstr(err, err_cap, "trajectory must be a JSON object");
    return false;
  }

  if (!require_string_value(trajectory, "trajectory_id", err, err_cap)) {
    return false;
  }

  char initial_state[96];
  if (!tf_json_object_get_string(trajectory, "initial_state", initial_state, sizeof(initial_state))) {
    tf_copy_cstr(err, err_cap, "initial_state is required and must be a string");
    return false;
  }
  if (!string_in_schema_states(schemas, initial_state)) {
    snprintf(err, err_cap, "initial_state has unsupported value %s", initial_state);
    return false;
  }

  TfJsonSpan steps;
  if (!require_array_value(trajectory, "steps", &steps, err, err_cap)) {
    return false;
  }
  if (!validate_array_items(
        steps,
        "steps",
        1,
        validate_trajectory_step,
        0,
        err,
        err_cap
      )) {
    return false;
  }

  if (!optional_string_value(trajectory, "description", err, err_cap) ||
      !optional_string_value(trajectory, "final_state", err, err_cap) ||
      !optional_string_value(trajectory, "final_answer", err, err_cap) ||
      !optional_string_value(trajectory, "teacher_id", err, err_cap)) {
    return false;
  }
  if (!optional_enum_value(
        trajectory,
        "difficulty",
        difficulty_values,
        sizeof(difficulty_values) / sizeof(difficulty_values[0]),
        err,
        err_cap
      )) {
    return false;
  }
  if (!optional_enum_value(
        trajectory,
        "source",
        source_values,
        sizeof(source_values) / sizeof(source_values[0]),
        err,
        err_cap
      )) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// General Draft 2020-12 (subset) validator: schema document vs. instance.
// ---------------------------------------------------------------------------

static bool single_type_match(const char *name, TfJsonType it, TfJsonSpan instance) {
  if (strcmp(name, "null") == 0) {
    return it == TF_JSON_NULL;
  }
  if (strcmp(name, "boolean") == 0) {
    return it == TF_JSON_BOOL;
  }
  if (strcmp(name, "object") == 0) {
    return it == TF_JSON_OBJECT;
  }
  if (strcmp(name, "array") == 0) {
    return it == TF_JSON_ARRAY;
  }
  if (strcmp(name, "string") == 0) {
    return it == TF_JSON_STRING;
  }
  if (strcmp(name, "number") == 0) {
    return it == TF_JSON_NUMBER;
  }
  if (strcmp(name, "integer") == 0) {
    return it == TF_JSON_NUMBER && tf_json_is_integer(instance);
  }
  return false;
}

static bool type_matches(TfJsonSpan type_val, TfJsonType it, TfJsonSpan instance) {
  TfJsonType tt = tf_json_type(type_val);
  if (tt == TF_JSON_STRING) {
    char name[32];
    const char *p = tf_json_skip_ws(type_val.start);
    if (!tf_json_read_string(&p, name, sizeof(name))) {
      return false;
    }
    return single_type_match(name, it, instance);
  }
  if (tt == TF_JSON_ARRAY) {
    size_t n = 0;
    if (!tf_json_array_count(type_val, &n)) {
      return false;
    }
    for (size_t i = 0; i < n; i++) {
      TfJsonSpan elem;
      char name[32];
      const char *p;
      if (!tf_json_array_get(type_val, i, &elem)) {
        continue;
      }
      p = tf_json_skip_ws(elem.start);
      if (tf_json_read_string(&p, name, sizeof(name)) && single_type_match(name, it, instance)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

// Decoded UTF-8 code-point count of a JSON string, or (size_t)-1 on error.
static size_t string_codepoints(TfJsonSpan str) {
  size_t bytelen = 0;
  if (!tf_json_string_decoded_size(str, &bytelen)) {
    return (size_t)-1;
  }
  char *buf = (char *)malloc(bytelen + 1);
  if (buf == 0) {
    return (size_t)-1;
  }
  const char *p = tf_json_skip_ws(str.start);
  if (!tf_json_read_string(&p, buf, bytelen + 1)) {
    free(buf);
    return (size_t)-1;
  }
  size_t cps = 0;
  for (size_t i = 0; i < bytelen; i++) {
    if (((unsigned char)buf[i] & 0xC0) != 0x80) {
      cps++;
    }
  }
  free(buf);
  return cps;
}

static bool get_number_keyword(TfJsonSpan schema, const char *kw, double *out) {
  TfJsonSpan v;
  if (!tf_json_object_get_value(schema, kw, &v)) {
    return false;
  }
  return tf_json_number(v, out);
}

// Decode a JSON string value into a freshly allocated NUL-terminated buffer
// (caller frees); *len_out gets the decoded byte length. Returns NULL on error.
static char *decode_string_alloc(TfJsonSpan str, size_t *len_out) {
  size_t bytelen = 0;
  if (!tf_json_string_decoded_size(str, &bytelen)) {
    return 0;
  }
  char *buf = (char *)malloc(bytelen + 1);
  if (buf == 0) {
    return 0;
  }
  const char *p = tf_json_skip_ws(str.start);
  if (!tf_json_read_string(&p, buf, bytelen + 1)) {
    free(buf);
    return 0;
  }
  if (len_out != 0) {
    *len_out = bytelen;
  }
  return buf;
}

#define TF_SCHEMA_MAX_DEPTH 96
#define TF_URI_CAP 512
#define TF_ID_INDEX_CAP 256

// An index of in-document $id / $anchor reference targets, keyed by absolute
// URI. `parent_base` is the base URI in scope *around* the node (before applying
// the node's own $id), so a bare-$id $ref can re-enter the node and apply its
// $id exactly once.
typedef struct {
  char uri[TF_URI_CAP];
  char parent_base[TF_URI_CAP];
  TfJsonSpan span;
  bool is_dynamic;  // true if registered from a $dynamicAnchor (vs a plain $anchor/$id)
} TfIdEntry;

typedef struct {
  TfIdEntry entries[TF_ID_INDEX_CAP];
  size_t count;
  bool overflow;  // set if a reference target was dropped (count exceeded the cap)
} TfIdIndex;

// The dynamic scope: the ordered list of schema-resource base URIs entered along
// the current validation path (outermost first). $dynamicRef resolves to the
// outermost $dynamicAnchor of its name found across this scope.
#define TF_DYN_SCOPE_CAP 64
typedef struct {
  char bases[TF_DYN_SCOPE_CAP][TF_URI_CAP];
  size_t count;
} TfDynScope;

static void dyn_push(TfDynScope *d, const char *base) {
  if (d == 0 || d->count >= TF_DYN_SCOPE_CAP) {
    return;
  }
  tf_copy_cstr(d->bases[d->count], TF_URI_CAP, base);
  d->count++;
}

// Annotation results collected while validating one instance, used by
// unevaluatedProperties / unevaluatedItems. `props` are object member names
// evaluated by this schema and its in-place applicators; `item_seen[i]` marks
// array item i as evaluated; `all_items` means every item was (e.g. via
// `items`). Kept compact because instances of this live on the recursion stack.
#define TF_EVAL_MAX_PROPS 64
#define TF_EVAL_MAX_ITEMS 1024
typedef struct {
  char props[TF_EVAL_MAX_PROPS][96];
  size_t prop_count;
  unsigned char item_seen[TF_EVAL_MAX_ITEMS];
  bool all_items;
} TfEvalSet;

static void eval_reset(TfEvalSet *e) {
  e->prop_count = 0;
  memset(e->item_seen, 0, sizeof(e->item_seen));
  e->all_items = false;
}

static bool eval_has_prop(const TfEvalSet *e, const char *name) {
  for (size_t i = 0; i < e->prop_count; i++) {
    if (strcmp(e->props[i], name) == 0) {
      return true;
    }
  }
  return false;
}

static void eval_add_prop(TfEvalSet *e, const char *name) {
  if (e == 0 || eval_has_prop(e, name)) {
    return;
  }
  if (e->prop_count < TF_EVAL_MAX_PROPS) {
    tf_copy_cstr(e->props[e->prop_count], 96, name);
    e->prop_count++;
  }
}

static void eval_mark_item(TfEvalSet *e, size_t i) {
  if (e != 0 && i < TF_EVAL_MAX_ITEMS) {
    e->item_seen[i] = 1;
  }
}

static bool eval_item_seen(const TfEvalSet *e, size_t i) {
  return e->all_items || (i < TF_EVAL_MAX_ITEMS && e->item_seen[i]);
}

static void eval_merge(TfEvalSet *dst, const TfEvalSet *src) {
  if (dst == 0 || src == 0) {
    return;
  }
  for (size_t i = 0; i < src->prop_count; i++) {
    eval_add_prop(dst, src->props[i]);
  }
  if (src->all_items) {
    dst->all_items = true;
  }
  for (size_t i = 0; i < TF_EVAL_MAX_ITEMS; i++) {
    if (src->item_seen[i]) {
      dst->item_seen[i] = 1;
    }
  }
}

// Remove RFC 3986 "." and ".." path segments in place.
static void remove_dot_segments(char *path) {
  char out[TF_URI_CAP];
  size_t o = 0;
  const char *in = path;
  while (*in != '\0' && o + 1 < sizeof(out)) {
    if (strncmp(in, "../", 3) == 0) {
      in += 3;
    } else if (strncmp(in, "./", 2) == 0) {
      in += 2;
    } else if (strncmp(in, "/./", 3) == 0) {
      in += 2;  // leave the leading '/'
    } else if (strcmp(in, "/.") == 0) {
      out[o++] = '/';
      in += 2;
    } else if (strncmp(in, "/../", 4) == 0 || strcmp(in, "/..") == 0) {
      in += (in[3] == '/') ? 3 : 2;
      while (o > 0 && out[o - 1] != '/') {
        o--;
      }
      if (o > 0) {
        o--;  // drop the '/'
      }
    } else {
      do {
        out[o++] = *in++;
      } while (*in != '\0' && *in != '/' && o + 1 < sizeof(out));
    }
  }
  out[o] = '\0';
  tf_copy_cstr(path, TF_URI_CAP, out);
}

// Resolve a URI-reference `ref` against `base` per RFC 3986 §5.3 into `out`.
static void resolve_uri(const char *base, const char *ref, char *out, size_t out_cap) {
  // Detect an absolute ref (has a scheme: letters then ':' before any '/?#').
  bool ref_has_scheme = false;
  for (const char *p = ref; *p != '\0'; p++) {
    if (*p == ':') {
      ref_has_scheme = (p != ref);
      break;
    }
    if (!isalnum((unsigned char)*p) && *p != '+' && *p != '-' && *p != '.') {
      break;
    }
  }
  if (ref_has_scheme) {
    tf_copy_cstr(out, out_cap, ref);
    return;
  }
  if (ref[0] == '#' || ref[0] == '\0') {
    // Same-document reference: base minus its fragment, plus ref's fragment.
    char b[TF_URI_CAP];
    tf_copy_cstr(b, sizeof(b), base);
    char *hash = strchr(b, '#');
    if (hash != 0) {
      *hash = '\0';
    }
    snprintf(out, out_cap, "%s%s", b, ref);
    return;
  }

  // Split base into scheme:, authority, path (ignore base query/fragment).
  char b[TF_URI_CAP];
  tf_copy_cstr(b, sizeof(b), base);
  char *bfrag = strchr(b, '#');
  if (bfrag) {
    *bfrag = '\0';
  }
  char *bquery = strchr(b, '?');
  if (bquery) {
    *bquery = '\0';
  }
  // scheme = up to and including the first ':'.
  char prefix[TF_URI_CAP] = "";  // scheme + authority
  char *bpath = b;
  char *colon = strchr(b, ':');
  if (colon != 0 && colon[1] == '/' && colon[2] == '/') {
    // scheme://authority/path
    char *auth_start = colon + 3;
    char *path_start = strchr(auth_start, '/');
    if (path_start != 0) {
      size_t plen = (size_t)(path_start - b);
      memcpy(prefix, b, plen);
      prefix[plen] = '\0';
      bpath = path_start;
    } else {
      tf_copy_cstr(prefix, sizeof(prefix), b);
      bpath = (char *)"";
    }
  } else if (colon != 0) {
    // opaque (e.g. urn:...): scheme + the rest is "path".
    size_t plen = (size_t)(colon + 1 - b);
    memcpy(prefix, b, plen);
    prefix[plen] = '\0';
    bpath = colon + 1;
  }

  char merged[TF_URI_CAP * 2];
  char rpath[TF_URI_CAP];
  char rfrag[TF_URI_CAP] = "";
  tf_copy_cstr(rpath, sizeof(rpath), ref);
  char *rh = strchr(rpath, '#');
  if (rh) {
    tf_copy_cstr(rfrag, sizeof(rfrag), rh);  // includes '#'
    *rh = '\0';
  }
  if (rpath[0] == '/') {
    tf_copy_cstr(merged, sizeof(merged), rpath);
  } else {
    // Replace the last segment of bpath with rpath.
    char bp[TF_URI_CAP];
    tf_copy_cstr(bp, sizeof(bp), bpath);
    char *slash = strrchr(bp, '/');
    if (slash != 0) {
      slash[1] = '\0';
      snprintf(merged, sizeof(merged), "%s%s", bp, rpath);
    } else {
      tf_copy_cstr(merged, sizeof(merged), rpath);
    }
  }
  remove_dot_segments(merged);
  char full[TF_URI_CAP * 4];
  snprintf(full, sizeof(full), "%s%s%s", prefix, merged, rfrag);
  tf_copy_cstr(out, out_cap, full);
}

static void index_add(
  TfIdIndex *index, const char *uri, const char *parent_base, TfJsonSpan span, bool is_dynamic
) {
  if (index->count >= TF_ID_INDEX_CAP) {
    index->overflow = true;  // surfaced by the caller; never mis-validate silently
    return;
  }
  tf_copy_cstr(index->entries[index->count].uri, TF_URI_CAP, uri);
  tf_copy_cstr(index->entries[index->count].parent_base, TF_URI_CAP, parent_base);
  index->entries[index->count].span = span;
  index->entries[index->count].is_dynamic = is_dynamic;
  index->count++;
}

static const TfIdEntry *index_lookup(const TfIdIndex *index, const char *uri) {
  for (size_t i = 0; i < index->count; i++) {
    if (strcmp(index->entries[i].uri, uri) == 0) {
      return &index->entries[i];
    }
  }
  return 0;
}

// Recursively index every $id and $anchor under `node`, threading the base URI.
static void index_build(TfJsonSpan node, const char *base, TfIdIndex *index, int depth) {
  if (depth > TF_SCHEMA_MAX_DEPTH || tf_json_type(node) != TF_JSON_OBJECT) {
    if (tf_json_type(node) == TF_JSON_ARRAY) {
      size_t n = 0;
      if (tf_json_array_count(node, &n)) {
        for (size_t i = 0; i < n; i++) {
          TfJsonSpan e;
          if (tf_json_array_get(node, i, &e)) {
            index_build(e, base, index, depth + 1);
          }
        }
      }
    }
    return;
  }
  char cur_base[TF_URI_CAP];
  tf_copy_cstr(cur_base, sizeof(cur_base), base);
  char idbuf[TF_URI_CAP];
  if (tf_json_object_get_string(node, "$id", idbuf, sizeof(idbuf))) {
    char resolved[TF_URI_CAP];
    resolve_uri(base, idbuf, resolved, sizeof(resolved));
    char *frag = strchr(resolved, '#');  // a bare $id should have no fragment
    if (frag) {
      *frag = '\0';
    }
    tf_copy_cstr(cur_base, sizeof(cur_base), resolved);
    index_add(index, resolved, base, node, false);
  }
  char anchor[TF_URI_CAP];
  if (tf_json_object_get_string(node, "$anchor", anchor, sizeof(anchor))) {
    char key[TF_URI_CAP * 2];
    snprintf(key, sizeof(key), "%s#%s", cur_base, anchor);
    index_add(index, key, cur_base, node, false);
  }
  // $dynamicAnchor is keyed like $anchor but flagged dynamic, so $dynamicRef can
  // both bookend-check it and walk the dynamic scope for the outermost match.
  if (tf_json_object_get_string(node, "$dynamicAnchor", anchor, sizeof(anchor))) {
    char key[TF_URI_CAP * 2];
    snprintf(key, sizeof(key), "%s#%s", cur_base, anchor);
    index_add(index, key, cur_base, node, true);
  }
  // Recurse into every member value (any of them may hold subschemas).
  char keys[TF_MAX_SCHEMA_PROPERTIES][96];
  size_t nk = 0;
  if (tf_json_collect_object_keys(node, keys, TF_MAX_SCHEMA_PROPERTIES, &nk)) {
    for (size_t i = 0; i < nk; i++) {
      TfJsonSpan v;
      if (tf_json_object_get_value(node, keys[i], &v)) {
        index_build(v, cur_base, index, depth + 1);
      }
    }
  }
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

// Percent-decode a URI fragment ("%25" -> "%", "%22" -> '"') into `out`.
// `$ref` is a URI reference, so its fragment is percent-encoded before the
// JSON-Pointer ~0/~1 escaping is applied.
static void percent_decode(const char *src, char *out, size_t out_cap) {
  size_t o = 0;
  for (size_t i = 0; src[i] != '\0' && o + 1 < out_cap; i++) {
    if (src[i] == '%' && hexval(src[i + 1]) >= 0 && hexval(src[i + 2]) >= 0) {
      out[o++] = (char)((hexval(src[i + 1]) << 4) | hexval(src[i + 2]));
      i += 2;
    } else {
      out[o++] = src[i];
    }
  }
  out[o] = '\0';
}

// Resolve a local JSON Pointer (the part after '#', e.g. "/$defs/Foo") against
// `root`. An empty pointer resolves to root itself. The fragment is
// percent-decoded, then each token is ~0/~1 unescaped; array indices are
// numeric. Returns false for unresolvable pointers.
static bool resolve_json_pointer(TfJsonSpan root, const char *fragment, TfJsonSpan *out) {
  char decoded[512];
  percent_decode(fragment, decoded, sizeof(decoded));
  const char *pointer = decoded;
  TfJsonSpan cur = root;
  while (*pointer == '/') {
    pointer++;
    char token[128];
    size_t ti = 0;
    while (*pointer != '\0' && *pointer != '/' && ti + 1 < sizeof(token)) {
      char c = *pointer++;
      if (c == '~' && (*pointer == '0' || *pointer == '1')) {
        c = (*pointer == '1') ? '/' : '~';
        pointer++;
      }
      token[ti++] = c;
    }
    token[ti] = '\0';
    TfJsonType t = tf_json_type(cur);
    if (t == TF_JSON_OBJECT) {
      TfJsonSpan v;
      if (!tf_json_object_get_value(cur, token, &v)) {
        return false;
      }
      cur = v;
    } else if (t == TF_JSON_ARRAY) {
      char *end = 0;
      long idx = strtol(token, &end, 10);
      TfJsonSpan v;
      if (*end != '\0' || idx < 0 || !tf_json_array_get(cur, (size_t)idx, &v)) {
        return false;
      }
      cur = v;
    } else {
      return false;
    }
  }
  *out = cur;
  return true;
}

// `validate_root` is the entry wrapper: it applies a subschema's own $id to the
// base URI and pushes the resulting resource base onto the dynamic scope (with a
// balanced pop), then delegates to `validate_root_body` for the keyword work.
// Recursing through the wrapper keeps the dynamic scope correct for $dynamicRef.
static bool validate_root(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  TfJsonSpan schema,
  TfJsonSpan instance,
  TfEvalSet *eval,
  int depth,
  char *err,
  size_t err_cap
);
static bool validate_root_body(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  TfJsonSpan schema,
  TfJsonSpan instance,
  TfEvalSet *eval,
  int depth,
  char *err,
  size_t err_cap
);

// Resolve a $ref URI against `base`, locate the target (local JSON pointer,
// $anchor/$dynamicAnchor, or another $id'd subschema) via the index, and validate
// `instance` against it, populating `out_eval`. Unknown (remote) documents are
// pass-through annotations.
static bool resolve_ref_into(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  const char *ref,
  TfJsonSpan instance,
  TfEvalSet *out_eval,
  int depth,
  char *err,
  size_t err_cap
) {
  char target[TF_URI_CAP];
  resolve_uri(base, ref, target, sizeof(target));
  char doc[TF_URI_CAP];
  tf_copy_cstr(doc, sizeof(doc), target);
  const char *frag = "";
  char *hash = strchr(doc, '#');
  if (hash != 0) {
    frag = hash + 1;
    *hash = '\0';
  }
  TfJsonSpan doc_span = root;
  const char *doc_base = base;
  const char *parent_base_for_id = base;
  char doc_base_buf[TF_URI_CAP];
  char parent_base_buf[TF_URI_CAP];
  if (doc[0] != '\0') {
    const TfIdEntry *found = index_lookup(index, doc);
    if (found == 0) {
      return true;  // remote / unresolvable document -> annotation, no constraint
    }
    doc_span = found->span;
    tf_copy_cstr(doc_base_buf, sizeof(doc_base_buf), doc);
    doc_base = doc_base_buf;
    tf_copy_cstr(parent_base_buf, sizeof(parent_base_buf), found->parent_base);
    parent_base_for_id = parent_base_buf;
  }
  TfJsonSpan resolved;
  bool got = false;
  const char *effective_base = doc_base;
  if (frag[0] == '\0') {
    resolved = doc_span;
    got = true;
    effective_base = parent_base_for_id;
  } else if (frag[0] == '/') {
    got = resolve_json_pointer(doc_span, frag, &resolved);
  } else {
    char akey[TF_URI_CAP * 2];
    snprintf(akey, sizeof(akey), "%s#%s", doc, frag);
    const TfIdEntry *a = index_lookup(index, akey);
    if (a != 0) {
      resolved = a->span;
      got = true;
    }
  }
  if (!got) {
    snprintf(err, err_cap, "could not resolve ref %s", ref);
    return false;
  }
  return validate_root(
    root, dyn, effective_base, index, resolved, instance, out_eval, depth + 1, err, err_cap
  );
}

// Resolve a $dynamicRef. If its static target (resolving "#name" against `base`
// like a plain $ref) is a $dynamicAnchor, the reference is "bookended" and binds
// instead to the OUTERMOST $dynamicAnchor of that name across the dynamic scope.
// Otherwise it behaves exactly like $ref.
static bool resolve_dynamic_ref_into(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  const char *ref,
  TfJsonSpan instance,
  TfEvalSet *out_eval,
  int depth,
  char *err,
  size_t err_cap
) {
  const char *hash = strchr(ref, '#');
  // Only a plain-name fragment (#name) takes part in dynamic-scope resolution;
  // pointer fragments / bare URIs fall through to ordinary $ref handling.
  if (hash == 0 || hash[1] == '\0' || hash[1] == '/') {
    return resolve_ref_into(root, dyn, base, index, ref, instance, out_eval, depth, err, err_cap);
  }
  const char *name = hash + 1;
  char target[TF_URI_CAP];
  resolve_uri(base, ref, target, sizeof(target));
  const TfIdEntry *stat = index_lookup(index, target);
  if (stat == 0 || !stat->is_dynamic) {
    // Not bookended by a $dynamicAnchor -> ordinary $ref semantics.
    return resolve_ref_into(root, dyn, base, index, ref, instance, out_eval, depth, err, err_cap);
  }
  // Bookended: bind to the outermost scope entry that defines this $dynamicAnchor.
  for (size_t i = 0; i < dyn->count; i++) {
    char key[TF_URI_CAP * 2];
    snprintf(key, sizeof(key), "%s#%s", dyn->bases[i], name);
    const TfIdEntry *e = index_lookup(index, key);
    if (e != 0 && e->is_dynamic) {
      return validate_root(
        root, dyn, dyn->bases[i], index, e->span, instance, out_eval, depth + 1, err, err_cap
      );
    }
  }
  // No scope match (shouldn't happen once bookended) -> the static target.
  return validate_root(
    root, dyn, base, index, stat->span, instance, out_eval, depth + 1, err, err_cap
  );
}

static bool validate_root_body(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  TfJsonSpan schema,
  TfJsonSpan instance,
  TfEvalSet *eval,
  int depth,
  char *err,
  size_t err_cap
) {
  if (err_cap > 0) {
    err[0] = '\0';
  }
  if (depth > TF_SCHEMA_MAX_DEPTH) {
    tf_copy_cstr(err, err_cap, "schema recursion too deep");
    return false;
  }

  // Annotations collected for unevaluatedProperties / unevaluatedItems; merged
  // into the caller's `eval` at the end so in-place applicators propagate up.
  TfEvalSet local;
  eval_reset(&local);

  // Boolean schema: true accepts anything, false rejects everything.
  TfJsonType st = tf_json_type(schema);
  if (st == TF_JSON_BOOL) {
    const char *p = tf_json_skip_ws(schema.start);
    if (*p == 't') {
      return true;
    }
    tf_copy_cstr(err, err_cap, "schema is false; nothing validates");
    return false;
  }
  if (st != TF_JSON_OBJECT) {
    tf_copy_cstr(err, err_cap, "schema must be an object or boolean");
    return false;
  }

  // The subschema $id was already applied to `base` by the validate_root wrapper
  // (which also pushed the resource onto the dynamic scope).

  // $ref: resolve the reference URI against the current base, locate the target
  // (a local JSON pointer, a $anchor, or another $id'd subschema in this
  // document) via the index, and validate it. Draft 2020-12 keeps sibling
  // keywords, so validation continues below. Unknown documents (remote refs)
  // are treated as pass-through annotations.
  const char *ref_keywords[] = {"$ref", "$dynamicRef"};
  for (size_t rk = 0; rk < sizeof(ref_keywords) / sizeof(ref_keywords[0]); rk++) {
    TfJsonSpan ref_val;
    if (!tf_json_object_get_value(schema, ref_keywords[rk], &ref_val)) {
      continue;
    }
    char ref[TF_URI_CAP];
    const char *rp = tf_json_skip_ws(ref_val.start);
    if (!tf_json_read_string(&rp, ref, sizeof(ref))) {
      continue;
    }
    TfEvalSet ref_eval;
    eval_reset(&ref_eval);
    bool ok = (rk == 0)
                ? resolve_ref_into(root, dyn, base, index, ref, instance, &ref_eval, depth, err, err_cap)
                : resolve_dynamic_ref_into(
                    root, dyn, base, index, ref, instance, &ref_eval, depth, err, err_cap
                  );
    if (!ok) {
      return false;
    }
    eval_merge(&local, &ref_eval);  // $ref/$dynamicRef are in-place applicators
  }

  TfJsonType it = tf_json_type(instance);
  if (it == TF_JSON_INVALID) {
    tf_copy_cstr(err, err_cap, "instance is not valid JSON");
    return false;
  }

  // type
  TfJsonSpan type_val;
  if (tf_json_object_get_value(schema, "type", &type_val) &&
      !type_matches(type_val, it, instance)) {
    tf_copy_cstr(err, err_cap, "instance type does not match schema 'type'");
    return false;
  }

  // enum
  TfJsonSpan enum_val;
  if (tf_json_object_get_value(schema, "enum", &enum_val)) {
    size_t n = 0;
    bool found = false;
    if (tf_json_array_count(enum_val, &n)) {
      for (size_t i = 0; i < n; i++) {
        TfJsonSpan e;
        if (tf_json_array_get(enum_val, i, &e) && tf_json_equal(e, instance)) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      tf_copy_cstr(err, err_cap, "instance is not one of the enum values");
      return false;
    }
  }

  // const
  TfJsonSpan const_val;
  if (tf_json_object_get_value(schema, "const", &const_val) &&
      !tf_json_equal(const_val, instance)) {
    tf_copy_cstr(err, err_cap, "instance does not equal const");
    return false;
  }

  // allOf / anyOf / oneOf / not
  // In-place applicators contribute their evaluated properties/items to `local`.
  TfJsonSpan comb;
  if (tf_json_object_get_value(schema, "allOf", &comb)) {
    size_t n = 0;
    tf_json_array_count(comb, &n);
    for (size_t i = 0; i < n; i++) {
      TfJsonSpan sub;
      TfEvalSet sub_eval;
      eval_reset(&sub_eval);
      if (!tf_json_array_get(comb, i, &sub) ||
          !validate_root(root, dyn, base, index, sub, instance, &sub_eval, depth + 1, err, err_cap)) {
        if (err_cap > 0 && err[0] == '\0') {
          tf_copy_cstr(err, err_cap, "failed an allOf subschema");
        }
        return false;
      }
      eval_merge(&local, &sub_eval);
    }
  }
  if (tf_json_object_get_value(schema, "anyOf", &comb)) {
    size_t n = 0;
    tf_json_array_count(comb, &n);
    bool any = false;
    // Evaluate every branch (no short-circuit) so all matching branches
    // contribute annotations.
    for (size_t i = 0; i < n; i++) {
      TfJsonSpan sub;
      TfEvalSet sub_eval;
      eval_reset(&sub_eval);
      char ignore[TF_MAX_ERROR];
      if (tf_json_array_get(comb, i, &sub) &&
          validate_root(root, dyn, base, index, sub, instance, &sub_eval, depth + 1, ignore, sizeof(ignore))) {
        any = true;
        eval_merge(&local, &sub_eval);
      }
    }
    if (!any) {
      tf_copy_cstr(err, err_cap, "failed every anyOf subschema");
      return false;
    }
  }
  if (tf_json_object_get_value(schema, "oneOf", &comb)) {
    size_t n = 0;
    tf_json_array_count(comb, &n);
    size_t matched = 0;
    TfEvalSet match_eval;
    eval_reset(&match_eval);
    for (size_t i = 0; i < n; i++) {
      TfJsonSpan sub;
      TfEvalSet sub_eval;
      eval_reset(&sub_eval);
      char ignore[TF_MAX_ERROR];
      if (tf_json_array_get(comb, i, &sub) &&
          validate_root(root, dyn, base, index, sub, instance, &sub_eval, depth + 1, ignore, sizeof(ignore))) {
        matched++;
        match_eval = sub_eval;
      }
    }
    if (matched != 1) {
      tf_copy_cstr(err, err_cap, "instance must match exactly one oneOf subschema");
      return false;
    }
    eval_merge(&local, &match_eval);
  }
  TfJsonSpan not_schema;
  if (tf_json_object_get_value(schema, "not", &not_schema)) {
    char ignore[TF_MAX_ERROR];  // a failed 'not' contributes no annotations
    if (validate_root(root, dyn, base, index, not_schema, instance, 0, depth + 1, ignore, sizeof(ignore))) {
      tf_copy_cstr(err, err_cap, "instance must not match the 'not' subschema");
      return false;
    }
  }

  // if / then / else
  TfJsonSpan if_schema;
  if (tf_json_object_get_value(schema, "if", &if_schema)) {
    char ignore[TF_MAX_ERROR];
    TfEvalSet if_eval;
    eval_reset(&if_eval);
    bool if_valid =
      validate_root(root, dyn, base, index, if_schema, instance, &if_eval, depth + 1, ignore, sizeof(ignore));
    TfJsonSpan branch;
    TfEvalSet branch_eval;
    eval_reset(&branch_eval);
    if (if_valid) {
      eval_merge(&local, &if_eval);  // 'if' annotations apply when it matches
      if (tf_json_object_get_value(schema, "then", &branch) &&
          !validate_root(root, dyn, base, index, branch, instance, &branch_eval, depth + 1, err, err_cap)) {
        return false;
      }
    } else {
      if (tf_json_object_get_value(schema, "else", &branch) &&
          !validate_root(root, dyn, base, index, branch, instance, &branch_eval, depth + 1, err, err_cap)) {
        return false;
      }
    }
    eval_merge(&local, &branch_eval);
  }

  // Numeric constraints.
  if (it == TF_JSON_NUMBER) {
    double v = 0.0;
    tf_json_number(instance, &v);
    double bound = 0.0;
    if (get_number_keyword(schema, "minimum", &bound) && v < bound) {
      tf_copy_cstr(err, err_cap, "number below minimum");
      return false;
    }
    if (get_number_keyword(schema, "maximum", &bound) && v > bound) {
      tf_copy_cstr(err, err_cap, "number above maximum");
      return false;
    }
    if (get_number_keyword(schema, "exclusiveMinimum", &bound) && v <= bound) {
      tf_copy_cstr(err, err_cap, "number not above exclusiveMinimum");
      return false;
    }
    if (get_number_keyword(schema, "exclusiveMaximum", &bound) && v >= bound) {
      tf_copy_cstr(err, err_cap, "number not below exclusiveMaximum");
      return false;
    }
    double mult = 0.0;
    if (get_number_keyword(schema, "multipleOf", &mult) && mult != 0.0) {
      double q = v / mult;
      // multipleOf is checked in double precision. Beyond the exactly-representable
      // integral range the quotient cannot be verified, so we fail closed (reject)
      // rather than accept an unverifiable multiple — Python computes this exactly
      // with bignum/decimal arithmetic and may differ for |value| > ~9e18.
      if (q < -9.0e18 || q > 9.0e18 || q != (double)(long long)q) {
        tf_copy_cstr(err, err_cap, "number is not a multiple of multipleOf");
        return false;
      }
    }
  }

  // String constraints.
  if (it == TF_JSON_STRING) {
    double bound = 0.0;
    if (get_number_keyword(schema, "minLength", &bound) ||
        get_number_keyword(schema, "maxLength", &bound)) {
      size_t cps = string_codepoints(instance);
      if (cps == (size_t)-1) {
        tf_copy_cstr(err, err_cap, "could not measure string length");
        return false;
      }
      double minl = 0.0;
      double maxl = 0.0;
      if (get_number_keyword(schema, "minLength", &minl) && (double)cps < minl) {
        tf_copy_cstr(err, err_cap, "string shorter than minLength");
        return false;
      }
      if (get_number_keyword(schema, "maxLength", &maxl) && (double)cps > maxl) {
        tf_copy_cstr(err, err_cap, "string longer than maxLength");
        return false;
      }
    }
    TfJsonSpan pattern_val;
    if (tf_json_object_get_value(schema, "pattern", &pattern_val)) {
      char pat[512];
      const char *pp = tf_json_skip_ws(pattern_val.start);
      if (tf_json_read_string(&pp, pat, sizeof(pat))) {
        size_t slen = 0;
        char *s = decode_string_alloc(instance, &slen);
        if (s == 0) {
          tf_copy_cstr(err, err_cap, "could not decode string for pattern");
          return false;
        }
        bool supported = true;
        bool matched = tf_regex_search(pat, s, slen, &supported);
        free(s);
        if (supported && !matched) {  // unsupported regex syntax -> annotation
          tf_copy_cstr(err, err_cap, "string does not match pattern");
          return false;
        }
      }
    }
  }

  // Array constraints.
  if (it == TF_JSON_ARRAY) {
    size_t count = 0;
    if (!tf_json_array_count(instance, &count)) {
      tf_copy_cstr(err, err_cap, "instance array is malformed");
      return false;
    }
    double bound = 0.0;
    if (get_number_keyword(schema, "minItems", &bound) && (double)count < bound) {
      tf_copy_cstr(err, err_cap, "array has fewer than minItems");
      return false;
    }
    if (get_number_keyword(schema, "maxItems", &bound) && (double)count > bound) {
      tf_copy_cstr(err, err_cap, "array has more than maxItems");
      return false;
    }
    TfJsonSpan contains_schema;
    if (tf_json_object_get_value(schema, "contains", &contains_schema)) {
      size_t matches = 0;
      for (size_t i = 0; i < count; i++) {
        TfJsonSpan elem;
        char ignore[TF_MAX_ERROR];
        if (tf_json_array_get(instance, i, &elem) &&
            validate_root(root, dyn, base, index, contains_schema, elem, 0, depth + 1, ignore, sizeof(ignore))) {
          matches++;
          eval_mark_item(&local, i);  // a contains match is an evaluated item
        }
      }
      double minc = 1.0;
      double maxc = 0.0;
      (void)get_number_keyword(schema, "minContains", &minc);
      if ((double)matches < minc) {
        tf_copy_cstr(err, err_cap, "array has too few items matching 'contains'");
        return false;
      }
      if (get_number_keyword(schema, "maxContains", &maxc) && (double)matches > maxc) {
        tf_copy_cstr(err, err_cap, "array has too many items matching 'contains'");
        return false;
      }
    }
    TfJsonSpan unique;
    if (tf_json_object_get_value(schema, "uniqueItems", &unique)) {
      const char *up = tf_json_skip_ws(unique.start);
      if (*up == 't') {
        for (size_t i = 0; i < count; i++) {
          for (size_t j = i + 1; j < count; j++) {
            TfJsonSpan ei;
            TfJsonSpan ej;
            if (tf_json_array_get(instance, i, &ei) && tf_json_array_get(instance, j, &ej) &&
                tf_json_equal(ei, ej)) {
              tf_copy_cstr(err, err_cap, "array items are not unique");
              return false;
            }
          }
        }
      }
    }
    TfJsonSpan prefix_items;
    size_t prefix_count = 0;
    bool has_prefix = tf_json_object_get_value(schema, "prefixItems", &prefix_items);
    if (has_prefix) {
      tf_json_array_count(prefix_items, &prefix_count);
    }
    TfJsonSpan items_schema;
    bool has_items = tf_json_object_get_value(schema, "items", &items_schema);
    for (size_t i = 0; i < count; i++) {
      TfJsonSpan elem;
      if (!tf_json_array_get(instance, i, &elem)) {
        continue;
      }
      if (has_prefix && i < prefix_count) {
        TfJsonSpan psub;
        if (tf_json_array_get(prefix_items, i, &psub) &&
            !validate_root(root, dyn, base, index, psub, elem, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_mark_item(&local, i);
      } else if (has_items) {
        if (!validate_root(root, dyn, base, index, items_schema, elem, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_mark_item(&local, i);
      }
    }
    if (has_items) {
      local.all_items = true;  // `items` covers every position from prefix onward
    }

    // unevaluatedItems: applies to items not evaluated by prefixItems/items/
    // contains or an in-place applicator.
    TfJsonSpan uneval_items;
    if (!local.all_items && tf_json_object_get_value(schema, "unevaluatedItems", &uneval_items)) {
      // The evaluated-item bitmap is bounded; for a larger array we cannot tell
      // which items past the cap were evaluated, so fail loud rather than risk a
      // false unevaluated-item rejection. (all_items short-circuits the common
      // `items`-covers-everything case above, so this only bites position-specific
      // evaluation on very large arrays.)
      if (count > TF_EVAL_MAX_ITEMS) {
        tf_copy_cstr(err, err_cap, "array too large for unevaluatedItems tracking");
        return false;
      }
      const char *up = tf_json_skip_ws(uneval_items.start);
      for (size_t i = 0; i < count; i++) {
        if (eval_item_seen(&local, i)) {
          continue;
        }
        if (*up == 'f') {
          tf_copy_cstr(err, err_cap, "array has an unevaluated item");
          return false;
        }
        TfJsonSpan elem;
        if (tf_json_array_get(instance, i, &elem) && *up != 't' &&
            !validate_root(root, dyn, base, index, uneval_items, elem, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_mark_item(&local, i);
      }
    }
  }

  // Object constraints.
  if (it == TF_JSON_OBJECT) {
    TfJsonSpan required;
    if (tf_json_object_get_value(schema, "required", &required)) {
      size_t n = 0;
      tf_json_array_count(required, &n);
      for (size_t i = 0; i < n; i++) {
        TfJsonSpan rk;
        char key[96];
        const char *p;
        if (!tf_json_array_get(required, i, &rk)) {
          continue;
        }
        p = tf_json_skip_ws(rk.start);
        if (tf_json_read_string(&p, key, sizeof(key)) && !tf_json_object_has_key(instance, key)) {
          snprintf(err, err_cap, "missing required property: %s", key);
          return false;
        }
      }
    }

    char keys[TF_MAX_SCHEMA_PROPERTIES][96];
    size_t nk = 0;
    if (!tf_json_collect_object_keys(instance, keys, TF_MAX_SCHEMA_PROPERTIES, &nk)) {
      tf_copy_cstr(err, err_cap, "object has too many properties to validate");
      return false;
    }
    double bound = 0.0;
    if (get_number_keyword(schema, "minProperties", &bound) && (double)nk < bound) {
      tf_copy_cstr(err, err_cap, "object has fewer than minProperties");
      return false;
    }
    if (get_number_keyword(schema, "maxProperties", &bound) && (double)nk > bound) {
      tf_copy_cstr(err, err_cap, "object has more than maxProperties");
      return false;
    }

    TfJsonSpan props;
    bool has_props = tf_json_object_get_object(schema, "properties", &props);
    TfJsonSpan pat_props;
    bool has_pat_props = tf_json_object_get_object(schema, "patternProperties", &pat_props);
    char pat_keys[TF_MAX_SCHEMA_PROPERTIES][96];
    size_t n_pat = 0;
    if (has_pat_props) {
      (void)tf_json_collect_object_keys(pat_props, pat_keys, TF_MAX_SCHEMA_PROPERTIES, &n_pat);
    }
    TfJsonSpan addprops;
    bool has_addprops = tf_json_object_get_value(schema, "additionalProperties", &addprops);
    bool addprops_is_false = false;
    if (has_addprops) {
      const char *ap = tf_json_skip_ws(addprops.start);
      addprops_is_false = (*ap == 'f');
    }
    for (size_t i = 0; i < nk; i++) {
      TfJsonSpan prop_schema;
      bool in_props = has_props && tf_json_object_get_value(props, keys[i], &prop_schema);
      TfJsonSpan value;
      if (!tf_json_object_get_value(instance, keys[i], &value)) {
        continue;
      }
      if (in_props) {
        if (!validate_root(root, dyn, base, index, prop_schema, value, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_add_prop(&local, keys[i]);
      }
      // patternProperties: a key matching any pattern is validated against that
      // subschema and is no longer "additional".
      bool matched_pattern = false;
      for (size_t pi = 0; pi < n_pat; pi++) {
        bool supported = true;
        if (tf_regex_search(pat_keys[pi], keys[i], strlen(keys[i]), &supported) && supported) {
          matched_pattern = true;
          TfJsonSpan psub;
          if (tf_json_object_get_value(pat_props, pat_keys[pi], &psub) &&
              !validate_root(root, dyn, base, index, psub, value, 0, depth + 1, err, err_cap)) {
            return false;
          }
        }
      }
      if (matched_pattern) {
        eval_add_prop(&local, keys[i]);
      }
      if (!in_props && !matched_pattern && has_addprops) {
        if (addprops_is_false) {
          snprintf(err, err_cap, "additional property not allowed: %s", keys[i]);
          return false;
        }
        const char *ap = tf_json_skip_ws(addprops.start);
        if (*ap != 't' && !validate_root(root, dyn, base, index, addprops, value, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_add_prop(&local, keys[i]);  // additionalProperties evaluates it
      }
    }

    // propertyNames: each property name (as a string instance) must validate.
    TfJsonSpan name_schema;
    if (tf_json_object_get_value(schema, "propertyNames", &name_schema)) {
      for (size_t i = 0; i < nk; i++) {
        char encoded[256];
        if (!tf_json_write_string(keys[i], encoded, sizeof(encoded))) {
          continue;
        }
        TfJsonSpan name_inst = {encoded, encoded + strlen(encoded)};
        if (!validate_root(root, dyn, base, index, name_schema, name_inst, 0, depth + 1, err, err_cap)) {
          return false;
        }
      }
    }

    // dependentRequired: {"a": ["b", ...]} — if a present, b... must be present.
    TfJsonSpan dep_req;
    if (tf_json_object_get_object(schema, "dependentRequired", &dep_req)) {
      char dep_keys[TF_MAX_SCHEMA_PROPERTIES][96];
      size_t ndep = 0;
      if (tf_json_collect_object_keys(dep_req, dep_keys, TF_MAX_SCHEMA_PROPERTIES, &ndep)) {
        for (size_t i = 0; i < ndep; i++) {
          if (!tf_json_object_has_key(instance, dep_keys[i])) {
            continue;
          }
          TfJsonSpan reqlist;
          if (!tf_json_object_get_value(dep_req, dep_keys[i], &reqlist)) {
            continue;
          }
          size_t rn = 0;
          tf_json_array_count(reqlist, &rn);
          for (size_t r = 0; r < rn; r++) {
            TfJsonSpan rk;
            char key[96];
            const char *p;
            if (!tf_json_array_get(reqlist, r, &rk)) {
              continue;
            }
            p = tf_json_skip_ws(rk.start);
            if (tf_json_read_string(&p, key, sizeof(key)) &&
                !tf_json_object_has_key(instance, key)) {
              snprintf(err, err_cap, "dependentRequired: %s requires %s", dep_keys[i], key);
              return false;
            }
          }
        }
      }
    }

    // dependentSchemas: {"a": {schema}} — if a present, instance must validate.
    TfJsonSpan dep_sch;
    if (tf_json_object_get_object(schema, "dependentSchemas", &dep_sch)) {
      char dep_keys[TF_MAX_SCHEMA_PROPERTIES][96];
      size_t ndep = 0;
      if (tf_json_collect_object_keys(dep_sch, dep_keys, TF_MAX_SCHEMA_PROPERTIES, &ndep)) {
        for (size_t i = 0; i < ndep; i++) {
          if (!tf_json_object_has_key(instance, dep_keys[i])) {
            continue;
          }
          TfJsonSpan sub;
          TfEvalSet sub_eval;
          eval_reset(&sub_eval);
          if (tf_json_object_get_value(dep_sch, dep_keys[i], &sub) &&
              !validate_root(root, dyn, base, index, sub, instance, &sub_eval, depth + 1, err, err_cap)) {
            return false;
          }
          eval_merge(&local, &sub_eval);  // dependentSchemas is in-place
        }
      }
    }

    // unevaluatedProperties: applies to instance members not evaluated above (by
    // properties / patternProperties / additionalProperties or any in-place
    // applicator). Runs last so `local` is fully populated.
    TfJsonSpan uneval;
    if (tf_json_object_get_value(schema, "unevaluatedProperties", &uneval)) {
      for (size_t i = 0; i < nk; i++) {
        if (eval_has_prop(&local, keys[i])) {
          continue;
        }
        TfJsonSpan value;
        if (!tf_json_object_get_value(instance, keys[i], &value)) {
          continue;
        }
        const char *up = tf_json_skip_ws(uneval.start);
        if (*up == 'f') {
          snprintf(err, err_cap, "unevaluated property not allowed: %s", keys[i]);
          return false;
        }
        if (*up != 't' &&
            !validate_root(root, dyn, base, index, uneval, value, 0, depth + 1, err, err_cap)) {
          return false;
        }
        eval_add_prop(&local, keys[i]);
      }
    }
  }

  eval_merge(eval, &local);
  return true;
}

// Wrapper: apply this subschema's own $id to the base URI, push the resulting
// resource base onto the dynamic scope while validating its subtree (balanced
// pop), then run the keyword body. Sibling/child schemas recurse back through
// here so the dynamic scope reflects exactly the resources on the active path.
static bool validate_root(
  TfJsonSpan root,
  TfDynScope *dyn,
  const char *base,
  const TfIdIndex *index,
  TfJsonSpan schema,
  TfJsonSpan instance,
  TfEvalSet *eval,
  int depth,
  char *err,
  size_t err_cap
) {
  char local_base[TF_URI_CAP];
  tf_copy_cstr(local_base, sizeof(local_base), base);
  char idbuf[TF_URI_CAP];
  if (tf_json_type(schema) == TF_JSON_OBJECT &&
      tf_json_object_get_string(schema, "$id", idbuf, sizeof(idbuf))) {
    char resolved[TF_URI_CAP];
    resolve_uri(base, idbuf, resolved, sizeof(resolved));
    char *frag = strchr(resolved, '#');
    if (frag) {
      *frag = '\0';
    }
    tf_copy_cstr(local_base, sizeof(local_base), resolved);
  }
  size_t mark = dyn != 0 ? dyn->count : 0;
  bool pushed = false;
  if (dyn != 0 && strcmp(local_base, base) != 0) {
    dyn_push(dyn, local_base);
    pushed = true;
  }
  bool ok = validate_root_body(
    root, dyn, local_base, index, schema, instance, eval, depth, err, err_cap
  );
  if (pushed) {
    dyn->count = mark;
  }
  return ok;
}

bool tf_jsonschema_validate_ex(
  TfJsonSpan schema,
  TfJsonSpan instance,
  const TfJsonSpan *registry,
  size_t registry_count,
  char *err,
  size_t err_cap
) {
  // Establish the root base URI from a top-level $id, then index every in-document
  // $id / $anchor / $dynamicAnchor so $ref and $dynamicRef can resolve.
  char root_base[TF_URI_CAP] = "";
  char idbuf[TF_URI_CAP];
  if (tf_json_type(schema) == TF_JSON_OBJECT &&
      tf_json_object_get_string(schema, "$id", idbuf, sizeof(idbuf))) {
    resolve_uri("", idbuf, root_base, sizeof(root_base));
    char *frag = strchr(root_base, '#');
    if (frag) {
      *frag = '\0';
    }
  }
  TfIdIndex index;
  index.count = 0;
  index.overflow = false;
  index_build(schema, "", &index, 0);
  index_add(&index, root_base, "", schema, false);  // root resolves by its base
  // Registry documents (e.g. the metaschema + its vocabularies) are each indexed
  // under their own absolute $id so remote $ref / $dynamicRef can resolve them.
  for (size_t i = 0; i < registry_count; i++) {
    index_build(registry[i], "", &index, 0);
  }
  if (index.overflow) {
    // A reference target was dropped — fail loud rather than mis-validate by
    // treating a now-unresolvable $ref as a pass-through annotation.
    tf_copy_cstr(err, err_cap, "schema reference index overflow (too many $id/$anchor targets)");
    return false;
  }
  TfDynScope dyn;
  dyn.count = 0;
  dyn_push(&dyn, root_base);  // the root resource is the outermost dynamic scope
  return validate_root(schema, &dyn, root_base, &index, schema, instance, 0, 0, err, err_cap);
}

bool tf_jsonschema_validate(TfJsonSpan schema, TfJsonSpan instance, char *err, size_t err_cap) {
  return tf_jsonschema_validate_ex(schema, instance, 0, 0, err, err_cap);
}

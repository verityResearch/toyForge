#ifndef TOYFORGE_JSON_H
#define TOYFORGE_JSON_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  const char *start;
  const char *end;
} TfJsonSpan;

typedef enum {
  TF_JSON_INVALID,
  TF_JSON_NULL,
  TF_JSON_BOOL,
  TF_JSON_NUMBER,
  TF_JSON_STRING,
  TF_JSON_ARRAY,
  TF_JSON_OBJECT
} TfJsonType;

// Classify the JSON value at the start of `span` (leading whitespace skipped).
TfJsonType tf_json_type(TfJsonSpan span);

// Parse a JSON number span into `*out`. Returns false if not a number.
bool tf_json_number(TfJsonSpan span, double *out);

// True if the JSON number span is an integer value (e.g. 2 or 2.0, not 2.5).
bool tf_json_is_integer(TfJsonSpan span);

// Deep structural equality of two JSON values (numbers compared by value,
// object members order-independent, array elements order-sensitive).
bool tf_json_equal(TfJsonSpan a, TfJsonSpan b);

const char *tf_json_skip_ws(const char *p);
bool tf_json_read_string(const char **cursor, char *out, size_t out_cap);
bool tf_json_string_decoded_size(TfJsonSpan string, size_t *out_size);
bool tf_json_write_string(const char *input, char *out, size_t out_cap);
bool tf_json_skip_value(const char **cursor);
bool tf_json_array_count(TfJsonSpan array, size_t *count);
bool tf_json_array_get(TfJsonSpan array, size_t index, TfJsonSpan *value);
bool tf_json_object_get_value(TfJsonSpan object, const char *key, TfJsonSpan *value);
bool tf_json_object_get_string(TfJsonSpan object, const char *key, char *out, size_t out_cap);
bool tf_json_object_get_object(TfJsonSpan object, const char *key, TfJsonSpan *out);
bool tf_json_object_has_key(TfJsonSpan object, const char *key);
bool tf_json_collect_object_keys(
  TfJsonSpan object,
  char keys[][96],
  size_t max_keys,
  size_t *count
);

#endif

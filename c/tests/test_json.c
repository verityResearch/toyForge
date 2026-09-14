#include "toyforge/json.h"

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

#define ASSERT_STR_EQ(actual, expected)                                                   \
  do {                                                                                    \
    if (strcmp((actual), (expected)) != 0) {                                               \
      fprintf(                                                                            \
        stderr,                                                                           \
        "ASSERT_STR_EQ failed at %s:%d: %s='%s' expected='%s'\n",                        \
        __FILE__,                                                                         \
        __LINE__,                                                                         \
        #actual,                                                                          \
        (actual),                                                                         \
        (expected)                                                                        \
      );                                                                                  \
      failures++;                                                                         \
    }                                                                                     \
  } while (0)

static void assert_parse_string(const char *json, const char *expected) {
  char out[128] = "";
  const char *cursor = json;
  ASSERT_TRUE(tf_json_read_string(&cursor, out, sizeof(out)));
  ASSERT_TRUE(*tf_json_skip_ws(cursor) == '\0');
  ASSERT_STR_EQ(out, expected);
}

static void test_json_string_decodes_basic_escapes(void) {
  assert_parse_string("\"quote:\\\" slash:\\\\ line:\\n\"", "quote:\" slash:\\ line:\n");
}

static void test_json_string_decodes_utf8_and_unicode_escapes(void) {
  const char cafe[] = {'c', 'a', 'f', (char)0xC3, (char)0xA9, '\0'};
  const char rocket[] = {(char)0xF0, (char)0x9F, (char)0x9A, (char)0x80, '\0'};

  assert_parse_string("\"caf\\u00e9\"", cafe);
  assert_parse_string("\"\\ud83d\\ude80\"", rocket);
  assert_parse_string("\"caf\xC3\xA9\"", cafe);
}

static void test_json_string_rejects_invalid_unicode(void) {
  char out[32] = "";
  const char *high = "\"\\ud83d\"";
  const char *low = "\"\\ude80\"";
  const char *bad_hex = "\"\\u12xz\"";

  ASSERT_TRUE(!tf_json_read_string(&high, out, sizeof(out)));
  ASSERT_TRUE(!tf_json_read_string(&low, out, sizeof(out)));
  ASSERT_TRUE(!tf_json_read_string(&bad_hex, out, sizeof(out)));
}

static void test_json_string_rejects_invalid_raw_utf8_and_controls(void) {
  char out[32] = "";
  const char invalid_utf8[] = {'"', (char)0xC3, '(', '"', '\0'};
  const char raw_control[] = {'"', 'a', '\n', 'b', '"', '\0'};
  const char *invalid_cursor = invalid_utf8;
  const char *control_cursor = raw_control;

  ASSERT_TRUE(!tf_json_read_string(&invalid_cursor, out, sizeof(out)));
  ASSERT_TRUE(!tf_json_read_string(&control_cursor, out, sizeof(out)));
}

static void test_json_string_fails_when_output_buffer_is_too_small(void) {
  char out[4] = "";
  const char *cursor = "\"abcd\"";
  ASSERT_TRUE(!tf_json_read_string(&cursor, out, sizeof(out)));
}

static void test_json_string_writer_escapes_and_preserves_utf8(void) {
  char out[128] = "";
  const char input[] = {
    'q', '"', '\n', (char)0xF0, (char)0x9F, (char)0x9A, (char)0x80, '\0'
  };
  const char expected[] = {
    '"',       'q',       '\\',      '"', '\\', 'n', (char)0xF0,
    (char)0x9F, (char)0x9A, (char)0x80, '"', '\0'
  };

  ASSERT_TRUE(tf_json_write_string(input, out, sizeof(out)));
  ASSERT_STR_EQ(out, expected);
}

static void test_json_string_writer_escapes_control_characters(void) {
  char out[128] = "";
  const char input[] = {'a', 1, 'b', '\0'};
  ASSERT_TRUE(tf_json_write_string(input, out, sizeof(out)));
  ASSERT_STR_EQ(out, "\"a\\u0001b\"");
}

static void test_json_string_writer_rejects_invalid_utf8_and_small_buffers(void) {
  char out[16] = "";
  const char invalid_utf8[] = {'x', (char)0xC3, '(', '\0'};

  ASSERT_TRUE(!tf_json_write_string(invalid_utf8, out, sizeof(out)));
  ASSERT_TRUE(!tf_json_write_string("abcd", out, 4));
}

static void test_json_string_decoded_size_has_no_output_buffer_limit(void) {
  char json[1104];
  json[0] = '"';
  memset(json + 1, 'x', 1050);
  json[1051] = '"';
  json[1052] = '\0';
  TfJsonSpan span = {json, json + strlen(json)};
  size_t decoded_size = 0;
  char out[32] = "";
  const char *cursor = json;

  ASSERT_TRUE(tf_json_string_decoded_size(span, &decoded_size));
  ASSERT_TRUE(decoded_size == 1050);
  ASSERT_TRUE(!tf_json_read_string(&cursor, out, sizeof(out)));
}

static void test_json_string_decoded_size_counts_utf8_bytes(void) {
  const char *json = "\"\\ud83d\\ude80\"";
  TfJsonSpan span = {json, json + strlen(json)};
  size_t decoded_size = 0;

  ASSERT_TRUE(tf_json_string_decoded_size(span, &decoded_size));
  ASSERT_TRUE(decoded_size == 4);
}

static void test_json_skip_value_accepts_numbers_after_whitespace(void) {
  const char *json = "{\"type\":\"string\",\"minLength\": 4}";
  const char *cursor = json;
  ASSERT_TRUE(tf_json_skip_value(&cursor));
  ASSERT_TRUE(*cursor == '\0');
}

static void test_json_object_get_string_accepts_inline_object_value(void) {
  const char *json = "{ \"type\": \"string\" }";
  TfJsonSpan object = {json, json + strlen(json)};
  char value[32] = "";
  ASSERT_TRUE(tf_json_object_get_string(object, "type", value, sizeof(value)));
  ASSERT_STR_EQ(value, "string");
}

static void test_json_array_helpers_count_and_get_items(void) {
  const char *json = "[{\"a\":1}, \"x\", []]";
  TfJsonSpan array = {json, json + strlen(json)};
  TfJsonSpan item;
  size_t count = 0;

  ASSERT_TRUE(tf_json_array_count(array, &count));
  ASSERT_TRUE(count == 3);
  ASSERT_TRUE(tf_json_array_get(array, 0, &item));
  ASSERT_TRUE(item.start[0] == '{');
  ASSERT_TRUE(tf_json_array_get(array, 1, &item));
  ASSERT_TRUE(item.start[0] == '"');
  ASSERT_TRUE(tf_json_array_get(array, 2, &item));
  ASSERT_TRUE(item.start[0] == '[');
  ASSERT_TRUE(!tf_json_array_get(array, 3, &item));
}

static void test_json_array_helpers_accept_empty_arrays(void) {
  const char *json = "[]";
  TfJsonSpan array = {json, json + strlen(json)};
  TfJsonSpan item;
  size_t count = 1;

  ASSERT_TRUE(tf_json_array_count(array, &count));
  ASSERT_TRUE(count == 0);
  ASSERT_TRUE(!tf_json_array_get(array, 0, &item));
}

static void test_json_array_helpers_reject_malformed_arrays(void) {
  const char *json = "[1,]";
  TfJsonSpan array = {json, json + strlen(json)};
  TfJsonSpan item;
  size_t count = 0;

  ASSERT_TRUE(!tf_json_array_count(array, &count));
  ASSERT_TRUE(!tf_json_array_get(array, 0, &item));
}

static void test_json_skip_value_bounds_recursion_depth(void) {
  // Deeply-nested input must fail closed, not overflow the stack (the parser is
  // recursive; without a depth bound "[[[[..." crashes every consumer).
  const size_t deep = 4000;
  char *buf = (char *)malloc(2 * deep + 1);
  ASSERT_TRUE(buf != 0);
  for (size_t i = 0; i < deep; i++) {
    buf[i] = '[';
    buf[2 * deep - 1 - i] = ']';
  }
  buf[2 * deep] = '\0';
  const char *cursor = buf;
  ASSERT_TRUE(!tf_json_skip_value(&cursor));  // rejected, no crash
  free(buf);

  // A modestly-nested array (well under the bound) still parses fully.
  const char *ok = "[[[[[[[[[[1]]]]]]]]]]";  // 10 deep
  const char *ok_cursor = ok;
  ASSERT_TRUE(tf_json_skip_value(&ok_cursor));
  ASSERT_TRUE(*ok_cursor == '\0');
}

int main(void) {
  test_json_string_decodes_basic_escapes();
  test_json_string_decodes_utf8_and_unicode_escapes();
  test_json_string_rejects_invalid_unicode();
  test_json_string_rejects_invalid_raw_utf8_and_controls();
  test_json_string_fails_when_output_buffer_is_too_small();
  test_json_string_writer_escapes_and_preserves_utf8();
  test_json_string_writer_escapes_control_characters();
  test_json_string_writer_rejects_invalid_utf8_and_small_buffers();
  test_json_string_decoded_size_has_no_output_buffer_limit();
  test_json_string_decoded_size_counts_utf8_bytes();
  test_json_skip_value_accepts_numbers_after_whitespace();
  test_json_object_get_string_accepts_inline_object_value();
  test_json_array_helpers_count_and_get_items();
  test_json_array_helpers_accept_empty_arrays();
  test_json_array_helpers_reject_malformed_arrays();
  test_json_skip_value_bounds_recursion_depth();

  if (failures != 0) {
    fprintf(stderr, "%d json test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("json tests passed");
  return EXIT_SUCCESS;
}

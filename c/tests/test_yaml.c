#include "toyforge/yaml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define ASSERT_TRUE(expr)                                                             \
  do {                                                                                \
    if (!(expr)) {                                                                    \
      fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      failures++;                                                                     \
    }                                                                                 \
  } while (0)

#define ASSERT_STR_EQ(actual, expected)                                  \
  do {                                                                   \
    const char *a_ = (actual);                                          \
    const char *e_ = (expected);                                        \
    if (a_ == NULL || strcmp(a_, e_) != 0) {                           \
      fprintf(                                                          \
        stderr,                                                         \
        "ASSERT_STR_EQ failed at %s:%d: got='%s' expected='%s'\n",     \
        __FILE__,                                                       \
        __LINE__,                                                       \
        a_ ? a_ : "(null)",                                            \
        e_                                                              \
      );                                                                \
      failures++;                                                       \
    }                                                                   \
  } while (0)

// Parse `yaml`, serialize to JSON, and assert it equals `expected_json`.
static void assert_json(const char *yaml, const char *expected_json) {
  char err[256] = "";
  TfYamlNode *root = tf_yaml_parse(yaml, err, sizeof(err));
  if (root == NULL) {
    fprintf(stderr, "parse failed for [%s]: %s\n", yaml, err);
    failures++;
    return;
  }
  char *json = tf_yaml_to_json(root);
  ASSERT_STR_EQ(json, expected_json);
  free(json);
  tf_yaml_free(root);
}

static void test_scalars(void) {
  assert_json("a: 1", "{\"a\":1}");
  assert_json("a: -5", "{\"a\":-5}");
  assert_json("a: hello", "{\"a\":\"hello\"}");
  assert_json("a: \"quoted: colon\"", "{\"a\":\"quoted: colon\"}");
  assert_json("a: true", "{\"a\":true}");
  assert_json("a: false", "{\"a\":false}");
  assert_json("a: null", "{\"a\":null}");
  assert_json("a:", "{\"a\":null}");
  assert_json("a: 2.5", "{\"a\":2.5}");
  // A quoted version number stays a string, not an int/float.
  assert_json("version: \"1\"", "{\"version\":\"1\"}");
}

static void test_block_mapping_and_nesting(void) {
  assert_json("a: 1\nb: 2", "{\"a\":1,\"b\":2}");
  assert_json("outer:\n  inner: x", "{\"outer\":{\"inner\":\"x\"}}");
}

static void test_block_sequence(void) {
  assert_json("items:\n  - NEW\n  - TRIAGED", "{\"items\":[\"NEW\",\"TRIAGED\"]}");
  // Sequence of mappings, including continuation keys under a "- key:" item.
  assert_json(
    "seeds:\n  - id: 1\n    name: a\n  - id: 2\n    name: b",
    "{\"seeds\":[{\"id\":1,\"name\":\"a\"},{\"id\":2,\"name\":\"b\"}]}"
  );
}

static void test_flow(void) {
  assert_json("m: { a: 1, b: two }", "{\"m\":{\"a\":1,\"b\":\"two\"}}");
  assert_json("s: [1, 2, 3]", "{\"s\":[1,2,3]}");
  assert_json("s: []", "{\"s\":[]}");
  assert_json("m: {}", "{\"m\":{}}");
  assert_json(
    "p: { steps: [\"h1\", \"h2\"], t: call }",
    "{\"p\":{\"steps\":[\"h1\",\"h2\"],\"t\":\"call\"}}"
  );
}

static void test_comments_and_blanks(void) {
  assert_json("# top comment\na: 1  # trailing\n\nb: 2", "{\"a\":1,\"b\":2}");
  // '#' inside quotes is not a comment.
  assert_json("a: \"x # y\"", "{\"a\":\"x # y\"}");
}

static void test_map_get(void) {
  char err[256] = "";
  TfYamlNode *root = tf_yaml_parse("a: 1\nb: hello", err, sizeof(err));
  ASSERT_TRUE(root != NULL);
  const TfYamlNode *b = tf_yaml_map_get(root, "b");
  ASSERT_TRUE(b != NULL && b->type == TF_YAML_STRING);
  if (b && b->type == TF_YAML_STRING) {
    ASSERT_STR_EQ(b->as.string, "hello");
  }
  ASSERT_TRUE(tf_yaml_map_get(root, "missing") == NULL);
  tf_yaml_free(root);
}

int main(void) {
  test_scalars();
  test_block_mapping_and_nesting();
  test_block_sequence();
  test_flow();
  test_comments_and_blanks();
  test_map_get();
  if (failures > 0) {
    fprintf(stderr, "%d yaml test assertion(s) failed\n", failures);
    return 1;
  }
  printf("all yaml tests passed\n");
  return 0;
}

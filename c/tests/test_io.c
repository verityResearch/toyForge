#include "toyforge/io.h"

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

static void test_copy_cstr_truncates_and_terminates(void) {
  char buf[4];
  tf_copy_cstr(buf, sizeof(buf), "abcdef");
  ASSERT_STR_EQ(buf, "abc");
  tf_copy_cstr(buf, sizeof(buf), 0);
  ASSERT_STR_EQ(buf, "");
  tf_copy_cstr(0, sizeof(buf), "ignored");
}

static void test_copy_span_truncates_and_handles_invalid_ranges(void) {
  char buf[5];
  const char *value = "abcdef";
  tf_copy_span(buf, sizeof(buf), value + 1, value + 6);
  ASSERT_STR_EQ(buf, "bcde");
  tf_copy_span(buf, sizeof(buf), value + 4, value + 2);
  ASSERT_STR_EQ(buf, "");
  tf_copy_span(0, sizeof(buf), value, value + 1);
}

static void test_join_path(void) {
  char path[64];
  ASSERT_TRUE(tf_join_path(path, sizeof(path), "schemas", "state-machine.yaml"));
  ASSERT_STR_EQ(path, "schemas/state-machine.yaml");
  ASSERT_TRUE(!tf_join_path(path, 8, "schemas", "state-machine.yaml"));
  ASSERT_TRUE(!tf_join_path(0, sizeof(path), "schemas", "state-machine.yaml"));
}

static void test_read_file(void) {
  char err[128] = "";
  char *text = tf_read_file("../../schemas/jsonrpc-methods.json", err, sizeof(err));
  ASSERT_TRUE(text != 0);
  if (text != 0) {
    ASSERT_TRUE(strstr(text, "\"methods\"") != 0);
    free(text);
  }
  ASSERT_TRUE(err[0] == '\0');

  text = tf_read_file("../../schemas/does-not-exist.json", err, sizeof(err));
  ASSERT_TRUE(text == 0);
  ASSERT_TRUE(strstr(err, "failed to open") != 0);

  ASSERT_TRUE(tf_read_file(0, 0, 0) == 0);
  ASSERT_TRUE(tf_read_file("../../schemas/does-not-exist.json", 0, 0) == 0);
}

static void test_read_file_bytes(void) {
  unsigned char *bytes = 0;
  size_t len = 0;
  char err[128] = "";
  ASSERT_TRUE(tf_read_file_bytes("../tests/fixtures/hash/abc.txt", &bytes, &len, err, sizeof(err)));
  ASSERT_TRUE(bytes != 0);
  ASSERT_TRUE(len == 4);
  if (bytes != 0) {
    ASSERT_TRUE(bytes[0] == 'a');
    ASSERT_TRUE(bytes[3] == '\n');
    free(bytes);
  }
  ASSERT_TRUE(!tf_read_file_bytes("../tests/fixtures/hash/abc.txt", 0, &len, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "output buffer") != 0);
}

static void test_read_line(void) {
  FILE *f = tmpfile();
  ASSERT_TRUE(f != 0);
  if (f == 0) {
    return;
  }
  fputs("first\nsecond", f);
  rewind(f);

  char *line = tf_read_line(f);
  ASSERT_TRUE(line != 0);
  if (line != 0) {
    ASSERT_STR_EQ(line, "first\n");
    free(line);
  }
  line = tf_read_line(f);
  ASSERT_TRUE(line != 0);
  if (line != 0) {
    ASSERT_STR_EQ(line, "second");
    free(line);
  }
  ASSERT_TRUE(tf_read_line(f) == 0);
  fclose(f);
  ASSERT_TRUE(tf_read_line(0) == 0);
}

int main(void) {
  test_copy_cstr_truncates_and_terminates();
  test_copy_span_truncates_and_handles_invalid_ranges();
  test_join_path();
  test_read_file();
  test_read_file_bytes();
  test_read_line();

  if (failures != 0) {
    fprintf(stderr, "%d io test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("io tests passed");
  return EXIT_SUCCESS;
}

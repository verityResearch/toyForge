#include "toyforge/sha256.h"

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

static void assert_sha256_hex(const char *input, const char *expected) {
  char actual[TF_SHA256_HEX_SIZE] = "";
  ASSERT_TRUE(tf_sha256_hex(input, strlen(input), actual));
  ASSERT_STR_EQ(actual, expected);
}

static void test_known_vectors(void) {
  assert_sha256_hex(
    "",
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
  );
  assert_sha256_hex(
    "abc",
    "ba7816bf8f01cfea414140de5dae2223"
    "b00361a396177a9cb410ff61f20015ad"
  );
  assert_sha256_hex(
    "The quick brown fox jumps over the lazy dog",
    "d7a8fbb307d7809469ca9abcb0082e4f"
    "8d5651e46d3cdb762d02d0bf37c9e592"
  );
}

static void test_multiblock_input(void) {
  char input[1000];
  memset(input, 'a', sizeof(input));
  char actual[TF_SHA256_HEX_SIZE] = "";
  ASSERT_TRUE(tf_sha256_hex(input, sizeof(input), actual));
  ASSERT_STR_EQ(
    actual,
    "41edece42d63e8d9bf515a9ba6932e1c"
    "20cbc9f5a5d134645adb5db1b9737ea3"
  );
}

static void test_raw_digest_and_invalid_arguments(void) {
  unsigned char digest[TF_SHA256_DIGEST_SIZE];
  char hex[TF_SHA256_HEX_SIZE] = "";

  ASSERT_TRUE(tf_sha256("abc", 3, digest));
  ASSERT_TRUE(tf_sha256_hex("abc", 3, hex));
  ASSERT_TRUE(digest[0] == 0xba);
  ASSERT_TRUE(digest[31] == 0xad);
  ASSERT_TRUE(!tf_sha256(0, 1, digest));
  ASSERT_TRUE(!tf_sha256("abc", 3, 0));
  ASSERT_TRUE(!tf_sha256_hex("abc", 3, 0));
}

static void test_file_hash(void) {
  char actual[TF_SHA256_HEX_SIZE] = "";
  char err[128] = "";
  ASSERT_TRUE(tf_sha256_file_hex("../tests/fixtures/hash/abc.txt", actual, err, sizeof(err)));
  ASSERT_STR_EQ(
    actual,
    "edeaaff3f1774ad2888673770c6d640"
    "97e391bc362d7d6fb34982ddf0efd18cb"
  );
  ASSERT_TRUE(err[0] == '\0');
  ASSERT_TRUE(!tf_sha256_file_hex("../tests/fixtures/hash/missing.txt", actual, err, sizeof(err)));
  ASSERT_TRUE(strstr(err, "failed to open") != 0);
}

int main(void) {
  test_known_vectors();
  test_multiblock_input();
  test_raw_digest_and_invalid_arguments();
  test_file_hash();

  if (failures != 0) {
    fprintf(stderr, "%d sha256 test failures\n", failures);
    return EXIT_FAILURE;
  }
  puts("sha256 tests passed");
  return EXIT_SUCCESS;
}

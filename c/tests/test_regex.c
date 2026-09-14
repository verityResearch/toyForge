#include "toyforge/regex.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(const char *pattern, const char *text, bool expect_supported, bool expect_match) {
  bool supported = false;
  bool got = tf_regex_search(pattern, text, strlen(text), &supported);
  if (supported != expect_supported || (supported && got != expect_match)) {
    fprintf(
      stderr,
      "regex FAIL: /%s/ on \"%s\": supported=%d (want %d) match=%d (want %d)\n",
      pattern, text, supported, expect_supported, got, expect_match
    );
    failures++;
  }
}

int main(void) {
  // Literals are an unanchored search.
  check("X_", "fooX_bar", true, true);
  check("X_", "fooXbar", true, false);

  // '.' and quantifiers.
  check("b.*", "abc", true, true);  // matches "bc"
  check("f.*o", "xfoobar", true, true);
  check("f.*o", "fbar", true, false);
  check("a*", "", true, true);
  check("a*", "xxx", true, true);  // matches empty
  check("a+", "xxx", true, false);
  check("a+", "xax", true, true);
  check("aaa*", "a", true, false);   // needs at least "aa"
  check("aaa*", "aa", true, true);   // "aa" + zero more
  check("aaa*", "aaa", true, true);

  // Anchors.
  check("^.*bar$", "foobar", true, true);
  check("^.*bar$", "foobarbaz", true, false);
  check("^a*$", "aaa", true, true);
  check("^a*$", "aab", true, false);

  // Character classes and {n,}.
  check("[0-9]{2,}", "a12b", true, true);
  check("[0-9]{2,}", "a1b", true, false);
  check("[0-9]{2,}", "x99999y", true, true);
  check("^[a-z]+$", "hello", true, true);
  check("^[a-z]+$", "Hello", true, false);
  check("[^0-9]", "123a", true, true);
  check("[^0-9]", "123", true, false);
  check("\\d{3}", "ab123", true, true);
  check("\\d{3}", "ab12", true, false);
  check("^\\w+$", "snake_case1", true, true);
  check("^\\w+$", "no spaces", true, false);
  check("a{2,3}", "baaaa", true, true);
  check("^a{2,3}$", "aaaa", true, false);
  check("colou?r", "color", true, true);
  check("colou?r", "colour", true, true);
  check("colou?r", "coluor", true, false);

  // Unsupported syntax → reported, not guessed.
  check("(ab)+", "abab", false, false);
  check("a|b", "a", false, false);
  check("^\\p{Letter}+$", "abc", false, false);
  check("a\\1", "aa", false, false);

  if (failures != 0) {
    fprintf(stderr, "%d regex test failures\n", failures);
    return 1;
  }
  printf("regex tests passed\n");
  return 0;
}

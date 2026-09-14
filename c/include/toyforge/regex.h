#ifndef TOYFORGE_REGEX_H
#define TOYFORGE_REGEX_H

#include <stdbool.h>
#include <stddef.h>

// A bounded ECMA-262 / Python-`re.search`-compatible matcher for the regex
// subset JSON Schema `pattern`/`patternProperties` commonly use: literal
// characters, `.`, character classes `[...]` (ranges, negation, and the
// \d/\w/\s/\D/\W/\S shorthands), the `^`/`$` anchors, and the `*`/`+`/`?` and
// `{n}`/`{n,}`/`{n,m}` quantifiers (greedy). Matching is unanchored unless `^`
// is present (search semantics).
//
// Unsupported syntax — groups `()`, alternation `|`, backreferences,
// lookaround, and Unicode property escapes `\p{...}` — is reported by setting
// `*supported` to false; callers then treat the pattern as a pass-through
// annotation rather than guessing.
//
// Returns true iff the pattern matches somewhere in text[0, text_len).
bool tf_regex_search(const char *pattern, const char *text, size_t text_len, bool *supported);

#endif

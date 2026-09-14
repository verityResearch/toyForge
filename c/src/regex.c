#include "toyforge/regex.h"

#include <string.h>

// A "flat" regex: a sequence of quantified atoms with no groups or alternation.
typedef enum { RX_CHAR, RX_ANY, RX_CLASS, RX_START, RX_END } RxAtomType;

typedef struct {
  RxAtomType type;
  unsigned char ch;        // RX_CHAR
  unsigned char set[256];  // RX_CLASS: set[c] != 0 when byte c matches
  int min;                 // quantifier lower bound (default 1)
  int max;                 // quantifier upper bound, -1 = unbounded (default 1)
} RxAtom;

#define RX_MAX_ATOMS 256

typedef struct {
  RxAtom atoms[RX_MAX_ATOMS];
  size_t count;
} Rx;

static void class_add_shorthand(unsigned char *set, char kind) {
  for (int c = 0; c < 256; c++) {
    bool member = false;
    if (kind == 'd' || kind == 'D') {
      member = (c >= '0' && c <= '9');
    } else if (kind == 'w' || kind == 'W') {
      member = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               c == '_';
    } else if (kind == 's' || kind == 'S') {
      member = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
    }
    if (kind == 'D' || kind == 'W' || kind == 'S') {
      member = !member;
    }
    if (member) {
      set[c] = 1;
    }
  }
}

// Translate a backslash escape that denotes a single literal byte. Returns the
// byte and advances *p past it; returns -1 for shorthand classes (handled
// separately) or unsupported escapes.
static int escape_literal(const char **p) {
  char e = **p;
  (*p)++;
  switch (e) {
    case 'n':
      return '\n';
    case 't':
      return '\t';
    case 'r':
      return '\r';
    case 'f':
      return '\f';
    case 'v':
      return '\v';
    case '0':
      return '\0';
    default:
      // Any other escaped char is the literal char (\. \\ \( \[ \* ...).
      return (unsigned char)e;
  }
}

static bool is_shorthand(char c) {
  return c == 'd' || c == 'D' || c == 'w' || c == 'W' || c == 's' || c == 'S';
}

// Parse the pattern into `rx`. Returns false if it uses unsupported syntax.
static bool rx_compile(const char *pattern, Rx *rx) {
  rx->count = 0;
  const char *p = pattern;
  while (*p != '\0') {
    if (rx->count >= RX_MAX_ATOMS) {
      return false;
    }
    RxAtom atom;
    memset(&atom, 0, sizeof(atom));
    atom.min = 1;
    atom.max = 1;

    char c = *p;
    if (c == '(' || c == ')' || c == '|') {
      return false;  // groups / alternation unsupported
    } else if (c == '^') {
      atom.type = RX_START;
      p++;
    } else if (c == '$') {
      atom.type = RX_END;
      p++;
    } else if (c == '.') {
      atom.type = RX_ANY;
      p++;
    } else if (c == '[') {
      atom.type = RX_CLASS;
      p++;
      bool negate = false;
      if (*p == '^') {
        negate = true;
        p++;
      }
      // A ']' immediately after '[' or '[^' is a literal.
      bool first = true;
      while (*p != '\0' && (*p != ']' || first)) {
        first = false;
        int lo;
        if (*p == '\\') {
          p++;
          if (*p == '\0') {
            return false;
          }
          if (is_shorthand(*p)) {
            class_add_shorthand(atom.set, *p);
            p++;
            continue;
          }
          if (*p == 'p' || *p == 'P') {
            return false;  // Unicode property escapes unsupported
          }
          lo = escape_literal(&p);
        } else {
          lo = (unsigned char)*p++;
        }
        if (*p == '-' && p[1] != ']' && p[1] != '\0') {
          p++;  // range
          int hi;
          if (*p == '\\') {
            p++;
            if (*p == '\0' || is_shorthand(*p)) {
              return false;
            }
            hi = escape_literal(&p);
          } else {
            hi = (unsigned char)*p++;
          }
          if (hi < lo) {
            return false;
          }
          for (int x = lo; x <= hi; x++) {
            atom.set[x] = 1;
          }
        } else {
          atom.set[lo] = 1;
        }
      }
      if (*p != ']') {
        return false;  // unterminated class
      }
      p++;
      if (negate) {
        for (int x = 0; x < 256; x++) {
          atom.set[x] = (unsigned char)(!atom.set[x]);
        }
      }
    } else if (c == '\\') {
      p++;
      if (*p == '\0') {
        return false;
      }
      if (is_shorthand(*p)) {
        atom.type = RX_CLASS;
        class_add_shorthand(atom.set, *p);
        p++;
      } else if (*p == 'p' || *p == 'P' || *p == 'b' || *p == 'B' ||
                 (*p >= '1' && *p <= '9')) {
        return false;  // \p, word boundaries, backreferences unsupported
      } else {
        atom.type = RX_CHAR;
        atom.ch = (unsigned char)escape_literal(&p);
      }
    } else if (c == '*' || c == '+' || c == '?' || c == '{') {
      return false;  // quantifier with no preceding atom
    } else {
      atom.type = RX_CHAR;
      atom.ch = (unsigned char)c;
      p++;
    }

    // Optional quantifier on the atom just parsed (anchors are not quantifiable
    // in any meaningful way; treat a following quantifier as on a zero-width op
    // by leaving min/max=1).
    if (atom.type != RX_START && atom.type != RX_END) {
      if (*p == '*') {
        atom.min = 0;
        atom.max = -1;
        p++;
      } else if (*p == '+') {
        atom.min = 1;
        atom.max = -1;
        p++;
      } else if (*p == '?') {
        atom.min = 0;
        atom.max = 1;
        p++;
      } else if (*p == '{') {
        const char *q = p + 1;
        int lo = 0;
        bool have_lo = false;
        while (*q >= '0' && *q <= '9') {
          lo = lo * 10 + (*q - '0');
          have_lo = true;
          q++;
        }
        int hi = lo;
        bool comma = false;
        if (*q == ',') {
          comma = true;
          q++;
          if (*q >= '0' && *q <= '9') {
            hi = 0;
            while (*q >= '0' && *q <= '9') {
              hi = hi * 10 + (*q - '0');
              q++;
            }
          } else {
            hi = -1;  // {n,}
          }
        }
        if (*q == '}' && have_lo) {
          atom.min = lo;
          atom.max = comma ? hi : lo;
          p = q + 1;
        }
        // A malformed {..} is treated as literal '{' (already stored as RX_CHAR
        // when c=='{' would have returned false above — but here c was an atom,
        // so a stray '{' after it stays literal). Leave p at '{'.
      }
    }

    // A '?' right after a quantifier is the lazy modifier. Greedy vs. lazy does
    // not change whether a match *exists* (only which one), so accept and
    // ignore it for our boolean search.
    if (*p == '?' && (atom.min != 1 || atom.max != 1)) {
      p++;
    }

    rx->atoms[rx->count++] = atom;
  }
  return true;
}

static bool atom_matches(const RxAtom *a, unsigned char c) {
  if (a->type == RX_ANY) {
    return c != '\n';
  }
  if (a->type == RX_CHAR) {
    return c == a->ch;
  }
  if (a->type == RX_CLASS) {
    return a->set[c] != 0;
  }
  return false;
}

static bool rx_match_from(const Rx *rx, size_t ai, const char *text, size_t tlen, size_t pos) {
  if (ai == rx->count) {
    return true;  // all atoms consumed → search match (trailing text is free)
  }
  const RxAtom *a = &rx->atoms[ai];
  if (a->type == RX_START) {
    return pos == 0 && rx_match_from(rx, ai + 1, text, tlen, pos);
  }
  if (a->type == RX_END) {
    return pos == tlen && rx_match_from(rx, ai + 1, text, tlen, pos);
  }
  // Single-character atom with a greedy quantifier [min, max].
  size_t k = 0;
  while ((a->max < 0 || k < (size_t)a->max) && pos + k < tlen &&
         atom_matches(a, (unsigned char)text[pos + k])) {
    k++;
  }
  if (k < (size_t)a->min) {
    return false;
  }
  for (size_t c = k;; c--) {
    if (rx_match_from(rx, ai + 1, text, tlen, pos + c)) {
      return true;
    }
    if (c == (size_t)a->min) {
      break;
    }
  }
  return false;
}

bool tf_regex_search(const char *pattern, const char *text, size_t text_len, bool *supported) {
  Rx rx;
  if (!rx_compile(pattern, &rx)) {
    if (supported != 0) {
      *supported = false;
    }
    return false;
  }
  if (supported != 0) {
    *supported = true;
  }
  // Unanchored search: try matching from each start position (and once past the
  // end so that an empty/`$`-only pattern can match the empty tail).
  for (size_t start = 0; start <= text_len; start++) {
    if (rx_match_from(&rx, 0, text, text_len, start)) {
      return true;
    }
  }
  return false;
}

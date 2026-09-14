#include "toyforge/json.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *tf_json_skip_ws(const char *p) {
  while (*p && isspace((unsigned char)*p)) {
    p++;
  }
  return p;
}

static int hex_value(char c) {
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

static bool read_hex4(const char **cursor, unsigned int *out) {
  const char *p = *cursor;
  unsigned int value = 0;
  for (int i = 0; i < 4; i++) {
    int h = hex_value(p[i]);
    if (h < 0) {
      return false;
    }
    value = (value << 4) | (unsigned int)h;
  }
  *cursor = p + 4;
  *out = value;
  return true;
}

static bool is_utf8_cont(unsigned char c) {
  return c >= 0x80 && c <= 0xBF;
}

static bool raw_utf8_len(const unsigned char *p, size_t *len) {
  unsigned char c = p[0];
  if (c < 0x80) {
    if (c < 0x20) {
      return false;
    }
    *len = 1;
    return true;
  }
  if (c >= 0xC2 && c <= 0xDF) {
    if (!is_utf8_cont(p[1])) {
      return false;
    }
    *len = 2;
    return true;
  }
  if (c == 0xE0) {
    if (!(p[1] >= 0xA0 && p[1] <= 0xBF) || !is_utf8_cont(p[2])) {
      return false;
    }
    *len = 3;
    return true;
  }
  if (c >= 0xE1 && c <= 0xEC) {
    if (!is_utf8_cont(p[1]) || !is_utf8_cont(p[2])) {
      return false;
    }
    *len = 3;
    return true;
  }
  if (c == 0xED) {
    if (!(p[1] >= 0x80 && p[1] <= 0x9F) || !is_utf8_cont(p[2])) {
      return false;
    }
    *len = 3;
    return true;
  }
  if (c >= 0xEE && c <= 0xEF) {
    if (!is_utf8_cont(p[1]) || !is_utf8_cont(p[2])) {
      return false;
    }
    *len = 3;
    return true;
  }
  if (c == 0xF0) {
    if (!(p[1] >= 0x90 && p[1] <= 0xBF) || !is_utf8_cont(p[2]) ||
        !is_utf8_cont(p[3])) {
      return false;
    }
    *len = 4;
    return true;
  }
  if (c >= 0xF1 && c <= 0xF3) {
    if (!is_utf8_cont(p[1]) || !is_utf8_cont(p[2]) || !is_utf8_cont(p[3])) {
      return false;
    }
    *len = 4;
    return true;
  }
  if (c == 0xF4) {
    if (!(p[1] >= 0x80 && p[1] <= 0x8F) || !is_utf8_cont(p[2]) ||
        !is_utf8_cont(p[3])) {
      return false;
    }
    *len = 4;
    return true;
  }
  return false;
}

static bool append_bytes(
  char *out,
  size_t out_cap,
  size_t *out_i,
  const char *bytes,
  size_t len
) {
  if (len > (size_t)-1 - *out_i) {
    return false;
  }
  if (out == 0) {
    *out_i += len;
    return true;
  }
  if (*out_i + len >= out_cap) {
    return false;
  }
  memcpy(out + *out_i, bytes, len);
  *out_i += len;
  out[*out_i] = '\0';
  return true;
}

static bool append_byte(char *out, size_t out_cap, size_t *out_i, unsigned char byte) {
  char c = (char)byte;
  return append_bytes(out, out_cap, out_i, &c, 1);
}

static bool append_codepoint(char *out, size_t out_cap, size_t *out_i, unsigned int cp) {
  unsigned char bytes[4];
  size_t len = 0;
  if (cp <= 0x7F) {
    bytes[0] = (unsigned char)cp;
    len = 1;
  } else if (cp <= 0x7FF) {
    bytes[0] = (unsigned char)(0xC0 | (cp >> 6));
    bytes[1] = (unsigned char)(0x80 | (cp & 0x3F));
    len = 2;
  } else if (cp >= 0xD800 && cp <= 0xDFFF) {
    return false;
  } else if (cp <= 0xFFFF) {
    bytes[0] = (unsigned char)(0xE0 | (cp >> 12));
    bytes[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    bytes[2] = (unsigned char)(0x80 | (cp & 0x3F));
    len = 3;
  } else if (cp <= 0x10FFFF) {
    bytes[0] = (unsigned char)(0xF0 | (cp >> 18));
    bytes[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    bytes[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    bytes[3] = (unsigned char)(0x80 | (cp & 0x3F));
    len = 4;
  } else {
    return false;
  }
  return append_bytes(out, out_cap, out_i, (const char *)bytes, len);
}

static bool read_json_string_impl(
  const char **cursor,
  char *out,
  size_t out_cap,
  size_t *decoded_size
) {
  const char *p = tf_json_skip_ws(*cursor);
  if (*p != '"') {
    return false;
  }
  if (out != 0 && out_cap == 0) {
    return false;
  }
  p++;
  size_t out_i = 0;
  while (*p && *p != '"') {
    unsigned char c = (unsigned char)*p++;
    if (c == '\\') {
      if (*p == '\0') {
        return false;
      }
      c = (unsigned char)*p++;
      switch (c) {
        case '"':
        case '\\':
        case '/':
          if (!append_byte(out, out_cap, &out_i, c)) {
            return false;
          }
          break;
        case 'b':
          if (!append_byte(out, out_cap, &out_i, '\b')) {
            return false;
          }
          break;
        case 'f':
          if (!append_byte(out, out_cap, &out_i, '\f')) {
            return false;
          }
          break;
        case 'n':
          if (!append_byte(out, out_cap, &out_i, '\n')) {
            return false;
          }
          break;
        case 'r':
          if (!append_byte(out, out_cap, &out_i, '\r')) {
            return false;
          }
          break;
        case 't':
          if (!append_byte(out, out_cap, &out_i, '\t')) {
            return false;
          }
          break;
        case 'u': {
          unsigned int high = 0;
          if (!read_hex4(&p, &high)) {
            return false;
          }
          if (high >= 0xD800 && high <= 0xDBFF) {
            unsigned int low = 0;
            if (p[0] != '\\' || p[1] != 'u') {
              return false;
            }
            p += 2;
            if (!read_hex4(&p, &low) || low < 0xDC00 || low > 0xDFFF) {
              return false;
            }
            high = 0x10000 + (((high - 0xD800) << 10) | (low - 0xDC00));
          } else if (high >= 0xDC00 && high <= 0xDFFF) {
            return false;
          }
          if (!append_codepoint(out, out_cap, &out_i, high)) {
            return false;
          }
          break;
        }
        default:
          return false;
      }
    } else if (c < 0x20) {
      return false;
    } else {
      size_t len = 0;
      p--;
      if (!raw_utf8_len((const unsigned char *)p, &len)) {
        return false;
      }
      if (!append_bytes(out, out_cap, &out_i, p, len)) {
        return false;
      }
      p += len;
    }
  }
  if (*p != '"') {
    return false;
  }
  if (out != 0) {
    out[out_i] = '\0';
  }
  if (decoded_size != 0) {
    *decoded_size = out_i;
  }
  *cursor = p + 1;
  return true;
}

bool tf_json_read_string(const char **cursor, char *out, size_t out_cap) {
  if (out == 0 || out_cap == 0) {
    return false;
  }
  out[0] = '\0';
  return read_json_string_impl(cursor, out, out_cap, 0);
}

bool tf_json_string_decoded_size(TfJsonSpan string, size_t *out_size) {
  if (out_size == 0) {
    return false;
  }
  const char *p = string.start;
  size_t size = 0;
  if (!read_json_string_impl(&p, 0, 0, &size) || p != string.end) {
    return false;
  }
  *out_size = size;
  return true;
}

bool tf_json_write_string(const char *input, char *out, size_t out_cap) {
  if (input == 0 || out == 0 || out_cap == 0) {
    return false;
  }
  out[0] = '\0';
  size_t out_i = 0;
  if (!append_byte(out, out_cap, &out_i, '"')) {
    return false;
  }
  const unsigned char *p = (const unsigned char *)input;
  while (*p) {
    unsigned char c = *p;
    switch (c) {
      case '"':
        if (!append_bytes(out, out_cap, &out_i, "\\\"", 2)) {
          return false;
        }
        p++;
        break;
      case '\\':
        if (!append_bytes(out, out_cap, &out_i, "\\\\", 2)) {
          return false;
        }
        p++;
        break;
      case '\b':
        if (!append_bytes(out, out_cap, &out_i, "\\b", 2)) {
          return false;
        }
        p++;
        break;
      case '\f':
        if (!append_bytes(out, out_cap, &out_i, "\\f", 2)) {
          return false;
        }
        p++;
        break;
      case '\n':
        if (!append_bytes(out, out_cap, &out_i, "\\n", 2)) {
          return false;
        }
        p++;
        break;
      case '\r':
        if (!append_bytes(out, out_cap, &out_i, "\\r", 2)) {
          return false;
        }
        p++;
        break;
      case '\t':
        if (!append_bytes(out, out_cap, &out_i, "\\t", 2)) {
          return false;
        }
        p++;
        break;
      default:
        if (c < 0x20) {
          char esc[7];
          snprintf(esc, sizeof(esc), "\\u%04x", (unsigned int)c);
          if (!append_bytes(out, out_cap, &out_i, esc, 6)) {
            return false;
          }
          p++;
        } else {
          size_t len = 0;
          if (!raw_utf8_len(p, &len)) {
            return false;
          }
          if (!append_bytes(out, out_cap, &out_i, (const char *)p, len)) {
            return false;
          }
          p += len;
        }
        break;
    }
  }
  if (!append_byte(out, out_cap, &out_i, '"')) {
    return false;
  }
  out[out_i] = '\0';
  return true;
}

static bool skip_json_string(const char **cursor) {
  return read_json_string_impl(cursor, 0, 0, 0);
}

static bool skip_number(const char **cursor) {
  const char *p = *cursor;
  if (*p == '-') {
    p++;
  }
  if (*p == '0') {
    p++;
  } else if (*p >= '1' && *p <= '9') {
    p++;
    while (isdigit((unsigned char)*p)) {
      p++;
    }
  } else {
    return false;
  }
  if (*p == '.') {
    p++;
    if (!isdigit((unsigned char)*p)) {
      return false;
    }
    while (isdigit((unsigned char)*p)) {
      p++;
    }
  }
  if (*p == 'e' || *p == 'E') {
    p++;
    if (*p == '+' || *p == '-') {
      p++;
    }
    if (!isdigit((unsigned char)*p)) {
      return false;
    }
    while (isdigit((unsigned char)*p)) {
      p++;
    }
  }
  *cursor = p;
  return true;
}

// Maximum object/array nesting the recursive value scanner will descend before
// failing closed. The deepest legitimate JSON in this project (schemas, the
// official JSON Schema suite, trajectory data) nests ~10 levels, so 1000 never
// rejects real input while bounding recursion far below a stack overflow on
// adversarial deeply-nested input — matching Python's json.loads, which raises
// RecursionError rather than crashing. Without this bound, input like
// "[[[[..." crashes the parser (and every consumer: jsonschema, verifier, ...).
#define TF_JSON_MAX_DEPTH 1000

static bool skip_value_depth(const char **cursor, int depth);

static bool skip_object_depth(const char **cursor, int depth) {
  const char *p = tf_json_skip_ws(*cursor);
  if (*p != '{') {
    return false;
  }
  p++;
  p = tf_json_skip_ws(p);
  if (*p == '}') {
    *cursor = p + 1;
    return true;
  }
  while (true) {
    if (!skip_json_string(&p)) {
      return false;
    }
    p = tf_json_skip_ws(p);
    if (*p != ':') {
      return false;
    }
    p++;
    if (!skip_value_depth(&p, depth)) {
      return false;
    }
    p = tf_json_skip_ws(p);
    if (*p == ',') {
      p++;
      p = tf_json_skip_ws(p);
      continue;
    }
    if (*p == '}') {
      *cursor = p + 1;
      return true;
    }
    return false;
  }
}

static bool skip_array_depth(const char **cursor, int depth) {
  const char *p = tf_json_skip_ws(*cursor);
  if (*p != '[') {
    return false;
  }
  p++;
  p = tf_json_skip_ws(p);
  if (*p == ']') {
    *cursor = p + 1;
    return true;
  }
  while (true) {
    if (!skip_value_depth(&p, depth)) {
      return false;
    }
    p = tf_json_skip_ws(p);
    if (*p == ',') {
      p++;
      p = tf_json_skip_ws(p);
      continue;
    }
    if (*p == ']') {
      *cursor = p + 1;
      return true;
    }
    return false;
  }
}

static bool skip_value_depth(const char **cursor, int depth) {
  if (depth > TF_JSON_MAX_DEPTH) {
    return false;
  }
  const char *p = tf_json_skip_ws(*cursor);
  if (*p == '"') {
    if (!skip_json_string(&p)) {
      return false;
    }
    *cursor = p;
    return true;
  }
  if (*p == '{') {
    return skip_object_depth(cursor, depth + 1);
  }
  if (*p == '[') {
    return skip_array_depth(cursor, depth + 1);
  }
  if (*p == '-' || isdigit((unsigned char)*p)) {
    if (!skip_number(&p)) {
      return false;
    }
    *cursor = p;
    return true;
  }
  if (strncmp(p, "true", 4) == 0) {
    *cursor = p + 4;
    return true;
  }
  if (strncmp(p, "false", 5) == 0) {
    *cursor = p + 5;
    return true;
  }
  if (strncmp(p, "null", 4) == 0) {
    *cursor = p + 4;
    return true;
  }
  return false;
}

bool tf_json_skip_value(const char **cursor) {
  return skip_value_depth(cursor, 0);
}

static const char *skip_ws_until(const char *p, const char *end) {
  while (p < end && isspace((unsigned char)*p)) {
    p++;
  }
  return p;
}

static bool array_span_is_valid(TfJsonSpan array) {
  const char *p = skip_ws_until(array.start, array.end);
  if (p >= array.end || *p != '[') {
    return false;
  }
  if (!tf_json_skip_value(&p)) {
    return false;
  }
  return skip_ws_until(p, array.end) == array.end;
}

static bool array_advance(
  const char **cursor,
  const char *end,
  TfJsonSpan *value,
  bool *has_value,
  bool *finished
) {
  const char *p = skip_ws_until(*cursor, end);
  *has_value = false;
  *finished = false;
  if (p >= end) {
    return false;
  }
  if (*p == ']') {
    *cursor = p + 1;
    *finished = true;
    return true;
  }

  const char *value_start = p;
  const char *value_end = value_start;
  if (!tf_json_skip_value(&value_end) || value_end > end) {
    return false;
  }
  value->start = value_start;
  value->end = value_end;
  *has_value = true;

  p = skip_ws_until(value_end, end);
  if (p >= end) {
    return false;
  }
  if (*p == ',') {
    *cursor = p + 1;
    return true;
  }
  if (*p == ']') {
    *cursor = p + 1;
    *finished = true;
    return true;
  }
  return false;
}

bool tf_json_array_count(TfJsonSpan array, size_t *count) {
  if (count == 0) {
    return false;
  }
  *count = 0;
  if (!array_span_is_valid(array)) {
    return false;
  }
  const char *p = skip_ws_until(array.start, array.end);
  p++;
  while (true) {
    TfJsonSpan value;
    bool has_value = false;
    bool finished = false;
    if (!array_advance(&p, array.end, &value, &has_value, &finished)) {
      return false;
    }
    if (has_value) {
      (*count)++;
    }
    if (finished) {
      return skip_ws_until(p, array.end) == array.end;
    }
  }
}

bool tf_json_array_get(TfJsonSpan array, size_t index, TfJsonSpan *value) {
  if (value == 0) {
    return false;
  }
  if (!array_span_is_valid(array)) {
    return false;
  }
  const char *p = skip_ws_until(array.start, array.end);
  p++;
  size_t current = 0;
  while (true) {
    TfJsonSpan item;
    bool has_value = false;
    bool finished = false;
    if (!array_advance(&p, array.end, &item, &has_value, &finished)) {
      return false;
    }
    if (!has_value) {
      return false;
    }
    if (current == index) {
      *value = item;
      return true;
    }
    if (finished) {
      return false;
    }
    current++;
  }
}

bool tf_json_object_get_value(TfJsonSpan object, const char *key, TfJsonSpan *value) {
  const char *p = tf_json_skip_ws(object.start);
  bool found = false;
  TfJsonSpan last_value = {0};
  if (p >= object.end || *p != '{') {
    return false;
  }
  p++;
  while (true) {
    p = tf_json_skip_ws(p);
    if (p >= object.end) {
      return false;
    }
    if (*p == '}') {
      return false;
    }
    char found_key[128];
    if (!tf_json_read_string(&p, found_key, sizeof(found_key))) {
      return false;
    }
    p = tf_json_skip_ws(p);
    if (*p != ':') {
      return false;
    }
    p++;
    const char *value_start = tf_json_skip_ws(p);
    p = value_start;
    if (!tf_json_skip_value(&p)) {
      return false;
    }
    if (strcmp(found_key, key) == 0) {
      last_value.start = value_start;
      last_value.end = p;
      found = true;
    }
    p = tf_json_skip_ws(p);
    if (*p == ',') {
      p++;
      continue;
    }
    if (*p == '}') {
      if (found) {
        *value = last_value;
      }
      return found;
    }
    return false;
  }
}

bool tf_json_object_get_string(TfJsonSpan object, const char *key, char *out, size_t out_cap) {
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value)) {
    return false;
  }
  const char *p = value.start;
  if (!tf_json_read_string(&p, out, out_cap)) {
    return false;
  }
  return p == value.end;
}

bool tf_json_object_get_object(TfJsonSpan object, const char *key, TfJsonSpan *out) {
  TfJsonSpan value;
  if (!tf_json_object_get_value(object, key, &value)) {
    return false;
  }
  const char *p = tf_json_skip_ws(value.start);
  if (*p != '{') {
    return false;
  }
  out->start = p;
  out->end = value.end;
  return true;
}

bool tf_json_object_has_key(TfJsonSpan object, const char *key) {
  TfJsonSpan ignored;
  return tf_json_object_get_value(object, key, &ignored);
}

TfJsonType tf_json_type(TfJsonSpan span) {
  const char *p = tf_json_skip_ws(span.start);
  if (*p == '\0') {
    return TF_JSON_INVALID;
  }
  switch (*p) {
    case '{':
      return TF_JSON_OBJECT;
    case '[':
      return TF_JSON_ARRAY;
    case '"':
      return TF_JSON_STRING;
    case 't':
    case 'f':
      return TF_JSON_BOOL;
    case 'n':
      return TF_JSON_NULL;
    default:
      if (*p == '-' || (*p >= '0' && *p <= '9')) {
        return TF_JSON_NUMBER;
      }
      return TF_JSON_INVALID;
  }
}

bool tf_json_number(TfJsonSpan span, double *out) {
  if (tf_json_type(span) != TF_JSON_NUMBER) {
    return false;
  }
  const char *p = tf_json_skip_ws(span.start);
  char *end = 0;
  double v = strtod(p, &end);
  if (end == p) {
    return false;
  }
  if (out != 0) {
    *out = v;
  }
  return true;
}

bool tf_json_is_integer(TfJsonSpan span) {
  // Text-first: a plain integer literal (optional sign + digits, no '.', 'e' or
  // 'E') denotes an integer at any magnitude, so classify it from the token
  // rather than from a double that cannot represent values beyond ~2^53/long long.
  const char *p = tf_json_skip_ws(span.start);
  const char *q = p;
  if (q < span.end && (*q == '-' || *q == '+')) {
    q++;
  }
  bool has_digit = false;
  bool fractional_or_exp = false;
  for (; q < span.end; q++) {
    char c = *q;
    if (c >= '0' && c <= '9') {
      has_digit = true;
    } else if (c == '.' || c == 'e' || c == 'E') {
      fractional_or_exp = true;
      break;
    } else {
      break;  // end of the number token (whitespace, etc.)
    }
  }
  if (has_digit && !fractional_or_exp) {
    return true;  // pure integer literal — integral at any magnitude
  }
  // Has a fraction/exponent (e.g. 1.0, 1.5e3): fall back to the numeric check,
  // bounded to the exactly-representable integral range.
  double v = 0.0;
  if (!tf_json_number(span, &v)) {
    return false;
  }
  if (v < -9.0e18 || v > 9.0e18) {
    return false;  // outside the exactly-representable integral range we check
  }
  return v == (double)(long long)v;
}

static bool string_decode_alloc(TfJsonSpan span, char **out, size_t *len) {
  size_t size = 0;
  if (!tf_json_string_decoded_size(span, &size)) {
    return false;
  }
  char *buf = (char *)malloc(size + 1);
  if (buf == 0) {
    return false;
  }
  const char *p = tf_json_skip_ws(span.start);
  if (!tf_json_read_string(&p, buf, size + 1)) {
    free(buf);
    return false;
  }
  *out = buf;
  *len = size;
  return true;
}

bool tf_json_equal(TfJsonSpan a, TfJsonSpan b) {
  TfJsonType ta = tf_json_type(a);
  TfJsonType tb = tf_json_type(b);
  if (ta != tb || ta == TF_JSON_INVALID) {
    return false;
  }
  switch (ta) {
    case TF_JSON_NULL:
      return true;
    case TF_JSON_BOOL: {
      const char *pa = tf_json_skip_ws(a.start);
      const char *pb = tf_json_skip_ws(b.start);
      return *pa == *pb;  // 't' or 'f'
    }
    case TF_JSON_NUMBER: {
      double va = 0.0;
      double vb = 0.0;
      return tf_json_number(a, &va) && tf_json_number(b, &vb) && va == vb;
    }
    case TF_JSON_STRING: {
      char *sa = 0;
      char *sb = 0;
      size_t la = 0;
      size_t lb = 0;
      const char *pa = tf_json_skip_ws(a.start);
      const char *pb = tf_json_skip_ws(b.start);
      TfJsonSpan na = {pa, a.end};
      TfJsonSpan nb = {pb, b.end};
      if (!string_decode_alloc(na, &sa, &la)) {
        return false;
      }
      if (!string_decode_alloc(nb, &sb, &lb)) {
        free(sa);
        return false;
      }
      bool eq = la == lb && memcmp(sa, sb, la) == 0;
      free(sa);
      free(sb);
      return eq;
    }
    case TF_JSON_ARRAY: {
      size_t ca = 0;
      size_t cb = 0;
      if (!tf_json_array_count(a, &ca) || !tf_json_array_count(b, &cb) || ca != cb) {
        return false;
      }
      for (size_t i = 0; i < ca; i++) {
        TfJsonSpan ea;
        TfJsonSpan eb;
        if (!tf_json_array_get(a, i, &ea) || !tf_json_array_get(b, i, &eb) ||
            !tf_json_equal(ea, eb)) {
          return false;
        }
      }
      return true;
    }
    case TF_JSON_OBJECT: {
#define TF_JSON_EQ_MAX_KEYS 128
      char keys_a[TF_JSON_EQ_MAX_KEYS][96];
      char keys_b[TF_JSON_EQ_MAX_KEYS][96];
      size_t na = 0;
      size_t nb = 0;
      if (!tf_json_collect_object_keys(a, keys_a, TF_JSON_EQ_MAX_KEYS, &na) ||
          !tf_json_collect_object_keys(b, keys_b, TF_JSON_EQ_MAX_KEYS, &nb) || na != nb) {
        return false;
      }
#undef TF_JSON_EQ_MAX_KEYS
      for (size_t i = 0; i < na; i++) {
        TfJsonSpan va;
        TfJsonSpan vb;
        if (!tf_json_object_get_value(a, keys_a[i], &va) ||
            !tf_json_object_get_value(b, keys_a[i], &vb) || !tf_json_equal(va, vb)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

bool tf_json_collect_object_keys(
  TfJsonSpan object,
  char keys[][96],
  size_t max_keys,
  size_t *count
) {
  const char *p = tf_json_skip_ws(object.start);
  *count = 0;
  if (p >= object.end || *p != '{') {
    return false;
  }
  p++;
  while (true) {
    p = tf_json_skip_ws(p);
    if (p >= object.end) {
      return false;
    }
    if (*p == '}') {
      return true;
    }
    if (*count >= max_keys) {
      return false;
    }
    if (!tf_json_read_string(&p, keys[*count], 96)) {
      return false;
    }
    (*count)++;
    p = tf_json_skip_ws(p);
    if (*p != ':') {
      return false;
    }
    p++;
    if (!tf_json_skip_value(&p)) {
      return false;
    }
    p = tf_json_skip_ws(p);
    if (*p == ',') {
      p++;
      continue;
    }
    if (*p == '}') {
      return true;
    }
    return false;
  }
}

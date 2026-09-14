#include "toyforge/yaml.h"

#include "toyforge/io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Growable byte buffer
// ---------------------------------------------------------------------------

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} Buf;

static bool buf_reserve(Buf *b, size_t extra) {
  if (b->len + extra + 1 <= b->cap) {
    return true;
  }
  size_t cap = b->cap ? b->cap : 64;
  while (cap < b->len + extra + 1) {
    cap *= 2;
  }
  char *data = (char *)realloc(b->data, cap);
  if (data == 0) {
    return false;
  }
  b->data = data;
  b->cap = cap;
  return true;
}

static bool buf_putn(Buf *b, const char *s, size_t n) {
  if (!buf_reserve(b, n)) {
    return false;
  }
  memcpy(b->data + b->len, s, n);
  b->len += n;
  b->data[b->len] = '\0';
  return true;
}

static bool buf_puts(Buf *b, const char *s) {
  return buf_putn(b, s, strlen(s));
}

static bool buf_putc(Buf *b, char c) {
  return buf_putn(b, &c, 1);
}

// ---------------------------------------------------------------------------
// Node constructors / destructor
// ---------------------------------------------------------------------------

static TfYamlNode *node_new(TfYamlType type) {
  TfYamlNode *n = (TfYamlNode *)calloc(1, sizeof(TfYamlNode));
  if (n != 0) {
    n->type = type;
  }
  return n;
}

void tf_yaml_free(TfYamlNode *node) {
  if (node == 0) {
    return;
  }
  switch (node->type) {
    case TF_YAML_STRING:
      free(node->as.string);
      break;
    case TF_YAML_SEQUENCE:
      for (size_t i = 0; i < node->as.seq.count; i++) {
        tf_yaml_free(node->as.seq.items[i]);
      }
      free(node->as.seq.items);
      break;
    case TF_YAML_MAPPING:
      for (size_t i = 0; i < node->as.map.count; i++) {
        free(node->as.map.keys[i]);
        tf_yaml_free(node->as.map.values[i]);
      }
      free(node->as.map.keys);
      free(node->as.map.values);
      break;
    default:
      break;
  }
  free(node);
}

static TfYamlNode *node_string_owned(char *owned) {
  TfYamlNode *n = node_new(TF_YAML_STRING);
  if (n == 0) {
    free(owned);
    return 0;
  }
  n->as.string = owned;
  return n;
}

static char *dup_range(const char *start, size_t len) {
  char *s = (char *)malloc(len + 1);
  if (s == 0) {
    return 0;
  }
  memcpy(s, start, len);
  s[len] = '\0';
  return s;
}

static bool seq_append(TfYamlNode *seq, TfYamlNode *item) {
  TfYamlNode **items =
    (TfYamlNode **)realloc(seq->as.seq.items, (seq->as.seq.count + 1) * sizeof(TfYamlNode *));
  if (items == 0) {
    return false;
  }
  seq->as.seq.items = items;
  seq->as.seq.items[seq->as.seq.count++] = item;
  return true;
}

static bool map_append(TfYamlNode *map, char *key, TfYamlNode *value) {
  char **keys = (char **)realloc(map->as.map.keys, (map->as.map.count + 1) * sizeof(char *));
  if (keys == 0) {
    return false;
  }
  map->as.map.keys = keys;
  TfYamlNode **values =
    (TfYamlNode **)realloc(map->as.map.values, (map->as.map.count + 1) * sizeof(TfYamlNode *));
  if (values == 0) {
    return false;
  }
  map->as.map.values = values;
  map->as.map.keys[map->as.map.count] = key;
  map->as.map.values[map->as.map.count] = value;
  map->as.map.count++;
  return true;
}

// ---------------------------------------------------------------------------
// Scalar typing (PyYAML core schema subset)
// ---------------------------------------------------------------------------

static bool str_is_int(const char *s) {
  if (*s == '+' || *s == '-') {
    s++;
  }
  if (*s == '\0') {
    return false;
  }
  for (; *s; s++) {
    if (!isdigit((unsigned char)*s)) {
      return false;
    }
  }
  return true;
}

static bool str_is_float(const char *s) {
  const char *p = s;
  if (*p == '+' || *p == '-') {
    p++;
  }
  bool has_digit = false;
  bool has_dot = false;
  bool has_exp = false;
  for (; *p; p++) {
    if (isdigit((unsigned char)*p)) {
      has_digit = true;
    } else if (*p == '.' && !has_dot && !has_exp) {
      has_dot = true;
    } else if ((*p == 'e' || *p == 'E') && has_digit && !has_exp) {
      has_exp = true;
      if (p[1] == '+' || p[1] == '-') {
        p++;
      }
    } else {
      return false;
    }
  }
  return has_digit && (has_dot || has_exp);
}

static TfYamlNode *resolve_plain_scalar(const char *text, size_t len) {
  // Trim surrounding whitespace.
  while (len > 0 && (text[0] == ' ' || text[0] == '\t')) {
    text++;
    len--;
  }
  while (len > 0 && (text[len - 1] == ' ' || text[len - 1] == '\t')) {
    len--;
  }
  char *s = dup_range(text, len);
  if (s == 0) {
    return 0;
  }
  TfYamlNode *n = 0;
  if (s[0] == '\0' || strcmp(s, "~") == 0 || strcmp(s, "null") == 0 || strcmp(s, "Null") == 0 ||
      strcmp(s, "NULL") == 0) {
    n = node_new(TF_YAML_NULL);
  } else if (strcmp(s, "true") == 0 || strcmp(s, "True") == 0 || strcmp(s, "TRUE") == 0) {
    n = node_new(TF_YAML_BOOL);
    if (n) {
      n->as.boolean = true;
    }
  } else if (strcmp(s, "false") == 0 || strcmp(s, "False") == 0 || strcmp(s, "FALSE") == 0) {
    n = node_new(TF_YAML_BOOL);
    if (n) {
      n->as.boolean = false;
    }
  } else if (str_is_int(s)) {
    n = node_new(TF_YAML_INT);
    if (n) {
      n->as.integer = strtoll(s, 0, 10);
    }
  } else if (str_is_float(s)) {
    n = node_new(TF_YAML_FLOAT);
    if (n) {
      n->as.number = strtod(s, 0);
    }
  } else {
    free(s);
    return node_string_owned(dup_range(text, len));
  }
  free(s);
  return n;
}

// ---------------------------------------------------------------------------
// Quoted scalar decoding
// ---------------------------------------------------------------------------

static int hex_digit(char c) {
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

static bool encode_utf8(Buf *b, unsigned int cp) {
  if (cp <= 0x7F) {
    return buf_putc(b, (char)cp);
  }
  if (cp <= 0x7FF) {
    return buf_putc(b, (char)(0xC0 | (cp >> 6))) && buf_putc(b, (char)(0x80 | (cp & 0x3F)));
  }
  if (cp <= 0xFFFF) {
    return buf_putc(b, (char)(0xE0 | (cp >> 12))) &&
           buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F))) &&
           buf_putc(b, (char)(0x80 | (cp & 0x3F)));
  }
  return buf_putc(b, (char)(0xF0 | (cp >> 18))) &&
         buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3F))) &&
         buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3F))) && buf_putc(b, (char)(0x80 | (cp & 0x3F)));
}

// Parse a quoted scalar at *p (which points at the opening quote). On success
// returns a string node and advances *p past the closing quote.
static TfYamlNode *parse_quoted(const char **p, char *err, size_t err_cap) {
  char quote = **p;
  (*p)++;
  Buf b = {0};
  while (**p != '\0') {
    char c = **p;
    if (c == quote) {
      if (quote == '\'' && (*p)[1] == '\'') {  // '' -> literal '
        if (!buf_putc(&b, '\'')) {
          goto oom;
        }
        *p += 2;
        continue;
      }
      (*p)++;  // consume closing quote
      TfYamlNode *n = node_string_owned(b.data ? b.data : dup_range("", 0));
      return n;
    }
    if (quote == '"' && c == '\\') {
      char e = (*p)[1];
      *p += 2;
      switch (e) {
        case 'n':
          if (!buf_putc(&b, '\n')) goto oom;
          break;
        case 't':
          if (!buf_putc(&b, '\t')) goto oom;
          break;
        case 'r':
          if (!buf_putc(&b, '\r')) goto oom;
          break;
        case '0':
          if (!buf_putc(&b, '\0')) goto oom;
          break;
        case '"':
        case '\\':
        case '/':
          if (!buf_putc(&b, e)) goto oom;
          break;
        case 'u': {
          unsigned int cp = 0;
          for (int i = 0; i < 4; i++) {
            int h = hex_digit((*p)[i]);
            if (h < 0) {
              tf_copy_cstr(err, err_cap, "invalid \\u escape in quoted string");
              free(b.data);
              return 0;
            }
            cp = (cp << 4) | (unsigned int)h;
          }
          *p += 4;
          if (!encode_utf8(&b, cp)) goto oom;
          break;
        }
        default:
          if (!buf_putc(&b, e)) goto oom;
          break;
      }
      continue;
    }
    if (!buf_putc(&b, c)) {
      goto oom;
    }
    (*p)++;
  }
  tf_copy_cstr(err, err_cap, "unterminated quoted string");
  free(b.data);
  return 0;
oom:
  tf_copy_cstr(err, err_cap, "out of memory parsing quoted string");
  free(b.data);
  return 0;
}

// ---------------------------------------------------------------------------
// Flow parsing  ({a: b}, [a, b])
// ---------------------------------------------------------------------------

static const char *skip_ws(const char *p) {
  while (*p == ' ' || *p == '\t') {
    p++;
  }
  return p;
}

static TfYamlNode *flow_value(const char **p, char *err, size_t err_cap);

static TfYamlNode *flow_mapping(const char **p, char *err, size_t err_cap) {
  TfYamlNode *map = node_new(TF_YAML_MAPPING);
  if (map == 0) {
    return 0;
  }
  (*p)++;  // consume '{'
  *p = skip_ws(*p);
  if (**p == '}') {
    (*p)++;
    return map;
  }
  for (;;) {
    *p = skip_ws(*p);
    // key: quoted or plain up to ':'
    char *key = 0;
    if (**p == '"' || **p == '\'') {
      TfYamlNode *k = parse_quoted(p, err, err_cap);
      if (k == 0) {
        tf_yaml_free(map);
        return 0;
      }
      key = k->as.string;
      k->as.string = 0;
      tf_yaml_free(k);
    } else {
      const char *start = *p;
      while (**p != '\0' && **p != ':') {
        (*p)++;
      }
      const char *end = *p;
      while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
        end--;
      }
      key = dup_range(start, (size_t)(end - start));
    }
    if (key == 0) {
      tf_yaml_free(map);
      tf_copy_cstr(err, err_cap, "out of memory parsing flow key");
      return 0;
    }
    *p = skip_ws(*p);
    if (**p != ':') {
      free(key);
      tf_yaml_free(map);
      tf_copy_cstr(err, err_cap, "expected ':' in flow mapping");
      return 0;
    }
    (*p)++;
    TfYamlNode *value = flow_value(p, err, err_cap);
    if (value == 0) {
      free(key);
      tf_yaml_free(map);
      return 0;
    }
    if (!map_append(map, key, value)) {
      free(key);
      tf_yaml_free(value);
      tf_yaml_free(map);
      tf_copy_cstr(err, err_cap, "out of memory building flow mapping");
      return 0;
    }
    *p = skip_ws(*p);
    if (**p == ',') {
      (*p)++;
      continue;
    }
    if (**p == '}') {
      (*p)++;
      return map;
    }
    tf_yaml_free(map);
    tf_copy_cstr(err, err_cap, "expected ',' or '}' in flow mapping");
    return 0;
  }
}

static TfYamlNode *flow_sequence(const char **p, char *err, size_t err_cap) {
  TfYamlNode *seq = node_new(TF_YAML_SEQUENCE);
  if (seq == 0) {
    return 0;
  }
  (*p)++;  // consume '['
  *p = skip_ws(*p);
  if (**p == ']') {
    (*p)++;
    return seq;
  }
  for (;;) {
    TfYamlNode *value = flow_value(p, err, err_cap);
    if (value == 0) {
      tf_yaml_free(seq);
      return 0;
    }
    if (!seq_append(seq, value)) {
      tf_yaml_free(value);
      tf_yaml_free(seq);
      tf_copy_cstr(err, err_cap, "out of memory building flow sequence");
      return 0;
    }
    *p = skip_ws(*p);
    if (**p == ',') {
      (*p)++;
      *p = skip_ws(*p);
      continue;
    }
    if (**p == ']') {
      (*p)++;
      return seq;
    }
    tf_yaml_free(seq);
    tf_copy_cstr(err, err_cap, "expected ',' or ']' in flow sequence");
    return 0;
  }
}

static TfYamlNode *flow_value(const char **p, char *err, size_t err_cap) {
  *p = skip_ws(*p);
  if (**p == '{') {
    return flow_mapping(p, err, err_cap);
  }
  if (**p == '[') {
    return flow_sequence(p, err, err_cap);
  }
  if (**p == '"' || **p == '\'') {
    return parse_quoted(p, err, err_cap);
  }
  const char *start = *p;
  while (**p != '\0' && **p != ',' && **p != ']' && **p != '}') {
    (*p)++;
  }
  return resolve_plain_scalar(start, (size_t)(*p - start));
}

// Parse a value that begins inline after "key:" / "- ": flow, quoted, or plain.
static TfYamlNode *parse_inline_value(const char *text, char *err, size_t err_cap) {
  const char *p = skip_ws(text);
  if (*p == '{' || *p == '[' || *p == '"' || *p == '\'') {
    const char *q = p;
    TfYamlNode *n = flow_value(&q, err, err_cap);
    return n;
  }
  return resolve_plain_scalar(p, strlen(p));
}

// ---------------------------------------------------------------------------
// Line model + block parsing
// ---------------------------------------------------------------------------

typedef struct {
  int indent;
  char *content;  // points into the owned working copy; comment-stripped, rtrimmed
} Line;

// Strip an unquoted '#' comment (preceded by whitespace or at start) and trailing
// whitespace, in place.
static void strip_comment(char *s) {
  bool in_s = false;
  bool in_d = false;
  for (size_t i = 0; s[i] != '\0'; i++) {
    char c = s[i];
    if (in_d) {
      if (c == '\\' && s[i + 1] != '\0') {
        i++;
      } else if (c == '"') {
        in_d = false;
      }
    } else if (in_s) {
      if (c == '\'') {
        in_s = false;
      }
    } else if (c == '"') {
      in_d = true;
    } else if (c == '\'') {
      in_s = true;
    } else if (c == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t')) {
      s[i] = '\0';
      break;
    }
  }
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r')) {
    s[--n] = '\0';
  }
}

static bool starts_with_dash(const char *c) {
  return c[0] == '-' && (c[1] == ' ' || c[1] == '\0');
}

// Find the index of the "key:" separator colon (a ':' followed by space or
// end-of-string, outside quotes), or -1.
static long find_kv_colon(const char *c) {
  bool in_s = false;
  bool in_d = false;
  for (long i = 0; c[i] != '\0'; i++) {
    char ch = c[i];
    if (in_d) {
      if (ch == '\\' && c[i + 1] != '\0') {
        i++;
      } else if (ch == '"') {
        in_d = false;
      }
    } else if (in_s) {
      if (ch == '\'') {
        in_s = false;
      }
    } else if (ch == '"') {
      in_d = true;
    } else if (ch == '\'') {
      in_s = true;
    } else if (ch == ':' && (c[i + 1] == ' ' || c[i + 1] == '\0')) {
      return i;
    }
  }
  return -1;
}

static TfYamlNode *parse_node(const Line *lines, size_t n, size_t *i, int indent, char *err,
                              size_t err_cap);

static char *key_text(const char *content, long colon, char *err, size_t err_cap) {
  const char *start = content;
  const char *end = content + colon;
  if (*start == '"' || *start == '\'') {
    const char *p = start;
    TfYamlNode *k = parse_quoted(&p, err, err_cap);
    if (k == 0) {
      return 0;
    }
    char *key = k->as.string;
    k->as.string = 0;
    tf_yaml_free(k);
    return key;
  }
  while (end > start && (end[-1] == ' ' || end[-1] == '\t')) {
    end--;
  }
  return dup_range(start, (size_t)(end - start));
}

// Add one "key: value" entry (line already consumed) to `map`. For an empty
// value the value is the nested block at indent > key_indent.
static bool parse_map_entry(TfYamlNode *map, const Line *lines, size_t n, size_t *i,
                            const char *content, int key_indent, char *err, size_t err_cap) {
  long colon = find_kv_colon(content);
  if (colon < 0) {
    tf_copy_cstr(err, err_cap, "expected 'key:' in mapping");
    return false;
  }
  char *key = key_text(content, colon, err, err_cap);
  if (key == 0) {
    return false;
  }
  const char *value_part = skip_ws(content + colon + 1);
  TfYamlNode *value = 0;
  if (*value_part == '\0') {
    if (*i < n && lines[*i].indent > key_indent) {
      value = parse_node(lines, n, i, lines[*i].indent, err, err_cap);
    } else {
      value = node_new(TF_YAML_NULL);
    }
  } else {
    value = parse_inline_value(value_part, err, err_cap);
  }
  if (value == 0) {
    free(key);
    return false;
  }
  if (!map_append(map, key, value)) {
    free(key);
    tf_yaml_free(value);
    tf_copy_cstr(err, err_cap, "out of memory building mapping");
    return false;
  }
  return true;
}

static TfYamlNode *parse_mapping(const Line *lines, size_t n, size_t *i, int indent, char *err,
                                 size_t err_cap) {
  TfYamlNode *map = node_new(TF_YAML_MAPPING);
  if (map == 0) {
    return 0;
  }
  while (*i < n && lines[*i].indent == indent && !starts_with_dash(lines[*i].content)) {
    const char *content = lines[*i].content;
    (*i)++;
    if (!parse_map_entry(map, lines, n, i, content, indent, err, err_cap)) {
      tf_yaml_free(map);
      return 0;
    }
  }
  return map;
}

static TfYamlNode *parse_sequence(const Line *lines, size_t n, size_t *i, int indent, char *err,
                                  size_t err_cap) {
  TfYamlNode *seq = node_new(TF_YAML_SEQUENCE);
  if (seq == 0) {
    return 0;
  }
  while (*i < n && lines[*i].indent == indent && starts_with_dash(lines[*i].content)) {
    const char *content = lines[*i].content;
    const char *rest = content + 1;  // past '-'
    while (*rest == ' ') {
      rest++;
    }
    int rest_col = indent + (int)(rest - content);
    (*i)++;
    TfYamlNode *item = 0;
    if (*rest == '\0') {
      if (*i < n && lines[*i].indent > indent) {
        item = parse_node(lines, n, i, lines[*i].indent, err, err_cap);
      } else {
        item = node_new(TF_YAML_NULL);
      }
    } else if (rest[0] == '{' || rest[0] == '[' || rest[0] == '"' || rest[0] == '\'') {
      item = parse_inline_value(rest, err, err_cap);
    } else if (find_kv_colon(rest) >= 0) {
      // Sequence item is a mapping whose first key sits at rest_col.
      item = node_new(TF_YAML_MAPPING);
      if (item != 0 &&
          parse_map_entry(item, lines, n, i, rest, rest_col, err, err_cap)) {
        while (*i < n && lines[*i].indent == rest_col &&
               !starts_with_dash(lines[*i].content)) {
          const char *c2 = lines[*i].content;
          (*i)++;
          if (!parse_map_entry(item, lines, n, i, c2, rest_col, err, err_cap)) {
            tf_yaml_free(item);
            item = 0;
            break;
          }
        }
      } else if (item != 0) {
        tf_yaml_free(item);
        item = 0;
      }
    } else {
      item = resolve_plain_scalar(rest, strlen(rest));
    }
    if (item == 0) {
      tf_yaml_free(seq);
      return 0;
    }
    if (!seq_append(seq, item)) {
      tf_yaml_free(item);
      tf_yaml_free(seq);
      tf_copy_cstr(err, err_cap, "out of memory building sequence");
      return 0;
    }
  }
  return seq;
}

static TfYamlNode *parse_node(const Line *lines, size_t n, size_t *i, int indent, char *err,
                              size_t err_cap) {
  if (starts_with_dash(lines[*i].content)) {
    return parse_sequence(lines, n, i, indent, err, err_cap);
  }
  return parse_mapping(lines, n, i, indent, err, err_cap);
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

TfYamlNode *tf_yaml_parse(const char *text, char *err, size_t err_cap) {
  if (err_cap > 0) {
    err[0] = '\0';
  }
  char *work = dup_range(text, strlen(text));
  if (work == 0) {
    tf_copy_cstr(err, err_cap, "out of memory copying YAML");
    return 0;
  }

  // Split into logical (comment-stripped, non-empty) lines.
  Line *lines = 0;
  size_t count = 0;
  size_t cap = 0;
  char *p = work;
  while (*p != '\0') {
    char *nl = strchr(p, '\n');
    if (nl != 0) {
      *nl = '\0';
    }
    int indent = 0;
    while (p[indent] == ' ') {
      indent++;
    }
    char *content = p + indent;
    strip_comment(content);
    if (*content != '\0') {
      if (count == cap) {
        size_t ncap = cap ? cap * 2 : 64;
        Line *grown = (Line *)realloc(lines, ncap * sizeof(Line));
        if (grown == 0) {
          free(lines);
          free(work);
          tf_copy_cstr(err, err_cap, "out of memory lexing YAML");
          return 0;
        }
        lines = grown;
        cap = ncap;
      }
      lines[count].indent = indent;
      lines[count].content = content;
      count++;
    }
    if (nl == 0) {
      break;
    }
    p = nl + 1;
  }

  TfYamlNode *root = 0;
  if (count == 0) {
    root = node_new(TF_YAML_NULL);
  } else {
    size_t i = 0;
    root = parse_node(lines, count, &i, lines[0].indent, err, err_cap);
    if (root != 0 && i != count) {
      tf_yaml_free(root);
      root = 0;
      tf_copy_cstr(err, err_cap, "trailing content after top-level node");
    }
  }
  free(lines);
  free(work);
  return root;
}

TfYamlNode *tf_yaml_new_null(void) {
  return node_new(TF_YAML_NULL);
}

TfYamlNode *tf_yaml_new_string(const char *s) {
  return node_string_owned(dup_range(s, strlen(s)));
}

TfYamlNode *tf_yaml_new_mapping(void) {
  return node_new(TF_YAML_MAPPING);
}

TfYamlNode *tf_yaml_new_sequence(void) {
  return node_new(TF_YAML_SEQUENCE);
}

bool tf_yaml_map_set(TfYamlNode *map, const char *key, TfYamlNode *value) {
  if (map == 0 || map->type != TF_YAML_MAPPING || value == 0) {
    tf_yaml_free(value);
    return false;
  }
  for (size_t i = 0; i < map->as.map.count; i++) {
    if (strcmp(map->as.map.keys[i], key) == 0) {
      tf_yaml_free(map->as.map.values[i]);
      map->as.map.values[i] = value;
      return true;
    }
  }
  char *key_copy = dup_range(key, strlen(key));
  if (key_copy == 0) {
    tf_yaml_free(value);
    return false;
  }
  if (!map_append(map, key_copy, value)) {
    free(key_copy);
    tf_yaml_free(value);
    return false;
  }
  return true;
}

bool tf_yaml_seq_append_node(TfYamlNode *seq, TfYamlNode *value) {
  if (seq == 0 || seq->type != TF_YAML_SEQUENCE || value == 0) {
    tf_yaml_free(value);
    return false;
  }
  if (!seq_append(seq, value)) {
    tf_yaml_free(value);
    return false;
  }
  return true;
}

TfYamlNode *tf_yaml_clone(const TfYamlNode *node) {
  if (node == 0) {
    return 0;
  }
  TfYamlNode *copy = node_new(node->type);
  if (copy == 0) {
    return 0;
  }
  switch (node->type) {
    case TF_YAML_NULL:
      break;
    case TF_YAML_BOOL:
      copy->as.boolean = node->as.boolean;
      break;
    case TF_YAML_INT:
      copy->as.integer = node->as.integer;
      break;
    case TF_YAML_FLOAT:
      copy->as.number = node->as.number;
      break;
    case TF_YAML_STRING:
      copy->as.string = dup_range(node->as.string, strlen(node->as.string));
      if (copy->as.string == 0) {
        free(copy);
        return 0;
      }
      break;
    case TF_YAML_SEQUENCE:
      for (size_t i = 0; i < node->as.seq.count; i++) {
        TfYamlNode *child = tf_yaml_clone(node->as.seq.items[i]);
        if (child == 0 || !tf_yaml_seq_append_node(copy, child)) {
          tf_yaml_free(copy);
          return 0;
        }
      }
      break;
    case TF_YAML_MAPPING:
      for (size_t i = 0; i < node->as.map.count; i++) {
        TfYamlNode *child = tf_yaml_clone(node->as.map.values[i]);
        if (child == 0 || !tf_yaml_map_set(copy, node->as.map.keys[i], child)) {
          tf_yaml_free(copy);
          return 0;
        }
      }
      break;
  }
  return copy;
}

const TfYamlNode *tf_yaml_map_get(const TfYamlNode *node, const char *key) {
  if (node == 0 || node->type != TF_YAML_MAPPING) {
    return 0;
  }
  for (size_t i = 0; i < node->as.map.count; i++) {
    if (strcmp(node->as.map.keys[i], key) == 0) {
      return node->as.map.values[i];
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

static bool json_string(Buf *b, const char *s) {
  if (!buf_putc(b, '"')) {
    return false;
  }
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    unsigned char c = *p;
    switch (c) {
      case '"':
        if (!buf_puts(b, "\\\"")) return false;
        break;
      case '\\':
        if (!buf_puts(b, "\\\\")) return false;
        break;
      case '\n':
        if (!buf_puts(b, "\\n")) return false;
        break;
      case '\t':
        if (!buf_puts(b, "\\t")) return false;
        break;
      case '\r':
        if (!buf_puts(b, "\\r")) return false;
        break;
      default:
        if (c < 0x20) {
          char esc[7];
          snprintf(esc, sizeof(esc), "\\u%04x", c);
          if (!buf_puts(b, esc)) return false;
        } else {
          if (!buf_putc(b, (char)c)) return false;
        }
        break;
    }
  }
  return buf_putc(b, '"');
}

static bool json_emit(Buf *b, const TfYamlNode *node) {
  char num[64];
  switch (node->type) {
    case TF_YAML_NULL:
      return buf_puts(b, "null");
    case TF_YAML_BOOL:
      return buf_puts(b, node->as.boolean ? "true" : "false");
    case TF_YAML_INT:
      snprintf(num, sizeof(num), "%lld", node->as.integer);
      return buf_puts(b, num);
    case TF_YAML_FLOAT:
      snprintf(num, sizeof(num), "%.17g", node->as.number);
      return buf_puts(b, num);
    case TF_YAML_STRING:
      return json_string(b, node->as.string);
    case TF_YAML_SEQUENCE:
      if (!buf_putc(b, '[')) return false;
      for (size_t i = 0; i < node->as.seq.count; i++) {
        if (i && !buf_putc(b, ',')) return false;
        if (!json_emit(b, node->as.seq.items[i])) return false;
      }
      return buf_putc(b, ']');
    case TF_YAML_MAPPING:
      if (!buf_putc(b, '{')) return false;
      for (size_t i = 0; i < node->as.map.count; i++) {
        if (i && !buf_putc(b, ',')) return false;
        if (!json_string(b, node->as.map.keys[i])) return false;
        if (!buf_putc(b, ':')) return false;
        if (!json_emit(b, node->as.map.values[i])) return false;
      }
      return buf_putc(b, '}');
  }
  return false;
}

char *tf_yaml_to_json(const TfYamlNode *node) {
  Buf b = {0};
  if (!json_emit(&b, node)) {
    free(b.data);
    return 0;
  }
  if (b.data == 0) {
    return dup_range("", 0);
  }
  return b.data;
}

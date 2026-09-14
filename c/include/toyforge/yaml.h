#ifndef TOYFORGE_YAML_H
#define TOYFORGE_YAML_H

#include <stdbool.h>
#include <stddef.h>

// A bounded YAML reader for the constructs toyForge's checked-in YAML uses:
// block mappings and sequences (indentation based), flow mappings `{a: b}` and
// flow sequences `[a, b]`, single- and double-quoted scalars, plain scalars with
// core scalar typing (`null`/`~`, `true`/`false`/`TRUE`, decimal int, float WITH
// a decimal point, else string), and `#` comments.
//
// Plain-scalar typing matches `yaml.safe_load` on the subset the repository YAML
// actually uses, but is intentionally NOT full YAML 1.1: it does not infer the
// 1.1 bool keywords (`yes`/`no`/`on`/`off` stay strings), `0x`/`0o` integer
// prefixes, `.inf`/`.nan`, or dotless exponents (`1e3` stays... a float here,
// whereas PyYAML keeps it a string). It also does NOT support block scalars
// (`|`/`>`), anchors, aliases, tags, or document separators — none of these
// appear in the repository YAML, and a differential fuzz confirmed the divergent
// forms are all outside that set.
//
// Parity is enforced against `yaml.safe_load` on every repository YAML file by
// scripts/diff_c_yaml.py.

typedef enum {
  TF_YAML_NULL,
  TF_YAML_BOOL,
  TF_YAML_INT,
  TF_YAML_FLOAT,
  TF_YAML_STRING,
  TF_YAML_SEQUENCE,
  TF_YAML_MAPPING
} TfYamlType;

typedef struct TfYamlNode TfYamlNode;

struct TfYamlNode {
  TfYamlType type;
  union {
    bool boolean;
    long long integer;
    double number;
    char *string;  // owned
    struct {
      TfYamlNode **items;  // owned
      size_t count;
    } seq;
    struct {
      char **keys;          // owned
      TfYamlNode **values;  // owned
      size_t count;
    } map;
  } as;
};

// Parse a NUL-terminated YAML document. Returns a heap node tree owned by the
// caller (free with tf_yaml_free) or NULL on error, writing a message to `err`.
TfYamlNode *tf_yaml_parse(const char *text, char *err, size_t err_cap);

// Free a node tree (NULL-safe).
void tf_yaml_free(TfYamlNode *node);

// Serialize a node tree to a compact JSON string (heap, caller frees) or NULL
// on allocation failure. Mapping key order is preserved.
char *tf_yaml_to_json(const TfYamlNode *node);

// Return the value for `key` in a mapping node, or NULL (type mismatch / absent).
const TfYamlNode *tf_yaml_map_get(const TfYamlNode *node, const char *key);

// Node constructors (heap; freed via tf_yaml_free or once attached to a parent).
TfYamlNode *tf_yaml_new_null(void);
TfYamlNode *tf_yaml_new_string(const char *s);  // copies s
TfYamlNode *tf_yaml_new_mapping(void);
TfYamlNode *tf_yaml_new_sequence(void);

// Mutation: both take ownership of `value` (freed by the parent). Replacing an
// existing mapping key frees the old value. Return false on allocation failure
// (and free `value`).
bool tf_yaml_map_set(TfYamlNode *map, const char *key, TfYamlNode *value);
bool tf_yaml_seq_append_node(TfYamlNode *seq, TfYamlNode *value);

// Deep copy a node tree (NULL on allocation failure).
TfYamlNode *tf_yaml_clone(const TfYamlNode *node);

#endif

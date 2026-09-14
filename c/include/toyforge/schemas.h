#ifndef TOYFORGE_SCHEMAS_H
#define TOYFORGE_SCHEMAS_H

#include "toyforge/common.h"

void tf_schemas_load_builtin(TfSchemas *schemas);
bool tf_schemas_load_dir(TfSchemas *schemas, const char *schemas_dir, char *err, size_t err_cap);
const TfMethodSpec *tf_schemas_find_method(const TfSchemas *schemas, const char *method);
const TfRubricPreset *tf_schemas_find_rubric_preset(
  const TfSchemas *schemas,
  const char *preset
);
bool tf_schemas_state_allowed(const TfSchemas *schemas, const char *state);
const char *tf_schemas_transition_to(
  const TfSchemas *schemas,
  const char *from,
  const char *trigger
);

#endif

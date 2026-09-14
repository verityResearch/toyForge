#ifndef TOYFORGE_VERIFIER_H
#define TOYFORGE_VERIFIER_H

#include "toyforge/common.h"

TfStepResult tf_verify_step(
  const TfVerifyContext *ctx,
  const char *model_output,
  const TfSchemas *schemas
);

#endif

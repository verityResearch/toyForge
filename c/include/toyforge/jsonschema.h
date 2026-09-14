#ifndef TOYFORGE_JSONSCHEMA_H
#define TOYFORGE_JSONSCHEMA_H

#include "toyforge/json.h"
#include "toyforge/common.h"

#include <stdbool.h>
#include <stddef.h>

bool tf_jsonschema_validate_method_params(
  const TfSchemas *schemas,
  const char *method,
  TfJsonSpan params,
  char *err,
  size_t err_cap
);

bool tf_jsonschema_validate_trajectory(
  const TfSchemas *schemas,
  TfJsonSpan trajectory,
  char *err,
  size_t err_cap
);

// General JSON Schema (Draft 2020-12 subset) validation: check `instance`
// against a schema document (a JSON object, or boolean true/false). Supported
// keywords: type (incl. integer), enum, const, allOf/anyOf/oneOf/not,
// if/then/else, properties, required, additionalProperties,
// minProperties/maxProperties, propertyNames, dependentRequired,
// dependentSchemas, prefixItems, items, contains/minContains/maxContains,
// minItems/maxItems, uniqueItems, minLength/maxLength (code points),
// minimum/maximum/exclusiveMinimum/exclusiveMaximum, multipleOf,
// $ref/$defs/$id/$anchor (RFC 3986 URI reference resolution against an in-document
// index, depth-bounded), pattern/patternProperties over the bounded regex engine
// (toyforge/regex.h), unevaluatedProperties/unevaluatedItems (annotation
// collection across in-place applicators), and $dynamicRef/$dynamicAnchor with
// full dynamic-scope resolution (a bookended $dynamicRef binds to the outermost
// $dynamicAnchor of its name across the resources on the active validation path).
// Cross-document ($id/URL) references resolve when the referenced documents are
// supplied via the registry (see tf_jsonschema_validate_ex) — this is enough to
// validate a schema against the Draft 2020-12 metaschema + its vocabularies.
// Not yet supported (treated as pass-through annotations): format assertion and
// Unicode-property (\p{}) regex escapes. Bounded for safety: objects to
// TF_MAX_SCHEMA_PROPERTIES (64) members, plus fixed capacities for the reference
// index and for unevaluatedItems array-item tracking (1024 items); exceeding any
// of these fails loud (an error) rather than silently mis-validating.
bool tf_jsonschema_validate(TfJsonSpan schema, TfJsonSpan instance, char *err, size_t err_cap);

// As tf_jsonschema_validate, but with a registry of additional schema documents
// (each indexed under its own absolute $id) so cross-document $ref/$dynamicRef —
// e.g. validating a schema against the Draft 2020-12 metaschema + its vocabulary
// documents — can resolve. The registry spans must outlive the call.
bool tf_jsonschema_validate_ex(
  TfJsonSpan schema,
  TfJsonSpan instance,
  const TfJsonSpan *registry,
  size_t registry_count,
  char *err,
  size_t err_cap
);

#endif

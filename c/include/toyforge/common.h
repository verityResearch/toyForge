#ifndef TOYFORGE_COMMON_H
#define TOYFORGE_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#define TF_MAX_ERROR 384
#define TF_MAX_TEXT 1024
#define TF_MAX_METHODS 16
#define TF_MAX_METHOD_TRIGGERS 4
#define TF_MAX_STATES 32
#define TF_MAX_TERMINAL_STATES 8
#define TF_MAX_TRANSITIONS 32
#define TF_MAX_RUBRIC_PRESETS 8
#define TF_MAX_SCHEMA_PROPERTIES 64
#define TF_MAX_SCHEMA_CHILD_PROPERTIES 8
#define TF_MAX_SCHEMA_PREFIX 64

typedef struct {
  double parse;
  double schema;
  double method_known;
  double precondition_met;
  double transition_valid;
  double sequence_optimal;
} TfSubscores;

typedef struct {
  bool passed;
  TfSubscores subscores;
  char parsed_thinking[TF_MAX_TEXT];
  char method[96];
  char new_state[96];
  char error_message[TF_MAX_ERROR];
} TfStepResult;

typedef enum {
  TF_SCHEMA_VALUE_STRING = 1,
  TF_SCHEMA_VALUE_OBJECT = 2,
} TfSchemaValueType;

typedef struct {
  char name[96];
  TfSchemaValueType type;
  bool required;
  bool has_min_length;
  size_t min_length;
  bool has_prefix;
  char prefix[TF_MAX_SCHEMA_PREFIX];
  bool format_annotation;
} TfSchemaChildProperty;

typedef struct {
  char name[96];
  TfSchemaValueType type;
  bool required;
  bool has_min_length;
  size_t min_length;
  bool has_prefix;
  char prefix[TF_MAX_SCHEMA_PREFIX];
  bool additional_properties;
  size_t child_property_count;
  TfSchemaChildProperty child_properties[TF_MAX_SCHEMA_CHILD_PROPERTIES];
} TfSchemaProperty;

typedef struct {
  bool loaded;
  bool additional_properties;
  size_t property_count;
  TfSchemaProperty properties[TF_MAX_SCHEMA_PROPERTIES];
} TfParamSchema;

typedef struct {
  const char *name;
  const char *triggers[TF_MAX_METHOD_TRIGGERS];
  size_t trigger_count;
  bool query_only;
  TfParamSchema params_schema;
} TfMethodSpec;

typedef struct {
  const char *from;
  const char *trigger;
  const char *to;
} TfTransition;

typedef struct {
  double parse;
  double schema;
  double method_known;
  double precondition_met;
  double transition_valid;
  double sequence_optimal;
} TfRubric;

typedef struct {
  const char *name;
  TfRubric weights;
} TfRubricPreset;

typedef struct {
  TfMethodSpec methods[TF_MAX_METHODS];
  size_t method_count;
  char states[TF_MAX_STATES][96];
  size_t state_count;
  char initial_state[96];
  char terminal_states[TF_MAX_TERMINAL_STATES][96];
  size_t terminal_state_count;
  TfTransition transitions[TF_MAX_TRANSITIONS];
  size_t transition_count;
  TfRubric rubric_default;
  TfRubricPreset rubric_presets[TF_MAX_RUBRIC_PRESETS];
  size_t rubric_preset_count;
  char rubric_aggregation[16];
  bool trajectory_schema_loaded;
  bool trajectory_initial_state_enum_ref_resolved;
  char trajectory_initial_state_enum_ref[96];
  size_t trajectory_initial_state_enum_count;
  char trajectory_initial_state_enum_values[TF_MAX_STATES][96];
  char method_name_storage[TF_MAX_METHODS][96];
  char trigger_storage[TF_MAX_METHODS][TF_MAX_METHOD_TRIGGERS][96];
  char transition_from_storage[TF_MAX_TRANSITIONS][96];
  char transition_trigger_storage[TF_MAX_TRANSITIONS][96];
  char transition_to_storage[TF_MAX_TRANSITIONS][96];
  char rubric_preset_name_storage[TF_MAX_RUBRIC_PRESETS][32];
} TfSchemas;

typedef struct {
  const char *prior_state;
  const char *expected_trigger;
  const char *expected_state_after;
  bool infer_trigger;
} TfVerifyContext;

#endif

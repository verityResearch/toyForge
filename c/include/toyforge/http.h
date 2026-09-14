#ifndef TOYFORGE_HTTP_H
#define TOYFORGE_HTTP_H

#include <stdbool.h>
#include <stddef.h>

// POST `body` as application/json to `url` with an optional Bearer `api_key`
// (pass NULL/"" to omit the Authorization header). On success writes the
// response body to *out (heap, caller frees) and returns true; on failure
// writes a message to `err` and leaves *out NULL.
//
// Only built when libcurl is available (TF_HAVE_CURL). Verifier-only builds do
// not compile or link this module; callers guard live-HTTP commands on the same
// macro and fail closed otherwise.
bool tf_http_post_json(
  const char *url,
  const char *api_key,
  const char *body,
  char **out,
  char *err,
  size_t err_cap
);

// As above but with caller-supplied request headers (e.g. Anthropic's
// `x-api-key:` + `anthropic-version:`) instead of a Bearer token.
// `Content-Type: application/json` is always added.
bool tf_http_post_json_headers(
  const char *url,
  const char *const *extra_headers,
  size_t n_extra_headers,
  const char *body,
  char **out,
  char *err,
  size_t err_cap
);

#endif

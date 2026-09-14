#include "toyforge/http.h"

#include "toyforge/io.h"

#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  char *data;
  size_t len;
} ResponseBuf;

static size_t write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  size_t n = size * nmemb;
  ResponseBuf *r = (ResponseBuf *)userdata;
  char *grown = (char *)realloc(r->data, r->len + n + 1);
  if (grown == 0) {
    return 0;  // signals an error to libcurl
  }
  r->data = grown;
  memcpy(r->data + r->len, ptr, n);
  r->len += n;
  r->data[r->len] = '\0';
  return n;
}

static char *empty_string(void) {
  char *s = (char *)malloc(1);
  if (s != 0) {
    s[0] = '\0';
  }
  return s;
}

// POST `body` as JSON with a prebuilt header list (which this function takes
// ownership of and frees). Internal core shared by the Bearer and custom-header
// public entry points.
static bool http_post(
  const char *url,
  struct curl_slist *headers,
  const char *body,
  char **out,
  char *err,
  size_t err_cap
) {
  *out = 0;
  CURL *curl = curl_easy_init();
  if (curl == 0) {
    curl_slist_free_all(headers);
    tf_copy_cstr(err, err_cap, "curl initialization failed");
    return false;
  }

  ResponseBuf resp = {0};
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

  CURLcode rc = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  if (rc != CURLE_OK) {
    snprintf(err, err_cap, "http request failed: %s", curl_easy_strerror(rc));
    free(resp.data);
    return false;
  }
  if (status < 200 || status >= 300) {
    snprintf(err, err_cap, "http status %ld", status);
    free(resp.data);
    return false;
  }

  *out = resp.data != 0 ? resp.data : empty_string();
  if (*out == 0) {
    tf_copy_cstr(err, err_cap, "out of memory reading http response");
    return false;
  }
  return true;
}

bool tf_http_post_json(
  const char *url,
  const char *api_key,
  const char *body,
  char **out,
  char *err,
  size_t err_cap
) {
  struct curl_slist *headers = curl_slist_append(0, "Content-Type: application/json");
  char auth[1024];
  if (api_key != 0 && api_key[0] != '\0') {
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", api_key);
    headers = curl_slist_append(headers, auth);
  }
  return http_post(url, headers, body, out, err, err_cap);
}

bool tf_http_post_json_headers(
  const char *url,
  const char *const *extra_headers,
  size_t n_extra_headers,
  const char *body,
  char **out,
  char *err,
  size_t err_cap
) {
  struct curl_slist *headers = curl_slist_append(0, "Content-Type: application/json");
  for (size_t i = 0; i < n_extra_headers; i++) {
    headers = curl_slist_append(headers, extra_headers[i]);
  }
  return http_post(url, headers, body, out, err, err_cap);
}

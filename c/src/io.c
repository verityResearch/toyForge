#include "toyforge/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void tf_copy_cstr(char *dst, size_t dst_cap, const char *src) {
  if (dst == 0 || dst_cap == 0) {
    return;
  }
  if (src == 0) {
    src = "";
  }
  size_t n = strlen(src);
  if (n >= dst_cap) {
    n = dst_cap - 1;
  }
  memcpy(dst, src, n);
  dst[n] = '\0';
}

void tf_copy_span(char *dst, size_t dst_cap, const char *start, const char *end) {
  if (dst == 0 || dst_cap == 0) {
    return;
  }
  if (start == 0 || end == 0 || end < start) {
    dst[0] = '\0';
    return;
  }
  size_t n = (size_t)(end - start);
  if (n >= dst_cap) {
    n = dst_cap - 1;
  }
  memcpy(dst, start, n);
  dst[n] = '\0';
}

static void set_error(char *err, size_t err_cap, const char *msg) {
  tf_copy_cstr(err, err_cap, msg);
}

static void set_path_error(char *err, size_t err_cap, const char *prefix, const char *path) {
  if (err == 0 || err_cap == 0) {
    return;
  }
  snprintf(err, err_cap, "%s %s", prefix, path);
}

bool tf_join_path(char *out, size_t out_cap, const char *dir, const char *name) {
  if (out == 0 || out_cap == 0 || dir == 0 || name == 0) {
    return false;
  }
  int n = snprintf(out, out_cap, "%s/%s", dir, name);
  return n >= 0 && (size_t)n < out_cap;
}

bool tf_read_file_bytes(
  const char *path,
  unsigned char **out,
  size_t *out_len,
  char *err,
  size_t err_cap
) {
  if (err != 0 && err_cap > 0) {
    err[0] = '\0';
  }
  if (out == 0 || out_len == 0) {
    set_error(err, err_cap, "output buffer is required");
    return false;
  }
  *out = 0;
  *out_len = 0;
  if (path == 0 || path[0] == '\0') {
    set_error(err, err_cap, "path is required");
    return false;
  }

  FILE *f = fopen(path, "rb");
  if (f == 0) {
    set_path_error(err, err_cap, "failed to open", path);
    return false;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    set_path_error(err, err_cap, "failed to seek", path);
    return false;
  }
  long size = ftell(f);
  if (size < 0) {
    fclose(f);
    set_path_error(err, err_cap, "failed to size", path);
    return false;
  }
  rewind(f);

  unsigned char *buf = (unsigned char *)malloc((size_t)size + 1);
  if (buf == 0) {
    fclose(f);
    set_error(err, err_cap, "out of memory reading file");
    return false;
  }
  size_t got = fread(buf, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) {
    free(buf);
    set_path_error(err, err_cap, "short read from", path);
    return false;
  }
  buf[got] = '\0';
  *out = buf;
  *out_len = got;
  return true;
}

char *tf_read_file(const char *path, char *err, size_t err_cap) {
  unsigned char *bytes = 0;
  size_t len = 0;
  if (!tf_read_file_bytes(path, &bytes, &len, err, err_cap)) {
    return 0;
  }
  (void)len;
  return (char *)bytes;
}

char *tf_read_line(FILE *f) {
  if (f == 0) {
    return 0;
  }
  size_t cap = 4096;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (buf == 0) {
    return 0;
  }

  int ch = 0;
  while ((ch = fgetc(f)) != EOF) {
    if (len + 1 >= cap) {
      size_t new_cap = cap * 2;
      char *new_buf = (char *)realloc(buf, new_cap);
      if (new_buf == 0) {
        free(buf);
        return 0;
      }
      buf = new_buf;
      cap = new_cap;
    }
    buf[len++] = (char)ch;
    if (ch == '\n') {
      break;
    }
  }
  if (len == 0 && ch == EOF) {
    free(buf);
    return 0;
  }
  buf[len] = '\0';
  return buf;
}

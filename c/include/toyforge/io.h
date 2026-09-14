#ifndef TOYFORGE_IO_H
#define TOYFORGE_IO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

void tf_copy_cstr(char *dst, size_t dst_cap, const char *src);
void tf_copy_span(char *dst, size_t dst_cap, const char *start, const char *end);
bool tf_join_path(char *out, size_t out_cap, const char *dir, const char *name);
bool tf_read_file_bytes(
  const char *path,
  unsigned char **out,
  size_t *out_len,
  char *err,
  size_t err_cap
);
char *tf_read_file(const char *path, char *err, size_t err_cap);
char *tf_read_line(FILE *f);

#endif

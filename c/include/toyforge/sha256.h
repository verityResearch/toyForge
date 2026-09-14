#ifndef TOYFORGE_SHA256_H
#define TOYFORGE_SHA256_H

#include <stdbool.h>
#include <stddef.h>

#define TF_SHA256_DIGEST_SIZE 32
#define TF_SHA256_HEX_SIZE 65

bool tf_sha256(const void *data, size_t len, unsigned char digest[TF_SHA256_DIGEST_SIZE]);
bool tf_sha256_hex(const void *data, size_t len, char out[TF_SHA256_HEX_SIZE]);
bool tf_sha256_file_hex(
  const char *path,
  char out[TF_SHA256_HEX_SIZE],
  char *err,
  size_t err_cap
);

#endif

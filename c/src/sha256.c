#include "toyforge/sha256.h"

#include "toyforge/io.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  uint8_t data[64];
  uint32_t state[8];
  uint64_t bitlen;
  size_t datalen;
} TfSha256Ctx;

static const uint32_t K[64] = {
  0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
  0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
  0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
  0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
  0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
  0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
  0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
  0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
  0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
  0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
  0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}

static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

static uint32_t ep0(uint32_t x) {
  return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static uint32_t ep1(uint32_t x) {
  return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static uint32_t sig0(uint32_t x) {
  return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static uint32_t sig1(uint32_t x) {
  return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

static void sha256_transform(TfSha256Ctx *ctx, const uint8_t data[64]) {
  uint32_t m[64];
  for (size_t i = 0; i < 16; i++) {
    size_t j = i * 4;
    m[i] = ((uint32_t)data[j] << 24) | ((uint32_t)data[j + 1] << 16) |
           ((uint32_t)data[j + 2] << 8) | (uint32_t)data[j + 3];
  }
  for (size_t i = 16; i < 64; i++) {
    m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
  }

  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t e = ctx->state[4];
  uint32_t f = ctx->state[5];
  uint32_t g = ctx->state[6];
  uint32_t h = ctx->state[7];

  for (size_t i = 0; i < 64; i++) {
    uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[i] + m[i];
    uint32_t t2 = ep0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void sha256_init(TfSha256Ctx *ctx) {
  ctx->datalen = 0;
  ctx->bitlen = 0;
  ctx->state[0] = 0x6a09e667u;
  ctx->state[1] = 0xbb67ae85u;
  ctx->state[2] = 0x3c6ef372u;
  ctx->state[3] = 0xa54ff53au;
  ctx->state[4] = 0x510e527fu;
  ctx->state[5] = 0x9b05688cu;
  ctx->state[6] = 0x1f83d9abu;
  ctx->state[7] = 0x5be0cd19u;
}

static void sha256_update(TfSha256Ctx *ctx, const uint8_t *data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    ctx->data[ctx->datalen] = data[i];
    ctx->datalen++;
    if (ctx->datalen == sizeof(ctx->data)) {
      sha256_transform(ctx, ctx->data);
      ctx->bitlen += 512u;
      ctx->datalen = 0;
    }
  }
}

static void sha256_final(TfSha256Ctx *ctx, unsigned char digest[TF_SHA256_DIGEST_SIZE]) {
  size_t i = ctx->datalen;

  ctx->data[i++] = 0x80u;
  if (i > 56) {
    while (i < 64) {
      ctx->data[i++] = 0;
    }
    sha256_transform(ctx, ctx->data);
    memset(ctx->data, 0, 56);
  } else {
    while (i < 56) {
      ctx->data[i++] = 0;
    }
  }

  ctx->bitlen += (uint64_t)ctx->datalen * 8u;
  ctx->data[63] = (uint8_t)(ctx->bitlen);
  ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
  ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
  ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
  ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
  ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
  ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
  ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
  sha256_transform(ctx, ctx->data);

  for (i = 0; i < 4; i++) {
    digest[i] = (unsigned char)((ctx->state[0] >> (24 - i * 8)) & 0xffu);
    digest[i + 4] = (unsigned char)((ctx->state[1] >> (24 - i * 8)) & 0xffu);
    digest[i + 8] = (unsigned char)((ctx->state[2] >> (24 - i * 8)) & 0xffu);
    digest[i + 12] = (unsigned char)((ctx->state[3] >> (24 - i * 8)) & 0xffu);
    digest[i + 16] = (unsigned char)((ctx->state[4] >> (24 - i * 8)) & 0xffu);
    digest[i + 20] = (unsigned char)((ctx->state[5] >> (24 - i * 8)) & 0xffu);
    digest[i + 24] = (unsigned char)((ctx->state[6] >> (24 - i * 8)) & 0xffu);
    digest[i + 28] = (unsigned char)((ctx->state[7] >> (24 - i * 8)) & 0xffu);
  }
}

bool tf_sha256(const void *data, size_t len, unsigned char digest[TF_SHA256_DIGEST_SIZE]) {
  if ((data == 0 && len != 0) || digest == 0) {
    return false;
  }
  TfSha256Ctx ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, (const uint8_t *)data, len);
  sha256_final(&ctx, digest);
  return true;
}

bool tf_sha256_hex(const void *data, size_t len, char out[TF_SHA256_HEX_SIZE]) {
  static const char hex[] = "0123456789abcdef";
  unsigned char digest[TF_SHA256_DIGEST_SIZE];
  if (out == 0 || !tf_sha256(data, len, digest)) {
    return false;
  }
  for (size_t i = 0; i < TF_SHA256_DIGEST_SIZE; i++) {
    out[i * 2] = hex[digest[i] >> 4];
    out[i * 2 + 1] = hex[digest[i] & 0x0fu];
  }
  out[TF_SHA256_HEX_SIZE - 1] = '\0';
  return true;
}

bool tf_sha256_file_hex(
  const char *path,
  char out[TF_SHA256_HEX_SIZE],
  char *err,
  size_t err_cap
) {
  if (out == 0) {
    tf_copy_cstr(err, err_cap, "output buffer is required");
    return false;
  }
  unsigned char *bytes = 0;
  size_t len = 0;
  if (!tf_read_file_bytes(path, &bytes, &len, err, err_cap)) {
    return false;
  }
  bool ok = tf_sha256_hex(bytes, len, out);
  free(bytes);
  if (!ok) {
    tf_copy_cstr(err, err_cap, "failed to hash file");
  }
  return ok;
}

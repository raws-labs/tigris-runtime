/**
 * @file tigris_lz4.h
 * @brief Minimal LZ4 block decoder - no malloc, decode-only.
 */

#ifndef TIGRIS_LZ4_H
#define TIGRIS_LZ4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Decompress an LZ4 block.
 *
 * @param src       Compressed data.
 * @param src_len   Size of compressed data in bytes.
 * @param dst       Output buffer (caller-allocated).
 * @param dst_cap   Capacity of output buffer in bytes.
 * @return Decompressed size on success, or negative on error:
 *         -1 = output buffer too small
 *         -2 = corrupted/truncated input
 */
int32_t tigris_lz4_decompress(
    const uint8_t *src, uint32_t src_len,
    uint8_t *dst, uint32_t dst_cap);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_LZ4_H */

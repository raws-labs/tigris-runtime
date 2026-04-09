/**
 * @file tigris_lz4.c
 * @brief Minimal LZ4 block decoder - no malloc, decode-only.
 *
 * Implements the LZ4 block format (not frame format):
 *   repeat {
 *       token (1 byte): high nibble = literal_len, low nibble = match_len
 *       [extra literal length bytes if nibble == 15]
 *       literal bytes
 *       [if not end of input:]
 *           offset (2 bytes LE): backward match distance
 *           [extra match length bytes if nibble == 15]
 *   }
 */

#include "tigris_lz4.h"
#include <string.h>

#define LZ4_MIN_MATCH 4

int32_t tigris_lz4_decompress(
    const uint8_t *src, uint32_t src_len,
    uint8_t *dst, uint32_t dst_cap)
{
    if (!src || !dst)
        return -2;

    const uint8_t *ip = src;
    const uint8_t *ip_end = src + src_len;
    uint8_t *op = dst;
    uint8_t *op_end = dst + dst_cap;

    while (ip < ip_end) {
        /* Read token */
        uint8_t token = *ip++;
        uint32_t lit_len = token >> 4;
        uint32_t match_len = token & 0x0F;

        /* Extended literal length */
        if (lit_len == 15) {
            uint8_t extra;
            do {
                if (ip >= ip_end)
                    return -2;
                extra = *ip++;
                lit_len += extra;
            } while (extra == 255);
        }

        /* Copy literals */
        if (lit_len > 0) {
            if (ip + lit_len > ip_end)
                return -2;
            if (op + lit_len > op_end)
                return -1;
            memcpy(op, ip, lit_len);
            ip += lit_len;
            op += lit_len;
        }

        /* Check if we're at the end of input (last sequence has no match) */
        if (ip >= ip_end)
            break;

        /* Read match offset (2 bytes LE) */
        if (ip + 2 > ip_end)
            return -2;
        uint32_t offset = (uint32_t)ip[0] | ((uint32_t)ip[1] << 8);
        ip += 2;

        if (offset == 0)
            return -2;  /* offset 0 is invalid */

        /* Extended match length */
        match_len += LZ4_MIN_MATCH;
        if ((token & 0x0F) == 15) {
            uint8_t extra;
            do {
                if (ip >= ip_end)
                    return -2;
                extra = *ip++;
                match_len += extra;
            } while (extra == 255);
        }

        /* Copy match */
        if (op + match_len > op_end)
            return -1;

        const uint8_t *match_src = op - offset;
        if (match_src < dst)
            return -2;  /* offset points before output buffer */

        /* Byte-by-byte copy handles overlapping matches */
        for (uint32_t i = 0; i < match_len; i++)
            op[i] = match_src[i];
        op += match_len;
    }

    return (int32_t)(op - dst);
}

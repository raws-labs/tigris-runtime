/**
 * @file tigris_transpose_band.h
 * @brief The permutations a banded transpose can execute.
 *
 * Internal to src/. Not installed and not part of the public API. The loader
 * and the executor both decide whether a transpose is bandable, so the rule
 * lives here rather than in a copy on each side.
 */
#ifndef TIGRIS_TRANSPOSE_BAND_H
#define TIGRIS_TRANSPOSE_BAND_H

#include <stdint.h>

/**
 * Split a stored permutation into the groups a banded transpose swaps.
 *
 * A transpose is a plain matrix transpose when it swaps two adjacent groups
 * of axes and leaves everything else in its relative order:
 *
 *     [prefix][A][B][suffix] -> [prefix][B][A][suffix]
 *
 * It then transposes prod(A) by prod(B), repeated over prod(prefix), and
 * carries prod(suffix) elements at each position. A layout conversion, an
 * attention block's key transpose and its head permutation are all instances
 * of this; the prefix and the suffix are what tell them apart.
 *
 * Writes the prefix length, the length of A, and the length of the middle,
 * and returns 1. Returns 0 when the permutation is not of that shape,
 * including for the identity, which moves nothing.
 */
static int transpose_band_groups(
    const uint8_t *perm, uint8_t rank,
    uint8_t *out_prefix, uint8_t *out_split, uint8_t *out_middle)
{
    if (!perm || rank < 2u)
        return 0;
    uint8_t prefix = 0;
    while (prefix < rank && perm[prefix] == prefix)
        prefix++;
    if (prefix >= rank)
        return 0;
    uint8_t suffix = 0;
    while (suffix < (uint8_t)(rank - prefix) &&
           perm[rank - 1u - suffix] == (uint8_t)(rank - 1u - suffix))
        suffix++;
    uint8_t middle = (uint8_t)(rank - prefix - suffix);
    if (middle < 2u)
        return 0;
    /* Inside the middle the permutation reads [B][A], so where A begins is
     * where the first index lands. */
    if (perm[prefix] < prefix)
        return 0;
    uint8_t split = (uint8_t)(perm[prefix] - prefix);
    if (split == 0u || split >= middle)
        return 0;
    for (uint8_t i = 0; i < middle; i++) {
        uint8_t want = (i < (uint8_t)(middle - split))
                           ? (uint8_t)(prefix + split + i)
                           : (uint8_t)(prefix + i - (middle - split));
        if (perm[prefix + i] != want)
            return 0;
    }
    *out_prefix = prefix;
    *out_split = split;
    *out_middle = middle;
    return 1;
}

#endif /* TIGRIS_TRANSPOSE_BAND_H */

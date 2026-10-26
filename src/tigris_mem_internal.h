/**
 * @file tigris_mem_internal.h
 * @brief Arena primitives the executor and the input interface share.
 *
 * Internal to src/. Not installed and not part of the public API. An
 * application places model inputs with tigris_input_alloc(),
 * tigris_input_write() or tigris_mem_alloc_slow(); the executor makes every
 * other allocation.
 */
#ifndef TIGRIS_MEM_INTERNAL_H
#define TIGRIS_MEM_INTERNAL_H

#include <stdint.h>

#include "tigris.h"
#include "tigris_mem.h"

/**
 * Bump-allocate in the fast arena, set tensor_ptrs[tensor_idx].
 *
 * @param mem         Memory manager.
 * @param tensor_idx  Tensor index to allocate for.
 * @param size_bytes  Number of bytes to allocate.
 * @return TIGRIS_MEM_OK, TIGRIS_MEM_ERR_NULL, TIGRIS_MEM_ERR_OOM, or
 *         TIGRIS_MEM_ERR_BAD_INDEX.
 */
tigris_mem_error_t tigris_mem_alloc_fast(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes);

/**
 * Load a tensor from slow to fast: allocate in fast arena, memcpy from
 * current (slow) location, update tensor_ptrs[tensor_idx] to fast.
 *
 * @param mem         Memory manager.
 * @param tensor_idx  Tensor index to load.
 * @param size_bytes  Number of bytes to copy.
 * @return TIGRIS_MEM_OK, TIGRIS_MEM_ERR_OOM, or TIGRIS_MEM_ERR_NOT_SET.
 */
tigris_mem_error_t tigris_mem_load(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes);

/**
 * Spill a tensor from fast to slow: allocate in slow buffer, memcpy from
 * current (fast) location, update tensor_ptrs[tensor_idx] to slow.
 *
 * @param mem         Memory manager.
 * @param tensor_idx  Tensor index to spill.
 * @param size_bytes  Number of bytes to copy.
 * @return TIGRIS_MEM_OK, TIGRIS_MEM_ERR_OOM, or TIGRIS_MEM_ERR_NOT_SET.
 */
tigris_mem_error_t tigris_mem_spill(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes);

/**
 * Reset the fast arena bump pointer to the reserved offset.
 * Leaves tensor_ptrs untouched, so callers must invalidate stale pointers.
 *
 * @param mem  Memory manager.
 */
void tigris_mem_reset_fast(tigris_mem_t *mem);

/**
 * Load an axis-1 band of an NHWC or NLC tensor from slow to fast.
 * Allocates N*tile_extent*W*C*elem in fast and copies [h_start, h_end).
 *
 * @param mem         Memory manager.
 * @param plan        Loaded plan (for tensor shape lookup).
 * @param tensor_idx  Tensor index to load a tile from.
 * @param h_start     First row (inclusive) of the height band.
 * @param h_end       Last row (exclusive) of the height band.
 * @return TIGRIS_MEM_OK on success, negative error code on failure.
 */
tigris_mem_error_t tigris_mem_load_tile(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tensor_idx, int32_t h_start, int32_t h_end);

/**
 * Spill an axis-1 band from fast to slow. Writes [h_start, h_end) into the
 * full NHWC/NLC tensor at slow_base, then restores tensor_ptrs[tensor_idx].
 *
 * @param mem         Memory manager.
 * @param plan        Loaded plan (for tensor shape lookup).
 * @param tensor_idx  Tensor index to spill a tile for.
 * @param slow_base   Destination buffer in slow memory for the full tensor.
 * @param h_start     First row (inclusive) of the height band.
 * @param h_end       Last row (exclusive) of the height band.
 * @return TIGRIS_MEM_OK on success, negative error code on failure.
 */
tigris_mem_error_t tigris_mem_spill_tile(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tensor_idx, void *slow_base, int32_t h_start, int32_t h_end);

/**
 * Load a 2D [h0:h1, w0:w1, :] sub-rectangle of an NHWC tensor from slow to
 * fast. Allocates N*(h1-h0)*(w1-w0)*C*elem in fast, packed with no gaps
 * between rows, and copies the strided source rectangle row by row.
 *
 * @param mem   Memory manager.
 * @param plan  Loaded plan (for tensor shape lookup).
 * @param tidx  Tensor index to load a tile from. Must be rank 4 (NHWC).
 * @param h0    First row (inclusive) of the height range. Must satisfy
 *              0 <= h0 < h1 <= H.
 * @param h1    Last row (exclusive) of the height range.
 * @param w0    First column (inclusive) of the width range. Must satisfy
 *              0 <= w0 < w1 <= W.
 * @param w1    Last column (exclusive) of the width range.
 * @return TIGRIS_MEM_OK on success, negative error code on failure.
 */
tigris_mem_error_t tigris_mem_load_tile_2d(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tidx, int32_t h0, int32_t h1, int32_t w0, int32_t w1);

/**
 * Spill a packed 2D [oh, ow, C] core from fast to the strided
 * [h0:h1, w0:w1, :] rectangle of the full NHWC tensor at slow_base, then
 * restore tensor_ptrs[tidx]. 2D output tiles have no halo, so the entire
 * packed core is written back.
 *
 * @param mem       Memory manager.
 * @param plan      Loaded plan (for tensor shape lookup).
 * @param tidx      Tensor index to spill a tile for.
 * @param slow_base Destination buffer in slow memory for the full tensor.
 * @param h0        First row (inclusive) of the height range. Must satisfy
 *                  0 <= h0 < h1 <= H.
 * @param h1        Last row (exclusive) of the height range.
 * @param w0        First column (inclusive) of the width range. Must
 *                  satisfy 0 <= w0 < w1 <= W.
 * @param w1        Last column (exclusive) of the width range.
 * @return TIGRIS_MEM_OK on success, negative error code on failure.
 */
tigris_mem_error_t tigris_mem_spill_tile_2d(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tidx, void *slow_base,
    int32_t h0, int32_t h1, int32_t w0, int32_t w1);

#endif /* TIGRIS_MEM_INTERNAL_H */

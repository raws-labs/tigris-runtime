/**
 * @file tigris_loader.h
 * @brief Plan loader API - parse a .tgrs buffer into a tigris_plan_t.
 */

#ifndef TIGRIS_LOADER_H
#define TIGRIS_LOADER_H

#include "tigris.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a binary plan from a buffer.
 *
 * Zero-copy, zero-alloc: all pointers in out_plan point directly into buf.
 * The caller must keep buf alive for the lifetime of out_plan.
 *
 * ALIGNMENT CONTRACT: buf must be aligned to TIGRIS_TENSOR_ALIGN (16 on DSP
 * Cortex-M) when the plan's weights are read in place (XIP / non-compressed)
 * and consumed by the optimized CMSIS-NN kernels. The compiler aligns every
 * weight blob to that boundary *relative to the file base*, so an unaligned buf
 * makes the weight pointers unaligned and the opt kernels fault (LDRD/SIMD).
 * bin2c emits the embedded blob aligned(16); mmap/flash bases satisfy this; a
 * malloc'd host buffer should use aligned_alloc(TIGRIS_TENSOR_ALIGN, ...). The
 * reference (s8/f32) kernels do not require this; only the opt backends do.
 *
 * @param buf       Pointer to the loaded .tgrs file contents.
 * @param buf_len   Size of the buffer in bytes.
 * @param out_plan  Output: parsed plan handle.
 * @return TIGRIS_OK on success, negative error code on failure.
 */
tigris_error_t tigris_plan_load(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan);

/**
 * Return a human-readable string for a loader error code.
 *
 * @param err  Loader error code.
 * @return Static string describing the error.
 */
const char *tigris_error_str(tigris_error_t err);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_LOADER_H */

/**
 * @file tigris_mem.h
 * @brief Memory manager API - arena allocator + tensor pointer table.
 */

#ifndef TIGRIS_MEM_H
#define TIGRIS_MEM_H

#include <stdint.h>
#include <stddef.h>

#include "tigris.h"  /* tigris_plan_t needed by tile load/spill */

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */

/** Memory manager error codes. */
typedef enum {
    TIGRIS_MEM_OK           =  0,   /**< Success */
    TIGRIS_MEM_ERR_NULL     = -1,   /**< Null pointer argument */
    TIGRIS_MEM_ERR_OOM      = -2,   /**< Out of memory in arena */
    TIGRIS_MEM_ERR_BAD_INDEX= -3,   /**< Tensor index >= num_tensors */
    TIGRIS_MEM_ERR_NOT_SET  = -4,   /**< Tensor pointer is NULL (load/spill source) */
} tigris_mem_error_t;

/* Tile context */

/** Runtime-only per-tile spatial context, active during tiled execution. */
typedef struct {
    uint8_t  active;       /* 0=full tensor, 1=tile active */
    uint8_t  _pad0;
    uint16_t pad_top;      /* effective top padding for this tile */
    uint16_t pad_bottom;   /* effective bottom padding */
    uint16_t _pad1;
    int32_t  in_h;         /* tile input H (with halo) */
    int32_t  out_h;        /* tile output H (core) */
    int32_t  in_w;         /* input W (same as full, for convenience) */
    int32_t  out_w;        /* output W */
    /* New-row offsets for line-buffered chains: a kernel computes exactly
     * out_h output rows, writing them starting at out_row_start in the
     * (possibly larger, persistent) output buffer, and reading input rows
     * starting at in_row_start in the input buffer handed to it. Both
     * default to 0, which reduces every existing full-tile kernel call to
     * today's behavior byte-for-byte. Set by the roll loop to skip
     * recomputing already-rolled overlap rows. */
    int32_t  out_row_start; /* rows already written before this call */
    int32_t  in_row_start;  /* rows already consumed before this call */
    /* 2D (HW axis) tile geometry. Consulted only when width_tiled == 1;
     * default to 0 like every other field above, so 1D-height and
     * non-tiled paths stay byte-identical. */
    uint16_t pad_left;      /* effective left padding for this tile */
    uint16_t pad_right;     /* effective right padding for this tile */
    uint8_t  width_tiled;   /* 0=height-only tile, 1=2D (HW) tile active */
} tigris_tile_ctx_t;

/* Tensor alignment */

/**
 * Minimum alignment for tensor allocations (bytes, must be power of 2).
 * Auto-detected from target architecture, override with -DTIGRIS_TENSOR_ALIGN=N.
 */
#ifndef TIGRIS_TENSOR_ALIGN
  #if defined(__ARM_FEATURE_SVE)
    #define TIGRIS_TENSOR_ALIGN 64   /* SVE (variable-length, 64 safe) */
  #elif defined(__aarch64__)
    #define TIGRIS_TENSOR_ALIGN 16   /* NEON */
  #elif defined(__x86_64__)
    #define TIGRIS_TENSOR_ALIGN 32   /* AVX2 */
  #elif defined(__riscv_vector)
    #define TIGRIS_TENSOR_ALIGN 16   /* RVV */
  #elif defined(__XTENSA__)
    #define TIGRIS_TENSOR_ALIGN 8    /* ESP32-S3 TIE (ee.vld.l.64.ip) */
  #elif defined(__ARM_NEON)
    #define TIGRIS_TENSOR_ALIGN 16   /* 32-bit ARM with NEON */
  #elif defined(__ARM_FEATURE_DSP)
    #define TIGRIS_TENSOR_ALIGN 16   /* Cortex-M4/M7/M33: CMSIS-NN opt kernels need aligned buffers */
  #else
    #define TIGRIS_TENSOR_ALIGN 4    /* Cortex-M0, 8-bit MCUs, etc. */
  #endif
#endif

/* Memory manager state */

/** Memory manager state - dual-arena allocator with tensor pointer table. */
typedef struct {
    void       **tensor_ptrs;       /* [num_tensors] - current data ptr per tensor */
    uint16_t     num_tensors;
    uint8_t     *fast_base;         /* fast arena buffer */
    uint32_t     fast_size;         /* capacity in bytes */
    uint32_t     fast_used;         /* bump offset */
    uint32_t     fast_reserved;    /* bytes reserved at start (survives reset) */
    uint32_t     fast_peak;         /* observed high-water mark for fast_used */
    uint8_t     *slow_base;         /* slow buffer */
    uint32_t     slow_size;         /* capacity in bytes */
    uint32_t     slow_used;         /* bump offset */
    tigris_tile_ctx_t tile;         /* per-tile context (active during tiled execution) */
} tigris_mem_t;

/**
 * Record a new fast-arena high-water mark if fast_used grew.
 * Called at every site that increases fast_used (the bump allocator and the
 * executor's direct scratch/weight bumps) so fast_peak is the observed arena
 * high-water mark, not a compile-time estimate. Normal execution compacts on
 * memory pressure, so a roomier arena can intentionally report a higher peak.
 */
static inline void tigris_mem_note_fast_peak(tigris_mem_t *mem)
{
    if (mem && mem->fast_used > mem->fast_peak)
        mem->fast_peak = mem->fast_used;
}

/* API */

/**
 * Initialise the memory manager. Zeroes all tensor pointers.
 * Aligns the initial bump offset to TIGRIS_TENSOR_ALIGN.
 *
 * @param mem          Memory manager to initialise.
 * @param tensor_ptrs  Caller-owned array of void* (one per tensor).
 * @param num_tensors  Number of tensors in the plan.
 * @param fast_buf     Fast arena buffer (SRAM on embedded targets).
 * @param fast_size    Size of fast_buf in bytes.
 * @param slow_buf     Slow buffer (PSRAM or heap on embedded targets).
 * @param slow_size    Size of slow_buf in bytes.
 * @return TIGRIS_MEM_OK on success, TIGRIS_MEM_ERR_NULL for a null argument,
 *         or TIGRIS_MEM_ERR_OOM if an unaligned buffer is too small to reach
 *         its first aligned allocation address.
 */
tigris_mem_error_t tigris_mem_init(
    tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t num_tensors,
    void *fast_buf, uint32_t fast_size,
    void *slow_buf, uint32_t slow_size);

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
 * Bump-allocate in the slow buffer, set tensor_ptrs[tensor_idx].
 *
 * @param mem         Memory manager.
 * @param tensor_idx  Tensor index to allocate for.
 * @param size_bytes  Number of bytes to allocate.
 * @return TIGRIS_MEM_OK, TIGRIS_MEM_ERR_NULL, TIGRIS_MEM_ERR_OOM, or
 *         TIGRIS_MEM_ERR_BAD_INDEX.
 */
tigris_mem_error_t tigris_mem_alloc_slow(
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
 * Does NOT touch tensor_ptrs - callers must invalidate stale pointers.
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

/**
 * Return a human-readable string for a memory error code.
 *
 * @param err  Memory error code.
 * @return Static string describing the error.
 */
const char *tigris_mem_error_str(tigris_mem_error_t err);

/* Inline helpers */

/**
 * Get the current data pointer for a tensor.
 *
 * @param mem  Memory manager (read-only).
 * @param idx  Tensor index.
 * @return Pointer to tensor data, or NULL if not allocated or index is invalid.
 */
static inline void *tigris_mem_tensor_ptr(const tigris_mem_t *mem, uint16_t idx)
{
    if (!mem || idx >= mem->num_tensors) return NULL;
    return mem->tensor_ptrs[idx];
}

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_MEM_H */

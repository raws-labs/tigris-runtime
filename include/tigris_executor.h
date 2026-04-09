/**
 * @file tigris_executor.h
 * @brief Stage executor API with pluggable kernel dispatch.
 */

#ifndef TIGRIS_EXECUTOR_H
#define TIGRIS_EXECUTOR_H

#include "tigris.h"
#include "tigris_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */

/** Executor error codes returned by tigris_run(). */
typedef enum {
    TIGRIS_EXEC_OK          =  0,   /**< Success */
    TIGRIS_EXEC_ERR_NULL    = -1,   /**< Null pointer argument */
    TIGRIS_EXEC_ERR_MEM     = -2,   /**< Memory allocation failed (arena OOM) */
    TIGRIS_EXEC_ERR_KERNEL  = -3,   /**< Kernel callback returned error */
    TIGRIS_EXEC_ERR_NO_STAGES = -4, /**< Plan has no stages */
    TIGRIS_EXEC_ERR_TILE    = -5,   /**< Tiled execution error */
} tigris_exec_error_t;

/* Kernel callback */

/**
 * Kernel dispatch callback - called once per op during stage execution.
 *
 * Input tensor pointers are already set in mem; the kernel should
 * read inputs and write outputs via tigris_mem_tensor_ptr().
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Opaque user context passed through from tigris_run().
 * @return 0 on success, negative on error.
 */
typedef int (*tigris_kernel_fn)(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

/* Execution statistics */

/** Per-inference execution statistics populated by tigris_run(). */
typedef struct {
    uint16_t stages_normal;        /* stages executed without tiling */
    uint16_t stages_tiled;         /* stages executed with spatial tiling */
    uint16_t stages_chain;         /* stages executed as chain tiles */
    uint16_t total_tiles;          /* total tile iterations across all stages */
    uint32_t slow_overflow_count;  /* intra-stage allocs that overflowed to slow */
    uint32_t slow_overflow_bytes;  /* total bytes overflowed to slow */
    uint32_t loads_bytes;          /* total bytes loaded slow -> fast */
    uint32_t spills_bytes;         /* total bytes spilled fast -> slow */
    uint32_t compactions;          /* number of fast-pool compactions */
    uint32_t slow_peak;            /* high-water mark for slow_used */
} tigris_exec_stats_t;

/* Executor API */

/**
 * Run inference on a loaded plan.
 *
 * Caller must:
 *   1. Load the plan with tigris_plan_load().
 *   2. Initialise tigris_mem_t with fast + slow buffers.
 *   3. Allocate model input tensors in slow and fill with data.
 *
 * After return, model output tensors are in the slow buffer.
 *
 * @param plan      Loaded plan (read-only).
 * @param mem       Initialised memory manager.
 * @param kernel    Kernel dispatch callback (e.g. tigris_dispatch_kernel_s8).
 * @param user_ctx  Opaque context forwarded to every kernel call.
 * @param stats     Optional stats output (may be NULL).
 * @return TIGRIS_EXEC_OK on success, negative error code on failure.
 */
tigris_exec_error_t tigris_run(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats);

/**
 * Return a human-readable string for an executor error code.
 *
 * @param err  Executor error code.
 * @return Static string describing the error.
 */
const char *tigris_exec_error_str(tigris_exec_error_t err);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_EXECUTOR_H */

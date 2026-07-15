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

/** Executor error codes returned by the executor entry points. */
typedef enum {
    TIGRIS_EXEC_OK          =  0,   /**< Success */
    TIGRIS_EXEC_ERR_NULL    = -1,   /**< Null pointer argument */
    TIGRIS_EXEC_ERR_MEM     = -2,   /**< Memory allocation failed (arena OOM) */
    TIGRIS_EXEC_ERR_KERNEL  = -3,   /**< Kernel callback returned error */
    TIGRIS_EXEC_ERR_NO_STAGES = -4, /**< Plan has no stages */
    TIGRIS_EXEC_ERR_TILE    = -5,   /**< Tiled execution error */
    TIGRIS_EXEC_ERR_WORKSPACE = -6, /**< Executor workspace unavailable/too small */
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

/* Executor workspace */

/**
 * Opaque workspace size derived from the configured plan limits.  Override
 * only to add target-specific headroom; a compile-time assertion prevents an
 * undersized value from building.
 */
#ifndef TIGRIS_EXECUTOR_WORKSPACE_BYTES
#define TIGRIS_EXECUTOR_WORKSPACE_BYTES (                                      \
    64u +                                                                      \
    2u * TIGRIS_MAX_TENSORS * sizeof(uint16_t) +                               \
    (TIGRIS_MAX_STAGE_INPUTS + TIGRIS_MAX_STAGE_OUTPUTS) * sizeof(void *) +     \
    TIGRIS_MAX_CHAIN_STAGES * (                                                \
        2u * sizeof(void *) +                                                  \
        (14u + 11u * TIGRIS_MAX_SPATIAL_OPS_PER_STAGE) * sizeof(int32_t)))
#endif

/** Caller-owned, naturally aligned executor working storage. */
typedef union {
    void     *_align_pointer;
    uint64_t  _align_u64;
    double    _align_double;
    uint8_t   bytes[TIGRIS_EXECUTOR_WORKSPACE_BYTES];
} tigris_executor_workspace_t;

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
 * This compatibility entry point uses one process-global workspace and is
 * therefore not re-entrant or safe for concurrent inference. Define
 * TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0 to omit that storage and use
 * tigris_run_with_workspace() instead; this function then returns
 * TIGRIS_EXEC_ERR_WORKSPACE.
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
 * Re-entrant executor entry point using explicit caller-owned workspace.
 * The workspace may be static, global, heap-backed, or task-local, but must
 * remain exclusively owned for the duration of the call.
 *
 * @return TIGRIS_EXEC_OK on success, TIGRIS_EXEC_ERR_WORKSPACE when workspace
 *         is NULL, or another negative executor error.
 */
tigris_exec_error_t tigris_run_with_workspace(
    const tigris_plan_t          *plan,
    tigris_mem_t                 *mem,
    tigris_kernel_fn              kernel,
    void                         *user_ctx,
    tigris_exec_stats_t          *stats,
    tigris_executor_workspace_t  *workspace);

/** Return sizeof(tigris_executor_workspace_t) for language bindings/allocators. */
size_t tigris_executor_workspace_size(void);

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

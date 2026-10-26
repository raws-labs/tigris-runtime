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
    TIGRIS_EXEC_ERR_MEM     = -2,   /**< Arena too small for the plan, found up front or in the run */
    TIGRIS_EXEC_ERR_KERNEL  = -3,   /**< Kernel callback returned error */
    TIGRIS_EXEC_ERR_NO_STAGES = -4, /**< Plan has no stages */
    TIGRIS_EXEC_ERR_TILE    = -5,   /**< Tiled execution error */
    TIGRIS_EXEC_ERR_WORKSPACE = -6, /**< Executor workspace unavailable/too small */
    TIGRIS_EXEC_ERR_STATE   = -7,   /**< Plan keeps state and none was given, or too little */
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
 * @param user_ctx  Opaque user context passed through from the run call.
 * @return 0 on success, negative on error.
 */
typedef int (*tigris_kernel_fn)(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

/* Execution statistics */

/** Per-inference execution statistics filled by the executor entry points. */
typedef struct tigris_exec_stats {
    uint16_t stages_normal;        /* stages executed without tiling */
    uint16_t stages_tiled;         /* stages executed with spatial tiling */
    uint16_t stages_chain;         /* stages executed as chain tiles */
    uint16_t total_tiles;          /* total tile iterations across all stages */
    uint32_t slow_overflow_count;  /* intra-stage allocs that overflowed to slow */
    uint32_t slow_overflow_bytes;  /* total bytes overflowed to slow */
    uint32_t loads_bytes;          /* total bytes loaded slow -> fast */
    uint32_t spills_bytes;         /* total bytes spilled fast -> slow */
    uint32_t compactions;          /* number of fast and slow pool compactions */
    uint32_t slow_peak;            /* high-water mark for slow_used */
    uint32_t weight_bytes;         /* uncompressed weight bytes written to fast */
    uint32_t copy_bytes;           /* state and control-flow bytes copied */
} tigris_exec_stats_t;

/* Executor workspace */

/** Portable C/C++ alignment constants used by the workspace sizing formula. */
typedef struct {
    char padding;
    void *value;
} tigris_executor_pointer_alignment_t;
typedef struct {
    char padding;
    int32_t value;
} tigris_executor_int32_alignment_t;
typedef struct {
    char padding;
    uint16_t value;
} tigris_executor_uint16_alignment_t;

#define TIGRIS_EXECUTOR_POINTER_ALIGNMENT \
    offsetof(tigris_executor_pointer_alignment_t, value)
#define TIGRIS_EXECUTOR_INT32_ALIGNMENT \
    offsetof(tigris_executor_int32_alignment_t, value)
#define TIGRIS_EXECUTOR_UINT16_ALIGNMENT \
    offsetof(tigris_executor_uint16_alignment_t, value)

/**
 * Exact caller-buffer capacity for a plan's executor metadata.
 *
 * The leading alignment allowances let an arbitrary byte buffer be passed to
 * tigris_run_with_workspace_buffer().  Generated code supplies the plan's own
 * figures, so the build limits consume no SRAM: chain_stages is the longest
 * chain, chain_spatial_ops the most height-spatial operators in one chain
 * (summed over its stages) and subgraph_depth the most control-flow operators
 * running one inside another.
 */
#define TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(                            \
    tensors, inputs, outputs, chain_stages, chain_spatial_ops, subgraph_depth) \
    ((TIGRIS_EXECUTOR_POINTER_ALIGNMENT - 1u) +                                \
     ((size_t)(inputs) + (size_t)(outputs) +                                   \
      ((chain_stages) != 0u ? (size_t)(outputs) : 0u) +                        \
      2u * (size_t)(chain_stages)) * sizeof(void *) +                          \
     (TIGRIS_EXECUTOR_INT32_ALIGNMENT - 1u) +                                  \
     (15u * (size_t)(chain_stages) + 13u * (size_t)(chain_spatial_ops) +       \
      2u * (size_t)(tensors) + (size_t)(outputs)) * sizeof(int32_t) +          \
     (TIGRIS_EXECUTOR_UINT16_ALIGNMENT - 1u) +                                 \
     (2u * (size_t)(tensors) + 4u + 4u * (size_t)(subgraph_depth)) *           \
         sizeof(uint16_t))

/**
 * The most height-spatial operators in one chain and the deepest control-flow
 * nesting a plan within the configured limits can have: a plan has at most
 * 65535 operators, and more graphs than its nesting depth.
 */
#define TIGRIS_EXECUTOR_BUILD_CHAIN_SPATIAL_OPS                                \
    ((uint64_t)TIGRIS_MAX_CHAIN_STAGES * TIGRIS_MAX_SPATIAL_OPS_PER_STAGE < 65535u \
         ? (uint64_t)TIGRIS_MAX_CHAIN_STAGES * TIGRIS_MAX_SPATIAL_OPS_PER_STAGE    \
         : 65535u)
#define TIGRIS_EXECUTOR_BUILD_SUBGRAPH_DEPTH                                   \
    (TIGRIS_MAX_SUBGRAPH_DEPTH < TIGRIS_MAX_SUBGRAPHS                          \
         ? TIGRIS_MAX_SUBGRAPH_DEPTH : TIGRIS_MAX_SUBGRAPHS - 1u)

/**
 * Opaque workspace size for the largest plan the configured limits admit, with
 * 64 bytes of alignment headroom.  Override only to add target-specific
 * headroom; a compile-time assertion prevents an undersized value from
 * building.
 */
#ifndef TIGRIS_EXECUTOR_WORKSPACE_BYTES
#define TIGRIS_EXECUTOR_WORKSPACE_BYTES (                                      \
    64u +                                                                      \
    (TIGRIS_MAX_STAGE_INPUTS + 2u * TIGRIS_MAX_STAGE_OUTPUTS +                 \
     2u * TIGRIS_MAX_CHAIN_STAGES) * sizeof(void *) +                          \
    (15u * TIGRIS_MAX_CHAIN_STAGES + 13u * TIGRIS_EXECUTOR_BUILD_CHAIN_SPATIAL_OPS + \
     2u * TIGRIS_MAX_TENSORS + TIGRIS_MAX_STAGE_OUTPUTS) * sizeof(int32_t) +    \
    (2u * TIGRIS_MAX_TENSORS + 4u + 4u * TIGRIS_EXECUTOR_BUILD_SUBGRAPH_DEPTH) * \
        sizeof(uint16_t))
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
 *   2. Initialize tigris_mem_t with a fast buffer and, unless the plan needs
 *      none, a slow buffer.
 *   3. Place every model input with tigris_input_alloc() or
 *      tigris_input_write(), or allocate it in the slow arena, and fill it.
 *
 * A plan of schema 11 or later is refused with TIGRIS_EXEC_ERR_MEM before any
 * stage runs when the fast arena above the run's floor is smaller than
 * tigris_fast_arena_required(). The floor is the arena's fill level on entry,
 * or, for inputs read in place, the start of the room left below them. Earlier
 * plans carry no requirement and are not checked.
 *
 * After return, model output tensors are in the slow buffer. A plan that reads
 * its inputs in place (see tigris_input_alloc()) consumes the inputs at the top
 * of the fast arena, decompresses its weights into the room left below them,
 * and leaves its outputs there, readable until the next allocation in that
 * arena.
 *
 * This is the primary entry point. It is re-entrant: concurrent inferences
 * each pass their own workspace and memory manager, with a kernel dispatcher
 * that is itself safe to call concurrently. The workspace may be static,
 * global, heap-backed, or task-local, but must remain exclusively owned for the
 * duration of the call. A caller that prefers a plan-sized buffer to the fixed
 * workspace type uses tigris_run_with_workspace_buffer().
 *
 * @param plan      Loaded plan (read-only).
 * @param mem       Initialised memory manager.
 * @param kernel    Kernel dispatch callback (e.g. tigris_dispatch_kernel_s8).
 * @param user_ctx  Opaque context forwarded to every kernel call.
 * @param stats     Optional stats output (may be NULL).
 * @param workspace Caller-owned executor workspace.
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

/**
 * Return the workspace capacity required by this loaded plan.
 *
 * The result includes alignment headroom, so a buffer of exactly this size may
 * start at any address.  Returns 0 when plan metadata is incomplete.
 */
size_t tigris_executor_workspace_required(const tigris_plan_t *plan);

/**
 * Convenience form of tigris_run_with_workspace() for a caller-owned byte
 * buffer instead of the fixed workspace type.
 *
 * This is the plan-sized API used by generated code.  The buffer may be
 * statically allocated, task-local, or heap-backed; the runtime never allocates
 * it and does not retain it after the call.
 *
 * @param workspace       Caller-owned byte buffer.
 * @param workspace_size  Available bytes, normally obtained from
 *                        tigris_executor_workspace_required().
 * @return TIGRIS_EXEC_ERR_WORKSPACE when the buffer is NULL or too small.
 */
tigris_exec_error_t tigris_run_with_workspace_buffer(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace,
    size_t               workspace_size);

/**
 * Compatibility entry point: tigris_run_with_workspace() with one
 * process-global workspace, so it is not re-entrant or safe for concurrent
 * inference. Define TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0 to omit that
 * storage; this function then returns TIGRIS_EXEC_ERR_WORKSPACE. Contract and
 * remaining parameters as tigris_run_with_workspace().
 */
tigris_exec_error_t tigris_run(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats);

/**
 * Run a plan that keeps variables across invocations.
 *
 * The state buffer holds every variable between runs: each run reads the
 * values the last one left and writes back the ones it assigns. Prepare it
 * once with tigris_state_init(), which is also how it is reset. A plan
 * without state also runs here, with state NULL and state_size 0.
 *
 * @return TIGRIS_EXEC_ERR_STATE when the plan keeps state and the buffer is
 *         NULL or smaller than tigris_state_required(); otherwise as
 *         tigris_run_with_workspace_buffer().
 */
tigris_exec_error_t tigris_run_with_state(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace,
    size_t               workspace_size,
    void                *state,
    size_t               state_size);

/** Bytes of state a plan keeps across invocations; 0 for a plan without. */
size_t tigris_state_required(const tigris_plan_t *plan);

/**
 * Write every variable's initial value into a state buffer, which prepares
 * it for the first run and resets it after any number of runs.
 *
 * @return TIGRIS_EXEC_OK, or TIGRIS_EXEC_ERR_STATE when the buffer is NULL
 *         or smaller than tigris_state_required().
 */
tigris_exec_error_t tigris_state_init(const tigris_plan_t *plan, void *state, size_t state_size);

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

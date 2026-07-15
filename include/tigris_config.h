/**
 * @file tigris_config.h
 * @brief Compile-time execution limits and workspace policy.
 *
 * Define any value on the compiler command line before including TiGrIS
 * headers to tune the runtime for a target.  The loader rejects plans that
 * exceed the selected limits before the executor can index bounded storage.
 */

#ifndef TIGRIS_CONFIG_H
#define TIGRIS_CONFIG_H

/** Maximum runtime tensors in one loaded plan. */
#ifndef TIGRIS_MAX_TENSORS
#define TIGRIS_MAX_TENSORS 512u
#endif

/** Maximum inputs and outputs carried by one execution stage. */
#ifndef TIGRIS_MAX_STAGE_INPUTS
#define TIGRIS_MAX_STAGE_INPUTS 16u
#endif
#ifndef TIGRIS_MAX_STAGE_OUTPUTS
#define TIGRIS_MAX_STAGE_OUTPUTS 16u
#endif

/** Maximum stages in one tile-through chain. */
#ifndef TIGRIS_MAX_CHAIN_STAGES
#define TIGRIS_MAX_CHAIN_STAGES 16u
#endif

/** Maximum spatial operators composed within one chained stage. */
#ifndef TIGRIS_MAX_SPATIAL_OPS_PER_STAGE
#define TIGRIS_MAX_SPATIAL_OPS_PER_STAGE 8u
#endif

/**
 * Keep the source-compatible tigris_run() wrapper and its process-global
 * workspace.  Set to 0 for re-entrant/multi-task builds; those builds call
 * tigris_run_with_workspace() with caller-owned storage instead.
 */
#ifndef TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE
#define TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE 1
#endif

#if TIGRIS_MAX_TENSORS < 1u || TIGRIS_MAX_TENSORS > 65535u
#error "TIGRIS_MAX_TENSORS must be in [1, 65535]"
#endif
#if TIGRIS_MAX_STAGE_INPUTS < 1u || TIGRIS_MAX_STAGE_INPUTS > 65535u
#error "TIGRIS_MAX_STAGE_INPUTS must be in [1, 65535]"
#endif
#if TIGRIS_MAX_STAGE_OUTPUTS < 1u || TIGRIS_MAX_STAGE_OUTPUTS > 65535u
#error "TIGRIS_MAX_STAGE_OUTPUTS must be in [1, 65535]"
#endif
#if TIGRIS_MAX_CHAIN_STAGES < 2u || TIGRIS_MAX_CHAIN_STAGES > 65535u
#error "TIGRIS_MAX_CHAIN_STAGES must be in [2, 65535]"
#endif
#if TIGRIS_MAX_SPATIAL_OPS_PER_STAGE < 1u
#error "TIGRIS_MAX_SPATIAL_OPS_PER_STAGE must be positive"
#endif
#if TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE != 0 && \
    TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE != 1
#error "TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE must be 0 or 1"
#endif

#endif /* TIGRIS_CONFIG_H */

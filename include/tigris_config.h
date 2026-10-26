/**
 * @file tigris_config.h
 * @brief Compile-time execution limits and workspace policy.
 *
 * Define any value on the compiler command line before including TiGrIS
 * headers to tune the runtime for a target.  The loader rejects plans that
 * exceed the selected limits, and tigris_executor_workspace_t is sized for the
 * largest plan they admit; other loader and executor storage follows the
 * loaded plan.
 */

#ifndef TIGRIS_CONFIG_H
#define TIGRIS_CONFIG_H

#include <stdint.h>

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

/** Maximum graphs, the main graph included, in a plan with control flow. */
#ifndef TIGRIS_MAX_SUBGRAPHS
#define TIGRIS_MAX_SUBGRAPHS 16u
#endif

/** Maximum control-flow operators running at once, one inside another. */
#ifndef TIGRIS_MAX_SUBGRAPH_DEPTH
#define TIGRIS_MAX_SUBGRAPH_DEPTH 4u
#endif

/** Maximum spatial operators composed within one chained stage. */
#ifndef TIGRIS_MAX_SPATIAL_OPS_PER_STAGE
#define TIGRIS_MAX_SPATIAL_OPS_PER_STAGE 8u
#endif

/**
 * Keep the compatibility tigris_run() wrapper and its process-global
 * workspace. Set to 0 for generated, re-entrant, or multi-task builds; those
 * call tigris_run_with_workspace() or tigris_run_with_workspace_buffer() with
 * caller-owned storage.
 */
#ifndef TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE
#define TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE 1
#endif

/**
 * Build limits in the terms of the limits above: what a plan needs, as
 * tigris_plan_load_ex() reports it, or what this build allows, as
 * tigris_build_limits() reports it. The loader refuses a plan that needs more
 * than this build allows.
 */
typedef struct {
    uint16_t tensors;
    uint16_t stage_inputs;
    uint16_t stage_outputs;
    uint16_t chain_stages;
    uint16_t spatial_ops_per_stage;
} tigris_plan_limits_t;

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
#if TIGRIS_MAX_SUBGRAPHS < 1u || TIGRIS_MAX_SUBGRAPHS > 65535u
#error "TIGRIS_MAX_SUBGRAPHS must be in [1, 65535]"
#endif
#if TIGRIS_MAX_SUBGRAPH_DEPTH < 1u || TIGRIS_MAX_SUBGRAPH_DEPTH > 65535u
#error "TIGRIS_MAX_SUBGRAPH_DEPTH must be in [1, 65535]"
#endif
#if TIGRIS_MAX_SPATIAL_OPS_PER_STAGE < 1u
#error "TIGRIS_MAX_SPATIAL_OPS_PER_STAGE must be positive"
#endif
#if TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE != 0 && \
    TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE != 1
#error "TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE must be 0 or 1"
#endif

#endif /* TIGRIS_CONFIG_H */

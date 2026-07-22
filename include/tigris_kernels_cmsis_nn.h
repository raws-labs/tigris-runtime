/**
 * @file tigris_kernels_cmsis_nn.h
 * @brief CMSIS-NN accelerated int8 kernel dispatch for TiGrIS runtime.
 *
 * Provides tigris_dispatch_kernel_cmsis_nn(), a concrete tigris_kernel_fn
 * that maps TiGrIS ops to ARM's CMSIS-NN optimized implementations.
 * Unsupported ops fall back to tigris_dispatch_kernel_s8().
 *
 * Requires TIGRIS_HAS_CMSIS_NN to be defined and CMSIS-NN headers available.
 */

#ifndef TIGRIS_KERNELS_CMSIS_NN_H
#define TIGRIS_KERNELS_CMSIS_NN_H

#ifdef TIGRIS_HAS_CMSIS_NN

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * CMSIS-NN accelerated int8 kernel dispatch - concrete tigris_kernel_fn.
 *
 * Supported ops: Conv, DepthwiseConv, FullyConnected, and compatible-
 * quantization AvgPool/GlobalAvg. The vendored generic Conv/Depthwise
 * implementations honor non-unit dilation, so non-tiled dilated ops remain
 * accelerated. Tiled spatial ops, requantizing pooling, and unsupported ops
 * fall back to tigris_dispatch_kernel_s8().
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Unused (may be NULL).
 * @return 0 on success, -1 on unsupported op type.
 */
int tigris_dispatch_kernel_cmsis_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

/**
 * Query the exact CMSIS-NN workspace reservation for a loaded plan.
 *
 * The result combines the linked CMSIS-NN buffer-size APIs with bounded
 * scalar-to-per-channel quantization expansion storage. Each region is
 * rounded up to the adapter's 16-byte alignment.
 *
 * @param plan  Loaded plan.
 * @return Workspace bytes, 0 when no native op needs workspace, or UINT32_MAX
 *         for an invalid plan or an unrepresentable result.
 */
uint32_t tigris_cmsis_nn_scratch_required(const tigris_plan_t *plan);

/**
 * Query the total fast arena required for CMSIS-NN execution.
 *
 * Combines tigris_fast_arena_required() with the exact CMSIS-NN workspace
 * reservation while preserving the plan's full core activation capacity.
 * The arena base must be aligned to TIGRIS_TENSOR_ALIGN.
 *
 * @param plan  Loaded plan.
 * @return Total bytes, or UINT32_MAX when the requirement is invalid or
 *         cannot be represented.
 */
uint32_t tigris_cmsis_nn_fast_arena_required(const tigris_plan_t *plan);

/**
 * Reserve the CMSIS-NN workspace from the top of the fast arena.
 *
 * Sizes the largest kernel buffer and scalar quantization expansion across the
 * plan and carves aligned regions from mem (reducing mem->fast_size), so no
 * per-op CMSIS-NN workspace is allocated on the stack. Repeated calls for the
 * same memory manager are idempotent when the existing reservation is large
 * enough; a different manager or larger reservation requires deinitialization.
 * Required before dispatch when tigris_cmsis_nn_scratch_required(plan) is
 * positive. Dispatch fails closed rather than falling back to hidden static
 * scratch when preparation was skipped.
 *
 * @param plan  Loaded plan.
 * @param mem   Memory manager (fast_size is reduced in place).
 * @return 0 on success, -1 if mem/plan is NULL or the arena is smaller than
 *         tigris_cmsis_nn_fast_arena_required(plan).
 */
int tigris_cmsis_nn_prepare(const tigris_plan_t *plan, tigris_mem_t *mem);

/**
 * Release the CMSIS-NN scratch reservation and restore the original fast-arena
 * size. Call only while no inference is using the memory manager.
 *
 * @param mem  The same memory manager passed to prepare.
 * @return 0 on success, -1 for NULL or a manager not owned by CMSIS-NN.
 */
int tigris_cmsis_nn_deinit(tigris_mem_t *mem);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_HAS_CMSIS_NN */

#endif /* TIGRIS_KERNELS_CMSIS_NN_H */

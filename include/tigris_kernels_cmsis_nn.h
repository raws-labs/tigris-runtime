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
 * Supported ops: Conv, DepthwiseConv, FullyConnected, AvgPool/GlobalAvg.
 * All others fall back to tigris_dispatch_kernel_s8().
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

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_HAS_CMSIS_NN */

#endif /* TIGRIS_KERNELS_CMSIS_NN_H */

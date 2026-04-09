/**
 * @file tigris_kernels.h
 * @brief Reference float32 kernel dispatch for TiGrIS runtime.
 *
 * Provides tigris_dispatch_kernel(), a concrete tigris_kernel_fn that
 * switches on op_type and calls naive float32 implementations for all
 * supported ops. Data layout: NHWC.
 */

#ifndef TIGRIS_KERNELS_H
#define TIGRIS_KERNELS_H

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Reference float32 kernel dispatch - concrete tigris_kernel_fn.
 *
 * Reads input tensors, writes output tensors via mem->tensor_ptrs.
 * Weight/bias data accessed via tigris_op_weight() / tigris_op_bias().
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Unused (may be NULL).
 * @return 0 on success, -1 on unsupported op type.
 */
int tigris_dispatch_kernel(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_KERNELS_H */

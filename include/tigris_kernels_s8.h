/**
 * @file tigris_kernels_s8.h
 * @brief Reference int8 (symmetric) kernel dispatch for TiGrIS runtime.
 *
 * Provides tigris_dispatch_kernel_s8(), a concrete tigris_kernel_fn that
 * switches on op_type and calls naive int8 implementations. Data layout: NHWC.
 *
 * Quantization convention (TFLite/CMSIS-NN compatible):
 *   acc = sum(input - input_zp) * weight + bias  (int32 accumulator)
 *   output = clamp(multiply_by_quantized_multiplier(acc, M, shift) + output_zp)
 */

#ifndef TIGRIS_KERNELS_S8_H
#define TIGRIS_KERNELS_S8_H

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Reference int8 quantized kernel dispatch - concrete tigris_kernel_fn.
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Unused (may be NULL).
 * @return 0 on success, -1 on unsupported op type.
 */
int tigris_dispatch_kernel_s8(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_KERNELS_S8_H */

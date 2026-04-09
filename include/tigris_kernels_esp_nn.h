/**
 * @file tigris_kernels_esp_nn.h
 * @brief ESP-NN accelerated int8 kernel dispatch for TiGrIS runtime.
 *
 * Provides tigris_dispatch_kernel_esp_nn(), a concrete tigris_kernel_fn
 * that maps TiGrIS ops to Espressif's ESP-NN optimized implementations.
 * Unsupported ops fall back to tigris_dispatch_kernel_s8().
 *
 * Requires TIGRIS_HAS_ESP_NN to be defined and ESP-NN headers available.
 */

#ifndef TIGRIS_KERNELS_ESP_NN_H
#define TIGRIS_KERNELS_ESP_NN_H

#ifdef TIGRIS_HAS_ESP_NN

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Pre-allocate ESP-NN scratch buffers from the arena.
 *
 * Scans all ops in the plan to compute maximum scratch, depthwise output,
 * and asymmetric-padding buffer sizes.  Carves them from the top of
 * mem->fast_size (shrinking the available arena) so that no heap allocation
 * occurs during inference.
 *
 * Must be called ONCE after tigris_mem_init() and before tigris_run().
 * The reduced fast_size persists across tigris_mem_init() re-inits as long
 * as they use the already-reduced mem->fast_size value.
 *
 * @param plan  Loaded plan (read-only, used to scan op shapes).
 * @param mem   Memory manager (fast_size is reduced in place).
 * @return 0 on success, -1 if the arena is too small.
 */
int tigris_esp_nn_prepare(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem);

/**
 * Print Conv dispatch statistics (SRAM scratch vs PSRAM scratch vs fallback).
 */
void tigris_esp_nn_print_conv_stats(void);

/**
 * ESP-NN accelerated int8 kernel dispatch - concrete tigris_kernel_fn.
 *
 * Supported ops: Conv, DepthwiseConv, FullyConnected, AvgPool/GlobalAvg, Add.
 * All others fall back to tigris_dispatch_kernel_s8().
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Unused (may be NULL).
 * @return 0 on success, -1 on unsupported op type.
 */
int tigris_dispatch_kernel_esp_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_HAS_ESP_NN */

#endif /* TIGRIS_KERNELS_ESP_NN_H */

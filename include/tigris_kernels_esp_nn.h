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
 * Prepare platform-managed ESP-NN workspace.
 *
 * Scans all ops in the plan to compute maximum scratch, depthwise output,
 * and asymmetric-padding buffer sizes. On ESP-IDF it obtains aligned PSRAM
 * workspace and may use otherwise-free internal SRAM for faster Conv scratch.
 * The supplied TiGrIS fast arena is not reduced. Missing optional workspace
 * causes the affected op to fall back to s8_ref rather than changing its
 * semantics, and no workspace allocation occurs inside tigris_run().
 *
 * Must be called ONCE after tigris_mem_init() and before tigris_run().
 * Workspace is held in process-lifetime adapter state; the current API has no
 * deinitialization or repeat-prepare ownership contract.
 *
 * @param plan  Loaded plan (read-only, used to scan op shapes).
 * @param mem   Initialized memory manager (read-only sizing input).
 * @return 0 after a valid scan/setup, or -1 for invalid/overflowing metadata
 *         or an unavailable mandatory depthwise output buffer.
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
 * Supported ops: unit-dilation Conv/DepthwiseConv, FullyConnected, compatible-
 * quantization AveragePool, and Add. Conv/DepthwiseConv with non-unit dilation,
 * tiled or requantizing AveragePool, GlobalAveragePool, and other unsupported
 * ops fall back to tigris_dispatch_kernel_s8().
 *
 * @param plan      Loaded plan (read-only).
 * @param op        Current operator descriptor.
 * @param op_index  Global index of the op in plan->ops[].
 * @param mem       Memory manager with tensor pointers set.
 * @param user_ctx  Forwarded to s8_ref when a fallback route is selected.
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

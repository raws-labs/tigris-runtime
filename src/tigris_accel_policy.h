/**
 * @file tigris_accel_policy.h
 * @brief Host-testable pre-routing policy for optional int8 accelerators.
 */

#ifndef TIGRIS_ACCEL_POLICY_H
#define TIGRIS_ACCEL_POLICY_H

#include "tigris_executor.h"

typedef enum {
    TIGRIS_ACCEL_ESP_NN,
    TIGRIS_ACCEL_CMSIS_NN,
} tigris_accel_backend_t;

typedef enum {
    TIGRIS_ACCEL_ROUTE_ADAPTER,
    TIGRIS_ACCEL_ROUTE_S8_REF,
} tigris_accel_route_t;

/**
 * Decide whether an op must bypass an accelerated adapter.
 *
 * ESP-NN's Conv/Depthwise APIs used by this runtime encode only unit dilation,
 * so either effective dilation axis above one routes to s8_ref. CMSIS-NN's
 * generic Conv/Depthwise implementations honor dilation. Both native
 * AveragePool adapters preserve input quantization and therefore route to
 * s8_ref when output quantization differs. Non-tile-aware pooling/spatial
 * adapter paths also route to the reference kernel here.
 */
tigris_accel_route_t tigris_accel_pre_route(
    tigris_accel_backend_t backend,
    const tigris_plan_t   *plan,
    const tigris_op_t     *op,
    int                    tile_active);

/**
 * Validate the workspace required to pre-pad an ESP-NN Conv input.
 *
 * The ESP adapter uses this for asymmetric padding, which ESP-NN cannot encode
 * directly. Overflow, invalid dimensions, and undersized workspace all return
 * false so dispatch can fall back to s8_ref without changing padding semantics.
 *
 * @param required_bytes  Optional exact byte count (one byte per int8 sample).
 * @return 1 when workspace_bytes is sufficient, otherwise 0.
 */
int tigris_accel_esp_pad_workspace_fits(
    int32_t input_h,
    int32_t input_w,
    int32_t input_c,
    uint16_t pad_top,
    uint16_t pad_bottom,
    uint16_t pad_left,
    uint16_t pad_right,
    uint32_t workspace_bytes,
    uint32_t *required_bytes);

/**
 * Execute the reference route selected by tigris_accel_pre_route().
 *
 * @param handled  Set to 1 when s8_ref ran, or 0 when the caller should invoke
 *                 its accelerated adapter.
 * @return Kernel result when handled, otherwise 0.
 */
int tigris_accel_try_s8_ref(
    tigris_accel_backend_t backend,
    const tigris_plan_t   *plan,
    const tigris_op_t     *op,
    uint16_t               op_index,
    tigris_mem_t          *mem,
    void                  *user_ctx,
    int                   *handled);

#endif /* TIGRIS_ACCEL_POLICY_H */

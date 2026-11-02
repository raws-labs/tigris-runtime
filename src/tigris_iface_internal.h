/**
 * @file tigris_iface_internal.h
 * @brief The int8 quantization rules the input interface and the QUANTIZE kernel share.
 *
 * Internal to src/. Not installed and not part of the public API.
 */
#ifndef TIGRIS_IFACE_INTERNAL_H
#define TIGRIS_IFACE_INTERNAL_H

#include <stdbool.h>
#include <stdint.h>

#include "tigris.h"

/**
 * A float32 value in an int8 encoding, as TFLite's QUANTIZE computes it: a
 * float32 division by the scale, rounded half away from zero, plus the zero
 * point, clamped to int8.
 */
int8_t tigris_quantize_value(float value, const tigris_quant_param_t *quant);

/**
 * A float32 value in an int8 encoding, as ONNX QuantizeLinear computes it: the
 * same float32 division, rounded half to even, plus the zero point, clamped
 * to int8. Exact for every float32 and independent of the rounding mode.
 */
int8_t tigris_quantize_value_even(float value, const tigris_quant_param_t *quant);

/**
 * Whether the plan quantizes float32 with ties to even: a schema 12 or later
 * plan with TIGRIS_FLAG_QUANTIZE_TIES_EVEN set. Any other plan rounds half away
 * from zero.
 */
bool tigris_plan_quantizes_ties_even(const tigris_plan_t *plan);

#endif /* TIGRIS_IFACE_INTERNAL_H */

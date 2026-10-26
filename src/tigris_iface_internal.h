/**
 * @file tigris_iface_internal.h
 * @brief The int8 quantization rule the input interface and the QUANTIZE kernel share.
 *
 * Internal to src/. Not installed and not part of the public API.
 */
#ifndef TIGRIS_IFACE_INTERNAL_H
#define TIGRIS_IFACE_INTERNAL_H

#include <stdint.h>

#include "tigris.h"

/**
 * A float32 value in an int8 encoding, as TFLite's QUANTIZE computes it: a
 * float32 division by the scale, rounded half away from zero, plus the zero
 * point, clamped to int8.
 */
int8_t tigris_quantize_value(float value, const tigris_quant_param_t *quant);

#endif /* TIGRIS_IFACE_INTERNAL_H */

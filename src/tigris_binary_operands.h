/**
 * @file tigris_binary_operands.h
 * @brief Where a binary operator's two operands come from, shared by the
 * float and int8 kernels.
 *
 * Internal to src/. Not installed and not part of the public API. Defined
 * in tigris_kernels.c.
 */
#ifndef TIGRIS_BINARY_OPERANDS_H
#define TIGRIS_BINARY_OPERANDS_H

#include "tigris.h"
#include "tigris_mem.h"

#include <stdint.h>

/** Operands of up to this rank are read by output coordinate when a
 * broadcast is not a repetition; the others need no coordinates. */
#define TIGRIS_BROADCAST_MAX_RANK 4u
#define TIGRIS_BINARY_MAX_RANK 8u

/** Both operands of a binary operator, in operand order, against the output
 * `output`. An operand is read densely (period 0, general 0) and follows a
 * tile; repeats every `period` elements, which is any broadcast over leading
 * axes, and is read whole under a tile; or, when `general` is set, by the
 * output coordinate through `stride` (0 on a broadcast axis) and is never
 * tiled. `dims` is the output's shape left-padded to the broadcast rank. */
typedef struct {
    const uint8_t *data[2];
    uint32_t period[2];
    uint8_t general[2];
    uint32_t stride[2][TIGRIS_BROADCAST_MAX_RANK];
    uint32_t dims[TIGRIS_BROADCAST_MAX_RANK];
    const tigris_quant_param_t *quant[2];
    uint16_t output;
} tigris_binary_operands_t;

/** Resolves the operands of `op`: two tensors, or one tensor and a constant
 * from the weight table, each of the output's shape or broadcast to it. The
 * constant-operand attribute says which operand the constant is, how it is
 * quantized and, in its long form, its shape; without it only a float Add or
 * Mul takes a constant, as its second operand, holding one value, one per
 * channel, or one per element. `dtype` is 1 (float32) or 3 (int8). Returns 0
 * for anything else. */
int tigris_binary_operands(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, uint8_t dtype, tigris_binary_operands_t *out);

/** The element of operand k that output element i reads, for a general
 * broadcast. */
uint32_t tigris_binary_general_index(const tigris_binary_operands_t *operands,
                                     uint8_t k, uint32_t i);

#endif /* TIGRIS_BINARY_OPERANDS_H */

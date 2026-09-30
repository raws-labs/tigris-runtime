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

/** Both operands of a binary operator, in operand order. An operand with
 * period 0 has the output's shape and follows a tile; one with period n
 * repeats every n elements and is read whole under a tile. */
typedef struct {
    const uint8_t *data[2];
    uint32_t period[2];
    const tigris_quant_param_t *quant[2];
} tigris_binary_operands_t;

/** Resolves the operands of `op`: two tensors, the second the first's shape
 * or one value per channel, or one tensor and a constant from the weight
 * table. The constant-operand attribute says which operand the constant is
 * and how it is quantized; without it only a float Add or Mul takes a
 * constant, as its second operand. `dtype` is 1 (float32) or 3 (int8).
 * Returns 0 for anything else. */
int tigris_binary_operands(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, uint8_t dtype, tigris_binary_operands_t *out);

#endif /* TIGRIS_BINARY_OPERANDS_H */

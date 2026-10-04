/**
 * @file tigris_binary_operands.h
 * @brief Operand resolution for binary operators and padding, shared by the
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
#define TIGRIS_BROADCAST_MAX_RANK 5u
#define TIGRIS_BINARY_MAX_RANK 8u

/** Two binary operands or three select operands, in operand order, against the output
 * `output`. An operand is read densely (period 0, general 0) and follows a
 * tile; repeats every `period` elements, which is any broadcast over leading
 * axes, and is read whole under a tile; or, when `general` is set, by the
 * output coordinate through `stride` (0 on a broadcast axis) and is never
 * tiled. `dims` is the output's shape left-padded to the broadcast rank. */
typedef struct {
    const uint8_t *data[3];
    uint32_t period[3];
    uint8_t general[3];
    uint32_t stride[3][TIGRIS_BROADCAST_MAX_RANK];
    uint32_t dims[TIGRIS_BROADCAST_MAX_RANK];
    const tigris_quant_param_t *quant[3];
    uint16_t output;
} tigris_binary_operands_t;

/** Resolves the operands of `op`: two or three tensors, with at most one replaced by a constant
 * from the weight table, each of the output's shape or broadcast to it. The
 * constant-operand attribute says which operand the constant is, how it is
 * quantized and, in its long form, its shape; without it only a float Add or
 * Mul takes a constant, as its second operand, holding one value, one per
 * channel, or one per element. `dtype` is 1 (float32), 3 (int8), or 9 (bool). Selection
 * always reads its first operand as bool. Returns 0
 * for anything else. */
int tigris_binary_operands(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, uint8_t dtype, tigris_binary_operands_t *out);

/** The element of operand k that output element i reads, for a general
 * broadcast. */
uint32_t tigris_binary_general_index(const tigris_binary_operands_t *operands,
                                     uint8_t k, uint32_t i);

/* Whether a reduction or movement preserves the existing row-band geometry. */
int tigris_op_independent_band(const tigris_plan_t *plan, const tigris_op_t *op,
                                uint16_t op_index, int row_tiled);

/* Bool operations and byte-preserving selection share both dispatchers. */
/** All of one axis of a rank-3 bool tensor, kept at one or dropped. */
int tigris_reduce_all_execute(const tigris_plan_t *plan, const tigris_op_t *op,
                              uint16_t op_index, tigris_mem_t *mem);

int tigris_bool_execute(const tigris_plan_t *plan, const tigris_op_t *op,
                        uint16_t op_index, tigris_mem_t *mem);

/** Pads `op`'s input into its output: the output is filled with `fill`
 * (one element of `element` bytes), then the input is copied in at the
 * leading pad of every axis, from the pads attribute in stored axis order.
 * Returns 0 on success, -1 for anything it cannot place. */
int tigris_pad(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
               const tigris_mem_t *mem, uint32_t element, const uint8_t *fill);

#endif /* TIGRIS_BINARY_OPERANDS_H */

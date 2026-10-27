/**
 * @file tigris_band.c
 * @brief Band geometry: how a stage and its tensors divide into bands.
 *
 * The rules the executor bands a stage by, the band kind it classifies a
 * stage as, and the tile facts the loader derives from them and checks.
 */

#include "tigris_band.h"
#include "tigris_loader.h"
#include "tigris_transpose_band.h"

#include <string.h>

/**
 * Find the first axis-1 spatial op (Conv/DWConv/Pool/Conv1D/ConvTranspose) in a
 * stage. Returns its index in plan->ops, or -1 if none. ConvTranspose is only
 * located here so the 2D executor can invert its rectangle; the 1D executor
 * fails closed on it (see exec_stage_tiled).
 */
int tigris_stage_spatial_op(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t t = plan->ops[sops[j]].op_type;
        if (tigris_height_spatial_op(t) || tigris_height_upsample_op(t) ||
            t == TIGRIS_OP_CONV1D || t == TIGRIS_OP_CONV_TRANSPOSE)
            return (int)sops[j];
    }
    return -1;
}

/**
 * The rows and columns a tensor presents to a row band, or 0 if it presents
 * none.
 *
 * A rank-2 matrix is one directly. A rank-3 tensor with a unit leading axis in
 * the model's own order is the same bytes as that matrix, which is what lets
 * the Reshape pair around a lowered matrix product join the band: it moves no
 * data, so the band means the same on both sides of it.
 */
int tigris_tensor_row_view(
    const tigris_plan_t *plan, const tigris_tensor_t *t,
    int32_t *batch, int32_t *rows, int32_t *cols)
{
    /* Read the rank and the pool before the shape: this runs on every stage
     * of every plan, including one that carries no shape pool at all, where
     * offsetting it is undefined rather than merely wrong. */
    if (plan->shape_pool == NULL)
        return 0;
    if (t->ndim < 2u)
        return 0;
    /* The last two axes are the matrix and everything before them is batch,
     * which is what a matrix product already means by its shape. Rank 2 is
     * one matrix, rank 3 a batch of them, rank 4 a batch of batches, which is
     * what an attention block's head axis is. Above rank 2 the tensor has to
     * be in the model's own axis order: a spatial tensor holds its channels
     * last, so its final two axes are not a matrix. */
    const int32_t *shape = tigris_tensor_shape(plan, t);
    if (t->ndim == 2u) {
        *batch = 1;
        *rows = shape[0];
        *cols = shape[1];
        return 1;
    }
    if ((t->flags & TIGRIS_TENSOR_LINEAR) == 0u)
        return 0;
    int32_t outer = 1;
    for (uint8_t axis = 0; axis + 2u < t->ndim; axis++) {
        if (shape[axis] <= 0)
            return 0;
        outer *= shape[axis];
    }
    *batch = outer;
    *rows = shape[t->ndim - 2u];
    *cols = shape[t->ndim - 1u];
    return 1;
}

static int tigris_band_whole_input(const tigris_op_t *op, uint8_t position)
{
    return (position == 1u && (op->op_type == TIGRIS_OP_GATHER ||
           op->op_type == TIGRIS_OP_GATHER_ND || op->op_type == TIGRIS_OP_EMBEDDING_LOOKUP)) ||
           (position == 2u && op->op_type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE);
}

/* Whether a binary operator's constant, held in a weight the caller has
 * checked, holds one value per element of `output`. Its element is the data
 * dtype's, or one byte for a selection's condition, as the kernel reads it;
 * a comparison's bool output does not size it. */
static int constant_is_dense(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_tensor_t *output)
{
    uint8_t len = 0u;
    const uint8_t *attr = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
    const int select = op->op_type == TIGRIS_OP_SELECT_V2;
    const uint8_t position = attr != NULL ? attr[0] : 1u;
    const uint8_t dtype = select ? output->dtype :
        plan->tensors[tigris_op_inputs(plan, op)[0]].dtype;
    const uint64_t element = select && position == 0u ? 1u : tigris_dtype_size(dtype);
    return (uint64_t)plan->weight_entries[op->weight_idx].size_bytes ==
        (uint64_t)tigris_tensor_elements(output) * element;
}

/* Whether a rank-2 binary operator's constant operand reads alike in every
 * row. A row band hands the constant to the kernel whole and unshifted, so one
 * that varies along the rows meets the wrong rows. At rank 2 the leading band
 * defers to the row band, so such an operator has to leave the row band for
 * the leading band, which offsets the constant. Mirrors
 * _constant_repeats_every_row in the compiler. */
static int constant_repeats_every_row(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_tensor_t *output)
{
    if (!tigris_binary_op(op->op_type) || op->weight_idx == TIGRIS_NO_WEIGHT || output->ndim != 2u)
        return 1;
    if (plan->weight_entries == NULL || op->weight_idx >= plan->header->num_weights ||
        op->num_inputs == 0u)
        return 0;
    uint8_t len = 0u;
    const uint8_t *attr = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
    if (attr != NULL && len != TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) {
        int32_t rows = 0;
        if (len != TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN + 8u)
            return 0;
        memcpy(&rows, attr + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN, sizeof(rows));
        return rows == 1;
    }
    /* The short form is one value, one per column, or one per element, told
     * apart by length; only the last varies along the rows. */
    return !constant_is_dense(plan, op, op_index, output);
}

/* The row band rule. The loader's form also admits a stage without
 * operators, a FullyConnected whose output is not rank 2 and a constant that
 * varies along the rows; the executor declines all three. */
int tigris_stage_row_band(const tigris_plan_t *plan, const tigris_stage_t *stage, int loader,
                          int32_t *out_batch, int32_t *out_rows)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);

    if ((stage->ops_count == 0u && !loader) || stage->inputs_count == 0u ||
        stage->outputs_count == 0u || stage->chain_len != 0u)
        return 0;

    /* Every tensor the band cuts has to agree on the batch and the row count.
     * The outputs settle it, because an output is always banded. */
    int32_t batch = -1;
    int32_t rows = -1;
    int32_t cols = 0;
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        int32_t b;
        int32_t r;
        if (!tigris_tensor_row_view(plan, &plan->tensors[sout[i]], &b, &r, &cols))
            return 0;
        if (rows < 0) {
            batch = b;
            rows = r;
        }
        if (b != batch || r != rows)
            return 0;
    }
    if (rows <= 1)
        return 0;

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        if ((!tigris_row_tiling_op(op->op_type) && !tigris_op_independent_band(plan, op, sops[j], 1)) || op->num_inputs == 0u ||
            op->num_outputs != 1u)
            return 0;
        /* Every intermediate carries the same band, or a band means something
         * different partway through the stage. */
        const uint16_t *outs = tigris_op_outputs(plan, op);
        const tigris_tensor_t *out = &plan->tensors[outs[0]];
        int32_t b;
        int32_t r;
        if (!tigris_tensor_row_view(plan, out, &b, &r, &cols) ||
            b != batch || r != rows || (!loader && !constant_repeats_every_row(plan, op, sops[j], out)))
            return 0;
        /* A row view accepts a rank-3 [1, rows, cols] as the same memory, but
         * kern_fully_connected reads its channel count out of y_shape[1],
         * which on a rank-3 output is the row count. The matrix product is
         * the one operator here whose rank the row view does not settle. */
        if (!loader && op->op_type == TIGRIS_OP_FULLY_CONN && out->ndim != 2u)
            return 0;
        const uint16_t *ins = tigris_op_inputs(plan, op);
        for (uint8_t k = 0; k < op->num_inputs; k++) {
            const tigris_tensor_t *in = &plan->tensors[ins[k]];
            int32_t ib;
            int32_t ir;
            int32_t ic;
            /* An operand the band does not cut is read whole, which only a
             * matrix product's second operand is entitled to be. The role
             * settles it before the shape does: an operand read whole is
             * never also cut, however its own row count reads. */
            if (k > 0u && tigris_whole_operand_op(op->op_type))
                continue;
            if (tigris_tensor_row_view(plan, in, &ib, &ir, &ic) &&
                ib == batch && ir == rows)
                continue;
            return 0;
        }
    }
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        int32_t b;
        int32_t r;
        if (!tigris_tensor_row_view(plan, &plan->tensors[sin[i]], &b, &r, &cols))
            return 0;
    }
    if (out_batch != NULL)
        *out_batch = batch;
    *out_rows = rows;
    return 1;
}

/**
 * Whether a stage is a rank-2 matrix pipeline that tiles along its rows.
 *
 * Rank 2 has no batch or spatial axis, so the stripe contract never reaches
 * it: a transformer's projections lower to exactly this shape and would
 * otherwise have to hold both operands whole. Every tensor the stage touches
 * must agree on the row count, which is what makes a band of rows mean the
 * same thing on each of them.
 */
int tigris_stage_row_tiled(
    const tigris_plan_t *plan, const tigris_stage_t *stage, int32_t *out_batch, int32_t *out_rows)
{
    return tigris_stage_row_band(plan, stage, 0, out_batch, out_rows);
}

/**
 * Whether a row band reads this stage tensor whole rather than cutting it.
 *
 * A matrix product reads every row of its second operand to produce one row
 * of its output. Which tensors the band cuts is a property of the role each
 * plays, not of its shape: an operand whose own row count happens to equal
 * the band's is still read whole, and cutting it computes a partial product.
 * Mirrors _whole_band_operands in the compiler.
 */
int tigris_whole_band_operand(
    const tigris_plan_t *plan, const tigris_stage_t *stage, uint16_t tidx)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        const uint16_t *ins = tigris_op_inputs(plan, op);
        for (uint8_t k = 1u; k < op->num_inputs; k++) {
            if (ins[k] == tidx && (tigris_whole_operand_op(op->op_type) || tigris_band_whole_input(op, k)))
                return 1;
        }
    }
    return 0;
}

/**
 * Whether a stage is a standalone global reduction that the output-driven
 * tile loop cannot serve.
 *
 * exec_stage_tiled sizes its tiles from the stage output height. A global
 * reduction collapses that height to 1, so it yields one tile holding the
 * whole input and tiles nothing. Such a stage is walked along its input
 * instead, a band of rows at a time, with the kernel carrying a running
 * partial between bands. Deliberately narrow: one op consuming the stage
 * input and producing the stage output, so nothing else has to stay resident
 * across bands.
 */
int tigris_stage_input_driven_reduction(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (stage->ops_count != 1 || stage->inputs_count != 1 ||
        stage->outputs_count != 1)
        return 0;

    uint16_t op_idx = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[op_idx];
    if (op->op_type != TIGRIS_OP_GLOBAL_AVG)
        return 0;
    if (op->num_inputs != 1 || op->num_outputs != 1)
        return 0;

    uint16_t in_idx = tigris_stage_inputs(plan, stage)[0];
    uint16_t out_idx = tigris_stage_outputs(plan, stage)[0];
    if (tigris_op_inputs(plan, op)[0] != in_idx ||
        tigris_op_outputs(plan, op)[0] != out_idx)
        return 0;

    const tigris_tensor_t *in = &plan->tensors[in_idx];
    const tigris_tensor_t *out = &plan->tensors[out_idx];
    if (in->ndim != 4 || out->ndim != 4)
        return 0;

    const int32_t *in_shape = tigris_tensor_shape(plan, in);
    const int32_t *out_shape = tigris_tensor_shape(plan, out);
    return in_shape[0] > 0 && (int64_t)in_shape[1] * in_shape[2] > 0 &&
           in_shape[2] > 0 && in_shape[3] > 0 &&
           out_shape[0] == in_shape[0] && out_shape[1] == 1 &&
           out_shape[2] == 1 && out_shape[3] == in_shape[3];
}

/**
 * The batch and the two extents a banded transpose swaps, or 0 if this
 * operator is not one.
 *
 * transpose_band_groups states the rule: the permutation swaps two adjacent
 * groups of axes and leaves the rest in order. Everything ahead of the pair
 * is a batch the transpose repeats over and everything behind it is a block
 * that travels with the element, so a layout conversion, an attention block's
 * key transpose and its head permutation all reduce to the same matrix
 * transpose.
 */
int tigris_transpose_extents(
    const tigris_plan_t *plan, uint16_t op_idx,
    uint16_t in_idx, int32_t *outer, int32_t *rows, int32_t *cols)
{
    const tigris_tensor_t *in = &plan->tensors[in_idx];
    uint8_t perm_rank = 0;
    const uint8_t *perm = tigris_op_attribute_data(
        plan, op_idx, TIGRIS_OP_ATTR_TRANSPOSE_PERM, &perm_rank);
    if (!perm || perm_rank != in->ndim || in->ndim < 2u)
        return 0;

    uint8_t prefix = 0;
    uint8_t split = 0;
    uint8_t middle = 0;
    if (!transpose_band_groups(perm, in->ndim, &prefix, &split, &middle))
        return 0;

    const int32_t *shape = tigris_tensor_shape(plan, in);
    int32_t batch = 1;
    int32_t a = 1;
    int32_t b = 1;
    for (uint8_t i = 0; i < prefix; i++)
        batch *= shape[i];
    for (uint8_t i = 0; i < split; i++)
        a *= shape[prefix + i];
    for (uint8_t i = split; i < middle; i++)
        b *= shape[prefix + i];
    if (batch <= 0 || a <= 0 || b <= 0)
        return 0;
    *outer = batch;
    *rows = a;
    *cols = b;
    return 1;
}

/**
 * Whether a stage is a standalone layout conversion the stripe contract
 * cannot serve.
 *
 * A conversion moves the channel axis past the spatial ones, so a tile of its
 * output needs a column range of its input, not a row range. Accepted only as
 * the whole stage, since nothing else could carry a height across it.
 */
int tigris_stage_tiled_transpose(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (stage->ops_count != 1 || stage->inputs_count != 1 ||
        stage->outputs_count != 1)
        return 0;

    uint16_t op_idx = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[op_idx];
    if (op->op_type != TIGRIS_OP_TRANSPOSE ||
        op->num_inputs != 1 || op->num_outputs != 1)
        return 0;

    uint16_t in_idx = tigris_stage_inputs(plan, stage)[0];
    uint16_t out_idx = tigris_stage_outputs(plan, stage)[0];
    if (tigris_op_inputs(plan, op)[0] != in_idx ||
        tigris_op_outputs(plan, op)[0] != out_idx)
        return 0;

    const tigris_tensor_t *in = &plan->tensors[in_idx];
    const tigris_tensor_t *out = &plan->tensors[out_idx];
    if (in->ndim != out->ndim)
        return 0;

    int32_t batch = 0;
    int32_t rows = 0;
    int32_t cols = 0;
    if (!tigris_transpose_extents(plan, op_idx, in_idx, &batch, &rows, &cols))
        return 0;

    return batch > 0 && rows > 0 && cols > 0 &&
           in->size_bytes == out->size_bytes;
}

uint8_t tigris_band_axis(const tigris_plan_t *plan, const tigris_tensor_t *tensor, int mode)
{
    if (mode == TIGRIS_BAND_BATCH) return 0u;
    if (mode == TIGRIS_BAND_TIME_MAJOR_BATCH) return tensor->ndim == 3u ? 1u : 0u;
    if (mode == TIGRIS_BAND_TRAILING) return (uint8_t)(tensor->ndim - 1u);
    if (mode >= (int)TIGRIS_BAND_AXIS_BASE) {
        int axis = mode - (int)TIGRIS_BAND_AXIS_BASE;
        return (uint8_t)axis;
    }
    if (mode != TIGRIS_BAND_LEADING) return mode ? (uint8_t)(tensor->ndim - 2u) : 1u;
    const int32_t *shape = tigris_tensor_shape(plan, tensor);
    uint8_t axis = 0u;
    while (axis + 1u < tensor->ndim && shape[axis] == 1) axis++;
    return axis;
}

/* Reads attribute payloads whose lengths the loader has checked. */
int tigris_op_independent_band(const tigris_plan_t *plan, const tigris_op_t *op,
                                uint16_t op_index, int row_tiled)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u ||
        (op->num_outputs != 1u && op->op_type != TIGRIS_OP_SPLIT)) return 0;
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *out = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    if (tigris_binary_op(op->op_type)) {
        if (row_tiled < (int)TIGRIS_BAND_AXIS_BASE || out->ndim < 2u || out->ndim > 5u)
            return 0;
        uint8_t axis = tigris_band_axis(plan, out, row_tiled);
        const int32_t *shape = tigris_tensor_shape(plan, out);
        if (axis >= out->ndim || shape[axis] <= 1) return 0;
        int broadcast = 0;
        for (uint8_t i = 0u; i < op->num_inputs; i++) {
            const tigris_tensor_t *t = &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (t->ndim != out->ndim || t->dtype == 6u) return 0;
            const int32_t *other = tigris_tensor_shape(plan, t);
            int dense = 1, channel = i == 1u;
            for (uint8_t a = 0u; a < out->ndim; a++) {
                if (other[a] != 1 && other[a] != shape[a]) return 0;
                if (other[a] != shape[a]) dense = 0;
                if (a + 1u < out->ndim && other[a] != 1) channel = 0;
            }
            if (!dense && !channel) broadcast = 1;
        }
        if (op->weight_idx != TIGRIS_NO_WEIGHT) {
            if (!plan->weight_entries || op->weight_idx >= plan->header->num_weights) return 0;
            if (constant_is_dense(plan, op, op_index, out)) broadcast = 1;
            uint8_t len = 0u;
            const uint8_t *attr = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
            if (attr && len == TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN + 4u * out->ndim) {
                for (uint8_t a = 0u; a + 1u < out->ndim; a++) {
                    int32_t extent;
                    memcpy(&extent, attr + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN + 4u * a, 4u);
                    if (extent > 1) broadcast = 1;
                }
            }
        }
        return broadcast;
    }
    uint8_t table_len = 0u;
    if (tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_TABLE_SHAPE, &table_len)) {
        if (row_tiled != TIGRIS_BAND_LEADING || op->num_inputs != 1u || op->bias_idx != TIGRIS_NO_WEIGHT)
            return 0;
        uint8_t length = 0u;
        const uint8_t *payload = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_MOVEMENT, &length);
        int32_t metadata[19] = {0};
        if (!payload) return 0;
        memcpy(metadata, payload, length);
        uint8_t ia = tigris_band_axis(plan, in, row_tiled), oa = tigris_band_axis(plan, out, row_tiled);
        if (ia >= in->ndim || oa >= out->ndim ||
            (op->op_type == TIGRIS_OP_GATHER_ND && ia + 1u >= in->ndim) ||
            (op->op_type != TIGRIS_OP_GATHER_ND && (metadata[0] != 0 || metadata[1] != 0))) return 0;
        const int32_t *is = tigris_tensor_shape(plan, in), *os = tigris_tensor_shape(plan, out);
        return is[ia] > 1 && is[ia] == os[oa];
    }
    if (in->ndim < 2u || in->ndim > 6u || out->ndim < 2u || out->ndim > 6u) return 0;
    int32_t ib, ir, ic, ob, orows, oc;
    uint8_t row = tigris_band_axis(plan, in, row_tiled);
    if (row >= in->ndim) return 0;
    if (op->op_type == TIGRIS_OP_TRANSPOSE && row_tiled < (int)TIGRIS_BAND_AXIS_BASE) return 0;
    if (row_tiled < (int)TIGRIS_BAND_LEADING && (op->op_type == TIGRIS_OP_SPLIT ||
        op->op_type == TIGRIS_OP_CONCAT || op->op_type == TIGRIS_OP_PAD)) return 0;
    if (row_tiled < (int)TIGRIS_BAND_LEADING && op->num_inputs > 1u &&
        tigris_band_whole_input(op, 1u)) return 0;
    if (row_tiled == TIGRIS_BAND_LEADING || row_tiled == TIGRIS_BAND_TRAILING ||
        row_tiled >= (int)TIGRIS_BAND_AXIS_BASE) {
        int trailing = row_tiled == TIGRIS_BAND_TRAILING;
        int explicit_axis = row_tiled >= (int)TIGRIS_BAND_AXIS_BASE;
        uint8_t axes_len = 0u;
        const uint8_t *axes = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &axes_len);
        int drop_after = (op->op_type == TIGRIS_OP_ARG_MAX || op->op_type == TIGRIS_OP_ARG_MIN) &&
            axes != NULL && axes_len == 1u && axes[0] > row;
        const int32_t *first = tigris_tensor_shape(plan, in);
        if (first[row] <= 1 || (!trailing && out->ndim != in->ndim &&
            !(drop_after && out->ndim + 1u == in->ndim))) return 0;
        for (uint8_t i = 0u; i < op->num_inputs + op->num_outputs; i++) {
            if (i < op->num_inputs && tigris_band_whole_input(op, i)) continue;
            const tigris_tensor_t *t = i >= op->num_inputs ?
                &plan->tensors[tigris_op_outputs(plan, op)[i - op->num_inputs]] :
                &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (t->ndim < 2u || (!trailing && t->ndim != in->ndim &&
                !(drop_after && i >= op->num_inputs && t->ndim + 1u == in->ndim))) return 0;
            const int32_t *shape = tigris_tensor_shape(plan, t);
            if (!trailing && !explicit_axis) for (uint8_t a = 0u; a < row; a++) if (shape[a] != 1) return 0;
            if (shape[tigris_band_axis(plan, t, row_tiled)] != first[row]) return 0;
        }
    } else if (row_tiled) {
        if (!tigris_tensor_row_view(plan, in, &ib, &ir, &ic) ||
            !tigris_tensor_row_view(plan, out, &ob, &orows, &oc) || ib != ob || ir != orows) return 0;
        for (uint8_t i = 1u; i < op->num_inputs; i++) {
            const tigris_tensor_t *t = &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (!tigris_tensor_row_view(plan, t, &ob, &orows, &oc) || ib != ob || ir != orows) return 0;
        }
    } else {
        if ((in->ndim != 3u && in->ndim != 4u) || out->ndim != in->ndim) return 0;
        const int32_t *first = tigris_tensor_shape(plan, in);
        const int32_t *last = tigris_tensor_shape(plan, out);
        if (first[0] != last[0] || first[1] != last[1] || first[1] <= 1) return 0;
        for (uint8_t i = 1u; i < op->num_inputs; i++) {
            const tigris_tensor_t *operand = &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (operand->ndim != in->ndim) return 0;
            const int32_t *other = tigris_tensor_shape(plan, operand);
            if (other[0] != first[0] || other[1] != first[1]) return 0;
        }
    }
    uint8_t length = 0u;
    uint8_t type = op->op_type;
    if (type == TIGRIS_OP_TRANSPOSE) {
        uint8_t prefix, split, middle;
        const uint8_t *perm = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_TRANSPOSE_PERM, &length);
        return perm && length == in->ndim && perm[row] == row &&
            !transpose_band_groups(perm, length, &prefix, &split, &middle);
    }
    if (type == TIGRIS_OP_SPLIT || type == TIGRIS_OP_CONCAT) {
        uint8_t axis = type == TIGRIS_OP_CONCAT && in->ndim == 4u && op->spatial.kernel_h == 0u
                     ? 3u : (uint8_t)op->spatial.kernel_h;
        return axis != row && (type == TIGRIS_OP_SPLIT ||
            ((axis + 1u != in->ndim || (in->flags & TIGRIS_TENSOR_LINEAR) != 0u) &&
             op->weight_idx == TIGRIS_NO_WEIGHT));
    }
    if (type == TIGRIS_OP_PAD) {
        const uint8_t *pads = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_PADS, &length);
        int32_t lead = -1, trail = -1;
        if (!pads || length != (uint32_t)in->ndim * 8u) return 0;
        memcpy(&lead, pads + (uint32_t)row * 8u, 4u);
        memcpy(&trail, pads + (uint32_t)row * 8u + 4u, 4u);
        return lead == 0 && trail == 0;
    }
    if (op->num_inputs == 1u && tigris_axis1_unary_pointwise_op(type) &&
        type != TIGRIS_OP_SOFTMAX && type != TIGRIS_OP_LOG_SOFTMAX &&
        type != TIGRIS_OP_L2_NORMALIZATION && type != TIGRIS_OP_LAYER_NORM && in->dtype != 6u)
        return in->ndim == 2u && row_tiled >= (int)TIGRIS_BAND_LEADING && tigris_tensor_shapes_equal(plan, in, out);
    if (type == TIGRIS_OP_REDUCE_MEAN || type == TIGRIS_OP_REDUCE_MAX ||
        type == TIGRIS_OP_REDUCE_MIN || type == TIGRIS_OP_REDUCE_SUM ||
        type == TIGRIS_OP_REDUCE_ALL || type == TIGRIS_OP_CUMSUM ||
        type == TIGRIS_OP_ARG_MAX || type == TIGRIS_OP_ARG_MIN) {
        const uint8_t *axes = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &length);
        return axes != NULL && length == 1u && axes[0] < in->ndim && axes[0] != row;
    }
    int32_t metadata[19] = {0};
    const uint8_t *payload = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_MOVEMENT, &length);
    if (!payload) return 0;
    memcpy(metadata, payload, length);
    const int32_t *shape = tigris_tensor_shape(plan, in);
    if (type == TIGRIS_OP_GATHER || type == TIGRIS_OP_GATHER_ND || type == TIGRIS_OP_EMBEDDING_LOOKUP) {
        if (op->num_inputs != (op->weight_idx == TIGRIS_NO_WEIGHT ? 2u : 1u)) return 0;
        if (type == TIGRIS_OP_GATHER_ND)
            return metadata[0] > 0 && metadata[0] <= 6 && metadata[metadata[0]] <= row;
        return metadata[0] != row && metadata[1] <= row;
    }
    if (type == TIGRIS_OP_STRIDED_SLICE) {
        uint32_t shrink = length > (uint32_t)in->ndim * 12u ? (uint32_t)metadata[3u * in->ndim] : 0u;
        return (shrink & (1u << row)) == 0u && metadata[3u * row] == 0 &&
               metadata[3u * row + 1u] == shape[row] && metadata[3u * row + 2u] == 1;
    }
    if (type == TIGRIS_OP_MIRROR_PAD)
        return metadata[1u + 2u * row] == 0 && metadata[2u + 2u * row] == 0;
    if (type == TIGRIS_OP_REVERSE_V2)
        return ((uint32_t)metadata[0] & (1u << row)) == 0u;
    if (type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE && op->num_inputs == 3u &&
        op->weight_idx == TIGRIS_NO_WEIGHT && length == (uint32_t)in->ndim * 4u)
        return metadata[row] == shape[row];
    return type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE && op->weight_idx == TIGRIS_NO_WEIGHT &&
           op->num_inputs == 2u && length == (uint32_t)in->ndim * 8u &&
           metadata[row] == 0 && metadata[in->ndim + row] == shape[row];
}

/* The band mode an SVDF or LSTM defines for itself: its batch. */
static int recurrent_band_mode(const tigris_plan_t *plan, uint16_t index, const tigris_op_t *op)
{
    if (op->op_type == TIGRIS_OP_SVDF) return TIGRIS_BAND_BATCH;
    const uint8_t *params = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_LSTM, NULL);
    int32_t time_major = 0;
    if (params) memcpy(&time_major, params, sizeof(time_major));
    return time_major ? TIGRIS_BAND_TIME_MAJOR_BATCH : TIGRIS_BAND_BATCH;
}

/* Whether a stage has the form a leading band needs: one unchained operator
 * with stage inputs and outputs, and a shape pool. The band mode helpers read
 * that operator and its shapes, so this holds before either runs. */
static int leading_band_stage(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    return plan->shape_pool != NULL && stage->ops_count == 1u && stage->chain_len == 0u &&
        stage->inputs_count != 0u && stage->outputs_count != 0u;
}

/* Below schema 11, the band mode a leading band runs by: the axis whose band
 * costs least, or the operator's own. The stage has a leading band's form
 * (leading_band_stage). */
static int derived_band_mode(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    /* The loader asks before it checks operand counts, so the operator may
     * have none. */
    if (op->num_inputs != 0u && op->num_outputs != 0u &&
        (op->op_type == TIGRIS_OP_SPLIT || op->op_type == TIGRIS_OP_CONCAT ||
         op->op_type == TIGRIS_OP_PAD || op->op_type == TIGRIS_OP_TRANSPOSE || tigris_binary_op(op->op_type) ||
         (op->num_inputs > 1u && tigris_band_whole_input(op, 1u)) ||
         (op->num_inputs > 2u && tigris_band_whole_input(op, 2u)))) {
        const tigris_tensor_t *input = &plan->tensors[tigris_binary_op(op->op_type) ?
            tigris_op_outputs(plan, op)[0] : tigris_op_inputs(plan, op)[0]];
        const int32_t *shape = tigris_tensor_shape(plan, input);
        int mode = TIGRIS_BAND_LEADING;
        int32_t extent = 1;
        uint64_t best_cost = UINT64_MAX;
        for (uint8_t axis = 0u; axis < input->ndim; axis++) {
            int candidate = (int)TIGRIS_BAND_AXIS_BASE + axis;
            uint64_t cost = 0u;
            if (tigris_binary_op(op->op_type)) {
                for (uint8_t i = 0u; i < op->num_inputs + op->num_outputs; i++) {
                    uint16_t tidx = i < op->num_inputs ? tigris_op_inputs(plan, op)[i] :
                        tigris_op_outputs(plan, op)[i - op->num_inputs];
                    int repeated = 0;
                    for (uint8_t j = 0u; j < i && j < op->num_inputs; j++)
                        if (tigris_op_inputs(plan, op)[j] == tidx) repeated = 1;
                    if (repeated) continue;
                    const tigris_tensor_t *t = &plan->tensors[tidx];
                    int32_t divisor = t->ndim == input->ndim ? tigris_tensor_shape(plan, t)[axis] : 1;
                    if (divisor <= 0) return TIGRIS_BAND_LEADING;
                    uint64_t bytes = t->size_bytes / (uint32_t)divisor;
                    cost += ((bytes + 31u) / 32u) * 32u;
                }
            }
            if ((cost < best_cost || (cost == best_cost && shape[axis] > extent)) &&
                tigris_op_independent_band(plan, op, index, candidate)) {
                extent = shape[axis];
                best_cost = cost;
                mode = candidate;
            }
        }
        if (best_cost != UINT64_MAX) return mode;
    }
    if (op->op_type == TIGRIS_OP_SVDF || op->op_type == TIGRIS_OP_LSTM)
        return recurrent_band_mode(plan, index, op);
    return !tigris_op_independent_band(plan, op, index, TIGRIS_BAND_LEADING) &&
        tigris_op_independent_band(plan, op, index, TIGRIS_BAND_TRAILING) ?
        TIGRIS_BAND_TRAILING : TIGRIS_BAND_LEADING;
}

/* The rows a one-operator stage bands along under a leading band mode, or 0
 * when the operator does not keep that axis independent. */
static int32_t leading_band_rows(const tigris_plan_t *plan, const tigris_stage_t *stage, int mode)
{
    if (!leading_band_stage(plan, stage)) return 0;
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    if (op->op_type == TIGRIS_OP_SVDF || op->op_type == TIGRIS_OP_LSTM) {
        if (op->num_inputs != stage->inputs_count || op->num_outputs != stage->outputs_count ||
            mode != recurrent_band_mode(plan, index, op))
            return 0;
        int32_t rows = 0;
        for (uint8_t i = 0; i < op->num_inputs + op->num_outputs; i++) {
            uint16_t tidx = i < op->num_inputs ? tigris_op_inputs(plan, op)[i] :
                tigris_op_outputs(plan, op)[i - op->num_inputs];
            const tigris_tensor_t *t = &plan->tensors[tidx];
            if (t->ndim < 2u || t->ndim > 3u) return 0;
            int32_t r = tigris_tensor_shape(plan, t)[tigris_band_axis(plan, t, mode)];
            if (r <= 1 || (rows && rows != r)) return 0;
            rows = r;
        }
        return rows;
    }
    if ((mode != (int)TIGRIS_BAND_LEADING && mode != (int)TIGRIS_BAND_TRAILING &&
         mode < (int)TIGRIS_BAND_AXIS_BASE) || !tigris_op_independent_band(plan, op, index, mode))
        return 0;
    for (uint16_t i = 0u; i < stage->inputs_count; i++) {
        int found = 0;
        for (uint8_t j = 0u; j < op->num_inputs; j++)
            if (tigris_stage_inputs(plan, stage)[i] == tigris_op_inputs(plan, op)[j]) found = 1;
        if (!found) return 0;
    }
    for (uint16_t i = 0u; i < stage->outputs_count; i++) {
        int found = 0;
        for (uint8_t j = 0u; j < op->num_outputs; j++)
            if (tigris_stage_outputs(plan, stage)[i] == tigris_op_outputs(plan, op)[j]) found = 1;
        if (!found) return 0;
    }
    for (uint8_t i = 0u; i < op->num_inputs; i++) {
        int found = 0;
        for (uint16_t j = 0u; j < stage->inputs_count; j++)
            if (tigris_op_inputs(plan, op)[i] == tigris_stage_inputs(plan, stage)[j]) found = 1;
        if (!found) return 0;
    }
    const tigris_tensor_t *out = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    return tigris_tensor_shape(plan, out)[tigris_band_axis(plan, out, mode)];
}

int32_t tigris_stage_leading_band(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (!leading_band_stage(plan, stage)) return 0;
    int32_t rows = leading_band_rows(plan, stage, derived_band_mode(plan, stage));
    if (rows == 0) return 0;
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    if (op->op_type == TIGRIS_OP_SVDF || op->op_type == TIGRIS_OP_LSTM) return rows;
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const int32_t *shape = tigris_tensor_shape(plan, in);
    /* Preserve the meaning of existing matrix-row and height records. A rank-2
     * stage the row band declines, such as one with a dense constant, stays leading. */
    int32_t row_band = 0;
    if (in->ndim == 2u && shape[0] > 1 && tigris_row_tiling_op(op->op_type) &&
        tigris_stage_row_tiled(plan, stage, NULL, &row_band)) return 0;
    if ((in->ndim >= 2u && shape[in->ndim - 2u] > 1 && tigris_op_independent_band(plan, op, index, 1)) ||
        tigris_op_independent_band(plan, op, index, 0)) return 0;
    return rows;
}

static int states_its_own_order(const tigris_tensor_t *t)
{
    return (t->flags & TIGRIS_TENSOR_LINEAR) != 0u || t->ndim <= 2u;
}

/**
 * Whether a reshape from in to out gives a tensor a new shape without moving a
 * byte.
 *
 * A reshape produces the same bytes in the same order, and only the shape the
 * consumer reads differs, in two cases. Two tensors that each state their own
 * axis order hold their elements in the order the shape lists them, so
 * regrouping the shape moves nothing. Two spatial tensors hold their channel
 * axis last and their spatial axes in order ahead of it, so regrouping only
 * the spatial axes moves nothing either: that is what an unfold is, and it is
 * how a vision transformer turns a feature map into patches. There the batch
 * and the channel extent have to survive, because they are the ends of the
 * stored order and a reshape that changes either is moving bytes.
 *
 * The output can then share the input's buffer instead of being allocated and
 * copied into. Mirrors pure_reinterpretations in the compiler, which counts
 * the two as one allocation when it decides whether a stage fits.
 */
int tigris_pure_reinterpretation(const tigris_plan_t *plan, const tigris_tensor_t *in,
                                 const tigris_tensor_t *out)
{
    if (in->size_bytes != out->size_bytes)
        return 0;
    /* The bytes are only the same bytes if they mean the same thing. A plan
     * may give the two tensors different quantization, and then the reshape
     * has to re-encode rather than pass through. */
    const tigris_quant_param_t *iq = tigris_tensor_quant(plan, in);
    const tigris_quant_param_t *oq = tigris_tensor_quant(plan, out);
    if (iq != oq && (!iq || !oq || iq->scale != oq->scale || iq->zero_point != oq->zero_point))
        return 0;
    int in_own = states_its_own_order(in);
    if (in_own != states_its_own_order(out))
        return 0;
    if (in_own)
        return 1;
    if (plan->shape_pool == NULL || in->ndim == 0u || out->ndim == 0u)
        return 0;
    const int32_t *a = tigris_tensor_shape(plan, in);
    const int32_t *b = tigris_tensor_shape(plan, out);
    return a[0] == b[0] && a[in->ndim - 1u] == b[out->ndim - 1u];
}

int tigris_reshape_band(const tigris_plan_t *plan, const tigris_stage_t *stage,
                        tigris_reshape_band_t *view)
{
    if (stage->ops_count != 1u || stage->inputs_count != 1u ||
        stage->outputs_count != 1u || stage->chain_len != 0u) return 0;
    const tigris_op_t *op = &plan->ops[tigris_stage_ops(plan, stage)[0]];
    if ((op->op_type != TIGRIS_OP_RESHAPE && op->op_type != TIGRIS_OP_FLATTEN) ||
        op->num_inputs != 1u || op->num_outputs != 1u) return 0;
    uint16_t x = tigris_op_inputs(plan, op)[0], y = tigris_op_outputs(plan, op)[0];
    if (tigris_stage_inputs(plan, stage)[0] != x || tigris_stage_outputs(plan, stage)[0] != y) return 0;
    const tigris_tensor_t *in = &plan->tensors[x], *out = &plan->tensors[y];
    if ((in->dtype != 1u && in->dtype != 3u) || in->dtype != out->dtype ||
        in->ndim < 2u || in->ndim > 6u || out->ndim < 2u || out->ndim > 6u ||
        !tigris_pure_reinterpretation(plan, in, out)) return 0;
    const int32_t *a = tigris_tensor_shape(plan, in), *b = tigris_tensor_shape(plan, out);
    int own_in = states_its_own_order(in);
    for (int mode = own_in ? 0 : 1; mode < 3; mode++) {
        uint8_t ia = mode == 0 ? (uint8_t)(in->ndim - 2u) : 1u;
        uint8_t ib = mode == 0 ? (uint8_t)(out->ndim - 2u) : 1u;
        if (mode == 2) {
            /* Unit prefixes leave a single contiguous interval on each side. */
            ia = 0u;
            ib = 0u;
            while (ia + 1u < in->ndim && a[ia] == 1) ia++;
            while (ib + 1u < out->ndim && b[ib] == 1) ib++;
        }
        int64_t blocks = 1, other = 1, ci = 1, co = 1;
        for (uint8_t i = 0u; i < in->ndim; i++) {
            if (a[i] <= 0) return 0;
            if (i < ia) blocks *= a[i];
            if (i > ia) ci *= a[i];
            if (blocks > INT32_MAX || ci > INT32_MAX) return 0;
        }
        for (uint8_t i = 0u; i < out->ndim; i++) {
            if (b[i] <= 0) return 0;
            if (i < ib) other *= b[i];
            if (i > ib) co *= b[i];
            if (other > INT32_MAX || co > INT32_MAX) return 0;
        }
        if (blocks == other && (int64_t)a[ia] * ci == (int64_t)b[ib] * co && a[ia] > 1) {
            view->blocks = (int32_t)blocks;
            view->input_rows = a[ia];
            view->input_width = (int32_t)ci;
            view->output_rows = b[ib];
            view->output_width = (int32_t)co;
            view->input_axis = ia;
            return 1;
        }
    }
    return 0;
}

int tigris_reshape_tile_valid(const tigris_tile_plan_t *tile,
                              const tigris_reshape_band_t *view)
{
    return tile->tileable == 1u && tile->axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH &&
           tile->tile_height > 0u && tile->tile_height <= view->input_rows &&
           tile->original_height == view->input_rows && tile->halo == 0u &&
           tile->receptive_field == 1u && tile->tile_width == 0u && tile->flags == 0u &&
           tile->num_tiles == ((uint32_t)view->input_rows + tile->tile_height - 1u) / tile->tile_height &&
           ((int64_t)tile->tile_height * view->input_width) % view->output_width == 0 &&
           ((int64_t)view->input_rows * view->input_width) % view->output_width == 0;
}

/* The band kind a schema 11 record states, when the stage meets that kind's
 * conditions: a 2D band exactly under an HW record, a band mode exactly for a
 * leading band, and each kind's own rule. The loader checks the stripe and 2D
 * contracts of the record's axis. */
static tigris_band_kind_t stated_band_kind(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                           uint8_t axis, tigris_stage_band_t *out)
{
    static const tigris_band_kind_t kinds[TIGRIS_BAND_KIND_STRIPE + 1] = {
        TIGRIS_BAND_KIND_NONE, TIGRIS_BAND_KIND_RESHAPE, TIGRIS_BAND_KIND_LEADING,
        TIGRIS_BAND_KIND_ROWS, TIGRIS_BAND_KIND_TRANSPOSE, TIGRIS_BAND_KIND_BY_INPUT,
        TIGRIS_BAND_KIND_HW, TIGRIS_BAND_KIND_STRIPE,
    };
    uint8_t stated = plan->tile_plans_v11[stage->tile_plan_idx].facts.band;
    tigris_band_kind_t kind = TIGRIS_TILE_BAND_KIND(stated) <= (uint8_t)TIGRIS_BAND_KIND_STRIPE ?
        kinds[TIGRIS_TILE_BAND_KIND(stated)] : TIGRIS_BAND_KIND_NONE;
    int mode = (int)TIGRIS_TILE_BAND_MODE(stated);
    if ((kind == TIGRIS_BAND_KIND_HW) != (axis == TIGRIS_TILE_AXIS_HW) ||
        (kind == TIGRIS_BAND_KIND_LEADING) != (mode != 0))
        return TIGRIS_BAND_KIND_NONE;
    tigris_reshape_band_t view;
    int holds = 0;
    switch (kind) {
    case TIGRIS_BAND_KIND_RESHAPE:
        holds = tigris_reshape_band(plan, stage, &view);
        out->rows = holds ? view.input_rows : 0;
        break;
    case TIGRIS_BAND_KIND_LEADING:
        out->rows = leading_band_rows(plan, stage, mode);
        out->mode = mode;
        holds = out->rows > 0;
        break;
    case TIGRIS_BAND_KIND_ROWS:
        holds = tigris_stage_row_tiled(plan, stage, &out->batch, &out->rows);
        break;
    case TIGRIS_BAND_KIND_TRANSPOSE:
        holds = tigris_stage_tiled_transpose(plan, stage);
        break;
    case TIGRIS_BAND_KIND_BY_INPUT:
        holds = tigris_stage_input_driven_reduction(plan, stage);
        break;
    case TIGRIS_BAND_KIND_HW:
    case TIGRIS_BAND_KIND_STRIPE:
        holds = 1;
        break;
    default:
        break;
    }
    if (!holds) {
        out->batch = 0;
        out->rows = 0;
        out->mode = 0;
        return TIGRIS_BAND_KIND_NONE;
    }
    return kind;
}

tigris_band_kind_t tigris_stage_band_kind(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                          tigris_stage_band_t *band)
{
    tigris_stage_band_t unused;
    tigris_stage_band_t *out = band != NULL ? band : &unused;
    out->batch = 0;
    out->rows = 0;
    out->mode = 0;
    if (plan == NULL || plan->header == NULL || stage == NULL)
        return TIGRIS_BAND_KIND_NONE;
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    int banded = tile.tileable != 0u && tile.axis != TIGRIS_TILE_AXIS_NONE;
    /* A schema 11 record comes from tile_plans_v11, which banded implies. */
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS)
        return banded ? stated_band_kind(plan, stage, tile.axis, out) : TIGRIS_BAND_KIND_NONE;
    if (banded) {
        tigris_reshape_band_t view;
        if (tigris_reshape_band(plan, stage, &view)) {
            out->rows = view.input_rows;
            return TIGRIS_BAND_KIND_RESHAPE;
        }
        int32_t leading = tigris_stage_leading_band(plan, stage);
        if (leading > 0) {
            out->rows = leading;
            out->mode = derived_band_mode(plan, stage);
            return TIGRIS_BAND_KIND_LEADING;
        }
    }
    if (tigris_stage_row_tiled(plan, stage, &out->batch, &out->rows))
        return TIGRIS_BAND_KIND_ROWS;
    if (tigris_stage_tiled_transpose(plan, stage))
        return TIGRIS_BAND_KIND_TRANSPOSE;
    if (tigris_stage_input_driven_reduction(plan, stage))
        return TIGRIS_BAND_KIND_BY_INPUT;
    if (banded && tile.axis == TIGRIS_TILE_AXIS_HW)
        return TIGRIS_BAND_KIND_HW;
    return TIGRIS_BAND_KIND_STRIPE;
}

/**
 * Whether every operator of a stage implements the stripe contract along
 * axis: a lone operator that keeps its rows independent, rank-3 unary
 * pointwise ops around at most one Conv1D (binary pointwise ops only without
 * one), or rank-4 height ops with at most one spatial or resampling op and no
 * Resize in a chain. Along TIGRIS_TILE_AXIS_HW a ConvTranspose also counts as
 * the spatial op, since only the 2D executor inverts its rectangle.
 *
 * Only the listed operators qualify: an operator that runs on a full tensor
 * does not necessarily consume the tile geometry correctly. Softmax and the
 * other normalizations reduce along the channel axis, which a band never cuts.
 */
int tigris_stage_stripe_contract(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                 uint8_t rank, uint8_t axis)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    uint16_t spatial_count = 0;

    if (stage->ops_count == 1u && stage->chain_len == 0u &&
        tigris_op_independent_band(plan, &plan->ops[sops[0]], sops[0], 0))
        return 1;
    if (rank == 3u) {
        uint16_t binary_count = 0;
        if (stage->chain_len != 0u)
            return 0;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint8_t type = plan->ops[sops[j]].op_type;
            if (type == TIGRIS_OP_CONV1D) {
                spatial_count++;
            } else if (tigris_axis1_binary_pointwise_op(type)) {
                binary_count++;
            } else if (!tigris_axis1_unary_pointwise_op(type)) {
                return 0;
            }
        }
        return spatial_count <= 1u && !(spatial_count == 1u && binary_count > 0u);
    }
    if (rank != 4u)
        return 0;
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t type = plan->ops[sops[j]].op_type;
        tigris_height_op_t kind = tigris_height_op(type);
        if (kind == TIGRIS_HEIGHT_OP_SPATIAL ||
            (axis == TIGRIS_TILE_AXIS_HW && type == TIGRIS_OP_CONV_TRANSPOSE)) {
            spatial_count++;
        } else if (kind == TIGRIS_HEIGHT_OP_RESAMPLE) {
            if (stage->chain_len != 0u)
                return 0;
            spatial_count++;
        } else if (kind == TIGRIS_HEIGHT_OP_REFUSED) {
            return 0;
        }
    }
    return spatial_count <= 1u;
}

/* Ceiling of a / b, or 0 when it does not fit 32 bits. */
static uint32_t bands_of(uint64_t extent, uint64_t band)
{
    uint64_t count = band == 0u ? 0u : (extent + band - 1u) / band;
    return count > UINT32_MAX ? 0u : (uint32_t)count;
}

/* Rows a band reads past its own, from the stage's spatial operator. */
static uint32_t stage_halo(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    int index = tigris_stage_spatial_op(plan, stage);
    if (index < 0)
        return 0u;
    const tigris_op_t *op = &plan->ops[index];
    if (op->op_type == TIGRIS_OP_RESIZE_LINEAR)
        return 1u;
    if (tigris_height_upsample_op(op->op_type) || op->op_type == TIGRIS_OP_CONV_TRANSPOSE)
        return 0u;
    uint32_t dilation = op->spatial.dilation_h != 0u ? op->spatial.dilation_h : 1u;
    return op->spatial.kernel_h == 0u ? 0u : (uint32_t)(op->spatial.kernel_h - 1u) * dilation;
}

int tigris_stage_tile_facts(const tigris_plan_t *plan, const tigris_stage_t *stage,
                            tigris_tile_facts_t *out)
{
    if (!plan || !plan->header || !stage || !out)
        return 0;
    memset(out, 0, sizeof(*out));
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    if (!tile.tileable)
        return 1;
    if (stage->inputs_count == 0u || stage->outputs_count == 0u || !plan->shape_pool)
        return 0;
    const uint16_t *inputs = tigris_stage_inputs(plan, stage);
    const tigris_tensor_t *in = &plan->tensors[inputs[0]];
    const tigris_tensor_t *output = &plan->tensors[tigris_stage_outputs(plan, stage)[0]];
    const int32_t *in_shape = tigris_tensor_shape(plan, in);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint64_t extent = 0u;
    uint64_t count = 0u;
    tigris_reshape_band_t view;
    tigris_stage_band_t stage_band;
    uint8_t prefix = 0u;
    uint8_t split = 0u;
    uint8_t middle = 0u;
    uint32_t halo = 0u;
    /* The kind the executor bands the stage by. */
    tigris_band_kind_t kind = tigris_stage_band_kind(plan, stage, &stage_band);
    if (kind == TIGRIS_BAND_KIND_RESHAPE && tigris_reshape_band(plan, stage, &view)) {
        out->band_axis = view.input_axis;
        extent = (uint32_t)view.input_rows;
        out->tile_extent = tile.input_band_extent;
    } else if (kind == TIGRIS_BAND_KIND_LEADING) {
        const tigris_op_t *op = &plan->ops[tigris_stage_ops(plan, stage)[0]];
        const tigris_tensor_t *band = &plan->tensors[tigris_op_inputs(plan, op)[0]];
        out->band_axis = tigris_band_axis(plan, band, stage_band.mode);
        extent = (uint32_t)stage_band.rows;
        out->tile_extent = tile.input_band_extent;
    } else if (kind == TIGRIS_BAND_KIND_ROWS) {
        /* The band input is the first stage input the band cuts. */
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            const tigris_tensor_t *t = &plan->tensors[inputs[i]];
            int32_t b = 0;
            int32_t r = 0;
            int32_t c = 0;
            if (!tigris_whole_band_operand(plan, stage, inputs[i]) &&
                tigris_tensor_row_view(plan, t, &b, &r, &c) && b == stage_band.batch && r == stage_band.rows) {
                out->band_axis = (uint8_t)(t->ndim - 2u);
                break;
            }
        }
        extent = (uint32_t)stage_band.rows;
        out->tile_extent = tile.input_band_extent;
    } else if (kind == TIGRIS_BAND_KIND_TRANSPOSE) {
        uint16_t index = tigris_stage_ops(plan, stage)[0];
        uint8_t rank = 0u;
        const uint8_t *perm = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_TRANSPOSE_PERM, &rank);
        if (perm == NULL || !transpose_band_groups(perm, rank, &prefix, &split, &middle))
            return 0;
        /* The run sizes a conversion's band itself, so the record's is only bounded. */
        int on_columns = (tile.flags & TIGRIS_TILE_FLAG_BAND_ON_COLUMNS) != 0u;
        uint8_t first = on_columns ? (uint8_t)(prefix + split) : prefix;
        uint8_t last = on_columns ? (uint8_t)(prefix + middle) : (uint8_t)(prefix + split);
        extent = 1u;
        for (uint8_t a = first; a < last; a++)
            extent *= (uint32_t)in_shape[a];
        out->band_axis = first;
        if (plan->tile_plans_v11 != NULL) {
            uint32_t stated = plan->tile_plans_v11[stage->tile_plan_idx].facts.tile_extent;
            if (stated != 0u && stated <= extent)
                out->tile_extent = stated;
        }
    } else if (kind == TIGRIS_BAND_KIND_BY_INPUT) {
        out->band_axis = 1u;
        if (tile.input_channel_group != 0u) {
            extent = (uint64_t)(uint32_t)in_shape[1] * (uint32_t)in_shape[2];
            out->tile_extent = tile.input_spatial_interval;
            count = (uint64_t)(uint32_t)in_shape[0] *
                    bands_of((uint32_t)in_shape[3], tile.input_channel_group) *
                    bands_of(extent, tile.input_spatial_interval);
        } else {
            extent = (uint32_t)in_shape[1];
            out->tile_extent = tile.input_band_extent;
        }
    } else if (kind == TIGRIS_BAND_KIND_HW) {
        if (in->ndim != 4u || output->ndim != 4u || tile.output_width_extent == 0u)
            return 0;
        out->band_axis = 1u;
        extent = (uint32_t)out_shape[1];
        out->tile_extent = tile.output_band_extent;
        count = (uint64_t)bands_of(extent, tile.output_band_extent) *
                bands_of((uint32_t)out_shape[2], tile.output_width_extent);
        halo = stage_halo(plan, stage);
    } else {
        /* A height stripe, Resize included, bands its output rows. */
        out->band_axis = 1u;
        extent = (uint32_t)out_shape[1];
        out->tile_extent = tile.output_band_extent;
        halo = stage_halo(plan, stage);
    }
    if (count == 0u)
        count = bands_of(extent, out->tile_extent);
    if (extent == 0u || extent > UINT32_MAX || count == 0u || count > UINT32_MAX ||
        out->tile_extent == 0u || halo > UINT16_MAX)
        return 0;
    out->original_extent = (uint32_t)extent;
    out->num_tiles = (uint32_t)count;
    out->halo = (uint16_t)halo;
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS)
        out->band = (uint8_t)((uint32_t)kind | (uint32_t)stage_band.mode << 4);
    return 1;
}

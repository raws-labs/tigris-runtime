/**
 * @file tigris_loader.c
 * @brief Zero-alloc binary plan parser.
 *
 * Validates the file header, walks the section directory, and sets
 * pointers into the buffer. Never allocates memory.
 */

#include "tigris_loader.h"
#include "tigris_binary_operands.h"
#include "tigris_op_config.h"
#include "tigris_op_contract.h"

#include "tigris_transpose_band.h"

#include <math.h>
#include <string.h>

/* Helpers */

/* Keep these in sync with the executor's fixed-size working arrays. */
/** Check that [offset, offset+size) fits within buf_len. */
static inline int bounds_ok(uint32_t offset, uint32_t size, uint32_t buf_len)
{
    return (offset <= buf_len) && (size <= buf_len - offset);
}

/** Check an element range without overflowing offset + count. */
static inline int elements_ok(uint32_t offset, uint32_t count, uint32_t total)
{
    return offset <= total && count <= total - offset;
}

/** Check that an offset names a terminated string inside the string section. */
static int string_ok(const char *strings, uint32_t strings_len, uint32_t offset)
{
    if (!strings || offset >= strings_len)
        return 0;
    return memchr(strings + offset, '\0', strings_len - offset) != NULL;
}

static int tensor_shapes_equal(
    const tigris_plan_t *plan,
    const tigris_tensor_t *a,
    const tigris_tensor_t *b)
{
    if (a->ndim != b->ndim)
        return 0;
    const int32_t *a_shape = tigris_tensor_shape(plan, a);
    const int32_t *b_shape = tigris_tensor_shape(plan, b);
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a_shape[i] != b_shape[i])
            return 0;
    }
    return 1;
}

static uint32_t tensor_elements(const tigris_tensor_t *tensor)
{
    return tensor->size_bytes / ((tensor->dtype == 1 || tensor->dtype == 6) ? sizeof(int32_t) : 1u);
}

static int weight_size_is(
    const tigris_plan_t *plan, uint16_t index, uint64_t expected)
{
    return index != TIGRIS_NO_WEIGHT && plan->weight_entries &&
           expected <= UINT32_MAX &&
           plan->weight_entries[index].size_bytes == (uint32_t)expected;
}

static int optional_bias_size_is(
    const tigris_plan_t *plan, uint16_t index, uint64_t expected)
{
    return index == TIGRIS_NO_WEIGHT || weight_size_is(plan, index, expected);
}

static int output_dim_is_valid(
    int32_t input, int32_t output, uint8_t kernel, uint8_t stride,
    uint16_t pad_before, uint16_t pad_after, uint16_t dilation)
{
    if (input <= 0 || output <= 0 || kernel == 0 || stride == 0)
        return 0;
    uint64_t effective_dilation = dilation ? dilation : 1u;
    uint64_t effective_kernel =
        ((uint64_t)kernel - 1u) * effective_dilation + 1u;
    uint64_t padded = (uint64_t)input + pad_before + pad_after;
    if (padded < effective_kernel)
        return 0;
    return (padded - effective_kernel) / stride + 1u == (uint64_t)output;
}

/* Whether an int8 activation an operator reads has its encoding stated. Its
 * record may also carry the per-channel requantization of the operator that
 * wrote it; a reader uses only the scale and zero point. */
static int quant_input_stated(const tigris_plan_t *plan, const tigris_tensor_t *tensor)
{
    return tensor->dtype != 3 ||
           (tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM &&
            tensor->quant_param_idx < plan->num_quant_params);
}

static int quant_channels_fit(
    const tigris_plan_t *plan, const tigris_tensor_t *tensor,
    uint32_t channels)
{
    if (tensor->dtype != 3 || tensor->quant_param_idx == TIGRIS_NO_QUANT_PARAM)
        return 1;
    const tigris_quant_param_t *qp =
        &plan->quant_params[tensor->quant_param_idx];
    return qp->num_channels == 1 || qp->num_channels == channels;
}

static int op_has_plain_io(
    const tigris_op_t *op, uint8_t inputs, uint8_t outputs)
{
    return op->num_inputs == inputs && op->num_outputs == outputs &&
           op->weight_idx == TIGRIS_NO_WEIGHT &&
           op->bias_idx == TIGRIS_NO_WEIGHT;
}

static int is_binary_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_BINARY) != 0u;
}

/* Whether a binary operand of this shape broadcasts to the output: its rank
 * at most the output's, and each axis, aligned from the innermost, either 1
 * or the output's extent. */
static int shape_broadcasts_to(const int32_t *shape, uint8_t ndim,
                               const tigris_plan_t *plan, const tigris_tensor_t *output)
{
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    if (ndim > output->ndim)
        return 0;
    for (uint8_t d = 0; d < ndim; d++) {
        int32_t extent = shape[d];
        int32_t target = out_shape[d + (uint8_t)(output->ndim - ndim)];
        if (extent <= 0 || (extent != 1 && extent != target))
            return 0;
    }
    return 1;
}

#define TIGRIS_MAX_TENSOR_RANK_FOR_ATTR 8u
/** Highest input rank an ArgMax or ArgMin reads its axis from. */
#define TIGRIS_ARG_MAX_RANK 6u

/* A binary operator's constant operand, held in its weight: one value, one
 * per channel, or one per element of `output`, or the shape the long form of
 * the attribute states. An int8 constant carries its
 * quantization in the constant-operand attribute; without the attribute only
 * a float Add or Mul takes a constant, as its second operand. */
static int constant_operand_is_valid(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_tensor_t *output, uint8_t dtype,
    const tigris_quant_param_t **quant)
{
    uint8_t len = 0u;
    const uint8_t *attr = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
    const uint32_t element = (dtype == 1u || dtype == 6u) ? 4u : 1u;
    *quant = NULL;
    if (op->num_inputs != (op->op_type == TIGRIS_OP_SELECT_V2 ? 2u : 1u) || op->num_outputs != 1u ||
        op->bias_idx != TIGRIS_NO_WEIGHT ||
        op->weight_idx == TIGRIS_NO_WEIGHT ||
        op->weight_idx >= plan->header->num_weights ||
        !plan->weight_entries ||
        (dtype != 1u && dtype != 3u && dtype != 9u && dtype != 6u))
        return 0;
    if (attr) {
        uint16_t index = (uint16_t)((uint16_t)attr[2] | (uint16_t)((uint16_t)attr[3] << 8));
        if (dtype == 3u) {
            if (index >= plan->num_quant_params ||
                plan->quant_params[index].num_channels != 1u)
                return 0;
            *quant = &plan->quant_params[index];
        } else if (index != TIGRIS_NO_QUANT_PARAM) {
            return 0;
        }
    } else if (dtype != 1u ||
               (op->op_type != TIGRIS_OP_ADD && op->op_type != TIGRIS_OP_MUL)) {
        return 0;
    }
    const uint32_t bytes = plan->weight_entries[op->weight_idx].size_bytes;
    if (attr != NULL && len != TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) {
        /* The long form states the constant's shape in stored axis order. */
        int32_t shape[TIGRIS_MAX_TENSOR_RANK_FOR_ATTR];
        uint64_t count = 1u;
        if (len != TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN + 4u * output->ndim ||
            output->ndim > TIGRIS_MAX_TENSOR_RANK_FOR_ATTR)
            return 0;
        memcpy(shape, attr + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN,
               sizeof(int32_t) * output->ndim);
        if (!shape_broadcasts_to(shape, output->ndim, plan, output))
            return 0;
        for (uint8_t d = 0; d < output->ndim; d++)
            count *= (uint64_t)shape[d];
        return count * element == bytes;
    }
    const uint32_t channels = output->ndim == 0u ? element : (uint32_t)
        tigris_tensor_shape(plan, output)[output->ndim - 1u] * element;
    return bytes == element || bytes == channels || bytes == tensor_elements(output) * element;
}

static int is_axis1_unary_pointwise_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_AXIS_UNARY) != 0u;
}

int tigris_height_spatial_op(uint8_t type)
{
    return type == TIGRIS_OP_CONV ||
           type == TIGRIS_OP_DEPTHWISE ||
           type == TIGRIS_OP_MAX_POOL ||
           type == TIGRIS_OP_AVG_POOL || type == TIGRIS_OP_L2_POOL;
}

/* Resize computes its source band from the full input and output extents. */
int tigris_height_upsample_op(uint8_t type)
{
    return type == TIGRIS_OP_RESIZE || type == TIGRIS_OP_RESIZE_LINEAR;
}

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

/**
 * Whether every op in a stage keeps a rank-2 matrix's rows independent.
 *
 * A matrix product against a constant reads one row of its left operand to
 * write one row of its output, and the pointwise operators beside it preserve
 * rows outright. Neither needs anything from a neighboring row, so a band of
 * rows computes exactly the rows it holds. A second activation operand is
 * excluded for the reason it is everywhere else: nothing co-tiles it.
 */
int tigris_row_tiling_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_ROW) != 0u;
}

/* The operators that may read an operand the band does not cut: a matrix
 * product reads every row of its second operand to produce one row of its
 * output, so that operand is resident whole exactly as a weight is. Mirrors
 * _ROW_TILING_WHOLE_OPERAND_OPS in the compiler. */
int tigris_whole_operand_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_WHOLE_OPERAND) != 0u;
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
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);

    if (stage->ops_count == 0u || stage->inputs_count == 0u ||
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
            b != batch || r != rows)
            return 0;
        /* A row view accepts a rank-3 [1, rows, cols] as the same memory, but
         * kern_fully_connected reads its channel count out of y_shape[1],
         * which on a rank-3 output is the row count. The matrix product is
         * the one operator here whose rank the row view does not settle. */
        if (op->op_type == TIGRIS_OP_FULLY_CONN && out->ndim != 2u)
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


int tigris_band_whole_input(const tigris_op_t *op, uint8_t position)
{
    return (position == 1u && (op->op_type == TIGRIS_OP_GATHER ||
           op->op_type == TIGRIS_OP_GATHER_ND || op->op_type == TIGRIS_OP_EMBEDDING_LOOKUP)) ||
           (position == 2u && op->op_type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE);
}

int tigris_op_independent_band(const tigris_plan_t *plan, const tigris_op_t *op,
                                uint16_t op_index, int row_tiled)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u ||
        (op->num_outputs != 1u && op->op_type != TIGRIS_OP_SPLIT)) return 0;
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *out = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    if (is_binary_op(op->op_type)) {
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
            uint32_t bytes = plan->weight_entries[op->weight_idx].size_bytes;
            if (bytes == out->size_bytes && bytes > tigris_dtype_size(out->dtype)) broadcast = 1;
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
        if (!payload || length > sizeof(metadata)) return 0;
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
    if (op->num_inputs == 1u && is_axis1_unary_pointwise_op(type) &&
        type != TIGRIS_OP_SOFTMAX && type != TIGRIS_OP_LOG_SOFTMAX &&
        type != TIGRIS_OP_L2_NORMALIZATION && type != TIGRIS_OP_LAYER_NORM && in->dtype != 6u)
        return in->ndim == 2u && row_tiled >= (int)TIGRIS_BAND_LEADING && tensor_shapes_equal(plan, in, out);
    if (type == TIGRIS_OP_REDUCE_MEAN || type == TIGRIS_OP_REDUCE_MAX ||
        type == TIGRIS_OP_REDUCE_MIN || type == TIGRIS_OP_REDUCE_SUM ||
        type == TIGRIS_OP_REDUCE_ALL || type == TIGRIS_OP_CUMSUM ||
        type == TIGRIS_OP_ARG_MAX || type == TIGRIS_OP_ARG_MIN) {
        const uint8_t *axes = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &length);
        return axes != NULL && length == 1u && axes[0] < in->ndim && axes[0] != row;
    }
    int32_t metadata[19] = {0};
    const uint8_t *payload = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_MOVEMENT, &length);
    if (!payload || length == 0u || length > sizeof(metadata) || length % 4u != 0u) return 0;
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

int tigris_stage_band_mode(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    /* The loader asks before it checks operand counts, and a legacy custom
     * operator may have none. */
    if (op->num_inputs != 0u && op->num_outputs != 0u &&
        (op->op_type == TIGRIS_OP_SPLIT || op->op_type == TIGRIS_OP_CONCAT ||
         op->op_type == TIGRIS_OP_PAD || op->op_type == TIGRIS_OP_TRANSPOSE || is_binary_op(op->op_type) ||
         (op->num_inputs > 1u && tigris_band_whole_input(op, 1u)) ||
         (op->num_inputs > 2u && tigris_band_whole_input(op, 2u)))) {
        const tigris_tensor_t *input = &plan->tensors[is_binary_op(op->op_type) ?
            tigris_op_outputs(plan, op)[0] : tigris_op_inputs(plan, op)[0]];
        const int32_t *shape = tigris_tensor_shape(plan, input);
        int mode = TIGRIS_BAND_LEADING;
        int32_t extent = 1;
        uint64_t best_cost = UINT64_MAX;
        for (uint8_t axis = 0u; axis < input->ndim; axis++) {
            int candidate = (int)TIGRIS_BAND_AXIS_BASE + axis;
            uint64_t cost = 0u;
            if (is_binary_op(op->op_type)) {
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
    if (op->op_type == TIGRIS_OP_SVDF) return TIGRIS_BAND_BATCH;
    if (op->op_type == TIGRIS_OP_LSTM) {
        const uint8_t *params = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_LSTM, NULL);
        int32_t time_major = 0;
        if (params) memcpy(&time_major, params, sizeof(time_major));
        return time_major ? TIGRIS_BAND_TIME_MAJOR_BATCH : TIGRIS_BAND_BATCH;
    }
    return !tigris_op_independent_band(plan, op, index, TIGRIS_BAND_LEADING) &&
        tigris_op_independent_band(plan, op, index, TIGRIS_BAND_TRAILING) ?
        TIGRIS_BAND_TRAILING : TIGRIS_BAND_LEADING;
}

int32_t tigris_stage_leading_band(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (!plan->shape_pool || stage->ops_count != 1u || stage->chain_len != 0u ||
        stage->inputs_count == 0u || stage->outputs_count == 0u) return 0;
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    if (op->op_type == TIGRIS_OP_SVDF || op->op_type == TIGRIS_OP_LSTM) {
        if (op->num_inputs != stage->inputs_count || op->num_outputs != stage->outputs_count)
            return 0;
        int mode = tigris_stage_band_mode(plan, stage);
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
    if (!tigris_op_independent_band(plan, op, index, tigris_stage_band_mode(plan, stage))) return 0;
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
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const int32_t *shape = tigris_tensor_shape(plan, in);
    /* Preserve the meaning of existing matrix-row and height records. */
    if (in->ndim == 2u && shape[0] > 1 && tigris_row_tiling_op(op->op_type)) return 0;
    if ((in->ndim >= 2u && shape[in->ndim - 2u] > 1 && tigris_op_independent_band(plan, op, index, 1)) ||
        tigris_op_independent_band(plan, op, index, 0)) return 0;
    const tigris_tensor_t *out = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    return tigris_tensor_shape(plan, out)[tigris_band_axis(plan, out, tigris_stage_band_mode(plan, stage))];
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
        in->size_bytes != out->size_bytes || in->ndim < 2u || in->ndim > 6u ||
        out->ndim < 2u || out->ndim > 6u) return 0;
    const tigris_quant_param_t *iq = tigris_tensor_quant(plan, in);
    const tigris_quant_param_t *oq = tigris_tensor_quant(plan, out);
    if (iq != oq && (!iq || !oq || iq->scale != oq->scale || iq->zero_point != oq->zero_point)) return 0;
    const int32_t *a = tigris_tensor_shape(plan, in), *b = tigris_tensor_shape(plan, out);
    int own_in = in->ndim == 2u || (in->flags & TIGRIS_TENSOR_LINEAR) != 0u;
    int own_out = out->ndim == 2u || (out->flags & TIGRIS_TENSOR_LINEAR) != 0u;
    if (own_in != own_out || (!own_in && (a[0] != b[0] || a[in->ndim - 1u] != b[out->ndim - 1u]))) return 0;
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

tigris_band_kind_t tigris_stage_band_kind(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                          tigris_stage_band_t *band)
{
    tigris_stage_band_t unused;
    tigris_stage_band_t *out = band != NULL ? band : &unused;
    out->batch = 0;
    out->rows = 0;
    if (plan == NULL || plan->header == NULL || stage == NULL)
        return TIGRIS_BAND_KIND_NONE;
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    int banded = tile.tileable != 0u && tile.axis != TIGRIS_TILE_AXIS_NONE;
    if (banded) {
        tigris_reshape_band_t view;
        if (tigris_reshape_band(plan, stage, &view)) {
            out->rows = view.input_rows;
            return TIGRIS_BAND_KIND_RESHAPE;
        }
        int32_t leading = tigris_stage_leading_band(plan, stage);
        if (leading > 0) {
            out->rows = leading;
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

static int grouped_reduction_valid(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                   const tigris_tile_plan_v10_t *tile)
{
    if (!tile->tileable || tile->axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH ||
        tile->flags != 0u || tile->input_band_extent != 0u || tile->output_band_extent != 0u ||
        tile->output_width_extent != 0u || tile->output_linear_segment != 0u ||
        stage->ops_count != 1u || stage->inputs_count != 1u || stage->outputs_count != 1u ||
        stage->chain_len != 0u || tile->input_channel_group == 0u || tile->input_spatial_interval == 0u)
        return 0;
    const tigris_op_t *op = &plan->ops[tigris_stage_ops(plan, stage)[0]];
    if (op->op_type != TIGRIS_OP_GLOBAL_AVG || op->num_inputs != 1u || op->num_outputs != 1u ||
        tigris_op_inputs(plan, op)[0] != tigris_stage_inputs(plan, stage)[0] ||
        tigris_op_outputs(plan, op)[0] != tigris_stage_outputs(plan, stage)[0])
        return 0;
    const tigris_tensor_t *input = &plan->tensors[tigris_stage_inputs(plan, stage)[0]];
    const tigris_tensor_t *output = &plan->tensors[tigris_stage_outputs(plan, stage)[0]];
    if (input->ndim != 4u || output->ndim != 4u || input->dtype != output->dtype ||
        (input->dtype != 1u && input->dtype != 3u))
        return 0;
    const int32_t *shape = tigris_tensor_shape(plan, input);
    uint64_t spatial = (uint64_t)(uint32_t)shape[1] * (uint32_t)shape[2];
    if (tile->input_channel_group > (uint32_t)shape[3] || tile->input_spatial_interval > spatial ||
        spatial > INT32_MAX || (input->dtype == 3u && spatial > (uint32_t)INT32_MAX / 255u))
        return 0;
    uint32_t element = tigris_dtype_size(input->dtype);
    uint64_t channel = tile->input_channel_group;
    uint64_t bytes[3] = {channel * 4u, channel * element,
                        channel * tile->input_spatial_interval * element};
    uint64_t required = 0u;
    for (uint8_t i = 0u; i < 3u; i++)
        required += (bytes[i] + TIGRIS_TENSOR_ALIGN - 1u) / TIGRIS_TENSOR_ALIGN * TIGRIS_TENSOR_ALIGN;
    return required <= tile->tiled_peak_bytes && tile->tiled_peak_bytes != 0u &&
           tile->tiled_peak_bytes <= plan->header->budget;
}

static int tile_v10_valid(const tigris_plan_t *plan, const tigris_stage_t *stage,
                          const tigris_tile_plan_v10_t *tile)
{
    if (tile->input_channel_group != 0u || tile->input_spatial_interval != 0u)
        return grouped_reduction_valid(plan, stage, tile);
    if (tile->output_linear_segment != 0u ||
        tile->flags > TIGRIS_TILE_FLAG_BAND_ON_COLUMNS)
        return 0;
    if (!tile->tileable)
        return tile->input_band_extent == 0u && tile->output_band_extent == 0u &&
               tile->output_width_extent == 0u && tile->flags == 0u && tile->tiled_peak_bytes == 0u;
    if (stage->inputs_count == 0u || stage->outputs_count == 0u ||
        tile->tiled_peak_bytes == 0u ||
        (plan->header->budget != 0u && tile->tiled_peak_bytes > plan->header->budget))
        return 0;
    const uint16_t *inputs = tigris_stage_inputs(plan, stage);
    const uint16_t *outputs = tigris_stage_outputs(plan, stage);
    const uint16_t *ops = tigris_stage_ops(plan, stage);
    const tigris_tensor_t *in = &plan->tensors[inputs[0]];
    const tigris_tensor_t *out = &plan->tensors[outputs[0]];
    const int32_t *b = tigris_tensor_shape(plan, out);
    if (tile->axis == TIGRIS_TILE_AXIS_HW)
        return in->ndim == 4u && out->ndim == 4u && tile->input_band_extent == 0u &&
               tile->output_band_extent > 0u && tile->output_band_extent <= (uint32_t)b[1] &&
               tile->output_width_extent > 0u && tile->output_width_extent <= (uint32_t)b[2] &&
               tile->flags == 0u;
    if (tile->output_width_extent != 0u)
        return 0;
    tigris_stage_band_t band;
    tigris_band_kind_t kind = tigris_stage_band_kind(plan, stage, &band);
    if (kind == TIGRIS_BAND_KIND_LEADING)
        return tile->flags == 0u && tile->output_band_extent == 0u &&
               tile->input_band_extent > 0u && tile->input_band_extent <= (uint32_t)band.rows;
    if (stage->ops_count == 1u && plan->ops[ops[0]].op_type == TIGRIS_OP_TRANSPOSE)
        return tile->input_band_extent == 0u && tile->output_band_extent == 0u;
    if (tile->flags != 0u)
        return 0;
    int resize = 0;
    for (uint16_t i = 0; i < stage->ops_count; i++)
        if (plan->ops[ops[i]].op_type == TIGRIS_OP_RESIZE || plan->ops[ops[i]].op_type == TIGRIS_OP_RESIZE_LINEAR)
            resize = 1;
    if (resize)
        return out->ndim >= 3u && tile->input_band_extent == 0u &&
               tile->output_band_extent > 0u && tile->output_band_extent <= (uint32_t)b[1];
    if (tile->output_band_extent != 0u || tile->input_band_extent == 0u ||
        tile->input_band_extent > INT32_MAX)
        return 0;
    tigris_reshape_band_t view;
    if (kind == TIGRIS_BAND_KIND_RESHAPE && tigris_reshape_band(plan, stage, &view))
        return tile->input_band_extent <= (uint32_t)view.input_rows &&
               ((int64_t)tile->input_band_extent * view.input_width) % view.output_width == 0;
    int32_t batch = 0, rows = 0, cols = 0;
    if (tigris_tensor_row_view(plan, out, &batch, &rows, &cols))
        if (tile->input_band_extent <= (uint32_t)rows)
            return 1;
    /* A reduction's output has no rows; a broadcast operand does not set them. */
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        const tigris_tensor_t *tensor = &plan->tensors[inputs[i]];
        const int32_t *shape = tigris_tensor_shape(plan, tensor);
        if (tensor->ndim >= 3u && tile->input_band_extent <= (uint32_t)shape[1])
            return 1;
    }
    return 0;
}

static int is_axis1_binary_pointwise_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_AXIS_BINARY) != 0u;
}

/**
 * Whether an attribute kind belongs to this operator.
 *
 * Schema 8 declares every kind at once so the operators behind them can land
 * one at a time, and this is what keeps that from loosening the format: a
 * payload is only accepted on the operator that will read it.
 */
static int attr_kind_matches_op(uint8_t kind, uint8_t op_type)
{
    const tigris_op_contract_t *row = tigris_op_contract(op_type);
    return row != NULL && kind < 32u &&
           (row->allowed_attributes & ((uint32_t)1u << kind)) != 0u;
}

/* Schemas v2/v3 let a custom dispatcher implement Transpose, MatMul and
 * opcodes the built-in table did not define yet, so only the built-in
 * operators whose meaning those schemas already fixed are validated there. */
static int legacy_semantics_are_builtin(uint8_t op_type)
{
    const tigris_op_contract_t *row = tigris_op_contract(op_type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_LEGACY) != 0u;
}

/* Whether two activations are encoded alike: the same scale and zero point,
 * whichever entry of the plan states them, so a byte copy between them is
 * exact. Per-channel requantization a record carries for the operator that
 * wrote the tensor is not part of its encoding. */
static int same_quantization(const tigris_quant_param_t *a, const tigris_quant_param_t *b)
{
    if (a == b)
        return 1;
    if (a == NULL || b == NULL)
        return 0;
    return memcmp(&a->scale, &b->scale, sizeof(a->scale)) == 0 &&
           a->zero_point == b->zero_point;
}

static int movement_is_gather(uint8_t type)
{
    return type == TIGRIS_OP_GATHER || type == TIGRIS_OP_GATHER_ND ||
           type == TIGRIS_OP_EMBEDDING_LOOKUP;
}

static int movement_metadata_valid(const tigris_plan_t *plan,
                                   const tigris_op_t *op, uint16_t index)
{
    int32_t metadata[19] = {0};
    int32_t expected[6] = {0};
    uint8_t length = 0u;
    const uint8_t *payload = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_MOVEMENT, &length);
    uint8_t table_length = 0u;
    const uint8_t *table = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_TABLE_SHAPE, &table_length);
    if (!payload || length == 0u || length > sizeof(metadata) || length % 4u != 0u ||
        op->num_outputs != 1u || (table == NULL && (op->num_inputs == 0u || op->bias_idx != TIGRIS_NO_WEIGHT)))
        return 0;
    memcpy(metadata, payload, length);
    const tigris_tensor_t *output = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    const tigris_tensor_t *input = table != NULL ? output : &plan->tensors[tigris_op_inputs(plan, op)[0]];
    int32_t table_shape[6];
    uint8_t rank = table != NULL ? (uint8_t)(table_length / 4u) : input->ndim;
    if (table != NULL) {
        if (table_length == 0u || table_length > sizeof(table_shape) || table_length % 4u != 0u ||
            op->weight_idx == TIGRIS_NO_WEIGHT) return 0;
        memcpy(table_shape, table, table_length);
        uint64_t bytes = tigris_dtype_size(output->dtype);
        for (uint8_t i = 0u; i < rank; i++) {
            if (table_shape[i] <= 0 || bytes == 0u || bytes > UINT32_MAX / (uint32_t)table_shape[i]) return 0;
            bytes *= (uint32_t)table_shape[i];
        }
        if (plan->weight_entries[op->weight_idx].size_bytes != bytes) return 0;
    }
    const int32_t *shape = table != NULL ? table_shape : tigris_tensor_shape(plan, input);
    const int32_t *target = tigris_tensor_shape(plan, output);
    uint8_t written = 0u;
    if (rank == 0u || rank > 6u || output->ndim > 6u ||
        (input->dtype != 1u && input->dtype != 3u) || input->dtype != output->dtype ||
        (input->dtype == 3u && !same_quantization(tigris_tensor_quant(plan, input), tigris_tensor_quant(plan, output))))
        return 0;
    if (movement_is_gather(op->op_type)) {
        uint16_t index_weight = table != NULL ? op->bias_idx : op->weight_idx;
        if (op->num_inputs != (index_weight == TIGRIS_NO_WEIGHT ? (table != NULL ? 1u : 2u) : (table != NULL ? 0u : 1u))) return 0;
        int32_t irank = metadata[0];
        uint8_t head = 1u;
        int32_t axis = 0, batch = 0;
        if (op->op_type != TIGRIS_OP_GATHER_ND) {
            head = 3u;
            axis = metadata[0]; batch = metadata[1]; irank = metadata[2];
            if (axis < 0 || axis >= rank || batch < 0 || batch > axis || batch > irank)
                return 0;
        }
        if (irank < 0 || irank > 6 || length != ((uint32_t)head + (uint32_t)irank) * 4u)
            return 0;
        uint64_t indices = 1u;
        for (int32_t i = 0; i < irank; i++) {
            int32_t dim = metadata[head + i];
            if (dim <= 0 || indices > UINT32_MAX / 4u / (uint32_t)dim) return 0;
            indices *= (uint32_t)dim;
        }
        if (index_weight != TIGRIS_NO_WEIGHT) {
            if (plan->weight_entries[index_weight].size_bytes != indices * 4u) return 0;
        } else {
            const tigris_tensor_t *index_tensor = &plan->tensors[tigris_op_inputs(plan, op)[table != NULL ? 0u : 1u]];
            if (index_tensor->dtype != 6u || index_tensor->ndim != irank || index_tensor->size_bytes != indices * 4u ||
                index_tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM ||
                memcmp(tigris_tensor_shape(plan, index_tensor), metadata + head, (size_t)irank * 4u) != 0)
                return 0;
        }
        if (op->op_type == TIGRIS_OP_GATHER_ND) {
            if (irank == 0) return 0;
            int32_t depth = metadata[irank];
            if (depth < 1 || depth > rank || depth > 5 || irank - 1 + rank - depth > 6) return 0;
            for (int32_t i = 0; i + 1 < irank; i++) expected[written++] = metadata[1 + i];
            for (int32_t i = depth; i < rank; i++) expected[written++] = shape[i];
        } else {
            if ((int32_t)rank - 1 + irank - batch > 6) return 0;
            if (op->op_type == TIGRIS_OP_EMBEDDING_LOOKUP &&
                (rank < 2u || axis != 0 || batch != 0 || irank != 1)) return 0;
            for (int32_t i = 0; i < batch; i++) if (metadata[head + i] != shape[i]) return 0;
            for (int32_t i = 0; i < axis; i++) expected[written++] = shape[i];
            for (int32_t i = batch; i < irank; i++) expected[written++] = metadata[head + i];
            for (int32_t i = axis + 1; i < rank; i++) expected[written++] = shape[i];
        }
    } else {
        if (op->op_type != TIGRIS_OP_DYNAMIC_UPDATE_SLICE &&
            (op->weight_idx != TIGRIS_NO_WEIGHT || op->num_inputs != 1u)) return 0;
        written = rank;
        if (op->op_type == TIGRIS_OP_STRIDED_SLICE) {
            if (rank > 4u || (length != (uint32_t)rank * 12u && length != (uint32_t)rank * 12u + 4u)) return 0;
            uint32_t shrink = 0u;
            if (length > (uint32_t)rank * 12u) shrink = (uint32_t)metadata[3u * rank];
            if (shrink >= (1u << rank)) return 0;
            written = 0u;
            for (uint8_t a = 0; a < rank; a++) {
                int64_t start = metadata[3u * a], end = metadata[3u * a + 1u], step = metadata[3u * a + 2u];
                if (step == 0 || start < 0 || start >= shape[a] || end < -1 || end > shape[a]) return 0;
                int64_t distance = step > 0 ? end - start : start - end;
                int64_t stride = step > 0 ? step : -step;
                int64_t count = (distance + stride - 1) / stride;
                int64_t last = start + (count - 1) * step;
                if (distance <= 0 || count > INT32_MAX || last < 0 || last >= shape[a]) return 0;
                if ((shrink & (1u << a)) != 0u) {
                    if (count != 1 || step <= 0) return 0;
                } else expected[written++] = (int32_t)count;
            }
        } else if (op->op_type == TIGRIS_OP_MIRROR_PAD) {
            if (length != (1u + 2u * rank) * 4u || metadata[0] < 0 || metadata[0] > 1) return 0;
            for (uint8_t a = 0; a < rank; a++) {
                int32_t before = metadata[1u + 2u * a], after = metadata[2u + 2u * a];
                int64_t count = (int64_t)shape[a] + before + after;
                int32_t limit = shape[a] - 1 + metadata[0];
                if (before < 0 || after < 0 || before > limit || after > limit || count > INT32_MAX) return 0;
                expected[a] = (int32_t)count;
            }
        } else if (op->op_type == TIGRIS_OP_REVERSE_V2) {
            if (length != 4u || metadata[0] <= 0 || (uint32_t)metadata[0] >= (1u << rank)) return 0;
            uint32_t mask = (uint32_t)metadata[0];
            while ((mask & 1u) == 0u) mask >>= 1u;
            if ((mask & (mask + 1u)) != 0u) return 0;
            memcpy(expected, shape, (size_t)rank * sizeof(int32_t));
        } else if (op->op_type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE) {
            int dynamic = length == (uint32_t)rank * 4u;
            uint8_t operands = op->weight_idx == TIGRIS_NO_WEIGHT ? 2u : 1u;
            if ((!dynamic && length != (uint32_t)rank * 8u) ||
                op->num_inputs != operands + (dynamic ? 1u : 0u)) return 0;
            if (dynamic) {
                const tigris_tensor_t *starts = &plan->tensors[tigris_op_inputs(plan, op)[operands]];
                if (starts->dtype != 6u || starts->ndim != 1u ||
                    tigris_tensor_shape(plan, starts)[0] != rank ||
                    starts->quant_param_idx != TIGRIS_NO_QUANT_PARAM) return 0;
            }
            const int32_t *ushape = metadata + (dynamic ? 0u : rank);
            uint64_t update_elements = 1u;
            for (uint8_t a = 0; a < rank; a++) {
                if (ushape[a] <= 0 || update_elements > UINT32_MAX / (uint32_t)ushape[a]) return 0;
                update_elements *= (uint32_t)ushape[a];
            }
            if (op->weight_idx != TIGRIS_NO_WEIGHT) {
                if (plan->weight_entries[op->weight_idx].size_bytes != update_elements * tigris_dtype_size(input->dtype)) return 0;
            } else {
                const tigris_tensor_t *update = &plan->tensors[tigris_op_inputs(plan, op)[1]];
                if (update->ndim != rank || update->dtype != input->dtype ||
                    memcmp(ushape, tigris_tensor_shape(plan, update), (size_t)rank * sizeof(int32_t)) != 0 ||
                    (input->dtype == 3u && !same_quantization(tigris_tensor_quant(plan, input), tigris_tensor_quant(plan, update))))
                    return 0;
            }
            for (uint8_t a = 0; a < rank; a++) {
                if (ushape[a] > shape[a] ||
                    (!dynamic && (metadata[a] < 0 || metadata[a] > shape[a] - ushape[a]))) return 0;
                expected[a] = shape[a];
            }
        } else return 0;
    }
    if (written != output->ndim) return 0;
    for (uint8_t a = 0; a < written; a++) if (expected[a] != target[a]) return 0;
    return 1;
}




uint8_t tigris_plan_data_dtype(const tigris_plan_t *plan)
{
    /* Float32 tensors in an int8 plan are admitted only by operator slots
     * that state float32, which the dtype signatures check. */
    for (uint16_t i = 0; i < plan->header->num_tensors; i++) {
        const tigris_tensor_t *tensor = &plan->tensors[i];
        if ((tensor->flags & TIGRIS_TENSOR_CONSTANT) == 0u && tensor->dtype == 3u) return 3u;
    }
    return 1u;
}

/* Role 0 is the plan's data dtype, 255 data or bool, 254 data or int16 state
 * (the tensor table admits int16 only on state tensors), 253 data or int32,
 * 252 bool or int32, 251 data, int32 or float32 (control flow carries float32
 * across the boundary of an int8 plan's subgraph). */
/* The weight index of an operator's constant operand k, or TIGRIS_NO_WEIGHT. */
static uint16_t constant_index(const tigris_plan_t *plan, uint16_t op_index, uint8_t k)
{
    uint8_t len = 0u;
    const uint8_t *list = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &len);
    if (!list || (uint32_t)k * 2u + 2u > len) return TIGRIS_NO_WEIGHT;
    return (uint16_t)((uint16_t)list[(uint32_t)k * 2u] |
                      (uint16_t)((uint16_t)list[(uint32_t)k * 2u + 1u] << 8));
}

/* DetectionPostProcess: box encodings [1, boxes, 4] and scores [1, boxes,
 * columns] in the data dtype, float32 anchors [boxes, 4] as its constant; float32
 * boxes [1, detections, 4], classes and scores [1, detections], the count [1]
 * and working memory of at least TIGRIS_DETECTION_SCRATCH_BYTES out. */
static int detection_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index)
{
    uint8_t len = 0u;
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_DETECTION, &len);
    uint8_t constants_len = 0u;
    if (op->num_inputs != 2u || op->num_outputs != 5u || op->weight_idx != TIGRIS_NO_WEIGHT ||
        op->bias_idx != TIGRIS_NO_WEIGHT || !params || len != TIGRIS_OP_ATTR_DETECTION_LEN ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 2u)
        return 0;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *encodings = &plan->tensors[ins[0]];
    const tigris_tensor_t *scores = &plan->tensors[ins[1]];
    uint16_t limits[3];
    float values[6];
    memcpy(limits, params, sizeof(limits));
    memcpy(values, params + 8, sizeof(values));
    const uint8_t dtype = encodings->dtype;
    if (encodings->ndim != 3u || scores->ndim != 3u || scores->dtype != dtype ||
        (dtype == 3u && (!tigris_tensor_quant(plan, encodings) || !tigris_tensor_quant(plan, scores))))
        return 0;
    const int32_t *es = tigris_tensor_shape(plan, encodings);
    const int32_t *ss = tigris_tensor_shape(plan, scores);
    const int32_t boxes = es[1];
    if (es[0] != 1 || es[2] != 4 || boxes <= 0 || boxes > 65535 || ss[0] != 1 || ss[1] != boxes ||
        limits[0] == 0u || limits[2] == 0u || ss[2] < (int32_t)limits[2] || ss[2] > (int32_t)limits[2] + 1 ||
        params[6] > 1u || params[7] != 0u || (params[6] == 1u && limits[1] == 0u))
        return 0;
    /* Thresholds and scales as TFLite Micro checks or divides by them. */
    if (!isfinite(values[0]) || !(values[1] > 0.0f) || !(values[1] <= 1.0f))
        return 0;
    for (uint8_t k = 2u; k < 6u; k++)
        if (!isfinite(values[k]) || values[k] == 0.0f)
            return 0;
    const int32_t expected[4][3] = {{1, (int32_t)limits[0], 4}, {1, (int32_t)limits[0], 0},
                                    {1, (int32_t)limits[0], 0}, {1, 0, 0}};
    const uint8_t ranks[4] = {3u, 2u, 2u, 1u};
    for (uint8_t i = 0u; i < 5u; i++) {
        const tigris_tensor_t *out = &plan->tensors[outs[i]];
        if (out->dtype != 1u || out->quant_param_idx != TIGRIS_NO_QUANT_PARAM)
            return 0;
        if (i < 4u) {
            const int32_t *shape = tigris_tensor_shape(plan, out);
            if (out->ndim != ranks[i]) return 0;
            for (uint8_t a = 0u; a < ranks[i]; a++)
                if (shape[a] != expected[i][a]) return 0;
        }
    }
    const size_t candidates = (size_t)limits[0] + (params[6] == 1u ? (size_t)limits[1] : 0u);
    if ((size_t)plan->tensors[outs[4]].size_bytes <
        TIGRIS_DETECTION_SCRATCH_BYTES(boxes, limits[0], candidates))
        return 0;
    return weight_size_is(plan, constant_index(plan, op_index, 0u), (uint64_t)boxes * 16u);
}

/* Svdf: input [batch, features] and state [batch, filters * memory] in, output
 * [batch, units] and the next state out; feature, time and bias constants. A
 * float Svdf keeps everything in float32; an int8 one keeps state and time
 * weights in int16 and its bias in int32, as TFLite Micro does. */
static int svdf_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index)
{
    uint8_t len = 0u;
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SVDF, &len);
    uint8_t constants_len = 0u;
    if (op->num_inputs != 2u || op->num_outputs != 2u || op->weight_idx != TIGRIS_NO_WEIGHT ||
        op->bias_idx != TIGRIS_NO_WEIGHT || !params ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 6u)
        return 0;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *x = &plan->tensors[ins[0]];
    const tigris_tensor_t *state = &plan->tensors[ins[1]];
    const tigris_tensor_t *y = &plan->tensors[outs[0]];
    const tigris_tensor_t *kept = &plan->tensors[outs[1]];
    int quantized = x->dtype == 3u;
    int32_t rank;
    memcpy(&rank, params, sizeof(rank));
    if (x->ndim != 2u || y->ndim != 2u || state->ndim != 2u ||
        (state->flags & TIGRIS_TENSOR_STATE) == 0u || (kept->flags & TIGRIS_TENSOR_STATE) == 0u ||
        !tensor_shapes_equal(plan, state, kept) || state->dtype != kept->dtype ||
        len != (quantized ? 24u : 4u) || rank <= 0 ||
        state->dtype != (quantized ? 5u : 1u) || y->dtype != x->dtype ||
        (quantized && (!tigris_tensor_quant(plan, x) || !tigris_tensor_quant(plan, y))))
        return 0;
    const int32_t *xs = tigris_tensor_shape(plan, x);
    const int32_t *ys = tigris_tensor_shape(plan, y);
    const int32_t *ss = tigris_tensor_shape(plan, state);
    uint64_t filters = (uint64_t)ys[1] * (uint64_t)rank;
    if (xs[0] != ys[0] || xs[0] != ss[0] || filters > INT32_MAX ||
        (uint64_t)ss[1] % filters != 0u)
        return 0;
    uint64_t memory = (uint64_t)ss[1] / filters;
    uint16_t feature = constant_index(plan, op_index, 0u);
    uint16_t time = constant_index(plan, op_index, 1u);
    uint16_t bias = constant_index(plan, op_index, 2u);
    return weight_size_is(plan, feature, filters * (uint64_t)xs[1] * (quantized ? 1u : 4u)) &&
           weight_size_is(plan, time, filters * memory * (quantized ? 2u : 4u)) &&
           weight_size_is(plan, bias, (uint64_t)ys[1] * 4u);
}

/* Lstm: sequence [batch, steps, features] ([steps, batch, features] when time
 * major), hidden and cell state [batch, units] in; the hidden sequence and
 * both states out. Weights [units, features] then [units, units], four of
 * each, and four biases [units]. Float32 throughout, or int8 sequence and
 * hidden state with an int16 cell, int8 weights and int32 biases. */
static int lstm_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index)
{
    uint8_t len = 0u;
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_LSTM, &len);
    uint8_t constants_len = 0u;
    if (op->num_inputs != 3u || op->num_outputs != 3u || op->weight_idx != TIGRIS_NO_WEIGHT ||
        op->bias_idx != TIGRIS_NO_WEIGHT || !params ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 24u)
        return 0;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *x = &plan->tensors[ins[0]];
    const tigris_tensor_t *y = &plan->tensors[outs[0]];
    int32_t time_major;
    memcpy(&time_major, params, sizeof(time_major));
    const int quantized = x->dtype == 3u;
    if (x->ndim != 3u || y->ndim != 3u || (x->dtype != 1u && !quantized) || y->dtype != x->dtype ||
        len != (quantized ? 112u : 8u) ||
        (quantized && (!tigris_tensor_quant(plan, x) || !tigris_tensor_quant(plan, &plan->tensors[ins[1]]))))
        return 0;
    if (quantized) {
        int32_t v[28];
        memcpy(v, params, sizeof(v));
        /* Zero points in int8, a cell scale between 2^-30 and 2^15, a clip in
         * cell units, and shifts the requantization can apply. */
        if (v[2] < -128 || v[2] > 127 || v[3] < -128 || v[3] > 127 || v[4] < -30 || v[4] > 15 ||
            v[5] < 0 || v[5] > 32767)
            return 0;
        for (uint8_t pair = 6u; pair < 28u; pair += 2u)
            if (v[pair] < 0 || v[pair + 1u] < -31 || v[pair + 1u] > 30)
                return 0;
    }
    const int32_t *xs = tigris_tensor_shape(plan, x);
    const int32_t *ys = tigris_tensor_shape(plan, y);
    const int32_t batch = time_major ? xs[1] : xs[0];
    const int32_t units = ys[2];
    if (xs[0] != ys[0] || xs[1] != ys[1])
        return 0;
    for (uint8_t i = 1u; i < 3u; i++) {
        const tigris_tensor_t *held = &plan->tensors[ins[i]];
        const tigris_tensor_t *kept = &plan->tensors[outs[i]];
        const int32_t *hs = tigris_tensor_shape(plan, held);
        const uint8_t dtype = quantized ? (i == 1u ? 3u : 5u) : 1u;
        if (held->ndim != 2u || held->dtype != dtype || hs[0] != batch || hs[1] != units ||
            (held->flags & TIGRIS_TENSOR_STATE) == 0u || (kept->flags & TIGRIS_TENSOR_STATE) == 0u ||
            kept->dtype != dtype || !tensor_shapes_equal(plan, held, kept))
            return 0;
    }
    for (uint8_t k = 0u; k < 12u; k++) {
        const uint64_t columns = k < 4u ? (uint64_t)xs[2] : (k < 8u ? (uint64_t)units : 1u);
        const uint64_t element = (quantized && k < 8u) ? 1u : 4u;
        if (!weight_size_is(plan, constant_index(plan, op_index, k),
                            (uint64_t)units * columns * element))
            return 0;
    }
    return 1;
}

static int dtype_matches_slot(uint8_t dtype, uint8_t role, uint8_t data_dtype)
{
    if (role == 250u) return dtype == 1u || dtype == data_dtype;
    if (role == 253u) return dtype == data_dtype || dtype == 6u;
    if (role == 252u) return dtype == 9u || dtype == 6u;
    if (role == 251u) return dtype == data_dtype || dtype == 6u || dtype == 1u;
    if (role == 254u) return dtype == data_dtype || dtype == 5u;
    return role == 255u ? dtype == data_dtype || dtype == 6u || dtype == 9u : dtype == (role == 0u ? data_dtype : role);
}

/* General broadcasting follows the reference rank limit; leading repeats need no coordinates. */
static int bool_broadcast_valid(const int32_t *shape, uint8_t ndim, const tigris_plan_t *plan,
                                const tigris_tensor_t *output, uint8_t limit)
{
    if (output->ndim > 8u || !shape_broadcasts_to(shape, ndim, plan, output)) return 0;
    if (output->ndim <= limit) return 1;
    const int32_t *target = tigris_tensor_shape(plan, output);
    uint8_t suffix = ndim;
    while (suffix > 0u && shape[suffix - 1u] == target[output->ndim - ndim + suffix - 1u]) suffix--;
    for (uint8_t i = 0; i < suffix; i++) if (shape[i] != 1) return 0;
    return 1;
}

static int bool_and_sum_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const tigris_tensor_t *x = &plan->tensors[inputs[0]];
    const tigris_tensor_t *y = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    const uint8_t type = op->op_type;
    if (op->num_outputs != 1u || op->bias_idx != TIGRIS_NO_WEIGHT) return 0;
    if (type == TIGRIS_OP_LOGICAL_NOT || type == TIGRIS_OP_CAST || type == TIGRIS_OP_ADD_N) {
        if (op->weight_idx != TIGRIS_NO_WEIGHT ||
            (type != TIGRIS_OP_ADD_N && op->num_inputs != 1u)) return 0;
        /* int32 casts to float32 only, as the converter writes a counter read as a float. */
        if (x->dtype == 6u && (type != TIGRIS_OP_CAST || y->dtype != 1u)) return 0;
        for (uint8_t i = 0; i < op->num_inputs; i++) {
            const tigris_tensor_t *t = &plan->tensors[inputs[i]];
            if (!tensor_shapes_equal(plan, t, y)) return 0;
            if (type == TIGRIS_OP_ADD_N && x->dtype == 3u &&
                !same_quantization(tigris_tensor_quant(plan, x), tigris_tensor_quant(plan, t))) return 0;
        }
        if (y->dtype == 3u) {
            const tigris_quant_param_t *qy = tigris_tensor_quant(plan, y);
            if (qy == NULL || qy->num_channels != 1u) return 0;
            if (type == TIGRIS_OP_CAST)
                return isfinite(qy->scale) && qy->scale > 0.0f &&
                       qy->zero_point >= -128 && qy->zero_point <= 127;
            const tigris_quant_param_t *qx = tigris_tensor_quant(plan, x);
            if (qx == NULL) return 0;
            /* The exact input multiplier 0.5 follows the reference's left shift 20. */
            const int64_t low = (-(int64_t)qx->zero_point + (int64_t)op->num_inputs * (-128 - (int64_t)qx->zero_point)) * 524288;
            const int64_t high = (-(int64_t)qx->zero_point + (int64_t)op->num_inputs * (127 - (int64_t)qx->zero_point)) * 524288;
            const double multiplier = 2.0 * (double)qx->scale / (1048576.0 * (double)qy->scale);
            if (low < INT32_MIN || high > INT32_MAX || multiplier <= 0.0 || multiplier >= 1.0) return 0;
        }
        return 1;
    }
    if (y->ndim > 8u) return 0;
    const uint8_t arity = type == TIGRIS_OP_SELECT_V2 ? 3u : 2u;
    const uint8_t rank_limit = arity == 3u ? 5u : 4u;
    uint8_t len = 0u;
    const uint8_t *constant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
    const uint8_t data_dtype = x->dtype == 1u || y->dtype == 1u ? 1u : tigris_plan_data_dtype(plan);
    const int comparison = type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_GREATER_EQUAL;
    if (op->num_inputs != arity && op->num_inputs != arity - 1u) return 0;
    if (op->num_inputs == arity && (constant != NULL || op->weight_idx != TIGRIS_NO_WEIGHT)) return 0;
    if (op->num_inputs != arity && (constant == NULL || constant[0] >= arity)) return 0;
    const tigris_quant_param_t *qy = tigris_tensor_quant(plan, y);
    uint8_t input_slot = 0u;
    for (uint8_t k = 0u; k < arity; k++) {
        const uint8_t dtype = comparison ? (x->dtype == 6u ? 6u : data_dtype)
                            : ((arity == 3u && k != 0u) ? data_dtype : 9u);
        const tigris_quant_param_t *q = NULL;
        if (constant != NULL && k == constant[0]) {
            if (!constant_operand_is_valid(plan, op, op_index, y, dtype, &q)) return 0;
            if (len > TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) {
                int32_t shape[8];
                memcpy(shape, constant + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN, 4u * y->ndim);
                if (!bool_broadcast_valid(shape, y->ndim, plan, y, rank_limit)) return 0;
            }
        } else {
            const tigris_tensor_t *t = &plan->tensors[inputs[input_slot]];
            input_slot++;
            if (!bool_broadcast_valid(tigris_tensor_shape(plan, t), t->ndim, plan, y, rank_limit)) return 0;
            if (comparison && t->dtype != dtype) return 0;
            q = tigris_tensor_quant(plan, t);
        }
        if (dtype == 3u) {
            if (q == NULL) return 0;
            if (comparison && (q->scale <= 0.0f || q->scale >= 1.0f)) return 0;
            if (arity == 3u && !same_quantization(q, qy)) return 0;
        }
    }
    uint8_t requant_len = 0u;
    const uint8_t *requant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_COMPARISON_REQUANT, &requant_len);
    if (comparison && data_dtype == 3u && x->dtype != 6u)
        return requant != NULL && requant_len == TIGRIS_OP_ATTR_COMPARISON_REQUANT_LEN;
    return requant == NULL;
}

/* The graph whose stage range holds stage s; 0 is the main graph. */
static uint16_t graph_of_stage(const tigris_plan_t *plan, uint16_t s)
{
    for (uint16_t g = 1; g < plan->num_subgraphs; g++) {
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        if (s >= graph->first_stage && s - graph->first_stage < graph->num_stages)
            return g;
    }
    return 0u;
}

/* Widen [lo, hi] to cover tensor t. */
static void cover(uint16_t *lo, uint16_t *hi, uint16_t t)
{
    if (t < *lo) *lo = t;
    if (t > *hi) *hi = t;
}

static int same_tensor_form(const tigris_plan_t *plan, uint16_t a, uint16_t b)
{
    const tigris_tensor_t *x = &plan->tensors[a];
    const tigris_tensor_t *y = &plan->tensors[b];
    /* Below rank 3 both layouts store the same bytes. */
    return x->dtype == y->dtype && x->size_bytes == y->size_bytes &&
           tensor_shapes_equal(plan, x, y) &&
           (x->ndim < 3u ||
            (x->flags & TIGRIS_TENSOR_LINEAR) == (y->flags & TIGRIS_TENSOR_LINEAR));
}

/* A While's loop variables, results, condition inputs, body inputs and body
 * outputs all have one form, position by position; its condition gives one
 * bool. */
static int while_graphs_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
                              const uint8_t *graphs, uint16_t container)
{
    const uint16_t condition = (uint16_t)((uint16_t)graphs[0] | (uint16_t)((uint16_t)graphs[1] << 8));
    const uint16_t body = (uint16_t)((uint16_t)graphs[2] | (uint16_t)((uint16_t)graphs[3] << 8));
    if (condition == 0u || body == 0u || condition == body || condition <= container ||
        body <= container ||
        condition >= plan->num_subgraphs || body >= plan->num_subgraphs)
        return 0;
    /* A loop variable starts from the next input, or from the constant the
     * constants attribute names for it. */
    uint8_t len = 0u;
    const uint8_t *list = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &len);
    const uint8_t variables = op->num_outputs;
    if (list != NULL && (len != 2u * variables || plan->num_weight_blocks != 0u))
        return 0;
    const tigris_subgraph_t *c = &plan->subgraphs[condition];
    const tigris_subgraph_t *b = &plan->subgraphs[body];
    if (c->inputs_count != variables || c->outputs_count != 1u ||
        b->inputs_count != variables || b->outputs_count != variables)
        return 0;
    const tigris_tensor_t *verdict = &plan->tensors[plan->index_pool[c->outputs_off]];
    if (verdict->dtype != 9u || verdict->size_bytes != 1u)
        return 0;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    uint8_t next = 0u;
    for (uint8_t i = 0u; i < variables; i++) {
        const uint16_t weight = list != NULL ? constant_index(plan, op_index, i) : TIGRIS_NO_WEIGHT;
        if (weight == TIGRIS_NO_WEIGHT) {
            if (next >= op->num_inputs || !same_tensor_form(plan, ins[next], outs[i]))
                return 0;
            next++;
        } else if (!weight_size_is(plan, weight, plan->tensors[outs[i]].size_bytes)) {
            return 0;
        }
        if (!same_tensor_form(plan, outs[i], plan->index_pool[c->inputs_off + i]) ||
            !same_tensor_form(plan, outs[i], plan->index_pool[b->inputs_off + i]) ||
            !same_tensor_form(plan, outs[i], plan->index_pool[b->outputs_off + i]))
            return 0;
    }
    return next == op->num_inputs;
}

/* Control flow: graphs are consecutive stage ranges covering every stage, the
 * main graph first and at stage 0; a subgraph's tensors belong to it alone; an
 * If sits alone in a main-graph stage and its operands and results match both
 * branches' inputs and outputs. A plan without the section has no If. */
static tigris_error_t validate_subgraphs(const tigris_plan_t *plan, uint32_t index_count)
{
    const uint16_t num_stages = plan->header->num_stages;
    if (plan->num_subgraphs == 0u) {
        for (uint16_t n = 0; n < plan->header->num_ops; n++)
            if (plan->ops[n].op_type == TIGRIS_OP_IF || plan->ops[n].op_type == TIGRIS_OP_WHILE)
                return TIGRIS_ERR_BAD_OPERATOR;
        return TIGRIS_OK;
    }
    if (plan->num_subgraphs > TIGRIS_MAX_SUBGRAPHS)
        return TIGRIS_ERR_BAD_SECTION;
    uint32_t next = 0u;
    for (uint16_t g = 0; g < plan->num_subgraphs; g++) {
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        if (graph->first_stage != next || (g > 0u && graph->num_stages == 0u) ||
            !elements_ok(graph->inputs_off, graph->inputs_count, index_count) ||
            !elements_ok(graph->outputs_off, graph->outputs_count, index_count))
            return TIGRIS_ERR_BAD_SECTION;
        next += graph->num_stages;
        const uint16_t *ins = plan->index_pool + graph->inputs_off;
        const uint16_t *outs = plan->index_pool + graph->outputs_off;
        for (uint16_t i = 0; i < graph->inputs_count; i++)
            if (ins[i] >= plan->header->num_tensors) return TIGRIS_ERR_BAD_SECTION;
        for (uint16_t i = 0; i < graph->outputs_count; i++)
            if (outs[i] >= plan->header->num_tensors) return TIGRIS_ERR_BAD_SECTION;
    }
    const tigris_subgraph_t *main = &plan->subgraphs[0];
    if (next != num_stages || main->inputs_count != plan->header->num_model_inputs ||
        main->outputs_count != plan->header->num_model_outputs ||
        memcmp(plan->index_pool + main->inputs_off, plan->model_inputs,
               (size_t)main->inputs_count * sizeof(uint16_t)) != 0 ||
        memcmp(plan->index_pool + main->outputs_off, plan->model_outputs,
               (size_t)main->outputs_count * sizeof(uint16_t)) != 0)
        return TIGRIS_ERR_BAD_SECTION;
    /* Each graph's tensors form one index range, the ranges disjoint, so no
     * tensor is shared between graphs; the main graph's range also holds the
     * model interface and the state. */
    uint16_t lo[TIGRIS_MAX_SUBGRAPHS];
    uint16_t hi[TIGRIS_MAX_SUBGRAPHS];
    for (uint16_t g = 0; g < plan->num_subgraphs; g++) {
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        lo[g] = UINT16_MAX;
        hi[g] = 0u;
        for (uint16_t s = graph->first_stage; s < graph->first_stage + graph->num_stages; s++) {
            const tigris_stage_t *stage = &plan->stages[s];
            const uint16_t *ops = tigris_stage_ops(plan, stage);
            for (uint16_t k = 0; k < stage->ops_count; k++) {
                const tigris_op_t *op = &plan->ops[ops[k]];
                const uint16_t *ins = tigris_op_inputs(plan, op);
                const uint16_t *outs = tigris_op_outputs(plan, op);
                for (uint8_t i = 0; i < op->num_inputs; i++) cover(&lo[g], &hi[g], ins[i]);
                for (uint8_t i = 0; i < op->num_outputs; i++) cover(&lo[g], &hi[g], outs[i]);
            }
        }
        for (uint16_t i = 0; i < graph->inputs_count; i++)
            cover(&lo[g], &hi[g], plan->index_pool[graph->inputs_off + i]);
        for (uint16_t i = 0; i < graph->outputs_count; i++)
            cover(&lo[g], &hi[g], plan->index_pool[graph->outputs_off + i]);
    }
    for (uint16_t i = 0; i < plan->num_state; i++) {
        cover(&lo[0], &hi[0], plan->state_entries[i].input);
        if (plan->state_entries[i].output != 0xFFFFu)
            cover(&lo[0], &hi[0], plan->state_entries[i].output);
    }
    for (uint16_t g = 0; g < plan->num_subgraphs; g++) {
        for (uint16_t h = (uint16_t)(g + 1u); h < plan->num_subgraphs; h++)
            if (lo[g] <= hi[g] && lo[h] <= hi[h] && lo[g] <= hi[h] && lo[h] <= hi[g])
                return TIGRIS_ERR_BAD_TENSOR;
        for (uint32_t t = lo[g]; g > 0u && t <= hi[g]; t++)
            if ((plan->tensors[t].flags & (TIGRIS_TENSOR_MODEL_INPUT | TIGRIS_TENSOR_MODEL_OUTPUT |
                                           TIGRIS_TENSOR_STATE)) != 0u)
                return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint16_t s = 0; s < num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        const uint16_t *ops = tigris_stage_ops(plan, stage);
        for (uint16_t k = 0; k < stage->ops_count; k++) {
            const uint16_t op_index = ops[k];
            const tigris_op_t *op = &plan->ops[op_index];
            if (op->op_type != TIGRIS_OP_IF && op->op_type != TIGRIS_OP_WHILE) continue;
            uint8_t len = 0u;
            const uint8_t *branches = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SUBGRAPHS, &len);
            /* A graph runs only graphs placed after it, so control flow
             * cannot recurse. */
            const uint16_t container = graph_of_stage(plan, s);
            if (stage->ops_count != 1u || stage->chain_len != 0u ||
                stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN || !branches || len != 4u ||
                op->num_inputs < 1u)
                return TIGRIS_ERR_BAD_OPERATOR;
            for (uint8_t b = 0u; b < 2u; b++) {
                uint16_t g = (uint16_t)((uint16_t)branches[2u * b] |
                                        (uint16_t)((uint16_t)branches[2u * b + 1u] << 8));
                if (g <= container)
                    return TIGRIS_ERR_BAD_OPERATOR;
            }
            const uint16_t *ins = tigris_op_inputs(plan, op);
            const uint16_t *outs = tigris_op_outputs(plan, op);
            if (op->op_type == TIGRIS_OP_WHILE) {
                if (!while_graphs_valid(plan, op, op_index, branches, container))
                    return TIGRIS_ERR_BAD_OPERATOR;
                continue;
            }
            const tigris_tensor_t *condition = &plan->tensors[ins[0]];
            if (condition->dtype != 9u || condition->size_bytes != 1u)
                return TIGRIS_ERR_BAD_OPERATOR;
            for (uint8_t b = 0u; b < 2u; b++) {
                uint16_t g = (uint16_t)((uint16_t)branches[2u * b] |
                                        (uint16_t)((uint16_t)branches[2u * b + 1u] << 8));
                if (g == 0u || g >= plan->num_subgraphs) return TIGRIS_ERR_BAD_OPERATOR;
                const tigris_subgraph_t *graph = &plan->subgraphs[g];
                if (graph->inputs_count + 1u != op->num_inputs ||
                    graph->outputs_count != op->num_outputs)
                    return TIGRIS_ERR_BAD_OPERATOR;
                for (uint8_t i = 1u; i < op->num_inputs; i++)
                    if (!same_tensor_form(plan, ins[i], plan->index_pool[graph->inputs_off + i - 1u]))
                        return TIGRIS_ERR_BAD_OPERATOR;
                for (uint8_t i = 0u; i < op->num_outputs; i++)
                    if (!same_tensor_form(plan, outs[i], plan->index_pool[graph->outputs_off + i]))
                        return TIGRIS_ERR_BAD_OPERATOR;
            }
        }
    }
    /* Nesting depth: a graph's is one more than the deepest graph any of its
     * control-flow operators runs; graphs after it are already known. */
    uint16_t nesting[TIGRIS_MAX_SUBGRAPHS];
    for (uint16_t k = 0; k < plan->num_subgraphs; k++) {
        const uint16_t g = (uint16_t)(plan->num_subgraphs - 1u - k);
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        nesting[g] = 0u;
        for (uint16_t s = graph->first_stage; s < graph->first_stage + graph->num_stages; s++) {
            const tigris_stage_t *stage = &plan->stages[s];
            if (stage->ops_count != 1u) continue;
            const uint16_t op_index = tigris_stage_ops(plan, stage)[0];
            const uint8_t type = plan->ops[op_index].op_type;
            if (type != TIGRIS_OP_IF && type != TIGRIS_OP_WHILE) continue;
            const uint8_t *runs = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SUBGRAPHS, NULL);
            for (uint8_t b = 0u; b < 2u; b++) {
                uint16_t child = (uint16_t)((uint16_t)runs[2u * b] | (uint16_t)((uint16_t)runs[2u * b + 1u] << 8));
                if ((uint16_t)(nesting[child] + 1u) > nesting[g])
                    nesting[g] = (uint16_t)(nesting[child] + 1u);
            }
        }
    }
    if (nesting[0] > TIGRIS_MAX_SUBGRAPH_DEPTH)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

/* Whether tensor t enters a subgraph; the subgraph section is validated. */
static int subgraph_input(const tigris_plan_t *plan, uint16_t t)
{
    for (uint16_t g = 1; g < plan->num_subgraphs; g++) {
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        for (uint16_t i = 0; i < graph->inputs_count; i++)
            if (plan->index_pool[graph->inputs_off + i] == t) return 1;
    }
    return 0;
}

static int float_section_contains(const uint8_t *section, uint16_t tensor)
{
    const uint8_t mask = (uint8_t)(1u << (tensor % 8u));
    return (section[tensor / 8u] & mask) != 0u;
}

static void float_section_add(uint8_t *section, uint16_t tensor)
{
    const uint8_t mask = (uint8_t)(1u << (tensor % 8u));
    section[tensor / 8u] = (uint8_t)(section[tensor / 8u] | mask);
}

/* A float activation must be reachable from a typed conversion or control flow. */
static void find_float_sections(const tigris_plan_t *plan, uint8_t *section)
{
    memset(section, 0, (TIGRIS_MAX_TENSORS + 7u) / 8u);
    for (uint16_t n = 0u; n < plan->header->num_ops; n++) {
        const tigris_op_t *op = &plan->ops[n];
        if (op->op_type != TIGRIS_OP_DEQUANTIZE && op->op_type != TIGRIS_OP_IF &&
            op->op_type != TIGRIS_OP_WHILE) continue;
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0u; k < op->num_outputs; k++)
            if (plan->tensors[outs[k]].dtype == 1u) float_section_add(section, outs[k]);
    }
    int changed = 1;
    while (changed) {
        changed = 0;
        for (uint16_t n = 0u; n < plan->header->num_ops; n++) {
            const tigris_op_t *op = &plan->ops[n];
            if (!tigris_float_operator(plan, op)) continue;
            const uint16_t *ins = tigris_op_inputs(plan, op);
            int valid = 1;
            for (uint8_t k = 0u; k < op->num_inputs; k++) {
                const tigris_tensor_t *t = &plan->tensors[ins[k]];
                if (t->dtype == 3u || (t->dtype == 1u && !float_section_contains(section, ins[k]) &&
                    (t->flags & TIGRIS_TENSOR_CONSTANT) == 0u)) valid = 0;
            }
            if (!valid) continue;
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t k = 0u; k < op->num_outputs; k++)
                if (plan->tensors[outs[k]].dtype == 1u && !float_section_contains(section, outs[k])) {
                    float_section_add(section, outs[k]);
                    changed = 1;
                }
        }
    }
}

static int kernel_linked(const tigris_plan_t *plan, const tigris_op_t *op)
{
    const tigris_op_contract_t *row = tigris_op_contract(op->op_type);
    int linked = 1;
    if (row != NULL && row->executor == 0u) {
        if (tigris_int32_operator(plan, op) != 0)
            linked = row->linked_i32;
        else if (tigris_plan_data_dtype(plan) == 1u || tigris_float_operator(plan, op) != 0)
            linked = row->linked_f32;
        else
            linked = row->linked_s8;
    }
    return linked;
}

static tigris_error_t validate_contract_equal(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    if (!bool_and_sum_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_gather(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    if (!movement_metadata_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_svdf(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    if (!svdf_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_lstm(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    if (!lstm_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_detection_postprocess(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    if (!detection_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_quantize(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    /* One tensor in and one out of the same shape; the int8 side has
     * a quantization (a producer's record may add per-channel
     * requantization, its scale and zero point stay per tensor), the
     * float32 side has none. A Quantize from int8 requantizes, so
     * both sides are int8 with a quantization each. */
    const tigris_tensor_t *encoded = op->op_type == TIGRIS_OP_QUANTIZE ? output : input;
    const tigris_tensor_t *real = op->op_type == TIGRIS_OP_QUANTIZE ? input : output;
    const int requantize = op->op_type == TIGRIS_OP_QUANTIZE && input->dtype == 3u;
    const tigris_quant_param_t *q = tigris_tensor_quant(plan, encoded);
    const tigris_quant_param_t *source = requantize ? tigris_tensor_quant(plan, input) : NULL;
    if (!op_has_plain_io(op, 1, 1) || !tensor_shapes_equal(plan, input, output) ||
        q == NULL || !isfinite(q->scale) || !(q->scale > 0.0f) ||
        q->zero_point < -128 || q->zero_point > 127 ||
        (requantize ? source == NULL || !isfinite(source->scale) || !(source->scale > 0.0f) ||
                      source->zero_point < -128 || source->zero_point > 127
                    : real->quant_param_idx != TIGRIS_NO_QUANT_PARAM))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_if(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    (void)plan;
    (void)op;
    (void)op_index;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_conv(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->num_inputs != 1 || op->num_outputs != 1 ||
        op->weight_idx == TIGRIS_NO_WEIGHT ||
        input->ndim != 4 || output->ndim != 4)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint32_t input_channels = (uint32_t)in_shape[3];
    uint32_t output_channels = (uint32_t)out_shape[3];
    if (in_shape[0] != out_shape[0] ||
        !output_dim_is_valid(
            in_shape[1], out_shape[1], op->spatial.kernel_h,
            op->spatial.stride_h, op->spatial.pad_top,
            op->spatial.pad_bottom, op->spatial.dilation_h) ||
        !output_dim_is_valid(
            in_shape[2], out_shape[2], op->spatial.kernel_w,
            op->spatial.stride_w, op->spatial.pad_left,
            op->spatial.pad_right, op->spatial.dilation_w))
        return TIGRIS_ERR_BAD_OPERATOR;

    uint64_t weight_elements =
        (uint64_t)op->spatial.kernel_h * op->spatial.kernel_w;
    if (op->op_type == TIGRIS_OP_CONV) {
        if (op->spatial.group != 1)
            return TIGRIS_ERR_BAD_OPERATOR;
        weight_elements *= (uint64_t)input_channels * output_channels;
    } else {
        /* One group per input channel, each with the same number of
         * filters: the channel multiplier. */
        if (op->spatial.group != input_channels || input_channels == 0u ||
            output_channels % input_channels != 0u)
            return TIGRIS_ERR_BAD_OPERATOR;
        weight_elements *= output_channels;
    }
    uint64_t weight_bytes = weight_elements *
        (dtype == 1 ? sizeof(float) : sizeof(int8_t));
    uint64_t bias_bytes = (uint64_t)output_channels * sizeof(int32_t);
    if (!weight_size_is(plan, op->weight_idx, weight_bytes) ||
        !optional_bias_size_is(plan, op->bias_idx, bias_bytes) ||
        !quant_channels_fit(plan, output, output_channels))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_conv1d(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->num_inputs != 1 || op->num_outputs != 1 ||
        op->weight_idx == TIGRIS_NO_WEIGHT ||
        input->ndim != 3 || output->ndim != 3)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint32_t input_channels = (uint32_t)in_shape[2];
    uint32_t output_channels = (uint32_t)out_shape[2];
    if (in_shape[0] != out_shape[0] || op->spatial.group != 1 ||
        !output_dim_is_valid(
            in_shape[1], out_shape[1], op->spatial.kernel_h,
            op->spatial.stride_h, op->spatial.pad_top,
            op->spatial.pad_bottom, op->spatial.dilation_h))
        return TIGRIS_ERR_BAD_OPERATOR;
    uint64_t weight_bytes =
        (uint64_t)output_channels * op->spatial.kernel_h *
        input_channels * (dtype == 1 ? sizeof(float) : sizeof(int8_t));
    uint64_t bias_bytes = (uint64_t)output_channels * sizeof(int32_t);
    if (!weight_size_is(plan, op->weight_idx, weight_bytes) ||
        !optional_bias_size_is(plan, op->bias_idx, bias_bytes) ||
        !quant_channels_fit(plan, output, output_channels))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_fully_conn(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->num_inputs != 1 || op->num_outputs != 1 ||
        op->weight_idx == TIGRIS_NO_WEIGHT || output->ndim == 0)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint32_t output_channels =
        (uint32_t)out_shape[output->ndim - 1u];
    uint32_t input_elements = tensor_elements(input);
    uint32_t output_elements = tensor_elements(output);
    uint32_t batches = 1;
    if (output->ndim == 2) {
        batches = (uint32_t)out_shape[0];
        output_channels = (uint32_t)out_shape[1];
    }
    if (output->ndim > 2 ||
        output_elements != batches * output_channels ||
        batches == 0u || input_elements % batches != 0u)
        return TIGRIS_ERR_BAD_OPERATOR;
    uint32_t inputs_per_batch = input_elements / batches;
    uint64_t weight_bytes =
        (uint64_t)output_channels * inputs_per_batch *
        (dtype == 1 ? sizeof(float) : sizeof(int8_t));
    uint64_t bias_bytes = (uint64_t)output_channels * sizeof(int32_t);
    if (!weight_size_is(plan, op->weight_idx, weight_bytes) ||
        !optional_bias_size_is(plan, op->bias_idx, bias_bytes) ||
        !quant_channels_fit(plan, output, output_channels))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_relu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
    !tensor_shapes_equal(plan, input, output) ||
    (op->op_type == TIGRIS_OP_SOFTMAX && input->ndim == 0))
    return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_reshape(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
    input->size_bytes != output->size_bytes || input->dtype != output->dtype)
    return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_transpose(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
    input->size_bytes != output->size_bytes || input->dtype != output->dtype)
    return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_leaky_relu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
        !tensor_shapes_equal(plan, input, output) ||
        ((op->op_type == TIGRIS_OP_LOG_SOFTMAX ||
          op->op_type == TIGRIS_OP_L2_NORMALIZATION) && input->ndim == 0u) ||
        (op->op_type == TIGRIS_OP_L2_NORMALIZATION && input->ndim > 4u))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (dtype == 3u) {
        if (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            output->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            !quant_input_stated(plan, input) || !quant_channels_fit(plan, output, 1))
            return TIGRIS_ERR_BAD_OPERATOR;
        const tigris_quant_param_t *q = &plan->quant_params[output->quant_param_idx];
        if ((op->op_type == TIGRIS_OP_LOG_SOFTMAX &&
             (q->scale != 0.0625f || q->zero_point != 127)) ||
            (op->op_type == TIGRIS_OP_L2_NORMALIZATION &&
             (q->scale != 0.0078125f || q->zero_point != 0)))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_abs(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
        !tensor_shapes_equal(plan, input, output))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (dtype == 3) {
        if ((op->op_type != TIGRIS_OP_ABS && op->op_type != TIGRIS_OP_RSQRT) ||
            input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            output->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            !quant_input_stated(plan, input) ||
            !quant_channels_fit(plan, output, 1))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_div(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    const tigris_quant_param_t *second_quant = NULL;
    if (op->op_type == TIGRIS_OP_PRELU) {
        uint8_t len = 0;
        const uint8_t *attr = tigris_op_attribute_data(
            plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
        if (op->num_inputs != 1u || !attr || len < 4u || attr[0] != 1u ||
            input->ndim == 0u || input->ndim > 4u ||
            !tensor_shapes_equal(plan, input, output))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    if (!shape_broadcasts_to(tigris_tensor_shape(plan, input), input->ndim,
                             plan, output))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->num_inputs == 1u) {
        if (!constant_operand_is_valid(plan, op, op_index, output, dtype, &second_quant))
            return TIGRIS_ERR_BAD_OPERATOR;
    } else {
        const tigris_tensor_t *other = &plan->tensors[inputs[1]];
        if (!op_has_plain_io(op, 2, 1) ||
            !shape_broadcasts_to(tigris_tensor_shape(plan, other), other->ndim,
                                 plan, output))
            return TIGRIS_ERR_BAD_OPERATOR;
        if (dtype == 3u) {
            const tigris_tensor_t *second = &plan->tensors[inputs[1]];
            if (second->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
                !quant_input_stated(plan, second))
                return TIGRIS_ERR_BAD_OPERATOR;
            second_quant = &plan->quant_params[second->quant_param_idx];
        }
    }
    if (dtype == 3) {
        if ((op->op_type != TIGRIS_OP_DIV && op->op_type != TIGRIS_OP_SQUARED_DIFFERENCE &&
             op->op_type != TIGRIS_OP_MAXIMUM && op->op_type != TIGRIS_OP_MINIMUM &&
             op->op_type != TIGRIS_OP_PRELU) ||
            input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            output->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            !quant_input_stated(plan, input) ||
            !quant_channels_fit(plan, output, 1))
            return TIGRIS_ERR_BAD_OPERATOR;
        if (op->op_type == TIGRIS_OP_MAXIMUM || op->op_type == TIGRIS_OP_MINIMUM) {
            const tigris_quant_param_t *a = &plan->quant_params[input->quant_param_idx];
            const tigris_quant_param_t *b = second_quant;
            const tigris_quant_param_t *y = &plan->quant_params[output->quant_param_idx];
            if (a->scale != b->scale || a->scale != y->scale ||
                a->zero_point != b->zero_point || a->zero_point != y->zero_point)
                return TIGRIS_ERR_BAD_OPERATOR;
        }
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_pad(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const tigris_file_header_t *hdr = plan->header;
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    /* A byte copy into a filled frame: the pads attribute places the
     * input, the optional weight is one element of fill, and int8
     * keeps its quantization. */
    uint8_t pads_len = 0u;
    const uint8_t *pads = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_PADS, &pads_len);
    const uint32_t element = (dtype == 1u) ? (uint32_t)sizeof(float) : 1u;
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        op->bias_idx != TIGRIS_NO_WEIGHT || pads == NULL ||
        input->ndim == 0u || input->ndim != output->ndim ||
        (dtype != 1u && dtype != 3u) ||
        (op->weight_idx != TIGRIS_NO_WEIGHT &&
         (op->weight_idx >= hdr->num_weights || !plan->weight_entries ||
          plan->weight_entries[op->weight_idx].size_bytes != element)) ||
        (dtype == 3u && !same_quantization(tigris_tensor_quant(plan, input),
                                           tigris_tensor_quant(plan, output))))
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    for (uint8_t d = 0; d < input->ndim; d++) {
        int32_t lead;
        int32_t trail;
        memcpy(&lead, pads + 8u * d, sizeof(lead));
        memcpy(&trail, pads + 8u * d + 4u, sizeof(trail));
        if ((int64_t)in_shape[d] + lead + trail != (int64_t)out_shape[d])
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_gelu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    uint8_t len = 0u;
    const uint8_t *mode = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_GELU_APPROXIMATE, &len);
    if (mode == NULL || len != 1u || !op_has_plain_io(op, 1, 1) ||
        input->ndim != output->ndim || input->size_bytes != output->size_bytes)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    for (uint8_t d = 0; d < input->ndim; d++) {
        if (in_shape[d] != out_shape[d]) return TIGRIS_ERR_BAD_OPERATOR;
    }
    if (dtype == 3u) {
        const tigris_quant_param_t *iq = tigris_tensor_quant(plan, input);
        const tigris_quant_param_t *oq = tigris_tensor_quant(plan, output);
        if (iq == NULL || oq == NULL || iq->num_channels != 1u || oq->num_channels != 1u)
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_hardswish(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
    input->ndim != output->ndim ||
    input->size_bytes != output->size_bytes)
    return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_split(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    /* A split needs no arithmetic: every output keeps the input's
     * shape but for the stored axis named in kernel_h, 0 being the
     * outermost, and their extents on that axis sum to the input's.
     * Each part is then one contiguous run per position ahead of the
     * axis. */
    if (op->num_inputs != 1u || op->num_outputs < 2u ||
        input->ndim == 0u || op->spatial.kernel_h >= input->ndim)
        return TIGRIS_ERR_BAD_OPERATOR;
    const uint8_t split_axis = (uint8_t)op->spatial.kernel_h;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    int64_t covered = 0;
    uint32_t total_bytes = 0;
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        const tigris_tensor_t *part = &plan->tensors[outputs[i]];
        if (part->ndim != input->ndim || part->dtype != input->dtype)
            return TIGRIS_ERR_BAD_OPERATOR;
        const int32_t *part_shape = tigris_tensor_shape(plan, part);
        for (uint8_t axis = 0; axis < input->ndim; axis++) {
            if (axis != split_axis && part_shape[axis] != in_shape[axis])
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        if (part_shape[split_axis] <= 0)
            return TIGRIS_ERR_BAD_OPERATOR;
        covered += part_shape[split_axis];
        if (total_bytes > UINT32_MAX - part->size_bytes)
            return TIGRIS_ERR_BAD_OPERATOR;
        total_bytes += part->size_bytes;
        if (!same_quantization(tigris_tensor_quant(plan, part),
                               tigris_tensor_quant(plan, input)))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    if (covered != (int64_t)in_shape[split_axis] ||
        total_bytes != input->size_bytes)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_cumsum(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    uint8_t num_axes = 0;
    const uint8_t *axes = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &num_axes);
    if (!op_has_plain_io(op, 1, 1) || input->ndim != 3u ||
        !tensor_shapes_equal(plan, input, output) || !axes || num_axes != 1u || axes[0] >= 3u)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (dtype == 3u) {
        if (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            output->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            !quant_input_stated(plan, input) || !quant_channels_fit(plan, output, 1))
            return TIGRIS_ERR_BAD_OPERATOR;
        const tigris_quant_param_t *iq = &plan->quant_params[input->quant_param_idx];
        const tigris_quant_param_t *oq = &plan->quant_params[output->quant_param_idx];
        if ((double)iq->scale / (double)oq->scale >= 524288.0)
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_arg_max(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    const int index_output = op->op_type == TIGRIS_OP_ARG_MAX || op->op_type == TIGRIS_OP_ARG_MIN;
    if (dtype == 3u && !index_output && op->op_type != TIGRIS_OP_REDUCE_MEAN) {
        if (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            output->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
            !quant_input_stated(plan, input) || !quant_channels_fit(plan, output, 1))
            return TIGRIS_ERR_BAD_OPERATOR;
        if ((op->op_type == TIGRIS_OP_REDUCE_MAX || op->op_type == TIGRIS_OP_REDUCE_MIN) &&
            !same_quantization(&plan->quant_params[input->quant_param_idx],
                               &plan->quant_params[output->quant_param_idx]))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    /* One activation in, one of its axes collapsed: rank 3 for the
     * reductions, any rank up to TIGRIS_ARG_MAX_RANK for an index. The
     * result keeps that axis at one or drops it, which is the
     * difference between keepdims and not, and every other extent
     * stays put. */
    uint8_t num_axes = 0u;
    const uint8_t *axes = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_AXES, &num_axes);
    const uint8_t rank = input->ndim;
    const int rank_fits = index_output ? (rank >= 1u && rank <= TIGRIS_ARG_MAX_RANK)
                                       : (rank == 3u);
    if (!op_has_plain_io(op, 1, 1) || !rank_fits ||
        axes == NULL || num_axes != 1u || axes[0] >= rank ||
        (output->ndim != rank && (uint8_t)(output->ndim + 1u) != rank))
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint8_t written = 0u;
    for (uint8_t axis = 0; axis < rank; axis++) {
        if (in_shape[axis] <= 0)
            return TIGRIS_ERR_BAD_OPERATOR;
        int32_t want = (axis == axes[0]) ? 1 : in_shape[axis];
        if (axis == axes[0] && output->ndim != rank)
            continue;  /* keepdims=0 drops the axis instead */
        if (written >= output->ndim || out_shape[written] != want)
            return TIGRIS_ERR_BAD_OPERATOR;
        written++;
    }
    if (written != output->ndim)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_layer_norm(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    /* One activation in and out, shape-preserving. Scale is a weight
     * and bias an optional one, each holding one value per position
     * along the axis the kernel normalizes, which is the last stored
     * one. */
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        input->ndim == 0u || input->ndim != output->ndim ||
        input->size_bytes != output->size_bytes ||
        op->weight_idx == TIGRIS_NO_WEIGHT)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *shape = tigris_tensor_shape(plan, input);
    int32_t width = shape[input->ndim - 1u];
    if (width <= 0)
        return TIGRIS_ERR_BAD_OPERATOR;
    uint32_t need = (uint32_t)width * (uint32_t)sizeof(float);
    if (plan->weight_entries[op->weight_idx].size_bytes != need)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->bias_idx != TIGRIS_NO_WEIGHT &&
        plan->weight_entries[op->bias_idx].size_bytes != need)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_matmul(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    /* Both operands are activations, so neither is a weight entry.
     * Axes are the model's own: the trailing pair is the matrix and
     * everything before it batches, with no broadcasting. A plan that
     * needs broadcast batches is rejected rather than guessed at. */
    if (!op_has_plain_io(op, 2, 1) || input->ndim < 2)
        return TIGRIS_ERR_BAD_OPERATOR;
    const tigris_tensor_t *rhs = &plan->tensors[inputs[1]];
    if (rhs->ndim != input->ndim || output->ndim != input->ndim)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *a_shape = tigris_tensor_shape(plan, input);
    const int32_t *b_shape = tigris_tensor_shape(plan, rhs);
    const int32_t *y_shape = tigris_tensor_shape(plan, output);
    uint8_t last = (uint8_t)(input->ndim - 1u);
    for (uint8_t axis = 0; axis + 2u <= last; axis++) {
        if (a_shape[axis] != b_shape[axis] ||
            a_shape[axis] != y_shape[axis])
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    if (a_shape[last] != b_shape[last - 1u] ||
        y_shape[last - 1u] != a_shape[last - 1u] ||
        y_shape[last] != b_shape[last])
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_add(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    /* Each operand has the output's shape or broadcasts to it; int32
     * operands and result go together, without an activation. */
    if ((dtype == 6u) != (output->dtype == 6u) ||
        (op->num_inputs == 2 && (plan->tensors[inputs[1]].dtype == 6u) != (dtype == 6u)) ||
        (dtype == 6u && op->fused_act != TIGRIS_ACT_NONE))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->num_outputs != 1 ||
        (op->num_inputs != 1 && op->num_inputs != 2) ||
        op->bias_idx != TIGRIS_NO_WEIGHT ||
        !shape_broadcasts_to(tigris_tensor_shape(plan, input), input->ndim,
                             plan, output))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->num_inputs == 2) {
        const tigris_tensor_t *second = &plan->tensors[inputs[1]];
        if (op->weight_idx != TIGRIS_NO_WEIGHT ||
            !shape_broadcasts_to(tigris_tensor_shape(plan, second), second->ndim,
                                 plan, output))
            return TIGRIS_ERR_BAD_OPERATOR;
    } else {
        /* Channels are stored innermost, so a per-channel constant
         * repeats on its own without shape metadata. */
        const tigris_quant_param_t *constant_quant;
        if (!constant_operand_is_valid(plan, op, op_index, output, dtype,
                                       &constant_quant) ||
            (dtype == 3u && (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
                             output->quant_param_idx == TIGRIS_NO_QUANT_PARAM)))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_max_pool(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->op_type == TIGRIS_OP_L2_POOL && dtype != 1u)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (!op_has_plain_io(op, 1, 1) ||
        input->ndim != 4 || output->ndim != 4 ||
        (op->spatial.dilation_h > 1) ||
        (op->spatial.dilation_w > 1) ||
        op->spatial.pad_top >= op->spatial.kernel_h ||
        op->spatial.pad_bottom >= op->spatial.kernel_h ||
        op->spatial.pad_left >= op->spatial.kernel_w ||
        op->spatial.pad_right >= op->spatial.kernel_w)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    if (in_shape[0] != out_shape[0] || in_shape[3] != out_shape[3] ||
        !output_dim_is_valid(
            in_shape[1], out_shape[1], op->spatial.kernel_h,
            op->spatial.stride_h, op->spatial.pad_top,
            op->spatial.pad_bottom, 1) ||
        !output_dim_is_valid(
            in_shape[2], out_shape[2], op->spatial.kernel_w,
            op->spatial.stride_w, op->spatial.pad_left,
            op->spatial.pad_right, 1))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_global_max(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    (void)op_index;
    if (!op_has_plain_io(op, 1, 1) ||
        input->ndim != 4 || output->ndim != 4)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    if (out_shape[0] != in_shape[0] || out_shape[1] != 1 ||
        out_shape[2] != 1 || out_shape[3] != in_shape[3])
        return TIGRIS_ERR_BAD_OPERATOR;
    if (input->dtype == 3u &&
        (uint64_t)(uint32_t)in_shape[1] * (uint32_t)in_shape[2] > (uint32_t)INT32_MAX / 255u)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_concat(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const tigris_file_header_t *hdr = plan->header;
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    (void)op_index;
    /* Concatenation runs along one stored axis after the batch axis,
     * named in kernel_h. The inputs have to agree on every other axis
     * and add up along this one. */
    const int32_t *out_shape;
    uint64_t along = 0;
    uint8_t last;
    uint8_t axis;
    uint32_t covered = 0;
    if (op->num_inputs == 0 || op->num_outputs != 1 ||
        op->bias_idx != TIGRIS_NO_WEIGHT ||
        output->ndim < 3u || output->ndim > 4u)
        return TIGRIS_ERR_BAD_OPERATOR;
    last = (uint8_t)(output->ndim - 1u);
    /* Rank 4 has always meant the channel axis when no axis is
     * recorded, so a plan from before the axis was recorded still
     * loads. Rank 3 has to name it. */
    if (output->ndim == 4u && op->spatial.kernel_h == 0)
        axis = last;
    else if (op->spatial.kernel_h >= 1u && op->spatial.kernel_h <= last)
        axis = (uint8_t)op->spatial.kernel_h;
    else
        return TIGRIS_ERR_BAD_OPERATOR;
    out_shape = tigris_tensor_shape(plan, output);
    for (uint8_t i = 0; i < op->num_inputs; i++) {
        const tigris_tensor_t *part = &plan->tensors[inputs[i]];
        const int32_t *shape;
        if (part->ndim != output->ndim)
            return TIGRIS_ERR_BAD_OPERATOR;
        shape = tigris_tensor_shape(plan, part);
        for (uint8_t j = 0; j <= last; j++) {
            if (j != axis && shape[j] != out_shape[j])
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        along += (uint32_t)shape[axis];
        if (covered > output->size_bytes - part->size_bytes)
            return TIGRIS_ERR_BAD_OPERATOR;
        covered += part->size_bytes;
    }
    /* A constant leading part is named as the operator's weight. It
     * has no shape of its own in the plan, so its extent is the one
     * the inputs leave over and its bytes have to be exactly the rest
     * of the output. */
    if (op->weight_idx != TIGRIS_NO_WEIGHT) {
        const tigris_weight_entry_t *constant;
        if (op->weight_idx >= hdr->num_weights || axis != last)
            return TIGRIS_ERR_BAD_OPERATOR;
        constant = &plan->weight_entries[op->weight_idx];
        if (along >= (uint64_t)out_shape[last] ||
            covered > output->size_bytes - constant->size_bytes ||
            covered + constant->size_bytes != output->size_bytes)
            return TIGRIS_ERR_BAD_OPERATOR;
        return TIGRIS_OK;
    }
    if (along != (uint64_t)out_shape[axis] ||
        covered != output->size_bytes)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_conv_transpose(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    (void)op_index;
    /* Untiled gather kernel (kern_conv_transpose / kern_conv_transpose_s8
     * in tigris_kernels.c / tigris_kernels_s8.c). group == 1 and
     * dilation == 1 are the only supported configuration in scope: the
     * gather kernels never read dilation_h/dilation_w, so a plan with
     * dilation != 1 would silently produce wrong output instead of
     * failing loudly. Unlike TIGRIS_OP_CONV, output_padding is
     * absorbed into the emitted output shape at compile time, so the
     * shrink-formula output_dim_is_valid (derived for the forward
     * strided-conv relation) does not apply here; the gather loop
     * bounds itself against the allocated output extent instead.
     * Validate structure only, mirroring the CONV case above minus
     * the output-shape derivation.
     *
     * The gather kernels invert the forward stride relation with
     * num_h % stride_h and num_h / stride_h (same for width). A zero
     * stride divides by zero at execution time, and the kernels do
     * not guard against it themselves, so the loader must reject it
     * here. */
    if (op->num_inputs != 1 || op->num_outputs != 1 ||
        op->weight_idx == TIGRIS_NO_WEIGHT ||
        input->ndim != 4 || output->ndim != 4)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->spatial.group != 1)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->spatial.stride_h == 0 || op->spatial.stride_w == 0)
        return TIGRIS_ERR_BAD_OPERATOR;
    /* dilation 0 encodes "1" (unset) in this codebase, same
     * convention the CONV case's output_dim_is_valid call relies on. */
    uint16_t dilation_h =
        op->spatial.dilation_h ? op->spatial.dilation_h : 1;
    uint16_t dilation_w =
        op->spatial.dilation_w ? op->spatial.dilation_w : 1;
    if (dilation_h != 1 || dilation_w != 1)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    if (in_shape[0] != out_shape[0])
        return TIGRIS_ERR_BAD_OPERATOR;
    /* Weight is OHWI [OC, KH, KW, IC], same element count as the
     * CONV case's [OC, KH, KW, IC]-order weight, just not
     * transposed the way regular Conv weights are; bias is OC
     * int32, same as CONV. */
    uint32_t input_channels = (uint32_t)in_shape[3];
    uint32_t output_channels = (uint32_t)out_shape[3];
    uint64_t weight_elements = (uint64_t)output_channels *
        op->spatial.kernel_h * op->spatial.kernel_w * input_channels;
    uint64_t weight_bytes = weight_elements *
        (dtype == 1 ? sizeof(float) : sizeof(int8_t));
    uint64_t bias_bytes = (uint64_t)output_channels * sizeof(int32_t);
    /* kern_conv_transpose_s8 indexes the output quant param by oc
     * for every oc in [0, output_channels) once num_channels > 1, so
     * a per-channel quant param must cover exactly output_channels
     * entries (or be per-tensor with num_channels == 1); otherwise
     * the kernel reads past the validated quant-data region. Same
     * guard as the CONV case above. */
    if (!weight_size_is(plan, op->weight_idx, weight_bytes) ||
        !optional_bias_size_is(plan, op->bias_idx, bias_bytes) ||
        !quant_channels_fit(plan, output, output_channels))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_resize_linear(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *output = &plan->tensors[outputs[0]];
    const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
    const uint8_t dtype = input->dtype;
    if (op->spatial.kernel_h > (op->op_type == TIGRIS_OP_RESIZE_LINEAR ? 2u : 3u))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->op_type == TIGRIS_OP_RESIZE_LINEAR &&
        ((dtype != 1 && dtype != 3) ||
         (dtype == 3 &&
          (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
           output->quant_param_idx == TIGRIS_NO_QUANT_PARAM))))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (!op_has_plain_io(op, 1, 1) ||
        input->ndim != 4 || output->ndim != 4)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    if (in_shape[0] != out_shape[0] || in_shape[3] != out_shape[3] ||
        (op->spatial.stride_h != 0u && (uint64_t)in_shape[1] * op->spatial.stride_h !=
            (uint32_t)out_shape[1]) ||
        (op->spatial.stride_w != 0u && (uint64_t)in_shape[2] * op->spatial.stride_w !=
            (uint32_t)out_shape[2]))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->op_type == TIGRIS_OP_RESIZE_LINEAR && dtype == 3) {
        const tigris_quant_param_t *iq = tigris_tensor_quant(plan, input);
        const tigris_quant_param_t *oq = tigris_tensor_quant(plan, output);
        if (iq->scale == oq->scale && iq->zero_point == oq->zero_point) {
            float scales[2] = {0.0f, 0.0f};
            const uint8_t *data = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_RESIZE_SCALES, NULL);
            if (data != NULL) memcpy(scales, data, sizeof(scales));
            for (int axis = 0; axis < 2; axis++) {
                int64_t ni = in_shape[axis + 1], no = out_shape[axis + 1];
                if (1024 * ni + no / 2 > INT32_MAX) return TIGRIS_ERR_BAD_OPERATOR;
                if (op->spatial.kernel_h == 2 && no > 1) { ni--; no--; }
                int64_t scale = (1024 * ni + no / 2) / no;
                if (scales[axis] > 0.0f && op->spatial.kernel_h != 2) {
                    double ratio = floor(1024.0 / (double)scales[axis] + 0.5);
                    if (ratio > (double)INT32_MAX) return TIGRIS_ERR_BAD_OPERATOR;
                    scale = (int64_t)ratio;
                }
                int64_t last = (int64_t)(out_shape[axis + 1] - 1) * scale;
                if (op->spatial.kernel_h == 0) last += scale / 2;
                if (last > INT32_MAX) return TIGRIS_ERR_BAD_OPERATOR;
                int64_t pos = last - (op->spatial.kernel_h == 0 ? 512 : 0);
                if (pos > INT32_MAX - 1023 || pos / 1024 >= in_shape[axis + 1])
                    return TIGRIS_ERR_BAD_OPERATOR;
            }
        }
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_operator_semantics(const tigris_plan_t *plan)
{

    const tigris_file_header_t *hdr = plan->header;
    const int legacy = hdr->version < TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES;
    const uint8_t data_dtype = tigris_plan_data_dtype(plan);
    uint8_t section[(TIGRIS_MAX_TENSORS + 7u) / 8u];
    find_float_sections(plan, section);
    for (uint16_t t = 0; t < hdr->num_tensors; t++) {
        if (!tigris_dtype_requires_source(plan->tensors[t].dtype)) continue;
        if ((plan->tensors[t].flags & TIGRIS_TENSOR_CONSTANT) != 0u)
            return TIGRIS_ERR_BAD_TENSOR;
        uint32_t producers = 0u;
        for (uint16_t n = 0; n < hdr->num_ops; n++) {
            const tigris_op_t *producer = &plan->ops[n];
            const uint16_t *outputs = tigris_op_outputs(plan, producer);
            for (uint8_t j = 0; j < producer->num_outputs; j++) {
                if (outputs[j] != t) continue;
                const tigris_op_contract_t *writer = tigris_op_contract(producer->op_type);
                if (writer == NULL ||
                    !dtype_matches_slot(plan->tensors[t].dtype,
                                        writer->dtype.output, data_dtype))
                    return TIGRIS_ERR_BAD_OPERATOR;
                producers++;
            }
        }
        /* A model input or a subgraph input is written before its graph runs. */
        uint32_t expected = ((plan->tensors[t].flags & TIGRIS_TENSOR_MODEL_INPUT) != 0u ||
                             subgraph_input(plan, t)) ? 0u : 1u;
        if (producers != expected) return TIGRIS_ERR_BAD_OPERATOR;
    }
    for (uint16_t op_index = 0; op_index < hdr->num_ops; op_index++) {
        const tigris_op_t *op = &plan->ops[op_index];
        if (legacy && !legacy_semantics_are_builtin(op->op_type))
            continue;
        uint8_t table_len = 0u;
        const uint8_t *table_attr = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_TABLE_SHAPE, &table_len);
        if ((op->num_inputs == 0 && table_attr == NULL) || op->num_outputs == 0 ||
            op->fused_act > TIGRIS_ACT_RELU6 || op->_pad1 != 0)
            return TIGRIS_ERR_BAD_OPERATOR;

        const uint16_t *inputs = tigris_op_inputs(plan, op);
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        const tigris_tensor_t *output = &plan->tensors[outputs[0]];
        const tigris_tensor_t *input = op->num_inputs == 0u ? output : &plan->tensors[inputs[0]];
        uint8_t dtype = input->dtype;
        const tigris_op_contract_t *contract = tigris_op_contract(op->op_type);
        if (contract == NULL) return TIGRIS_ERR_BAD_OPERATOR;
        const dtype_signature_t *signature = &contract->dtype;
        const uint8_t op_dtype = data_dtype == 3u && tigris_float_operator(plan, op) ? 1u : data_dtype;
        if (data_dtype == 3u && op_dtype == 1u) {
            for (uint8_t i = 0u; i < op->num_inputs; i++)
                if (plan->tensors[inputs[i]].dtype == 1u && !float_section_contains(section, inputs[i]) &&
                    (plan->tensors[inputs[i]].flags & TIGRIS_TENSOR_CONSTANT) == 0u)
                    return TIGRIS_ERR_BAD_OPERATOR;
            for (uint8_t i = 0u; i < op->num_outputs; i++)
                if (plan->tensors[outputs[i]].dtype == 1u && !float_section_contains(section, outputs[i]))
                    return TIGRIS_ERR_BAD_OPERATOR;
        }
        uint8_t slot_constant_len = 0u;
        const uint8_t *slot_constant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &slot_constant_len);
        for (uint8_t i = 0; i < op->num_inputs; i++) {
            uint8_t position = i;
            if (slot_constant != NULL && slot_constant[0] <= position) position++;
            else if (op->op_type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE &&
                     op->weight_idx != TIGRIS_NO_WEIGHT && position >= 1u) position++;
            if (table_attr != NULL) position++;
            uint8_t role = signature->inputs[position < 3u ? position : 2u];
            if (!dtype_matches_slot(plan->tensors[inputs[i]].dtype, role, op_dtype))
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        for (uint8_t i = 0; i < op->num_outputs; i++) {
            if (!dtype_matches_slot(plan->tensors[outputs[i]].dtype, signature->output, op_dtype))
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        if (dtype == 3 && op->act_min > op->act_max)
            return TIGRIS_ERR_BAD_OPERATOR;

        /* Add carries an activation because a quantizer writes the residual
         * block's Relu after the sum, on the edge that the sum's own output
         * requantization already covers. */
        int allows_fused_activation = (contract->flags & TIGRIS_CONTRACT_FUSED) != 0u ||
            ((contract->flags & TIGRIS_CONTRACT_FUSED_FLOAT) != 0u && dtype == 1u);
        if (!allows_fused_activation && op->fused_act != TIGRIS_ACT_NONE)
            return TIGRIS_ERR_BAD_OPERATOR;

        if (op->num_inputs < contract->min_inputs ||
            op->num_inputs > contract->max_inputs ||
            op->num_outputs < contract->min_outputs ||
            op->num_outputs > contract->max_outputs ||
            contract->validate == NULL)
            return TIGRIS_ERR_BAD_OPERATOR;
        if (contract->validate(plan, op, op_index) != TIGRIS_OK)
            return TIGRIS_ERR_BAD_OPERATOR;

    }

    return TIGRIS_OK;
}

/* Public API */

void tigris_build_limits(tigris_plan_limits_t *out)
{
    if (!out)
        return;
    out->tensors = (uint16_t)TIGRIS_MAX_TENSORS;
    out->stage_inputs = (uint16_t)TIGRIS_MAX_STAGE_INPUTS;
    out->stage_outputs = (uint16_t)TIGRIS_MAX_STAGE_OUTPUTS;
    out->chain_stages = (uint16_t)TIGRIS_MAX_CHAIN_STAGES;
    out->spatial_ops_per_stage = (uint16_t)TIGRIS_MAX_SPATIAL_OPS_PER_STAGE;
}

/* Raise a reported requirement to cover one more observation. */
static void note_requirement(uint16_t *slot, uint32_t seen)
{
    if (slot != NULL && seen > (uint32_t)*slot)
        *slot = (seen > 65535u) ? 65535u : (uint16_t)seen;
}

tigris_error_t tigris_plan_load(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan)
{
    return tigris_plan_load_ex(buf, buf_len, out_plan, NULL);
}

typedef struct {
    const uint8_t * buf;
    uint32_t buf_len;
    tigris_plan_t * out_plan;
    tigris_plan_limits_t * out_required;
    tigris_plan_limits_t required;
    tigris_plan_t candidate;
    const tigris_file_header_t * hdr;
    uint32_t sec_off;
    uint32_t section_offsets[TIGRIS_SEC_MAX];
    uint32_t section_ends[TIGRIS_SEC_MAX];
    uint8_t section_seen[TIGRIS_SEC_MAX];
    int found_sentinel;
    uint32_t sec_cursor;
    uint32_t previous_type;
    uint32_t previous_offset;
    uint32_t dir_end;
    uint32_t index_bytes;
    uint32_t shape_bytes;
    uint32_t strings_len;
    uint32_t index_count;
    uint32_t shape_count;
    uint32_t model_io_count;
    uint16_t next_scheduled_op;
} tigris_load_state_t;

static tigris_error_t load_header_and_directory(tigris_load_state_t *ctx)
{
    /* A plan that exceeds this build is refused, but the refusal says which
     * build would run it, so the requirement is gathered as the tables are
     * walked rather than returned from the first breach. */

    memset(&ctx->required, 0, sizeof(ctx->required));
    if (ctx->out_required)
        memset(ctx->out_required, 0, sizeof(*ctx->out_required));

    if (!ctx->out_plan)
        return TIGRIS_ERR_NULL;

    /* Never expose pointers from a plan that later fails validation. */
    memset(ctx->out_plan, 0, sizeof(*ctx->out_plan));
    if (!ctx->buf)
        return TIGRIS_ERR_NULL;

    memset(&ctx->candidate, 0, sizeof(ctx->candidate));

    /* Verify little-endian platform (plan format assumes LE) */
    {
        const uint32_t endian_test = 1;
        if (*(const uint8_t *)&endian_test != 1)
            return TIGRIS_ERR_ENDIAN;
    }

    /* Header validation */

    if (ctx->buf_len < sizeof(tigris_file_header_t))
        return TIGRIS_ERR_TOO_SMALL;

    ctx->hdr = (const tigris_file_header_t *)ctx->buf;

    if (memcmp(ctx->hdr->magic, TIGRIS_MAGIC_BYTES, 4) != 0)
        return TIGRIS_ERR_BAD_MAGIC;

    if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_MIN ||
        ctx->hdr->version > TIGRIS_SCHEMA_VERSION)
        return TIGRIS_ERR_BAD_VERSION;

    if (ctx->hdr->file_size != ctx->buf_len)
        return TIGRIS_ERR_BAD_SIZE;

    ctx->candidate.header = ctx->hdr;

    /* Section directory */

    ctx->sec_off = ctx->hdr->section_dir_off;
    if (!bounds_ok(ctx->sec_off, sizeof(tigris_section_entry_t), ctx->buf_len))
        return TIGRIS_ERR_BAD_SECTION;

    /* Track which sections we found via offsets into buf (0 = not found) */



    memset(ctx->section_offsets, 0, sizeof(ctx->section_offsets));
    memset(ctx->section_ends, 0, sizeof(ctx->section_ends));
    memset(ctx->section_seen, 0, sizeof(ctx->section_seen));

    ctx->found_sentinel = 0;
    ctx->sec_cursor = ctx->sec_off;
    ctx->previous_type = 0;
    ctx->previous_offset = 0;
    while (bounds_ok(ctx->sec_cursor, sizeof(tigris_section_entry_t), ctx->buf_len)) {
        const tigris_section_entry_t *sec =
            (const tigris_section_entry_t *)(ctx->buf + ctx->sec_cursor);
        if (sec->type == 0)
        {
            ctx->found_sentinel = 1;
            break;
        }

        if (sec->type >= TIGRIS_SEC_MAX)
            return TIGRIS_ERR_BAD_SECTION;

        if (ctx->section_seen[sec->type])
            return TIGRIS_ERR_BAD_SECTION;

        /* The writer emits an ordered directory and non-overlapping section
         * starts. Empty sections may share an offset with the next section. */
        if (sec->type <= ctx->previous_type || sec->offset < ctx->previous_offset ||
            sec->offset > ctx->buf_len)
            return TIGRIS_ERR_BAD_SECTION;

        ctx->section_seen[sec->type] = 1;
        ctx->section_offsets[sec->type] = sec->offset;
        ctx->previous_type = sec->type;
        ctx->previous_offset = sec->offset;
        ctx->sec_cursor += sizeof(tigris_section_entry_t);
    }

    if (!ctx->found_sentinel)
        return TIGRIS_ERR_BAD_SECTION;

    /* No section may overlap the parsed directory. Derive each section's
     * exclusive end from the next directory entry, including empty sections. */
    ctx->dir_end = ctx->sec_cursor + sizeof(tigris_section_entry_t);
    for (uint32_t i = 1; i < TIGRIS_SEC_MAX; i++) {
        if (!ctx->section_seen[i])
            continue;
        if (ctx->section_offsets[i] < ctx->dir_end)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t end = ctx->buf_len;
        for (uint32_t j = i + 1; j < TIGRIS_SEC_MAX; j++) {
            if (ctx->section_seen[j] &&
                ctx->section_offsets[j] < end)
                end = ctx->section_offsets[j];
        }
        ctx->section_ends[i] = end;
    }

    /* Required sections */

    if (!ctx->section_offsets[TIGRIS_SEC_TENSORS])     return TIGRIS_ERR_MISSING_SEC;
    if (!ctx->section_offsets[TIGRIS_SEC_OPS])         return TIGRIS_ERR_MISSING_SEC;
    if (!ctx->section_offsets[TIGRIS_SEC_INDEX_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!ctx->section_offsets[TIGRIS_SEC_SHAPE_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!ctx->section_offsets[TIGRIS_SEC_STRINGS])     return TIGRIS_ERR_MISSING_SEC;
    if (ctx->hdr->num_stages && !ctx->section_offsets[TIGRIS_SEC_STAGES])
        return TIGRIS_ERR_MISSING_SEC;
    if (ctx->hdr->num_tile_plans && !ctx->section_offsets[TIGRIS_SEC_TILE_PLANS])
        return TIGRIS_ERR_MISSING_SEC;
    if (ctx->hdr->num_weights && !ctx->section_offsets[TIGRIS_SEC_WEIGHTS])
        return TIGRIS_ERR_MISSING_SEC;
    if (ctx->hdr->num_quant_params && !ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS])
        return TIGRIS_ERR_MISSING_SEC;
    if (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK &&
        !ctx->section_offsets[TIGRIS_SEC_MEMORY])
        return TIGRIS_ERR_MISSING_SEC;

    /* The on-disk tables are otherwise byte-addressable, but these pools and
     * the unpacked state and subgraph tables are dereferenced through
     * uint16_t/int32_t pointers below.  Reject malformed offsets before
     * exposing an unaligned pointer to a target that faults on it (or to UBSan
     * during loader fuzzing). */
    if ((ctx->section_offsets[TIGRIS_SEC_INDEX_POOL] % sizeof(uint16_t)) != 0 &&
        (ctx->hdr->num_ops || ctx->hdr->num_stages || ctx->hdr->num_model_inputs ||
         ctx->hdr->num_model_outputs))
        return TIGRIS_ERR_BAD_SECTION;
    if ((ctx->section_offsets[TIGRIS_SEC_SHAPE_POOL] % sizeof(int32_t)) != 0 &&
        ctx->hdr->num_tensors)
        return TIGRIS_ERR_BAD_SECTION;
    if ((ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS] % sizeof(int32_t)) != 0 &&
        ctx->hdr->num_quant_params)
        return TIGRIS_ERR_BAD_SECTION;
    if ((ctx->section_offsets[TIGRIS_SEC_STATE] % sizeof(uint32_t)) != 0 ||
        (ctx->section_offsets[TIGRIS_SEC_SUBGRAPHS] % sizeof(uint16_t)) != 0)
        return TIGRIS_ERR_BAD_SECTION;

    /* Set pointers */

    /* Tensors */
    {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_TENSORS];
        uint32_t need = (uint32_t)ctx->hdr->num_tensors * sizeof(tigris_tensor_t);
        if (!bounds_ok(off, need, ctx->section_ends[TIGRIS_SEC_TENSORS]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.tensors = (const tigris_tensor_t *)(ctx->buf + off);
    }

    /* Ops */
    {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_OPS];
        uint32_t need = (uint32_t)ctx->hdr->num_ops * sizeof(tigris_op_t);
        if (!bounds_ok(off, need, ctx->section_ends[TIGRIS_SEC_OPS]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.ops = (const tigris_op_t *)(ctx->buf + off);
    }
    return TIGRIS_OK;
}

static tigris_error_t locate_optional_sections(tigris_load_state_t *ctx)
{
    /* Stages (optional - 0 stages is valid for un-partitioned graphs) */
    if (ctx->section_offsets[TIGRIS_SEC_STAGES] && ctx->hdr->num_stages > 0) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_STAGES];
        uint32_t need = (uint32_t)ctx->hdr->num_stages * sizeof(tigris_stage_t);
        if (!bounds_ok(off, need, ctx->section_ends[TIGRIS_SEC_STAGES]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.stages = (const tigris_stage_t *)(ctx->buf + off);
    }

    /* Tile plans (optional) */
    if (ctx->section_offsets[TIGRIS_SEC_TILE_PLANS] && ctx->hdr->num_tile_plans > 0) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_TILE_PLANS];
        uint32_t stride = ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS
            ? sizeof(tigris_tile_plan_v11_t)
            : ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_STATE
            ? sizeof(tigris_tile_plan_v10_t) : sizeof(tigris_tile_plan_t);
        uint32_t need = (uint32_t)ctx->hdr->num_tile_plans * stride;
        if (!bounds_ok(off, need, ctx->section_ends[TIGRIS_SEC_TILE_PLANS]))
            return TIGRIS_ERR_BAD_SECTION;
        if (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS)
            ctx->candidate.tile_plans_v11 = (const tigris_tile_plan_v11_t *)(ctx->buf + off);
        else if (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_STATE)
            ctx->candidate.tile_plans_v10 = (const tigris_tile_plan_v10_t *)(ctx->buf + off);
        else
            ctx->candidate.tile_plans = (const tigris_tile_plan_t *)(ctx->buf + off);
        for (uint16_t i = 0; i < ctx->hdr->num_tile_plans; i++) {
            const tigris_tile_plan_v10_t *wide = ctx->candidate.tile_plans_v11
                ? &ctx->candidate.tile_plans_v11[i].choices : ctx->candidate.tile_plans_v10
                ? &ctx->candidate.tile_plans_v10[i] : NULL;
            uint8_t tileable = wide ? wide->tileable : ctx->candidate.tile_plans[i].tileable;
            uint8_t axis = wide ? wide->axis : ctx->candidate.tile_plans[i].axis;
            if (tileable > 1u)
                return TIGRIS_ERR_BAD_SECTION;
            if (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS &&
                ((tileable && axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH && axis != TIGRIS_TILE_AXIS_HW) ||
                 (!tileable && axis != TIGRIS_TILE_AXIS_NONE)))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    /* Weights (optional) */
    if (ctx->section_offsets[TIGRIS_SEC_WEIGHTS] && ctx->hdr->num_weights > 0) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)ctx->hdr->num_weights * sizeof(tigris_weight_entry_t);
        if (!bounds_ok(off, entries_size, ctx->section_ends[TIGRIS_SEC_WEIGHTS]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.weight_entries = (const tigris_weight_entry_t *)(ctx->buf + off);
        /* weight_blob set below only if no compressed blocks */
    }

    /* Weight blocks - compressed per-stage weight data (optional) */
    if (ctx->section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS]) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        if (!bounds_ok(off, 4, ctx->section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t num_blocks, compression;
        memcpy(&num_blocks, ctx->buf + off, 2);
        memcpy(&compression, ctx->buf + off + 2, 2);
        ctx->candidate.num_weight_blocks = num_blocks;
        ctx->candidate.weight_compression = compression;
        uint32_t entries_size = (uint32_t)num_blocks * sizeof(tigris_weight_block_t);
        if (!bounds_ok(off + 4, entries_size, ctx->section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.weight_blocks = (const tigris_weight_block_t *)(ctx->buf + off + 4);
        ctx->candidate.weight_blocks_data = ctx->buf + off + 4 + entries_size;
    }

    /* Per-operator attributes (optional, schema v4+). Compare against the
     * feature's introduction version, not the latest schema: otherwise a
     * future schema bump would accidentally make valid v4 plans fail. */
    if (ctx->section_offsets[TIGRIS_SEC_OP_ATTRIBUTES]) {
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_OP_ATTRIBUTES];
        if (!bounds_ok(off, 4, ctx->section_ends[TIGRIS_SEC_OP_ATTRIBUTES]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t count;
        memcpy(&count, ctx->buf + off, sizeof(count));
        uint32_t entries_size = (uint32_t)count * sizeof(tigris_op_attribute_t);
        if (count == 0 ||
            !bounds_ok(off + 4u, entries_size,
                       ctx->section_ends[TIGRIS_SEC_OP_ATTRIBUTES]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.num_op_attributes = count;
        ctx->candidate.op_attributes =
            (const tigris_op_attribute_t *)(ctx->buf + off + 4u);
        ctx->candidate.op_attribute_data = ctx->buf + off + 4u + entries_size;
    }

    /* Variables kept across invocations (optional, schema v10+): their
     * entries and initial values, checked against the tensors below. */
    if (ctx->section_offsets[TIGRIS_SEC_STATE]) {
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_STATE)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_STATE];
        uint32_t end = ctx->section_ends[TIGRIS_SEC_STATE];
        if (!bounds_ok(off, 8u, end))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t count;
        uint16_t reserved;
        memcpy(&count, ctx->buf + off, sizeof(count));
        memcpy(&reserved, ctx->buf + off + 2u, sizeof(reserved));
        memcpy(&ctx->candidate.state_bytes, ctx->buf + off + 4u, sizeof(ctx->candidate.state_bytes));
        uint32_t entries_size = (uint32_t)count * sizeof(tigris_state_entry_t);
        if (count == 0u || reserved != 0u || !bounds_ok(off + 8u, entries_size, end))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.num_state = count;
        ctx->candidate.state_entries = (const tigris_state_entry_t *)(ctx->buf + off + 8u);
        ctx->candidate.state_section = ctx->buf + off;
        for (uint16_t i = 0; i < count; i++) {
            const tigris_state_entry_t *entry = &ctx->candidate.state_entries[i];
            if (entry->bytes == 0u || (entry->offset % 4u) != 0u ||
                entry->offset > ctx->candidate.state_bytes ||
                entry->bytes > ctx->candidate.state_bytes - entry->offset ||
                !bounds_ok(off + entry->initial, entry->bytes, end))
                return TIGRIS_ERR_BAD_SECTION;
            for (uint16_t j = 0; j < i; j++) {
                const tigris_state_entry_t *other = &ctx->candidate.state_entries[j];
                if (entry->offset < other->offset + other->bytes &&
                    other->offset < entry->offset + entry->bytes)
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
    }

    /* The plan's fast-arena figure (schema v11+). */
    if (ctx->section_offsets[TIGRIS_SEC_MEMORY]) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_MEMORY];
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK ||
            !bounds_ok(off, sizeof(tigris_memory_t), ctx->section_ends[TIGRIS_SEC_MEMORY]))
            return TIGRIS_ERR_BAD_SECTION;
        tigris_memory_t memory;
        memcpy(&memory, ctx->buf + off, sizeof(memory));
        ctx->candidate.scheduled_peak = memory.scheduled_peak;
    }

    /* Set weight_blob for XIP only when no compressed blocks */
    if (ctx->candidate.weight_entries && !ctx->candidate.weight_blocks) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)ctx->hdr->num_weights * sizeof(tigris_weight_entry_t);
        ctx->candidate.weight_blob = ctx->buf + off + entries_size;
    }

    /* Quant params (optional) */
    if (ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        /* First 4 bytes: uint16_t num_quant_params + quant-data length (v2)
         * or page count (v3). */
        if (!bounds_ok(off, 4, ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t nqp, qd_field;
        memcpy(&nqp, ctx->buf + off, 2);
        memcpy(&qd_field, ctx->buf + off + 2, 2);
        ctx->candidate.num_quant_params = nqp;
        /* The header states a count and so does the section. Everything below
         * bounds a quant_param_idx against one of them and indexes the array
         * sized by the other, so they have to be reconciled here rather than
         * after the tables that read them. A header claiming more than the
         * section holds read past the plan. */
        if (nqp != ctx->hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t entries_size = (uint32_t)nqp * sizeof(tigris_quant_param_t);
        uint32_t data_size;
        if (ctx->hdr->version == TIGRIS_SCHEMA_VERSION_V2)
            data_size = (uint32_t)qd_field * sizeof(int32_t);
        else {
            if (ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] < off + 4u + entries_size)
                return TIGRIS_ERR_BAD_SECTION;
            data_size = ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] - off - 4u - entries_size;
            if ((data_size % sizeof(int32_t)) != 0)
                return TIGRIS_ERR_BAD_SECTION;
            uint32_t physical_pages = (data_size / sizeof(int32_t) +
                                       TIGRIS_QUANT_PAGE_ELEMS - 1u) /
                                      TIGRIS_QUANT_PAGE_ELEMS;
            /* The next aligned section can contribute up to 15 padding bytes
             * to this section span. It may therefore extend an otherwise
             * page-aligned pool into one physical page without changing the
             * semantic v3 page count recorded by the compiler. */
            if (qd_field == 0 || physical_pages < qd_field ||
                physical_pages > (uint32_t)qd_field + 1u)
                return TIGRIS_ERR_BAD_SECTION;
        }
        if (!bounds_ok(off + 4, entries_size + data_size,
                       ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.quant_params = (const tigris_quant_param_t *)(ctx->buf + off + 4);
        if (data_size > 0)
            ctx->candidate.quant_data = (const int32_t *)(ctx->buf + off + 4 + entries_size);
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_tables_and_ops(tigris_load_state_t *ctx)
{
    /* Index pool */
    {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_INDEX_POOL];
        ctx->candidate.index_pool = (const uint16_t *)(ctx->buf + off);
    }

    /* Shape pool */
    {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_SHAPE_POOL];
        ctx->candidate.shape_pool = (const int32_t *)(ctx->buf + off);
    }

    /* String table */
    {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_STRINGS];
        ctx->candidate.strings = (const char *)(ctx->buf + off);
    }

    /* Validate every table span and cross-reference before exposing the plan
     * to the executor. Section boundaries above ensure these counts cannot
     * borrow bytes from the next table. */
    ctx->index_bytes = ctx->section_ends[TIGRIS_SEC_INDEX_POOL] -
                           ctx->section_offsets[TIGRIS_SEC_INDEX_POOL];
    ctx->shape_bytes = ctx->section_ends[TIGRIS_SEC_SHAPE_POOL] -
                           ctx->section_offsets[TIGRIS_SEC_SHAPE_POOL];
    ctx->strings_len = ctx->section_ends[TIGRIS_SEC_STRINGS] -
                           ctx->section_offsets[TIGRIS_SEC_STRINGS];
    if (((ctx->index_bytes % sizeof(uint16_t)) != 0 &&
         (ctx->hdr->num_ops || ctx->hdr->num_stages || ctx->hdr->num_model_inputs ||
          ctx->hdr->num_model_outputs)) ||
        ((ctx->shape_bytes % sizeof(int32_t)) != 0 && ctx->hdr->num_tensors) ||
        ctx->strings_len == 0)
        return TIGRIS_ERR_BAD_SECTION;
    ctx->index_count = ctx->index_bytes / sizeof(uint16_t);
    ctx->shape_count = ctx->shape_bytes / sizeof(int32_t);

    if (!string_ok(ctx->candidate.strings, ctx->strings_len, ctx->hdr->model_name_str))
        return TIGRIS_ERR_BAD_SECTION;

    ctx->model_io_count = (uint32_t)ctx->hdr->num_model_inputs +
                              (uint32_t)ctx->hdr->num_model_outputs;
    if (!elements_ok(ctx->hdr->model_io_off, ctx->model_io_count, ctx->index_count))
        return TIGRIS_ERR_BAD_SECTION;
    ctx->candidate.model_inputs  = ctx->candidate.index_pool + ctx->hdr->model_io_off;
    ctx->candidate.model_outputs = ctx->candidate.model_inputs + ctx->hdr->num_model_inputs;
    for (uint32_t i = 0; i < ctx->model_io_count; i++) {
        if (ctx->candidate.model_inputs[i] >= ctx->hdr->num_tensors)
            return TIGRIS_ERR_BAD_SECTION;
    }

    note_requirement(&ctx->required.tensors, ctx->hdr->num_tensors);
    if (ctx->hdr->num_tensors > TIGRIS_MAX_TENSORS) {
        if (ctx->out_required)
            *ctx->out_required = ctx->required;
        return TIGRIS_ERR_PLAN_LIMITS;
    }

    for (uint16_t i = 0; i < ctx->hdr->num_tensors; i++) {
        const tigris_tensor_t *tensor = &ctx->candidate.tensors[i];
        if (!string_ok(ctx->candidate.strings, ctx->strings_len, tensor->name_str) ||
            !elements_ok(tensor->shape_off, tensor->ndim, ctx->shape_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM &&
            tensor->quant_param_idx >= ctx->hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
        if ((tensor->flags & ~(TIGRIS_TENSOR_CONSTANT |
                               TIGRIS_TENSOR_MODEL_INPUT |
                               TIGRIS_TENSOR_MODEL_OUTPUT |
                               TIGRIS_TENSOR_LINEAR |
                               TIGRIS_TENSOR_STATE)) != 0 ||
            ((tensor->flags & TIGRIS_TENSOR_STATE) != 0u &&
             (tensor->flags & (TIGRIS_TENSOR_CONSTANT | TIGRIS_TENSOR_MODEL_INPUT |
                               TIGRIS_TENSOR_MODEL_OUTPUT)) != 0u) ||
            ((tensor->dtype == 1 || tensor->dtype == 6 || tensor->dtype == 9) &&
             tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM))
            return TIGRIS_ERR_BAD_TENSOR;
        /* The byte was reserved and written as zero before the interface dtype
         * existed, so an older plan reads as "no declared interface". A
         * declared interface belongs to a boundary tensor and says something
         * only when it differs from what the plan stores. */
        if (tensor->iface_dtype != 0) {
            if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_INTERFACE_DTYPE ||
                (tensor->flags & (TIGRIS_TENSOR_MODEL_INPUT |
                                  TIGRIS_TENSOR_MODEL_OUTPUT)) == 0 ||
                tensor->iface_dtype == tensor->dtype)
                return TIGRIS_ERR_BAD_TENSOR;
            /* The conversions this runtime performs are from a float or a
             * uint8 interface to a quantized int8 plan tensor, and
             * tigris_iface.c refuses anything else. Saying so here refuses
             * the plan when it is loaded rather than when a caller first
             * writes an input. */
            if (!((tensor->iface_dtype == 7u && tensor->dtype == 6u) ||
                  ((tensor->iface_dtype == 1u || tensor->iface_dtype == 2u) &&
                   tensor->dtype == 3u && tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM)))
                return TIGRIS_ERR_BAD_TENSOR;
        }
        if (tensor->iface_dtype == 7u && tensor->size_bytes > UINT32_MAX / 2u)
            return TIGRIS_ERR_BAD_TENSOR;
        uint32_t elements = 1;
        for (uint8_t dim = 0; dim < tensor->ndim; dim++) {
            int32_t value = ctx->candidate.shape_pool[tensor->shape_off + dim];
            if (value <= 0 || (uint32_t)value > UINT32_MAX / elements)
                return TIGRIS_ERR_BAD_TENSOR;
            elements *= (uint32_t)value;
        }
        uint32_t element_size = tigris_dtype_size(tensor->dtype);
        if (tensor->dtype != 1u && tensor->dtype != 3u && tensor->dtype != 6u && tensor->dtype != 9u &&
            !(tensor->dtype == 5u && (tensor->flags & TIGRIS_TENSOR_STATE) != 0u))
            return TIGRIS_ERR_BAD_TENSOR;
        if (elements > UINT32_MAX / element_size ||
            tensor->size_bytes != elements * element_size)
            return TIGRIS_ERR_BAD_TENSOR;
        if (tensor->dtype == 3 &&
            tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM) {
            const tigris_quant_param_t *qp =
                &ctx->candidate.quant_params[tensor->quant_param_idx];
            if (!isfinite(qp->scale) || qp->scale <= 0.0f ||
                qp->zero_point < -128 || qp->zero_point > 127)
                return TIGRIS_ERR_BAD_TENSOR;
        }
    }

    for (uint8_t i = 0; i < ctx->hdr->num_model_inputs; i++) {
        if (!(ctx->candidate.tensors[ctx->candidate.model_inputs[i]].flags &
              TIGRIS_TENSOR_MODEL_INPUT))
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint8_t i = 0; i < ctx->hdr->num_model_outputs; i++) {
        if (!(ctx->candidate.tensors[ctx->candidate.model_outputs[i]].flags &
              TIGRIS_TENSOR_MODEL_OUTPUT))
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint16_t tensor_idx = 0; tensor_idx < ctx->hdr->num_tensors; tensor_idx++) {
        const tigris_tensor_t *tensor = &ctx->candidate.tensors[tensor_idx];
        int found_input = 0;
        int found_output = 0;
        for (uint8_t i = 0; i < ctx->hdr->num_model_inputs; i++)
            found_input |= ctx->candidate.model_inputs[i] == tensor_idx;
        for (uint8_t i = 0; i < ctx->hdr->num_model_outputs; i++)
            found_output |= ctx->candidate.model_outputs[i] == tensor_idx;
        if (((tensor->flags & TIGRIS_TENSOR_MODEL_INPUT) != 0) != found_input ||
            ((tensor->flags & TIGRIS_TENSOR_MODEL_OUTPUT) != 0) != found_output)
            return TIGRIS_ERR_BAD_TENSOR;
    }

    if (ctx->section_offsets[TIGRIS_SEC_SUBGRAPHS]) {
        uint32_t off = ctx->section_offsets[TIGRIS_SEC_SUBGRAPHS];
        uint32_t end = ctx->section_ends[TIGRIS_SEC_SUBGRAPHS];
        uint16_t count;
        uint16_t reserved;
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_STATE || !bounds_ok(off, 4u, end))
            return TIGRIS_ERR_BAD_SECTION;
        memcpy(&count, ctx->buf + off, sizeof(count));
        memcpy(&reserved, ctx->buf + off + 2u, sizeof(reserved));
        if (count < 2u || reserved != 0u ||
            !bounds_ok(off + 4u, (uint32_t)count * sizeof(tigris_subgraph_t), end))
            return TIGRIS_ERR_BAD_SECTION;
        ctx->candidate.subgraphs = (const tigris_subgraph_t *)(ctx->buf + off + 4u);
        ctx->candidate.num_subgraphs = count;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_attribute_low(
    tigris_load_state_t *ctx, const tigris_op_attribute_t *attr,
    const tigris_op_t *op, const tigris_tensor_t *input,
    const tigris_tensor_t *output)
{
    const tigris_file_header_t *hdr = ctx->hdr;
    (void)hdr; (void)op; (void)input; (void)output;
    switch (attr->type) {
    case TIGRIS_OP_ATTR_TRANSPOSE_PERM:
    {
        if (op->num_inputs != 1u ||
                            attr->data_len != input->ndim ||
                            output->ndim != input->ndim ||
                            output->size_bytes != input->size_bytes)
                            return TIGRIS_ERR_BAD_SECTION;
                        const uint8_t *perm =
                            ctx->candidate.op_attribute_data + attr->data_offset;
                        const int32_t *in_shape =
                            tigris_tensor_shape(&ctx->candidate, input);
                        const int32_t *out_shape =
                            tigris_tensor_shape(&ctx->candidate, output);
                        for (uint8_t axis = 0; axis < attr->data_len; axis++) {
                            if (perm[axis] >= attr->data_len)
                                return TIGRIS_ERR_BAD_SECTION;
                            for (uint8_t prior = 0; prior < axis; prior++) {
                                if (perm[prior] == perm[axis])
                                    return TIGRIS_ERR_BAD_SECTION;
                            }
                            if (out_shape[axis] != in_shape[perm[axis]])
                                return TIGRIS_ERR_BAD_SECTION;
                        }
        break;
    }
    case TIGRIS_OP_ATTR_EPSILON:
    case TIGRIS_OP_ATTR_ALPHA:
    {
        float value;
                        if (attr->data_len != sizeof(value))
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(&value,
                               ctx->candidate.op_attribute_data + attr->data_offset,
                               sizeof(value));
                        /* A variance floor must be positive; a negative slope must be
                         * finite. Both would otherwise reach a kernel as a divisor or
                         * a multiplier with no defined result. */
                        if (!isfinite(value) ||
                            (attr->type == TIGRIS_OP_ATTR_EPSILON &&
                             (value < 0.0f || (value == 0.0f && op->op_type != TIGRIS_OP_L2_NORMALIZATION))))
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_RESIZE_SCALES:
    {
        float scales[2];
                        if (attr->data_len != sizeof(scales) || input->ndim != 4u || output->ndim != 4u)
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(scales, ctx->candidate.op_attribute_data + attr->data_offset, sizeof(scales));
                        const int32_t *in_shape = tigris_tensor_shape(&ctx->candidate, input);
                        const int32_t *out_shape = tigris_tensor_shape(&ctx->candidate, output);
                        for (int axis = 0; axis < 2; axis++) {
                            if (!isfinite(scales[axis]) || scales[axis] <= 0.0f ||
                                floorf((float)in_shape[axis + 1] * scales[axis]) !=
                                    (float)out_shape[axis + 1])
                                return TIGRIS_ERR_BAD_SECTION;
                        }
        break;
    }
    case TIGRIS_OP_ATTR_CLIP_BOUNDS:
    {
        float bounds[2];
                        if (attr->data_len != sizeof(bounds))
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(bounds,
                               ctx->candidate.op_attribute_data + attr->data_offset,
                               sizeof(bounds));
                        if (!isfinite(bounds[0]) || !isfinite(bounds[1]) ||
                            bounds[0] > bounds[1])
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_PADS:
    {
        /* Two int32 per axis, leading then trailing. data_len counts
                         * bytes, so a rank beyond 31 cannot be expressed and is
                         * refused by the width check rather than truncated. */
                        uint32_t pads_bytes =
                            2u * (uint32_t)input->ndim * (uint32_t)sizeof(int32_t);
                        if (input->ndim == 0u || pads_bytes > 0xFFu ||
                            (uint32_t)attr->data_len != pads_bytes ||
                            (attr->data_offset % 4u) != 0u)
                            return TIGRIS_ERR_BAD_SECTION;
                        for (uint8_t byte = 0; byte < attr->data_len; byte += 4u) {
                            int32_t pad;
                            memcpy(&pad,
                                   ctx->candidate.op_attribute_data + attr->data_offset +
                                       byte,
                                   sizeof(pad));
                            if (pad < 0)
                                return TIGRIS_ERR_BAD_SECTION;
                        }
        break;
    }
    case TIGRIS_OP_ATTR_BINARY_REQUANT:
    {
        /* Three Q0.31 pairs. A shift outside the range the requant
                         * helper can apply would shift by more than the width of the
                         * accumulator, which is undefined rather than saturating. */
                        if (attr->data_len != TIGRIS_OP_ATTR_BINARY_REQUANT_LEN ||
                            ctx->hdr->version < TIGRIS_SCHEMA_VERSION_BINARY_REQUANT)
                            return TIGRIS_ERR_BAD_SECTION;
                        for (uint8_t pair = 0; pair < 3u; pair++) {
                            int32_t multiplier;
                            int32_t shift;
                            memcpy(&multiplier,
                                   ctx->candidate.op_attribute_data + attr->data_offset +
                                       (uint32_t)pair * 8u,
                                   sizeof(multiplier));
                            memcpy(&shift,
                                   ctx->candidate.op_attribute_data + attr->data_offset +
                                       (uint32_t)pair * 8u + 4u,
                                   sizeof(shift));
                            if (multiplier < 0 || shift < -31 || shift > 31)
                                return TIGRIS_ERR_BAD_SECTION;
                        }
        break;
    }
    case TIGRIS_OP_ATTR_CONSTANT_OPERAND:
    {
        /* Operand position, a zero byte, a quant param index and
                         * optionally a shape, which the operator's own validation
                         * resolves. */
                        const uint8_t *payload = ctx->candidate.op_attribute_data + attr->data_offset;
                        if (attr->data_len < TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN ||
                            (attr->data_len - TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) % 4u != 0u ||
                            payload[0] > (op->op_type == TIGRIS_OP_SELECT_V2 ? 2u : 1u) || payload[1] != 0u)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_POOL_ROUNDING:
    {
        if (attr->data_len != 1u ||
                            ctx->candidate.op_attribute_data[attr->data_offset] !=
                                TIGRIS_POOL_ROUNDING_AVERAGE)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_AXES:
    {
        if (attr->data_len == 0u || attr->data_len > input->ndim)
                            return TIGRIS_ERR_BAD_SECTION;
                        const uint8_t *axes =
                            ctx->candidate.op_attribute_data + attr->data_offset;
                        for (uint8_t axis = 0; axis < attr->data_len; axis++) {
                            if (axes[axis] >= input->ndim)
                                return TIGRIS_ERR_BAD_SECTION;
                            for (uint8_t prior = 0; prior < axis; prior++) {
                                if (axes[prior] >= axes[axis])
                                    return TIGRIS_ERR_BAD_SECTION;
                            }
                        }
        break;
    }
    default:
        return TIGRIS_ERR_BAD_SECTION;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_attribute_high(
    tigris_load_state_t *ctx, const tigris_op_attribute_t *attr,
    const tigris_op_t *op, const tigris_tensor_t *input,
    const tigris_tensor_t *output)
{
    const tigris_file_header_t *hdr = ctx->hdr;
    (void)hdr; (void)op; (void)input; (void)output;
    switch (attr->type) {
    case TIGRIS_OP_ATTR_GELU_APPROXIMATE:
    {
        if (attr->data_len != 1u ||
                            ctx->candidate.op_attribute_data[attr->data_offset] > 1u)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_COMPARISON_REQUANT:
    {
        int32_t values[5];
                        if (attr->data_len != TIGRIS_OP_ATTR_COMPARISON_REQUANT_LEN)
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(values, ctx->candidate.op_attribute_data + attr->data_offset, sizeof(values));
                        if (values[0] != 8 || values[1] < 0 || values[3] < 0 ||
                            values[2] < -31 || values[2] > 0 || values[4] < -31 || values[4] > 0)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_MOVEMENT:
    {
        if (!movement_metadata_valid(&ctx->candidate, op, attr->op_index)) return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_CONSTANTS:
    {
        const uint8_t *list = ctx->candidate.op_attribute_data + attr->data_offset;
                        if (attr->data_len == 0u || (attr->data_len % 2u) != 0u)
                            return TIGRIS_ERR_BAD_SECTION;
                        for (uint8_t byte = 0; byte < attr->data_len; byte += 2u) {
                            uint16_t index = (uint16_t)((uint16_t)list[byte] |
                                                        (uint16_t)((uint16_t)list[byte + 1u] << 8));
                            if (index != TIGRIS_NO_WEIGHT && index >= ctx->hdr->num_weights)
                                return TIGRIS_ERR_BAD_SECTION;
                        }
        break;
    }
    case TIGRIS_OP_ATTR_TABLE_SHAPE:
    {
        if (attr->data_len == 0u || attr->data_len > 24u || attr->data_len % 4u != 0u)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_DETECTION:
    {
        /* Its values are checked with the operator. */
                        if (attr->data_len != TIGRIS_OP_ATTR_DETECTION_LEN)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_SUBGRAPHS:
    {
        /* Graph indices are checked with the section. */
                        if (attr->data_len != 4u)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_LSTM:
    {
        int32_t time_major;
                        float clip;
                        /* Float32, or int8 with its 26 integer parameters. */
                        if (attr->data_len != 8u && attr->data_len != 112u)
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(&time_major, ctx->candidate.op_attribute_data + attr->data_offset, sizeof(time_major));
                        memcpy(&clip, ctx->candidate.op_attribute_data + attr->data_offset + 4u, sizeof(clip));
                        if ((time_major != 0 && time_major != 1) || !isfinite(clip) || clip < 0.0f)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_SVDF:
    {
        int32_t values[6];
                        if (attr->data_len != 4u && attr->data_len != sizeof(values))
                            return TIGRIS_ERR_BAD_SECTION;
                        memcpy(values, ctx->candidate.op_attribute_data + attr->data_offset, attr->data_len);
                        if (values[0] <= 0)
                            return TIGRIS_ERR_BAD_SECTION;
                        /* A requantization shift beyond the accumulator width would be
                         * undefined rather than saturating. */
                        if (attr->data_len == sizeof(values) &&
                            (values[1] < INT16_MIN || values[1] > INT16_MAX ||
                             values[2] < 0 || values[3] < -31 || values[3] > 0 ||
                             values[4] < 0 || values[5] < -31 || values[5] > 0))
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_CUMSUM_OPTIONS:
    {
        const uint8_t *options = ctx->candidate.op_attribute_data + attr->data_offset;
                        if (attr->data_len != 2u || options[0] > 1u || options[1] > 1u)
                            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    default:
        return TIGRIS_ERR_BAD_SECTION;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_state_and_attributes(tigris_load_state_t *ctx)
{
    /* Every state tensor is one variable's value on entry or on exit, the
     * two alike, sized as the entry states. */
    for (uint16_t tensor_idx = 0; tensor_idx < ctx->hdr->num_tensors; tensor_idx++) {
        uint16_t roles = 0u;
        for (uint16_t i = 0; i < ctx->candidate.num_state; i++) {
            const tigris_state_entry_t *entry = &ctx->candidate.state_entries[i];
            if (entry->input == tensor_idx)
                roles++;
            if (entry->output == tensor_idx)
                roles++;
        }
        const uint16_t flagged =
            ((ctx->candidate.tensors[tensor_idx].flags & TIGRIS_TENSOR_STATE) != 0u) ? 1u : 0u;
        if (roles != flagged)
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint16_t i = 0; i < ctx->candidate.num_state; i++) {
        const tigris_state_entry_t *entry = &ctx->candidate.state_entries[i];
        if (entry->input >= ctx->hdr->num_tensors ||
            (entry->output != 0xFFFFu && entry->output >= ctx->hdr->num_tensors))
            return TIGRIS_ERR_BAD_TENSOR;
        const tigris_tensor_t *in = &ctx->candidate.tensors[entry->input];
        if (in->size_bytes != entry->bytes || (in->dtype != 1u && in->dtype != 3u && in->dtype != 5u))
            return TIGRIS_ERR_BAD_TENSOR;
        if (entry->output != 0xFFFFu) {
            const tigris_tensor_t *out = &ctx->candidate.tensors[entry->output];
            if (out->dtype != in->dtype || out->size_bytes != in->size_bytes ||
                !tensor_shapes_equal(&ctx->candidate, in, out) ||
                (out->flags & TIGRIS_TENSOR_LINEAR) != (in->flags & TIGRIS_TENSOR_LINEAR) ||
                (in->dtype == 3u && !same_quantization(tigris_tensor_quant(&ctx->candidate, in),
                                                       tigris_tensor_quant(&ctx->candidate, out))))
                return TIGRIS_ERR_BAD_TENSOR;
        }
    }

    for (uint16_t i = 0; i < ctx->hdr->num_ops; i++) {
        const tigris_op_t *op = &ctx->candidate.ops[i];
        if (!string_ok(ctx->candidate.strings, ctx->strings_len, op->name_str) ||
            !elements_ok(op->inputs_off, op->num_inputs, ctx->index_count) ||
            !elements_ok(op->outputs_off, op->num_outputs, ctx->index_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
            ctx->hdr->num_stages && op->stage >= ctx->hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        if ((op->weight_idx != TIGRIS_NO_WEIGHT &&
             op->weight_idx >= ctx->hdr->num_weights) ||
            (op->bias_idx != TIGRIS_NO_WEIGHT &&
             op->bias_idx >= ctx->hdr->num_weights))
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t j = 0; j < op->num_inputs; j++) {
            if (ctx->candidate.index_pool[op->inputs_off + j] >= ctx->hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
        for (uint8_t j = 0; j < op->num_outputs; j++) {
            if (ctx->candidate.index_pool[op->outputs_off + j] >= ctx->hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    /* Attribute records are ordered and typed.  Validate their payloads
     * against the referenced operators before any kernel can use them.
     *
     * Schemas v2/v3 predate typed attributes and allowed custom dispatch
     * callbacks to implement operators such as legacy Transpose, MatMul, and
     * application opcodes; semantic validation leaves exactly those open
     * there (legacy_semantics_are_builtin). */
    if (ctx->candidate.op_attributes) {
        uint32_t attrs_off = ctx->section_offsets[TIGRIS_SEC_OP_ATTRIBUTES];
        uint32_t attrs_data_off = attrs_off + 4u +
            (uint32_t)ctx->candidate.num_op_attributes *
            sizeof(tigris_op_attribute_t);
        uint32_t attrs_data_len = ctx->section_ends[TIGRIS_SEC_OP_ATTRIBUTES] -
            attrs_data_off;
        /* An attribute may refer to a later attribute of the same operator.
         * Check all spans before any payload validator follows such a link. */
        for (uint16_t i = 0; i < ctx->candidate.num_op_attributes; i++) {
            const tigris_op_attribute_t *attr = &ctx->candidate.op_attributes[i];
            if (attr->op_index >= ctx->hdr->num_ops ||
                !bounds_ok(attr->data_offset, attr->data_len, attrs_data_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
        uint16_t previous_op = 0;
        uint8_t previous_attr_type = 0;
        for (uint16_t i = 0; i < ctx->candidate.num_op_attributes; i++) {
            const tigris_op_attribute_t *attr = &ctx->candidate.op_attributes[i];
            if (attr->op_index >= ctx->hdr->num_ops ||
                (i > 0 &&
                 (attr->op_index < previous_op ||
                  (attr->op_index == previous_op &&
                   attr->type <= previous_attr_type))) ||
                !bounds_ok(attr->data_offset, attr->data_len, attrs_data_len))
                return TIGRIS_ERR_BAD_SECTION;
            const tigris_op_t *op = &ctx->candidate.ops[attr->op_index];
            /* Every kind a schema-8 plan may carry is validated here whether
             * or not an operator consumes it yet, so an operator behind one
             * of them lands without a further version. A kind this build does
             * not know still fails closed, in the default arm. */
            if (attr->type > TIGRIS_OP_ATTR_TRANSPOSE_PERM &&
                ctx->hdr->version < TIGRIS_SCHEMA_VERSION_V8)
                return TIGRIS_ERR_BAD_SECTION;
            /* Svdf and Lstm also write their next states; If gives each
             * result its branches compute; DetectionPostProcess writes four
             * results and its working memory. */
            if ((op->num_inputs < 1u && !(movement_is_gather(op->op_type) &&
                 op->weight_idx != TIGRIS_NO_WEIGHT && op->bias_idx != TIGRIS_NO_WEIGHT)) ||
                (op->num_outputs != 1u && !(op->op_type == TIGRIS_OP_SVDF && op->num_outputs == 2u) &&
                 !(op->op_type == TIGRIS_OP_LSTM && op->num_outputs == 3u) &&
                 !(op->op_type == TIGRIS_OP_DETECTION_POSTPROCESS && op->num_outputs == 5u) &&
                 !((op->op_type == TIGRIS_OP_IF || op->op_type == TIGRIS_OP_WHILE) &&
                   op->num_outputs >= 1u)))
                return TIGRIS_ERR_BAD_SECTION;
            /* A kind belongs to the operator that reads it, whether or not a
             * kernel for that operator exists yet, so a stray payload cannot
             * ride along on an unrelated op. */
            if (!attr_kind_matches_op(attr->type, op->op_type))
                return TIGRIS_ERR_BAD_SECTION;
            uint16_t input_idx = ctx->candidate.index_pool[op->inputs_off];
            uint16_t output_idx = ctx->candidate.index_pool[op->outputs_off];
            const tigris_tensor_t *input = &ctx->candidate.tensors[input_idx];
            const tigris_tensor_t *output = &ctx->candidate.tensors[output_idx];
            const tigris_attribute_contract_t *attr_contract =
                tigris_attribute_contract(attr->type);
            if (attr_contract == NULL || attr_contract->step == 0u ||
                attr->data_len < attr_contract->min_len ||
                attr->data_len > attr_contract->max_len ||
                ((uint16_t)attr->data_len - attr_contract->min_len) % attr_contract->step != 0u ||
                (attr_contract->exact_len != 0u && attr->data_len != attr_contract->exact_len))
                return TIGRIS_ERR_BAD_SECTION;
            tigris_error_t value_error = attr->type <= TIGRIS_OP_ATTR_RESIZE_SCALES
                ? validate_attribute_low(ctx, attr, op, input, output)
                : validate_attribute_high(ctx, attr, op, input, output);
            if (value_error != TIGRIS_OK) return value_error;

            previous_op = attr->op_index;
            previous_attr_type = attr->type;
        }
        uint16_t attr_cursor = 0u;
        for (uint16_t op_idx = 0u; op_idx < ctx->hdr->num_ops; op_idx++) {
            uint32_t present = 0u;
            while (attr_cursor < ctx->candidate.num_op_attributes &&
                   ctx->candidate.op_attributes[attr_cursor].op_index == op_idx) {
                present |= 1u << ctx->candidate.op_attributes[attr_cursor].type;
                attr_cursor++;
            }
            const tigris_op_contract_t *row =
                tigris_op_contract(ctx->candidate.ops[op_idx].op_type);
            if (row != NULL &&
                (present & row->required_attributes) != row->required_attributes)
                return TIGRIS_ERR_BAD_SECTION;
        }
    } else if (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES) {
        for (uint16_t op_idx = 0u; op_idx < ctx->hdr->num_ops; op_idx++) {
            const tigris_op_contract_t *row =
                tigris_op_contract(ctx->candidate.ops[op_idx].op_type);
            if (row != NULL && row->required_attributes != 0u)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }
    return TIGRIS_OK;
}

static int stage_row_banded(tigris_load_state_t *ctx, const tigris_stage_t *stage)
{
    int32_t band_batch = -1;
    int32_t band_rows = -1;
    int32_t band_cols = 0;
    int row_banded = stage->chain_len == 0 &&
                     stage->outputs_count > 0;
    if (row_banded) {
        for (uint16_t j = 0; j < stage->outputs_count; j++) {
            int32_t b = 0;
            int32_t r = 0;
            if (!tigris_tensor_row_view(
                    &ctx->candidate,
                    &ctx->candidate.tensors[
                        ctx->candidate.index_pool[
                            stage->outputs_off + j]],
                    &b, &r, &band_cols)) {
                row_banded = 0;
                break;
            }
            if (band_rows < 0) {
                band_batch = b;
                band_rows = r;
            }
            if (b != band_batch || r != band_rows) {
                row_banded = 0;
                break;
            }
        }
    }
    if (row_banded && band_rows <= 1)
        row_banded = 0;
    if (row_banded) {
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            const tigris_op_t *rop = &ctx->candidate.ops[
                ctx->candidate.index_pool[stage->ops_off + j]];
            int32_t b = 0;
            int32_t r = 0;
            if (rop->num_inputs == 0u || rop->num_outputs != 1u ||
                (!tigris_row_tiling_op(rop->op_type) && !tigris_op_independent_band(
                    &ctx->candidate, rop, ctx->candidate.index_pool[stage->ops_off + j], 1)) ||
                !tigris_tensor_row_view(
                    &ctx->candidate,
                    &ctx->candidate.tensors[
                        ctx->candidate.index_pool[rop->outputs_off]],
                    &b, &r, &band_cols) ||
                b != band_batch || r != band_rows) {
                row_banded = 0;
                break;
            }
            for (uint8_t k = 0; k < rop->num_inputs; k++) {
                int32_t ib = 0;
                int32_t ir = 0;
                int32_t ic = 0;
                if (tigris_tensor_row_view(
                        &ctx->candidate,
                        &ctx->candidate.tensors[
                            ctx->candidate.index_pool[
                                rop->inputs_off + k]],
                        &ib, &ir, &ic) &&
                    ib == band_batch && ir == band_rows)
                    continue;
                if (k == 0u ||
                    !tigris_whole_operand_op(rop->op_type)) {
                    row_banded = 0;
                    break;
                }
            }
            if (!row_banded)
                break;
        }
    }
    if (row_banded) {
        for (uint16_t j = 0; j < stage->inputs_count; j++) {
            int32_t b = 0;
            int32_t r = 0;
            if (!tigris_tensor_row_view(
                    &ctx->candidate,
                    &ctx->candidate.tensors[
                        ctx->candidate.index_pool[
                            stage->inputs_off + j]],
                    &b, &r, &band_cols)) {
                row_banded = 0;
                break;
            }
        }
    }

    return row_banded;
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
        out->band_axis = tigris_band_axis(plan, band, tigris_stage_band_mode(plan, stage));
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
    } else if (tile.output_band_extent != 0u) {
        /* A Resize bands its output rows. */
        out->band_axis = 1u;
        extent = (uint32_t)out_shape[1];
        out->tile_extent = tile.output_band_extent;
        halo = stage_halo(plan, stage);
    } else {
        /* The height stripe runs over the first input more than one row high. */
        const tigris_tensor_t *band = in;
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            const tigris_tensor_t *t = &plan->tensors[inputs[i]];
            if (t->ndim != 4u || tigris_tensor_shape(plan, t)[1] != 1) {
                band = t;
                break;
            }
        }
        if (band->ndim < 2u)
            return 0;
        out->band_axis = 1u;
        extent = (uint32_t)tigris_tensor_shape(plan, band)[1];
        out->tile_extent = tile.input_band_extent;
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
    return 1;
}

static tigris_error_t validate_stage_tile_band(
    tigris_load_state_t *ctx, const tigris_stage_t *stage,
    const tigris_tile_plan_t *tile, uint8_t axis, int reshape,
    int32_t leading_rows)
{
    if (!reshape && !leading_rows && tile->tileable &&
        axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH) {
        if (stage->inputs_count == 0 || stage->outputs_count == 0)
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t first_input =
            ctx->candidate.index_pool[stage->inputs_off];
        uint16_t first_output =
            ctx->candidate.index_pool[stage->outputs_off];
        /* A row-banded stage is checked on its own terms first: its
         * tensors need not share a rank, only a batch and a row
         * count, because a Reshape that regroups the leading axes
         * moves no data. The outputs settle the band, since an output
         * is always banded; an input the band does not cut is read
         * whole, which only a matrix product's second operand is
         * entitled to be. Mirrors tigris_stage_row_tiled, the executor's rule,
         * and accepting a stage it will decline sends the stage
         * through the normal path instead of refusing the plan. */
        int row_banded = stage_row_banded(ctx, stage);
        uint8_t rank = ctx->candidate.tensors[first_input].ndim;
        int transpose = rank >= 3u && rank <= 6u && stage->ops_count == 1u && stage->chain_len == 0u &&
            ctx->candidate.ops[ctx->candidate.index_pool[stage->ops_off]].op_type == TIGRIS_OP_TRANSPOSE;
        if (!row_banded &&
            ((!transpose && rank != 3 && rank != 4) ||
             ctx->candidate.tensors[first_output].ndim != rank))
            return TIGRIS_ERR_BAD_SECTION;
        /* A row band agrees on rows, not on rank: the Reshape pair
         * around a lowered matrix product drops a unit leading axis
         * without moving a byte, and its row count is checked above. */
        if (!row_banded) {
            for (uint16_t j = 0; j < stage->inputs_count; j++) {
                uint16_t tensor_idx =
                    ctx->candidate.index_pool[stage->inputs_off + j];
                if (ctx->candidate.tensors[tensor_idx].ndim != rank)
                    return TIGRIS_ERR_BAD_SECTION;
            }
            for (uint16_t j = 0; j < stage->outputs_count; j++) {
                uint16_t tensor_idx =
                    ctx->candidate.index_pool[stage->outputs_off + j];
                if (ctx->candidate.tensors[tensor_idx].ndim != rank)
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }

        /* Schema v5 adds the rank-3 NLC length contract. Unary
         * pointwise ops may surround one Conv1D. Dynamic binary ops
         * remain pointwise-only: mixing an external binary operand
         * with a strided Conv1D can expose different length
         * resolutions. Rank-4 stages retain the existing audited
         * height contract and are checked again by the executor. */
        /* A per-channel operand is one row high, so a rank-4 height
         * stripe loads it whole for every band. The 2D, chain, row
         * band and rank-3 executors have no such rule, so there a
         * binary op may only take full-shape operands. */
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            const tigris_op_t *bop = &ctx->candidate.ops[
                ctx->candidate.index_pool[stage->ops_off + j]];
            if (is_axis1_binary_pointwise_op(bop->op_type) &&
                bop->num_inputs >= 2) {
                const uint16_t *bin = tigris_op_inputs(&ctx->candidate, bop);
                for (uint8_t operand = 1; operand < bop->num_inputs; operand++) {
                    if (!tensor_shapes_equal(&ctx->candidate,
                                             &ctx->candidate.tensors[bin[0]],
                                             &ctx->candidate.tensors[bin[operand]]) &&
                        (stage->chain_len != 0 || tile->tile_width != 0 ||
                         rank != 4 || row_banded))
                        return TIGRIS_ERR_BAD_OPERATOR;
                }
            }
        }

        if (row_banded) {
            /* Already validated on its own terms above. */
        } else if (stage->ops_count == 1 &&
            ctx->candidate.ops[
                ctx->candidate.index_pool[stage->ops_off]
            ].op_type == TIGRIS_OP_TRANSPOSE) {
            /* A transpose is banded along the longer of the two
             * extents it swaps, and only as the whole stage.
             * transpose_band_groups states which permutations that
             * is: the ones that swap two adjacent groups of axes and
             * leave the rest in order. The executor reads the same
             * rule from the same header rather than from a copy. */
            uint16_t conv_op = ctx->candidate.index_pool[stage->ops_off];
            uint8_t perm_rank = 0;
            const uint8_t *perm = tigris_op_attribute_data(
                &ctx->candidate, conv_op,
                TIGRIS_OP_ATTR_TRANSPOSE_PERM, &perm_rank);
            uint8_t tb_prefix = 0;
            uint8_t tb_split = 0;
            uint8_t tb_middle = 0;
            int convertible =
                perm && perm_rank == rank &&
                transpose_band_groups(perm, rank, &tb_prefix,
                                      &tb_split, &tb_middle);
            if (!convertible ||
                stage->inputs_count != 1 ||
                stage->outputs_count != 1 ||
                stage->chain_len != 0)
                return TIGRIS_ERR_BAD_OPERATOR;
        } else if (stage->ops_count == 1u && stage->chain_len == 0u &&
                   tigris_op_independent_band(&ctx->candidate,
                       &ctx->candidate.ops[ctx->candidate.index_pool[stage->ops_off]],
                       ctx->candidate.index_pool[stage->ops_off], 0)) {
            /* The operator preserves serialized height on every operand. */
        } else if (rank == 3) {
            uint16_t conv_count = 0;
            uint16_t binary_count = 0;
            if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_TILE_AXIS ||
                stage->chain_len != 0)
                return TIGRIS_ERR_BAD_OPERATOR;
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                uint8_t type = ctx->candidate.ops[
                    ctx->candidate.index_pool[stage->ops_off + j]
                ].op_type;
                if (type == TIGRIS_OP_CONV1D) {
                    conv_count++;
                } else if (is_axis1_binary_pointwise_op(type)) {
                    binary_count++;
                } else if (!is_axis1_unary_pointwise_op(type)) {
                    return TIGRIS_ERR_BAD_OPERATOR;
                }
            }
            if (conv_count > 1 ||
                (conv_count == 1 && binary_count > 0))
                return TIGRIS_ERR_BAD_OPERATOR;
        } else if (stage->ops_count == 1 &&
                   ctx->candidate.ops[
                       ctx->candidate.index_pool[stage->ops_off]
                   ].op_type == TIGRIS_OP_GLOBAL_AVG) {
            /* A global reduction collapses the height the stripe
             * contract propagates, so the executor walks its input
             * instead. Permitted only as the whole stage and only
             * when the output really is collapsed: with anything
             * else present there would be a height to carry. */
            const int32_t *out_shape = tigris_tensor_shape(
                &ctx->candidate, &ctx->candidate.tensors[first_output]);
            if (stage->inputs_count != 1 ||
                stage->outputs_count != 1 ||
                stage->chain_len != 0 ||
                rank != 4 ||
                out_shape[1] != 1 || out_shape[2] != 1)
                return TIGRIS_ERR_BAD_OPERATOR;
        } else {
            uint16_t spatial_count = 0;
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                uint8_t type = ctx->candidate.ops[
                    ctx->candidate.index_pool[stage->ops_off + j]
                ].op_type;
                const tigris_op_contract_t *row = tigris_op_contract(type);
                if (row != NULL && (row->flags & TIGRIS_CONTRACT_SPATIAL) != 0u) {
                    spatial_count++;
                } else if (row != NULL && (row->flags & TIGRIS_CONTRACT_RESAMPLE) != 0u) {
                    if (stage->chain_len != 0u) return TIGRIS_ERR_BAD_OPERATOR;
                    spatial_count++;
                } else if (row == NULL || (row->flags & TIGRIS_CONTRACT_HEIGHT) == 0u) {
                    return TIGRIS_ERR_BAD_OPERATOR;
                }
            }
            if (spatial_count > 1)
                return TIGRIS_ERR_BAD_OPERATOR;
        }
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_stage(tigris_load_state_t *ctx, uint16_t i)
{
    const tigris_stage_t *stage = &ctx->candidate.stages[i];
    if (!elements_ok(stage->ops_off, stage->ops_count, ctx->index_count) ||
        !elements_ok(stage->inputs_off, stage->inputs_count, ctx->index_count) ||
        !elements_ok(stage->outputs_off, stage->outputs_count, ctx->index_count) ||
        (stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN &&
         stage->tile_plan_idx >= ctx->hdr->num_tile_plans) ||
        (stage->chain_len == 0 && stage->chain_id != TIGRIS_NO_CHAIN) ||
        (stage->chain_len > 0 && stage->chain_id == TIGRIS_NO_CHAIN))
        return TIGRIS_ERR_BAD_SECTION;
    note_requirement(&ctx->required.stage_inputs, stage->inputs_count);
    note_requirement(&ctx->required.stage_outputs, stage->outputs_count);
    note_requirement(&ctx->required.chain_stages, stage->chain_len);
    if (stage->inputs_count > TIGRIS_MAX_STAGE_INPUTS ||
        stage->outputs_count > TIGRIS_MAX_STAGE_OUTPUTS ||
        stage->chain_len > TIGRIS_MAX_CHAIN_STAGES) {
        if (ctx->out_required)
            *ctx->out_required = ctx->required;
        return TIGRIS_ERR_PLAN_LIMITS;
    }

    /* A chain is represented redundantly on every member.  The executor
     * only starts it at its declared head and skips the other members, so
     * accepting an incomplete or displaced chain silently drops work. */
    if (stage->chain_len > 0) {
        if (stage->chain_len < 2 || stage->chain_id > i ||
            stage->chain_len > ctx->hdr->num_stages - stage->chain_id)
            return TIGRIS_ERR_BAD_SECTION;
        const tigris_stage_t *head = &ctx->candidate.stages[stage->chain_id];
        if (head->chain_id != stage->chain_id ||
            head->chain_len != stage->chain_len ||
            head->chain_tile_h == 0 ||
            stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN ||
            (i != stage->chain_id && stage->chain_tile_h != 0))
            return TIGRIS_ERR_BAD_SECTION;
        if (i == stage->chain_id) {
            for (uint16_t member = i + 1u;
                 member < i + stage->chain_len; member++) {
                const tigris_stage_t *next = &ctx->candidate.stages[member];
                if (next->chain_id != i ||
                    next->chain_len != stage->chain_len ||
                    next->chain_tile_h != 0 ||
                    next->tile_plan_idx != TIGRIS_NO_TILE_PLAN)
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
    } else if (stage->chain_tile_h != 0) {
        return TIGRIS_ERR_BAD_SECTION;
    }

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = ctx->candidate.index_pool[stage->ops_off + j];
        if (op_idx >= ctx->hdr->num_ops)
            return TIGRIS_ERR_BAD_SECTION;
        /* The binary plan contract is a forward, partitioned schedule.
         * Enforcing it here prevents a malformed stage list from silently
         * omitting work or running the same operation more than once. */
        if (op_idx != ctx->next_scheduled_op)
            return TIGRIS_ERR_BAD_SECTION;
        /* In schema v5 the stage table is authoritative and the legacy
         * operator byte is a canonical low-byte hint. This preserves the
         * zero-copy v2-v4 record layout without imposing a 256-stage
         * ceiling on larger v5 plans. */
        if ((ctx->hdr->version < TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
             ctx->candidate.ops[op_idx].stage != i) ||
            (ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
             ctx->candidate.ops[op_idx].stage != (uint8_t)i))
            return TIGRIS_ERR_BAD_OPERATOR;
        ctx->next_scheduled_op++;
    }
    for (uint16_t j = 0; j < stage->inputs_count; j++) {
        if (ctx->candidate.index_pool[stage->inputs_off + j] >= ctx->hdr->num_tensors)
            return TIGRIS_ERR_BAD_SECTION;
    }
    for (uint16_t j = 0; j < stage->outputs_count; j++) {
        if (ctx->candidate.index_pool[stage->outputs_off + j] >= ctx->hdr->num_tensors)
            return TIGRIS_ERR_BAD_SECTION;
    }

    if (stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN) {
        tigris_tile_plan_t legacy = {0};
        const tigris_tile_plan_t *tile;
        const tigris_tile_plan_v10_t *wide = ctx->candidate.tile_plans_v11
            ? &ctx->candidate.tile_plans_v11[stage->tile_plan_idx].choices
            : ctx->candidate.tile_plans_v10 ? &ctx->candidate.tile_plans_v10[stage->tile_plan_idx] : NULL;
        if (wide) {
            if (!tile_v10_valid(&ctx->candidate, stage, wide))
                return TIGRIS_ERR_BAD_OPERATOR;
            legacy.tileable = wide->tileable;
            legacy.axis = wide->axis;
            legacy.tile_width = wide->output_width_extent != 0u ? 1u : 0u;
            tile = &legacy;
        } else {
            tile = &ctx->candidate.tile_plans[stage->tile_plan_idx];
        }
        uint8_t axis =
            ctx->hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS
                ? tile->axis
                : (tile->tileable
                       ? TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH
                       : TIGRIS_TILE_AXIS_NONE);
        tigris_reshape_band_t reshape_view;
        int reshape = tigris_reshape_band(&ctx->candidate, stage, &reshape_view);
        if (reshape && tile->tileable && !wide && !tigris_reshape_tile_valid(tile, &reshape_view))
            return TIGRIS_ERR_BAD_OPERATOR;
        int32_t leading_rows = tigris_stage_leading_band(&ctx->candidate, stage);
        if (leading_rows && tile->tileable) {
            tigris_reshape_band_t band = {1, leading_rows, 1, leading_rows, 1, 0u};
            if (!wide && !tigris_reshape_tile_valid(tile, &band)) return TIGRIS_ERR_BAD_OPERATOR;
        }
        tigris_error_t band_error = validate_stage_tile_band(
            ctx, stage, tile, axis, reshape, leading_rows);
        if (band_error != TIGRIS_OK) return band_error;
        if (ctx->candidate.tile_plans_v11) {
            tigris_tile_facts_t derived;
            const tigris_tile_facts_t *stated =
                &ctx->candidate.tile_plans_v11[stage->tile_plan_idx].facts;
            if (!tigris_stage_tile_facts(&ctx->candidate, stage, &derived) ||
                memcmp(&derived, stated, sizeof(derived)) != 0)
                return TIGRIS_ERR_TILE_FACTS;
        }
    }

    if (stage->chain_len >= 2) {
        if (stage->chain_id == TIGRIS_NO_CHAIN ||
            stage->chain_id >= ctx->hdr->num_stages ||
            stage->chain_len > ctx->hdr->num_stages - stage->chain_id)
            return TIGRIS_ERR_BAD_SECTION;

        /* Chain execution stores bounded spatial-op metadata in its
         * caller-provided workspace. Validate the stage-op slice and the
         * audited height-stripe operator contract before execution. */
        uint32_t pool_off = ctx->section_offsets[TIGRIS_SEC_INDEX_POOL];
        uint32_t prefix = (uint32_t)stage->ops_off * sizeof(uint16_t);
        uint32_t bytes = (uint32_t)stage->ops_count * sizeof(uint16_t);
        if (!bounds_ok(pool_off, prefix, ctx->buf_len) ||
            !bounds_ok(pool_off + prefix, bytes, ctx->buf_len))
            return TIGRIS_ERR_BAD_SECTION;

        const uint8_t *op_indices = ctx->buf + pool_off + prefix;
        uint32_t spatial_count = 0;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx;
            memcpy(&op_idx, op_indices + (uint32_t)j * sizeof(op_idx),
                   sizeof(op_idx));
            if (op_idx >= ctx->hdr->num_ops)
                return TIGRIS_ERR_BAD_SECTION;

            uint8_t op_type = ctx->candidate.ops[op_idx].op_type;
            const tigris_op_contract_t *row = tigris_op_contract(op_type);
            if (row != NULL && (row->flags & TIGRIS_CONTRACT_SPATIAL) != 0u) {
                spatial_count++;
                note_requirement(&ctx->required.spatial_ops_per_stage, spatial_count);
                if (spatial_count > TIGRIS_MAX_SPATIAL_OPS_PER_STAGE) {
                    if (ctx->out_required) *ctx->out_required = ctx->required;
                    return TIGRIS_ERR_PLAN_LIMITS;
                }
            } else if (row == NULL || (row->flags & TIGRIS_CONTRACT_HEIGHT) == 0u) {
                return TIGRIS_ERR_BAD_OPERATOR;
            }
        }
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_weights_and_stages(tigris_load_state_t *ctx)
{
    if (ctx->candidate.weight_entries && !ctx->candidate.weight_blocks) {
        uint32_t weights_off = ctx->section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)ctx->hdr->num_weights *
                                sizeof(tigris_weight_entry_t);
        uint32_t blob_len = ctx->section_ends[TIGRIS_SEC_WEIGHTS] -
                            weights_off - entries_size;
        for (uint16_t i = 0; i < ctx->hdr->num_weights; i++) {
            const tigris_weight_entry_t *weight = &ctx->candidate.weight_entries[i];
            if (!string_ok(ctx->candidate.strings, ctx->strings_len, weight->name_str) ||
                !bounds_ok(weight->offset, weight->size_bytes, blob_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t qp_off = ctx->section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        uint16_t nqp, qd_field;
        memcpy(&nqp, ctx->buf + qp_off, sizeof(nqp));
        memcpy(&qd_field, ctx->buf + qp_off + sizeof(nqp), sizeof(qd_field));
        uint32_t qd_len = ctx->hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? qd_field :
            (ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] - qp_off - 4u -
             (uint32_t)nqp * sizeof(tigris_quant_param_t)) / sizeof(int32_t);
        for (uint16_t i = 0; i < nqp; i++) {
            const tigris_quant_param_t *qp = &ctx->candidate.quant_params[i];
            uint32_t page = ctx->hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
            uint32_t moff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off;
            uint32_t soff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off;
            if (!isfinite(qp->scale) || qp->scale <= 0.0f)
                return TIGRIS_ERR_BAD_TENSOR;
            if (qp->num_channels == 0 ||
                (ctx->hdr->version != TIGRIS_SCHEMA_VERSION_V2 &&
                 (page >= qd_field ||
                  (uint32_t)qp->multiplier_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS ||
                  (uint32_t)qp->shift_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS)) ||
                !elements_ok(moff, qp->num_channels, qd_len) ||
                !elements_ok(soff, qp->num_channels, qd_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (ctx->candidate.weight_blocks) {
        uint32_t wb_off = ctx->section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        uint32_t block_entries = (uint32_t)ctx->candidate.num_weight_blocks *
                                 sizeof(tigris_weight_block_t);
        uint32_t blobs_off = wb_off + 4u + block_entries;
        uint32_t blobs_len = ctx->section_ends[TIGRIS_SEC_WEIGHT_BLOCKS] - blobs_off;
        if (ctx->candidate.weight_compression != TIGRIS_COMPRESS_NONE &&
            ctx->candidate.weight_compression != TIGRIS_COMPRESS_LZ4)
            return TIGRIS_ERR_BAD_SECTION;
        if (ctx->candidate.num_weight_blocks == 0 || ctx->hdr->num_weights == 0 ||
            ctx->candidate.num_weight_blocks > ctx->hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t previous_stage = TIGRIS_NO_CHAIN;
        for (uint16_t i = 0; i < ctx->candidate.num_weight_blocks; i++) {
            const tigris_weight_block_t *block = &ctx->candidate.weight_blocks[i];
            if (block->stage_idx >= ctx->hdr->num_stages ||
                (i > 0 && block->stage_idx <= previous_stage) ||
                !elements_ok(block->first_weight_idx, block->num_weights,
                             ctx->hdr->num_weights) ||
                block->num_weights == 0 || block->compressed_size == 0 ||
                block->uncompressed_size == 0 ||
                !bounds_ok(block->blob_offset, block->compressed_size, blobs_len) ||
                (ctx->candidate.weight_compression == TIGRIS_COMPRESS_NONE &&
                 block->compressed_size != block->uncompressed_size))
                return TIGRIS_ERR_BAD_SECTION;
            previous_stage = block->stage_idx;
        }
    }

    /* Plan limits validation */

    ctx->next_scheduled_op = 0;
    for (uint16_t i = 0; i < ctx->hdr->num_stages; i++) {
        tigris_error_t stage_error = validate_stage(ctx, i);
        if (stage_error != TIGRIS_OK) return stage_error;
    }
    return TIGRIS_OK;
}

static tigris_error_t validate_graph_and_publish(tigris_load_state_t *ctx)
{
    if (ctx->next_scheduled_op != ctx->hdr->num_ops)
        return TIGRIS_ERR_BAD_SECTION;

    /* A compressed stage redirects weight_entry offsets into its own decoded
     * block.  Check that every referenced weight/bias is present in that
     * block before any executor or kernel can dereference the offset. */
    if (ctx->candidate.weight_blocks) {
        uint16_t block_cursor = 0;
        for (uint16_t s = 0; s < ctx->hdr->num_stages; s++) {
            const tigris_weight_block_t *block = NULL;
            if (block_cursor < ctx->candidate.num_weight_blocks &&
                ctx->candidate.weight_blocks[block_cursor].stage_idx == s) {
                block = &ctx->candidate.weight_blocks[block_cursor++];
            }

            const tigris_stage_t *stage = &ctx->candidate.stages[s];
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                const tigris_op_t *op =
                    &ctx->candidate.ops[ctx->candidate.index_pool[stage->ops_off + j]];
                uint16_t refs[2] = {op->weight_idx, op->bias_idx};
                for (uint8_t r = 0; r < 2; r++) {
                    if (refs[r] == TIGRIS_NO_WEIGHT)
                        continue;
                    if (!block ||
                        !bounds_ok(ctx->candidate.weight_entries[refs[r]].offset,
                                   ctx->candidate.weight_entries[refs[r]].size_bytes,
                                   block->uncompressed_size))
                        return TIGRIS_ERR_BAD_SECTION;
                }
            }
        }
    }

    {
        tigris_error_t graph_error = validate_subgraphs(&ctx->candidate, ctx->index_count);
        if (graph_error != TIGRIS_OK)
            return graph_error;
    }

    {
        tigris_error_t semantic_error = validate_operator_semantics(&ctx->candidate);
        if (semantic_error != TIGRIS_OK)
            return semantic_error;
    }

    for (uint16_t n = 0u; n < ctx->candidate.header->num_ops; n++) {
        if (kernel_linked(&ctx->candidate, &ctx->candidate.ops[n]) == 0)
            return TIGRIS_ERR_KERNEL_DISABLED;
    }

    if (ctx->out_required)
        *ctx->out_required = ctx->required;

    *ctx->out_plan = ctx->candidate;
    return TIGRIS_OK;
}

tigris_error_t tigris_plan_load_ex(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan,
    tigris_plan_limits_t *out_required)
{
    tigris_load_state_t state;
    memset(&state, 0, sizeof(state));
    state.buf = buf;
    state.buf_len = buf_len;
    state.out_plan = out_plan;
    state.out_required = out_required;
    tigris_error_t error;
    error = load_header_and_directory(&state);
    if (error != TIGRIS_OK) return error;
    error = locate_optional_sections(&state);
    if (error != TIGRIS_OK) return error;
    error = validate_tables_and_ops(&state);
    if (error != TIGRIS_OK) return error;
    error = validate_state_and_attributes(&state);
    if (error != TIGRIS_OK) return error;
    error = validate_weights_and_stages(&state);
    if (error != TIGRIS_OK) return error;
    error = validate_graph_and_publish(&state);
    if (error != TIGRIS_OK) return error;
    return TIGRIS_OK;
}


const char *tigris_error_str(tigris_error_t err)
{
    switch (err) {
        case TIGRIS_OK:             return "OK";
        case TIGRIS_ERR_NULL:       return "null pointer argument";
        case TIGRIS_ERR_TOO_SMALL:  return "buffer too small for header";
        case TIGRIS_ERR_BAD_MAGIC:  return "bad magic (expected TGRS)";
        case TIGRIS_ERR_BAD_VERSION:return "unsupported plan version";
        case TIGRIS_ERR_BAD_SIZE:   return "file_size field mismatch";
        case TIGRIS_ERR_BAD_SECTION:return "section offset out of bounds";
        case TIGRIS_ERR_MISSING_SEC:return "required section missing";
        case TIGRIS_ERR_ENDIAN:     return "platform is not little-endian";
        case TIGRIS_ERR_PLAN_LIMITS:return "plan exceeds executor limits";
        case TIGRIS_ERR_BAD_TENSOR: return "tensor contract is not executable";
        case TIGRIS_ERR_BAD_OPERATOR:return "operator contract is not executable";
        case TIGRIS_ERR_BAD_INTERFACE:
            return "declared model interface is not convertible";
        case TIGRIS_ERR_KERNEL_DISABLED:
            return "operator kernel not built into this runtime";
        case TIGRIS_ERR_MEM:        return "arena cannot hold the tensor";
        case TIGRIS_ERR_TENSOR_SLOTS:
            return "memory manager has fewer tensor slots than the plan";
        case TIGRIS_ERR_TILE_FACTS:
            return "tile record disagrees with the plan's tiling";
        default:                    return "unknown error";
    }
}

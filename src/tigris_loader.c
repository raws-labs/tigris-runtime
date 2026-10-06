/**
 * @file tigris_loader.c
 * @brief Zero-alloc binary plan parser.
 *
 * Validates the file header, walks the section directory, and sets
 * pointers into the buffer. Never allocates memory.
 */

#include "tigris_loader.h"
#include "tigris_binary_operands.h"

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
    return type == TIGRIS_OP_ADD || type == TIGRIS_OP_SUB ||
           type == TIGRIS_OP_MUL || type == TIGRIS_OP_DIV ||
           type == TIGRIS_OP_SQUARED_DIFFERENCE ||
           type == TIGRIS_OP_MAXIMUM || type == TIGRIS_OP_MINIMUM ||
           type == TIGRIS_OP_FLOOR_DIV || type == TIGRIS_OP_FLOOR_MOD || type == TIGRIS_OP_PRELU ||
           (type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_LOGICAL_OR) || type == TIGRIS_OP_SELECT_V2;
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
    const uint32_t element = (dtype == 1u) ? (uint32_t)sizeof(float) : 1u;
    *quant = NULL;
    if (op->num_inputs != (op->op_type == TIGRIS_OP_SELECT_V2 ? 2u : 1u) || op->num_outputs != 1u ||
        op->bias_idx != TIGRIS_NO_WEIGHT ||
        op->weight_idx == TIGRIS_NO_WEIGHT ||
        op->weight_idx >= plan->header->num_weights ||
        !plan->weight_entries ||
        (dtype != 1u && dtype != 3u && dtype != 9u))
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
    /* Softmax is a reduction, but along the final stored dimension, which is
     * the channel axis in both NLC and NHWC. The length axis a tile cuts is
     * ahead of it, so a tile holds whole normalization rows and the operator
     * behaves like the shape-preserving unary ones here. */
    return type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH ||
           type == TIGRIS_OP_SOFTMAX ||
           type == TIGRIS_OP_LEAKY_RELU ||
           type == TIGRIS_OP_ELU ||
           type == TIGRIS_OP_LOG_SOFTMAX ||
           type == TIGRIS_OP_L2_NORMALIZATION ||
           type == TIGRIS_OP_ERF ||
           type == TIGRIS_OP_HARDSWISH ||
           type == TIGRIS_OP_ABS ||
           type == TIGRIS_OP_RSQRT ||
           type == TIGRIS_OP_NEG || type == TIGRIS_OP_LOGICAL_NOT || type == TIGRIS_OP_CAST ||
           type == TIGRIS_OP_EXP ||
           type == TIGRIS_OP_LOG ||
           type == TIGRIS_OP_SQRT ||
           type == TIGRIS_OP_SQUARE ||
           type == TIGRIS_OP_FLOOR ||
           type == TIGRIS_OP_CEIL ||
           type == TIGRIS_OP_ROUND ||
           type == TIGRIS_OP_SIN ||
           type == TIGRIS_OP_COS ||
           type == TIGRIS_OP_LAYER_NORM;
}

/**
 * The rows and columns a tensor presents to a row band, or 0 if it presents
 * none. Mirrors tensor_row_view in the executor.
 */
static int tensor_row_view(
    const tigris_plan_t *plan, const tigris_tensor_t *t,
    int32_t *batch, int32_t *rows, int32_t *cols)
{
    if (plan->shape_pool == NULL)
        return 0;
    if (t->ndim < 2u)
        return 0;
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
 * Whether an operator keeps a rank-2 matrix's rows independent, so a band of
 * rows computes exactly the rows it holds. Mirrors is_row_tiling_op in the
 * executor.
 */
/* Mirrors is_whole_operand_op in the executor. */
static int is_whole_operand_op(uint8_t type)
{
    return type == TIGRIS_OP_FULLY_CONN || type == TIGRIS_OP_MATMUL;
}

static int is_row_tiling_op(uint8_t type)
{
    return type == TIGRIS_OP_FULLY_CONN ||
           type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH ||
           type == TIGRIS_OP_ERF ||
           type == TIGRIS_OP_HARDSWISH ||
           type == TIGRIS_OP_ABS ||
           type == TIGRIS_OP_RSQRT ||
           type == TIGRIS_OP_NEG || type == TIGRIS_OP_LOGICAL_NOT || type == TIGRIS_OP_CAST ||
           type == TIGRIS_OP_EXP ||
           type == TIGRIS_OP_LOG ||
           type == TIGRIS_OP_SQRT ||
           type == TIGRIS_OP_SQUARE ||
           type == TIGRIS_OP_FLOOR ||
           type == TIGRIS_OP_CEIL ||
           type == TIGRIS_OP_ROUND ||
           type == TIGRIS_OP_SIN ||
           type == TIGRIS_OP_COS ||
           type == TIGRIS_OP_SOFTMAX ||
           type == TIGRIS_OP_LEAKY_RELU ||
           type == TIGRIS_OP_ELU ||
           type == TIGRIS_OP_LOG_SOFTMAX ||
           type == TIGRIS_OP_L2_NORMALIZATION ||
           type == TIGRIS_OP_LAYER_NORM ||
           type == TIGRIS_OP_RESHAPE ||
           type == TIGRIS_OP_FLATTEN ||
           type == TIGRIS_OP_MATMUL ||
           type == TIGRIS_OP_ADD ||
           type == TIGRIS_OP_SUB ||
           type == TIGRIS_OP_DIV ||
           type == TIGRIS_OP_SQUARED_DIFFERENCE ||
           type == TIGRIS_OP_MAXIMUM ||
           type == TIGRIS_OP_MINIMUM ||
           type == TIGRIS_OP_FLOOR_DIV ||
           type == TIGRIS_OP_FLOOR_MOD ||
           type == TIGRIS_OP_PRELU ||
           (type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_LOGICAL_OR) ||
           type == TIGRIS_OP_SELECT_V2 || type == TIGRIS_OP_ADD_N ||
           type == TIGRIS_OP_MUL;
}


uint8_t tigris_band_axis(const tigris_plan_t *plan, const tigris_tensor_t *tensor, int mode)
{
    if (mode != TIGRIS_BAND_LEADING) return mode ? (uint8_t)(tensor->ndim - 2u) : 1u;
    const int32_t *shape = tigris_tensor_shape(plan, tensor);
    uint8_t axis = 0u;
    while (axis + 1u < tensor->ndim && shape[axis] == 1) axis++;
    return axis;
}


int tigris_op_independent_band(const tigris_plan_t *plan, const tigris_op_t *op,
                                uint16_t op_index, int row_tiled)
{
    if (op->num_inputs == 0u || op->num_outputs != 1u) return 0;
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *out = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    if (in->ndim < 2u || in->ndim > 6u || out->ndim < 2u || out->ndim > 6u) return 0;
    int32_t ib, ir, ic, ob, orows, oc;
    uint8_t row = tigris_band_axis(plan, in, row_tiled);
    if (row_tiled == TIGRIS_BAND_LEADING) {
        const int32_t *first = tigris_tensor_shape(plan, in);
        if (first[row] <= 1 || out->ndim != in->ndim) return 0;
        for (uint8_t i = 0u; i <= op->num_inputs; i++) {
            const tigris_tensor_t *t = i == op->num_inputs ? out :
                &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (t->ndim != in->ndim) return 0;
            const int32_t *shape = tigris_tensor_shape(plan, t);
            for (uint8_t a = 0u; a < row; a++) if (shape[a] != 1) return 0;
            if (shape[row] != first[row]) return 0;
        }
    } else if (row_tiled) {
        if (!tensor_row_view(plan, in, &ib, &ir, &ic) ||
            !tensor_row_view(plan, out, &ob, &orows, &oc) || ib != ob || ir != orows) return 0;
        for (uint8_t i = 1u; i < op->num_inputs; i++) {
            const tigris_tensor_t *t = &plan->tensors[tigris_op_inputs(plan, op)[i]];
            if (!tensor_row_view(plan, t, &ob, &orows, &oc) || ib != ob || ir != orows) return 0;
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
        if (op->num_inputs != 1u || op->weight_idx == TIGRIS_NO_WEIGHT) return 0;
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
    return type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE && op->weight_idx == TIGRIS_NO_WEIGHT &&
           op->num_inputs == 2u && length == (uint32_t)in->ndim * 8u &&
           metadata[row] == 0 && metadata[in->ndim + row] == shape[row];
}

int32_t tigris_stage_leading_band(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (!plan->shape_pool || stage->ops_count != 1u || stage->chain_len != 0u ||
        stage->inputs_count == 0u || stage->outputs_count != 1u) return 0;
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    if (op->num_inputs != stage->inputs_count || op->num_outputs != 1u ||
        tigris_op_outputs(plan, op)[0] != tigris_stage_outputs(plan, stage)[0] ||
        !tigris_op_independent_band(plan, op, index, TIGRIS_BAND_LEADING)) return 0;
    for (uint8_t i = 0u; i < op->num_inputs; i++) {
        int found = 0;
        for (uint16_t j = 0u; j < stage->inputs_count; j++)
            if (tigris_op_inputs(plan, op)[i] == tigris_stage_inputs(plan, stage)[j]) found = 1;
        if (!found) return 0;
    }
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const int32_t *shape = tigris_tensor_shape(plan, in);
    /* Preserve the meaning of existing matrix-row and height records. */
    if ((shape[in->ndim - 2u] > 1 && tigris_op_independent_band(plan, op, index, 1)) ||
        tigris_op_independent_band(plan, op, index, 0)) return 0;
    return shape[tigris_band_axis(plan, in, TIGRIS_BAND_LEADING)];
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

static int is_axis1_binary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_ADD || type == TIGRIS_OP_SUB ||
           type == TIGRIS_OP_DIV ||
           type == TIGRIS_OP_SQUARED_DIFFERENCE ||
           type == TIGRIS_OP_MAXIMUM ||
           type == TIGRIS_OP_MINIMUM ||
           type == TIGRIS_OP_FLOOR_DIV ||
           type == TIGRIS_OP_FLOOR_MOD ||
           type == TIGRIS_OP_PRELU ||
           (type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_LOGICAL_OR) ||
           type == TIGRIS_OP_SELECT_V2 || type == TIGRIS_OP_ADD_N ||
           type == TIGRIS_OP_MUL;
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
    switch (kind) {
    case TIGRIS_OP_ATTR_COMPARISON_REQUANT: return op_type >= TIGRIS_OP_EQUAL && op_type <= TIGRIS_OP_GREATER_EQUAL;
    case TIGRIS_OP_ATTR_MOVEMENT: return op_type >= TIGRIS_OP_GATHER && op_type <= TIGRIS_OP_DYNAMIC_UPDATE_SLICE;
    case TIGRIS_OP_ATTR_TRANSPOSE_PERM: return op_type == TIGRIS_OP_TRANSPOSE;
    case TIGRIS_OP_ATTR_EPSILON:        return op_type == TIGRIS_OP_LAYER_NORM ||
                                                op_type == TIGRIS_OP_L2_NORMALIZATION;
    case TIGRIS_OP_ATTR_ALPHA:          return op_type == TIGRIS_OP_LEAKY_RELU;
    case TIGRIS_OP_ATTR_CLIP_BOUNDS:    return op_type == TIGRIS_OP_CLIP;
    case TIGRIS_OP_ATTR_PADS:           return op_type == TIGRIS_OP_PAD;
    case TIGRIS_OP_ATTR_AXES:           return op_type == TIGRIS_OP_REDUCE_MEAN ||
                                                op_type == TIGRIS_OP_REDUCE_MAX ||
                                                op_type == TIGRIS_OP_REDUCE_MIN ||
                                                op_type == TIGRIS_OP_REDUCE_SUM ||
                                                op_type == TIGRIS_OP_CUMSUM ||
                                                op_type == TIGRIS_OP_ARG_MAX ||
                                                op_type == TIGRIS_OP_ARG_MIN ||
                                                op_type == TIGRIS_OP_REDUCE_ALL;
    case TIGRIS_OP_ATTR_CUMSUM_OPTIONS: return op_type == TIGRIS_OP_CUMSUM;
    case TIGRIS_OP_ATTR_POOL_ROUNDING:  return op_type == TIGRIS_OP_GLOBAL_AVG;
    case TIGRIS_OP_ATTR_RESIZE_SCALES:  return op_type == TIGRIS_OP_RESIZE ||
                                             op_type == TIGRIS_OP_RESIZE_LINEAR;
    case TIGRIS_OP_ATTR_BINARY_REQUANT: return op_type == TIGRIS_OP_ADD ||
                                               op_type == TIGRIS_OP_SUB;
    case TIGRIS_OP_ATTR_CONSTANT_OPERAND: return is_binary_op(op_type);
    case TIGRIS_OP_ATTR_CONSTANTS:      return op_type == TIGRIS_OP_SVDF || op_type == TIGRIS_OP_LSTM;
    case TIGRIS_OP_ATTR_SVDF:           return op_type == TIGRIS_OP_SVDF;
    case TIGRIS_OP_ATTR_LSTM:           return op_type == TIGRIS_OP_LSTM;
    default:                            return 0;
    }
}

/* Schemas v2/v3 let a custom dispatcher implement Transpose, MatMul and
 * opcodes the built-in table did not define yet, so only the built-in
 * operators whose meaning those schemas already fixed are validated there. */
static int legacy_semantics_are_builtin(uint8_t op_type)
{
    switch ((tigris_op_type_t)op_type) {
    case TIGRIS_OP_CONV:
    case TIGRIS_OP_DEPTHWISE:
    case TIGRIS_OP_CONV1D:
    case TIGRIS_OP_FULLY_CONN:
    case TIGRIS_OP_CONV_TRANSPOSE:
    case TIGRIS_OP_RELU:
    case TIGRIS_OP_RELU6:
    case TIGRIS_OP_SIGMOID:
    case TIGRIS_OP_TANH:
    case TIGRIS_OP_SOFTMAX:
    case TIGRIS_OP_RESHAPE:
    case TIGRIS_OP_FLATTEN:
    case TIGRIS_OP_REDUCE_MEAN:
    case TIGRIS_OP_ADD:
    case TIGRIS_OP_SUB:
    case TIGRIS_OP_MUL:
    case TIGRIS_OP_MAX_POOL:
    case TIGRIS_OP_AVG_POOL:
    case TIGRIS_OP_GLOBAL_MAX:
    case TIGRIS_OP_GLOBAL_AVG:
    case TIGRIS_OP_CONCAT:
    case TIGRIS_OP_RESIZE:
        return 1;
    default:
        return 0;
    }
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
    if (!payload || length == 0u || length > sizeof(metadata) || length % 4u != 0u ||
        op->num_outputs != 1u || op->num_inputs == 0u ||
        op->bias_idx != TIGRIS_NO_WEIGHT)
        return 0;
    memcpy(metadata, payload, length);
    const tigris_tensor_t *input = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *output = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    const int32_t *shape = tigris_tensor_shape(plan, input);
    const int32_t *target = tigris_tensor_shape(plan, output);
    uint8_t rank = input->ndim;
    uint8_t written = 0u;
    if (rank == 0u || rank > 6u || output->ndim > 6u ||
        (input->dtype != 1u && input->dtype != 3u) || input->dtype != output->dtype ||
        (input->dtype == 3u && !same_quantization(tigris_tensor_quant(plan, input), tigris_tensor_quant(plan, output))))
        return 0;
    if (movement_is_gather(op->op_type)) {
        if (op->num_inputs != (op->weight_idx == TIGRIS_NO_WEIGHT ? 2u : 1u)) return 0;
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
        if (op->weight_idx != TIGRIS_NO_WEIGHT) {
            if (plan->weight_entries[op->weight_idx].size_bytes != indices * 4u) return 0;
        } else {
            const tigris_tensor_t *index_tensor = &plan->tensors[tigris_op_inputs(plan, op)[1]];
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

typedef struct {
    uint8_t inputs[3];
    uint8_t output;
} dtype_signature_t;


uint8_t tigris_plan_data_dtype(const tigris_plan_t *plan)
{
    uint8_t dtype = 0u;
    for (uint16_t i = 0; i < plan->header->num_tensors; i++) {
        const tigris_tensor_t *tensor = &plan->tensors[i];
        if ((tensor->flags & TIGRIS_TENSOR_CONSTANT) != 0u ||
            (tensor->dtype != 1u && tensor->dtype != 3u)) continue;
        if (dtype != 0u && dtype != tensor->dtype) return 0u;
        dtype = tensor->dtype;
    }
    return dtype == 0u ? 1u : dtype;
}

/* Role 0 is the plan's data dtype, 255 data or bool, 254 data or int16 state
 * (the tensor table admits int16 only on state tensors). */
/* The weight index of an operator's constant operand k, or TIGRIS_NO_WEIGHT. */
static uint16_t constant_index(const tigris_plan_t *plan, uint16_t op_index, uint8_t k)
{
    uint8_t len = 0u;
    const uint8_t *list = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &len);
    if (!list || (uint32_t)k * 2u + 2u > len) return TIGRIS_NO_WEIGHT;
    return (uint16_t)((uint16_t)list[(uint32_t)k * 2u] |
                      (uint16_t)((uint16_t)list[(uint32_t)k * 2u + 1u] << 8));
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
    if (role == 254u) return dtype == data_dtype || dtype == 5u;
    return role == 255u ? dtype == data_dtype || dtype == 9u : dtype == (role == 0u ? data_dtype : role);
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
    const uint8_t data_dtype = tigris_plan_data_dtype(plan);
    const int comparison = type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_GREATER_EQUAL;
    if (op->num_inputs != arity && op->num_inputs != arity - 1u) return 0;
    if (op->num_inputs == arity && (constant != NULL || op->weight_idx != TIGRIS_NO_WEIGHT)) return 0;
    if (op->num_inputs != arity && (constant == NULL || constant[0] >= arity)) return 0;
    const tigris_quant_param_t *qy = tigris_tensor_quant(plan, y);
    uint8_t input_slot = 0u;
    for (uint8_t k = 0u; k < arity; k++) {
        const uint8_t dtype = comparison || (arity == 3u && k != 0u) ? data_dtype : 9u;
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
    if (comparison && data_dtype == 3u)
        return requant != NULL && requant_len == TIGRIS_OP_ATTR_COMPARISON_REQUANT_LEN;
    return requant == NULL;
}

static tigris_error_t validate_operator_semantics(const tigris_plan_t *plan)
{
    /* Zero is the data dtype, 255 also permits bool; the final input slot repeats. */
    static const dtype_signature_t dtype_signatures[TIGRIS_OP_LSTM + 1] = {
        [TIGRIS_OP_CONV] = {{0, 0, 0}, 0},
        [TIGRIS_OP_DEPTHWISE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_RELU] = {{0, 0, 0}, 0},
        [TIGRIS_OP_RELU6] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MAX_POOL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_AVG_POOL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ADD] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MUL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_FULLY_CONN] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SOFTMAX] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CLIP] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SIGMOID] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CONCAT] = {{0, 0, 0}, 0},
        [TIGRIS_OP_PAD] = {{0, 0, 0}, 0},
        [TIGRIS_OP_GLOBAL_AVG] = {{0, 0, 0}, 0},
        [TIGRIS_OP_FLATTEN] = {{255, 255, 255}, 255},
        [TIGRIS_OP_RESHAPE] = {{255, 255, 255}, 255},
        [TIGRIS_OP_SUB] = {{0, 0, 0}, 0},
        [TIGRIS_OP_DIV] = {{0, 0, 0}, 0},
        [TIGRIS_OP_TANH] = {{0, 0, 0}, 0},
        [TIGRIS_OP_LEAKY_RELU] = {{0, 0, 0}, 0},
        [TIGRIS_OP_BATCH_NORM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_INST_NORM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CONV_TRANSPOSE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MATMUL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REDUCE_MEAN] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SQUEEZE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_UNSQUEEZE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_TRANSPOSE] = {{255, 255, 255}, 255},
        [TIGRIS_OP_RESIZE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_GLOBAL_MAX] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CONV1D] = {{0, 0, 0}, 0},
        [TIGRIS_OP_LAYER_NORM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ERF] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SPLIT] = {{0, 0, 0}, 0},
        [TIGRIS_OP_RESIZE_LINEAR] = {{0, 0, 0}, 0},
        [TIGRIS_OP_HARDSWISH] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ABS] = {{0, 0, 0}, 0},
        [TIGRIS_OP_RSQRT] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SQUARED_DIFFERENCE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MAXIMUM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MINIMUM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_NEG] = {{0, 0, 0}, 0},
        [TIGRIS_OP_EXP] = {{0, 0, 0}, 0},
        [TIGRIS_OP_LOG] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SQRT] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SQUARE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_FLOOR] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CEIL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ROUND] = {{0, 0, 0}, 0},
        [TIGRIS_OP_SIN] = {{0, 0, 0}, 0},
        [TIGRIS_OP_COS] = {{0, 0, 0}, 0},
        [TIGRIS_OP_FLOOR_DIV] = {{0, 0, 0}, 0},
        [TIGRIS_OP_FLOOR_MOD] = {{0, 0, 0}, 0},
        [TIGRIS_OP_PRELU] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ELU] = {{0, 0, 0}, 0},
        [TIGRIS_OP_LOG_SOFTMAX] = {{0, 0, 0}, 0},
        [TIGRIS_OP_L2_NORMALIZATION] = {{0, 0, 0}, 0},
        [TIGRIS_OP_L2_POOL] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REDUCE_MAX] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REDUCE_MIN] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REDUCE_SUM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_CUMSUM] = {{0, 0, 0}, 0},
        [TIGRIS_OP_ARG_MAX] = {{0, 0, 0}, 6},
        [TIGRIS_OP_ARG_MIN] = {{0, 0, 0}, 6},
        [TIGRIS_OP_GATHER] = {{0, 6, 6}, 0},
        [TIGRIS_OP_GATHER_ND] = {{0, 6, 6}, 0},
        [TIGRIS_OP_STRIDED_SLICE] = {{0, 0, 0}, 0},
        [TIGRIS_OP_MIRROR_PAD] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REVERSE_V2] = {{0, 0, 0}, 0},
        [TIGRIS_OP_EMBEDDING_LOOKUP] = {{0, 6, 6}, 0},
        [TIGRIS_OP_DYNAMIC_UPDATE_SLICE] = {{0, 0, 6}, 0},
        [TIGRIS_OP_EQUAL] = {{0, 0, 0}, 9},
        [TIGRIS_OP_LESS] = {{0, 0, 0}, 9},
        [TIGRIS_OP_LESS_EQUAL] = {{0, 0, 0}, 9},
        [TIGRIS_OP_GREATER] = {{0, 0, 0}, 9},
        [TIGRIS_OP_GREATER_EQUAL] = {{0, 0, 0}, 9},
        [TIGRIS_OP_LOGICAL_AND] = {{9, 9, 9}, 9},
        [TIGRIS_OP_LOGICAL_OR] = {{9, 9, 9}, 9},
        [TIGRIS_OP_LOGICAL_NOT] = {{9, 9, 9}, 9},
        [TIGRIS_OP_SELECT_V2] = {{9, 0, 0}, 0},
        [TIGRIS_OP_CAST] = {{9, 9, 9}, 0},
        [TIGRIS_OP_ADD_N] = {{0, 0, 0}, 0},
        [TIGRIS_OP_REDUCE_ALL] = {{9, 9, 9}, 9},
        [TIGRIS_OP_SVDF] = {{0, 254, 254}, 254},
        [TIGRIS_OP_LSTM] = {{0, 254, 254}, 254},
    };
    const tigris_file_header_t *hdr = plan->header;
    const int legacy = hdr->version < TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES;
    const uint8_t data_dtype = tigris_plan_data_dtype(plan);
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
                if (producer->op_type >= sizeof(dtype_signatures) / sizeof(dtype_signatures[0]) ||
                    dtype_signatures[producer->op_type].output != plan->tensors[t].dtype)
                    return TIGRIS_ERR_BAD_OPERATOR;
                producers++;
            }
        }
        uint32_t expected = (plan->tensors[t].flags & TIGRIS_TENSOR_MODEL_INPUT) != 0u ? 0u : 1u;
        if (producers != expected) return TIGRIS_ERR_BAD_OPERATOR;
    }
    for (uint16_t op_index = 0; op_index < hdr->num_ops; op_index++) {
        const tigris_op_t *op = &plan->ops[op_index];
        if (legacy && !legacy_semantics_are_builtin(op->op_type))
            continue;
        if (op->num_inputs == 0 || op->num_outputs == 0 ||
            op->fused_act > TIGRIS_ACT_RELU6 || op->_pad1 != 0)
            return TIGRIS_ERR_BAD_OPERATOR;

        const uint16_t *inputs = tigris_op_inputs(plan, op);
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        const tigris_tensor_t *input = &plan->tensors[inputs[0]];
        const tigris_tensor_t *output = &plan->tensors[outputs[0]];
        uint8_t dtype = input->dtype;
        int index_output = op->op_type == TIGRIS_OP_ARG_MAX || op->op_type == TIGRIS_OP_ARG_MIN;
        if (op->op_type >= sizeof(dtype_signatures) / sizeof(dtype_signatures[0]))
            return TIGRIS_ERR_BAD_OPERATOR;
        const dtype_signature_t *signature = &dtype_signatures[op->op_type];
        uint8_t slot_constant_len = 0u;
        const uint8_t *slot_constant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &slot_constant_len);
        for (uint8_t i = 0; i < op->num_inputs; i++) {
            uint8_t position = i;
            if (slot_constant != NULL && slot_constant[0] <= position) position++;
            else if (op->op_type == TIGRIS_OP_DYNAMIC_UPDATE_SLICE &&
                     op->weight_idx != TIGRIS_NO_WEIGHT && position >= 1u) position++;
            uint8_t role = signature->inputs[position < 3u ? position : 2u];
            if (!dtype_matches_slot(plan->tensors[inputs[i]].dtype, role, data_dtype))
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        for (uint8_t i = 0; i < op->num_outputs; i++) {
            if (!dtype_matches_slot(plan->tensors[outputs[i]].dtype, signature->output, data_dtype))
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        if (dtype == 3 && op->act_min > op->act_max)
            return TIGRIS_ERR_BAD_OPERATOR;

        /* Add carries an activation because a quantizer writes the residual
         * block's Relu after the sum, on the edge that the sum's own output
         * requantization already covers. */
        int allows_fused_activation =
            op->op_type == TIGRIS_OP_CONV ||
            op->op_type == TIGRIS_OP_DEPTHWISE ||
            op->op_type == TIGRIS_OP_FULLY_CONN ||
            op->op_type == TIGRIS_OP_CONV1D ||
            op->op_type == TIGRIS_OP_ADD ||
            (op->op_type == TIGRIS_OP_SVDF && dtype == 1u);
        if (!allows_fused_activation && op->fused_act != TIGRIS_ACT_NONE)
            return TIGRIS_ERR_BAD_OPERATOR;

        switch ((tigris_op_type_t)op->op_type) {
        case TIGRIS_OP_EQUAL:
        case TIGRIS_OP_LESS:
        case TIGRIS_OP_LESS_EQUAL:
        case TIGRIS_OP_GREATER:
        case TIGRIS_OP_GREATER_EQUAL:
        case TIGRIS_OP_LOGICAL_AND:
        case TIGRIS_OP_LOGICAL_OR:
        case TIGRIS_OP_LOGICAL_NOT:
        case TIGRIS_OP_SELECT_V2:
        case TIGRIS_OP_CAST:
        case TIGRIS_OP_ADD_N:
            if (!bool_and_sum_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
            break;
        case TIGRIS_OP_GATHER:
        case TIGRIS_OP_GATHER_ND:
        case TIGRIS_OP_STRIDED_SLICE:
        case TIGRIS_OP_MIRROR_PAD:
        case TIGRIS_OP_REVERSE_V2:
        case TIGRIS_OP_EMBEDDING_LOOKUP:
        case TIGRIS_OP_DYNAMIC_UPDATE_SLICE:
            if (!movement_metadata_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
            break;
        case TIGRIS_OP_SVDF:
            if (!svdf_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
            break;
        case TIGRIS_OP_LSTM:
            if (!lstm_valid(plan, op, op_index)) return TIGRIS_ERR_BAD_OPERATOR;
            break;
        case TIGRIS_OP_CONV:
        case TIGRIS_OP_DEPTHWISE: {
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
            break;
        }

        case TIGRIS_OP_CONV1D: {
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
            break;
        }

        case TIGRIS_OP_FULLY_CONN: {
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
            break;
        }

        case TIGRIS_OP_RELU:
        case TIGRIS_OP_RELU6:
        case TIGRIS_OP_SIGMOID:
        case TIGRIS_OP_TANH:
        case TIGRIS_OP_SOFTMAX:
            if (!op_has_plain_io(op, 1, 1) ||
                !tensor_shapes_equal(plan, input, output) ||
                (op->op_type == TIGRIS_OP_SOFTMAX && input->ndim == 0))
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_RESHAPE:
        case TIGRIS_OP_FLATTEN:
            if (!op_has_plain_io(op, 1, 1) ||
                input->size_bytes != output->size_bytes || input->dtype != output->dtype)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_TRANSPOSE:
            if (!op_has_plain_io(op, 1, 1) ||
                input->size_bytes != output->size_bytes || input->dtype != output->dtype)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_LEAKY_RELU:
        case TIGRIS_OP_ELU:
        case TIGRIS_OP_LOG_SOFTMAX:
        case TIGRIS_OP_L2_NORMALIZATION:
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
            break;

        case TIGRIS_OP_ABS:
        case TIGRIS_OP_RSQRT:
        case TIGRIS_OP_NEG:
        case TIGRIS_OP_EXP:
        case TIGRIS_OP_LOG:
        case TIGRIS_OP_SQRT:
        case TIGRIS_OP_SQUARE:
        case TIGRIS_OP_FLOOR:
        case TIGRIS_OP_CEIL:
        case TIGRIS_OP_ROUND:
        case TIGRIS_OP_SIN:
        case TIGRIS_OP_COS:
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
            break;

        case TIGRIS_OP_DIV:
        case TIGRIS_OP_SQUARED_DIFFERENCE:
        case TIGRIS_OP_MAXIMUM:
        case TIGRIS_OP_MINIMUM:
        case TIGRIS_OP_FLOOR_DIV:
        case TIGRIS_OP_FLOOR_MOD:
        case TIGRIS_OP_PRELU:
        {
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
            break;
        }

        case TIGRIS_OP_PAD: {
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
            break;
        }

        case TIGRIS_OP_HARDSWISH:
        case TIGRIS_OP_ERF:
            if (!op_has_plain_io(op, 1, 1) ||
                input->ndim != output->ndim ||
                input->size_bytes != output->size_bytes)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_SPLIT: {
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
            break;
        }

        case TIGRIS_OP_CUMSUM: {
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
            break;
        }

        case TIGRIS_OP_ARG_MAX:
        case TIGRIS_OP_ARG_MIN:
        case TIGRIS_OP_REDUCE_MAX:
        case TIGRIS_OP_REDUCE_MIN:
        case TIGRIS_OP_REDUCE_SUM:
        case TIGRIS_OP_REDUCE_ALL:
        case TIGRIS_OP_REDUCE_MEAN: {
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
            break;
        }

        case TIGRIS_OP_LAYER_NORM: {
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
            break;
        }

        case TIGRIS_OP_MATMUL: {
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
            break;
        }

        case TIGRIS_OP_ADD:
        case TIGRIS_OP_SUB:
        case TIGRIS_OP_MUL:
            /* Each operand has the output's shape or broadcasts to it. */
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
            break;

        case TIGRIS_OP_MAX_POOL:
        case TIGRIS_OP_AVG_POOL:
        case TIGRIS_OP_L2_POOL: {
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
            break;
        }

        case TIGRIS_OP_GLOBAL_MAX:
        case TIGRIS_OP_GLOBAL_AVG: {
            if (!op_has_plain_io(op, 1, 1) ||
                input->ndim != 4 || output->ndim != 4)
                return TIGRIS_ERR_BAD_OPERATOR;
            const int32_t *in_shape = tigris_tensor_shape(plan, input);
            const int32_t *out_shape = tigris_tensor_shape(plan, output);
            if (out_shape[0] != in_shape[0] || out_shape[1] != 1 ||
                out_shape[2] != 1 || out_shape[3] != in_shape[3])
                return TIGRIS_ERR_BAD_OPERATOR;
            break;
        }

        case TIGRIS_OP_CONCAT: {
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
                break;
            }
            if (along != (uint64_t)out_shape[axis] ||
                covered != output->size_bytes)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;
        }

        case TIGRIS_OP_CONV_TRANSPOSE: {
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
            break;
        }

        case TIGRIS_OP_RESIZE_LINEAR:
        case TIGRIS_OP_RESIZE: {
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
            break;
        }

        default:
            return TIGRIS_ERR_BAD_OPERATOR;
        }
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

tigris_error_t tigris_plan_load_ex(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan,
    tigris_plan_limits_t *out_required)
{
    /* A plan that exceeds this build is refused, but the refusal says which
     * build would run it, so the requirement is gathered as the tables are
     * walked rather than returned from the first breach. */
    tigris_plan_limits_t required;
    memset(&required, 0, sizeof(required));
    if (out_required)
        memset(out_required, 0, sizeof(*out_required));

    if (!out_plan)
        return TIGRIS_ERR_NULL;

    /* Never expose pointers from a plan that later fails validation. */
    memset(out_plan, 0, sizeof(*out_plan));
    if (!buf)
        return TIGRIS_ERR_NULL;
    tigris_plan_t candidate;
    memset(&candidate, 0, sizeof(candidate));

    /* Verify little-endian platform (plan format assumes LE) */
    {
        const uint32_t endian_test = 1;
        if (*(const uint8_t *)&endian_test != 1)
            return TIGRIS_ERR_ENDIAN;
    }

    /* Header validation */

    if (buf_len < sizeof(tigris_file_header_t))
        return TIGRIS_ERR_TOO_SMALL;

    const tigris_file_header_t *hdr = (const tigris_file_header_t *)buf;

    if (memcmp(hdr->magic, TIGRIS_MAGIC_BYTES, 4) != 0)
        return TIGRIS_ERR_BAD_MAGIC;

    if (hdr->version < TIGRIS_SCHEMA_VERSION_MIN ||
        hdr->version > TIGRIS_SCHEMA_VERSION)
        return TIGRIS_ERR_BAD_VERSION;

    if (hdr->file_size != buf_len)
        return TIGRIS_ERR_BAD_SIZE;

    candidate.header = hdr;

    /* Section directory */

    uint32_t sec_off = hdr->section_dir_off;
    if (!bounds_ok(sec_off, sizeof(tigris_section_entry_t), buf_len))
        return TIGRIS_ERR_BAD_SECTION;

    /* Track which sections we found via offsets into buf (0 = not found) */
    uint32_t section_offsets[TIGRIS_SEC_MAX];
    uint32_t section_ends[TIGRIS_SEC_MAX];
    uint8_t section_seen[TIGRIS_SEC_MAX];
    memset(section_offsets, 0, sizeof(section_offsets));
    memset(section_ends, 0, sizeof(section_ends));
    memset(section_seen, 0, sizeof(section_seen));

    int found_sentinel = 0;
    uint32_t sec_cursor = sec_off;
    uint32_t previous_type = 0;
    uint32_t previous_offset = 0;
    while (bounds_ok(sec_cursor, sizeof(tigris_section_entry_t), buf_len)) {
        const tigris_section_entry_t *sec =
            (const tigris_section_entry_t *)(buf + sec_cursor);
        if (sec->type == 0)
        {
            found_sentinel = 1;
            break;
        }

        if (sec->type >= TIGRIS_SEC_MAX)
            return TIGRIS_ERR_BAD_SECTION;

        if (section_seen[sec->type])
            return TIGRIS_ERR_BAD_SECTION;

        /* The writer emits an ordered directory and non-overlapping section
         * starts. Empty sections may share an offset with the next section. */
        if (sec->type <= previous_type || sec->offset < previous_offset ||
            sec->offset > buf_len)
            return TIGRIS_ERR_BAD_SECTION;

        section_seen[sec->type] = 1;
        section_offsets[sec->type] = sec->offset;
        previous_type = sec->type;
        previous_offset = sec->offset;
        sec_cursor += sizeof(tigris_section_entry_t);
    }

    if (!found_sentinel)
        return TIGRIS_ERR_BAD_SECTION;

    /* No section may overlap the parsed directory. Derive each section's
     * exclusive end from the next directory entry, including empty sections. */
    uint32_t dir_end = sec_cursor + sizeof(tigris_section_entry_t);
    for (uint32_t i = 1; i < TIGRIS_SEC_MAX; i++) {
        if (!section_seen[i])
            continue;
        if (section_offsets[i] < dir_end)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t end = buf_len;
        for (uint32_t j = i + 1; j < TIGRIS_SEC_MAX; j++) {
            if (section_seen[j] &&
                section_offsets[j] < end)
                end = section_offsets[j];
        }
        section_ends[i] = end;
    }

    /* Required sections */

    if (!section_offsets[TIGRIS_SEC_TENSORS])     return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_OPS])         return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_INDEX_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_SHAPE_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_STRINGS])     return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_stages && !section_offsets[TIGRIS_SEC_STAGES])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_tile_plans && !section_offsets[TIGRIS_SEC_TILE_PLANS])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_weights && !section_offsets[TIGRIS_SEC_WEIGHTS])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_quant_params && !section_offsets[TIGRIS_SEC_QUANT_PARAMS])
        return TIGRIS_ERR_MISSING_SEC;

    /* The on-disk tables are otherwise byte-addressable, but these pools are
     * dereferenced through uint16_t/int32_t pointers below.  Reject malformed
     * offsets before exposing an unaligned pointer to a target that faults on
     * it (or to UBSan during loader fuzzing). */
    if ((section_offsets[TIGRIS_SEC_INDEX_POOL] % sizeof(uint16_t)) != 0 &&
        (hdr->num_ops || hdr->num_stages || hdr->num_model_inputs ||
         hdr->num_model_outputs))
        return TIGRIS_ERR_BAD_SECTION;
    if ((section_offsets[TIGRIS_SEC_SHAPE_POOL] % sizeof(int32_t)) != 0 &&
        hdr->num_tensors)
        return TIGRIS_ERR_BAD_SECTION;
    if ((section_offsets[TIGRIS_SEC_QUANT_PARAMS] % sizeof(int32_t)) != 0 &&
        hdr->num_quant_params)
        return TIGRIS_ERR_BAD_SECTION;

    /* Set pointers */

    /* Tensors */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_TENSORS];
        uint32_t need = (uint32_t)hdr->num_tensors * sizeof(tigris_tensor_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_TENSORS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.tensors = (const tigris_tensor_t *)(buf + off);
    }

    /* Ops */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_OPS];
        uint32_t need = (uint32_t)hdr->num_ops * sizeof(tigris_op_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_OPS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.ops = (const tigris_op_t *)(buf + off);
    }

    /* Stages (optional - 0 stages is valid for un-partitioned graphs) */
    if (section_offsets[TIGRIS_SEC_STAGES] && hdr->num_stages > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_STAGES];
        uint32_t need = (uint32_t)hdr->num_stages * sizeof(tigris_stage_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_STAGES]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.stages = (const tigris_stage_t *)(buf + off);
    }

    /* Tile plans (optional) */
    if (section_offsets[TIGRIS_SEC_TILE_PLANS] && hdr->num_tile_plans > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_TILE_PLANS];
        uint32_t need = (uint32_t)hdr->num_tile_plans * sizeof(tigris_tile_plan_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_TILE_PLANS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.tile_plans = (const tigris_tile_plan_t *)(buf + off);
        for (uint16_t i = 0; i < hdr->num_tile_plans; i++) {
            const tigris_tile_plan_t *tile = &candidate.tile_plans[i];
            if (tile->tileable > 1)
                return TIGRIS_ERR_BAD_SECTION;
            if (hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS) {
                if ((tile->tileable &&
                     tile->axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH &&
                     tile->axis != TIGRIS_TILE_AXIS_HW) ||
                    (!tile->tileable &&
                     tile->axis != TIGRIS_TILE_AXIS_NONE))
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
    }

    /* Weights (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHTS] && hdr->num_weights > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights * sizeof(tigris_weight_entry_t);
        if (!bounds_ok(off, entries_size, section_ends[TIGRIS_SEC_WEIGHTS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.weight_entries = (const tigris_weight_entry_t *)(buf + off);
        /* weight_blob set below only if no compressed blocks */
    }

    /* Weight blocks - compressed per-stage weight data (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS]) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        if (!bounds_ok(off, 4, section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t num_blocks, compression;
        memcpy(&num_blocks, buf + off, 2);
        memcpy(&compression, buf + off + 2, 2);
        candidate.num_weight_blocks = num_blocks;
        candidate.weight_compression = compression;
        uint32_t entries_size = (uint32_t)num_blocks * sizeof(tigris_weight_block_t);
        if (!bounds_ok(off + 4, entries_size, section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.weight_blocks = (const tigris_weight_block_t *)(buf + off + 4);
        candidate.weight_blocks_data = buf + off + 4 + entries_size;
    }

    /* Per-operator attributes (optional, schema v4+). Compare against the
     * feature's introduction version, not the latest schema: otherwise a
     * future schema bump would accidentally make valid v4 plans fail. */
    if (section_offsets[TIGRIS_SEC_OP_ATTRIBUTES]) {
        if (hdr->version < TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t off = section_offsets[TIGRIS_SEC_OP_ATTRIBUTES];
        if (!bounds_ok(off, 4, section_ends[TIGRIS_SEC_OP_ATTRIBUTES]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t count;
        memcpy(&count, buf + off, sizeof(count));
        uint32_t entries_size = (uint32_t)count * sizeof(tigris_op_attribute_t);
        if (count == 0 ||
            !bounds_ok(off + 4u, entries_size,
                       section_ends[TIGRIS_SEC_OP_ATTRIBUTES]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.num_op_attributes = count;
        candidate.op_attributes =
            (const tigris_op_attribute_t *)(buf + off + 4u);
        candidate.op_attribute_data = buf + off + 4u + entries_size;
    }

    /* Variables kept across invocations (optional, schema v10+): their
     * entries and initial values, checked against the tensors below. */
    if (section_offsets[TIGRIS_SEC_STATE]) {
        if (hdr->version < TIGRIS_SCHEMA_VERSION_STATE)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t off = section_offsets[TIGRIS_SEC_STATE];
        uint32_t end = section_ends[TIGRIS_SEC_STATE];
        if (!bounds_ok(off, 8u, end))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t count;
        uint16_t reserved;
        memcpy(&count, buf + off, sizeof(count));
        memcpy(&reserved, buf + off + 2u, sizeof(reserved));
        memcpy(&candidate.state_bytes, buf + off + 4u, sizeof(candidate.state_bytes));
        uint32_t entries_size = (uint32_t)count * sizeof(tigris_state_entry_t);
        if (count == 0u || reserved != 0u || !bounds_ok(off + 8u, entries_size, end))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.num_state = count;
        candidate.state_entries = (const tigris_state_entry_t *)(buf + off + 8u);
        candidate.state_section = buf + off;
        for (uint16_t i = 0; i < count; i++) {
            const tigris_state_entry_t *entry = &candidate.state_entries[i];
            if (entry->bytes == 0u || (entry->offset % 4u) != 0u ||
                entry->offset > candidate.state_bytes ||
                entry->bytes > candidate.state_bytes - entry->offset ||
                !bounds_ok(off + entry->initial, entry->bytes, end))
                return TIGRIS_ERR_BAD_SECTION;
            for (uint16_t j = 0; j < i; j++) {
                const tigris_state_entry_t *other = &candidate.state_entries[j];
                if (entry->offset < other->offset + other->bytes &&
                    other->offset < entry->offset + entry->bytes)
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
    }

    /* Set weight_blob for XIP only when no compressed blocks */
    if (candidate.weight_entries && !candidate.weight_blocks) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights * sizeof(tigris_weight_entry_t);
        candidate.weight_blob = buf + off + entries_size;
    }

    /* Quant params (optional) */
    if (section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t off = section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        /* First 4 bytes: uint16_t num_quant_params + quant-data length (v2)
         * or page count (v3). */
        if (!bounds_ok(off, 4, section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t nqp, qd_field;
        memcpy(&nqp, buf + off, 2);
        memcpy(&qd_field, buf + off + 2, 2);
        candidate.num_quant_params = nqp;
        /* The header states a count and so does the section. Everything below
         * bounds a quant_param_idx against one of them and indexes the array
         * sized by the other, so they have to be reconciled here rather than
         * after the tables that read them. A header claiming more than the
         * section holds read past the plan. */
        if (nqp != hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t entries_size = (uint32_t)nqp * sizeof(tigris_quant_param_t);
        uint32_t data_size;
        if (hdr->version == TIGRIS_SCHEMA_VERSION_V2)
            data_size = (uint32_t)qd_field * sizeof(int32_t);
        else {
            if (section_ends[TIGRIS_SEC_QUANT_PARAMS] < off + 4u + entries_size)
                return TIGRIS_ERR_BAD_SECTION;
            data_size = section_ends[TIGRIS_SEC_QUANT_PARAMS] - off - 4u - entries_size;
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
                       section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        candidate.quant_params = (const tigris_quant_param_t *)(buf + off + 4);
        if (data_size > 0)
            candidate.quant_data = (const int32_t *)(buf + off + 4 + entries_size);
    }

    /* Index pool */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_INDEX_POOL];
        candidate.index_pool = (const uint16_t *)(buf + off);
    }

    /* Shape pool */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_SHAPE_POOL];
        candidate.shape_pool = (const int32_t *)(buf + off);
    }

    /* String table */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_STRINGS];
        candidate.strings = (const char *)(buf + off);
    }

    /* Validate every table span and cross-reference before exposing the plan
     * to the executor. Section boundaries above ensure these counts cannot
     * borrow bytes from the next table. */
    uint32_t index_bytes = section_ends[TIGRIS_SEC_INDEX_POOL] -
                           section_offsets[TIGRIS_SEC_INDEX_POOL];
    uint32_t shape_bytes = section_ends[TIGRIS_SEC_SHAPE_POOL] -
                           section_offsets[TIGRIS_SEC_SHAPE_POOL];
    uint32_t strings_len = section_ends[TIGRIS_SEC_STRINGS] -
                           section_offsets[TIGRIS_SEC_STRINGS];
    if (((index_bytes % sizeof(uint16_t)) != 0 &&
         (hdr->num_ops || hdr->num_stages || hdr->num_model_inputs ||
          hdr->num_model_outputs)) ||
        ((shape_bytes % sizeof(int32_t)) != 0 && hdr->num_tensors) ||
        strings_len == 0)
        return TIGRIS_ERR_BAD_SECTION;
    uint32_t index_count = index_bytes / sizeof(uint16_t);
    uint32_t shape_count = shape_bytes / sizeof(int32_t);

    if (!string_ok(candidate.strings, strings_len, hdr->model_name_str))
        return TIGRIS_ERR_BAD_SECTION;

    uint32_t model_io_count = (uint32_t)hdr->num_model_inputs +
                              (uint32_t)hdr->num_model_outputs;
    if (!elements_ok(hdr->model_io_off, model_io_count, index_count))
        return TIGRIS_ERR_BAD_SECTION;
    candidate.model_inputs  = candidate.index_pool + hdr->model_io_off;
    candidate.model_outputs = candidate.model_inputs + hdr->num_model_inputs;
    for (uint32_t i = 0; i < model_io_count; i++) {
        if (candidate.model_inputs[i] >= hdr->num_tensors)
            return TIGRIS_ERR_BAD_SECTION;
    }

    note_requirement(&required.tensors, hdr->num_tensors);
    if (hdr->num_tensors > TIGRIS_MAX_TENSORS) {
        if (out_required)
            *out_required = required;
        return TIGRIS_ERR_PLAN_LIMITS;
    }

    for (uint16_t i = 0; i < hdr->num_tensors; i++) {
        const tigris_tensor_t *tensor = &candidate.tensors[i];
        if (!string_ok(candidate.strings, strings_len, tensor->name_str) ||
            !elements_ok(tensor->shape_off, tensor->ndim, shape_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM &&
            tensor->quant_param_idx >= hdr->num_quant_params)
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
            if (hdr->version < TIGRIS_SCHEMA_VERSION_INTERFACE_DTYPE ||
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
            int32_t value = candidate.shape_pool[tensor->shape_off + dim];
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
                &candidate.quant_params[tensor->quant_param_idx];
            if (!isfinite(qp->scale) || qp->scale <= 0.0f ||
                qp->zero_point < -128 || qp->zero_point > 127)
                return TIGRIS_ERR_BAD_TENSOR;
        }
    }

    for (uint8_t i = 0; i < hdr->num_model_inputs; i++) {
        if (!(candidate.tensors[candidate.model_inputs[i]].flags &
              TIGRIS_TENSOR_MODEL_INPUT))
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint8_t i = 0; i < hdr->num_model_outputs; i++) {
        if (!(candidate.tensors[candidate.model_outputs[i]].flags &
              TIGRIS_TENSOR_MODEL_OUTPUT))
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint16_t tensor_idx = 0; tensor_idx < hdr->num_tensors; tensor_idx++) {
        const tigris_tensor_t *tensor = &candidate.tensors[tensor_idx];
        int found_input = 0;
        int found_output = 0;
        for (uint8_t i = 0; i < hdr->num_model_inputs; i++)
            found_input |= candidate.model_inputs[i] == tensor_idx;
        for (uint8_t i = 0; i < hdr->num_model_outputs; i++)
            found_output |= candidate.model_outputs[i] == tensor_idx;
        if (((tensor->flags & TIGRIS_TENSOR_MODEL_INPUT) != 0) != found_input ||
            ((tensor->flags & TIGRIS_TENSOR_MODEL_OUTPUT) != 0) != found_output)
            return TIGRIS_ERR_BAD_TENSOR;
    }

    /* Every state tensor is one variable's value on entry or on exit, the
     * two alike, sized as the entry states. */
    for (uint16_t tensor_idx = 0; tensor_idx < hdr->num_tensors; tensor_idx++) {
        uint16_t roles = 0u;
        for (uint16_t i = 0; i < candidate.num_state; i++) {
            const tigris_state_entry_t *entry = &candidate.state_entries[i];
            if (entry->input == tensor_idx)
                roles++;
            if (entry->output == tensor_idx)
                roles++;
        }
        const uint16_t flagged =
            ((candidate.tensors[tensor_idx].flags & TIGRIS_TENSOR_STATE) != 0u) ? 1u : 0u;
        if (roles != flagged)
            return TIGRIS_ERR_BAD_TENSOR;
    }
    for (uint16_t i = 0; i < candidate.num_state; i++) {
        const tigris_state_entry_t *entry = &candidate.state_entries[i];
        if (entry->input >= hdr->num_tensors ||
            (entry->output != 0xFFFFu && entry->output >= hdr->num_tensors))
            return TIGRIS_ERR_BAD_TENSOR;
        const tigris_tensor_t *in = &candidate.tensors[entry->input];
        if (in->size_bytes != entry->bytes || (in->dtype != 1u && in->dtype != 3u && in->dtype != 5u))
            return TIGRIS_ERR_BAD_TENSOR;
        if (entry->output != 0xFFFFu) {
            const tigris_tensor_t *out = &candidate.tensors[entry->output];
            if (out->dtype != in->dtype || out->size_bytes != in->size_bytes ||
                !tensor_shapes_equal(&candidate, in, out) ||
                (out->flags & TIGRIS_TENSOR_LINEAR) != (in->flags & TIGRIS_TENSOR_LINEAR) ||
                (in->dtype == 3u && !same_quantization(tigris_tensor_quant(&candidate, in),
                                                       tigris_tensor_quant(&candidate, out))))
                return TIGRIS_ERR_BAD_TENSOR;
        }
    }

    for (uint16_t i = 0; i < hdr->num_ops; i++) {
        const tigris_op_t *op = &candidate.ops[i];
        if (!string_ok(candidate.strings, strings_len, op->name_str) ||
            !elements_ok(op->inputs_off, op->num_inputs, index_count) ||
            !elements_ok(op->outputs_off, op->num_outputs, index_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (hdr->version < TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
            hdr->num_stages && op->stage >= hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        if ((op->weight_idx != TIGRIS_NO_WEIGHT &&
             op->weight_idx >= hdr->num_weights) ||
            (op->bias_idx != TIGRIS_NO_WEIGHT &&
             op->bias_idx >= hdr->num_weights))
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t j = 0; j < op->num_inputs; j++) {
            if (candidate.index_pool[op->inputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
        for (uint8_t j = 0; j < op->num_outputs; j++) {
            if (candidate.index_pool[op->outputs_off + j] >= hdr->num_tensors)
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
    if (candidate.op_attributes) {
        uint32_t attrs_off = section_offsets[TIGRIS_SEC_OP_ATTRIBUTES];
        uint32_t attrs_data_off = attrs_off + 4u +
            (uint32_t)candidate.num_op_attributes *
            sizeof(tigris_op_attribute_t);
        uint32_t attrs_data_len = section_ends[TIGRIS_SEC_OP_ATTRIBUTES] -
            attrs_data_off;
        uint16_t previous_op = 0;
        uint8_t previous_attr_type = 0;
        for (uint16_t i = 0; i < candidate.num_op_attributes; i++) {
            const tigris_op_attribute_t *attr = &candidate.op_attributes[i];
            if (attr->op_index >= hdr->num_ops ||
                (i > 0 &&
                 (attr->op_index < previous_op ||
                  (attr->op_index == previous_op &&
                   attr->type <= previous_attr_type))) ||
                !bounds_ok(attr->data_offset, attr->data_len, attrs_data_len))
                return TIGRIS_ERR_BAD_SECTION;
            const tigris_op_t *op = &candidate.ops[attr->op_index];
            /* Every kind a schema-8 plan may carry is validated here whether
             * or not an operator consumes it yet, so an operator behind one
             * of them lands without a further version. A kind this build does
             * not know still fails closed, in the default arm. */
            if (attr->type > TIGRIS_OP_ATTR_TRANSPOSE_PERM &&
                hdr->version < TIGRIS_SCHEMA_VERSION_V8)
                return TIGRIS_ERR_BAD_SECTION;
            /* Svdf and Lstm also write their next states. */
            if (op->num_inputs < 1u ||
                (op->num_outputs != 1u && !(op->op_type == TIGRIS_OP_SVDF && op->num_outputs == 2u) &&
                 !(op->op_type == TIGRIS_OP_LSTM && op->num_outputs == 3u)))
                return TIGRIS_ERR_BAD_SECTION;
            /* A kind belongs to the operator that reads it, whether or not a
             * kernel for that operator exists yet, so a stray payload cannot
             * ride along on an unrelated op. */
            if (!attr_kind_matches_op(attr->type, op->op_type))
                return TIGRIS_ERR_BAD_SECTION;
            uint16_t input_idx = candidate.index_pool[op->inputs_off];
            uint16_t output_idx = candidate.index_pool[op->outputs_off];
            const tigris_tensor_t *input = &candidate.tensors[input_idx];
            const tigris_tensor_t *output = &candidate.tensors[output_idx];
            switch (attr->type) {
            case TIGRIS_OP_ATTR_TRANSPOSE_PERM: {
                if (op->num_inputs != 1u ||
                    attr->data_len != input->ndim ||
                    output->ndim != input->ndim ||
                    output->size_bytes != input->size_bytes)
                    return TIGRIS_ERR_BAD_SECTION;
                const uint8_t *perm =
                    candidate.op_attribute_data + attr->data_offset;
                const int32_t *in_shape =
                    tigris_tensor_shape(&candidate, input);
                const int32_t *out_shape =
                    tigris_tensor_shape(&candidate, output);
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
            case TIGRIS_OP_ATTR_ALPHA: {
                float value;
                if (attr->data_len != sizeof(value))
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(&value,
                       candidate.op_attribute_data + attr->data_offset,
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
            case TIGRIS_OP_ATTR_RESIZE_SCALES: {
                float scales[2];
                if (attr->data_len != sizeof(scales) || input->ndim != 4u || output->ndim != 4u)
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(scales, candidate.op_attribute_data + attr->data_offset, sizeof(scales));
                const int32_t *in_shape = tigris_tensor_shape(&candidate, input);
                const int32_t *out_shape = tigris_tensor_shape(&candidate, output);
                for (int axis = 0; axis < 2; axis++) {
                    if (!isfinite(scales[axis]) || scales[axis] <= 0.0f ||
                        floorf((float)in_shape[axis + 1] * scales[axis]) !=
                            (float)out_shape[axis + 1])
                        return TIGRIS_ERR_BAD_SECTION;
                }
                break;
            }
            case TIGRIS_OP_ATTR_CLIP_BOUNDS: {
                float bounds[2];
                if (attr->data_len != sizeof(bounds))
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(bounds,
                       candidate.op_attribute_data + attr->data_offset,
                       sizeof(bounds));
                if (!isfinite(bounds[0]) || !isfinite(bounds[1]) ||
                    bounds[0] > bounds[1])
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_PADS: {
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
                           candidate.op_attribute_data + attr->data_offset +
                               byte,
                           sizeof(pad));
                    if (pad < 0)
                        return TIGRIS_ERR_BAD_SECTION;
                }
                break;
            }
            case TIGRIS_OP_ATTR_COMPARISON_REQUANT: {
                int32_t values[5];
                if (attr->data_len != TIGRIS_OP_ATTR_COMPARISON_REQUANT_LEN)
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(values, candidate.op_attribute_data + attr->data_offset, sizeof(values));
                if (values[0] != 8 || values[1] < 0 || values[3] < 0 ||
                    values[2] < -31 || values[2] > 0 || values[4] < -31 || values[4] > 0)
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_BINARY_REQUANT: {
                /* Three Q0.31 pairs. A shift outside the range the requant
                 * helper can apply would shift by more than the width of the
                 * accumulator, which is undefined rather than saturating. */
                if (attr->data_len != TIGRIS_OP_ATTR_BINARY_REQUANT_LEN ||
                    hdr->version < TIGRIS_SCHEMA_VERSION_BINARY_REQUANT)
                    return TIGRIS_ERR_BAD_SECTION;
                for (uint8_t pair = 0; pair < 3u; pair++) {
                    int32_t multiplier;
                    int32_t shift;
                    memcpy(&multiplier,
                           candidate.op_attribute_data + attr->data_offset +
                               (uint32_t)pair * 8u,
                           sizeof(multiplier));
                    memcpy(&shift,
                           candidate.op_attribute_data + attr->data_offset +
                               (uint32_t)pair * 8u + 4u,
                           sizeof(shift));
                    if (multiplier < 0 || shift < -31 || shift > 31)
                        return TIGRIS_ERR_BAD_SECTION;
                }
                break;
            }
            case TIGRIS_OP_ATTR_CONSTANT_OPERAND: {
                /* Operand position, a zero byte, a quant param index and
                 * optionally a shape, which the operator's own validation
                 * resolves. */
                const uint8_t *payload = candidate.op_attribute_data + attr->data_offset;
                if (attr->data_len < TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN ||
                    (attr->data_len - TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) % 4u != 0u ||
                    payload[0] > (op->op_type == TIGRIS_OP_SELECT_V2 ? 2u : 1u) || payload[1] != 0u)
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_POOL_ROUNDING: {
                if (attr->data_len != 1u ||
                    candidate.op_attribute_data[attr->data_offset] !=
                        TIGRIS_POOL_ROUNDING_AVERAGE)
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_MOVEMENT:
                if (!movement_metadata_valid(&candidate, op, attr->op_index)) return TIGRIS_ERR_BAD_SECTION;
                break;
            case TIGRIS_OP_ATTR_CONSTANTS: {
                const uint8_t *list = candidate.op_attribute_data + attr->data_offset;
                if (attr->data_len == 0u || (attr->data_len % 2u) != 0u)
                    return TIGRIS_ERR_BAD_SECTION;
                for (uint8_t byte = 0; byte < attr->data_len; byte += 2u) {
                    uint16_t index = (uint16_t)((uint16_t)list[byte] |
                                                (uint16_t)((uint16_t)list[byte + 1u] << 8));
                    if (index != TIGRIS_NO_WEIGHT && index >= hdr->num_weights)
                        return TIGRIS_ERR_BAD_SECTION;
                }
                break;
            }
            case TIGRIS_OP_ATTR_LSTM: {
                int32_t time_major;
                float clip;
                /* Float32, or int8 with its 26 integer parameters. */
                if (attr->data_len != 8u && attr->data_len != 112u)
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(&time_major, candidate.op_attribute_data + attr->data_offset, sizeof(time_major));
                memcpy(&clip, candidate.op_attribute_data + attr->data_offset + 4u, sizeof(clip));
                if ((time_major != 0 && time_major != 1) || !isfinite(clip) || clip < 0.0f)
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_SVDF: {
                int32_t values[6];
                if (attr->data_len != 4u && attr->data_len != sizeof(values))
                    return TIGRIS_ERR_BAD_SECTION;
                memcpy(values, candidate.op_attribute_data + attr->data_offset, attr->data_len);
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
            case TIGRIS_OP_ATTR_CUMSUM_OPTIONS: {
                const uint8_t *options = candidate.op_attribute_data + attr->data_offset;
                if (attr->data_len != 2u || options[0] > 1u || options[1] > 1u)
                    return TIGRIS_ERR_BAD_SECTION;
                break;
            }
            case TIGRIS_OP_ATTR_AXES: {
                if (attr->data_len == 0u || attr->data_len > input->ndim)
                    return TIGRIS_ERR_BAD_SECTION;
                const uint8_t *axes =
                    candidate.op_attribute_data + attr->data_offset;
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
            previous_op = attr->op_index;
            previous_attr_type = attr->type;
        }
        uint16_t attr_cursor = 0;
        for (uint16_t op_idx = 0; op_idx < hdr->num_ops; op_idx++) {
            int has_transpose_perm = 0;
            int has_epsilon = 0;
            int has_axes = 0;
            int has_cumsum_options = 0;
            while (attr_cursor < candidate.num_op_attributes &&
                   candidate.op_attributes[attr_cursor].op_index == op_idx) {
                if (candidate.op_attributes[attr_cursor].type ==
                    TIGRIS_OP_ATTR_TRANSPOSE_PERM)
                    has_transpose_perm = 1;
                if (candidate.op_attributes[attr_cursor].type ==
                    TIGRIS_OP_ATTR_EPSILON)
                    has_epsilon = 1;
                if (candidate.op_attributes[attr_cursor].type ==
                    TIGRIS_OP_ATTR_AXES)
                    has_axes = 1;
                if (candidate.op_attributes[attr_cursor].type == TIGRIS_OP_ATTR_CUMSUM_OPTIONS)
                    has_cumsum_options = 1;
                attr_cursor++;
            }
            if ((candidate.ops[op_idx].op_type == TIGRIS_OP_TRANSPOSE) !=
                has_transpose_perm)
                return TIGRIS_ERR_BAD_SECTION;
            /* A normalization divides by a variance floor it cannot default,
             * so the attribute is required rather than optional. */
            if ((candidate.ops[op_idx].op_type == TIGRIS_OP_LAYER_NORM ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_L2_NORMALIZATION) !=
                has_epsilon)
                return TIGRIS_ERR_BAD_SECTION;
            /* A reduction cannot default the axes it collapses either. */
            if ((candidate.ops[op_idx].op_type == TIGRIS_OP_CUMSUM) != has_cumsum_options)
                return TIGRIS_ERR_BAD_SECTION;
            if ((candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MEAN ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MAX ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MIN ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_SUM ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_CUMSUM ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_ARG_MAX ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_ARG_MIN ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_ALL) !=
                has_axes)
                return TIGRIS_ERR_BAD_SECTION;
        }
    } else if (hdr->version >= TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES) {
        /* No attribute section at all, so every operator that requires one is
         * refused here on the same terms the loop above applies when there
         * is a section. */
        for (uint16_t op_idx = 0; op_idx < hdr->num_ops; op_idx++) {
            if (candidate.ops[op_idx].op_type == TIGRIS_OP_TRANSPOSE ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_LAYER_NORM ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_L2_NORMALIZATION ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MEAN ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MAX ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_MIN ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_SUM ||
                candidate.ops[op_idx].op_type == TIGRIS_OP_CUMSUM ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_ARG_MAX ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_ARG_MIN ||
                 candidate.ops[op_idx].op_type == TIGRIS_OP_REDUCE_ALL)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (candidate.weight_entries && !candidate.weight_blocks) {
        uint32_t weights_off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights *
                                sizeof(tigris_weight_entry_t);
        uint32_t blob_len = section_ends[TIGRIS_SEC_WEIGHTS] -
                            weights_off - entries_size;
        for (uint16_t i = 0; i < hdr->num_weights; i++) {
            const tigris_weight_entry_t *weight = &candidate.weight_entries[i];
            if (!string_ok(candidate.strings, strings_len, weight->name_str) ||
                !bounds_ok(weight->offset, weight->size_bytes, blob_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t qp_off = section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        uint16_t nqp, qd_field;
        memcpy(&nqp, buf + qp_off, sizeof(nqp));
        memcpy(&qd_field, buf + qp_off + sizeof(nqp), sizeof(qd_field));
        uint32_t qd_len = hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? qd_field :
            (section_ends[TIGRIS_SEC_QUANT_PARAMS] - qp_off - 4u -
             (uint32_t)nqp * sizeof(tigris_quant_param_t)) / sizeof(int32_t);
        for (uint16_t i = 0; i < nqp; i++) {
            const tigris_quant_param_t *qp = &candidate.quant_params[i];
            uint32_t page = hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
            uint32_t moff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off;
            uint32_t soff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off;
            if (!isfinite(qp->scale) || qp->scale <= 0.0f)
                return TIGRIS_ERR_BAD_TENSOR;
            if (qp->num_channels == 0 ||
                (hdr->version != TIGRIS_SCHEMA_VERSION_V2 &&
                 (page >= qd_field ||
                  (uint32_t)qp->multiplier_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS ||
                  (uint32_t)qp->shift_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS)) ||
                !elements_ok(moff, qp->num_channels, qd_len) ||
                !elements_ok(soff, qp->num_channels, qd_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (candidate.weight_blocks) {
        uint32_t wb_off = section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        uint32_t block_entries = (uint32_t)candidate.num_weight_blocks *
                                 sizeof(tigris_weight_block_t);
        uint32_t blobs_off = wb_off + 4u + block_entries;
        uint32_t blobs_len = section_ends[TIGRIS_SEC_WEIGHT_BLOCKS] - blobs_off;
        if (candidate.weight_compression != TIGRIS_COMPRESS_NONE &&
            candidate.weight_compression != TIGRIS_COMPRESS_LZ4)
            return TIGRIS_ERR_BAD_SECTION;
        if (candidate.num_weight_blocks == 0 || hdr->num_weights == 0 ||
            candidate.num_weight_blocks > hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t previous_stage = TIGRIS_NO_CHAIN;
        for (uint16_t i = 0; i < candidate.num_weight_blocks; i++) {
            const tigris_weight_block_t *block = &candidate.weight_blocks[i];
            if (block->stage_idx >= hdr->num_stages ||
                (i > 0 && block->stage_idx <= previous_stage) ||
                !elements_ok(block->first_weight_idx, block->num_weights,
                             hdr->num_weights) ||
                block->num_weights == 0 || block->compressed_size == 0 ||
                block->uncompressed_size == 0 ||
                !bounds_ok(block->blob_offset, block->compressed_size, blobs_len) ||
                (candidate.weight_compression == TIGRIS_COMPRESS_NONE &&
                 block->compressed_size != block->uncompressed_size))
                return TIGRIS_ERR_BAD_SECTION;
            previous_stage = block->stage_idx;
        }
    }

    /* Plan limits validation */

    uint16_t next_scheduled_op = 0;
    for (uint16_t i = 0; i < hdr->num_stages; i++) {
        const tigris_stage_t *stage = &candidate.stages[i];
        if (!elements_ok(stage->ops_off, stage->ops_count, index_count) ||
            !elements_ok(stage->inputs_off, stage->inputs_count, index_count) ||
            !elements_ok(stage->outputs_off, stage->outputs_count, index_count) ||
            (stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN &&
             stage->tile_plan_idx >= hdr->num_tile_plans) ||
            (stage->chain_len == 0 && stage->chain_id != TIGRIS_NO_CHAIN) ||
            (stage->chain_len > 0 && stage->chain_id == TIGRIS_NO_CHAIN))
            return TIGRIS_ERR_BAD_SECTION;
        note_requirement(&required.stage_inputs, stage->inputs_count);
        note_requirement(&required.stage_outputs, stage->outputs_count);
        note_requirement(&required.chain_stages, stage->chain_len);
        if (stage->inputs_count > TIGRIS_MAX_STAGE_INPUTS ||
            stage->outputs_count > TIGRIS_MAX_STAGE_OUTPUTS ||
            stage->chain_len > TIGRIS_MAX_CHAIN_STAGES) {
            if (out_required)
                *out_required = required;
            return TIGRIS_ERR_PLAN_LIMITS;
        }

        /* A chain is represented redundantly on every member.  The executor
         * only starts it at its declared head and skips the other members, so
         * accepting an incomplete or displaced chain silently drops work. */
        if (stage->chain_len > 0) {
            if (stage->chain_len < 2 || stage->chain_id > i ||
                stage->chain_len > hdr->num_stages - stage->chain_id)
                return TIGRIS_ERR_BAD_SECTION;
            const tigris_stage_t *head = &candidate.stages[stage->chain_id];
            if (head->chain_id != stage->chain_id ||
                head->chain_len != stage->chain_len ||
                head->chain_tile_h == 0 ||
                stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN ||
                (i != stage->chain_id && stage->chain_tile_h != 0))
                return TIGRIS_ERR_BAD_SECTION;
            if (i == stage->chain_id) {
                for (uint16_t member = i + 1u;
                     member < i + stage->chain_len; member++) {
                    const tigris_stage_t *next = &candidate.stages[member];
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
            uint16_t op_idx = candidate.index_pool[stage->ops_off + j];
            if (op_idx >= hdr->num_ops)
                return TIGRIS_ERR_BAD_SECTION;
            /* The binary plan contract is a forward, partitioned schedule.
             * Enforcing it here prevents a malformed stage list from silently
             * omitting work or running the same operation more than once. */
            if (op_idx != next_scheduled_op)
                return TIGRIS_ERR_BAD_SECTION;
            /* In schema v5 the stage table is authoritative and the legacy
             * operator byte is a canonical low-byte hint. This preserves the
             * zero-copy v2-v4 record layout without imposing a 256-stage
             * ceiling on larger v5 plans. */
            if ((hdr->version < TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
                 candidate.ops[op_idx].stage != i) ||
                (hdr->version >= TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY &&
                 candidate.ops[op_idx].stage != (uint8_t)i))
                return TIGRIS_ERR_BAD_OPERATOR;
            next_scheduled_op++;
        }
        for (uint16_t j = 0; j < stage->inputs_count; j++) {
            if (candidate.index_pool[stage->inputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
        for (uint16_t j = 0; j < stage->outputs_count; j++) {
            if (candidate.index_pool[stage->outputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }

        if (stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN) {
            const tigris_tile_plan_t *tile =
                &candidate.tile_plans[stage->tile_plan_idx];
            uint8_t axis =
                hdr->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS
                    ? tile->axis
                    : (tile->tileable
                           ? TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH
                           : TIGRIS_TILE_AXIS_NONE);
            tigris_reshape_band_t reshape_view;
            int reshape = tigris_reshape_band(&candidate, stage, &reshape_view);
            if (reshape && tile->tileable && !tigris_reshape_tile_valid(tile, &reshape_view))
                return TIGRIS_ERR_BAD_OPERATOR;
            int32_t leading_rows = tigris_stage_leading_band(&candidate, stage);
            if (leading_rows && tile->tileable) {
                tigris_reshape_band_t band = {1, leading_rows, 1, leading_rows, 1};
                if (!tigris_reshape_tile_valid(tile, &band)) return TIGRIS_ERR_BAD_OPERATOR;
            }
            if (!reshape && !leading_rows && tile->tileable &&
                axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH) {
                if (stage->inputs_count == 0 || stage->outputs_count == 0)
                    return TIGRIS_ERR_BAD_SECTION;
                uint16_t first_input =
                    candidate.index_pool[stage->inputs_off];
                uint16_t first_output =
                    candidate.index_pool[stage->outputs_off];
                /* A row-banded stage is checked on its own terms first: its
                 * tensors need not share a rank, only a batch and a row
                 * count, because a Reshape that regroups the leading axes
                 * moves no data. The outputs settle the band, since an output
                 * is always banded; an input the band does not cut is read
                 * whole, which only a matrix product's second operand is
                 * entitled to be. Mirrors stage_is_row_tiled in the executor,
                 * and accepting a stage it will decline sends the stage
                 * through the normal path instead of refusing the plan. */
                int32_t band_batch = -1;
                int32_t band_rows = -1;
                int32_t band_cols = 0;
                int row_banded = stage->chain_len == 0 &&
                                 stage->outputs_count > 0;
                if (row_banded) {
                    for (uint16_t j = 0; j < stage->outputs_count; j++) {
                        int32_t b = 0;
                        int32_t r = 0;
                        if (!tensor_row_view(
                                &candidate,
                                &candidate.tensors[
                                    candidate.index_pool[
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
                        const tigris_op_t *rop = &candidate.ops[
                            candidate.index_pool[stage->ops_off + j]];
                        int32_t b = 0;
                        int32_t r = 0;
                        if (rop->num_inputs == 0u || rop->num_outputs != 1u ||
                            (!is_row_tiling_op(rop->op_type) && !tigris_op_independent_band(
                                &candidate, rop, candidate.index_pool[stage->ops_off + j], 1)) ||
                            !tensor_row_view(
                                &candidate,
                                &candidate.tensors[
                                    candidate.index_pool[rop->outputs_off]],
                                &b, &r, &band_cols) ||
                            b != band_batch || r != band_rows) {
                            row_banded = 0;
                            break;
                        }
                        for (uint8_t k = 0; k < rop->num_inputs; k++) {
                            int32_t ib = 0;
                            int32_t ir = 0;
                            int32_t ic = 0;
                            if (tensor_row_view(
                                    &candidate,
                                    &candidate.tensors[
                                        candidate.index_pool[
                                            rop->inputs_off + k]],
                                    &ib, &ir, &ic) &&
                                ib == band_batch && ir == band_rows)
                                continue;
                            if (k == 0u ||
                                !is_whole_operand_op(rop->op_type)) {
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
                        if (!tensor_row_view(
                                &candidate,
                                &candidate.tensors[
                                    candidate.index_pool[
                                        stage->inputs_off + j]],
                                &b, &r, &band_cols)) {
                            row_banded = 0;
                            break;
                        }
                    }
                }

                uint8_t rank = candidate.tensors[first_input].ndim;
                int transpose = rank >= 3u && rank <= 6u && stage->ops_count == 1u && stage->chain_len == 0u &&
                    candidate.ops[candidate.index_pool[stage->ops_off]].op_type == TIGRIS_OP_TRANSPOSE;
                if (!row_banded &&
                    ((!transpose && rank != 3 && rank != 4) ||
                     candidate.tensors[first_output].ndim != rank))
                    return TIGRIS_ERR_BAD_SECTION;
                /* A row band agrees on rows, not on rank: the Reshape pair
                 * around a lowered matrix product drops a unit leading axis
                 * without moving a byte, and its row count is checked above. */
                if (!row_banded) {
                    for (uint16_t j = 0; j < stage->inputs_count; j++) {
                        uint16_t tensor_idx =
                            candidate.index_pool[stage->inputs_off + j];
                        if (candidate.tensors[tensor_idx].ndim != rank)
                            return TIGRIS_ERR_BAD_SECTION;
                    }
                    for (uint16_t j = 0; j < stage->outputs_count; j++) {
                        uint16_t tensor_idx =
                            candidate.index_pool[stage->outputs_off + j];
                        if (candidate.tensors[tensor_idx].ndim != rank)
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
                    const tigris_op_t *bop = &candidate.ops[
                        candidate.index_pool[stage->ops_off + j]];
                    if (is_axis1_binary_pointwise_op(bop->op_type) &&
                        bop->num_inputs >= 2) {
                        const uint16_t *bin = tigris_op_inputs(&candidate, bop);
                        for (uint8_t operand = 1; operand < bop->num_inputs; operand++) {
                            if (!tensor_shapes_equal(&candidate,
                                                     &candidate.tensors[bin[0]],
                                                     &candidate.tensors[bin[operand]]) &&
                                (stage->chain_len != 0 || tile->tile_width != 0 ||
                                 rank != 4 || row_banded))
                                return TIGRIS_ERR_BAD_OPERATOR;
                        }
                    }
                }

                if (row_banded) {
                    /* Already validated on its own terms above. */
                } else if (stage->ops_count == 1 &&
                    candidate.ops[
                        candidate.index_pool[stage->ops_off]
                    ].op_type == TIGRIS_OP_TRANSPOSE) {
                    /* A transpose is banded along the longer of the two
                     * extents it swaps, and only as the whole stage.
                     * transpose_band_groups states which permutations that
                     * is: the ones that swap two adjacent groups of axes and
                     * leave the rest in order. The executor reads the same
                     * rule from the same header rather than from a copy. */
                    uint16_t conv_op = candidate.index_pool[stage->ops_off];
                    uint8_t perm_rank = 0;
                    const uint8_t *perm = tigris_op_attribute_data(
                        &candidate, conv_op,
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
                           tigris_op_independent_band(&candidate,
                               &candidate.ops[candidate.index_pool[stage->ops_off]],
                               candidate.index_pool[stage->ops_off], 0)) {
                    /* The operator preserves serialized height on every operand. */
                } else if (rank == 3) {
                    uint16_t conv_count = 0;
                    uint16_t binary_count = 0;
                    if (hdr->version < TIGRIS_SCHEMA_VERSION_TILE_AXIS ||
                        stage->chain_len != 0)
                        return TIGRIS_ERR_BAD_OPERATOR;
                    for (uint16_t j = 0; j < stage->ops_count; j++) {
                        uint8_t type = candidate.ops[
                            candidate.index_pool[stage->ops_off + j]
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
                           candidate.ops[
                               candidate.index_pool[stage->ops_off]
                           ].op_type == TIGRIS_OP_GLOBAL_AVG) {
                    /* A global reduction collapses the height the stripe
                     * contract propagates, so the executor walks its input
                     * instead. Permitted only as the whole stage and only
                     * when the output really is collapsed: with anything
                     * else present there would be a height to carry. */
                    const int32_t *out_shape = tigris_tensor_shape(
                        &candidate, &candidate.tensors[first_output]);
                    if (stage->inputs_count != 1 ||
                        stage->outputs_count != 1 ||
                        stage->chain_len != 0 ||
                        rank != 4 ||
                        out_shape[1] != 1 || out_shape[2] != 1)
                        return TIGRIS_ERR_BAD_OPERATOR;
                } else {
                    uint16_t spatial_count = 0;
                    for (uint16_t j = 0; j < stage->ops_count; j++) {
                        uint8_t type = candidate.ops[
                            candidate.index_pool[stage->ops_off + j]
                        ].op_type;
                        if (type == TIGRIS_OP_CONV ||
                            type == TIGRIS_OP_DEPTHWISE ||
                            type == TIGRIS_OP_MAX_POOL ||
                            type == TIGRIS_OP_AVG_POOL || type == TIGRIS_OP_L2_POOL) {
                            spatial_count++;
                        } else if (type == TIGRIS_OP_RESIZE ||
                                   type == TIGRIS_OP_RESIZE_LINEAR) {
                            /* A resample carries the height the other way:
                             * the tile loop divides to find its source band
                             * instead of multiplying. It counts as the
                             * stage's one spatial operator, and it may not
                             * be chained, because the chain executor
                             * composes a single receptive field across its
                             * members and has no fractional stride. */
                            if (stage->chain_len != 0)
                                return TIGRIS_ERR_BAD_OPERATOR;
                            spatial_count++;
                        } else if (type != TIGRIS_OP_RELU &&
                                   type != TIGRIS_OP_RELU6 &&
                                   type != TIGRIS_OP_SIGMOID &&
                                   type != TIGRIS_OP_TANH &&
                                   /* Elementwise, and a normalization whose
                                    * reduced axis both tile axes are ahead
                                    * of: shape-preserving and halo-free. */
                                   type != TIGRIS_OP_ERF &&
                                   type != TIGRIS_OP_HARDSWISH &&
                                   type != TIGRIS_OP_ABS &&
                                   type != TIGRIS_OP_RSQRT &&
                                   type != TIGRIS_OP_NEG &&
                                   type != TIGRIS_OP_EXP &&
                                   type != TIGRIS_OP_LOG &&
                                   type != TIGRIS_OP_SQRT &&
                                   type != TIGRIS_OP_SQUARE &&
                                   type != TIGRIS_OP_FLOOR &&
                                   type != TIGRIS_OP_CEIL &&
                                   type != TIGRIS_OP_ROUND &&
                                   type != TIGRIS_OP_SIN &&
                                   type != TIGRIS_OP_COS &&
                                   !(type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_ADD_N) &&
                                   type != TIGRIS_OP_DIV &&
                                   type != TIGRIS_OP_SQUARED_DIFFERENCE &&
                                   type != TIGRIS_OP_MAXIMUM &&
                                   type != TIGRIS_OP_MINIMUM &&
                                   type != TIGRIS_OP_FLOOR_DIV &&
                                   type != TIGRIS_OP_FLOOR_MOD &&
                                   type != TIGRIS_OP_LAYER_NORM &&
                                   /* Softmax normalizes along the final
                                    * stored dimension, which a height stripe
                                    * never cuts, so it belongs beside the
                                    * unary operators here exactly as it does
                                    * in the rank-3 contract above and in
                                    * is_height_tiling_op. */
                                   type != TIGRIS_OP_SOFTMAX &&
                                   type != TIGRIS_OP_LEAKY_RELU &&
                                   type != TIGRIS_OP_PRELU &&
                                   type != TIGRIS_OP_ELU &&
                                   type != TIGRIS_OP_LOG_SOFTMAX &&
                                   type != TIGRIS_OP_L2_NORMALIZATION &&
                                   type != TIGRIS_OP_ADD &&
                                   type != TIGRIS_OP_SUB &&
                                   type != TIGRIS_OP_MUL &&
                                   type != TIGRIS_OP_CONCAT) {
                            return TIGRIS_ERR_BAD_OPERATOR;
                        }
                    }
                    if (spatial_count > 1)
                        return TIGRIS_ERR_BAD_OPERATOR;
                }
            }
        }

        if (stage->chain_len >= 2) {
            if (stage->chain_id == TIGRIS_NO_CHAIN ||
                stage->chain_id >= hdr->num_stages ||
                stage->chain_len > hdr->num_stages - stage->chain_id)
                return TIGRIS_ERR_BAD_SECTION;

            /* Chain execution stores bounded spatial-op metadata in its
             * caller-provided workspace. Validate the stage-op slice and the
             * audited height-stripe operator contract before execution. */
            uint32_t pool_off = section_offsets[TIGRIS_SEC_INDEX_POOL];
            uint32_t prefix = (uint32_t)stage->ops_off * sizeof(uint16_t);
            uint32_t bytes = (uint32_t)stage->ops_count * sizeof(uint16_t);
            if (!bounds_ok(pool_off, prefix, buf_len) ||
                !bounds_ok(pool_off + prefix, bytes, buf_len))
                return TIGRIS_ERR_BAD_SECTION;

            const uint8_t *op_indices = buf + pool_off + prefix;
            uint32_t spatial_count = 0;
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                uint16_t op_idx;
                memcpy(&op_idx, op_indices + (uint32_t)j * sizeof(op_idx),
                       sizeof(op_idx));
                if (op_idx >= hdr->num_ops)
                    return TIGRIS_ERR_BAD_SECTION;

                uint8_t op_type = candidate.ops[op_idx].op_type;
                if (op_type == TIGRIS_OP_CONV ||
                    op_type == TIGRIS_OP_DEPTHWISE ||
                    op_type == TIGRIS_OP_MAX_POOL ||
                    op_type == TIGRIS_OP_AVG_POOL || op_type == TIGRIS_OP_L2_POOL) {
                    spatial_count++;
                    note_requirement(&required.spatial_ops_per_stage,
                                     spatial_count);
                    if (spatial_count > TIGRIS_MAX_SPATIAL_OPS_PER_STAGE) {
                        if (out_required)
                            *out_required = required;
                        return TIGRIS_ERR_PLAN_LIMITS;
                    }
                } else if (op_type != TIGRIS_OP_RELU &&
                           op_type != TIGRIS_OP_RELU6 &&
                           op_type != TIGRIS_OP_SIGMOID &&
                           op_type != TIGRIS_OP_TANH &&
                           /* The same shape-preserving, halo-free set the
                            * stripe contract admits above: a chain is a run
                            * of stripe-tileable stages, so its operator list
                            * is that one. */
                           op_type != TIGRIS_OP_SOFTMAX &&
                           op_type != TIGRIS_OP_LEAKY_RELU &&
                           op_type != TIGRIS_OP_PRELU &&
                           op_type != TIGRIS_OP_ELU &&
                           op_type != TIGRIS_OP_LOG_SOFTMAX &&
                           op_type != TIGRIS_OP_L2_NORMALIZATION &&
                           op_type != TIGRIS_OP_ERF &&
                           op_type != TIGRIS_OP_HARDSWISH &&
                           op_type != TIGRIS_OP_ABS &&
                           op_type != TIGRIS_OP_RSQRT &&
                           op_type != TIGRIS_OP_NEG &&
                           op_type != TIGRIS_OP_EXP &&
                           op_type != TIGRIS_OP_LOG &&
                           op_type != TIGRIS_OP_SQRT &&
                           op_type != TIGRIS_OP_SQUARE &&
                           op_type != TIGRIS_OP_FLOOR &&
                           op_type != TIGRIS_OP_CEIL &&
                           op_type != TIGRIS_OP_ROUND &&
                           op_type != TIGRIS_OP_SIN &&
                           op_type != TIGRIS_OP_COS &&
                           !(op_type >= TIGRIS_OP_EQUAL && op_type <= TIGRIS_OP_ADD_N) &&
                           op_type != TIGRIS_OP_DIV &&
                           op_type != TIGRIS_OP_SQUARED_DIFFERENCE &&
                           op_type != TIGRIS_OP_MAXIMUM &&
                           op_type != TIGRIS_OP_MINIMUM &&
                           op_type != TIGRIS_OP_FLOOR_DIV &&
                           op_type != TIGRIS_OP_FLOOR_MOD &&
                           op_type != TIGRIS_OP_LAYER_NORM &&
                           op_type != TIGRIS_OP_ADD &&
                           op_type != TIGRIS_OP_SUB &&
                           op_type != TIGRIS_OP_MUL &&
                           op_type != TIGRIS_OP_CONCAT) {
                    return TIGRIS_ERR_BAD_OPERATOR;
                }
            }
        }
    }

    if (next_scheduled_op != hdr->num_ops)
        return TIGRIS_ERR_BAD_SECTION;

    /* A compressed stage redirects weight_entry offsets into its own decoded
     * block.  Check that every referenced weight/bias is present in that
     * block before any executor or kernel can dereference the offset. */
    if (candidate.weight_blocks) {
        uint16_t block_cursor = 0;
        for (uint16_t s = 0; s < hdr->num_stages; s++) {
            const tigris_weight_block_t *block = NULL;
            if (block_cursor < candidate.num_weight_blocks &&
                candidate.weight_blocks[block_cursor].stage_idx == s) {
                block = &candidate.weight_blocks[block_cursor++];
            }

            const tigris_stage_t *stage = &candidate.stages[s];
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                const tigris_op_t *op =
                    &candidate.ops[candidate.index_pool[stage->ops_off + j]];
                uint16_t refs[2] = {op->weight_idx, op->bias_idx};
                for (uint8_t r = 0; r < 2; r++) {
                    if (refs[r] == TIGRIS_NO_WEIGHT)
                        continue;
                    if (!block ||
                        !bounds_ok(candidate.weight_entries[refs[r]].offset,
                                   candidate.weight_entries[refs[r]].size_bytes,
                                   block->uncompressed_size))
                        return TIGRIS_ERR_BAD_SECTION;
                }
            }
        }
    }

    {
        tigris_error_t semantic_error = validate_operator_semantics(&candidate);
        if (semantic_error != TIGRIS_OK)
            return semantic_error;
    }

    if (out_required)
        *out_required = required;

    *out_plan = candidate;
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
        default:                    return "unknown error";
    }
}

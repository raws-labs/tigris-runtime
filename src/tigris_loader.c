/**
 * @file tigris_loader.c
 * @brief Zero-alloc binary plan parser.
 *
 * Validates the file header, walks the section directory, and sets
 * pointers into the buffer. Never allocates memory.
 */

#include "tigris_loader.h"
#include "tigris_band.h"
#include "tigris_binary_operands.h"
#include "tigris_op_config.h"
#include "tigris_op_contract.h"

#include "tigris_transpose_band.h"

#include <math.h>
#include <string.h>

/* Helpers */

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

static int op_has_no_weights(const tigris_op_t *op)
{
    return op->weight_idx == TIGRIS_NO_WEIGHT && op->bias_idx == TIGRIS_NO_WEIGHT;
}

static const tigris_tensor_t *op_first_output(const tigris_plan_t *plan, const tigris_op_t *op)
{
    return &plan->tensors[tigris_op_outputs(plan, op)[0]];
}

/* An operator's first operand, or its first result when it reads none. */
static const tigris_tensor_t *op_first_input(const tigris_plan_t *plan, const tigris_op_t *op)
{
    return op->num_inputs == 0u ? op_first_output(plan, op)
                                : &plan->tensors[tigris_op_inputs(plan, op)[0]];
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
    return bytes == element || bytes == channels || bytes == tigris_tensor_elements(output) * element;
}

/* Operator classes, read from the contract table only this file includes. */
int tigris_binary_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_BINARY) != 0u;
}

int tigris_axis1_unary_pointwise_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_AXIS_UNARY) != 0u;
}

int tigris_axis1_binary_pointwise_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    return row != NULL && (row->flags & TIGRIS_CONTRACT_AXIS_BINARY) != 0u;
}

tigris_height_op_t tigris_height_op(uint8_t type)
{
    const tigris_op_contract_t *row = tigris_op_contract(type);
    const uint16_t flags = row != NULL ? row->flags : 0u;
    tigris_height_op_t kind = TIGRIS_HEIGHT_OP_REFUSED;
    if ((flags & TIGRIS_CONTRACT_SPATIAL) != 0u)
        kind = TIGRIS_HEIGHT_OP_SPATIAL;
    else if ((flags & TIGRIS_CONTRACT_RESAMPLE) != 0u)
        kind = TIGRIS_HEIGHT_OP_RESAMPLE;
    else if ((flags & TIGRIS_CONTRACT_HEIGHT) != 0u)
        kind = TIGRIS_HEIGHT_OP_POINTWISE;
    return kind;
}

int tigris_height_spatial_op(uint8_t type)
{
    return tigris_height_op(type) == TIGRIS_HEIGHT_OP_SPATIAL;
}

int tigris_height_upsample_op(uint8_t type)
{
    return tigris_height_op(type) == TIGRIS_HEIGHT_OP_RESAMPLE;
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
    /* A stripe or an HW tile bands the rows of its spatial op's output. */
    const tigris_tensor_t *band_out = &plan->tensors[tigris_stage_band_output(plan, stage)];
    const int32_t *b = tigris_tensor_shape(plan, band_out);
    if (tile->axis == TIGRIS_TILE_AXIS_HW)
        return in->ndim == 4u && band_out->ndim == 4u && tile->input_band_extent == 0u &&
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
    /* A Resize, and from schema 11 every height stripe, states its band in output rows. */
    int output_rows = kind == TIGRIS_BAND_KIND_STRIPE &&
                      plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS;
    for (uint16_t i = 0; i < stage->ops_count; i++)
        if (plan->ops[ops[i]].op_type == TIGRIS_OP_RESIZE || plan->ops[ops[i]].op_type == TIGRIS_OP_RESIZE_LINEAR)
            output_rows = 1;
    if (output_rows)
        return band_out->ndim >= 3u && tile->input_band_extent == 0u &&
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

/* Whether an operator may read no activation: a gather from a constant table,
 * with the table in its weight and the indices in its bias. */
static int reads_constant_table(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index)
{
    return movement_is_gather(op->op_type) && op->weight_idx != TIGRIS_NO_WEIGHT &&
           op->bias_idx != TIGRIS_NO_WEIGHT &&
           tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_TABLE_SHAPE, NULL) != NULL;
}

/* The attribute pass calls this once every payload length meets the contract. */
static int movement_metadata_valid(const tigris_plan_t *plan, const tigris_op_t *op, uint16_t index,
                                   const uint8_t *payload, uint8_t length)
{
    int32_t metadata[19] = {0};
    int32_t expected[6] = {0};
    uint8_t table_length = 0u;
    const uint8_t *table = tigris_op_attribute_data(plan, index, TIGRIS_OP_ATTR_TABLE_SHAPE, &table_length);
    if (table == NULL && op->bias_idx != TIGRIS_NO_WEIGHT)
        return 0;
    memcpy(metadata, payload, length);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const tigris_tensor_t *input = table != NULL ? output : op_first_input(plan, op);
    int32_t table_shape[6];
    uint8_t rank = table != NULL ? (uint8_t)(table_length / 4u) : input->ndim;
    if (table != NULL) {
        if (op->weight_idx == TIGRIS_NO_WEIGHT) return 0;
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
static tigris_error_t validate_contract_detection_postprocess(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_DETECTION, NULL);
    uint8_t constants_len = 0u;
    if (!op_has_no_weights(op) || !params ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 2u)
        return TIGRIS_ERR_BAD_OPERATOR;
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
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *es = tigris_tensor_shape(plan, encodings);
    const int32_t *ss = tigris_tensor_shape(plan, scores);
    const int32_t boxes = es[1];
    if (es[0] != 1 || es[2] != 4 || boxes <= 0 || boxes > 65535 || ss[0] != 1 || ss[1] != boxes ||
        limits[0] == 0u || limits[2] == 0u || ss[2] < (int32_t)limits[2] || ss[2] > (int32_t)limits[2] + 1 ||
        params[6] > 1u || params[7] != 0u || (params[6] == 1u && limits[1] == 0u))
        return TIGRIS_ERR_BAD_OPERATOR;
    /* Thresholds and scales as TFLite Micro checks or divides by them. */
    if (!isfinite(values[0]) || !(values[1] > 0.0f) || !(values[1] <= 1.0f))
        return TIGRIS_ERR_BAD_OPERATOR;
    for (uint8_t k = 2u; k < 6u; k++)
        if (!isfinite(values[k]) || values[k] == 0.0f)
            return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t expected[4][3] = {{1, (int32_t)limits[0], 4}, {1, (int32_t)limits[0], 0},
                                    {1, (int32_t)limits[0], 0}, {1, 0, 0}};
    const uint8_t ranks[4] = {3u, 2u, 2u, 1u};
    for (uint8_t i = 0u; i < 5u; i++) {
        const tigris_tensor_t *out = &plan->tensors[outs[i]];
        if (out->dtype != 1u || out->quant_param_idx != TIGRIS_NO_QUANT_PARAM)
            return TIGRIS_ERR_BAD_OPERATOR;
        if (i < 4u) {
            const int32_t *shape = tigris_tensor_shape(plan, out);
            if (out->ndim != ranks[i]) return TIGRIS_ERR_BAD_OPERATOR;
            for (uint8_t a = 0u; a < ranks[i]; a++)
                if (shape[a] != expected[i][a]) return TIGRIS_ERR_BAD_OPERATOR;
        }
    }
    const size_t candidates = (size_t)limits[0] + (params[6] == 1u ? (size_t)limits[1] : 0u);
    if ((size_t)plan->tensors[outs[4]].size_bytes <
        TIGRIS_DETECTION_SCRATCH_BYTES(boxes, limits[0], candidates))
        return TIGRIS_ERR_BAD_OPERATOR;
    return weight_size_is(plan, constant_index(plan, op_index, 0u), (uint64_t)boxes * 16u)
        ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR;
}

/* Svdf: input [batch, features] and state [batch, filters * memory] in, output
 * [batch, units] and the next state out; feature, time and bias constants. A
 * float Svdf keeps everything in float32; an int8 one keeps state and time
 * weights in int16 and its bias in int32, as TFLite Micro does. */
static tigris_error_t validate_contract_svdf(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    uint8_t len = 0u;
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SVDF, &len);
    uint8_t constants_len = 0u;
    if (!op_has_no_weights(op) || !params ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 6u)
        return TIGRIS_ERR_BAD_OPERATOR;
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
        !tigris_tensor_shapes_equal(plan, state, kept) || state->dtype != kept->dtype ||
        len != (quantized ? 24u : 4u) || rank <= 0 ||
        state->dtype != (quantized ? 5u : 1u) || y->dtype != x->dtype ||
        (quantized && (!tigris_tensor_quant(plan, x) || !tigris_tensor_quant(plan, y))))
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *xs = tigris_tensor_shape(plan, x);
    const int32_t *ys = tigris_tensor_shape(plan, y);
    const int32_t *ss = tigris_tensor_shape(plan, state);
    uint64_t filters = (uint64_t)ys[1] * (uint64_t)rank;
    if (xs[0] != ys[0] || xs[0] != ss[0] || filters > INT32_MAX ||
        (uint64_t)ss[1] % filters != 0u)
        return TIGRIS_ERR_BAD_OPERATOR;
    uint64_t memory = (uint64_t)ss[1] / filters;
    uint16_t feature = constant_index(plan, op_index, 0u);
    uint16_t time = constant_index(plan, op_index, 1u);
    uint16_t bias = constant_index(plan, op_index, 2u);
    return (weight_size_is(plan, feature, filters * (uint64_t)xs[1] * (quantized ? 1u : 4u)) &&
            weight_size_is(plan, time, filters * memory * (quantized ? 2u : 4u)) &&
            weight_size_is(plan, bias, (uint64_t)ys[1] * 4u)) ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR;
}

/* Lstm: sequence [batch, steps, features] ([steps, batch, features] when time
 * major), hidden and cell state [batch, units] in; the hidden sequence and
 * both states out. Weights [units, features] then [units, units], four of
 * each, and four biases [units]. Float32 throughout, or int8 sequence and
 * hidden state with an int16 cell, int8 weights and int32 biases. */
static tigris_error_t validate_contract_lstm(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    uint8_t len = 0u;
    const uint8_t *params = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_LSTM, &len);
    uint8_t constants_len = 0u;
    if (!op_has_no_weights(op) || !params ||
        !tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &constants_len) ||
        constants_len != 24u)
        return TIGRIS_ERR_BAD_OPERATOR;
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
        return TIGRIS_ERR_BAD_OPERATOR;
    if (quantized) {
        int32_t v[28];
        memcpy(v, params, sizeof(v));
        /* Zero points in int8, a cell scale between 2^-30 and 2^15, a clip in
         * cell units, and shifts the requantization can apply. */
        if (v[2] < -128 || v[2] > 127 || v[3] < -128 || v[3] > 127 || v[4] < -30 || v[4] > 15 ||
            v[5] < 0 || v[5] > 32767)
            return TIGRIS_ERR_BAD_OPERATOR;
        for (uint8_t pair = 6u; pair < 28u; pair += 2u)
            if (v[pair] < 0 || v[pair + 1u] < -31 || v[pair + 1u] > 30)
                return TIGRIS_ERR_BAD_OPERATOR;
    }
    const int32_t *xs = tigris_tensor_shape(plan, x);
    const int32_t *ys = tigris_tensor_shape(plan, y);
    const int32_t batch = time_major ? xs[1] : xs[0];
    const int32_t units = ys[2];
    if (xs[0] != ys[0] || xs[1] != ys[1])
        return TIGRIS_ERR_BAD_OPERATOR;
    for (uint8_t i = 1u; i < 3u; i++) {
        const tigris_tensor_t *held = &plan->tensors[ins[i]];
        const tigris_tensor_t *kept = &plan->tensors[outs[i]];
        const int32_t *hs = tigris_tensor_shape(plan, held);
        const uint8_t dtype = quantized ? (i == 1u ? 3u : 5u) : 1u;
        if (held->ndim != 2u || held->dtype != dtype || hs[0] != batch || hs[1] != units ||
            (held->flags & TIGRIS_TENSOR_STATE) == 0u || (kept->flags & TIGRIS_TENSOR_STATE) == 0u ||
            kept->dtype != dtype || !tigris_tensor_shapes_equal(plan, held, kept))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    for (uint8_t k = 0u; k < 12u; k++) {
        const uint64_t columns = k < 4u ? (uint64_t)xs[2] : (k < 8u ? (uint64_t)units : 1u);
        const uint64_t element = (quantized && k < 8u) ? 1u : 4u;
        if (!weight_size_is(plan, constant_index(plan, op_index, k),
                            (uint64_t)units * columns * element))
            return TIGRIS_ERR_BAD_OPERATOR;
    }
    return TIGRIS_OK;
}

/* Role 0 is the plan's data dtype, 255 data, int32 or bool, 254 data or int16
 * state (the tensor table admits int16 only on state tensors), 253 data or
 * int32, 252 bool or int32, 251 data, int32 or float32 (control flow carries
 * float32 across the boundary of an int8 plan's subgraph), 250 data or
 * float32, and any other role that dtype. */
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

static tigris_error_t validate_contract_equal(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const tigris_tensor_t *x = &plan->tensors[inputs[0]];
    const tigris_tensor_t *y = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    const uint8_t type = op->op_type;
    if (op->bias_idx != TIGRIS_NO_WEIGHT) return TIGRIS_ERR_BAD_OPERATOR;
    if (type == TIGRIS_OP_LOGICAL_NOT || type == TIGRIS_OP_CAST || type == TIGRIS_OP_ADD_N) {
        if (op->weight_idx != TIGRIS_NO_WEIGHT) return TIGRIS_ERR_BAD_OPERATOR;
        /* int32 casts to float32 only, as the converter writes a counter read as a float. */
        if (x->dtype == 6u && (type != TIGRIS_OP_CAST || y->dtype != 1u)) return TIGRIS_ERR_BAD_OPERATOR;
        for (uint8_t i = 0; i < op->num_inputs; i++) {
            const tigris_tensor_t *t = &plan->tensors[inputs[i]];
            if (!tigris_tensor_shapes_equal(plan, t, y)) return TIGRIS_ERR_BAD_OPERATOR;
            if (type == TIGRIS_OP_ADD_N && x->dtype == 3u &&
                !same_quantization(tigris_tensor_quant(plan, x), tigris_tensor_quant(plan, t))) return TIGRIS_ERR_BAD_OPERATOR;
        }
        if (y->dtype == 3u) {
            const tigris_quant_param_t *qy = tigris_tensor_quant(plan, y);
            if (qy == NULL || qy->num_channels != 1u) return TIGRIS_ERR_BAD_OPERATOR;
            if (type == TIGRIS_OP_CAST)
                return (isfinite(qy->scale) && qy->scale > 0.0f &&
                        qy->zero_point >= -128 && qy->zero_point <= 127) ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR;
            const tigris_quant_param_t *qx = tigris_tensor_quant(plan, x);
            if (qx == NULL) return TIGRIS_ERR_BAD_OPERATOR;
            /* The exact input multiplier 0.5 follows the reference's left shift 20. */
            const int64_t low = (-(int64_t)qx->zero_point + (int64_t)op->num_inputs * (-128 - (int64_t)qx->zero_point)) * 524288;
            const int64_t high = (-(int64_t)qx->zero_point + (int64_t)op->num_inputs * (127 - (int64_t)qx->zero_point)) * 524288;
            const double multiplier = 2.0 * (double)qx->scale / (1048576.0 * (double)qy->scale);
            if (low < INT32_MIN || high > INT32_MAX || multiplier <= 0.0 || multiplier >= 1.0) return TIGRIS_ERR_BAD_OPERATOR;
        }
        return TIGRIS_OK;
    }
    if (y->ndim > 8u) return TIGRIS_ERR_BAD_OPERATOR;
    const uint8_t arity = type == TIGRIS_OP_SELECT_V2 ? 3u : 2u;
    const uint8_t rank_limit = arity == 3u ? 5u : 4u;
    uint8_t len = 0u;
    const uint8_t *constant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &len);
    const uint8_t data_dtype = x->dtype == 1u || y->dtype == 1u ? 1u : tigris_plan_data_dtype(plan);
    const int comparison = type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_GREATER_EQUAL;
    if (op->num_inputs == arity && (constant != NULL || op->weight_idx != TIGRIS_NO_WEIGHT)) return TIGRIS_ERR_BAD_OPERATOR;
    if (op->num_inputs != arity && (constant == NULL || constant[0] >= arity)) return TIGRIS_ERR_BAD_OPERATOR;
    const tigris_quant_param_t *qy = tigris_tensor_quant(plan, y);
    uint8_t input_slot = 0u;
    for (uint8_t k = 0u; k < arity; k++) {
        const uint8_t dtype = comparison ? (x->dtype == 6u ? 6u : data_dtype)
                            : ((arity == 3u && k != 0u) ? data_dtype : 9u);
        const tigris_quant_param_t *q = NULL;
        if (constant != NULL && k == constant[0]) {
            if (!constant_operand_is_valid(plan, op, op_index, y, dtype, &q)) return TIGRIS_ERR_BAD_OPERATOR;
            if (len > TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN) {
                int32_t shape[8];
                memcpy(shape, constant + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN, 4u * y->ndim);
                if (!bool_broadcast_valid(shape, y->ndim, plan, y, rank_limit)) return TIGRIS_ERR_BAD_OPERATOR;
            }
        } else {
            const tigris_tensor_t *t = &plan->tensors[inputs[input_slot]];
            input_slot++;
            if (!bool_broadcast_valid(tigris_tensor_shape(plan, t), t->ndim, plan, y, rank_limit)) return TIGRIS_ERR_BAD_OPERATOR;
            if (comparison && t->dtype != dtype) return TIGRIS_ERR_BAD_OPERATOR;
            q = tigris_tensor_quant(plan, t);
        }
        if (dtype == 3u) {
            if (q == NULL) return TIGRIS_ERR_BAD_OPERATOR;
            if (comparison && (q->scale <= 0.0f || q->scale >= 1.0f)) return TIGRIS_ERR_BAD_OPERATOR;
            if (arity == 3u && !same_quantization(q, qy)) return TIGRIS_ERR_BAD_OPERATOR;
        }
    }
    const uint8_t *requant = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_COMPARISON_REQUANT, NULL);
    const int requant_needed = comparison && data_dtype == 3u && x->dtype != 6u;
    return (requant != NULL) == requant_needed ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR;
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

/* Whether a plan's count is above a build limit. The limit is a parameter so
 * the comparison holds for every limit up to 65535. */
static int above_limit(uint32_t seen, uint32_t limit)
{
    return seen > limit;
}

/* The index range [lo, hi] of the tensors graph g reads and writes, empty when
 * lo > hi; the main graph's also holds the state. */
static void graph_tensor_range(const tigris_plan_t *plan, uint16_t g, uint16_t *lo, uint16_t *hi)
{
    const tigris_subgraph_t *graph = &plan->subgraphs[g];
    *lo = UINT16_MAX;
    *hi = 0u;
    for (uint16_t s = graph->first_stage; s < graph->first_stage + graph->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        const uint16_t *ops = tigris_stage_ops(plan, stage);
        for (uint16_t k = 0; k < stage->ops_count; k++) {
            const tigris_op_t *op = &plan->ops[ops[k]];
            const uint16_t *ins = tigris_op_inputs(plan, op);
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t i = 0; i < op->num_inputs; i++) cover(lo, hi, ins[i]);
            for (uint8_t i = 0; i < op->num_outputs; i++) cover(lo, hi, outs[i]);
        }
    }
    for (uint16_t i = 0; i < graph->inputs_count; i++)
        cover(lo, hi, plan->index_pool[graph->inputs_off + i]);
    for (uint16_t i = 0; i < graph->outputs_count; i++)
        cover(lo, hi, plan->index_pool[graph->outputs_off + i]);
    for (uint16_t i = 0; g == 0u && i < plan->num_state; i++) {
        cover(lo, hi, plan->state_entries[i].input);
        if (plan->state_entries[i].output != 0xFFFFu)
            cover(lo, hi, plan->state_entries[i].output);
    }
}

/* Graph slot 0 or 1 of a control-flow operator's subgraphs attribute names. */
static uint16_t named_graph(const uint8_t *runs, uint32_t slot)
{
    return (uint16_t)((uint16_t)runs[2u * slot] | (uint16_t)((uint16_t)runs[2u * slot + 1u] << 8));
}

/* The subgraphs attribute of the i-th attribute record, or NULL for another kind. */
static const uint8_t *subgraphs_record(const tigris_plan_t *plan, uint16_t i)
{
    const tigris_op_attribute_t *attr = &plan->op_attributes[i];
    return attr->type == TIGRIS_OP_ATTR_SUBGRAPHS ? plan->op_attribute_data + attr->data_offset : NULL;
}

/* The graph run at position p = 2 * stage + slot, slot 0 or 1 of a
 * control-flow operator's two graphs; 0, never a run graph, when the stage has
 * no control flow. */
static uint16_t graph_run_at(const tigris_plan_t *plan, uint32_t p)
{
    const tigris_stage_t *stage = &plan->stages[p / 2u];
    if (stage->ops_count != 1u) return 0u;
    const uint16_t op_index = tigris_stage_ops(plan, stage)[0];
    const uint8_t type = plan->ops[op_index].op_type;
    if (type != TIGRIS_OP_IF && type != TIGRIS_OP_WHILE) return 0u;
    return named_graph(tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SUBGRAPHS, NULL), p % 2u);
}

/* The stage of the one control-flow operator that runs graph g, placed before
 * g's own stages and alone in its stage. */
static uint16_t runner_stage(const tigris_plan_t *plan, uint16_t g)
{
    uint16_t runner = 0u;
    for (uint16_t i = 0u; i < plan->num_op_attributes; i++) {
        const uint8_t *runs = subgraphs_record(plan, i);
        if (runs != NULL && (named_graph(runs, 0u) == g || named_graph(runs, 1u) == g))
            runner = plan->op_attributes[i].op_index;
    }
    uint16_t s = plan->subgraphs[g].first_stage;
    while (s > 0u) {
        s--;
        if (plan->stages[s].ops_count == 1u && tigris_stage_ops(plan, &plan->stages[s])[0] == runner)
            break;
    }
    return s;
}

/* How deep control flow nests. The graphs form a tree below the main graph,
 * each run by one operator, so the walk enters each graph an operator runs and
 * returns to the position after the slot that ran it, keeping no storage per
 * graph. An If whose two branches name one graph enters it once. */
static uint32_t subgraph_depth(const tigris_plan_t *plan)
{
    uint32_t depth = 0u, deepest = 0u;
    uint16_t graph = 0u;
    uint32_t p = 0u;
    for (;;) {
        const tigris_subgraph_t *running = &plan->subgraphs[graph];
        const uint32_t end = 2u * ((uint32_t)running->first_stage + running->num_stages);
        uint16_t child = 0u;
        while (p < end && child == 0u) child = graph_run_at(plan, p++);
        if (child != 0u) {
            depth++;
            if (depth > deepest) deepest = depth;
            graph = child;
            p = 2u * (uint32_t)plan->subgraphs[child].first_stage;
        } else if (graph == 0u) {
            return deepest;
        } else {
            const uint16_t s = runner_stage(plan, graph);
            p = 2u * (uint32_t)s + (graph_run_at(plan, 2u * (uint32_t)s + 1u) == graph ? 2u : 1u);
            graph = graph_of_stage(plan, s);
            depth--;
        }
    }
}

static int same_tensor_form(const tigris_plan_t *plan, uint16_t a, uint16_t b)
{
    const tigris_tensor_t *x = &plan->tensors[a];
    const tigris_tensor_t *y = &plan->tensors[b];
    /* Below rank 3 both layouts store the same bytes. */
    return x->dtype == y->dtype && x->size_bytes == y->size_bytes &&
           tigris_tensor_shapes_equal(plan, x, y) &&
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
static tigris_error_t validate_subgraphs(const tigris_plan_t *plan, uint32_t index_count,
                                         uint16_t *depth)
{
    const uint16_t num_stages = plan->header->num_stages;
    *depth = 0u;
    if (plan->num_subgraphs == 0u) {
        for (uint16_t n = 0; n < plan->header->num_ops; n++)
            if (plan->ops[n].op_type == TIGRIS_OP_IF || plan->ops[n].op_type == TIGRIS_OP_WHILE)
                return TIGRIS_ERR_BAD_OPERATOR;
        return TIGRIS_OK;
    }
    if (above_limit(plan->num_subgraphs, TIGRIS_MAX_SUBGRAPHS))
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
    /* A graph runs its stage range alone, so a chain stays inside one graph. */
    for (uint16_t s = 0; s < num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        if (stage->chain_len >= 2u && stage->chain_id == s &&
            graph_of_stage(plan, s) != graph_of_stage(plan, (uint16_t)(s + stage->chain_len - 1u)))
            return TIGRIS_ERR_BAD_SECTION;
    }
    /* Each graph's tensors form one index range, the ranges disjoint, so no
     * tensor is shared between graphs; the main graph's range also holds the
     * model interface and the state. A range is derived where it is compared,
     * so no storage grows with the graph count. */
    for (uint16_t g = 0; g < plan->num_subgraphs; g++) {
        uint16_t lo = 0u;
        uint16_t hi = 0u;
        graph_tensor_range(plan, g, &lo, &hi);
        for (uint16_t h = (uint16_t)(g + 1u); h < plan->num_subgraphs; h++) {
            uint16_t h_lo = 0u;
            uint16_t h_hi = 0u;
            graph_tensor_range(plan, h, &h_lo, &h_hi);
            if (lo <= hi && h_lo <= h_hi && lo <= h_hi && h_lo <= hi)
                return TIGRIS_ERR_BAD_TENSOR;
        }
        for (uint32_t t = lo; g > 0u && t <= hi; t++)
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
    /* Each subgraph is run by one control-flow operator at most, so the depth
     * walk can return from a graph to the operator that ran it. Each such
     * operator has one subgraphs record. */
    for (uint16_t i = 0u; i < plan->num_op_attributes; i++) {
        const uint8_t *runs = subgraphs_record(plan, i);
        for (uint16_t j = (uint16_t)(i + 1u); runs != NULL && j < plan->num_op_attributes; j++) {
            const uint8_t *other = subgraphs_record(plan, j);
            for (uint32_t k = 0u; other != NULL && k < 4u; k++)
                if (named_graph(runs, k / 2u) == named_graph(other, k % 2u))
                    return TIGRIS_ERR_BAD_OPERATOR;
        }
    }
    const uint32_t nesting = subgraph_depth(plan);
    if (nesting > TIGRIS_MAX_SUBGRAPH_DEPTH)
        return TIGRIS_ERR_BAD_OPERATOR;
    *depth = (uint16_t)nesting;
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

/* A float activation of an int8 plan is in a float section: an operator
 * placed before index `before` writes it and is a Dequantize, an If, a While,
 * or a float operator without int8 operands, whose own operands are checked
 * in its turn. Operators run in index order, so an operand is written first. */
static int float_section_written(const tigris_plan_t *plan, uint16_t tensor, uint16_t before)
{
    for (uint16_t n = 0u; n < before; n++) {
        const tigris_op_t *op = &plan->ops[n];
        const uint16_t *outs = tigris_op_outputs(plan, op);
        int writes = 0;
        for (uint8_t k = 0u; k < op->num_outputs; k++)
            if (outs[k] == tensor) writes = 1;
        if (!writes) continue;
        if (op->op_type == TIGRIS_OP_DEQUANTIZE || op->op_type == TIGRIS_OP_IF ||
            op->op_type == TIGRIS_OP_WHILE)
            return 1;
        if (!tigris_float_operator(plan, op)) continue;
        const uint16_t *ins = tigris_op_inputs(plan, op);
        int int8_operand = 0;
        for (uint8_t k = 0u; k < op->num_inputs; k++)
            if (plan->tensors[ins[k]].dtype == 3u) int8_operand = 1;
        if (!int8_operand) return 1;
    }
    return 0;
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

/* The attribute pass has checked the movement metadata. */
static tigris_error_t validate_contract_gather(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    (void)op;
    return tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_MOVEMENT, NULL) != NULL
        ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR;
}

static tigris_error_t validate_contract_quantize(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
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
    if (!op_has_no_weights(op) || !tigris_tensor_shapes_equal(plan, input, output) ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->weight_idx == TIGRIS_NO_WEIGHT || input->ndim != 4 || output->ndim != 4)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->weight_idx == TIGRIS_NO_WEIGHT || input->ndim != 3 || output->ndim != 3)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->weight_idx == TIGRIS_NO_WEIGHT || output->ndim == 0)
        return TIGRIS_ERR_BAD_OPERATOR;
    const int32_t *out_shape = tigris_tensor_shape(plan, output);
    uint32_t output_channels =
        (uint32_t)out_shape[output->ndim - 1u];
    uint32_t input_elements = tigris_tensor_elements(input);
    uint32_t output_elements = tigris_tensor_elements(output);
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    if (!op_has_no_weights(op) || !tigris_tensor_shapes_equal(plan, input, output) ||
        (op->op_type == TIGRIS_OP_SOFTMAX && input->ndim == 0))
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_reshape(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    if (!op_has_no_weights(op) ||
        input->size_bytes != output->size_bytes || input->dtype != output->dtype)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_leaky_relu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (!op_has_no_weights(op) ||
        !tigris_tensor_shapes_equal(plan, input, output) ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (!op_has_no_weights(op) || !tigris_tensor_shapes_equal(plan, input, output))
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    const tigris_quant_param_t *second_quant = NULL;
    if (op->op_type == TIGRIS_OP_PRELU) {
        const uint8_t *attr = tigris_op_attribute_data(
            plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, NULL);
        if (op->num_inputs != 1u || !attr || attr[0] != 1u ||
            input->ndim == 0u || input->ndim > 4u ||
            !tigris_tensor_shapes_equal(plan, input, output))
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
        if (!op_has_no_weights(op) ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    /* A byte copy into a filled frame: the pads attribute places the
     * input, the optional weight is one element of fill, and int8
     * keeps its quantization. */
    uint8_t pads_len = 0u;
    const uint8_t *pads = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_PADS, &pads_len);
    const uint32_t element = (dtype == 1u) ? (uint32_t)sizeof(float) : 1u;
    if (op->bias_idx != TIGRIS_NO_WEIGHT || pads == NULL ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    const uint8_t *mode = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_GELU_APPROXIMATE, NULL);
    if (mode == NULL || !op_has_no_weights(op) ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    if (!op_has_no_weights(op) || input->ndim != output->ndim ||
        input->size_bytes != output->size_bytes)
        return TIGRIS_ERR_BAD_OPERATOR;
    return TIGRIS_OK;
}

static tigris_error_t validate_contract_split(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index)
{
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *input = op_first_input(plan, op);
    (void)op_index;
    /* A split needs no arithmetic: every output keeps the input's
     * shape but for the stored axis named in kernel_h, 0 being the
     * outermost, and their extents on that axis sum to the input's.
     * Each part is then one contiguous run per position ahead of the
     * axis. */
    if (input->ndim == 0u || op->spatial.kernel_h >= input->ndim)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    uint8_t num_axes = 0;
    const uint8_t *axes = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &num_axes);
    if (!op_has_no_weights(op) || input->ndim != 3u ||
        !tigris_tensor_shapes_equal(plan, input, output) || !axes || num_axes != 1u || axes[0] >= 3u)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
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
    if (!op_has_no_weights(op) || !rank_fits ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    /* One activation in and out, shape-preserving. Scale is a weight
     * and bias an optional one, each holding one value per position
     * along the axis the kernel normalizes, which is the last stored
     * one. */
    if (input->ndim == 0u || input->ndim != output->ndim ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    /* Both operands are activations, so neither is a weight entry.
     * Axes are the model's own: the trailing pair is the matrix and
     * everything before it batches, with no broadcasting. A plan that
     * needs broadcast batches is rejected rather than guessed at. */
    if (!op_has_no_weights(op) || input->ndim < 2)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    /* Each operand has the output's shape or broadcasts to it; int32
     * operands and result go together, without an activation. */
    if ((dtype == 6u) != (output->dtype == 6u) ||
        (op->num_inputs == 2 && (plan->tensors[inputs[1]].dtype == 6u) != (dtype == 6u)) ||
        (dtype == 6u && op->fused_act != TIGRIS_ACT_NONE))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->bias_idx != TIGRIS_NO_WEIGHT ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    (void)op_index;
    if (op->op_type == TIGRIS_OP_L2_POOL && dtype != 1u)
        return TIGRIS_ERR_BAD_OPERATOR;
    if (!op_has_no_weights(op) ||
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    (void)op_index;
    if (!op_has_no_weights(op) || input->ndim != 4 || output->ndim != 4)
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
    if (op->bias_idx != TIGRIS_NO_WEIGHT || output->ndim < 3u || output->ndim > 4u)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
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
    if (op->weight_idx == TIGRIS_NO_WEIGHT || input->ndim != 4 || output->ndim != 4)
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
    const tigris_tensor_t *input = op_first_input(plan, op);
    const tigris_tensor_t *output = op_first_output(plan, op);
    const uint8_t dtype = input->dtype;
    if (op->spatial.kernel_h > (op->op_type == TIGRIS_OP_RESIZE_LINEAR ? 2u : 3u))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (op->op_type == TIGRIS_OP_RESIZE_LINEAR &&
        ((dtype != 1 && dtype != 3) ||
         (dtype == 3 &&
          (input->quant_param_idx == TIGRIS_NO_QUANT_PARAM ||
           output->quant_param_idx == TIGRIS_NO_QUANT_PARAM))))
        return TIGRIS_ERR_BAD_OPERATOR;
    if (!op_has_no_weights(op) || input->ndim != 4 || output->ndim != 4)
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
        const uint8_t *table_attr = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_TABLE_SHAPE, NULL);
        if ((op->num_inputs == 0 && !reads_constant_table(plan, op, op_index)) || op->num_outputs == 0 ||
            op->fused_act > TIGRIS_ACT_RELU6 || op->_pad1 != 0)
            return TIGRIS_ERR_BAD_OPERATOR;

        const uint16_t *inputs = tigris_op_inputs(plan, op);
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        uint8_t dtype = op_first_input(plan, op)->dtype;
        const tigris_op_contract_t *contract = tigris_op_contract(op->op_type);
        if (contract == NULL) return TIGRIS_ERR_BAD_OPERATOR;
        const dtype_signature_t *signature = &contract->dtype;
        const uint8_t op_dtype = data_dtype == 3u && tigris_float_operator(plan, op) ? 1u : data_dtype;
        if (data_dtype == 3u && op_dtype == 1u) {
            for (uint8_t i = 0u; i < op->num_inputs; i++)
                if (plan->tensors[inputs[i]].dtype == 1u &&
                    (plan->tensors[inputs[i]].flags & TIGRIS_TENSOR_CONSTANT) == 0u &&
                    !float_section_written(plan, inputs[i], op_index))
                    return TIGRIS_ERR_BAD_OPERATOR;
            for (uint8_t i = 0u; i < op->num_outputs; i++)
                if (plan->tensors[outputs[i]].dtype == 1u &&
                    !float_section_written(plan, outputs[i], hdr->num_ops))
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

/* Raise a reported requirement to cover one more observation; nonzero when
 * the observation is above this build's limit. */
static int note_requirement(uint16_t *slot, uint32_t seen, uint32_t limit)
{
    if (slot != NULL && seen > (uint32_t)*slot)
        *slot = (seen > 65535u) ? 65535u : (uint16_t)seen;
    return above_limit(seen, limit);
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

    /* Per-operator attributes (optional). */
    if (ctx->section_offsets[TIGRIS_SEC_OP_ATTRIBUTES]) {
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
        if (ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] < off + 4u + entries_size)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t data_size = ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] - off - 4u - entries_size;
        if ((data_size % sizeof(int32_t)) != 0)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t physical_pages = (data_size / sizeof(int32_t) +
                                   TIGRIS_QUANT_PAGE_ELEMS - 1u) /
                                  TIGRIS_QUANT_PAGE_ELEMS;
        /* The next aligned section can contribute up to 15 padding bytes
         * to this section span. It may therefore extend an otherwise
         * page-aligned pool into one physical page without changing the
         * semantic page count recorded by the compiler. */
        if (qd_field == 0 || physical_pages < qd_field ||
            physical_pages > (uint32_t)qd_field + 1u)
            return TIGRIS_ERR_BAD_SECTION;
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

    if (note_requirement(&ctx->required.tensors, ctx->hdr->num_tensors, TIGRIS_MAX_TENSORS)) {
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

/* The values of one attribute payload, whose length the contract has checked. */
static tigris_error_t validate_attribute_value(
    tigris_load_state_t *ctx, const tigris_op_attribute_t *attr,
    const tigris_op_t *op, const tigris_tensor_t *input,
    const tigris_tensor_t *output)
{
    const uint8_t *data = ctx->candidate.op_attribute_data + attr->data_offset;
    switch (attr->type) {
    case TIGRIS_OP_ATTR_TRANSPOSE_PERM:
    {
        if (op->num_inputs != 1u || attr->data_len != input->ndim ||
            output->ndim != input->ndim || output->size_bytes != input->size_bytes)
            return TIGRIS_ERR_BAD_SECTION;
        const int32_t *in_shape = tigris_tensor_shape(&ctx->candidate, input);
        const int32_t *out_shape = tigris_tensor_shape(&ctx->candidate, output);
        for (uint8_t axis = 0; axis < attr->data_len; axis++) {
            if (data[axis] >= attr->data_len)
                return TIGRIS_ERR_BAD_SECTION;
            for (uint8_t prior = 0; prior < axis; prior++) {
                if (data[prior] == data[axis])
                    return TIGRIS_ERR_BAD_SECTION;
            }
            if (out_shape[axis] != in_shape[data[axis]])
                return TIGRIS_ERR_BAD_SECTION;
        }
        break;
    }
    case TIGRIS_OP_ATTR_EPSILON:
    case TIGRIS_OP_ATTR_ALPHA:
    {
        float value;
        memcpy(&value, data, sizeof(value));
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
        if (input->ndim != 4u || output->ndim != 4u)
            return TIGRIS_ERR_BAD_SECTION;
        memcpy(scales, data, sizeof(scales));
        const int32_t *in_shape = tigris_tensor_shape(&ctx->candidate, input);
        const int32_t *out_shape = tigris_tensor_shape(&ctx->candidate, output);
        for (int axis = 0; axis < 2; axis++) {
            if (!isfinite(scales[axis]) || scales[axis] <= 0.0f ||
                floorf((float)in_shape[axis + 1] * scales[axis]) != (float)out_shape[axis + 1])
                return TIGRIS_ERR_BAD_SECTION;
        }
        break;
    }
    case TIGRIS_OP_ATTR_CLIP_BOUNDS:
    {
        float bounds[2];
        memcpy(bounds, data, sizeof(bounds));
        if (!isfinite(bounds[0]) || !isfinite(bounds[1]) || bounds[0] > bounds[1])
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_PADS:
    {
        /* Two int32 per axis, leading then trailing. data_len counts
         * bytes, so a rank beyond 31 cannot be expressed and is
         * refused by the width check rather than truncated. */
        uint32_t pads_bytes = 2u * (uint32_t)input->ndim * (uint32_t)sizeof(int32_t);
        if (input->ndim == 0u || pads_bytes > 0xFFu ||
            (uint32_t)attr->data_len != pads_bytes ||
            (attr->data_offset % 4u) != 0u)
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t byte = 0; byte < attr->data_len; byte += 4u) {
            int32_t pad;
            memcpy(&pad, data + byte, sizeof(pad));
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
        if (ctx->hdr->version < TIGRIS_SCHEMA_VERSION_BINARY_REQUANT)
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t pair = 0; pair < 3u; pair++) {
            int32_t multiplier;
            int32_t shift;
            memcpy(&multiplier, data + (uint32_t)pair * 8u, sizeof(multiplier));
            memcpy(&shift, data + (uint32_t)pair * 8u + 4u, sizeof(shift));
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
        if (data[0] > (op->op_type == TIGRIS_OP_SELECT_V2 ? 2u : 1u) || data[1] != 0u)
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_POOL_ROUNDING:
    {
        if (data[0] != TIGRIS_POOL_ROUNDING_AVERAGE)
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_AXES:
    {
        if (attr->data_len > input->ndim)
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t axis = 0; axis < attr->data_len; axis++) {
            if (data[axis] >= input->ndim)
                return TIGRIS_ERR_BAD_SECTION;
            for (uint8_t prior = 0; prior < axis; prior++) {
                if (data[prior] >= data[axis])
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
        break;
    }
    case TIGRIS_OP_ATTR_GELU_APPROXIMATE:
    {
        if (data[0] > 1u)
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_COMPARISON_REQUANT:
    {
        int32_t values[5];
        memcpy(values, data, sizeof(values));
        if (values[0] != 8 || values[1] < 0 || values[3] < 0 ||
            values[2] < -31 || values[2] > 0 || values[4] < -31 || values[4] > 0)
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_MOVEMENT:
    {
        if (!movement_metadata_valid(&ctx->candidate, op, attr->op_index, data, attr->data_len))
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_CONSTANTS:
    {
        for (uint8_t byte = 0; byte < attr->data_len; byte += 2u) {
            uint16_t index = (uint16_t)((uint16_t)data[byte] |
                                        (uint16_t)((uint16_t)data[byte + 1u] << 8));
            if (index != TIGRIS_NO_WEIGHT && index >= ctx->hdr->num_weights)
                return TIGRIS_ERR_BAD_SECTION;
        }
        break;
    }
    case TIGRIS_OP_ATTR_TABLE_SHAPE:
    case TIGRIS_OP_ATTR_DETECTION:
    case TIGRIS_OP_ATTR_SUBGRAPHS:
        /* Checked with the gather, the operator and the subgraph section. */
        break;
    case TIGRIS_OP_ATTR_LSTM:
    {
        int32_t time_major;
        float clip;
        memcpy(&time_major, data, sizeof(time_major));
        memcpy(&clip, data + 4u, sizeof(clip));
        if ((time_major != 0 && time_major != 1) || !isfinite(clip) || clip < 0.0f)
            return TIGRIS_ERR_BAD_SECTION;
        break;
    }
    case TIGRIS_OP_ATTR_SVDF:
    {
        int32_t values[6];
        memcpy(values, data, attr->data_len);
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
        if (data[0] > 1u || data[1] > 1u)
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
                !tigris_tensor_shapes_equal(&ctx->candidate, in, out) ||
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
     * against the referenced operators before any kernel can use them. */
    if (ctx->candidate.op_attributes) {
        uint32_t attrs_off = ctx->section_offsets[TIGRIS_SEC_OP_ATTRIBUTES];
        uint32_t attrs_data_off = attrs_off + 4u +
            (uint32_t)ctx->candidate.num_op_attributes *
            sizeof(tigris_op_attribute_t);
        uint32_t attrs_data_len = ctx->section_ends[TIGRIS_SEC_OP_ATTRIBUTES] -
            attrs_data_off;
        /* An attribute may refer to a later attribute of the same operator.
         * Check every span and payload length before any payload validator
         * follows such a link. */
        for (uint16_t i = 0; i < ctx->candidate.num_op_attributes; i++) {
            const tigris_op_attribute_t *attr = &ctx->candidate.op_attributes[i];
            const tigris_attribute_contract_t *lengths = tigris_attribute_contract(attr->type);
            if (attr->op_index >= ctx->hdr->num_ops ||
                !bounds_ok(attr->data_offset, attr->data_len, attrs_data_len) ||
                lengths == NULL || lengths->step == 0u ||
                attr->data_len < lengths->min_len || attr->data_len > lengths->max_len ||
                (attr->data_len - lengths->min_len) % lengths->step != 0u)
                return TIGRIS_ERR_BAD_SECTION;
        }
        uint16_t previous_op = 0;
        uint8_t previous_attr_type = 0;
        for (uint16_t i = 0; i < ctx->candidate.num_op_attributes; i++) {
            const tigris_op_attribute_t *attr = &ctx->candidate.op_attributes[i];
            if (i > 0 &&
                (attr->op_index < previous_op ||
                 (attr->op_index == previous_op && attr->type <= previous_attr_type)))
                return TIGRIS_ERR_BAD_SECTION;
            const tigris_op_t *op = &ctx->candidate.ops[attr->op_index];
            /* Every kind a schema-8 plan may carry is validated here whether
             * or not an operator consumes it yet, so an operator behind one
             * of them lands without a further version. A kind this build does
             * not know still fails closed, in the default arm. */
            if (attr->type > TIGRIS_OP_ATTR_TRANSPOSE_PERM &&
                ctx->hdr->version < TIGRIS_SCHEMA_VERSION_V8)
                return TIGRIS_ERR_BAD_SECTION;
            /* An operator reads an activation unless it gathers from a constant
             * table, and writes one result or as many as its row allows. */
            const tigris_op_contract_t *row = tigris_op_contract(op->op_type);
            if ((op->num_inputs == 0u && !reads_constant_table(&ctx->candidate, op, attr->op_index)) ||
                (op->num_outputs != 1u && (row == NULL || op->num_outputs < row->min_outputs ||
                                           op->num_outputs > row->max_outputs)))
                return TIGRIS_ERR_BAD_SECTION;
            /* A kind belongs to the operator that reads it, whether or not a
             * kernel for that operator exists yet, so a stray payload cannot
             * ride along on an unrelated op. */
            if (!attr_kind_matches_op(attr->type, op->op_type))
                return TIGRIS_ERR_BAD_SECTION;
            tigris_error_t value_error = validate_attribute_value(
                ctx, attr, op, op_first_input(&ctx->candidate, op), op_first_output(&ctx->candidate, op));
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
    } else {
        for (uint16_t op_idx = 0u; op_idx < ctx->hdr->num_ops; op_idx++) {
            const tigris_op_contract_t *row =
                tigris_op_contract(ctx->candidate.ops[op_idx].op_type);
            if (row != NULL && row->required_attributes != 0u)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }
    return TIGRIS_OK;
}

/* Whether a stage's inputs and outputs are rank 4, with operators' operands and results too. */
static int stage_tensors_rank4(const tigris_plan_t *plan, const tigris_stage_t *stage,
                               int operators)
{
    const uint16_t *inputs = tigris_stage_inputs(plan, stage);
    const uint16_t *outputs = tigris_stage_outputs(plan, stage);
    for (uint16_t i = 0; i < stage->inputs_count; i++)
        if (plan->tensors[inputs[i]].ndim != 4u) return 0;
    for (uint16_t i = 0; i < stage->outputs_count; i++)
        if (plan->tensors[outputs[i]].ndim != 4u) return 0;
    const uint16_t *ops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; operators && j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[ops[j]];
        const uint16_t *op_inputs = tigris_op_inputs(plan, op);
        const uint16_t *op_outputs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_inputs; k++)
            if (plan->tensors[op_inputs[k]].ndim != 4u) return 0;
        for (uint8_t k = 0; k < op->num_outputs; k++)
            if (plan->tensors[op_outputs[k]].ndim != 4u) return 0;
    }
    return 1;
}

/* Whether every binary pointwise operator of a stage takes operands of one shape. */
static int stage_binary_operands_equal(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    const uint16_t *ops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[ops[j]];
        if (!tigris_axis1_binary_pointwise_op(op->op_type) || op->num_inputs < 2u) continue;
        const uint16_t *inputs = tigris_op_inputs(plan, op);
        for (uint8_t k = 1; k < op->num_inputs; k++)
            if (!tigris_tensor_shapes_equal(plan, &plan->tensors[inputs[0]], &plan->tensors[inputs[k]]))
                return 0;
    }
    return 1;
}

static tigris_error_t validate_stage_tile_band(
    tigris_load_state_t *ctx, const tigris_stage_t *stage,
    const tigris_tile_plan_v10_t *tile, int reshape, int32_t leading_rows)
{
    if (!reshape && !leading_rows && tile->tileable && tile->axis == TIGRIS_TILE_AXIS_HW) {
        /* The 2D executor bands rank-4 stage tensors, in the stripe contract along HW. */
        if (stage->inputs_count == 0 || stage->outputs_count == 0 ||
            !stage_tensors_rank4(&ctx->candidate, stage, 0))
            return TIGRIS_ERR_BAD_SECTION;
        if (tigris_stage_band_kind(&ctx->candidate, stage, NULL) != TIGRIS_BAND_KIND_HW ||
            !stage_binary_operands_equal(&ctx->candidate, stage))
            return TIGRIS_ERR_BAD_OPERATOR;
        return tigris_stage_stripe_banding(&ctx->candidate, stage, TIGRIS_BAND_HELD_BY_HW_TILE);
    }
    if (!reshape && !leading_rows && tile->tileable &&
        tile->axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH) {
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
         * entitled to be. This is the executor's row band rule in its
         * loader form: accepting a stage the executor declines sends the
         * stage through the normal path instead of refusing the plan. */
        int32_t band_rows = 0;
        int row_banded = tigris_stage_row_band(&ctx->candidate, stage, 1, NULL, &band_rows);
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
         * resolutions. tigris_stage_stripe_contract states both
         * rules, and the executor routes by the same predicate. */
        /* A per-channel operand is one row high, so a rank-4 height
         * stripe loads it whole for every band. The 2D, chain, row
         * band and rank-3 executors have no such rule, so there a
         * binary op may only take full-shape operands. */
        if ((stage->chain_len != 0 || tile->output_width_extent != 0u || rank != 4 || row_banded) &&
            !stage_binary_operands_equal(&ctx->candidate, stage))
            return TIGRIS_ERR_BAD_OPERATOR;

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
        } else if ((rank == 3 && ctx->hdr->version < TIGRIS_SCHEMA_VERSION_TILE_AXIS) ||
                   !tigris_stage_stripe_contract(&ctx->candidate, stage, rank,
                                                 TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH)) {
            return TIGRIS_ERR_BAD_OPERATOR;
        }
        /* The executor bands a stage by its band kind, not by the branch
         * above that admitted it, so a stage of the stripe's kind meets the
         * stripe's check whichever branch that was. */
        if (tigris_stage_band_kind(&ctx->candidate, stage, NULL) == TIGRIS_BAND_KIND_STRIPE)
            return tigris_stage_stripe_banding(&ctx->candidate, stage, TIGRIS_BAND_HELD_BY_STRIPE);
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
    const int inputs_above = note_requirement(&ctx->required.stage_inputs, stage->inputs_count,
                                              TIGRIS_MAX_STAGE_INPUTS);
    const int outputs_above = note_requirement(&ctx->required.stage_outputs, stage->outputs_count,
                                               TIGRIS_MAX_STAGE_OUTPUTS);
    const int chain_above = note_requirement(&ctx->required.chain_stages, stage->chain_len,
                                             TIGRIS_MAX_CHAIN_STAGES);
    if (inputs_above || outputs_above || chain_above) {
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
        /* The record of schema 9 and below; later schemas carry the wide one. */
        const tigris_tile_plan_t *narrow = tigris_stage_tile_plan(&ctx->candidate, stage);
        tigris_tile_plan_v10_t tile = tigris_stage_tiles(&ctx->candidate, stage);
        /* From schema 11 a tileable record states the band kind the stage
         * runs by, and the stage has to meet that kind's conditions. */
        tigris_stage_band_t stated_band = {0, 0, 0};
        tigris_band_kind_t kind = TIGRIS_BAND_KIND_NONE;
        if (ctx->candidate.tile_plans_v11) {
            kind = tigris_stage_band_kind(&ctx->candidate, stage, &stated_band);
            if (tile.tileable && kind == TIGRIS_BAND_KIND_NONE)
                return TIGRIS_ERR_TILE_FACTS;
        }
        if (narrow == NULL && !tile_v10_valid(&ctx->candidate, stage, &tile))
            return TIGRIS_ERR_BAD_OPERATOR;
        tigris_reshape_band_t reshape_view;
        int reshape;
        int32_t leading_rows;
        if (ctx->candidate.tile_plans_v11) {
            reshape = kind == TIGRIS_BAND_KIND_RESHAPE;
            leading_rows = kind == TIGRIS_BAND_KIND_LEADING ? stated_band.rows : 0;
        } else {
            reshape = tigris_reshape_band(&ctx->candidate, stage, &reshape_view);
            if (reshape && narrow != NULL && narrow->tileable &&
                !tigris_reshape_tile_valid(narrow, &reshape_view))
                return TIGRIS_ERR_BAD_OPERATOR;
            leading_rows = tigris_stage_leading_band(&ctx->candidate, stage);
        }
        if (leading_rows && narrow != NULL && narrow->tileable) {
            tigris_reshape_band_t band = {1, leading_rows, 1, leading_rows, 1, 0u};
            if (!tigris_reshape_tile_valid(narrow, &band)) return TIGRIS_ERR_BAD_OPERATOR;
        }
        tigris_error_t band_error = validate_stage_tile_band(
            ctx, stage, &tile, reshape, leading_rows);
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
        /* Chain execution keeps bounded spatial-op metadata in its workspace. */
        uint32_t spatial_count = 0;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx = ctx->candidate.index_pool[stage->ops_off + j];
            tigris_height_op_t kind = tigris_height_op(ctx->candidate.ops[op_idx].op_type);
            if (kind == TIGRIS_HEIGHT_OP_SPATIAL) {
                spatial_count++;
                if (note_requirement(&ctx->required.spatial_ops_per_stage, spatial_count,
                                     TIGRIS_MAX_SPATIAL_OPS_PER_STAGE)) {
                    if (ctx->out_required) *ctx->out_required = ctx->required;
                    return TIGRIS_ERR_PLAN_LIMITS;
                }
            } else if (kind != TIGRIS_HEIGHT_OP_POINTWISE) {
                return TIGRIS_ERR_BAD_OPERATOR;
            }
        }
        /* The chain executor sizes every band it holds as [N, rows, W, C]. */
        if (!stage_tensors_rank4(&ctx->candidate, stage, 1))
            return TIGRIS_ERR_BAD_SECTION;
        if (!tigris_stage_reads_its_band(&ctx->candidate, stage, TIGRIS_BAND_HELD_BY_CHAIN))
            return TIGRIS_ERR_CHAIN_BAND;
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
        uint32_t qd_len = (ctx->section_ends[TIGRIS_SEC_QUANT_PARAMS] - qp_off - 4u -
                           (uint32_t)nqp * sizeof(tigris_quant_param_t)) / sizeof(int32_t);
        for (uint16_t i = 0; i < nqp; i++) {
            const tigris_quant_param_t *qp = &ctx->candidate.quant_params[i];
            /* Indices rather than pointers: the offsets are checked below. */
            uint32_t moff = tigris_quant_page_start(qp) + qp->multiplier_off;
            uint32_t soff = tigris_quant_page_start(qp) + qp->shift_off;
            if (!isfinite(qp->scale) || qp->scale <= 0.0f)
                return TIGRIS_ERR_BAD_TENSOR;
            if (qp->num_channels == 0 || qp->_pad >= qd_field ||
                (uint32_t)qp->multiplier_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS ||
                (uint32_t)qp->shift_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS ||
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

/* Whether weight `index` (or none) lies inside a stage's decoded block. */
static int weight_in_block(const tigris_plan_t *plan, uint16_t index,
                           const tigris_weight_block_t *block)
{
    if (index == TIGRIS_NO_WEIGHT)
        return 1;
    return block != NULL &&
           bounds_ok(plan->weight_entries[index].offset, plan->weight_entries[index].size_bytes,
                     block->uncompressed_size);
}

/* The weights a stage's operators read, in turn: each operator's weight and
 * bias, then every constant its attribute records list. */
typedef struct {
    uint16_t op, end_op;     /* operators still to read; a stage's are consecutive */
    uint16_t attr, end_attr; /* their attribute records, sorted by operator */
    uint8_t slot;            /* 0 weight or 1 bias, then the constant in the record */
} weight_reads_t;

static int next_weight_read(const tigris_plan_t *plan, weight_reads_t *reads, uint16_t *index)
{
    if (reads->op < reads->end_op) {
        const tigris_op_t *op = &plan->ops[reads->op];
        if (reads->slot == 0u) {
            *index = op->weight_idx;
            reads->slot = 1u;
        } else {
            *index = op->bias_idx;
            reads->slot = 0u;
            reads->op++;
        }
        return 1;
    }
    while (reads->attr < reads->end_attr) {
        const tigris_op_attribute_t *attr = &plan->op_attributes[reads->attr];
        const uint32_t at = (uint32_t)reads->slot * 2u;
        if (attr->type == TIGRIS_OP_ATTR_CONSTANTS && at + 2u <= attr->data_len) {
            const uint8_t *list = plan->op_attribute_data + attr->data_offset;
            *index = (uint16_t)((uint16_t)list[at] | (uint16_t)((uint16_t)list[at + 1u] << 8));
            reads->slot++;
            return 1;
        }
        reads->attr++;
        reads->slot = 0u;
    }
    return 0;
}

/* Whether weight `index` shares bytes with a weight read after it; all of
 * them already lie inside the stage's block, so no end overflows. */
static int weight_overlaps_later(const tigris_plan_t *plan, const weight_reads_t *later,
                                 uint16_t index)
{
    if (index == TIGRIS_NO_WEIGHT || plan->weight_entries[index].size_bytes == 0u)
        return 0;
    const tigris_weight_entry_t *weight = &plan->weight_entries[index];
    weight_reads_t reads = *later;
    uint16_t other = TIGRIS_NO_WEIGHT;
    while (next_weight_read(plan, &reads, &other)) {
        if (other == TIGRIS_NO_WEIGHT || other == index)
            continue;
        const tigris_weight_entry_t *entry = &plan->weight_entries[other];
        if (entry->size_bytes != 0u && entry->offset < weight->offset + weight->size_bytes &&
            weight->offset < entry->offset + entry->size_bytes)
            return 1;
    }
    return 0;
}

static tigris_error_t validate_graph_and_publish(tigris_load_state_t *ctx)
{
    if (ctx->next_scheduled_op != ctx->hdr->num_ops)
        return TIGRIS_ERR_BAD_SECTION;

    /* A compressed stage redirects weight_entry offsets into its own decoded
     * block.  Check that every weight an operator reads (weight, bias and
     * listed constants) lies inside that block, and that no two of them
     * share bytes, before any executor or kernel can dereference the offset. */
    if (ctx->candidate.weight_blocks) {
        uint16_t block_cursor = 0;
        uint16_t attr_cursor = 0;
        for (uint16_t s = 0; s < ctx->hdr->num_stages; s++) {
            const tigris_weight_block_t *block = NULL;
            if (block_cursor < ctx->candidate.num_weight_blocks &&
                ctx->candidate.weight_blocks[block_cursor].stage_idx == s) {
                block = &ctx->candidate.weight_blocks[block_cursor++];
            }

            const tigris_stage_t *stage = &ctx->candidate.stages[s];
            weight_reads_t stage_reads;
            stage_reads.op = stage->ops_count > 0u ? ctx->candidate.index_pool[stage->ops_off] : 0u;
            stage_reads.end_op = (uint16_t)(stage_reads.op + stage->ops_count);
            while (attr_cursor < ctx->candidate.num_op_attributes &&
                   ctx->candidate.op_attributes[attr_cursor].op_index < stage_reads.op)
                attr_cursor++;
            stage_reads.attr = attr_cursor;
            while (attr_cursor < ctx->candidate.num_op_attributes &&
                   ctx->candidate.op_attributes[attr_cursor].op_index < stage_reads.end_op)
                attr_cursor++;
            stage_reads.end_attr = attr_cursor;
            stage_reads.slot = 0u;

            weight_reads_t reads = stage_reads;
            uint16_t index = TIGRIS_NO_WEIGHT;
            while (next_weight_read(&ctx->candidate, &reads, &index)) {
                if (!weight_in_block(&ctx->candidate, index, block))
                    return TIGRIS_ERR_BAD_SECTION;
            }
            reads = stage_reads;
            while (next_weight_read(&ctx->candidate, &reads, &index)) {
                if (weight_overlaps_later(&ctx->candidate, &reads, index))
                    return TIGRIS_ERR_BAD_SECTION;
            }
        }
    }

    {
        uint16_t depth = 0u;
        tigris_error_t graph_error = validate_subgraphs(&ctx->candidate, ctx->index_count, &depth);
        if (graph_error != TIGRIS_OK)
            return graph_error;
        ctx->candidate.subgraph_depth = depth;
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
            return "tile record disagrees with the tiling the runtime derives from the plan";
        case TIGRIS_ERR_CHAIN_BAND:
            return "chained stage reads a tensor outside its band; recompile the model with this TiGrIS release";
        case TIGRIS_ERR_TILE_BAND:
            return "tiled stage reads or writes a tensor outside its band; recompile the model with this TiGrIS release";
        default:                    return "unknown error";
    }
}

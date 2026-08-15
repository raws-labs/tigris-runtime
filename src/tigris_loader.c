/**
 * @file tigris_loader.c
 * @brief Zero-alloc binary plan parser.
 *
 * Validates the file header, walks the section directory, and sets
 * pointers into the buffer. Never allocates memory.
 */

#include "tigris_loader.h"

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
    return tensor->size_bytes / (tensor->dtype == 1 ? sizeof(float) : 1u);
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

static int is_axis1_unary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH;
}

static int is_axis1_binary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_ADD || type == TIGRIS_OP_MUL;
}

static tigris_error_t validate_operator_semantics(const tigris_plan_t *plan)
{
    const tigris_file_header_t *hdr = plan->header;

    for (uint16_t op_index = 0; op_index < hdr->num_ops; op_index++) {
        const tigris_op_t *op = &plan->ops[op_index];
        if (op->num_inputs == 0 || op->num_outputs == 0 ||
            op->fused_act > TIGRIS_ACT_RELU6 || op->_pad1 != 0)
            return TIGRIS_ERR_BAD_OPERATOR;

        const uint16_t *inputs = tigris_op_inputs(plan, op);
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        const tigris_tensor_t *input = &plan->tensors[inputs[0]];
        const tigris_tensor_t *output = &plan->tensors[outputs[0]];
        uint8_t dtype = input->dtype;

        for (uint8_t i = 0; i < op->num_inputs; i++) {
            if (plan->tensors[inputs[i]].dtype != dtype)
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        for (uint8_t i = 0; i < op->num_outputs; i++) {
            if (plan->tensors[outputs[i]].dtype != dtype)
                return TIGRIS_ERR_BAD_OPERATOR;
        }
        if (dtype == 3 && op->act_min > op->act_max)
            return TIGRIS_ERR_BAD_OPERATOR;

        int allows_fused_activation =
            op->op_type == TIGRIS_OP_CONV ||
            op->op_type == TIGRIS_OP_DEPTHWISE ||
            op->op_type == TIGRIS_OP_FULLY_CONN ||
            op->op_type == TIGRIS_OP_CONV1D;
        if (!allows_fused_activation && op->fused_act != TIGRIS_ACT_NONE)
            return TIGRIS_ERR_BAD_OPERATOR;

        switch ((tigris_op_type_t)op->op_type) {
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
                (uint64_t)op->spatial.kernel_h * op->spatial.kernel_w *
                input_channels;
            if (op->op_type == TIGRIS_OP_CONV) {
                if (op->spatial.group != 1)
                    return TIGRIS_ERR_BAD_OPERATOR;
                weight_elements *= output_channels;
            } else {
                if (op->spatial.group != input_channels ||
                    output_channels != input_channels)
                    return TIGRIS_ERR_BAD_OPERATOR;
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
            if (dtype == 3 && output->ndim == 2)
                batches = (uint32_t)out_shape[0];
            if ((dtype == 1 && output_elements != output_channels) ||
                (dtype == 3 &&
                 (output->ndim > 2 || output_elements != batches * output_channels ||
                  input_elements % batches != 0)))
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
                input->size_bytes != output->size_bytes)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_TRANSPOSE:
            if (!op_has_plain_io(op, 1, 1) ||
                input->size_bytes != output->size_bytes)
                return TIGRIS_ERR_BAD_OPERATOR;
            break;

        case TIGRIS_OP_ADD:
        case TIGRIS_OP_MUL:
            if (op->num_outputs != 1 ||
                (op->num_inputs != 1 && op->num_inputs != 2) ||
                op->bias_idx != TIGRIS_NO_WEIGHT ||
                !tensor_shapes_equal(plan, input, output))
                return TIGRIS_ERR_BAD_OPERATOR;
            if (op->num_inputs == 2) {
                if (op->weight_idx != TIGRIS_NO_WEIGHT ||
                    !tensor_shapes_equal(
                        plan, input, &plan->tensors[inputs[1]]))
                    return TIGRIS_ERR_BAD_OPERATOR;
            } else {
                if (dtype != 1 || op->weight_idx == TIGRIS_NO_WEIGHT ||
                    !plan->weight_entries ||
                    (plan->weight_entries[op->weight_idx].size_bytes !=
                         sizeof(float) &&
                     plan->weight_entries[op->weight_idx].size_bytes !=
                         input->size_bytes))
                    return TIGRIS_ERR_BAD_OPERATOR;
            }
            break;

        case TIGRIS_OP_MAX_POOL:
        case TIGRIS_OP_AVG_POOL: {
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
            if (op->num_inputs == 0 || op->num_outputs != 1 ||
                op->weight_idx != TIGRIS_NO_WEIGHT ||
                op->bias_idx != TIGRIS_NO_WEIGHT || output->ndim != 4)
                return TIGRIS_ERR_BAD_OPERATOR;
            const int32_t *out_shape = tigris_tensor_shape(plan, output);
            uint64_t channels = 0;
            for (uint8_t i = 0; i < op->num_inputs; i++) {
                const tigris_tensor_t *part = &plan->tensors[inputs[i]];
                if (part->ndim != 4)
                    return TIGRIS_ERR_BAD_OPERATOR;
                const int32_t *shape = tigris_tensor_shape(plan, part);
                if (shape[0] != out_shape[0] || shape[1] != out_shape[1] ||
                    shape[2] != out_shape[2])
                    return TIGRIS_ERR_BAD_OPERATOR;
                channels += (uint32_t)shape[3];
            }
            if (channels != (uint32_t)out_shape[3])
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
            if (!weight_size_is(plan, op->weight_idx, weight_bytes) ||
                !optional_bias_size_is(plan, op->bias_idx, bias_bytes))
                return TIGRIS_ERR_BAD_OPERATOR;
            break;
        }

        case TIGRIS_OP_RESIZE: {
            if (!op_has_plain_io(op, 1, 1) ||
                input->ndim != 4 || output->ndim != 4 ||
                op->spatial.stride_h == 0 || op->spatial.stride_w == 0)
                return TIGRIS_ERR_BAD_OPERATOR;
            const int32_t *in_shape = tigris_tensor_shape(plan, input);
            const int32_t *out_shape = tigris_tensor_shape(plan, output);
            if (in_shape[0] != out_shape[0] || in_shape[3] != out_shape[3] ||
                (uint64_t)in_shape[1] * op->spatial.stride_h !=
                    (uint32_t)out_shape[1] ||
                (uint64_t)in_shape[2] * op->spatial.stride_w !=
                    (uint32_t)out_shape[2])
                return TIGRIS_ERR_BAD_OPERATOR;
            break;
        }

        default:
            return TIGRIS_ERR_BAD_OPERATOR;
        }
    }

    return TIGRIS_OK;
}

/* Public API */

tigris_error_t tigris_plan_load(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan)
{
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
     * exclusive end from the next distinct section start. */
    uint32_t dir_end = sec_cursor + sizeof(tigris_section_entry_t);
    for (uint32_t i = 1; i < TIGRIS_SEC_MAX; i++) {
        if (!section_seen[i])
            continue;
        if (section_offsets[i] < dir_end)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t end = buf_len;
        for (uint32_t j = 1; j < TIGRIS_SEC_MAX; j++) {
            if (section_seen[j] && section_offsets[j] > section_offsets[i] &&
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

    if (hdr->num_tensors > TIGRIS_MAX_TENSORS)
        return TIGRIS_ERR_PLAN_LIMITS;

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
                               TIGRIS_TENSOR_MODEL_OUTPUT)) != 0 ||
            tensor->_pad != 0 ||
            (tensor->dtype == 1 &&
             tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM))
            return TIGRIS_ERR_BAD_TENSOR;
        uint32_t elements = 1;
        for (uint8_t dim = 0; dim < tensor->ndim; dim++) {
            int32_t value = candidate.shape_pool[tensor->shape_off + dim];
            if (value <= 0 || (uint32_t)value > UINT32_MAX / elements)
                return TIGRIS_ERR_BAD_TENSOR;
            elements *= (uint32_t)value;
        }
        uint32_t element_size;
        if (tensor->dtype == 1)
            element_size = sizeof(float);
        else if (tensor->dtype == 3)
            element_size = sizeof(int8_t);
        else
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
     * Schemas v2/v3 predate typed attributes and intentionally allowed custom
     * dispatch callbacks to implement operators such as legacy Transpose,
     * MatMul, and application opcodes. Preserve that structural-loader
     * contract; schema v4 introduced the strict built-in semantic contract. */
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
            if (attr->type != TIGRIS_OP_ATTR_TRANSPOSE_PERM ||
                op->op_type != TIGRIS_OP_TRANSPOSE ||
                op->num_inputs != 1 || op->num_outputs != 1)
                return TIGRIS_ERR_BAD_SECTION;
            uint16_t input_idx = candidate.index_pool[op->inputs_off];
            uint16_t output_idx = candidate.index_pool[op->outputs_off];
            const tigris_tensor_t *input = &candidate.tensors[input_idx];
            const tigris_tensor_t *output = &candidate.tensors[output_idx];
            if (attr->data_len != input->ndim || output->ndim != input->ndim)
                return TIGRIS_ERR_BAD_SECTION;
            const uint8_t *perm = candidate.op_attribute_data + attr->data_offset;
            for (uint8_t axis = 0; axis < attr->data_len; axis++) {
                if (perm[axis] >= attr->data_len ||
                    output->size_bytes != input->size_bytes)
                    return TIGRIS_ERR_BAD_SECTION;
                for (uint8_t prior = 0; prior < axis; prior++) {
                    if (perm[prior] == perm[axis])
                        return TIGRIS_ERR_BAD_SECTION;
                }
                const int32_t *in_shape = tigris_tensor_shape(&candidate, input);
                const int32_t *out_shape = tigris_tensor_shape(&candidate, output);
                if (out_shape[axis] != in_shape[perm[axis]])
                    return TIGRIS_ERR_BAD_SECTION;
            }
            previous_op = attr->op_index;
            previous_attr_type = attr->type;
        }
        uint16_t attr_cursor = 0;
        for (uint16_t op_idx = 0; op_idx < hdr->num_ops; op_idx++) {
            int has_transpose_perm = 0;
            while (attr_cursor < candidate.num_op_attributes &&
                   candidate.op_attributes[attr_cursor].op_index == op_idx) {
                if (candidate.op_attributes[attr_cursor].type ==
                    TIGRIS_OP_ATTR_TRANSPOSE_PERM)
                    has_transpose_perm = 1;
                attr_cursor++;
            }
            if ((candidate.ops[op_idx].op_type == TIGRIS_OP_TRANSPOSE) !=
                has_transpose_perm)
                return TIGRIS_ERR_BAD_SECTION;
        }
    } else if (hdr->version >= TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES) {
        for (uint16_t op_idx = 0; op_idx < hdr->num_ops; op_idx++) {
            if (candidate.ops[op_idx].op_type == TIGRIS_OP_TRANSPOSE)
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
        if (nqp != hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
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
        if (stage->inputs_count > TIGRIS_MAX_STAGE_INPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->outputs_count > TIGRIS_MAX_STAGE_OUTPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->chain_len > TIGRIS_MAX_CHAIN_STAGES)
            return TIGRIS_ERR_PLAN_LIMITS;

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
            if (tile->tileable &&
                axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH) {
                if (stage->inputs_count == 0 || stage->outputs_count == 0)
                    return TIGRIS_ERR_BAD_SECTION;
                uint16_t first_input =
                    candidate.index_pool[stage->inputs_off];
                uint16_t first_output =
                    candidate.index_pool[stage->outputs_off];
                uint8_t rank = candidate.tensors[first_input].ndim;
                if ((rank != 3 && rank != 4) ||
                    candidate.tensors[first_output].ndim != rank)
                    return TIGRIS_ERR_BAD_SECTION;
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

                /* Schema v5 adds the rank-3 NLC length contract. Unary
                 * pointwise ops may surround one Conv1D. Dynamic binary ops
                 * remain pointwise-only: mixing an external binary operand
                 * with a strided Conv1D can expose different length
                 * resolutions. Rank-4 stages retain the existing audited
                 * height contract and are checked again by the executor. */
                if (rank == 3) {
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
                } else {
                    uint16_t spatial_count = 0;
                    for (uint16_t j = 0; j < stage->ops_count; j++) {
                        uint8_t type = candidate.ops[
                            candidate.index_pool[stage->ops_off + j]
                        ].op_type;
                        if (type == TIGRIS_OP_CONV ||
                            type == TIGRIS_OP_DEPTHWISE ||
                            type == TIGRIS_OP_MAX_POOL ||
                            type == TIGRIS_OP_AVG_POOL) {
                            spatial_count++;
                        } else if (type != TIGRIS_OP_RELU &&
                                   type != TIGRIS_OP_RELU6 &&
                                   type != TIGRIS_OP_SIGMOID &&
                                   type != TIGRIS_OP_TANH &&
                                   type != TIGRIS_OP_ADD &&
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
                    op_type == TIGRIS_OP_AVG_POOL) {
                    spatial_count++;
                    if (spatial_count > TIGRIS_MAX_SPATIAL_OPS_PER_STAGE)
                        return TIGRIS_ERR_PLAN_LIMITS;
                } else if (op_type != TIGRIS_OP_RELU &&
                           op_type != TIGRIS_OP_RELU6 &&
                           op_type != TIGRIS_OP_SIGMOID &&
                           op_type != TIGRIS_OP_TANH &&
                           op_type != TIGRIS_OP_ADD &&
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

    if (hdr->version >= TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES) {
        tigris_error_t semantic_error = validate_operator_semantics(&candidate);
        if (semantic_error != TIGRIS_OK)
            return semantic_error;
    }

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
        default:                    return "unknown error";
    }
}

/**
 * @file tigris_kernels_cmsis_nn.c
 * @brief CMSIS-NN accelerated int8 kernel adapter for TiGrIS runtime.
 *
 * Thin adapter mapping TiGrIS plan structures to CMSIS-NN API calls.
 * Unsupported ops fall back to the reference tigris_dispatch_kernel_s8().
 *
 * CMSIS-NN API reference: https://github.com/ARM-software/CMSIS-NN
 * Data layout: NHWC (matches CMSIS-NN convention).
 *
 * CMSIS-NN requires a scratch buffer (cmsis_nn_context) for some ops.
 * This adapter allocates scratch from the fast arena's remaining space.
 */

#ifdef TIGRIS_HAS_CMSIS_NN

#include "tigris_kernels_cmsis_nn.h"
#include "tigris_accel_policy.h"
#include "tigris_kernels_s8.h"

#include "arm_nnfunctions.h"
#include <limits.h>

/* Helpers */

static inline int32_t get_mult(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off + ch];
}

static inline int32_t get_shft(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off + ch];
}

/* Scratch reservation.
 *
 * tigris_cmsis_nn_prepare() reserves a single 16-aligned scratch buffer, sized
 * to the largest CMSIS-NN kernel buffer across the plan, carved from the TOP of
 * the fast arena (mirrors tigris_esp_nn_prepare). Kernels then use that buffer
 * instead of stack VLAs - so the per-op CMSIS scratch (conv im2col can be many
 * KB on large models) can never overflow the stack.
 *
 * A plan with a positive scratch query must be prepared before dispatch. */
static uint8_t  *s_scratch = NULL;       /* arena region set by prepare(), or NULL */
static uint32_t  s_scratch_size = 0;
static int32_t  *s_quant_mult = NULL;
static int32_t  *s_quant_shift = NULL;
static uint32_t  s_quant_channels = 0;
static uint32_t  s_workspace_size = 0;
static tigris_mem_t *s_prepared_mem = NULL;
static uint32_t s_prepared_fast_size = 0;
static uint8_t   s_zero_scratch[16]
    __attribute__((aligned(16)));

static void cmsis_partition_workspace(
    uint8_t *workspace, uint32_t workspace_size, uint32_t quant_bytes)
{
    s_quant_mult = NULL;
    s_quant_shift = NULL;
    s_quant_channels = 0;
    if (quant_bytes) {
        s_quant_mult = (int32_t *)workspace;
        s_quant_channels = quant_bytes / (2u * sizeof(int32_t));
        s_quant_shift = s_quant_mult + s_quant_channels;
    }
    s_scratch = workspace + quant_bytes;
    s_scratch_size = workspace_size - quant_bytes;
}

/* Return a 16-aligned scratch buffer of at least `need` bytes, or NULL. */
static uint8_t *cmsis_scratch(uint32_t need)
{
    if (s_scratch && need <= s_scratch_size)
        return s_scratch;
    if (need == 0)
        return s_zero_scratch;
    return NULL;
}

static int cmsis_per_channel_quant(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp,
    uint32_t channels, cmsis_nn_per_channel_quant_params *quant)
{
    if (!plan || !quant || channels == 0)
        return -1;

    if (qp && qp->num_channels == channels) {
        uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2
            ? 0u : qp->_pad * TIGRIS_QUANT_PAGE_ELEMS;
        quant->multiplier = (int32_t *)&plan->quant_data[
            page + qp->multiplier_off];
        quant->shift = (int32_t *)&plan->quant_data[page + qp->shift_off];
        return 0;
    }

    if ((qp && qp->num_channels != 1u) ||
        !s_quant_mult || !s_quant_shift || channels > s_quant_channels)
        return -1;

    for (uint32_t channel = 0; channel < channels; channel++) {
        s_quant_mult[channel] = qp ? get_mult(plan, qp, 0) : 1;
        s_quant_shift[channel] = qp ? get_shft(plan, qp, 0) : 0;
    }
    quant->multiplier = s_quant_mult;
    quant->shift = s_quant_shift;
    return 0;
}

/* Conv2D */

static int adapt_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1], IW = x_shape[2], IC = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2], OC = y_shape[3];
    int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

    cmsis_nn_conv_params conv_params = {
        .input_offset  = in_qp  ? -in_qp->zero_point  : 0,
        .output_offset = out_qp ? out_qp->zero_point : 0,
        .stride   = { .w = op->spatial.stride_w,  .h = op->spatial.stride_h },
        .padding  = { .w = op->spatial.pad_left,   .h = op->spatial.pad_top },
        .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                      .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    cmsis_nn_per_channel_quant_params quant;
    if (OC <= 0 || cmsis_per_channel_quant(
            plan, out_qp, (uint32_t)OC, &quant) != 0)
        return -1;

    cmsis_nn_dims input_dims  = { .n = N, .h = IH, .w = IW, .c = IC };
    cmsis_nn_dims filter_dims = { .n = OC, .h = KH, .w = KW, .c = IC };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims output_dims = { .n = N, .h = OH, .w = OW, .c = OC };

    /* Wrapper auto-selects the optimal kernel per shape: arm_convolve_1x1_s8_fast
     * for 1x1 / stride-1 / pad-0 / dilation-1 (the pointwise convs), the 1xN
     * kernel for row convs, and the generic im2col path otherwise. Weights/bias
     * are 16-aligned by the compiler and the scratch below is 16-aligned, which
     * the fast/DSP/MVE paths need (they LDRD / SIMD-load these buffers). */
    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(
        &conv_params, &input_dims, &filter_dims, &output_dims);
    uint8_t *scratch = cmsis_scratch(buf_size > 0 ? (uint32_t)buf_size : 0);
    if (!scratch)
        return -1;
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
        &ctx, &conv_params, &quant,
        &input_dims, X,
        &filter_dims, W,
        &bias_dims, B,
        &output_dims, Y);

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Depthwise Conv2D */

static int adapt_depthwise_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1], IW = x_shape[2], C = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2];
    int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

    /* Weight layout: [KH, KW, C] (HWC) from compiler.
     * CMSIS-NN expects [1, KH, KW, C] via filter_dims.n=1 - same data. */

    cmsis_nn_dw_conv_params dw_params = {
        .input_offset  = in_qp  ? -in_qp->zero_point  : 0,
        .output_offset = out_qp ? out_qp->zero_point : 0,
        .ch_mult = 1,
        .stride   = { .w = op->spatial.stride_w,  .h = op->spatial.stride_h },
        .padding  = { .w = op->spatial.pad_left,   .h = op->spatial.pad_top },
        .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                      .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    cmsis_nn_per_channel_quant_params quant;
    if (C <= 0 || cmsis_per_channel_quant(
            plan, out_qp, (uint32_t)C, &quant) != 0)
        return -1;

    cmsis_nn_dims input_dims  = { .n = N, .h = IH, .w = IW, .c = C };
    cmsis_nn_dims filter_dims = { .n = 1, .h = KH, .w = KW, .c = C };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = C };
    cmsis_nn_dims output_dims = { .n = N, .h = OH, .w = OW, .c = C };

    /* Wrapper picks the optimized depthwise kernel (DSP/MVE SIMD) for the
     * common 3x3 / stride-1 / ch_mult-1 case. The opt kernel reads weights/bias
     * with LDRD / wide SIMD loads, so they must be aligned: the compiler aligns
     * every weight blob to 16 in the plan (writer.py WEIGHT_ALIGN), and the
     * activation buffers use TIGRIS_TENSOR_ALIGN (16 on DSP Cortex-M). The
     * scratch below is over-allocated and 16-aligned. */
    int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
        &dw_params, &input_dims, &filter_dims, &output_dims);
    uint8_t *scratch = cmsis_scratch(buf_size > 0 ? (uint32_t)buf_size : 0);
    if (!scratch)
        return -1;
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(
        &ctx, &dw_params, &quant,
        &input_dims, X,
        &filter_dims, W,
        &bias_dims, B,
        &output_dims, Y);

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Fully Connected */

static int adapt_fully_connected(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int N  = (y_tensor->ndim >= 2) ? y_shape[0] : 1;
    int OC = (y_tensor->ndim >= 2) ? y_shape[1] : y_shape[0];

    const tigris_tensor_t *x_tensor = &plan->tensors[ins[0]];
    const int32_t *x_shape = tigris_tensor_shape(plan, x_tensor);
    uint32_t x_numel = 1;
    for (uint8_t i = 0; i < x_tensor->ndim; i++)
        x_numel *= (uint32_t)x_shape[i];
    int IC = (int)(x_numel / (uint32_t)N);

    /* TFLite FC weights are commonly per-channel (one scale per output class).
     * The compiler already emits the per-output-channel multiplier/shift as two
     * contiguous arrays in the plan's quant_data, so pass them to CMSIS-NN by
     * pointer - no per-inference copy onto the stack. */
    int per_channel = (out_qp && out_qp->num_channels > 1);

    cmsis_nn_fc_params fc_params = {
        .input_offset  = in_qp  ? -in_qp->zero_point  : 0,
        .filter_offset = 0,  /* symmetric weight quant */
        .output_offset = out_qp ? out_qp->zero_point : 0,
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    /* CMSIS-NN FC filter_dims format is [N, C] where N = accumulation depth
     * (IC) and C = output depth (OC) - NOT the conv [C_OUT,..,C_IN] layout. */
    cmsis_nn_dims input_dims  = { .n = N,  .h = 1, .w = 1, .c = IC };
    cmsis_nn_dims filter_dims = { .n = IC, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims bias_dims   = { .n = 1,  .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims output_dims = { .n = N,  .h = 1, .w = 1, .c = OC };

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    uint8_t *scratch = cmsis_scratch(buf_size > 0 ? (uint32_t)buf_size : 0);
    if (!scratch)
        return -1;
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };
    if (buf_size > 0) {
        /* The DSP/MVE arm_fully_connected_s8 reads a precomputed per-output
         * kernel sum (input_offset folded) from ctx->buf; the plain-C path
         * ignores it. Compute it the way CMSIS-NN does internally. */
        arm_vector_sum_s8((int32_t *)scratch, IC, OC, W,
                          fc_params.input_offset, 0, NULL);
    }

    arm_cmsis_nn_status status;
    if (per_channel) {
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)&plan->quant_data[(plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : out_qp->_pad * TIGRIS_QUANT_PAGE_ELEMS) + out_qp->multiplier_off],
            .shift      = (int32_t *)&plan->quant_data[(plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : out_qp->_pad * TIGRIS_QUANT_PAGE_ELEMS) + out_qp->shift_off] };
        status = arm_fully_connected_per_channel_s8(
            &ctx, &fc_params, &quant, &input_dims, X,
            &filter_dims, W, &bias_dims, B, &output_dims, Y);
    } else {
        cmsis_nn_per_tensor_quant_params quant = {
            .multiplier = out_qp ? get_mult(plan, out_qp, 0) : 1,
            .shift      = out_qp ? get_shft(plan, out_qp, 0) : 0 };
        status = arm_fully_connected_s8(
            &ctx, &fc_params, &quant, &input_dims, X,
            &filter_dims, W, &bias_dims, B, &output_dims, Y);
    }

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Average Pool / Global Average Pool */

static int adapt_avg_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem, void *user_ctx)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* arm_avgpool_s8 preserves the input scale/zero-point - it does not
     * requantize. If the op rescales (in/out quant differ), fall back to the
     * reference kernel which requantizes. */
    const tigris_quant_param_t *pin_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *pout_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    if (pin_qp && pout_qp &&
        (pin_qp->scale != pout_qp->scale || pin_qp->zero_point != pout_qp->zero_point))
        return tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1], IW = x_shape[2], C = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2];

    cmsis_nn_pool_params pool_params;
    pool_params.activation.min = op->act_min;
    pool_params.activation.max = op->act_max;

    if ((tigris_op_type_t)op->op_type == TIGRIS_OP_GLOBAL_AVG) {
        pool_params.stride.h  = 1;
        pool_params.stride.w  = 1;
        pool_params.padding.h = 0;
        pool_params.padding.w = 0;
    } else {
        pool_params.stride.h  = op->spatial.stride_h;
        pool_params.stride.w  = op->spatial.stride_w;
        pool_params.padding.h = op->spatial.pad_top;
        pool_params.padding.w = op->spatial.pad_left;
    }

    int FH = ((tigris_op_type_t)op->op_type == TIGRIS_OP_GLOBAL_AVG) ? IH : op->spatial.kernel_h;
    int FW = ((tigris_op_type_t)op->op_type == TIGRIS_OP_GLOBAL_AVG) ? IW : op->spatial.kernel_w;

    cmsis_nn_dims input_dims  = { .n = N, .h = IH, .w = IW, .c = C };
    cmsis_nn_dims filter_dims = { .n = 1, .h = FH, .w = FW, .c = 1 };
    cmsis_nn_dims output_dims = { .n = N, .h = OH, .w = OW, .c = C };

    int32_t buf_size = arm_avgpool_s8_get_buffer_size(OW, C);
    uint8_t *scratch = cmsis_scratch(buf_size > 0 ? (uint32_t)buf_size : 0);
    if (!scratch)
        return -1;
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_avgpool_s8(
        &ctx, &pool_params,
        &input_dims, X,
        &filter_dims,
        &output_dims, Y);

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Scratch sizing and reservation. */

static uint32_t cmsis_workspace_required(
    const tigris_plan_t *plan, uint32_t *quant_bytes_out)
{
    if (!plan || !plan->header ||
        (plan->header->num_ops > 0 &&
         (!plan->ops || !plan->tensors || !plan->index_pool ||
          !plan->shape_pool)))
        return UINT32_MAX;

    uint32_t max_scratch = 0;
    uint32_t max_broadcast_channels = 0;
    for (uint16_t i = 0; i < plan->header->num_ops; i++) {
        const tigris_op_t *op = &plan->ops[i];
        if (tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, plan, op, 0) ==
            TIGRIS_ACCEL_ROUTE_S8_REF)
            continue;
        const uint16_t *ins  = tigris_op_inputs(plan, op);
        const uint16_t *outs = tigris_op_outputs(plan, op);
        const int32_t *xs = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
        const int32_t *ys = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
        const tigris_quant_param_t *out_qp =
            tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
        int32_t s = 0;

        switch ((tigris_op_type_t)op->op_type) {
        case TIGRIS_OP_CONV: {
            cmsis_nn_dims id = { .n = xs[0], .h = xs[1], .w = xs[2], .c = xs[3] };
            cmsis_nn_dims fd = { .n = ys[3], .h = op->spatial.kernel_h,
                                 .w = op->spatial.kernel_w, .c = xs[3] };
            cmsis_nn_dims od = { .n = ys[0], .h = ys[1], .w = ys[2], .c = ys[3] };
            cmsis_nn_conv_params cp = {
                .stride   = { .w = op->spatial.stride_w, .h = op->spatial.stride_h },
                .padding  = { .w = op->spatial.pad_left, .h = op->spatial.pad_top },
                .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                              .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
            };
            s = arm_convolve_wrapper_s8_get_buffer_size(&cp, &id, &fd, &od);
            if ((!out_qp || out_qp->num_channels == 1u) && ys[3] > 0 &&
                (uint32_t)ys[3] > max_broadcast_channels)
                max_broadcast_channels = (uint32_t)ys[3];
            break;
        }
        case TIGRIS_OP_DEPTHWISE: {
            cmsis_nn_dims id = { .n = xs[0], .h = xs[1], .w = xs[2], .c = xs[3] };
            cmsis_nn_dims fd = { .n = 1, .h = op->spatial.kernel_h,
                                 .w = op->spatial.kernel_w, .c = xs[3] };
            cmsis_nn_dims od = { .n = ys[0], .h = ys[1], .w = ys[2], .c = xs[3] };
            cmsis_nn_dw_conv_params dp = {
                .ch_mult  = 1,
                .stride   = { .w = op->spatial.stride_w, .h = op->spatial.stride_h },
                .padding  = { .w = op->spatial.pad_left, .h = op->spatial.pad_top },
                .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                              .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
            };
            s = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dp, &id, &fd, &od);
            if ((!out_qp || out_qp->num_channels == 1u) && xs[3] > 0 &&
                (uint32_t)xs[3] > max_broadcast_channels)
                max_broadcast_channels = (uint32_t)xs[3];
            break;
        }
        case TIGRIS_OP_FULLY_CONN: {
            int OC = (plan->tensors[outs[0]].ndim >= 2) ? ys[1] : ys[0];
            const tigris_tensor_t *xt = &plan->tensors[ins[0]];
            int N = (plan->tensors[outs[0]].ndim >= 2) ? ys[0] : 1;
            if (N <= 0 || xt->size_bytes > (uint32_t)INT_MAX)
                return UINT32_MAX;
            cmsis_nn_dims fd = { .n = (int)(xt->size_bytes / (uint32_t)N),
                                 .h = 1, .w = 1, .c = OC };
            s = arm_fully_connected_s8_get_buffer_size(&fd);
            break;
        }
        case TIGRIS_OP_AVG_POOL:
        case TIGRIS_OP_GLOBAL_AVG:
            s = arm_avgpool_s8_get_buffer_size(ys[2], xs[3]);
            break;
        default:
            break;
        }
        if (s < 0)
            return UINT32_MAX;
        if (s > 0 && (uint32_t)s > max_scratch)
            max_scratch = (uint32_t)s;
    }

    if (max_scratch > UINT32_MAX - 15u ||
        max_broadcast_channels > (UINT32_MAX - 15u) / (2u * sizeof(int32_t)))
        return UINT32_MAX;
    uint32_t kernel_bytes = (max_scratch + 15u) & ~15u;
    uint32_t quant_bytes = (
        max_broadcast_channels * 2u * sizeof(int32_t) + 15u) & ~15u;
    if (quant_bytes > UINT32_MAX - kernel_bytes)
        return UINT32_MAX;
    if (quant_bytes_out)
        *quant_bytes_out = quant_bytes;
    return kernel_bytes + quant_bytes;
}

uint32_t tigris_cmsis_nn_scratch_required(const tigris_plan_t *plan)
{
    return cmsis_workspace_required(plan, NULL);
}

uint32_t tigris_cmsis_nn_fast_arena_required(const tigris_plan_t *plan)
{
    uint32_t core = tigris_fast_arena_required(plan);
    uint32_t scratch = tigris_cmsis_nn_scratch_required(plan);
    if (core == UINT32_MAX || scratch == UINT32_MAX)
        return UINT32_MAX;
    if (scratch == 0)
        return core;
    if (core > UINT32_MAX - 15u)
        return UINT32_MAX;

    uint32_t aligned_core = (core + 15u) & ~15u;
    if (scratch > UINT32_MAX - aligned_core)
        return UINT32_MAX;
    return aligned_core + scratch;
}

int tigris_cmsis_nn_prepare(const tigris_plan_t *plan, tigris_mem_t *mem)
{
    if (!plan || !plan->header || !mem || !mem->fast_base)
        return -1;

    uint32_t quant_bytes = 0;
    uint32_t want = cmsis_workspace_required(plan, &quant_bytes);
    if (want == UINT32_MAX)
        return -1;

    if (s_prepared_mem) {
        /* The adapter has one global CMSIS context. Never silently shrink a
         * second arena or replace a reservation which may still be in use. */
        if (s_prepared_mem != mem || want > s_workspace_size)
            return -1;
        cmsis_partition_workspace(
            mem->fast_base + mem->fast_size, s_workspace_size, quant_bytes);
        return 0;
    }

    if (want == 0)                    /* no op needs scratch */
        return 0;

    uint32_t total_required = tigris_cmsis_nn_fast_arena_required(plan);
    if (total_required == UINT32_MAX || mem->fast_size < total_required)
        return -1;

    /* Carve a 16-aligned region from the top of the fast arena, reducing
     * fast_size so the executor's activation allocations never overlap it. */
    uint32_t new_size = (mem->fast_size - want) & ~15u;
    uint8_t *workspace = mem->fast_base + new_size;
    s_workspace_size = mem->fast_size - new_size; /* >= want, aligned base */
    cmsis_partition_workspace(workspace, s_workspace_size, quant_bytes);
    s_prepared_mem = mem;
    s_prepared_fast_size = mem->fast_size;
    mem->fast_size = new_size;
    return 0;
}

int tigris_cmsis_nn_deinit(tigris_mem_t *mem)
{
    if (!mem || s_prepared_mem != mem)
        return -1;
    mem->fast_size = s_prepared_fast_size;
    s_scratch = NULL;
    s_scratch_size = 0;
    s_quant_mult = NULL;
    s_quant_shift = NULL;
    s_quant_channels = 0;
    s_workspace_size = 0;
    s_prepared_mem = NULL;
    s_prepared_fast_size = 0;
    return 0;
}

/* Dispatch */

int tigris_dispatch_kernel_cmsis_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    /* CMSIS-NN's vendored generic Conv/Depthwise kernels explicitly consume
     * dilation.h/.w, so non-tiled dilation remains native. The shared policy
     * routes tiled spatial ops and pooling that requires requantization to
     * s8_ref before any vendor call. */
    int handled = 0;
    int route_rc = tigris_accel_try_s8_ref(
        TIGRIS_ACCEL_CMSIS_NN, plan, op, op_index, mem, user_ctx, &handled);
    if (handled)
        return route_rc;

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return adapt_conv2d(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return adapt_depthwise_conv2d(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return adapt_fully_connected(plan, op, mem);
    case TIGRIS_OP_AVG_POOL:    return adapt_avg_pool(plan, op, op_index, mem, user_ctx);
    case TIGRIS_OP_GLOBAL_AVG:  return adapt_avg_pool(plan, op, op_index, mem, user_ctx);

    /* Ops without CMSIS-NN equivalent: fall back to reference s8 kernels */
    default:
        return tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);
    }
}

#else
/* ISO C forbids empty translation units */
typedef int tigris_kernels_cmsis_nn_unused_;
#endif /* TIGRIS_HAS_CMSIS_NN */

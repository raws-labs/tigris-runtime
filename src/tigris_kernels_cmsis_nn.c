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
#include "tigris_op_config.h"

#include "arm_nnfunctions.h"
#include <limits.h>

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
static int32_t  *s_quant_buf = NULL;     /* per-tensor to per-channel expansion */
static uint32_t  s_quant_channels = 0;
static uint32_t  s_workspace_size = 0;
static tigris_mem_t *s_prepared_mem = NULL;
static uint32_t s_prepared_fast_size = 0;
static uint8_t   s_zero_scratch[16]
    __attribute__((aligned(16)));

static void cmsis_partition_workspace(
    uint8_t *workspace, uint32_t workspace_size, uint32_t quant_bytes)
{
    s_quant_buf = NULL;
    s_quant_channels = 0;
    if (quant_bytes) {
        s_quant_buf = (int32_t *)workspace;
        s_quant_channels = quant_bytes / (2u * sizeof(int32_t));
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
    const int32_t *multipliers = NULL;
    const int32_t *shifts = NULL;
    if (!quant || !tigris_accel_channel_quant(plan, qp, channels, s_quant_buf,
                                              s_quant_channels, &multipliers, &shifts))
        return -1;
    /* MISRA 11.8 deviation: CMSIS-NN declares these fields non-const but
     * only reads them, so the read-only arrays are passed as is. */
    quant->multiplier = (int32_t *)multipliers;
    quant->shift = (int32_t *)shifts;
    return 0;
}

/* Conv2D and Depthwise Conv2D
 *
 * The call covers the current tile (tigris_accel_tile_view). CMSIS-NN takes
 * only the leading pads: rows and columns past the input extent read as the
 * input zero point, which covers the bottom and right pads of an edge tile. */

/* Stride, pads, dilation (0 reads as 1) and filter dims of a Conv call, for
 * sizing and dispatch; dispatch adds the offsets and the activation. */
static void cmsis_conv_setup(
    const tigris_op_t *op, const cmsis_nn_dims *input, const cmsis_nn_dims *output,
    int32_t pad_top, int32_t pad_left, cmsis_nn_conv_params *params, cmsis_nn_dims *filter)
{
    *params = (cmsis_nn_conv_params){
        .stride   = { .w = op->spatial.stride_w, .h = op->spatial.stride_h },
        .padding  = { .w = pad_left, .h = pad_top },
        .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                      .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
    };
    *filter = (cmsis_nn_dims){ .n = output->c, .h = op->spatial.kernel_h,
                               .w = op->spatial.kernel_w, .c = input->c };
}

/* The same for a Depthwise call; depthwise weights are [KH, KW, OC]. */
static void cmsis_dw_setup(
    const tigris_op_t *op, const cmsis_nn_dims *input, const cmsis_nn_dims *output,
    int32_t pad_top, int32_t pad_left, cmsis_nn_dw_conv_params *params, cmsis_nn_dims *filter)
{
    *params = (cmsis_nn_dw_conv_params){
        .ch_mult  = output->c / input->c,
        .stride   = { .w = op->spatial.stride_w, .h = op->spatial.stride_h },
        .padding  = { .w = pad_left, .h = pad_top },
        .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                      .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
    };
    *filter = (cmsis_nn_dims){ .n = 1, .h = op->spatial.kernel_h,
                               .w = op->spatial.kernel_w, .c = output->c };
}

static int adapt_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, const tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    tigris_accel_tile_view_t v;
    tigris_accel_tile_view(op, mem, x_shape, y_shape, &v);
    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]) + v.in_offset;
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]) + v.out_offset;
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    int N = x_shape[0], OC = y_shape[3];
    cmsis_nn_dims input_dims  = { .n = N, .h = v.in_h, .w = v.in_w, .c = x_shape[3] };
    cmsis_nn_dims output_dims = { .n = N, .h = v.out_h, .w = v.out_w, .c = OC };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims filter_dims;
    cmsis_nn_conv_params conv_params;
    cmsis_conv_setup(op, &input_dims, &output_dims, v.pad_top, v.pad_left,
                     &conv_params, &filter_dims);
    conv_params.input_offset = in_qp ? -in_qp->zero_point : 0;
    conv_params.output_offset = out_qp ? out_qp->zero_point : 0;
    conv_params.activation = (cmsis_nn_activation){ .min = op->act_min, .max = op->act_max };

    cmsis_nn_per_channel_quant_params quant;
    if (OC <= 0 || cmsis_per_channel_quant(
            plan, out_qp, (uint32_t)OC, &quant) != 0)
        return -1;

    /* The wrapper picks the kernel per shape: arm_convolve_1x1_s8_fast for
     * 1x1 / stride-1 / pad-0 / dilation-1, the 1xN kernel for row convs, and
     * im2col otherwise. The fast/DSP/MVE paths read weights, bias and scratch
     * with wide loads: weights are 16-aligned by the compiler, scratch here. */
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

static int adapt_depthwise_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, const tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    tigris_accel_tile_view_t v;
    tigris_accel_tile_view(op, mem, x_shape, y_shape, &v);
    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]) + v.in_offset;
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]) + v.out_offset;
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    int N = x_shape[0], C = x_shape[3], OC = y_shape[3];
    cmsis_nn_per_channel_quant_params quant;
    if (C <= 0 || cmsis_per_channel_quant(
            plan, out_qp, (uint32_t)OC, &quant) != 0)
        return -1;

    cmsis_nn_dims input_dims  = { .n = N, .h = v.in_h, .w = v.in_w, .c = C };
    cmsis_nn_dims output_dims = { .n = N, .h = v.out_h, .w = v.out_w, .c = OC };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims filter_dims;
    cmsis_nn_dw_conv_params dw_params;
    cmsis_dw_setup(op, &input_dims, &output_dims, v.pad_top, v.pad_left,
                   &dw_params, &filter_dims);
    dw_params.input_offset = in_qp ? -in_qp->zero_point : 0;
    dw_params.output_offset = out_qp ? out_qp->zero_point : 0;
    dw_params.activation = (cmsis_nn_activation){ .min = op->act_min, .max = op->act_max };

    /* The wrapper picks the DSP/MVE kernel for 3x3 / stride-1 / ch_mult-1,
     * which needs the same alignment as the Conv kernels. */
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
        /* MISRA 11.8 deviation: read-only plan arrays, see cmsis_per_channel_quant. */
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)tigris_quant_multipliers(plan, out_qp),
            .shift      = (int32_t *)tigris_quant_shifts(plan, out_qp) };
        status = arm_fully_connected_per_channel_s8(
            &ctx, &fc_params, &quant, &input_dims, X,
            &filter_dims, W, &bias_dims, B, &output_dims, Y);
    } else {
        cmsis_nn_per_tensor_quant_params quant = {
            .multiplier = out_qp ? tigris_quant_multipliers(plan, out_qp)[0] : 1,
            .shift      = out_qp ? tigris_quant_shifts(plan, out_qp)[0] : 0 };
        status = arm_fully_connected_s8(
            &ctx, &fc_params, &quant, &input_dims, X,
            &filter_dims, W, &bias_dims, B, &output_dims, Y);
    }

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Average Pool / Global Average Pool */

static int adapt_avg_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

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
        if (tigris_accel_pre_route(TIGRIS_ACCEL_CMSIS_NN, plan, op, i, 0) ==
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
        case TIGRIS_OP_CONV:
        case TIGRIS_OP_DEPTHWISE: {
            cmsis_nn_dims id = { .n = xs[0], .h = xs[1], .w = xs[2], .c = xs[3] };
            cmsis_nn_dims od = { .n = ys[0], .h = ys[1], .w = ys[2], .c = ys[3] };
            cmsis_nn_dims fd;
            if ((tigris_op_type_t)op->op_type == TIGRIS_OP_CONV) {
                cmsis_nn_conv_params cp;
                cmsis_conv_setup(op, &id, &od, op->spatial.pad_top,
                                 op->spatial.pad_left, &cp, &fd);
                s = arm_convolve_wrapper_s8_get_buffer_size(&cp, &id, &fd, &od);
            } else {
                cmsis_nn_dw_conv_params dp;
                cmsis_dw_setup(op, &id, &od, op->spatial.pad_top,
                               op->spatial.pad_left, &dp, &fd);
                s = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dp, &id, &fd, &od);
            }
            if ((!out_qp || out_qp->num_channels == 1u) && ys[3] > 0 &&
                (uint32_t)ys[3] > max_broadcast_channels)
                max_broadcast_channels = (uint32_t)ys[3];
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
    s_quant_buf = NULL;
    s_quant_channels = 0;
    s_workspace_size = 0;
    s_prepared_mem = NULL;
    s_prepared_fast_size = 0;
    return 0;
}

static int adapt_binary(const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    tigris_accel_binary_t params;
    if (!tigris_accel_binary_params(plan, op, mem, &params)) return -1;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const int8_t *a = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    const int8_t *b = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[1]);
    int8_t *y = (int8_t *)tigris_mem_tensor_ptr(mem, tigris_op_outputs(plan, op)[0]);
    if (!a || !b || !y) return -1;
    arm_cmsis_nn_status status;
    if (op->op_type == TIGRIS_OP_MUL) {
        status = arm_elementwise_mul_s8(a, b, params.input1_offset, params.input2_offset, y,
                    params.output_offset, params.multiplier, params.shift,
                    params.activation_min, params.activation_max, params.count);
    } else {
        cmsis_nn_context ctx = { .buf = NULL, .size = 0 };
        cmsis_nn_dims dims = { .n = 1, .h = 1, .w = 1, .c = params.count };
        status = op->op_type == TIGRIS_OP_MAXIMUM
            ? arm_maximum_s8(&ctx, a, &dims, b, &dims, y, &dims)
            : arm_minimum_s8(&ctx, a, &dims, b, &dims, y, &dims);
    }
    return status == ARM_CMSIS_NN_SUCCESS ? 0 : -1;
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
     * routes tiled spatial ops, pooling that requires requantization or
     * rounds as a mean, and the single-rounding exceptions to s8_ref before
     * any vendor call. */
    int handled = 0;
    int route_rc = tigris_accel_try_s8_ref(
        TIGRIS_ACCEL_CMSIS_NN, plan, op, op_index, mem, user_ctx, &handled);
    if (handled)
        return route_rc;

    switch ((tigris_op_type_t)op->op_type) {
#if TIGRIS_KERNEL_S8_CONV
    case TIGRIS_OP_CONV: return adapt_conv2d(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_DEPTHWISE
    case TIGRIS_OP_DEPTHWISE: return adapt_depthwise_conv2d(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_FULLY_CONN
    case TIGRIS_OP_FULLY_CONN: return adapt_fully_connected(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_AVG_POOL
    case TIGRIS_OP_AVG_POOL: return adapt_avg_pool(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_GLOBAL_AVG
    case TIGRIS_OP_GLOBAL_AVG: return adapt_avg_pool(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_MUL
    case TIGRIS_OP_MUL: return adapt_binary(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_MAXIMUM
    case TIGRIS_OP_MAXIMUM: return adapt_binary(plan, op, mem);
#endif
#if TIGRIS_KERNEL_S8_MINIMUM
    case TIGRIS_OP_MINIMUM: return adapt_binary(plan, op, mem);
#endif

    /* Ops without CMSIS-NN equivalent: fall back to reference s8 kernels */
    default:
        return tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);
    }
}

#else
/* ISO C forbids empty translation units */
typedef int tigris_kernels_cmsis_nn_unused_;
#endif /* TIGRIS_HAS_CMSIS_NN */

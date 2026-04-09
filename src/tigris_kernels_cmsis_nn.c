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
#include "tigris_kernels_s8.h"

#include "arm_nnfunctions.h"

/* Helpers */

static inline int32_t get_mult(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    return plan->quant_data[qp->multiplier_off + ch];
}

static inline int32_t get_shft(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    return plan->quant_data[qp->shift_off + ch];
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

    /* Build per-channel multiplier/shift arrays */
    int32_t mult_arr[OC], shift_arr[OC];
    for (int oc = 0; oc < OC; oc++) {
        int ch = (out_qp && out_qp->num_channels > 1) ? oc : 0;
        mult_arr[oc]  = out_qp ? get_mult(plan, out_qp, ch) : 1;
        shift_arr[oc] = out_qp ? get_shft(plan, out_qp, ch) : 0;
    }

    cmsis_nn_conv_params conv_params = {
        .input_offset  = in_qp  ? -in_qp->zero_point  : 0,
        .output_offset = out_qp ? out_qp->zero_point : 0,
        .stride   = { .w = op->spatial.stride_w,  .h = op->spatial.stride_h },
        .padding  = { .w = op->spatial.pad_left,   .h = op->spatial.pad_top },
        .dilation = { .w = op->spatial.dilation_w ? op->spatial.dilation_w : 1,
                      .h = op->spatial.dilation_h ? op->spatial.dilation_h : 1 },
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    cmsis_nn_per_channel_quant_params quant = {
        .multiplier = mult_arr,
        .shift = shift_arr,
    };

    cmsis_nn_dims input_dims  = { .n = N, .h = IH, .w = IW, .c = IC };
    cmsis_nn_dims filter_dims = { .n = OC, .h = KH, .w = KW, .c = IC };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims output_dims = { .n = N, .h = OH, .w = OW, .c = OC };

    /* CMSIS-NN upscale dims (set to 1 for standard conv) */
    cmsis_nn_dims upscale_dims = { .n = 1, .h = 1, .w = 1, .c = 1 };

    /* Scratch buffer from fast arena remaining space */
    int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
    uint8_t scratch[buf_size > 0 ? (uint32_t)buf_size : 1];
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_convolve_s8(
        &ctx, &conv_params, &quant,
        &input_dims, X,
        &filter_dims, W,
        &bias_dims, B,
        &upscale_dims,
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

    int32_t mult_arr[C], shift_arr[C];
    for (int c = 0; c < C; c++) {
        int ch = (out_qp && out_qp->num_channels > 1) ? c : 0;
        mult_arr[c]  = out_qp ? get_mult(plan, out_qp, ch) : 1;
        shift_arr[c] = out_qp ? get_shft(plan, out_qp, ch) : 0;
    }

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

    cmsis_nn_per_channel_quant_params quant = {
        .multiplier = mult_arr,
        .shift = shift_arr,
    };

    cmsis_nn_dims input_dims  = { .n = N, .h = IH, .w = IW, .c = C };
    cmsis_nn_dims filter_dims = { .n = 1, .h = KH, .w = KW, .c = C };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = C };
    cmsis_nn_dims output_dims = { .n = N, .h = OH, .w = OW, .c = C };

    int32_t buf_size = arm_depthwise_conv_s8_get_buffer_size(&input_dims, &filter_dims);
    uint8_t scratch[buf_size > 0 ? (uint32_t)buf_size : 1];
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_depthwise_conv_s8(
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

    /* CMSIS-NN FC uses per-tensor quant */
    int32_t m = out_qp ? get_mult(plan, out_qp, 0) : 1;
    int32_t s = out_qp ? get_shft(plan, out_qp, 0) : 0;

    cmsis_nn_fc_params fc_params = {
        .input_offset  = in_qp  ? -in_qp->zero_point  : 0,
        .filter_offset = 0,  /* symmetric weight quant */
        .output_offset = out_qp ? out_qp->zero_point : 0,
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    cmsis_nn_per_tensor_quant_params quant = {
        .multiplier = m,
        .shift = s,
    };

    cmsis_nn_dims input_dims  = { .n = N, .h = 1, .w = 1, .c = IC };
    cmsis_nn_dims filter_dims = { .n = OC, .h = 1, .w = 1, .c = IC };
    cmsis_nn_dims bias_dims   = { .n = 1, .h = 1, .w = 1, .c = OC };
    cmsis_nn_dims output_dims = { .n = N, .h = 1, .w = 1, .c = OC };

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    uint8_t scratch[buf_size > 0 ? (uint32_t)buf_size : 1];
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &ctx, &fc_params, &quant,
        &input_dims, X,
        &filter_dims, W,
        &bias_dims, B,
        &output_dims, Y);

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
    uint8_t scratch[buf_size > 0 ? (uint32_t)buf_size : 1];
    cmsis_nn_context ctx = { .buf = scratch, .size = buf_size };

    arm_cmsis_nn_status status = arm_avgpool_s8(
        &ctx, &pool_params,
        &input_dims, X,
        &filter_dims,
        &output_dims, Y);

    return (status == ARM_CMSIS_NN_SUCCESS) ? 0 : -1;
}

/* Dispatch */

int tigris_dispatch_kernel_cmsis_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    (void)op_index;
    (void)user_ctx;

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return adapt_conv2d(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return adapt_depthwise_conv2d(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return adapt_fully_connected(plan, op, mem);
    case TIGRIS_OP_AVG_POOL:    return adapt_avg_pool(plan, op, mem);
    case TIGRIS_OP_GLOBAL_AVG:  return adapt_avg_pool(plan, op, mem);

    /* Ops without CMSIS-NN equivalent: fall back to reference s8 kernels */
    default:
        return tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);
    }
}

#else
/* ISO C forbids empty translation units */
typedef int tigris_kernels_cmsis_nn_unused_;
#endif /* TIGRIS_HAS_CMSIS_NN */

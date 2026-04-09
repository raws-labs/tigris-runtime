/**
 * @file tigris_kernels_s8.c
 * @brief Reference int8 (symmetric quantized) kernels + dispatch.
 *
 * All kernels are static. Data layout: NHWC, int8.
 * Quantization parameters accessed via tigris_tensor_quant().
 * Weight/bias data accessed via tigris_op_weight() / tigris_op_bias().
 *
 * Accumulation in int32. Output requantization uses pre-computed
 * Q0.31 multiplier and shift stored in the plan's quant data.
 */

#include "tigris_kernels_s8.h"

#include <math.h>
#include <string.h>

/* Helpers */

/** Clamp an int32 to int8 range. */
static inline int8_t clamp_s8(int32_t x)
{
    if (x < -128) return -128;
    if (x > 127) return 127;
    return (int8_t)x;
}

/** Clamp an int32 to [act_min, act_max] range (fused activation). */
static inline int8_t clamp_act(int32_t x, int8_t act_min, int8_t act_max)
{
    if (x < act_min) return act_min;
    if (x > act_max) return act_max;
    return (int8_t)x;
}

/**
 * Multiply-by-quantized-multiplier (TFLite convention).
 *
 * Two-step: SaturatingRoundingDoublingHighMul (x*m >> 31 with rounding),
 * then apply remaining shift with rounding. Matches TFLite/CMSIS-NN/ESP-NN.
 */
static inline int32_t multiply_by_quantized_multiplier(
    int32_t x, int32_t multiplier, int32_t shift)
{
    /* Step 1: SaturatingRoundingDoublingHighMul - high 32 bits of x*m*2 */
    int64_t ab = (int64_t)x * (int64_t)multiplier;
    int32_t nudge = ab >= 0 ? (1 << 30) : -(1 << 30);
    int32_t high = (int32_t)((ab + nudge) >> 31);

    /* Step 2: Apply remaining shift with rounding */
    if (shift >= 0) {
        return high << shift;
    }
    int right = -shift;
    int32_t mask = (1 << right) - 1;
    int32_t threshold = mask >> 1;
    int32_t remainder = high & mask;
    return (high >> right) + (remainder > threshold ? 1 : 0);
}

/** Get multiplier from quant data (works for both per-tensor and per-channel). */
static inline int32_t get_multiplier(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    return plan->quant_data[qp->multiplier_off + ch];
}

/** Get shift from quant data (works for both per-tensor and per-channel). */
static inline int32_t get_shift(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    return plan->quant_data[qp->shift_off + ch];
}

/** Total number of elements in a tensor. */
static uint32_t tensor_numel(const tigris_plan_t *plan, uint16_t tidx)
{
    const tigris_tensor_t *t = &plan->tensors[tidx];
    const int32_t *shape = tigris_tensor_shape(plan, t);
    uint32_t n = 1;
    for (uint8_t i = 0; i < t->ndim; i++)
        n *= (uint32_t)shape[i];
    return n;
}

/** Tile-aware element count: uses tile dimensions when tiling active. */
static uint32_t tile_aware_numel(const tigris_plan_t *plan, uint16_t tidx,
                                 const tigris_mem_t *mem)
{
    if (!mem->tile.active)
        return tensor_numel(plan, tidx);
    const tigris_tensor_t *t = &plan->tensors[tidx];
    const int32_t *shape = tigris_tensor_shape(plan, t);
    /* NHWC: N * out_h * out_w * C */
    return (uint32_t)shape[0] * (uint32_t)mem->tile.out_h *
           (uint32_t)mem->tile.out_w * (uint32_t)shape[3];
}

/* Int8 Kernels */

static int kern_conv2d_s8(
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

    /* Input quant param for zero point */
    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;

    /* Output quant param for requantization */
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;

    /* Output qp holds the pre-computed per-channel multiplier/shift for
     * requantization (effective_scale = input_scale * weight_scale / output_scale). */

    /* NHWC layout */
    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int IC = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];
    int OC = y_shape[3];
    int KH = op->spatial.kernel_h;
    int KW = op->spatial.kernel_w;
    int SH = op->spatial.stride_h;
    int SW = op->spatial.stride_w;
    int PT = op->spatial.pad_top;
    int PL = op->spatial.pad_left;
    int DH = op->spatial.dilation_h ? op->spatial.dilation_h : 1;
    int DW = op->spatial.dilation_w ? op->spatial.dilation_w : 1;

    /* Tile override */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    /* Weight layout: [OC, KH, KW, IC] - OHWI */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int oc = 0; oc < OC; oc++) {
                    int32_t acc = B ? B[oc] : 0;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PT + kh * DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PL + kw * DW;
                            if (iw < 0 || iw >= IW) continue;
                            for (int ic = 0; ic < IC; ic++) {
                                int32_t x_val = (int32_t)X[((n*IH+ih)*IW+iw)*IC+ic] - input_zp;
                                int32_t w_val = (int32_t)W[((oc*KH+kh)*KW+kw)*IC+ic];
                                acc += x_val * w_val;
                            }
                        }
                    }
                    /* Requantize: per-channel uses oc, per-tensor uses 0 */
                    int ch = (out_qp && out_qp->num_channels > 1) ? oc : 0;
                    int32_t m = out_qp ? get_multiplier(plan, out_qp, ch) : 1;
                    int32_t s = out_qp ? get_shift(plan, out_qp, ch) : 0;
                    int32_t scaled = multiply_by_quantized_multiplier(acc, m, s);
                    Y[((n*OH+oh)*OW+ow)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

static int kern_depthwise_conv2d_s8(
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

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];
    int KH = op->spatial.kernel_h;
    int KW = op->spatial.kernel_w;
    int SH = op->spatial.stride_h;
    int SW = op->spatial.stride_w;
    int PT = op->spatial.pad_top;
    int PL = op->spatial.pad_left;
    int DH = op->spatial.dilation_h ? op->spatial.dilation_h : 1;
    int DW = op->spatial.dilation_w ? op->spatial.dilation_w : 1;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    /* Weight layout: [KH, KW, C] (HWC) */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int c = 0; c < C; c++) {
                    int32_t acc = B ? B[c] : 0;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PT + kh * DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PL + kw * DW;
                            if (iw < 0 || iw >= IW) continue;
                            int32_t x_val = (int32_t)X[((n*IH+ih)*IW+iw)*C+c] - input_zp;
                            int32_t w_val = (int32_t)W[(kh*KW+kw)*C+c];
                            acc += x_val * w_val;
                        }
                    }
                    int ch = (out_qp && out_qp->num_channels > 1) ? c : 0;
                    int32_t m = out_qp ? get_multiplier(plan, out_qp, ch) : 1;
                    int32_t s = out_qp ? get_shift(plan, out_qp, ch) : 0;
                    int32_t scaled = multiply_by_quantized_multiplier(acc, m, s);
                    Y[((n*OH+oh)*OW+ow)*C+c] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

static int kern_fully_connected_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;

    /* Input: [N, IC], Weight: [OC, IC], Output: [N, OC] */
    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];
    uint32_t x_numel = tensor_numel(plan, ins[0]);
    int N, OC;
    if (y_tensor->ndim >= 2) {
        N  = y_shape[0];
        OC = y_shape[1];
    } else {
        N  = 1;
        OC = y_shape[0];
    }
    int IC = (int)(x_numel / (uint32_t)N);

    /* Compute: [N, OC] */
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            int32_t acc = B ? B[oc] : 0;
            for (int ic = 0; ic < IC; ic++) {
                int32_t x_val = (int32_t)X[n * IC + ic] - input_zp;
                int32_t w_val = (int32_t)W[oc * IC + ic];
                acc += x_val * w_val;
            }
            int ch = (out_qp && out_qp->num_channels > 1) ? oc : 0;
            int32_t m = out_qp ? get_multiplier(plan, out_qp, ch) : 1;
            int32_t s = out_qp ? get_shift(plan, out_qp, ch) : 0;
            int32_t scaled = multiply_by_quantized_multiplier(acc, m, s);
            Y[n * OC + oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
        }
    }
    return 0;
}

static int kern_relu_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* Quantized relu: max(x, zero_point) */
    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    int8_t zp = in_qp ? (int8_t)in_qp->zero_point : 0;

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = X[i] > zp ? X[i] : zp;
    }
    return 0;
}

static int kern_relu6_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* Quantized relu6: clamp to [zp_0, zp_6] */
    const tigris_quant_param_t *qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    float scale = qp ? qp->scale : 1.0f;
    int32_t zp = qp ? qp->zero_point : 0;
    int8_t lo = clamp_s8(zp);  /* quantized 0 */
    int8_t hi = clamp_s8((int32_t)(6.0f / scale) + zp);  /* quantized 6 */

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        int8_t v = X[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        Y[i] = v;
    }
    return 0;
}

static int kern_add_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *A = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    const int8_t *B_ptr = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[1]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const tigris_quant_param_t *qp_a = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *qp_b = tigris_tensor_quant(plan, &plan->tensors[ins[1]]);
    const tigris_quant_param_t *qp_y = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    float sa = qp_a ? qp_a->scale : 1.0f;
    int32_t za = qp_a ? qp_a->zero_point : 0;
    float sb = qp_b ? qp_b->scale : 1.0f;
    int32_t zb = qp_b ? qp_b->zero_point : 0;
    float sy = qp_y ? qp_y->scale : 1.0f;
    int32_t zy = qp_y ? qp_y->zero_point : 0;

    /* real_a = sa * (a - za), real_b = sb * (b - zb)
     * real_y = real_a + real_b
     * y = real_y / sy + zy = (sa/sy)*(a-za) + (sb/sy)*(b-zb) + zy
     */
    float ma = sa / sy;
    float mb = sb / sy;

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        float real_sum = ma * (float)((int32_t)A[i] - za)
                       + mb * (float)((int32_t)B_ptr[i] - zb);
        int32_t q = (int32_t)roundf(real_sum) + zy;
        Y[i] = clamp_s8(q);
    }
    return 0;
}

static int kern_global_avg_pool_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;

    /* NHWC: [N, H, W, C] -> [N, 1, 1, C] */
    int N = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    int C = x_shape[3];
    int HW = H * W;

    float rescale = in_scale / out_scale;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            int32_t sum = 0;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    sum += (int32_t)X[((n*H+h)*W+w)*C+c] - input_zp;
                }
            }
            float avg = (float)sum / (float)HW;
            int32_t q = (int32_t)(avg * rescale + 0.5f) + output_zp;
            Y[n * C + c] = clamp_s8(q);
        }
    }
    return 0;
}

static int kern_reshape_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    if (X != Y) {
        memcpy(Y, X, n);
    }
    return 0;
}

static int kern_max_pool_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];
    int KH = op->spatial.kernel_h;
    int KW = op->spatial.kernel_w;
    int SH = op->spatial.stride_h ? op->spatial.stride_h : 1;
    int SW = op->spatial.stride_w ? op->spatial.stride_w : 1;
    int PT = op->spatial.pad_top;
    int PL = op->spatial.pad_left;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    /* Max preserves scale/zp - no requantization needed */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int c = 0; c < C; c++) {
                    int8_t max_val = -128;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PT + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PL + kw;
                            if (iw < 0 || iw >= IW) continue;
                            int8_t v = X[((n * IH + ih) * IW + iw) * C + c];
                            if (v > max_val) max_val = v;
                        }
                    }
                    Y[((n * OH + oh) * OW + ow) * C + c] = max_val;
                }
            }
        }
    }
    return 0;
}

static int kern_concat_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    int8_t *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = y_shape[0];
    int H  = y_shape[1];
    int W  = y_shape[2];

    if (mem->tile.active) {
        H = mem->tile.out_h;
        W = mem->tile.out_w;
    }

    int out_c_offset = 0;
    int total_OC = y_shape[3];

    const tigris_quant_param_t *qp_y =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    float out_scale = qp_y ? qp_y->scale : 1.0f;
    int32_t out_zp  = qp_y ? qp_y->zero_point : 0;

    for (int i = 0; i < op->num_inputs; i++) {
        const int8_t *Xi = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[i]);
        const int32_t *xi_shape = tigris_tensor_shape(plan, &plan->tensors[ins[i]]);
        int Ci = xi_shape[3];

        const tigris_quant_param_t *qp_i =
            tigris_tensor_quant(plan, &plan->tensors[ins[i]]);
        float in_scale = qp_i ? qp_i->scale : 1.0f;
        int32_t in_zp  = qp_i ? qp_i->zero_point : 0;

        /* Fast path: identical quant params -> memcpy */
        int same_qp = (fabsf(in_scale - out_scale) < 1e-7f && in_zp == out_zp);

        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    const int8_t *src = Xi + ((n * H + h) * W + w) * Ci;
                    int8_t *dst = Y + ((n * H + h) * W + w) * total_OC + out_c_offset;
                    if (same_qp) {
                        memcpy(dst, src, (size_t)Ci);
                    } else {
                        float M = in_scale / out_scale;
                        for (int c = 0; c < Ci; c++) {
                            float real = M * (float)((int32_t)src[c] - in_zp);
                            dst[c] = clamp_s8((int32_t)roundf(real) + out_zp);
                        }
                    }
                }
            }
        }
        out_c_offset += Ci;
    }
    return 0;
}

static int kern_resize_nearest_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];

    int scale_h = op->spatial.stride_h ? op->spatial.stride_h : 1;
    int scale_w = op->spatial.stride_w ? op->spatial.stride_w : 1;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
    }

    /* Nearest-neighbor: just copy bytes, no arithmetic */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih = oh / scale_h;
            if (ih >= IH) ih = IH - 1;
            for (int ow = 0; ow < OW; ow++) {
                int iw = ow / scale_w;
                if (iw >= IW) iw = IW - 1;
                const int8_t *src = X + ((n * IH + ih) * IW + iw) * C;
                int8_t *dst = Y + ((n * OH + oh) * OW + ow) * C;
                memcpy(dst, src, (size_t)C);
            }
        }
    }
    return 0;
}

static int kern_sigmoid_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    float in_scale = in_qp ? in_qp->scale : 1.0f;
    int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t out_zp = out_qp ? out_qp->zero_point : 0;

    /* Build 256-entry LUT indexed by (uint8_t)int8_value.
     * The cast (int8_t)i gives the two's complement mapping:
     *   i=0->0, i=127->127, i=128->-128, i=255->-1 */
    int8_t lut[256];
    for (int i = 0; i < 256; i++) {
        int8_t x_val = (int8_t)i;
        float real = in_scale * (float)((int32_t)x_val - in_zp);
        float sig = 1.0f / (1.0f + expf(-real));
        int32_t q = (int32_t)roundf(sig / out_scale) + out_zp;
        lut[i] = clamp_s8(q);
    }

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = lut[(uint8_t)X[i]];
    }
    return 0;
}

static int kern_mul_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *A = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    const int8_t *B_ptr = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[1]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const tigris_quant_param_t *qp_a = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *qp_b = tigris_tensor_quant(plan, &plan->tensors[ins[1]]);
    const tigris_quant_param_t *qp_y = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    float sa = qp_a ? qp_a->scale : 1.0f;
    int32_t za = qp_a ? qp_a->zero_point : 0;
    float sb = qp_b ? qp_b->scale : 1.0f;
    int32_t zb = qp_b ? qp_b->zero_point : 0;
    float sy = qp_y ? qp_y->scale : 1.0f;
    int32_t zy = qp_y ? qp_y->zero_point : 0;

    float combined_scale = (sa * sb) / sy;

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        float real_prod = (float)((int32_t)A[i] - za) *
                          (float)((int32_t)B_ptr[i] - zb) * combined_scale;
        int32_t q = (int32_t)roundf(real_prod) + zy;
        Y[i] = clamp_s8(q);
    }
    return 0;
}

/* Dispatch */

int tigris_dispatch_kernel_s8(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    (void)op_index;
    (void)user_ctx;

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return kern_conv2d_s8(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return kern_depthwise_conv2d_s8(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return kern_fully_connected_s8(plan, op, mem);
    case TIGRIS_OP_RELU:        return kern_relu_s8(plan, op, mem);
    case TIGRIS_OP_RELU6:       return kern_relu6_s8(plan, op, mem);
    case TIGRIS_OP_ADD:         return kern_add_s8(plan, op, mem);
    case TIGRIS_OP_GLOBAL_AVG:  return kern_global_avg_pool_s8(plan, op, mem);
    case TIGRIS_OP_RESHAPE:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_FLATTEN:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_MAX_POOL:    return kern_max_pool_s8(plan, op, mem);
    case TIGRIS_OP_CONCAT:      return kern_concat_s8(plan, op, mem);
    case TIGRIS_OP_RESIZE:      return kern_resize_nearest_s8(plan, op, mem);
    case TIGRIS_OP_SIGMOID:     return kern_sigmoid_s8(plan, op, mem);
    case TIGRIS_OP_MUL:         return kern_mul_s8(plan, op, mem);
    default:
        return -1;  /* unsupported op type */
    }
}

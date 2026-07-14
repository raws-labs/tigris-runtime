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
#include "tigris_accel_policy.h"

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

/** Load a possibly unaligned int32 bias value from the packed weight blob. */
static inline int32_t load_bias_s32(const uint8_t *bias, uint32_t index)
{
    int32_t value;
    memcpy(&value, bias + (size_t)index * sizeof(value), sizeof(value));
    return value;
}

/* gemmlowp SaturatingRoundingDoublingHighMul - the high 32 bits of 2*a*b with
 * rounding (TFLite/CMSIS-NN/ESP-NN convention). */
static inline int32_t sat_round_dbl_high_mul(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / ((int64_t)1 << 31));
}

/* gemmlowp RoundingDivideByPOT - round-to-nearest divide by 2^exponent, with the
 * sign-correct tie threshold. */
static inline int32_t round_div_by_pot(int32_t x, int exponent)
{
    if (exponent <= 0) return x;
    int32_t mask = ((int32_t)1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

/**
 * Multiply-by-quantized-multiplier - bit-exact with TFLite/CMSIS-NN gemmlowp.
 * shift > 0 left-shifts the input; shift <= 0 right-shifts the result.
 */
static inline int32_t multiply_by_quantized_multiplier(
    int32_t x, int32_t multiplier, int32_t shift)
{
    int left  = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    return round_div_by_pot(
        sat_round_dbl_high_mul(x * ((int32_t)1 << left), multiplier), right);
}

/* TFLite QuantizeMultiplier: frexp(scale) -> (Q0.31 multiplier, shift=exp).
 * Used at runtime where the effective multiplier is not precomputed in the plan
 * (the quantized Mean folds 1/HW into it). */
static inline void compute_quant_mult(double scale, int32_t *mult, int *shift)
{
    if (scale <= 0.0) { *mult = 0; *shift = 0; return; }
    int e;
    double q = frexp(scale, &e);
    int64_t qf = (int64_t)(q * 2147483648.0 + 0.5);  /* round(q * 2^31), q>0 */
    if (qf == ((int64_t)1 << 31)) { qf /= 2; e += 1; }
    *mult = (int32_t)qf;
    *shift = e;
}

/** Fold a positive sample count into a quantized scale for a mean. */
static inline void compute_mean_quant_mult(
    double scale, int32_t count, int32_t *mult, int *shift)
{
    compute_quant_mult(scale, mult, shift);
    if (count > 1) {
        int s = 63 - __builtin_clzll((unsigned long long)count);
        if (s > 32) s = 32;
        int max_s = 31 + *shift;
        if (max_s < 0) max_s = 0;
        if (s > max_s) s = max_s;
        *mult = (int32_t)(((int64_t)*mult << s) / count);
        *shift -= s;
    }
}

/** Vendor/TFLite AveragePool rounding: nearest, with exact ties away from 0. */
static inline int32_t round_divide_away_from_zero(int32_t value, int32_t divisor)
{
    int32_t half = divisor / 2;
    return value > 0 ? (value + half) / divisor
                     : (value - half) / divisor;
}

/** Get multiplier from quant data (works for both per-tensor and per-channel). */
static inline int32_t get_multiplier(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off + ch];
}

/** Get shift from quant data (works for both per-tensor and per-channel). */
static inline int32_t get_shift(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off + ch];
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

/** Validate exact-shape, two-dynamic-input int8 elementwise operands. */
static int binary_s8_io_is_valid(
    const tigris_plan_t *plan, const tigris_op_t *op, const tigris_mem_t *mem)
{
    if (!plan || !op || !mem || !plan->header || !plan->tensors ||
        !plan->index_pool || !plan->shape_pool || !mem->tensor_ptrs ||
        op->num_inputs != 2 || op->num_outputs != 1 ||
        op->weight_idx != TIGRIS_NO_WEIGHT ||
        op->bias_idx != TIGRIS_NO_WEIGHT)
        return 0;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    uint16_t indices[3] = {ins[0], ins[1], outs[0]};
    for (uint8_t i = 0; i < 3; i++) {
        if (indices[i] >= plan->header->num_tensors ||
            indices[i] >= mem->num_tensors ||
            !tigris_mem_tensor_ptr(mem, indices[i]))
            return 0;
    }

    const tigris_tensor_t *a = &plan->tensors[indices[0]];
    const tigris_tensor_t *b = &plan->tensors[indices[1]];
    const tigris_tensor_t *y = &plan->tensors[indices[2]];
    if (a->dtype != 3 || b->dtype != 3 || y->dtype != 3 ||
        a->ndim != b->ndim || a->ndim != y->ndim ||
        a->size_bytes != b->size_bytes || a->size_bytes != y->size_bytes ||
        (mem->tile.active && a->ndim != 4))
        return 0;

    uint64_t numel = 1;
    const int32_t *a_shape = tigris_tensor_shape(plan, a);
    const int32_t *b_shape = tigris_tensor_shape(plan, b);
    const int32_t *y_shape = tigris_tensor_shape(plan, y);
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a_shape[i] <= 0 || a_shape[i] != b_shape[i] ||
            a_shape[i] != y_shape[i])
            return 0;
        numel *= (uint32_t)a_shape[i];
        if (numel > UINT32_MAX)
            return 0;
    }
    return (uint32_t)numel == a->size_bytes;
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
    const uint8_t *B = (const uint8_t *)tigris_op_bias(plan, op);

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
                    int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
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

/* INT8 1D convolution. NLC activation layout, weights [OC, K, IC], per-channel
 * requant - the int8 analogue of kern_conv1d (float) / kern_conv2d_s8. Conv1D
 * tensors are 3D so they are never tiled (the tile path requires 4D I/O), hence
 * no mem->tile handling, matching the float kernel. */
static int kern_conv1d_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const uint8_t *B = (const uint8_t *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;

    /* NLC layout (transposed from NCL) */
    int N  = x_shape[0];
    int IT = x_shape[1];   /* input length (time) */
    int IC = x_shape[2];
    int OT = y_shape[1];   /* output length (time) */
    int OC = y_shape[2];
    int K  = op->spatial.kernel_h;                              /* kernel size in kernel_h */
    int S  = op->spatial.stride_h   ? op->spatial.stride_h   : 1;
    int D  = op->spatial.dilation_h ? op->spatial.dilation_h : 1;
    int PB = op->spatial.pad_top;                              /* pad_begin in pad_top */

    for (int n = 0; n < N; n++) {
        for (int ot = 0; ot < OT; ot++) {
            for (int oc = 0; oc < OC; oc++) {
                int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
                for (int k = 0; k < K; k++) {
                    int it = ot * S - PB + k * D;
                    if (it < 0 || it >= IT) continue;
                    for (int ic = 0; ic < IC; ic++) {
                        int32_t x_val = (int32_t)X[(n*IT+it)*IC+ic] - input_zp;
                        int32_t w_val = (int32_t)W[(oc*K+k)*IC+ic];
                        acc += x_val * w_val;
                    }
                }
                int ch = (out_qp && out_qp->num_channels > 1) ? oc : 0;
                int32_t m = out_qp ? get_multiplier(plan, out_qp, ch) : 1;
                int32_t s = out_qp ? get_shift(plan, out_qp, ch) : 0;
                int32_t scaled = multiply_by_quantized_multiplier(acc, m, s);
                Y[(n*OT+ot)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
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
    const uint8_t *B = (const uint8_t *)tigris_op_bias(plan, op);

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
                    int32_t acc = B ? load_bias_s32(B, (uint32_t)c) : 0;
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
    const uint8_t *B = (const uint8_t *)tigris_op_bias(plan, op);

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
            int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
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
    /* Round (not truncate) the quantized 6.0 ceiling to match TFLite's activation
     * max and the compiler's fused-Relu6 bound; truncation is 1 LSB too low when
     * 6/scale has a fractional part >= 0.5. */
    int8_t hi = clamp_s8((int32_t)lroundf(6.0f / scale) + zp);  /* quantized 6 */

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
    /* Constants are absent from the tensor/quant tables in the current wire
     * format. Without their scale and zero point, one-input + weight_idx Add
     * cannot be interpreted safely, so fail closed before reading ins[1]. */
    if (!binary_s8_io_is_valid(plan, op, mem))
        return -1;

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

    /* TFLite-exact quantized Add (reference/integer_ops/add.h): shift both inputs
     * left by 20, scale each by an integer multiplier derived from its scale
     * relative to twice-the-max-input-scale, sum, then requantize to the output
     * scale. All integer gemmlowp - bit-exact with tflite/CMSIS-NN. The old float
     * (sa/sy)*(a-za)+... drifted ~1 LSB/op, accumulating to tens of LSB across a
     * residual network like MobileNetV2. */
    const int LEFT_SHIFT = 20;
    double twice_max = 2.0 * ((sa > sb) ? (double)sa : (double)sb);
    int32_t a_mult, b_mult, y_mult;
    int a_shift, b_shift, y_shift;
    compute_quant_mult((double)sa / twice_max, &a_mult, &a_shift);
    compute_quant_mult((double)sb / twice_max, &b_mult, &b_shift);
    compute_quant_mult(twice_max / (((double)(1 << LEFT_SHIFT)) * (double)sy),
                       &y_mult, &y_shift);

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        int32_t av = ((int32_t)A[i] - za) * (1 << LEFT_SHIFT);
        int32_t bv = ((int32_t)B_ptr[i] - zb) * (1 << LEFT_SHIFT);
        int32_t scaled = multiply_by_quantized_multiplier(av, a_mult, a_shift)
                       + multiply_by_quantized_multiplier(bv, b_mult, b_shift);
        int32_t q = multiply_by_quantized_multiplier(scaled, y_mult, y_shift) + zy;
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

    /* Guard a missing/zero output scale (unquantized output tensor): degrade to
     * preserving the input quant instead of dividing by zero. */
    if (!(out_scale > 0.0f)) {
        out_scale = in_scale;
        output_zp = input_zp;
    }

    /* TFLite QuantizedMeanOrSum (reference/reduce.h), bit-exact:
     *   - requant multiplier/shift from (in_scale / out_scale)
     *   - fold the 1/HW mean into the multiplier
     *   - per channel: MBQM(Σx - in_zp*HW, mult, shift) + out_zp */
    int32_t mult;
    int shift0;
    compute_mean_quant_mult(
        (double)in_scale / (double)out_scale, HW, &mult, &shift0);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            /* Walk the channel with a stride-C pointer - no per-element index
             * arithmetic. temp_sum = Σ raw x (TFLite subtracts the zp after). */
            const int8_t *xp = X + (size_t)(n * HW) * C + c;
            int32_t sum = 0;
            for (int p = 0; p < HW; p++) {
                sum += *xp;
                xp += C;
            }
            int32_t shifted = sum - input_zp * HW;
            int32_t q = multiply_by_quantized_multiplier(shifted, mult, shift0) + output_zp;
            Y[n * C + c] = clamp_s8(q);
        }
    }
    return 0;
}

static int kern_avg_pool_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape =
        tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape =
        tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp =
        tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    if (!(in_scale > 0.0f))
        in_scale = 1.0f;
    if (!(out_scale > 0.0f)) {
        out_scale = in_scale;
        output_zp = input_zp;
    }
    int same_quant = in_scale == out_scale && input_zp == output_zp;

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
    if (KH <= 0 || KW <= 0)
        return -1;

    /* ONNX AveragePool defaults count_include_pad=0, the only mode representable
     * by the current schema. Both ESP-NN and CMSIS-NN use the same valid-sample
     * divisor. */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih_base = oh * SH - PT;
            int kh_start = ih_base < 0 ? -ih_base : 0;
            int kh_end = KH;
            if (ih_base + kh_end > IH)
                kh_end = IH - ih_base;

            for (int ow = 0; ow < OW; ow++) {
                int iw_base = ow * SW - PL;
                int kw_start = iw_base < 0 ? -iw_base : 0;
                int kw_end = KW;
                if (iw_base + kw_end > IW)
                    kw_end = IW - iw_base;

                int32_t count = (kh_end - kh_start) * (kw_end - kw_start);
                if (count <= 0)
                    return -1;

                for (int c = 0; c < C; c++) {
                    int32_t sum = 0;
                    for (int kh = kh_start; kh < kh_end; kh++) {
                        int ih = ih_base + kh;
                        for (int kw = kw_start; kw < kw_end; kw++) {
                            int iw = iw_base + kw;
                            sum += X[((n * IH + ih) * IW + iw) * C + c];
                        }
                    }

                    int32_t q;
                    if (same_quant) {
                        q = round_divide_away_from_zero(sum, count);
                    } else {
                        /* TFLite rejects mismatched input/output quantization
                         * for int8 AveragePool, so there is no canonical
                         * integer-kernel result for this extension. Define it
                         * as dequantize -> valid-sample mean -> requantize,
                         * using round() so half ties follow the away-from-zero
                         * convention of the native ESP/CMSIS pool kernels. */
                        int32_t centered = sum - input_zp * count;
                        double scaled =
                            ((double)centered * (double)in_scale) /
                            ((double)count * (double)out_scale);
                        double quantized = round(scaled) + (double)output_zp;
                        if (quantized < (double)op->act_min)
                            q = op->act_min;
                        else if (quantized > (double)op->act_max)
                            q = op->act_max;
                        else
                            q = (int32_t)quantized;
                    }
                    Y[((n * OH + oh) * OW + ow) * C + c] =
                        clamp_act(q, op->act_min, op->act_max);
                }
            }
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
                    /* Apply the fused activation clamp, as TFLM/CMSIS MaxPool do
                     * (no-op when act range is the full [-128, 127]). */
                    Y[((n * OH + oh) * OW + ow) * C + c] =
                        clamp_act(max_val, op->act_min, op->act_max);
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

/* Tanh via a 256-entry LUT, mirroring kern_sigmoid_s8 (tanh in place of the
 * logistic). int8 input -> int8 output with its own output quant. */
static int kern_tanh_s8(
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

    int8_t lut[256];
    for (int i = 0; i < 256; i++) {
        int8_t x_val = (int8_t)i;
        float real = in_scale * (float)((int32_t)x_val - in_zp);
        float th = tanhf(real);
        int32_t q = (int32_t)roundf(th / out_scale) + out_zp;
        lut[i] = clamp_s8(q);
    }

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = lut[(uint8_t)X[i]];
    }
    return 0;
}

/* Softmax over the final tensor dimension.  The binary plan format does not
 * carry a Softmax axis attribute; the compiler's supported quantized graphs
 * use ONNX's default/final-axis form.  Keep this scalar reference path for
 * terminal classifiers (and as the accelerator fallback), where correctness
 * matters more than throughput. */
static int kern_softmax_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    if (!plan || !op || !mem || op->num_inputs != 1 || op->num_outputs != 1 ||
        mem->tile.active)
        return -1;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    if (ins[0] >= plan->header->num_tensors || outs[0] >= plan->header->num_tensors ||
        ins[0] >= mem->num_tensors || outs[0] >= mem->num_tensors)
        return -1;

    const tigris_tensor_t *x_tensor = &plan->tensors[ins[0]];
    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];
    if (x_tensor->dtype != 3 || y_tensor->dtype != 3 || x_tensor->ndim == 0 ||
        x_tensor->ndim != y_tensor->ndim || x_tensor->size_bytes != y_tensor->size_bytes)
        return -1;

    const int32_t *x_shape = tigris_tensor_shape(plan, x_tensor);
    const int32_t *y_shape = tigris_tensor_shape(plan, y_tensor);
    uint64_t count = 1;
    for (uint8_t i = 0; i < x_tensor->ndim; i++) {
        if (x_shape[i] <= 0 || x_shape[i] != y_shape[i])
            return -1;
        count *= (uint32_t)x_shape[i];
        if (count > UINT32_MAX)
            return -1;
    }
    const uint32_t classes = (uint32_t)x_shape[x_tensor->ndim - 1];
    if (classes == 0 || count != x_tensor->size_bytes || count % classes != 0)
        return -1;

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, x_tensor);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, y_tensor);
    const float in_scale = in_qp ? in_qp->scale : 1.0f;
    const int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    const float out_scale = out_qp ? out_qp->scale : 1.0f;
    const int32_t out_zp = out_qp ? out_qp->zero_point : 0;
    if (in_scale <= 0.0f || out_scale <= 0.0f)
        return -1;

    for (uint32_t row = 0; row < (uint32_t)count / classes; row++) {
        const int8_t *x = X + (size_t)row * classes;
        int8_t *y = Y + (size_t)row * classes;
        int32_t max_q = x[0];
        for (uint32_t c = 1; c < classes; c++)
            if (x[c] > max_q) max_q = x[c];

        float sum = 0.0f;
        for (uint32_t c = 0; c < classes; c++)
            sum += expf(in_scale * (float)((int32_t)x[c] - max_q));
        if (sum == 0.0f)
            return -1;

        for (uint32_t c = 0; c < classes; c++) {
            float probability = expf(in_scale * (float)((int32_t)x[c] - max_q)) / sum;
            y[c] = clamp_s8((int32_t)roundf(probability / out_scale) + out_zp);
        }
    }
    (void)in_zp; /* Translation invariance cancels the input zero point. */
    return 0;
}

static int kern_mul_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    /* See kern_add_s8: constant quantization is not recoverable from weight_idx. */
    if (!binary_s8_io_is_valid(plan, op, mem))
        return -1;

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
    case TIGRIS_OP_AVG_POOL:     return kern_avg_pool_s8(plan, op, mem);
    case TIGRIS_OP_RESHAPE:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_FLATTEN:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_MAX_POOL:    return kern_max_pool_s8(plan, op, mem);
    case TIGRIS_OP_CONCAT:      return kern_concat_s8(plan, op, mem);
    case TIGRIS_OP_RESIZE:      return kern_resize_nearest_s8(plan, op, mem);
    case TIGRIS_OP_SIGMOID:     return kern_sigmoid_s8(plan, op, mem);
    case TIGRIS_OP_TANH:        return kern_tanh_s8(plan, op, mem);
    case TIGRIS_OP_SOFTMAX:     return kern_softmax_s8(plan, op, mem);
    case TIGRIS_OP_MUL:         return kern_mul_s8(plan, op, mem);
    case TIGRIS_OP_CONV1D:      return kern_conv1d_s8(plan, op, mem);
    default:
        return -1;  /* unsupported op type */
    }
}

/* Optional accelerator pre-routing. This lives with s8_ref so integrations
 * that enumerate the historical runtime sources explicitly do not need a new
 * translation unit merely to use the fallback policy. */

static uint16_t effective_dilation(uint16_t dilation)
{
    return dilation ? dilation : 1u;
}

static int is_conv_or_depthwise(const tigris_op_t *op)
{
    return op && (op->op_type == TIGRIS_OP_CONV ||
                  op->op_type == TIGRIS_OP_DEPTHWISE);
}

static int cmsis_tiled_spatial_uses_reference(const tigris_op_t *op)
{
    if (!op)
        return 0;

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:
    case TIGRIS_OP_DEPTHWISE:
    case TIGRIS_OP_AVG_POOL:
    case TIGRIS_OP_GLOBAL_AVG:
    case TIGRIS_OP_MAX_POOL:
        return 1;
    default:
        return 0;
    }
}

static int pool_quantization_is_compatible(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    if (!op || (op->op_type != TIGRIS_OP_AVG_POOL &&
                op->op_type != TIGRIS_OP_GLOBAL_AVG))
        return 1;
    if (!plan)
        return 0;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_quant_param_t *in_qp =
        tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    int32_t out_zp = out_qp ? out_qp->zero_point : 0;

    return in_scale > 0.0f && out_scale > 0.0f &&
           in_scale == out_scale && in_zp == out_zp;
}

int tigris_accel_esp_pad_workspace_fits(
    int32_t input_h,
    int32_t input_w,
    int32_t input_c,
    uint16_t pad_top,
    uint16_t pad_bottom,
    uint16_t pad_left,
    uint16_t pad_right,
    uint32_t workspace_bytes,
    uint32_t *required_bytes)
{
    if (required_bytes)
        *required_bytes = 0;
    if (input_h <= 0 || input_w <= 0 || input_c <= 0)
        return 0;

    uint64_t padded_h = (uint64_t)(uint32_t)input_h +
                        pad_top + pad_bottom;
    uint64_t padded_w = (uint64_t)(uint32_t)input_w +
                        pad_left + pad_right;
    if (padded_h > INT32_MAX || padded_w > INT32_MAX ||
        padded_h > UINT32_MAX / padded_w)
        return 0;
    uint64_t area = padded_h * padded_w;
    if (area > UINT32_MAX / (uint32_t)input_c)
        return 0;
    uint64_t needed = area * (uint32_t)input_c;

    if (required_bytes)
        *required_bytes = (uint32_t)needed;
    return needed <= workspace_bytes;
}

tigris_accel_route_t tigris_accel_pre_route(
    tigris_accel_backend_t backend,
    const tigris_plan_t   *plan,
    const tigris_op_t     *op,
    int                    tile_active)
{
    if (backend == TIGRIS_ACCEL_CMSIS_NN && tile_active &&
        cmsis_tiled_spatial_uses_reference(op))
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    if (backend == TIGRIS_ACCEL_ESP_NN && tile_active && op &&
        op->op_type == TIGRIS_OP_AVG_POOL)
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    if ((backend == TIGRIS_ACCEL_ESP_NN ||
         backend == TIGRIS_ACCEL_CMSIS_NN) &&
        !pool_quantization_is_compatible(plan, op))
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    if (backend == TIGRIS_ACCEL_ESP_NN && op &&
        op->op_type == TIGRIS_OP_GLOBAL_AVG)
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    if (backend == TIGRIS_ACCEL_ESP_NN && is_conv_or_depthwise(op) &&
        (effective_dilation(op->spatial.dilation_h) != 1u ||
         effective_dilation(op->spatial.dilation_w) != 1u))
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    return TIGRIS_ACCEL_ROUTE_ADAPTER;
}

int tigris_accel_try_s8_ref(
    tigris_accel_backend_t backend,
    const tigris_plan_t   *plan,
    const tigris_op_t     *op,
    uint16_t               op_index,
    tigris_mem_t          *mem,
    void                  *user_ctx,
    int                   *handled)
{
    if (!plan || !op || !mem || !handled)
        return -1;

    *handled = tigris_accel_pre_route(
                   backend, plan, op, mem->tile.active) ==
               TIGRIS_ACCEL_ROUTE_S8_REF;
    if (!*handled)
        return 0;

    return tigris_dispatch_kernel_s8(
        plan, op, op_index, mem, user_ctx);
}

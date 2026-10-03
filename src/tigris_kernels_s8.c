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
#include "tigris_kernels.h"
#include "tigris_accel_policy.h"
#include "tigris_kernel_window.h"
#include "tigris_binary_operands.h"

#include <math.h>
#include <string.h>

/* Each kernel stays out of line. Inlined, they share one register allocation
 * and one stack frame across the whole dispatcher, so editing any kernel
 * reallocates spill slots in every other one: a four-byte change to the Add
 * kernel moved the frame from 204 to 212 bytes and cost 6.6% on a DS-CNN that
 * contains no Add at all. The call happens once per operator, not per element. */
#if defined(__GNUC__) || defined(__clang__)
#  define TIGRIS_KERNEL_NOINLINE __attribute__((noinline))
#else
#  define TIGRIS_KERNEL_NOINLINE
#endif

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
    uint32_t mask = ((uint32_t)1u << exponent) - 1u;
    uint32_t remainder = (uint32_t)x & mask;
    uint32_t threshold = (mask >> 1) + (x < 0 ? 1u : 0u);
    int32_t quotient = x >= 0 ? (int32_t)((uint32_t)x >> exponent)
                             : -1 - (int32_t)(~(uint32_t)x >> exponent);
    return quotient + (remainder > threshold ? 1 : 0);
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
        int s = 0;
        for (uint32_t remaining = (uint32_t)count; remaining > 1u; remaining >>= 1)
            ++s;
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

/**
 * The int8 result of averaging `count` samples that sum to `sum`, with TFLite
 * AVERAGE_POOL_2D rounding. With equal input and output quantization this is
 * TFLite's integer division with ties away from zero. TFLite rejects differing
 * quantization for this op, so that case is defined as dequantize, mean and
 * requantize with round(), which keeps ties away from zero as the native
 * ESP/CMSIS pool kernels do.
 */
static inline int32_t average_pool_quantize(
    int32_t sum, int32_t count, int same_quant, int32_t input_zp,
    float in_scale, float out_scale, int32_t output_zp,
    int8_t act_min, int8_t act_max)
{
    int32_t q;
    if (same_quant) {
        q = round_divide_away_from_zero(sum, count);
    } else {
        int32_t centered = sum - input_zp * count;
        double scaled = ((double)centered * (double)in_scale) /
                        ((double)count * (double)out_scale);
        double quantized = round(scaled) + (double)output_zp;
        if (quantized < (double)act_min)
            q = act_min;
        else if (quantized > (double)act_max)
            q = act_max;
        else
            q = (int32_t)quantized;
    }
    return clamp_act(q, act_min, act_max);
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

/** Re-encode a value that a kernel carried through unchanged.
 *
 * Max preserves the value, but not its encoding: when the output tensor
 * declares a different scale or zero point from the input's, the same value
 * is a different int8. Defined the way kern_avg_pool_s8 defines the mismatched
 * case, dequantize then requantize with round() so half ties go away from
 * zero, since TFLite rejects mismatched pool quantization and there is no
 * canonical integer-kernel result to match.
 */
static inline int32_t requantize_passthrough(
    int32_t q, int32_t in_zp, float in_scale, int32_t out_zp, float out_scale)
{
    if (in_scale == out_scale && in_zp == out_zp)
        return q;
    double value = (double)(q - in_zp) * (double)in_scale;
    double scaled = round(value / (double)out_scale) + (double)out_zp;
    if (scaled < -128.0)
        return -128;
    if (scaled > 127.0)
        return 127;
    return (int32_t)scaled;
}

/** The input and output quantization of a value-preserving operator. */
typedef struct {
    int32_t in_zp;
    int32_t out_zp;
    float   in_scale;
    float   out_scale;
    int     same;
} passthrough_quant_t;

static inline passthrough_quant_t passthrough_quant(
    const tigris_plan_t *plan, uint16_t in_idx, uint16_t out_idx)
{
    const tigris_quant_param_t *in_qp =
        tigris_tensor_quant(plan, &plan->tensors[in_idx]);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[out_idx]);
    passthrough_quant_t q;
    q.in_zp = in_qp ? in_qp->zero_point : 0;
    q.in_scale = in_qp ? in_qp->scale : 1.0f;
    if (!(q.in_scale > 0.0f))
        q.in_scale = 1.0f;
    /* A tensor with no quantization parameters states nothing to re-encode
     * to, so the pair counts as matching and the value passes through. Only a
     * plan that declares both, and declares them different, asks for the
     * conversion. */
    q.out_zp = q.in_zp;
    q.out_scale = q.in_scale;
    q.same = 1;
    if (in_qp && out_qp && out_qp->scale > 0.0f) {
        q.out_zp = out_qp->zero_point;
        q.out_scale = out_qp->scale;
        q.same = (q.in_scale == q.out_scale) && (q.in_zp == q.out_zp);
    }
    return q;
}

/** The requantization parameters of an output tensor, resolved once.
 *
 * get_multiplier and get_shift chase plan->header and recompute the page
 * offset on every call, and the weighted kernels called them once per output
 * element for two numbers that vary only with the channel. The arrays sit
 * next to each other in one page, so one resolution serves the whole kernel.
 */
typedef struct {
    const int32_t *multiplier;  /* NULL when the tensor carries no params */
    const int32_t *shift;
    int            per_channel; /* index by the channel, otherwise by 0 */
} requant_params_t;

static inline requant_params_t requant_resolve(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp)
{
    requant_params_t rq;
    rq.multiplier = NULL;
    rq.shift = NULL;
    rq.per_channel = 0;
    if (qp) {
        uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2
                            ? 0u : qp->_pad;
        const int32_t *base = plan->quant_data +
                              (size_t)page * TIGRIS_QUANT_PAGE_ELEMS;
        rq.multiplier = base + qp->multiplier_off;
        rq.shift = base + qp->shift_off;
        rq.per_channel = qp->num_channels > 1 ? 1 : 0;
    }
    return rq;
}

/* The multiplier and shift a channel requantizes with. A tensor without quant
 * params keeps the (1, 0) the kernels used before, which is a Q0.31 multiply
 * rather than the identity, so the result is unchanged. */
static inline int32_t requant_apply(
    const requant_params_t *rq, int32_t acc, int channel)
{
    int32_t m = 1;
    int32_t sh = 0;
    if (rq->multiplier) {
        int idx = rq->per_channel ? channel : 0;
        m = rq->multiplier[idx];
        sh = rq->shift[idx];
    }
    return multiply_by_quantized_multiplier(acc, m, sh);
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
    /* A row band cuts the second to last axis, so out_h counts the rows in
     * the band, the trailing axis is the row, and everything ahead of the
     * pair is batch that the band spans rather than cuts. */
    if (mem->tile.row_tiled) {
        uint32_t batch = 1;
        for (uint8_t axis = 0; axis + 2u < t->ndim; axis++)
            batch *= (uint32_t)shape[axis];
        return batch * (uint32_t)mem->tile.out_h *
               (uint32_t)shape[t->ndim - 1];
    }
    /* Serialized activations are NHWC or NLC. */
    uint32_t width = t->ndim == 4 ? (uint32_t)mem->tile.out_w : 1u;
    return (uint32_t)shape[0] * (uint32_t)mem->tile.out_h *
           width * (uint32_t)shape[t->ndim - 1];
}

/** Per-row element count for pointwise row-offset pointer arithmetic:
 * the same width*channels term tile_aware_numel multiplies by out_h. */
static uint32_t tile_row_elems(const tigris_plan_t *plan, uint16_t tidx,
                               const tigris_mem_t *mem)
{
    const tigris_tensor_t *t = &plan->tensors[tidx];
    const int32_t *shape = tigris_tensor_shape(plan, t);
    uint32_t width = t->ndim == 4 ? (uint32_t)mem->tile.out_w : 1u;
    return width * (uint32_t)shape[t->ndim - 1];
}

/** Shift a pointwise (height-preserving) kernel's input/output base pointers
 * to the current sub-range: X by in_row_start rows, Y by out_row_start rows.
 * No-op when tiling is inactive (both offsets are then 0). */
static void apply_pointwise_row_offset_s8(
    const tigris_plan_t *plan, uint16_t tidx, const tigris_mem_t *mem,
    const int8_t **x, int8_t **y)
{
    if (!mem->tile.active)
        return;
    uint32_t row_elems = tile_row_elems(plan, tidx, mem);
    *x += (size_t)mem->tile.in_row_start * row_elems;
    *y += (size_t)mem->tile.out_row_start * row_elems;
}

/* The operands of an int8 binary operator and its output, advanced to the
 * tile's rows. An operand that repeats (one value, or one per channel) is
 * read whole; see tigris_binary_operands. */
static int binary_s8_begin(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, tigris_binary_operands_t *operands,
    int8_t **y, uint32_t *count)
{
    if (!tigris_binary_operands(plan, op, op_index, mem, 3u, operands))
        return 0;
    const uint16_t result = operands->output;
    *y = (int8_t *)tigris_mem_tensor_ptr(mem, result);
    *count = tile_aware_numel(plan, result, mem);
    if (mem->tile.active) {
        uint32_t row_elems = tile_row_elems(plan, result, mem);
        for (uint8_t k = 0; k < 2u; k++) {
            if (operands->period[k] == 0u)
                operands->data[k] += (size_t)mem->tile.in_row_start * row_elems;
        }
        *y += (size_t)mem->tile.out_row_start * row_elems;
    }
    return 1;
}

/* The element of operand k that output element i reads. */
static inline int32_t binary_s8_at(const tigris_binary_operands_t *operands, uint8_t k,
                                   uint32_t i)
{
    uint32_t index = operands->general[k] ? tigris_binary_general_index(operands, k, i)
                   : operands->period[k] ? i % operands->period[k] : i;
    return (int32_t)((const int8_t *)operands->data[k])[index];
}

/* Int8 Kernels */

static TIGRIS_KERNEL_NOINLINE int kern_conv2d_s8(
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
    requant_params_t rq = requant_resolve(plan, out_qp);

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
    int32_t out_row_start = 0;
    int32_t in_row_start = 0;

    /* Tile override */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
        /* Left pad only applies to a packed 2D (HW) tile; the right edge is
         * already implicit in the IW bound check below, the same way the
         * bottom edge is implicit in IH without a PB local. */
        if (mem->tile.width_tiled)
            PL = mem->tile.pad_left;
        out_row_start = mem->tile.out_row_start;
        in_row_start  = mem->tile.in_row_start;
    }

    /* Weight layout: [OC, KH, KW, IC] - OHWI */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int oh_g = oh + out_row_start;
            int base_h = oh_g * SH - PT - in_row_start;
            int kh0, kh1;
            tap_range(base_h, IH, KH, DH, &kh0, &kh1);
            for (int ow = 0; ow < OW; ow++) {
                int base_w = ow * SW - PL;
                int kw0, kw1;
                tap_range(base_w, IW, KW, DW, &kw0, &kw1);
                for (int oc = 0; oc < OC; oc++) {
                    int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh * DH;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw * DW;
                            for (int ic = 0; ic < IC; ic++) {
                                int32_t x_val = (int32_t)X[((n*IH+ih)*IW+iw)*IC+ic] - input_zp;
                                int32_t w_val = (int32_t)W[((oc*KH+kh)*KW+kw)*IC+ic];
                                acc += x_val * w_val;
                            }
                        }
                    }
                    int32_t scaled = requant_apply(&rq, acc, oc);
                    Y[((n*OH+oh_g)*OW+ow)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

/** ConvTranspose (transposed convolution) via input-gather, int8.
 * Same transpose-gather index relation as kern_conv_transpose (f32,
 * tigris_kernels.c): for each output pixel, gather from the input
 * positions that scattered into it under the forward strided-transpose
 * relation (oh = ih*SH - PT + kh, ow = iw*SW - PL + kw), inverted to solve
 * for ih/iw given oh/ow/kh/kw. A tap is skipped when it does not land on
 * the stride lattice (num_h/num_w not divisible by SH/SW) or when the
 * resulting ih/iw falls outside the input. group == 1 and dilation == 1
 * are the only supported configuration in scope.
 * Accumulation in int32 with per-channel requantization, matching
 * kern_conv2d_s8.
 */
static TIGRIS_KERNEL_NOINLINE int kern_conv_transpose_s8(
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
    requant_params_t rq = requant_resolve(plan, out_qp);

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

    /* Tile override. Structurally identical to kern_conv2d_s8: the tile's
     * global offset is folded into the effective pads PT/PL by the
     * executor, so the gather below runs unchanged in local packed
     * coordinates. Left pad only applies to a packed 2D (HW) tile, same as
     * kern_conv2d_s8. ConvTranspose never uses out_row_start / in_row_start
     * (2D-only, no line-buffered chain). */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
        if (mem->tile.width_tiled)
            PL = mem->tile.pad_left;
    }

    /* Weight layout: [OC, KH, KW, IC] (OHWI) */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int oc = 0; oc < OC; oc++) {
                    int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
                    for (int kh = 0; kh < KH; kh++) {
                        int num_h = oh + PT - kh;
                        if (num_h % SH != 0)
                            continue;
                        int ih = num_h / SH;
                        if (ih < 0 || ih >= IH)
                            continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int num_w = ow + PL - kw;
                            if (num_w % SW != 0)
                                continue;
                            int iw = num_w / SW;
                            if (iw < 0 || iw >= IW)
                                continue;
                            for (int ic = 0; ic < IC; ic++) {
                                int32_t x_val = (int32_t)X[((n*IH+ih)*IW+iw)*IC+ic] - input_zp;
                                int32_t w_val = (int32_t)W[((oc*KH+kh)*KW+kw)*IC+ic];
                                acc += x_val * w_val;
                            }
                        }
                    }
                    int32_t scaled = requant_apply(&rq, acc, oc);
                    Y[((n*OH+oh)*OW+ow)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

/* INT8 1D convolution. NLC activation layout, weights [OC, K, IC], per-channel
 * requant - the int8 analogue of kern_conv1d (float) / kern_conv2d_s8. */
static TIGRIS_KERNEL_NOINLINE int kern_conv1d_s8(
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
    requant_params_t rq = requant_resolve(plan, out_qp);

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

    if (mem->tile.active) {
        IT = mem->tile.in_h;
        OT = mem->tile.out_h;
        PB = mem->tile.pad_top;
    }

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
                int32_t scaled = requant_apply(&rq, acc, oc);
                Y[(n*OT+ot)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
            }
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_depthwise_conv2d_s8(
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
    requant_params_t rq = requant_resolve(plan, out_qp);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OC = y_shape[3];
    int M  = OC / C;
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
    int32_t out_row_start = 0;
    int32_t in_row_start = 0;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
        /* Left pad only applies to a packed 2D (HW) tile; the right edge is
         * already implicit in the IW bound check below, the same way the
         * bottom edge is implicit in IH without a PB local. */
        if (mem->tile.width_tiled)
            PL = mem->tile.pad_left;
        out_row_start = mem->tile.out_row_start;
        in_row_start  = mem->tile.in_row_start;
    }

    /* Output channel oc reads input channel oc / M, M the channel multiplier.
     * Weight layout: [KH, KW, OC] (HWC) */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int oh_g = oh + out_row_start;
            int base_h = oh_g * SH - PT - in_row_start;
            int kh0, kh1;
            tap_range(base_h, IH, KH, DH, &kh0, &kh1);
            for (int ow = 0; ow < OW; ow++) {
                int base_w = ow * SW - PL;
                int kw0, kw1;
                tap_range(base_w, IW, KW, DW, &kw0, &kw1);
                for (int oc = 0; oc < OC; oc++) {
                    int c = oc / M;
                    int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh * DH;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw * DW;
                            int32_t x_val = (int32_t)X[((n*IH+ih)*IW+iw)*C+c] - input_zp;
                            int32_t w_val = (int32_t)W[(kh*KW+kw)*OC+oc];
                            acc += x_val * w_val;
                        }
                    }
                    int32_t scaled = requant_apply(&rq, acc, oc);
                    Y[((n*OH+oh_g)*OW+ow)*OC+oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_fully_connected_s8(
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
    requant_params_t rq = requant_resolve(plan, out_qp);

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
    /* A row band computes its own rows against the whole weight. IC is taken
     * above from the full element count, so the narrowing happens after it. */
    if (mem->tile.active && mem->tile.row_tiled)
        N = (int)mem->tile.out_h;

    /* Compute: [N, OC] */
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            int32_t acc = B ? load_bias_s32(B, (uint32_t)oc) : 0;
            for (int ic = 0; ic < IC; ic++) {
                int32_t x_val = (int32_t)X[n * IC + ic] - input_zp;
                int32_t w_val = (int32_t)W[oc * IC + ic];
                acc += x_val * w_val;
            }
            int32_t scaled = requant_apply(&rq, acc, oc);
            Y[n * OC + oc] = clamp_act(scaled + output_zp, op->act_min, op->act_max);
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_relu_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* Quantized relu: max(x, zero_point). The clamp is in the input's domain,
     * so the result has to be re-encoded when the output declares a different
     * scale or zero point. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);
    int8_t zp = clamp_s8(q.in_zp);

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    if (q.same) {
        for (uint32_t i = 0; i < n; i++)
            Y[i] = X[i] > zp ? X[i] : zp;
    } else {
        for (uint32_t i = 0; i < n; i++) {
            int32_t v = X[i] > zp ? X[i] : zp;
            Y[i] = (int8_t)requantize_passthrough(
                v, q.in_zp, q.in_scale, q.out_zp, q.out_scale);
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_relu6_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* Quantized relu6: clamp to [zp_0, zp_6] in the input's domain, then
     * re-encode if the output declares a different scale or zero point. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);
    float scale = q.in_scale;
    int32_t zp = q.in_zp;
    int8_t lo = clamp_s8(zp);  /* quantized 0 */
    /* Round (not truncate) the quantized 6.0 ceiling to match TFLite's activation
     * max and the compiler's fused-Relu6 bound; truncation is 1 LSB too low when
     * 6/scale has a fractional part >= 0.5. */
    int8_t hi = clamp_s8((int32_t)lroundf(6.0f / scale) + zp);  /* quantized 6 */

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++) {
        int8_t v = X[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        Y[i] = q.same ? v : (int8_t)requantize_passthrough(
            v, q.in_zp, q.in_scale, q.out_zp, q.out_scale);
    }
    return 0;
}

/* Add and Sub differ only in the sign of the second scaled term; everything
 * around it, including the requantization, is identical. */
static TIGRIS_KERNEL_NOINLINE int kern_addsub_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem, int subtract)
{
    tigris_binary_operands_t operands;
    int8_t *Y;
    uint32_t n;
    if (!binary_s8_begin(plan, op, op_index, mem, &operands, &Y, &n))
        return -1;

    const tigris_quant_param_t *qp_a = operands.quant[0];
    const tigris_quant_param_t *qp_b = operands.quant[1];
    const tigris_quant_param_t *qp_y = tigris_tensor_quant(
        plan, &plan->tensors[tigris_op_outputs(plan, op)[0]]);

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
    int32_t a_mult, b_mult, y_mult;
    int a_shift, b_shift, y_shift;
    uint8_t requant_len = 0u;
    const uint8_t *requant = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_BINARY_REQUANT, &requant_len);
    if (requant != NULL &&
        requant_len == TIGRIS_OP_ATTR_BINARY_REQUANT_LEN) {
        /* The three pairs are compile-time constants of the operand scales,
         * so a plan that states them spares the kernel three frexp-and-round
         * sequences per call and gives a vendor kernel what it needs. */
        int32_t pairs[6];
        memcpy(pairs, requant, sizeof(pairs));
        a_mult = pairs[0]; a_shift = (int)pairs[1];
        b_mult = pairs[2]; b_shift = (int)pairs[3];
        y_mult = pairs[4]; y_shift = (int)pairs[5];
    } else {
        double twice_max = 2.0 * ((sa > sb) ? (double)sa : (double)sb);
        compute_quant_mult((double)sa / twice_max, &a_mult, &a_shift);
        compute_quant_mult((double)sb / twice_max, &b_mult, &b_shift);
        compute_quant_mult(
            twice_max / (((double)(1 << LEFT_SHIFT)) * (double)sy),
            &y_mult, &y_shift);
    }

    for (uint32_t i = 0; i < n; i++) {
        int32_t av = (binary_s8_at(&operands, 0u, i) - za) *
                     (1 << LEFT_SHIFT);
        int32_t bv = (binary_s8_at(&operands, 1u, i) - zb) *
                     (1 << LEFT_SHIFT);
        int32_t scaled_b = multiply_by_quantized_multiplier(bv, b_mult, b_shift);
        int32_t scaled = multiply_by_quantized_multiplier(av, a_mult, a_shift)
                       + (subtract ? -scaled_b : scaled_b);
        int32_t q = multiply_by_quantized_multiplier(scaled, y_mult, y_shift) + zy;
        /* A fused activation is a clamp on the requantized result: the compiler
         * derives the bounds from the output scale and zero point. Older plans
         * carry no bounds for this op, so fall back to the plain int8 range. */
        Y[i] = (op->fused_act == TIGRIS_ACT_NONE)
             ? clamp_s8(q)
             : clamp_act(q, op->act_min, op->act_max);
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_add_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    return kern_addsub_s8(plan, op, op_index, mem, 0);
}

static TIGRIS_KERNEL_NOINLINE int kern_sub_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    return kern_addsub_s8(plan, op, op_index, mem, 1);
}

static TIGRIS_KERNEL_NOINLINE int kern_global_avg_pool_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
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

    /* TIGRIS_OP_ATTR_POOL_ROUNDING selects TFLite AVERAGE_POOL_2D rounding for
     * a pool that was a whole-map AveragePool; without it the pool is a mean. */
    uint8_t rounding_len = 0u;
    int average = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_POOL_ROUNDING, &rounding_len) != NULL;
    int same_quant = in_scale == out_scale && input_zp == output_zp;

    /* TFLite QuantizedMeanOrSum (reference/reduce.h), bit-exact:
     *   - requant multiplier/shift from (in_scale / out_scale)
     *   - fold the 1/HW mean into the multiplier
     *   - per channel: MBQM(Σx - in_zp*HW, mult, shift) + out_zp */
    int32_t mult;
    int shift0;
    compute_mean_quant_mult(
        (double)in_scale / (double)out_scale, HW, &mult, &shift0);

    /* A band of rows while the executor walks the input, the whole tensor
     * otherwise. Only the raw int32 sum is split: the zero-point subtraction
     * and the requantization still run once over the full HW, so a banded
     * reduction yields the same byte as a single pass. */
    int32_t *acc = (int32_t *)mem->tile.reduce_acc;
    int rows = (mem->tile.active && mem->tile.in_h > 0) ? mem->tile.in_h : H;
    int band = rows * W;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            /* Walk the channel with a stride-C pointer - no per-element index
             * arithmetic. temp_sum = Σ raw x (TFLite subtracts the zp after). */
            const int8_t *xp = X + (size_t)(n * band) * C + c;
            int32_t sum = (acc && !mem->tile.reduce_first) ? acc[n * C + c] : 0;
            for (int p = 0; p < band; p++) {
                sum += *xp;
                xp += C;
            }
            if (acc) {
                acc[n * C + c] = sum;
                if (!mem->tile.reduce_last)
                    continue;
            }
            if (average) {
                Y[n * C + c] = (int8_t)average_pool_quantize(
                    sum, HW, same_quant, input_zp, in_scale, out_scale,
                    output_zp, op->act_min, op->act_max);
                continue;
            }
            int32_t shifted = sum - input_zp * HW;
            int32_t q = multiply_by_quantized_multiplier(shifted, mult, shift0) + output_zp;
            Y[n * C + c] = clamp_s8(q);
        }
    }
    return 0;
}

/**
 * Mean over one axis of a rank-3 tensor, the int8 twin of kern_reduce_mean.
 *
 * Follows TFLite QuantizedMeanOrSum: sum the raw bytes, subtract the zero
 * point once over the full count, and fold the reciprocal into the requant
 * multiplier so the division never leaves the integer domain.
 */
static TIGRIS_KERNEL_NOINLINE int kern_reduce_mean_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *in = &plan->tensors[ins[0]];
    uint8_t num_axes = 0u;
    const uint8_t *axes = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_AXES, &num_axes);
    if (axes == NULL || num_axes != 1u || in->ndim != 3u || axes[0] >= 3u)
        return -1;

    const int32_t *shape = tigris_tensor_shape(plan, in);
    int32_t outer = 1;
    int32_t inner = 1;
    for (uint8_t axis = 0; axis < 3u; axis++) {
        if (shape[axis] <= 0)
            return -1;
        if (axis < axes[0])
            outer *= shape[axis];
        else if (axis > axes[0])
            inner *= shape[axis];
    }
    int32_t reduced = shape[axes[0]];

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, in);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t input_zp = in_qp ? in_qp->zero_point : 0;
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t output_zp = out_qp ? out_qp->zero_point : 0;
    if (!(out_scale > 0.0f)) {
        out_scale = in_scale;
        output_zp = input_zp;
    }

    int32_t mult;
    int shift0;
    compute_mean_quant_mult(
        (double)in_scale / (double)out_scale, (int)reduced, &mult, &shift0);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    for (int32_t o = 0; o < outer; o++) {
        const int8_t *src = X + (size_t)o * (size_t)reduced * (size_t)inner;
        int8_t *dst = Y + (size_t)o * (size_t)inner;
        for (int32_t i = 0; i < inner; i++) {
            const int8_t *xp = src + i;
            int32_t sum = 0;
            for (int32_t r = 0; r < reduced; r++) {
                sum += *xp;
                xp += inner;
            }
            int32_t shifted = sum - input_zp * reduced;
            int32_t q = multiply_by_quantized_multiplier(shifted, mult, shift0) +
                        output_zp;
            dst[i] = clamp_s8(q);
        }
    }
    return 0;
}

/**
 * Split a tensor into contiguous parts along its outermost stored axis.
 *
 * The loader admits only the split whose parts are runs of the input, so each
 * output is that many bytes taken in order. Nothing is interleaved and no
 * shape arithmetic is needed; the parts differ only in how much they take.
 */
static TIGRIS_KERNEL_NOINLINE int kern_pad_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index, tigris_mem_t *mem)
{
    /* The fill is the operator's weight, already quantized to the output,
     * when the model states one; else the output's zero point, a real 0. */
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[tigris_op_outputs(plan, op)[0]]);
    int8_t zero = (int8_t)(out_qp ? out_qp->zero_point : 0);
    const uint8_t *fill = (const uint8_t *)&zero;
    if (op->weight_idx != TIGRIS_NO_WEIGHT) {
        if (plan->weight_entries[op->weight_idx].size_bytes != 1u)
            return -1;
        fill = (const uint8_t *)tigris_op_weight(plan, op);
    }
    return tigris_pad(plan, op, op_index, mem, 1u, fill);
}

static TIGRIS_KERNEL_NOINLINE int kern_split_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const uint8_t *source = (const uint8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    const tigris_tensor_t *input = &plan->tensors[ins[0]];
    const int32_t *in_shape = tigris_tensor_shape(plan, input);
    if (source == NULL || op->spatial.kernel_h >= input->ndim)
        return -1;
    /* Each part is one run per position ahead of the split axis. */
    uint32_t positions = 1;
    for (uint32_t axis = 0; axis < op->spatial.kernel_h; axis++)
        positions *= (uint32_t)in_shape[axis];
    uint32_t stride = input->size_bytes / positions;
    uint32_t offset = 0;
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        const tigris_tensor_t *part = &plan->tensors[outs[i]];
        uint8_t *destination = (uint8_t *)tigris_mem_tensor_ptr(mem, outs[i]);
        uint32_t run = part->size_bytes / positions;
        if (destination == NULL)
            return -1;
        for (uint32_t q = 0; q < positions; q++)
            memcpy(destination + (size_t)q * run, source + (size_t)q * stride + offset, run);
        offset += run;
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_avg_pool_s8(
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
    int32_t out_row_start = 0;
    int32_t in_row_start = 0;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
        /* Left pad only applies to a packed 2D (HW) tile; the right edge is
         * already implicit in the IW bound check below, the same way the
         * bottom edge is implicit in IH without a PB local. */
        if (mem->tile.width_tiled)
            PL = mem->tile.pad_left;
        out_row_start = mem->tile.out_row_start;
        in_row_start  = mem->tile.in_row_start;
    }
    if (KH <= 0 || KW <= 0)
        return -1;

    /* ONNX AveragePool defaults count_include_pad=0, the only mode representable
     * by the current schema. Both ESP-NN and CMSIS-NN use the same valid-sample
     * divisor. */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int oh_g = oh + out_row_start;
            int ih_base = oh_g * SH - PT - in_row_start;
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

                    Y[((n * OH + oh_g) * OW + ow) * C + c] = (int8_t)average_pool_quantize(
                        sum, count, same_quant, input_zp, in_scale, out_scale,
                        output_zp, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

/* Y = A * B with both operands activations. Neither is a weight entry, so
 * both zero points come from the tensor table and the requantization uses the
 * output's own multiplier, which the compiler derived from the product of the
 * two input scales. Axes are the model's own: the trailing pair is the matrix
 * and everything before it batches. */
static TIGRIS_KERNEL_NOINLINE int kern_matmul_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int8_t *A = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    const int8_t *B = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[1]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const tigris_tensor_t *a_tensor = &plan->tensors[ins[0]];
    const tigris_tensor_t *b_tensor = &plan->tensors[ins[1]];
    const int32_t *a_shape = tigris_tensor_shape(plan, a_tensor);
    const int32_t *b_shape = tigris_tensor_shape(plan, b_tensor);

    const tigris_quant_param_t *qp_a = tigris_tensor_quant(plan, a_tensor);
    const tigris_quant_param_t *qp_b = tigris_tensor_quant(plan, b_tensor);
    const tigris_quant_param_t *qp_y =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    int32_t a_zp = qp_a ? qp_a->zero_point : 0;
    int32_t b_zp = qp_b ? qp_b->zero_point : 0;
    int32_t y_zp = qp_y ? qp_y->zero_point : 0;
    int32_t mult = qp_y ? get_multiplier(plan, qp_y, 0) : 1;
    int32_t shift = qp_y ? get_shift(plan, qp_y, 0) : 0;

    uint8_t last = (uint8_t)(a_tensor->ndim - 1u);
    int M = a_shape[last - 1u];
    int K = a_shape[last];
    int N = b_shape[last];

    /* A row band states its row count in tile.out_h, not in the shape. See
     * kern_matmul. */
    if (mem->tile.active && mem->tile.row_tiled && mem->tile.out_h > 0)
        M = mem->tile.out_h;

    int batches = 1;
    for (uint8_t axis = 0; axis + 2u <= last; axis++)
        batches *= a_shape[axis];

    for (int b = 0; b < batches; b++) {
        const int8_t *a = A + (size_t)b * (size_t)M * (size_t)K;
        const int8_t *w = B + (size_t)b * (size_t)K * (size_t)N;
        int8_t *y = Y + (size_t)b * (size_t)M * (size_t)N;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++) {
                    int32_t av = (int32_t)a[m * K + k] - a_zp;
                    int32_t bv = (int32_t)w[k * N + n] - b_zp;
                    acc += av * bv;
                }
                int32_t scaled =
                    multiply_by_quantized_multiplier(acc, mult, shift);
                y[m * N + n] =
                    clamp_act(scaled + y_zp, op->act_min, op->act_max);
            }
        }
    }
    return 0;
}

/* GlobalMaxPool: the maximum over every spatial position, per channel. Like
 * MaxPool it needs no requantization, because the result is one of the input
 * values and so already sits in the input's scale and zero point. */
static TIGRIS_KERNEL_NOINLINE int kern_global_max_pool_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    int N = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    int C = x_shape[3];

    /* Max preserves the value, not its encoding: see kern_max_pool_s8. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            int32_t best = -128;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int32_t v = (int32_t)X[((n * H + h) * W + w) * C + c];
                    if (v > best) best = v;
                }
            }
            int32_t out_q = requantize_passthrough(
                best, q.in_zp, q.in_scale, q.out_zp, q.out_scale);
            Y[n * C + c] = clamp_act(out_q, op->act_min, op->act_max);
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_reshape_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    /* A reshape moves no value, so with matching quantization it is a copy or
     * nothing at all. A plan that declares a different output encoding still
     * has to be honored, and then it is a pass over the elements. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);
    if (!q.same) {
        for (uint32_t i = 0; i < n; i++)
            Y[i] = (int8_t)requantize_passthrough(
                X[i], q.in_zp, q.in_scale, q.out_zp, q.out_scale);
    } else if (X != Y) {
        memcpy(Y, X, n);
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_max_pool_s8(
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
    int32_t out_row_start = 0;
    int32_t in_row_start = 0;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
        /* Left pad only applies to a packed 2D (HW) tile; the right edge is
         * already implicit in the IW bound check below, the same way the
         * bottom edge is implicit in IH without a PB local. */
        if (mem->tile.width_tiled)
            PL = mem->tile.pad_left;
        out_row_start = mem->tile.out_row_start;
        in_row_start  = mem->tile.in_row_start;
    }

    /* Max preserves the value. The encoding is only preserved when the two
     * tensors declare the same quantization, which nothing enforces. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int oh_g = oh + out_row_start;
            int base_h = oh_g * SH - PT - in_row_start;
            int kh0, kh1;
            tap_range(base_h, IH, KH, 1, &kh0, &kh1);
            for (int ow = 0; ow < OW; ow++) {
                int base_w = ow * SW - PL;
                int kw0, kw1;
                tap_range(base_w, IW, KW, 1, &kw0, &kw1);
                for (int c = 0; c < C; c++) {
                    int8_t max_val = -128;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw;
                            int8_t v = X[((n * IH + ih) * IW + iw) * C + c];
                            if (v > max_val) max_val = v;
                        }
                    }
                    /* Apply the fused activation clamp, as TFLM/CMSIS MaxPool do
                     * (no-op when act range is the full [-128, 127]). */
                    int32_t out_q = requantize_passthrough(
                        max_val, q.in_zp, q.in_scale, q.out_zp, q.out_scale);
                    Y[((n * OH + oh_g) * OW + ow) * C + c] =
                        clamp_act(out_q, op->act_min, op->act_max);
                }
            }
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_concat_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    int8_t *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    /* Concat along one stored axis; see kern_concat for the shape of it. */
    uint8_t rank = plan->tensors[outs[0]].ndim;
    uint8_t axis = (rank == 4u && op->spatial.kernel_h == 0)
                 ? 3u : (uint8_t)op->spatial.kernel_h;
    int positions = 1;
    int inner = 1;
    int out_c_offset = 0;

    /* A constant operand carries no scale or zero point in the plan, so an
     * int8 concatenation cannot place one. The compiler keeps these float. */
    if (op->weight_idx != TIGRIS_NO_WEIGHT)
        return -1;

    if (axis == 0u || axis >= rank || (mem->tile.active && axis != rank - 1u))
        return -1;
    for (uint8_t j = 0; j < axis; j++)
        positions *= y_shape[j];
    for (uint8_t j = (uint8_t)(axis + 1u); j < rank; j++)
        inner *= y_shape[j];
    if (mem->tile.active) {
        positions = y_shape[0] * mem->tile.out_h;
        if (rank == 4u)
            positions *= mem->tile.out_w;
    }
    int total_OC = y_shape[axis] * inner;

    const tigris_quant_param_t *qp_y =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    float out_scale = qp_y ? qp_y->scale : 1.0f;
    int32_t out_zp  = qp_y ? qp_y->zero_point : 0;

    for (int i = 0; i < op->num_inputs; i++) {
        const int8_t *Xi = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[i]);
        const int32_t *xi_shape = tigris_tensor_shape(plan, &plan->tensors[ins[i]]);
        int Ci = xi_shape[axis] * inner;

        const tigris_quant_param_t *qp_i =
            tigris_tensor_quant(plan, &plan->tensors[ins[i]]);
        float in_scale = qp_i ? qp_i->scale : 1.0f;
        int32_t in_zp  = qp_i ? qp_i->zero_point : 0;

        /* Fast path: identical quant params -> memcpy */
        int same_qp = (fabsf(in_scale - out_scale) < 1e-7f && in_zp == out_zp);

        if (Xi == NULL)
            return -1;
        for (int q = 0; q < positions; q++) {
            const int8_t *src = Xi + (size_t)q * Ci;
            int8_t *dst = Y + (size_t)q * total_OC + out_c_offset;
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
        out_c_offset += Ci;
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_resize_nearest_s8(
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

    if (X == NULL || Y == NULL || op->spatial.kernel_h > 3) return -1;

    /* Under a tile the buffers hold a band, and which source row an output
     * row takes is a function of its global index, not of its position in
     * the band. See kern_resize_nearest. */
    float scale_h = tigris_resize_explicit_scale(plan, op, 0);
    float scale_w = tigris_resize_explicit_scale(plan, op, 1);
    int out_origin = 0;
    int in_origin = 0;
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        out_origin = mem->tile.out_row_origin;
        in_origin = mem->tile.in_row_origin;
    }

    /* Nearest-neighbor picks a value, it does not compute one, so with
     * matching quantization it is a byte copy. A plan that declares a
     * different output encoding still has to be honored. */
    passthrough_quant_t q = passthrough_quant(plan, ins[0], outs[0]);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih = tigris_resize_nearest(oh + out_origin, x_shape[1], y_shape[1],
                                           op->spatial.kernel_h, scale_h) - in_origin;
            if (ih < 0 || ih >= IH) return -1;
            for (int ow = 0; ow < OW; ow++) {
                int iw = tigris_resize_nearest(ow, x_shape[2], y_shape[2], op->spatial.kernel_h, scale_w);
                const int8_t *src = X + ((n * IH + ih) * IW + iw) * C;
                int8_t *dst = Y + ((n * OH + oh) * OW + ow) * C;
                if (q.same) {
                    memcpy(dst, src, (size_t)C);
                } else {
                    for (int c = 0; c < C; c++)
                        dst[c] = (int8_t)requantize_passthrough(
                            src[c], q.in_zp, q.in_scale,
                            q.out_zp, q.out_scale);
                }
            }
        }
    }
    return 0;
}

/* Matching quantization uses TFLite's 10-bit coordinates and one rounding at 2^20. */
static TIGRIS_KERNEL_NOINLINE int kern_resize_linear_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];
    float scale_h = tigris_resize_explicit_scale(plan, op, 0);
    float scale_w = tigris_resize_explicit_scale(plan, op, 1);
    int out_origin = 0;
    int in_origin = 0;

    if (X == NULL || Y == NULL || in_qp == NULL || out_qp == NULL ||
        op->spatial.kernel_h > 2 || !(out_qp->scale > 0.0f))
        return -1;
    float in_scale = in_qp->scale;
    float in_zp = (float)in_qp->zero_point;
    float inv_out = 1.0f / out_qp->scale;
    int32_t out_zp = out_qp->zero_point;
    int integer = in_qp->scale == out_qp->scale && in_qp->zero_point == out_qp->zero_point;
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        out_origin = mem->tile.out_row_origin;
        in_origin = mem->tile.in_row_origin;
    }

    if (integer) {
        int32_t scale_h_10 = tigris_resize_scale_s8(x_shape[1], y_shape[1], op->spatial.kernel_h, scale_h);
        int32_t scale_w_10 = tigris_resize_scale_s8(x_shape[2], y_shape[2], op->spatial.kernel_h, scale_w);
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                int32_t py, gy0, gy1;
                if (!tigris_resize_position_s8(oh + out_origin, x_shape[1],
                                               op->spatial.kernel_h, scale_h_10, &py, &gy0, &gy1))
                    return -1;
                int32_t y0 = gy0 - in_origin;
                int32_t y1 = gy1 - in_origin;
                if (y0 < 0 || y1 > IH - 1)
                    return -1;
                int32_t dy = py - (1 << 10) * gy0;
                for (int ow = 0; ow < OW; ow++) {
                    int32_t px, x0, x1;
                    if (!tigris_resize_position_s8(ow, x_shape[2],
                                                   op->spatial.kernel_h, scale_w_10, &px, &x0, &x1))
                        return -1;
                    int32_t dx = px - (1 << 10) * x0;
                    const int8_t *r0 = X + ((size_t)(n * IH + y0) * (size_t)IW) * (size_t)C;
                    const int8_t *r1 = X + ((size_t)(n * IH + y1) * (size_t)IW) * (size_t)C;
                    int8_t *dst = Y + ((size_t)(n * OH + oh) * (size_t)OW + (size_t)ow) *
                                  (size_t)C;
                    for (int c = 0; c < C; c++) {
                        int64_t sum =
                            (int64_t)r0[(size_t)x0 * C + c] * (int64_t)(((1 << 10) - dy) * ((1 << 10) - dx)) +
                            (int64_t)r1[(size_t)x0 * C + c] * (int64_t)(dy * ((1 << 10) - dx)) +
                            (int64_t)r0[(size_t)x1 * C + c] * (int64_t)(((1 << 10) - dy) * dx) +
                            (int64_t)r1[(size_t)x1 * C + c] * (int64_t)(dy * dx);
                        int64_t half = (sum > 0) ? (1 << 19) : -(1 << 19);
                        dst[c] = clamp_s8((int32_t)((sum + half) / (1 << 20)));
                    }
                }
            }
        }
        return 0;
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            float fy;
            int y0, y1;
            tigris_resize_position(oh + out_origin, x_shape[1], y_shape[1],
                                   op->spatial.kernel_h, scale_h, &fy, &y0, &y1);
            float wy = fy - (float)y0;
            y0 -= in_origin;
            y1 -= in_origin;
            if (y0 < 0 || y1 >= IH) return -1;
            for (int ow = 0; ow < OW; ow++) {
                float fx;
                int x0, x1;
                tigris_resize_position(ow, x_shape[2], y_shape[2],
                                       op->spatial.kernel_h, scale_w, &fx, &x0, &x1);
                float wx = fx - (float)x0;
                const int8_t *r0 = X + ((size_t)(n * IH + y0) * (size_t)IW) * (size_t)C;
                const int8_t *r1 = X + ((size_t)(n * IH + y1) * (size_t)IW) * (size_t)C;
                int8_t *dst = Y + ((size_t)(n * OH + oh) * (size_t)OW + (size_t)ow) *
                              (size_t)C;
                for (int c = 0; c < C; c++) {
                    float top = ((float)r0[(size_t)x0 * C + c] - in_zp) * (1.0f - wx) +
                                ((float)r0[(size_t)x1 * C + c] - in_zp) * wx;
                    float bot = ((float)r1[(size_t)x0 * C + c] - in_zp) * (1.0f - wx) +
                                ((float)r1[(size_t)x1 * C + c] - in_zp) * wx;
                    float real = (top * (1.0f - wy) + bot * wy) * in_scale;
                    dst[c] = clamp_s8((int32_t)roundf(real * inv_out) + out_zp);
                }
            }
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_sigmoid_s8(
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
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = lut[(uint8_t)X[i]];
    }
    return 0;
}

/* Erf via a 256-entry LUT, mirroring kern_sigmoid_s8. int8 in, int8 out with
 * its own output quant. */
static TIGRIS_KERNEL_NOINLINE int kern_erf_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    const tigris_quant_param_t *in_qp =
        tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t out_zp = out_qp ? out_qp->zero_point : 0;
    if (!(out_scale > 0.0f))
        return -1;

    int8_t lut[256];
    for (int i = 0; i < 256; i++) {
        int8_t x_val = (int8_t)i;
        float real = in_scale * (float)((int32_t)x_val - in_zp);
        int32_t q = (int32_t)roundf(erff(real) / out_scale) + out_zp;
        lut[i] = clamp_s8(q);
    }

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++)
        Y[i] = lut[(uint8_t)X[i]];
    return 0;
}

/**
 * Layer normalization over the final stored dimension, int8 in and out.
 *
 * The statistics are taken in float from the dequantized row, because a
 * normalization divides by a per-row standard deviation that no fixed
 * multiplier can stand in for. Scale and bias stay float weights, as they are
 * in the plan, and only the result is requantized.
 */
static TIGRIS_KERNEL_NOINLINE int kern_layer_norm_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    const tigris_tensor_t *in = &plan->tensors[ins[0]];
    const int32_t *shape = tigris_tensor_shape(plan, in);
    if (in->ndim == 0u)
        return -1;
    int32_t width = shape[in->ndim - 1u];
    if (width <= 0)
        return -1;

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    if (n % (uint32_t)width != 0u)
        return -1;
    uint32_t rows = n / (uint32_t)width;

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, in);
    const tigris_quant_param_t *out_qp =
        tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    float in_scale = in_qp ? in_qp->scale : 1.0f;
    int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    float out_scale = out_qp ? out_qp->scale : 1.0f;
    int32_t out_zp = out_qp ? out_qp->zero_point : 0;
    if (!(out_scale > 0.0f))
        return -1;

    const float *scale = (const float *)tigris_op_weight(plan, op);
    const float *bias = (const float *)tigris_op_bias(plan, op);
    if (!scale)
        return -1;

    uint8_t attr_len = 0;
    const uint8_t *raw = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_EPSILON, &attr_len);
    if (!raw || attr_len != sizeof(float))
        return -1;
    float epsilon;
    memcpy(&epsilon, raw, sizeof(epsilon));

    for (uint32_t r = 0; r < rows; r++) {
        const int8_t *row = X + (size_t)r * (size_t)width;
        int8_t *out_row = Y + (size_t)r * (size_t)width;
        float sum = 0.0f;
        for (int32_t c = 0; c < width; c++)
            sum += in_scale * (float)((int32_t)row[c] - in_zp);
        float mean = sum / (float)width;
        float sq = 0.0f;
        for (int32_t c = 0; c < width; c++) {
            float d = in_scale * (float)((int32_t)row[c] - in_zp) - mean;
            sq += d * d;
        }
        float inv = 1.0f / sqrtf(sq / (float)width + epsilon);
        for (int32_t c = 0; c < width; c++) {
            float real = in_scale * (float)((int32_t)row[c] - in_zp);
            float norm = (real - mean) * inv * scale[c] +
                         (bias ? bias[c] : 0.0f);
            int32_t q = (int32_t)roundf(norm / out_scale) + out_zp;
            out_row[c] = clamp_s8(q);
        }
    }
    return 0;
}

static int32_t elementwise_shift_left(int32_t value, int shift)
{
    int64_t result = (int64_t)value * ((int64_t)1 << shift);
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    return (int32_t)result;
}

static int32_t elementwise_multiply(int32_t value, int32_t multiplier, int shift)
{
    int left = shift > 0 ? shift : 0;
    int right = shift < 0 ? -shift : 0;
    int32_t product = sat_round_dbl_high_mul(
        elementwise_shift_left(value, left), multiplier);
    if (right > 31) return 0;
    uint32_t mask = ((uint32_t)1u << right) - 1u;
    uint32_t remainder = (uint32_t)product & mask;
    uint32_t threshold = (mask >> 1) + (product < 0 ? 1u : 0u);
    return (int32_t)((int64_t)product >> right) + (remainder > threshold ? 1 : 0);
}

static int elementwise_multiplier(double scale, int32_t *multiplier, int *shift)
{
    if (!isfinite(scale) || scale < 0.0) return 0;
    compute_quant_mult(scale, multiplier, shift);
    if (*shift < -31) {
        *shift = 0;
        *multiplier = 0;
    }
    return *shift <= 30;
}

static int elementwise_requantize(
    int32_t value, int32_t multiplier, int shift, int32_t zero, int8_t *output)
{
    /* Reject shifts and products outside the reference fixed-point domain. */
    if (shift < -31 || shift > 30) return 0;
    int left = shift > 0 ? shift : 0;
    int64_t shifted = (int64_t)value * ((int64_t)1 << left);
    if (shifted > INT32_MAX || shifted < INT32_MIN) return 0;
    int32_t result = elementwise_multiply(value, multiplier, shift);
    if (result > 127 - zero) *output = 127;
    else if (result < -128 - zero) *output = -128;
    else *output = (int8_t)(result + zero);
    return 1;
}

static int elementwise_leading_zeros(uint32_t value)
{
    int count = 0;
    if (value == 0u) return 32;
    uint32_t remaining = value;
    while ((remaining & 0x80000000u) == 0u) {
        remaining <<= 1;
        count++;
    }
    return count;
}

static int32_t elementwise_inverse_sqrt(int32_t value, int *output_shift)
{
    *output_shift = 0;
    if (value <= 1) return INT32_MAX;
    int32_t reduced = value;
    int shift = 11;
    while (reduced >= ((int32_t)1 << 29)) {
        reduced /= 4;
        shift++;
    }
    int pairs = (elementwise_leading_zeros((uint32_t)reduced) - 1) / 2 - 1;
    shift -= pairs;
    int32_t normalized = reduced * ((int32_t)1 << (2 * pairs));
    int32_t half = round_div_by_pot(normalized >> 1, 1);
    int32_t estimate = 268435456;
    for (int step = 0; step < 5; step++) {
        int32_t cube = sat_round_dbl_high_mul(
            sat_round_dbl_high_mul(estimate, estimate), estimate);
        cube = elementwise_shift_left(cube, 6);
        int32_t next = sat_round_dbl_high_mul(402653184, estimate) -
                       sat_round_dbl_high_mul(half, cube);
        estimate = elementwise_shift_left(next, 3);
    }
    estimate = sat_round_dbl_high_mul(estimate, 1518500250);
    if (shift < 0) return elementwise_shift_left(estimate, -shift);
    *output_shift = -shift;
    return estimate;
}

static int32_t elementwise_reciprocal(int32_t value, int *shift)
{
    int leading = elementwise_leading_zeros((uint32_t)value);
    uint32_t normalized = ((uint32_t)value << leading) - 0x80000000u;
    int32_t half = (int32_t)(((int64_t)normalized + INT32_MAX + 1) / 2);
    int32_t estimate = 1515870810 + sat_round_dbl_high_mul(half, -1010580540);
    for (int step = 0; step < 3; step++) {
        int32_t error = 536870912 - sat_round_dbl_high_mul(half, estimate);
        estimate += elementwise_shift_left(sat_round_dbl_high_mul(estimate, error), 2);
    }
    *shift = 31 - leading;
    return elementwise_shift_left(estimate, 1);
}

static int32_t div_round_by_pot(int32_t value, int exponent)
{
    /* An int32 value is strictly below half a unit for exponents >= 63. */
    if (exponent >= 63) return 0;
    uint64_t mask = ((uint64_t)1u << exponent) - 1u;
    uint64_t remainder = (uint64_t)(int64_t)value & mask;
    uint64_t threshold = (mask >> 1) + (value < 0 ? 1u : 0u);
    return (int32_t)((int64_t)value >> exponent) + (remainder > threshold ? 1 : 0);
}

static int elementwise_divide(
    int32_t numerator, int32_t denominator, int32_t multiplier, int shift,
    int32_t zero, int8_t *output)
{
    if (denominator == 0) return 0;
    int32_t a = denominator < 0 ? -numerator : numerator;
    int32_t b = denominator < 0 ? -denominator : denominator;
    int reciprocal_shift;
    int32_t inverse = elementwise_reciprocal(b, &reciprocal_shift);
    uint32_t sign_bits = a < 0 ? ~(uint32_t)a : (uint32_t)a;
    int headroom = elementwise_leading_zeros(sign_bits) - 1;
    int32_t scaled = (int32_t)((int64_t)a * ((int64_t)1 << headroom));
    int32_t quotient = sat_round_dbl_high_mul(scaled, inverse);
    int exponent = reciprocal_shift + headroom - shift;
    if (exponent < 0) return 0;
    int32_t result = div_round_by_pot(sat_round_dbl_high_mul(quotient, multiplier), exponent);
    if (result > 127 - zero) *output = 127;
    else if (result < -128 - zero) *output = -128;
    else *output = (int8_t)(result + zero);
    return 1;
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    const tigris_quant_param_t *input_quant;
    const tigris_quant_param_t *output_quant;
    uint32_t count;
    uint32_t width;
} activation_s8_t;

static int activation_s8_begin(
    const tigris_plan_t *plan, const tigris_op_t *op, const tigris_mem_t *mem,
    activation_s8_t *view)
{
    if (op->num_inputs != 1u || op->num_outputs != 1u) return 0;
    uint16_t input_index = tigris_op_inputs(plan, op)[0];
    uint16_t output_index = tigris_op_outputs(plan, op)[0];
    const tigris_tensor_t *input = &plan->tensors[input_index];
    const tigris_tensor_t *output = &plan->tensors[output_index];
    if (input->dtype != 3u || output->dtype != 3u ||
        input->ndim != output->ndim || input->size_bytes != output->size_bytes ||
        (input->ndim == 0u && (op->op_type == TIGRIS_OP_LOG_SOFTMAX ||
                              op->op_type == TIGRIS_OP_L2_NORMALIZATION))) return 0;
    const int32_t *shape = tigris_tensor_shape(plan, input);
    const int32_t *output_shape = tigris_tensor_shape(plan, output);
    for (uint8_t axis = 0; axis < input->ndim; axis++)
        if (shape[axis] <= 0 || shape[axis] != output_shape[axis]) return 0;
    view->input = (const int8_t *)tigris_mem_tensor_ptr(mem, input_index);
    view->output = (int8_t *)tigris_mem_tensor_ptr(mem, output_index);
    view->input_quant = tigris_tensor_quant(plan, input);
    view->output_quant = tigris_tensor_quant(plan, output);
    if (!view->input || !view->output || !view->input_quant || !view->output_quant ||
        !isfinite(view->input_quant->scale) || !isfinite(view->output_quant->scale) ||
        view->input_quant->scale <= 0.0f || view->output_quant->scale <= 0.0f) return 0;
    view->count = tile_aware_numel(plan, input_index, mem);
    view->width = input->ndim == 0u ? 1u : (uint32_t)shape[input->ndim - 1u];
    if (view->count == 0u || view->count % view->width != 0u) return 0;
    apply_pointwise_row_offset_s8(plan, input_index, mem, &view->input, &view->output);
    return 1;
}

static TIGRIS_KERNEL_NOINLINE int kern_leaky_relu_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    activation_s8_t view;
    if (!activation_s8_begin(plan, op, mem, &view)) return -1;
    float alpha = 0.01f;
    uint8_t length = 0;
    const uint8_t *raw = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_ALPHA, &length);
    if (raw) {
        if (length != sizeof(alpha)) return -1;
        memcpy(&alpha, raw, sizeof(alpha));
    }
    if (!isfinite(alpha)) return -1;
    int32_t multipliers[2];
    int shifts[2];
    float identity_scale = view.input_quant->scale / view.output_quant->scale;
    float alpha_scale = view.input_quant->scale * alpha / view.output_quant->scale;
    if (!elementwise_multiplier((double)identity_scale, &multipliers[0], &shifts[0]) ||
        !elementwise_multiplier((double)fabsf(alpha_scale), &multipliers[1], &shifts[1])) return -1;
    for (uint32_t i = 0; i < view.count; i++) {
        int32_t value = (int32_t)view.input[i] - view.input_quant->zero_point;
        int branch = value < 0 ? 1 : 0;
        int left = shifts[branch] > 0 ? shifts[branch] : 0;
        int64_t shifted = (int64_t)value * ((int64_t)1 << left);
        if (shifted < INT32_MIN || shifted > INT32_MAX) return -1;
        int32_t result = elementwise_multiply(value, multipliers[branch], shifts[branch]);
        if (branch == 1 && alpha_scale < 0.0f) {
            if (result == INT32_MIN) return -1;
            result = -result;
        }
        int64_t offset = (int64_t)result + view.output_quant->zero_point;
        if (offset < INT32_MIN || offset > INT32_MAX) return -1;
        view.output[i] = clamp_s8((int32_t)offset);
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_prelu_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    tigris_binary_operands_t operands;
    int8_t *output;
    uint32_t count;
    if (!binary_s8_begin(plan, op, op_index, mem, &operands, &output, &count)) return -1;
    const tigris_quant_param_t *iq = operands.quant[0];
    const tigris_quant_param_t *aq = operands.quant[1];
    const tigris_quant_param_t *oq = tigris_tensor_quant(plan, &plan->tensors[operands.output]);
    if (!iq || !aq || !oq || iq->scale <= 0.0f || aq->scale <= 0.0f || oq->scale <= 0.0f) return -1;
    int32_t multipliers[2];
    int shifts[2];
    if (!elementwise_multiplier((double)iq->scale / (double)oq->scale, &multipliers[0], &shifts[0]) ||
        !elementwise_multiplier((double)iq->scale * (double)aq->scale / (double)oq->scale,
                                &multipliers[1], &shifts[1])) return -1;
    for (uint32_t i = 0; i < count; i++) {
        int32_t value = binary_s8_at(&operands, 0u, i) - iq->zero_point;
        int branch = value < 0 ? 1 : 0;
        if (branch == 1) value *= binary_s8_at(&operands, 1u, i) - aq->zero_point;
        if (!elementwise_requantize(value, multipliers[branch], shifts[branch],
                                    oq->zero_point, &output[i])) return -1;
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_elu_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    activation_s8_t view;
    if (!activation_s8_begin(plan, op, mem, &view)) return -1;
    int8_t table[256];
    float inverse_scale = 1.0f / view.output_quant->scale;
    for (int32_t value = -128; value <= 127; value++) {
        float real = view.input_quant->scale * (float)(value - view.input_quant->zero_point);
        float transformed = real < 0.0f ? expf(real) - 1.0f : real;
        float rescaled = roundf(transformed * inverse_scale);
        float quantized = rescaled + (float)view.output_quant->zero_point;
        if (!isfinite(quantized) || (double)quantized < (double)INT32_MIN ||
            (double)quantized > (double)INT32_MAX) return -1;
        table[(uint8_t)(int8_t)value] = clamp_s8((int32_t)quantized);
    }
    for (uint32_t i = 0; i < view.count; i++) view.output[i] = table[(uint8_t)view.input[i]];
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_l2_normalization_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    activation_s8_t view;
    if (!activation_s8_begin(plan, op, mem, &view) ||
        view.output_quant->scale != 1.0f / 128.0f || view.output_quant->zero_point != 0) return -1;
    for (uint32_t row = 0; row < view.count / view.width; row++) {
        const int8_t *input = view.input + (size_t)row * view.width;
        int8_t *output = view.output + (size_t)row * view.width;
        int64_t sum = 0;
        for (uint32_t c = 0; c < view.width; c++) {
            int32_t centered = (int32_t)input[c] - view.input_quant->zero_point;
            sum += (int64_t)centered * (int64_t)centered;
            if (sum > INT32_MAX) return -1;
        }
        int shift;
        int32_t inverse = elementwise_inverse_sqrt((int32_t)sum, &shift);
        for (uint32_t c = 0; c < view.width; c++) {
            int32_t centered = (int32_t)input[c] - view.input_quant->zero_point;
            if (!elementwise_requantize(centered, inverse, shift + 7, 0, &output[c])) return -1;
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_elementwise_unary_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const int8_t *input = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t *output = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const tigris_quant_param_t *iq = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *oq = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);
    if (!input || !output || !iq || !oq || iq->scale <= 0.0f || oq->scale <= 0.0f)
        return -1;
    double scale = op->op_type == TIGRIS_OP_ABS
        ? (double)(iq->scale / oq->scale)
        : 1.0 / (double)(sqrtf(iq->scale) * oq->scale);
    int32_t multiplier;
    int shift;
    if (!elementwise_multiplier(scale, &multiplier, &shift)) return -1;
    uint32_t count = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &input, &output);
    for (uint32_t i = 0; i < count; i++) {
        int32_t value = (int32_t)input[i] - iq->zero_point;
        if (op->op_type == TIGRIS_OP_ABS) {
            value = value < 0 ? -value : value;
            if (iq->scale != oq->scale) {
                if (!elementwise_requantize(value, multiplier, shift, oq->zero_point, &output[i]))
                    return -1;
                continue;
            }
        } else {
            if (value < 0) return -1;
            if (value == 0) {
                output[i] = 127;
                continue;
            }
            int inverse_shift;
            int32_t inverse = elementwise_inverse_sqrt(value, &inverse_shift);
            value = elementwise_multiply(1, inverse, inverse_shift + 20);
            if (!elementwise_requantize(value, multiplier, shift - 20, oq->zero_point, &output[i]))
                return -1;
            continue;
        }
        output[i] = clamp_s8(value + oq->zero_point);
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_elementwise_binary_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    tigris_binary_operands_t operands;
    int8_t *y;
    uint32_t count;
    if (!binary_s8_begin(plan, op, op_index, mem, &operands, &y, &count)) return -1;
    uint16_t out = tigris_op_outputs(plan, op)[0];
    const tigris_quant_param_t *aq = operands.quant[0];
    const tigris_quant_param_t *bq = operands.quant[1];
    const tigris_quant_param_t *yq = tigris_tensor_quant(plan, &plan->tensors[out]);
    if (!aq || !bq || !yq || aq->scale <= 0.0f || bq->scale <= 0.0f || yq->scale <= 0.0f)
        return -1;
    int32_t multipliers[3] = {0, 0, 0};
    int shifts[3] = {0, 0, 0};
    if (op->op_type == TIGRIS_OP_SQUARED_DIFFERENCE) {
        double twice_max = 2.0 * (double)fmaxf(aq->scale, bq->scale);
        if (!elementwise_multiplier((double)aq->scale / twice_max, &multipliers[0], &shifts[0]) ||
            !elementwise_multiplier((double)bq->scale / twice_max, &multipliers[1], &shifts[1]) ||
            !elementwise_multiplier(twice_max * twice_max / (double)(16384.0f * yq->scale),
                                    &multipliers[2], &shifts[2]))
            return -1;
    } else if (op->op_type == TIGRIS_OP_DIV) {
        double scale = (double)(aq->scale / (bq->scale * yq->scale));
        if (!isfinite(scale) || scale < 0.0) return -1;
        compute_quant_mult(scale, &multipliers[2], &shifts[2]);
        if (shifts[2] < -31) {
            multipliers[2] = 0;
            shifts[2] = 0;
        }
    } else if (aq->scale != bq->scale || aq->scale != yq->scale ||
               aq->zero_point != bq->zero_point || aq->zero_point != yq->zero_point) {
        return -1;
    }
    for (uint32_t i = 0; i < count; i++) {
        int32_t av = binary_s8_at(&operands, 0u, i);
        int32_t bv = binary_s8_at(&operands, 1u, i);
        if (op->op_type == TIGRIS_OP_MAXIMUM || op->op_type == TIGRIS_OP_MINIMUM) {
            int32_t result = op->op_type == TIGRIS_OP_MAXIMUM ? (av > bv ? av : bv) : (av < bv ? av : bv);
            y[i] = (int8_t)result;
            continue;
        }
        av -= aq->zero_point;
        bv -= bq->zero_point;
        if (op->op_type == TIGRIS_OP_DIV) {
            if (!elementwise_divide(av, bv, multipliers[2], shifts[2], yq->zero_point, &y[i]))
                return -1;
            continue;
        }
        av = elementwise_multiply(av * 128, multipliers[0], shifts[0]);
        bv = elementwise_multiply(bv * 128, multipliers[1], shifts[1]);
        int32_t difference = av - bv;
        if (!elementwise_requantize(difference * difference, multipliers[2], shifts[2],
                                    yq->zero_point, &y[i]))
            return -1;
    }
    return 0;
}

/* HardSwish via a 256-entry LUT, mirroring kern_sigmoid_s8:
 * x * relu6(x + 3) / 6 on the dequantized input, requantized to the output. */
static TIGRIS_KERNEL_NOINLINE int kern_hardswish_s8(
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
        float gate = real + 3.0f;
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 6.0f) gate = 6.0f;
        int32_t q = (int32_t)roundf(real * gate / 6.0f / out_scale) + out_zp;
        lut[i] = clamp_s8(q);
    }

    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = lut[(uint8_t)X[i]];
    }
    return 0;
}

/* Tanh via a 256-entry LUT, mirroring kern_sigmoid_s8 (tanh in place of the
 * logistic). int8 input -> int8 output with its own output quant. */
static TIGRIS_KERNEL_NOINLINE int kern_tanh_s8(
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
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);
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
/* Fixed-point int8 Softmax, bit-exact with the TFLite reference kernel and
 * CMSIS-NN arm_softmax_s8. Input differences are Q5.26, exponentials and the
 * reciprocal Q0.31, the sum of exponentials Q12.19 (gemmlowp FixedPoint). */
#define SOFTMAX_DIFF_INT_BITS 5
#define SOFTMAX_ACCUM_INT_BITS 12

/* gemmlowp SaturatingRoundingMultiplyByPOT for a positive exponent. */
static inline int32_t sat_shift_left(int32_t x, int exponent)
{
    int32_t threshold = (int32_t)(((uint32_t)1u << (31 - exponent)) - 1u);
    if (x > threshold) return INT32_MAX;
    if (x < -threshold) return INT32_MIN;
    return (int32_t)((uint32_t)x << exponent);
}

static inline int32_t wrapping_add(int32_t a, int32_t b)
{
    return (int32_t)((uint32_t)a + (uint32_t)b);
}

/* exp(a) for a in [-1/4, 0), Q0.31 in and out: Taylor series around -1/8. */
static int32_t softmax_exp_quarter(int32_t a)
{
    const int32_t exp_minus_one_eighth = 1895147668;
    const int32_t one_third = 715827883;
    int32_t x = wrapping_add(a, (int32_t)1 << 28);
    int32_t x2 = sat_round_dbl_high_mul(x, x);
    int32_t x3 = sat_round_dbl_high_mul(x2, x);
    int32_t x4 = sat_round_dbl_high_mul(x2, x2);
    int32_t x4_over_4 = round_div_by_pot(x4, 2);
    int32_t poly = round_div_by_pot(
        wrapping_add(sat_round_dbl_high_mul(wrapping_add(x4_over_4, x3), one_third), x2), 1);
    return wrapping_add(exp_minus_one_eighth,
                        sat_round_dbl_high_mul(exp_minus_one_eighth, wrapping_add(x, poly)));
}

/* exp(a) for a <= 0, a in Q5.26, result in Q0.31 (gemmlowp exp_on_negative_values). */
static int32_t softmax_exp_negative(int32_t a)
{
    static const int32_t multipliers[7] = {
        1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242
    };
    const int32_t one_quarter = (int32_t)1 << 24;
    int32_t a_mod_quarter_minus_quarter = (a & (one_quarter - 1)) - one_quarter;
    int32_t result = softmax_exp_quarter(sat_shift_left(a_mod_quarter_minus_quarter, SOFTMAX_DIFF_INT_BITS));
    int32_t remainder = a_mod_quarter_minus_quarter - a;
    for (int k = 0; k < 7; k++) {
        /* Bit 24 + k of the remainder stands for exp(-2^(k-2)). */
        if ((remainder & ((int32_t)1 << (24 + k))) != 0)
            result = sat_round_dbl_high_mul(result, multipliers[k]);
    }
    return (a == 0) ? INT32_MAX : result;
}

/* 1 / (1 + x) for x in [0, 1), Q0.31 in and out: Newton-Raphson in Q2.29. */
static int32_t softmax_one_over_one_plus_x(int32_t x)
{
    int64_t sum = (int64_t)x + INT32_MAX;
    int32_t half_denominator = (int32_t)((sum + (sum >= 0 ? 1 : -1)) / 2);
    int32_t estimate = wrapping_add(1515870810, sat_round_dbl_high_mul(half_denominator, -1010580540));
    for (int i = 0; i < 3; i++) {
        int32_t product = sat_round_dbl_high_mul(half_denominator, estimate);
        int32_t one_minus_product = wrapping_add((int32_t)1 << 29, -product);
        estimate = wrapping_add(estimate,
            sat_shift_left(sat_round_dbl_high_mul(estimate, one_minus_product), 2));
    }
    return sat_shift_left(estimate, 1);
}

static inline int count_leading_zeros(uint32_t x)
{
    int n = 0;
    if (x == 0u) return 32;
    while ((x & (0x80000000u >> n)) == 0u) n++;
    return n;
}

typedef struct {
    int32_t multiplier;
    int left_shift;
    int32_t diff_min;
} softmax_params_t;

/* TFLite PreprocessSoftmaxScaling and CalculateInputRadius for beta = 1.
 * Returns 0 when the input scale is too small for a non-negative shift. */
static int softmax_params(float in_scale, softmax_params_t *p)
{
    double real = (double)in_scale * (double)(1L << (31 - SOFTMAX_DIFF_INT_BITS));
    if (real > 2147483647.0) real = 2147483647.0;
    compute_quant_mult(real, &p->multiplier, &p->left_shift);
    p->diff_min = 0;
    if (p->left_shift < 0 || p->left_shift > 30)
        return 0;
    double radius = ((double)((1 << SOFTMAX_DIFF_INT_BITS) - 1) *
                     (double)(1L << (31 - SOFTMAX_DIFF_INT_BITS))) /
                    (double)(1L << p->left_shift);
    p->diff_min = -(int32_t)floor(radius);
    return 1;
}

static inline int32_t softmax_scaled_diff(int32_t diff, const softmax_params_t *p)
{
    return sat_round_dbl_high_mul(diff * ((int32_t)1 << p->left_shift), p->multiplier);
}

/* One row with output scale 1/256 and zero point -128. */
static void softmax_row_fixed_point(
    const int8_t *x, int8_t *y, uint32_t classes, const softmax_params_t *p)
{
    int32_t max_q = x[0];
    for (uint32_t c = 1; c < classes; c++)
        if (x[c] > max_q) max_q = x[c];

    int32_t sum = 0;
    for (uint32_t c = 0; c < classes; c++) {
        int32_t diff = (int32_t)x[c] - max_q;
        if (diff >= p->diff_min)
            sum = wrapping_add(sum, round_div_by_pot(
                softmax_exp_negative(softmax_scaled_diff(diff, p)), SOFTMAX_ACCUM_INT_BITS));
    }

    int headroom = count_leading_zeros((uint32_t)sum);
    int bits_over_unit = SOFTMAX_ACCUM_INT_BITS - headroom;
    int32_t shifted_sum_minus_one =
        (int32_t)(((uint32_t)sum << headroom) - 0x80000000u);
    int32_t reciprocal = softmax_one_over_one_plus_x(shifted_sum_minus_one);

    for (uint32_t c = 0; c < classes; c++) {
        int32_t diff = (int32_t)x[c] - max_q;
        if (diff >= p->diff_min) {
            int32_t e = softmax_exp_negative(softmax_scaled_diff(diff, p));
            int32_t q = round_div_by_pot(sat_round_dbl_high_mul(reciprocal, e),
                                         bits_over_unit + 31 - 8);
            y[c] = clamp_s8(q - 128);
        } else {
            y[c] = (int8_t)-128;
        }
    }
}

static int32_t log_softmax_saturate(int64_t value)
{
    if (value > INT32_MAX) return INT32_MAX;
    if (value < INT32_MIN) return INT32_MIN;
    return (int32_t)value;
}

static int32_t log_softmax_power(int32_t value, int exponent)
{
    return exponent < 0 ? round_div_by_pot(value, -exponent) : sat_shift_left(value, exponent);
}

/* The fixed-point logarithm maps the Q12.19 sum to Q5.26. */
static int32_t log_softmax_log_sum(int32_t sum)
{
    int headroom_a = count_leading_zeros((uint32_t)sum);
    int32_t r_a = sat_shift_left(sat_round_dbl_high_mul(
        log_softmax_power(sum, headroom_a - 1), 1518500250), 1);
    int32_t power_a = log_softmax_saturate((int64_t)sat_shift_left(12 - headroom_a, 25) + 8388608);
    int32_t z_b = sat_round_dbl_high_mul(sum, 1518500250);
    int headroom_b = count_leading_zeros((uint32_t)z_b) - 1;
    int32_t r_b = log_softmax_power(sum, headroom_b);
    int32_t power_b = log_softmax_saturate((int64_t)sat_shift_left(12 - headroom_b, 25) - 8388608);
    int32_t r = r_a < r_b ? r_a : r_b;
    int32_t power = power_a > power_b ? power_a : power_b;
    int32_t p = (int32_t)(((int64_t)r + 1805811301 + 1) / 2);
    int32_t q = wrapping_add(r, -1805811301);
    q = wrapping_add(q, q);
    int32_t square = sat_round_dbl_high_mul(q, q);
    int32_t numerator = wrapping_add(sat_round_dbl_high_mul(q, r),
        sat_round_dbl_high_mul(sat_round_dbl_high_mul(q, square), 117049297));
    int32_t denominator = wrapping_add(sat_round_dbl_high_mul(p,
        wrapping_add(wrapping_add(1057819769, q), sat_round_dbl_high_mul(127690142, square))),
        sat_round_dbl_high_mul(638450708, q));
    int32_t reciprocal = softmax_one_over_one_plus_x(denominator);
    return sat_shift_left(wrapping_add(sat_round_dbl_high_mul(power, 1488522236),
        sat_round_dbl_high_mul(round_div_by_pot(numerator, 6), reciprocal)), 1);
}

static int log_softmax_row(
    const int8_t *input, int8_t *output, uint32_t width,
    const softmax_params_t *forward, int32_t reverse_multiplier, int reverse_shift)
{
    int32_t maximum = input[0];
    for (uint32_t c = 1; c < width; c++)
        if (input[c] > maximum) maximum = input[c];
    int64_t sum = 0;
    for (uint32_t c = 0; c < width; c++) {
        int32_t diff = (int32_t)input[c] - maximum;
        if (diff >= forward->diff_min) {
            sum += round_div_by_pot(softmax_exp_negative(softmax_scaled_diff(diff, forward)), 12);
            if (sum > INT32_MAX) return -1;
        }
    }
    int32_t log_sum = log_softmax_log_sum((int32_t)sum);
    if (log_sum < 0) return -1;
    int32_t shifted_log_sum = log_sum + INT32_MIN;
    int32_t minimum = elementwise_multiply(shifted_log_sum, reverse_multiplier, reverse_shift);
    if (minimum < forward->diff_min - 1) minimum = forward->diff_min - 1;
    for (uint32_t c = 0; c < width; c++) {
        int32_t diff = (int32_t)input[c] - maximum;
        if (diff > minimum) {
            int64_t centered = (int64_t)softmax_scaled_diff(diff, forward) - log_sum;
            if (centered < INT32_MIN || centered > INT32_MAX) return -1;
            output[c] = clamp_s8(round_div_by_pot((int32_t)centered, 22) + 127);
        } else {
            output[c] = -128;
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_log_softmax_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    activation_s8_t view;
    if (!activation_s8_begin(plan, op, mem, &view) ||
        view.output_quant->scale != 1.0f / 16.0f || view.output_quant->zero_point != 127) return -1;
    softmax_params_t forward;
    if (!softmax_params(view.input_quant->scale, &forward) || forward.left_shift < 1) return -1;
    double reverse = (double)((int32_t)1 << (31 - forward.left_shift)) / (double)forward.multiplier;
    int32_t reverse_multiplier;
    int reverse_shift;
    if (reverse <= 0.0 || reverse >= 1.0 ||
        !elementwise_multiplier(reverse, &reverse_multiplier, &reverse_shift)) return -1;
    for (uint32_t row = 0; row < view.count / view.width; row++) {
        if (log_softmax_row(view.input + (size_t)row * view.width,
                            view.output + (size_t)row * view.width, view.width,
                            &forward, reverse_multiplier, reverse_shift) != 0) return -1;
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_softmax_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    if (!plan || !op || !mem || op->num_inputs != 1 || op->num_outputs != 1)
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
    uint64_t whole = 1;
    for (uint8_t i = 0; i < x_tensor->ndim; i++) {
        if (x_shape[i] <= 0 || x_shape[i] != y_shape[i])
            return -1;
        whole *= (uint32_t)x_shape[i];
        if (whole > UINT32_MAX)
            return -1;
    }
    const uint32_t classes = (uint32_t)x_shape[x_tensor->ndim - 1];
    if (classes == 0 || whole != x_tensor->size_bytes)
        return -1;
    /* Normalization runs along the final stored dimension while a tile cuts
     * stored axis 1, so a tile always holds whole rows and needs nothing from
     * its neighbours. tile_aware_numel counts in those same terms. */
    const uint32_t count = tile_aware_numel(plan, ins[0], mem);
    if (count == 0 || count % classes != 0)
        return -1;

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;
    apply_pointwise_row_offset_s8(plan, ins[0], mem, &X, &Y);

    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, x_tensor);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, y_tensor);
    const float in_scale = in_qp ? in_qp->scale : 1.0f;
    const int32_t in_zp = in_qp ? in_qp->zero_point : 0;
    const float out_scale = out_qp ? out_qp->scale : 1.0f;
    const int32_t out_zp = out_qp ? out_qp->zero_point : 0;
    if (in_scale <= 0.0f || out_scale <= 0.0f)
        return -1;

    /* TFLite's int8 Softmax output quantization selects its fixed-point
     * kernel. Other output quantizations have no TFLite counterpart and are
     * computed in float. */
    softmax_params_t fixed;
    const int fixed_valid = softmax_params(in_scale, &fixed);
    const int use_fixed = fixed_valid && out_scale == 1.0f / 256.0f && out_zp == -128;

    for (uint32_t row = 0; row < (uint32_t)count / classes; row++) {
        const int8_t *x = X + (size_t)row * classes;
        int8_t *y = Y + (size_t)row * classes;
        if (use_fixed) {
            softmax_row_fixed_point(x, y, classes, &fixed);
            continue;
        }
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

static TIGRIS_KERNEL_NOINLINE int kern_mul_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    tigris_binary_operands_t operands;
    int8_t *Y;
    uint32_t n;
    if (!binary_s8_begin(plan, op, op_index, mem, &operands, &Y, &n))
        return -1;

    const tigris_quant_param_t *qp_a = operands.quant[0];
    const tigris_quant_param_t *qp_b = operands.quant[1];
    const tigris_quant_param_t *qp_y = tigris_tensor_quant(
        plan, &plan->tensors[tigris_op_outputs(plan, op)[0]]);

    float sa = qp_a ? qp_a->scale : 1.0f;
    int32_t za = qp_a ? qp_a->zero_point : 0;
    float sb = qp_b ? qp_b->scale : 1.0f;
    int32_t zb = qp_b ? qp_b->zero_point : 0;
    float sy = qp_y ? qp_y->scale : 1.0f;
    int32_t zy = qp_y ? qp_y->zero_point : 0;

    /* TFLite-exact quantized Mul (reference/integer_ops/mul.h): the integer
     * product of the offset operands, requantized by one fixed-point multiplier
     * derived in double from the three scales as TFLite's Prepare does. */
    int32_t y_mult;
    int y_shift;
    compute_quant_mult((double)sa * (double)sb / (double)sy, &y_mult, &y_shift);

    for (uint32_t i = 0; i < n; i++) {
        int32_t product = (binary_s8_at(&operands, 0u, i) - za) *
                          (binary_s8_at(&operands, 1u, i) - zb);
        int32_t q = multiply_by_quantized_multiplier(product, y_mult, y_shift) + zy;
        Y[i] = (op->fused_act == TIGRIS_ACT_NONE)
             ? clamp_s8(q)
             : clamp_act(q, op->act_min, op->act_max);
    }
    return 0;
}

typedef struct {
    const int8_t *input;
    int8_t *output;
    const tigris_quant_param_t *input_quant;
    const tigris_quant_param_t *output_quant;
    int32_t outer, reduced, inner;
} reduction_s8_t;

static int reduction_s8_begin(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, reduction_s8_t *view)
{
    if (mem->tile.active || op->num_inputs != 1u || op->num_outputs != 1u) return 0;
    uint16_t x = tigris_op_inputs(plan, op)[0];
    uint16_t y = tigris_op_outputs(plan, op)[0];
    const tigris_tensor_t *input = &plan->tensors[x];
    const tigris_tensor_t *output = &plan->tensors[y];
    uint8_t length = 0;
    const uint8_t *axis = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_AXES, &length);
    if (input->dtype != 3u || output->dtype != 3u || input->ndim != 3u ||
        !axis || length != 1u || axis[0] >= 3u) return 0;
    const int32_t *shape = tigris_tensor_shape(plan, input);
    uint32_t count = 1u;
    view->outer = 1;
    view->inner = 1;
    for (uint8_t a = 0; a < 3u; a++) {
        if (shape[a] <= 0 || count > UINT32_MAX / (uint32_t)shape[a]) return 0;
        count *= (uint32_t)shape[a];
        if (a < axis[0]) {
            if (view->outer > INT32_MAX / shape[a]) return 0;
            view->outer *= shape[a];
        } else if (a > axis[0]) {
            if (view->inner > INT32_MAX / shape[a]) return 0;
            view->inner *= shape[a];
        }
    }
    view->reduced = shape[axis[0]];
    uint32_t outputs = op->op_type == TIGRIS_OP_CUMSUM ? count : count / (uint32_t)view->reduced;
    if (input->size_bytes != count || output->size_bytes != outputs) return 0;
    if (op->op_type == TIGRIS_OP_CUMSUM) {
        const int32_t *out_shape = tigris_tensor_shape(plan, output);
        if (output->ndim != 3u) return 0;
        for (uint8_t a = 0; a < 3u; a++) if (shape[a] != out_shape[a]) return 0;
    }
    view->input = (const int8_t *)tigris_mem_tensor_ptr(mem, x);
    view->output = (int8_t *)tigris_mem_tensor_ptr(mem, y);
    view->input_quant = tigris_tensor_quant(plan, input);
    view->output_quant = tigris_tensor_quant(plan, output);
    if (!view->input || !view->output || !view->input_quant || !view->output_quant ||
        view->input_quant->num_channels != 1u || view->output_quant->num_channels != 1u ||
        !isfinite(view->input_quant->scale) || !isfinite(view->output_quant->scale) ||
        view->input_quant->scale <= 0.0f || view->output_quant->scale <= 0.0f ||
        view->input_quant->zero_point < -128 || view->input_quant->zero_point > 127 ||
        view->output_quant->zero_point < -128 || view->output_quant->zero_point > 127) return 0;
    return 1;
}

static int reduction_s8_requantize(
    int32_t value, int32_t multiplier, int shift, int32_t zero, int8_t *output)
{
    int left = shift > 0 ? shift : 0;
    int64_t shifted = (int64_t)value * ((int64_t)1 << left);
    if (shifted < INT32_MIN || shifted > INT32_MAX) return 0;
    int64_t result = (int64_t)elementwise_multiply(value, multiplier, shift) + zero;
    if (result < INT32_MIN || result > INT32_MAX) return 0;
    *output = clamp_s8((int32_t)result);
    return 1;
}

static TIGRIS_KERNEL_NOINLINE int kern_reduce_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    reduction_s8_t view;
    if (!reduction_s8_begin(plan, op, op_index, mem, &view)) return -1;
    int32_t multiplier = 0;
    int shift = 0;
    if (op->op_type == TIGRIS_OP_REDUCE_SUM) {
        if (!elementwise_multiplier((double)view.input_quant->scale / view.output_quant->scale,
                                    &multiplier, &shift)) return -1;
    } else if (view.input_quant->scale != view.output_quant->scale ||
               view.input_quant->zero_point != view.output_quant->zero_point) return -1;
    for (int32_t o = 0; o < view.outer; o++) {
        const int8_t *src = view.input + (size_t)o * (size_t)view.reduced * (size_t)view.inner;
        int8_t *dst = view.output + (size_t)o * (size_t)view.inner;
        for (int32_t i = 0; i < view.inner; i++) {
            int64_t value = 0;
            if (op->op_type == TIGRIS_OP_REDUCE_MAX) value = -128;
            else if (op->op_type == TIGRIS_OP_REDUCE_MIN) value = 127;
            for (int32_t r = 0; r < view.reduced; r++) {
                int32_t sample = src[(size_t)r * (size_t)view.inner + (size_t)i];
                if (op->op_type == TIGRIS_OP_REDUCE_SUM) {
                    value += sample;
                    if (value < INT32_MIN || value > INT32_MAX) return -1;
                } else if ((op->op_type == TIGRIS_OP_REDUCE_MAX && sample > value) ||
                           (op->op_type == TIGRIS_OP_REDUCE_MIN && sample < value)) value = sample;
            }
            if (op->op_type == TIGRIS_OP_REDUCE_SUM) {
                value -= (int64_t)view.input_quant->zero_point * view.reduced;
                if (value < INT32_MIN || value > INT32_MAX ||
                    !reduction_s8_requantize((int32_t)value, multiplier, shift,
                                             view.output_quant->zero_point, &dst[i])) return -1;
            } else dst[i] = (int8_t)value;
        }
    }
    return 0;
}

static TIGRIS_KERNEL_NOINLINE int kern_cumsum_s8(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    reduction_s8_t view;
    if (!reduction_s8_begin(plan, op, op_index, mem, &view)) return -1;
    uint8_t length = 0;
    const uint8_t *options = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_CUMSUM_OPTIONS, &length);
    if (!options || length != 2u || options[0] > 1u || options[1] > 1u) return -1;
    double scale = (2.0 * view.input_quant->scale) / (1048576.0 * view.output_quant->scale);
    int32_t multiplier;
    int shift;
    if (scale <= 0.0 || scale >= 1.0 || !elementwise_multiplier(scale, &multiplier, &shift) ||
        shift > 0) return -1;
    int32_t initial = sat_round_dbl_high_mul(-view.input_quant->zero_point * 1048576, 1073741824);
    for (int32_t o = 0; o < view.outer; o++) {
        for (int32_t i = 0; i < view.inner; i++) {
            /* The reference seeds the accumulator with the scaled input offset. */
            int64_t sum = initial;
            for (int32_t r = 0; r < view.reduced; r++) {
                int32_t step = options[1] != 0u ? view.reduced - 1 - r : r;
                size_t index = ((size_t)o * (size_t)view.reduced + (size_t)step) * (size_t)view.inner + (size_t)i;
                int32_t sample = (int32_t)view.input[index] - view.input_quant->zero_point;
                int32_t scaled = sat_round_dbl_high_mul(sample * 1048576, 1073741824);
                int32_t value = (int32_t)sum;
                sum += scaled;
                if (sum < INT32_MIN || sum > INT32_MAX) return -1;
                if (options[0] == 0u) value = (int32_t)sum;
                if (!reduction_s8_requantize(value, multiplier, shift,
                                             view.output_quant->zero_point, &view.output[index])) return -1;
            }
        }
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
    (void)user_ctx;

#ifdef TIGRIS_COUNT_KERNEL_ROWS
    extern unsigned long g_tigris_kernel_rows;
    if (mem->tile.active)
        g_tigris_kernel_rows += (unsigned long)mem->tile.out_h;
#endif

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return kern_conv2d_s8(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return kern_depthwise_conv2d_s8(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return kern_fully_connected_s8(plan, op, mem);
    case TIGRIS_OP_RELU:        return kern_relu_s8(plan, op, mem);
    case TIGRIS_OP_RELU6:       return kern_relu6_s8(plan, op, mem);
    case TIGRIS_OP_ADD:         return kern_add_s8(plan, op, op_index, mem);
    case TIGRIS_OP_GLOBAL_AVG:  return kern_global_avg_pool_s8(plan, op, op_index, mem);
    case TIGRIS_OP_AVG_POOL:     return kern_avg_pool_s8(plan, op, mem);
    case TIGRIS_OP_RESHAPE:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_FLATTEN:     return kern_reshape_s8(plan, op, mem);
    case TIGRIS_OP_MAX_POOL:    return kern_max_pool_s8(plan, op, mem);
    case TIGRIS_OP_CONCAT:      return kern_concat_s8(plan, op, mem);
    case TIGRIS_OP_RESIZE:      return kern_resize_nearest_s8(plan, op, mem);
    case TIGRIS_OP_SIGMOID:     return kern_sigmoid_s8(plan, op, mem);
    case TIGRIS_OP_TANH:        return kern_tanh_s8(plan, op, mem);
    case TIGRIS_OP_SOFTMAX:     return kern_softmax_s8(plan, op, mem);
    case TIGRIS_OP_LEAKY_RELU: return kern_leaky_relu_s8(plan, op, op_index, mem);
    case TIGRIS_OP_PRELU: return kern_prelu_s8(plan, op, op_index, mem);
    case TIGRIS_OP_ELU: return kern_elu_s8(plan, op, mem);
    case TIGRIS_OP_LOG_SOFTMAX: return kern_log_softmax_s8(plan, op, mem);
    case TIGRIS_OP_L2_NORMALIZATION: return kern_l2_normalization_s8(plan, op, mem);
    case TIGRIS_OP_MUL:         return kern_mul_s8(plan, op, op_index, mem);
    case TIGRIS_OP_CONV1D:      return kern_conv1d_s8(plan, op, mem);
    case TIGRIS_OP_CONV_TRANSPOSE: return kern_conv_transpose_s8(plan, op, mem);
    case TIGRIS_OP_SUB:         return kern_sub_s8(plan, op, op_index, mem);
    case TIGRIS_OP_GLOBAL_MAX:  return kern_global_max_pool_s8(plan, op, mem);
    case TIGRIS_OP_MATMUL:      return kern_matmul_s8(plan, op, mem);
    case TIGRIS_OP_TRANSPOSE:   return tigris_transpose_execute(plan, op, op_index, mem);
    case TIGRIS_OP_ERF:         return kern_erf_s8(plan, op, mem);
    case TIGRIS_OP_HARDSWISH:   return kern_hardswish_s8(plan, op, mem);
    case TIGRIS_OP_ABS:        return kern_elementwise_unary_s8(plan, op, mem);
    case TIGRIS_OP_RSQRT:      return kern_elementwise_unary_s8(plan, op, mem);
    case TIGRIS_OP_DIV:        return kern_elementwise_binary_s8(plan, op, op_index, mem);
    case TIGRIS_OP_SQUARED_DIFFERENCE: return kern_elementwise_binary_s8(plan, op, op_index, mem);
    case TIGRIS_OP_MAXIMUM:    return kern_elementwise_binary_s8(plan, op, op_index, mem);
    case TIGRIS_OP_MINIMUM:    return kern_elementwise_binary_s8(plan, op, op_index, mem);
    case TIGRIS_OP_RESIZE_LINEAR: return kern_resize_linear_s8(plan, op, mem);
    case TIGRIS_OP_LAYER_NORM:  return kern_layer_norm_s8(plan, op, op_index, mem);
    case TIGRIS_OP_REDUCE_MEAN: return kern_reduce_mean_s8(plan, op, op_index, mem);
    case TIGRIS_OP_REDUCE_MAX:
    case TIGRIS_OP_REDUCE_MIN:
    case TIGRIS_OP_REDUCE_SUM: return kern_reduce_s8(plan, op, op_index, mem);
    case TIGRIS_OP_ARG_MAX:
    case TIGRIS_OP_ARG_MIN: return tigris_arg_execute(plan, op, op_index, mem);
    case TIGRIS_OP_CUMSUM: return kern_cumsum_s8(plan, op, op_index, mem);
    case TIGRIS_OP_SPLIT:       return kern_split_s8(plan, op, mem);
    case TIGRIS_OP_PAD:         return kern_pad_s8(plan, op, op_index, mem);
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

static int32_t depthwise_multiplier(const tigris_plan_t *plan, const tigris_op_t *op)
{
    const tigris_tensor_t *x = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *y = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    int32_t c = tigris_tensor_shape(plan, x)[3];
    return c > 0 ? tigris_tensor_shape(plan, y)[3] / c : 0;
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
        /* Plain-height tiles run on the CMSIS-NN adapter, which honors the tile
         * context (in_h/out_h/pad_top/pad_bottom) like the ESP-NN adapter does.
         * Dilated tiles still fall back to the reference kernel until native
         * dilated-tile parity is covered; rolled and 2D-width tiles are already
         * routed to reference by tigris_accel_try_s8_ref before this policy. */
        return (effective_dilation(op->spatial.dilation_h) != 1u ||
                effective_dilation(op->spatial.dilation_w) != 1u);
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

int tigris_accel_row_band(
    int32_t   in_h,
    uint16_t  pad_top,
    uint16_t  stride_h,
    uint16_t  kernel_h,
    int32_t   out_start,
    int32_t   out_rows,
    int32_t  *in_start,
    int32_t  *in_rows,
    uint16_t *band_pad_top)
{
    if (!in_start || !in_rows || !band_pad_top || in_h <= 0 ||
        stride_h == 0u || kernel_h == 0u || out_start < 0 || out_rows <= 0)
        return 0;

    /* The band's first window starts `lead` padding rows above the input when
     * lead is positive, and -lead rows into it otherwise; `end` is one past
     * the last row its last window reads. */
    int64_t first_row = (int64_t)out_start * (int64_t)stride_h;
    int64_t lead = (int64_t)pad_top - first_row;
    int64_t end = first_row + ((int64_t)out_rows - 1) * (int64_t)stride_h
                - (int64_t)pad_top + (int64_t)kernel_h;
    int64_t start = 0;
    int64_t pad = 0;
    if (lead > 0)
        pad = lead;
    else
        start = -lead;
    int64_t stop = (end < (int64_t)in_h) ? end : (int64_t)in_h;
    if (stop <= start || pad > (int64_t)UINT16_MAX)
        return 0;

    *in_start = (int32_t)start;
    *in_rows = (int32_t)(stop - start);
    *band_pad_top = (uint16_t)pad;
    return 1;
}

int tigris_accel_binary_params(const tigris_plan_t *plan, const tigris_op_t *op,
                               const tigris_mem_t *mem, tigris_accel_binary_t *params)
{
    if (!plan || !op || !params || op->num_inputs != 2 || op->num_outputs != 1 ||
        op->weight_idx != TIGRIS_NO_WEIGHT || op->bias_idx != TIGRIS_NO_WEIGHT)
        return 0;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const tigris_tensor_t *a = &plan->tensors[ins[0]];
    const tigris_tensor_t *b = &plan->tensors[ins[1]];
    const tigris_tensor_t *y = &plan->tensors[tigris_op_outputs(plan, op)[0]];
    if (a->dtype != 3 || b->dtype != 3 || y->dtype != 3 ||
        a->ndim != b->ndim || a->ndim != y->ndim || a->size_bytes > INT32_MAX ||
        a->size_bytes != b->size_bytes || a->size_bytes != y->size_bytes ||
        ((a->flags | b->flags) & TIGRIS_TENSOR_CONSTANT) != 0)
        return 0;
    const int32_t *as = tigris_tensor_shape(plan, a);
    const int32_t *bs = tigris_tensor_shape(plan, b);
    const int32_t *ys = tigris_tensor_shape(plan, y);
    for (uint8_t i = 0; i < a->ndim; i++)
        if (as[i] != bs[i] || as[i] != ys[i]) return 0;
    const tigris_quant_param_t *aq = tigris_tensor_quant(plan, a);
    const tigris_quant_param_t *bq = tigris_tensor_quant(plan, b);
    const tigris_quant_param_t *yq = tigris_tensor_quant(plan, y);
    if (!aq || !bq || !yq || aq->num_channels != 1 || bq->num_channels != 1 || yq->num_channels != 1)
        return 0;
    if (op->op_type == TIGRIS_OP_MUL) {
        double scale = (double)aq->scale * (double)bq->scale / (double)yq->scale;
        if (!isfinite(scale) || scale <= 0.0) return 0;
        int shift;
        compute_quant_mult(scale, &params->multiplier, &shift);
        if (shift < -31 || shift > 15) return 0;
        params->shift = shift;
    } else if (op->op_type == TIGRIS_OP_MAXIMUM || op->op_type == TIGRIS_OP_MINIMUM) {
        if (aq->scale != bq->scale || aq->scale != yq->scale ||
            aq->zero_point != bq->zero_point || aq->zero_point != yq->zero_point)
            return 0;
        params->multiplier = 0;
        params->shift = 0;
    } else {
        return 0;
    }
    params->input1_offset = -aq->zero_point;
    params->input2_offset = -bq->zero_point;
    params->output_offset = yq->zero_point;
    params->activation_min = op->fused_act == TIGRIS_ACT_NONE ? -128 : op->act_min;
    params->activation_max = op->fused_act == TIGRIS_ACT_NONE ? 127 : op->act_max;
    params->count = (int32_t)(mem ? tile_aware_numel(plan, ins[0], mem) : a->size_bytes);
    return 1;
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

    /* ESP-NN depthwise has no batch dimension. */
    if ((backend == TIGRIS_ACCEL_ESP_NN || backend == TIGRIS_ACCEL_CMSIS_NN) &&
        op && op->op_type == TIGRIS_OP_DEPTHWISE && plan &&
        (depthwise_multiplier(plan, op) < 1 ||
         (backend == TIGRIS_ACCEL_ESP_NN && depthwise_multiplier(plan, op) > 1 &&
          tigris_tensor_shape(plan, &plan->tensors[tigris_op_inputs(plan, op)[0]])[0] != 1)))
        return TIGRIS_ACCEL_ROUTE_S8_REF;

    if (op && (op->op_type == TIGRIS_OP_MUL || op->op_type == TIGRIS_OP_MAXIMUM ||
               op->op_type == TIGRIS_OP_MINIMUM)) {
        tigris_accel_binary_t params;
        if ((backend == TIGRIS_ACCEL_ESP_NN && op->op_type != TIGRIS_OP_MUL) ||
            !tigris_accel_binary_params(plan, op, NULL, &params))
            return TIGRIS_ACCEL_ROUTE_S8_REF;
    }

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

    /* 2D tiles carry mem->tile.width_tiled and partition width via
     * pad_left/tile-in_w as well as height (see exec_stage_tiled_2d). The
     * ESP-NN/CMSIS-NN Conv/Depthwise adapters honor the packed-width tile
     * natively (in_w/out_w/pad_left, right pad implicit via IW clipping); every
     * other op still routes to s8_ref. */
    if (mem->tile.active && mem->tile.width_tiled &&
        !is_conv_or_depthwise(op)) {
        *handled = 1;
        return tigris_dispatch_kernel_s8(
            plan, op, op_index, mem, user_ctx);
    }

    /* A row band hands the kernel a buffer holding only the band's rows, and
     * says so in mem->tile.out_h rather than in the tensor shape. The vendor
     * FullyConnected kernels take their row count from the output tensor, so
     * they would read and write the whole matrix into a band-sized allocation.
     * The s8 reference kernel honors the band, so route there. */
    if (mem->tile.active && mem->tile.row_tiled) {
        *handled = 1;
        return tigris_dispatch_kernel_s8(
            plan, op, op_index, mem, user_ctx);
    }

    /* Line-buffered chains roll overlap rows across tiles by setting
     * mem->tile.out_row_start / in_row_start (see exec_chain_tiled). The
     * ESP-NN/CMSIS-NN Conv/Depthwise adapters honor the roll natively - they
     * fold out_row_start into the output pointer and (out_row_start*stride -
     * in_row_start) into the input pointer, matching the s8 kernel's index math.
     * Every other op ignores the offsets and would write rows at the wrong
     * position, so a rolled non-conv/depthwise op still routes to s8_ref. */
    if (mem->tile.active &&
        (mem->tile.out_row_start != 0 || mem->tile.in_row_start != 0) &&
        !is_conv_or_depthwise(op)) {
        *handled = 1;
        return tigris_dispatch_kernel_s8(
            plan, op, op_index, mem, user_ctx);
    }

    *handled = tigris_accel_pre_route(
                   backend, plan, op, mem->tile.active) ==
               TIGRIS_ACCEL_ROUTE_S8_REF;
    if (!*handled)
        return 0;

    return tigris_dispatch_kernel_s8(
        plan, op, op_index, mem, user_ctx);
}

/**
 * @file tigris_kernels.c
 * @brief Reference float32 kernels + dispatch.
 *
 * All kernels are static. Data layout: NHWC, float32.
 * Weight/bias accessed via tigris_op_weight() / tigris_op_bias().
 */

#include "tigris_kernels.h"

#include "tigris_kernel_window.h"
#include "tigris_binary_operands.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef TIGRIS_COUNT_KERNEL_ROWS
/* Test-only instrumentation: total output rows computed across all kernel
 * dispatches. Incremented by mem->tile.out_h per tiled dispatch so a test can
 * compare the line-buffered roll (new rows only) against the recompute path
 * (full back-propagated range every tile). Never compiled into shipping code. */
unsigned long g_tigris_kernel_rows = 0;
#endif

/* Helpers */

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

/** True when two tensor descriptors have the same logical shape. */
static int tensor_shapes_equal(
    const tigris_plan_t *plan,
    const tigris_tensor_t *a,
    const tigris_tensor_t *b)
{
    if (a->ndim != b->ndim)
        return 0;
    if (a->ndim == 0)
        return 1;

    const int32_t *a_shape = tigris_tensor_shape(plan, a);
    const int32_t *b_shape = tigris_tensor_shape(plan, b);
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a_shape[i] != b_shape[i])
            return 0;
    }
    return 1;
}

/** Validate that a tensor's shape describes exactly its float32 byte size. */
static int tensor_is_f32(const tigris_plan_t *plan, const tigris_tensor_t *t)
{
    /* ONNX TensorProto.DataType.FLOAT == 1. */
    if (!plan || !t || t->dtype != 1)
        return 0;

    uint64_t numel = 1;
    if (t->ndim > 0) {
        const int32_t *shape = tigris_tensor_shape(plan, t);
        if (!shape)
            return 0;
        for (uint8_t i = 0; i < t->ndim; i++) {
            if (shape[i] <= 0)
                return 0;
            numel *= (uint32_t)shape[i];
            if (numel > UINT32_MAX / sizeof(float))
                return 0;
        }
    }
    return (uint32_t)(numel * sizeof(float)) == t->size_bytes;
}

/** Load float32 weight data without assuming the packed blob is aligned. */
static float load_weight_f32(const uint8_t *data, uint32_t index)
{
    float value;
    memcpy(&value, data + (size_t)index * sizeof(value), sizeof(value));
    return value;
}

/** Apply fused activation (float). */
static inline float apply_fused_act_f32(float v, uint8_t fused_act)
{
    if (fused_act == TIGRIS_ACT_RELU)  return v > 0.f ? v : 0.f;
    if (fused_act == TIGRIS_ACT_RELU6) { v = v > 0.f ? v : 0.f; return v < 6.f ? v : 6.f; }
    return v;
}

/** Shift a pointwise (height-preserving) kernel's input/output base pointers
 * to the current sub-range: X by in_row_start rows, Y by out_row_start rows.
 * No-op when tiling is inactive (both offsets are then 0). */
static void apply_pointwise_row_offset_f32(
    const tigris_plan_t *plan, uint16_t tidx, const tigris_mem_t *mem,
    const float **x, float **y)
{
    if (!mem->tile.active)
        return;
    uint32_t row_elems = tile_row_elems(plan, tidx, mem);
    *x += (size_t)mem->tile.in_row_start * row_elems;
    *y += (size_t)mem->tile.out_row_start * row_elems;
}

/* Kernels */

static int kern_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    const float *W = (const float *)tigris_op_weight(plan, op);
    const float *B = (const float *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

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

    /* Weight layout: [OC, KH, KW, IC] (OHWI) */
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
                    float sum = B ? B[oc] : 0.0f;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh * DH;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw * DW;
                            for (int ic = 0; ic < IC; ic++) {
                                float x_val = X[((n * IH + ih) * IW + iw) * IC + ic];
                                float w_val = W[((oc * KH + kh) * KW + kw) * IC + ic];
                                sum += x_val * w_val;
                            }
                        }
                    }
                    Y[((n * OH + oh_g) * OW + ow) * OC + oc] = apply_fused_act_f32(sum, op->fused_act);
                }
            }
        }
    }
    return 0;
}

/** ConvTranspose (transposed convolution) via input-gather.
 * Untiled: runs as a whole standalone stage. group == 1 and dilation == 1
 * are the only supported configuration in scope; output shape (including
 * output_padding) is already resolved by the compiler into y_shape.
 *
 * For each output pixel, gather from the input positions that scattered
 * into it under the forward strided-transpose relation
 * (oh = ih*SH - PT + kh, ow = iw*SW - PL + kw), inverted to solve for
 * ih/iw given oh/ow/kh/kw. A tap is skipped when it does not land on the
 * stride lattice (num_h/num_w not divisible by SH/SW) or when the
 * resulting ih/iw falls outside the input.
 */
static int kern_conv_transpose(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    const float *W = (const float *)tigris_op_weight(plan, op);
    const float *B = (const float *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

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

    /* Tile override. Structurally identical to kern_conv2d: the tile's
     * global offset is folded into the effective pads PT/PL by the executor
     * (PT_eff = oh0 + PT - ih0*SH), so the gather below runs unchanged in
     * local packed coordinates. Left pad only applies to a packed 2D (HW)
     * tile, same as kern_conv2d. ConvTranspose never uses out_row_start /
     * in_row_start (2D-only, no line-buffered chain). */
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
                    float sum = B ? B[oc] : 0.0f;
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
                                float x_val = X[((n * IH + ih) * IW + iw) * IC + ic];
                                float w_val = W[((oc * KH + kh) * KW + kw) * IC + ic];
                                sum += x_val * w_val;
                            }
                        }
                    }
                    Y[((n * OH + oh) * OW + ow) * OC + oc] = apply_fused_act_f32(sum, op->fused_act);
                }
            }
        }
    }
    return 0;
}

static int kern_depthwise_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    const float *W = (const float *)tigris_op_weight(plan, op);
    const float *B = (const float *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    /* NHWC layout */
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

    /* Depthwise: group == C, each input channel convolved independently by
     * M filters; output channel oc reads input channel oc / M.
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
                    float sum = B ? B[oc] : 0.0f;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh * DH;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw * DW;
                            float x_val = X[((n * IH + ih) * IW + iw) * C + c];
                            float w_val = W[(kh * KW + kw) * OC + oc];
                            sum += x_val * w_val;
                        }
                    }
                    Y[((n * OH + oh_g) * OW + ow) * OC + oc] = apply_fused_act_f32(sum, op->fused_act);
                }
            }
        }
    }
    return 0;
}

static int kern_relu(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++)
        Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
    return 0;
}

static int kern_relu6(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++) {
        float v = X[i] > 0.0f ? X[i] : 0.0f;
        Y[i] = v < 6.0f ? v : 6.0f;
    }
    return 0;
}

static int kern_sigmoid(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++)
        Y[i] = 1.0f / (1.0f + expf(-X[i]));
    return 0;
}

static float elementwise_round(float value)
{
    float base = floorf(value);
    float fraction = value - base;
    if (fraction < 0.5f) return base;
    if (fraction > 0.5f) return base + 1.0f;
    return fmodf(base, 2.0f) == 0.0f ? base : base + 1.0f;
}

static int kern_elementwise_unary_f32(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *input = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float *output = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t count = tile_aware_numel(plan, ins[0], mem);
    if (!input || !output) return -1;
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &input, &output);
    for (uint32_t i = 0; i < count; i++) {
        float value = input[i];
        switch (op->op_type) {
        case TIGRIS_OP_ABS: value = fabsf(value); break;
        case TIGRIS_OP_RSQRT: value = 1.0f / sqrtf(value); break;
        case TIGRIS_OP_NEG: value = -value; break;
        case TIGRIS_OP_EXP: value = expf(value); break;
        case TIGRIS_OP_LOG: value = logf(value); break;
        case TIGRIS_OP_SQRT: value = sqrtf(value); break;
        case TIGRIS_OP_SQUARE: value *= value; break;
        case TIGRIS_OP_FLOOR: value = floorf(value); break;
        case TIGRIS_OP_CEIL: value = ceilf(value); break;
        case TIGRIS_OP_ROUND: value = elementwise_round(value); break;
        case TIGRIS_OP_SIN: value = sinf(value); break;
        case TIGRIS_OP_COS: value = cosf(value); break;
        default: return -1;
        }
        output[i] = value;
    }
    return 0;
}

/* HardSwish: x * relu6(x + 3) / 6. */
static int kern_hardswish(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++) {
        float gate = X[i] + 3.0f;
        if (gate < 0.0f) gate = 0.0f;
        if (gate > 6.0f) gate = 6.0f;
        Y[i] = X[i] * gate / 6.0f;
    }
    return 0;
}

static int kern_erf(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++)
        Y[i] = erff(X[i]);
    return 0;
}

/**
 * Layer normalization over the final stored dimension.
 *
 * The normalizer places the operand in the layout whose last stored axis is
 * the one the model normalizes, so the kernel always walks the trailing axis
 * and a height tile holds whole normalization rows, exactly as for Softmax.
 * Scale is the operator's weight and bias its optional bias, both indexed by
 * position along that axis.
 */
static int kern_layer_norm(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
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
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    if (n % (uint32_t)width != 0u)
        return -1;
    uint32_t rows = n / (uint32_t)width;

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
        const float *row = X + (size_t)r * (size_t)width;
        float *out_row = Y + (size_t)r * (size_t)width;
        float sum = 0.0f;
        for (int32_t c = 0; c < width; c++)
            sum += row[c];
        float mean = sum / (float)width;
        float sq = 0.0f;
        for (int32_t c = 0; c < width; c++) {
            float d = row[c] - mean;
            sq += d * d;
        }
        float inv = 1.0f / sqrtf(sq / (float)width + epsilon);
        for (int32_t c = 0; c < width; c++) {
            float norm = (row[c] - mean) * inv;
            out_row[c] = norm * scale[c] + (bias ? bias[c] : 0.0f);
        }
    }
    return 0;
}

static int kern_tanh(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    uint32_t n = tile_aware_numel(plan, ins[0], mem);
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);
    for (uint32_t i = 0; i < n; i++)
        Y[i] = tanhf(X[i]);
    return 0;
}

/* Softmax over the final dimension.  Supported plans use ONNX's default
 * final-axis form; Softmax is deliberately not tileable. */
static int kern_softmax(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    if (!plan || !op || !mem || op->num_inputs != 1 || op->num_outputs != 1)
        return -1;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *x_tensor = &plan->tensors[ins[0]];
    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];
    if (!tensor_is_f32(plan, x_tensor) || !tensor_is_f32(plan, y_tensor) ||
        !tensor_shapes_equal(plan, x_tensor, y_tensor) || x_tensor->ndim == 0)
        return -1;

    const int32_t *shape = tigris_tensor_shape(plan, x_tensor);
    const uint32_t classes = (uint32_t)shape[x_tensor->ndim - 1];
    /* Normalization runs along the final stored dimension while a tile cuts
     * stored axis 1, so a tile always holds whole rows and needs nothing from
     * its neighbours. tile_aware_numel counts in those same terms. */
    const uint32_t count = tile_aware_numel(plan, ins[0], mem);
    if (classes == 0 || count % classes != 0)
        return -1;

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;
    apply_pointwise_row_offset_f32(plan, ins[0], mem, &X, &Y);

    for (uint32_t row = 0; row < count / classes; row++) {
        const float *x = X + (size_t)row * classes;
        float *y = Y + (size_t)row * classes;
        float maximum = x[0];
        for (uint32_t c = 1; c < classes; c++)
            if (x[c] > maximum) maximum = x[c];

        /* Keep each exponential rather than evaluating it again to
         * normalize. The value, the summation order and the division are
         * unchanged; only the second call to expf goes. Writing into y as we
         * read x is safe in place, since each index is read before it is
         * written and the maximum is already known. */
        float sum = 0.0f;
        for (uint32_t c = 0; c < classes; c++) {
            y[c] = expf(x[c] - maximum);
            sum += y[c];
        }
        if (sum == 0.0f)
            return -1;
        for (uint32_t c = 0; c < classes; c++)
            y[c] = y[c] / sum;
    }
    return 0;
}

static int kern_conv1d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    const float *W = (const float *)tigris_op_weight(plan, op);
    const float *B = (const float *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    /* NLC layout (transposed from NCL) */
    int N  = x_shape[0];
    int IT = x_shape[1];   /* input length (time) */
    int IC = x_shape[2];
    int OT = y_shape[1];   /* output length (time) */
    int OC = y_shape[2];
    int K  = op->spatial.kernel_h;   /* kernel size stored in kernel_h */
    int S  = op->spatial.stride_h  ? op->spatial.stride_h  : 1;
    int D  = op->spatial.dilation_h ? op->spatial.dilation_h : 1;
    int PB = op->spatial.pad_top;    /* pad_begin stored in pad_top */

    if (mem->tile.active) {
        IT = mem->tile.in_h;
        OT = mem->tile.out_h;
        PB = mem->tile.pad_top;
    }

    /* Weight layout: [OC, K, IC] (transposed from [OC, IC, K]) */
    for (int n = 0; n < N; n++) {
        for (int ot = 0; ot < OT; ot++) {
            for (int oc = 0; oc < OC; oc++) {
                float sum = B ? B[oc] : 0.0f;
                for (int k = 0; k < K; k++) {
                    int it = ot * S - PB + k * D;
                    if (it >= 0 && it < IT) {
                        for (int ic = 0; ic < IC; ic++) {
                            sum += X[(n * IT + it) * IC + ic]
                                 * W[(oc * K + k) * IC + ic];
                        }
                    }
                }
                Y[(n * OT + ot) * OC + oc] = apply_fused_act_f32(sum, op->fused_act);
            }
        }
    }
    return 0;
}

/* Elementwise binary operations. Add and Mul commute, so a constant operand can
 * be applied from either side; Sub and Div do not, and the wire format records
 * only that an operand is constant, not which side it was on. Those two
 * therefore take two tensor operands and fail closed on the constant form. */
#define BINARY_ADD 0
#define BINARY_MUL 1
#define BINARY_SUB 2
#define BINARY_DIV 3
#define BINARY_SQUARED_DIFFERENCE 4
#define BINARY_MAXIMUM 5
#define BINARY_MINIMUM 6
#define BINARY_FLOOR_DIV 7
#define BINARY_FLOOR_MOD 8

/* Declared in tigris_binary_operands.h, shared with the int8 kernels. */
int tigris_binary_operands(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    const tigris_mem_t *mem, uint8_t dtype, tigris_binary_operands_t *out)
{
    const uint32_t element = (dtype == 1u) ? 4u : 1u;
    if ((plan == NULL) || (op == NULL) || (mem == NULL) || (out == NULL) ||
        (plan->header == NULL) || (plan->tensors == NULL) ||
        (plan->index_pool == NULL) || (plan->shape_pool == NULL) ||
        (mem->tensor_ptrs == NULL) ||
        (dtype != 1u && dtype != 3u) || op->num_outputs != 1u ||
        op->bias_idx != TIGRIS_NO_WEIGHT ||
        (op->num_inputs != 1u && op->num_inputs != 2u))
        return 0;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t y_index = tigris_op_outputs(plan, op)[0];
    const uint16_t x_index = ins[0];
    if (x_index >= plan->header->num_tensors || y_index >= plan->header->num_tensors ||
        x_index >= mem->num_tensors || y_index >= mem->num_tensors)
        return 0;
    const tigris_tensor_t *x = &plan->tensors[x_index];
    const tigris_tensor_t *y = &plan->tensors[y_index];
    if (x->dtype != dtype || y->dtype != dtype || x->ndim == 0u ||
        x->ndim != y->ndim || x->size_bytes != y->size_bytes ||
        (mem->tile.active && x->ndim != 3u && x->ndim != 4u))
        return 0;
    const int32_t *x_shape = tigris_tensor_shape(plan, x);
    const int32_t *y_shape = tigris_tensor_shape(plan, y);
    uint64_t count = 1u;
    for (uint8_t d = 0; d < x->ndim; d++) {
        if (x_shape[d] <= 0 || x_shape[d] != y_shape[d])
            return 0;
        count *= (uint64_t)x_shape[d];
    }
    const uint32_t channels = (uint32_t)x_shape[x->ndim - 1u];
    if (count * element != x->size_bytes)
        return 0;

    const uint8_t *x_data = (const uint8_t *)tigris_mem_tensor_ptr(mem, x_index);
    if ((x_data == NULL) || (tigris_mem_tensor_ptr(mem, y_index) == NULL))
        return 0;
    const tigris_quant_param_t *x_quant = tigris_tensor_quant(plan, x);

    uint8_t attr_len = 0u;
    const uint8_t *attr = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_CONSTANT_OPERAND, &attr_len);

    if (op->num_inputs == 2u) {
        const uint16_t b_index = ins[1];
        if ((attr != NULL) || (op->weight_idx != TIGRIS_NO_WEIGHT) ||
            b_index >= plan->header->num_tensors || b_index >= mem->num_tensors)
            return 0;
        const tigris_tensor_t *b = &plan->tensors[b_index];
        const int32_t *b_shape = tigris_tensor_shape(plan, b);
        uint32_t period = 0u;
        if (b->dtype != dtype || b->ndim != x->ndim)
            return 0;
        for (uint8_t d = 0; d < x->ndim; d++) {
            if (b_shape[d] != x_shape[d])
                period = channels;
        }
        if (period != 0u) {
            /* One value per channel: every axis but the innermost is 1. */
            for (uint8_t d = 0; d + 1u < x->ndim; d++) {
                if (b_shape[d] != 1)
                    return 0;
            }
            if (b_shape[x->ndim - 1u] != x_shape[x->ndim - 1u] ||
                b->size_bytes != channels * element)
                return 0;
        } else if (b->size_bytes != x->size_bytes) {
            return 0;
        }
        out->data[0] = x_data;
        out->data[1] = (const uint8_t *)tigris_mem_tensor_ptr(mem, b_index);
        out->period[0] = 0u;
        out->period[1] = period;
        out->quant[0] = x_quant;
        out->quant[1] = tigris_tensor_quant(plan, b);
        return (out->data[1] != NULL) ? 1 : 0;
    }

    if ((op->weight_idx == TIGRIS_NO_WEIGHT) || (plan->weight_entries == NULL) ||
        op->weight_idx >= plan->header->num_weights)
        return 0;
    uint8_t position = 1u;
    const tigris_quant_param_t *constant_quant = NULL;
    if (attr != NULL) {
        uint16_t quant_index;
        if (attr_len != TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN || attr[0] > 1u || attr[1] != 0u)
            return 0;
        position = attr[0];
        quant_index = (uint16_t)((uint16_t)attr[2] | (uint16_t)((uint16_t)attr[3] << 8));
        if (dtype == 3u) {
            if ((quant_index >= plan->num_quant_params) || (plan->quant_params == NULL))
                return 0;
            constant_quant = &plan->quant_params[quant_index];
        } else if (quant_index != TIGRIS_NO_QUANT_PARAM) {
            return 0;
        } else {
            constant_quant = NULL;
        }
    } else if (dtype != 1u || (op->op_type != TIGRIS_OP_ADD && op->op_type != TIGRIS_OP_MUL)) {
        return 0;
    }

    const uint32_t constant_bytes = plan->weight_entries[op->weight_idx].size_bytes;
    uint32_t period;
    if (constant_bytes == element) {
        period = 1u;
    } else if (constant_bytes == channels * element) {
        /* Channels are stored innermost, so this repeats wherever a tile starts. */
        period = channels;
    } else if (constant_bytes == x->size_bytes && !mem->tile.active) {
        /* One per element carries no tile offset, so only whole tensors. */
        period = 0u;
    } else {
        return 0;
    }
    const uint8_t *constant = (const uint8_t *)tigris_op_weight(plan, op);
    if (constant == NULL)
        return 0;
    out->data[position] = constant;
    out->period[position] = period;
    out->quant[position] = constant_quant;
    out->data[1u - position] = x_data;
    out->period[1u - position] = 0u;
    out->quant[1u - position] = x_quant;
    return 1;
}

static int kern_binary_f32(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem, int operation)
{
    tigris_binary_operands_t operands;
    if (!tigris_binary_operands(plan, op, op_index, mem, 1u, &operands))
        return -1;
    const uint16_t a_idx = tigris_op_inputs(plan, op)[0];
    float *Y = (float *)tigris_mem_tensor_ptr(mem, tigris_op_outputs(plan, op)[0]);

    uint32_t n = tile_aware_numel(plan, a_idx, mem);
    if (mem->tile.active) {
        uint32_t row_elems = tile_row_elems(plan, a_idx, mem);
        for (uint8_t k = 0; k < 2u; k++) {
            if (operands.period[k] == 0u)
                operands.data[k] += (size_t)mem->tile.in_row_start * row_elems * sizeof(float);
        }
        Y += (size_t)mem->tile.out_row_start * row_elems;
    }
    for (uint32_t i = 0; i < n; i++) {
        float a = load_weight_f32(operands.data[0],
                                  operands.period[0] ? i % operands.period[0] : i);
        float b = load_weight_f32(operands.data[1],
                                  operands.period[1] ? i % operands.period[1] : i);
        float v;
        switch (operation) {
        case BINARY_MUL: v = a * b; break;
        case BINARY_SUB: v = a - b; break;
        case BINARY_DIV: v = a / b; break;
        case BINARY_SQUARED_DIFFERENCE: v = a - b; v *= v; break;
        case BINARY_MAXIMUM: v = a < b ? b : a; break;
        case BINARY_MINIMUM: v = b < a ? b : a; break;
        case BINARY_FLOOR_DIV:
            if (b == 0.0f) return -1;
            v = floorf(a / b);
            break;
        case BINARY_FLOOR_MOD:
            v = fmodf(a, b);
            if (v != 0.0f && ((b < 0.0f) != (v < 0.0f))) v += b;
            break;
        default:         v = a + b; break;
        }
        Y[i] = apply_fused_act_f32(v, op->fused_act);
    }
    return 0;
}

static int kern_mul(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    return kern_binary_f32(plan, op, op_index, mem, BINARY_MUL);
}

static int kern_add(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    return kern_binary_f32(plan, op, op_index, mem, BINARY_ADD);
}

static int kern_sub(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    return kern_binary_f32(plan, op, op_index, mem, BINARY_SUB);
}

static int kern_global_avg_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

    /* NHWC layout */
    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    int N  = x_shape[0];
    int H  = x_shape[1];
    int W  = x_shape[2];
    int C  = x_shape[3];
    int HW = H * W;

    /* A band of rows while the executor walks the input, the whole tensor
     * otherwise. The divisor stays the full HW either way, and the band sums
     * add into the accumulator in the same order as the single-pass loop, so
     * both produce the same float. */
    float *acc = (float *)mem->tile.reduce_acc;
    int rows = (mem->tile.active && mem->tile.in_h > 0) ? mem->tile.in_h : H;

    /* Output: [N, 1, 1, C] in NHWC */
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = (acc && !mem->tile.reduce_first) ? acc[n * C + c] : 0.0f;
            for (int h = 0; h < rows; h++) {
                for (int w = 0; w < W; w++) {
                    sum += X[((n * rows + h) * W + w) * C + c];
                }
            }
            if (acc) {
                acc[n * C + c] = sum;
                if (!mem->tile.reduce_last)
                    continue;
            }
            Y[n * C + c] = sum / (float)HW;
        }
    }
    return 0;
}

/**
 * Mean over one axis of a rank-3 tensor.
 *
 * The plan names a serialized axis, so the operator reads the same whichever
 * order a tensor states its own axes in: the token axis of a sequence and the
 * channel axis of a feature map are both just one axis of [outer, reduced,
 * inner]. Collapsing the token axis is what a mean-pooled sequence head does.
 */
static int reduce_mean_extents(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    int32_t *outer, int32_t *reduced, int32_t *inner)
{
    const tigris_tensor_t *in = &plan->tensors[tigris_op_inputs(plan, op)[0]];
    uint8_t num_axes = 0u;
    const uint8_t *axes = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_AXES, &num_axes);
    if (axes == NULL || num_axes != 1u || in->ndim != 3u || axes[0] >= 3u)
        return -1;

    const int32_t *shape = tigris_tensor_shape(plan, in);
    int32_t lead = 1;
    int32_t trail = 1;
    for (uint8_t axis = 0; axis < 3u; axis++) {
        if (shape[axis] <= 0)
            return -1;
        if (axis < axes[0])
            lead *= shape[axis];
        else if (axis > axes[0])
            trail *= shape[axis];
    }
    *outer = lead;
    *reduced = shape[axes[0]];
    *inner = trail;
    return 0;
}

static int kern_reduce_mean(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem)
{
    int32_t outer;
    int32_t reduced;
    int32_t inner;
    if (reduce_mean_extents(plan, op, op_index, &outer, &reduced, &inner) != 0)
        return -1;

    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    for (int32_t o = 0; o < outer; o++) {
        const float *src = X + (size_t)o * (size_t)reduced * (size_t)inner;
        float *dst = Y + (size_t)o * (size_t)inner;
        for (int32_t i = 0; i < inner; i++)
            dst[i] = 0.0f;
        for (int32_t r = 0; r < reduced; r++) {
            const float *row = src + (size_t)r * (size_t)inner;
            for (int32_t i = 0; i < inner; i++)
                dst[i] += row[i];
        }
        for (int32_t i = 0; i < inner; i++)
            dst[i] /= (float)reduced;
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
static int kern_split(
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

static int kern_fully_connected(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    const float *W = (const float *)tigris_op_weight(plan, op);
    const float *B = (const float *)tigris_op_bias(plan, op);

    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];
    const int32_t *y_shape = tigris_tensor_shape(plan, y_tensor);
    /* Gemm: Y = X * W^T + B. Input [N, IC], weight [OC, IC], output [N, OC].
     * N is 1 for a classifier head and the row count for a linear layer
     * applied per sequence position, which is the same shape the int8 kernel
     * has always accepted. */
    int N, OC;
    if (y_tensor->ndim >= 2) {
        N  = y_shape[0];
        OC = y_shape[1];
    } else {
        N  = 1;
        OC = y_shape[y_tensor->ndim - 1];
    }
    /* A row band computes its own rows against the whole weight, so only the
     * row count changes; the loaded operand already starts at the band. */
    if (mem->tile.active && mem->tile.row_tiled)
        N = (int)mem->tile.out_h;
    /* IC from weight: total weight elements / OC */
    const tigris_weight_entry_t *we = &plan->weight_entries[op->weight_idx];
    int IC = (int)(we->size_bytes / sizeof(float)) / OC;

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float sum = B ? B[oc] : 0.0f;
            for (int ic = 0; ic < IC; ic++)
                sum += W[oc * IC + ic] * X[n * IC + ic];
            Y[n * OC + oc] = apply_fused_act_f32(sum, op->fused_act);
        }
    }
    return 0;
}

/* Y = A * B with both operands activations. Axes are the model's own: the
 * trailing pair is the matrix, everything before it batches. The loader has
 * already checked that the batch extents agree and that the inner dimensions
 * meet, so this only walks them. MatMul is not in the loader's fused-activation
 * allow-list, so there is no activation to apply here. */
static int kern_matmul(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *A = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    const float *B = (const float *)tigris_mem_tensor_ptr(mem, ins[1]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

    const tigris_tensor_t *a_tensor = &plan->tensors[ins[0]];
    const tigris_tensor_t *b_tensor = &plan->tensors[ins[1]];
    const int32_t *a_shape = tigris_tensor_shape(plan, a_tensor);
    const int32_t *b_shape = tigris_tensor_shape(plan, b_tensor);

    uint8_t last = (uint8_t)(a_tensor->ndim - 1u);
    int M = a_shape[last - 1u];
    int K = a_shape[last];
    int N = b_shape[last];

    /* A row band hands this kernel a band of the first operand's rows and the
     * whole of the second, and says how many rows in tile.out_h rather than in
     * the shape. The batch count and both inner extents are unchanged: a band
     * cuts the second to last axis and nothing else. */
    if (mem->tile.active && mem->tile.row_tiled && mem->tile.out_h > 0)
        M = mem->tile.out_h;

    int batches = 1;
    for (uint8_t axis = 0; axis + 2u <= last; axis++)
        batches *= a_shape[axis];

    for (int b = 0; b < batches; b++) {
        const float *a = A + (size_t)b * (size_t)M * (size_t)K;
        const float *w = B + (size_t)b * (size_t)K * (size_t)N;
        float *y = Y + (size_t)b * (size_t)M * (size_t)N;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += a[m * K + k] * w[k * N + n];
                y[m * N + n] = sum;
            }
        }
    }
    return 0;
}

/* GlobalMaxPool: the maximum over every spatial position, per channel. The
 * counterpart of GlobalAveragePool, and like MaxPool it needs no requantization
 * because a maximum is one of the input values. */
static int kern_global_max_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    int N = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    int C = x_shape[3];

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float best = X[(n * H * W) * C + c];
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float v = X[((n * H + h) * W + w) * C + c];
                    if (v > best) best = v;
                }
            }
            Y[n * C + c] = best;
        }
    }
    return 0;
}

static int kern_reshape(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const void *X = tigris_mem_tensor_ptr(mem, ins[0]);
    void       *Y = tigris_mem_tensor_ptr(mem, outs[0]);
    /* A reshape moves no data, so under a tile it moves exactly the tile: the
     * element count the tile holds, not the tensor's. The int8 twin has
     * always done this; the float one copied the whole tensor, which a row
     * band would write past the end of. */
    const tigris_tensor_t *in = &plan->tensors[ins[0]];
    uint32_t elems = tensor_numel(plan, ins[0]);
    uint32_t sz = in->size_bytes;
    if (elems != 0u)
        sz = (in->size_bytes / elems) * tile_aware_numel(plan, ins[0], mem);
    if (X != Y)
        memcpy(Y, X, sz);
    return 0;
}

static int kern_max_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

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
                    float max_val = -FLT_MAX;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw;
                            float v = X[((n * IH + ih) * IW + iw) * C + c];
                            if (v > max_val) max_val = v;
                        }
                    }
                    Y[((n * OH + oh_g) * OW + ow) * C + c] = max_val;
                }
            }
        }
    }
    return 0;
}

/* ONNX AveragePool with the representable default: exclude padded samples
 * from the divisor. The tiled executor supplies tile-local height and padding
 * so the same bounds checks are valid for full and tiled tensors. */
static int kern_avg_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

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

    if (!X || !Y || KH <= 0 || KW <= 0)
        return -1;

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
                /* The window's sample count is the size of the two ranges,
                 * the same number the per-tap counter arrived at. */
                int count = (kh1 - kh0) * (kw1 - kw0);
                if (count == 0)
                    return -1;
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    for (int kh = kh0; kh < kh1; kh++) {
                        int ih = base_h + kh;
                        for (int kw = kw0; kw < kw1; kw++) {
                            int iw = base_w + kw;
                            sum += X[((n * IH + ih) * IW + iw) * C + c];
                        }
                    }
                    Y[((n * OH + oh_g) * OW + ow) * C + c] = sum / (float)count;
                }
            }
        }
    }
    return 0;
}

static int kern_concat(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    float *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    /* Concat along one stored axis: each input contributes one contiguous
     * run per position, a run being its extent along the axis times every
     * axis after it, and the positions are the product of the axes ahead of
     * it. Along the last axis, an image's channels or a sequence's features,
     * the run is the input's own extent and a tile only shortens the axes
     * ahead; any other axis is cut whole and never tiled. */
    uint8_t rank = plan->tensors[outs[0]].ndim;
    uint8_t axis = (rank == 4u && op->spatial.kernel_h == 0)
                 ? 3u : (uint8_t)op->spatial.kernel_h;
    int positions = 1;
    int inner = 1;
    int out_c_offset = 0;

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

    /* A constant leading part, which is how a learned token is prepended to a
     * sequence. The plan names it as the operator's weight and the loader has
     * already checked that its bytes are the ones the inputs leave over, so
     * its extent is whatever they do not cover. */
    if (op->weight_idx != TIGRIS_NO_WEIGHT) {
        const float *K = (const float *)tigris_op_weight(plan, op);
        int leading = total_OC;
        if (K == NULL || mem->tile.active)
            return -1;
        for (int i = 0; i < op->num_inputs; i++) {
            const tigris_tensor_t *part = &plan->tensors[ins[i]];
            leading -= tigris_tensor_shape(plan, part)[axis] * inner;
        }
        if (leading <= 0)
            return -1;
        for (int q = 0; q < positions; q++) {
            memcpy(Y + (size_t)q * total_OC, K + (size_t)q * leading,
                   (size_t)leading * sizeof(float));
        }
        out_c_offset = leading;
    }

    for (int i = 0; i < op->num_inputs; i++) {
        const float *Xi = (const float *)tigris_mem_tensor_ptr(mem, ins[i]);
        const int32_t *xi_shape = tigris_tensor_shape(plan, &plan->tensors[ins[i]]);
        int Ci = xi_shape[axis] * inner;

        if (Xi == NULL)
            return -1;
        for (int q = 0; q < positions; q++) {
            memcpy(Y + (size_t)q * total_OC + out_c_offset,
                   Xi + (size_t)q * Ci, (size_t)Ci * sizeof(float));
        }
        out_c_offset += Ci;
    }
    return 0;
}

/**
 * Bilinear upsample by an integer factor, half-pixel coordinates.
 *
 * ONNX Resize with mode "linear" mixes the two rows and two columns around
 * the source coordinate. Which coordinate that is depends on the convention
 * the graph was written in, and kernel_h says which: half-pixel places an
 * output sample at (o + 0.5) / scale - 0.5, and asymmetric, which is what
 * every opset before 11 meant, places it at o / scale. Coordinates outside
 * the input are clamped at the border either way.
 */
static int kern_resize_linear(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];
    float scale_h = (float)(op->spatial.stride_h ? op->spatial.stride_h : 1);
    float scale_w = (float)(op->spatial.stride_w ? op->spatial.stride_w : 1);
    float shift = (op->spatial.kernel_h == 1) ? 0.0f : 0.5f;
    int out_origin = 0;
    int in_origin = 0;

    if (X == NULL || Y == NULL || op->spatial.kernel_h > 1)
        return -1;
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        out_origin = mem->tile.out_row_origin;
        in_origin = mem->tile.in_row_origin;
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            float fy = ((float)(oh + out_origin) + shift) / scale_h - shift;
            int y0, y1;
            float wy;
            if (fy < 0.0f) fy = 0.0f;
            y0 = (int)fy - in_origin;
            if (y0 < 0) y0 = 0;
            if (y0 > IH - 1) y0 = IH - 1;
            y1 = (y0 + 1 < IH) ? y0 + 1 : IH - 1;
            wy = fy - (float)(y0 + in_origin);
            for (int ow = 0; ow < OW; ow++) {
                float fx = ((float)ow + shift) / scale_w - shift;
                int x0, x1;
                float wx;
                const float *r0;
                const float *r1;
                float *dst;
                if (fx < 0.0f) fx = 0.0f;
                x0 = (int)fx;
                if (x0 > IW - 1) x0 = IW - 1;
                x1 = (x0 + 1 < IW) ? x0 + 1 : IW - 1;
                wx = fx - (float)x0;
                r0 = X + ((size_t)(n * IH + y0) * (size_t)IW) * (size_t)C;
                r1 = X + ((size_t)(n * IH + y1) * (size_t)IW) * (size_t)C;
                dst = Y + ((size_t)(n * OH + oh) * (size_t)OW + (size_t)ow) *
                      (size_t)C;
                for (int c = 0; c < C; c++) {
                    float top = r0[(size_t)x0 * C + c] * (1.0f - wx) +
                                r0[(size_t)x1 * C + c] * wx;
                    float bot = r1[(size_t)x0 * C + c] * (1.0f - wx) +
                                r1[(size_t)x1 * C + c] * wx;
                    dst[c] = top * (1.0f - wy) + bot * wy;
                }
            }
        }
    }
    return 0;
}

static int kern_resize_nearest(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float       *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int N  = x_shape[0];
    int IH = x_shape[1];
    int IW = x_shape[2];
    int C  = x_shape[3];
    int OH = y_shape[1];
    int OW = y_shape[2];

    /* Integer scale factors stored in stride_h/w by the compiler */
    int scale_h = op->spatial.stride_h ? op->spatial.stride_h : 1;
    int scale_w = op->spatial.stride_w ? op->spatial.stride_w : 1;

    /* Under a tile the buffers hold a band, and which source row an output
     * row takes is a function of its global index, not of its position in
     * the band. The executor publishes where the band starts. */
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

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih = (oh + out_origin) / scale_h - in_origin;
            if (ih >= IH) ih = IH - 1;
            if (ih < 0) ih = 0;
            for (int ow = 0; ow < OW; ow++) {
                int iw = ow / scale_w;
                if (iw >= IW) iw = IW - 1;
                const float *src = X + ((n * IH + ih) * IW + iw) * C;
                float *dst = Y + ((n * OH + oh) * OW + ow) * C;
                memcpy(dst, src, (size_t)C * sizeof(float));
            }
        }
    }
    return 0;
}

/* The permutation walks below move one element at a time. Handing each of
 * those to memcpy costs a call plus the library's own size descent before a
 * single byte moves: 27 instructions per element on Cortex-M4 against 3 for a
 * typed store. Every dtype the runtime carries is 1 or 4 bytes wide, and a
 * tensor is aligned to at least TIGRIS_TENSOR_ALIGN, so the typed forms cover
 * every plan in practice and memcpy stays for anything else. */
/* Ranks above this fall back to the general index walk rather than carry a
 * larger odometer. Every shape the compiler emits is rank 4 or less. */
#define TIGRIS_TRANSPOSE_MAX_RANK 8

static int transpose_elements_are_typed(
    const void *dst, const void *src, uint32_t element_size)
{
    uintptr_t bits = (uintptr_t)dst | (uintptr_t)src;
    if (element_size == 1u)
        return 1;
    if (element_size != 4u)
        return 0;
    return (bits & (uintptr_t)3) == 0u;
}

/* One packed [rows, cols] matrix transposed into [cols, rows]. `typed` is
 * what transpose_elements_are_typed said about this pair of pointers. */
static void transpose_matrix(
    uint8_t *dst, const uint8_t *src,
    int32_t rows, int32_t cols, uint32_t element_size, int typed)
{
    if (typed && element_size == 4u) {
        const uint32_t *s32 = (const uint32_t *)(const void *)src;
        uint32_t *d32 = (uint32_t *)(void *)dst;
        for (int32_t r = 0; r < rows; r++) {
            const uint32_t *row = s32 + (size_t)r * (size_t)cols;
            uint32_t *col = d32 + (size_t)r;
            for (int32_t c = 0; c < cols; c++)
                col[(size_t)c * (size_t)rows] = row[c];
        }
    } else if (typed && element_size == 1u) {
        for (int32_t r = 0; r < rows; r++) {
            const uint8_t *row = src + (size_t)r * (size_t)cols;
            uint8_t *col = dst + (size_t)r;
            for (int32_t c = 0; c < cols; c++)
                col[(size_t)c * (size_t)rows] = row[c];
        }
    } else {
        for (int32_t r = 0; r < rows; r++) {
            for (int32_t c = 0; c < cols; c++) {
                memcpy(dst + ((size_t)c * (size_t)rows + (size_t)r) *
                                 element_size,
                       src + ((size_t)r * (size_t)cols + (size_t)c) *
                                 element_size,
                       element_size);
            }
        }
    }
}

int tigris_transpose_execute(
    const tigris_plan_t *plan, const tigris_op_t *op,
    uint16_t op_index, tigris_mem_t *mem)
{
    if (!plan || !op || !mem || op->num_inputs != 1 || op->num_outputs != 1)
        return -1;
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    const tigris_tensor_t *input = &plan->tensors[ins[0]];
    const tigris_tensor_t *output = &plan->tensors[outs[0]];
    uint8_t rank = 0;
    const uint8_t *perm = tigris_op_attribute_data(
        plan, op_index, TIGRIS_OP_ATTR_TRANSPOSE_PERM, &rank);
    if (!perm || rank != input->ndim || output->ndim != rank ||
        input->size_bytes != output->size_bytes)
        return -1;
    const uint8_t *src = (const uint8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    uint8_t *dst = (uint8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!src || !dst)
        return -1;
    const int32_t *input_shape = tigris_tensor_shape(plan, input);
    const int32_t *output_shape = tigris_tensor_shape(plan, output);
    uint32_t elements = 1;
    for (uint8_t axis = 0; axis < rank; axis++) {
        if (output_shape[axis] <= 0 || perm[axis] >= rank)
            return -1;
        elements *= (uint32_t)output_shape[axis];
    }
    if (elements == 0 || input->size_bytes % elements != 0)
        return -1;
    uint32_t element_size = input->size_bytes / elements;

    /* A conversion stage hands this kernel a packed [batch, in_h, in_w] slice
     * of the input and expects a packed [batch, out_h, out_w] slice of the
     * output. Whatever the tensor's rank, the permutation a conversion
     * performs is a plain matrix transpose once the axes that keep their
     * relative order are read as one, so the slice is transposed directly
     * rather than through the general index walk below. */
    if (mem->tile.active && mem->tile.transposed_tile) {
        int32_t batch = mem->tile.transpose_outer;
        int32_t rows = mem->tile.in_h;
        int32_t cols = mem->tile.in_w;
        if (batch <= 0 || rows <= 0 || cols <= 0 ||
            mem->tile.out_h != cols || mem->tile.out_w != rows)
            return -1;
        /* The axes behind the swapped pair travel with the element, so the
         * slice is a matrix of blocks rather than of scalars. The executor
         * states how many elements a position holds, because the slice is a
         * band and the tensor's own shape describes the whole. */
        int32_t block = mem->tile.transpose_block > 0
                            ? mem->tile.transpose_block : 1;
        uint32_t block_size = element_size * (uint32_t)block;
        int typed = transpose_elements_are_typed(dst, src, block_size);
        for (int32_t n = 0; n < batch; n++) {
            const uint8_t *src_n =
                src + (size_t)n * (size_t)rows * (size_t)cols * block_size;
            uint8_t *dst_n =
                dst + (size_t)n * (size_t)rows * (size_t)cols * block_size;
            transpose_matrix(dst_n, src_n, rows, cols, block_size, typed);
        }
        return 0;
    }

    /* The input index the next output element reads is the current one plus a
     * per-axis step, so the walk carries an odometer instead of rebuilding the
     * whole index. What it replaces is a divide and a modulo per axis per
     * element, plus an inner loop that rebuilt each axis stride from scratch,
     * which made the old walk quadratic in the rank. Measured on a rank-4
     * [1,64,64,16] float transpose: 157 instructions per element before, 16.3
     * after, bit-identical over all 65,536. */
    if (rank <= TIGRIS_TRANSPOSE_MAX_RANK) {
        uint32_t input_stride[TIGRIS_TRANSPOSE_MAX_RANK];
        uint32_t step[TIGRIS_TRANSPOSE_MAX_RANK];
        int32_t coordinate[TIGRIS_TRANSPOSE_MAX_RANK];
        uint32_t stride = 1;
        for (uint8_t reverse_axis = rank; reverse_axis > 0u; reverse_axis--) {
            uint8_t axis = reverse_axis - 1u;
            input_stride[axis] = stride;
            stride *= (uint32_t)input_shape[axis];
        }
        for (uint8_t axis = 0; axis < rank; axis++) {
            step[axis] = input_stride[perm[axis]];
            coordinate[axis] = 0;
        }

        int typed = transpose_elements_are_typed(dst, src, element_size);
        uint8_t last = rank - 1u;
        uint32_t input_index = 0;
        for (uint32_t output_index = 0; output_index < elements;
             output_index++) {
            if (typed && element_size == 4u) {
                ((uint32_t *)(void *)dst)[output_index] =
                    ((const uint32_t *)(const void *)src)[input_index];
            } else if (typed && element_size == 1u) {
                dst[output_index] = src[input_index];
            } else {
                memcpy(dst + (size_t)output_index * element_size,
                       src + (size_t)input_index * element_size, element_size);
            }
            /* The last axis carries for one element in output_shape[last],
             * so it is worth stepping without entering the carry loop. */
            coordinate[last]++;
            input_index += step[last];
            if (coordinate[last] >= output_shape[last]) {
                input_index -= step[last] * (uint32_t)output_shape[last];
                coordinate[last] = 0;
                for (uint8_t reverse_axis = last; reverse_axis > 0u;
                     reverse_axis--) {
                    uint8_t axis = reverse_axis - 1u;
                    coordinate[axis]++;
                    input_index += step[axis];
                    if (coordinate[axis] < output_shape[axis])
                        break;
                    input_index -= step[axis] * (uint32_t)output_shape[axis];
                    coordinate[axis] = 0;
                }
            }
        }
        return 0;
    }

    for (uint32_t output_index = 0; output_index < elements; output_index++) {
        uint32_t remainder = output_index;
        uint32_t input_index = 0;
        for (uint8_t reverse_axis = rank; reverse_axis > 0; reverse_axis--) {
            uint8_t output_axis = reverse_axis - 1u;
            uint32_t coordinate = remainder % (uint32_t)output_shape[output_axis];
            remainder /= (uint32_t)output_shape[output_axis];
            uint32_t stride = 1;
            for (uint8_t axis = perm[output_axis] + 1u; axis < rank; axis++)
                stride *= (uint32_t)input_shape[axis];
            input_index += coordinate * stride;
        }
        memcpy(dst + (size_t)output_index * element_size,
               src + (size_t)input_index * element_size, element_size);
    }
    return 0;
}

/* Dispatch */

int tigris_dispatch_kernel(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    (void)user_ctx;

#ifdef TIGRIS_COUNT_KERNEL_ROWS
    if (mem->tile.active)
        g_tigris_kernel_rows += (unsigned long)mem->tile.out_h;
#endif

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return kern_conv2d(plan, op, mem);
    case TIGRIS_OP_CONV_TRANSPOSE: return kern_conv_transpose(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return kern_depthwise_conv2d(plan, op, mem);
    case TIGRIS_OP_RELU:        return kern_relu(plan, op, mem);
    case TIGRIS_OP_RELU6:       return kern_relu6(plan, op, mem);
    case TIGRIS_OP_SIGMOID:     return kern_sigmoid(plan, op, mem);
    case TIGRIS_OP_TANH:        return kern_tanh(plan, op, mem);
    case TIGRIS_OP_SOFTMAX:     return kern_softmax(plan, op, mem);
    case TIGRIS_OP_ADD:         return kern_add(plan, op, op_index, mem);
    case TIGRIS_OP_MUL:         return kern_mul(plan, op, op_index, mem);
    case TIGRIS_OP_CONV1D:      return kern_conv1d(plan, op, mem);
    case TIGRIS_OP_GLOBAL_AVG:  return kern_global_avg_pool(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return kern_fully_connected(plan, op, mem);
    case TIGRIS_OP_RESHAPE:     return kern_reshape(plan, op, mem);
    case TIGRIS_OP_FLATTEN:     return kern_reshape(plan, op, mem);
    case TIGRIS_OP_MAX_POOL:    return kern_max_pool(plan, op, mem);
    case TIGRIS_OP_AVG_POOL:    return kern_avg_pool(plan, op, mem);
    case TIGRIS_OP_CONCAT:      return kern_concat(plan, op, mem);
    case TIGRIS_OP_RESIZE:      return kern_resize_nearest(plan, op, mem);
    case TIGRIS_OP_RESIZE_LINEAR:
                                return kern_resize_linear(plan, op, mem);
    case TIGRIS_OP_SUB:         return kern_sub(plan, op, op_index, mem);
    case TIGRIS_OP_GLOBAL_MAX: return kern_global_max_pool(plan, op, mem);
    case TIGRIS_OP_MATMUL:      return kern_matmul(plan, op, mem);
    case TIGRIS_OP_TRANSPOSE:   return tigris_transpose_execute(plan, op, op_index, mem);
    case TIGRIS_OP_ERF:         return kern_erf(plan, op, mem);
    case TIGRIS_OP_ABS: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_RSQRT: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_NEG: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_EXP: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_LOG: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_SQRT: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_SQUARE: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_FLOOR: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_CEIL: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_ROUND: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_SIN: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_COS: return kern_elementwise_unary_f32(plan, op, mem);
    case TIGRIS_OP_DIV: return kern_binary_f32(plan, op, op_index, mem, BINARY_DIV);
    case TIGRIS_OP_SQUARED_DIFFERENCE: return kern_binary_f32(plan, op, op_index, mem, BINARY_SQUARED_DIFFERENCE);
    case TIGRIS_OP_MAXIMUM: return kern_binary_f32(plan, op, op_index, mem, BINARY_MAXIMUM);
    case TIGRIS_OP_MINIMUM: return kern_binary_f32(plan, op, op_index, mem, BINARY_MINIMUM);
    case TIGRIS_OP_FLOOR_DIV: return kern_binary_f32(plan, op, op_index, mem, BINARY_FLOOR_DIV);
    case TIGRIS_OP_FLOOR_MOD: return kern_binary_f32(plan, op, op_index, mem, BINARY_FLOOR_MOD);
    case TIGRIS_OP_HARDSWISH:   return kern_hardswish(plan, op, mem);
    case TIGRIS_OP_LAYER_NORM:  return kern_layer_norm(plan, op, op_index, mem);
    case TIGRIS_OP_REDUCE_MEAN: return kern_reduce_mean(plan, op, op_index, mem);
    case TIGRIS_OP_SPLIT:       return kern_split(plan, op, mem);
    default:
        return -1;  /* unsupported op type */
    }
}

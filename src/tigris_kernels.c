/**
 * @file tigris_kernels.c
 * @brief Reference float32 kernels + dispatch.
 *
 * All kernels are static. Data layout: NHWC, float32.
 * Weight/bias accessed via tigris_op_weight() / tigris_op_bias().
 */

#include "tigris_kernels.h"

#include <float.h>
#include <math.h>
#include <string.h>

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
    /* Serialized activations are NHWC or NLC. */
    uint32_t width = t->ndim == 4 ? (uint32_t)mem->tile.out_w : 1u;
    return (uint32_t)shape[0] * (uint32_t)mem->tile.out_h *
           width * (uint32_t)shape[t->ndim - 1];
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

    /* Tile override */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    /* Weight layout: [OC, KH, KW, IC] (OHWI) */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int oc = 0; oc < OC; oc++) {
                    float sum = B ? B[oc] : 0.0f;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PT + kh * DH;
                            int iw = ow * SW - PL + kw * DW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                for (int ic = 0; ic < IC; ic++) {
                                    float x_val = X[((n * IH + ih) * IW + iw) * IC + ic];
                                    float w_val = W[((oc * KH + kh) * KW + kw) * IC + ic];
                                    sum += x_val * w_val;
                                }
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

    /* Tile override */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    /* Depthwise: group == C, each channel convolved independently.
     * Weight layout: [KH, KW, C] (HWC) */
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int c = 0; c < C; c++) {
                    float sum = B ? B[c] : 0.0f;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PT + kh * DH;
                            int iw = ow * SW - PL + kw * DW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                float x_val = X[((n * IH + ih) * IW + iw) * C + c];
                                float w_val = W[(kh * KW + kw) * C + c];
                                sum += x_val * w_val;
                            }
                        }
                    }
                    Y[((n * OH + oh) * OW + ow) * C + c] = apply_fused_act_f32(sum, op->fused_act);
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
    for (uint32_t i = 0; i < n; i++)
        Y[i] = 1.0f / (1.0f + expf(-X[i]));
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
    for (uint32_t i = 0; i < n; i++)
        Y[i] = tanhf(X[i]);
    return 0;
}

/* Softmax over the final dimension.  Supported plans use ONNX's default
 * final-axis form; Softmax is deliberately not tileable. */
static int kern_softmax(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    if (!plan || !op || !mem || op->num_inputs != 1 || op->num_outputs != 1 ||
        mem->tile.active)
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
    const uint32_t count = tensor_numel(plan, ins[0]);
    if (classes == 0 || count % classes != 0)
        return -1;

    const float *X = (const float *)tigris_mem_tensor_ptr(mem, ins[0]);
    float *Y = (float *)tigris_mem_tensor_ptr(mem, outs[0]);
    if (!X || !Y)
        return -1;

    for (uint32_t row = 0; row < count / classes; row++) {
        const float *x = X + (size_t)row * classes;
        float *y = Y + (size_t)row * classes;
        float maximum = x[0];
        for (uint32_t c = 1; c < classes; c++)
            if (x[c] > maximum) maximum = x[c];

        float sum = 0.0f;
        for (uint32_t c = 0; c < classes; c++)
            sum += expf(x[c] - maximum);
        if (sum == 0.0f)
            return -1;
        for (uint32_t c = 0; c < classes; c++)
            y[c] = expf(x[c] - maximum) / sum;
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

static int kern_binary_f32(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem,
    int multiply)
{
    if (!plan || !op || !mem || !plan->header || !plan->tensors ||
        !plan->index_pool || !plan->shape_pool || !mem->tensor_ptrs ||
        op->num_outputs != 1 ||
        (op->num_inputs != 1 && op->num_inputs != 2))
        return -1;

    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    uint16_t a_idx = ins[0];
    uint16_t y_idx = outs[0];
    if (a_idx >= plan->header->num_tensors ||
        y_idx >= plan->header->num_tensors ||
        a_idx >= mem->num_tensors || y_idx >= mem->num_tensors)
        return -1;

    const tigris_tensor_t *a_tensor = &plan->tensors[a_idx];
    const tigris_tensor_t *y_tensor = &plan->tensors[y_idx];
    if (!tensor_is_f32(plan, a_tensor) ||
        !tensor_is_f32(plan, y_tensor) ||
        !tensor_shapes_equal(plan, a_tensor, y_tensor))
        return -1;
    if (mem->tile.active && a_tensor->ndim != 3 && a_tensor->ndim != 4)
        return -1;

    const float *A = (const float *)tigris_mem_tensor_ptr(mem, a_idx);
    float *Y = (float *)tigris_mem_tensor_ptr(mem, y_idx);
    if (!A || !Y)
        return -1;

    const float *dynamic_b = NULL;
    const uint8_t *constant_b = NULL;
    uint32_t constant_count = 0;

    if (op->num_inputs == 2) {
        /* Dynamic elementwise Add/Mul supports exact-shape operands only. */
        if (op->weight_idx != TIGRIS_NO_WEIGHT ||
            op->bias_idx != TIGRIS_NO_WEIGHT)
            return -1;
        uint16_t b_idx = ins[1];
        if (b_idx >= plan->header->num_tensors || b_idx >= mem->num_tensors)
            return -1;
        const tigris_tensor_t *b_tensor = &plan->tensors[b_idx];
        if (!tensor_is_f32(plan, b_tensor) ||
            !tensor_shapes_equal(plan, a_tensor, b_tensor))
            return -1;
        dynamic_b = (const float *)tigris_mem_tensor_ptr(mem, b_idx);
        if (!dynamic_b)
            return -1;
    } else {
        /* The compiler omits constants from the tensor input list and stores
         * the sole constant operand in weight_idx. The wire format has only a
         * byte size, so scalar broadcast and exact-size elementwise constants
         * are the only unambiguous cases. */
        if (op->weight_idx == TIGRIS_NO_WEIGHT ||
            op->weight_idx >= plan->header->num_weights ||
            op->bias_idx != TIGRIS_NO_WEIGHT || !plan->weight_entries ||
            !plan->weight_blob)
            return -1;

        const tigris_weight_entry_t *weight =
            &plan->weight_entries[op->weight_idx];
        if (weight->size_bytes == sizeof(float)) {
            constant_count = 1;
        } else if (weight->size_bytes == a_tensor->size_bytes &&
                   !mem->tile.active) {
            /* A full constant has no shape/tile-offset metadata. It is safe
             * only for non-tiled exact-shape execution. */
            constant_count = a_tensor->size_bytes / sizeof(float);
        } else {
            return -1;
        }
        constant_b = (const uint8_t *)tigris_op_weight(plan, op);
        if (!constant_b)
            return -1;
    }

    uint32_t n = tile_aware_numel(plan, a_idx, mem);
    for (uint32_t i = 0; i < n; i++) {
        float b = dynamic_b
            ? dynamic_b[i]
            : load_weight_f32(constant_b, constant_count == 1 ? 0 : i);
        Y[i] = multiply ? A[i] * b : A[i] + b;
    }
    return 0;
}

static int kern_mul(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    return kern_binary_f32(plan, op, mem, 1);
}

static int kern_add(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    return kern_binary_f32(plan, op, mem, 0);
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

    /* Output: [N, 1, 1, C] in NHWC */
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    sum += X[((n * H + h) * W + w) * C + c];
                }
            }
            Y[n * C + c] = sum / (float)HW;
        }
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
    /* Gemm: Y = X * W^T + B. OC is the last dim of output. */
    int OC = y_shape[y_tensor->ndim - 1];
    /* IC from weight: total weight elements / OC */
    const tigris_weight_entry_t *we = &plan->weight_entries[op->weight_idx];
    int IC = (int)(we->size_bytes / sizeof(float)) / OC;

    for (int oc = 0; oc < OC; oc++) {
        float sum = B ? B[oc] : 0.0f;
        for (int ic = 0; ic < IC; ic++)
            sum += W[oc * IC + ic] * X[ic];
        Y[oc] = apply_fused_act_f32(sum, op->fused_act);
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
    uint32_t sz = plan->tensors[ins[0]].size_bytes;
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

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int c = 0; c < C; c++) {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PT + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PL + kw;
                            if (iw < 0 || iw >= IW) continue;
                            float v = X[((n * IH + ih) * IW + iw) * C + c];
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

    if (!X || !Y || KH <= 0 || KW <= 0)
        return -1;

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
        PT = mem->tile.pad_top;
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PT + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PL + kw;
                            if (iw < 0 || iw >= IW) continue;
                            sum += X[((n * IH + ih) * IW + iw) * C + c];
                            count++;
                        }
                    }
                    if (count == 0)
                        return -1;
                    Y[((n * OH + oh) * OW + ow) * C + c] = sum / (float)count;
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

    int N  = y_shape[0];
    int H  = y_shape[1];
    int W  = y_shape[2];

    if (mem->tile.active) {
        H = mem->tile.out_h;
        W = mem->tile.out_w;
    }

    /* Channel-axis concat (axis=3 in NHWC).
     * For each spatial position, memcpy each input's channel slice. */
    int out_c_offset = 0;
    int total_OC = y_shape[3];

    for (int i = 0; i < op->num_inputs; i++) {
        const float *Xi = (const float *)tigris_mem_tensor_ptr(mem, ins[i]);
        const int32_t *xi_shape = tigris_tensor_shape(plan, &plan->tensors[ins[i]]);
        int Ci = xi_shape[3];

        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    const float *src = Xi + ((n * H + h) * W + w) * Ci;
                    float *dst = Y + ((n * H + h) * W + w) * total_OC + out_c_offset;
                    memcpy(dst, src, (size_t)Ci * sizeof(float));
                }
            }
        }
        out_c_offset += Ci;
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

    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        IW = mem->tile.in_w;
        OW = mem->tile.out_w;
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih = oh / scale_h;
            if (ih >= IH) ih = IH - 1;
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

    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        return kern_conv2d(plan, op, mem);
    case TIGRIS_OP_DEPTHWISE:   return kern_depthwise_conv2d(plan, op, mem);
    case TIGRIS_OP_RELU:        return kern_relu(plan, op, mem);
    case TIGRIS_OP_RELU6:       return kern_relu6(plan, op, mem);
    case TIGRIS_OP_SIGMOID:     return kern_sigmoid(plan, op, mem);
    case TIGRIS_OP_TANH:        return kern_tanh(plan, op, mem);
    case TIGRIS_OP_SOFTMAX:     return kern_softmax(plan, op, mem);
    case TIGRIS_OP_ADD:         return kern_add(plan, op, mem);
    case TIGRIS_OP_MUL:         return kern_mul(plan, op, mem);
    case TIGRIS_OP_CONV1D:      return kern_conv1d(plan, op, mem);
    case TIGRIS_OP_GLOBAL_AVG:  return kern_global_avg_pool(plan, op, mem);
    case TIGRIS_OP_FULLY_CONN:  return kern_fully_connected(plan, op, mem);
    case TIGRIS_OP_RESHAPE:     return kern_reshape(plan, op, mem);
    case TIGRIS_OP_FLATTEN:     return kern_reshape(plan, op, mem);
    case TIGRIS_OP_MAX_POOL:    return kern_max_pool(plan, op, mem);
    case TIGRIS_OP_AVG_POOL:    return kern_avg_pool(plan, op, mem);
    case TIGRIS_OP_CONCAT:      return kern_concat(plan, op, mem);
    case TIGRIS_OP_RESIZE:      return kern_resize_nearest(plan, op, mem);
    case TIGRIS_OP_TRANSPOSE:   return tigris_transpose_execute(plan, op, op_index, mem);
    default:
        return -1;  /* unsupported op type */
    }
}

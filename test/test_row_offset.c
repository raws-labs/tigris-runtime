/**
 * @file test_row_offset.c
 * @brief Kernel sub-range parity tests for the line-buffered chains
 *        new-row offset contract (mem->tile.out_row_start / in_row_start).
 *
 * Each test runs a kernel twice through tigris_dispatch_kernel[_s8]():
 *   1. A full run (tile inactive) that computes every output row, captured
 *      as the reference.
 *   2. A sub-range run with tile.active=1, tile.out_row_start = k and
 *      tile.in_row_start set to the matching input row, tile.out_h reduced
 *      to cover only rows [k, OH). For spatial kernels the input pointer is
 *      a windowed buffer starting at the global row tile.in_row_start (as
 *      the tiled executor already hands kernels a pre-sliced input); for
 *      pointwise kernels the input/output pointers are the same full
 *      buffers used by the full run, since pointwise ops are height
 *      preserving and shift both pointers internally by the same offset.
 *
 * The sub-range run's computed rows must be byte-identical to the full
 * run's rows, and the rows before out_row_start must be untouched (still
 * zero, since the output buffer is zeroed before the sub-range call).
 */

#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "test_harness.h"

static const char g_strings[] = "\0";

enum { IH = 8, IW = 2, KH = 3, KW = 2, SH = 2 };

/* Pointwise inputs over a 1x8x2x1 map. */
static const int32_t k_mixed[16] = {-3, 5, -1, 2, 4, -7, 0, 8, -2, 6, 1, -9, 3, -4, 7, -5};
static const int32_t k_mixed_relu6[16] = {-3, 5, -1, 2, 4, 9, 0, 8, -2, 6, 1, -9, 3, -4, 7, -5};
static const int32_t k_mixed_s8[16] = {-5, -2, 0, 10, 3, 7, -8, 1, 4, -3, 9, -6, 2, 5, -1, 8};
static const int32_t k_relu6_s8[16] = {-10, 0, 30, 80, -5, 20, 60, 90, -20, 10, 40, 70, -1, 5, 25, 50};
static const int32_t k_lhs[16] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2};
static const int32_t k_rhs[16] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1};

/* Conv and depthwise filters (3x2 window, one or two channels) and biases. */
static const float k_conv_w[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
static const float k_dw_w[12] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
static const float k_conv_b[1] = {0.25f};
static const float k_dw_b[2] = {0.25f, -0.1f};
static const float k_conv_w_s8[6] = {1, 0, 0, 1, 1, 0};
static const float k_dw_w_s8[12] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
static const float k_zero_b[2] = {0, 0};

/*
 * A window op (conv, depthwise, pools) reads a 3x2 window at stride 2 over a
 * 1x8x2xC map into 1x3x1xC; a pointwise op keeps the 1x8x2x1 shape. The
 * sub-range run starts at output row skip. Scales of 0 mean no quantization;
 * a NULL input pattern is a ramp.
 */
typedef struct {
    const char *name;
    uint8_t op, s8, n_in, channels, skip, per_channel;
    float in_scale; int32_t in_zp;
    float out_scale; int32_t out_zp;
    const int32_t *a, *b;
    const float *weights, *bias;
} row_case_t;

static const row_case_t k_row_cases[] = {
    {"conv2d", TIGRIS_OP_CONV, 0, 1, 1, 1, 0, 0, 0, 0, 0, NULL, NULL, k_conv_w, k_conv_b},
    {"depthwise", TIGRIS_OP_DEPTHWISE, 0, 1, 2, 1, 0, 0, 0, 0, 0, NULL, NULL, k_dw_w, k_dw_b},
    {"max_pool", TIGRIS_OP_MAX_POOL, 0, 1, 1, 1, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL},
    {"relu", TIGRIS_OP_RELU, 0, 1, 1, 3, 0, 0, 0, 0, 0, k_mixed, NULL, NULL, NULL},
    {"relu6", TIGRIS_OP_RELU6, 0, 1, 1, 3, 0, 0, 0, 0, 0, k_mixed_relu6, NULL, NULL, NULL},
    {"sigmoid", TIGRIS_OP_SIGMOID, 0, 1, 1, 3, 0, 0, 0, 0, 0, k_mixed, NULL, NULL, NULL},
    {"tanh", TIGRIS_OP_TANH, 0, 1, 1, 3, 0, 0, 0, 0, 0, k_mixed, NULL, NULL, NULL},
    {"add", TIGRIS_OP_ADD, 0, 2, 1, 2, 0, 0, 0, 0, 0, k_lhs, k_rhs, NULL, NULL},
    {"mul", TIGRIS_OP_MUL, 0, 2, 1, 2, 0, 0, 0, 0, 0, k_lhs, k_rhs, NULL, NULL},
    {"conv2d_s8", TIGRIS_OP_CONV, 1, 1, 1, 1, 0, 1.0f, 0, 1.0f, 0, NULL, NULL, k_conv_w_s8, k_zero_b},
    {"depthwise_s8", TIGRIS_OP_DEPTHWISE, 1, 1, 2, 1, 1, 1.0f, 0, 1.0f, 0, NULL, NULL, k_dw_w_s8, k_zero_b},
    {"avg_pool_s8", TIGRIS_OP_AVG_POOL, 1, 1, 1, 1, 0, 1.0f, 0, 1.0f, 0, NULL, NULL, NULL, NULL},
    {"max_pool_s8", TIGRIS_OP_MAX_POOL, 1, 1, 1, 1, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL},
    {"relu_s8", TIGRIS_OP_RELU, 1, 1, 1, 3, 0, 0.5f, -2, 0, 0, k_mixed_s8, NULL, NULL, NULL},
    {"relu6_s8", TIGRIS_OP_RELU6, 1, 1, 1, 3, 0, 0.1f, 0, 0, 0, k_relu6_s8, NULL, NULL, NULL},
    {"sigmoid_s8", TIGRIS_OP_SIGMOID, 1, 1, 1, 3, 0, 0.5f, -2, 1.0f / 256.0f, -128, k_mixed_s8, NULL, NULL, NULL},
    {"tanh_s8", TIGRIS_OP_TANH, 1, 1, 1, 3, 0, 0.5f, -2, 1.0f / 128.0f, 0, k_mixed_s8, NULL, NULL, NULL},
    {"add_s8", TIGRIS_OP_ADD, 1, 2, 1, 2, 0, 1.0f, 0, 1.0f, 0, k_lhs, k_rhs, NULL, NULL},
    {"mul_s8", TIGRIS_OP_MUL, 1, 2, 1, 2, 0, 1.0f, 0, 1.0f, 0, k_lhs, k_rhs, NULL, NULL},
};

static int bytes_match(const void *a, const void *b, size_t n)
{
    return memcmp(a, b, n) == 0;
}

static int bytes_all_zero(const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++)
        if (b[i] != 0)
            return 0;
    return 1;
}

static int is_window_op(uint8_t op)
{
    return op == TIGRIS_OP_CONV || op == TIGRIS_OP_DEPTHWISE ||
           op == TIGRIS_OP_MAX_POOL || op == TIGRIS_OP_AVG_POOL;
}

static void store_value(uint8_t *buf, uint32_t i, int s8, float v)
{
    if (s8) {
        buf[i] = (uint8_t)(int8_t)v;
    } else {
        memcpy(buf + (size_t)i * sizeof(float), &v, sizeof(float));
    }
}

static void run_row_case(const row_case_t *rc)
{
    printf("  test_%s_row_offset...\n", rc->name);

    const int window = is_window_op(rc->op);
    const int c = rc->channels, k = rc->skip, s8 = rc->s8;
    const int sh = window ? SH : 1;
    const int oh = window ? 3 : IH, ow = window ? 1 : IW;
    const uint32_t elem = s8 ? 1u : (uint32_t)sizeof(float);
    const uint32_t in_elems = (uint32_t)(IH * IW * c);
    const uint32_t out_elems = (uint32_t)(oh * ow * c);
    const uint16_t out = rc->n_in;

    int32_t shape_pool[8] = {1, IH, IW, c, 1, oh, ow, c};
    uint16_t index_pool[3] = {0, 1, 2};
    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i <= out; i++) {
        tensors[i].shape_off = i == out ? 4 : 0;
        tensors[i].ndim = 4;
        tensors[i].dtype = s8 ? 3 : 1;
        tensors[i].size_bytes = (i == out ? out_elems : in_elems) * elem;
    }

    tigris_quant_param_t qp[3];
    int32_t quant_data[4] = {1073741824, 1073741824, 1, 1};  /* identity requant, two channels */
    memset(qp, 0, sizeof(qp));
    for (int i = 0; i <= out; i++) {
        float scale = i == out ? rc->out_scale : rc->in_scale;
        qp[i].scale = scale;
        qp[i].zero_point = i == out ? rc->out_zp : rc->in_zp;
        qp[i].num_channels = (uint16_t)(i == out && rc->per_channel ? c : 1);
        qp[i].shift_off = 2;
        if (s8)
            tensors[i].quant_param_idx = scale > 0.0f ? (uint16_t)i : TIGRIS_NO_QUANT_PARAM;
    }

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = rc->op;
    op.num_inputs = rc->n_in; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = out;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;
    if (window) {
        op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
        op.spatial.stride_h = SH; op.spatial.stride_w = 1;
        if (s8) { op.act_min = -128; op.act_max = 127; }
    }

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = (uint16_t)(out + 1); header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    if (s8) {
        plan.quant_params = qp;
        plan.quant_data = quant_data;
        plan.num_quant_params = 3;
    }

    /* Weights then bias; int8 filters carry an int32 bias right behind them. */
    tigris_weight_entry_t weights[2];
    _Alignas(4) uint8_t weight_blob[16 * sizeof(float)];
    if (rc->weights) {
        uint32_t nw = (uint32_t)(KH * KW * c);
        memset(weights, 0, sizeof(weights));
        for (uint32_t i = 0; i < nw; i++)
            store_value(weight_blob, i, s8, rc->weights[i]);
        for (int i = 0; i < c; i++) {
            if (s8) {
                int32_t b = (int32_t)rc->bias[i];
                memcpy(weight_blob + nw + (size_t)i * sizeof(b), &b, sizeof(b));
            } else {
                store_value(weight_blob, nw + (uint32_t)i, 0, rc->bias[i]);
            }
        }
        weights[0].size_bytes = nw * elem;
        weights[1].offset = nw * elem;
        weights[1].size_bytes = (uint32_t)c * (uint32_t)sizeof(float);
        op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
        op.weight_idx = 0; op.bias_idx = 1;
        header.num_weights = 2;
        plan.weight_entries = weights;
        plan.weight_blob = weight_blob;
    }

    _Alignas(4) uint8_t inputs[2][IH * IW * 2 * sizeof(float)];
    for (int j = 0; j < rc->n_in; j++) {
        const int32_t *pattern = j == 0 ? rc->a : rc->b;
        for (uint32_t i = 0; i < in_elems; i++) {
            float ramp = s8 ? (float)((i % 7) + 1) : (float)(i + 1);
            store_value(inputs[j], i, s8, pattern ? (float)pattern[i] : ramp);
        }
    }

    _Alignas(4) uint8_t full_out[IH * IW * sizeof(float)];
    _Alignas(4) uint8_t tiled_out[IH * IW * sizeof(float)];
    tigris_kernel_fn kernel = s8 ? tigris_dispatch_kernel_s8 : tigris_dispatch_kernel;
    void *ptrs[3];
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = (uint16_t)(out + 1);
    char msg[96];

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = inputs[0];
    ptrs[1] = inputs[1];
    ptrs[out] = full_out;
    snprintf(msg, sizeof(msg), "%s full run dispatch returns 0", rc->name);
    TEST_ASSERT(kernel(&plan, &op, 0, &mem, NULL) == 0, msg);

    /* A window op gets its input sliced from the first row it needs; a
     * pointwise op shifts both pointers itself. */
    memset(tiled_out, 0, sizeof(tiled_out));
    if (window)
        ptrs[0] = inputs[0] + (size_t)(k * sh) * IW * (size_t)c * elem;
    ptrs[out] = tiled_out;
    mem.tile.active = 1;
    mem.tile.out_h = oh - k;
    mem.tile.out_w = ow;
    mem.tile.out_row_start = k;
    mem.tile.in_row_start = k * sh;
    if (window) {
        mem.tile.in_h = IH - k * sh;
        mem.tile.in_w = IW;
    }
    snprintf(msg, sizeof(msg), "%s tiled run dispatch returns 0", rc->name);
    TEST_ASSERT(kernel(&plan, &op, 0, &mem, NULL) == 0, msg);

    size_t row_bytes = (size_t)ow * (size_t)c * elem;
    snprintf(msg, sizeof(msg), "%s sub-range rows byte-identical to full run", rc->name);
    TEST_ASSERT(bytes_match(full_out + (size_t)k * row_bytes,
                            tiled_out + (size_t)k * row_bytes,
                            (size_t)(oh - k) * row_bytes), msg);
    snprintf(msg, sizeof(msg), "%s rows before out_row_start are untouched", rc->name);
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)k * row_bytes), msg);
}

int main(void)
{
    printf("TiGrIS Row Offset Tests\n\n");

    for (size_t i = 0; i < sizeof(k_row_cases) / sizeof(k_row_cases[0]); i++)
        run_row_case(&k_row_cases[i]);

    return test_report();
}

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

/* Test infrastructure */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

static const char g_strings[] = "\0";

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

/* f32: Conv2D */

static void test_conv2d_row_offset_f32(void)
{
    printf("  test_conv2d_row_offset_f32...\n");

    enum { IH = 8, IW = 2, IC = 1, OC = 1, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    float X[IH * IW * IC];
    for (int i = 0; i < IH * IW * IC; i++)
        X[i] = (float)(i + 1);

    float W[OC * KH * KW * IC] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float B[OC] = {0.25f};

    int32_t x_shape[] = {1, IH, IW, IC};
    int32_t y_shape[] = {1, OH, OW, OC};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * OC * sizeof(float); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0; weights[1].offset = sizeof(W); weights[1].size_bytes = sizeof(B);

    uint8_t weight_blob[sizeof(W) + sizeof(B)];
    memcpy(weight_blob, W, sizeof(W));
    memcpy(weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1; header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = weight_blob;

    void *ptrs[2];
    float full_out[OH * OW * OC];
    float tiled_out[OH * OW * OC];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "conv2d full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * IC;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "conv2d tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * OC, tiled_out + (size_t)K * OW * OC,
        (size_t)(OH - K) * OW * OC * sizeof(float)),
        "conv2d sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * OC * sizeof(float)),
        "conv2d rows before out_row_start are untouched");
}

/* f32: DepthwiseConv2D */

static void test_depthwise_conv2d_row_offset_f32(void)
{
    printf("  test_depthwise_conv2d_row_offset_f32...\n");

    enum { IH = 8, IW = 2, C = 2, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    float X[IH * IW * C];
    for (int i = 0; i < IH * IW * C; i++)
        X[i] = (float)(i + 1);

    float W[KH * KW * C] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
    };
    float B[C] = {0.25f, -0.1f};

    int32_t x_shape[] = {1, IH, IW, C};
    int32_t y_shape[] = {1, OH, OW, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * C * sizeof(float); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0; weights[1].offset = sizeof(W); weights[1].size_bytes = sizeof(B);

    uint8_t weight_blob[sizeof(W) + sizeof(B)];
    memcpy(weight_blob, W, sizeof(W));
    memcpy(weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_DEPTHWISE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1; header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = weight_blob;

    void *ptrs[2];
    float full_out[OH * OW * C];
    float tiled_out[OH * OW * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "depthwise full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * C;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "depthwise tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * C, tiled_out + (size_t)K * OW * C,
        (size_t)(OH - K) * OW * C * sizeof(float)),
        "depthwise sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * C * sizeof(float)),
        "depthwise rows before out_row_start are untouched");
}

/* f32: MaxPool */

static void test_max_pool_row_offset_f32(void)
{
    printf("  test_max_pool_row_offset_f32...\n");

    enum { IH = 8, IW = 2, C = 1, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    float X[IH * IW * C];
    for (int i = 0; i < IH * IW * C; i++)
        X[i] = (float)(i + 1);

    int32_t x_shape[] = {1, IH, IW, C};
    int32_t y_shape[] = {1, OH, OW, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * C * sizeof(float); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_MAX_POOL;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;

    void *ptrs[2];
    float full_out[OH * OW * C];
    float tiled_out[OH * OW * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "max_pool full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * C;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "max_pool tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * C, tiled_out + (size_t)K * OW * C,
        (size_t)(OH - K) * OW * C * sizeof(float)),
        "max_pool sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * C * sizeof(float)),
        "max_pool rows before out_row_start are untouched");
}

/* f32: Relu (unary pointwise) */

static void test_relu_row_offset_f32(void)
{
    printf("  test_relu_row_offset_f32...\n");

    enum { H = 8, W = 2, C = 1, K = 3 };

    float X[H * W * C] = {
        -3, 5, -1, 2, 4, -7, 0, 8, -2, 6, 1, -9, 3, -4, 7, -5
    };

    int32_t shape[] = {1, H, W, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 0; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;

    int32_t shape_pool[4];
    memcpy(shape_pool, shape, sizeof(shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RELU;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;

    void *ptrs[2];
    float full_out[H * W * C];
    float tiled_out[H * W * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "relu full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.out_h = H - K;
    mem.tile.out_w = W;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K;

    ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "relu tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * W * C, tiled_out + (size_t)K * W * C,
        (size_t)(H - K) * W * C * sizeof(float)),
        "relu sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * W * C * sizeof(float)),
        "relu rows before out_row_start are untouched");
}

/* f32: Add (binary pointwise, dynamic second input) */

static void test_add_row_offset_f32(void)
{
    printf("  test_add_row_offset_f32...\n");

    enum { H = 8, W = 2, C = 1, K = 2 };

    float A[H * W * C];
    float Bv[H * W * C];
    for (int i = 0; i < H * W * C; i++) {
        A[i] = (float)(i % 5) - 2.0f;
        Bv[i] = (float)(i % 3) - 1.0f;
    }

    int32_t shape[] = {1, H, W, C};

    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 3; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(A); tensors[i].dtype = 1;
    }

    int32_t shape_pool[4];
    memcpy(shape_pool, shape, sizeof(shape));

    uint16_t index_pool[3] = {0, 1, 2}; /* inputs=[0,1] at off 0, output=[2] at off 2 */

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_ADD;
    op.num_inputs = 2; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 2;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 3; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;

    void *ptrs[3];
    float full_out[H * W * C];
    float tiled_out[H * W * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 3;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = A;
    ptrs[1] = Bv;
    ptrs[2] = full_out;
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "add full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = A;
    ptrs[1] = Bv;
    ptrs[2] = tiled_out;

    mem.tile.active = 1;
    mem.tile.out_h = H - K;
    mem.tile.out_w = W;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K;

    ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "add tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * W * C, tiled_out + (size_t)K * W * C,
        (size_t)(H - K) * W * C * sizeof(float)),
        "add sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * W * C * sizeof(float)),
        "add rows before out_row_start are untouched");
}

/* s8: Conv2D */

static void test_conv2d_row_offset_s8(void)
{
    printf("  test_conv2d_row_offset_s8...\n");

    enum { IH = 8, IW = 2, IC = 1, OC = 1, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    int8_t X[IH * IW * IC];
    for (int i = 0; i < IH * IW * IC; i++)
        X[i] = (int8_t)((i % 7) + 1);

    int8_t W[OC * KH * KW * IC] = {1, 0, 0, 1, 1, 0};
    int32_t B[OC] = {0};

    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    int32_t quant_data[4] = {mults[0], shifts[0], mults[0], shifts[0]};

    tigris_quant_param_t qp[2];
    memset(qp, 0, sizeof(qp));
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[0].multiplier_off = 0; qp[0].shift_off = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 1;
    qp[1].multiplier_off = 2; qp[1].shift_off = 3;

    int32_t x_shape[] = {1, IH, IW, IC};
    int32_t y_shape[] = {1, OH, OW, OC};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 3;
    tensors[0].quant_param_idx = 0;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * OC; tensors[1].dtype = 3;
    tensors[1].quant_param_idx = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0; weights[1].offset = sizeof(W); weights[1].size_bytes = sizeof(B);

    uint8_t weight_blob[sizeof(W) + sizeof(B)];
    memcpy(weight_blob, W, sizeof(W));
    memcpy(weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.weight_idx = 0; op.bias_idx = 1;
    op.act_min = -128; op.act_max = 127;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1; header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = weight_blob;
    plan.quant_params = qp;
    plan.quant_data = quant_data;
    plan.num_quant_params = 2;

    void *ptrs[2];
    int8_t full_out[OH * OW * OC];
    int8_t tiled_out[OH * OW * OC];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "conv2d_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * IC;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "conv2d_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * OC, tiled_out + (size_t)K * OW * OC,
        (size_t)(OH - K) * OW * OC),
        "conv2d_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * OC),
        "conv2d_s8 rows before out_row_start are untouched");
}

/* s8: DepthwiseConv2D */

static void test_depthwise_conv2d_row_offset_s8(void)
{
    printf("  test_depthwise_conv2d_row_offset_s8...\n");

    enum { IH = 8, IW = 2, C = 2, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    int8_t X[IH * IW * C];
    for (int i = 0; i < IH * IW * C; i++)
        X[i] = (int8_t)((i % 7) + 1);

    int8_t W[KH * KW * C] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
    int32_t B[C] = {0, 0};

    /* Per-channel identity requant for the output: mult=2^30, shift=1. */
    int32_t quant_data[6] = {
        1073741824, 1, /* input single channel (unused mult/shift, but present) */
        1073741824, 1073741824, /* output multipliers, ch 0 and 1 */
        1, 1                    /* output shifts, ch 0 and 1 */
    };

    tigris_quant_param_t qp[2];
    memset(qp, 0, sizeof(qp));
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[0].multiplier_off = 0; qp[0].shift_off = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 2;
    qp[1].multiplier_off = 2; qp[1].shift_off = 4;

    int32_t x_shape[] = {1, IH, IW, C};
    int32_t y_shape[] = {1, OH, OW, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 3;
    tensors[0].quant_param_idx = 0;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * C; tensors[1].dtype = 3;
    tensors[1].quant_param_idx = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0; weights[1].offset = sizeof(W); weights[1].size_bytes = sizeof(B);

    uint8_t weight_blob[sizeof(W) + sizeof(B)];
    memcpy(weight_blob, W, sizeof(W));
    memcpy(weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_DEPTHWISE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.weight_idx = 0; op.bias_idx = 1;
    op.act_min = -128; op.act_max = 127;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1; header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = weight_blob;
    plan.quant_params = qp;
    plan.quant_data = quant_data;
    plan.num_quant_params = 2;

    void *ptrs[2];
    int8_t full_out[OH * OW * C];
    int8_t tiled_out[OH * OW * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "depthwise_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * C;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "depthwise_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * C, tiled_out + (size_t)K * OW * C,
        (size_t)(OH - K) * OW * C),
        "depthwise_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * C),
        "depthwise_s8 rows before out_row_start are untouched");
}

/* s8: AveragePool (kh_start/kh_end clamp style, distinct from MaxPool) */

static void test_avg_pool_row_offset_s8(void)
{
    printf("  test_avg_pool_row_offset_s8...\n");

    enum { IH = 8, IW = 2, C = 1, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    int8_t X[IH * IW * C];
    for (int i = 0; i < IH * IW * C; i++)
        X[i] = (int8_t)((i % 7) + 1);

    tigris_quant_param_t qp[2];
    memset(qp, 0, sizeof(qp));
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 1;

    int32_t x_shape[] = {1, IH, IW, C};
    int32_t y_shape[] = {1, OH, OW, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 3;
    tensors[0].quant_param_idx = 0;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * C; tensors[1].dtype = 3;
    tensors[1].quant_param_idx = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_AVG_POOL;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;
    op.act_min = -128; op.act_max = 127;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.quant_params = qp;
    plan.num_quant_params = 2;

    void *ptrs[2];
    int8_t full_out[OH * OW * C];
    int8_t tiled_out[OH * OW * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "avg_pool_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * C;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "avg_pool_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * C, tiled_out + (size_t)K * OW * C,
        (size_t)(OH - K) * OW * C),
        "avg_pool_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * C),
        "avg_pool_s8 rows before out_row_start are untouched");
}

/* s8: MaxPool */

static void test_max_pool_row_offset_s8(void)
{
    printf("  test_max_pool_row_offset_s8...\n");

    enum { IH = 8, IW = 2, C = 1, KH = 3, KW = 2, SH = 2, SW = 1,
           OH = 3, OW = 1, K = 1 };

    int8_t X[IH * IW * C];
    for (int i = 0; i < IH * IW * C; i++)
        X[i] = (int8_t)((i % 7) + 1);

    int32_t x_shape[] = {1, IH, IW, C};
    int32_t y_shape[] = {1, OH, OW, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 3;
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = OH * OW * C; tensors[1].dtype = 3;
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_MAX_POOL;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = KH; op.spatial.kernel_w = KW;
    op.spatial.stride_h = SH; op.spatial.stride_w = SW;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;
    op.act_min = -128; op.act_max = 127;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;

    void *ptrs[2];
    int8_t full_out[OH * OW * C];
    int8_t tiled_out[OH * OW * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "max_pool_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X + (size_t)(K * SH) * IW * C;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.in_h = IH - K * SH;
    mem.tile.out_h = OH - K;
    mem.tile.in_w = IW;
    mem.tile.out_w = OW;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K * SH;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "max_pool_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * OW * C, tiled_out + (size_t)K * OW * C,
        (size_t)(OH - K) * OW * C),
        "max_pool_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * OW * C),
        "max_pool_s8 rows before out_row_start are untouched");
}

/* s8: Relu (unary pointwise) */

static void test_relu_row_offset_s8(void)
{
    printf("  test_relu_row_offset_s8...\n");

    enum { H = 8, W = 2, C = 1, K = 3 };

    int8_t X[H * W * C] = {
        -5, -2, 0, 10, 3, 7, -8, 1, 4, -3, 9, -6, 2, 5, -1, 8
    };

    tigris_quant_param_t qp[1];
    memset(qp, 0, sizeof(qp));
    qp[0].scale = 0.5f; qp[0].zero_point = -2; qp[0].num_channels = 1;

    int32_t shape[] = {1, H, W, C};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 3;
    tensors[0].quant_param_idx = 0;
    tensors[1].shape_off = 0; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 3;
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    int32_t shape_pool[4];
    memcpy(shape_pool, shape, sizeof(shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RELU;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.quant_params = qp;
    plan.num_quant_params = 1;

    void *ptrs[2];
    int8_t full_out[H * W * C];
    int8_t tiled_out[H * W * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = X;
    ptrs[1] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "relu_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = X;
    ptrs[1] = tiled_out;

    mem.tile.active = 1;
    mem.tile.out_h = H - K;
    mem.tile.out_w = W;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "relu_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * W * C, tiled_out + (size_t)K * W * C,
        (size_t)(H - K) * W * C),
        "relu_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * W * C),
        "relu_s8 rows before out_row_start are untouched");
}

/* s8: Add (binary pointwise, two dynamic inputs) */

static void test_add_row_offset_s8(void)
{
    printf("  test_add_row_offset_s8...\n");

    enum { H = 8, W = 2, C = 1, K = 2 };

    int8_t A[H * W * C];
    int8_t Bv[H * W * C];
    for (int i = 0; i < H * W * C; i++) {
        A[i] = (int8_t)((i % 5) - 2);
        Bv[i] = (int8_t)((i % 3) - 1);
    }

    tigris_quant_param_t qp[3];
    memset(qp, 0, sizeof(qp));
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 1;
    qp[2].scale = 1.0f; qp[2].zero_point = 0; qp[2].num_channels = 1;

    int32_t shape[] = {1, H, W, C};

    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 3; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(A); tensors[i].dtype = 3;
        tensors[i].quant_param_idx = (uint16_t)i;
    }

    int32_t shape_pool[4];
    memcpy(shape_pool, shape, sizeof(shape));

    uint16_t index_pool[3] = {0, 1, 2};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_ADD;
    op.num_inputs = 2; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 2;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 3; header.num_ops = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.quant_params = qp;
    plan.num_quant_params = 3;

    void *ptrs[3];
    int8_t full_out[H * W * C];
    int8_t tiled_out[H * W * C];

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 3;

    memset(full_out, 0, sizeof(full_out));
    ptrs[0] = A;
    ptrs[1] = Bv;
    ptrs[2] = full_out;
    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "add_s8 full run dispatch returns 0");

    memset(tiled_out, 0, sizeof(tiled_out));
    ptrs[0] = A;
    ptrs[1] = Bv;
    ptrs[2] = tiled_out;

    mem.tile.active = 1;
    mem.tile.out_h = H - K;
    mem.tile.out_w = W;
    mem.tile.out_row_start = K;
    mem.tile.in_row_start = K;

    ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "add_s8 tiled run dispatch returns 0");

    TEST_ASSERT(bytes_match(
        full_out + (size_t)K * W * C, tiled_out + (size_t)K * W * C,
        (size_t)(H - K) * W * C),
        "add_s8 sub-range rows byte-identical to full run");
    TEST_ASSERT(bytes_all_zero(tiled_out, (size_t)K * W * C),
        "add_s8 rows before out_row_start are untouched");
}

int main(void)
{
    printf("TiGrIS Row Offset Tests\n\n");

    test_conv2d_row_offset_f32();
    test_depthwise_conv2d_row_offset_f32();
    test_max_pool_row_offset_f32();
    test_relu_row_offset_f32();
    test_add_row_offset_f32();

    test_conv2d_row_offset_s8();
    test_depthwise_conv2d_row_offset_s8();
    test_avg_pool_row_offset_s8();
    test_max_pool_row_offset_s8();
    test_relu_row_offset_s8();
    test_add_row_offset_s8();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed ? 1 : 0;
}

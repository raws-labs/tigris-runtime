/**
 * @file test_kernels.c
 * @brief Unit tests for reference float32 kernels.
 *
 * Each test constructs a minimal tigris_plan_t in memory (no binary file),
 * populates tensors/ops/weights, and runs a single kernel through dispatch.
 *
 * Data layout: NHWC, float32.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tigris.h"
#include "tigris_mem.h"
#include "tigris_kernels.h"

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

#define TEST_ASSERT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (eps)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %.6f, expected %.6f)\n", \
                __func__, __LINE__, msg, (double)(a), (double)(b)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

#define EPS 1e-5f

/* Plan builder helpers */

/*
 * Instead of constructing a full binary plan, we build a tigris_plan_t
 * by pointing its fields to local stack/heap arrays. This is valid
 * because the kernels only access plan->tensors, plan->ops,
 * plan->index_pool, plan->shape_pool, plan->weight_entries,
 * plan->weight_blob via the inline helpers.
 */

/* A string table with just a NUL at offset 0. */
static const char g_strings[] = "\0";

/* Conv2D test */

static void test_conv2d(void)
{
    printf("  test_conv2d...\n");

    /* NHWC: 1x4x4x1 input, kernel 3x3, stride=1, pad=0 -> 1x2x2x1 output */
    int32_t x_shape[] = {1, 4, 4, 1};
    int32_t y_shape[] = {1, 2, 2, 1};

    /* Input data in NHWC (single channel, same as row-major) */
    float X[16];
    for (int i = 0; i < 16; i++) X[i] = (float)(i + 1);
    /* X = 1..16 in row-major 4x4 */

    /* Weight [OC=1, KH=3, KW=3, IC=1] (OHWI layout) */
    float W[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    float B[1] = {0.5f};

    /* Expected: conv with this kernel computes column-diff + bias
     * out[0,0] = (1+5+9)*1 + (2+6+10)*0 + (3+7+11)*(-1) + 0.5 = 15-21+0.5 = -5.5
     * out[0,1] = (2+6+10)*1 + (3+7+11)*0 + (4+8+12)*(-1) + 0.5 = 18-24+0.5 = -5.5
     * out[1,0] = (5+9+13)*1 + (6+10+14)*0 + (7+11+15)*(-1) + 0.5 = 27-33+0.5 = -5.5
     * out[1,1] = (6+10+14)*1 + (7+11+15)*0 + (8+12+16)*(-1) + 0.5 = 30-36+0.5 = -5.5
     */
    float expected[4] = {-5.5f, -5.5f, -5.5f, -5.5f};

    /* Build plan */
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    /* Shape pool: x_shape at [0..3], y_shape at [4..7] */
    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    /* Index pool: inputs=[0] at idx 0, outputs=[1] at idx 1 */
    uint16_t index_pool[2] = {0, 1};

    /* Weight entry */
    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0;
    weights[1].offset = sizeof(W);
    weights[1].size_bytes = sizeof(B);

    /* Weight blob: W then B */
    float weight_blob[10];
    memcpy(weight_blob, W, sizeof(W));
    memcpy((char *)weight_blob + sizeof(W), B, sizeof(B));

    /* Op */
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 3; op.spatial.kernel_w = 3;
    op.spatial.stride_h = 1; op.spatial.stride_w = 1;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    /* Assemble plan */
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = (const uint8_t *)weight_blob;

    /* Memory */
    void *ptrs[2] = {NULL, NULL};
    float out_buf[4];
    ptrs[0] = X;
    ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    /* Run */
    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "conv2d output");
}

/* ConvTranspose tests */

static void test_conv_transpose_stride2(void)
{
    printf("  test_conv_transpose_stride2...\n");

    /* NHWC: 1x2x2x1 input, kernel 2x2, stride=2, pad=0 -> 1x4x4x1 output.
     * With stride == kernel and no pad, each input pixel maps to a
     * disjoint 2x2 output block:
     *   out[2*ih+kh, 2*iw+kw] = X[ih,iw] * W[kh,kw]
     * Hand-verified per input pixel:
     *   ih=0,iw=0 (X=1): out[0,0]=10 out[0,1]=20 out[1,0]=30 out[1,1]=40
     *   ih=0,iw=1 (X=2): out[0,2]=20 out[0,3]=40 out[1,2]=60 out[1,3]=80
     *   ih=1,iw=0 (X=3): out[2,0]=30 out[2,1]=60 out[3,0]=90 out[3,1]=120
     *   ih=1,iw=1 (X=4): out[2,2]=40 out[2,3]=80 out[3,2]=120 out[3,3]=160
     */
    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 4, 4, 1};

    float X[4] = {1, 2,
                  3, 4};
    /* Weight [OC=1, KH=2, KW=2, IC=1] (OHWI layout) */
    float W[4] = {10, 20,
                  30, 40};
    float B[1] = {0.0f};

    float expected[16] = {
        10,  20,  20,  40,
        30,  40,  60,  80,
        30,  60,  40,  80,
        90, 120, 120, 160,
    };

    /* Build plan */
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0;
    weights[1].offset = sizeof(W);
    weights[1].size_bytes = sizeof(B);

    float weight_blob[5];
    memcpy(weight_blob, W, sizeof(W));
    memcpy((char *)weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV_TRANSPOSE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 2; op.spatial.kernel_w = 2;
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.spatial.group = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = (const uint8_t *)weight_blob;

    void *ptrs[2] = {NULL, NULL};
    float out_buf[16];
    memset(out_buf, 0, sizeof(out_buf));
    ptrs[0] = X;
    ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 16; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "conv_transpose stride2 output");
}

static void test_conv_transpose_pad_stride_skip(void)
{
    printf("  test_conv_transpose_pad_stride_skip...\n");

    /* NHWC: 1x2x2x1 input, kernel 3x3, stride=2, pad=1 -> 1x3x3x1 output.
     * This case exercises the skip branches in the gather loop:
     *   - num_h/num_w % stride != 0 (some kh/kw taps land off the stride
     *     lattice and must be skipped)
     *   - ih/iw < 0 or >= input extent (pad pushes the tap out of range)
     *
     * Scatter-form cross-check (oh = ih*SH - PT + kh, same for w) gives the
     * same result as the gather form used by the kernel; both were computed
     * by hand and agree (see task-1-report.md for the full derivation).
     *
     * Weight [OC=1, KH=3, KW=3, IC=1] (OHWI), row-major kh,kw = 1..9:
     *   W = [[1,2,3],[4,5,6],[7,8,9]]
     * Input X[ih,iw] = [[1,2],[3,4]]
     *
     * Valid (ih,kh,oh) triples (num_h % 2 == 0 and 0<=ih<2):
     *   oh=0: (ih=0,kh=1)
     *   oh=1: (ih=1,kh=0), (ih=0,kh=2)
     *   oh=2: (ih=1,kh=1)
     * Same structure for (iw,kw,ow). Expected (row-major oh,ow):
     *   Y[0,0]=5   Y[0,1]=14  Y[0,2]=10
     *   Y[1,0]=14  Y[1,1]=36  Y[1,2]=24
     *   Y[2,0]=15  Y[2,1]=34  Y[2,2]=20
     */
    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 3, 3, 1};

    float X[4] = {1, 2,
                  3, 4};
    float W[9] = {1, 2, 3,
                  4, 5, 6,
                  7, 8, 9};
    float B[1] = {0.0f};

    float expected[9] = {
         5, 14, 10,
        14, 36, 24,
        15, 34, 20,
    };

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0;
    weights[1].offset = sizeof(W);
    weights[1].size_bytes = sizeof(B);

    float weight_blob[10];
    memcpy(weight_blob, W, sizeof(W));
    memcpy((char *)weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONV_TRANSPOSE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 3; op.spatial.kernel_w = 3;
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
    op.spatial.pad_top = 1; op.spatial.pad_bottom = 1;
    op.spatial.pad_left = 1; op.spatial.pad_right = 1;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.spatial.group = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = (const uint8_t *)weight_blob;

    void *ptrs[2] = {NULL, NULL};
    float out_buf[9];
    memset(out_buf, 0, sizeof(out_buf));
    ptrs[0] = X;
    ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 9; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "conv_transpose pad/stride-skip output");
}

/* DepthwiseConv2D test */

static void test_depthwise_conv2d(void)
{
    printf("  test_depthwise_conv2d...\n");

    /* NHWC: 1x4x4x2 input, 2-channel DW kernel 3x3, stride=1, pad=0 -> 1x2x2x2 output */
    int32_t x_shape[] = {1, 4, 4, 2};
    int32_t y_shape[] = {1, 2, 2, 2};

    /* Input in NHWC: channel-interleaved.
     * NCHW ch0 = 1..16, ch1 = 17..32
     * NHWC: for each spatial position, interleave channels */
    float X[32];
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            int nchw_ch0 = h * 4 + w + 1;       /* 1..16 */
            int nchw_ch1 = h * 4 + w + 17;      /* 17..32 */
            X[(h * 4 + w) * 2 + 0] = (float)nchw_ch0;
            X[(h * 4 + w) * 2 + 1] = (float)nchw_ch1;
        }
    }

    /* Weight [KH=3, KW=3, C=2] in HWC layout.
     * ch0 = all 1s, ch1 = all -1s
     * HWC interleaves channels: for each (kh,kw), store [ch0, ch1] */
    float W[18];
    for (int i = 0; i < 9; i++) {
        W[i * 2 + 0] =  1.0f;   /* ch0 */
        W[i * 2 + 1] = -1.0f;   /* ch1 */
    }
    float B[2] = {0.0f, 10.0f};

    /* Channel 0: sum of each 3x3 patch (all ones kernel)
     * patch[0,0]: 1+2+3+5+6+7+9+10+11 = 54
     * patch[0,1]: 2+3+4+6+7+8+10+11+12 = 63
     * patch[1,0]: 5+6+7+9+10+11+13+14+15 = 90
     * patch[1,1]: 6+7+8+10+11+12+14+15+16 = 99
     * Channel 1: -(sum of 3x3 patch) + 10
     * patch[0,0]: -(17+18+19+21+22+23+25+26+27) + 10 = -198+10 = -188
     * patch[0,1]: -(18+19+20+22+23+24+26+27+28) + 10 = -207+10 = -197
     * patch[1,0]: -(21+22+23+25+26+27+29+30+31) + 10 = -234+10 = -224
     * patch[1,1]: -(22+23+24+26+27+28+30+31+32) + 10 = -243+10 = -233
     *
     * NHWC output order: [h=0,w=0,c=0], [h=0,w=0,c=1], [h=0,w=1,c=0], ...
     */
    float expected[8] = {54, -188, 63, -197, 90, -224, 99, -233};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0;
    weights[1].offset = sizeof(W);
    weights[1].size_bytes = sizeof(B);

    float weight_blob[20];
    memcpy(weight_blob, W, sizeof(W));
    memcpy((char *)weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_DEPTHWISE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 3; op.spatial.kernel_w = 3;
    op.spatial.stride_h = 1; op.spatial.stride_w = 1;
    op.spatial.dilation_h = 1; op.spatial.dilation_w = 1;
    op.spatial.group = 2;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = (const uint8_t *)weight_blob;

    void *ptrs[2] = {NULL, NULL};
    float out_buf[8];
    ptrs[0] = X;
    ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 8; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "depthwise output");
}

/* Relu test */

static void test_relu(void)
{
    printf("  test_relu...\n");

    float X[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float expected[4] = {0.0f, 0.0f, 1.0f, 2.0f};
    /* NHWC: 1x2x2x1 */
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 0; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;

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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[2];
    float out_buf[4];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "relu output");
}

/* Relu6 test */

static void test_relu6(void)
{
    printf("  test_relu6...\n");

    float X[4] = {-1.0f, 3.0f, 7.0f, 6.0f};
    float expected[4] = {0.0f, 3.0f, 6.0f, 6.0f};
    /* NHWC: 1x2x2x1 */
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 0; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RELU6;
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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[2];
    float out_buf[4];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "relu6 output");
}

/* Add test */

static void test_add(void)
{
    printf("  test_add...\n");

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float expected[4] = {11.0f, 22.0f, 33.0f, 44.0f};
    /* NHWC: 1x2x2x1 */
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 3; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(A); tensors[i].dtype = 1;
    }

    /* Index pool: inputs=[0,1] at idx 0, outputs=[2] at idx 2 */
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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[3];
    float out_buf[4];
    ptrs[0] = A; ptrs[1] = B; ptrs[2] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 3;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "add output");
}

/* GlobalAvgPool test */

static void test_global_avg_pool(void)
{
    printf("  test_global_avg_pool...\n");

    /* NHWC: 1x3x3x2 -> 1x1x1x2 */
    int32_t x_shape[] = {1, 3, 3, 2};
    int32_t y_shape[] = {1, 1, 1, 2};

    /* Input in NHWC: channel-interleaved.
     * NCHW ch0 = 1..9, ch1 = 10..18
     * NHWC: for each (h,w), interleave channels */
    float X[18];
    for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
            int nchw_ch0 = h * 3 + w + 1;       /* 1..9 */
            int nchw_ch1 = h * 3 + w + 10;      /* 10..18 */
            X[(h * 3 + w) * 2 + 0] = (float)nchw_ch0;
            X[(h * 3 + w) * 2 + 1] = (float)nchw_ch1;
        }
    }

    /* Expected: avg(1..9) = 5.0, avg(10..18) = 14.0 */
    float expected[2] = {5.0f, 14.0f};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_GLOBAL_AVG;
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
    float out_buf[2];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 2; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "global_avg_pool output");
}

/* AveragePool test */

static void test_avg_pool(void)
{
    printf("  test_avg_pool...\n");

    /* NHWC: 1x4x4x1, 2x2 stride 2 -> 1x2x2x1. */
    int32_t x_shape[] = {1, 4, 4, 1};
    int32_t y_shape[] = {1, 2, 2, 1};
    float input[16];
    for (int i = 0; i < 16; i++) input[i] = (float)(i + 1);
    float expected[] = {3.5f, 5.5f, 11.5f, 13.5f};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(input); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));
    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_AVG_POOL;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 2; op.spatial.kernel_w = 2;
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
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
    float output[4];
    ptrs[0] = input; ptrs[1] = output;
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(output[i], expected[i], EPS, "avg_pool output");
}

/* FullyConnected (Gemm) test */

static void test_fully_connected(void)
{
    printf("  test_fully_connected...\n");

    /* W: 2x3, X: 3-vector, B: 2-vector -> Y: 2-vector */
    int32_t x_shape[] = {1, 3};
    int32_t y_shape[] = {1, 2};

    float X[3] = {1.0f, 2.0f, 3.0f};
    float W[6] = {
        1, 0, 0,   /* row 0 */
        0, 1, 1    /* row 1 */
    };
    float B[2] = {0.5f, -0.5f};

    /* Y[0] = 1*1 + 0*2 + 0*3 + 0.5 = 1.5
     * Y[1] = 0*1 + 1*2 + 1*3 - 0.5 = 4.5 */
    float expected[2] = {1.5f, 4.5f};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 2;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 2; tensors[1].ndim = 2;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[4];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 2, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(W);
    weights[1].name_str = 0;
    weights[1].offset = sizeof(W);
    weights[1].size_bytes = sizeof(B);

    float weight_blob[8];
    memcpy(weight_blob, W, sizeof(W));
    memcpy((char *)weight_blob + sizeof(W), B, sizeof(B));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_FULLY_CONN;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = 0; op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = (const uint8_t *)weight_blob;

    void *ptrs[2];
    float out_buf[2];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 2; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "fc output");
}

/* Reshape test */

static void test_reshape(void)
{
    printf("  test_reshape...\n");

    float X[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int32_t x_shape[] = {1, 6};
    int32_t y_shape[] = {2, 3};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 2;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 2; tensors[1].ndim = 2;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;

    int32_t shape_pool[4];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 2, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RESHAPE;
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
    float out_buf[6];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    TEST_ASSERT(memcmp(out_buf, X, sizeof(X)) == 0, "reshape memcpy matches");
}

/* MaxPool test */

static void test_max_pool(void)
{
    printf("  test_max_pool...\n");

    /* NHWC: 1x4x4x1 input, kernel 2x2, stride=2, pad=0 -> 1x2x2x1 output */
    int32_t x_shape[] = {1, 4, 4, 1};
    int32_t y_shape[] = {1, 2, 2, 1};

    float X[16];
    for (int i = 0; i < 16; i++) X[i] = (float)(i + 1);
    /* X = 1..16 in row-major 4x4
     * Pool patches (2x2, stride 2):
     * [0,0]: max(1,2,5,6) = 6
     * [0,1]: max(3,4,7,8) = 8
     * [1,0]: max(9,10,13,14) = 14
     * [1,1]: max(11,12,15,16) = 16
     */
    float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_MAX_POOL;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.kernel_h = 2; op.spatial.kernel_w = 2;
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
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
    float out_buf[4];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "max_pool output");
}

/* Concat test */

static void test_concat(void)
{
    printf("  test_concat...\n");

    /* Two inputs NHWC: 1x2x2x1 each -> output 1x2x2x2 (channel concat) */
    int32_t xa_shape[] = {1, 2, 2, 1};
    int32_t xb_shape[] = {1, 2, 2, 1};
    int32_t y_shape[]  = {1, 2, 2, 2};

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    /* Expected NHWC output: for each (h,w), channels = [A_val, B_val]
     * [0,0]: [1, 10], [0,1]: [2, 20], [1,0]: [3, 30], [1,1]: [4, 40]
     */
    float expected[8] = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f};

    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(A); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(B); tensors[1].dtype = 1;
    tensors[2].shape_off = 8; tensors[2].ndim = 4;
    tensors[2].size_bytes = sizeof(expected); tensors[2].dtype = 1;

    int32_t shape_pool[12];
    memcpy(shape_pool, xa_shape, sizeof(xa_shape));
    memcpy(shape_pool + 4, xb_shape, sizeof(xb_shape));
    memcpy(shape_pool + 8, y_shape, sizeof(y_shape));

    /* Index pool: inputs=[0,1] at idx 0, outputs=[2] at idx 2 */
    uint16_t index_pool[3] = {0, 1, 2};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONCAT;
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
    float out_buf[8];
    ptrs[0] = A; ptrs[1] = B; ptrs[2] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 3;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 8; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "concat output");
}

/* Resize nearest test */

static void test_resize_nearest(void)
{
    printf("  test_resize_nearest...\n");

    /* NHWC: 1x2x2x1 input, scale 2x -> 1x4x4x1 output */
    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 4, 4, 1};

    float X[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    /* 2x nearest: each pixel duplicated 2x2
     * [1, 1, 2, 2]
     * [1, 1, 2, 2]
     * [3, 3, 4, 4]
     * [3, 3, 4, 4]
     */
    float expected[16] = {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
    };

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(expected); tensors[1].dtype = 1;

    int32_t shape_pool[8];
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RESIZE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    /* Scale factors stored in stride_h/w */
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
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
    float out_buf[16];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 16; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "resize_nearest output");
}

/* Sigmoid test */

static void test_sigmoid(void)
{
    printf("  test_sigmoid...\n");

    float X[4] = {0.0f, -1.0f, 1.0f, 2.0f};
    /* sigmoid(0)=0.5, sigmoid(-1)~=0.2689, sigmoid(1)~=0.7311, sigmoid(2)~=0.8808 */
    float expected[4] = {0.5f, 0.268941f, 0.731059f, 0.880797f};
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 0; tensors[1].ndim = 4;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;

    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_SIGMOID;
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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[2];
    float out_buf[4];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], 1e-4f, "sigmoid output");
}

static void test_softmax(void)
{
    printf("  test_softmax...\n");

    /* Two independent class vectors: [0,1,-1] and [0,0,0]. */
    float X[6] = {0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f};
    float expected[6] = {0.244728f, 0.665241f, 0.090031f,
                         0.333333f, 0.333333f, 0.333333f};
    int32_t shape[] = {2, 3};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 2;
    tensors[0].size_bytes = sizeof(X); tensors[0].dtype = 1;
    tensors[1].shape_off = 0; tensors[1].ndim = 2;
    tensors[1].size_bytes = sizeof(X); tensors[1].dtype = 1;
    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_SOFTMAX;
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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[2];
    float out_buf[6];
    ptrs[0] = X; ptrs[1] = out_buf;
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "softmax dispatch returns 0");
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], 1e-5f, "softmax output");
}

/* Mul test */

static void test_mul(void)
{
    printf("  test_mul...\n");

    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {10.0f, 0.5f, -1.0f, 2.0f};
    float expected[4] = {10.0f, 1.0f, -3.0f, 8.0f};
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 3; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(A); tensors[i].dtype = 1;
    }

    uint16_t index_pool[3] = {0, 1, 2};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_MUL;
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
    plan.shape_pool = shape;
    plan.strings = g_strings;

    void *ptrs[3];
    float out_buf[4];
    ptrs[0] = A; ptrs[1] = B; ptrs[2] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 3;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == 0, "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "mul output");
}

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    int32_t shapes[8];
    uint16_t indices[2];
    tigris_weight_entry_t weight;
    uint8_t weight_blob[1 + 4 * sizeof(float)];
    tigris_plan_t plan;
    tigris_mem_t mem;
    void *ptrs[2];
    float input[4];
    float output[4];
} const_binary_fixture_t;

static void init_const_binary_fixture(
    const_binary_fixture_t *fx, tigris_op_type_t op_type,
    const float *constant, uint32_t constant_bytes)
{
    memset(fx, 0, sizeof(*fx));

    const int32_t shape[4] = {1, 2, 2, 1};
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->header.num_weights = 1;
    memcpy(fx->shapes, shape, sizeof(shape));
    memcpy(fx->shapes + 4, shape, sizeof(shape));
    memcpy(fx->input, input, sizeof(input));
    for (int i = 0; i < 4; i++)
        fx->output[i] = -123.0f;

    for (int i = 0; i < 2; i++) {
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].dtype = 1; /* ONNX FLOAT */
        fx->tensors[i].size_bytes = sizeof(fx->input);
        fx->tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }

    /* Compiler encoding: the dynamic input is the only tensor input; the
     * constant is raw float32 bytes selected through weight_idx. Offset it by
     * one byte to verify that packed, unaligned weight blobs are safe. */
    fx->indices[0] = 0;
    fx->indices[1] = 1;
    fx->op.op_type = (uint8_t)op_type;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;
    fx->op.weight_idx = 0;
    fx->op.bias_idx = TIGRIS_NO_WEIGHT;
    fx->weight.offset = 1;
    fx->weight.size_bytes = constant_bytes;
    memcpy(fx->weight_blob + 1, constant, constant_bytes);

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.weight_entries = &fx->weight;
    fx->plan.weight_blob = fx->weight_blob;

    fx->ptrs[0] = fx->input;
    fx->ptrs[1] = fx->output;
    fx->mem.tensor_ptrs = fx->ptrs;
    fx->mem.num_tensors = 2;
}

static void assert_f32_array(
    const float *actual, const float *expected, const char *message)
{
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(actual[i], expected[i], EPS, message);
}

static void test_constant_binary_f32(void)
{
    printf("  test_constant_binary_f32...\n");

    const_binary_fixture_t fx;
    float scalar = 2.0f;
    const float exact[4] = {10.0f, 0.5f, -1.0f, 2.0f};
    const float add_scalar_expected[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    const float mul_scalar_expected[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    const float add_exact_expected[4] = {11.0f, 2.5f, 2.0f, 6.0f};
    const float mul_exact_expected[4] = {10.0f, 1.0f, -3.0f, 8.0f};

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, &scalar, (uint32_t)sizeof(scalar));
    TEST_ASSERT(((uintptr_t)tigris_op_weight(&fx.plan, &fx.op) %
                 _Alignof(float)) != 0,
                "constant fixture is deliberately unaligned");
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "scalar constant Add succeeds");
    assert_f32_array(fx.output, add_scalar_expected,
                     "scalar constant Add output");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_MUL, &scalar, (uint32_t)sizeof(scalar));
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "scalar constant Mul succeeds");
    assert_f32_array(fx.output, mul_scalar_expected,
                     "scalar constant Mul output");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, exact, (uint32_t)sizeof(exact));
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "exact constant Add succeeds");
    assert_f32_array(fx.output, add_exact_expected,
                     "exact constant Add output");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_MUL, exact, (uint32_t)sizeof(exact));
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "exact constant Mul succeeds");
    assert_f32_array(fx.output, mul_exact_expected,
                     "exact constant Mul output");

    /* Scalar broadcast remains well-defined during tiling. */
    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, &scalar, (uint32_t)sizeof(scalar));
    fx.mem.tile.active = 1;
    fx.mem.tile.out_h = 2;
    fx.mem.tile.out_w = 2;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "scalar constant Add supports a tile");
    assert_f32_array(fx.output, add_scalar_expected,
                     "tiled scalar constant Add output");

    /* NLC tiles use length and the final channel dimension, not NHWC width. */
    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, &scalar, (uint32_t)sizeof(scalar));
    fx.tensors[0].ndim = 3;
    fx.tensors[1].ndim = 3;
    fx.shapes[0] = fx.shapes[4] = 1;
    fx.shapes[1] = fx.shapes[5] = 2;
    fx.shapes[2] = fx.shapes[6] = 2;
    fx.mem.tile.active = 1;
    fx.mem.tile.out_h = 2;
    fx.mem.tile.out_w = 99;  /* ignored for NLC */
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "scalar constant Add supports a rank-3 tile");
    assert_f32_array(fx.output, add_scalar_expected,
                     "tiled rank-3 scalar constant Add output");
}

static void test_malformed_constant_binary_f32(void)
{
    printf("  test_malformed_constant_binary_f32...\n");

    const_binary_fixture_t fx;
    const float partial[2] = {2.0f, 3.0f};
    const float exact[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, partial, (uint32_t)sizeof(partial));
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "non-scalar non-exact constant is rejected");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, exact, (uint32_t)sizeof(exact));
    fx.mem.tile.active = 1;
    fx.mem.tile.out_h = 1;
    fx.mem.tile.out_w = 2;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "exact constant without a tile offset is rejected");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_MUL, exact, (uint32_t)sizeof(exact));
    fx.op.weight_idx = TIGRIS_NO_WEIGHT;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "one-input Mul without a constant is rejected");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, exact, (uint32_t)sizeof(exact));
    fx.op.weight_idx = 1;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "out-of-range constant index is rejected");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, exact, (uint32_t)sizeof(exact));
    fx.shapes[4] = 2;
    fx.shapes[5] = 2;
    fx.shapes[6] = 1;
    fx.shapes[7] = 1;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "same-byte-count output with a different shape is rejected");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_MUL, exact, (uint32_t)sizeof(exact));
    fx.op.num_inputs = 3;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == -1,
                "unsupported binary arity is rejected before index access");
}

/* Unsupported op type test */

static void test_unsupported_op(void)
{
    printf("  test_unsupported_op...\n");

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_UNKNOWN;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
    TEST_ASSERT(ret == -1, "unsupported op returns -1");
}

static void test_transpose(void)
{
    printf("  test_transpose...\n");
    int32_t shapes[] = {2, 3, 3, 2};
    uint16_t indices[] = {0, 1};
    float input[] = {1, 2, 3, 4, 5, 6};
    float expected[] = {1, 4, 2, 5, 3, 6};
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 2;
    tensors[0].dtype = 1; tensors[0].size_bytes = sizeof(input);
    tensors[1].shape_off = 2; tensors[1].ndim = 2;
    tensors[1].dtype = 1; tensors[1].size_bytes = sizeof(expected);
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_TRANSPOSE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_TRANSPOSE_PERM, 2, 0};
    const uint8_t perm[] = {1, 0};
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = indices; plan.shape_pool = shapes;
    plan.op_attributes = &attr; plan.op_attribute_data = perm;
    plan.num_op_attributes = 1;
    void *ptrs[2]; uint8_t fast[128], slow[128]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, 2, fast, sizeof(fast), slow, sizeof(slow));
    tigris_mem_alloc_fast(&mem, 0, sizeof(input));
    memcpy(ptrs[0], input, sizeof(input));
    tigris_mem_alloc_fast(&mem, 1, sizeof(expected));
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "transpose returns 0");
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_NEAR(((float *)ptrs[1])[i], expected[i], EPS,
                         "transpose value");
}

/* Main */

int main(void)
{
    printf("TiGrIS Kernel Tests\n\n");

    test_conv2d();
    test_conv_transpose_stride2();
    test_conv_transpose_pad_stride_skip();
    test_depthwise_conv2d();
    test_relu();
    test_relu6();
    test_add();
    test_global_avg_pool();
    test_avg_pool();
    test_fully_connected();
    test_reshape();
    test_max_pool();
    test_concat();
    test_resize_nearest();
    test_sigmoid();
    test_softmax();
    test_mul();
    test_constant_binary_f32();
    test_malformed_constant_binary_f32();
    test_transpose();
    test_unsupported_op();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

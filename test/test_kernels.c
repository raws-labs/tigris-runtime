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

static void test_conv_transpose_out_of_range_skip(void)
{
    printf("  test_conv_transpose_out_of_range_skip...\n");

    /* NHWC: 1x2x2x1 input, kernel 3x3, stride=1, pad=2 -> 1x3x3x1 output.
     * stride_h=stride_w=1 means num_h % SH and num_w % SW are always 0, so
     * every kh/kw tap passes the modulo check in kern_conv_transpose and the
     * only thing that can skip a tap is the ih/iw bounds check itself
     * (tigris_kernels.c:265,272). This isolates that branch from the
     * modulo-skip branch, which the earlier stride2/pad-stride-skip tests
     * exercise instead (see test_conv_transpose_pad_stride_skip; every
     * non-divisible num_h/num_w there is caught by the modulo check before
     * the bounds check line is ever reached).
     *
     * Gather-form derivation, ih = oh + PT - kh (PT=2, SH=1):
     *   oh=0: kh=0 -> ih=2 (>=IH=2, skip -- the bounds branch, hit
     *                directly, not via modulo); kh=1 -> ih=1 (valid);
     *                kh=2 -> ih=0 (valid).
     *   oh=1: kh=0 -> ih=3 (skip, bounds); kh=1 -> ih=2 (skip, bounds);
     *                kh=2 -> ih=1 (valid).
     *   oh=2: kh=0 -> ih=4 (skip, bounds); kh=1 -> ih=3 (skip, bounds);
     *                kh=2 -> ih=2 (skip, bounds). All three kh taps skip,
     *                so this row's sum never leaves the initial bias.
     * Width is the same structure (IW=2, PL=2, KW=3, SW=1), so ow=2 is
     * also an all-skip column. Cross-checked against the scatter-form
     * relation oh = ih*SH - PT + kh (ih in {0,1}, kh in {0,1,2}), which
     * gives the same three valid (ih,kh,oh) triples: (0,2,0), (1,1,0),
     * (1,2,1); same for w. Full expected grid (row-major oh,ow), bias=7,
     * X[ih,iw] = {1,2,3,4}, W[kh,kw] = {1..9} row-major:
     *   Y[0,0] = 7 + X[0,0]*W[2,2] + X[0,1]*W[2,1]
     *              + X[1,0]*W[1,2] + X[1,1]*W[1,1]
     *          = 7 + 1*9 + 2*8 + 3*6 + 4*5 = 7 + 9+16+18+20 = 70
     *   Y[0,1] = 7 + X[0,1]*W[2,2] + X[1,1]*W[1,2] = 7 + 2*9 + 4*6 = 49
     *   Y[0,2] = 7                                  (w-side all-skip)
     *   Y[1,0] = 7 + X[1,0]*W[2,2] + X[1,1]*W[2,1] = 7 + 3*9 + 4*8 = 66
     *   Y[1,1] = 7 + X[1,1]*W[2,2]                 = 7 + 4*9        = 43
     *   Y[1,2] = 7                                  (w-side all-skip)
     *   Y[2,*] = 7                                  (h-side all-skip)
     */
    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 3, 3, 1};

    float X[4] = {1, 2,
                  3, 4};
    float W[9] = {1, 2, 3,
                  4, 5, 6,
                  7, 8, 9};
    float B[1] = {7.0f};

    float expected[9] = {
        70, 49, 7,
        66, 43, 7,
         7,  7, 7,
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
    op.spatial.stride_h = 1; op.spatial.stride_w = 1;
    op.spatial.pad_top = 2; op.spatial.pad_bottom = 2;
    op.spatial.pad_left = 2; op.spatial.pad_right = 2;
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
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "conv_transpose out-of-range-skip output");
}

static void test_conv_transpose_tile_2d(void)
{
    printf("  test_conv_transpose_tile_2d...\n");

    /* NHWC: 1x4x4x1 input, kernel 2x2, stride=2, pad=0 -> 1x8x8x1 output.
     * Run the op WHOLE (tile inactive) to get Y_full, the oracle. Then run
     * it TILED for one interior 2D output tile and assert byte-equality
     * with the whole-op sub-region. No expected values are hardcoded.
     *
     * Interior 2D output tile [oh0..oh1) x [ow0..ow1) = [4..6) x [2..6):
     *   ih_lo = floor_div(oh0 + PT - (KH-1), SH) = floor_div(4+0-1, 2) = 1
     *   ih_hi = floor_div(oh1-1 + PT, SH)        = floor_div(5+0, 2)   = 2
     *   iw_lo = floor_div(ow0 + PL - (KW-1), SW) = floor_div(2+0-1, 2) = 0
     *   iw_hi = floor_div(ow1-1 + PL, SW)        = floor_div(5+0, 2)   = 2
     * loaded input rect: rows [1,3) cols [0,3)  -> in_h=2, in_w=3, ih0=1, iw0=0
     * effective pads:  PT_eff = oh0 + PT - ih0*SH = 4 - 2 = 2
     *                  PL_eff = ow0 + PL - iw0*SW = 2 - 0 = 2
     */
    int32_t x_shape[] = {1, 4, 4, 1};
    int32_t y_shape[] = {1, 8, 8, 1};

    float X_full[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    };
    /* Weight [OC=1, KH=2, KW=2, IC=1] (OHWI layout) */
    float W[4] = {1, 2,
                  3, 4};
    float B[1] = {0.5f};

    /* Build plan (shared by both the whole and the tiled dispatch: only
     * N/IC/OC are read from x_shape/y_shape once the tile override is
     * active, so the same "full" shape metadata is valid for both calls). */
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4;
    tensors[0].size_bytes = sizeof(X_full); tensors[0].dtype = 1;
    tensors[1].shape_off = 4; tensors[1].ndim = 4;
    tensors[1].size_bytes = 8 * 8 * sizeof(float); tensors[1].dtype = 1;

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

    /* 1) Run WHOLE: X[1,4,4,1], W OHWI [1,2,2,1] -> Y_full[1,8,8,1]. */
    void *ptrs_full[2] = {NULL, NULL};
    float Y_full[64];
    memset(Y_full, 0, sizeof(Y_full));
    ptrs_full[0] = X_full;
    ptrs_full[1] = Y_full;

    tigris_mem_t mem_full;
    memset(&mem_full, 0, sizeof(mem_full));
    mem_full.tensor_ptrs = ptrs_full;
    mem_full.num_tensors = 2;

    int ret_full = tigris_dispatch_kernel(&plan, &op, 0, &mem_full, NULL);
    TEST_ASSERT(ret_full == 0, "dispatch (whole) returns 0");

    /* 2) Build packed X_tile = X_full[1:3, 0:3] (in_h=2, in_w=3, IC=1). */
    float X_tile[6];
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++)
            X_tile[r * 3 + c] = X_full[(1 + r) * 4 + (0 + c)];

    /* 3) Run TILED for output tile [4..6) x [2..6): set mem.tile active=1,
     * width_tiled=1, out_h=2, out_w=4, in_h=2, in_w=3, pad_top=2,
     * pad_left=2, pad_bottom=0, pad_right=0, out_row_start=0,
     * in_row_start=0; point tensor_ptrs at X_tile / Y_tile[2*4]. */
    void *ptrs_tile[2] = {NULL, NULL};
    float Y_tile[2 * 4];
    memset(Y_tile, 0, sizeof(Y_tile));
    ptrs_tile[0] = X_tile;
    ptrs_tile[1] = Y_tile;

    tigris_mem_t mem_tile;
    memset(&mem_tile, 0, sizeof(mem_tile));
    mem_tile.tensor_ptrs = ptrs_tile;
    mem_tile.num_tensors = 2;
    mem_tile.tile.active = 1;
    mem_tile.tile.width_tiled = 1;
    mem_tile.tile.in_h = 2;
    mem_tile.tile.out_h = 2;
    mem_tile.tile.in_w = 3;
    mem_tile.tile.out_w = 4;
    mem_tile.tile.pad_top = 2;
    mem_tile.tile.pad_left = 2;
    mem_tile.tile.pad_bottom = 0;
    mem_tile.tile.pad_right = 0;
    mem_tile.tile.out_row_start = 0;
    mem_tile.tile.in_row_start = 0;

    /* 4) dispatch kern_conv_transpose, then: tiled == whole[4:6, 2:6]. */
    int ret_tile = tigris_dispatch_kernel(&plan, &op, 0, &mem_tile, NULL);
    TEST_ASSERT(ret_tile == 0, "dispatch (tile) returns 0");

    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 4; c++)
            TEST_ASSERT_NEAR(Y_tile[r * 4 + c], Y_full[(4 + r) * 8 + (2 + c)], EPS,
                             "conv_transpose 2D tile == whole sub-region");
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

static void test_sub(void)
{
    printf("  test_sub...\n");

    /* Sub does not commute, so the operand order is what this checks: a
     * swapped pair would give the negated result. */
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float expected[4] = {-9.0f, -18.0f, -27.0f, -36.0f};
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
    op.op_type = TIGRIS_OP_SUB;
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
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "sub output");

    /* The constant form carries no operand side, so it must fail closed. */
    op.num_inputs = 1;
    op.weight_idx = 0;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "constant-operand Sub is refused");
}

/* GlobalAvgPool test */

static void test_global_max_pool(void)
{
    printf("  test_global_max_pool...\n");

    /* NHWC: 1x3x3x2 -> 1x1x1x2, the same layout test_global_avg_pool uses.
     * Channel 0 holds 1..9 and channel 1 holds 10..18, so the maxima are the
     * last value of each channel and a per-channel mix-up would show. */
    int32_t x_shape[] = {1, 3, 3, 2};
    int32_t y_shape[] = {1, 1, 1, 2};

    float X[18];
    for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 3; w++) {
            X[(h * 3 + w) * 2 + 0] = (float)(h * 3 + w + 1);
            X[(h * 3 + w) * 2 + 1] = (float)(h * 3 + w + 10);
        }
    }
    /* A negative in channel 0 must not become the maximum. */
    X[0] = -42.0f;
    float expected[2] = {9.0f, 18.0f};

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
    op.op_type = TIGRIS_OP_GLOBAL_MAX;
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
        TEST_ASSERT_NEAR(out_buf[i], expected[i], EPS, "global_max_pool output");
}

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

/* HardSwish at -4, -1, 0, 2, 4: 0, -1/3, 0, 5/3, 4. */
static void test_hardswish(void)
{
    printf("  test_hardswish...\n");
    int32_t shape_pool[4] = {1, 5, 1, 1};
    float X[5] = {-4.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    float Y[5];
    const float expected[5] = {0.0f, -1.0f / 3.0f, 0.0f, 5.0f / 3.0f, 4.0f};
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 2; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 2; tensors[i].dtype = 1;
        tensors[i].size_bytes = sizeof(X);
    }
    uint16_t index_pool[2] = {0, 1};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_HARDSWISH;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = TIGRIS_NO_WEIGHT; op.bias_idx = TIGRIS_NO_WEIGHT;
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = index_pool; plan.shape_pool = shape_pool; plan.strings = g_strings;
    void *ptrs[2] = {X, Y};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs; mem.num_tensors = 2;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0, "hardswish runs");
    for (int i = 0; i < 5; i++)
        TEST_ASSERT_NEAR(Y[i], expected[i], EPS, "hardswish value");
}

/* A squeeze-and-excitation gate: one value per channel of the map it scales,
 * repeating every C elements, whole under a height band. */
static void test_per_channel_mul(void)
{
    printf("  test_per_channel_mul...\n");
    /* NHWC map 1x2x2x2 and gate 1x1x1x2. */
    int32_t shape_pool[12] = {1, 2, 2, 2,  1, 1, 1, 2,  1, 2, 2, 2};
    float A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float B[2] = {10, -1};
    float Y[8];
    const float expected[8] = {10, -2, 30, -4, 50, -6, 70, -8};
    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    const uint16_t offs[3] = {0, 4, 8};
    const uint32_t sizes[3] = {sizeof(A), sizeof(B), sizeof(Y)};
    for (int i = 0; i < 3; i++) {
        tensors[i].shape_off = offs[i]; tensors[i].ndim = 4; tensors[i].dtype = 1;
        tensors[i].size_bytes = sizes[i];
    }
    uint16_t index_pool[3] = {0, 1, 2};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_MUL;
    op.num_inputs = 2; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 2;
    op.weight_idx = TIGRIS_NO_WEIGHT; op.bias_idx = TIGRIS_NO_WEIGHT;
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 3; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = index_pool; plan.shape_pool = shape_pool; plan.strings = g_strings;
    void *ptrs[3] = {A, B, Y};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs; mem.num_tensors = 3;

    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0, "per-channel Mul runs");
    for (int i = 0; i < 8; i++)
        TEST_ASSERT_NEAR(Y[i], expected[i], EPS, "per-channel Mul value");

    /* The second row alone under a height band: the map is offset by the band's
     * row, the gate is not. */
    memset(Y, 0, sizeof(Y));
    ptrs[0] = A + 4;
    ptrs[2] = Y;
    mem.tile.active = 1;
    mem.tile.in_h = 1; mem.tile.out_h = 1;
    mem.tile.in_w = 2; mem.tile.out_w = 2;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "per-channel Mul runs in a band");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(Y[i], expected[4 + i], EPS, "per-channel Mul band value");
    mem.tile.active = 0;
    ptrs[0] = A;

    /* A row-wise operand is neither full-shape nor per-channel. */
    shape_pool[5] = 2; shape_pool[6] = 1; shape_pool[7] = 1;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "a row-wise operand is refused");
}

/* Concat on the last stored axis at rank 3, and with a constant leading part
 * (how a learned token is prepended to a sequence). */
static void test_concat_last_axis_variants(void)
{
    printf("  test_concat_last_axis_variants...\n");

    /* NLC: A 1x2x1, B 1x2x2 -> Y 1x2x3; K is the constant leading part. */
    int32_t shape_pool[12] = {1, 2, 1,  1, 2, 2,  1, 2, 3,  1, 2, 4};
    float A[2] = {1.0f, 2.0f};
    float B[4] = {10.0f, 11.0f, 20.0f, 21.0f};
    float K[2] = {100.0f, 200.0f};
    float Y[8];

    tigris_tensor_t tensors[4];
    memset(tensors, 0, sizeof(tensors));
    const uint16_t offs[4] = {0, 3, 6, 9};
    const uint32_t sizes[4] = {sizeof(A), sizeof(B), 6 * sizeof(float),
                               8 * sizeof(float)};
    for (int i = 0; i < 4; i++) {
        tensors[i].shape_off = offs[i];
        tensors[i].ndim = 3;
        tensors[i].dtype = 1;
        tensors[i].size_bytes = sizes[i];
    }

    tigris_weight_entry_t weight;
    memset(&weight, 0, sizeof(weight));
    weight.offset = 0;
    weight.size_bytes = sizeof(K);

    uint16_t index_pool[5] = {0, 1, 2, 1, 3};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_CONCAT;
    op.num_inputs = 2; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 2;
    op.weight_idx = TIGRIS_NO_WEIGHT;
    op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 4; header.num_ops = 1; header.num_weights = 1;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape_pool;
    plan.strings = g_strings;
    plan.weight_entries = &weight;
    plan.weight_blob = (const uint8_t *)K;

    void *ptrs[4] = {A, B, Y, Y};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 4;

    const float rank3[6] = {1.0f, 10.0f, 11.0f, 2.0f, 20.0f, 21.0f};
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "rank-3 concat succeeds");
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_NEAR(Y[i], rank3[i], EPS, "rank-3 concat output");

    /* [K | B | A]: the constant takes the channels the inputs leave over. */
    index_pool[3] = 1; index_pool[4] = 3;
    op.inputs_off = 3; op.num_inputs = 1; op.outputs_off = 4;
    op.weight_idx = 0;
    tensors[3].size_bytes = 6 * sizeof(float);
    shape_pool[11] = 3;
    const float lead[6] = {100.0f, 10.0f, 11.0f, 200.0f, 20.0f, 21.0f};
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "constant leading part succeeds");
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_NEAR(Y[i], lead[i], EPS, "constant leading part output");

    mem.tile.active = 1;
    mem.tile.out_h = 2;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "constant leading part refuses a tile");
    mem.tile.active = 0;

    shape_pool[11] = 2;  /* inputs cover every channel: nothing left over */
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "constant with no leading extent is refused");

    shape_pool[11] = 3;
    ptrs[1] = NULL;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "missing concat input is refused");
}

/* Bilinear Resize, checked against ONNX Runtime (opset 13, mode linear) on an
 * NHWC 1x2x3x2 input holding 1..12, scale 2 on both spatial axes. */
static const float k_linear_half_pixel[48] = {
    1, 2, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f, 4.5f, 4.5f, 5.5f, 5, 6,
    2.5f, 3.5f, 3, 4, 4, 5, 5, 6, 6, 7, 6.5f, 7.5f,
    5.5f, 6.5f, 6, 7, 7, 8, 8, 9, 9, 10, 9.5f, 10.5f,
    7, 8, 7.5f, 8.5f, 8.5f, 9.5f, 9.5f, 10.5f, 10.5f, 11.5f, 11, 12,
};
static const float k_linear_asymmetric[48] = {
    1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 5, 6,
    4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 8, 9,
    7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 11, 12,
    7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 11, 12,
};

static int run_resize_linear(uint16_t convention, const float *X, float *Y,
                             const tigris_tile_ctx_t *tile)
{
    int32_t shape_pool[8] = {1, 2, 3, 2, 1, 4, 6, 2};
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 1;
    tensors[0].size_bytes = 12 * sizeof(float);
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 1;
    tensors[1].size_bytes = 48 * sizeof(float);

    uint16_t index_pool[2] = {0, 1};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RESIZE_LINEAR;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.spatial.stride_h = 2; op.spatial.stride_w = 2;
    op.spatial.kernel_h = convention;  /* 0 half-pixel, 1 asymmetric */
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

    void *ptrs[2] = {(void *)X, Y};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;
    if (tile)
        mem.tile = *tile;
    return tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL);
}

static void test_resize_linear(void)
{
    printf("  test_resize_linear...\n");

    float X[12];
    float Y[48];
    for (int i = 0; i < 12; i++)
        X[i] = (float)(i + 1);

    TEST_ASSERT(run_resize_linear(0, X, Y, NULL) == 0,
                "half-pixel bilinear succeeds");
    for (int i = 0; i < 48; i++)
        TEST_ASSERT_NEAR(Y[i], k_linear_half_pixel[i], EPS,
                         "half-pixel bilinear matches ONNX Runtime");

    TEST_ASSERT(run_resize_linear(1, X, Y, NULL) == 0,
                "asymmetric bilinear succeeds");
    for (int i = 0; i < 48; i++)
        TEST_ASSERT_NEAR(Y[i], k_linear_asymmetric[i], EPS,
                         "asymmetric bilinear matches ONNX Runtime");

    /* A height tile: output row 3 alone, from input row 1 alone. The origins
     * place the band in the full image, so the result is that row of the
     * untiled output. */
    tigris_tile_ctx_t tile;
    memset(&tile, 0, sizeof(tile));
    tile.active = 1;
    tile.in_h = 1;  tile.out_h = 1;
    tile.in_w = 3;  tile.out_w = 6;
    tile.out_row_origin = 3;
    tile.in_row_origin = 1;
    TEST_ASSERT(run_resize_linear(0, X + 6, Y, &tile) == 0,
                "tiled bilinear succeeds");
    for (int i = 0; i < 12; i++)
        TEST_ASSERT_NEAR(Y[i], k_linear_half_pixel[36 + i], EPS,
                         "tiled bilinear reproduces its rows");

    TEST_ASSERT(run_resize_linear(2, X, Y, NULL) != 0,
                "unknown coordinate convention is refused");
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

static void test_erf(void)
{
    printf("  test_erf...\n");

    float X[4] = {0.0f, 0.5f, -1.0f, 2.0f};
    float expected[4] = {0.0f, 0.5204999f, -0.8427008f, 0.9953223f};
    int32_t shape[] = {1, 2, 2, 1};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 2; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(X); tensors[i].dtype = 1;
    }
    uint16_t index_pool[2] = {0, 1};

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_ERF;
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

    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "dispatch returns 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(out_buf[i], expected[i], 1e-5f, "erf output");
}

/**
 * Layer normalization over the trailing axis, checked against the definition
 * rather than against a recorded output: each row must come out zero-mean and
 * unit-variance before scale and bias, so the assertions below read the
 * result back through that.
 */
static void test_layer_norm(void)
{
    printf("  test_layer_norm...\n");

    /* Two rows of four, deliberately different in mean and in spread. */
    float X[8] = {1.0f, 2.0f, 3.0f, 4.0f, -6.0f, -2.0f, 2.0f, 6.0f};
    float scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t shape[] = {1, 1, 2, 4};

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 2; i++) {
        tensors[i].shape_off = 0; tensors[i].ndim = 4;
        tensors[i].size_bytes = sizeof(X); tensors[i].dtype = 1;
    }
    uint16_t index_pool[2] = {0, 1};

    tigris_weight_entry_t weights[2];
    memset(weights, 0, sizeof(weights));
    weights[0].offset = 0;
    weights[0].size_bytes = sizeof(scale);
    weights[1].offset = sizeof(scale);
    weights[1].size_bytes = sizeof(bias);
    uint8_t blob[sizeof(scale) + sizeof(bias)];
    memcpy(blob, scale, sizeof(scale));
    memcpy(blob + sizeof(scale), bias, sizeof(bias));

    float epsilon = 1e-5f;
    tigris_op_attribute_t attribute;
    memset(&attribute, 0, sizeof(attribute));
    attribute.op_index = 0;
    attribute.type = TIGRIS_OP_ATTR_EPSILON;
    attribute.data_len = (uint8_t)sizeof(epsilon);
    attribute.data_offset = 0;
    uint8_t attr_data[sizeof(epsilon)];
    memcpy(attr_data, &epsilon, sizeof(epsilon));

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_LAYER_NORM;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = 0;
    op.bias_idx = 1;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1; header.num_weights = 2;

    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = &op;
    plan.index_pool = index_pool;
    plan.shape_pool = shape;
    plan.strings = g_strings;
    plan.weight_entries = weights;
    plan.weight_blob = blob;
    plan.op_attributes = &attribute;
    plan.op_attribute_data = attr_data;
    plan.num_op_attributes = 1;

    void *ptrs[2];
    float out_buf[8];
    ptrs[0] = X; ptrs[1] = out_buf;

    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;

    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "dispatch returns 0");
    for (int row = 0; row < 2; row++) {
        float sum = 0.0f;
        float sq = 0.0f;
        for (int c = 0; c < 4; c++) {
            sum += out_buf[row * 4 + c];
            sq += out_buf[row * 4 + c] * out_buf[row * 4 + c];
        }
        TEST_ASSERT_NEAR(sum / 4.0f, 0.0f, 1e-5f, "normalized row is centered");
        TEST_ASSERT_NEAR(sq / 4.0f, 1.0f, 1e-4f, "normalized row has unit variance");
    }
    /* Order is preserved, so the two rows normalize to the same shape even
     * though their spreads differ by a factor of four. */
    TEST_ASSERT(out_buf[0] < out_buf[1] && out_buf[1] < out_buf[2],
                "normalization preserves order");

    /* A missing variance floor is refused rather than defaulted. */
    plan.num_op_attributes = 0;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "a missing variance floor is refused");
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

    /* One value per channel: two positions of two channels, channels
     * innermost, so the constant repeats every two elements. */
    const float per_channel[2] = {10.0f, -1.0f};
    const float add_channel_expected[4] = {11.0f, 1.0f, 13.0f, 3.0f};
    const float mul_channel_expected[4] = {10.0f, -2.0f, 30.0f, -4.0f};
    init_const_binary_fixture(
        &fx, TIGRIS_OP_ADD, per_channel, (uint32_t)sizeof(per_channel));
    fx.shapes[2] = fx.shapes[6] = 1;
    fx.shapes[3] = fx.shapes[7] = 2;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "per-channel constant Add succeeds");
    assert_f32_array(fx.output, add_channel_expected,
                     "per-channel constant Add output");

    init_const_binary_fixture(
        &fx, TIGRIS_OP_MUL, per_channel, (uint32_t)sizeof(per_channel));
    fx.shapes[2] = fx.shapes[6] = 1;
    fx.shapes[3] = fx.shapes[7] = 2;
    fx.mem.tile.active = 1;
    fx.mem.tile.out_h = 2;
    fx.mem.tile.out_w = 1;
    TEST_ASSERT(tigris_dispatch_kernel(
                    &fx.plan, &fx.op, 0, &fx.mem, NULL) == 0,
                "per-channel constant Mul supports a tile");
    assert_f32_array(fx.output, mul_channel_expected,
                     "tiled per-channel constant Mul output");
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


/* The transpose kernel moves elements with typed stores where the pointers
 * allow it and memcpy where they do not. Both have to agree with the plain
 * index walk, on every rank and on both paths through the kernel. */
static void transpose_reference(
    uint8_t *out, const uint8_t *in, int rank, const int32_t *in_shape,
    const uint8_t *perm, uint32_t elem_size)
{
    int32_t out_shape[4];
    uint32_t elements = 1;
    for (int a = 0; a < rank; a++) {
        out_shape[a] = in_shape[perm[a]];
        elements *= (uint32_t)in_shape[a];
    }
    for (uint32_t o = 0; o < elements; o++) {
        uint32_t remainder = o;
        uint32_t in_index = 0;
        for (int a = rank - 1; a >= 0; a--) {
            uint32_t coord = remainder % (uint32_t)out_shape[a];
            remainder /= (uint32_t)out_shape[a];
            uint32_t stride = 1;
            for (int b = perm[a] + 1; b < rank; b++)
                stride *= (uint32_t)in_shape[b];
            in_index += coord * stride;
        }
        memcpy(out + (size_t)o * elem_size,
               in + (size_t)in_index * elem_size, elem_size);
    }
}

static void run_transpose_case_block(
    int rank, const int32_t *in_shape, const uint8_t *perm,
    uint32_t elem_size, int misalign, int banded,
    int32_t outer, int32_t rows, int32_t cols, int32_t block,
    const char *what)
{
    int32_t shapes[8];
    uint32_t elements = 1;
    for (int a = 0; a < rank; a++) {
        shapes[a] = in_shape[a];
        shapes[4 + a] = in_shape[perm[a]];
        elements *= (uint32_t)in_shape[a];
    }

    static uint8_t in_buf[1024];
    static uint8_t out_buf[1024];
    static uint8_t ref_buf[1024];
    uint8_t *in = in_buf + misalign;
    uint8_t *out = out_buf + misalign;
    for (uint32_t i = 0; i < elements * elem_size; i++)
        in[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
    memset(out, 0, elements * elem_size);
    transpose_reference(ref_buf, in, rank, in_shape, perm, elem_size);

    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = (uint8_t)rank;
    tensors[0].size_bytes = elements * elem_size;
    tensors[1].shape_off = 4; tensors[1].ndim = (uint8_t)rank;
    tensors[1].size_bytes = elements * elem_size;

    uint16_t indices[2] = {0, 1};
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_TRANSPOSE;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    tigris_op_attribute_t attr = {
        0, TIGRIS_OP_ATTR_TRANSPOSE_PERM, (uint8_t)rank, 0};
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = indices; plan.shape_pool = shapes;
    plan.op_attributes = &attr; plan.op_attribute_data = perm;
    plan.num_op_attributes = 1;

    void *ptrs[2] = { in, out };
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs; mem.num_tensors = 2;
    if (banded) {
        mem.tile.active = 1;
        mem.tile.transposed_tile = 1;
        mem.tile.transpose_outer = outer;
        mem.tile.transpose_block = block;
        mem.tile.in_h = rows; mem.tile.in_w = cols;
        mem.tile.out_h = cols; mem.tile.out_w = rows;
    }

    TEST_ASSERT(tigris_transpose_execute(&plan, &op, 0, &mem) == 0, what);
    TEST_ASSERT(memcmp(out, ref_buf, elements * elem_size) == 0, what);
}

static void run_transpose_case(
    int rank, const int32_t *in_shape, const uint8_t *perm,
    uint32_t elem_size, int misalign, int banded,
    int32_t rows, int32_t cols, const char *what)
{
    run_transpose_case_block(rank, in_shape, perm, elem_size, misalign,
                             banded, 1, rows, cols, 1, what);
}

/* A transpose the band path accepts swaps two adjacent groups of axes and
 * leaves the rest in order, so the axes behind the swapped pair travel with
 * the element and the slice is a matrix of blocks. An attention block's head
 * permutation is the first one with such a suffix; the three the path used to
 * name were the cases with none. */
static void test_transpose_carries_a_block(void)
{
    printf("  test_transpose_carries_a_block...\n");

    /* [1, 6, 2, 3] -> [1, 2, 6, 3] under stored perm (0, 2, 1, 3): a 6 by 2
     * transpose carrying three elements at each position. */
    const int32_t shape[4] = {1, 6, 2, 3};
    const uint8_t perm[4] = {0, 2, 1, 3};
    run_transpose_case_block(4, shape, perm, 4, 0, 1, 1, 6, 2, 3,
                             "rank-4 float banded with a block of three");
    run_transpose_case_block(4, shape, perm, 1, 0, 1, 1, 6, 2, 3,
                             "rank-4 int8 banded with a block of three");

    /* The same permutation with a batch ahead of the swapped pair. */
    const int32_t batched[4] = {4, 6, 2, 3};
    run_transpose_case_block(4, batched, perm, 4, 0, 1, 4, 6, 2, 3,
                             "a batch ahead of the swapped pair");
}

static void test_transpose_element_paths(void)
{
    printf("  test_transpose_element_paths...\n");

    const int32_t r4[4] = {1, 5, 6, 4};
    const uint8_t p4[4] = {0, 3, 1, 2};
    run_transpose_case(4, r4, p4, 4, 0, 0, 0, 0,
                       "rank-4 float general walk");
    run_transpose_case(4, r4, p4, 1, 0, 0, 0, 0,
                       "rank-4 int8 general walk");
    /* One byte in defeats the typed stores and takes the memcpy fallback. */
    run_transpose_case(4, r4, p4, 4, 1, 0, 0, 0,
                       "rank-4 float misaligned falls back");

    const int32_t r3[3] = {1, 12, 7};
    const uint8_t p3[3] = {0, 2, 1};
    run_transpose_case(3, r3, p3, 4, 0, 1, 12, 7,
                       "rank-3 float banded");
    run_transpose_case(3, r3, p3, 1, 0, 1, 12, 7,
                       "rank-3 int8 banded");
    run_transpose_case(3, r3, p3, 4, 1, 1, 12, 7,
                       "rank-3 float banded misaligned falls back");

    const int32_t r4b[4] = {1, 4, 3, 5};
    const uint8_t p4b[4] = {0, 2, 3, 1};
    run_transpose_case(4, r4b, p4b, 4, 0, 1, 4, 15,
                       "rank-4 float banded the other way");
}


/* A row band cuts the second to last axis, so on a rank-4 tensor the head
 * axis is batch that the band spans rather than cuts. The element count a
 * pointwise kernel works from has to include it: while a band meant one
 * matrix there was no batch to multiply by, and the first rank-4 band left
 * every batch but the first untouched. */
static void test_pointwise_row_band_covers_every_batch(void)
{
    printf("  test_pointwise_row_band_covers_every_batch...\n");

    enum { BATCH = 3, ROWS = 4, COLS = 2, BAND = 2 };
    int32_t shape[] = {1, BATCH, ROWS, COLS};
    uint16_t indices[] = {0, 1};
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    for (int i = 0; i < 2; i++) {
        tensors[i].shape_off = 0;
        tensors[i].ndim = 4;
        tensors[i].dtype = 1;
        tensors[i].flags = TIGRIS_TENSOR_LINEAR;
        tensors[i].size_bytes =
            (uint32_t)(BATCH * ROWS * COLS) * sizeof(float);
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }

    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_RELU;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    op.weight_idx = TIGRIS_NO_WEIGHT; op.bias_idx = TIGRIS_NO_WEIGHT;

    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = indices; plan.shape_pool = shape;

    /* The band buffers hold BAND rows for every batch, laid out the way the
     * band gather writes them. */
    float in[BATCH * BAND * COLS];
    float out[BATCH * BAND * COLS];
    for (int i = 0; i < BATCH * BAND * COLS; i++)
        in[i] = -1.0f - (float)i;
    for (int i = 0; i < BATCH * BAND * COLS; i++)
        out[i] = 12345.0f;

    void *ptrs[2] = { in, out };
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs; mem.num_tensors = 2;
    mem.tile.active = 1;
    mem.tile.row_tiled = 1;
    mem.tile.in_h = BAND;
    mem.tile.out_h = BAND;
    mem.tile.in_w = 1;
    mem.tile.out_w = 1;

    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "relu over a rank-4 row band returns 0");
    for (int i = 0; i < BATCH * BAND * COLS; i++)
        TEST_ASSERT_NEAR(out[i], 0.0f, EPS,
                         "every batch of the band was written");
}

/* Main */

/* A mean over one axis of a rank-3 tensor. The axis the plan names is a
 * serialized one, so the same kernel collapses a sequence's token axis and a
 * feature map's channel axis; both are checked, plus the two ways a plan may
 * state the result's rank. */
static void run_reduce_mean_case(
    uint8_t axis, uint8_t out_ndim, const float *expected, int count)
{
    /* [2, 3, 2], values 0..11 */
    int32_t shapes[] = {2, 3, 2, 0, 0, 0};
    float input[12];
    for (int i = 0; i < 12; i++)
        input[i] = (float)i;
    int32_t kept[3];
    uint8_t written = 0;
    for (uint8_t a = 0; a < 3u; a++) {
        if (a == axis && out_ndim == 2u)
            continue;
        kept[written++] = (a == axis) ? 1 : shapes[a];
    }
    for (uint8_t a = 0; a < written; a++)
        shapes[3 + a] = kept[a];

    uint16_t indices[] = {0, 1};
    tigris_tensor_t tensors[2];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 3;
    tensors[0].dtype = 1; tensors[0].size_bytes = sizeof(input);
    tensors[1].shape_off = 3; tensors[1].ndim = out_ndim;
    tensors[1].dtype = 1;
    tensors[1].size_bytes = (uint32_t)count * sizeof(float);
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_REDUCE_MEAN;
    op.num_inputs = 1; op.num_outputs = 1;
    op.inputs_off = 0; op.outputs_off = 1;
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_AXES, 1, 0};
    const uint8_t axes[] = {axis};
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 2; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = indices; plan.shape_pool = shapes;
    plan.op_attributes = &attr; plan.op_attribute_data = axes;
    plan.num_op_attributes = 1;
    void *ptrs[2]; uint8_t fast[256], slow[256]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, 2, fast, sizeof(fast), slow, sizeof(slow));
    tigris_mem_alloc_fast(&mem, 0, sizeof(input));
    memcpy(ptrs[0], input, sizeof(input));
    tigris_mem_alloc_fast(&mem, 1, tensors[1].size_bytes);
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "reduce mean returns 0");
    for (int i = 0; i < count; i++)
        TEST_ASSERT_NEAR(((float *)ptrs[1])[i], expected[i], EPS,
                         "reduce mean value");
}

static void test_reduce_mean(void)
{
    printf("  test_reduce_mean...\n");
    /* Mean over the middle axis: rows 0,2,4 and 1,3,5 of each batch. */
    const float over_rows[] = {2.0f, 3.0f, 8.0f, 9.0f};
    run_reduce_mean_case(1, 3, over_rows, 4);
    run_reduce_mean_case(1, 2, over_rows, 4);
    /* Mean over the innermost axis pairs neighbours. */
    const float over_inner[] = {0.5f, 2.5f, 4.5f, 6.5f, 8.5f, 10.5f};
    run_reduce_mean_case(2, 3, over_inner, 6);
    /* Mean over the outermost axis averages the two batches. */
    const float over_outer[] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    run_reduce_mean_case(0, 3, over_outer, 6);
}


/* A split hands out contiguous runs of its input, so the parts are the input
 * read in order and nothing is interleaved. Uneven parts are here because the
 * kernel sizes each one from its own tensor rather than dividing. */
static void test_split(void)
{
    printf("  test_split...\n");
    int32_t shapes[] = {3, 2, 2, 2, 1, 2};
    float input[6] = {1, 2, 3, 4, 5, 6};
    uint16_t indices[] = {0, 1, 2};
    tigris_tensor_t tensors[3];
    memset(tensors, 0, sizeof(tensors));
    tensors[0].shape_off = 0; tensors[0].ndim = 2;
    tensors[0].dtype = 1; tensors[0].size_bytes = sizeof(input);
    tensors[1].shape_off = 2; tensors[1].ndim = 2;
    tensors[1].dtype = 1; tensors[1].size_bytes = 4u * sizeof(float);
    tensors[2].shape_off = 4; tensors[2].ndim = 2;
    tensors[2].dtype = 1; tensors[2].size_bytes = 2u * sizeof(float);
    tigris_op_t op;
    memset(&op, 0, sizeof(op));
    op.op_type = TIGRIS_OP_SPLIT;
    op.num_inputs = 1; op.num_outputs = 2;
    op.inputs_off = 0; op.outputs_off = 1;
    tigris_file_header_t header;
    memset(&header, 0, sizeof(header));
    header.num_tensors = 3; header.num_ops = 1;
    tigris_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    plan.header = &header; plan.tensors = tensors; plan.ops = &op;
    plan.index_pool = indices; plan.shape_pool = shapes;
    void *ptrs[3]; uint8_t fast[256], slow[256]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, 3, fast, sizeof(fast), slow, sizeof(slow));
    tigris_mem_alloc_fast(&mem, 0, sizeof(input));
    memcpy(ptrs[0], input, sizeof(input));
    tigris_mem_alloc_fast(&mem, 1, tensors[1].size_bytes);
    tigris_mem_alloc_fast(&mem, 2, tensors[2].size_bytes);
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) == 0,
                "split returns 0");
    const float first[] = {1, 2, 3, 4};
    const float second[] = {5, 6};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_NEAR(((float *)ptrs[1])[i], first[i], EPS, "split part 0");
    for (int i = 0; i < 2; i++)
        TEST_ASSERT_NEAR(((float *)ptrs[2])[i], second[i], EPS, "split part 1");

    /* A part with nowhere to go is refused rather than written past. */
    mem.tensor_ptrs[2] = NULL;
    TEST_ASSERT(tigris_dispatch_kernel(&plan, &op, 0, &mem, NULL) != 0,
                "split without a destination is refused");
}


int main(void)
{
    printf("TiGrIS Kernel Tests\n\n");

    test_conv2d();
    test_conv_transpose_stride2();
    test_conv_transpose_pad_stride_skip();
    test_conv_transpose_out_of_range_skip();
    test_conv_transpose_tile_2d();
    test_depthwise_conv2d();
    test_relu();
    test_relu6();
    test_add();
    test_sub();
    test_global_avg_pool();
    test_global_max_pool();
    test_avg_pool();
    test_fully_connected();
    test_reshape();
    test_max_pool();
    test_concat();
    test_concat_last_axis_variants();
    test_hardswish();
    test_per_channel_mul();
    test_resize_nearest();
    test_resize_linear();
    test_sigmoid();
    test_erf();
    test_layer_norm();
    test_reduce_mean();
    test_split();
    test_softmax();
    test_mul();
    test_constant_binary_f32();
    test_malformed_constant_binary_f32();
    test_transpose();
    test_transpose_element_paths();
    test_transpose_carries_a_block();
    test_pointwise_row_band_covers_every_batch();
    test_unsupported_op();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

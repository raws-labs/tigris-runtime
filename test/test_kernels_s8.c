/**
 * @file test_kernels_s8.c
 * @brief Unit tests for int8 quantized kernels.
 *
 * Tests use hand-computed expected values with known quant params.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
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

#define TEST_ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) != (b)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %ld, expected %ld)\n", \
                __func__, __LINE__, msg, (long)(a), (long)(b)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* Synthetic plan builder helpers */

/*
 * For unit testing, we build a minimal plan in memory with just enough
 * structure for kernels to read shapes, quant params, and weight data.
 */

/* Max sizes for our tiny test plans */
#define MAX_TENSORS  8
#define MAX_OPS      4
#define MAX_SHAPES  32
#define MAX_INDICES 32
#define MAX_STRINGS 256
#define MAX_WEIGHTS  4
#define MAX_QP       8
#define MAX_QD      32
#define MAX_WEIGHT_BLOB 4096

/* We use static arrays instead of allocations */
static tigris_file_header_t test_header;
static tigris_tensor_t test_tensors[MAX_TENSORS];
static tigris_op_t test_ops[MAX_OPS];
static int32_t test_shapes[MAX_SHAPES];
static uint16_t test_indices[MAX_INDICES];
static char test_strings[MAX_STRINGS];
static tigris_weight_entry_t test_weight_entries[MAX_WEIGHTS];
static uint8_t test_weight_blob[MAX_WEIGHT_BLOB];
static tigris_quant_param_t test_qp[MAX_QP];
static int32_t test_qd[MAX_QD];

static int shape_cursor, index_cursor, string_cursor, weight_cursor, qp_cursor, qd_cursor, weight_blob_cursor;

static void plan_reset(void)
{
    memset(&test_header, 0, sizeof(test_header));
    memset(test_tensors, 0, sizeof(test_tensors));
    memset(test_ops, 0, sizeof(test_ops));
    memset(test_shapes, 0, sizeof(test_shapes));
    memset(test_indices, 0, sizeof(test_indices));
    memset(test_strings, 0, sizeof(test_strings));
    memset(test_weight_entries, 0, sizeof(test_weight_entries));
    memset(test_weight_blob, 0, sizeof(test_weight_blob));
    memset(test_qp, 0, sizeof(test_qp));
    memset(test_qd, 0, sizeof(test_qd));
    shape_cursor = 0;
    index_cursor = 0;
    string_cursor = 1;  /* offset 0 = empty string */
    weight_cursor = 0;
    qp_cursor = 0;
    qd_cursor = 0;
    weight_blob_cursor = 0;
}

static uint16_t add_shape(const int32_t *dims, int ndim)
{
    uint16_t off = (uint16_t)shape_cursor;
    for (int i = 0; i < ndim; i++)
        test_shapes[shape_cursor++] = dims[i];
    return off;
}

static uint16_t add_indices(const uint16_t *vals, int count)
{
    uint16_t off = (uint16_t)index_cursor;
    for (int i = 0; i < count; i++)
        test_indices[index_cursor++] = vals[i];
    return off;
}

static uint32_t add_string(const char *s)
{
    uint32_t off = (uint32_t)string_cursor;
    int len = (int)strlen(s);
    memcpy(test_strings + string_cursor, s, len + 1);
    string_cursor += len + 1;
    return off;
}

static uint16_t add_tensor(tigris_plan_t *plan, const char *name,
                           const int32_t *shape, int ndim, uint8_t dtype,
                           uint32_t size_bytes, uint16_t qp_idx)
{
    (void)plan;
    uint16_t idx = test_header.num_tensors++;
    test_tensors[idx].name_str = add_string(name);
    test_tensors[idx].shape_off = add_shape(shape, ndim);
    test_tensors[idx].ndim = (uint8_t)ndim;
    test_tensors[idx].dtype = dtype;
    test_tensors[idx].size_bytes = size_bytes;
    test_tensors[idx].quant_param_idx = qp_idx;
    test_tensors[idx].flags = 0;
    return idx;
}

static uint16_t add_weight(const void *data, uint32_t size, const char *name)
{
    uint16_t idx = (uint16_t)weight_cursor++;
    test_weight_entries[idx].name_str = add_string(name);
    test_weight_entries[idx].offset = (uint32_t)weight_blob_cursor;
    test_weight_entries[idx].size_bytes = size;
    memcpy(test_weight_blob + weight_blob_cursor, data, size);
    weight_blob_cursor += (int)size;
    test_header.num_weights = weight_cursor;
    return idx;
}

static uint16_t add_quant_param(float scale, int32_t zp, uint16_t num_ch,
                                const int32_t *multipliers, const int32_t *shifts)
{
    uint16_t idx = (uint16_t)qp_cursor++;
    test_qp[idx].scale = scale;
    test_qp[idx].zero_point = zp;
    test_qp[idx].num_channels = num_ch;
    if (multipliers && shifts) {
        test_qp[idx].multiplier_off = (uint16_t)qd_cursor;
        for (uint16_t i = 0; i < num_ch; i++)
            test_qd[qd_cursor++] = multipliers[i];
        test_qp[idx].shift_off = (uint16_t)qd_cursor;
        for (uint16_t i = 0; i < num_ch; i++)
            test_qd[qd_cursor++] = shifts[i];
    }
    return idx;
}

static void build_plan(tigris_plan_t *plan)
{
    plan->header = &test_header;
    plan->tensors = test_tensors;
    plan->ops = test_ops;
    plan->stages = NULL;
    plan->tile_plans = NULL;
    plan->index_pool = test_indices;
    plan->shape_pool = test_shapes;
    plan->strings = test_strings;
    plan->weight_entries = test_weight_entries;
    plan->weight_blob = test_weight_blob;
    plan->quant_params = test_qp;
    plan->quant_data = test_qd;
    plan->num_quant_params = (uint16_t)qp_cursor;
    plan->model_inputs = NULL;
    plan->model_outputs = NULL;
}

/* Tests */

static void test_relu_s8(void)
{
    printf("  test_relu_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /* Quant param: scale=0.5, zp=-2 */
    uint16_t qp = add_quant_param(0.5f, -2, 1, NULL, NULL);

    int32_t shape[] = {1, 4};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 4, qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 4, TIGRIS_NO_QUANT_PARAM);

    test_ops[0].op_type = TIGRIS_OP_RELU;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;

    build_plan(&plan);

    /* Set up mem */
    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));

    /* Input: [-5, -2, 0, 10]. zp=-2, so relu clamps to >= -2 */
    int8_t input_data[] = {-5, -2, 0, 10};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 4);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "relu_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], -2, "relu(-5, zp=-2) = -2");
    TEST_ASSERT_EQ(out[1], -2, "relu(-2, zp=-2) = -2");
    TEST_ASSERT_EQ(out[2], 0, "relu(0, zp=-2) = 0");
    TEST_ASSERT_EQ(out[3], 10, "relu(10, zp=-2) = 10");
}

static void test_relu6_s8(void)
{
    printf("  test_relu6_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /* Quant param: scale=0.1, zp=0. quantized(6) = 6/0.1 + 0 = 60 */
    uint16_t qp = add_quant_param(0.1f, 0, 1, NULL, NULL);

    int32_t shape[] = {1, 4};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 4, qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 4, TIGRIS_NO_QUANT_PARAM);

    test_ops[0].op_type = TIGRIS_OP_RELU6;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));

    /* scale=0.1, zp=0: quantized range [0, 60] */
    int8_t input_data[] = {-10, 0, 30, 80};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 4);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "relu6_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 0, "relu6(-10) = 0");
    TEST_ASSERT_EQ(out[1], 0, "relu6(0) = 0");
    TEST_ASSERT_EQ(out[2], 30, "relu6(30) = 30 (in range)");
    TEST_ASSERT_EQ(out[3], 60, "relu6(80) = 60 (clamped)");
}

static void test_reshape_s8(void)
{
    printf("  test_reshape_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    int32_t in_shape[] = {1, 2, 3};
    int32_t out_shape[] = {1, 6};
    uint16_t t_in = add_tensor(&plan, "in", in_shape, 3, 3, 6, TIGRIS_NO_QUANT_PARAM);
    uint16_t t_out = add_tensor(&plan, "out", out_shape, 2, 3, 6, TIGRIS_NO_QUANT_PARAM);

    test_ops[0].op_type = TIGRIS_OP_RESHAPE;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));

    int8_t input_data[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_in, 6);
    memcpy(ptrs[t_in], input_data, 6);
    tigris_mem_alloc_fast(&mem, t_out, 6);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "reshape_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQ(out[i], input_data[i], "reshape preserves data");
    }
}

static void test_conv2d_s8_per_tensor(void)
{
    printf("  test_conv2d_s8_per_tensor...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * Per-tensor requantization: 1 output channel, num_channels=1.
     * 3x3 input, 2x2 kernel, no padding, stride 1.
     * Input shape NHWC: [1, 3, 3, 1], Output: [1, 2, 2, 1]
     * Weight shape OHWI: [1, 2, 2, 1]
     *
     * Input (int8, zp=0): 1..9
     * Weight: [1, 0, 0, 1]  Bias: [0]
     *
     * Accumulator: 6, 8, 12, 14
     * Identity requant (multiplier=2^30, shift=1) -> output = acc
     */
    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 3, 3, 1};
    int32_t y_shape[] = {1, 2, 2, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 9, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 4, out_qp);

    int8_t weight[] = {1, 0, 0, 1};
    uint16_t w_idx = add_weight(weight, 4, "w");

    int32_t bias[] = {0};
    uint16_t b_idx = add_weight(bias, 4, "b");

    test_ops[0].op_type = TIGRIS_OP_CONV;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 2;
    test_ops[0].spatial.kernel_w = 2;
    test_ops[0].spatial.stride_h = 1;
    test_ops[0].spatial.stride_w = 1;
    test_ops[0].spatial.dilation_h = 1;
    test_ops[0].spatial.dilation_w = 1;
    test_ops[0].spatial.group = 1;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tigris_mem_alloc_fast(&mem, t_in, 9);
    memcpy(ptrs[t_in], input_data, 9);
    tigris_mem_alloc_fast(&mem, t_out, 4);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv2d_s8_per_tensor returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 6, "y[0,0,0,0]=6");
    TEST_ASSERT_EQ(out[1], 8, "y[0,0,1,0]=8");
    TEST_ASSERT_EQ(out[2], 12, "y[0,1,0,0]=12");
    TEST_ASSERT_EQ(out[3], 14, "y[0,1,1,0]=14");
}

static void test_conv2d_s8_v2(void)
{
    printf("  test_conv2d_s8_v2...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * 1-channel conv: 3x3 input, 2x2 kernel, no padding, stride 1
     * Output has 2 channels to test per-channel requantization.
     *
     * Input shape NHWC: [1, 3, 3, 1], Output: [1, 2, 2, 2]
     * Weight shape OHWI: [2, 2, 2, 1]
     *
     * Input (int8, zp=0):
     *   1  2  3
     *   4  5  6
     *   7  8  9
     *
     * Weight OC=0 (identity-like): [1,0,0,1]
     * Weight OC=1 (all ones):      [1,1,1,1]
     *
     * Bias (int32): [0, 0]
     *
     * OC=0 accumulator: 6, 8, 12, 14 (same as single-channel)
     * OC=1 accumulator: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
     *
     * Multiplier = 2^30 (=1073741824), shift = 1
     * Output = acc * 1 = acc (identity requant)
     */
    int32_t mults[] = {1073741824, 1073741824};
    int32_t shifts[] = {1, 1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.0f, 0, 2, mults, shifts);

    int32_t x_shape[] = {1, 3, 3, 1};
    int32_t y_shape[] = {1, 2, 2, 2};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 9, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 8, out_qp);

    /* Weight [2, 2, 2, 1] = {1,0,0,1, 1,1,1,1} */
    int8_t weight[] = {1, 0, 0, 1, 1, 1, 1, 1};
    uint16_t w_idx = add_weight(weight, 8, "w");

    int32_t bias[] = {0, 0};
    uint16_t b_idx = add_weight(bias, 8, "b");

    test_ops[0].op_type = TIGRIS_OP_CONV;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 2;
    test_ops[0].spatial.kernel_w = 2;
    test_ops[0].spatial.stride_h = 1;
    test_ops[0].spatial.stride_w = 1;
    test_ops[0].spatial.dilation_h = 1;
    test_ops[0].spatial.dilation_w = 1;
    test_ops[0].spatial.group = 1;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    tigris_mem_alloc_fast(&mem, t_in, 9);
    memcpy(ptrs[t_in], input_data, 9);
    tigris_mem_alloc_fast(&mem, t_out, 8);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv2d_s8_v2 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    /* NHWC output: interleaved [y00_oc0, y00_oc1, y01_oc0, y01_oc1, ...] */
    TEST_ASSERT_EQ(out[0], 6, "y[0,0,0,0]=6");
    TEST_ASSERT_EQ(out[1], 12, "y[0,0,0,1]=12");
    TEST_ASSERT_EQ(out[2], 8, "y[0,0,1,0]=8");
    TEST_ASSERT_EQ(out[3], 16, "y[0,0,1,1]=16");
    TEST_ASSERT_EQ(out[4], 12, "y[0,1,0,0]=12");
    TEST_ASSERT_EQ(out[5], 24, "y[0,1,0,1]=24");
    TEST_ASSERT_EQ(out[6], 14, "y[0,1,1,0]=14");
    TEST_ASSERT_EQ(out[7], 28, "y[0,1,1,1]=28");
}

static void test_fc_s8(void)
{
    printf("  test_fc_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * FC: [1, 3] * [2, 3]^T + [2] = [1, 2]
     * Input (int8, zp=0): [1, 2, 3]
     * Weight (int8): [[1,0,0], [0,1,0]] (identity-like)
     * Bias (int32): [10, 20]
     *
     * acc[0] = 1*1 + 2*0 + 3*0 + 10 = 11
     * acc[1] = 1*0 + 2*1 + 3*0 + 20 = 22
     *
     * multiplier = 2^30, shift = 1 -> identity requant
     */
    int32_t mults[] = {1073741824, 1073741824};
    int32_t shifts[] = {1, 1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.0f, 0, 2, mults, shifts);

    int32_t x_shape[] = {1, 3};
    int32_t y_shape[] = {1, 2};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 2, 3, 3, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 2, 3, 2, out_qp);

    int8_t weight[] = {1, 0, 0, 0, 1, 0};
    uint16_t w_idx = add_weight(weight, 6, "w");

    int32_t bias[] = {10, 20};
    uint16_t b_idx = add_weight(bias, 8, "b");

    test_ops[0].op_type = TIGRIS_OP_FULLY_CONN;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {1, 2, 3};
    tigris_mem_alloc_fast(&mem, t_in, 3);
    memcpy(ptrs[t_in], input_data, 3);
    tigris_mem_alloc_fast(&mem, t_out, 2);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "fc_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    /* acc[0]=11, acc[1]=22. Identity requant (mult=2^30, shift=1) has
     * a +1 rounding nudge for odd values: 11->12, 22->22 (even, no nudge). */
    TEST_ASSERT_EQ(out[0], 12, "fc[0] = 12");
    TEST_ASSERT_EQ(out[1], 22, "fc[1] = 22");
}

static void test_global_avg_pool_s8(void)
{
    printf("  test_global_avg_pool_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * GAP: [1, 2, 2, 1] -> [1, 1, 1, 1]
     * Input (int8, zp=0, scale=1.0): [2, 4, 6, 8]
     * avg = (2+4+6+8)/4 = 5
     * Output (zp=0, scale=1.0): 5
     */
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp_idx = add_quant_param(1.0f, 0, 1, NULL, NULL);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 1, 1, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 1, out_qp_idx);

    test_ops[0].op_type = TIGRIS_OP_GLOBAL_AVG;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {2, 4, 6, 8};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 1);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "global_avg_pool_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 5, "avg(2,4,6,8) = 5");
}

static void test_unsupported_op_s8(void)
{
    printf("  test_unsupported_op_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    test_ops[0].op_type = TIGRIS_OP_SOFTMAX;  /* Not supported in s8 */
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[64], slow_buf[64];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, 1, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, -1, "unsupported op returns -1");
}

/* Main */

int main(void)
{
    printf("TiGrIS Int8 Kernel Tests\n\n");

    test_relu_s8();
    test_relu6_s8();
    test_reshape_s8();
    test_conv2d_s8_per_tensor();
    test_conv2d_s8_v2();
    test_fc_s8();
    test_global_avg_pool_s8();
    test_unsupported_op_s8();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

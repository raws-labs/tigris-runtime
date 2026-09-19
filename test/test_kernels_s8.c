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
static _Alignas(int32_t) uint8_t test_weight_blob[MAX_WEIGHT_BLOB];
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
    /* A test that wants attributes sets them after this call. Leaving the
     * fields alone would let a stack-local plan carry whatever the previous
     * frame left there, which any kernel that reads an attribute follows. */
    plan->op_attributes = NULL;
    plan->op_attribute_data = NULL;
    plan->num_op_attributes = 0;
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

static void test_binary_s8_contract(void)
{
    printf("  test_binary_s8_contract...\n");

    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 4};
    uint16_t t_a = add_tensor(&plan, "a", shape, 2, 3, 4, qp);
    uint16_t t_b = add_tensor(&plan, "b", shape, 2, 3, 4, qp);
    uint16_t t_y = add_tensor(&plan, "y", shape, 2, 3, 4, qp);

    uint16_t inputs[] = {t_a, t_b};
    uint16_t outputs[] = {t_y};
    test_ops[0].op_type = TIGRIS_OP_ADD;
    test_ops[0].num_inputs = 2;
    test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(inputs, 2);
    test_ops[0].outputs_off = add_indices(outputs, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(
        &mem, ptrs, test_header.num_tensors,
        fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {1, 2, 3, 4};
    const int8_t b[] = {2, -1, 3, 0};
    tigris_mem_alloc_fast(&mem, t_a, sizeof(a));
    memcpy(ptrs[t_a], a, sizeof(a));
    tigris_mem_alloc_fast(&mem, t_b, sizeof(b));
    memcpy(ptrs[t_b], b, sizeof(b));
    tigris_mem_alloc_fast(&mem, t_y, sizeof(a));

    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   0, "two-dynamic-input Add remains supported");
    const int8_t add_expected[] = {3, 1, 6, 4};
    TEST_ASSERT(memcmp(ptrs[t_y], add_expected, sizeof(add_expected)) == 0,
                "two-dynamic-input Add output");

    /* Sub shares Add's path and differs only in the sign of the second term,
     * so the operand order is what this checks: a swap would negate it. */
    test_ops[0].op_type = TIGRIS_OP_SUB;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   0, "two-dynamic-input Sub is supported");
    const int8_t sub_expected[] = {-1, 3, 0, 4};
    TEST_ASSERT(memcmp(ptrs[t_y], sub_expected, sizeof(sub_expected)) == 0,
                "two-dynamic-input Sub output");
    test_ops[0].op_type = TIGRIS_OP_ADD;

    test_ops[0].op_type = TIGRIS_OP_MUL;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   0, "two-dynamic-input Mul remains supported");
    const int8_t mul_expected[] = {2, -2, 9, 0};
    TEST_ASSERT(memcmp(ptrs[t_y], mul_expected, sizeof(mul_expected)) == 0,
                "two-dynamic-input Mul output");

    /* Equal byte counts are insufficient: general dynamic broadcasting is
     * not implemented, so logical shapes must match exactly. */
    uint16_t y_shape_off = test_tensors[t_y].shape_off;
    test_shapes[y_shape_off] = 2;
    test_shapes[y_shape_off + 1] = 2;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   -1, "same-size differently-shaped dynamic output rejected");
    test_shapes[y_shape_off] = 1;
    test_shapes[y_shape_off + 1] = 4;

    test_ops[0].weight_idx = 0;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   -1, "ambiguous dynamic inputs plus weight rejected");

    /* Compiler one-input + weight encoding must fail before ins[1] is read:
     * the weight entry carries bytes but no constant quant scale/zero point. */
    const int8_t constant[] = {1, 1, 1, 1};
    test_ops[0].weight_idx = add_weight(
        constant, (uint32_t)sizeof(constant), "constant");
    test_ops[0].num_inputs = 1;
    test_indices[1] = t_y;
    test_ops[0].outputs_off = 1;
    test_ops[0].op_type = TIGRIS_OP_ADD;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   -1, "quantized constant Add fails closed");
    test_ops[0].op_type = TIGRIS_OP_MUL;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   -1, "quantized constant Mul fails closed");

    test_ops[0].num_inputs = 3;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   -1, "unsupported quantized binary arity rejected");
}

static void test_tiled_rank3_binary_s8(void)
{
    printf("  test_tiled_rank3_binary_s8...\n");

    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 4, 2};
    uint16_t t_a = add_tensor(&plan, "a", shape, 3, 3, 8, qp);
    uint16_t t_b = add_tensor(&plan, "b", shape, 3, 3, 8, qp);
    uint16_t t_y = add_tensor(&plan, "y", shape, 3, 3, 8, qp);
    uint16_t inputs[] = {t_a, t_b};
    uint16_t outputs[] = {t_y};

    test_ops[0].op_type = TIGRIS_OP_ADD;
    test_ops[0].num_inputs = 2;
    test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(inputs, 2);
    test_ops[0].outputs_off = add_indices(outputs, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(
        &mem, ptrs, test_header.num_tensors,
        fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {1, 2, 3, 4, 50, 60, 70, 80};
    const int8_t b[] = {2, -1, 3, 0, 1, 1, 1, 1};
    const int8_t add_expected[] = {3, 1, 6, 4};
    const int8_t mul_expected[] = {2, -2, 9, 0};
    tigris_mem_alloc_fast(&mem, t_a, sizeof(a));
    memcpy(ptrs[t_a], a, sizeof(a));
    tigris_mem_alloc_fast(&mem, t_b, sizeof(b));
    memcpy(ptrs[t_b], b, sizeof(b));
    tigris_mem_alloc_fast(&mem, t_y, sizeof(a));
    memset(ptrs[t_y], 0x55, sizeof(a));
    mem.tile.active = 1;
    mem.tile.out_h = 2;
    mem.tile.out_w = 1;

    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   0, "tiled rank-3 Add succeeds");
    TEST_ASSERT(memcmp(ptrs[t_y], add_expected, sizeof(add_expected)) == 0,
                "tiled rank-3 Add processes the selected length range");
    TEST_ASSERT(((int8_t *)ptrs[t_y])[4] == 0x55,
                "tiled rank-3 Add does not process the full tensor");

    memset(ptrs[t_y], 0x55, sizeof(a));
    test_ops[0].op_type = TIGRIS_OP_MUL;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                       &plan, &test_ops[0], 0, &mem, NULL),
                   0, "tiled rank-3 Mul succeeds");
    TEST_ASSERT(memcmp(ptrs[t_y], mul_expected, sizeof(mul_expected)) == 0,
                "tiled rank-3 Mul processes the selected length range");
    TEST_ASSERT(((int8_t *)ptrs[t_y])[4] == 0x55,
                "tiled rank-3 Mul does not process the full tensor");
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

/* ConvTranspose s8 tests */

static void test_conv_transpose_s8_stride2(void)
{
    printf("  test_conv_transpose_s8_stride2...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * Same no-overlap stride2 gather geometry as the f32
     * test_conv_transpose_stride2 (kernel_h/kernel_w == stride_h/stride_w,
     * no padding): each input pixel maps to a disjoint 2x2 output block,
     * out[2*ih+kh, 2*iw+kw] = X[ih,iw] * W[kh,kw] + bias.
     * Identity requant (multiplier=2^30, shift=1), zp=0, bias=1 (oc=0).
     * X[ih,iw]: 1 2 / 3 4   W[kh,kw]: 1 2 / 3 4
     * Hand-computed per input pixel (product + bias 1):
     *   ih=0,iw=0 (X=1): out[0,0]=2 out[0,1]=3 out[1,0]=4 out[1,1]=5
     *   ih=0,iw=1 (X=2): out[0,2]=3 out[0,3]=5 out[1,2]=7 out[1,3]=9
     *   ih=1,iw=0 (X=3): out[2,0]=4 out[2,1]=7 out[3,0]=10 out[3,1]=13
     *   ih=1,iw=1 (X=4): out[2,2]=5 out[2,3]=9 out[3,2]=13 out[3,3]=17
     */
    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 4, 4, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 16, out_qp);

    /* Weight [OC=1, KH=2, KW=2, IC=1] (OHWI) */
    int8_t weight[] = {1, 2, 3, 4};
    uint16_t w_idx = add_weight(weight, sizeof(weight), "w");

    int32_t bias[] = {1};
    uint16_t b_idx = add_weight(bias, sizeof(bias), "b");

    test_ops[0].op_type = TIGRIS_OP_CONV_TRANSPOSE;
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
    test_ops[0].spatial.stride_h = 2;
    test_ops[0].spatial.stride_w = 2;
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

    int8_t input_data[] = {1, 2, 3, 4};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 16);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv_transpose_s8_stride2 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    int8_t expected[16] = {
        2,  3,  3,  5,
        4,  5,  7,  9,
        4,  7,  5,  9,
        10, 13, 13, 17,
    };
    for (int i = 0; i < 16; i++)
        TEST_ASSERT_EQ(out[i], expected[i], "conv_transpose_s8 stride2 output");
}

static void test_conv_transpose_s8_pad_stride_skip(void)
{
    printf("  test_conv_transpose_s8_pad_stride_skip...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * Same padded/indivisible-stride geometry as the f32
     * test_conv_transpose_pad_stride_skip (kernel 3x3, stride 2, pad 1,
     * 1x2x2x1 input -> 1x3x3x1 output): exercises the modulo-skip branch
     * (num_h/num_w % stride != 0) in the s8 gather loop. Identity requant
     * (multiplier=2^30, shift=1), zp=0, bias=0.
     * X[ih,iw]: 1 2 / 3 4   W[kh,kw] row-major 1..9: 1 2 3 / 4 5 6 / 7 8 9
     * Expected (same valid (ih,kh,oh)/(iw,kw,ow) triples as the f32 test):
     *   Y[0,0]=5   Y[0,1]=14  Y[0,2]=10
     *   Y[1,0]=14  Y[1,1]=36  Y[1,2]=24
     *   Y[2,0]=15  Y[2,1]=34  Y[2,2]=20
     */
    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 3, 3, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 9, out_qp);

    int8_t weight[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint16_t w_idx = add_weight(weight, sizeof(weight), "w");

    int32_t bias[] = {0};
    uint16_t b_idx = add_weight(bias, sizeof(bias), "b");

    test_ops[0].op_type = TIGRIS_OP_CONV_TRANSPOSE;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 3;
    test_ops[0].spatial.kernel_w = 3;
    test_ops[0].spatial.stride_h = 2;
    test_ops[0].spatial.stride_w = 2;
    test_ops[0].spatial.pad_top = 1;
    test_ops[0].spatial.pad_bottom = 1;
    test_ops[0].spatial.pad_left = 1;
    test_ops[0].spatial.pad_right = 1;
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

    int8_t input_data[] = {1, 2, 3, 4};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 9);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv_transpose_s8_pad_stride_skip returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    int8_t expected[9] = {
         5, 14, 10,
        14, 36, 24,
        15, 34, 20,
    };
    for (int i = 0; i < 9; i++)
        TEST_ASSERT_EQ(out[i], expected[i], "conv_transpose_s8 pad/stride-skip output");
}

static void test_conv_transpose_s8_out_of_range_skip(void)
{
    printf("  test_conv_transpose_s8_out_of_range_skip...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * Same geometry as the f32 test_conv_transpose_out_of_range_skip
     * (kernel 3x3, stride 1, pad 2, 1x2x2x1 input -> 1x3x3x1 output):
     * stride_h=stride_w=1 makes num_h % SH and num_w % SW always 0, so the
     * only skip that can happen here is the ih/iw >= input-extent bounds
     * check, isolating that branch in the s8 kernel from the modulo-skip
     * branch already covered above. Identity requant (multiplier=2^30,
     * shift=1), zp=0, bias=7.
     * X[ih,iw]: 1 2 / 3 4   W[kh,kw] row-major 1..9: 1 2 3 / 4 5 6 / 7 8 9
     * Expected (same derivation as the f32 test):
     *   Y[0,0]=70  Y[0,1]=49  Y[0,2]=7
     *   Y[1,0]=66  Y[1,1]=43  Y[1,2]=7
     *   Y[2,0]=7   Y[2,1]=7   Y[2,2]=7
     */
    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 3, 3, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 9, out_qp);

    int8_t weight[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint16_t w_idx = add_weight(weight, sizeof(weight), "w");

    int32_t bias[] = {7};
    uint16_t b_idx = add_weight(bias, sizeof(bias), "b");

    test_ops[0].op_type = TIGRIS_OP_CONV_TRANSPOSE;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 3;
    test_ops[0].spatial.kernel_w = 3;
    test_ops[0].spatial.stride_h = 1;
    test_ops[0].spatial.stride_w = 1;
    test_ops[0].spatial.pad_top = 2;
    test_ops[0].spatial.pad_bottom = 2;
    test_ops[0].spatial.pad_left = 2;
    test_ops[0].spatial.pad_right = 2;
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

    int8_t input_data[] = {1, 2, 3, 4};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 9);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv_transpose_s8_out_of_range_skip returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    int8_t expected[9] = {
        70, 49, 7,
        66, 43, 7,
         7,  7, 7,
    };
    for (int i = 0; i < 9; i++)
        TEST_ASSERT_EQ(out[i], expected[i], "conv_transpose_s8 out-of-range-skip output");
}

static void test_conv_transpose_s8_tile_2d(void)
{
    printf("  test_conv_transpose_s8_tile_2d...\n");

    plan_reset();
    tigris_plan_t plan;

    /*
     * Same interior 2D output tile geometry as the f32 test
     * (test_conv_transpose_tile_2d): NHWC 1x4x4x1 input, kernel 2x2,
     * stride=2, pad=0 -> 1x8x8x1 output. Interior output tile
     * [oh0..oh1) x [ow0..ow1) = [4..6) x [2..6):
     *   ih_lo = floor_div(oh0 + PT - (KH-1), SH) = floor_div(4+0-1, 2) = 1
     *   ih_hi = floor_div(oh1-1 + PT, SH)        = floor_div(5+0, 2)   = 2
     *   iw_lo = floor_div(ow0 + PL - (KW-1), SW) = floor_div(2+0-1, 2) = 0
     *   iw_hi = floor_div(ow1-1 + PL, SW)        = floor_div(5+0, 2)   = 2
     * loaded input rect: rows [1,3) cols [0,3) -> in_h=2, in_w=3, ih0=1, iw0=0
     * effective pads: PT_eff = oh0 + PT - ih0*SH = 4 - 2 = 2
     *                 PL_eff = ow0 + PL - iw0*SW = 2 - 0 = 2
     * out_h=2, out_w=4.
     * Identity requant (multiplier=2^30, shift=1), zp=0, bias=1 (oc=0), same
     * scaffolding as test_conv_transpose_s8_stride2. Run the op WHOLE (tile
     * inactive) to get Y_full, the oracle; no expected values are
     * hardcoded. Then run it TILED for the interior tile and assert exact
     * int8 equality against Y_full's sub-region.
     */
    int32_t mults[] = {1073741824};
    int32_t shifts[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 4, 4, 1};
    int32_t y_shape[] = {1, 8, 8, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 16, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 64, out_qp);

    /* Weight [OC=1, KH=2, KW=2, IC=1] (OHWI) */
    int8_t weight[] = {1, 2, 3, 4};
    uint16_t w_idx = add_weight(weight, sizeof(weight), "w");

    int32_t bias[] = {1};
    uint16_t b_idx = add_weight(bias, sizeof(bias), "b");

    test_ops[0].op_type = TIGRIS_OP_CONV_TRANSPOSE;
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
    test_ops[0].spatial.stride_h = 2;
    test_ops[0].spatial.stride_w = 2;
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

    int8_t input_data[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    };
    tigris_mem_alloc_fast(&mem, t_in, 16);
    memcpy(ptrs[t_in], input_data, 16);
    tigris_mem_alloc_fast(&mem, t_out, 64);

    /* 1) Run WHOLE (tile inactive): Y_full is the oracle. */
    int ret_full = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret_full, 0, "conv_transpose_s8_tile_2d whole dispatch returns 0");

    int8_t Y_full[64];
    memcpy(Y_full, ptrs[t_out], sizeof(Y_full));

    /* 2) Build packed X_tile = X_full[1:3, 0:3] (in_h=2, in_w=3, IC=1). */
    int8_t input_tile[6];
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++)
            input_tile[r * 3 + c] = input_data[(1 + r) * 4 + (0 + c)];

    /* 3) Run TILED for output tile [4..6) x [2..6). */
    tigris_mem_reset_fast(&mem);
    tigris_mem_alloc_fast(&mem, t_in, 6);
    memcpy(ptrs[t_in], input_tile, 6);
    tigris_mem_alloc_fast(&mem, t_out, 8);

    mem.tile.active = 1;
    mem.tile.width_tiled = 1;
    mem.tile.in_h = 2;
    mem.tile.out_h = 2;
    mem.tile.in_w = 3;
    mem.tile.out_w = 4;
    mem.tile.pad_top = 2;
    mem.tile.pad_left = 2;
    mem.tile.pad_bottom = 0;
    mem.tile.pad_right = 0;
    mem.tile.out_row_start = 0;
    mem.tile.in_row_start = 0;

    int ret_tile = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret_tile, 0, "conv_transpose_s8_tile_2d tiled dispatch returns 0");

    int8_t *out_tile = (int8_t *)ptrs[t_out];
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 4; c++)
            TEST_ASSERT_EQ(out_tile[r * 4 + c], Y_full[(4 + r) * 8 + (2 + c)],
                           "conv_transpose_s8 2D tile == whole sub-region");
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

    TEST_ASSERT(((uintptr_t)tigris_op_bias(&plan, &test_ops[0]) %
                 _Alignof(int32_t)) != 0,
                "FC bias fixture is deliberately unaligned");

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
    /* acc[0]=11, acc[1]=22. Identity requant (mult=2^30, shift=1, i.e. scale=1)
     * maps each value to itself under the gemmlowp/TFLite rounding (the old
     * nudge biased odd values up by 1, giving 12 - that was the bug). */
    TEST_ASSERT_EQ(out[0], 11, "fc[0] = 11");
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

static void test_global_max_pool_s8(void)
{
    printf("  test_global_max_pool_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /* A maximum is one of the input values, so with matching quantization it
     * passes through unchanged. The negative rules out an unsigned read of the
     * input. The sibling test below covers the case where the two tensors
     * declare different quantization, where the value survives but its
     * encoding does not. */
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp_idx = add_quant_param(1.0f, 0, 1, NULL, NULL);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 1, 1, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 1, out_qp_idx);

    test_ops[0].op_type = TIGRIS_OP_GLOBAL_MAX;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                    slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {-9, 4, 7, 3};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 1);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "global_max_pool_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 7, "max(-9,4,7,3) = 7");
}

/* A maximum preserves the value, not the encoding. When the output tensor
 * declares a different scale or zero point, the same value is a different
 * int8, and both max kernels used to write the input's encoding straight out.
 * kern_avg_pool_s8 beside them has always defined this case; these now match
 * it. */
static void test_max_pool_s8_requantizes(void)
{
    printf("  test_max_pool_s8_requantizes...\n");

    plan_reset();
    tigris_plan_t plan;

    /* Input at scale 1.0, zero point 0; output at scale 0.5, zero point -10.
     * max(-9, 4, 7, 3) is 7, so the value is 7.0 and its encoding at the
     * output is round(7.0 / 0.5) + (-10) = 4. */
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp_idx = add_quant_param(0.5f, -10, 1, NULL, NULL);

    int32_t x_shape[] = {1, 2, 2, 1};
    int32_t y_shape[] = {1, 1, 1, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 1, out_qp_idx);

    test_ops[0].op_type = TIGRIS_OP_GLOBAL_MAX;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in};
    uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                    slow_buf, sizeof(slow_buf));

    int8_t input_data[] = {-9, 4, 7, 3};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 1);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "global_max_pool_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 4, "max 7 at scale 1.0 is 4 at scale 0.5, zp -10");
}

/* A row band hands the matrix product a band of its first operand's rows and
 * the whole of its second, and says how many rows in tile.out_h rather than in
 * the shape. Taking the row count from the shape would compute the whole
 * matrix into a band-sized output. */
static void test_matmul_s8_row_band(void)
{
    printf("  test_matmul_s8_row_band...\n");

    plan_reset();
    tigris_plan_t plan;

    /* An effective scale of 1: the Q0.31 multiplier for 1.0 is 2^30 with a
     * shift of 1, which is the identity the other kernels use. */
    int32_t mult[] = {1 << 30};
    int32_t shift[] = {1};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mult, shift);

    enum { ROWS = 4, INNER = 2, COLS = 2, BAND = 2 };
    int32_t a_shape[] = {ROWS, INNER};
    int32_t b_shape[] = {INNER, COLS};
    int32_t y_shape[] = {ROWS, COLS};
    uint16_t t_a = add_tensor(&plan, "a", a_shape, 2, 3,
                              ROWS * INNER, in_qp);
    uint16_t t_b = add_tensor(&plan, "b", b_shape, 2, 3,
                              INNER * COLS, in_qp);
    uint16_t t_y = add_tensor(&plan, "y", y_shape, 2, 3,
                              ROWS * COLS, out_qp);

    test_ops[0].op_type = TIGRIS_OP_MATMUL;
    test_ops[0].num_inputs = 2;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_a, t_b};
    uint16_t outp[] = {t_y};
    test_ops[0].inputs_off = add_indices(inp, 2);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;

    build_plan(&plan);

    /* The band buffer holds the first two rows of a; b is whole. */
    int8_t a_band[BAND * INNER] = {1, 2, 3, 4};
    int8_t b_whole[INNER * COLS] = {1, 0, 0, 1};   /* identity */
    int8_t y_band[BAND * COLS];
    memset(y_band, 99, sizeof(y_band));

    void *ptrs[MAX_TENSORS];
    memset(ptrs, 0, sizeof(ptrs));
    ptrs[t_a] = a_band;
    ptrs[t_b] = b_whole;
    ptrs[t_y] = y_band;
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = test_header.num_tensors;
    mem.tile.active = 1;
    mem.tile.row_tiled = 1;
    mem.tile.in_h = BAND;
    mem.tile.out_h = BAND;
    mem.tile.in_w = 1;
    mem.tile.out_w = 1;

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "matmul_s8 over a row band returns 0");
    TEST_ASSERT_EQ(y_band[0], 1, "band row 0 column 0");
    TEST_ASSERT_EQ(y_band[1], 2, "band row 0 column 1");
    TEST_ASSERT_EQ(y_band[2], 3, "band row 1 column 0");
    TEST_ASSERT_EQ(y_band[3], 4, "band row 1 column 1");
}

static void test_unsupported_op_s8(void)
{
    printf("  test_unsupported_op_s8...\n");

    plan_reset();
    tigris_plan_t plan;

    /* Pad has a schema opcode but no runtime dispatcher. Softmax is
     * intentionally supported by s8_ref for terminal classifiers, and MatMul
     * gained a dispatcher, so neither is the unsupported exemplar any more. */
    test_ops[0].op_type = TIGRIS_OP_PAD;
    test_header.num_ops = 1;

    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[64], slow_buf[64];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, 1, fast, sizeof(fast), slow_buf, sizeof(slow_buf));

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, -1, "unsupported op returns -1");
}

/* Relu6 ceiling must be ROUNDED, not truncated (regression for the
 * (int32_t)(6/scale) truncation bug). scale=0.11 -> 6/scale=54.545 -> round 55;
 * truncation would give 54, clamping a legitimate 55 down to 54. */
static void test_relu6_s8_rounds_ceiling(void)
{
    printf("  test_relu6_s8_rounds_ceiling...\n");
    plan_reset();
    tigris_plan_t plan;

    uint16_t qp = add_quant_param(0.11f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 3};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 3, qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 3, TIGRIS_NO_QUANT_PARAM);

    test_ops[0].op_type = TIGRIS_OP_RELU6;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in}; uint16_t outp[] = {t_out};
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
    int8_t input_data[] = {55, 100, -10};
    tigris_mem_alloc_fast(&mem, t_in, 3);
    memcpy(ptrs[t_in], input_data, 3);
    tigris_mem_alloc_fast(&mem, t_out, 3);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "relu6_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 55, "relu6(55): ceiling rounds to 55, not truncated to 54");
    TEST_ASSERT_EQ(out[1], 55, "relu6(100) clamps to 55");
    TEST_ASSERT_EQ(out[2], 0,  "relu6(-10) clamps to 0");
}

/* Tanh via LUT (regression: s8 dispatch must handle TIGRIS_OP_TANH). */
static void test_tanh_s8(void)
{
    printf("  test_tanh_s8...\n");
    plan_reset();
    tigris_plan_t plan;

    uint16_t in_qp = add_quant_param(0.1f, 0, 1, NULL, NULL);          /* real = 0.1*x */
    uint16_t out_qp = add_quant_param(1.0f / 127.0f, 0, 1, NULL, NULL); /* [-1,1]->[-127,127] */
    int32_t shape[] = {1, 3};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 3, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 3, out_qp);

    test_ops[0].op_type = TIGRIS_OP_TANH;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in}; uint16_t outp[] = {t_out};
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
    int8_t input_data[] = {0, 20, -20};   /* real 0, +2.0, -2.0 */
    tigris_mem_alloc_fast(&mem, t_in, 3);
    memcpy(ptrs[t_in], input_data, 3);
    tigris_mem_alloc_fast(&mem, t_out, 3);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "tanh_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    /* tanh(0)=0; tanh(2)=0.9640 -> round(0.9640*127)=122; tanh(-2)=-122 */
    TEST_ASSERT_EQ(out[0], 0,    "tanh(0) = 0");
    TEST_ASSERT_EQ(out[1], 122,  "tanh(2.0) = 122");
    TEST_ASSERT_EQ(out[2], -122, "tanh(-2.0) = -122");
}

/* Softmax is a terminal classifier operation and must be available through
 * the reference fallback used by accelerator backends. */
static void test_softmax_s8(void)
{
    printf("  test_softmax_s8...\n");
    plan_reset();
    tigris_plan_t plan;

    uint16_t in_qp = add_quant_param(0.5f, -3, 1, NULL, NULL);
    /* TFLite's common int8 Softmax representation: [0,1] -> [-128,127]. */
    uint16_t out_qp = add_quant_param(1.0f / 256.0f, -128, 1, NULL, NULL);
    int32_t shape[] = {2, 3};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 6, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 6, out_qp);

    test_ops[0].op_type = TIGRIS_OP_SOFTMAX;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in}; uint16_t outp[] = {t_out};
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
    /* Rows encode real logits [0, 1, -1] and [0, 0, 0]. */
    int8_t input_data[] = {-3, -1, -5, -3, -3, -3};
    tigris_mem_alloc_fast(&mem, t_in, 6);
    memcpy(ptrs[t_in], input_data, 6);
    tigris_mem_alloc_fast(&mem, t_out, 6);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "softmax_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    /* Quantized probabilities: [0.2447, .6652, .0900] and [1/3,1/3,1/3]. */
    TEST_ASSERT_EQ(out[0], -65, "softmax row 0 class 0");
    TEST_ASSERT_EQ(out[1], 42,  "softmax row 0 class 1");
    TEST_ASSERT_EQ(out[2], -105,"softmax row 0 class 2");
    TEST_ASSERT_EQ(out[3], -43, "softmax row 1 class 0");
    TEST_ASSERT_EQ(out[4], -43, "softmax row 1 class 1");
    TEST_ASSERT_EQ(out[5], -43, "softmax row 1 class 2");
}

/* INT8 Conv1D (regression: s8 dispatch must handle TIGRIS_OP_CONV1D). NLC input
 * [1,3,2], weights [OC=1,K=2,IC=2] all ones, identity requant -> output = acc. */
static void test_conv1d_s8(void)
{
    printf("  test_conv1d_s8...\n");
    plan_reset();
    tigris_plan_t plan;

    int32_t mults[] = {1073741824}; int32_t shifts[] = {1};  /* identity (2^30, <<1) */
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, mults, shifts);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, mults, shifts);

    int32_t x_shape[] = {1, 3, 2};   /* N, IT, IC */
    int32_t y_shape[] = {1, 2, 1};   /* N, OT, OC */
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 3, 3, 6, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 3, 3, 2, out_qp);

    int8_t weight[] = {1, 1, 1, 1};  /* [OC,K,IC] = [1,2,2] all ones */
    uint16_t w_idx = add_weight(weight, 4, "w");
    int32_t bias[] = {0};
    uint16_t b_idx = add_weight(bias, 4, "b");

    test_ops[0].op_type = TIGRIS_OP_CONV1D;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in}; uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = w_idx;
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 2;   /* K */
    test_ops[0].spatial.stride_h = 1;
    test_ops[0].spatial.dilation_h = 1;
    test_ops[0].spatial.pad_top = 0;    /* pad_begin */
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[4096], slow_buf[4096];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow_buf, sizeof(slow_buf));
    /* NLC: t0=[1,2] t1=[3,4] t2=[5,6] */
    int8_t input_data[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_in, 6);
    memcpy(ptrs[t_in], input_data, 6);
    tigris_mem_alloc_fast(&mem, t_out, 2);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv1d_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    /* ot0 = (1+2)+(3+4)=10 ; ot1 = (3+4)+(5+6)=18 */
    TEST_ASSERT_EQ(out[0], 10, "conv1d y[0]=10");
    TEST_ASSERT_EQ(out[1], 18, "conv1d y[1]=18");

    tigris_mem_reset_fast(&mem);
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 1);
    mem.tile.active = 1;
    mem.tile.in_h = 2;
    mem.tile.out_h = 1;
    mem.tile.pad_top = 0;
    ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "tiled conv1d_s8 returns 0");
    out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 10, "tiled conv1d y[0]=10");
}

static void test_transpose_s8(void)
{
    printf("  test_transpose_s8...\n");
    plan_reset();
    tigris_plan_t plan;
    int32_t input_shape[] = {2, 3};
    int32_t output_shape[] = {3, 2};
    uint16_t input = add_tensor(&plan, "x", input_shape, 2, 3, 6,
                                TIGRIS_NO_QUANT_PARAM);
    uint16_t output = add_tensor(&plan, "y", output_shape, 2, 3, 6,
                                 TIGRIS_NO_QUANT_PARAM);
    uint16_t ins[] = {input}, outs[] = {output};
    test_ops[0].op_type = TIGRIS_OP_TRANSPOSE;
    test_ops[0].num_inputs = 1; test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(ins, 1);
    test_ops[0].outputs_off = add_indices(outs, 1);
    test_header.num_ops = 1;
    build_plan(&plan);
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_TRANSPOSE_PERM, 2, 0};
    const uint8_t perm[] = {1, 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = perm;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[256]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                    slow, sizeof(slow));
    const int8_t values[] = {1, 2, 3, 4, 5, 6};
    const int8_t expected[] = {1, 4, 2, 5, 3, 6};
    tigris_mem_alloc_fast(&mem, input, sizeof(values));
    memcpy(ptrs[input], values, sizeof(values));
    tigris_mem_alloc_fast(&mem, output, sizeof(expected));
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "transpose_s8 returns 0");
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[output])[i], expected[i],
                       "transpose_s8 value");
}


/* The int8 mean folds the reciprocal into the requant multiplier, so a plan
 * whose output declares a different encoding than its input has to come back
 * on the output's scale. A rescaling case and a same-scale case, because the
 * first is what a quantizer actually assigns and the second is the arithmetic
 * anyone can check by hand. */
static void test_reduce_mean_s8(void)
{
    printf("  test_reduce_mean_s8...\n");
    plan_reset();
    tigris_plan_t plan;

    /* [1, 4, 2], mean over the token axis.
     * columns: (2+4+6+8)/4 = 5 and (10+12+14+16)/4 = 13 */
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t x_shape[] = {1, 4, 2};
    int32_t y_shape[] = {1, 1, 2};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 3, 3, 8, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 3, 3, 2, out_qp);
    uint16_t ins[] = {t_in}, outs[] = {t_out};
    test_ops[0].op_type = TIGRIS_OP_REDUCE_MEAN;
    test_ops[0].num_inputs = 1; test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(ins, 1);
    test_ops[0].outputs_off = add_indices(outs, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;
    build_plan(&plan);
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_AXES, 1, 0};
    const uint8_t axes[] = {1};
    plan.op_attributes = &attr;
    plan.op_attribute_data = axes;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[4096], slow[4096]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                    slow, sizeof(slow));
    const int8_t values[] = {2, 10, 4, 12, 6, 14, 8, 16};
    tigris_mem_alloc_fast(&mem, t_in, sizeof(values));
    memcpy(ptrs[t_in], values, sizeof(values));
    tigris_mem_alloc_fast(&mem, t_out, 2);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "reduce_mean_s8 returns 0");
    TEST_ASSERT_EQ(((int8_t *)ptrs[t_out])[0], 5, "reduce_mean_s8 column 0");
    TEST_ASSERT_EQ(((int8_t *)ptrs[t_out])[1], 13, "reduce_mean_s8 column 1");

    /* Same values through a half-sized output scale and a shifted zero point:
     * 5 and 13 in input units become 10 and 26 steps, less the shift. */
    test_qp[out_qp].scale = 0.5f;
    test_qp[out_qp].zero_point = -4;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "rescaled reduce_mean_s8 returns 0");
    TEST_ASSERT_EQ(((int8_t *)ptrs[t_out])[0], 6, "rescaled column 0");
    TEST_ASSERT_EQ(((int8_t *)ptrs[t_out])[1], 22, "rescaled column 1");

    /* A plan that does not say which axis it collapses is refused. */
    plan.num_op_attributes = 0;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                          NULL) != 0,
                "reduce_mean_s8 without axes is refused");
}


/* A plan may state the three pairs that scale a quantized sum's operands and
 * its result. The kernel has to read them when they are there and derive them
 * when they are not: a plan whose pairs are ignored would run on numbers the
 * vendor kernels were never given. That the stated pairs equal the derived
 * ones is the cross-repository gate's job, since only the compiler produces
 * them; what is checkable here is that each path is taken. */
static void test_binary_requant_is_read_from_the_plan(void)
{
    printf("  test_binary_requant_is_read_from_the_plan...\n");
    plan_reset();
    tigris_plan_t plan;

    uint16_t qa = add_quant_param(0.05f, 0, 1, NULL, NULL);
    uint16_t qb = add_quant_param(0.03f, -4, 1, NULL, NULL);
    uint16_t qy = add_quant_param(0.06f, 2, 1, NULL, NULL);
    int32_t shape[] = {1, 16};
    uint16_t t_a = add_tensor(&plan, "a", shape, 2, 3, 16, qa);
    uint16_t t_b = add_tensor(&plan, "b", shape, 2, 3, 16, qb);
    uint16_t t_y = add_tensor(&plan, "y", shape, 2, 3, 16, qy);
    uint16_t ins[] = {t_a, t_b}, outs[] = {t_y};
    test_ops[0].op_type = TIGRIS_OP_ADD;
    test_ops[0].num_inputs = 2; test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(ins, 2);
    test_ops[0].outputs_off = add_indices(outs, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS]; uint8_t fast[4096], slow[4096]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                    slow, sizeof(slow));
    int8_t left[16], right[16];
    for (int i = 0; i < 16; i++) {
        left[i] = (int8_t)(i * 7 - 50);
        right[i] = (int8_t)(60 - i * 5);
    }
    tigris_mem_alloc_fast(&mem, t_a, sizeof(left));
    memcpy(ptrs[t_a], left, sizeof(left));
    tigris_mem_alloc_fast(&mem, t_b, sizeof(right));
    memcpy(ptrs[t_b], right, sizeof(right));
    tigris_mem_alloc_fast(&mem, t_y, 16);

    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "add without stated pairs returns 0");
    int8_t derived[16];
    memcpy(derived, ptrs[t_y], sizeof(derived));

    /* Halving the first operand's multiplier is a change the derived path
     * cannot produce, so the output moving proves the plan was read. */
    int32_t pairs[6] = {1 << 30, 0, 1 << 30, 0, 1 << 30, -20};
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_BINARY_REQUANT,
                                  (uint8_t)TIGRIS_OP_ATTR_BINARY_REQUANT_LEN,
                                  0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = (const uint8_t *)pairs;
    plan.num_op_attributes = 1;
    memset(ptrs[t_y], 0, 16);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "add with stated pairs returns 0");
    int differs = 0;
    for (int i = 0; i < 16; i++)
        if (((int8_t *)ptrs[t_y])[i] != derived[i])
            differs = 1;
    TEST_ASSERT(differs, "stated pairs reach the arithmetic");

    /* A payload of the wrong length is not a stated pair set, so the kernel
     * falls back rather than reading past it. */
    attr.data_len = 8;
    memset(ptrs[t_y], 0, 16);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "add with a short payload returns 0");
    for (int i = 0; i < 16; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], derived[i],
                       "a short payload falls back to the derived pairs");
}

/* Main */

int main(void)
{
    printf("TiGrIS Int8 Kernel Tests\n\n");

    test_relu_s8();
    test_relu6_s8();
    test_relu6_s8_rounds_ceiling();
    test_binary_s8_contract();
    test_tiled_rank3_binary_s8();
    test_reshape_s8();
    test_conv2d_s8_per_tensor();
    test_conv2d_s8_v2();
    test_conv_transpose_s8_stride2();
    test_conv_transpose_s8_pad_stride_skip();
    test_conv_transpose_s8_out_of_range_skip();
    test_conv_transpose_s8_tile_2d();
    test_fc_s8();
    test_global_avg_pool_s8();
    test_global_max_pool_s8();
    test_max_pool_s8_requantizes();
    test_matmul_s8_row_band();
    test_tanh_s8();
    test_softmax_s8();
    test_conv1d_s8();
    test_transpose_s8();
    test_reduce_mean_s8();
    test_binary_requant_is_read_from_the_plan();
    test_unsupported_op_s8();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

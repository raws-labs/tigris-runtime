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
#include "elementwise_s8_golden.h"
#include "test_harness.h"

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

/* One-operator plan with the given tensors. */
static void one_op_s8(tigris_plan_t *plan, uint8_t type, const uint16_t *ins, int n_in,
                      const uint16_t *outs, int n_out, uint16_t weight)
{
    test_ops[0].op_type = type;
    test_ops[0].num_inputs = (uint8_t)n_in; test_ops[0].num_outputs = (uint8_t)n_out;
    test_ops[0].inputs_off = add_indices(ins, n_in);
    test_ops[0].outputs_off = add_indices(outs, n_out);
    test_ops[0].weight_idx = weight;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128; test_ops[0].act_max = 127;
    test_header.num_ops = 1;
    build_plan(plan);
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
    one_op_s8(&plan, TIGRIS_OP_RELU, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    /* Input: [-5, -2, 0, 10]. zp=-2, so relu clamps to >= -2 */
    int8_t input_data[] = {-5, -2, 0, 10};
    tigris_mem_alloc_fast(&mem, t_in, 4); memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 4);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "relu_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], -2, "relu(-5, zp=-2) = -2");
    TEST_ASSERT_EQ(out[1], -2, "relu(-2, zp=-2) = -2");
    TEST_ASSERT_EQ(out[2], 0, "relu(0, zp=-2) = 0");
    TEST_ASSERT_EQ(out[3], 10, "relu(10, zp=-2) = 10");
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
    one_op_s8(&plan, TIGRIS_OP_RESHAPE, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_in, 6); memcpy(ptrs[t_in], input_data, 6);
    tigris_mem_alloc_fast(&mem, t_out, 6);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "reshape_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQ(out[i], input_data[i], "reshape preserves data");
    }
}

static void test_conv2d_s8(void)
{
    printf("  test_conv2d_s8...\n");

    /*
     * 1-channel conv: 3x3 input, 2x2 kernel, no padding, stride 1, two output
     * channels. Input NHWC [1, 3, 3, 1], output [1, 2, 2, 2], weight OHWI
     * [2, 2, 2, 1].
     *
     * Input (int8, zp=0):
     *   1  2  3
     *   4  5  6
     *   7  8  9
     *
     * Weight OC=0 (identity-like): [1,0,0,1]
     * Weight OC=1 (all ones):      [1,1,1,1]
     * Bias (int32): [0, 0]
     *
     * OC=0 accumulator: 6, 8, 12, 14
     * OC=1 accumulator: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
     *
     * Multiplier = 2^30, shift = 1: output = acc. The requantization is stated
     * once for both channels (per-tensor), then once per channel.
     */
    const int8_t expected[] = {6, 12, 8, 16, 12, 24, 14, 28};
    for (uint16_t channels = 1; channels <= 2; channels++) {
        plan_reset();
        tigris_plan_t plan;
        int32_t mults[] = {1073741824, 1073741824};
        int32_t shifts[] = {1, 1};
        uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
        uint16_t out_qp = add_quant_param(1.0f, 0, channels, mults, shifts);

        int32_t x_shape[] = {1, 3, 3, 1};
        int32_t y_shape[] = {1, 2, 2, 2};
        uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 9, in_qp);
        uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 8, out_qp);
        int8_t weight[] = {1, 0, 0, 1, 1, 1, 1, 1};
        int32_t bias[] = {0, 0};
        uint16_t w_idx = add_weight(weight, 8, "w");
        uint16_t b_idx = add_weight(bias, 8, "b");
        one_op_s8(&plan, TIGRIS_OP_CONV, &t_in, 1, &t_out, 1, w_idx);
        test_ops[0].bias_idx = b_idx;
        test_ops[0].spatial.kernel_h = 2;
        test_ops[0].spatial.kernel_w = 2;
        test_ops[0].spatial.stride_h = 1;
        test_ops[0].spatial.stride_w = 1;
        test_ops[0].spatial.dilation_h = 1;
        test_ops[0].spatial.dilation_w = 1;
        test_ops[0].spatial.group = 1;

        void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
        tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
        int8_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        tigris_mem_alloc_fast(&mem, t_in, 9); memcpy(ptrs[t_in], input_data, 9);
        tigris_mem_alloc_fast(&mem, t_out, 8);

        int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
        TEST_ASSERT_EQ(ret, 0, "conv2d_s8 returns 0");

        /* NHWC output: interleaved [y00_oc0, y00_oc1, y01_oc0, y01_oc1, ...] */
        int8_t *out = (int8_t *)ptrs[t_out];
        for (int i = 0; i < 8; i++)
            TEST_ASSERT_EQ(out[i], expected[i],
                           channels == 1 ? "per-tensor conv2d_s8 output" : "per-channel conv2d_s8 output");
    }
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
    int32_t bias[] = {10, 20};
    uint16_t w_idx = add_weight(weight, 6, "w");
    uint16_t b_idx = add_weight(bias, 8, "b");
    one_op_s8(&plan, TIGRIS_OP_FULLY_CONN, &t_in, 1, &t_out, 1, w_idx);
    test_ops[0].bias_idx = b_idx;

    TEST_ASSERT(((uintptr_t)tigris_op_bias(&plan, &test_ops[0]) %
                 _Alignof(int32_t)) != 0,
                "FC bias fixture is deliberately unaligned");

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {1, 2, 3};
    tigris_mem_alloc_fast(&mem, t_in, 3); memcpy(ptrs[t_in], input_data, 3);
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

static void test_fc_s8_shift_beyond_31_bits(void)
{
    printf("  test_fc_s8_shift_beyond_31_bits...\n");

    plan_reset();
    tigris_plan_t plan;

    /* Plans from older compilers carry shifts below -31 for effective scales
     * under 2^-32. acc = +-2^30 gives a high product of +-2^29, which rounds
     * to 0 at either shift, so both outputs are the zero point. */
    int32_t mults[] = {1073741824, 1073741824};
    int32_t shifts[] = {-32, -35};
    uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.0f, 5, 2, mults, shifts);

    int32_t x_shape[] = {1, 1};
    int32_t y_shape[] = {1, 2};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 2, 3, 1, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 2, 3, 2, out_qp);

    int8_t weight[] = {0, 0};
    uint16_t w_idx = add_weight(weight, 2, "w");
    int32_t bias[] = {1073741824, -1073741824};
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

    tigris_mem_alloc_fast(&mem, t_in, 1);
    ((int8_t *)ptrs[t_in])[0] = 1;
    tigris_mem_alloc_fast(&mem, t_out, 2);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "fc_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 5, "shift -32 rounds to the zero point");
    TEST_ASSERT_EQ(out[1], 5, "shift -35 rounds to the zero point");
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
    one_op_s8(&plan, TIGRIS_OP_GLOBAL_AVG, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {2, 4, 6, 8};
    tigris_mem_alloc_fast(&mem, t_in, 4); memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 1);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "global_avg_pool_s8 returns 0");

    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 5, "avg(2,4,6,8) = 5");
}

/* A whole-map AveragePool lowered to a global pool keeps TFLite
 * AVERAGE_POOL_2D rounding through TIGRIS_OP_ATTR_POOL_ROUNDING: the sum over
 * the 25x5 map divided by 125, ties away from zero. Without the attribute the
 * pool is a TFLite MEAN, whose folded multiplier rounds some sums differently. */
static void test_global_avg_pool_s8_average_rounding(void)
{
    printf("  test_global_avg_pool_s8_average_rounding...\n");

    plan_reset();
    tigris_plan_t plan;
    uint16_t in_qp = add_quant_param(0.25f, -3, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(0.25f, -3, 1, NULL, NULL);
    int32_t x_shape[] = {1, 25, 5, 1};
    int32_t y_shape[] = {1, 1, 1, 1};
    uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 125, in_qp);
    uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 1, out_qp);
    uint16_t ins[] = {t_in}, outs[] = {t_out};
    test_ops[0].op_type = TIGRIS_OP_GLOBAL_AVG;
    test_ops[0].num_inputs = 1; test_ops[0].num_outputs = 1;
    test_ops[0].inputs_off = add_indices(ins, 1);
    test_ops[0].outputs_off = add_indices(outs, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;
    build_plan(&plan);
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_POOL_ROUNDING, 1, 0};
    const uint8_t average[] = {TIGRIS_POOL_ROUNDING_AVERAGE};

    void *ptrs[MAX_TENSORS]; uint8_t fast[4096], slow[4096]; tigris_mem_t mem;
    int mismatches = 0, mean_differs = 0;
    for (int32_t target = -15000; target <= 15000; target += 7) {
        int8_t values[125];
        int32_t remaining = target;
        for (int i = 0; i < 125; i++) {
            int32_t v = remaining / (125 - i);
            values[i] = (int8_t)v;
            remaining -= v;
        }
        int32_t expected = target > 0 ? (target + 62) / 125 : (target - 62) / 125;
        int8_t results[2];
        for (int with_attr = 0; with_attr < 2; with_attr++) {
            plan.op_attributes = with_attr ? &attr : NULL;
            plan.op_attribute_data = with_attr ? average : NULL;
            plan.num_op_attributes = with_attr ? 1 : 0;
            tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast),
                            slow, sizeof(slow));
            tigris_mem_alloc_fast(&mem, t_in, sizeof(values));
            memcpy(ptrs[t_in], values, sizeof(values));
            tigris_mem_alloc_fast(&mem, t_out, 1);
            if (tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0)
                mismatches++;
            results[with_attr] = *(int8_t *)ptrs[t_out];
        }
        if (results[1] != expected)
            mismatches++;
        if (results[0] != results[1])
            mean_differs++;
    }
    TEST_ASSERT_EQ(mismatches, 0, "average rounding matches sum/125 with ties away from zero");
    TEST_ASSERT(mean_differs > 0, "mean rounding differs from average rounding on some sums");
}

/* A maximum preserves the value, not the encoding. With matching
 * quantization it passes through unchanged; the negative rules out an
 * unsigned read of the input. When the output declares a different scale or
 * zero point, the same value is a different int8: both max kernels used to
 * write the input's encoding straight out, and now match kern_avg_pool_s8,
 * which has always defined this case. */
static void test_global_max_pool_s8(void)
{
    printf("  test_global_max_pool_s8...\n");

    /* max(-9, 4, 7, 3) is 7.0; at scale 0.5, zero point -10 it encodes as
     * round(7.0 / 0.5) + (-10) = 4. */
    static const struct {
        float out_scale;
        int32_t out_zp;
        int8_t expected;
        const char *what;
    } cases[] = {
        {1.0f, 0, 7, "max(-9,4,7,3) = 7"},
        {0.5f, -10, 4, "max 7 at scale 1.0 is 4 at scale 0.5, zp -10"},
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        plan_reset();
        tigris_plan_t plan;
        uint16_t in_qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
        uint16_t out_qp_idx = add_quant_param(cases[k].out_scale, cases[k].out_zp, 1, NULL, NULL);
        int32_t x_shape[] = {1, 2, 2, 1};
        int32_t y_shape[] = {1, 1, 1, 1};
        uint16_t t_in = add_tensor(&plan, "x", x_shape, 4, 3, 4, in_qp);
        uint16_t t_out = add_tensor(&plan, "y", y_shape, 4, 3, 1, out_qp_idx);
        one_op_s8(&plan, TIGRIS_OP_GLOBAL_MAX, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

        void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
        tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
        int8_t input_data[] = {-9, 4, 7, 3};
        tigris_mem_alloc_fast(&mem, t_in, 4); memcpy(ptrs[t_in], input_data, 4);
        tigris_mem_alloc_fast(&mem, t_out, 1);

        int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
        TEST_ASSERT_EQ(ret, 0, "global_max_pool_s8 returns 0");
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_out])[0], cases[k].expected, cases[k].what);
    }
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

    uint16_t inp[] = {t_a, t_b};
    one_op_s8(&plan, TIGRIS_OP_MATMUL, inp, 2, &t_y, 1, TIGRIS_NO_WEIGHT);

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

    /* An opcode no dispatcher handles reaches the default branch. */
    test_ops[0].op_type = TIGRIS_OP_UNKNOWN;
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
    int32_t shape[] = {1, 4};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 4, qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 4, TIGRIS_NO_QUANT_PARAM);
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
    int8_t input_data[] = {55, 100, -10, 30};
    tigris_mem_alloc_fast(&mem, t_in, 4);
    memcpy(ptrs[t_in], input_data, 4);
    tigris_mem_alloc_fast(&mem, t_out, 4);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "relu6_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 55, "relu6(55): ceiling rounds to 55, not truncated to 54");
    TEST_ASSERT_EQ(out[1], 55, "relu6(100) clamps to 55");
    TEST_ASSERT_EQ(out[2], 0,  "relu6(-10) clamps to 0");
    TEST_ASSERT_EQ(out[3], 30, "relu6(30) passes through inside the range");
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

    one_op_s8(&plan, TIGRIS_OP_TANH, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {0, 20, -20};   /* real 0, +2.0, -2.0 */
    tigris_mem_alloc_fast(&mem, t_in, 3); memcpy(ptrs[t_in], input_data, 3);
    tigris_mem_alloc_fast(&mem, t_out, 3);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "tanh_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    /* tanh(0)=0; tanh(2)=0.9640 -> round(0.9640*127)=122; tanh(-2)=-122 */
    TEST_ASSERT_EQ(out[0], 0,    "tanh(0) = 0");
    TEST_ASSERT_EQ(out[1], 122,  "tanh(2.0) = 122");
    TEST_ASSERT_EQ(out[2], -122, "tanh(-2.0) = -122");
}

static void test_gelu_s8(void)
{
    printf("  test_gelu_s8...\n");
    plan_reset();
    tigris_plan_t plan;

    uint16_t in_qp = add_quant_param(0.1f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(0.01f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 3};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 3, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 3, out_qp);

    test_ops[0].op_type = TIGRIS_OP_GELU;
    test_ops[0].num_inputs = 1;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_in}; uint16_t outp[] = {t_out};
    test_ops[0].inputs_off = add_indices(inp, 1);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_header.num_ops = 1;
    build_plan(&plan);

    uint8_t approximate = 0;
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_GELU_APPROXIMATE, 1, 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = &approximate;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {0, 20, -20};   /* real 0, +2.0, -2.0 */
    tigris_mem_alloc_fast(&mem, t_in, 3);
    memcpy(ptrs[t_in], input_data, 3);
    tigris_mem_alloc_fast(&mem, t_out, 3);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "gelu_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    TEST_ASSERT_EQ(out[0], 0, "gelu(0)");
    TEST_ASSERT_EQ(out[1], 127, "positive tail saturates");
    TEST_ASSERT_EQ(out[2], -5, "negative tail rounds");
    approximate = 1;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "approximate GELU");
    TEST_ASSERT_EQ(out[2], -5, "approximate negative tail rounds");
    plan.num_op_attributes = 0;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0, "missing GELU options");
}


/* HardSwish: x * relu6(x + 3) / 6 through the LUT. Real inputs -4, -1, 0, 2, 4
 * at scale 0.1 give 0, -1/3, 0, 5/3, 4; at output scale 0.05 that is 0, -7,
 * 0, 33, 80 (half away from zero). */
static void test_hardswish_s8(void)
{
    printf("  test_hardswish_s8...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t in_qp = add_quant_param(0.1f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(0.05f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 5};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 5, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 5, out_qp);
    one_op_s8(&plan, TIGRIS_OP_HARDSWISH, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t input_data[] = {-40, -10, 0, 20, 40};
    tigris_mem_alloc_fast(&mem, t_in, 5); memcpy(ptrs[t_in], input_data, 5);
    tigris_mem_alloc_fast(&mem, t_out, 5);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "hardswish_s8 returns 0");
    const int8_t expected[] = {0, -7, 0, 33, 80};
    int8_t *out = (int8_t *)ptrs[t_out];
    for (int i = 0; i < 5; i++)
        TEST_ASSERT_EQ(out[i], expected[i], "hardswish_s8 value");
}

/* An int8 Quantize from int8 requantizes as TFLite's Requantize does:
 * (x - in_zp) * in_scale / out_scale through the gemmlowp multiplier, plus
 * out_zp. A ratio of 2 is exact; a ratio of 1/2 rounds a half up, so 3 gives 2
 * and -3 gives -1. */
static void run_requantize_s8(float in_scale, int32_t in_zp, float out_scale, int32_t out_zp,
                              const int8_t *input, const int8_t *expected)
{
    plan_reset();
    tigris_plan_t plan;
    uint16_t in_qp = add_quant_param(in_scale, in_zp, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(out_scale, out_zp, 1, NULL, NULL);
    int32_t shape[] = {1, 5};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, 5, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, 5, out_qp);
    test_ops[0].op_type = TIGRIS_OP_QUANTIZE;
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
    tigris_mem_alloc_fast(&mem, t_in, 5);
    memcpy(ptrs[t_in], input, 5);
    tigris_mem_alloc_fast(&mem, t_out, 5);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "requantize_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    for (int i = 0; i < 5; i++)
        TEST_ASSERT_EQ(out[i], expected[i], "requantize_s8 value");
}

static void test_requantize_s8(void)
{
    printf("  test_requantize_s8...\n");
    const int8_t wide_in[] = {-100, 0, 1, 3, 60};
    const int8_t wide_out[] = {-128, -5, -3, 1, 115};
    run_requantize_s8(0.5f, 1, 0.25f, -3, wide_in, wide_out);
    const int8_t narrow_in[] = {3, -3, 4, -128, 127};
    const int8_t narrow_out[] = {2, -1, 2, -64, 64};
    run_requantize_s8(0.25f, 0, 0.5f, 0, narrow_in, narrow_out);
}

/* A per-channel second operand repeats every C elements of the first. */
static void run_per_channel_binary_s8(uint8_t op_type, const int8_t *expected,
                                      const char *what)
{
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t full[] = {1, 1, 2, 3};
    int32_t gate[] = {1, 1, 1, 3};
    uint16_t t_a = add_tensor(&plan, "a", full, 4, 3, 6, qp);
    uint16_t t_b = add_tensor(&plan, "b", gate, 4, 3, 3, qp);
    uint16_t t_y = add_tensor(&plan, "y", full, 4, 3, 6, qp);
    test_ops[0].op_type = op_type;
    test_ops[0].num_inputs = 2;
    test_ops[0].num_outputs = 1;
    uint16_t inp[] = {t_a, t_b}; uint16_t outp[] = {t_y};
    test_ops[0].inputs_off = add_indices(inp, 2);
    test_ops[0].outputs_off = add_indices(outp, 1);
    test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
    test_ops[0].act_min = -128;
    test_ops[0].act_max = 127;
    test_header.num_ops = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS];
    uint8_t fast[1024], slow[1024];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {1, 2, 3, 4, 5, 6};
    const int8_t b[] = {2, -1, 3};
    tigris_mem_alloc_fast(&mem, t_a, 6); memcpy(ptrs[t_a], a, 6);
    tigris_mem_alloc_fast(&mem, t_b, 3); memcpy(ptrs[t_b], b, 3);
    tigris_mem_alloc_fast(&mem, t_y, 6);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, what);
    int8_t *y = (int8_t *)ptrs[t_y];
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_EQ(y[i], expected[i], what);

    /* One value per column: a broadcast inside the operand, read by output
     * coordinate and never tiled. */
    test_shapes[test_tensors[t_b].shape_off + 2] = 2;
    test_shapes[test_tensors[t_b].shape_off + 3] = 1;
    test_tensors[t_b].size_bytes = 2;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "a column-wise operand runs");
    const int8_t columns_product[] = {2, 4, 6, -4, -5, -6};
    const int8_t columns_sum[] = {3, 4, 5, 3, 4, 5};
    const int8_t *columns = (op_type == TIGRIS_OP_MUL) ? columns_product : columns_sum;
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_EQ(y[i], columns[i], "column-wise operand value");
    mem.tile.active = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "a column-wise operand is refused under a tile");
    mem.tile.active = 0;
}

static void test_per_channel_binary_s8(void)
{
    printf("  test_per_channel_binary_s8...\n");
    const int8_t product[] = {2, -2, 9, 8, -5, 18};
    const int8_t sum[] = {3, 1, 6, 6, 4, 9};
    run_per_channel_binary_s8(TIGRIS_OP_MUL, product, "per-channel Mul");
    run_per_channel_binary_s8(TIGRIS_OP_ADD, sum, "per-channel Add");
}

/* Softmax rows against the TFLite reference and CMSIS-NN arm_softmax_s8, which
 * use fixed point for the common [0,1] -> [-128,127] output encoding; another
 * output quantization keeps float. */
static int run_softmax_s8(float in_scale, int32_t in_zp, float out_scale, int32_t out_zp,
                          const int8_t *data, uint32_t rows, uint32_t classes, int8_t *result)
{
    plan_reset();
    tigris_plan_t plan;
    uint16_t in_qp = add_quant_param(in_scale, in_zp, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(out_scale, out_zp, 1, NULL, NULL);
    int32_t shape[] = {(int32_t)rows, (int32_t)classes};
    uint16_t t_in = add_tensor(&plan, "in", shape, 2, 3, rows * classes, in_qp);
    uint16_t t_out = add_tensor(&plan, "out", shape, 2, 3, rows * classes, out_qp);
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
    uint8_t fast[256], slow[64];
    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    tigris_mem_alloc_fast(&mem, t_in, rows * classes);
    memcpy(ptrs[t_in], data, rows * classes);
    tigris_mem_alloc_fast(&mem, t_out, rows * classes);
    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    memcpy(result, ptrs[t_out], rows * classes);
    return ret;
}

static void test_softmax_s8(void)
{
    printf("  test_softmax_s8...\n");
    static const struct {
        float scale;
        int32_t zero_point;
        uint32_t rows, classes;
        int8_t data[6];
        int8_t expected[6];
    } cases[] = {
        /* Two rows encoding real logits [0, 1, -1] and [0, 0, 0]: each row
         * normalizes on its own, to [.2447, .6652, .0900] and thirds. */
        {0.5f, -3, 2, 3, {-3, -1, -5, -3, -3, -3}, {-65, 42, -105, -43, -43, -43}},
        /* Rows where a float softmax would round differently. */
        {0.053988829255104065f, 0, 1, 6, {-21, -37, 19, -9, 39, -11}, {-121, -125, -71, -116, 38, -117}},
        {0.01572207175195217f, 0, 1, 4, {33, -7, -39, 7}, {-26, -74, -95, -61}},
        {0.1154790073633194f, 0, 1, 6, {-37, 34, -25, 4, 4, -32}, {-128, 113, -128, -120, -120, -128}},
    };
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        int8_t out[6];
        uint32_t n = cases[k].rows * cases[k].classes;
        TEST_ASSERT_EQ(run_softmax_s8(cases[k].scale, cases[k].zero_point, 1.0f / 256.0f, -128,
                                      cases[k].data, cases[k].rows, cases[k].classes, out), 0,
                       "fixed-point softmax returns 0");
        for (uint32_t i = 0; i < n; i++)
            TEST_ASSERT_EQ(out[i], cases[k].expected[i], "fixed-point softmax matches TFLite");
    }

    /* Output zero point -127 is not TFLite's: float, round half away. */
    int8_t out[6];
    TEST_ASSERT_EQ(run_softmax_s8(0.053988829255104065f, 0, 1.0f / 256.0f, -127,
                                  cases[1].data, 1, 6, out), 0, "float softmax returns 0");
    static const int8_t float_expected[6] = {-120, -124, -70, -115, 40, -116};
    TEST_ASSERT(memcmp(out, float_expected, 6) == 0, "float softmax keeps its rounding");
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

    one_op_s8(&plan, TIGRIS_OP_CONV1D, &t_in, 1, &t_out, 1, w_idx);
    test_ops[0].bias_idx = b_idx;
    test_ops[0].spatial.kernel_h = 2;   /* K */
    test_ops[0].spatial.stride_h = 1;
    test_ops[0].spatial.dilation_h = 1;
    test_ops[0].spatial.pad_top = 0;    /* pad_begin */

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    /* NLC: t0=[1,2] t1=[3,4] t2=[5,6] */
    int8_t input_data[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_in, 6); memcpy(ptrs[t_in], input_data, 6);
    tigris_mem_alloc_fast(&mem, t_out, 2);

    int ret = tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "conv1d_s8 returns 0");
    int8_t *out = (int8_t *)ptrs[t_out];
    /* ot0 = (1+2)+(3+4)=10 ; ot1 = (3+4)+(5+6)=18 */
    TEST_ASSERT_EQ(out[0], 10, "conv1d y[0]=10");
    TEST_ASSERT_EQ(out[1], 18, "conv1d y[1]=18");

    tigris_mem_reset_fast(&mem);
    tigris_mem_alloc_fast(&mem, t_in, 4); memcpy(ptrs[t_in], input_data, 4);
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
    one_op_s8(&plan, TIGRIS_OP_REDUCE_MEAN, &t_in, 1, &t_out, 1, TIGRIS_NO_WEIGHT);
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_AXES, 1, 0};
    const uint8_t axes[] = {1};
    plan.op_attributes = &attr;
    plan.op_attribute_data = axes;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t values[] = {2, 10, 4, 12, 6, 14, 8, 16};
    tigris_mem_alloc_fast(&mem, t_in, sizeof(values)); memcpy(ptrs[t_in], values, sizeof(values));
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


/* The int8 split is the same contiguous hand-out as the float one: the parts
 * carry the input's encoding because they are its bytes. */
static void test_split_s8(void)
{
    printf("  test_split_s8...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(0.25f, -3, 1, NULL, NULL);
    int32_t whole[] = {3, 4};
    int32_t part_shape[] = {1, 4};
    uint16_t t_in = add_tensor(&plan, "x", whole, 2, 3, 12, qp);
    uint16_t t_a = add_tensor(&plan, "a", part_shape, 2, 3, 4, qp);
    uint16_t t_b = add_tensor(&plan, "b", part_shape, 2, 3, 4, qp);
    uint16_t t_c = add_tensor(&plan, "c", part_shape, 2, 3, 4, qp);
    uint16_t outs[] = {t_a, t_b, t_c};
    one_op_s8(&plan, TIGRIS_OP_SPLIT, &t_in, 1, outs, 3, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    int8_t values[12];
    for (int i = 0; i < 12; i++)
        values[i] = (int8_t)(i - 6);
    tigris_mem_alloc_fast(&mem, t_in, sizeof(values)); memcpy(ptrs[t_in], values, sizeof(values));
    tigris_mem_alloc_fast(&mem, t_a, 4);
    tigris_mem_alloc_fast(&mem, t_b, 4);
    tigris_mem_alloc_fast(&mem, t_c, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                             NULL), 0,
                   "split_s8 returns 0");
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_a])[i], values[i], "split_s8 part 0");
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_b])[i], values[4 + i],
                       "split_s8 part 1");
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_c])[i], values[8 + i],
                       "split_s8 part 2");
    }
    mem.tensor_ptrs[t_in] = NULL;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem,
                                          NULL) != 0,
                "split_s8 without a source is refused");
}

/* Operand pairs where TFLite's fixed-point product and a float product
 * rounded once differ by one step; expected values follow TFLite's
 * MultiplyByQuantizedMultiplier. */
static void test_mul_s8_tflite_rounding(void)
{
    printf("  test_mul_s8_tflite_rounding...\n");
    plan_reset();
    tigris_plan_t plan;
    int32_t shape[] = {1, 4};
    uint16_t t_a = add_tensor(&plan, "a", shape, 2, 3, 4, add_quant_param(0.0235f, -3, 1, NULL, NULL));
    uint16_t t_b = add_tensor(&plan, "b", shape, 2, 3, 4, add_quant_param(0.0173f, 5, 1, NULL, NULL));
    uint16_t t_y = add_tensor(&plan, "y", shape, 2, 3, 4, add_quant_param(0.0311f, -7, 1, NULL, NULL));
    uint16_t ins[] = {t_a, t_b}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_MUL, ins, 2, outs, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {-126, -126, -125, -125}, b[] = {-32, 42, -53, 63};
    const int8_t expected[] = {53, -67, 86, -100};
    tigris_mem_alloc_fast(&mem, t_a, 4); memcpy(ptrs[t_a], a, 4);
    tigris_mem_alloc_fast(&mem, t_b, 4); memcpy(ptrs[t_b], b, 4);
    tigris_mem_alloc_fast(&mem, t_y, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "mul_s8 runs");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], expected[i], "mul_s8 rounds like TFLite");
}

/* A split along the inner axis of [2, 3] into [2, 1] and [2, 2]: one run per
 * row from each part. */
static void test_split_s8_inner_axis(void)
{
    printf("  test_split_s8_inner_axis...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(0.25f, -3, 1, NULL, NULL);
    int32_t whole[] = {2, 3}, narrow[] = {2, 1}, wide[] = {2, 2};
    uint16_t t_in = add_tensor(&plan, "x", whole, 2, 3, 6, qp);
    uint16_t t_a = add_tensor(&plan, "a", narrow, 2, 3, 2, qp);
    uint16_t t_b = add_tensor(&plan, "b", wide, 2, 3, 4, qp);
    uint16_t ins[] = {t_in}, outs[] = {t_a, t_b};
    one_op_s8(&plan, TIGRIS_OP_SPLIT, ins, 1, outs, 2, TIGRIS_NO_WEIGHT);
    test_ops[0].spatial.kernel_h = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t values[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_in, 6); memcpy(ptrs[t_in], values, 6);
    tigris_mem_alloc_fast(&mem, t_a, 2);
    tigris_mem_alloc_fast(&mem, t_b, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "inner split runs");
    const int8_t first[] = {1, 4}, second[] = {2, 3, 5, 6};
    for (int i = 0; i < 2; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_a])[i], first[i], "inner split part 0");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_b])[i], second[i], "inner split part 1");
    test_ops[0].spatial.kernel_h = 2;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "split past the last axis is refused");
}

/* NHWC [1,1,2,2] and [1,2,2,2] joined along height; the second input is
 * quantized differently and is requantized on the way. */
static void test_concat_s8_height(void)
{
    printf("  test_concat_s8_height...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t out_qp = add_quant_param(0.25f, 0, 1, NULL, NULL);
    int32_t top[] = {1, 1, 2, 2}, bottom[] = {1, 2, 2, 2}, whole[] = {1, 3, 2, 2};
    uint16_t t_a = add_tensor(&plan, "a", top, 4, 3, 4, out_qp);
    uint16_t t_b = add_tensor(&plan, "b", bottom, 4, 3, 8, add_quant_param(0.5f, 0, 1, NULL, NULL));
    uint16_t t_y = add_tensor(&plan, "y", whole, 4, 3, 12, out_qp);
    uint16_t ins[] = {t_a, t_b}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_CONCAT, ins, 2, outs, 1, TIGRIS_NO_WEIGHT);
    test_ops[0].spatial.kernel_h = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {1, 2, 3, 4}, b[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int8_t expected[] = {1, 2, 3, 4, 2, 4, 6, 8, 10, 12, 14, 16};
    tigris_mem_alloc_fast(&mem, t_a, 4); memcpy(ptrs[t_a], a, 4);
    tigris_mem_alloc_fast(&mem, t_b, 8); memcpy(ptrs[t_b], b, 8);
    tigris_mem_alloc_fast(&mem, t_y, 12);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "height concat runs");
    for (int i = 0; i < 12; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], expected[i], "height concat value");
    mem.tile.active = 1; mem.tile.out_h = 1; mem.tile.out_w = 2;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "a tile across the joined axis is refused");
}

/* Two [1,2,2,4] inputs at scale 1e-6 joined on channels into scale 1.05e-6.
 * The scales differ by only 5e-8, and code 100 still becomes 95. */
static void test_concat_s8_small_scales(void)
{
    printf("  test_concat_s8_small_scales...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t in_qp = add_quant_param(1.0e-6f, 0, 1, NULL, NULL);
    uint16_t out_qp = add_quant_param(1.05e-6f, 0, 1, NULL, NULL);
    int32_t part[] = {1, 2, 2, 4}, whole[] = {1, 2, 2, 8};
    uint16_t t_a = add_tensor(&plan, "a", part, 4, 3, 16, in_qp);
    uint16_t t_b = add_tensor(&plan, "b", part, 4, 3, 16, in_qp);
    uint16_t t_y = add_tensor(&plan, "y", whole, 4, 3, 32, out_qp);
    uint16_t ins[] = {t_a, t_b}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_CONCAT, ins, 2, outs, 1, TIGRIS_NO_WEIGHT);
    test_ops[0].spatial.kernel_h = 3;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    tigris_mem_alloc_fast(&mem, t_a, 16); memset(ptrs[t_a], 100, 16);
    tigris_mem_alloc_fast(&mem, t_b, 16); memset(ptrs[t_b], 100, 16);
    tigris_mem_alloc_fast(&mem, t_y, 32);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "small-scale concat runs");
    for (int i = 0; i < 32; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], 95, "small-scale concat requantizes");
}

/* One input channel, two filters of 1x1: output channel oc reads channel
 * oc / 2. The requantization is the identity. */
static void test_depthwise_multiplier_s8(void)
{
    printf("  test_depthwise_multiplier_s8...\n");
    plan_reset();
    tigris_plan_t plan;
    const int32_t mult[] = {1 << 30}, shift[] = {1};
    int32_t in_shape[] = {1, 1, 2, 1}, out_shape[] = {1, 1, 2, 2};
    uint16_t t_x = add_tensor(&plan, "x", in_shape, 4, 3, 2, add_quant_param(1.0f, 0, 1, NULL, NULL));
    uint16_t t_y = add_tensor(&plan, "y", out_shape, 4, 3, 4, add_quant_param(1.0f, 0, 1, mult, shift));
    const int8_t weights[] = {2, -1};
    uint16_t ins[] = {t_x}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_DEPTHWISE, ins, 1, outs, 1, add_weight(weights, 2, "w"));
    test_ops[0].spatial.kernel_h = 1; test_ops[0].spatial.kernel_w = 1;
    test_ops[0].spatial.stride_h = 1; test_ops[0].spatial.stride_w = 1;
    test_ops[0].spatial.group = 1;
    build_plan(&plan);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t x[] = {3, -2};
    const int8_t expected[] = {6, -3, -4, 2};
    tigris_mem_alloc_fast(&mem, t_x, 2); memcpy(ptrs[t_x], x, 2);
    tigris_mem_alloc_fast(&mem, t_y, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "depthwise multiplier runs");
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], expected[i], "depthwise multiplier value");
}

static void test_elementwise_reference_goldens(void)
{
    printf("  test_elementwise_reference_goldens...\n");
    for (size_t c = 0; c < sizeof(elementwise_goldens) / sizeof(elementwise_goldens[0]); c++) {
        const elementwise_golden_t *gold = &elementwise_goldens[c];
        plan_reset();
        tigris_plan_t plan;
        uint16_t quant[3];
        for (int q = 0; q < 3; q++)
            quant[q] = add_quant_param(gold->scales[q], gold->zeros[q], 1, NULL, NULL);
        int32_t shape[] = {1, (int32_t)gold->count / 16, 16, 1};
        uint16_t a = add_tensor(&plan, "a", shape, 4, 3, gold->count, quant[0]);
        uint16_t b = add_tensor(&plan, "b", shape, 4, 3, gold->count, quant[1]);
        uint16_t y = add_tensor(&plan, "y", shape, 4, 3, gold->count, quant[2]);
        uint16_t inputs[] = {a, b};
        uint16_t outputs[] = {y};
        test_ops[0].op_type = gold->operation;
        test_ops[0].num_inputs = gold->input_b ? 2 : 1;
        test_ops[0].num_outputs = 1;
        test_ops[0].inputs_off = add_indices(inputs, test_ops[0].num_inputs);
        test_ops[0].outputs_off = add_indices(outputs, 1);
        test_ops[0].weight_idx = TIGRIS_NO_WEIGHT;
        test_ops[0].bias_idx = TIGRIS_NO_WEIGHT;
        test_ops[0].act_min = -128;
        test_ops[0].act_max = 127;
        test_header.num_ops = 1;
        build_plan(&plan);
        int8_t actual[768];
        void *pointers[MAX_TENSORS] = {0};
        pointers[a] = (void *)gold->input_a;
        pointers[b] = (void *)gold->input_b;
        pointers[y] = actual;
        tigris_mem_t mem;
        memset(&mem, 0, sizeof(mem));
        mem.tensor_ptrs = pointers;
        mem.num_tensors = test_header.num_tensors;
        memset(actual, 85, sizeof(actual));
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                       0, "elementwise reference runs");
        for (uint32_t i = 0; i < gold->count; i++) {
            if (actual[i] != gold->output[i])
                fprintf(stderr, "    golden case %zu, element %u\n", c, i);
            TEST_ASSERT_EQ(actual[i], gold->output[i], "TFLM reference byte");
        }
        memset(actual, 85, sizeof(actual));
        mem.tile.active = 1;
        mem.tile.in_h = 2;
        mem.tile.out_h = 2;
        mem.tile.in_w = 16;
        mem.tile.out_w = 16;
        mem.tile.in_row_start = 1;
        mem.tile.out_row_start = 1;
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                       0, "elementwise band runs");
        for (uint32_t i = 0; i < gold->count; i++) {
            int8_t expected = i >= 16u && i < 48u ? gold->output[i] : 85;
            TEST_ASSERT_EQ(actual[i], expected, "band matches TFLM and preserves surrounding bytes");
        }
        mem.tile.active = 0;
        if (gold->operation == TIGRIS_OP_DIV) {
            int8_t invalid[768];
            memcpy(invalid, gold->input_b, gold->count);
            invalid[0] = (int8_t)gold->zeros[1];
            pointers[b] = invalid;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "zero centered divisor is rejected");
            pointers[b] = (void *)gold->input_b;
            test_qp[quant[2]].scale = gold->scales[0] / (gold->scales[1] * 1099511627776.0f);
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "negative reference rounding exponent is rejected");
        }
        if (gold->operation == TIGRIS_OP_RSQRT) {
            int8_t invalid[256];
            memcpy(invalid, gold->input_a, sizeof(invalid));
            invalid[0] = (int8_t)(gold->zeros[0] - 1);
            pointers[a] = invalid;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "negative reciprocal square root is rejected");
            pointers[a] = (void *)gold->input_a;
            test_qp[quant[2]].scale = 65536.0f;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "out-of-range reference right shift is rejected");
        }
        if (gold->operation == TIGRIS_OP_ABS || gold->operation == TIGRIS_OP_SQUARED_DIFFERENCE ||
            gold->operation == TIGRIS_OP_DIV) {
            test_qp[quant[2]].scale = 1.0e-20f;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "out-of-range reference multiplier is rejected");
        }
        if (gold->operation == TIGRIS_OP_ABS) {
            test_qp[quant[2]].scale = gold->scales[0] / 16777216.0f;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                        "overflowing reference product is rejected");
        }
        test_qp[quant[2]].scale = gold->scales[2];
        test_tensors[a].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                    "missing quantization is rejected");
    }
}

/* A binary operator whose other operand is a constant from the weight table,
 * placed and quantized by the constant-operand attribute. */
static void test_binary_s8_constant_operand(void)
{
    printf("  test_binary_s8_constant_operand...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(0.1f, 0, 1, NULL, NULL);
    uint16_t constant_qp = add_quant_param(0.1f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 1, 1, 4};
    uint16_t t_x = add_tensor(&plan, "x", shape, 4, 3, 4, qp);
    uint16_t t_y = add_tensor(&plan, "y", shape, 4, 3, 4, qp);
    const int8_t ten = 10;
    uint16_t ins[] = {t_x}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_SUB, ins, 1, outs, 1, add_weight(&ten, 1, "c"));
    build_plan(&plan);

    uint8_t payload[4] = {0, 0, (uint8_t)constant_qp, 0};
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_CONSTANT_OPERAND,
                                  (uint8_t)TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN, 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = payload;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t x[] = {0, 3, -4, 7};
    tigris_mem_alloc_fast(&mem, t_x, 4); memcpy(ptrs[t_x], x, 4);
    tigris_mem_alloc_fast(&mem, t_y, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "constant minuend runs");
    const int8_t minuend[] = {10, 7, 14, 3};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], minuend[i], "constant minuend value");

    payload[0] = 1;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "constant subtrahend runs");
    const int8_t subtrahend[] = {-10, -7, -14, -3};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], subtrahend[i], "constant subtrahend value");

    /* One value per channel, repeated for every position. */
    const int8_t per_channel[] = {1, 2, 3, 4};
    test_ops[0].op_type = TIGRIS_OP_MUL;
    test_ops[0].weight_idx = add_weight(per_channel, 4, "per_channel");
    build_plan(&plan);
    plan.op_attributes = &attr;
    plan.op_attribute_data = payload;
    plan.num_op_attributes = 1;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "per-channel constant product runs");
    const int8_t product[] = {0, 1, -1, 3};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], product[i], "per-channel constant product");

    payload[0] = 2;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "a third operand position is refused");
    payload[0] = 1;
    payload[2] = 0xFF; payload[3] = 0xFF;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "an int8 constant without quantization is refused");
    plan.num_op_attributes = 0;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "an int8 constant without the attribute is refused");
}

/* Both operands broadcast: [1,2,1,2] + [1,1,3,2] -> [1,2,3,2]. The first
 * varies along height and channels, the second along width and channels. */
static void test_binary_s8_both_broadcast(void)
{
    printf("  test_binary_s8_both_broadcast...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t a_shape[] = {1, 2, 1, 2}, b_shape[] = {1, 1, 3, 2}, y_shape[] = {1, 2, 3, 2};
    uint16_t t_a = add_tensor(&plan, "a", a_shape, 4, 3, 4, qp);
    uint16_t t_b = add_tensor(&plan, "b", b_shape, 4, 3, 6, qp);
    uint16_t t_y = add_tensor(&plan, "y", y_shape, 4, 3, 12, qp);
    uint16_t ins[] = {t_a, t_b}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_ADD, ins, 2, outs, 1, TIGRIS_NO_WEIGHT);

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t a[] = {10, 20, 30, 40}, b[] = {1, 2, 3, 4, 5, 6};
    tigris_mem_alloc_fast(&mem, t_a, 4); memcpy(ptrs[t_a], a, 4);
    tigris_mem_alloc_fast(&mem, t_b, 6); memcpy(ptrs[t_b], b, 6);
    tigris_mem_alloc_fast(&mem, t_y, 12);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "both-broadcast Add runs");
    const int8_t expected[] = {11, 22, 13, 24, 15, 26, 31, 42, 33, 44, 35, 46};
    for (int i = 0; i < 12; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], expected[i], "both-broadcast Add value");
}

/* A constant that states its shape in the long attribute form: one value per
 * row of a [1,2,2,1] map. */
static void test_binary_s8_constant_shape(void)
{
    printf("  test_binary_s8_constant_shape...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 2, 2, 1};
    uint16_t t_x = add_tensor(&plan, "x", shape, 4, 3, 4, qp);
    uint16_t t_y = add_tensor(&plan, "y", shape, 4, 3, 4, qp);
    const int8_t rows[] = {5, -3};
    uint16_t ins[] = {t_x}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_SUB, ins, 1, outs, 1, add_weight(rows, 2, "rows"));
    uint8_t payload[20] = {1, 0, (uint8_t)qp, 0};
    const int32_t constant_shape[4] = {1, 2, 1, 1};
    memcpy(payload + 4, constant_shape, sizeof(constant_shape));
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_CONSTANT_OPERAND, 20u, 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = payload;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t x[] = {1, 2, 3, 4};
    tigris_mem_alloc_fast(&mem, t_x, 4); memcpy(ptrs[t_x], x, 4);
    tigris_mem_alloc_fast(&mem, t_y, 4);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "shaped constant runs");
    const int8_t expected[] = {-4, -3, 6, 7};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], expected[i], "shaped constant value");
    memcpy(payload + 8, &(int32_t){3}, sizeof(int32_t));
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "a constant shape that does not broadcast is refused");
}

/* A [1,2,2,1] map padded by one row above and one column on the right:
 * the fill is the output zero point, or the weight when one is stated. */
static void test_pad_s8(void)
{
    printf("  test_pad_s8...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t qp = add_quant_param(0.5f, -3, 1, NULL, NULL);
    int32_t in_shape[] = {1, 2, 2, 1}, out_shape[] = {1, 3, 3, 1};
    uint16_t t_x = add_tensor(&plan, "x", in_shape, 4, 3, 4, qp);
    uint16_t t_y = add_tensor(&plan, "y", out_shape, 4, 3, 9, qp);
    uint16_t ins[] = {t_x}, outs[] = {t_y};
    one_op_s8(&plan, TIGRIS_OP_PAD, ins, 1, outs, 1, TIGRIS_NO_WEIGHT);
    const int32_t pads[8] = {0, 0, 1, 0, 0, 1, 0, 0};
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_PADS, (uint8_t)sizeof(pads), 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = (const uint8_t *)pads;
    plan.num_op_attributes = 1;

    void *ptrs[MAX_TENSORS]; uint8_t fast[256], slow[16]; tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, test_header.num_tensors, fast, sizeof(fast), slow, sizeof(slow));
    const int8_t x[] = {1, 2, 3, 4};
    tigris_mem_alloc_fast(&mem, t_x, 4); memcpy(ptrs[t_x], x, 4);
    tigris_mem_alloc_fast(&mem, t_y, 9);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0, "pad runs");
    const int8_t zero_fill[] = {-3, -3, -3, 1, 2, -3, 3, 4, -3};
    for (int i = 0; i < 9; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], zero_fill[i], "pad with the zero point");

    const int8_t seven = 7;
    test_ops[0].weight_idx = add_weight(&seven, 1, "fill");
    build_plan(&plan);
    plan.op_attributes = &attr;
    plan.op_attribute_data = (const uint8_t *)pads;
    plan.num_op_attributes = 1;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "pad with a fill runs");
    const int8_t stated_fill[] = {7, 7, 7, 1, 2, 7, 3, 4, 7};
    for (int i = 0; i < 9; i++)
        TEST_ASSERT_EQ(((int8_t *)ptrs[t_y])[i], stated_fill[i], "pad with a stated fill");

    uint16_t fill = test_ops[0].weight_idx;
    const int8_t two_fills[] = {7, 7};
    test_ops[0].weight_idx = add_weight(two_fills, 2, "fills");
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "pad refuses a fill that is not one value");
    test_ops[0].weight_idx = fill;

    mem.tile.active = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "pad refuses a tile");
    test_shapes[test_tensors[t_x].shape_off + 3u] = 2;
    test_shapes[test_tensors[t_y].shape_off + 3u] = 2;
    test_tensors[t_x].size_bytes = 8u;
    test_tensors[t_y].size_bytes = 18u;
    mem.tile.row_tiled = 9u;
    mem.tile.in_h = mem.tile.out_h = 1;
    memset(ptrs[t_y], 0x55, 18u);
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL), 0,
                   "pad bands an unpadded channel axis");
    TEST_ASSERT(memcmp(ptrs[t_y], stated_fill, 9u) == 0, "channel band retains fill and coordinates");
    for (size_t i = 9u; i < 18u; i++)
        TEST_ASSERT_EQ(((uint8_t *)ptrs[t_y])[i], 0x55, "pad band output canary");
    mem.tile.active = 0;
    plan.num_op_attributes = 0;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "pad without its widths is refused");
}

static void test_activation_s8_domains(void)
{
    printf("  test_activation_s8_domains...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t iq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t oq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t x = add_tensor(&plan, "x", NULL, 0, 3, 1, iq);
    uint16_t y = add_tensor(&plan, "y", NULL, 0, 3, 1, oq);
    one_op_s8(&plan, TIGRIS_OP_LEAKY_RELU, &x, 1, &y, 1, TIGRIS_NO_WEIGHT);
    int8_t input = -100, output = 85;
    void *pointers[MAX_TENSORS] = {0};
    pointers[x] = &input;
    pointers[y] = &output;
    tigris_mem_t mem = {0};
    mem.tensor_ptrs = pointers;
    mem.num_tensors = test_header.num_tensors;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "scalar LeakyRelu runs");
    TEST_ASSERT_EQ(output, -1, "scalar LeakyRelu applies its default alpha");

    float alpha = 16777216.0f;
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_ALPHA, sizeof(alpha), 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = (const uint8_t *)&alpha;
    plan.num_op_attributes = 1;
    input = -128;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LeakyRelu rejects overflow in the alpha branch left shift");
    alpha = 0.01f;
    input = 127;
    test_qp[iq].scale = 16777216.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LeakyRelu rejects overflow in the identity branch left shift");
    test_qp[iq].scale = 1073741824.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LeakyRelu rejects a shift beyond the reference multiplier domain");
    test_qp[iq].scale = 1.0f;
    alpha = NAN;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LeakyRelu rejects nonfinite alpha");
    alpha = 0.01f;
    attr.data_len = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LeakyRelu rejects truncated alpha");

    test_ops[0].op_type = TIGRIS_OP_ELU;
    plan.num_op_attributes = 0;
    input = -1;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "scalar Elu runs");
    TEST_ASSERT_EQ(output, -1, "scalar Elu rounds its transformed value");
    output = 85;
    test_qp[oq].scale = 0x1p-31f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "Elu rejects an out-of-int32 LUT conversion even for an unused input value");
    TEST_ASSERT_EQ(output, 85, "Elu validates the table before writing output");
    test_qp[oq].scale = 0x1p-140f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "Elu rejects a nonfinite LUT rescaling");
    test_qp[oq].scale = 1.0f;
    test_qp[iq].scale = NAN;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "activation rejects nonfinite input scale");
    test_qp[iq].scale = 1.0f;
    test_tensors[x].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "activation rejects missing input quantization");
    test_tensors[x].quant_param_idx = iq;
    test_ops[0].op_type = TIGRIS_OP_LOG_SOFTMAX;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LogSoftmax rejects a scalar without a normalization axis");
    test_ops[0].op_type = TIGRIS_OP_L2_NORMALIZATION;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "L2Normalization rejects a scalar without a normalization axis");
}

static void test_prelu_s8_domains(void)
{
    printf("  test_prelu_s8_domains...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t iq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t aq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t oq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t shape[] = {1};
    uint16_t x = add_tensor(&plan, "x", shape, 1, 3, 1, iq);
    uint16_t y = add_tensor(&plan, "y", shape, 1, 3, 1, oq);
    int8_t slope = 127;
    one_op_s8(&plan, TIGRIS_OP_PRELU, &x, 1, &y, 1, add_weight(&slope, 1, "alpha"));
    uint8_t payload[4] = {1, 0, (uint8_t)aq, 0};
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_CONSTANT_OPERAND, sizeof(payload), 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = payload;
    plan.num_op_attributes = 1;
    int8_t input = -128, output = 85;
    void *pointers[MAX_TENSORS] = {0};
    pointers[x] = &input;
    pointers[y] = &output;
    tigris_mem_t mem = {0};
    mem.tensor_ptrs = pointers;
    mem.num_tensors = test_header.num_tensors;
    test_qp[oq].scale = 0x1p-16f;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "PRelu accepts a large representable negative product");
    TEST_ASSERT_EQ(output, -128, "PRelu clamps a representable product to int8");
    test_qp[oq].scale = 0x1p-17f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "PRelu rejects overflow while shifting the negative-branch product");
    input = 127;
    test_qp[oq].scale = 1.0f;
    test_qp[iq].scale = 16777216.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "PRelu rejects overflow in its identity branch left shift");
    test_qp[iq].scale = 1.0f;
    test_qp[aq].scale = 1073741824.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "PRelu rejects an alpha multiplier outside the reference domain");
    test_qp[aq].scale = 1.0f;
    payload[2] = 0xFFu;
    payload[3] = 0xFFu;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "PRelu rejects a constant alpha without quantization");
}

static void test_normalization_s8_domains(void)
{
    printf("  test_normalization_s8_domains...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t iq = add_quant_param(0.125f, 0, 1, NULL, NULL);
    uint16_t oq = add_quant_param(0.0625f, 127, 1, NULL, NULL);
    int32_t shape[] = {1, 4095};
    uint16_t x = add_tensor(&plan, "x", shape, 2, 3, 4095, iq);
    uint16_t y = add_tensor(&plan, "y", shape, 2, 3, 4095, oq);
    one_op_s8(&plan, TIGRIS_OP_LOG_SOFTMAX, &x, 1, &y, 1, TIGRIS_NO_WEIGHT);
    static int8_t input[33026], output[33026];
    memset(input, 0, sizeof(input));
    void *pointers[MAX_TENSORS] = {0};
    pointers[x] = input;
    pointers[y] = output;
    tigris_mem_t mem = {0};
    mem.tensor_ptrs = pointers;
    mem.num_tensors = test_header.num_tensors;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "LogSoftmax accepts 4095 equal exponentials");
    TEST_ASSERT_EQ(output[0], -6, "LogSoftmax normalizes the last safe equal row");
    test_shapes[test_tensors[x].shape_off + 1] = 4096;
    test_shapes[test_tensors[y].shape_off + 1] = 4096;
    test_tensors[x].size_bytes = test_tensors[y].size_bytes = 4096;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LogSoftmax rejects int32 overflow from 4096 equal exponentials");

    test_shapes[test_tensors[x].shape_off + 1] = 1024;
    test_shapes[test_tensors[y].shape_off + 1] = 1024;
    test_tensors[x].size_bytes = test_tensors[y].size_bytes = 1024;
    input[0] = -127;
    float boundary = (32.0f - logf(1023.0f)) / 127.0f;
    const float scales[] = {nextafterf(boundary, 0.0f), boundary, nextafterf(boundary, INFINITY)};
    for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); i++) {
        test_qp[iq].scale = scales[i];
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                       0, "LogSoftmax cutoff prevents subtraction overflow");
        TEST_ASSERT_EQ(output[0], -128, "LogSoftmax saturates the sample at its subtraction boundary");
        TEST_ASSERT_EQ(output[1], 16, "LogSoftmax preserves the other normalized samples");
    }
    const float invalid_scales[] = {0x1p-27f, 0x1p-26f, 16.0f};
    for (size_t i = 0; i < sizeof(invalid_scales) / sizeof(invalid_scales[0]); i++) {
        test_qp[iq].scale = invalid_scales[i];
        TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                    "LogSoftmax rejects an undefined reference scaling domain");
    }
    test_qp[iq].scale = 0.125f;
    test_qp[oq].zero_point = 0;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LogSoftmax rejects a noncanonical output zero point");
    test_qp[oq].zero_point = 127;
    test_qp[oq].scale = 0.125f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "LogSoftmax rejects a noncanonical output scale");

    test_ops[0].op_type = TIGRIS_OP_L2_NORMALIZATION;
    test_qp[oq].scale = 1.0f / 128.0f;
    test_qp[oq].zero_point = 0;
    test_qp[iq].zero_point = -128;
    memset(input, 127, sizeof(input));
    test_shapes[test_tensors[x].shape_off + 1] = 33025;
    test_shapes[test_tensors[y].shape_off + 1] = 33025;
    test_tensors[x].size_bytes = test_tensors[y].size_bytes = 33025;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "L2Normalization accepts 33025 maximal centered squares");
    TEST_ASSERT_EQ(output[0], 1, "L2Normalization scales a near-limit accumulator");
    test_shapes[test_tensors[x].shape_off + 1] = 33026;
    test_shapes[test_tensors[y].shape_off + 1] = 33026;
    test_tensors[x].size_bytes = test_tensors[y].size_bytes = 33026;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "L2Normalization rejects int32 sum-of-squares overflow");
    memset(input, -128, sizeof(input));
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "L2Normalization accepts a large zero-norm vector");
    TEST_ASSERT_EQ(output[0], 0, "L2Normalization emits zero for a zero-norm vector");
    test_qp[oq].zero_point = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "L2Normalization rejects a nonzero output zero point");
    test_qp[oq].zero_point = 0;
    test_qp[oq].scale = 1.0f / 256.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "L2Normalization rejects a noncanonical output scale");
}

static void test_reduce_s8_domains(void)
{
    printf("  test_reduce_s8_domains...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t iq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t oq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t input_shape[] = {1, 3, 1}, output_shape[] = {1, 1, 1};
    uint16_t x = add_tensor(&plan, "x", input_shape, 3, 3, 3, iq);
    uint16_t y = add_tensor(&plan, "y", output_shape, 3, 3, 1, oq);
    one_op_s8(&plan, TIGRIS_OP_REDUCE_MAX, &x, 1, &y, 1, TIGRIS_NO_WEIGHT);
    uint8_t axis = 1;
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_AXES, 1, 0};
    plan.op_attributes = &attr;
    plan.op_attribute_data = &axis;
    plan.num_op_attributes = 1;
    int8_t small[] = {-4, 3, -2}, output = 85;
    void *pointers[MAX_TENSORS] = {0};
    pointers[x] = small; pointers[y] = &output;
    tigris_mem_t mem = {0};
    mem.tensor_ptrs = pointers; mem.num_tensors = test_header.num_tensors;
    const uint8_t kinds[] = {TIGRIS_OP_REDUCE_MAX, TIGRIS_OP_REDUCE_MIN};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        test_ops[0].op_type = kinds[k];
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                       0, "extreme reduction with identical encoding runs");
        TEST_ASSERT_EQ(output, k == 0u ? 3 : -4, "extreme reduction selects the input byte");
        test_qp[oq].scale = 2.0f;
        TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                    "extreme reduction rejects changed scale");
        test_qp[oq].scale = 1.0f;
        test_qp[oq].zero_point = 1;
        TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                    "extreme reduction rejects changed zero point");
        test_qp[oq].zero_point = 0;
    }
    test_ops[0].op_type = TIGRIS_OP_REDUCE_SUM;
    test_qp[iq].scale = 16777216.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) == 0,
                "SUM accepts a large multiplier when the actual shifted sum fits");
    small[0] = 127;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "SUM rejects a left-shift overflow before clamping");
    test_qp[iq].scale = 1073741824.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "SUM rejects a multiplier shift outside the reference domain");
    test_qp[iq].scale = 1.0f;

    const uint32_t max_positive_count = 16909320u;
    int8_t *large = malloc(max_positive_count + 1u);
    TEST_ASSERT(large != NULL, "SUM overflow fixture allocated");
    if (!large) return;
    pointers[x] = large;
    memset(large, 127, max_positive_count + 1u);
    test_qp[iq].zero_point = 127;
    test_shapes[test_tensors[x].shape_off + 1u] = (int32_t)max_positive_count;
    test_tensors[x].size_bytes = max_positive_count;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "SUM accepts the last representable positive raw accumulator");
    TEST_ASSERT_EQ(output, 0, "SUM subtracts the zero-point contribution after accumulation");
    test_shapes[test_tensors[x].shape_off + 1u]++;
    test_tensors[x].size_bytes++;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "SUM rejects raw accumulator overflow even when centered values are zero");
    test_qp[iq].zero_point = -128;
    test_shapes[test_tensors[x].shape_off + 1u] = 8421505;
    test_tensors[x].size_bytes = 8421505u;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "SUM rejects a centered sum outside int32 despite a valid raw sum");
    memset(large, -127, max_positive_count);
    test_shapes[test_tensors[x].shape_off + 1u] = (int32_t)max_positive_count;
    test_tensors[x].size_bytes = max_positive_count;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "SUM permits the reference's int64 zero-point product");
    TEST_ASSERT_EQ(output, 127, "SUM clamps a valid corrected sum");
    memset(large, -128, 16777217u);
    test_shapes[test_tensors[x].shape_off + 1u] = 16777216;
    test_tensors[x].size_bytes = 16777216u;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "SUM accepts INT32_MIN as a raw accumulator");
    TEST_ASSERT_EQ(output, 0, "SUM corrects the valid negative boundary");
    test_shapes[test_tensors[x].shape_off + 1u]++;
    test_tensors[x].size_bytes++;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "SUM rejects raw accumulator underflow");
    free(large);
}

static void test_cumsum_s8_domains(void)
{
    printf("  test_cumsum_s8_domains...\n");
    plan_reset();
    tigris_plan_t plan;
    uint16_t iq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    uint16_t oq = add_quant_param(1.0f, 0, 1, NULL, NULL);
    int32_t shape[] = {1, 32, 1};
    uint16_t x = add_tensor(&plan, "x", shape, 3, 3, 32, iq);
    uint16_t y = add_tensor(&plan, "y", shape, 3, 3, 32, oq);
    one_op_s8(&plan, TIGRIS_OP_CUMSUM, &x, 1, &y, 1, TIGRIS_NO_WEIGHT);
    uint8_t data[] = {1, 0, 0};
    tigris_op_attribute_t attrs[] = {
        {0, TIGRIS_OP_ATTR_AXES, 1, 0}, {0, TIGRIS_OP_ATTR_CUMSUM_OPTIONS, 2, 1},
    };
    plan.op_attributes = attrs; plan.op_attribute_data = data; plan.num_op_attributes = 2;
    int8_t input[33], output[33];
    void *pointers[MAX_TENSORS] = {0};
    pointers[x] = input; pointers[y] = output;
    tigris_mem_t mem = {0};
    mem.tensor_ptrs = pointers; mem.num_tensors = test_header.num_tensors;
    for (int negative = 0; negative < 2; negative++) {
        memset(input, negative ? -128 : 127, sizeof(input));
        for (uint8_t exclusive = 0; exclusive < 2u; exclusive++) {
            for (uint8_t reverse = 0; reverse < 2u; reverse++) {
                data[1] = exclusive; data[2] = reverse;
                for (int t = 0; t < 2; t++) {
                    test_shapes[test_tensors[t].shape_off + 1u] = 32;
                    test_tensors[t].size_bytes = 32;
                }
                TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                               0, "CumSum accepts a representable boundary prefix");
                TEST_ASSERT_EQ(output[reverse ? 0 : 31], negative ? -128 : 127,
                               "CumSum clamps each representable prefix");
                for (int t = 0; t < 2; t++) {
                    test_shapes[test_tensors[t].shape_off + 1u] = 33;
                    test_tensors[t].size_bytes = 33;
                }
                TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                            "CumSum rejects overflow even in the final unused exclusive update");
            }
        }
    }
    for (int t = 0; t < 2; t++) {
        test_shapes[test_tensors[t].shape_off + 1u] = 3;
        test_tensors[t].size_bytes = 3;
    }
    data[1] = 1; data[2] = 0;
    test_qp[iq].zero_point = 5;
    memset(input, 5, sizeof(input));
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL),
                   0, "CumSum executes the reference's nonzero initial offset");
    for (int i = 0; i < 3; i++) TEST_ASSERT_EQ(output[i], -5, "exclusive CumSum retains the initial offset");
    test_qp[iq].zero_point = 0;
    test_qp[iq].scale = 524288.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects an output multiplier equal to one");
    test_qp[iq].scale = 1048576.0f;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects an output multiplier greater than one");
    test_qp[iq].scale = 1.0f;
    attrs[1].data_len = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects truncated options");
    attrs[1].data_len = 2;
    data[1] = 2;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects a nonboolean exclusive flag");
    data[1] = 0; data[2] = 2;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects a nonboolean reverse flag");
    data[2] = 0;
    plan.num_op_attributes = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum requires explicit options");
    plan.num_op_attributes = 2;
    data[0] = 3;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum rejects an out-of-range axis");
    data[0] = 1;
    mem.tile.active = 1;
    TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &test_ops[0], 0, &mem, NULL) != 0,
                "CumSum refuses partial-axis execution");
}

/* Main */

int main(void)
{
    printf("TiGrIS Int8 Kernel Tests\n\n");

    test_relu_s8();
    test_relu6_s8_rounds_ceiling();
    test_binary_s8_contract();
    test_tiled_rank3_binary_s8();
    test_reshape_s8();
    test_conv2d_s8();
    test_fc_s8();
    test_fc_s8_shift_beyond_31_bits();
    test_global_avg_pool_s8();
    test_global_avg_pool_s8_average_rounding();
    test_global_max_pool_s8();
    test_matmul_s8_row_band();
    test_hardswish_s8();
    test_requantize_s8();
    test_per_channel_binary_s8();
    test_tanh_s8();
    test_gelu_s8();
    test_softmax_s8();
    test_conv1d_s8();
    test_reduce_mean_s8();
    test_binary_requant_is_read_from_the_plan();
    test_split_s8();
    test_mul_s8_tflite_rounding();
    test_split_s8_inner_axis();
    test_concat_s8_height();
    test_concat_s8_small_scales();
    test_depthwise_multiplier_s8();
    test_pad_s8();
    test_binary_s8_constant_operand();
    test_binary_s8_both_broadcast();
    test_binary_s8_constant_shape();
    test_unsupported_op_s8();
    test_elementwise_reference_goldens();
    test_activation_s8_domains();
    test_prelu_s8_domains();
    test_normalization_s8_domains();
    test_reduce_s8_domains();
    test_cumsum_s8_domains();

    return test_report();
}

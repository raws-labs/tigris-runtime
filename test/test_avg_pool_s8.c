/**
 * @file test_avg_pool_s8.c
 * @brief Reference int8 AveragePool execution and accelerator-route tests.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_accel_policy.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"
#include "tigris_mem.h"

#define TEST_H             16
#define TEST_W              4
#define TEST_C              8
#define TEST_ELEMENTS      (TEST_H * TEST_W * TEST_C)
#define NORMAL_FAST_SIZE 2048u
#define TILED_FAST_SIZE   512u
#define TEST_SLOW_SIZE   2048u

static int tests_run;
static int tests_passed;
static int tests_failed;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", \
                __func__, __LINE__, msg); \
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

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    tigris_stage_t stage;
    int32_t shapes[8];
    uint16_t indices[7];
    tigris_quant_param_t quant_params[2];
    tigris_plan_t plan;
} avg_fixture_t;

static void build_fixture(avg_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->header.num_stages = 1;
    fx->header.num_quant_params = 2;
    fx->header.num_model_inputs = 1;
    fx->header.num_model_outputs = 1;
    fx->header.model_io_off = 5;

    for (uint16_t i = 0; i < 2; i++) {
        fx->tensors[i].size_bytes = TEST_ELEMENTS;
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].dtype = 3; /* ONNX INT8 */
        fx->tensors[i].quant_param_idx = i;
        fx->shapes[i * 4] = 1;
        fx->shapes[i * 4 + 1] = TEST_H;
        fx->shapes[i * 4 + 2] = TEST_W;
        fx->shapes[i * 4 + 3] = TEST_C;
    }
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    fx->quant_params[0].scale = 0.5f;
    fx->quant_params[0].zero_point = -3;
    fx->quant_params[0].num_channels = 1;
    fx->quant_params[1].scale = 0.25f;
    fx->quant_params[1].zero_point = 5;
    fx->quant_params[1].num_channels = 1;

    fx->indices[0] = 0; /* op input */
    fx->indices[1] = 1; /* op output */
    fx->indices[2] = 0; /* stage op */
    fx->indices[3] = 0; /* stage input */
    fx->indices[4] = 1; /* stage output */
    fx->indices[5] = 0; /* model input */
    fx->indices[6] = 1; /* model output */

    fx->op.op_type = TIGRIS_OP_AVG_POOL;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;
    fx->op.weight_idx = TIGRIS_NO_WEIGHT;
    fx->op.bias_idx = TIGRIS_NO_WEIGHT;
    fx->op.act_min = -128;
    fx->op.act_max = 127;
    fx->op.spatial.kernel_h = 3;
    fx->op.spatial.kernel_w = 3;
    fx->op.spatial.stride_h = 1;
    fx->op.spatial.stride_w = 1;
    fx->op.spatial.pad_top = 1;
    fx->op.spatial.pad_bottom = 1;
    fx->op.spatial.pad_left = 1;
    fx->op.spatial.pad_right = 1;

    fx->stage.ops_off = 2;
    fx->stage.ops_count = 1;
    fx->stage.inputs_off = 3;
    fx->stage.inputs_count = 1;
    fx->stage.outputs_off = 4;
    fx->stage.outputs_count = 1;
    fx->stage.tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    fx->stage.chain_id = TIGRIS_NO_CHAIN;

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.stages = &fx->stage;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.quant_params = fx->quant_params;
    fx->plan.num_quant_params = 2;
    fx->plan.model_inputs = &fx->indices[5];
    fx->plan.model_outputs = &fx->indices[6];
}

static int32_t round_divide_away_from_zero(int32_t value, int32_t divisor)
{
    int32_t half = divisor / 2;
    return value > 0 ? (value + half) / divisor
                     : (value - half) / divisor;
}

static void compute_expected(const int8_t *input, int8_t *expected)
{
    for (int oh = 0; oh < TEST_H; oh++) {
        for (int ow = 0; ow < TEST_W; ow++) {
            int ih0 = oh - 1;
            int iw0 = ow - 1;
            for (int c = 0; c < TEST_C; c++) {
                int32_t sum = 0;
                int32_t count = 0;
                for (int kh = 0; kh < 3; kh++) {
                    int ih = ih0 + kh;
                    if (ih < 0 || ih >= TEST_H)
                        continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int iw = iw0 + kw;
                        if (iw < 0 || iw >= TEST_W)
                            continue;
                        sum += input[(ih * TEST_W + iw) * TEST_C + c];
                        count++;
                    }
                }

                /* scale_in / scale_out == 2, input zp=-3, output zp=5. */
                int32_t centered = sum + 3 * count;
                int32_t q = round_divide_away_from_zero(
                                2 * centered, count) + 5;
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                expected[(oh * TEST_W + ow) * TEST_C + c] = (int8_t)q;
            }
        }
    }
}

static tigris_exec_error_t run_executor(
    avg_fixture_t *fx, const int8_t *input, int8_t *output,
    uint32_t fast_size, tigris_exec_stats_t *stats)
{
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[NORMAL_FAST_SIZE];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[TEST_SLOW_SIZE];
    tigris_mem_t mem;

    if (tigris_mem_init(&mem, ptrs, 2, fast, fast_size,
                        slow, sizeof(slow)) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    if (tigris_mem_alloc_slow(&mem, 0, TEST_ELEMENTS) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(ptrs[0], input, TEST_ELEMENTS);

    tigris_exec_error_t err = tigris_run(
        &fx->plan, &mem, tigris_dispatch_kernel_s8, NULL, stats);
    if (err == TIGRIS_EXEC_OK && ptrs[1])
        memcpy(output, ptrs[1], TEST_ELEMENTS);
    return err;
}

static int run_direct(avg_fixture_t *fx, const int8_t *input, int8_t *output,
                      int tile_active)
{
    void *ptrs[2] = {(void *)input, output};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;
    if (tile_active) {
        mem.tile.active = 1;
        mem.tile.in_h = TEST_H;
        mem.tile.out_h = TEST_H;
        mem.tile.in_w = TEST_W;
        mem.tile.out_w = TEST_W;
        mem.tile.pad_top = fx->op.spatial.pad_top;
        mem.tile.pad_bottom = fx->op.spatial.pad_bottom;
    }
    return tigris_dispatch_kernel_s8(
        &fx->plan, &fx->op, 0, &mem, NULL);
}

static int run_route(avg_fixture_t *fx, tigris_accel_backend_t backend,
                     const int8_t *input, int8_t *output,
                     int tile_active, int *handled)
{
    void *ptrs[2] = {(void *)input, output};
    tigris_mem_t mem;
    memset(&mem, 0, sizeof(mem));
    mem.tensor_ptrs = ptrs;
    mem.num_tensors = 2;
    if (tile_active) {
        mem.tile.active = 1;
        mem.tile.in_h = TEST_H;
        mem.tile.out_h = TEST_H;
        mem.tile.in_w = TEST_W;
        mem.tile.out_w = TEST_W;
        mem.tile.pad_top = fx->op.spatial.pad_top;
        mem.tile.pad_bottom = fx->op.spatial.pad_bottom;
    }
    return tigris_accel_try_s8_ref(
        backend, &fx->plan, &fx->op, 0, &mem, NULL, handled);
}

static void assert_array_eq(const int8_t *actual, const int8_t *expected,
                            const char *message)
{
    tests_run++;
    if (memcmp(actual, expected, TEST_ELEMENTS) != 0) {
        int mismatch = 0;
        while (mismatch < TEST_ELEMENTS &&
               actual[mismatch] == expected[mismatch])
            mismatch++;
        fprintf(stderr,
                "  FAIL: %s: %s (index %d got %d, expected %d)\n",
                __func__, message, mismatch,
                mismatch < TEST_ELEMENTS ? actual[mismatch] : 0,
                mismatch < TEST_ELEMENTS ? expected[mismatch] : 0);
        tests_failed++;
    } else {
        tests_passed++;
    }
}

static void test_same_quant_tie_rounding(void)
{
    printf("  test_same_quant_tie_rounding...\n");
    avg_fixture_t fx;
    int8_t output[1];
    void *ptrs[2];
    tigris_mem_t mem;
    build_fixture(&fx);

    fx.shapes[1] = 1; fx.shapes[2] = 2; fx.shapes[3] = 1;
    fx.shapes[5] = 1; fx.shapes[6] = 1; fx.shapes[7] = 1;
    fx.tensors[0].size_bytes = 2;
    fx.tensors[1].size_bytes = 1;
    fx.quant_params[0].scale = 1.0f;
    fx.quant_params[0].zero_point = 0;
    fx.quant_params[1] = fx.quant_params[0];
    fx.op.spatial.kernel_h = 1;
    fx.op.spatial.kernel_w = 2;
    fx.op.spatial.pad_top = 0;
    fx.op.spatial.pad_bottom = 0;
    fx.op.spatial.pad_left = 0;
    fx.op.spatial.pad_right = 0;

    {
        int8_t input[2] = {-1, 0};
        ptrs[0] = input; ptrs[1] = output;
        memset(&mem, 0, sizeof(mem));
        mem.tensor_ptrs = ptrs; mem.num_tensors = 2;
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                           &fx.plan, &fx.op, 0, &mem, NULL), 0,
                       "negative tie AveragePool succeeds");
        TEST_ASSERT_EQ(output[0], -1,
                       "negative half tie rounds away from zero");
    }
    {
        int8_t input[2] = {0, 1};
        ptrs[0] = input; ptrs[1] = output;
        TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(
                           &fx.plan, &fx.op, 0, &mem, NULL), 0,
                       "positive tie AveragePool succeeds");
        TEST_ASSERT_EQ(output[0], 1,
                       "positive half tie rounds away from zero");
    }
}

static void test_asymmetric_quant_normal_and_tiled(void)
{
    printf("  test_asymmetric_quant_normal_and_tiled...\n");
    avg_fixture_t fx;
    int8_t input[TEST_ELEMENTS];
    int8_t expected[TEST_ELEMENTS];
    int8_t normal[TEST_ELEMENTS];
    int8_t tiled[TEST_ELEMENTS];
    tigris_exec_stats_t normal_stats, tiled_stats;
    build_fixture(&fx);

    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (int8_t)(((i * 7) % 31) - 15);
    compute_expected(input, expected);

    TEST_ASSERT_EQ(run_executor(&fx, input, normal, NORMAL_FAST_SIZE,
                                &normal_stats),
                   TIGRIS_EXEC_OK, "normal asymmetric AveragePool succeeds");
    TEST_ASSERT_EQ(normal_stats.stages_normal, 1,
                   "large arena selects normal execution");
    assert_array_eq(normal, expected,
                    "normal asymmetric output matches scalar reference");

    TEST_ASSERT_EQ(run_executor(&fx, input, tiled, TILED_FAST_SIZE,
                                &tiled_stats),
                   TIGRIS_EXEC_OK, "tiled asymmetric AveragePool succeeds");
    TEST_ASSERT_EQ(tiled_stats.stages_tiled, 1,
                   "tight arena selects tiled execution");
    assert_array_eq(tiled, expected,
                    "tiled asymmetric output matches scalar reference");
    assert_array_eq(tiled, normal,
                    "normal and tiled asymmetric outputs agree");
}

static void test_accelerator_pool_routes(void)
{
    printf("  test_accelerator_pool_routes...\n");
    avg_fixture_t fx;
    int8_t input[TEST_ELEMENTS];
    int8_t direct[TEST_ELEMENTS];
    int8_t esp[TEST_ELEMENTS];
    int8_t cmsis[TEST_ELEMENTS];
    int handled = 0;
    build_fixture(&fx);
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (int8_t)(((i * 5) % 29) - 14);

    TEST_ASSERT_EQ(run_direct(&fx, input, direct, 0), 0,
                   "direct asymmetric AveragePool succeeds");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "ESP-NN asymmetric AveragePool routes to s8_ref");
    TEST_ASSERT_EQ(run_route(&fx, TIGRIS_ACCEL_ESP_NN,
                             input, esp, 0, &handled), 0,
                   "ESP-NN asymmetric fallback succeeds");
    TEST_ASSERT_EQ(handled, 1, "ESP-NN AveragePool adapter is bypassed");
    assert_array_eq(esp, direct,
                    "ESP-NN asymmetric fallback matches s8_ref");

    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN asymmetric AveragePool routes to s8_ref");
    TEST_ASSERT_EQ(run_route(&fx, TIGRIS_ACCEL_CMSIS_NN,
                             input, cmsis, 0, &handled), 0,
                   "CMSIS-NN asymmetric fallback succeeds");
    TEST_ASSERT_EQ(handled, 1, "CMSIS-NN AveragePool adapter is bypassed");
    assert_array_eq(cmsis, direct,
                    "CMSIS-NN asymmetric fallback matches s8_ref");

    fx.quant_params[1] = fx.quant_params[0];
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "ESP-NN compatible normal AveragePool remains native");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "CMSIS-NN compatible normal AveragePool remains native");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 1),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "ESP-NN tiled AveragePool bypasses non-tile-aware adapter");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 1),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN tiled AveragePool remains on s8_ref route");

    fx.tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "ESP-NN routes one-sided quant metadata to s8_ref");
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_CMSIS_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_S8_REF,
                   "CMSIS-NN routes one-sided quant metadata to s8_ref");

    fx.tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    TEST_ASSERT_EQ(tigris_accel_pre_route(
                       TIGRIS_ACCEL_ESP_NN, &fx.plan, &fx.op, 0),
                   TIGRIS_ACCEL_ROUTE_ADAPTER,
                   "unquantized input/output preserve native byte averaging");
}

int main(void)
{
    printf("TiGrIS Int8 AveragePool Tests\n\n");
    test_same_quant_tie_rounding();
    test_asymmetric_quant_normal_and_tiled();
    test_accelerator_pool_routes();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed ? 1 : 0;
}

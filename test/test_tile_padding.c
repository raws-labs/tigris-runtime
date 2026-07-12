/**
 * @file test_tile_padding.c
 * @brief Schema-v2 large-padding and dilated-convolution execution tests.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_mem.h"

#define INPUT_H             512
#define OUTPUT_H            712
#define HIGH_PADDING        300u
#define DILATION_H          100u
#define KERNEL_H            5
#define NORMAL_FAST_SIZE    8192u
#define TILED_FAST_SIZE     4096u
#define SLOW_SIZE           8192u

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
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stages[2];
    int32_t shapes[12];
    uint16_t indices[12];
    tigris_weight_entry_t weight_entries[2];
    _Alignas(float) uint8_t weight_blob[6 * sizeof(float)];
    char strings[1];
    tigris_plan_t plan;
} plan_fixture_t;

typedef enum {
    RUN_NORMAL,
    RUN_TILED,
    RUN_CHAIN
} run_mode_t;

typedef struct {
    uint32_t max_pad_top;
    uint32_t max_pad_bottom;
    uint32_t dilated_calls;
    uint32_t tiled_dilated_calls;
} observe_ctx_t;

static const float first_weights[KERNEL_H] = {
    0.5f, -1.0f, 1.5f, 2.0f, -0.75f
};
static const float second_weight = 1.25f;

static void build_plan(plan_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.num_tensors = 3;
    fx->header.num_ops = 2;
    fx->header.num_stages = 2;
    fx->header.num_weights = 2;
    fx->header.num_model_inputs = 1;
    fx->header.num_model_outputs = 1;
    fx->header.model_io_off = 10;

    const int32_t heights[3] = {INPUT_H, OUTPUT_H, OUTPUT_H};
    for (uint16_t i = 0; i < 3; i++) {
        fx->tensors[i].size_bytes = (uint32_t)heights[i] * sizeof(float);
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].dtype = 1; /* ONNX FLOAT */
        fx->tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        fx->shapes[i * 4] = 1;
        fx->shapes[i * 4 + 1] = heights[i];
        fx->shapes[i * 4 + 2] = 1;
        fx->shapes[i * 4 + 3] = 1;
    }
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    fx->indices[0] = 0;  /* op 0 input */
    fx->indices[1] = 1;  /* op 0 output */
    fx->indices[2] = 1;  /* op 1 input */
    fx->indices[3] = 2;  /* op 1 output */
    fx->indices[4] = 0;  /* stage 0 op */
    fx->indices[5] = 0;  /* stage 0 input */
    fx->indices[6] = 1;  /* stage 0 output */
    fx->indices[7] = 1;  /* stage 1 op */
    fx->indices[8] = 1;  /* stage 1 input */
    fx->indices[9] = 2;  /* stage 1 output */
    fx->indices[10] = 0; /* model input */
    fx->indices[11] = 2; /* model output */

    for (uint16_t i = 0; i < 2; i++) {
        fx->ops[i].op_type = TIGRIS_OP_CONV;
        fx->ops[i].num_inputs = 1;
        fx->ops[i].num_outputs = 1;
        fx->ops[i].stage = (uint8_t)i;
        fx->ops[i].inputs_off = (uint16_t)(i * 2);
        fx->ops[i].outputs_off = (uint16_t)(i * 2 + 1);
        fx->ops[i].weight_idx = i;
        fx->ops[i].bias_idx = TIGRIS_NO_WEIGHT;
        fx->ops[i].spatial.kernel_w = 1;
        fx->ops[i].spatial.stride_h = 1;
        fx->ops[i].spatial.stride_w = 1;
        fx->ops[i].spatial.dilation_h = 1;
        fx->ops[i].spatial.dilation_w = 1;
        fx->ops[i].spatial.group = 1;
    }
    fx->ops[0].spatial.kernel_h = KERNEL_H;
    fx->ops[0].spatial.pad_top = HIGH_PADDING;
    fx->ops[0].spatial.pad_bottom = HIGH_PADDING;
    fx->ops[0].spatial.dilation_h = DILATION_H;
    fx->ops[1].spatial.kernel_h = 1;

    fx->weight_entries[0].offset = 0;
    fx->weight_entries[0].size_bytes = sizeof(first_weights);
    fx->weight_entries[1].offset = sizeof(first_weights);
    fx->weight_entries[1].size_bytes = sizeof(second_weight);
    memcpy(fx->weight_blob, first_weights, sizeof(first_weights));
    memcpy(fx->weight_blob + sizeof(first_weights),
           &second_weight, sizeof(second_weight));

    fx->stages[0].ops_off = 4;
    fx->stages[0].ops_count = 1;
    fx->stages[0].inputs_off = 5;
    fx->stages[0].inputs_count = 1;
    fx->stages[0].outputs_off = 6;
    fx->stages[0].outputs_count = 1;
    fx->stages[1].ops_off = 7;
    fx->stages[1].ops_count = 1;
    fx->stages[1].inputs_off = 8;
    fx->stages[1].inputs_count = 1;
    fx->stages[1].outputs_off = 9;
    fx->stages[1].outputs_count = 1;
    for (uint16_t i = 0; i < 2; i++) {
        fx->stages[i].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        fx->stages[i].chain_id = TIGRIS_NO_CHAIN;
    }

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = fx->ops;
    fx->plan.stages = fx->stages;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.strings = fx->strings;
    fx->plan.weight_entries = fx->weight_entries;
    fx->plan.weight_blob = fx->weight_blob;
    fx->plan.model_inputs = &fx->indices[10];
    fx->plan.model_outputs = &fx->indices[11];
}

static void configure_mode(plan_fixture_t *fx, run_mode_t mode)
{
    for (uint16_t i = 0; i < 2; i++) {
        fx->stages[i].chain_id = TIGRIS_NO_CHAIN;
        fx->stages[i].chain_len = 0;
        fx->stages[i].chain_tile_h = 0;
    }
    if (mode == RUN_CHAIN) {
        for (uint16_t i = 0; i < 2; i++) {
            fx->stages[i].chain_id = 0;
            fx->stages[i].chain_len = 2;
        }
        fx->stages[0].chain_tile_h = 64;
    }
}

static int observing_reference_kernel(
    const tigris_plan_t *plan, const tigris_op_t *op,
    uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    observe_ctx_t *ctx = (observe_ctx_t *)user_ctx;
    if (op_index == 0) {
        ctx->dilated_calls++;
        if (mem->tile.active) {
            uint32_t pad_top = mem->tile.pad_top;
            uint32_t pad_bottom = mem->tile.pad_bottom;
            ctx->tiled_dilated_calls++;
            if (pad_top > ctx->max_pad_top)
                ctx->max_pad_top = pad_top;
            if (pad_bottom > ctx->max_pad_bottom)
                ctx->max_pad_bottom = pad_bottom;
        }
    }
    return tigris_dispatch_kernel(plan, op, op_index, mem, NULL);
}

static tigris_exec_error_t run_plan(
    plan_fixture_t *fx, run_mode_t mode, const float *input, float *output,
    observe_ctx_t *observe, tigris_exec_stats_t *stats)
{
    void *ptrs[3];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[NORMAL_FAST_SIZE];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[SLOW_SIZE];
    tigris_mem_t mem;
    uint32_t fast_size = mode == RUN_NORMAL
        ? NORMAL_FAST_SIZE : TILED_FAST_SIZE;

    configure_mode(fx, mode);
    memset(observe, 0, sizeof(*observe));
    memset(stats, 0, sizeof(*stats));

    tigris_mem_error_t merr = tigris_mem_init(
        &mem, ptrs, 3, fast, fast_size, slow, sizeof(slow));
    if (merr != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    merr = tigris_mem_alloc_slow(&mem, 0, INPUT_H * sizeof(float));
    if (merr != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(ptrs[0], input, INPUT_H * sizeof(float));

    tigris_exec_error_t err = tigris_run(
        &fx->plan, &mem, observing_reference_kernel, observe, stats);
    if (err == TIGRIS_EXEC_OK && ptrs[2])
        memcpy(output, ptrs[2], OUTPUT_H * sizeof(float));
    return err;
}

static void compute_expected(const float *input, float *expected)
{
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        float acc = 0.0f;
        for (int kh = 0; kh < KERNEL_H; kh++) {
            int ih = oh - (int)HIGH_PADDING + kh * (int)DILATION_H;
            if (ih >= 0 && ih < INPUT_H)
                acc += input[ih] * first_weights[kh];
        }
        expected[oh] = acc * second_weight;
    }
}

static int arrays_close(const float *actual, const float *expected)
{
    for (int i = 0; i < OUTPUT_H; i++) {
        if (fabsf(actual[i] - expected[i]) > 1.0e-6f) {
            fprintf(stderr,
                    "  mismatch at row %d: got %.9g, expected %.9g\n",
                    i, (double)actual[i], (double)expected[i]);
            return 0;
        }
    }
    return 1;
}

static void test_large_padding_and_dilation_modes(void)
{
    printf("  test_large_padding_and_dilation_modes...\n");

    plan_fixture_t fx;
    float input[INPUT_H];
    float expected[OUTPUT_H];
    float normal[OUTPUT_H];
    float tiled[OUTPUT_H];
    float chain[OUTPUT_H];
    observe_ctx_t normal_observe, tiled_observe, chain_observe;
    tigris_exec_stats_t normal_stats, tiled_stats, chain_stats;

    build_plan(&fx);
    for (int i = 0; i < INPUT_H; i++)
        input[i] = (float)((i % 23) - 11) * 0.125f;
    compute_expected(input, expected);

    TEST_ASSERT_EQ(sizeof(((tigris_tile_ctx_t *)0)->pad_top), sizeof(uint16_t),
                   "tile top padding stores schema-v2 width");
    TEST_ASSERT_EQ(sizeof(((tigris_tile_ctx_t *)0)->pad_bottom), sizeof(uint16_t),
                   "tile bottom padding stores schema-v2 width");

    TEST_ASSERT_EQ(run_plan(&fx, RUN_NORMAL, input, normal,
                            &normal_observe, &normal_stats),
                   TIGRIS_EXEC_OK, "normal dilated plan succeeds");
    TEST_ASSERT_EQ(normal_stats.stages_normal, 2,
                   "normal mode executes both stages normally");
    TEST_ASSERT_EQ(normal_observe.dilated_calls, 1,
                   "normal mode executes dilated reference kernel");
    TEST_ASSERT_EQ(normal_observe.tiled_dilated_calls, 0,
                   "normal reference kernel has no tile override");
    TEST_ASSERT(arrays_close(normal, expected),
                "normal output matches scalar reference");

    TEST_ASSERT_EQ(run_plan(&fx, RUN_TILED, input, tiled,
                            &tiled_observe, &tiled_stats),
                   TIGRIS_EXEC_OK, "standalone tiled dilated plan succeeds");
    TEST_ASSERT_EQ(tiled_stats.stages_tiled, 2,
                   "standalone mode tiles both stages");
    TEST_ASSERT(tiled_observe.tiled_dilated_calls > 1,
                "standalone dilated convolution spans tiles");
    TEST_ASSERT_EQ(tiled_observe.max_pad_top, HIGH_PADDING,
                   "standalone tile retains 300-row top padding");
    TEST_ASSERT_EQ(tiled_observe.max_pad_bottom, HIGH_PADDING,
                   "standalone tile retains 300-row bottom padding");
    TEST_ASSERT(arrays_close(tiled, expected),
                "standalone tiled output matches scalar reference");

    TEST_ASSERT_EQ(run_plan(&fx, RUN_CHAIN, input, chain,
                            &chain_observe, &chain_stats),
                   TIGRIS_EXEC_OK, "chained dilated plan succeeds");
    TEST_ASSERT_EQ(chain_stats.stages_chain, 2,
                   "chain mode streams both stages");
    TEST_ASSERT(chain_observe.tiled_dilated_calls > 1,
                "chained dilated convolution spans tiles");
    TEST_ASSERT_EQ(chain_observe.max_pad_top, HIGH_PADDING,
                   "chain tile retains 300-row top padding");
    TEST_ASSERT_EQ(chain_observe.max_pad_bottom, HIGH_PADDING,
                   "chain tile retains 300-row bottom padding");
    TEST_ASSERT(arrays_close(chain, expected),
                "chained output matches scalar reference");
    TEST_ASSERT(arrays_close(tiled, normal),
                "standalone tiled and normal outputs agree");
    TEST_ASSERT(arrays_close(chain, normal),
                "chained and normal outputs agree");
}

int main(void)
{
    printf("TiGrIS Tile Padding Tests\n\n");
    test_large_padding_and_dilation_modes();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed ? 1 : 0;
}

/**
 * @file test_global_reduce.c
 * @brief A global reduction executed in bands must equal the single pass.
 *
 * The executor walks a global reduction along its input, one band of rows at
 * a time, carrying a running partial between bands. This compares the tiled
 * result against the same plan run whole, for both dispatchers, and checks
 * that the tiled run really was tiled.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_mem.h"

#define TEST_N   1
#define TEST_H  16
#define TEST_W   4
#define TEST_C   8
#define TEST_ELEMENTS (TEST_N * TEST_H * TEST_W * TEST_C)

#define WHOLE_FAST_SIZE 16384u
#define SLOW_SIZE        8192u

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
    long got_ = (long)(a); \
    long want_ = (long)(b); \
    tests_run++; \
    if (got_ != want_) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %ld, expected %ld)\n", \
                __func__, __LINE__, msg, got_, want_); \
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
    tigris_tile_plan_t tile_plan;
    tigris_quant_param_t quant_params[2];
    int32_t shapes[8];
    uint16_t indices[7];
    tigris_plan_t plan;
} reduce_fixture_t;

/** One GlobalAveragePool stage carrying a height tile plan. */
static void build_fixture(reduce_fixture_t *fx, uint32_t elem_size)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.version = TIGRIS_SCHEMA_VERSION;
    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->header.num_stages = 1;
    fx->header.num_tile_plans = 1;
    fx->header.num_quant_params = 2;
    fx->header.num_model_inputs = 1;
    fx->header.num_model_outputs = 1;

    fx->shapes[0] = TEST_N; fx->shapes[1] = TEST_H;
    fx->shapes[2] = TEST_W; fx->shapes[3] = TEST_C;
    fx->shapes[4] = TEST_N; fx->shapes[5] = 1;
    fx->shapes[6] = 1;      fx->shapes[7] = TEST_C;

    fx->tensors[0].size_bytes = TEST_ELEMENTS * elem_size;
    fx->tensors[0].shape_off = 0;
    fx->tensors[0].ndim = 4;
    fx->tensors[0].dtype = elem_size == 1 ? 3 : 1;  /* ONNX INT8 / FLOAT */
    fx->tensors[0].quant_param_idx = 0;
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;

    fx->tensors[1].size_bytes = TEST_N * TEST_C * elem_size;
    fx->tensors[1].shape_off = 4;
    fx->tensors[1].ndim = 4;
    fx->tensors[1].dtype = fx->tensors[0].dtype;
    fx->tensors[1].quant_param_idx = 1;
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    fx->quant_params[0].scale = 0.5f;
    fx->quant_params[0].zero_point = -3;
    fx->quant_params[0].num_channels = 1;
    fx->quant_params[1].scale = 0.25f;
    fx->quant_params[1].zero_point = 7;
    fx->quant_params[1].num_channels = 1;

    fx->op.op_type = TIGRIS_OP_GLOBAL_AVG;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;

    fx->indices[0] = 0;  /* op input */
    fx->indices[1] = 1;  /* op output */
    fx->indices[2] = 0;  /* stage op */
    fx->indices[3] = 0;  /* stage input */
    fx->indices[4] = 1;  /* stage output */
    fx->indices[5] = 0;  /* model input */
    fx->indices[6] = 1;  /* model output */

    fx->stage.ops_off = 2;
    fx->stage.ops_count = 1;
    fx->stage.inputs_off = 3;
    fx->stage.inputs_count = 1;
    fx->stage.outputs_off = 4;
    fx->stage.outputs_count = 1;
    fx->stage.tile_plan_idx = 0;
    fx->stage.chain_id = TIGRIS_NO_CHAIN;

    fx->tile_plan.tileable = 1;
    fx->tile_plan.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    fx->tile_plan.tile_height = 1;
    fx->tile_plan.num_tiles = TEST_H;
    fx->tile_plan.original_height = TEST_H;

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.stages = &fx->stage;
    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.quant_params = fx->quant_params;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.model_inputs = &fx->indices[5];
    fx->plan.model_outputs = &fx->indices[6];
}

/**
 * Run the fixture with the given fast arena and copy the output out.
 * Returns the executor result; *tiled reports whether the stage was tiled.
 */
static tigris_exec_error_t run_case(
    reduce_fixture_t *fx, tigris_kernel_fn kernel,
    uint8_t *fast, uint32_t fast_size,
    const void *input, uint32_t input_bytes,
    void *output, uint32_t output_bytes, uint16_t *tiled)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t slow[SLOW_SIZE];
    uint8_t workspace[TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(2, 1, 1, 0, 0)];
    tigris_exec_stats_t stats;
    tigris_mem_t mem;
    void *ptrs[2];

    memset(&stats, 0, sizeof(stats));
    memset(slow, 0, sizeof(slow));

    if (tigris_mem_init(&mem, ptrs, 2, fast, fast_size,
                        slow, sizeof(slow)) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    if (tigris_mem_alloc_slow(&mem, 0, input_bytes) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(mem.tensor_ptrs[0], input, input_bytes);

    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &fx->plan, &mem, kernel, NULL, &stats, workspace, sizeof(workspace));
    if (err != TIGRIS_EXEC_OK)
        return err;

    memcpy(output, mem.tensor_ptrs[1], output_bytes);
    *tiled = stats.stages_tiled;
    return TIGRIS_EXEC_OK;
}

static void test_float_reduction_is_band_invariant(void)
{
    printf("  test_float_reduction_is_band_invariant...\n");

    reduce_fixture_t whole;
    reduce_fixture_t banded;
    build_fixture(&whole, sizeof(float));
    build_fixture(&banded, sizeof(float));

    float input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (float)((i % 37) - 18) * 0.125f;

    float whole_out[TEST_N * TEST_C];
    float banded_out[TEST_N * TEST_C];
    uint16_t whole_tiled = 0;
    uint16_t banded_tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    /* 1024 leaves room for a 7-row band, so the 16 rows arrive as 7+7+2. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];

    TEST_ASSERT_EQ(
        run_case(&whole, tigris_dispatch_kernel, big, sizeof(big),
                 input, sizeof(input), whole_out, sizeof(whole_out),
                 &whole_tiled),
        TIGRIS_EXEC_OK, "float whole-tensor run");
    TEST_ASSERT_EQ(whole_tiled, 0, "float whole-tensor run is not tiled");

    TEST_ASSERT_EQ(
        run_case(&banded, tigris_dispatch_kernel, small, sizeof(small),
                 input, sizeof(input), banded_out, sizeof(banded_out),
                 &banded_tiled),
        TIGRIS_EXEC_OK, "float banded run");
    TEST_ASSERT_EQ(banded_tiled, 1, "float banded run is tiled");

    TEST_ASSERT(memcmp(whole_out, banded_out, sizeof(whole_out)) == 0,
                "float banded reduction is byte-identical");
}

static void test_s8_reduction_is_band_invariant(void)
{
    printf("  test_s8_reduction_is_band_invariant...\n");

    reduce_fixture_t whole;
    reduce_fixture_t banded;
    build_fixture(&whole, 1u);
    build_fixture(&banded, 1u);

    int8_t input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (int8_t)((i * 7) % 251 - 125);

    int8_t whole_out[TEST_N * TEST_C];
    int8_t banded_out[TEST_N * TEST_C];
    uint16_t whole_tiled = 0;
    uint16_t banded_tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    /* 256 leaves room for a 6-row band, so the 16 rows arrive as 6+6+4. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[256];

    TEST_ASSERT_EQ(
        run_case(&whole, tigris_dispatch_kernel_s8, big, sizeof(big),
                 input, sizeof(input), whole_out, sizeof(whole_out),
                 &whole_tiled),
        TIGRIS_EXEC_OK, "int8 whole-tensor run");
    TEST_ASSERT_EQ(whole_tiled, 0, "int8 whole-tensor run is not tiled");

    TEST_ASSERT_EQ(
        run_case(&banded, tigris_dispatch_kernel_s8, small, sizeof(small),
                 input, sizeof(input), banded_out, sizeof(banded_out),
                 &banded_tiled),
        TIGRIS_EXEC_OK, "int8 banded run");
    TEST_ASSERT_EQ(banded_tiled, 1, "int8 banded run is tiled");

    TEST_ASSERT(memcmp(whole_out, banded_out, sizeof(whole_out)) == 0,
                "int8 banded reduction is byte-identical");
}

int main(void)
{
    printf("Global reduction tiling tests:\n");
    test_float_reduction_is_band_invariant();
    test_s8_reduction_is_band_invariant();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}

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
#include "test_harness.h"

#define TEST_N   1
#define TEST_H  16
#define TEST_W   4
#define TEST_C   8
#define TEST_ELEMENTS (TEST_N * TEST_H * TEST_W * TEST_C)

#define WHOLE_FAST_SIZE 16384u
#define SLOW_SIZE        8192u

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    tigris_stage_t stage;
    tigris_tile_plan_t tile_plan;
    tigris_tile_plan_v10_t grouped;
    tigris_quant_param_t quant_params[2];
    int32_t shapes[8];
    uint16_t indices[7];
    tigris_plan_t plan;
} reduce_fixture_t;

#include "trace_assert.h"

/** One GlobalAveragePool stage carrying a height tile plan. */
static void build_fixture(reduce_fixture_t *fx, uint32_t elem_size)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.version = TIGRIS_SCHEMA_VERSION_V9;
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

typedef struct {
    const char *name;
    uint32_t elem_size;
    tigris_kernel_fn kernel;
    uint32_t small_fast_size;
} reduce_case_t;

/* The small arenas force 7+7+2 float bands and 6+6+4 int8 bands. */
static const reduce_case_t k_reduce_cases[] = {
    { "float", sizeof(float), tigris_dispatch_kernel, 1024u },
    { "int8", 1u, tigris_dispatch_kernel_s8, 256u },
};

static void test_reduction_is_band_invariant(void)
{
    printf("  test_reduction_is_band_invariant...\n");

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];

    for (size_t c = 0; c < sizeof(k_reduce_cases) / sizeof(k_reduce_cases[0]); c++) {
        const reduce_case_t *rc = &k_reduce_cases[c];
        reduce_fixture_t whole;
        reduce_fixture_t banded;
        build_fixture(&whole, rc->elem_size);
        build_fixture(&banded, rc->elem_size);
        memset(big, 0, sizeof(big));
        memset(small, 0, sizeof(small));

        float input_f32[TEST_ELEMENTS];
        int8_t input_s8[TEST_ELEMENTS];
        for (int i = 0; i < TEST_ELEMENTS; i++) {
            input_f32[i] = (float)((i % 37) - 18) * 0.125f;
            input_s8[i] = (int8_t)((i * 7) % 251 - 125);
        }
        const void *input = rc->elem_size == 1u ? (const void *)input_s8
                                                : (const void *)input_f32;
        const uint32_t input_bytes = TEST_ELEMENTS * rc->elem_size;
        const uint32_t output_bytes = TEST_N * TEST_C * rc->elem_size;

        float whole_out[TEST_N * TEST_C];
        float banded_out[TEST_N * TEST_C];
        uint16_t whole_tiled = 0;
        uint16_t banded_tiled = 0;
        char msg[96];

        snprintf(msg, sizeof(msg), "%s whole-tensor run", rc->name);
        TEST_ASSERT_EQ(
            run_case(&whole, rc->kernel, big, sizeof(big), input, input_bytes,
                     whole_out, output_bytes, &whole_tiled),
            TIGRIS_EXEC_OK, msg);
        snprintf(msg, sizeof(msg), "%s whole-tensor run is not tiled", rc->name);
        TEST_ASSERT_EQ(whole_tiled, 0, msg);

        snprintf(msg, sizeof(msg), "%s banded run", rc->name);
        TEST_ASSERT_EQ(
            run_case(&banded, rc->kernel, small, rc->small_fast_size, input,
                     input_bytes, banded_out, output_bytes, &banded_tiled),
            TIGRIS_EXEC_OK, msg);
        snprintf(msg, sizeof(msg), "%s banded run is tiled", rc->name);
        TEST_ASSERT_EQ(banded_tiled, 1, msg);

        snprintf(msg, sizeof(msg), "%s banded reduction is byte-identical", rc->name);
        TEST_ASSERT(memcmp(whole_out, banded_out, output_bytes) == 0, msg);
    }
}

static int fail_reduction(const tigris_plan_t *plan, const tigris_op_t *op,
                          uint16_t index, tigris_mem_t *mem, void *context)
{
    (void)plan; (void)op; (void)index; (void)mem; (void)context;
    return -1;
}

static void test_channel_and_spatial_groups(void)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[128];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t input[2 * TEST_ELEMENTS * sizeof(float)];
    uint8_t expected[2 * TEST_C * sizeof(float)];
    uint8_t actual[2 * TEST_C * sizeof(float)];
    for (uint32_t element = 1u; element <= 4u; element += 3u) {
        reduce_fixture_t whole, grouped;
        build_fixture(&whole, element);
        build_fixture(&grouped, element);
        whole.shapes[0] = whole.shapes[4] = grouped.shapes[0] = grouped.shapes[4] = 2;
        for (uint8_t i = 0; i < 2u; i++) {
            whole.tensors[i].size_bytes *= 2u;
            grouped.tensors[i].size_bytes *= 2u;
        }
        if (element == 1u) {
            int8_t *values = (int8_t *)input;
            for (uint32_t i = 0; i < 2u * TEST_ELEMENTS; i++) values[i] = (int8_t)((i * 7u) % 251u - 125u);
        } else {
            float *values = (float *)input;
            for (uint32_t i = 0; i < 2u * TEST_ELEMENTS; i++) values[i] = (float)((int32_t)(i % 37u) - 18) * 0.125f;
        }
        grouped.header.version = TIGRIS_SCHEMA_VERSION_V10;
        grouped.grouped.tileable = 1u;
        grouped.grouped.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
        grouped.grouped.input_channel_group = 3u;
        grouped.grouped.input_spatial_interval = 5u;
        grouped.grouped.tiled_peak_bytes = sizeof(small);
        grouped.plan.tile_plans = NULL;
        grouped.plan.tile_plans_v10 = &grouped.grouped;
        tigris_kernel_fn kernel = element == 1u ? tigris_dispatch_kernel_s8 : tigris_dispatch_kernel;
        uint16_t tiled = 0u;
        uint32_t input_bytes = 2u * TEST_ELEMENTS * element;
        uint32_t output_bytes = 2u * TEST_C * element;
        TEST_ASSERT_EQ(run_case(&whole, kernel, big, sizeof(big), input, input_bytes,
                       expected, output_bytes, &tiled), TIGRIS_EXEC_OK, "whole batch reference");
        TEST_ASSERT_EQ(run_case(&grouped, kernel, small, sizeof(small), input, input_bytes,
                       actual, output_bytes, &tiled), TIGRIS_EXEC_OK, "partial channel and spatial groups");
        TEST_ASSERT_EQ(tiled, 1u, "grouped stage executes tiles");
        TEST_ASSERT(memcmp(actual, expected, output_bytes) == 0, "grouped batches preserve output bits");
        TEST_ASSERT_EQ(run_case(&grouped, kernel, small, 16u, input, input_bytes,
                       actual, output_bytes, &tiled), TIGRIS_EXEC_ERR_MEM, "accumulator needs arena space");
        TEST_ASSERT_EQ(run_case(&grouped, kernel, small, 63u, input, input_bytes,
                       actual, output_bytes, &tiled), TIGRIS_EXEC_ERR_MEM, "packed buffers need arena space");
        TEST_ASSERT_EQ(run_case(&grouped, fail_reduction, small, sizeof(small), input, input_bytes,
                       actual, output_bytes, &tiled), TIGRIS_EXEC_ERR_KERNEL, "group kernel failure propagates");
    }
}

int main(void)
{
    printf("Global reduction tiling tests:\n");
    test_reduction_is_band_invariant();
    test_channel_and_spatial_groups();

#ifdef TIGRIS_TRACE
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_BY_INPUT)) != 0u,
                   1, "by_input path checked against trace");
#endif

    return test_report();
}

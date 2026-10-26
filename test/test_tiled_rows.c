/**
 * @file test_tiled_rows.c
 * @brief A rank-2 matrix pipeline executed in row bands must equal one pass.
 *
 * Rank 2 has no batch or spatial axis, so the height contract never reaches
 * it, yet a matrix product against a constant reads one row to write one row.
 * This runs the same plan whole and in bands and compares.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "test_harness.h"
#include "trace_assert.h"
#include "plan_fixture.h"

#define TEST_ROWS  64
#define TEST_IN    8
#define TEST_OUT   4
#define IN_ELEMS   (TEST_ROWS * TEST_IN)
#define OUT_ELEMS  (TEST_ROWS * TEST_OUT)

#define WHOLE_FAST_SIZE 16384u
#define SLOW_SIZE        8192u

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stage;
    tigris_tile_plan_t tile_plan;
    tigris_weight_entry_t weights[2];
    float blob[TEST_IN * TEST_OUT + TEST_OUT];
    int32_t shapes[9];   /* 6 used; the tail holds a rank-3 view */
    uint16_t indices[PLAN_FIXTURE_POOL_ONE_STAGE(2)];
    tigris_plan_t plan;
} rows_fixture_t;

/** One [rows, IN] x [OUT, IN] product followed by a Relu, all rank 2. */
static void build_fixture(rows_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));
    plan_fixture_chain(&fx->plan, &fx->header, fx->tensors, fx->ops, &fx->stage,
                       fx->indices, fx->shapes, 2, 1);

    fx->header.version = TIGRIS_SCHEMA_VERSION_V9;
    fx->header.num_tile_plans = 1;
    fx->header.num_weights = 2;

    fx->shapes[0] = TEST_ROWS; fx->shapes[1] = TEST_IN;
    fx->shapes[2] = TEST_ROWS; fx->shapes[3] = TEST_OUT;
    fx->shapes[4] = TEST_ROWS; fx->shapes[5] = TEST_OUT;

    for (uint16_t i = 0; i < 3u; i++) {
        fx->tensors[i].shape_off = (uint16_t)(i * 2u);
        fx->tensors[i].ndim = 2;
        fx->tensors[i].dtype = 1;
        fx->tensors[i].quant_param_idx = 0xFFFFu;
    }
    fx->tensors[0].size_bytes = IN_ELEMS * (uint32_t)sizeof(float);
    fx->tensors[1].size_bytes = OUT_ELEMS * (uint32_t)sizeof(float);
    fx->tensors[2].size_bytes = OUT_ELEMS * (uint32_t)sizeof(float);
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    /* Weight [OUT, IN] then bias [OUT], both in the plan's own blob. */
    for (int i = 0; i < TEST_IN * TEST_OUT; i++)
        fx->blob[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < TEST_OUT; i++)
        fx->blob[TEST_IN * TEST_OUT + i] = (float)i * 0.5f - 1.0f;
    fx->weights[0].offset = 0;
    fx->weights[0].size_bytes = TEST_IN * TEST_OUT * (uint32_t)sizeof(float);
    fx->weights[1].offset = fx->weights[0].size_bytes;
    fx->weights[1].size_bytes = TEST_OUT * (uint32_t)sizeof(float);

    fx->ops[0].op_type = TIGRIS_OP_FULLY_CONN;
    fx->ops[0].weight_idx = 0; fx->ops[0].bias_idx = 1;

    fx->ops[1].op_type = TIGRIS_OP_RELU;
    fx->ops[1].weight_idx = TIGRIS_NO_WEIGHT;
    fx->ops[1].bias_idx = TIGRIS_NO_WEIGHT;

    fx->stage.tile_plan_idx = 0;

    fx->tile_plan.tileable = 1;
    fx->tile_plan.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    fx->tile_plan.tile_height = 1;
    fx->tile_plan.num_tiles = TEST_ROWS;
    fx->tile_plan.original_height = TEST_ROWS;

    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.weight_entries = fx->weights;
    fx->plan.weight_blob = (const uint8_t *)fx->blob;
}

/* A kernel that fails once a given number of calls have succeeded, so a
 * failure inside a band can be observed rather than argued about. */
typedef struct {
    int calls;
    int fail_after;
} failing_ctx_t;

static int failing_kernel(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem, void *user_ctx)
{
    failing_ctx_t *ctx = (failing_ctx_t *)user_ctx;
    if (ctx->calls++ >= ctx->fail_after)
        return -1;
    return tigris_dispatch_kernel(plan, op, op_index, mem, NULL);
}

static tigris_exec_error_t run_case_full(
    rows_fixture_t *fx, uint8_t *fast, uint32_t fast_size, uint32_t slow_size,
    tigris_kernel_fn kernel, void *user_ctx,
    const float *input, float *output, uint16_t *tiled)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t slow[SLOW_SIZE];
    uint8_t workspace[TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(3, 1, 1, 0, 0, 0)];
    return run_fixture(&fx->plan, kernel, user_ctx, fast, fast_size, slow, slow_size,
                       workspace, sizeof(workspace), input, output, tiled);
}

static tigris_exec_error_t run_case_slow(
    rows_fixture_t *fx, uint8_t *fast, uint32_t fast_size, uint32_t slow_size,
    const float *input, float *output, uint16_t *tiled)
{
    return run_case_full(fx, fast, fast_size, slow_size,
                         tigris_dispatch_kernel, NULL, input, output, tiled);
}

static tigris_exec_error_t run_case(
    rows_fixture_t *fx, uint8_t *fast, uint32_t fast_size,
    const float *input, float *output, uint16_t *tiled)
{
    return run_case_slow(fx, fast, fast_size, SLOW_SIZE, input, output, tiled);
}

static void test_rows_are_band_invariant(void)
{
    printf("  test_rows_are_band_invariant...\n");

    rows_fixture_t whole;
    rows_fixture_t banded;
    build_fixture(&whole);
    build_fixture(&banded);

    float input[IN_ELEMS];
    for (int i = 0; i < IN_ELEMS; i++)
        input[i] = (float)((i % 13) - 6) * 0.125f;

    float whole_out[OUT_ELEMS];
    float banded_out[OUT_ELEMS];
    uint16_t whole_tiled = 0;
    uint16_t banded_tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    /* A row costs 8 input floats and two lots of 4 output floats, so 512
     * bytes leaves room for a band far short of all 64 rows. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[512];

    TEST_ASSERT_EQ(
        run_case(&whole, big, sizeof(big), input, whole_out, &whole_tiled),
        TIGRIS_EXEC_OK, "whole-matrix run");
    TEST_ASSERT_EQ(whole_tiled, 0, "whole-matrix run is not tiled");

    TEST_ASSERT_EQ(
        run_case(&banded, small, sizeof(small), input, banded_out,
                 &banded_tiled),
        TIGRIS_EXEC_OK, "row-banded run");
    TEST_ASSERT_EQ(banded_tiled, 1, "row-banded run is tiled");

    TEST_ASSERT(memcmp(whole_out, banded_out, sizeof(whole_out)) == 0,
                "row-banded result is byte-identical");

    /* Relu leaves nothing negative, and a row that is all zero stays zero:
     * enough to show the band computed something rather than nothing. */
    int nonzero = 0;
    for (int i = 0; i < OUT_ELEMS; i++) {
        TEST_ASSERT(banded_out[i] >= 0.0f, "Relu clamps the banded result");
        if (banded_out[i] > 0.0f)
            nonzero++;
    }
    TEST_ASSERT(nonzero > 0, "the banded result is not uniformly zero");
}

/**
 * A stage whose tensors disagree on their row count is not a row band: a band
 * would mean a different thing on each of them. It must run untiled instead
 * of taking the path, which is what this asserts, with an arena roomy enough
 * that not tiling is not itself a failure.
 */
static void test_mismatched_rows_are_not_banded(void)
{
    printf("  test_mismatched_rows_are_not_banded...\n");

    rows_fixture_t fx;
    build_fixture(&fx);
    fx.shapes[4] = TEST_ROWS / 2;   /* the final output loses half its rows */
    fx.tensors[2].size_bytes =
        (TEST_ROWS / 2) * TEST_OUT * (uint32_t)sizeof(float);

    float input[IN_ELEMS];
    for (int i = 0; i < IN_ELEMS; i++)
        input[i] = 1.0f;
    float out[OUT_ELEMS];
    uint16_t tiled = 1;

    /* An arena too small for the whole stage, so the band is genuinely
     * considered and genuinely declined: the stage falls back to running
     * whole, which then does not fit. Not tiling is the assertion; the memory
     * error is how a declined band surfaces. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[512];
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_MEM, "mismatched rows are not banded");

    /* Same for a stage input that presents no row view at all. */
    rows_fixture_t ranked;
    build_fixture(&ranked);
    ranked.tensors[0].ndim = 4;
    ranked.shapes[0] = 1;
    TEST_ASSERT_EQ(run_case(&ranked, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_MEM, "a rank-4 input is not banded");

    /* And for an operator that does not keep rows independent. */
    rows_fixture_t mixed;
    build_fixture(&mixed);
    mixed.ops[1].op_type = TIGRIS_OP_CONCAT;
    TEST_ASSERT_EQ(run_case(&mixed, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_MEM,
                   "an operator that mixes rows is not banded");

    /* And for a rank-3 input that is not in the model's own order, where a
     * row band would mean something the stored bytes do not. */
    rows_fixture_t spatial;
    build_fixture(&spatial);
    spatial.tensors[0].ndim = 3;
    spatial.shapes[0] = 1;
    spatial.shapes[1] = TEST_ROWS;
    TEST_ASSERT_EQ(run_case(&spatial, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_MEM, "a spatial rank-3 input is not banded");
}

/**
 * A rank-3 linear [1, rows, cols] is the same memory as the rank-2 matrix and
 * passes the row view, but kern_fully_connected reads its channel count out
 * of y_shape[1], which on a rank 3 is the row count. The row view alone does
 * not settle the matrix product's rank, so the stage contract has to. The
 * loader refuses such a plan on its own terms; this is the executor's half of
 * the same contract, which the file's comments say re-checks what the loader
 * promised.
 */
static void test_a_rank3_matrix_product_output_is_not_banded(void)
{
    printf("  test_a_rank3_matrix_product_output_is_not_banded...\n");

    rows_fixture_t fx;
    build_fixture(&fx);
    fx.shapes[6] = 1;
    fx.shapes[7] = TEST_ROWS;
    fx.shapes[8] = TEST_OUT;
    fx.tensors[1].ndim = 3;
    fx.tensors[1].shape_off = 6;
    fx.tensors[1].flags |= TIGRIS_TENSOR_LINEAR;

    float input[IN_ELEMS];
    for (int i = 0; i < IN_ELEMS; i++)
        input[i] = 1.0f;
    float out[OUT_ELEMS];
    uint16_t tiled = 1;

    /* An arena too small for the whole stage, so the band is genuinely
     * considered. Declining it leaves the stage to run whole, which then does
     * not fit: the memory error is how the refusal surfaces. Admitting it
     * instead returns success with a matrix product computed against the row
     * count as its channel count. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[512];
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_MEM,
                   "a rank-3 matrix product output is not banded");
}

/**
 * The two ways a row band can fail to start. Neither may produce a partial
 * result: a band that cannot be sized and an assembled output that has
 * nowhere to live both have to say so.
 */
static void test_row_band_failures_are_reported(void)
{
    printf("  test_row_band_failures_are_reported...\n");

    float input[IN_ELEMS];
    for (int i = 0; i < IN_ELEMS; i++)
        input[i] = 1.0f;
    float out[OUT_ELEMS];
    uint16_t tiled = 0;

    /* One row costs an aligned allocation on each of the three tensors, so an
     * arena below that cannot hold even a single-row band. */
    rows_fixture_t narrow;
    build_fixture(&narrow);
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t tiny[64];
    TEST_ASSERT_EQ(
        run_case(&narrow, tiny, sizeof(tiny), input, out, &tiled),
        TIGRIS_EXEC_ERR_TILE, "a band of one that does not fit is reported");

    /* The band assembles its output in slow. 2560 bytes holds the 2048-byte
     * input and not the 1024-byte output, so the failure lands on the output
     * allocation inside the band path rather than before it. */
    rows_fixture_t shallow;
    build_fixture(&shallow);
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[512];
    TEST_ASSERT_EQ(
        run_case_slow(&shallow, small, sizeof(small), 2560u, input, out,
                      &tiled),
        TIGRIS_EXEC_ERR_MEM, "an output with nowhere to live is reported");

    /* A kernel that fails partway through a band stops the whole run rather
     * than spilling whatever the band happened to hold. */
    rows_fixture_t interrupted;
    build_fixture(&interrupted);
    failing_ctx_t ctx = {0, 3};
    TEST_ASSERT_EQ(
        run_case_full(&interrupted, small, sizeof(small), SLOW_SIZE,
                      failing_kernel, &ctx, input, out, &tiled),
        TIGRIS_EXEC_ERR_KERNEL, "a kernel failure inside a band is reported");
    TEST_ASSERT_EQ(ctx.calls, 4, "the run stops at the failing call");
}

/* A second stage whose single input dies there and matches its output row for
 * row: the band gathers that input before any op runs and scatters the result
 * back to the same rows, so one slow buffer can carry both tensors. */
#define ALIAS_IN   2
#define ALIAS_OUT  4
#define ALIAS_IN_BYTES  (TEST_ROWS * ALIAS_IN * (uint32_t)sizeof(float))
#define ALIAS_OUT_BYTES (TEST_ROWS * ALIAS_OUT * (uint32_t)sizeof(float))

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stages[2];
    tigris_tile_plan_t tile_plan;
    tigris_weight_entry_t weights[2];
    float blob[ALIAS_IN * ALIAS_OUT + ALIAS_OUT];
    int32_t shapes[6];
    uint16_t indices[PLAN_FIXTURE_POOL(2)];
    tigris_plan_t plan;
} alias_fixture_t;

static void build_alias_fixture(alias_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));
    plan_fixture_chain(&fx->plan, &fx->header, fx->tensors, fx->ops, fx->stages,
                       fx->indices, fx->shapes, 2, 0);

    fx->header.version = TIGRIS_SCHEMA_VERSION_V9;
    fx->header.num_tile_plans = 1;
    fx->header.num_weights = 2;

    fx->shapes[0] = TEST_ROWS; fx->shapes[1] = ALIAS_IN;
    fx->shapes[2] = TEST_ROWS; fx->shapes[3] = ALIAS_OUT;
    fx->shapes[4] = TEST_ROWS; fx->shapes[5] = ALIAS_OUT;

    for (uint16_t i = 0; i < 3u; i++) {
        fx->tensors[i].shape_off = (uint16_t)(i * 2u);
        fx->tensors[i].ndim = 2;
        fx->tensors[i].dtype = 1;
        fx->tensors[i].quant_param_idx = 0xFFFFu;
    }
    fx->tensors[0].size_bytes = ALIAS_IN_BYTES;
    fx->tensors[1].size_bytes = ALIAS_OUT_BYTES;
    fx->tensors[2].size_bytes = ALIAS_OUT_BYTES;
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    for (int i = 0; i < ALIAS_IN * ALIAS_OUT; i++)
        fx->blob[i] = (float)((i % 5) - 2) * 0.5f;
    for (int i = 0; i < ALIAS_OUT; i++)
        fx->blob[ALIAS_IN * ALIAS_OUT + i] = (float)i * 0.25f - 0.5f;
    fx->weights[0].offset = 0;
    fx->weights[0].size_bytes =
        ALIAS_IN * ALIAS_OUT * (uint32_t)sizeof(float);
    fx->weights[1].offset = fx->weights[0].size_bytes;
    fx->weights[1].size_bytes = ALIAS_OUT * (uint32_t)sizeof(float);

    fx->ops[0].op_type = TIGRIS_OP_FULLY_CONN;
    fx->ops[0].weight_idx = 0; fx->ops[0].bias_idx = 1;

    fx->ops[1].op_type = TIGRIS_OP_RELU;
    fx->ops[1].weight_idx = TIGRIS_NO_WEIGHT;
    fx->ops[1].bias_idx = TIGRIS_NO_WEIGHT;

    fx->stages[0].tile_plan_idx = 0;
    fx->stages[1].tile_plan_idx = 0;

    fx->tile_plan.tileable = 1;
    fx->tile_plan.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    fx->tile_plan.tile_height = 1;
    fx->tile_plan.num_tiles = TEST_ROWS;
    fx->tile_plan.original_height = TEST_ROWS;

    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.weight_entries = fx->weights;
    fx->plan.weight_blob = (const uint8_t *)fx->blob;
}

static tigris_exec_error_t run_alias_case(
    alias_fixture_t *fx, uint32_t slow_size, const float *input, float *output)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t fast[512];
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t slow[SLOW_SIZE];
    uint8_t workspace[TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(3, 1, 1, 0, 0, 0)];
    tigris_mem_t mem;
    void *ptrs[3];

    memset(slow, 0, sizeof(slow));
    if (tigris_mem_init(&mem, ptrs, 3, fast, sizeof(fast),
                        slow, slow_size) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    if (tigris_mem_alloc_slow(&mem, 0, fx->tensors[0].size_bytes)
            != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(mem.tensor_ptrs[0], input, fx->tensors[0].size_bytes);

    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &fx->plan, &mem, tigris_dispatch_kernel, NULL, NULL,
        workspace, sizeof(workspace));
    if (err != TIGRIS_EXEC_OK)
        return err;
    memcpy(output, mem.tensor_ptrs[2], fx->tensors[2].size_bytes);
    return TIGRIS_EXEC_OK;
}

/**
 * The shared buffer is what makes this fit: 1,792 bytes of slow memory hold
 * the model input beside one 1,024-byte buffer, and not two of them.
 */
static void test_a_band_writes_over_an_input_that_dies(void)
{
    printf("  test_a_band_writes_over_an_input_that_dies...\n");

    alias_fixture_t tight;
    alias_fixture_t roomy;
    build_alias_fixture(&tight);
    build_alias_fixture(&roomy);

    float input[TEST_ROWS * ALIAS_IN];
    for (int i = 0; i < TEST_ROWS * ALIAS_IN; i++)
        input[i] = (float)((i % 11) - 5) * 0.25f;

    float tight_out[TEST_ROWS * ALIAS_OUT];
    float roomy_out[TEST_ROWS * ALIAS_OUT];

    TEST_ASSERT_EQ(run_alias_case(&roomy, SLOW_SIZE, input, roomy_out),
                   TIGRIS_EXEC_OK, "the roomy run completes");
    TEST_ASSERT_EQ(run_alias_case(&tight, 1792u, input, tight_out),
                   TIGRIS_EXEC_OK,
                   "the run fits a pool too small to hold both tensors");
    TEST_ASSERT(memcmp(tight_out, roomy_out, sizeof(roomy_out)) == 0,
                "writing over the input leaves the same result");

    int nonzero = 0;
    for (int i = 0; i < TEST_ROWS * ALIAS_OUT; i++) {
        TEST_ASSERT(tight_out[i] >= 0.0f, "Relu clamps the result");
        if (tight_out[i] > 0.0f)
            nonzero++;
    }
    TEST_ASSERT(nonzero > 0, "the result is not uniformly zero");
}

/**
 * A model input is the caller's buffer and may still hold values the caller
 * wants after the run, so a band never writes over one. Here the stage that
 * would alias reads the model input directly, and the run needs a pool big
 * enough for both tensors.
 */
static void test_a_band_never_writes_over_a_model_input(void)
{
    printf("  test_a_band_never_writes_over_a_model_input...\n");

    alias_fixture_t fx;
    build_alias_fixture(&fx);
    /* Drop the first stage: the banded stage now reads the model input. */
    fx.header.num_stages = 1;
    fx.stages[0] = fx.stages[1];
    fx.tensors[0].size_bytes = ALIAS_OUT_BYTES;
    fx.shapes[1] = ALIAS_OUT;
    fx.indices[2] = 0;   /* the Relu now reads the model input */
    fx.indices[8] = 0;   /* and so does the stage */

    float input[TEST_ROWS * ALIAS_OUT];
    for (int i = 0; i < TEST_ROWS * ALIAS_OUT; i++)
        input[i] = (float)((i % 11) - 5) * 0.25f;
    float out[TEST_ROWS * ALIAS_OUT];

    TEST_ASSERT_EQ(run_alias_case(&fx, 1792u, input, out),
                   TIGRIS_EXEC_ERR_MEM,
                   "a pool sized for one buffer is not enough");
    TEST_ASSERT_EQ(run_alias_case(&fx, SLOW_SIZE, input, out),
                   TIGRIS_EXEC_OK, "both tensors fit a roomy pool");
    for (int i = 0; i < TEST_ROWS * ALIAS_OUT; i++) {
        float want = input[i] > 0.0f ? input[i] : 0.0f;
        if (out[i] != want) {
            TEST_ASSERT(0, "the model input is read, not overwritten");
            break;
        }
    }
    TEST_ASSERT(1, "the model input survives the run");
}

/* One rank-2 binary operator against a constant held as its weight: dense
 * [rows, cols], one value per row [rows, 1], or one row [cols]. A selection
 * takes a second input and holds its bool condition as the constant. */
#define CONST_COLS 8
#define CONST_ELEMS (TEST_ROWS * CONST_COLS)

typedef enum { CONST_DENSE, CONST_PER_ROW, CONST_ONE_ROW } const_form_t;

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t op;
    tigris_stage_t stage;
    tigris_tile_plan_t tile_plan;
    tigris_weight_entry_t weight;
    tigris_op_attribute_t attribute;
    uint8_t attribute_data[12];
    float blob[CONST_ELEMS];
    uint8_t condition[CONST_ELEMS];
    int32_t shapes[6];
    uint16_t indices[4];
    tigris_plan_t plan;
} const_fixture_t;

static void build_const_fixture(const_fixture_t *fx, uint8_t op_type, const_form_t form)
{
    const int select = op_type == TIGRIS_OP_SELECT_V2;
    const uint16_t inputs = select ? 2u : 1u;
    memset(fx, 0, sizeof(*fx));
    fx->header.version = TIGRIS_SCHEMA_VERSION_V9;
    fx->header.num_tensors = (uint16_t)(inputs + 1u);
    fx->header.num_ops = 1;
    fx->header.num_stages = 1;
    fx->header.num_tile_plans = 1;
    fx->header.num_weights = 1;
    fx->header.num_model_inputs = inputs;
    fx->header.num_model_outputs = 1;

    for (uint16_t i = 0; i < 3u; i++) {
        fx->shapes[2u * i] = TEST_ROWS;
        fx->shapes[2u * i + 1u] = CONST_COLS;
        fx->tensors[i].shape_off = (uint16_t)(2u * i);
        fx->tensors[i].ndim = 2;
        fx->tensors[i].dtype = 1;
        fx->tensors[i].quant_param_idx = 0xFFFFu;
        fx->tensors[i].size_bytes = CONST_ELEMS * (uint32_t)sizeof(float);
        fx->tensors[i].flags = TIGRIS_TENSOR_MODEL_INPUT;
    }
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    if (op_type == TIGRIS_OP_LESS) {
        fx->tensors[1].dtype = 9;
        fx->tensors[1].size_bytes = CONST_ELEMS;
    }

    uint32_t count = form == CONST_DENSE ? CONST_ELEMS : form == CONST_PER_ROW ? TEST_ROWS : CONST_COLS;
    for (uint32_t i = 0; i < count; i++) {
        fx->blob[i] = (float)((int)(i % 9u) - 4) * 0.5f;
        fx->condition[i] = (uint8_t)(i % 3u == 0u);
    }
    fx->weight.size_bytes = count * (select ? 1u : (uint32_t)sizeof(float));

    fx->op.op_type = op_type;
    fx->op.num_inputs = (uint8_t)inputs; fx->op.num_outputs = 1;
    fx->op.inputs_off = 0; fx->op.outputs_off = 2;
    fx->op.weight_idx = 0;
    fx->op.bias_idx = TIGRIS_NO_WEIGHT;

    /* A float Add reads its constant without the attribute; the others and
     * the per-row shape need it, the latter in its long form. */
    if (op_type != TIGRIS_OP_ADD || form == CONST_PER_ROW) {
        const int32_t shape[2] = {TEST_ROWS, 1};
        fx->attribute_data[0] = select ? 0u : 1u;
        fx->attribute_data[2] = 0xFFu;
        fx->attribute_data[3] = 0xFFu;
        fx->attribute.type = TIGRIS_OP_ATTR_CONSTANT_OPERAND;
        fx->attribute.data_len = TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN;
        if (form == CONST_PER_ROW) {
            memcpy(fx->attribute_data + TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN, shape, sizeof(shape));
            fx->attribute.data_len += (uint8_t)sizeof(shape);
        }
        fx->plan.num_op_attributes = 1;
        fx->plan.op_attributes = &fx->attribute;
        fx->plan.op_attribute_data = fx->attribute_data;
    }

    fx->indices[0] = 0;  /* op inputs, stage inputs, model inputs */
    fx->indices[1] = 2;
    fx->indices[2] = 1;  /* op output, stage output, model output */
    fx->indices[3] = 0;  /* stage op */
    fx->stage.ops_off = 3;
    fx->stage.ops_count = 1;
    fx->stage.inputs_off = 0;
    fx->stage.inputs_count = inputs;
    fx->stage.outputs_off = 2;
    fx->stage.outputs_count = 1;
    fx->stage.tile_plan_idx = 0;
    fx->stage.chain_id = TIGRIS_NO_CHAIN;

    fx->tile_plan.tileable = 1;
    fx->tile_plan.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    fx->tile_plan.tile_height = 1;
    fx->tile_plan.num_tiles = TEST_ROWS;
    fx->tile_plan.original_height = TEST_ROWS;

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.stages = &fx->stage;
    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.weight_entries = &fx->weight;
    fx->plan.weight_blob = select ? fx->condition : (const uint8_t *)fx->blob;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.model_inputs = &fx->indices[0];
    fx->plan.model_outputs = &fx->indices[2];
}

static tigris_exec_error_t run_const_case(const_fixture_t *fx, const float *input, const float *second,
                                          uint8_t *output)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t fast[512];
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t slow[SLOW_SIZE];
    uint8_t workspace[TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(3, 2, 1, 0, 0, 0)];
    tigris_exec_stats_t stats;
    tigris_mem_t mem;
    void *ptrs[3];

    memset(&stats, 0, sizeof(stats));
    memset(slow, 0, sizeof(slow));
    if (tigris_mem_init(&mem, ptrs, fx->header.num_tensors, fast, sizeof(fast), slow, sizeof(slow)) != TIGRIS_MEM_OK ||
        tigris_mem_alloc_slow(&mem, 0, fx->tensors[0].size_bytes) != TIGRIS_MEM_OK ||
        (fx->op.num_inputs == 2u &&
         tigris_mem_alloc_slow(&mem, 2, fx->tensors[2].size_bytes) != TIGRIS_MEM_OK))
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(mem.tensor_ptrs[0], input, fx->tensors[0].size_bytes);
    if (fx->op.num_inputs == 2u)
        memcpy(mem.tensor_ptrs[2], second, fx->tensors[2].size_bytes);
    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &fx->plan, &mem, tigris_dispatch_kernel, NULL, &stats, workspace, sizeof(workspace));
    if (err != TIGRIS_EXEC_OK)
        return err;
    if (stats.stages_tiled != 1u)
        return TIGRIS_EXEC_ERR_TILE;
    memcpy(output, mem.tensor_ptrs[1], fx->tensors[1].size_bytes);
    return TIGRIS_EXEC_OK;
}

/**
 * A row band hands a constant operand to the kernel whole, so only a constant
 * that every row reads alike may ride it. A dense or per-row constant takes
 * the leading band, which offsets it to the band's rows; one row of values
 * stays a row band. A dense constant is one element of its own type per
 * output element, whatever the output's type: a comparison writes bool, a
 * selection's condition is bool. Each runs banded and matches the operator
 * computed here.
 */
static void test_a_constant_that_varies_by_row_takes_the_leading_band(void)
{
    printf("  test_a_constant_that_varies_by_row_takes_the_leading_band...\n");
    static const struct {
        uint8_t op_type;
        const_form_t form;
        tigris_band_kind_t kind;
        const char *name;
    } cases[] = {
        {TIGRIS_OP_ADD, CONST_DENSE, TIGRIS_BAND_KIND_LEADING, "Add, dense constant"},
        {TIGRIS_OP_MAXIMUM, CONST_DENSE, TIGRIS_BAND_KIND_LEADING, "Max, dense constant"},
        {TIGRIS_OP_LESS, CONST_DENSE, TIGRIS_BAND_KIND_LEADING, "Less, dense constant"},
        {TIGRIS_OP_SELECT_V2, CONST_DENSE, TIGRIS_BAND_KIND_LEADING, "Select, dense condition"},
        {TIGRIS_OP_ADD, CONST_PER_ROW, TIGRIS_BAND_KIND_LEADING, "Add, one value per row"},
        {TIGRIS_OP_ADD, CONST_ONE_ROW, TIGRIS_BAND_KIND_ROWS, "Add, one row of values"},
        {TIGRIS_OP_MAXIMUM, CONST_ONE_ROW, TIGRIS_BAND_KIND_ROWS, "Max, one row of values"},
    };
    float input[CONST_ELEMS];
    float second[CONST_ELEMS];
    for (int i = 0; i < CONST_ELEMS; i++) {
        input[i] = (float)((i % 13) - 6) * 0.25f;
        second[i] = (float)((i % 7) - 3) * 0.75f;
    }
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const_fixture_t fx;
        build_const_fixture(&fx, cases[c].op_type, cases[c].form);
        TEST_ASSERT_EQ(tigris_stage_band_kind(&fx.plan, &fx.stage, NULL), cases[c].kind, cases[c].name);
        float out[CONST_ELEMS];
        uint8_t *bytes = (uint8_t *)out;
        TEST_ASSERT_EQ(run_const_case(&fx, input, second, bytes), TIGRIS_EXEC_OK, cases[c].name);
        int same = 1;
        for (int i = 0; i < CONST_ELEMS; i++) {
            int row = i / CONST_COLS, col = i % CONST_COLS;
            int at = cases[c].form == CONST_DENSE ? i : cases[c].form == CONST_PER_ROW ? row : col;
            float k = fx.blob[at];
            if (cases[c].op_type == TIGRIS_OP_LESS) {
                if (bytes[i] != (uint8_t)(input[i] < k))
                    same = 0;
                continue;
            }
            float want = cases[c].op_type == TIGRIS_OP_ADD ? input[i] + k :
                cases[c].op_type == TIGRIS_OP_SELECT_V2 ? (fx.condition[at] ? input[i] : second[i]) :
                (input[i] > k ? input[i] : k);
            if (out[i] != want)
                same = 0;
        }
        TEST_ASSERT(same, cases[c].name);
    }
}

int main(void)
{
    printf("Row-banded matrix tiling tests:\n");
    test_rows_are_band_invariant();
    test_mismatched_rows_are_not_banded();
    test_a_rank3_matrix_product_output_is_not_banded();
    test_row_band_failures_are_reported();
    test_a_band_writes_over_an_input_that_dies();
    test_a_band_never_writes_over_a_model_input();
    test_a_constant_that_varies_by_row_takes_the_leading_band();

#ifdef TIGRIS_TRACE
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_ROWS)) != 0u,
                   1, "rows path checked against trace");
#endif

    return test_report();
}

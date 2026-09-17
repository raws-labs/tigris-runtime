/**
 * @file test_tiled_transpose.c
 * @brief A layout conversion executed in bands must equal the single pass.
 *
 * The executor tiles a standalone rank-3 last-two-axis Transpose along its
 * output rows, gathering the input columns each band transposes. This compares
 * the banded result against the same plan run whole and checks that the banded
 * run really was tiled.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_mem.h"

#define TEST_N     1
#define TEST_LONG 64
#define TEST_SHORT 6
#define TEST_ELEMENTS (TEST_N * TEST_LONG * TEST_SHORT)

#define WHOLE_FAST_SIZE 8192u
#define SLOW_SIZE       8192u

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
    tigris_op_attribute_t attribute;
    uint8_t perm[4];
    int32_t shapes[8];
    uint16_t indices[7];
    tigris_plan_t plan;
} transpose_fixture_t;

/** One Transpose stage permuting [N, rows, cols] to [N, cols, rows]. */
static void build_fixture(transpose_fixture_t *fx, int32_t rows, int32_t cols)
{
    memset(fx, 0, sizeof(*fx));

    fx->header.version = TIGRIS_SCHEMA_VERSION;
    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->header.num_stages = 1;
    fx->header.num_tile_plans = 1;
    fx->header.num_model_inputs = 1;
    fx->header.num_model_outputs = 1;

    fx->shapes[0] = TEST_N; fx->shapes[1] = rows; fx->shapes[2] = cols;
    fx->shapes[3] = TEST_N; fx->shapes[4] = cols;  fx->shapes[5] = rows;

    for (uint16_t i = 0; i < 2; i++) {
        fx->tensors[i].size_bytes = TEST_ELEMENTS * (uint32_t)sizeof(float);
        fx->tensors[i].shape_off = (uint16_t)(i * 3);
        fx->tensors[i].ndim = 3;
        fx->tensors[i].dtype = 1; /* ONNX FLOAT */
        fx->tensors[i].quant_param_idx = 0xFFFFu;
    }
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    fx->op.op_type = TIGRIS_OP_TRANSPOSE;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;

    fx->perm[0] = 0; fx->perm[1] = 2; fx->perm[2] = 1;
    fx->attribute.op_index = 0;
    fx->attribute.type = TIGRIS_OP_ATTR_TRANSPOSE_PERM;
    fx->attribute.data_offset = 0;
    fx->attribute.data_len = 3;

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
    fx->tile_plan.num_tiles = (uint16_t)(rows > cols ? rows : cols);
    fx->tile_plan.original_height = (uint16_t)(rows > cols ? rows : cols);

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.stages = &fx->stage;
    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.op_attributes = &fx->attribute;
    fx->plan.op_attribute_data = fx->perm;
    fx->plan.num_op_attributes = 1;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.model_inputs = &fx->indices[5];
    fx->plan.model_outputs = &fx->indices[6];
}

static tigris_exec_error_t run_case(
    transpose_fixture_t *fx, uint8_t *fast, uint32_t fast_size,
    const float *input, float *output, uint16_t *tiled)
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
    if (tigris_mem_alloc_slow(&mem, 0, fx->tensors[0].size_bytes)
            != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(mem.tensor_ptrs[0], input, fx->tensors[0].size_bytes);

    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &fx->plan, &mem, tigris_dispatch_kernel, NULL, &stats,
        workspace, sizeof(workspace));
    if (err != TIGRIS_EXEC_OK)
        return err;

    memcpy(output, mem.tensor_ptrs[1], fx->tensors[1].size_bytes);
    *tiled = stats.stages_tiled;
    return TIGRIS_EXEC_OK;
}

/**
 * Run the same conversion whole and in bands and compare. `rows` and `cols`
 * are the input's two permuted extents; swapping them swaps which side of the
 * conversion takes the strided transfer.
 */
static void check_band_invariance(int32_t rows, int32_t cols, const char *what)
{
    printf("  check_band_invariance %s...\n", what);

    transpose_fixture_t whole;
    transpose_fixture_t banded;
    build_fixture(&whole, rows, cols);
    build_fixture(&banded, rows, cols);

    float input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (float)i * 0.5f - 96.0f;

    float whole_out[TEST_ELEMENTS];
    float banded_out[TEST_ELEMENTS];
    uint16_t whole_tiled = 0;
    uint16_t banded_tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    /* A slice holds TEST_SHORT x band floats on each side, so 1024 bytes
     * leaves room for a band well short of the full extent. */
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];

    TEST_ASSERT_EQ(
        run_case(&whole, big, sizeof(big), input, whole_out, &whole_tiled),
        TIGRIS_EXEC_OK, "whole-tensor conversion");
    TEST_ASSERT_EQ(whole_tiled, 0, "whole-tensor run is not tiled");

    TEST_ASSERT_EQ(
        run_case(&banded, small, sizeof(small), input, banded_out,
                 &banded_tiled),
        TIGRIS_EXEC_OK, "banded conversion");
    TEST_ASSERT_EQ(banded_tiled, 1, "banded run is tiled");

    TEST_ASSERT(memcmp(whole_out, banded_out, sizeof(whole_out)) == 0,
                "banded conversion is byte-identical");

    /* Independent check against the definition, so both runs agreeing on a
     * wrong answer cannot pass. */
    int mismatches = 0;
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t c = 0; c < cols; c++) {
            if (banded_out[c * rows + r] != input[r * cols + c])
                mismatches++;
        }
    }
    TEST_ASSERT_EQ(mismatches, 0, "banded conversion matches the permutation");
}

/**
 * Re-shape a rank-3 fixture as the rank-4 conversion that carries the same
 * matrix. `spatial_first` selects [N, H, W, C] -> [N, C, H, W] (perm 0,3,1,2),
 * the direction that goes to linear order; otherwise the way back.
 */
static void make_rank4(transpose_fixture_t *fx, int32_t h, int32_t w,
                       int32_t c, int spatial_first)
{
    fx->tensors[0].ndim = 4;
    fx->tensors[1].ndim = 4;
    fx->tensors[0].shape_off = 0;
    fx->tensors[1].shape_off = 4;
    fx->attribute.data_len = 4;

    if (spatial_first) {
        fx->shapes[0] = TEST_N; fx->shapes[1] = h;
        fx->shapes[2] = w;      fx->shapes[3] = c;
        fx->shapes[4] = TEST_N; fx->shapes[5] = c;
        fx->shapes[6] = h;      fx->shapes[7] = w;
        fx->perm[0] = 0; fx->perm[1] = 3; fx->perm[2] = 1; fx->perm[3] = 2;
    } else {
        fx->shapes[0] = TEST_N; fx->shapes[1] = c;
        fx->shapes[2] = h;      fx->shapes[3] = w;
        fx->shapes[4] = TEST_N; fx->shapes[5] = h;
        fx->shapes[6] = w;      fx->shapes[7] = c;
        fx->perm[0] = 0; fx->perm[1] = 2; fx->perm[2] = 3; fx->perm[3] = 1;
    }
}

/**
 * A rank-4 conversion leaves the two spatial axes in their relative order, so
 * it carries the same matrix as a rank-3 one with H*W read as a single
 * extent. Both directions must band identically to the single pass.
 */
static void check_rank4(int32_t h, int32_t w, int32_t c, int spatial_first,
                        const char *what)
{
    printf("  check_rank4 %s...\n", what);

    transpose_fixture_t whole;
    transpose_fixture_t banded;
    build_fixture(&whole, TEST_SHORT, TEST_LONG);
    build_fixture(&banded, TEST_SHORT, TEST_LONG);
    make_rank4(&whole, h, w, c, spatial_first);
    make_rank4(&banded, h, w, c, spatial_first);

    float input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (float)i * 0.25f - 48.0f;

    float whole_out[TEST_ELEMENTS];
    float banded_out[TEST_ELEMENTS];
    uint16_t whole_tiled = 0;
    uint16_t banded_tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t big[WHOLE_FAST_SIZE];
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];

    TEST_ASSERT_EQ(
        run_case(&whole, big, sizeof(big), input, whole_out, &whole_tiled),
        TIGRIS_EXEC_OK, "rank-4 whole-tensor conversion");
    TEST_ASSERT_EQ(whole_tiled, 0, "rank-4 whole-tensor run is not tiled");

    TEST_ASSERT_EQ(
        run_case(&banded, small, sizeof(small), input, banded_out,
                 &banded_tiled),
        TIGRIS_EXEC_OK, "rank-4 banded conversion");
    TEST_ASSERT_EQ(banded_tiled, 1, "rank-4 banded run is tiled");

    TEST_ASSERT(memcmp(whole_out, banded_out, sizeof(whole_out)) == 0,
                "rank-4 banded conversion is byte-identical");

    /* Against the definition: the collapsed matrix is rows x cols, where the
     * spatial pair is the row axis going to linear order and the column axis
     * coming back. */
    int32_t rows = spatial_first ? h * w : c;
    int32_t cols = spatial_first ? c : h * w;
    int mismatches = 0;
    for (int32_t r = 0; r < rows; r++) {
        for (int32_t k = 0; k < cols; k++) {
            if (banded_out[k * rows + r] != input[r * cols + k])
                mismatches++;
        }
    }
    TEST_ASSERT_EQ(mismatches, 0, "rank-4 conversion matches the permutation");
}

/** A permutation that is not a layout conversion must not take the path. */
static void check_unconvertible_perm_is_refused(void)
{
    printf("  check_unconvertible_perm_is_refused...\n");

    transpose_fixture_t fx;
    build_fixture(&fx, TEST_SHORT, TEST_LONG);
    make_rank4(&fx, 4, 4, 24, 1);
    /* Swap the two spatial axes as well: no longer a conversion. */
    fx.perm[0] = 0; fx.perm[1] = 3; fx.perm[2] = 2; fx.perm[3] = 1;
    fx.shapes[5] = 24; fx.shapes[6] = 4; fx.shapes[7] = 4;

    float input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (float)i;
    float out[TEST_ELEMENTS];
    uint16_t tiled = 0;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];
    /* The plan still promises a height tile, and the runtime cannot honor it
     * for this permutation, so it fails closed rather than guessing. */
    TEST_ASSERT_EQ(
        run_case(&fx, small, sizeof(small), input, out, &tiled),
        TIGRIS_EXEC_ERR_TILE, "unconvertible permutation fails closed");
    TEST_ASSERT_EQ(tiled, 0, "unconvertible permutation is not tiled");
}

int main(void)
{
    printf("Layout conversion tiling tests:\n");
    check_band_invariance(TEST_SHORT, TEST_LONG, "gathering input columns");
    check_band_invariance(TEST_LONG, TEST_SHORT, "scattering output columns");
    check_rank4(4, 4, 24, 1, "rank 4 to linear order");
    check_rank4(4, 4, 24, 0, "rank 4 back to spatial order");
    check_unconvertible_perm_is_refused();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed == 0 ? 0 : 1;
}

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
#include "test_harness.h"
#include "trace_assert.h"
#include "plan_fixture.h"

#define TEST_N     1
#define TEST_LONG 64
#define TEST_SHORT 6
#define TEST_ELEMENTS (TEST_N * TEST_LONG * TEST_SHORT)

#define WHOLE_FAST_SIZE 8192u
#define SLOW_SIZE       8192u

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    tigris_stage_t stage;
    tigris_tile_plan_t tile_plan;
    tigris_op_attribute_t attribute;
    uint8_t perm[4];
    int32_t shapes[8];
    uint16_t indices[PLAN_FIXTURE_POOL(1)];
    tigris_plan_t plan;
} transpose_fixture_t;

/** One Transpose stage permuting [N, rows, cols] to [N, cols, rows]. */
static void build_fixture(transpose_fixture_t *fx, int32_t rows, int32_t cols)
{
    memset(fx, 0, sizeof(*fx));
    plan_fixture_chain(&fx->plan, &fx->header, fx->tensors, &fx->op, &fx->stage,
                       fx->indices, fx->shapes, 1, 0);

    fx->header.version = TIGRIS_SCHEMA_VERSION_V9;
    fx->header.num_tile_plans = 1;

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

    fx->perm[0] = 0; fx->perm[1] = 2; fx->perm[2] = 1;
    fx->attribute.op_index = 0;
    fx->attribute.type = TIGRIS_OP_ATTR_TRANSPOSE_PERM;
    fx->attribute.data_offset = 0;
    fx->attribute.data_len = 3;

    fx->stage.tile_plan_idx = 0;

    fx->tile_plan.tileable = 1;
    fx->tile_plan.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    fx->tile_plan.tile_height = 1;
    fx->tile_plan.num_tiles = (uint16_t)(rows > cols ? rows : cols);
    fx->tile_plan.original_height = (uint16_t)(rows > cols ? rows : cols);
    /* The solver bands the longer axis and says which one that is, so the
     * fixture states what the compiler would. */
    fx->tile_plan.flags = (uint16_t)((cols >= rows)
                                     ? TIGRIS_TILE_FLAG_BAND_ON_COLUMNS : 0u);

    fx->plan.tile_plans = &fx->tile_plan;
    fx->plan.op_attributes = &fx->attribute;
    fx->plan.op_attribute_data = fx->perm;
    fx->plan.num_op_attributes = 1;
}

static tigris_exec_error_t run_case(
    transpose_fixture_t *fx, uint8_t *fast, uint32_t fast_size,
    const float *input, float *output, uint16_t *tiled)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t slow[SLOW_SIZE];
    uint8_t workspace[TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(2, 1, 1, 0, 0, 0)];
    return run_fixture(&fx->plan, tigris_dispatch_kernel, NULL, fast, fast_size,
                       slow, sizeof(slow), workspace, sizeof(workspace),
                       input, output, tiled);
}

/** A rank-4 conversion: [N, H, W, C] -> [N, C, H, W] or the way back. */
typedef struct {
    int32_t h;
    int32_t w;
    int32_t c;
    int spatial_first;
} rank4_layout_t;

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
 * Run the same conversion whole and in bands and compare. `rows` and `cols`
 * are the input's two permuted extents; swapping them swaps which side of the
 * conversion takes the strided transfer. A non-NULL `rank4` re-shapes the
 * fixture as a rank-4 conversion, which leaves the two spatial axes in their
 * relative order and so carries the same matrix with H*W read as one extent.
 */
static void check_band_invariance(int32_t rows, int32_t cols,
                                  const rank4_layout_t *rank4, const char *what)
{
    printf("  check_band_invariance %s...\n", what);

    transpose_fixture_t whole;
    transpose_fixture_t banded;
    build_fixture(&whole, rows, cols);
    build_fixture(&banded, rows, cols);
    if (rank4) {
        make_rank4(&whole, rank4->h, rank4->w, rank4->c, rank4->spatial_first);
        make_rank4(&banded, rank4->h, rank4->w, rank4->c, rank4->spatial_first);
        /* The spatial pair is the row axis going to linear order and the
         * column axis coming back. */
        rows = rank4->spatial_first ? rank4->h * rank4->w : rank4->c;
        cols = rank4->spatial_first ? rank4->c : rank4->h * rank4->w;
    }

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


/* Which axis the band runs along is a memory choice, not a correctness one,
 * so a plan that states the other side still has to produce the permutation.
 * That is what makes the stated bit safe to obey: the executor no longer has
 * to agree with the solver by coincidence. A plan from before schema 9 states
 * nothing, and the executor derives it as it always did. */
static void check_band_orientation(int32_t rows, int32_t cols)
{
    printf("  check_band_orientation...\n");

    float input[TEST_ELEMENTS];
    for (int i = 0; i < TEST_ELEMENTS; i++)
        input[i] = (float)i * 0.25f - 48.0f;

    _Alignas(TIGRIS_TENSOR_ALIGN) static uint8_t small[1024];
    float expected[TEST_ELEMENTS];
    uint16_t tiled = 0;
    transpose_fixture_t fx;

    build_fixture(&fx, rows, cols);
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, expected, &tiled),
                   TIGRIS_EXEC_OK, "conversion on the stated axis");
    TEST_ASSERT_EQ(tiled, 1, "stated-axis run is tiled");

    float other[TEST_ELEMENTS];
    build_fixture(&fx, rows, cols);
    fx.tile_plan.flags = (uint16_t)(fx.tile_plan.flags ^
                                    TIGRIS_TILE_FLAG_BAND_ON_COLUMNS);
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, other, &tiled),
                   TIGRIS_EXEC_OK, "conversion on the other axis");
    TEST_ASSERT(memcmp(expected, other, sizeof(expected)) == 0,
                "both orientations produce the permutation");

    float derived[TEST_ELEMENTS];
    build_fixture(&fx, rows, cols);
    fx.header.version = TIGRIS_SCHEMA_VERSION_V8;
    fx.tile_plan.flags = 0;
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, derived, &tiled),
                   TIGRIS_EXEC_OK, "conversion on a plan that states nothing");
    TEST_ASSERT(memcmp(expected, derived, sizeof(expected)) == 0,
                "a plan from before schema 9 still converts");
}

/** A fixed axis can band a permutation outside the matrix-conversion path. */
static void check_fixed_axis_permutation(void)
{
    printf("  check_fixed_axis_permutation...\n");

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
    TEST_ASSERT_EQ(
        run_case(&fx, small, sizeof(small), input, out, &tiled),
        TIGRIS_EXEC_OK, "fixed-axis permutation executes");
    TEST_ASSERT_EQ(tiled, 1, "fixed-axis permutation is tiled");
    for (int h = 0; h < 4; h++)
        for (int w = 0; w < 4; w++)
            for (int c = 0; c < 24; c++)
                TEST_ASSERT(out[(c * 4 + w) * 4 + h] == input[(h * 4 + w) * 24 + c],
                            "band preserves permutation coordinates");
    fx.perm[0] = 3; fx.perm[1] = 2; fx.perm[2] = 1; fx.perm[3] = 0;
    fx.shapes[4] = 24; fx.shapes[5] = 4; fx.shapes[6] = 4; fx.shapes[7] = 1;
    tiled = 0u;
    TEST_ASSERT_EQ(run_case(&fx, small, sizeof(small), input, out, &tiled),
                   TIGRIS_EXEC_ERR_TILE, "permutation without a fixed axis is refused");
    TEST_ASSERT_EQ(tiled, 0, "unsupported permutation is not tiled");
}

int main(void)
{
    printf("Layout conversion tiling tests:\n");
    static const rank4_layout_t to_linear = {4, 4, 24, 1};
    static const rank4_layout_t to_spatial = {4, 4, 24, 0};

    check_band_invariance(TEST_SHORT, TEST_LONG, NULL, "gathering input columns");
    check_band_invariance(TEST_LONG, TEST_SHORT, NULL, "scattering output columns");
    check_band_orientation(TEST_SHORT, TEST_LONG);
    check_band_invariance(TEST_SHORT, TEST_LONG, &to_linear,
                          "rank 4 to linear order");
    check_band_invariance(TEST_SHORT, TEST_LONG, &to_spatial,
                          "rank 4 back to spatial order");
    check_fixed_axis_permutation();

#ifdef TIGRIS_TRACE
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_TRANSPOSE)) != 0u,
                   1, "transpose path checked against trace");
#endif

    return test_report();
}

/**
 * @file test_tile_2d.c
 * @brief Runtime-side checks for Phase 1.3b (2D spatial tiling).
 *
 * Task 4: the TIGRIS_TILE_AXIS_HW constant, the width fields on
 *        tigris_tile_ctx_t, and the loader accepting a tileable tile plan
 *        whose axis is HW instead of only HEIGHT_OR_LENGTH.
 *
 * The HW-axis fixture is a real plan emitted by the compiler's 2D tile
 * solver (feature/2d-tiling-core): a [1,16,16,16] float32 3x3 stride-1
 * pad-1 Conv compiled at a 2K fast-memory budget, small enough that even a
 * single-row height tile still overflows the budget, forcing the compiler
 * to fall back to a 2x2 HW (height+width) core tile. The executor does not
 * yet dispatch HW-axis tile plans (that lands in Task 7); the Task 4 test
 * only exercises the loader's acceptance of the axis value.
 *
 * Task 5: tigris_mem_load_tile_2d and tigris_mem_spill_tile_2d, the strided
 *        2D sub-rectangle load and spill primitives that Task 7's HW-axis
 *        executor path will call. These tests exercise the mem-level
 *        primitives directly, independent of a loaded plan file, using a
 *        minimal in-memory tensor descriptor and shape pool.
 *
 * Task 6: per-tile in_w and left/right pad overrides in kern_conv2d and
 *        kern_avg_pool (reference float32 and int8). Each test builds one
 *        op describing the full untiled tensor, runs it once with
 *        mem->tile inactive (the golden full-canvas output), then runs the
 *        SAME op again against a packed 2D tile loaded with
 *        tigris_mem_load_tile_2d and mem->tile.width_tiled set, and asserts
 *        the packed core is byte-identical to the matching sub-block of the
 *        golden output. Before Task 6, PL stayed at the op's global
 *        pad_left even for a width-tiled call, so an interior tile (whose
 *        effective pad_left is 0) still shifted every column by the
 *        model's global pad_left.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"

/* Test infrastructure */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

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

/* Same pattern as test_loader.c: malloc gives sufficiently aligned storage
 * for tigris_plan_load, which only requires natural alignment of the typed
 * pool sections within the buffer. */
static uint8_t *load_file(const char *path, uint32_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "  Cannot open: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = (uint8_t *)malloc(len);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if ((long)fread(buf, 1, len, f) != len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (uint32_t)len;
    return buf;
}

static void test_loader_accepts_hw_axis(void)
{
    printf("  test_loader_accepts_hw_axis...\n");

    char path[512];
    int path_len = snprintf(path, sizeof(path), "%s/hw-axis-2d-conv.tgrs",
                             TIGRIS_TILE2D_FIXTURE_DIR);
    if (path_len <= 0 || (size_t)path_len >= sizeof(path)) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: fixture path does not fit\n");
        return;
    }

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    tests_run++;
    if (!buf) {
        tests_failed++;
        fprintf(stderr, "  FAIL: HW axis fixture is readable\n");
        return;
    }
    tests_passed++;

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "loader accepts axis=HW plan");

    if (err == TIGRIS_OK) {
        TEST_ASSERT_EQ(plan.header->num_tile_plans, 1,
                       "HW axis fixture has one tile plan");
        if (plan.header->num_tile_plans == 1) {
            TEST_ASSERT_EQ(plan.tile_plans[0].tileable, 1,
                           "HW axis fixture tile plan is tileable");
            TEST_ASSERT_EQ(plan.tile_plans[0].axis, TIGRIS_TILE_AXIS_HW,
                           "HW axis fixture tile plan axis is HW");
        }
    }

    free(buf);
}

static void test_tile_ctx_has_width_fields(void)
{
    printf("  test_tile_ctx_has_width_fields...\n");

    tigris_tile_ctx_t t;
    memset(&t, 0, sizeof t);
    t.pad_left = 1;
    t.pad_right = 2;
    t.width_tiled = 1;
    TEST_ASSERT_EQ(t.pad_left + t.pad_right + t.width_tiled, 4,
                   "width fields present");
}

/* Task 5: tigris_mem_load_tile_2d / tigris_mem_spill_tile_2d */

#define T2D_N    1
#define T2D_H    8
#define T2D_W    8
#define T2D_C    2
#define T2D_TIDX 0

/* Fill an [N,H,W,C] int8 tensor so element [h,w,c] == (int8_t)(h*100+w*10+c).
 * A distinct value per axis catches any axis or stride mixup in the copy
 * loop, not just an off-by-one in a single dimension. */
static void t2d_fill_source(int8_t *buf)
{
    for (int32_t h = 0; h < T2D_H; h++)
        for (int32_t w = 0; w < T2D_W; w++)
            for (int32_t c = 0; c < T2D_C; c++)
                buf[(h * T2D_W + w) * T2D_C + c] = (int8_t)(h * 100 + w * 10 + c);
}

/* Minimal plan with a single [T2D_N,T2D_H,T2D_W,T2D_C] int8 tensor. The 2D
 * load/spill primitives only touch plan->tensors and plan->shape_pool, so
 * nothing else needs to be populated. */
static void t2d_build_plan(tigris_tensor_t *tensor, int32_t *shape, tigris_plan_t *plan)
{
    shape[0] = T2D_N;
    shape[1] = T2D_H;
    shape[2] = T2D_W;
    shape[3] = T2D_C;

    memset(tensor, 0, sizeof(*tensor));
    tensor->ndim = 4;
    tensor->shape_off = 0;
    tensor->size_bytes = (uint32_t)(T2D_N * T2D_H * T2D_W * T2D_C) * (uint32_t)sizeof(int8_t);
    tensor->quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    memset(plan, 0, sizeof(*plan));
    plan->tensors = tensor;
    plan->shape_pool = shape;
}

static void test_load_tile_2d_roundtrip(void)
{
    printf("  test_load_tile_2d_roundtrip...\n");

    tigris_tensor_t tensor;
    int32_t shape[4];
    tigris_plan_t plan;
    t2d_build_plan(&tensor, shape, &plan);

    int8_t slow_buf[T2D_N * T2D_H * T2D_W * T2D_C];
    t2d_fill_source(slow_buf);

    uint8_t fast_buf[128];
    uint8_t slow_arena[128];
    void *tensor_ptrs[1] = { NULL };

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, tensor_ptrs, 1,
                                               fast_buf, sizeof(fast_buf),
                                               slow_arena, sizeof(slow_arena));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init ok");

    /* Point tensor_ptrs at the slow storage, mirroring the executor's
     * convention of setting tensor_ptrs[tidx] to the slow base right before
     * calling a tile load. */
    mem.tensor_ptrs[T2D_TIDX] = slow_buf;

    tigris_mem_error_t e = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX, 2, 6, 3, 7);
    TEST_ASSERT_EQ(e, TIGRIS_MEM_OK, "2D load ok");

    const int8_t *fast = (const int8_t *)tigris_mem_tensor_ptr(&mem, T2D_TIDX);
    if (!fast) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: fast tile pointer is NULL\n");
        return;
    }

    for (int r = 0; r < 4; r++)
        for (int cc = 0; cc < 4; cc++)
            for (int ch = 0; ch < 2; ch++) {
                int8_t got = fast[((r * 4) + cc) * 2 + ch];
                int8_t want = (int8_t)((2 + r) * 100 + (3 + cc) * 10 + ch);
                TEST_ASSERT_EQ(got, want, "packed element matches source rectangle");
            }
}

static void test_spill_tile_2d_roundtrip(void)
{
    printf("  test_spill_tile_2d_roundtrip...\n");

    tigris_tensor_t tensor;
    int32_t shape[4];
    tigris_plan_t plan;
    t2d_build_plan(&tensor, shape, &plan);

    int8_t slow_buf[T2D_N * T2D_H * T2D_W * T2D_C];
    memset(slow_buf, 0xFF, sizeof(slow_buf)); /* every element == -1 */

    int8_t core[4 * 4 * 2];
    for (int r = 0; r < 4; r++)
        for (int cc = 0; cc < 4; cc++)
            for (int ch = 0; ch < 2; ch++)
                core[((r * 4) + cc) * 2 + ch] = (int8_t)(50 + r * 4 + cc * 2 + ch);

    uint8_t fast_buf[128];
    uint8_t slow_arena[128];
    void *tensor_ptrs[1] = { NULL };

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, tensor_ptrs, 1,
                                               fast_buf, sizeof(fast_buf),
                                               slow_arena, sizeof(slow_arena));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init ok");

    /* tensor_ptrs points at the already-computed packed core, mirroring the
     * executor's convention right before calling a tile spill. */
    mem.tensor_ptrs[T2D_TIDX] = core;

    tigris_mem_error_t e = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                      slow_buf, 2, 6, 3, 7);
    TEST_ASSERT_EQ(e, TIGRIS_MEM_OK, "2D spill ok");

    for (int32_t h = 0; h < T2D_H; h++) {
        for (int32_t w = 0; w < T2D_W; w++) {
            int in_rect = (h >= 2 && h < 6 && w >= 3 && w < 7);
            for (int32_t c = 0; c < T2D_C; c++) {
                int8_t got = slow_buf[(h * T2D_W + w) * T2D_C + c];
                if (in_rect) {
                    int8_t want = core[(((h - 2) * 4) + (w - 3)) * 2 + c];
                    TEST_ASSERT_EQ(got, want, "spilled rectangle matches core");
                } else {
                    TEST_ASSERT_EQ(got, (int8_t)-1, "outside rectangle stays untouched");
                }
            }
        }
    }

    int restored = (tigris_mem_tensor_ptr(&mem, T2D_TIDX) == slow_buf);
    TEST_ASSERT_EQ(restored, 1, "tensor_ptrs restored to slow_base");
}

#define T2D_NDIM_TIDX 1  /* second tensor: not rank 4, for ndim rejection */

/* Two-tensor plan: index T2D_TIDX is the valid [T2D_N,T2D_H,T2D_W,T2D_C]
 * rank-4 tensor from t2d_build_plan, index T2D_NDIM_TIDX is a rank-3 [N,L,C]
 * tensor with no width axis, for exercising the ndim != 4 rejection path in
 * both tigris_mem_load_tile_2d and tigris_mem_spill_tile_2d. Both tensors
 * share one shape pool, the same convention t2d_build_plan uses. */
static void t2d_build_plan_with_bad_ndim(tigris_tensor_t *tensors, int32_t *shape,
                                          tigris_plan_t *plan)
{
    shape[0] = T2D_N;
    shape[1] = T2D_H;
    shape[2] = T2D_W;
    shape[3] = T2D_C;
    shape[4] = T2D_N;
    shape[5] = T2D_H;
    shape[6] = T2D_C;

    memset(&tensors[T2D_TIDX], 0, sizeof(tensors[T2D_TIDX]));
    tensors[T2D_TIDX].ndim = 4;
    tensors[T2D_TIDX].shape_off = 0;
    tensors[T2D_TIDX].size_bytes =
        (uint32_t)(T2D_N * T2D_H * T2D_W * T2D_C) * (uint32_t)sizeof(int8_t);
    tensors[T2D_TIDX].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    memset(&tensors[T2D_NDIM_TIDX], 0, sizeof(tensors[T2D_NDIM_TIDX]));
    tensors[T2D_NDIM_TIDX].ndim = 3;
    tensors[T2D_NDIM_TIDX].shape_off = 4;
    tensors[T2D_NDIM_TIDX].size_bytes =
        (uint32_t)(T2D_N * T2D_H * T2D_C) * (uint32_t)sizeof(int8_t);
    tensors[T2D_NDIM_TIDX].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    memset(plan, 0, sizeof(*plan));
    plan->tensors = tensors;
    plan->shape_pool = shape;
}

static void test_load_tile_2d_rejects_bad_bounds(void)
{
    printf("  test_load_tile_2d_rejects_bad_bounds...\n");

    tigris_tensor_t tensors[2];
    int32_t shape[7];
    tigris_plan_t plan;
    t2d_build_plan_with_bad_ndim(tensors, shape, &plan);

    int8_t slow_buf[T2D_N * T2D_H * T2D_W * T2D_C];
    t2d_fill_source(slow_buf);

    uint8_t fast_buf[128];
    uint8_t slow_arena[128];
    void *tensor_ptrs[2] = { NULL, NULL };

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, tensor_ptrs, 2,
                                               fast_buf, sizeof(fast_buf),
                                               slow_arena, sizeof(slow_arena));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init ok");
    mem.tensor_ptrs[T2D_TIDX] = slow_buf;

    /* tidx >= num_tensors must be rejected before any tensor lookup. */
    tigris_mem_error_t e_tidx = tigris_mem_load_tile_2d(&mem, &plan, 2, 0, 1, 0, 1);
    TEST_ASSERT_EQ(e_tidx, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range tidx rejected");

    /* ndim != 4 must be rejected before any bounds check. */
    tigris_mem_error_t e_ndim = tigris_mem_load_tile_2d(&mem, &plan, T2D_NDIM_TIDX,
                                                          0, 1, 0, 1);
    TEST_ASSERT_EQ(e_ndim, TIGRIS_MEM_ERR_BAD_INDEX, "non-4D tensor rejected");

    /* h0 < 0 */
    tigris_mem_error_t e_h0_neg = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                            -1, 6, 3, 7);
    TEST_ASSERT_EQ(e_h0_neg, TIGRIS_MEM_ERR_BAD_INDEX, "negative h0 rejected");

    /* h1 <= h0, tested at equality */
    tigris_mem_error_t e_h1_eq = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                           3, 3, 3, 7);
    TEST_ASSERT_EQ(e_h1_eq, TIGRIS_MEM_ERR_BAD_INDEX, "h1 == h0 rejected");

    /* h1 > H */
    tigris_mem_error_t e_h1_over = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                             2, T2D_H + 1, 3, 7);
    TEST_ASSERT_EQ(e_h1_over, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range h1 rejected");

    /* w0 < 0 */
    tigris_mem_error_t e_w0_neg = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                            2, 6, -1, 7);
    TEST_ASSERT_EQ(e_w0_neg, TIGRIS_MEM_ERR_BAD_INDEX, "negative w0 rejected");

    /* w1 <= w0, tested at equality */
    tigris_mem_error_t e_w1_eq = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                           2, 6, 4, 4);
    TEST_ASSERT_EQ(e_w1_eq, TIGRIS_MEM_ERR_BAD_INDEX, "w1 == w0 rejected");

    /* w1 > W must be rejected before any bytes move. */
    tigris_mem_error_t e_w1_over = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                             2, 6, 3, T2D_W + 1);
    TEST_ASSERT_EQ(e_w1_over, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range w1 rejected");
}

static void test_spill_tile_2d_rejects_bad_bounds(void)
{
    printf("  test_spill_tile_2d_rejects_bad_bounds...\n");

    tigris_tensor_t tensors[2];
    int32_t shape[7];
    tigris_plan_t plan;
    t2d_build_plan_with_bad_ndim(tensors, shape, &plan);

    int8_t core[4 * 4 * 2];
    memset(core, 0, sizeof(core));

    int8_t slow_buf[T2D_N * T2D_H * T2D_W * T2D_C];
    memset(slow_buf, 0xFF, sizeof(slow_buf));

    uint8_t fast_buf[128];
    uint8_t slow_arena[128];
    void *tensor_ptrs[2] = { NULL, NULL };

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, tensor_ptrs, 2,
                                               fast_buf, sizeof(fast_buf),
                                               slow_arena, sizeof(slow_arena));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init ok");
    mem.tensor_ptrs[T2D_TIDX] = core;

    /* tidx >= num_tensors must be rejected before any tensor lookup. */
    tigris_mem_error_t e_tidx = tigris_mem_spill_tile_2d(&mem, &plan, 2, slow_buf,
                                                           0, 1, 0, 1);
    TEST_ASSERT_EQ(e_tidx, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range tidx rejected");

    /* ndim != 4 must be rejected before any bounds check. */
    tigris_mem_error_t e_ndim = tigris_mem_spill_tile_2d(&mem, &plan, T2D_NDIM_TIDX,
                                                           slow_buf, 0, 1, 0, 1);
    TEST_ASSERT_EQ(e_ndim, TIGRIS_MEM_ERR_BAD_INDEX, "non-4D tensor rejected");

    /* h0 < 0 */
    tigris_mem_error_t e_h0_neg = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                             slow_buf, -1, 6, 3, 7);
    TEST_ASSERT_EQ(e_h0_neg, TIGRIS_MEM_ERR_BAD_INDEX, "negative h0 rejected");

    /* h1 <= h0, tested at equality */
    tigris_mem_error_t e_h1_eq = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                            slow_buf, 3, 3, 3, 7);
    TEST_ASSERT_EQ(e_h1_eq, TIGRIS_MEM_ERR_BAD_INDEX, "h1 == h0 rejected");

    /* h1 > H */
    tigris_mem_error_t e_h1_over = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                              slow_buf, 2, T2D_H + 1, 3, 7);
    TEST_ASSERT_EQ(e_h1_over, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range h1 rejected");

    /* w0 < 0 */
    tigris_mem_error_t e_w0_neg = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                             slow_buf, 2, 6, -1, 7);
    TEST_ASSERT_EQ(e_w0_neg, TIGRIS_MEM_ERR_BAD_INDEX, "negative w0 rejected");

    /* w1 <= w0, tested at equality */
    tigris_mem_error_t e_w1_eq = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                            slow_buf, 2, 6, 4, 4);
    TEST_ASSERT_EQ(e_w1_eq, TIGRIS_MEM_ERR_BAD_INDEX, "w1 == w0 rejected");

    /* w1 > W must be rejected before any bytes move. */
    tigris_mem_error_t e_w1_over = tigris_mem_spill_tile_2d(&mem, &plan, T2D_TIDX,
                                                              slow_buf, 2, 6, 3, T2D_W + 1);
    TEST_ASSERT_EQ(e_w1_over, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range w1 rejected");

    /* Nothing in slow_buf should have moved: every rejected call must fail
     * before the copy loop runs. */
    for (size_t i = 0; i < sizeof(slow_buf); i++) {
        tests_run++;
        if (slow_buf[i] != (int8_t)-1) {
            tests_failed++;
            fprintf(stderr,
                    "  FAIL: %s (byte %zu): slow_buf untouched by rejected spills "
                    "(got %d, expected -1)\n",
                    __func__, i, (int)slow_buf[i]);
        } else {
            tests_passed++;
        }
    }
}

/* Task 6: per-tile in_w / left-right pad overrides */

/* Shared geometry: a 3x3 stride-1 pad-1 spatial op over a 6x6 canvas keeps
 * the output the same size as the input, so a golden run and a tiled run
 * can share one op/weight setup and only the tile bounds differ. */
#define TC_H  6
#define TC_W  6
#define TC_C  2
#define TC_OC 2
#define TC_K  3
#define TS_C  1   /* s8 tests use a single channel to keep quant math trivial */

static void tc_fill_input_f32(float *x)
{
    for (int h = 0; h < TC_H; h++)
        for (int w = 0; w < TC_W; w++)
            for (int c = 0; c < TC_C; c++)
                x[(h * TC_W + w) * TC_C + c] = (float)(h * 100 + w * 10 + c);
}

static void tc_fill_weight_f32(float *w)
{
    for (int oc = 0; oc < TC_OC; oc++)
        for (int kh = 0; kh < TC_K; kh++)
            for (int kw = 0; kw < TC_K; kw++)
                for (int ic = 0; ic < TC_C; ic++)
                    w[((oc * TC_K + kh) * TC_K + kw) * TC_C + ic] =
                        (float)(oc * 1000 + kh * 100 + kw * 10 + ic) * 0.01f;
}

static void ts_fill_input_s8(int8_t *x)
{
    for (int h = 0; h < TC_H; h++)
        for (int w = 0; w < TC_W; w++)
            x[h * TC_W + w] = (int8_t)(h * TC_W + w);
}

/* Column-difference kernel: distinct per (kh,kw) so a left/right indexing
 * bug shows up as a wrong value, not an accidental match. */
static const int8_t ts_weight[TC_K * TC_K] = {
    1, 0, -1,
    1, 0, -1,
    1, 0, -1
};

static void tc_build_conv_plan_f32(
    tigris_tensor_t *tensors, int32_t *shape_pool, uint16_t *index_pool,
    tigris_weight_entry_t *weights, float *weight_blob,
    tigris_file_header_t *header, tigris_op_t *op, tigris_plan_t *plan,
    const float *W, uint32_t w_bytes, const float *B, uint32_t b_bytes)
{
    int32_t x_shape[4] = {1, TC_H, TC_W, TC_C};
    int32_t y_shape[4] = {1, TC_H, TC_W, TC_OC};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 1;
    tensors[0].size_bytes = (uint32_t)(TC_H * TC_W * TC_C) * (uint32_t)sizeof(float);
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 1;
    tensors[1].size_bytes = (uint32_t)(TC_H * TC_W * TC_OC) * (uint32_t)sizeof(float);
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    index_pool[0] = 0; index_pool[1] = 1;

    /* weight_blob is declared float[] by the caller (not uint8_t[]) so the
     * f32 kernels' direct B[oc] dereference stays naturally aligned, the
     * same convention test_kernels.c uses. */
    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = w_bytes;
    weights[1].name_str = 0; weights[1].offset = w_bytes; weights[1].size_bytes = b_bytes;
    memcpy(weight_blob, W, w_bytes);
    memcpy((uint8_t *)weight_blob + w_bytes, B, b_bytes);

    memset(op, 0, sizeof(*op));
    op->op_type = TIGRIS_OP_CONV;
    op->num_inputs = 1; op->num_outputs = 1;
    op->inputs_off = 0; op->outputs_off = 1;
    op->spatial.kernel_h = TC_K; op->spatial.kernel_w = TC_K;
    op->spatial.stride_h = 1; op->spatial.stride_w = 1;
    op->spatial.dilation_h = 1; op->spatial.dilation_w = 1;
    op->spatial.pad_top = 1; op->spatial.pad_bottom = 1;
    op->spatial.pad_left = 1; op->spatial.pad_right = 1;
    op->spatial.group = 1;
    op->weight_idx = 0; op->bias_idx = 1;
    op->fused_act = TIGRIS_ACT_NONE;

    memset(header, 0, sizeof(*header));
    header->num_tensors = 2; header->num_ops = 1; header->num_weights = 2;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = op;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
    plan->weight_entries = weights;
    plan->weight_blob = (const uint8_t *)weight_blob;
}

static void tc_build_conv_plan_s8(
    tigris_tensor_t *tensors, int32_t *shape_pool, uint16_t *index_pool,
    tigris_weight_entry_t *weights, uint8_t *weight_blob,
    tigris_quant_param_t *qp, int32_t *qd,
    tigris_file_header_t *header, tigris_op_t *op, tigris_plan_t *plan)
{
    int32_t x_shape[4] = {1, TC_H, TC_W, TS_C};
    int32_t y_shape[4] = {1, TC_H, TC_W, TS_C};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    /* Output quant: identity requant (multiplier=2^30, shift=1), the same
     * per-tensor convention test_conv2d_s8_per_tensor uses in
     * test_kernels_s8.c, so the int8 output equals the raw accumulator. */
    memset(qp, 0, sizeof(*qp) * 2);
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 1;
    qp[1].multiplier_off = 0; qp[1].shift_off = 1;
    qd[0] = 1073741824; /* 2^30 */
    qd[1] = 1;

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 3;
    tensors[0].size_bytes = (uint32_t)(TC_H * TC_W * TS_C);
    tensors[0].quant_param_idx = 0;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 3;
    tensors[1].size_bytes = (uint32_t)(TC_H * TC_W * TS_C);
    tensors[1].quant_param_idx = 1;

    index_pool[0] = 0; index_pool[1] = 1;

    static const int32_t bias[1] = {0};
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(ts_weight);
    weights[1].name_str = 0; weights[1].offset = sizeof(ts_weight);
    weights[1].size_bytes = sizeof(bias);
    memcpy(weight_blob, ts_weight, sizeof(ts_weight));
    memcpy(weight_blob + sizeof(ts_weight), bias, sizeof(bias));

    memset(op, 0, sizeof(*op));
    op->op_type = TIGRIS_OP_CONV;
    op->num_inputs = 1; op->num_outputs = 1;
    op->inputs_off = 0; op->outputs_off = 1;
    op->spatial.kernel_h = TC_K; op->spatial.kernel_w = TC_K;
    op->spatial.stride_h = 1; op->spatial.stride_w = 1;
    op->spatial.dilation_h = 1; op->spatial.dilation_w = 1;
    op->spatial.pad_top = 1; op->spatial.pad_bottom = 1;
    op->spatial.pad_left = 1; op->spatial.pad_right = 1;
    op->spatial.group = 1;
    op->weight_idx = 0; op->bias_idx = 1;
    op->act_min = -128; op->act_max = 127;

    memset(header, 0, sizeof(*header));
    header->num_tensors = 2; header->num_ops = 1; header->num_weights = 2;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = op;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
    plan->weight_entries = weights;
    plan->weight_blob = weight_blob;
    plan->quant_params = qp;
    plan->quant_data = qd;
    plan->num_quant_params = 2;
}

static void tc_build_avgpool_plan_f32(
    tigris_tensor_t *tensors, int32_t *shape_pool, uint16_t *index_pool,
    tigris_file_header_t *header, tigris_op_t *op, tigris_plan_t *plan)
{
    int32_t x_shape[4] = {1, TC_H, TC_W, TC_C};
    int32_t y_shape[4] = {1, TC_H, TC_W, TC_C};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 1;
    tensors[0].size_bytes = (uint32_t)(TC_H * TC_W * TC_C) * (uint32_t)sizeof(float);
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 1;
    tensors[1].size_bytes = (uint32_t)(TC_H * TC_W * TC_C) * (uint32_t)sizeof(float);
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    index_pool[0] = 0; index_pool[1] = 1;

    memset(op, 0, sizeof(*op));
    op->op_type = TIGRIS_OP_AVG_POOL;
    op->num_inputs = 1; op->num_outputs = 1;
    op->inputs_off = 0; op->outputs_off = 1;
    op->spatial.kernel_h = TC_K; op->spatial.kernel_w = TC_K;
    op->spatial.stride_h = 1; op->spatial.stride_w = 1;
    op->spatial.pad_top = 1; op->spatial.pad_bottom = 1;
    op->spatial.pad_left = 1; op->spatial.pad_right = 1;
    op->weight_idx = TIGRIS_NO_WEIGHT; op->bias_idx = TIGRIS_NO_WEIGHT;

    memset(header, 0, sizeof(*header));
    header->num_tensors = 2; header->num_ops = 1;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = op;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
}

static void tc_build_avgpool_plan_s8(
    tigris_tensor_t *tensors, int32_t *shape_pool, uint16_t *index_pool,
    tigris_file_header_t *header, tigris_op_t *op, tigris_plan_t *plan)
{
    int32_t x_shape[4] = {1, TC_H, TC_W, TS_C};
    int32_t y_shape[4] = {1, TC_H, TC_W, TS_C};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 3;
    tensors[0].size_bytes = (uint32_t)(TC_H * TC_W * TS_C);
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 3;
    tensors[1].size_bytes = (uint32_t)(TC_H * TC_W * TS_C);
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    index_pool[0] = 0; index_pool[1] = 1;

    memset(op, 0, sizeof(*op));
    op->op_type = TIGRIS_OP_AVG_POOL;
    op->num_inputs = 1; op->num_outputs = 1;
    op->inputs_off = 0; op->outputs_off = 1;
    op->spatial.kernel_h = TC_K; op->spatial.kernel_w = TC_K;
    op->spatial.stride_h = 1; op->spatial.stride_w = 1;
    op->spatial.pad_top = 1; op->spatial.pad_bottom = 1;
    op->spatial.pad_left = 1; op->spatial.pad_right = 1;
    op->weight_idx = TIGRIS_NO_WEIGHT; op->bias_idx = TIGRIS_NO_WEIGHT;
    op->act_min = -128; op->act_max = 127;

    memset(header, 0, sizeof(*header));
    header->num_tensors = 2; header->num_ops = 1;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = op;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
}

/* Compare a packed [2,2,channels] tile core against the sub-block of the
 * golden full output starting at (oh0, ow0). golden_w is the golden
 * tensor's full output width (row-major NHWC indexing). */
static void tc_compare_f32(
    const char *name, const float *tile, const float *golden_full,
    int golden_w, int channels, int oh0, int ow0)
{
    for (int oh = 0; oh < 2; oh++) {
        for (int ow = 0; ow < 2; ow++) {
            int gh = oh0 + oh, gw = ow0 + ow;
            for (int c = 0; c < channels; c++) {
                float got  = tile[(oh * 2 + ow) * channels + c];
                float want = golden_full[(gh * golden_w + gw) * channels + c];
                tests_run++;
                if (got != want) {
                    tests_failed++;
                    fprintf(stderr,
                        "  FAIL: %s: tile[%d,%d,%d]=%.6f != golden[%d,%d,%d]=%.6f\n",
                        name, oh, ow, c, (double)got, gh, gw, c, (double)want);
                } else {
                    tests_passed++;
                }
            }
        }
    }
}

static void tc_compare_s8(
    const char *name, const int8_t *tile, const int8_t *golden_full,
    int golden_w, int channels, int oh0, int ow0)
{
    for (int oh = 0; oh < 2; oh++) {
        for (int ow = 0; ow < 2; ow++) {
            int gh = oh0 + oh, gw = ow0 + ow;
            for (int c = 0; c < channels; c++) {
                int8_t got  = tile[(oh * 2 + ow) * channels + c];
                int8_t want = golden_full[(gh * golden_w + gw) * channels + c];
                tests_run++;
                if (got != want) {
                    tests_failed++;
                    fprintf(stderr,
                        "  FAIL: %s: tile[%d,%d,%d]=%d != golden[%d,%d,%d]=%d\n",
                        name, oh, ow, c, (int)got, gh, gw, c, (int)want);
                } else {
                    tests_passed++;
                }
            }
        }
    }
}

/* Pack input[h0:h1, w0:w1, :] via tigris_mem_load_tile_2d, set a 2D tile
 * context with the given effective top/left pads and zero row-start
 * offsets (2D tiles have no line-buffered halo), and dispatch the op. */
static void tc_run_tile_f32(
    int (*dispatch)(const tigris_plan_t *, const tigris_op_t *, uint16_t,
                     tigris_mem_t *, void *),
    const tigris_plan_t *plan, const tigris_op_t *op, const float *X_full,
    int32_t h0, int32_t h1, int32_t w0, int32_t w1,
    uint16_t pad_top, uint16_t pad_left, float *out_tile)
{
    uint8_t fast_buf[512];
    uint8_t slow_buf[16];
    void *tile_ptrs[2] = { NULL, NULL };
    tigris_mem_t tile_mem;
    tigris_mem_error_t merr = tigris_mem_init(&tile_mem, tile_ptrs, 2,
                                               fast_buf, sizeof(fast_buf),
                                               slow_buf, sizeof(slow_buf));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "tile mem init ok");

    tile_mem.tensor_ptrs[0] = (void *)X_full;
    tigris_mem_error_t lerr = tigris_mem_load_tile_2d(&tile_mem, plan, 0, h0, h1, w0, w1);
    TEST_ASSERT_EQ(lerr, TIGRIS_MEM_OK, "2D tile load ok");

    tile_mem.tensor_ptrs[1] = out_tile;

    tile_mem.tile.active = 1;
    tile_mem.tile.width_tiled = 1;
    tile_mem.tile.in_h = h1 - h0;
    tile_mem.tile.out_h = 2;
    tile_mem.tile.in_w = w1 - w0;
    tile_mem.tile.out_w = 2;
    tile_mem.tile.pad_top = pad_top;
    tile_mem.tile.pad_bottom = 0;
    tile_mem.tile.pad_left = pad_left;
    tile_mem.tile.pad_right = 0;
    tile_mem.tile.out_row_start = 0;
    tile_mem.tile.in_row_start = 0;

    int ret = dispatch(plan, op, 0, &tile_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "tiled dispatch ok");
}

static void tc_run_tile_s8(
    int (*dispatch)(const tigris_plan_t *, const tigris_op_t *, uint16_t,
                     tigris_mem_t *, void *),
    const tigris_plan_t *plan, const tigris_op_t *op, const int8_t *X_full,
    int32_t h0, int32_t h1, int32_t w0, int32_t w1,
    uint16_t pad_top, uint16_t pad_left, int8_t *out_tile)
{
    uint8_t fast_buf[256];
    uint8_t slow_buf[16];
    void *tile_ptrs[2] = { NULL, NULL };
    tigris_mem_t tile_mem;
    tigris_mem_error_t merr = tigris_mem_init(&tile_mem, tile_ptrs, 2,
                                               fast_buf, sizeof(fast_buf),
                                               slow_buf, sizeof(slow_buf));
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "tile mem init ok");

    tile_mem.tensor_ptrs[0] = (void *)X_full;
    tigris_mem_error_t lerr = tigris_mem_load_tile_2d(&tile_mem, plan, 0, h0, h1, w0, w1);
    TEST_ASSERT_EQ(lerr, TIGRIS_MEM_OK, "2D tile load ok");

    tile_mem.tensor_ptrs[1] = out_tile;

    tile_mem.tile.active = 1;
    tile_mem.tile.width_tiled = 1;
    tile_mem.tile.in_h = h1 - h0;
    tile_mem.tile.out_h = 2;
    tile_mem.tile.in_w = w1 - w0;
    tile_mem.tile.out_w = 2;
    tile_mem.tile.pad_top = pad_top;
    tile_mem.tile.pad_bottom = 0;
    tile_mem.tile.pad_left = pad_left;
    tile_mem.tile.pad_right = 0;
    tile_mem.tile.out_row_start = 0;
    tile_mem.tile.in_row_start = 0;

    int ret = dispatch(plan, op, 0, &tile_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "tiled dispatch ok");
}

/* Conv2D: interior tile (all effective pads 0) and top-left edge tile
 * (effective pad_top = pad_left = 1, the tensor boundary). f32 and s8. */

static void test_conv_2d_tile_matches_subregion_f32(void)
{
    printf("  test_conv_2d_tile_matches_subregion_f32...\n");

    float X[TC_H * TC_W * TC_C];
    tc_fill_input_f32(X);
    float W[TC_OC * TC_K * TC_K * TC_C];
    tc_fill_weight_f32(W);
    float B[TC_OC] = {0.5f, -0.25f};

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_weight_entry_t weights[2];
    float weight_blob[TC_OC * TC_K * TC_K * TC_C + TC_OC];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_conv_plan_f32(tensors, shape_pool, index_pool, weights,
                            weight_blob, &header, &op, &plan,
                            W, sizeof(W), B, sizeof(B));

    float Y_full[TC_H * TC_W * TC_OC];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden conv dispatch ok");

    /* Interior: input[1:5,1:5,:] packed as in_h=in_w=4, core [2:4,2:4]. */
    float Y_tile[2 * 2 * TC_OC];
    tc_run_tile_f32(tigris_dispatch_kernel, &plan, &op, X, 1, 5, 1, 5, 0, 0, Y_tile);
    tc_compare_f32(__func__, Y_tile, Y_full, TC_W, TC_OC, 2, 2);
}

static void test_conv_2d_tile_edge_f32(void)
{
    printf("  test_conv_2d_tile_edge_f32...\n");

    float X[TC_H * TC_W * TC_C];
    tc_fill_input_f32(X);
    float W[TC_OC * TC_K * TC_K * TC_C];
    tc_fill_weight_f32(W);
    float B[TC_OC] = {0.5f, -0.25f};

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_weight_entry_t weights[2];
    float weight_blob[TC_OC * TC_K * TC_K * TC_C + TC_OC];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_conv_plan_f32(tensors, shape_pool, index_pool, weights,
                            weight_blob, &header, &op, &plan,
                            W, sizeof(W), B, sizeof(B));

    float Y_full[TC_H * TC_W * TC_OC];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden conv dispatch ok");

    /* Top-left corner: input[0:3,0:3,:], effective pad_top=pad_left=1,
     * core [0:2,0:2]. */
    float Y_tile[2 * 2 * TC_OC];
    tc_run_tile_f32(tigris_dispatch_kernel, &plan, &op, X, 0, 3, 0, 3, 1, 1, Y_tile);
    tc_compare_f32(__func__, Y_tile, Y_full, TC_W, TC_OC, 0, 0);
}

static void test_conv_2d_tile_matches_subregion_s8(void)
{
    printf("  test_conv_2d_tile_matches_subregion_s8...\n");

    int8_t X[TC_H * TC_W * TS_C];
    ts_fill_input_s8(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_weight_entry_t weights[2];
    _Alignas(int32_t) uint8_t weight_blob[sizeof(ts_weight) + sizeof(int32_t)];
    tigris_quant_param_t qp[2];
    int32_t qd[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_conv_plan_s8(tensors, shape_pool, index_pool, weights, weight_blob,
                           qp, qd, &header, &op, &plan);

    int8_t Y_full[TC_H * TC_W * TS_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden conv_s8 dispatch ok");

    int8_t Y_tile[2 * 2 * TS_C];
    tc_run_tile_s8(tigris_dispatch_kernel_s8, &plan, &op, X, 1, 5, 1, 5, 0, 0, Y_tile);
    tc_compare_s8(__func__, Y_tile, Y_full, TC_W, TS_C, 2, 2);
}

static void test_conv_2d_tile_edge_s8(void)
{
    printf("  test_conv_2d_tile_edge_s8...\n");

    int8_t X[TC_H * TC_W * TS_C];
    ts_fill_input_s8(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_weight_entry_t weights[2];
    _Alignas(int32_t) uint8_t weight_blob[sizeof(ts_weight) + sizeof(int32_t)];
    tigris_quant_param_t qp[2];
    int32_t qd[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_conv_plan_s8(tensors, shape_pool, index_pool, weights, weight_blob,
                           qp, qd, &header, &op, &plan);

    int8_t Y_full[TC_H * TC_W * TS_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden conv_s8 dispatch ok");

    int8_t Y_tile[2 * 2 * TS_C];
    tc_run_tile_s8(tigris_dispatch_kernel_s8, &plan, &op, X, 0, 3, 0, 3, 1, 1, Y_tile);
    tc_compare_s8(__func__, Y_tile, Y_full, TC_W, TS_C, 0, 0);
}

/* AveragePool: the pool variant called for by the task brief. Its width
 * bound is computed differently from conv/max_pool (a valid-sample window
 * count, not a per-tap continue), so it exercises the PL fix on a distinct
 * code path. Same interior/edge geometry as the conv tests above. */

static void test_avg_pool_2d_tile_matches_subregion_f32(void)
{
    printf("  test_avg_pool_2d_tile_matches_subregion_f32...\n");

    float X[TC_H * TC_W * TC_C];
    tc_fill_input_f32(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_avgpool_plan_f32(tensors, shape_pool, index_pool, &header, &op, &plan);

    float Y_full[TC_H * TC_W * TC_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden avg_pool dispatch ok");

    float Y_tile[2 * 2 * TC_C];
    tc_run_tile_f32(tigris_dispatch_kernel, &plan, &op, X, 1, 5, 1, 5, 0, 0, Y_tile);
    tc_compare_f32(__func__, Y_tile, Y_full, TC_W, TC_C, 2, 2);
}

static void test_avg_pool_2d_tile_edge_f32(void)
{
    printf("  test_avg_pool_2d_tile_edge_f32...\n");

    float X[TC_H * TC_W * TC_C];
    tc_fill_input_f32(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_avgpool_plan_f32(tensors, shape_pool, index_pool, &header, &op, &plan);

    float Y_full[TC_H * TC_W * TC_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden avg_pool dispatch ok");

    float Y_tile[2 * 2 * TC_C];
    tc_run_tile_f32(tigris_dispatch_kernel, &plan, &op, X, 0, 3, 0, 3, 1, 1, Y_tile);
    tc_compare_f32(__func__, Y_tile, Y_full, TC_W, TC_C, 0, 0);
}

static void test_avg_pool_2d_tile_matches_subregion_s8(void)
{
    printf("  test_avg_pool_2d_tile_matches_subregion_s8...\n");

    int8_t X[TC_H * TC_W * TS_C];
    ts_fill_input_s8(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_avgpool_plan_s8(tensors, shape_pool, index_pool, &header, &op, &plan);

    int8_t Y_full[TC_H * TC_W * TS_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden avg_pool_s8 dispatch ok");

    int8_t Y_tile[2 * 2 * TS_C];
    tc_run_tile_s8(tigris_dispatch_kernel_s8, &plan, &op, X, 1, 5, 1, 5, 0, 0, Y_tile);
    tc_compare_s8(__func__, Y_tile, Y_full, TC_W, TS_C, 2, 2);
}

static void test_avg_pool_2d_tile_edge_s8(void)
{
    printf("  test_avg_pool_2d_tile_edge_s8...\n");

    int8_t X[TC_H * TC_W * TS_C];
    ts_fill_input_s8(X);

    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[2];
    tigris_file_header_t header;
    tigris_op_t op;
    tigris_plan_t plan;
    tc_build_avgpool_plan_s8(tensors, shape_pool, index_pool, &header, &op, &plan);

    int8_t Y_full[TC_H * TC_W * TS_C];
    void *golden_ptrs[2] = { X, Y_full };
    tigris_mem_t golden_mem;
    memset(&golden_mem, 0, sizeof(golden_mem));
    golden_mem.tensor_ptrs = golden_ptrs;
    golden_mem.num_tensors = 2;

    int ret = tigris_dispatch_kernel_s8(&plan, &op, 0, &golden_mem, NULL);
    TEST_ASSERT_EQ(ret, 0, "golden avg_pool_s8 dispatch ok");

    int8_t Y_tile[2 * 2 * TS_C];
    tc_run_tile_s8(tigris_dispatch_kernel_s8, &plan, &op, X, 0, 3, 0, 3, 1, 1, Y_tile);
    tc_compare_s8(__func__, Y_tile, Y_full, TC_W, TS_C, 0, 0);
}

/* Task 7: the 2D standalone-tiled executor path.
 *
 * These tests drive a full single-stage Conv plan through the executor. The
 * same plan struct is run twice: once with a large fast arena (which keeps
 * total_io below budget, so the executor takes the untiled exec_stage_normal
 * path and produces the whole-op golden) and once with a small fast arena
 * (which forces the axis=HW standalone tiled path, exec_stage_tiled_2d). The
 * two full output tensors must be byte-identical. The tile geometry is chosen
 * so H and W are NOT divisible by the tile, exercising a partial last row, a
 * partial last column, AND the bottom-right corner tile (partial in both). */

#define TE_H   10
#define TE_W   10
#define TE_C    2   /* f32 in/out channels */
#define TE_OC   2
#define TE_SC   1   /* s8 uses a single channel to keep the requant trivial */
#define TE_K    3
#define TE_TH   4   /* emitted tile height: 10 % 4 == 2, partial last row */
#define TE_TW   4   /* emitted tile width:  10 % 4 == 2, partial last col  */

/* Fill an [1,H,W,C] f32 activation with a value that varies in every axis so a
 * tiling bug in any tile (interior, edge, or corner) shows up as a mismatch
 * rather than an accidental agreement. */
static void te_fill_input_f32(float *x)
{
    for (int h = 0; h < TE_H; h++)
        for (int w = 0; w < TE_W; w++)
            for (int c = 0; c < TE_C; c++)
                x[(h * TE_W + w) * TE_C + c] = (float)(h * 100 + w * 10 + c);
}

/* A [OC,K,K,IC] weight distinct per tap and channel, so a left/right or
 * top/bottom halo mistake yields a wrong sum, not a coincidental match. */
static void te_fill_weight_f32(float *w)
{
    for (int oc = 0; oc < TE_OC; oc++)
        for (int kh = 0; kh < TE_K; kh++)
            for (int kw = 0; kw < TE_K; kw++)
                for (int ic = 0; ic < TE_C; ic++)
                    w[((oc * TE_K + kh) * TE_K + kw) * TE_C + ic] =
                        (float)(oc * 1000 + kh * 100 + kw * 10 + ic) * 0.013f;
}

static void te_fill_input_s8(int8_t *x)
{
    for (int h = 0; h < TE_H; h++)
        for (int w = 0; w < TE_W; w++)
            x[h * TE_W + w] = (int8_t)(((h * 7 + w * 3) % 21) - 10);
}

/* Single-channel [1,3,3,1] weight, asymmetric in both axes. */
static const int8_t te_weight_s8[TE_K * TE_K] = {
    1,  2, -1,
    0,  1, -2,
   -1,  1,  1
};

/* Assemble a full executor-ready single-stage Conv plan (f32). All backing
 * storage is caller-owned; the executor only reads through the plan pointers. */
static void te_build_conv_plan_f32(
    tigris_file_header_t *header, tigris_tensor_t *tensors,
    int32_t *shape_pool, uint16_t *index_pool, tigris_op_t *ops,
    tigris_stage_t *stages, tigris_tile_plan_t *tile_plans,
    tigris_weight_entry_t *weights, float *weight_blob,
    const float *W, uint32_t w_bytes, const float *B, uint32_t b_bytes,
    uint16_t *model_io, tigris_plan_t *plan)
{
    int32_t x_shape[4] = {1, TE_H, TE_W, TE_C};
    int32_t y_shape[4] = {1, TE_H, TE_W, TE_OC};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 1;
    tensors[0].size_bytes = (uint32_t)(TE_H * TE_W * TE_C) * (uint32_t)sizeof(float);
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 1;
    tensors[1].size_bytes = (uint32_t)(TE_H * TE_W * TE_OC) * (uint32_t)sizeof(float);
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    index_pool[0] = 0;  /* op input   */
    index_pool[1] = 1;  /* op output  */
    index_pool[2] = 0;  /* stage input  */
    index_pool[3] = 1;  /* stage output */
    index_pool[4] = 0;  /* stage op    */

    weights[0].name_str = 0; weights[0].offset = 0; weights[0].size_bytes = w_bytes;
    weights[1].name_str = 0; weights[1].offset = w_bytes; weights[1].size_bytes = b_bytes;
    memcpy(weight_blob, W, w_bytes);
    memcpy((uint8_t *)weight_blob + w_bytes, B, b_bytes);

    memset(ops, 0, sizeof(*ops));
    ops[0].op_type = TIGRIS_OP_CONV;
    ops[0].num_inputs = 1; ops[0].num_outputs = 1;
    ops[0].inputs_off = 0; ops[0].outputs_off = 1;
    ops[0].spatial.kernel_h = TE_K; ops[0].spatial.kernel_w = TE_K;
    ops[0].spatial.stride_h = 1; ops[0].spatial.stride_w = 1;
    ops[0].spatial.dilation_h = 1; ops[0].spatial.dilation_w = 1;
    ops[0].spatial.pad_top = 1; ops[0].spatial.pad_bottom = 1;
    ops[0].spatial.pad_left = 1; ops[0].spatial.pad_right = 1;
    ops[0].spatial.group = 1;
    ops[0].weight_idx = 0; ops[0].bias_idx = 1;
    ops[0].fused_act = TIGRIS_ACT_NONE;

    memset(stages, 0, sizeof(*stages));
    stages[0].peak_bytes = tensors[0].size_bytes + tensors[1].size_bytes;
    stages[0].ops_off = 4; stages[0].ops_count = 1;
    stages[0].inputs_off = 2; stages[0].inputs_count = 1;
    stages[0].outputs_off = 3; stages[0].outputs_count = 1;
    stages[0].tile_plan_idx = 0;
    stages[0].chain_id = TIGRIS_NO_CHAIN;
    stages[0].chain_len = 0;

    memset(tile_plans, 0, sizeof(*tile_plans));
    tile_plans[0].tileable = 1;
    tile_plans[0].axis = TIGRIS_TILE_AXIS_HW;
    tile_plans[0].tile_height = TE_TH;
    tile_plans[0].num_tiles = 9;              /* ceil(10/4) * ceil(10/4) */
    tile_plans[0].halo = TE_K - 1;
    tile_plans[0].receptive_field = TE_K;
    tile_plans[0].original_height = TE_H;
    tile_plans[0]._reserved = (uint32_t)TE_TW; /* tile_width in low 16 bits */

    model_io[0] = 0; model_io[1] = 1;

    memset(header, 0, sizeof(*header));
    memcpy(header->magic, TIGRIS_MAGIC_BYTES, 4);
    header->version = TIGRIS_SCHEMA_VERSION_V5;
    header->num_tensors = 2; header->num_ops = 1; header->num_stages = 1;
    header->num_tile_plans = 1; header->num_weights = 2;
    header->num_model_inputs = 1; header->num_model_outputs = 1;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = ops;
    plan->stages = stages;
    plan->tile_plans = tile_plans;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
    plan->weight_entries = weights;
    plan->weight_blob = (const uint8_t *)weight_blob;
    plan->model_inputs = &model_io[0];
    plan->model_outputs = &model_io[1];
}

/* s8 counterpart: single channel, per-tensor identity requant (multiplier
 * 2^30, shift 1) so the int8 output equals the clamped accumulator, matching
 * the convention used by the Task 6 s8 tests above. */
static void te_build_conv_plan_s8(
    tigris_file_header_t *header, tigris_tensor_t *tensors,
    int32_t *shape_pool, uint16_t *index_pool, tigris_op_t *ops,
    tigris_stage_t *stages, tigris_tile_plan_t *tile_plans,
    tigris_weight_entry_t *weights, uint8_t *weight_blob,
    tigris_quant_param_t *qp, int32_t *qd,
    uint16_t *model_io, tigris_plan_t *plan)
{
    int32_t x_shape[4] = {1, TE_H, TE_W, TE_SC};
    int32_t y_shape[4] = {1, TE_H, TE_W, TE_SC};
    memcpy(shape_pool, x_shape, sizeof(x_shape));
    memcpy(shape_pool + 4, y_shape, sizeof(y_shape));

    memset(qp, 0, sizeof(*qp) * 2);
    qp[0].scale = 1.0f; qp[0].zero_point = 0; qp[0].num_channels = 1;
    qp[1].scale = 1.0f; qp[1].zero_point = 0; qp[1].num_channels = 1;
    qp[1].multiplier_off = 0; qp[1].shift_off = 1;
    qd[0] = 1073741824; /* 2^30 */
    qd[1] = 1;

    memset(tensors, 0, sizeof(*tensors) * 2);
    tensors[0].shape_off = 0; tensors[0].ndim = 4; tensors[0].dtype = 3;
    tensors[0].size_bytes = (uint32_t)(TE_H * TE_W * TE_SC);
    tensors[0].quant_param_idx = 0;
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[1].shape_off = 4; tensors[1].ndim = 4; tensors[1].dtype = 3;
    tensors[1].size_bytes = (uint32_t)(TE_H * TE_W * TE_SC);
    tensors[1].quant_param_idx = 1;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    index_pool[0] = 0;
    index_pool[1] = 1;
    index_pool[2] = 0;
    index_pool[3] = 1;
    index_pool[4] = 0;

    static const int32_t bias[1] = {0};
    weights[0].name_str = 0; weights[0].offset = 0;
    weights[0].size_bytes = sizeof(te_weight_s8);
    weights[1].name_str = 0; weights[1].offset = sizeof(te_weight_s8);
    weights[1].size_bytes = sizeof(bias);
    memcpy(weight_blob, te_weight_s8, sizeof(te_weight_s8));
    memcpy(weight_blob + sizeof(te_weight_s8), bias, sizeof(bias));

    memset(ops, 0, sizeof(*ops));
    ops[0].op_type = TIGRIS_OP_CONV;
    ops[0].num_inputs = 1; ops[0].num_outputs = 1;
    ops[0].inputs_off = 0; ops[0].outputs_off = 1;
    ops[0].spatial.kernel_h = TE_K; ops[0].spatial.kernel_w = TE_K;
    ops[0].spatial.stride_h = 1; ops[0].spatial.stride_w = 1;
    ops[0].spatial.dilation_h = 1; ops[0].spatial.dilation_w = 1;
    ops[0].spatial.pad_top = 1; ops[0].spatial.pad_bottom = 1;
    ops[0].spatial.pad_left = 1; ops[0].spatial.pad_right = 1;
    ops[0].spatial.group = 1;
    ops[0].weight_idx = 0; ops[0].bias_idx = 1;
    ops[0].act_min = -128; ops[0].act_max = 127;

    memset(stages, 0, sizeof(*stages));
    stages[0].peak_bytes = tensors[0].size_bytes + tensors[1].size_bytes;
    stages[0].ops_off = 4; stages[0].ops_count = 1;
    stages[0].inputs_off = 2; stages[0].inputs_count = 1;
    stages[0].outputs_off = 3; stages[0].outputs_count = 1;
    stages[0].tile_plan_idx = 0;
    stages[0].chain_id = TIGRIS_NO_CHAIN;
    stages[0].chain_len = 0;

    memset(tile_plans, 0, sizeof(*tile_plans));
    tile_plans[0].tileable = 1;
    tile_plans[0].axis = TIGRIS_TILE_AXIS_HW;
    tile_plans[0].tile_height = TE_TH;
    tile_plans[0].num_tiles = 9;
    tile_plans[0].halo = TE_K - 1;
    tile_plans[0].receptive_field = TE_K;
    tile_plans[0].original_height = TE_H;
    tile_plans[0]._reserved = (uint32_t)TE_TW;

    model_io[0] = 0; model_io[1] = 1;

    memset(header, 0, sizeof(*header));
    memcpy(header->magic, TIGRIS_MAGIC_BYTES, 4);
    header->version = TIGRIS_SCHEMA_VERSION_V5;
    header->num_tensors = 2; header->num_ops = 1; header->num_stages = 1;
    header->num_tile_plans = 1; header->num_weights = 2;
    header->num_model_inputs = 1; header->num_model_outputs = 1;
    header->num_quant_params = 2;

    memset(plan, 0, sizeof(*plan));
    plan->header = header;
    plan->tensors = tensors;
    plan->ops = ops;
    plan->stages = stages;
    plan->tile_plans = tile_plans;
    plan->index_pool = index_pool;
    plan->shape_pool = shape_pool;
    plan->strings = "";
    plan->weight_entries = weights;
    plan->weight_blob = weight_blob;
    plan->quant_params = qp;
    plan->quant_data = qd;
    plan->num_quant_params = 2;
    plan->model_inputs = &model_io[0];
    plan->model_outputs = &model_io[1];
}

/* Run the plan through the executor with a caller-chosen fast arena size, then
 * copy the whole output tensor back. The input lives in slow, exactly as an
 * embedded application stages a model input before tigris_run. */
static tigris_exec_error_t te_run(
    tigris_plan_t *plan, tigris_kernel_fn kernel,
    uint16_t in_tidx, const void *in_data, uint32_t in_bytes,
    uint16_t out_tidx, uint32_t out_bytes,
    uint32_t fast_size, void *out_buf)
{
    uint32_t slow_size = 256u * 1024u;
    void *fast = malloc(fast_size);
    void *slow = malloc(slow_size);
    uint16_t nt = plan->header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    if (!fast || !slow || !ptrs) {
        free(fast); free(slow); free(ptrs);
        return TIGRIS_EXEC_ERR_MEM;
    }

    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, nt, fast, fast_size, slow, slow_size);
    tigris_mem_alloc_slow(&mem, in_tidx, in_bytes);
    memcpy(ptrs[in_tidx], in_data, in_bytes);

    size_t ws_size = tigris_executor_workspace_required(plan);
    void *ws = malloc(ws_size);
    tigris_exec_error_t err = TIGRIS_EXEC_ERR_MEM;
    if (ws) {
        err = tigris_run_with_workspace_buffer(
            plan, &mem, kernel, NULL, NULL, ws, ws_size);
        if (err == TIGRIS_EXEC_OK && out_buf)
            memcpy(out_buf, ptrs[out_tidx], out_bytes);
    }

    free(ws); free(ptrs); free(slow); free(fast);
    return err;
}

/* Large enough that total_io stays under budget: the executor runs the whole
 * op untiled and produces the golden. */
#define TE_GOLDEN_FAST (64u * 1024u)
/* Small enough that total_io overflows it (forcing the tiled path) but a
 * single interior tile working set still fits. The f32 stage moves 1600 I/O
 * bytes; the single-channel s8 stage only 200, so it needs its own tighter
 * arena to overflow yet still admit one tile. */
#define TE_TILED_FAST     (1u * 1024u)
#define TE_TILED_FAST_S8  160u

static void test_exec_2d_tiled_matches_whole_f32(void)
{
    printf("  test_exec_2d_tiled_matches_whole_f32...\n");

    float X[TE_H * TE_W * TE_C];
    te_fill_input_f32(X);
    float W[TE_OC * TE_K * TE_K * TE_C];
    te_fill_weight_f32(W);
    float B[TE_OC] = {0.5f, -0.25f};

    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[5];
    tigris_op_t ops[1];
    tigris_stage_t stages[1];
    tigris_tile_plan_t tile_plans[1];
    tigris_weight_entry_t weights[2];
    float weight_blob[TE_OC * TE_K * TE_K * TE_C + TE_OC];
    uint16_t model_io[2];
    tigris_plan_t plan;
    te_build_conv_plan_f32(&header, tensors, shape_pool, index_pool, ops,
                            stages, tile_plans, weights, weight_blob,
                            W, sizeof(W), B, sizeof(B), model_io, &plan);

    uint32_t out_bytes = tensors[1].size_bytes;
    float golden[TE_H * TE_W * TE_OC];
    float tiled[TE_H * TE_W * TE_OC];

    tigris_exec_error_t ge = te_run(&plan, tigris_dispatch_kernel, 0, X,
                                     sizeof(X), 1, out_bytes,
                                     TE_GOLDEN_FAST, golden);
    TEST_ASSERT_EQ(ge, TIGRIS_EXEC_OK, "golden (untiled) run ok");

    tigris_exec_error_t te = te_run(&plan, tigris_dispatch_kernel, 0, X,
                                     sizeof(X), 1, out_bytes,
                                     TE_TILED_FAST, tiled);
    TEST_ASSERT_EQ(te, TIGRIS_EXEC_OK, "2D tiled run ok");

    TEST_ASSERT_EQ(memcmp(tiled, golden, out_bytes), 0,
                   "f32 2D tiled output byte-identical to whole op");
}

static void test_exec_2d_tiled_matches_whole_s8(void)
{
    printf("  test_exec_2d_tiled_matches_whole_s8...\n");

    int8_t X[TE_H * TE_W * TE_SC];
    te_fill_input_s8(X);

    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[5];
    tigris_op_t ops[1];
    tigris_stage_t stages[1];
    tigris_tile_plan_t tile_plans[1];
    tigris_weight_entry_t weights[2];
    _Alignas(int32_t) uint8_t weight_blob[sizeof(te_weight_s8) + sizeof(int32_t)];
    tigris_quant_param_t qp[2];
    int32_t qd[2];
    uint16_t model_io[2];
    tigris_plan_t plan;
    te_build_conv_plan_s8(&header, tensors, shape_pool, index_pool, ops,
                           stages, tile_plans, weights, weight_blob,
                           qp, qd, model_io, &plan);

    uint32_t out_bytes = tensors[1].size_bytes;
    int8_t golden[TE_H * TE_W * TE_SC];
    int8_t tiled[TE_H * TE_W * TE_SC];

    tigris_exec_error_t ge = te_run(&plan, tigris_dispatch_kernel_s8, 0, X,
                                     sizeof(X), 1, out_bytes,
                                     TE_GOLDEN_FAST, golden);
    TEST_ASSERT_EQ(ge, TIGRIS_EXEC_OK, "golden (untiled) s8 run ok");

    tigris_exec_error_t te = te_run(&plan, tigris_dispatch_kernel_s8, 0, X,
                                     sizeof(X), 1, out_bytes,
                                     TE_TILED_FAST_S8, tiled);
    TEST_ASSERT_EQ(te, TIGRIS_EXEC_OK, "2D tiled s8 run ok");

    TEST_ASSERT_EQ(memcmp(tiled, golden, out_bytes), 0,
                   "s8 2D tiled output byte-identical to whole op");
}

static void test_exec_2d_shape_exceeds_arena_fails(void)
{
    printf("  test_exec_2d_shape_exceeds_arena_fails...\n");

    float X[TE_H * TE_W * TE_C];
    te_fill_input_f32(X);
    float W[TE_OC * TE_K * TE_K * TE_C];
    te_fill_weight_f32(W);
    float B[TE_OC] = {0.5f, -0.25f};

    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    int32_t shape_pool[8];
    uint16_t index_pool[5];
    tigris_op_t ops[1];
    tigris_stage_t stages[1];
    tigris_tile_plan_t tile_plans[1];
    tigris_weight_entry_t weights[2];
    float weight_blob[TE_OC * TE_K * TE_K * TE_C + TE_OC];
    uint16_t model_io[2];
    tigris_plan_t plan;
    te_build_conv_plan_f32(&header, tensors, shape_pool, index_pool, ops,
                            stages, tile_plans, weights, weight_blob,
                            W, sizeof(W), B, sizeof(B), model_io, &plan);

    /* A fast arena below the emitted tile working set but still below total_io
     * (so the stage is routed to the tiled path, which must then fail closed
     * rather than overrun the arena). */
    float sink[TE_H * TE_W * TE_OC];
    tigris_exec_error_t err = te_run(&plan, tigris_dispatch_kernel, 0, X,
                                      sizeof(X), 1, tensors[1].size_bytes,
                                      200u, sink);
    TEST_ASSERT_EQ(err, TIGRIS_EXEC_ERR_TILE,
                   "tile working set exceeding fast arena fails closed");
}

/* Main */

int main(void)
{
    printf("TiGrIS 2D Tiling Runtime Tests\n\n");

    test_tile_ctx_has_width_fields();
    test_loader_accepts_hw_axis();
    test_load_tile_2d_roundtrip();
    test_spill_tile_2d_roundtrip();
    test_load_tile_2d_rejects_bad_bounds();
    test_spill_tile_2d_rejects_bad_bounds();

    test_conv_2d_tile_matches_subregion_f32();
    test_conv_2d_tile_edge_f32();
    test_conv_2d_tile_matches_subregion_s8();
    test_conv_2d_tile_edge_s8();
    test_avg_pool_2d_tile_matches_subregion_f32();
    test_avg_pool_2d_tile_edge_f32();
    test_avg_pool_2d_tile_matches_subregion_s8();
    test_avg_pool_2d_tile_edge_s8();

    test_exec_2d_tiled_matches_whole_f32();
    test_exec_2d_tiled_matches_whole_s8();
    test_exec_2d_shape_exceeds_arena_fails();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

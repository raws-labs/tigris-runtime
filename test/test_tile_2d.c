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
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

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

static void test_load_tile_2d_rejects_bad_bounds(void)
{
    printf("  test_load_tile_2d_rejects_bad_bounds...\n");

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
    mem.tensor_ptrs[T2D_TIDX] = slow_buf;

    /* w1 > W must be rejected before any bytes move. */
    tigris_mem_error_t e = tigris_mem_load_tile_2d(&mem, &plan, T2D_TIDX,
                                                     2, 6, 3, T2D_W + 1);
    TEST_ASSERT_EQ(e, TIGRIS_MEM_ERR_BAD_INDEX, "out-of-range w1 rejected");
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

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

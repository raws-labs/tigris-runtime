/**
 * @file test_tile_2d.c
 * @brief Runtime-side checks for Phase 1.3b (2D spatial tiling), Task 4:
 *        the TIGRIS_TILE_AXIS_HW constant, the width fields on
 *        tigris_tile_ctx_t, and the loader accepting a tileable tile plan
 *        whose axis is HW instead of only HEIGHT_OR_LENGTH.
 *
 * The HW-axis fixture is a real plan emitted by the compiler's 2D tile
 * solver (feature/2d-tiling-core): a [1,16,16,16] float32 3x3 stride-1
 * pad-1 Conv compiled at a 2K fast-memory budget, small enough that even a
 * single-row height tile still overflows the budget, forcing the compiler
 * to fall back to a 2x2 HW (height+width) core tile. The executor does not
 * yet dispatch HW-axis tile plans (that lands in Task 7); this test only
 * exercises the loader's acceptance of the axis value.
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

/* Main */

int main(void)
{
    printf("TiGrIS 2D Tiling Runtime Tests\n\n");

    test_tile_ctx_has_width_fields();
    test_loader_accepts_hw_axis();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

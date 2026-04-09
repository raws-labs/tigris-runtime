/**
 * @file test_loader.c
 * @brief Native POSIX tests for the tigris plan loader.
 *
 * Usage: ./test_loader [fixture1.tgrs] [fixture2.tgrs] ...
 *
 * If no arguments given, runs built-in error tests only.
 * When fixture files are provided, loads each and verifies the parsed plan.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"

/* Test infrastructure */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
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

/* Error tests (no fixtures needed) */

static void test_null_args(void)
{
    printf("  test_null_args...\n");
    tigris_plan_t plan;
    uint8_t buf[48] = {0};

    TEST_ASSERT_EQ(tigris_plan_load(NULL, 48, &plan), TIGRIS_ERR_NULL, "null buf");
    TEST_ASSERT_EQ(tigris_plan_load(buf, 48, NULL), TIGRIS_ERR_NULL, "null plan");
}

static void test_too_small(void)
{
    printf("  test_too_small...\n");
    tigris_plan_t plan;
    uint8_t buf[4] = {'T', 'G', 'R', 'S'};

    TEST_ASSERT_EQ(tigris_plan_load(buf, 4, &plan), TIGRIS_ERR_TOO_SMALL, "tiny buf");
}

static void test_bad_magic(void)
{
    printf("  test_bad_magic...\n");
    tigris_plan_t plan;
    uint8_t buf[48];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, "XXXX", 4);

    TEST_ASSERT_EQ(tigris_plan_load(buf, 48, &plan), TIGRIS_ERR_BAD_MAGIC, "bad magic");
}

static void test_bad_version(void)
{
    printf("  test_bad_version...\n");
    tigris_plan_t plan;
    uint8_t buf[48];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, "TGRS", 4);
    /* version at offset 4, set to 99 (LE) */
    buf[4] = 99;
    buf[5] = 0;
    /* file_size at offset 8 */
    uint32_t fs = 48;
    memcpy(buf + 8, &fs, 4);

    TEST_ASSERT_EQ(tigris_plan_load(buf, 48, &plan), TIGRIS_ERR_BAD_VERSION, "bad version");
}

static void test_size_mismatch(void)
{
    printf("  test_size_mismatch...\n");
    tigris_plan_t plan;
    uint8_t buf[48];
    memset(buf, 0, sizeof(buf));
    memcpy(buf, "TGRS", 4);
    buf[4] = TIGRIS_SCHEMA_VERSION;
    /* file_size = 999 but buf_len = 48 */
    uint32_t fs = 999;
    memcpy(buf + 8, &fs, 4);

    TEST_ASSERT_EQ(tigris_plan_load(buf, 48, &plan), TIGRIS_ERR_BAD_SIZE, "size mismatch");
}

static void test_error_strings(void)
{
    printf("  test_error_strings...\n");
    TEST_ASSERT(strlen(tigris_error_str(TIGRIS_OK)) > 0, "OK string");
    TEST_ASSERT(strlen(tigris_error_str(TIGRIS_ERR_BAD_MAGIC)) > 0, "magic string");
    TEST_ASSERT(strlen(tigris_error_str((tigris_error_t)999)) > 0, "unknown string");
}

/* Struct size validation */

static void test_struct_sizes(void)
{
    printf("  test_struct_sizes...\n");
    TEST_ASSERT_EQ(sizeof(tigris_file_header_t), 48, "header size");
    TEST_ASSERT_EQ(sizeof(tigris_section_entry_t), 8, "section entry size");
    TEST_ASSERT_EQ(sizeof(tigris_tensor_t), 16, "tensor size");
    TEST_ASSERT_EQ(sizeof(tigris_op_t), 32, "op size");
    TEST_ASSERT_EQ(sizeof(tigris_stage_t), 28, "stage size");
    TEST_ASSERT_EQ(sizeof(tigris_tile_plan_t), 24, "tile plan size");
    TEST_ASSERT_EQ(sizeof(tigris_spatial_attrs_t), 12, "spatial attrs size");
    TEST_ASSERT_EQ(sizeof(tigris_weight_entry_t), 12, "weight entry size");
    TEST_ASSERT_EQ(sizeof(tigris_quant_param_t), 16, "quant param size");
}

/* Fixture loading tests */

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

static void test_fixture(const char *path)
{
    printf("  test_fixture(%s)...\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load fixture\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "load succeeds");

    if (err != TIGRIS_OK) {
        fprintf(stderr, "  Load error: %s\n", tigris_error_str(err));
        free(buf);
        return;
    }

    /* Header checks */
    TEST_ASSERT(memcmp(plan.header->magic, "TGRS", 4) == 0, "magic");
    TEST_ASSERT_EQ(plan.header->version, TIGRIS_SCHEMA_VERSION, "version");
    TEST_ASSERT_EQ(plan.header->file_size, buf_len, "file_size");
    TEST_ASSERT(plan.header->num_ops > 0, "has ops");
    TEST_ASSERT(plan.header->num_tensors > 0, "has tensors");

    /* Model name is non-empty */
    const char *name = tigris_model_name(&plan);
    TEST_ASSERT(name != NULL, "model name not null");
    TEST_ASSERT(strlen(name) > 0, "model name non-empty");

    /* Tensors are accessible */
    for (uint16_t i = 0; i < plan.header->num_tensors; i++) {
        const tigris_tensor_t *t = &plan.tensors[i];
        const char *tname = tigris_tensor_name(&plan, t);
        TEST_ASSERT(strlen(tname) > 0, "tensor has name");
        TEST_ASSERT(t->size_bytes > 0, "tensor has size");
        TEST_ASSERT(t->ndim > 0, "tensor has dims");

        const int32_t *shape = tigris_tensor_shape(&plan, t);
        for (uint8_t d = 0; d < t->ndim; d++) {
            TEST_ASSERT(shape[d] > 0, "shape dim positive");
        }
    }

    /* Ops are accessible */
    for (uint16_t i = 0; i < plan.header->num_ops; i++) {
        const tigris_op_t *op = &plan.ops[i];
        const char *oname = tigris_op_name(&plan, op);
        TEST_ASSERT(strlen(oname) > 0, "op has name");
        TEST_ASSERT(op->op_type != 0, "op has type");
        /* Check that input/output indices are valid tensor indices */
        const uint16_t *inputs = tigris_op_inputs(&plan, op);
        for (uint8_t j = 0; j < op->num_inputs; j++) {
            TEST_ASSERT(inputs[j] < plan.header->num_tensors, "input idx valid");
        }
        const uint16_t *outputs = tigris_op_outputs(&plan, op);
        for (uint8_t j = 0; j < op->num_outputs; j++) {
            TEST_ASSERT(outputs[j] < plan.header->num_tensors, "output idx valid");
        }
    }

    /* Stages (if present) */
    for (uint16_t i = 0; i < plan.header->num_stages; i++) {
        const tigris_stage_t *stage = &plan.stages[i];
        /* All op indices in stage should be valid */
        const uint16_t *sops = tigris_stage_ops(&plan, stage);
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            TEST_ASSERT(sops[j] < plan.header->num_ops, "stage op idx valid");
        }

        /* Tile plan reference (if any) */
        const tigris_tile_plan_t *tp = tigris_stage_tile_plan(&plan, stage);
        if (tp != NULL && tp->tileable) {
            TEST_ASSERT(tp->original_height > 0, "tile plan has height");
        }
    }

    /* Model I/O */
    TEST_ASSERT(plan.header->num_model_inputs > 0, "has model inputs");
    TEST_ASSERT(plan.header->num_model_outputs > 0, "has model outputs");
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        TEST_ASSERT(plan.model_inputs[i] < plan.header->num_tensors, "model input idx valid");
    }
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        TEST_ASSERT(plan.model_outputs[i] < plan.header->num_tensors, "model output idx valid");
    }

    /* Weights (if present) */
    if (plan.header->num_weights > 0) {
        TEST_ASSERT(plan.weight_entries != NULL, "weight entries not null");
        /* weight_blob is NULL for compressed plans (weight_blocks present instead) */
        if (plan.weight_blocks == NULL)
            TEST_ASSERT(plan.weight_blob != NULL, "weight blob not null");
        else
            TEST_ASSERT(plan.weight_blocks_data != NULL, "weight blocks data not null");
        for (uint16_t i = 0; i < plan.header->num_weights; i++) {
            const tigris_weight_entry_t *w = &plan.weight_entries[i];
            const char *wname = tigris_weight_name(&plan, w);
            TEST_ASSERT(strlen(wname) > 0, "weight has name");
            TEST_ASSERT(w->size_bytes > 0, "weight has size");
        }

        /* Verify op weight/bias references are valid */
        for (uint16_t i = 0; i < plan.header->num_ops; i++) {
            const tigris_op_t *op = &plan.ops[i];
            if (op->weight_idx != TIGRIS_NO_WEIGHT) {
                TEST_ASSERT(op->weight_idx < plan.header->num_weights, "weight_idx valid");
            }
            if (op->bias_idx != TIGRIS_NO_WEIGHT) {
                TEST_ASSERT(op->bias_idx < plan.header->num_weights, "bias_idx valid");
            }
        }
    }

    free(buf);
}

/* Main */

int main(int argc, char *argv[])
{
    printf("TiGrIS Plan Loader Tests\n\n");

    printf("Error handling tests:\n");
    test_null_args();
    test_too_small();
    test_bad_magic();
    test_bad_version();
    test_size_mismatch();
    test_error_strings();

    printf("\nStruct size tests:\n");
    test_struct_sizes();

    if (argc > 1) {
        printf("\nFixture tests:\n");
        for (int i = 1; i < argc; i++) {
            test_fixture(argv[i]);
        }
    } else {
        printf("\n(No fixture files provided - pass .tgrs paths as arguments)\n");
    }

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

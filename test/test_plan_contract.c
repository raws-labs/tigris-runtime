/**
 * @file test_plan_contract.c
 * @brief Binary format contract tests.
 *
 * Verify that .tgrs fixture files have known field values.
 * The same values are checked by test_plan_contract.py in tigris/.
 * If either side changes the format, this contract catches the mismatch.
 *
 * Usage: ./test_plan_contract <fixtures_dir>
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

/* File loader */

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

    uint8_t *buf = (uint8_t *)malloc((size_t)len);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if ((long)fread(buf, 1, (size_t)len, f) != len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (uint32_t)len;
    return buf;
}

/* Contract: linear_3op */

static void test_linear_3op(const char *fixtures_dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/linear_3op.tgrs", fixtures_dir);
    printf("  test_linear_3op(%s)...\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load linear_3op.tgrs\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "load succeeds");
    if (err != TIGRIS_OK) { free(buf); return; }

    TEST_ASSERT(memcmp(plan.header->magic, TIGRIS_MAGIC_BYTES, 4) == 0, "magic");
    TEST_ASSERT_EQ(plan.header->version, 1, "version");
    TEST_ASSERT_EQ(plan.header->file_size, buf_len, "file_size matches");
    TEST_ASSERT_EQ(plan.header->num_tensors, 4, "num_tensors");
    TEST_ASSERT_EQ(plan.header->num_ops, 3, "num_ops");
    TEST_ASSERT_EQ(plan.header->num_stages, 1, "num_stages");
    TEST_ASSERT_EQ(plan.header->num_tile_plans, 0, "num_tile_plans");
    TEST_ASSERT_EQ(plan.header->budget, 4096, "budget");
    TEST_ASSERT(plan.header->peak > 0, "peak > 0");
    TEST_ASSERT_EQ(plan.header->num_model_inputs, 1, "num_model_inputs");
    TEST_ASSERT_EQ(plan.header->num_model_outputs, 1, "num_model_outputs");
    TEST_ASSERT_EQ(plan.header->num_weights, 2, "num_weights");

    /* Model name should be "linear_3op" */
    const char *name = tigris_model_name(&plan);
    TEST_ASSERT(name != NULL && strcmp(name, "linear_3op") == 0, "model name");

    free(buf);
}

/* Contract: conv_relu_chain */

static void test_conv_relu_chain(const char *fixtures_dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/conv_relu_chain.tgrs", fixtures_dir);
    printf("  test_conv_relu_chain(%s)...\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load conv_relu_chain.tgrs\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "load succeeds");
    if (err != TIGRIS_OK) { free(buf); return; }

    TEST_ASSERT(memcmp(plan.header->magic, TIGRIS_MAGIC_BYTES, 4) == 0, "magic");
    TEST_ASSERT_EQ(plan.header->version, 1, "version");
    TEST_ASSERT_EQ(plan.header->file_size, buf_len, "file_size matches");
    TEST_ASSERT_EQ(plan.header->num_tensors, 3, "num_tensors");
    TEST_ASSERT_EQ(plan.header->num_ops, 2, "num_ops (relu fused)");
    TEST_ASSERT_EQ(plan.header->num_stages, 1, "num_stages");
    TEST_ASSERT_EQ(plan.header->num_tile_plans, 0, "num_tile_plans");
    TEST_ASSERT_EQ(plan.header->budget, 32768, "budget");
    TEST_ASSERT(plan.header->peak > 0, "peak > 0");
    TEST_ASSERT_EQ(plan.header->num_model_inputs, 1, "num_model_inputs");
    TEST_ASSERT_EQ(plan.header->num_model_outputs, 1, "num_model_outputs");
    TEST_ASSERT_EQ(plan.header->num_weights, 4, "num_weights");

    const char *name = tigris_model_name(&plan);
    TEST_ASSERT(name != NULL && strcmp(name, "conv_relu_chain") == 0, "model name");

    /* Verify first op is Conv with 3x3 kernel */
    TEST_ASSERT_EQ(plan.ops[0].op_type, TIGRIS_OP_CONV, "op0 is Conv");
    TEST_ASSERT_EQ(plan.ops[0].spatial.kernel_h, 3, "op0 kernel_h");
    TEST_ASSERT_EQ(plan.ops[0].spatial.kernel_w, 3, "op0 kernel_w");
    TEST_ASSERT_EQ(plan.ops[0].spatial.stride_h, 1, "op0 stride_h");
    TEST_ASSERT_EQ(plan.ops[0].spatial.stride_w, 1, "op0 stride_w");

    /* First conv should have fused relu */
    TEST_ASSERT_EQ(plan.ops[0].fused_act, TIGRIS_ACT_RELU, "op0 fused relu");

    free(buf);
}

/* Contract: ds_cnn */

static void test_ds_cnn(const char *fixtures_dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/ds_cnn.tgrs", fixtures_dir);
    printf("  test_ds_cnn(%s)...\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load ds_cnn.tgrs\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "load succeeds");
    if (err != TIGRIS_OK) { free(buf); return; }

    TEST_ASSERT(memcmp(plan.header->magic, TIGRIS_MAGIC_BYTES, 4) == 0, "magic");
    TEST_ASSERT_EQ(plan.header->version, 1, "version");
    TEST_ASSERT_EQ(plan.header->file_size, buf_len, "file_size matches");
    TEST_ASSERT_EQ(plan.header->num_tensors, 13, "num_tensors");
    TEST_ASSERT_EQ(plan.header->num_ops, 12, "num_ops");
    TEST_ASSERT_EQ(plan.header->num_stages, 1, "num_stages");
    TEST_ASSERT_EQ(plan.header->num_tile_plans, 0, "num_tile_plans");
    TEST_ASSERT_EQ(plan.header->budget, 262144, "budget");
    TEST_ASSERT(plan.header->peak > 0, "peak > 0");
    TEST_ASSERT_EQ(plan.header->num_model_inputs, 1, "num_model_inputs");
    TEST_ASSERT_EQ(plan.header->num_model_outputs, 1, "num_model_outputs");
    TEST_ASSERT_EQ(plan.header->num_weights, 20, "num_weights");

    const char *name = tigris_model_name(&plan);
    TEST_ASSERT(name != NULL && strcmp(name, "ds_cnn") == 0, "model name");

    /* Verify plan has depthwise conv ops */
    int dw_count = 0;
    for (uint16_t i = 0; i < plan.header->num_ops; i++) {
        if (plan.ops[i].op_type == TIGRIS_OP_DEPTHWISE)
            dw_count++;
    }
    TEST_ASSERT(dw_count > 0, "has depthwise conv ops");

    /* Verify plan has FC (fully connected) op */
    int fc_count = 0;
    for (uint16_t i = 0; i < plan.header->num_ops; i++) {
        if (plan.ops[i].op_type == TIGRIS_OP_FULLY_CONN)
            fc_count++;
    }
    TEST_ASSERT_EQ(fc_count, 1, "has one FC op");

    free(buf);
}

/* Struct size contract */

static void test_struct_sizes(void)
{
    printf("  test_struct_sizes...\n");
    TEST_ASSERT_EQ(sizeof(tigris_file_header_t), 48, "header = 48 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_section_entry_t), 8, "section entry = 8 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_tensor_t), 16, "tensor = 16 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_op_t), 32, "op = 32 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_stage_t), 28, "stage = 28 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_tile_plan_t), 24, "tile plan = 24 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_spatial_attrs_t), 12, "spatial attrs = 12 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_weight_entry_t), 12, "weight entry = 12 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_quant_param_t), 16, "quant param = 16 bytes");
    TEST_ASSERT_EQ(sizeof(tigris_weight_block_t), 20, "weight block = 20 bytes");
}

/* Main */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <fixtures_dir>\n", argv[0]);
        return 1;
    }

    const char *fixtures_dir = argv[1];

    printf("TiGrIS Plan Contract Tests\n\n");

    printf("Struct size contract:\n");
    test_struct_sizes();

    printf("\nlinear_3op contract:\n");
    test_linear_3op(fixtures_dir);

    printf("\nconv_relu_chain contract:\n");
    test_conv_relu_chain(fixtures_dir);

    printf("\nds_cnn contract:\n");
    test_ds_cnn(fixtures_dir);

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

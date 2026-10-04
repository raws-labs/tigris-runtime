/**
 * @file test_loader.c
 * @brief Native POSIX tests for the tigris plan loader.
 *
 * Usage: ./test_loader [fixture1.tgrs] [fixture2.tgrs] ...
 *
 * If no arguments given, runs built-in error tests only.
 * When fixture files are provided, loads each and verifies the parsed plan.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "movement_golden.h"
#include "bool_golden.h"
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

    memset(&plan, 0xA5, sizeof(plan));
    TEST_ASSERT_EQ(tigris_plan_load(NULL, 48, &plan), TIGRIS_ERR_NULL, "null buf");
    TEST_ASSERT(plan.header == NULL && plan.ops == NULL,
                "null buffer clears output plan");
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
    TEST_ASSERT(strlen(tigris_error_str(TIGRIS_ERR_BAD_TENSOR)) > 0,
                "tensor contract string");
    TEST_ASSERT(strlen(tigris_error_str(TIGRIS_ERR_BAD_OPERATOR)) > 0,
                "operator contract string");
    TEST_ASSERT(strlen(tigris_error_str((tigris_error_t)999)) > 0, "unknown string");
}

/* Minimal self-contained plans for malformed-input tests. */

#define MINIMAL_PLAN_SIZE 100u

static void build_minimal_plan(uint8_t *buf)
{
    memset(buf, 0, MINIMAL_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = MINIMAL_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    const uint32_t required[] = {
        TIGRIS_SEC_TENSORS,
        TIGRIS_SEC_OPS,
        TIGRIS_SEC_INDEX_POOL,
        TIGRIS_SEC_SHAPE_POOL,
        TIGRIS_SEC_STRINGS,
    };
    for (uint32_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        dir[i].type = required[i];
        dir[i].offset = MINIMAL_PLAN_SIZE - 1u;
    }
    buf[MINIMAL_PLAN_SIZE - 1u] = '\0';
    /* dir[5] remains the zero-valued sentinel. */
}

static void test_section_directory_guards(void)
{
    printf("  test_section_directory_guards...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;

    build_minimal_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "minimal plan loads");

    build_minimal_plan(buf);
    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    dir[5].type = TIGRIS_SEC_WEIGHTS;
    dir[5].offset = MINIMAL_PLAN_SIZE;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "missing directory sentinel");

    build_minimal_plan(buf);
    dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    dir[1].type = TIGRIS_SEC_TENSORS;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "duplicate section type");

    build_minimal_plan(buf);
    dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    dir[0].type = TIGRIS_SEC_OPS;
    dir[1].type = TIGRIS_SEC_TENSORS;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "unordered section directory");

    build_minimal_plan(buf);
    dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    dir[0].offset = sizeof(tigris_file_header_t);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "section overlaps directory");
}

static void test_pool_alignment_guards(void)
{
    printf("  test_pool_alignment_guards...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_file_header_t *hdr;

    /* The minimal plan deliberately puts all empty pools at byte 99.  They
     * are valid while unused, but a count must not make an unaligned typed
     * pool dereference reachable. */
    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_model_inputs = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "unaligned index pool rejected");

    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_tensors = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "unaligned shape pool rejected");
}

static void test_v2_plan_header_is_still_accepted(void)
{
    printf("  test_v2_plan_header_is_still_accepted...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;

    build_minimal_plan(buf);
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "v2 minimal plan loads");
}

#define V2_QUANT_PLAN_SIZE 136u

static void build_v2_quant_plan(uint8_t *buf)
{
    memset(buf, 0, V2_QUANT_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION_V2;
    hdr->file_size = V2_QUANT_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_quant_params = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    const uint32_t required[] = {
        TIGRIS_SEC_TENSORS,
        TIGRIS_SEC_OPS,
        TIGRIS_SEC_INDEX_POOL,
        TIGRIS_SEC_SHAPE_POOL,
        TIGRIS_SEC_STRINGS,
    };
    for (uint32_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        dir[i].type = required[i];
        dir[i].offset = 104u;
    }
    dir[5].type = TIGRIS_SEC_QUANT_PARAMS;
    dir[5].offset = 108u;

    uint16_t *quant_header = (uint16_t *)(buf + 108u);
    quant_header[0] = 1;  /* num params */
    quant_header[1] = 2;  /* v2 quant-data length */
    tigris_quant_param_t *qp = (tigris_quant_param_t *)(buf + 112u);
    qp->scale = 0.25f;
    qp->num_channels = 1;
    qp->multiplier_off = 0;
    qp->shift_off = 1;
    int32_t *quant_data = (int32_t *)(buf + 128u);
    quant_data[0] = 123;
    quant_data[1] = -7;
}

static void test_v2_quant_plan_is_still_accepted(void)
{
    printf("  test_v2_quant_plan_is_still_accepted...\n");
    _Alignas(4) uint8_t buf[V2_QUANT_PLAN_SIZE];
    tigris_plan_t plan;

    build_v2_quant_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "v2 quantized plan loads");
    TEST_ASSERT_EQ(plan.num_quant_params, 1, "v2 quant param count");
    TEST_ASSERT_EQ(plan.quant_data[0], 123, "v2 multiplier data");
    TEST_ASSERT_EQ(plan.quant_data[1], -7, "v2 shift data");

    build_v2_quant_plan(buf);
    ((tigris_quant_param_t *)(buf + 112u))->scale = 0.0f;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR, "non-positive quant scale rejected");
}

static void test_counted_sections_required(void)
{
    printf("  test_counted_sections_required...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_file_header_t *hdr;

    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_stages = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_MISSING_SEC, "counted stages require section");

    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_tile_plans = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_MISSING_SEC, "counted tile plans require section");

    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_weights = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_MISSING_SEC, "counted weights require section");

    build_minimal_plan(buf);
    hdr = (tigris_file_header_t *)buf;
    hdr->num_quant_params = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_MISSING_SEC, "counted quant params require section");
}

#define STAGED_PLAN_SIZE 704u
#define STAGED_TENSORS_OFF 112u
#define STAGED_OPS_OFF   128u
#define STAGED_STAGES_OFF 480u
#define STAGED_INDEX_OFF 544u
#define STAGED_SHAPES_OFF 640u
#define STAGED_STRINGS_OFF 656u
#define STAGED_WEIGHTS_OFF 660u

static void build_staged_plan(uint8_t *buf)
{
    memset(buf, 0, STAGED_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = STAGED_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 1;
    hdr->num_ops = 9;
    hdr->num_stages = 2;
    hdr->num_weights = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, STAGED_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, STAGED_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, STAGED_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, STAGED_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, STAGED_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, STAGED_STRINGS_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, STAGED_WEIGHTS_OFF};

    tigris_tensor_t *tensor =
        (tigris_tensor_t *)(buf + STAGED_TENSORS_OFF);
    tensor->size_bytes = sizeof(float);
    tensor->ndim = 4;
    tensor->dtype = 1;
    tensor->quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    int32_t *shape = (int32_t *)(buf + STAGED_SHAPES_OFF);
    shape[0] = 1;
    shape[1] = 1;
    shape[2] = 1;
    shape[3] = 1;

    tigris_weight_entry_t *weight =
        (tigris_weight_entry_t *)(buf + STAGED_WEIGHTS_OFF);
    weight->size_bytes = sizeof(float);

    tigris_op_t *ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    uint16_t *indices = (uint16_t *)(buf + STAGED_INDEX_OFF);
    for (uint16_t i = 0; i < hdr->num_ops; i++) {
        ops[i].op_type = TIGRIS_OP_CONV;
        ops[i].num_inputs = 1;
        ops[i].num_outputs = 1;
        ops[i].stage = i < 4 ? 0 : 1;
        ops[i].inputs_off = 2u * i;
        ops[i].outputs_off = 2u * i + 1u;
        ops[i].spatial.kernel_h = 1;
        ops[i].spatial.kernel_w = 1;
        ops[i].spatial.stride_h = 1;
        ops[i].spatial.stride_w = 1;
        ops[i].spatial.dilation_h = 1;
        ops[i].spatial.dilation_w = 1;
        ops[i].spatial.group = 1;
        ops[i].weight_idx = 0;
        ops[i].bias_idx = TIGRIS_NO_WEIGHT;
        indices[2u * i] = 0;
        indices[2u * i + 1u] = 0;
    }
    for (uint16_t i = 0; i < hdr->num_ops; i++)
        indices[18u + i] = i;

    tigris_stage_t *stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stages[0].chain_id = TIGRIS_NO_CHAIN;
    stages[0].ops_off = 18;
    stages[0].ops_count = 4;
    stages[0].inputs_off = 27;
    stages[0].inputs_count = 1;
    stages[0].outputs_off = 28;
    stages[0].outputs_count = 1;
    stages[1].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stages[1].chain_id = TIGRIS_NO_CHAIN;
    stages[1].ops_off = 22;
    stages[1].ops_count = 5;
    stages[1].inputs_off = 29;
    stages[1].inputs_count = 1;
    stages[1].outputs_off = 30;
    stages[1].outputs_count = 1;
}

static void test_chain_limit_guards(void)
{
    printf("  test_chain_limit_guards...\n");
    _Alignas(4) uint8_t buf[STAGED_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_stage_t *stages;

    build_staged_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "base staged plan loads");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].inputs_count = 17;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "stage input limit");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].outputs_count = 17;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "stage output limit");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = 0;
    stages[0].chain_len = 17;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "chain length limit");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = 1;
    stages[0].chain_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "chain end within stage table");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = TIGRIS_NO_CHAIN;
    stages[0].chain_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "chain requires valid head");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = 0;
    stages[0].chain_len = 2;
    stages[0].chain_tile_h = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "chain requires every member");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = 0;
    stages[0].chain_len = 2;
    stages[0].chain_tile_h = 1;
    stages[1].chain_id = 0;
    stages[1].chain_len = 2;
    stages[0].ops_count = 8;
    stages[1].ops_off = 26;
    stages[1].ops_count = 1;
    tigris_op_t *ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    for (uint16_t i = 0; i < 8; i++)
        ops[i].stage = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "eight chain spatial ops fit fixed metadata");

    ops[7].op_type = TIGRIS_OP_AVG_POOL;
    ops[7].weight_idx = TIGRIS_NO_WEIGHT;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "pool counts as supported chain spatial metadata");

    stages[0].ops_count = 9;
    stages[1].ops_off = 27;
    stages[1].ops_count = 0;
    ops[8].stage = 0;
    ops[8].op_type = TIGRIS_OP_MAX_POOL;
    ops[8].weight_idx = TIGRIS_NO_WEIGHT;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "ninth chain spatial op rejected");

    build_staged_plan(buf);
    stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].chain_id = 0;
    stages[0].chain_len = 2;
    stages[0].chain_tile_h = 1;
    stages[1].chain_id = 0;
    stages[1].chain_len = 2;
    ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    ops[0].op_type = TIGRIS_OP_GLOBAL_AVG;
    ops[0].weight_idx = TIGRIS_NO_WEIGHT;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "global reduction cannot enter height chain");
}

static void test_stage_schedule_guards(void)
{
    printf("  test_stage_schedule_guards...\n");
    _Alignas(4) uint8_t buf[STAGED_PLAN_SIZE];
    tigris_plan_t plan;

    build_staged_plan(buf);
    uint16_t *indices = (uint16_t *)(buf + STAGED_INDEX_OFF);
    indices[4] = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "duplicated stage operation");

    build_staged_plan(buf);
    tigris_stage_t *stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[1].ops_count = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "missing scheduled operation");
}

static uint32_t align4(uint32_t value)
{
    return (value + 3u) & ~3u;
}

static uint8_t *build_many_stage_plan(uint16_t stage_count, uint32_t *out_size)
{
    const uint32_t tensors_off = 104u;
    const uint32_t ops_off = tensors_off + sizeof(tigris_tensor_t);
    const uint32_t stages_off = align4(
        ops_off + (uint32_t)stage_count * sizeof(tigris_op_t));
    const uint32_t index_off = stages_off +
        (uint32_t)stage_count * sizeof(tigris_stage_t);
    const uint32_t index_count = (uint32_t)stage_count * 3u;
    const uint32_t shapes_off = align4(
        index_off + index_count * sizeof(uint16_t));
    const uint32_t strings_off = shapes_off + sizeof(int32_t);
    const uint32_t plan_size = strings_off + 1u;
    uint8_t *buf = calloc(1, plan_size);
    if (!buf)
        return NULL;

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = plan_size;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 1;
    hdr->num_ops = stage_count;
    hdr->num_stages = stage_count;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, tensors_off};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, ops_off};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, stages_off};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, index_off};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, shapes_off};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, strings_off};

    tigris_tensor_t *tensor = (tigris_tensor_t *)(buf + tensors_off);
    tensor->size_bytes = sizeof(float);
    tensor->ndim = 1;
    tensor->dtype = 1;
    tensor->quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    *(int32_t *)(buf + shapes_off) = 1;

    tigris_op_t *ops = (tigris_op_t *)(buf + ops_off);
    tigris_stage_t *stages = (tigris_stage_t *)(buf + stages_off);
    uint16_t *indices = (uint16_t *)(buf + index_off);
    for (uint16_t i = 0; i < stage_count; i++) {
        ops[i].op_type = TIGRIS_OP_RELU;
        ops[i].num_inputs = 1;
        ops[i].num_outputs = 1;
        ops[i].stage = (uint8_t)i;
        ops[i].inputs_off = 2u * i;
        ops[i].outputs_off = 2u * i + 1u;
        ops[i].weight_idx = TIGRIS_NO_WEIGHT;
        ops[i].bias_idx = TIGRIS_NO_WEIGHT;
        stages[i].ops_off = (uint16_t)(2u * stage_count + i);
        stages[i].ops_count = 1;
        stages[i].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        stages[i].chain_id = TIGRIS_NO_CHAIN;
        indices[2u * stage_count + i] = i;
    }

    *out_size = plan_size;
    return buf;
}

static void test_schema_v5_many_stage_schedule(void)
{
    printf("  test_schema_v5_many_stage_schedule...\n");
    const uint16_t stage_count = 300;
    uint32_t size = 0;
    uint8_t *buf = build_many_stage_plan(stage_count, &size);
    tigris_plan_t plan;

    TEST_ASSERT(buf != NULL, "many-stage plan allocation");
    if (!buf)
        return;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK,
                   "schema-v5 stage table supports more than 256 stages");

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    tigris_op_t *ops = (tigris_op_t *)(buf + dir[1].offset);
    ops[stage_count - 1u].stage++;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "schema-v5 low-byte stage hint remains canonical");

    free(buf);
    buf = build_many_stage_plan(stage_count, &size);
    TEST_ASSERT(buf != NULL, "legacy many-stage plan allocation");
    if (!buf)
        return;
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "schema-v4 retains its authoritative uint8 stage limit");
    free(buf);
}

/* A compact valid compressed-weight plan.  It lets the loader tests exercise
 * block metadata without depending on generated model fixtures. */
#define WEIGHT_BLOCK_PLAN_SIZE 191u
#define WB_STAGES_OFF          120u
#define WB_INDEX_OFF           148u
#define WB_STRINGS_OFF         150u
#define WB_WEIGHTS_OFF         151u
#define WB_BLOCKS_OFF          163u

static void build_compressed_weight_plan(uint8_t *buf)
{
    memset(buf, 0, WEIGHT_BLOCK_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = WEIGHT_BLOCK_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_stages = 1;
    hdr->num_weights = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, WB_STAGES_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, WB_STAGES_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, WB_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, WB_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, WB_STRINGS_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, WB_STRINGS_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, WB_WEIGHTS_OFF};
    dir[7] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHT_BLOCKS, WB_BLOCKS_OFF};

    tigris_stage_t *stage = (tigris_stage_t *)(buf + WB_STAGES_OFF);
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    tigris_weight_entry_t *weight =
        (tigris_weight_entry_t *)(buf + WB_WEIGHTS_OFF);
    weight->size_bytes = 4;

    uint16_t num_blocks = 1;
    uint16_t compression = TIGRIS_COMPRESS_NONE;
    memcpy(buf + WB_BLOCKS_OFF, &num_blocks, sizeof(num_blocks));
    memcpy(buf + WB_BLOCKS_OFF + sizeof(num_blocks), &compression,
           sizeof(compression));
    tigris_weight_block_t *block =
        (tigris_weight_block_t *)(buf + WB_BLOCKS_OFF + 4u);
    block->stage_idx = 0;
    block->first_weight_idx = 0;
    block->num_weights = 1;
    block->compressed_size = 4;
    block->uncompressed_size = 4;
}

static void test_compressed_block_guards(void)
{
    printf("  test_compressed_block_guards...\n");
    _Alignas(4) uint8_t buf[WEIGHT_BLOCK_PLAN_SIZE];
    tigris_plan_t plan;

    build_compressed_weight_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "minimal compressed block loads");

    build_compressed_weight_plan(buf);
    tigris_weight_block_t *block =
        (tigris_weight_block_t *)(buf + WB_BLOCKS_OFF + 4u);
    block->compressed_size = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "uncompressed blocks require all copied source bytes");

    build_compressed_weight_plan(buf);
    block = (tigris_weight_block_t *)(buf + WB_BLOCKS_OFF + 4u);
    block->uncompressed_size = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "empty weight block rejected");
}

/* A tiny valid plan with one tensor and a model-input reference. */

#define REFERENCED_PLAN_SIZE 133u
#define REF_TENSORS_OFF 96u
#define REF_INDEX_OFF   112u
#define REF_SHAPES_OFF  116u
#define REF_STRINGS_OFF 132u

static void build_referenced_plan(uint8_t *buf)
{
    memset(buf, 0, REFERENCED_PLAN_SIZE);
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = REFERENCED_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 1;
    hdr->num_model_inputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, REF_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, REF_INDEX_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, REF_INDEX_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, REF_SHAPES_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, REF_STRINGS_OFF};

    tigris_tensor_t *tensor = (tigris_tensor_t *)(buf + REF_TENSORS_OFF);
    tensor->size_bytes = sizeof(float);
    tensor->ndim = 1;
    tensor->dtype = 1;
    tensor->flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensor->quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    *(uint16_t *)(buf + REF_INDEX_OFF) = 0;
    *(int32_t *)(buf + REF_SHAPES_OFF) = 1;
    buf[REF_STRINGS_OFF] = '\0';
}

static void test_cross_reference_guards(void)
{
    printf("  test_cross_reference_guards...\n");
    _Alignas(4) uint8_t buf[REFERENCED_PLAN_SIZE];
    tigris_plan_t plan;

    build_referenced_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "referenced plan loads");

    build_referenced_plan(buf);
    *(uint16_t *)(buf + REF_INDEX_OFF) = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "model IO tensor index");

    build_referenced_plan(buf);
    ((tigris_tensor_t *)(buf + REF_TENSORS_OFF))->shape_off = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR, "tensor shape range");

    build_referenced_plan(buf);
    ((tigris_tensor_t *)(buf + REF_TENSORS_OFF))->name_str = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "tensor string range");

    build_referenced_plan(buf);
    ((tigris_tensor_t *)(buf + REF_TENSORS_OFF))->size_bytes = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR, "tensor size matches shape and dtype");

    build_referenced_plan(buf);
    ((tigris_tensor_t *)(buf + REF_TENSORS_OFF))->dtype = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR, "unsupported activation dtype");
}

/* A compact schema-v4 plan that carries the required Transpose permutation
 * attribute.  This exercises both the optional metadata section and the
 * loader's semantic validation before an executor can consume it. */
#define TRANSPOSE_PLAN_SIZE      259u
#define TRANSPOSE_TENSORS_OFF    112u
#define TRANSPOSE_OPS_OFF        144u
#define TRANSPOSE_STAGES_OFF     182u
#define TRANSPOSE_INDEX_OFF      210u
#define TRANSPOSE_SHAPES_OFF     220u
#define TRANSPOSE_STRINGS_OFF    236u
#define TRANSPOSE_ATTRS_OFF      237u
#define TRANSPOSE_ATTR_DATA_OFF  249u

static void build_transpose_plan(uint8_t *buf)
{
    memset(buf, 0, TRANSPOSE_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = TRANSPOSE_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, TRANSPOSE_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, TRANSPOSE_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, TRANSPOSE_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, TRANSPOSE_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, TRANSPOSE_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, TRANSPOSE_STRINGS_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, TRANSPOSE_ATTRS_OFF};

    tigris_tensor_t *tensors =
        (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[0].size_bytes = 6u * sizeof(float);
    tensors[0].shape_off = 0;
    tensors[0].ndim = 2;
    tensors[0].dtype = 1;
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].size_bytes = 6u * sizeof(float);
    tensors[1].shape_off = 2;
    tensors[1].ndim = 2;
    tensors[1].dtype = 1;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;

    tigris_op_t *op = (tigris_op_t *)(buf + TRANSPOSE_OPS_OFF);
    op->op_type = TIGRIS_OP_TRANSPOSE;
    op->num_inputs = 1;
    op->num_outputs = 1;
    op->inputs_off = 0;
    op->outputs_off = 1;
    op->weight_idx = TIGRIS_NO_WEIGHT;
    op->bias_idx = TIGRIS_NO_WEIGHT;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + TRANSPOSE_STAGES_OFF);
    stage->ops_off = 4;
    stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    uint16_t *indices = (uint16_t *)(buf + TRANSPOSE_INDEX_OFF);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 0;
    indices[3] = 1;
    indices[4] = 0;
    int32_t *shapes = (int32_t *)(buf + TRANSPOSE_SHAPES_OFF);
    shapes[0] = 2;
    shapes[1] = 3;
    shapes[2] = 3;
    shapes[3] = 2;
    buf[TRANSPOSE_STRINGS_OFF] = '\0';

    uint16_t count = 1;
    memcpy(buf + TRANSPOSE_ATTRS_OFF, &count, sizeof(count));
    tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(
        buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->op_index = 0;
    attr->type = TIGRIS_OP_ATTR_TRANSPOSE_PERM;
    attr->data_len = 2;
    attr->data_offset = 0;
    buf[TRANSPOSE_ATTR_DATA_OFF] = 1;
    buf[TRANSPOSE_ATTR_DATA_OFF + 1u] = 0;
}

/* A schema-v7 plan whose single stage carries a height tile plan. The tiled
 * contract admits two stage shapes the stripe rules cannot express: a global
 * reduction, which collapses the height, and a layout conversion, which
 * permutes it. Both are validated here, where the executor is not reached. */
#define TILED_PLAN_SIZE       308u
#define TILED_TENSORS_OFF     120u
#define TILED_OPS_OFF         152u
#define TILED_STAGES_OFF      190u
#define TILED_TILE_PLANS_OFF  218u
#define TILED_INDEX_OFF       242u
#define TILED_SHAPES_OFF      256u
#define TILED_STRINGS_OFF     288u
#define TILED_ATTRS_OFF       292u
#define TILED_ATTR_DATA_OFF   304u

/**
 * Build the plan above. `op_type` selects the stage's single operator;
 * `ndim` its tensor rank. Shapes are the conversion's by default and are
 * overwritten by the caller for a reduction.
 */
static void build_tiled_stage_plan(uint8_t *buf, uint8_t op_type, uint8_t ndim)
{
    memset(buf, 0, TILED_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = TILED_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_tile_plans = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, TILED_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, TILED_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, TILED_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_TILE_PLANS,
                                      TILED_TILE_PLANS_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, TILED_INDEX_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, TILED_SHAPES_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, TILED_STRINGS_OFF};
    /* The section exists only when it has a record: an empty one is not a
     * shape the emitter produces. A Transpose carries its permutation and a
     * reduction the axis it collapses. */
    if (op_type == TIGRIS_OP_TRANSPOSE || op_type == TIGRIS_OP_REDUCE_MEAN)
        dir[7] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES,
                                          TILED_ATTRS_OFF};

    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    uint32_t elems;
    if (ndim == 3u) {
        shapes[0] = 1; shapes[1] = 4; shapes[2] = 8;
        shapes[4] = 1; shapes[5] = 8; shapes[6] = 4;
        elems = 32u;
    } else {
        shapes[0] = 1; shapes[1] = 2; shapes[2] = 4; shapes[3] = 8;
        shapes[4] = 1; shapes[5] = 8; shapes[6] = 2; shapes[7] = 4;
        elems = 64u;
    }

    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    for (uint16_t i = 0; i < 2u; i++) {
        tensors[i].size_bytes = elems * (uint32_t)sizeof(float);
        tensors[i].shape_off = (uint16_t)(i * 4u);
        tensors[i].ndim = ndim;
        tensors[i].dtype = 1;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    tigris_op_t *op = (tigris_op_t *)(buf + TILED_OPS_OFF);
    op->op_type = op_type;
    op->num_inputs = 1;
    op->num_outputs = 1;
    op->inputs_off = 0;
    op->outputs_off = 1;
    op->weight_idx = TIGRIS_NO_WEIGHT;
    op->bias_idx = TIGRIS_NO_WEIGHT;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + TILED_STAGES_OFF);
    stage->ops_off = 4;
    stage->ops_count = 1;
    stage->inputs_off = 5;
    stage->inputs_count = 1;
    stage->outputs_off = 6;
    stage->outputs_count = 1;
    stage->tile_plan_idx = 0;
    stage->chain_id = TIGRIS_NO_CHAIN;

    tigris_tile_plan_t *tile =
        (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->tileable = 1;
    tile->axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    tile->tile_height = 1;
    tile->num_tiles = 4;
    tile->original_height = 4;

    uint16_t *indices = (uint16_t *)(buf + TILED_INDEX_OFF);
    indices[0] = 0;  /* op input */
    indices[1] = 1;  /* op output */
    indices[2] = 0;  /* model input */
    indices[3] = 1;  /* model output */
    indices[4] = 0;  /* stage op */
    indices[5] = 0;  /* stage input */
    indices[6] = 1;  /* stage output */

    buf[TILED_STRINGS_OFF] = '\0';

    uint16_t count = (op_type == TIGRIS_OP_TRANSPOSE ||
                      op_type == TIGRIS_OP_REDUCE_MEAN) ? 1u : 0u;
    memcpy(buf + TILED_ATTRS_OFF, &count, sizeof(count));
    if (count == 0u)
        return;

    tigris_op_attribute_t *attr =
        (tigris_op_attribute_t *)(buf + TILED_ATTRS_OFF + 4u);
    attr->op_index = 0;
    attr->data_offset = 0;
    if (op_type == TIGRIS_OP_REDUCE_MEAN) {
        attr->type = TIGRIS_OP_ATTR_AXES;
        attr->data_len = 1;
        buf[TILED_ATTR_DATA_OFF] = 1;
        return;
    }
    attr->type = TIGRIS_OP_ATTR_TRANSPOSE_PERM;
    attr->data_len = ndim;
    uint8_t *perm = buf + TILED_ATTR_DATA_OFF;
    if (ndim == 3u) {
        perm[0] = 0; perm[1] = 2; perm[2] = 1;
    } else {
        perm[0] = 0; perm[1] = 3; perm[2] = 1; perm[3] = 2;
    }
}

/* A reduction states the one axis it collapses, and the result either keeps
 * that axis at one or drops it. Everything else about the shape has to stay
 * put, and a plan that does not say which axis it means is refused rather
 * than defaulted, the way a normalization's variance floor is. */
static void test_reduce_mean_contract(void)
{
    printf("  test_reduce_mean_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;

    /* [1, 4, 8] with the middle axis collapsed, kept at one. */
    build_tiled_stage_plan(buf, TIGRIS_OP_REDUCE_MEAN, 3u);
    tigris_stage_t *stage = (tigris_stage_t *)(buf + TILED_STAGES_OFF);
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    shapes[4] = 1; shapes[5] = 1; shapes[6] = 8;
    tensors[1].size_bytes = 8u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "kept reduced axis accepted");

    /* The same result with the axis dropped instead. */
    tensors[1].ndim = 2u;
    shapes[4] = 1; shapes[5] = 8; shapes[6] = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "dropped reduced axis accepted");

    /* A result that keeps the collapsed extent is not a reduction. Its byte
     * count follows the shape, or the tensor check fires before the operator
     * one and the test would prove nothing about the reduction. */
    shapes[4] = 1; shapes[5] = 4; shapes[6] = 0;
    tensors[1].size_bytes = 4u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "uncollapsed axis rejected");

    /* An axis outside the rank names nothing. */
    tensors[1].ndim = 3u;
    shapes[4] = 1; shapes[5] = 1; shapes[6] = 8;
    tensors[1].size_bytes = 8u * (uint32_t)sizeof(float);
    buf[TILED_ATTR_DATA_OFF] = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "axis outside the rank rejected");

    /* Two axes name a reduction this contract does not cover. */
    buf[TILED_ATTR_DATA_OFF] = 0;
    buf[TILED_ATTR_DATA_OFF + 1u] = 1;
    tigris_op_attribute_t *attr =
        (tigris_op_attribute_t *)(buf + TILED_ATTRS_OFF + 4u);
    attr->data_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "multi-axis reduction rejected");

    /* And a plan that states no axis at all is refused on the section, the
     * way a normalization with no variance floor is. */
    build_tiled_stage_plan(buf, TIGRIS_OP_REDUCE_MEAN, 3u);
    stage = (tigris_stage_t *)(buf + TILED_STAGES_OFF);
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    memset(&dir[7], 0, sizeof(dir[7]));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "attribute-free reduction rejected");
}

/**
 * Schema 8 declares every attribute kind at once so the operators behind them
 * land one at a time. Each kind is validated on load whether or not a kernel
 * reads it yet, and belongs only to the operator that will.
 */
static void test_op_attribute_kinds(void)
{
    printf("  test_op_attribute_kinds...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_op_attribute_t *attr;

    /* Every kind other than the permutation belongs to another operator, so
     * on a Transpose each one is refused. */
    const uint8_t kinds[] = {
        TIGRIS_OP_ATTR_EPSILON,
        TIGRIS_OP_ATTR_ALPHA,
        TIGRIS_OP_ATTR_CLIP_BOUNDS,
        TIGRIS_OP_ATTR_PADS,
        TIGRIS_OP_ATTR_AXES,
        TIGRIS_OP_ATTR_POOL_ROUNDING,
    };
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++) {
        build_transpose_plan(buf);
        attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
        attr->type = kinds[i];
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       TIGRIS_ERR_BAD_SECTION,
                       "a kind on the wrong operator is refused");
    }

    build_transpose_plan(buf);
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_MAX + 1u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "a kind beyond the set is refused");

    /* A schema-7 plan carries no kind but the permutation, whatever the
     * operator, because the loader that reads it knows no other. */
    build_transpose_plan(buf);
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V7;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_EPSILON;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a schema-7 plan carries no kind beyond the permutation");
}

/**
 * The payload checks each kind applies. The operator each belongs to has no
 * kernel yet, so the plan is refused later for that; what these assert is the
 * error the payload itself produces first.
 */
static void test_op_attribute_payloads(void)
{
    printf("  test_op_attribute_payloads...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_op_attribute_t *attr;
    float value;

    /* A variance floor must be positive and finite. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_LAYER_NORM;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_EPSILON;
    attr->data_len = (uint8_t)sizeof(float);
    value = 0.0f;
    memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &value, sizeof(value));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "a zero variance floor is refused");

    /* The same record with a positive floor gets past the payload check and
     * is refused for the operator instead, which is a different error. */
    value = 1e-5f;
    memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &value, sizeof(value));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "a positive variance floor passes the payload check");

    /* The pool rounding kind carries exactly one byte, the average value. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_GLOBAL_AVG;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_POOL_ROUNDING;
    attr->data_len = 1;
    buf[TRANSPOSE_ATTR_DATA_OFF] = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "an unknown pool rounding is refused");
    buf[TRANSPOSE_ATTR_DATA_OFF] = TIGRIS_POOL_ROUNDING_AVERAGE;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_ERR_BAD_SECTION,
                "average pool rounding passes the payload check");
    attr->data_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "a two-byte pool rounding is refused");

    /* A wrong payload width is refused whatever the value. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_LEAKY_RELU;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_ALPHA;
    attr->data_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "a short slope payload is refused");

    /* Clip bounds must be ordered. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_CLIP;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_CLIP_BOUNDS;
    attr->data_len = (uint8_t)(2u * sizeof(float));
    float bounds[2] = {1.0f, -1.0f};
    memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, bounds, sizeof(bounds));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "reversed clip bounds are refused");

    /* Pads are two per axis and never negative. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_PAD;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_PADS;
    attr->data_len = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a pad list that is not two per axis is refused");

    /* Reduction axes are in range and strictly increasing. */
    build_transpose_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type =
        TIGRIS_OP_REDUCE_MEAN;
    attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_AXES;
    attr->data_len = 2;
    buf[TRANSPOSE_ATTR_DATA_OFF] = 1;
    buf[TRANSPOSE_ATTR_DATA_OFF + 1u] = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "repeated reduction axes refused");

    buf[TRANSPOSE_ATTR_DATA_OFF] = 0;
    buf[TRANSPOSE_ATTR_DATA_OFF + 1u] = 9;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a reduction axis past the rank is refused");
}

static void test_tiled_conversion_contract(void)
{
    printf("  test_tiled_conversion_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;

    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "rank-3 conversion may carry a height tile plan");

    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 4);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "rank-4 conversion may carry a height tile plan");

    /* A permutation that also reorders the spatial pair is not a conversion. */
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 4);
    buf[TILED_ATTR_DATA_OFF + 2u] = 2;
    buf[TILED_ATTR_DATA_OFF + 3u] = 1;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[5] = 8;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[6] = 4;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[7] = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "a non-conversion permutation may not claim a tile plan");

    /* Rank 2 has only the row band, and a Transpose does not keep rows
     * independent, so the stage qualifies for no tiling contract at all. */
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3);
    ((tigris_tensor_t *)(buf + TILED_TENSORS_OFF))[0].ndim = 2;
    ((tigris_tensor_t *)(buf + TILED_TENSORS_OFF))[1].ndim = 2;
    ((tigris_op_attribute_t *)(buf + TILED_ATTRS_OFF + 4u))->data_len = 2;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[0] = 4;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[1] = 8;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[4] = 8;
    ((int32_t *)(buf + TILED_SHAPES_OFF))[5] = 4;
    buf[TILED_ATTR_DATA_OFF] = 1;
    buf[TILED_ATTR_DATA_OFF + 1u] = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a rank-2 transpose may not claim a tile plan");
}

static void test_tiled_reduction_contract(void)
{
    printf("  test_tiled_reduction_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    int32_t *shapes;

    /* [1, 2, 4, 8] reduced to [1, 1, 1, 8]. */
    build_tiled_stage_plan(buf, TIGRIS_OP_GLOBAL_AVG, 4);
    shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 1; shapes[6] = 1; shapes[7] = 8;
    ((tigris_tensor_t *)(buf + TILED_TENSORS_OFF))[1].size_bytes =
        8u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "a collapsed reduction may carry a height tile plan");

    /* An output that still has spatial extent is not a global reduction. */
    build_tiled_stage_plan(buf, TIGRIS_OP_GLOBAL_AVG, 4);
    shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 1; shapes[6] = 2; shapes[7] = 8;
    ((tigris_tensor_t *)(buf + TILED_TENSORS_OFF))[1].size_bytes =
        16u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "an uncollapsed reduction may not claim a tile plan");
}

static void test_transpose_attribute_guards(void)
{
    printf("  test_transpose_attribute_guards...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;

    build_transpose_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid transpose attribute loads");
    TEST_ASSERT_EQ(plan.num_op_attributes, 1, "one transpose attribute");

    build_transpose_plan(buf);
    buf[TRANSPOSE_ATTR_DATA_OFF + 1u] = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "duplicate permutation axis rejected");

    build_transpose_plan(buf);
    ((tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u))->data_len = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "wrong permutation rank rejected");

    build_transpose_plan(buf);
    ((tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u))->type = 99;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "unknown attribute type rejected");

    build_transpose_plan(buf);
    uint16_t two_attributes = 2;
    memcpy(buf + TRANSPOSE_ATTRS_OFF, &two_attributes,
           sizeof(two_attributes));
    tigris_op_attribute_t *attrs = (tigris_op_attribute_t *)(
        buf + TRANSPOSE_ATTRS_OFF + 4u);
    attrs[1] = attrs[0];
    buf[TRANSPOSE_ATTR_DATA_OFF + sizeof(tigris_op_attribute_t)] = 1;
    buf[TRANSPOSE_ATTR_DATA_OFF + sizeof(tigris_op_attribute_t) + 1u] = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "duplicate operator attribute type rejected");

    build_transpose_plan(buf);
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "v2 cannot claim the v4 attribute section");

    /* v2/v3 plans predate typed attributes and may rely on the public custom
     * dispatcher for legacy Transpose and application operators. The hardened
     * loader must preserve that structural compatibility while v4 remains
     * strict. */
    build_transpose_plan(buf);
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(
        buf + sizeof(tigris_file_header_t));
    memset(&dir[6], 0, sizeof(dir[6]));
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "attribute-free v2 custom Transpose loads");

    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "attribute-free v3 custom Transpose loads");

    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "attribute-free v4 Transpose remains rejected");

    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V2;
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_UNKNOWN;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "v2 custom opcode loads for caller dispatcher");

    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "v4 custom opcode remains fail-closed");

    /* A normalization needs its variance floor as surely as a Transpose needs
     * its permutation, and the arm taken when there is no attribute section
     * at all has to say so too. This fixture carries no scale weight, so it
     * would be refused either way; the error says which guard fired, and
     * before the section arm covered normalizations it was the operator
     * check. A plan whose normalization is otherwise well formed had nothing
     * left to catch it and failed at inference instead. */
    build_transpose_plan(buf);
    dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    memset(&dir[6], 0, sizeof(dir[6]));
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type =
        TIGRIS_OP_LAYER_NORM;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "attribute-free normalization rejected");
}


/* A declared interface dtype is a promise the runtime converts on every call,
 * and the only conversion it has is a float interface over a quantized int8
 * tensor. Saying so on load refuses the plan once rather than on whichever
 * call first reaches tigris_iface.c. */
static void test_declared_interface_dtype_contract(void)
{
    printf("  test_declared_interface_dtype_contract...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_tensor_t *tensors;

    build_transpose_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "undeclared interface loads");

    /* A dtype the runtime has no conversion for at all. */
    build_transpose_plan(buf);
    tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[0].iface_dtype = 11u;  /* float64 */
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR,
                   "unconvertible declared dtype rejected");

    /* An int8 interface over a float tensor is the conversion in the
     * direction the runtime does not perform. */
    build_transpose_plan(buf);
    tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[0].iface_dtype = 3u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR,
                   "int8 interface over a float tensor rejected");

    /* A float interface over a tensor that carries no quantization has
     * nothing to convert with. */
    build_transpose_plan(buf);
    tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[0].dtype = 3u;
    tensors[0].size_bytes = 6u;
    tensors[0].iface_dtype = 1u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR,
                   "float interface without quantization rejected");
}


/* A plan whose single stage splits one rank-3 input into two parts. */
#define SPLIT_PLAN_SIZE     276u
#define SPLIT_TENSORS_OFF   104u
#define SPLIT_OPS_OFF       152u
#define SPLIT_STAGES_OFF    190u
#define SPLIT_INDEX_OFF     218u
#define SPLIT_SHAPES_OFF    236u
#define SPLIT_STRINGS_OFF   272u

static void build_split_plan(uint8_t *buf)
{
    memset(buf, 0, SPLIT_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = SPLIT_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 3;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 3;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, SPLIT_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, SPLIT_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, SPLIT_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, SPLIT_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, SPLIT_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, SPLIT_STRINGS_OFF};

    int32_t *shapes = (int32_t *)(buf + SPLIT_SHAPES_OFF);
    shapes[0] = 4; shapes[1] = 2; shapes[2] = 8;   /* input  */
    shapes[3] = 3; shapes[4] = 2; shapes[5] = 8;   /* part 0 */
    shapes[6] = 1; shapes[7] = 2; shapes[8] = 8;   /* part 1 */

    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SPLIT_TENSORS_OFF);
    for (uint16_t i = 0; i < 3u; i++) {
        tensors[i].shape_off = (uint16_t)(i * 3u);
        tensors[i].ndim = 3;
        tensors[i].dtype = 1;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        tensors[i].flags = TIGRIS_TENSOR_LINEAR;
    }
    tensors[0].size_bytes = 64u * (uint32_t)sizeof(float);
    tensors[1].size_bytes = 48u * (uint32_t)sizeof(float);
    tensors[2].size_bytes = 16u * (uint32_t)sizeof(float);
    tensors[0].flags |= TIGRIS_TENSOR_MODEL_INPUT;
    tensors[2].flags |= TIGRIS_TENSOR_MODEL_OUTPUT;

    tigris_op_t *op = (tigris_op_t *)(buf + SPLIT_OPS_OFF);
    op->op_type = TIGRIS_OP_SPLIT;
    op->num_inputs = 1;
    op->num_outputs = 2;
    op->inputs_off = 0;
    op->outputs_off = 1;
    op->weight_idx = TIGRIS_NO_WEIGHT;
    op->bias_idx = TIGRIS_NO_WEIGHT;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + SPLIT_STAGES_OFF);
    stage->ops_off = 5;
    stage->ops_count = 1;
    stage->inputs_off = 6;
    stage->inputs_count = 1;
    stage->outputs_off = 7;
    stage->outputs_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    uint16_t *indices = (uint16_t *)(buf + SPLIT_INDEX_OFF);
    indices[0] = 0;  /* op input */
    indices[1] = 1;  /* part 0 */
    indices[2] = 2;  /* part 1 */
    indices[3] = 0;  /* model input */
    indices[4] = 2;  /* model output */
    indices[5] = 0;  /* stage op */
    indices[6] = 0;  /* stage input */
    indices[7] = 2;  /* stage output */

    buf[SPLIT_STRINGS_OFF] = '\0';
}

/* A split's parts are one run of its input per position ahead of the split
 * axis, which needs no arithmetic. The loader has to say so: a part that
 * differs anywhere but the axis kernel_h names, or a set that does not cover
 * the input exactly, would be copied wrongly rather than refused. */
static void test_split_contract(void)
{
    printf("  test_split_contract...\n");
    _Alignas(4) uint8_t buf[SPLIT_PLAN_SIZE];
    tigris_plan_t plan;
    int32_t *shapes;
    tigris_tensor_t *tensors;

    build_split_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "contiguous parts accepted");

    /* Parts that do not cover the input are not a split of it. */
    build_split_plan(buf);
    shapes = (int32_t *)(buf + SPLIT_SHAPES_OFF);
    tensors = (tigris_tensor_t *)(buf + SPLIT_TENSORS_OFF);
    shapes[3] = 2;
    tensors[1].size_bytes = 32u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "parts that leave a gap rejected");

    /* A part that differs on an inner axis would be interleaved, not cut. */
    build_split_plan(buf);
    shapes = (int32_t *)(buf + SPLIT_SHAPES_OFF);
    tensors = (tigris_tensor_t *)(buf + SPLIT_TENSORS_OFF);
    shapes[5] = 4;
    tensors[1].size_bytes = 24u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "a part cut on an inner axis rejected");

    /* The same cut is a split along the stored axis kernel_h names. */
    ((tigris_op_t *)(buf + SPLIT_OPS_OFF))->spatial.kernel_h = 2;
    shapes[3] = 4; shapes[5] = 6;
    shapes[6] = 4; shapes[8] = 2;
    tensors[1].size_bytes = 48u * (uint32_t)sizeof(float);
    tensors[2].size_bytes = 16u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "a split along a named inner axis accepted");
    ((tigris_op_t *)(buf + SPLIT_OPS_OFF))->spatial.kernel_h = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "a split axis past the rank rejected");

    /* One part is not a split. */
    build_split_plan(buf);
    ((tigris_op_t *)(buf + SPLIT_OPS_OFF))->num_outputs = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "a single part rejected");
}



/* A build states its limits at compile time because they size fixed storage,
 * and a plan carries what it needs in its own tables. When the two disagree
 * the refusal has to name the build that would run the plan, or the only way
 * forward is guessing. */
static void test_a_refused_plan_names_the_build_it_wants(void)
{
    printf("  test_a_refused_plan_names_the_build_it_wants...\n");
    _Alignas(4) uint8_t buf[STAGED_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_plan_limits_t required;
    tigris_plan_limits_t build;

    tigris_build_limits(&build);
    TEST_ASSERT_EQ(build.tensors, TIGRIS_MAX_TENSORS,
                   "the build reports its own tensor limit");
    TEST_ASSERT_EQ(build.stage_inputs, TIGRIS_MAX_STAGE_INPUTS,
                   "the build reports its own stage input limit");

    /* A plan this build can run reports what it needed anyway, so a caller
     * sizing a target from a plan has the number without failing first. */
    build_staged_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load_ex(buf, sizeof(buf), &plan, &required),
                   TIGRIS_OK, "an accepted plan still reports its needs");
    TEST_ASSERT(required.tensors > 0u && required.tensors <= build.tensors,
                "an accepted plan needs no more than the build allows");

    /* One past what this build carries: refused, and the refusal says how
     * many the plan wanted. */
    build_staged_plan(buf);
    ((tigris_stage_t *)(buf + STAGED_STAGES_OFF))->inputs_count =
        (uint16_t)(TIGRIS_MAX_STAGE_INPUTS + 1u);
    TEST_ASSERT_EQ(tigris_plan_load_ex(buf, sizeof(buf), &plan, &required),
                   TIGRIS_ERR_PLAN_LIMITS, "over-limit stage inputs refused");
    TEST_ASSERT_EQ(required.stage_inputs, TIGRIS_MAX_STAGE_INPUTS + 1u,
                   "the refusal names the stage input count wanted");

    /* The old entry point keeps its behaviour, and takes no reporting. A
     * table too large for this build is refused the same way; reaching that
     * check needs a real table of that size, so it is covered by the plans
     * the cross-repository gate compiles rather than by an edited fixture,
     * which the section bounds refuse first and rightly. */
    build_staged_plan(buf);
    ((tigris_stage_t *)(buf + STAGED_STAGES_OFF))->outputs_count =
        (uint16_t)(TIGRIS_MAX_STAGE_OUTPUTS + 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "the plain entry point is unchanged");
}

static void build_unary_plan(uint8_t *buf)
{
    build_transpose_plan(buf);

    /* End the directory before the optional attribute section and turn the
     * Transpose into a plain exact-shape Relu plan.  The remaining bytes are
     * harmless string-section tail data. */
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(
        buf + sizeof(tigris_file_header_t));
    memset(&dir[6], 0, sizeof(dir[6]));

    tigris_op_t *op = (tigris_op_t *)(buf + TRANSPOSE_OPS_OFF);
    op->op_type = TIGRIS_OP_RELU;
    int32_t *shapes = (int32_t *)(buf + TRANSPOSE_SHAPES_OFF);
    shapes[2] = 2;
    shapes[3] = 3;
}

/* A row band cuts the second to last axis, so on a rank-4 tensor the leading
 * axes are batch the band spans rather than cuts. The loader mirrors that
 * because it decides whether the stage is row-banded before it applies the
 * stripe contract, and a stage it accepts that the executor declines runs
 * through the wrong path instead of being refused. */
static void test_rank4_row_band_contract(void)
{
    printf("  test_rank4_row_band_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;

    /* Both sides are the same [1, 2, 4, 8] matrix batch in the model's own
     * axis order: batch 2, four rows of eight. */
    build_tiled_stage_plan(buf, TIGRIS_OP_ERF, 4);
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 2; shapes[6] = 4; shapes[7] = 8;
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].flags |= TIGRIS_TENSOR_LINEAR;
    tensors[1].flags |= TIGRIS_TENSOR_LINEAR;
    tigris_tile_plan_t *tile =
        (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->num_tiles = 2;
    tile->tile_height = 2;
    tile->original_height = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "a rank-4 row band loads");

    /* The same shapes held channels-last are not a matrix batch, so the band
     * does not apply and the stripe contract decides instead. */
    build_tiled_stage_plan(buf, TIGRIS_OP_ERF, 4);
    shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 2; shapes[6] = 4; shapes[7] = 8;
    tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->num_tiles = 2;
    tile->tile_height = 1;
    tile->original_height = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "the spatial reading of the same shapes still loads");
}

static void test_operator_semantic_guards(void)
{
    printf("  test_operator_semantic_guards...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;

    build_unary_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid unary plan loads");

    build_unary_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_UNKNOWN;
    memset(&plan, 0xA5, sizeof(plan));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "unknown opcode rejected");
    TEST_ASSERT(plan.header == NULL && plan.ops == NULL,
                "failed load exposes no partial plan");

    build_unary_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_CLIP;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "undispatched opcode rejected");

    build_unary_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->num_inputs = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "empty operator input rejected");

    build_unary_plan(buf);
    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->fused_act = TIGRIS_ACT_RELU;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "fused activation on unsupported operator rejected");

    build_unary_plan(buf);
    int32_t *shapes = (int32_t *)(buf + TRANSPOSE_SHAPES_OFF);
    shapes[2] = 1;
    shapes[3] = 6;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "unary shape mismatch rejected");

    build_unary_plan(buf);
    tigris_tensor_t *tensors =
        (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[1].dtype = 3;
    tensors[1].size_bytes = 6;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "mixed activation dtype rejected");

    build_unary_plan(buf);
    tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[1].flags = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_TENSOR, "model output flag mismatch rejected");
}

/* Compact one-op plans for exercising the loader's non-weighted operator
 * decision table. Keeping these plans in memory makes every condition
 * reproducible without generated fixtures or a compiler checkout. */
#define SEMANTIC_PLAN_SIZE    289u
#define SEMANTIC_TENSORS_OFF  112u
#define SEMANTIC_OPS_OFF      160u
#define SEMANTIC_STAGES_OFF   200u
#define SEMANTIC_INDEX_OFF    228u
#define SEMANTIC_SHAPES_OFF   240u
#define SEMANTIC_STRINGS_OFF  288u

static void semantic_set_shape(
    uint8_t *buf, uint16_t tensor_index, uint8_t ndim,
    int32_t d0, int32_t d1, int32_t d2, int32_t d3)
{
    tigris_tensor_t *tensor =
        &((tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF))[tensor_index];
    int32_t *shape = (int32_t *)(buf + SEMANTIC_SHAPES_OFF) + 4u * tensor_index;
    const int32_t dims[4] = {d0, d1, d2, d3};
    uint32_t elements = 1;
    tensor->ndim = ndim;
    for (uint8_t i = 0; i < 4; i++) {
        shape[i] = dims[i];
        if (i < ndim)
            elements *= (uint32_t)dims[i];
    }
    tensor->size_bytes = elements * sizeof(float);
}

static void build_semantic_plan(
    uint8_t *buf, tigris_op_type_t op_type, uint8_t num_inputs)
{
    memset(buf, 0, SEMANTIC_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = SEMANTIC_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 3;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 3;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, SEMANTIC_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, SEMANTIC_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, SEMANTIC_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, SEMANTIC_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, SEMANTIC_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, SEMANTIC_STRINGS_OFF};

    tigris_tensor_t *tensors =
        (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF);
    for (uint16_t i = 0; i < 3; i++) {
        tensors[i].shape_off = 4u * i;
        tensors[i].dtype = 1;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        semantic_set_shape(buf, i, 4, 1, 2, 2, 2);
    }
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->op_type = (uint8_t)op_type;
    op->num_inputs = num_inputs;
    op->num_outputs = 1;
    op->inputs_off = 0;
    op->outputs_off = 2;
    op->weight_idx = TIGRIS_NO_WEIGHT;
    op->bias_idx = TIGRIS_NO_WEIGHT;
    op->spatial.kernel_h = 1;
    op->spatial.kernel_w = 1;
    op->spatial.stride_h = 1;
    op->spatial.stride_w = 1;
    op->spatial.dilation_h = 1;
    op->spatial.dilation_w = 1;
    op->spatial.group = 1;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + SEMANTIC_STAGES_OFF);
    stage->ops_off = 5;
    stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    uint16_t *indices = (uint16_t *)(buf + SEMANTIC_INDEX_OFF);
    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 1;
    indices[3] = 0;
    indices[4] = 1;
    indices[5] = 0;
    buf[SEMANTIC_STRINGS_OFF] = '\0';
}

#define ELEMENTWISE_QUANT_OFF 300u
#define ELEMENTWISE_PLAN_SIZE 376u

static void build_elementwise_quant_plan(
    uint8_t *buf, tigris_op_type_t type, uint8_t inputs)
{
    memset(buf, 0, ELEMENTWISE_PLAN_SIZE);
    build_semantic_plan(buf, type, inputs);
    memmove(buf + SEMANTIC_TENSORS_OFF + 8u, buf + SEMANTIC_TENSORS_OFF,
            SEMANTIC_PLAN_SIZE - SEMANTIC_TENSORS_OFF);
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    header->file_size = ELEMENTWISE_PLAN_SIZE;
    header->num_quant_params = 3;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    for (int i = 0; i < 6; i++) dir[i].offset += 8u;
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, ELEMENTWISE_QUANT_OFF};
    uint16_t *quant_header = (uint16_t *)(buf + ELEMENTWISE_QUANT_OFF);
    quant_header[0] = 3;
    quant_header[1] = 1;
    tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + ELEMENTWISE_QUANT_OFF + 4u);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 8u);
    for (uint16_t i = 0; i < 3; i++) {
        tensors[i].dtype = 3;
        tensors[i].size_bytes /= sizeof(float);
        tensors[i].quant_param_idx = i;
        quant[i].scale = 0.125f;
        quant[i].zero_point = -17;
        quant[i].num_channels = 1;
        quant[i].shift_off = 1;
    }
    tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + 8u);
    op->act_min = -128;
    op->act_max = 127;
}

#define NATIVE_REDUCTION_PLAN_SIZE (ELEMENTWISE_PLAN_SIZE + 24u)

static void build_native_reduction_plan(
    uint8_t *buf, tigris_op_type_t type, int quantized, uint8_t axis)
{
    memset(buf, 0, NATIVE_REDUCTION_PLAN_SIZE);
    if (quantized) build_elementwise_quant_plan(buf, type, 1);
    else build_semantic_plan(buf, type, 1);
    uint32_t offset = quantized ? 8u : 0u;
    uint32_t attrs_off = quantized ? ELEMENTWISE_PLAN_SIZE : 292u;
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    header->file_size = attrs_off + 24u;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    dir[quantized ? 7 : 6] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, attrs_off};
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + offset);
    int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + offset);
    uint32_t bytes = quantized ? 1u : sizeof(float);
    for (uint16_t t = 0; t < 2u; t++) {
        tensors[t].ndim = 3;
        tensors[t].size_bytes = 24u * bytes;
        dims[4u * t] = 2; dims[4u * t + 1u] = 3; dims[4u * t + 2u] = 4;
    }
    if (type != TIGRIS_OP_CUMSUM) {
        tensors[1].size_bytes /= (uint32_t)dims[axis];
        dims[4u + axis] = 1;
    }
    uint16_t count = type == TIGRIS_OP_CUMSUM ? 2u : 1u;
    memcpy(buf + attrs_off, &count, sizeof(count));
    tigris_op_attribute_t *attrs = (tigris_op_attribute_t *)(buf + attrs_off + 4u);
    attrs[0] = (tigris_op_attribute_t){0, TIGRIS_OP_ATTR_AXES, 1, 0};
    if (count == 2u)
        attrs[1] = (tigris_op_attribute_t){0, TIGRIS_OP_ATTR_CUMSUM_OPTIONS, 2, 1};
    buf[attrs_off + 4u + count * sizeof(*attrs)] = axis;
}

static void test_arg_contract(void)
{
    _Alignas(4) uint8_t buf[NATIVE_REDUCTION_PLAN_SIZE];
    tigris_plan_t plan;
    for (int q = 0; q < 2; q++) {
        for (uint8_t kind = TIGRIS_OP_ARG_MAX; kind <= TIGRIS_OP_ARG_MIN; kind++) {
            for (uint8_t axis = 0; axis < 3u; axis++) {
                build_native_reduction_plan(buf, kind, q, axis);
                uint32_t off = q ? 8u : 0u;
                tigris_file_header_t *header = (tigris_file_header_t *)buf;
                tigris_tensor_t *t = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + off);
                tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + off);
                int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + off);
                t[1].dtype = 6;
                t[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
                if (q) t[1].size_bytes *= 4u;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "int32 arg output accepted");
                t[1].iface_dtype = 7;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "int64 arg interface accepted");
                t[1].ndim = 2;
                uint8_t written = 0;
                for (uint8_t a = 0; a < 3u; a++) if (a != axis) dims[4u + written++] = dims[a];
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "arg drops axis");
                t[1].flags = TIGRIS_TENSOR_MODEL_INPUT | TIGRIS_TENSOR_MODEL_OUTPUT;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_TENSOR, "index cannot be input");
                t[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
                t[1].dtype = 1;
                t[1].iface_dtype = 0;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR, "arg must write int32");
                t[1].dtype = 6;
                op->op_type = TIGRIS_OP_REDUCE_MAX;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR, "only arg may produce an index");
                op->op_type = kind;
                op->num_outputs = 0;
                TEST_ASSERT(tigris_plan_load(buf, header->file_size, &plan) != TIGRIS_OK, "orphan index refused");
            }
            /* An index reads its axis from any rank: [2, 12] -> [2], and
             * [2, 3, 2, 2] -> [2, 3, 2] over the last axis. */
            build_native_reduction_plan(buf, kind, q, 1);
            uint32_t off = q ? 8u : 0u;
            tigris_file_header_t *header = (tigris_file_header_t *)buf;
            tigris_tensor_t *t = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + off);
            int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + off);
            t[1].dtype = 6;
            t[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
            t[0].ndim = 2; dims[0] = 2; dims[1] = 12;
            t[1].ndim = 1; dims[4] = 2;
            t[1].size_bytes = 2u * 4u;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "rank-two arg accepted");
            t[1].ndim = 2; dims[5] = 1;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "rank-two arg keeps its axis");
            dims[5] = 2; t[1].size_bytes = 4u * 4u;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR, "kept axis must be one");
            t[0].ndim = 4; dims[0] = 2; dims[1] = 3; dims[2] = 2; dims[3] = 2;
            t[1].ndim = 3; dims[4] = 2; dims[5] = 3; dims[6] = 2;
            t[1].size_bytes = 12u * 4u;
            buf[header->file_size - 24u + 12u] = 3;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "rank-four arg accepted");
            t[1].ndim = 1; dims[4] = 12;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR, "arg drops only one axis");
        }
    }
    /* The reductions keep the rank-three contract. */
    build_native_reduction_plan(buf, TIGRIS_OP_REDUCE_MAX, 0, 1);
    {
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF);
        int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF);
        t[0].ndim = 2; dims[0] = 2; dims[1] = 12;
        t[1].ndim = 1; dims[4] = 2;
        t[1].size_bytes = 2u * 4u;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR, "rank-two reduction refused");
    }
}

static void test_reduce_all_contract(void)
{
    _Alignas(4) uint8_t buf[NATIVE_REDUCTION_PLAN_SIZE];
    tigris_plan_t plan;
    for (uint8_t axis = 0; axis < 3u; axis++) {
        build_native_reduction_plan(buf, TIGRIS_OP_REDUCE_ALL, 0, axis);
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF);
        t[0].dtype = 9; t[0].size_bytes = 24u;
        t[1].dtype = 9; t[1].size_bytes /= 4u;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK, "bool reduce all accepted");
        t[1].dtype = 1; t[1].size_bytes *= 4u;
        TEST_ASSERT(tigris_plan_load(buf, header->file_size, &plan) != TIGRIS_OK, "reduce all writes bool");
    }
}

static void test_native_reduction_contract(void)
{
    _Alignas(4) uint8_t buf[NATIVE_REDUCTION_PLAN_SIZE];
    tigris_plan_t plan;
    const uint8_t kinds[] = {TIGRIS_OP_REDUCE_MAX, TIGRIS_OP_REDUCE_MIN, TIGRIS_OP_REDUCE_SUM};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        for (int quantized = 0; quantized < 2; quantized++) {
            uint32_t offset = quantized ? 8u : 0u;
            for (uint8_t axis = 0; axis < 3u; axis++) {
                build_native_reduction_plan(buf, kinds[k], quantized, axis);
                tigris_file_header_t *header = (tigris_file_header_t *)buf;
                tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + offset);
                int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + offset);
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK,
                               "rank-three reduction accepts every kept axis");
                tensors[1].ndim = 2;
                uint8_t written = 0;
                for (uint8_t a = 0; a < 3u; a++) if (a != axis) dims[4u + written++] = dims[a];
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK,
                               "rank-three reduction accepts every dropped axis");
                int32_t first = dims[4]; dims[4] = dims[5]; dims[5] = first;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                               "reduction rejects changed nonreduced extents");
            }
            build_native_reduction_plan(buf, kinds[k], quantized, 1);
            tigris_file_header_t *header = (tigris_file_header_t *)buf;
            tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + offset);
            tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + offset);
            int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + offset);
            uint32_t attrs_off = quantized ? ELEMENTWISE_PLAN_SIZE : 292u;
            tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + attrs_off + 4u);
            op->num_inputs = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "native reduction rejects a second activation input");
            op->num_inputs = 1;
            tensors[0].ndim = 4; dims[2] = 2; dims[3] = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "native reduction rejects rank-four input");
            tensors[0].ndim = 3; dims[2] = 4;
            tensors[1].ndim = 4; dims[7] = 1;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "native reduction rejects rank-four output");
            tensors[1].ndim = 3;
            attr->data_len = 2;
            buf[attrs_off + 12u] = 0; buf[attrs_off + 13u] = 1;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "native reduction rejects multiple axes");
            attr->data_len = 1;
            buf[attrs_off + 12u] = 3;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                           "native reduction rejects an axis outside the rank");
            buf[attrs_off + 12u] = 1;
            if (quantized) {
                tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + ELEMENTWISE_QUANT_OFF + 4u);
                for (uint16_t t = 0; t < 2u; t++) {
                    tensors[t].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                                   "int8 reduction requires input and output quantization");
                    tensors[t].quant_param_idx = t;
                    quant[t].num_channels = 2; quant[t].shift_off = 2;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                                   t == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                                   "int8 reduction reads its input's encoding; its output is per tensor");
                    quant[t].num_channels = 1; quant[t].shift_off = 1;
                }
                quant[1].scale *= 2.0f;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                               k == 2u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                               "only SUM can change the reduction output scale");
                quant[1].scale /= 2.0f;
                quant[1].zero_point++;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                               k == 2u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                               "only SUM can change the reduction output zero point");
            }
            tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
            memset(&dir[quantized ? 7 : 6], 0, sizeof(*dir));
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                           "native reduction requires an axes attribute");
        }
    }
}

static void test_cumsum_loader_contract(void)
{
    _Alignas(4) uint8_t buf[NATIVE_REDUCTION_PLAN_SIZE];
    tigris_plan_t plan;
    for (int quantized = 0; quantized < 2; quantized++) {
        uint32_t offset = quantized ? 8u : 0u;
        uint32_t attrs_off = quantized ? ELEMENTWISE_PLAN_SIZE : 292u;
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        for (uint8_t axis = 0; axis < 3u; axis++) {
            build_native_reduction_plan(buf, TIGRIS_OP_CUMSUM, quantized, axis);
            for (uint8_t flags = 0; flags < 4u; flags++) {
                buf[attrs_off + 21u] = flags & 1u;
                buf[attrs_off + 22u] = flags >> 1;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK,
                               "CumSum accepts every axis and exclusive/reverse combination");
            }
        }
        build_native_reduction_plan(buf, TIGRIS_OP_CUMSUM, quantized, 1);
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + offset);
        tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + offset);
        int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + offset);
        tigris_op_attribute_t *attrs = (tigris_op_attribute_t *)(buf + attrs_off + 4u);
        op->num_inputs = 2;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "CumSum rejects a second activation input");
        op->num_inputs = 1;
        dims[5] = 4; dims[6] = 3;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "CumSum rejects equal-sized tensors with different shapes");
        dims[5] = 3; dims[6] = 4;
        for (uint16_t t = 0; t < 2u; t++) {
            tensors[t].ndim = 2; dims[4u * t] = 6; dims[4u * t + 1u] = 4;
        }
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "CumSum rejects matching rank-two tensors");
        for (uint16_t t = 0; t < 2u; t++) {
            tensors[t].ndim = 3; dims[4u * t] = 2; dims[4u * t + 1u] = 3;
        }
        if (quantized) {
            tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + ELEMENTWISE_QUANT_OFF + 4u);
            for (uint16_t t = 0; t < 2u; t++) {
                tensors[t].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                               "int8 CumSum requires both quantization records");
                tensors[t].quant_param_idx = t;
                quant[t].num_channels = 2; quant[t].shift_off = 2;
                TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                               t == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                               "int8 CumSum reads its input's encoding; its output is per tensor");
                quant[t].num_channels = 1; quant[t].shift_off = 1;
            }
            quant[0].scale = 65535.0f;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK,
                           "CumSum accepts a scale ratio below its fixed-point limit");
            quant[0].scale = 65536.0f;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "CumSum rejects a scale ratio at its fixed-point limit");
            quant[0].scale = 131072.0f;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "CumSum rejects a scale ratio above its fixed-point limit");
            quant[0].scale = 0.125f;
        }
        attrs[1].data_len = 1;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum requires two option bytes");
        attrs[1].data_len = 2;
        for (uint8_t option = 0; option < 2u; option++) {
            buf[attrs_off + 21u + option] = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                           "CumSum rejects a nonboolean option");
            buf[attrs_off + 21u + option] = 0;
        }
        attrs[0].data_len = 2; buf[attrs_off + 20u] = 0; buf[attrs_off + 21u] = 1;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "CumSum rejects multiple axes");
        attrs[0].data_len = 1; buf[attrs_off + 20u] = 3;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum rejects an axis outside the rank");
        buf[attrs_off + 20u] = 1;
        op->op_type = TIGRIS_OP_REDUCE_SUM;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum options cannot attach to a reduction");
        op->op_type = TIGRIS_OP_CUMSUM;
        uint16_t count = 1;
        memcpy(buf + attrs_off, &count, sizeof(count));
        attrs[0].data_offset = 8;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum requires explicit options beside its axis");
        attrs[0] = attrs[1]; attrs[0].data_offset = 9;
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum requires an axis beside its options");
        tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
        memset(&dir[quantized ? 7 : 6], 0, sizeof(*dir));
        TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_SECTION,
                       "CumSum requires an attribute section");
    }
}

static void test_native_reduction_tiling_contract(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE + 12u];
    tigris_plan_t plan;
    const uint8_t kinds[] = {TIGRIS_OP_REDUCE_MAX, TIGRIS_OP_REDUCE_MIN,
                             TIGRIS_OP_REDUCE_SUM, TIGRIS_OP_CUMSUM};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        memset(buf, 0, sizeof(buf));
        build_tiled_stage_plan(buf, TIGRIS_OP_REDUCE_MEAN, 3);
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        header->file_size = sizeof(buf);
        ((tigris_op_t *)(buf + TILED_OPS_OFF))->op_type = kinds[k];
        tigris_stage_t *stage = (tigris_stage_t *)(buf + TILED_STAGES_OFF);
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
        int32_t *dims = (int32_t *)(buf + TILED_SHAPES_OFF);
        memcpy(dims + 4, dims, 3u * sizeof(*dims));
        if (kinds[k] == TIGRIS_OP_CUMSUM) {
            uint16_t count = 2;
            memcpy(buf + TILED_ATTRS_OFF, &count, sizeof(count));
            tigris_op_attribute_t *attrs = (tigris_op_attribute_t *)(buf + TILED_ATTRS_OFF + 4u);
            attrs[1] = (tigris_op_attribute_t){0, TIGRIS_OP_ATTR_CUMSUM_OPTIONS, 2, 1};
            buf[TILED_ATTRS_OFF + 20u] = 1;
        } else {
            dims[5] = 1;
            tensors[1].size_bytes /= 4u;
        }
        stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       "native reduction plan loads without tiling");
        stage->tile_plan_idx = 0;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "native reductions and scans refuse height tiling");
    }
}

static void test_activation_attribute_contract(void)
{
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;
    const uint8_t kinds[] = {TIGRIS_OP_LEAKY_RELU, TIGRIS_OP_L2_NORMALIZATION};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        build_transpose_plan(buf);
        tigris_op_t *op = (tigris_op_t *)(buf + TRANSPOSE_OPS_OFF);
        op->op_type = kinds[k];
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
        tensors[1].shape_off = tensors[0].shape_off;
        tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
        attr->type = k == 0u ? TIGRIS_OP_ATTR_ALPHA : TIGRIS_OP_ATTR_EPSILON;
        attr->data_len = sizeof(float);
        const float valid[] = {0.0f, 0.125f};
        for (size_t v = 0; v < sizeof(valid) / sizeof(valid[0]); v++) {
            memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &valid[v], sizeof(float));
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                           "finite activation attribute loads, including zero epsilon");
        }
        float negative = -0.125f;
        memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &negative, sizeof(float));
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       k == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_SECTION,
                       "negative alpha is supported and negative epsilon is rejected");
        const float invalid[] = {NAN, INFINITY};
        for (size_t v = 0; v < sizeof(invalid) / sizeof(invalid[0]); v++) {
            memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &invalid[v], sizeof(float));
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                           "nonfinite activation attribute rejected");
        }
        memcpy(buf + TRANSPOSE_ATTR_DATA_OFF, &valid[1], sizeof(float));
        attr->data_len = 1;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                       "truncated activation attribute rejected");
        attr->data_len = sizeof(float);
        op->op_type = TIGRIS_OP_ELU;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                       "activation attributes cannot attach to Elu");
        op->op_type = kinds[k];
        tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
        memset(&dir[6], 0, sizeof(dir[6]));
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       k == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_SECTION,
                       "LeakyRelu defaults alpha but L2Normalization requires epsilon");
    }
}

static void test_activation_semantic_contract(void)
{
    _Alignas(4) uint8_t buf[ELEMENTWISE_PLAN_SIZE + 16u];
    tigris_plan_t plan;
    const uint8_t kinds[] = {TIGRIS_OP_LEAKY_RELU, TIGRIS_OP_ELU,
                             TIGRIS_OP_LOG_SOFTMAX, TIGRIS_OP_L2_NORMALIZATION};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        for (int quantized = 0; quantized < 2; quantized++) {
            uint32_t offset = quantized ? 8u : 0u;
            if (quantized) build_elementwise_quant_plan(buf, kinds[k], 1);
            else build_semantic_plan(buf, kinds[k], 1);
            tigris_file_header_t *header = (tigris_file_header_t *)buf;
            tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
            tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + offset);
            tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + offset);
            if (kinds[k] == TIGRIS_OP_L2_NORMALIZATION) {
                uint32_t attrs_off = quantized ? ELEMENTWISE_PLAN_SIZE : SEMANTIC_PLAN_SIZE;
                header->file_size = attrs_off + 16u;
                dir[quantized ? 7 : 6] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, attrs_off};
                memset(buf + attrs_off, 0, 16u);
                uint16_t count = 1;
                memcpy(buf + attrs_off, &count, sizeof(count));
                tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_EPSILON, sizeof(float), 0};
                memcpy(buf + attrs_off + 4u, &attr, sizeof(attr));
                float epsilon = 1.0e-6f;
                memcpy(buf + attrs_off + 12u, &epsilon, sizeof(epsilon));
            }
            tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + ELEMENTWISE_QUANT_OFF + 4u);
            if (quantized && k >= 2u) {
                quant[1].scale = k == 2u ? 0.0625f : 0.0078125f;
                quant[1].zero_point = k == 2u ? 127 : 0;
            }
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_OK,
                           "activation with matching shapes and encoding loads");
            op->num_inputs = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "activation rejects extra input");
            op->num_inputs = 1;
            int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + offset);
            dims[5] = 1; dims[6] = 4;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "activation rejects equal-size output with different shape");
            dims[5] = 2; dims[6] = 2;
            if (quantized) {
                for (uint16_t t = 0; t < 2u; t++) {
                    tensors[t].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                                   "activation requires input and output quantization");
                    tensors[t].quant_param_idx = t;
                    quant[t].num_channels = 2;
                    quant[t].shift_off = 2;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                                   t == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                                   "activation reads its input's encoding; its output is per tensor");
                    quant[t].num_channels = 1;
                    quant[t].shift_off = 1;
                }
                if (k >= 2u) {
                    quant[1].scale *= 2.0f;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                                   "normalization requires its fixed output scale");
                    quant[1].scale /= 2.0f;
                    quant[1].zero_point--;
                    TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan), TIGRIS_ERR_BAD_OPERATOR,
                                   "normalization requires its fixed output zero point");
                    quant[1].zero_point++;
                }
            }
            for (uint16_t t = 0; t < 2u; t++) {
                tensors[t].ndim = 0;
                tensors[t].size_bytes = quantized ? 1u : sizeof(float);
            }
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                           k < 2u ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                           "scalar pointwise activations load and scalar normalizations are refused");
            for (uint16_t t = 0; t < 2u; t++) {
                tensors[t].ndim = 5;
                tensors[t].shape_off = 0;
                tensors[t].size_bytes = quantized ? 8u : 8u * sizeof(float);
            }
            dims[0] = 1; dims[1] = 1; dims[2] = 1; dims[3] = 1; dims[4] = 8;
            TEST_ASSERT_EQ(tigris_plan_load(buf, header->file_size, &plan),
                           k == 3u ? TIGRIS_ERR_BAD_OPERATOR : TIGRIS_OK,
                           "only L2Normalization has a rank-four limit");
        }
    }
}

#define PRELU_PLAN_SIZE 420u
#define PRELU_ATTRS_OFF 404u
#define PRELU_WEIGHTS_OFF 308u
#define PRELU_QUANT_OFF 328u

static void build_prelu_contract_plan(uint8_t *buf, int quantized)
{
    memset(buf, 0, PRELU_PLAN_SIZE);
    build_elementwise_quant_plan(buf, TIGRIS_OP_PRELU, 1);
    memmove(buf + SEMANTIC_TENSORS_OFF + 16u, buf + SEMANTIC_TENSORS_OFF + 8u,
            ELEMENTWISE_PLAN_SIZE - SEMANTIC_TENSORS_OFF - 8u);
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    header->file_size = PRELU_PLAN_SIZE;
    header->num_weights = 1;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    for (int i = 0; i < 7; i++) dir[i].offset += 8u;
    memmove(buf + PRELU_QUANT_OFF, buf + ELEMENTWISE_QUANT_OFF + 8u,
            ELEMENTWISE_PLAN_SIZE - ELEMENTWISE_QUANT_OFF);
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, PRELU_WEIGHTS_OFF};
    dir[7] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, PRELU_QUANT_OFF};
    dir[8] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, PRELU_ATTRS_OFF};
    memset(&dir[9], 0, sizeof(dir[9]));
    uint16_t count = 1;
    memcpy(buf + PRELU_ATTRS_OFF, &count, sizeof(count));
    tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_CONSTANT_OPERAND, 4, 0};
    memcpy(buf + PRELU_ATTRS_OFF + 4u, &attr, sizeof(attr));
    const uint8_t payload[] = {1, 0, quantized ? 2u : 255u, quantized ? 0u : 255u};
    memcpy(buf + PRELU_ATTRS_OFF + 12u, payload, sizeof(payload));
    tigris_weight_entry_t weight = {0, 0, quantized ? 2u : 8u};
    memcpy(buf + PRELU_WEIGHTS_OFF, &weight, sizeof(weight));
    tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + 16u);
    op->weight_idx = 0;
    if (!quantized) {
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 16u);
        for (int i = 0; i < 3; i++) {
            tensors[i].dtype = 1;
            tensors[i].size_bytes *= sizeof(float);
            tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        }
    }
}

static void test_prelu_loader_contract(void)
{
    _Alignas(4) uint8_t buf[PRELU_PLAN_SIZE];
    tigris_plan_t plan;
    for (int quantized = 0; quantized < 2; quantized++) {
        build_prelu_contract_plan(buf, quantized);
        tigris_op_t *op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF + 16u);
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 16u);
        uint8_t *payload = buf + PRELU_ATTRS_OFF + 12u;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       "PRelu constant alpha loads");
        payload[0] = 0;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses a constant first operand");
        payload[0] = 1;
        op->num_inputs = 2;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses dynamic alpha with a constant record");
        op->num_inputs = 1;
        op->weight_idx = TIGRIS_NO_WEIGHT;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu requires the alpha weight");
        op->weight_idx = 0;
        int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + 16u);
        dims[5] = 1; dims[6] = 4;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu requires unchanged output shape");
        dims[5] = 2; dims[6] = 2;
        if (quantized) {
            payload[2] = 255; payload[3] = 255;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "int8 PRelu requires alpha quantization");
            payload[2] = 2; payload[3] = 0;
            tigris_quant_param_t *q = (tigris_quant_param_t *)(buf + PRELU_QUANT_OFF + 4u);
            q[2].num_channels = 2; q[2].shift_off = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "int8 PRelu refuses per-channel alpha encoding");
            q[2].num_channels = 1; q[2].shift_off = 1;
        }
        for (uint16_t t = 0; t < 2u; t++) {
            tensors[t].ndim = 5; tensors[t].shape_off = 0;
        }
        dims[0] = 1; dims[1] = 1; dims[2] = 1; dims[3] = 1; dims[4] = 8;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses rank five");
        for (uint16_t t = 0; t < 2u; t++) {
            tensors[t].ndim = 0;
            tensors[t].size_bytes = quantized ? 1u : sizeof(float);
        }
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses rank zero");
        build_prelu_contract_plan(buf, quantized);
        tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
        memset(&dir[8], 0, sizeof(dir[8]));
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses absent operand placement metadata");
    }
    build_semantic_plan(buf, TIGRIS_OP_PRELU, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "PRelu refuses two dynamic operands");
}

static void test_prelu_tiling_contract(void)
{
    _Alignas(4) uint8_t buf[PRELU_PLAN_SIZE + 32u];
    tigris_plan_t plan;
    for (uint8_t rank = 2; rank <= 4u; rank++) {
        build_prelu_contract_plan(buf, 0);
        memmove(buf + SEMANTIC_TENSORS_OFF + 24u, buf + SEMANTIC_TENSORS_OFF + 16u,
                PRELU_PLAN_SIZE - SEMANTIC_TENSORS_OFF - 16u);
        tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
        for (int i = 0; i < 9; i++) dir[i].offset += 8u;
        const uint32_t tile_off = SEMANTIC_INDEX_OFF + 24u;
        memmove(buf + tile_off + sizeof(tigris_tile_plan_t), buf + tile_off,
                PRELU_PLAN_SIZE + 8u - tile_off);
        for (int i = 0; i < 9; i++)
            if (dir[i].offset >= tile_off) dir[i].offset += sizeof(tigris_tile_plan_t);
        memmove(&dir[4], &dir[3], 6u * sizeof(dir[0]));
        dir[3] = (tigris_section_entry_t){TIGRIS_SEC_TILE_PLANS, tile_off};
        memset(&dir[10], 0, sizeof(dir[10]));
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        header->file_size = sizeof(buf);
        header->num_tile_plans = 1;
        tigris_stage_t *stage = (tigris_stage_t *)(buf + SEMANTIC_STAGES_OFF + 24u);
        stage->tile_plan_idx = 0;
        stage->inputs_off = 3; stage->inputs_count = 1;
        stage->outputs_off = 4; stage->outputs_count = 1;
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 24u);
        int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + 48u);
        const int32_t shapes[3][4] = {{4, 2, 1, 1}, {1, 4, 2, 1}, {1, 2, 2, 2}};
        for (int t = 0; t < 2; t++) {
            tensors[t].ndim = rank;
            memcpy(dims + 4 * t, shapes[rank - 2u], 4u * sizeof(int32_t));
        }
        tigris_tile_plan_t tile = {0};
        tile.tileable = 1;
        tile.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
        tile.tile_height = 1;
        tile.original_height = rank == 4u ? 2u : 4u;
        tile.num_tiles = tile.original_height;
        memcpy(buf + tile_off, &tile, sizeof(tile));
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       "constant-alpha PRelu accepts row, sequence, and image height tiling");
    }
}

static void test_activation_tiling_contract(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    const uint8_t kinds[] = {TIGRIS_OP_LEAKY_RELU, TIGRIS_OP_ELU,
                             TIGRIS_OP_LOG_SOFTMAX, TIGRIS_OP_L2_NORMALIZATION};
    for (size_t k = 0; k < sizeof(kinds); k++) {
        for (uint8_t rank = 2; rank <= 4u; rank++) {
            build_tiled_stage_plan(buf, kinds[k], rank);
            tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
            int32_t *dims = (int32_t *)(buf + TILED_SHAPES_OFF);
            if (rank == 2u) {
                dims[0] = 4; dims[1] = 8;
                tensors[0].size_bytes = 32u * sizeof(float);
            }
            memcpy(dims + 4, dims, 4u * sizeof(int32_t));
            tensors[1].size_bytes = tensors[0].size_bytes;
            tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
            tile->original_height = rank == 2u ? dims[0] : dims[1];
            tile->num_tiles = tile->original_height;
            if (kinds[k] == TIGRIS_OP_L2_NORMALIZATION) {
                tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
                dir[7] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, TILED_ATTRS_OFF};
                uint16_t count = 1;
                memcpy(buf + TILED_ATTRS_OFF, &count, sizeof(count));
                tigris_op_attribute_t attr = {0, TIGRIS_OP_ATTR_EPSILON, sizeof(float), 0};
                memcpy(buf + TILED_ATTRS_OFF + 4u, &attr, sizeof(attr));
                float epsilon = 1.0e-6f;
                memcpy(buf + TILED_ATTR_DATA_OFF, &epsilon, sizeof(epsilon));
            }
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                           "activation accepts row, sequence, and image height tiling");
        }
    }
    _Alignas(4) uint8_t pool_buf[ELEMENTWISE_PLAN_SIZE];
    build_semantic_plan(pool_buf, TIGRIS_OP_L2_POOL, 1);
    TEST_ASSERT_EQ(tigris_plan_load(pool_buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_OK,
                   "float L2Pool loads");
    semantic_set_shape(pool_buf, 1, 4, 1, 1, 4, 2);
    TEST_ASSERT_EQ(tigris_plan_load(pool_buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "L2Pool rejects inconsistent output geometry");
    build_elementwise_quant_plan(pool_buf, TIGRIS_OP_L2_POOL, 1);
    TEST_ASSERT_EQ(tigris_plan_load(pool_buf, sizeof(pool_buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "L2Pool rejects int8 activations");
    build_tiled_stage_plan(buf, TIGRIS_OP_L2_POOL, 4);
    tigris_op_t *op = (tigris_op_t *)(buf + TILED_OPS_OFF);
    op->spatial.kernel_h = 1; op->spatial.kernel_w = 1;
    op->spatial.stride_h = 1; op->spatial.stride_w = 1;
    int32_t *dims = (int32_t *)(buf + TILED_SHAPES_OFF);
    memcpy(dims + 4, dims, 4u * sizeof(int32_t));
    tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->original_height = 2; tile->num_tiles = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "L2Pool supports image height tiling");
}

static void test_resize_scale_guards(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE + 4u];
    tigris_plan_t plan;
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 4);
    ((tigris_file_header_t *)buf)->file_size = sizeof(buf);
    tigris_op_t *op = (tigris_op_t *)(buf + TILED_OPS_OFF);
    op->op_type = TIGRIS_OP_RESIZE_LINEAR;
    int32_t shapes[8] = {1, 10, 7, 4, 1, 7, 4, 4};
    memcpy(buf + TILED_SHAPES_OFF, shapes, sizeof(shapes));
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].size_bytes = 10u * 7u * 4u * 4u;
    tensors[1].size_bytes = 7u * 4u * 4u * 4u;
    tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->num_tiles = 7;
    tile->original_height = 7;
    tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + TILED_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_RESIZE_SCALES;
    attr->data_len = 8;
    float scales[2] = {0.7f, 0.6f};
    memcpy(buf + TILED_ATTR_DATA_OFF, scales, sizeof(scales));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "noninteger scales preserve float32 output shape rounding");
    scales[0] = 0.8f;
    memcpy(buf + TILED_ATTR_DATA_OFF, scales, sizeof(scales));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "scale/output disagreement rejected");
    scales[0] = 0.0f;
    memcpy(buf + TILED_ATTR_DATA_OFF, scales, sizeof(scales));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "zero scale rejected");
    attr->data_len = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "short scales payload rejected");

    _Alignas(4) uint8_t quant_buf[ELEMENTWISE_PLAN_SIZE];
    build_elementwise_quant_plan(quant_buf, TIGRIS_OP_RESIZE_LINEAR, 1);
    tensors = (tigris_tensor_t *)(quant_buf + SEMANTIC_TENSORS_OFF + 8u);
    int32_t *dims = (int32_t *)(quant_buf + SEMANTIC_SHAPES_OFF + 8u);
    for (int i = 0; i < 2; i++) {
        dims[4 * i] = 1; dims[4 * i + 1] = i == 0 ? 3 : 4096;
        dims[4 * i + 2] = 1; dims[4 * i + 3] = 1;
        tensors[i].ndim = 4;
        tensors[i].size_bytes = (uint32_t)dims[4 * i + 1];
    }
    op = (tigris_op_t *)(quant_buf + SEMANTIC_OPS_OFF + 8u);
    op->spatial.kernel_h = 2;
    op->spatial.stride_h = 0;
    op->spatial.stride_w = 0;
    TEST_ASSERT_EQ(tigris_plan_load(quant_buf, sizeof(quant_buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "undefined reference coordinates rejected");
    dims[5] = 2049;
    tensors[1].size_bytes = 2049;
    TEST_ASSERT_EQ(tigris_plan_load(quant_buf, sizeof(quant_buf), &plan), TIGRIS_OK,
                   "defined neighboring coordinates accepted");
}

static void test_elementwise_semantics(void)
{
    static const tigris_op_type_t kinds[] = {
        TIGRIS_OP_ABS, TIGRIS_OP_RSQRT, TIGRIS_OP_NEG, TIGRIS_OP_EXP,
        TIGRIS_OP_LOG, TIGRIS_OP_SQRT, TIGRIS_OP_SQUARE, TIGRIS_OP_FLOOR,
        TIGRIS_OP_CEIL, TIGRIS_OP_ROUND, TIGRIS_OP_SIN, TIGRIS_OP_COS,
        TIGRIS_OP_DIV, TIGRIS_OP_SQUARED_DIFFERENCE, TIGRIS_OP_MAXIMUM,
        TIGRIS_OP_MINIMUM, TIGRIS_OP_FLOOR_DIV, TIGRIS_OP_FLOOR_MOD,
    };
    _Alignas(4) uint8_t buf[ELEMENTWISE_PLAN_SIZE];
    tigris_plan_t plan;
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); i++) {
        uint8_t inputs = i < 12u ? 1u : 2u;
        build_semantic_plan(buf, kinds[i], inputs);
        TEST_ASSERT_EQ(tigris_plan_load(buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_OK,
                       "elementwise float plan loads");
        semantic_set_shape(buf, 1, 4, 1, 1, 4, 2);
        TEST_ASSERT_EQ(tigris_plan_load(buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "elementwise mismatched shape rejected");
        build_semantic_plan(buf, kinds[i], inputs == 1u ? 2u : 1u);
        TEST_ASSERT_EQ(tigris_plan_load(buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "elementwise wrong arity rejected");

        build_elementwise_quant_plan(buf, kinds[i], inputs);
        int quantized = kinds[i] == TIGRIS_OP_ABS || kinds[i] == TIGRIS_OP_RSQRT ||
            kinds[i] == TIGRIS_OP_DIV ||
            kinds[i] == TIGRIS_OP_SQUARED_DIFFERENCE || kinds[i] == TIGRIS_OP_MAXIMUM ||
            kinds[i] == TIGRIS_OP_MINIMUM;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       quantized ? TIGRIS_OK : TIGRIS_ERR_BAD_OPERATOR,
                       "elementwise int8 capability checked");
        if (!quantized) continue;
        tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + ELEMENTWISE_QUANT_OFF + 4u);
        /* An input's record may carry the per-channel requantization of the
         * operator that wrote it; the operator's own output is per tensor. */
        quant[0].num_channels = 2;
        quant[0].shift_off = 2;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       "elementwise input reads only its scale and zero point");
        quant[0].num_channels = 1;
        if (inputs == 1u) {
            quant[1].num_channels = 2;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "unary output quantization is per tensor");
            quant[1].num_channels = 1;
        }
        if (kinds[i] == TIGRIS_OP_DIV) {
            quant[1].scale = 0.25f;
            quant[1].zero_point = 31;
            quant[2].scale = 0.0625f;
            quant[2].zero_point = -63;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                           "division accepts distinct input and output quantization");
        }
        if (kinds[i] == TIGRIS_OP_MAXIMUM || kinds[i] == TIGRIS_OP_MINIMUM) {
            quant[2].scale = 0.25f;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "comparison quantization mismatch rejected");
            quant[2].scale = 0.125f;
            quant[1].zero_point++;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                           "comparison zero point mismatch rejected");
        }
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 8u);
        tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "elementwise missing quantization rejected");
    }
}

static void test_nonweighted_operator_semantics(void)
{
    printf("  test_nonweighted_operator_semantics...\n");
    _Alignas(4) uint8_t buf[SEMANTIC_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_op_t *op;

    build_semantic_plan(buf, TIGRIS_OP_SOFTMAX, 1);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid Softmax loads");

    build_semantic_plan(buf, TIGRIS_OP_RESHAPE, 1);
    semantic_set_shape(buf, 1, 2, 1, 8, 0, 0);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "equal-size Reshape loads");

    build_semantic_plan(buf, TIGRIS_OP_FLATTEN, 1);
    semantic_set_shape(buf, 1, 2, 1, 8, 0, 0);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "equal-size Flatten loads");

    build_semantic_plan(buf, TIGRIS_OP_RESHAPE, 1);
    semantic_set_shape(buf, 1, 2, 2, 8, 0, 0);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "size-changing Reshape rejected");

    build_semantic_plan(buf, TIGRIS_OP_ADD, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid binary Add loads");

    build_semantic_plan(buf, TIGRIS_OP_MUL, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid binary Mul loads");

    build_semantic_plan(buf, TIGRIS_OP_ADD, 2);
    semantic_set_shape(buf, 2, 4, 1, 1, 4, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "binary Add shape mismatch rejected");

    for (int average = 0; average <= 1; average++) {
        build_semantic_plan(
            buf, average ? TIGRIS_OP_AVG_POOL : TIGRIS_OP_MAX_POOL, 1);
        semantic_set_shape(buf, 0, 4, 1, 4, 4, 2);
        semantic_set_shape(buf, 1, 4, 1, 2, 2, 2);
        op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
        op->spatial.kernel_h = 2;
        op->spatial.kernel_w = 2;
        op->spatial.stride_h = 2;
        op->spatial.stride_w = 2;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       average ? "valid AveragePool loads" :
                                 "valid MaxPool loads");
    }

    build_semantic_plan(buf, TIGRIS_OP_MAX_POOL, 1);
    semantic_set_shape(buf, 0, 4, 1, 4, 4, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 2, 2);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 2;
    op->spatial.kernel_w = 2;
    op->spatial.stride_h = 2;
    op->spatial.stride_w = 2;
    op->spatial.dilation_h = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "dilated pooling rejected");

    build_semantic_plan(buf, TIGRIS_OP_AVG_POOL, 1);
    semantic_set_shape(buf, 0, 4, 1, 4, 4, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 2, 2);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 2;
    op->spatial.kernel_w = 2;
    op->spatial.stride_h = 2;
    op->spatial.stride_w = 2;
    op->spatial.pad_left = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "oversized pool padding rejected");

    build_semantic_plan(buf, TIGRIS_OP_GLOBAL_AVG, 1);
    semantic_set_shape(buf, 0, 4, 1, 4, 4, 2);
    semantic_set_shape(buf, 1, 4, 1, 1, 1, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid GlobalAveragePool loads");

    build_semantic_plan(buf, TIGRIS_OP_GLOBAL_AVG, 1);
    semantic_set_shape(buf, 0, 4, 1, 4, 4, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 1, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "non-global GlobalAveragePool output rejected");

    /* The compiler names the stored axis in kernel_h: 3 for channels. */
    build_semantic_plan(buf, TIGRIS_OP_CONCAT, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 2, 4);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid channel Concat loads");
    op->spatial.kernel_h = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "rank-4 Concat without an axis means channels");

    build_semantic_plan(buf, TIGRIS_OP_CONCAT, 2);
    semantic_set_shape(buf, 1, 4, 1, 4, 2, 2);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid height Concat loads");
    op->spatial.kernel_h = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "Concat along the wrong axis rejected");
    op->spatial.kernel_h = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "Concat axis past the rank rejected");

    build_semantic_plan(buf, TIGRIS_OP_CONCAT, 2);
    semantic_set_shape(buf, 2, 4, 1, 1, 4, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 2, 4);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "Concat spatial mismatch rejected");

    build_semantic_plan(buf, TIGRIS_OP_CONCAT, 2);
    semantic_set_shape(buf, 1, 4, 1, 2, 2, 3);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.kernel_h = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "Concat channel sum rejected");

    build_semantic_plan(buf, TIGRIS_OP_RESIZE, 1);
    semantic_set_shape(buf, 0, 4, 1, 2, 3, 2);
    semantic_set_shape(buf, 1, 4, 1, 4, 6, 2);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.stride_h = 2;
    op->spatial.stride_w = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid integer Resize loads");

    op->spatial.stride_h = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_OK, "Resize derives a zero-stride axis from tensor shapes");
    op->spatial.kernel_h = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "unknown Resize coordinate convention rejected");

    build_semantic_plan(buf, TIGRIS_OP_RESIZE, 1);
    semantic_set_shape(buf, 0, 4, 1, 2, 3, 2);
    semantic_set_shape(buf, 1, 4, 1, 5, 6, 2);
    op = (tigris_op_t *)(buf + SEMANTIC_OPS_OFF);
    op->spatial.stride_h = 2;
    op->spatial.stride_w = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "Resize geometry mismatch rejected");
}

static void test_convolution_semantic_guards(void)
{
    printf("  test_convolution_semantic_guards...\n");
    _Alignas(4) uint8_t buf[STAGED_PLAN_SIZE];
    tigris_plan_t plan;

    build_staged_plan(buf);
    tigris_op_t *ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    ops[0].spatial.group = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "unsupported Conv group rejected");

    build_staged_plan(buf);
    ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    ops[0].spatial.stride_h = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "zero Conv stride rejected");

    build_staged_plan(buf);
    tigris_weight_entry_t *weight =
        (tigris_weight_entry_t *)(buf + STAGED_WEIGHTS_OFF);
    weight->size_bytes = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "undersized Conv weights rejected");

    build_staged_plan(buf);
    ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    ops[0].stage = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR, "operator stage mismatch rejected");
}

/* Compact one-op plan for exercising ConvTranspose loader acceptance.
 * Two tensors (input, output) plus a one-element weight, mirroring the
 * weight_idx convention kern_conv_transpose[_s8] actually reads (num_inputs
 * stays 1; the weight is addressed through op->weight_idx like Conv, not
 * appended to the input list). */
#define CONVT_PLAN_SIZE    276u
#define CONVT_TENSORS_OFF  112u
#define CONVT_OPS_OFF      144u
#define CONVT_STAGES_OFF   184u
#define CONVT_INDEX_OFF    212u
#define CONVT_SHAPES_OFF   224u
#define CONVT_STRINGS_OFF  256u
#define CONVT_WEIGHTS_OFF  260u

static void build_conv_transpose_plan(uint8_t *buf)
{
    memset(buf, 0, CONVT_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = CONVT_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_weights = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, CONVT_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, CONVT_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, CONVT_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, CONVT_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, CONVT_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, CONVT_STRINGS_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, CONVT_WEIGHTS_OFF};

    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + CONVT_TENSORS_OFF);
    int32_t *shapes = (int32_t *)(buf + CONVT_SHAPES_OFF);

    /* Tensor 0: input, NHWC 1x2x2x1 */
    tensors[0].shape_off = 0;
    tensors[0].ndim = 4;
    tensors[0].dtype = 1;
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[0].size_bytes = 4 * sizeof(float);
    shapes[0] = 1; shapes[1] = 2; shapes[2] = 2; shapes[3] = 1;

    /* Tensor 1: output, NHWC 1x2x2x1. output_padding/spatial derivation is
     * out of scope for this loader case, so the shape need only be rank-4
     * with a matching batch dimension. */
    tensors[1].shape_off = 4;
    tensors[1].ndim = 4;
    tensors[1].dtype = 1;
    tensors[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    tensors[1].size_bytes = 4 * sizeof(float);
    shapes[4] = 1; shapes[5] = 2; shapes[6] = 2; shapes[7] = 1;

    tigris_weight_entry_t *weight =
        (tigris_weight_entry_t *)(buf + CONVT_WEIGHTS_OFF);
    weight->name_str = 0;
    weight->offset = 0;
    weight->size_bytes = sizeof(float);

    tigris_op_t *op = (tigris_op_t *)(buf + CONVT_OPS_OFF);
    op->op_type = TIGRIS_OP_CONV_TRANSPOSE;
    op->num_inputs = 1;
    op->num_outputs = 1;
    op->stage = 0;
    op->inputs_off = 0;
    op->outputs_off = 1;
    op->spatial.kernel_h = 1;
    op->spatial.kernel_w = 1;
    op->spatial.stride_h = 1;
    op->spatial.stride_w = 1;
    op->spatial.dilation_h = 1;
    op->spatial.dilation_w = 1;
    op->spatial.group = 1;
    op->weight_idx = 0;
    op->bias_idx = TIGRIS_NO_WEIGHT;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + CONVT_STAGES_OFF);
    stage->ops_off = 4;
    stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    uint16_t *indices = (uint16_t *)(buf + CONVT_INDEX_OFF);
    indices[0] = 0;   /* op input: tensor 0 */
    indices[1] = 1;   /* op output: tensor 1 */
    indices[2] = 0;   /* model input: tensor 0 */
    indices[3] = 1;   /* model output: tensor 1 */
    indices[4] = 0;   /* stage op list: op 0 */

    buf[CONVT_STRINGS_OFF] = '\0';
}

static void test_conv_transpose_semantic_guards(void)
{
    printf("  test_conv_transpose_semantic_guards...\n");
    _Alignas(4) uint8_t buf[CONVT_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_op_t *op;
    tigris_weight_entry_t *weight;

    build_conv_transpose_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "valid ConvTranspose loads");

    build_conv_transpose_plan(buf);
    op = (tigris_op_t *)(buf + CONVT_OPS_OFF);
    op->spatial.stride_h = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "zero ConvTranspose stride_h rejected");

    build_conv_transpose_plan(buf);
    op = (tigris_op_t *)(buf + CONVT_OPS_OFF);
    op->spatial.stride_w = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "zero ConvTranspose stride_w rejected");

    build_conv_transpose_plan(buf);
    op = (tigris_op_t *)(buf + CONVT_OPS_OFF);
    op->spatial.group = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "unsupported ConvTranspose group rejected");

    build_conv_transpose_plan(buf);
    op = (tigris_op_t *)(buf + CONVT_OPS_OFF);
    op->spatial.dilation_h = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "dilated ConvTranspose rejected");

    build_conv_transpose_plan(buf);
    weight = (tigris_weight_entry_t *)(buf + CONVT_WEIGHTS_OFF);
    weight->size_bytes = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "undersized ConvTranspose weight rejected");
}

/* One-op int8 ConvTranspose plan with an output quant param, exercising the
 * quant_channels_fit guard (mirrors the CONV case's check). Input is NHWC
 * 1x2x2x2 (IC=2), output is NHWC 1x2x2x3 (OC=3), kernel/stride 1x1 so the
 * weight is a plain [OC, KH, KW, IC] = [3,1,1,2] int8 blob (6 bytes). The
 * quant data blob is sized for the largest case (num_channels==OC==3, 3
 * multiplier + 3 shift int32 elements = 24 bytes) and reused unchanged for
 * num_channels==1 and the malformed num_channels==2 case; only the quant
 * param's num_channels/shift_off fields move within that fixed span. */
#define CONVTQ_PLAN_SIZE    328u
#define CONVTQ_TENSORS_OFF  120u
#define CONVTQ_OPS_OFF      152u
#define CONVTQ_STAGES_OFF   190u
#define CONVTQ_INDEX_OFF    218u
#define CONVTQ_SHAPES_OFF   228u
#define CONVTQ_STRINGS_OFF  260u
#define CONVTQ_WEIGHTS_OFF  264u
#define CONVTQ_QUANT_OFF    284u

static void build_conv_transpose_quant_plan(uint8_t *buf, uint16_t num_channels)
{
    memset(buf, 0, CONVTQ_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = CONVTQ_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_weights = 1;
    hdr->num_quant_params = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, CONVTQ_TENSORS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, CONVTQ_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, CONVTQ_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, CONVTQ_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, CONVTQ_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, CONVTQ_STRINGS_OFF};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, CONVTQ_WEIGHTS_OFF};
    dir[7] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, CONVTQ_QUANT_OFF};

    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + CONVTQ_TENSORS_OFF);
    int32_t *shapes = (int32_t *)(buf + CONVTQ_SHAPES_OFF);

    /* Tensor 0: input, int8 NHWC 1x2x2x2 (IC=2), no quant param needed. */
    tensors[0].shape_off = 0;
    tensors[0].ndim = 4;
    tensors[0].dtype = 3;
    tensors[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    tensors[0].size_bytes = 8;
    shapes[0] = 1; shapes[1] = 2; shapes[2] = 2; shapes[3] = 2;

    /* Tensor 1: output, int8 NHWC 1x2x2x3 (OC=3), carries the quant param
     * under test. */
    tensors[1].shape_off = 4;
    tensors[1].ndim = 4;
    tensors[1].dtype = 3;
    tensors[1].quant_param_idx = 0;
    tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    tensors[1].size_bytes = 12;
    shapes[4] = 1; shapes[5] = 2; shapes[6] = 2; shapes[7] = 3;

    tigris_weight_entry_t *weight =
        (tigris_weight_entry_t *)(buf + CONVTQ_WEIGHTS_OFF);
    weight->name_str = 0;
    weight->offset = 0;
    weight->size_bytes = 6; /* OC(3) * KH(1) * KW(1) * IC(2), int8 */

    tigris_op_t *op = (tigris_op_t *)(buf + CONVTQ_OPS_OFF);
    op->op_type = TIGRIS_OP_CONV_TRANSPOSE;
    op->num_inputs = 1;
    op->num_outputs = 1;
    op->stage = 0;
    op->inputs_off = 0;
    op->outputs_off = 1;
    op->spatial.kernel_h = 1;
    op->spatial.kernel_w = 1;
    op->spatial.stride_h = 1;
    op->spatial.stride_w = 1;
    op->spatial.dilation_h = 1;
    op->spatial.dilation_w = 1;
    op->spatial.group = 1;
    op->weight_idx = 0;
    op->bias_idx = TIGRIS_NO_WEIGHT;
    op->act_min = -128;
    op->act_max = 127;

    tigris_stage_t *stage = (tigris_stage_t *)(buf + CONVTQ_STAGES_OFF);
    stage->ops_off = 4;
    stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;

    uint16_t *indices = (uint16_t *)(buf + CONVTQ_INDEX_OFF);
    indices[0] = 0;   /* op input: tensor 0 */
    indices[1] = 1;   /* op output: tensor 1 */
    indices[2] = 0;   /* model input: tensor 0 */
    indices[3] = 1;   /* model output: tensor 1 */
    indices[4] = 0;   /* stage op list: op 0 */

    buf[CONVTQ_STRINGS_OFF] = '\0';

    /* Quant params section: 1 param, 1 v3 page. Data blob is fixed at 6
     * int32 elements (3 multiplier + 3 shift, the OC==3 case) regardless of
     * num_channels so num_channels==1/2/3 all address elements within the
     * validated quant-data span; only quant_channels_fit distinguishes them. */
    uint16_t *quant_header = (uint16_t *)(buf + CONVTQ_QUANT_OFF);
    quant_header[0] = 1; /* num quant params */
    quant_header[1] = 1; /* v3 quant-data page count */
    tigris_quant_param_t *qp =
        (tigris_quant_param_t *)(buf + CONVTQ_QUANT_OFF + 4u);
    qp->scale = 0.5f;
    qp->zero_point = 0;
    qp->num_channels = num_channels;
    qp->multiplier_off = 0;
    qp->shift_off = num_channels;
    qp->_pad = 0; /* page 0 */
    int32_t *quant_data =
        (int32_t *)(buf + CONVTQ_QUANT_OFF + 4u + sizeof(tigris_quant_param_t));
    for (int i = 0; i < 3; i++)
        quant_data[i] = 100; /* multipliers */
    for (int i = 0; i < 3; i++)
        quant_data[3 + i] = 1; /* shifts */
}

static void test_conv_transpose_quant_channels_guard(void)
{
    printf("  test_conv_transpose_quant_channels_guard...\n");
    _Alignas(4) uint8_t buf[CONVTQ_PLAN_SIZE];
    tigris_plan_t plan;

    build_conv_transpose_quant_plan(buf, 3);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "ConvTranspose quant num_channels==OC loads");

    build_conv_transpose_quant_plan(buf, 1);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "ConvTranspose quant num_channels==1 (per-tensor) loads");

    build_conv_transpose_quant_plan(buf, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "ConvTranspose quant num_channels strictly between 1 and OC rejected");
}

/* Struct size validation */

static void test_struct_sizes(void)
{
    printf("  test_struct_sizes...\n");
    TEST_ASSERT_EQ(sizeof(tigris_file_header_t), 48, "header size");
    TEST_ASSERT_EQ(sizeof(tigris_section_entry_t), 8, "section entry size");
    TEST_ASSERT_EQ(sizeof(tigris_tensor_t), 16, "tensor size");
    TEST_ASSERT_EQ(sizeof(tigris_op_t), 38, "op size");
    TEST_ASSERT_EQ(sizeof(tigris_stage_t), 28, "stage size");
    TEST_ASSERT_EQ(sizeof(tigris_tile_plan_t), 24, "tile plan size");
    TEST_ASSERT_EQ(sizeof(tigris_spatial_attrs_t), 18, "spatial attrs size");
    TEST_ASSERT_EQ(sizeof(tigris_weight_entry_t), 12, "weight entry size");
    TEST_ASSERT_EQ(sizeof(tigris_quant_param_t), 16, "quant param size");
    TEST_ASSERT_EQ(sizeof(tigris_op_attribute_t), 8, "op attribute size");
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
    TEST_ASSERT(plan.header->version == TIGRIS_SCHEMA_VERSION ||
                plan.header->version == TIGRIS_SCHEMA_VERSION_V3 ||
                plan.header->version == TIGRIS_SCHEMA_VERSION_V2,
                "supported version");
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

/* The header states how many quantization parameters a plan has and so does
 * the section that holds them. Everything downstream bounds an index against
 * one and reads the array sized by the other, so a header claiming more than
 * the section holds read past the plan: a four-byte out-of-bounds read inside
 * tigris_plan_load, from input the loader exists to police. */
static void test_quant_param_counts_are_reconciled(void)
{
    printf("  test_quant_param_counts_are_reconciled...\n");
    char path[512];
    int path_len = snprintf(path, sizeof(path), "%s/%s",
                            TIGRIS_SCHEMA_COMPAT_DIR,
                            "schema-v3-qdq-conv.tgrs");
    if (path_len <= 0 || (size_t)path_len >= sizeof(path))
        return;
    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    TEST_ASSERT(buf != NULL, "quantized fixture is readable");
    if (!buf)
        return;

    tigris_plan_t plan;
    TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan), TIGRIS_OK,
                   "the fixture loads as emitted");

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    uint16_t present = hdr->num_quant_params;
    TEST_ASSERT(present > 0u, "the fixture carries quantization");
    const tigris_section_entry_t *dir =
        (const tigris_section_entry_t *)(buf + hdr->section_dir_off);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + dir[0].offset);

    /* A header that claims more than the section holds, and a tensor naming
     * an index inside the claim but outside the section. */
    hdr->num_quant_params = 300;
    for (uint16_t i = 0; i < hdr->num_tensors; i++) {
        if (tensors[i].quant_param_idx != TIGRIS_NO_QUANT_PARAM) {
            tensors[i].quant_param_idx = 200;
            break;
        }
    }
    TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a header claiming more quant params than the section "
                   "holds is refused");

    /* The reverse mismatch is no better founded. */
    hdr->num_quant_params = (uint16_t)(present - 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "a header claiming fewer is refused too");

    free(buf);
}

/* Schema v3 predates typed attributes, but its built-in operators already had
 * their current meaning, so a record that no kernel can execute is refused.
 * An opcode the v3 table did not define stays open to custom dispatch. */
static void test_legacy_schema_validates_builtin_operators(void)
{
    printf("  test_legacy_schema_validates_builtin_operators...\n");
    char path[512];
    int path_len = snprintf(path, sizeof(path), "%s/%s", TIGRIS_SCHEMA_COMPAT_DIR,
                            "schema-v3-qdq-conv.tgrs");
    TEST_ASSERT(path_len > 0 && (size_t)path_len < sizeof(path), "fixture path fits");
    uint32_t len = 0;
    uint8_t *buf = load_file(path, &len);
    TEST_ASSERT(buf != NULL, "v3 fixture is readable");
    if (!buf)
        return;

    tigris_plan_t plan;
    TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK, "v3 fixture loads");
    tigris_op_t *conv = NULL;
    for (uint16_t i = 0; i < plan.header->num_ops; i++) {
        if (plan.ops[i].op_type == TIGRIS_OP_CONV)
            conv = (tigris_op_t *)&plan.ops[i];
    }
    TEST_ASSERT(conv != NULL, "v3 fixture has a Conv");
    if (conv) {
        uint16_t group = conv->spatial.group;
        conv->spatial.group = 2;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "v3 Conv with a group no kernel runs is refused");
        conv->spatial.group = group;

        uint8_t op_type = conv->op_type;
        conv->op_type = 200;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK,
                       "v3 opcode outside the built-in table is left to dispatch");
        conv->op_type = op_type;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK,
                       "restored v3 fixture loads");
    }
    free(buf);
}

static void test_schema_compatibility_fixtures(void)
{
    /*
     * Immutable compiler-emitted artifacts, selected to exercise the feature
     * introduced by each supported schema:
     *   v2 linear:   tigris b47664926e7a484d6638ecd0fd372da477236619
     *   v3 QDQ Conv: tigris 0fe37d3a53292cf532ba8628d2284c00a896fbb2
     *   v4 Transpose:tigris 208b322cab7f97c9960c63a8075944374fdfff2c
     *   v5 Conv1D:   explicit serialized length axis
     *   v6 QDQ Conv: a boundary tensor declaring its float model interface
     *   v7 MatMul:   tensors recording whether they are stored in model order
     *   v8 LayerNorm:the typed attribute kinds
     *   v9 QDQ Add:  a quantized sum stating the multipliers that scale its
     *                two operands and its result
     */
    static const struct {
        const char *filename;
        uint32_t version;
        uint16_t quant_params;
        uint16_t op_attributes;
        uint16_t tile_plans;
        uint8_t tile_axis;
    } fixtures[] = {
        {"schema-v2-linear.tgrs", TIGRIS_SCHEMA_VERSION_V2, 0, 0, 0, 0},
        {"schema-v3-qdq-conv.tgrs", TIGRIS_SCHEMA_VERSION_V3, 3, 0, 0, 0},
        {"schema-v4-transpose.tgrs", TIGRIS_SCHEMA_VERSION_V4, 0, 1, 0, 0},
        {"schema-v5-conv1d-axis.tgrs", TIGRIS_SCHEMA_VERSION_V5, 0, 0, 1,
         TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH},
        {"schema-v6-interface-dtype.tgrs", TIGRIS_SCHEMA_VERSION_V6, 3, 0, 0, 0},
        {"schema-v7-tensor-layout.tgrs", TIGRIS_SCHEMA_VERSION_V7, 0, 2, 0, 0},
        {"schema-v8-layer-norm.tgrs", TIGRIS_SCHEMA_VERSION_V8, 0, 3, 0, 0},
        {"schema-v9-binary-requant.tgrs", TIGRIS_SCHEMA_VERSION_V9, 3, 1, 0, 0},
    };
    const size_t fixture_count = sizeof(fixtures) / sizeof(fixtures[0]);

    printf("  test_schema_compatibility_fixtures...\n");
    TEST_ASSERT_EQ(fixture_count,
                   TIGRIS_SCHEMA_VERSION - TIGRIS_SCHEMA_VERSION_MIN + 1u,
                   "every supported schema has a fixed fixture");
    for (size_t i = 0; i < fixture_count; i++) {
        TEST_ASSERT_EQ(fixtures[i].version,
                       TIGRIS_SCHEMA_VERSION_MIN + i,
                       "schema fixture versions are contiguous");
        char path[512];
        int path_len = snprintf(path, sizeof(path), "%s/%s",
                                TIGRIS_SCHEMA_COMPAT_DIR,
                                fixtures[i].filename);
        TEST_ASSERT(path_len > 0 && (size_t)path_len < sizeof(path),
                    "schema fixture path fits");
        if (path_len <= 0 || (size_t)path_len >= sizeof(path))
            continue;

        uint32_t buf_len = 0;
        uint8_t *buf = load_file(path, &buf_len);
        TEST_ASSERT(buf != NULL, "schema fixture is readable");
        if (!buf)
            continue;

        tigris_plan_t plan;
        tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
        TEST_ASSERT_EQ(err, TIGRIS_OK, "schema fixture loads");
        if (err == TIGRIS_OK) {
            TEST_ASSERT_EQ(plan.header->version, fixtures[i].version,
                           "schema fixture has expected version");
            TEST_ASSERT_EQ(plan.header->num_quant_params,
                           fixtures[i].quant_params,
                           "schema fixture has expected quant metadata");
            TEST_ASSERT_EQ(plan.num_op_attributes, fixtures[i].op_attributes,
                           "schema fixture has expected operator attributes");
            TEST_ASSERT_EQ(plan.header->num_tile_plans,
                           fixtures[i].tile_plans,
                           "schema fixture has expected tile plans");
            if (fixtures[i].version == TIGRIS_SCHEMA_VERSION_INTERFACE_DTYPE) {
                /* The quantized boundary states the float interface the model
                 * declares, which is what schema 6 added. Held to that one
                 * fixture: a later schema's exemplar need not be quantized. */
                for (uint8_t m = 0; m < plan.header->num_model_inputs; m++) {
                    const tigris_tensor_t *t =
                        &plan.tensors[plan.model_inputs[m]];
                    TEST_ASSERT_EQ(t->dtype, 3, "v6 boundary stores int8");
                    TEST_ASSERT_EQ(t->iface_dtype, 1,
                                   "v6 boundary declares float32");
                }
            }
            if (fixtures[i].version == TIGRIS_SCHEMA_VERSION_TENSOR_LAYOUT) {
                /* Schema 7 records whether a tensor is stored in the model's
                 * own axis order. Internal tensors of this matrix product are;
                 * its boundaries keep the channels-last convention callers
                 * already rely on, which is the distinction the flag exists to
                 * make readable. */
                uint16_t linear = 0;
                for (uint16_t t = 0; t < plan.header->num_tensors; t++) {
                    if ((plan.tensors[t].flags & TIGRIS_TENSOR_LINEAR) != 0u)
                        linear++;
                }
                TEST_ASSERT(linear > 0,
                            "v7 fixture records linear tensors");
                for (uint8_t m = 0; m < plan.header->num_model_inputs; m++) {
                    TEST_ASSERT_EQ(
                        plan.tensors[plan.model_inputs[m]].flags &
                            TIGRIS_TENSOR_LINEAR,
                        0, "v7 model input keeps the channels-last convention");
                }
                for (uint8_t m = 0; m < plan.header->num_model_outputs; m++) {
                    TEST_ASSERT_EQ(
                        plan.tensors[plan.model_outputs[m]].flags &
                            TIGRIS_TENSOR_LINEAR,
                        0, "v7 model output keeps the channels-last convention");
                }
            }
            if (fixtures[i].tile_plans > 0) {
                TEST_ASSERT_EQ(plan.tile_plans[0].axis,
                               fixtures[i].tile_axis,
                               "schema fixture has expected explicit tile axis");

                /* Exercise every rank-3 tile allow-list branch without
                 * introducing another generated fixture. The Conv1D record's
                 * weight metadata makes these opcode-only mutations invalid
                 * operator records after their tile classification. */
                size_t op_offset = (const uint8_t *)plan.ops - buf;
                tigris_op_t *op = (tigris_op_t *)(buf + op_offset);
                const uint8_t mutations[] = {
                    TIGRIS_OP_RELU,
                    TIGRIS_OP_RELU6,
                    TIGRIS_OP_SIGMOID,
                    TIGRIS_OP_TANH,
                    TIGRIS_OP_ADD,
                    TIGRIS_OP_MUL,
                    TIGRIS_OP_CONCAT,
                };
                for (size_t m = 0;
                     m < sizeof(mutations) / sizeof(mutations[0]); m++) {
                    op->op_type = mutations[m];
                    TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan),
                                   TIGRIS_ERR_BAD_OPERATOR,
                                   "malformed rank-3 tiled opcode fails closed");
                }
                op->op_type = TIGRIS_OP_CONV1D;
                TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan),
                               TIGRIS_OK,
                               "restored rank-3 Conv1D fixture loads");

                size_t tile_offset =
                    (const uint8_t *)plan.tile_plans - buf;
                ((tigris_tile_plan_t *)(buf + tile_offset))->axis =
                    TIGRIS_TILE_AXIS_WIDTH;
                TEST_ASSERT_EQ(tigris_plan_load(buf, buf_len, &plan),
                               TIGRIS_ERR_BAD_SECTION,
                               "unsupported v5 tile axis fails closed");
            }
        }
        free(buf);
    }
}

/* Main */

#define MOVEMENT_PLAN_SIZE 1280u

static void build_movement_plan(uint8_t *buf, const movement_golden_t *gold)
{
    memset(buf, 0, MOVEMENT_PLAN_SIZE);
    tigris_file_header_t *h = (tigris_file_header_t *)buf;
    memcpy(h->magic, TIGRIS_MAGIC_BYTES, 4);
    h->version = TIGRIS_SCHEMA_VERSION;
    h->file_size = MOVEMENT_PLAN_SIZE;
    h->section_dir_off = sizeof(*h);
    h->num_tensors = 3;
    h->num_ops = h->num_stages = 1;
    h->num_weights = gold->index_count != 0u ? 1 : 0;
    h->num_quant_params = gold->dtype == 3 ? 3 : 0;
    h->num_model_inputs = gold->update_count != 0u ? 2 : 1;
    h->num_model_outputs = 1;
    h->model_io_off = 4;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + h->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, 144};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, 192};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, 232};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, 260};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, 288};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, 360};
    uint8_t section = 6;
    if (h->num_weights != 0u) dir[section++] = (tigris_section_entry_t){TIGRIS_SEC_WEIGHTS, 368};
    if (h->num_quant_params != 0u) dir[section++] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, 1024};
    dir[section] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, 1152};
    tigris_tensor_t *t = (tigris_tensor_t *)(buf + 144);
    uint32_t counts[] = {gold->input_count, gold->update_count ? gold->update_count : 1u, gold->output_count};
    uint8_t ranks[] = {gold->rank, gold->update_rank ? gold->update_rank : 1u, gold->output_rank};
    for (int i = 0; i < 3; i++) {
        t[i].shape_off = (uint32_t)i * 6u;
        t[i].ndim = ranks[i];
        t[i].dtype = gold->dtype;
        t[i].quant_param_idx = gold->dtype == 3 ? (uint16_t)i : TIGRIS_NO_QUANT_PARAM;
        t[i].size_bytes = counts[i] * tigris_dtype_size(gold->dtype);
    }
    t[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    if (h->num_model_inputs == 2) t[1].flags = TIGRIS_TENSOR_MODEL_INPUT;
    t[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    memcpy(buf + 288, gold->shapes, sizeof(gold->shapes));
    tigris_op_t *op = (tigris_op_t *)(buf + 192);
    op->op_type = gold->op;
    op->num_inputs = h->num_model_inputs;
    op->num_outputs = 1;
    op->outputs_off = 2;
    op->weight_idx = h->num_weights ? 0 : TIGRIS_NO_WEIGHT;
    op->bias_idx = TIGRIS_NO_WEIGHT;
    op->act_min = -128; op->act_max = 127;
    tigris_stage_t *stage = (tigris_stage_t *)(buf + 232);
    stage->ops_off = 3; stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;
    uint16_t *indices = (uint16_t *)(buf + 260);
    indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 0;
    indices[4] = 0; indices[5] = h->num_model_inputs == 2 ? 1 : 2; indices[6] = 2;
    if (h->num_weights) {
        tigris_weight_entry_t *weight = (tigris_weight_entry_t *)(buf + 368);
        weight->size_bytes = gold->index_count * 4u;
        memcpy(buf + 368 + sizeof(*weight), gold->indices, weight->size_bytes);
    }
    if (h->num_quant_params) {
        uint16_t *qh = (uint16_t *)(buf + 1024);
        qh[0] = 3; qh[1] = 1;
        tigris_quant_param_t *q = (tigris_quant_param_t *)(buf + 1028);
        for (int i = 0; i < 3; i++) {
            q[i].scale = 0.125f; q[i].zero_point = -17;
            q[i].num_channels = 1; q[i].shift_off = 1;
        }
    }
    uint16_t count = 1;
    memcpy(buf + 1152, &count, sizeof(count));
    tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + 1156);
    attr->type = TIGRIS_OP_ATTR_MOVEMENT;
    attr->data_len = (uint8_t)(gold->metadata_count * 4u);
    memcpy(buf + 1164, gold->metadata, attr->data_len);
}

static void test_movement_contract(void)
{
    _Alignas(4) uint8_t buf[MOVEMENT_PLAN_SIZE];
    tigris_plan_t plan;
    for (size_t g = 0; g < sizeof(movement_goldens) / sizeof(movement_goldens[0]); g++) {
        build_movement_plan(buf, &movement_goldens[g]);
        tigris_error_t error = tigris_plan_load(buf, sizeof(buf), &plan);
        if (error != TIGRIS_OK) fprintf(stderr, "movement loader case %zu: %d\n", g, error);
        TEST_ASSERT_EQ(error, TIGRIS_OK, "movement reference metadata loads");
        tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + 1156);
        attr->data_len--;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "truncated movement metadata refused");
        attr->data_len++;
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + 144);
        t[2].dtype = 9;
        t[2].size_bytes = movement_goldens[g].output_count;
        t[2].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "movement cannot produce bool");
        build_movement_plan(buf, &movement_goldens[g]);
        int32_t invalid = -99;
        memcpy(buf + 1164, &invalid, sizeof(invalid));
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "invalid movement metadata refused");
        if (movement_goldens[g].dtype == 3) {
            build_movement_plan(buf, &movement_goldens[g]);
            ((tigris_quant_param_t *)(buf + 1028))[2].scale = 0.25f;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "movement cannot rescale int8");
        }
    }
}

#define BOOL_PLAN_SIZE 2048u

static void build_bool_plan(uint8_t *buf, const bool_golden_t *gold)
{
    memset(buf, 0, BOOL_PLAN_SIZE);
    tigris_file_header_t *h = (tigris_file_header_t *)buf;
    memcpy(h->magic, TIGRIS_MAGIC_BYTES, 4);
    h->version = TIGRIS_SCHEMA_VERSION;
    h->file_size = BOOL_PLAN_SIZE;
    h->section_dir_off = sizeof(*h);
    h->num_tensors = gold->inputs + 1u;
    h->num_ops = h->num_stages = 1;
    h->num_model_inputs = gold->inputs;
    h->num_model_outputs = 1;
    h->model_io_off = 40;
    h->num_quant_params = 4;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + h->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, 160};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, 240};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, 280};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, 320};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, 448};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, 528};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, 544};
    const int compare_s8 = gold->op >= TIGRIS_OP_EQUAL && gold->op <= TIGRIS_OP_GREATER_EQUAL && gold->dtypes[0] == 3;
    if (compare_s8) dir[7] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, 672};
    tigris_tensor_t *t = (tigris_tensor_t *)(buf + 160);
    int32_t *shapes = (int32_t *)(buf + 448);
    uint16_t *indices = (uint16_t *)(buf + 320);
    for (uint8_t i = 0; i < h->num_tensors; i++) {
        const uint8_t slot = i == gold->inputs ? 3u : i;
        t[i].shape_off = (uint32_t)i * 5u;
        t[i].ndim = gold->ranks[slot]; t[i].dtype = gold->dtypes[slot];
        t[i].size_bytes = gold->sizes[slot];
        t[i].quant_param_idx = t[i].dtype == 3u ? slot : TIGRIS_NO_QUANT_PARAM;
        t[i].flags = i == gold->inputs ? TIGRIS_TENSOR_MODEL_OUTPUT : TIGRIS_TENSOR_MODEL_INPUT;
        memcpy(shapes + i * 5u, gold->shapes + slot * 5u, 5u * sizeof(int32_t));
        indices[i] = i;
        indices[40u + i] = i;
    }
    tigris_op_t *op = (tigris_op_t *)(buf + 240);
    op->op_type = gold->op; op->num_inputs = gold->inputs; op->num_outputs = 1;
    op->outputs_off = gold->inputs;
    op->weight_idx = op->bias_idx = TIGRIS_NO_WEIGHT;
    op->act_min = -128; op->act_max = 127;
    tigris_stage_t *stage = (tigris_stage_t *)(buf + 280);
    stage->ops_off = 39; stage->ops_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN; stage->chain_id = TIGRIS_NO_CHAIN;
    uint16_t *qh = (uint16_t *)(buf + 544);
    qh[0] = 4; qh[1] = 1;
    tigris_quant_param_t *q = (tigris_quant_param_t *)(buf + 548);
    for (uint8_t i = 0; i < 4; i++) {
        q[i].scale = gold->scales[i]; q[i].zero_point = gold->zeros[i];
        q[i].num_channels = 1; q[i].shift_off = 1;
    }
    if (compare_s8) {
        uint16_t n = 1;
        memcpy(buf + 672, &n, sizeof(n));
        tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + 676);
        attr->type = TIGRIS_OP_ATTR_COMPARISON_REQUANT; attr->data_len = 20;
        memcpy(buf + 684, gold->requant, 20);
    }
}

static void test_bool_contract(void)
{
    _Alignas(4) uint8_t buf[BOOL_PLAN_SIZE];
    tigris_plan_t plan;
    for (size_t g = 0; g < sizeof(bool_goldens) / sizeof(bool_goldens[0]); g++) {
        const bool_golden_t *gold = &bool_goldens[g];
        build_bool_plan(buf, gold);
        tigris_error_t error = tigris_plan_load(buf, sizeof(buf), &plan);
        if (error != TIGRIS_OK) fprintf(stderr, "bool loader case %zu: %d\n", g, error);
        TEST_ASSERT_EQ(error, TIGRIS_OK, "bool reference metadata loads");
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + 160);
        tigris_op_t *op = (tigris_op_t *)(buf + 240);
        tigris_quant_param_t *q = (tigris_quant_param_t *)(buf + 548);
        const uint8_t out = gold->inputs;
        if (gold->ranks[3] == 0u && gold->dtypes[0] != 3u && gold->dtypes[3] != 3u) {
            tigris_file_header_t *h = (tigris_file_header_t *)buf;
            tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + h->section_dir_off);
            h->num_quant_params = 0; h->file_size = 531;
            dir[4].offset = 528;
            dir[6] = (tigris_section_entry_t){0, 0};
            for (uint8_t i = 0; i < h->num_tensors; i++) t[i].shape_off = 0;
            TEST_ASSERT_EQ(tigris_plan_load(buf, h->file_size, &plan), TIGRIS_OK,
                           "scalar plan permits an empty shape pool before an odd-sized string table");
            build_bool_plan(buf, gold);
        }
        t[out].dtype = 6; t[out].size_bytes = gold->sizes[3] / tigris_dtype_size(gold->dtypes[3]) * 4u;
        t[out].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "bool operator refuses int32 output");
        build_bool_plan(buf, gold);
        op->fused_act = TIGRIS_ACT_RELU;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "bool operator refuses fused activation");
        build_bool_plan(buf, gold);
        if (gold->op <= TIGRIS_OP_GREATER_EQUAL && gold->dtypes[0] == 3) {
            q[0].scale = 1.0f;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "comparison refuses unit scale");
            build_bool_plan(buf, gold);
            ((int32_t *)(buf + 684))[0] = 7;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "comparison refuses invalid left shift");
            build_bool_plan(buf, gold);
            ((tigris_op_attribute_t *)(buf + 676))->data_len = 16;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "comparison refuses short requantization");
        }
        if (gold->op == TIGRIS_OP_SELECT_V2 && gold->dtypes[3] == 3) {
            q[3].scale *= 2.0f;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "selection preserves encoding");
        }
        if (gold->op == TIGRIS_OP_CAST && gold->dtypes[3] == 3) {
            q[3].scale = 0.25f;
            q[3].zero_point = -3;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "cast quantizes to any int8 encoding");
            q[3].scale = 0.0f;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "cast refuses a zero scale");
        }
        if (gold->op == TIGRIS_OP_ADD_N && gold->dtypes[0] == 3) {
            q[1].scale *= 2.0f;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "sum requires equal input scales");
            build_bool_plan(buf, gold);
            uint16_t *indices = (uint16_t *)(buf + 320);
            indices[36] = out; op->outputs_off = 36;
            for (uint8_t i = 0; i < 33; i++) indices[i] = 0;
            q[0].zero_point = -128;
            op->num_inputs = 15;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "15 extreme inputs have a defined accumulator");
            op->num_inputs = 16;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "16 extreme inputs overflow reference accumulator");
            q[0].zero_point = 0; op->num_inputs = 32;
            TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "32 centered inputs have a defined accumulator");
            op->num_inputs = 33;
            TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "33 centered inputs overflow reference accumulator");
        }
    }
}

int main(int argc, char *argv[])
{
    test_movement_contract();
    test_bool_contract();
    printf("TiGrIS Plan Loader Tests\n\n");

    printf("Error handling tests:\n");
    test_null_args();
    test_too_small();
    test_bad_magic();
    test_bad_version();
    test_size_mismatch();
    test_error_strings();
    test_section_directory_guards();
    test_pool_alignment_guards();
    test_v2_plan_header_is_still_accepted();
    test_v2_quant_plan_is_still_accepted();
    test_counted_sections_required();
    test_chain_limit_guards();
    test_stage_schedule_guards();
    test_schema_v5_many_stage_schedule();
    test_compressed_block_guards();
    test_cross_reference_guards();
    test_transpose_attribute_guards();
    test_op_attribute_kinds();
    test_declared_interface_dtype_contract();
    test_a_refused_plan_names_the_build_it_wants();
    test_split_contract();
    test_reduce_mean_contract();
    test_op_attribute_payloads();
    test_resize_scale_guards();
    test_tiled_conversion_contract();
    test_tiled_reduction_contract();
    test_rank4_row_band_contract();
    test_operator_semantic_guards();
    test_nonweighted_operator_semantics();
    test_elementwise_semantics();
    test_native_reduction_contract();
    test_arg_contract();
    test_reduce_all_contract();
    test_cumsum_loader_contract();
    test_native_reduction_tiling_contract();
    test_activation_attribute_contract();
    test_activation_semantic_contract();
    test_prelu_loader_contract();
    test_prelu_tiling_contract();
    test_activation_tiling_contract();
    test_convolution_semantic_guards();
    test_conv_transpose_semantic_guards();
    test_conv_transpose_quant_channels_guard();

    printf("\nSchema compatibility fixtures:\n");
    test_schema_compatibility_fixtures();
    test_legacy_schema_validates_builtin_operators();
    test_quant_param_counts_are_reconciled();

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

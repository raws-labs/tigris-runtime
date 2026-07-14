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
#define STAGED_OPS_OFF   128u
#define STAGED_STAGES_OFF 480u
#define STAGED_INDEX_OFF 544u
#define STAGED_SHAPES_OFF 640u
#define STAGED_STRINGS_OFF 644u

static void build_staged_plan(uint8_t *buf)
{
    memset(buf, 0, STAGED_PLAN_SIZE);

    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = STAGED_PLAN_SIZE;
    hdr->section_dir_off = sizeof(*hdr);
    hdr->num_ops = 9;
    hdr->num_stages = 2;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, STAGED_OPS_OFF};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, STAGED_OPS_OFF};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, STAGED_STAGES_OFF};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, STAGED_INDEX_OFF};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, STAGED_SHAPES_OFF};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, STAGED_STRINGS_OFF};

    tigris_op_t *ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    uint16_t *indices = (uint16_t *)(buf + STAGED_INDEX_OFF);
    for (uint16_t i = 0; i < hdr->num_ops; i++) {
        ops[i].op_type = TIGRIS_OP_CONV;
        ops[i].weight_idx = TIGRIS_NO_WEIGHT;
        ops[i].bias_idx = TIGRIS_NO_WEIGHT;
        indices[i] = i;
    }

    tigris_stage_t *stages = (tigris_stage_t *)(buf + STAGED_STAGES_OFF);
    stages[0].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stages[0].chain_id = TIGRIS_NO_CHAIN;
    stages[1].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stages[1].chain_id = TIGRIS_NO_CHAIN;
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
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "eight chain spatial ops fit fixed metadata");

    stages[0].ops_count = 9;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_PLAN_LIMITS, "ninth chain spatial op rejected");
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
                   TIGRIS_ERR_BAD_SECTION, "tensor shape range");

    build_referenced_plan(buf);
    ((tigris_tensor_t *)(buf + REF_TENSORS_OFF))->name_str = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION, "tensor string range");
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
    test_section_directory_guards();
    test_v2_plan_header_is_still_accepted();
    test_v2_quant_plan_is_still_accepted();
    test_counted_sections_required();
    test_chain_limit_guards();
    test_compressed_block_guards();
    test_cross_reference_guards();

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

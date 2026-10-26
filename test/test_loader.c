/**
 * @file test_loader.c
 * @brief Native POSIX tests for the tigris plan loader.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "movement_golden.h"
#include "bool_golden.h"
#include "state_golden.h"
#include "tigris_loader.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_iface.h"
#include "tigris_binary_operands.h"
#include "test_harness.h"

/* Error tests (no fixtures needed) */

#include "trace_assert.h"

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
    TEST_ASSERT_EQ(TIGRIS_ERR_KERNEL_DISABLED, -13, "existing codes keep their values");
    TEST_ASSERT_EQ(TIGRIS_ERR_MEM, -14, "memory code follows the last one");
    TEST_ASSERT(strcmp(tigris_error_str(TIGRIS_ERR_MEM),
                       tigris_error_str((tigris_error_t)999)) != 0, "memory string");
    TEST_ASSERT_EQ(TIGRIS_ERR_TENSOR_SLOTS, -15, "tensor-slot code follows the memory code");
    TEST_ASSERT(strcmp(tigris_error_str(TIGRIS_ERR_TENSOR_SLOTS),
                       tigris_error_str((tigris_error_t)999)) != 0, "tensor-slot string");
    TEST_ASSERT_EQ(TIGRIS_ERR_TILE_FACTS, -16, "tile-facts code follows the tensor-slot code");
    TEST_ASSERT(strcmp(tigris_error_str(TIGRIS_ERR_TILE_FACTS),
                       tigris_error_str((tigris_error_t)999)) != 0, "tile-facts string");
}

/* Minimal self-contained plans for malformed-input tests. */

#define MINIMAL_PLAN_SIZE 112u

typedef struct {
    tigris_file_header_t *header;
    tigris_section_entry_t *sections;
} plan_builder_t;

static plan_builder_t plan_builder_begin(uint8_t *buf, uint32_t size, uint16_t version)
{
    memset(buf, 0, size);
    plan_builder_t builder;
    builder.header = (tigris_file_header_t *)buf;
    memcpy(builder.header->magic, TIGRIS_MAGIC_BYTES, 4);
    builder.header->version = version;
    builder.header->file_size = size;
    builder.header->section_dir_off = sizeof(*builder.header);
    builder.sections = (tigris_section_entry_t *)(buf + builder.header->section_dir_off);
    return builder;
}

static void plan_builder_section(plan_builder_t *builder, uint32_t index,
                                 uint32_t type, uint32_t offset)
{
    builder->sections[index].type = type;
    builder->sections[index].offset = offset;
}

/* Ends a current-schema plan: a memory record of peak 0 in its last 4 bytes, entered last. */
static void plan_finish(uint8_t *buf)
{
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    uint32_t i = 0;
    while (dir[i].type != 0u && dir[i].type != TIGRIS_SEC_MEMORY)
        i++;
    dir[i].type = TIGRIS_SEC_MEMORY;
    dir[i].offset = header->file_size - (uint32_t)sizeof(tigris_memory_t);
    memset(buf + dir[i].offset, 0, sizeof(tigris_memory_t));
}

/* Removes a directory entry; the later ones, the memory entry among them, move up. */
static void plan_drop_section(uint8_t *buf, uint32_t index)
{
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    for (uint32_t i = index; dir[i].type != 0u; i++)
        dir[i] = dir[i + 1u];
}

/* Rewrites a plan's schema; below 11 the memory entry, the last one, becomes the sentinel. */
static void plan_set_version(uint8_t *buf, uint32_t version)
{
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    header->version = version;
    if (version >= TIGRIS_SCHEMA_VERSION_V11)
        return;
    for (uint32_t i = 0; dir[i].type != 0u; i++)
        if (dir[i].type == TIGRIS_SEC_MEMORY)
            memset(&dir[i], 0, sizeof(dir[i]));
}

#define MINIMAL_POOLS_OFF 105u

static void build_minimal_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, MINIMAL_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    const uint32_t required[] = {
        TIGRIS_SEC_TENSORS,
        TIGRIS_SEC_OPS,
        TIGRIS_SEC_INDEX_POOL,
        TIGRIS_SEC_SHAPE_POOL,
        TIGRIS_SEC_STRINGS,
    };
    for (uint32_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        plan_builder_section(&builder, i, required[i], MINIMAL_POOLS_OFF);
    }
    buf[MINIMAL_POOLS_OFF] = '\0';
    plan_finish(buf);
    /* Section 6 remains the zero-valued sentinel. */
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
    dir[6] = dir[5];
    dir[5].type = TIGRIS_SEC_WEIGHTS;
    dir[5].offset = MINIMAL_POOLS_OFF;
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

static void test_memory_section(void)
{
    printf("  test_memory_section...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
    const uint32_t peak = 192u;

    build_minimal_plan(buf);
    ((tigris_file_header_t *)buf)->budget = 256u;
    memcpy(buf + dir[5].offset, &peak, sizeof(peak));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "schema 11 plan with its memory record loads");
    TEST_ASSERT_EQ(plan.scheduled_peak, 192u, "scheduled peak read from the plan");
    TEST_ASSERT_EQ(tigris_fast_arena_required(&plan), 192u,
                   "requirement is the scheduled peak, not the budget");

    build_minimal_plan(buf);
    memset(&dir[5], 0, sizeof(dir[5]));
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_MISSING_SEC,
                   "schema 11 requires the memory section");

    build_minimal_plan(buf);
    ((tigris_file_header_t *)buf)->version = TIGRIS_SCHEMA_VERSION_V10;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "an older plan carries no memory section");

    build_minimal_plan(buf);
    dir[5].offset = MINIMAL_PLAN_SIZE - 2u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "a truncated memory record is refused");

    build_minimal_plan(buf);
    ((tigris_file_header_t *)buf)->budget = 256u;
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V10);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "an older plan loads without the section");
    TEST_ASSERT_EQ(plan.scheduled_peak, 0u, "an older plan states no scheduled peak");
    TEST_ASSERT_EQ(tigris_fast_arena_required(&plan), 256u,
                   "an older plan's requirement stays its budget");
}

static void test_pool_alignment_guards(void)
{
    printf("  test_pool_alignment_guards...\n");
    _Alignas(4) uint8_t buf[MINIMAL_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_file_header_t *hdr;

    /* The minimal plan deliberately puts all empty pools at byte 105.  They
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

/* Seven directory entries and the sentinel end at 112, where the empty pools
 * sit; the quant section follows, and the memory record ends the plan. */
#define QUANT_PLAN_POOLS_OFF 112u
#define QUANT_PLAN_QUANT_OFF 116u
#define QUANT_PLAN_SIZE 148u

static void build_quant_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, QUANT_PLAN_SIZE,
                                                TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_quant_params = 1;

    const uint32_t required[] = {
        TIGRIS_SEC_TENSORS,
        TIGRIS_SEC_OPS,
        TIGRIS_SEC_INDEX_POOL,
        TIGRIS_SEC_SHAPE_POOL,
        TIGRIS_SEC_STRINGS,
    };
    for (uint32_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        plan_builder_section(&builder, i, required[i], QUANT_PLAN_POOLS_OFF);
    }
    plan_builder_section(&builder, 5u, TIGRIS_SEC_QUANT_PARAMS, QUANT_PLAN_QUANT_OFF);
    plan_finish(buf);

    uint16_t *quant_header = (uint16_t *)(buf + QUANT_PLAN_QUANT_OFF);
    quant_header[0] = 1;  /* num params */
    quant_header[1] = 1;  /* quant-data pages */
    tigris_quant_param_t *qp = (tigris_quant_param_t *)(buf + QUANT_PLAN_QUANT_OFF + 4u);
    qp->scale = 0.25f;
    qp->num_channels = 1;
    qp->multiplier_off = 0;
    qp->shift_off = 1;
    int32_t *quant_data = (int32_t *)(buf + QUANT_PLAN_QUANT_OFF + 4u + sizeof(*qp));
    quant_data[0] = 123;
    quant_data[1] = -7;
}

static void test_quant_plan_tables(void)
{
    printf("  test_quant_plan_tables...\n");
    _Alignas(4) uint8_t buf[QUANT_PLAN_SIZE];
    tigris_plan_t plan;

    build_quant_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "quantized plan loads");
    TEST_ASSERT_EQ(plan.num_quant_params, 1, "quant param count");
    TEST_ASSERT_EQ(plan.quant_data[0], 123, "multiplier data");
    TEST_ASSERT_EQ(plan.quant_data[1], -7, "shift data");

    build_quant_plan(buf);
    ((tigris_quant_param_t *)(buf + QUANT_PLAN_QUANT_OFF + 4u))->scale = 0.0f;
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

#define STAGED_PLAN_SIZE 724u
#define STAGED_TENSORS_OFF 128u
#define STAGED_OPS_OFF   144u
#define STAGED_STAGES_OFF 496u
#define STAGED_INDEX_OFF 560u
#define STAGED_SHAPES_OFF 656u
#define STAGED_STRINGS_OFF 672u
#define STAGED_WEIGHTS_OFF 676u

static void build_staged_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, STAGED_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 1;
    hdr->num_ops = 9;
    hdr->num_stages = 2;
    hdr->num_weights = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, STAGED_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, STAGED_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, STAGED_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, STAGED_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, STAGED_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, STAGED_STRINGS_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_WEIGHTS, STAGED_WEIGHTS_OFF);
    plan_finish(buf);

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

/* Stages of one Relu each, stage i on its own tensor i of the given rank. With
 * graph_count > 0 the plan has a subgraph section whose graph g starts at
 * stage graph_first[g]; no operator runs the subgraphs, as the loader needs
 * none to check their stage ranges. */
static uint8_t *build_chain_plan(uint16_t stage_count, uint8_t ndim, const uint16_t *graph_first,
                                 uint16_t graph_count, uint32_t *out_size)
{
    const uint32_t tensors_off = 128u;
    const uint32_t ops_off = tensors_off + (uint32_t)stage_count * sizeof(tigris_tensor_t);
    const uint32_t stages_off = align4(ops_off + (uint32_t)stage_count * sizeof(tigris_op_t));
    const uint32_t index_off = stages_off + (uint32_t)stage_count * sizeof(tigris_stage_t);
    const uint32_t shapes_off = align4(index_off + 5u * stage_count * (uint32_t)sizeof(uint16_t));
    const uint32_t strings_off = shapes_off + 4u * stage_count * (uint32_t)sizeof(int32_t);
    const uint32_t graphs_off = align4(strings_off + 1u);
    const uint32_t plan_size = graphs_off +
        (graph_count ? 4u + (uint32_t)graph_count * sizeof(tigris_subgraph_t) : 0u) +
        (uint32_t)sizeof(tigris_memory_t);
    uint8_t *buf = calloc(1, plan_size);
    if (!buf)
        return NULL;

    plan_builder_t builder = plan_builder_begin(buf, plan_size, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = stage_count;
    hdr->num_ops = stage_count;
    hdr->num_stages = stage_count;
    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, tensors_off);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, ops_off);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, stages_off);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, index_off);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, shapes_off);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, strings_off);
    if (graph_count)
        plan_builder_section(&builder, 6u, TIGRIS_SEC_SUBGRAPHS, graphs_off);
    plan_finish(buf);

    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + tensors_off);
    tigris_op_t *ops = (tigris_op_t *)(buf + ops_off);
    tigris_stage_t *stages = (tigris_stage_t *)(buf + stages_off);
    uint16_t *indices = (uint16_t *)(buf + index_off);
    int32_t *shapes = (int32_t *)(buf + shapes_off);
    for (uint16_t i = 0; i < stage_count; i++) {
        for (uint8_t d = 0; d < 4u; d++)
            shapes[4u * i + d] = d == 0u ? 1 : 2;
        tensors[i].size_bytes = (ndim == 4u ? 8u : 4u) * (uint32_t)sizeof(float);
        tensors[i].shape_off = (uint16_t)(4u * i);
        tensors[i].ndim = ndim;
        tensors[i].dtype = 1;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        ops[i].op_type = TIGRIS_OP_RELU;
        ops[i].num_inputs = 1;
        ops[i].num_outputs = 1;
        ops[i].stage = (uint8_t)i;
        ops[i].inputs_off = (uint16_t)(5u * i);
        ops[i].outputs_off = (uint16_t)(5u * i + 1u);
        ops[i].weight_idx = TIGRIS_NO_WEIGHT;
        ops[i].bias_idx = TIGRIS_NO_WEIGHT;
        stages[i].ops_off = (uint16_t)(5u * i + 2u);
        stages[i].ops_count = 1;
        stages[i].inputs_off = (uint16_t)(5u * i + 3u);
        stages[i].inputs_count = 1;
        stages[i].outputs_off = (uint16_t)(5u * i + 4u);
        stages[i].outputs_count = 1;
        stages[i].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        stages[i].chain_id = TIGRIS_NO_CHAIN;
        indices[5u * i] = i;
        indices[5u * i + 1u] = i;
        indices[5u * i + 2u] = i;
        indices[5u * i + 3u] = i;
        indices[5u * i + 4u] = i;
    }
    if (graph_count) {
        const uint16_t count = graph_count;
        memcpy(buf + graphs_off, &count, sizeof(count));
        tigris_subgraph_t *graphs = (tigris_subgraph_t *)(buf + graphs_off + 4u);
        for (uint16_t g = 0; g < graph_count; g++) {
            uint16_t end = g + 1u < graph_count ? graph_first[g + 1u] : stage_count;
            graphs[g].first_stage = graph_first[g];
            graphs[g].num_stages = (uint16_t)(end - graph_first[g]);
        }
    }
    *out_size = plan_size;
    return buf;
}

/* Marks stages head .. head + len - 1 of a build_chain_plan plan as one chain. */
static void chain_plan_link(uint8_t *buf, uint16_t head, uint16_t len)
{
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    tigris_stage_t *stages = (tigris_stage_t *)(buf + dir[2].offset);
    for (uint16_t s = head; s < head + len; s++) {
        stages[s].chain_id = head;
        stages[s].chain_len = len;
    }
    stages[head].chain_tile_h = 1;
}

static void test_chain_shape_guards(void)
{
    printf("  test_chain_shape_guards...\n");
    tigris_plan_t plan;
    uint32_t size = 0;
    uint8_t *buf = build_chain_plan(2, 4u, NULL, 0, &size);
    TEST_ASSERT(buf != NULL, "chain plan allocation");
    if (!buf)
        return;
    chain_plan_link(buf, 0, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "rank-4 chain loads");
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + dir[0].offset);
    tensors[1].ndim = 3;
    tensors[1].size_bytes = 4u * (uint32_t)sizeof(float);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_ERR_BAD_SECTION,
                   "a chain member's rank-3 tensor is refused");
    free(buf);

    buf = build_chain_plan(2, 3u, NULL, 0, &size);
    TEST_ASSERT(buf != NULL, "rank-3 plan allocation");
    if (!buf)
        return;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "rank-3 stages load unchained");
    chain_plan_link(buf, 0, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_ERR_BAD_SECTION,
                   "a rank-3 chain is refused");
    free(buf);
}

static void test_chain_graph_guards(void)
{
    printf("  test_chain_graph_guards...\n");
    tigris_plan_t plan;
    uint32_t size = 0;
    static const uint16_t two[2] = {0, 2};
    uint8_t *buf = build_chain_plan(3, 4u, two, 2, &size);
    TEST_ASSERT(buf != NULL, "graph plan allocation");
    if (!buf)
        return;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "a plan of two graphs loads");
    chain_plan_link(buf, 0, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "a chain inside the main graph loads");
    free(buf);

    buf = build_chain_plan(3, 4u, two, 2, &size);
    TEST_ASSERT(buf != NULL, "graph plan allocation");
    if (!buf)
        return;
    chain_plan_link(buf, 1, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_ERR_BAD_SECTION,
                   "a chain from the main graph into a subgraph is refused");
    free(buf);

    static const uint16_t three[3] = {0, 1, 2};
    buf = build_chain_plan(3, 4u, three, 3, &size);
    TEST_ASSERT(buf != NULL, "graph plan allocation");
    if (!buf)
        return;
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "a plan of three graphs loads");
    chain_plan_link(buf, 1, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_ERR_BAD_SECTION,
                   "a chain past the end of a subgraph is refused");
    free(buf);
}

/* The executor runs a chain from its head, so a hand-built plan that reaches a member first is refused. */
static void test_chain_member_runs_from_its_head(void)
{
    printf("  test_chain_member_runs_from_its_head...\n");
    tigris_plan_t plan;
    uint32_t size = 0;
    uint8_t *buf = build_chain_plan(2, 4u, NULL, 0, &size);
    TEST_ASSERT(buf != NULL, "chain plan allocation");
    if (!buf)
        return;
    chain_plan_link(buf, 0, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan), TIGRIS_OK, "chain plan loads");
    /* The loaded plan reads its stage table in place. */
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    ((tigris_stage_t *)(buf + dir[2].offset))[0].chain_id = 1;
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256], slow[256];
    void *pointers[2] = {NULL, NULL};
    tigris_mem_t mem;
    TEST_ASSERT_EQ(tigris_mem_init(&mem, pointers, 2, fast, sizeof(fast), slow, sizeof(slow)),
                   TIGRIS_MEM_OK, "chain arenas");
    TEST_ASSERT_EQ(tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, NULL), TIGRIS_EXEC_ERR_TILE,
                   "a chain member reached before its head is refused");
    free(buf);
}

static uint8_t *build_many_stage_plan(uint16_t stage_count, uint32_t *out_size)
{
    const uint32_t tensors_off = 112u;
    const uint32_t ops_off = tensors_off + sizeof(tigris_tensor_t);
    const uint32_t stages_off = align4(
        ops_off + (uint32_t)stage_count * sizeof(tigris_op_t));
    const uint32_t index_off = stages_off +
        (uint32_t)stage_count * sizeof(tigris_stage_t);
    const uint32_t index_count = (uint32_t)stage_count * 3u;
    const uint32_t shapes_off = align4(
        index_off + index_count * sizeof(uint16_t));
    const uint32_t strings_off = shapes_off + sizeof(int32_t);
    const uint32_t plan_size = align4(strings_off + 1u) + (uint32_t)sizeof(tigris_memory_t);
    uint8_t *buf = calloc(1, plan_size);
    if (!buf)
        return NULL;

    plan_builder_t builder = plan_builder_begin(buf, plan_size, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 1;
    hdr->num_ops = stage_count;
    hdr->num_stages = stage_count;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, tensors_off);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, ops_off);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, stages_off);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, index_off);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, shapes_off);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, strings_off);
    plan_finish(buf);

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
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V4);
    TEST_ASSERT_EQ(tigris_plan_load(buf, size, &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "schema-v4 retains its authoritative uint8 stage limit");
    free(buf);
}

/* A compact valid compressed-weight plan.  It lets the loader tests exercise
 * block metadata without depending on generated model fixtures. */
#define WEIGHT_BLOCK_PLAN_SIZE 212u
#define WB_STAGES_OFF          136u
#define WB_INDEX_OFF           164u
#define WB_STRINGS_OFF         166u
#define WB_WEIGHTS_OFF         167u
#define WB_BLOCKS_OFF          179u

static void build_compressed_weight_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, WEIGHT_BLOCK_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_stages = 1;
    hdr->num_weights = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, WB_STAGES_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, WB_STAGES_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, WB_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, WB_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, WB_STRINGS_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, WB_STRINGS_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_WEIGHTS, WB_WEIGHTS_OFF);
    plan_builder_section(&builder, 7u, TIGRIS_SEC_WEIGHT_BLOCKS, WB_BLOCKS_OFF);
    plan_finish(buf);

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

#define REFERENCED_PLAN_SIZE 153u
#define REF_TENSORS_OFF 112u
#define REF_INDEX_OFF   128u
#define REF_SHAPES_OFF  132u
#define REF_STRINGS_OFF 148u

static void build_referenced_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, REFERENCED_PLAN_SIZE,
                                                TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 1;
    hdr->num_model_inputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, REF_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, REF_INDEX_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_INDEX_POOL, REF_INDEX_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_SHAPE_POOL, REF_SHAPES_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_STRINGS, REF_STRINGS_OFF);
    plan_finish(buf);

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
#define TRANSPOSE_PLAN_SIZE      280u
#define TRANSPOSE_TENSORS_OFF    128u
#define TRANSPOSE_OPS_OFF        160u
#define TRANSPOSE_STAGES_OFF     198u
#define TRANSPOSE_INDEX_OFF      226u
#define TRANSPOSE_SHAPES_OFF     236u
#define TRANSPOSE_STRINGS_OFF    252u
#define TRANSPOSE_ATTRS_OFF      253u
#define TRANSPOSE_ATTR_DATA_OFF  265u

static void build_transpose_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, TRANSPOSE_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, TRANSPOSE_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, TRANSPOSE_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, TRANSPOSE_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, TRANSPOSE_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, TRANSPOSE_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, TRANSPOSE_STRINGS_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_OP_ATTRIBUTES, TRANSPOSE_ATTRS_OFF);
    plan_finish(buf);

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

/* A plan whose single stage carries a height tile plan. The tiled contract
 * admits two stage shapes the stripe rules cannot express: a global
 * reduction, which collapses the height, and a layout conversion, which
 * permutes it. Both are validated here, where the executor is not reached. */
/* The tile section holds one schema 11 record; a schema 9 record fits in it too. */
#define TILED_PLAN_SIZE       352u
#define TILED_TENSORS_OFF     136u
#define TILED_OPS_OFF         168u
#define TILED_STAGES_OFF      206u
#define TILED_TILE_PLANS_OFF  234u
#define TILED_INDEX_OFF       282u
#define TILED_SHAPES_OFF      296u
#define TILED_STRINGS_OFF     328u
#define TILED_ATTRS_OFF       332u
#define TILED_ATTR_DATA_OFF   344u

static tigris_tile_plan_v11_t *tiled_record(uint8_t *buf)
{
    return (tigris_tile_plan_v11_t *)(buf + TILED_TILE_PLANS_OFF);
}

/* What the record states about its tiling. */
static void tiled_state_facts(uint8_t *buf, uint32_t tile_extent, uint32_t num_tiles,
                              uint32_t original_extent, uint16_t halo, uint8_t band_axis)
{
    tigris_tile_facts_t *facts = &tiled_record(buf)->facts;
    facts->tile_extent = tile_extent;
    facts->num_tiles = num_tiles;
    facts->original_extent = original_extent;
    facts->halo = halo;
    facts->band_axis = band_axis;
    facts->_reserved = 0u;
}

/**
 * Build the plan above. `op_type` selects the stage's single operator;
 * `ndim` its tensor rank. Shapes are the conversion's by default and are
 * overwritten by the caller for a reduction.
 */
static void build_tiled_stage_plan(uint8_t *buf, uint8_t op_type, uint8_t ndim)
{
    plan_builder_t builder = plan_builder_begin(buf, TILED_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_tile_plans = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    tigris_section_entry_t *dir =
        (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, TILED_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, TILED_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, TILED_STAGES_OFF);
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_TILE_PLANS,
                                      TILED_TILE_PLANS_OFF};
    plan_builder_section(&builder, 4u, TIGRIS_SEC_INDEX_POOL, TILED_INDEX_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_SHAPE_POOL, TILED_SHAPES_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_STRINGS, TILED_STRINGS_OFF);
    /* The section exists only when it has a record: an empty one is not a
     * shape the emitter produces. A Transpose carries its permutation and a
     * reduction the axis it collapses. */
    if (op_type == TIGRIS_OP_TRANSPOSE || op_type == TIGRIS_OP_REDUCE_MEAN)
        dir[7] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES,
                                          TILED_ATTRS_OFF};
    plan_finish(buf);

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

    /* A conversion's run sizes its band, so the record states one: one row of the swapped group. */
    tigris_tile_plan_v11_t *tile = tiled_record(buf);
    tile->choices.tileable = 1;
    tile->choices.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    tile->choices.tiled_peak_bytes = 128u;
    tiled_state_facts(buf, 1u, ndim == 3u ? 4u : 8u, ndim == 3u ? 4u : 8u, 0u, 1u);

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

/* The same plan at schema 9, whose record states a band of one height row of four. */
static void tiled_plan_to_v9(uint8_t *buf)
{
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V9);
    tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    memset(buf + TILED_TILE_PLANS_OFF, 0, sizeof(tigris_tile_plan_v11_t));
    tile->tileable = 1;
    tile->axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    tile->tile_height = 1;
    tile->num_tiles = 4;
    tile->original_height = 4;
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
    plan_drop_section(buf, 7u);
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
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V7);
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

/* Schema 10's record states the scheduling choices alone, in 32 bytes. */
static void test_compact_tile_record(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3u);
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V10);
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    header->budget = 128u;
    tigris_tile_plan_v10_t *tile = (tigris_tile_plan_v10_t *)(buf + TILED_TILE_PLANS_OFF);
    memset(buf + TILED_TILE_PLANS_OFF, 0, sizeof(tigris_tile_plan_v11_t));
    tile->tileable = 1u;
    tile->axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    tile->tiled_peak_bytes = 128u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "compact transpose record loads");
    TEST_ASSERT(plan.tile_plans == NULL && plan.tile_plans_v10 != NULL &&
                plan.tile_plans_v11 == NULL, "record layouts have separate pointers");
    tile->input_channel_group = 1u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "unsupported grouping is refused");
    tile->input_channel_group = 0u;
    tile->flags = 2u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "unknown scheduling flags are refused");
    tile->flags = 0u;
    tile->input_band_extent = 1u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "transpose cannot carry unrelated input row geometry");
    tile->input_band_extent = 0u;
    tile->tiled_peak_bytes = 129u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "working-set ceiling must fit the plan budget");
    tile->tiled_peak_bytes = 128u;
    /* The section ends where the record would if it were short. */
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    dir[4].offset = TILED_TILE_PLANS_OFF + (uint32_t)sizeof(*tile) - 2u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "truncated record is refused");
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

    /* [1, 2, 4, 8] reduced to [1, 1, 1, 8], walked one input row at a time. */
    build_tiled_stage_plan(buf, TIGRIS_OP_GLOBAL_AVG, 4);
    shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 1; shapes[6] = 1; shapes[7] = 8;
    ((tigris_tensor_t *)(buf + TILED_TENSORS_OFF))[1].size_bytes =
        8u * (uint32_t)sizeof(float);
    tiled_record(buf)->choices.input_band_extent = 1u;
    tiled_state_facts(buf, 1u, 2u, 2u, 0u, 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "a collapsed reduction may carry a height tile plan");

    /* An output that still has spatial extent is not a global reduction. */
    build_tiled_stage_plan(buf, TIGRIS_OP_GLOBAL_AVG, 4);
    tiled_record(buf)->choices.input_band_extent = 0u;
    tiled_record(buf)->choices.output_band_extent = 1u;
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

    /* No supported schema leaves a Transpose without its permutation, or an
     * opcode outside the built-in table, to a caller's dispatcher. */
    build_transpose_plan(buf);
    plan_drop_section(buf, 6u);
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_MIN);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_SECTION,
                   "attribute-free v4 Transpose is rejected");

    ((tigris_op_t *)(buf + TRANSPOSE_OPS_OFF))->op_type = TIGRIS_OP_UNKNOWN;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "v4 custom opcode is rejected");

    /* A normalization needs its variance floor as surely as a Transpose needs
     * its permutation, and the arm taken when there is no attribute section
     * at all has to say so too. This fixture carries no scale weight, so it
     * would be refused either way; the error says which guard fired, and
     * before the section arm covered normalizations it was the operator
     * check. A plan whose normalization is otherwise well formed had nothing
     * left to catch it and failed at inference instead. */
    build_transpose_plan(buf);
    plan_drop_section(buf, 6u);
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
#define SPLIT_PLAN_SIZE     296u
#define SPLIT_TENSORS_OFF   120u
#define SPLIT_OPS_OFF       168u
#define SPLIT_STAGES_OFF    206u
#define SPLIT_INDEX_OFF     234u
#define SPLIT_SHAPES_OFF    252u
#define SPLIT_STRINGS_OFF   288u

static void build_split_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, SPLIT_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 3;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 3;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, SPLIT_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, SPLIT_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, SPLIT_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, SPLIT_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, SPLIT_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, SPLIT_STRINGS_OFF);
    plan_finish(buf);

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

    /* Drop the attribute section and make the Transpose an exact-shape Relu; the rest is string tail. */
    plan_drop_section(buf, 6u);

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
    tiled_record(buf)->choices.input_band_extent = 2u;
    tiled_state_facts(buf, 2u, 2u, 4u, 0u, 2u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "a rank-4 row band loads");

    /* The same shapes held channels-last are not a matrix batch, so the band
     * does not apply and the stripe contract decides instead. */
    build_tiled_stage_plan(buf, TIGRIS_OP_ERF, 4);
    shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = 2; shapes[6] = 4; shapes[7] = 8;
    tiled_record(buf)->choices.output_band_extent = 1u;
    tiled_state_facts(buf, 1u, 2u, 2u, 0u, 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "the spatial reading of the same shapes still loads");
}

static void check_reshape_band_execution(const tigris_plan_t *plan, uint32_t budget)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256], slow[512];
    void *pointers[2] = {NULL, NULL};
    float input[32], output[32];
    for (uint32_t i = 0; i < 32u; i++) input[i] = (float)i - 15.0f;
    memset(fast, 0xA5, sizeof(fast));
    tigris_mem_t mem;
    tigris_exec_stats_t stats;
    TEST_ASSERT_EQ(tigris_mem_init(&mem, pointers, 2, fast, budget, slow, sizeof(slow)),
                   TIGRIS_MEM_OK, "reshape arenas");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 0, sizeof(input)), TIGRIS_MEM_OK, "input allocation");
    TEST_ASSERT_EQ(tigris_input_write(plan, &mem, 0, input, sizeof(input)), TIGRIS_OK, "input write");
    TEST_ASSERT_EQ(tigris_run(plan, &mem, tigris_dispatch_kernel, NULL, &stats), TIGRIS_EXEC_OK,
                   "banded reshape execution");
    TEST_ASSERT_EQ(stats.stages_tiled, 1u, "reshape actually tiled");
    TEST_ASSERT(mem.fast_peak <= budget, "reshape fits its scheduled arena");
    TEST_ASSERT_EQ(tigris_output_read(plan, &mem, 1, output, sizeof(output)), TIGRIS_OK, "output read");
    TEST_ASSERT(memcmp(input, output, sizeof(input)) == 0, "reshape preserves every byte");
    for (uint32_t i = budget; i < sizeof(fast); i++)
        TEST_ASSERT_EQ(fast[i], 0xA5, "fast arena canary");
}

static void test_reshape_band_contract(void)
{
    printf("  test_reshape_band_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    /* Schema 9's record states the band's counts; schema 11 has its own test. */
    build_tiled_stage_plan(buf, TIGRIS_OP_RESHAPE, 3);
    tiled_plan_to_v9(buf);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].flags |= TIGRIS_TENSOR_LINEAR;
    tensors[1].flags |= TIGRIS_TENSOR_LINEAR;
    tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->receptive_field = 1;
    tile->tile_height = 3;
    tile->num_tiles = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "split rows with a final partial band");
    check_reshape_band_execution(&plan, 96u);
    tile->original_height = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "incorrect original extent refused");
    tile->original_height = 4;
    tile->num_tiles = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "incorrect tile count refused");
    tile->num_tiles = 2;
    tile->halo = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "reshape has no halo");
    tile->halo = 0;
    tile->tile_width = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "reshape has no width tile");
    tile->tile_width = 0;
    tile->tile_height = 0;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "zero band refused");
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[1] = 8; shapes[2] = 4;
    shapes[5] = 4; shapes[6] = 8;
    tile->original_height = 8;
    tile->tile_height = 2;
    tile->num_tiles = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "integral row merge");
    check_reshape_band_execution(&plan, 32u);
    tile->tile_height = 3;
    tile->num_tiles = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "non-integral endpoint refused");
    tile->tile_height = 2;
    tile->num_tiles = 4;
    shapes[4] = 2; shapes[5] = 2;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK,
                "non-integral contiguous endpoint refused");

    shapes[0] = 1; shapes[1] = 4; shapes[2] = 8;
    shapes[4] = 8; shapes[5] = 1; shapes[6] = 4;
    tile->original_height = 4;
    tile->tile_height = 3;
    tile->num_tiles = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "height interval maps onto output axis zero");
    check_reshape_band_execution(&plan, 96u);
    tile->original_height = 8;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "output extent cannot replace input extent");

    shapes[0] = 8; shapes[1] = 1; shapes[2] = 4;
    shapes[4] = 1; shapes[5] = 4; shapes[6] = 8;
    tile->tile_height = 2;
    tile->num_tiles = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "axis-zero interval maps onto output height");
    check_reshape_band_execution(&plan, 32u);
    tile->tile_height = 3;
    tile->num_tiles = 3;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "axis-zero endpoint must map to a whole output row");

    shapes[0] = 2; shapes[1] = 4; shapes[2] = 4;
    tile->original_height = 4;
    tile->tile_height = 2;
    tile->num_tiles = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "multiple leading input blocks are not a contiguous height band");
    shapes[0] = 1; shapes[1] = 4; shapes[2] = 8;
    shapes[4] = 2; shapes[5] = 4; shapes[6] = 4;
    tile->tile_height = 1;
    tile->num_tiles = 4;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "multiple leading output blocks are not a contiguous height band");
}

/* A schema 11 tile record states its tiling; the loader derives it and refuses a disagreeing record. */
static int tile_facts_load(uint8_t *buf, tigris_plan_t *plan)
{
    return tigris_plan_load(buf, TILED_PLAN_SIZE, plan);
}

static void test_tile_facts_contract(void)
{
    printf("  test_tile_facts_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_tile_facts_t derived;
    tigris_tile_facts_t *facts = &tiled_record(buf)->facts;

    /* A conversion of [1, 4, 8]: bands of one of the four rows. */
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3u);
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "stated conversion facts load");
    TEST_ASSERT(plan.tile_plans_v11 != NULL && plan.tile_plans_v10 == NULL &&
                plan.tile_plans == NULL, "schema 11 records have their own pointer");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, &plan.stages[0], &derived), 1, "facts derived");
    TEST_ASSERT(memcmp(&derived, facts, sizeof(derived)) == 0, "derived facts equal the record");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(NULL, &plan.stages[0], &derived), 0, "NULL plan");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, NULL, &derived), 0, "NULL stage");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, &plan.stages[0], NULL), 0, "NULL output");

    facts->num_tiles = 5u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "wrong tile count refused");
    facts->num_tiles = 4u;
    facts->original_extent = 8u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "the unbanded axis is not the extent");
    facts->original_extent = 4u;
    facts->halo = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "a conversion has no halo");
    facts->halo = 0u;
    facts->band_axis = 2u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "wrong band axis refused");
    facts->band_axis = 1u;
    facts->_reserved = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "reserved byte must be zero");
    facts->_reserved = 0u;

    /* The run sizes a conversion's band, so the record's is only bounded, and the count follows it. */
    facts->tile_extent = 3u;
    facts->num_tiles = 2u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "a stated band of three loads");
    facts->num_tiles = 4u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "the count follows the stated band");
    facts->tile_extent = 0u;
    facts->num_tiles = 0u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "an empty band refused");
    facts->tile_extent = 5u;
    facts->num_tiles = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "a band past the extent refused");

    /* Columns instead: the eight of them, one at a time. */
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3u);
    tiled_record(buf)->choices.flags = TIGRIS_TILE_FLAG_BAND_ON_COLUMNS;
    tiled_state_facts(buf, 1u, 8u, 8u, 0u, 2u);
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "a band along the columns loads");

    /* An untileable record states no tiling. */
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 3u);
    memset(tiled_record(buf), 0, sizeof(tigris_tile_plan_v11_t));
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "an untileable record loads");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, &plan.stages[0], &derived), 1,
                   "an untileable record has facts");
    TEST_ASSERT(derived.num_tiles == 0u && derived.original_extent == 0u &&
                derived.tile_extent == 0u, "and they are zero");
    facts->num_tiles = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "an untileable record with a count refused");

    /* A reshape of [1, 4, 8] to [1, 8, 4] in bands of three of its four input rows. */
    build_tiled_stage_plan(buf, TIGRIS_OP_RESHAPE, 3u);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].flags |= TIGRIS_TENSOR_LINEAR;
    tensors[1].flags |= TIGRIS_TENSOR_LINEAR;
    tiled_record(buf)->choices.input_band_extent = 3u;
    tiled_state_facts(buf, 3u, 2u, 4u, 0u, 1u);
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "stated reshape facts load");
    check_reshape_band_execution(&plan, 96u);
    facts->original_extent = 8u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "the output extent is not the reshape's");
    facts->original_extent = 4u;
    facts->num_tiles = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "wrong reshape count refused");
    facts->num_tiles = 2u;
    facts->tile_extent = 2u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "the band is the record's input band");

    /* A sum over the last axis of [7, 1, 4] bands the leading axis, three rows at a time. */
    build_tiled_stage_plan(buf, TIGRIS_OP_REDUCE_MEAN, 3u);
    ((tigris_op_t *)(buf + TILED_OPS_OFF))->op_type = TIGRIS_OP_REDUCE_SUM;
    int32_t *dims = (int32_t *)(buf + TILED_SHAPES_OFF);
    dims[0] = 7; dims[1] = 1; dims[2] = 4;
    dims[4] = 7; dims[5] = 1; dims[6] = 1;
    tensors[0].size_bytes = 112u;
    tensors[1].size_bytes = 28u;
    buf[TILED_ATTR_DATA_OFF] = 2;
    tiled_record(buf)->choices.input_band_extent = 3u;
    tiled_state_facts(buf, 3u, 3u, 7u, 0u, 0u);
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_OK, "stated leading-band facts load");
    facts->band_axis = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS,
                   "the leading band is the first axis");
    facts->band_axis = 0u;
    facts->original_extent = 1u;
    facts->num_tiles = 1u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "wrong leading extent refused");
    facts->original_extent = 7u;
    facts->num_tiles = 3u;
    facts->halo = 2u;
    TEST_ASSERT_EQ(tile_facts_load(buf, &plan), TIGRIS_ERR_TILE_FACTS, "a leading band has no halo");
}

/* A 3x1 max pool, stride 2 and one row of padding each side, of [1, 8, 2, 2] to [1, 4, 2, 2],
 * in a fast arena of 128 bytes. The schema 11 record states `band` output rows. */
static void build_strided_stripe_plan(uint8_t *buf, uint32_t band)
{
    build_tiled_stage_plan(buf, TIGRIS_OP_MAX_POOL, 4u);
    ((tigris_file_header_t *)buf)->budget = 128u;
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[0] = 1; shapes[1] = 8; shapes[2] = 2; shapes[3] = 2;
    shapes[4] = 1; shapes[5] = 4; shapes[6] = 2; shapes[7] = 2;
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].size_bytes = 32u * (uint32_t)sizeof(float);
    tensors[1].size_bytes = 16u * (uint32_t)sizeof(float);
    tigris_op_t *op = (tigris_op_t *)(buf + TILED_OPS_OFF);
    op->spatial.kernel_h = 3; op->spatial.kernel_w = 1;
    op->spatial.stride_h = 2; op->spatial.stride_w = 1;
    op->spatial.pad_top = 1; op->spatial.pad_bottom = 1;
    op->spatial.dilation_h = 1; op->spatial.dilation_w = 1;
    tiled_record(buf)->choices.output_band_extent = band;
    tiled_state_facts(buf, band, (4u + band - 1u) / band, 4u, 2u, 1u);
}

static tigris_exec_error_t run_strided_stripe(const tigris_plan_t *plan, const float *input,
                                              float *output, tigris_exec_stats_t *stats)
{
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[128], slow[512];
    void *pointers[2] = {NULL, NULL};
    tigris_mem_t mem;
    TEST_ASSERT_EQ(tigris_mem_init(&mem, pointers, 2, fast, sizeof(fast), slow, sizeof(slow)),
                   TIGRIS_MEM_OK, "stripe arenas");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 0, 32u * sizeof(float)), TIGRIS_MEM_OK, "input allocation");
    TEST_ASSERT_EQ(tigris_input_write(plan, &mem, 0, input, 32u * sizeof(float)), TIGRIS_OK, "input write");
    memset(stats, 0, sizeof(*stats));
    tigris_exec_error_t err = tigris_run(plan, &mem, tigris_dispatch_kernel, NULL, stats);
    if (err == TIGRIS_EXEC_OK)
        TEST_ASSERT_EQ(tigris_output_read(plan, &mem, 1, output, 16u * sizeof(float)), TIGRIS_OK,
                       "output read");
    return err;
}

/* From schema 11 a height stripe states its band in output rows, and the run takes that band. */
static void test_stripe_runs_its_stated_band(void)
{
    printf("  test_stripe_runs_its_stated_band...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    tigris_tile_facts_t derived;
    tigris_exec_stats_t stats;
    float input[32], output[16], expected[16];
    for (uint32_t i = 0; i < 32u; i++)
        input[i] = (float)((i * 7u) % 13u) - 6.0f;
    for (uint32_t o = 0; o < 4u; o++) {
        for (uint32_t k = 0; k < 4u; k++) {
            float best = -1.0e30f;
            for (int32_t r = 2 * (int32_t)o - 1; r <= 2 * (int32_t)o + 1; r++)
                if (r >= 0 && r < 8 && input[(uint32_t)r * 4u + k] > best)
                    best = input[(uint32_t)r * 4u + k];
            expected[o * 4u + k] = best;
        }
    }

    /* One output row per band: four bands, where the arena would take two rows at once. */
    build_strided_stripe_plan(buf, 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "an output band loads");
    TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, &plan.stages[0], &derived), 1, "facts derived");
    TEST_ASSERT(memcmp(&derived, &tiled_record(buf)->facts, sizeof(derived)) == 0,
                "the facts count output bands of the output extent");
    TEST_ASSERT_EQ(run_strided_stripe(&plan, input, output, &stats), TIGRIS_EXEC_OK, "the stripe runs");
    TEST_ASSERT_EQ(stats.stages_tiled, 1u, "the stage ran as a height stripe");
    TEST_ASSERT_EQ(stats.total_tiles, 4u, "the run takes the stated band");
    TEST_ASSERT(memcmp(output, expected, sizeof(output)) == 0, "outputs match the reference");

    /* A count of input rows over the band does not describe the run. */
    tiled_record(buf)->facts.num_tiles = 8u;
    tiled_record(buf)->facts.original_extent = 8u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_TILE_FACTS,
                   "an input-row count refused");
    /* Nor does a band of input rows. */
    build_strided_stripe_plan(buf, 1u);
    tiled_record(buf)->choices.output_band_extent = 0u;
    tiled_record(buf)->choices.input_band_extent = 1u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "an input band refused");
    build_strided_stripe_plan(buf, 5u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "a band past the output refused");

    /* The whole output in one band needs all eight input rows, more than the arena holds. */
    build_strided_stripe_plan(buf, 4u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "a whole-output band loads");
    memset(output, 0, sizeof(output));
    TEST_ASSERT_EQ(run_strided_stripe(&plan, input, output, &stats), TIGRIS_EXEC_ERR_TILE,
                   "a band that does not fit fails closed");
    TEST_ASSERT_EQ(stats.total_tiles, 0u, "before the first tile");
}

/* The tiled plan from [1, 2, 4, 8] to [1, out_h, out_w, 8] with a 2D record of one row by one column. */
static void build_hw_record_plan(uint8_t *buf, uint8_t op_type, int32_t out_h, int32_t out_w)
{
    build_tiled_stage_plan(buf, op_type, 4u);
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[4] = 1; shapes[5] = out_h; shapes[6] = out_w; shapes[7] = 8;
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[1].size_bytes = (uint32_t)(out_h * out_w * 8) * (uint32_t)sizeof(float);
    tigris_tile_plan_v11_t *tile = tiled_record(buf);
    tile->choices.axis = TIGRIS_TILE_AXIS_HW;
    tile->choices.output_band_extent = 1u;
    tile->choices.output_width_extent = 1u;
    tiled_state_facts(buf, 1u, (uint32_t)(out_h * out_w), (uint32_t)out_h, 0u, 1u);
}

/* A stage over t0 [1, 4, 4, 8] and t1 of `shape` under a 2D record of one row
 * by one column: one Add of both to t2, or with `pair` a Relu of each, t1's
 * to its own output t3 of t1's shape. */
#define HW_TWO_PLAN_SIZE 512u
static void build_hw_two_input_plan(uint8_t *buf, const int32_t *shape, uint8_t rank, int pair)
{
    const uint16_t tensors_count = pair ? 4u : 3u;
    const uint32_t tensors_off = 136u;
    const uint32_t ops_off = tensors_off + tensors_count * (uint32_t)sizeof(tigris_tensor_t);
    const uint32_t stages_off = ops_off + 2u * (uint32_t)sizeof(tigris_op_t);
    const uint32_t tiles_off = stages_off + (uint32_t)sizeof(tigris_stage_t);
    const uint32_t index_off = align4(tiles_off + (uint32_t)sizeof(tigris_tile_plan_v11_t));
    const uint32_t shapes_off = index_off + 32u;
    const uint32_t strings_off = shapes_off + 64u;
    memset(buf, 0, HW_TWO_PLAN_SIZE);
    plan_builder_t builder = plan_builder_begin(buf, HW_TWO_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = tensors_count;
    hdr->num_ops = pair ? 2u : 1u;
    hdr->num_stages = 1;
    hdr->num_tile_plans = 1;
    hdr->model_io_off = pair ? 4u : 3u;
    hdr->num_model_inputs = 2;
    hdr->num_model_outputs = pair ? 2u : 1u;
    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, tensors_off);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, ops_off);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, stages_off);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_TILE_PLANS, tiles_off);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_INDEX_POOL, index_off);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_SHAPE_POOL, shapes_off);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_STRINGS, strings_off);
    plan_finish(buf);

    static const int32_t full[4] = {1, 4, 4, 8};
    int32_t *shapes = (int32_t *)(buf + shapes_off);
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + tensors_off);
    for (uint16_t i = 0; i < tensors_count; i++) {
        int second = i == 1u || i == 3u;
        uint8_t ndim = second ? rank : 4u;
        uint32_t elements = 1u;
        for (uint8_t d = 0; d < ndim; d++) {
            shapes[4u * i + d] = second ? shape[d] : full[d];
            elements *= (uint32_t)shapes[4u * i + d];
        }
        tensors[i].size_bytes = elements * (uint32_t)sizeof(float);
        tensors[i].shape_off = (uint16_t)(4u * i);
        tensors[i].ndim = ndim;
        tensors[i].dtype = 1;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        tensors[i].flags = i < 2u ? TIGRIS_TENSOR_MODEL_INPUT : TIGRIS_TENSOR_MODEL_OUTPUT;
    }

    static const uint16_t add_pool[] = {0, 1, 2, 0, 1, 2, 0, 0, 1, 2};
    static const uint16_t pair_pool[] = {0, 2, 1, 3, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3};
    memcpy(buf + index_off, pair ? pair_pool : add_pool, pair ? sizeof(pair_pool) : sizeof(add_pool));
    tigris_op_t *ops = (tigris_op_t *)(buf + ops_off);
    for (uint16_t j = 0; j < hdr->num_ops; j++) {
        ops[j].op_type = pair ? TIGRIS_OP_RELU : TIGRIS_OP_ADD;
        ops[j].num_inputs = pair ? 1u : 2u;
        ops[j].num_outputs = 1;
        ops[j].inputs_off = (uint16_t)(2u * j);
        ops[j].outputs_off = pair ? (uint16_t)(2u * j + 1u) : 2u;
        ops[j].weight_idx = TIGRIS_NO_WEIGHT;
        ops[j].bias_idx = TIGRIS_NO_WEIGHT;
    }
    tigris_stage_t *stage = (tigris_stage_t *)(buf + stages_off);
    stage->ops_off = pair ? 8u : 6u;
    stage->ops_count = hdr->num_ops;
    stage->inputs_off = pair ? 10u : 7u;
    stage->inputs_count = 2;
    stage->outputs_off = pair ? 12u : 9u;
    stage->outputs_count = pair ? 2u : 1u;
    stage->tile_plan_idx = 0;
    stage->chain_id = TIGRIS_NO_CHAIN;

    tigris_tile_plan_v11_t *tile = (tigris_tile_plan_v11_t *)(buf + tiles_off);
    tile->choices.tileable = 1;
    tile->choices.axis = TIGRIS_TILE_AXIS_HW;
    tile->choices.tiled_peak_bytes = 128u;
    tile->choices.output_band_extent = 1u;
    tile->choices.output_width_extent = 1u;
    tile->facts = (tigris_tile_facts_t){.tile_extent = 1u, .num_tiles = 16u, .original_extent = 4u,
                                        .halo = 0u, .band_axis = 1u};
}

/* A 2D record sends its stage to the 2D executor, which bands rank-4 stage
 * tensors under the stripe contract along HW; a stage it cannot band that way
 * is refused at load rather than on its first run. */
static void test_hw_record_contract(void)
{
    printf("  test_hw_record_contract...\n");
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;

    build_hw_record_plan(buf, TIGRIS_OP_RELU, 2, 4);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "a pointwise 2D record loads");

    build_hw_record_plan(buf, TIGRIS_OP_GLOBAL_MAX, 1, 1);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "a 2D record on an operator without a height contract is refused");

    /* A global average bands by its input; a schema 10 record states no facts to disagree with. */
    build_hw_record_plan(buf, TIGRIS_OP_GLOBAL_AVG, 1, 1);
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_V10);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "a 2D record on a stage of another band kind is refused");

    /* A schema 9 record carries no ranks of its own. */
    build_tiled_stage_plan(buf, TIGRIS_OP_RELU, 3u);
    int32_t *shapes = (int32_t *)(buf + TILED_SHAPES_OFF);
    shapes[5] = 4; shapes[6] = 8;
    tiled_plan_to_v9(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "a rank-3 height record loads");
    ((tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF))->axis = TIGRIS_TILE_AXIS_HW;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "a 2D record on a rank-3 stage is refused");

    /* A 2D band cuts every operand to its rectangle, so a binary op takes full-shape operands only. */
    _Alignas(4) uint8_t two[HW_TWO_PLAN_SIZE];
    static const int32_t full[4] = {1, 4, 4, 8}, channel[4] = {1, 1, 1, 8}, rows[2] = {4, 8};
    build_hw_two_input_plan(two, full, 4u, 0);
    TEST_ASSERT_EQ(tigris_plan_load(two, sizeof(two), &plan), TIGRIS_OK, "a full-shape Add on a 2D record loads");
    build_hw_two_input_plan(two, channel, 4u, 0);
    TEST_ASSERT_EQ(tigris_plan_load(two, sizeof(two), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "a per-channel Add on a 2D record is refused");
    build_hw_two_input_plan(two, full, 4u, 1);
    TEST_ASSERT_EQ(tigris_plan_load(two, sizeof(two), &plan), TIGRIS_OK,
                   "a 2D record on a stage of two rank-4 inputs loads");
    build_hw_two_input_plan(two, rows, 2u, 1);
    TEST_ASSERT_EQ(tigris_plan_load(two, sizeof(two), &plan), TIGRIS_ERR_BAD_SECTION,
                   "a 2D record on a stage with a rank-2 input is refused");
}

/* The executor's order, read off the band predicates one at a time. */
static tigris_band_kind_t replayed_band_kind(const tigris_plan_t *plan,
                                             const tigris_stage_t *stage, int32_t *rows)
{
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    int banded = tile.tileable != 0u && tile.axis != TIGRIS_TILE_AXIS_NONE;
    tigris_reshape_band_t view;
    int32_t batch = 0;
    *rows = 0;
    if (banded && tigris_reshape_band(plan, stage, &view)) {
        *rows = view.input_rows;
        return TIGRIS_BAND_KIND_RESHAPE;
    }
    if (banded && tigris_stage_leading_band(plan, stage) > 0) {
        *rows = tigris_stage_leading_band(plan, stage);
        return TIGRIS_BAND_KIND_LEADING;
    }
    if (tigris_stage_row_tiled(plan, stage, &batch, rows))
        return TIGRIS_BAND_KIND_ROWS;
    if (tigris_stage_tiled_transpose(plan, stage))
        return TIGRIS_BAND_KIND_TRANSPOSE;
    if (tigris_stage_input_driven_reduction(plan, stage))
        return TIGRIS_BAND_KIND_BY_INPUT;
    return banded && tile.axis == TIGRIS_TILE_AXIS_HW ? TIGRIS_BAND_KIND_HW : TIGRIS_BAND_KIND_STRIPE;
}

/* One stage of one operator over up to two inputs, held in memory. */
typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t op;
    tigris_stage_t stage;
    uint16_t pool[7];
    int32_t shapes[24];
    tigris_op_attribute_t attribute;
    uint8_t attribute_data[48];
    tigris_tile_plan_v11_t tile;
    tigris_weight_entry_t weight;
    tigris_plan_t plan;
} band_sweep_t;

typedef struct {
    long cases;
    long exclusive_failures;
    long order_failures;
    long reshape_rows;
    long leading_rows;
} band_sweep_counts_t;

static uint32_t sweep_elements(const int32_t *shape, uint8_t rank)
{
    uint32_t elements = 1u;
    for (uint8_t i = 0u; i < rank; i++)
        elements *= (uint32_t)shape[i];
    return elements;
}

/* Every permutation of `rank` axes, the index-th in lexicographic order. */
static void sweep_permutation(uint32_t index, uint8_t rank, uint8_t *perm)
{
    uint8_t left[6] = {0, 1, 2, 3, 4, 5};
    uint8_t count = rank;
    for (uint8_t i = 0u; i < rank; i++) {
        uint32_t block = 1u;
        for (uint8_t j = 2u; j < count; j++)
            block *= j;
        uint8_t pick = (uint8_t)(index / block);
        index %= block;
        perm[i] = left[pick];
        for (uint8_t j = pick; j + 1u < count; j++)
            left[j] = left[j + 1u];
        count--;
    }
}

static void sweep_case(band_sweep_t *sweep, band_sweep_counts_t *counts)
{
    tigris_plan_t *plan = &sweep->plan;
    const tigris_stage_t *stage = &sweep->stage;
    tigris_reshape_band_t view;
    int32_t batch = 0;
    int32_t rows = 0;
    static const uint8_t records[3][2] = {
        {0u, TIGRIS_TILE_AXIS_NONE},
        {1u, TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH},
        {1u, TIGRIS_TILE_AXIS_HW},
    };
    int reshape = tigris_reshape_band(plan, stage, &view);
    int leading = tigris_stage_leading_band(plan, stage) > 0;
    int row = tigris_stage_row_tiled(plan, stage, &batch, &rows);
    int transpose = tigris_stage_tiled_transpose(plan, stage);
    int by_input = tigris_stage_input_driven_reduction(plan, stage);
    counts->cases++;
    if ((reshape && leading) ||
        (transpose && (reshape || leading || row || by_input)) ||
        (by_input && (reshape || leading || row)))
        counts->exclusive_failures++;
    if (reshape && row)
        counts->reshape_rows++;
    if (leading && row)
        counts->leading_rows++;
    for (uint8_t r = 0u; r < 3u; r++) {
        sweep->stage.tile_plan_idx = records[r][0] != 0u ? 0u : TIGRIS_NO_TILE_PLAN;
        sweep->tile.choices.tileable = records[r][0];
        sweep->tile.choices.axis = records[r][1];
        int banded = records[r][0] != 0u;
        tigris_band_kind_t want =
            banded && reshape ? TIGRIS_BAND_KIND_RESHAPE :
            banded && leading ? TIGRIS_BAND_KIND_LEADING :
            row ? TIGRIS_BAND_KIND_ROWS :
            transpose ? TIGRIS_BAND_KIND_TRANSPOSE :
            by_input ? TIGRIS_BAND_KIND_BY_INPUT :
            records[r][1] == TIGRIS_TILE_AXIS_HW ? TIGRIS_BAND_KIND_HW : TIGRIS_BAND_KIND_STRIPE;
        int32_t want_rows = want == TIGRIS_BAND_KIND_RESHAPE ? view.input_rows :
            want == TIGRIS_BAND_KIND_LEADING ? tigris_stage_leading_band(plan, stage) :
            want == TIGRIS_BAND_KIND_ROWS ? rows : 0;
        tigris_stage_band_t band;
        if (tigris_stage_band_kind(plan, stage, &band) != want || band.rows != want_rows ||
            band.batch != (want == TIGRIS_BAND_KIND_ROWS ? batch : 0))
            counts->order_failures++;
    }
}

/* The operator's own attributes: a permutation, one reduced axis, or pads on one axis. */
static void sweep_attributes(band_sweep_t *sweep, band_sweep_counts_t *counts)
{
    uint8_t rank = sweep->tensors[0].ndim;
    sweep->plan.num_op_attributes = 0u;
    sweep_case(sweep, counts);
    sweep->plan.num_op_attributes = 1u;
    if (sweep->op.op_type == TIGRIS_OP_TRANSPOSE) {
        uint32_t perms = 1u;
        for (uint8_t i = 2u; i <= rank; i++)
            perms *= i;
        sweep->attribute.type = TIGRIS_OP_ATTR_TRANSPOSE_PERM;
        sweep->attribute.data_len = rank;
        for (uint32_t p = 0u; p < perms; p++) {
            sweep_permutation(p, rank, sweep->attribute_data);
            sweep_case(sweep, counts);
        }
    } else if (sweep->op.op_type == TIGRIS_OP_REDUCE_MEAN || sweep->op.op_type == TIGRIS_OP_REDUCE_SUM ||
               sweep->op.op_type == TIGRIS_OP_REDUCE_MAX || sweep->op.op_type == TIGRIS_OP_ARG_MAX ||
               sweep->op.op_type == TIGRIS_OP_CUMSUM) {
        sweep->attribute.type = TIGRIS_OP_ATTR_AXES;
        sweep->attribute.data_len = 1u;
        for (uint8_t axis = 0u; axis < rank; axis++) {
            sweep->attribute_data[0] = axis;
            sweep_case(sweep, counts);
        }
    } else if (sweep->op.op_type == TIGRIS_OP_PAD) {
        sweep->attribute.type = TIGRIS_OP_ATTR_PADS;
        sweep->attribute.data_len = (uint8_t)(8u * rank);
        for (uint8_t axis = 0u; axis <= rank; axis++) {
            memset(sweep->attribute_data, 0, sizeof(sweep->attribute_data));
            if (axis > 0u)
                sweep->attribute_data[8u * (axis - 1u)] = 1u;
            sweep_case(sweep, counts);
        }
    }
    sweep->plan.num_op_attributes = 0u;
}

/*
 * The executor and the tile facts take a stage's band kind in one order:
 * reshape, leading, rows, transpose, by input, 2D, stripe; a tile record
 * checks leading before reshape. Over single-operator stages of every
 * operator, shapes of rank 2 to 4 with extents 1 and 3, either axis order
 * flag, unit and full second operands and a full-size constant, no Reshape is
 * a leading band, and a banded transpose or a global reduction by input is no
 * other kind. A row band can also be a Reshape or a leading band, and both
 * orders put rows after them.
 */
static void test_band_kind_order(void)
{
    printf("  test_band_kind_order...\n");
    band_sweep_t sweep;
    band_sweep_counts_t counts;
    memset(&sweep, 0, sizeof(sweep));
    memset(&counts, 0, sizeof(counts));
    tigris_plan_t *plan = &sweep.plan;
    sweep.header.version = TIGRIS_SCHEMA_VERSION;
    sweep.header.num_tensors = 3u;
    sweep.header.num_ops = 1u;
    sweep.header.num_stages = 1u;
    sweep.header.num_tile_plans = 1u;
    sweep.header.num_weights = 1u;
    plan->header = &sweep.header;
    plan->tensors = sweep.tensors;
    plan->ops = &sweep.op;
    plan->stages = &sweep.stage;
    plan->index_pool = sweep.pool;
    plan->shape_pool = sweep.shapes;
    plan->op_attributes = &sweep.attribute;
    plan->op_attribute_data = sweep.attribute_data;
    plan->tile_plans_v11 = &sweep.tile;
    plan->weight_entries = &sweep.weight;
    /* Operator inputs 0 and 1, its output, the stage operator, stage inputs, stage output. */
    const uint16_t pool[7] = {0u, 1u, 2u, 0u, 0u, 1u, 2u};
    memcpy(sweep.pool, pool, sizeof(pool));
    sweep.op.inputs_off = 0u;
    sweep.op.outputs_off = 2u;
    sweep.op.num_outputs = 1u;
    sweep.op.bias_idx = TIGRIS_NO_WEIGHT;
    sweep.stage.ops_off = 3u;
    sweep.stage.ops_count = 1u;
    sweep.stage.inputs_off = 4u;
    sweep.stage.outputs_off = 6u;
    sweep.stage.outputs_count = 1u;
    sweep.stage.chain_id = TIGRIS_NO_CHAIN;
    for (uint16_t t = 0u; t < 3u; t++) {
        sweep.tensors[t].shape_off = (uint16_t)(8u * t);
        sweep.tensors[t].dtype = 1u;
        sweep.tensors[t].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }
    int32_t *in_shape = &sweep.shapes[0];
    int32_t *second_shape = &sweep.shapes[8];
    int32_t *out_shape = &sweep.shapes[16];
    for (uint16_t type = 1u; type <= (uint16_t)TIGRIS_OP_GELU; type++) {
        sweep.op.op_type = (uint8_t)type;
        for (uint8_t rank = 2u; rank <= 4u; rank++) {
            for (uint32_t extents = 0u; extents < (1u << rank); extents++) {
                for (uint8_t i = 0u; i < rank; i++)
                    in_shape[i] = ((extents >> i) & 1u) != 0u ? 3 : 1;
                /* Outputs: the input's shape, one axis at 1, the last two swapped or merged. */
                for (uint8_t form = 0u; form <= rank + 2u; form++) {
                    uint8_t out_rank = rank;
                    memcpy(out_shape, in_shape, sizeof(int32_t) * rank);
                    if (form > 0u && form <= rank) {
                        out_shape[form - 1u] = 1;
                    } else if (form == rank + 1u) {
                        out_shape[rank - 2u] = in_shape[rank - 1u];
                        out_shape[rank - 1u] = in_shape[rank - 2u];
                    } else if (form == rank + 2u) {
                        out_shape[rank - 2u] *= in_shape[rank - 1u];
                        out_rank = (uint8_t)(rank - 1u);
                        if (out_rank < 2u)
                            continue;
                    }
                    for (uint8_t layout = 0u; layout < 4u; layout++) {
                        sweep.tensors[0].ndim = rank;
                        sweep.tensors[0].flags = (layout & 1u) != 0u ? TIGRIS_TENSOR_LINEAR : 0u;
                        sweep.tensors[0].size_bytes = 4u * sweep_elements(in_shape, rank);
                        sweep.tensors[2].ndim = out_rank;
                        sweep.tensors[2].flags = (layout & 2u) != 0u ? TIGRIS_TENSOR_LINEAR : 0u;
                        sweep.tensors[2].size_bytes = 4u * sweep_elements(out_shape, out_rank);
                        sweep.weight.size_bytes = sweep.tensors[2].size_bytes;
                        /* A second operand of the output's shape with any subset of its axes at 1. */
                        for (uint32_t unit = 0u; unit <= (1u << out_rank); unit++) {
                            uint8_t inputs = unit == (1u << out_rank) ? 1u : 2u;
                            sweep.op.num_inputs = inputs;
                            sweep.stage.inputs_count = inputs;
                            for (uint8_t i = 0u; i < out_rank; i++)
                                second_shape[i] = ((unit >> i) & 1u) != 0u ? 1 : out_shape[i];
                            sweep.tensors[1] = sweep.tensors[2];
                            sweep.tensors[1].shape_off = 8u;
                            sweep.tensors[1].size_bytes = 4u * sweep_elements(second_shape, out_rank);
                            sweep.op.weight_idx = TIGRIS_NO_WEIGHT;
                            sweep_attributes(&sweep, &counts);
                            sweep.op.weight_idx = 0u;
                            sweep_attributes(&sweep, &counts);
                        }
                    }
                }
            }
        }
    }
    printf("    %ld stages, %ld reshape row bands, %ld leading row bands\n",
           counts.cases, counts.reshape_rows, counts.leading_rows);
    TEST_ASSERT_EQ(counts.exclusive_failures, 0, "reshape, leading, transpose and by-input bands exclude each other");
    TEST_ASSERT_EQ(counts.order_failures, 0, "the band kind follows the executor's order");
    TEST_ASSERT(counts.reshape_rows > 0, "the sweep reaches a Reshape that is also a row band");
    TEST_ASSERT(counts.leading_rows > 0, "the sweep reaches a leading band that is also a row band");

    TEST_ASSERT_EQ(tigris_stage_band_kind(NULL, &sweep.stage, NULL), TIGRIS_BAND_KIND_NONE, "NULL plan");
    TEST_ASSERT_EQ(tigris_stage_band_kind(plan, NULL, NULL), TIGRIS_BAND_KIND_NONE, "NULL stage");
}

/* Every stage of the fixed fixtures of each schema has the replayed kind. */
static void test_band_kind_fixtures(void)
{
    static const char *const names[] = {
        "schema-v4-transpose.tgrs", "schema-v5-conv1d-axis.tgrs",
        "schema-v6-interface-dtype.tgrs",
        "schema-v7-tensor-layout.tgrs", "schema-v8-layer-norm.tgrs",
        "schema-v9-binary-requant.tgrs", "schema-v10-state.tgrs",
        "schema-v11-scheduled-peak.tgrs",
    };
    printf("  test_band_kind_fixtures...\n");
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", TIGRIS_SCHEMA_COMPAT_DIR, names[i]);
        uint32_t len = 0;
        uint8_t *buf = load_file(path, &len);
        TEST_ASSERT(buf != NULL, "fixture readable");
        if (buf == NULL)
            continue;
        tigris_plan_t plan;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK, names[i]);
        long mismatches = 0;
        for (uint16_t s = 0; s < plan.header->num_stages; s++) {
            const tigris_stage_t *stage = &plan.stages[s];
            if (stage->chain_len != 0u)
                continue;
            int32_t rows = 0;
            tigris_stage_band_t band;
            if (tigris_stage_band_kind(&plan, stage, &band) != replayed_band_kind(&plan, stage, &rows))
                mismatches++;
        }
        TEST_ASSERT_EQ(mismatches, 0, names[i]);
        free(buf);
    }
}

static void test_float_section(void)
{
    printf("  test_float_section...\n");
    _Alignas(32) uint8_t buf[352] = {0};
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    memcpy(hdr->magic, TIGRIS_MAGIC_BYTES, 4);
    hdr->version = TIGRIS_SCHEMA_VERSION;
    hdr->file_size = sizeof(buf);
    hdr->section_dir_off = 48;
    hdr->num_tensors = 3;
    hdr->num_ops = 2;
    hdr->num_stages = 1;
    hdr->budget = 256;
    hdr->model_io_off = 6;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;
    hdr->num_quant_params = 1;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + 48);
    dir[0] = (tigris_section_entry_t){TIGRIS_SEC_TENSORS, 128};
    dir[1] = (tigris_section_entry_t){TIGRIS_SEC_OPS, 176};
    dir[2] = (tigris_section_entry_t){TIGRIS_SEC_STAGES, 256};
    dir[3] = (tigris_section_entry_t){TIGRIS_SEC_INDEX_POOL, 284};
    dir[4] = (tigris_section_entry_t){TIGRIS_SEC_SHAPE_POOL, 304};
    dir[5] = (tigris_section_entry_t){TIGRIS_SEC_STRINGS, 316};
    dir[6] = (tigris_section_entry_t){TIGRIS_SEC_QUANT_PARAMS, 320};
    plan_finish(buf);
    tigris_tensor_t *t = (tigris_tensor_t *)(buf + 128);
    for (uint16_t i = 0; i < 3; i++) {
        t[i].dtype = i == 0 ? 3 : 1;
        t[i].ndim = 1;
        t[i].shape_off = i;
        t[i].size_bytes = i == 0 ? 4 : 16;
        t[i].quant_param_idx = i == 0 ? 0 : TIGRIS_NO_QUANT_PARAM;
    }
    t[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    t[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    const int32_t shape[3] = {4, 4, 4};
    memcpy(buf + 304, shape, sizeof(shape));
    const uint16_t indices[10] = {0, 1, 1, 2, 0, 1, 0, 2, 0, 2};
    memcpy(buf + 284, indices, sizeof(indices));
    tigris_op_t *ops = (tigris_op_t *)(buf + 176);
    for (uint8_t i = 0; i < 2; i++) {
        ops[i].op_type = i == 0 ? TIGRIS_OP_DEQUANTIZE : TIGRIS_OP_RELU;
        ops[i].num_inputs = 1;
        ops[i].num_outputs = 1;
        ops[i].inputs_off = 2u * i;
        ops[i].outputs_off = 2u * i + 1u;
        ops[i].weight_idx = TIGRIS_NO_WEIGHT;
        ops[i].bias_idx = TIGRIS_NO_WEIGHT;
    }
    tigris_stage_t *stage = (tigris_stage_t *)(buf + 256);
    stage->ops_off = 4;
    stage->ops_count = 2;
    stage->inputs_off = 8;
    stage->inputs_count = 1;
    stage->outputs_off = 9;
    stage->outputs_count = 1;
    stage->tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage->chain_id = TIGRIS_NO_CHAIN;
    uint16_t *quant_header = (uint16_t *)(buf + 320);
    quant_header[0] = 1;
    quant_header[1] = 1;
    tigris_quant_param_t *quant = (tigris_quant_param_t *)(buf + 324);
    quant->scale = 0.25f;
    quant->num_channels = 1;
    quant->shift_off = 1;
    tigris_plan_t plan;
    tigris_error_t error = tigris_plan_load(buf, sizeof(buf), &plan);
    TEST_ASSERT_EQ(error, TIGRIS_OK, "Dequantize and float Relu load in an int8 plan");
    if (error != TIGRIS_OK) return;
    _Alignas(32) uint8_t arena[256];
    void *ptrs[3];
    tigris_mem_t mem;
    TEST_ASSERT_EQ(tigris_mem_init(&mem, ptrs, 3, arena, sizeof(arena), arena, 0),
                   TIGRIS_MEM_OK, "float section memory initializes");
    int8_t input[4] = {-128, -1, 1, 127};
    _Alignas(32) float real[4], result[4];
    ptrs[0] = input; ptrs[1] = real; ptrs[2] = result;
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &plan.ops[0], 0, &mem, NULL), 0,
                   "Dequantize executes through int8 dispatch");
    TEST_ASSERT_EQ(tigris_dispatch_kernel_s8(&plan, &plan.ops[1], 1, &mem, NULL), 0,
                   "float Relu executes through int8 dispatch");
    const float expected[4] = {0.0f, 0.0f, 0.25f, 31.75f};
    TEST_ASSERT(memcmp(ptrs[2], expected, sizeof(expected)) == 0,
                "raw output tensor contains float results");
    ops[0].op_type = TIGRIS_OP_RELU;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "a float activation without a conversion is refused");
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
/* The directory has room for the sections the plans built on this one add. */
#define SEMANTIC_PLAN_SIZE    312u
#define SEMANTIC_TENSORS_OFF  128u
#define SEMANTIC_OPS_OFF      176u
#define SEMANTIC_STAGES_OFF   216u
#define SEMANTIC_INDEX_OFF    244u
#define SEMANTIC_SHAPES_OFF   256u
#define SEMANTIC_STRINGS_OFF  304u

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
    plan_builder_t builder = plan_builder_begin(buf, SEMANTIC_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 3;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->model_io_off = 3;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, SEMANTIC_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, SEMANTIC_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, SEMANTIC_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, SEMANTIC_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, SEMANTIC_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, SEMANTIC_STRINGS_OFF);
    plan_finish(buf);

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

#define ELEMENTWISE_QUANT_OFF 320u
#define ELEMENTWISE_PLAN_SIZE 400u

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
    plan_finish(buf);
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

/* The attribute section follows the plan it is built on, and the record follows it. */
#define NATIVE_ATTRS_OFF(quantized) ((quantized) ? ELEMENTWISE_PLAN_SIZE : SEMANTIC_PLAN_SIZE)
#define NATIVE_REDUCTION_PLAN_SIZE (ELEMENTWISE_PLAN_SIZE + 28u)

static void build_native_reduction_plan(
    uint8_t *buf, tigris_op_type_t type, int quantized, uint8_t axis)
{
    memset(buf, 0, NATIVE_REDUCTION_PLAN_SIZE);
    if (quantized) build_elementwise_quant_plan(buf, type, 1);
    else build_semantic_plan(buf, type, 1);
    uint32_t offset = quantized ? 8u : 0u;
    uint32_t attrs_off = NATIVE_ATTRS_OFF(quantized);
    tigris_file_header_t *header = (tigris_file_header_t *)buf;
    header->file_size = attrs_off + 28u;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + header->section_dir_off);
    dir[quantized ? 7 : 6] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, attrs_off};
    plan_finish(buf);
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
            buf[NATIVE_ATTRS_OFF(q) + 12u] = 3;
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
            uint32_t attrs_off = NATIVE_ATTRS_OFF(quantized);
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
            plan_drop_section(buf, quantized ? 7u : 6u);
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
        uint32_t attrs_off = NATIVE_ATTRS_OFF(quantized);
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
        plan_drop_section(buf, quantized ? 7u : 6u);
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
        plan_finish(buf);
        /* A band of one of the four rows. */
        tiled_record(buf)->choices.input_band_extent = 1u;
        tiled_state_facts(buf, 1u, 4u, 4u, 0u, 1u);
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
        /* The stage bands along its trailing axis, so a stated height band disagrees with the plan. */
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_TILE_FACTS,
                       "native reductions and scans refuse height tiling");
    }
}

static void test_leading_band_contract(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE];
    tigris_plan_t plan;
    /* Schema 9's record states the band's counts; schema 11 has its own test. */
    build_tiled_stage_plan(buf, TIGRIS_OP_REDUCE_MEAN, 3);
    tiled_plan_to_v9(buf);
    ((tigris_op_t *)(buf + TILED_OPS_OFF))->op_type = TIGRIS_OP_REDUCE_SUM;
    tigris_tensor_t *t = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    int32_t *dims = (int32_t *)(buf + TILED_SHAPES_OFF);
    dims[0] = 7; dims[1] = 1; dims[2] = 4;
    dims[4] = 7; dims[5] = 1; dims[6] = 1;
    t[0].size_bytes = 112u;
    t[1].size_bytes = 28u;
    buf[TILED_ATTR_DATA_OFF] = 2;
    tigris_tile_plan_t *tile = (tigris_tile_plan_t *)(buf + TILED_TILE_PLANS_OFF);
    tile->original_height = 7;
    tile->tile_height = 3;
    tile->num_tiles = 3;
    tile->receptive_field = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "leading band loads");
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[128], slow[256];
    void *pointers[2] = {NULL, NULL};
    float input[28], output[7];
    for (uint32_t i = 0; i < 28u; i++) input[i] = (float)i - 10.0f;
    memset(fast, 0xA5, sizeof(fast));
    tigris_mem_t mem;
    tigris_exec_stats_t stats;
    TEST_ASSERT_EQ(tigris_mem_init(&mem, pointers, 2, fast, 96u, slow, sizeof(slow)), TIGRIS_MEM_OK, "arenas");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 0, sizeof(input)), TIGRIS_MEM_OK, "input storage");
    TEST_ASSERT_EQ(tigris_input_write(&plan, &mem, 0, input, sizeof(input)), TIGRIS_OK, "input");
    TEST_ASSERT_EQ(tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, &stats), TIGRIS_EXEC_OK, "leading execution");
    TEST_ASSERT_EQ(stats.stages_tiled, 1u, "leading stage actually tiled");
    TEST_ASSERT(mem.fast_peak <= 96u, "leading band fits arena");
    TEST_ASSERT_EQ(tigris_output_read(&plan, &mem, 1, output, sizeof(output)), TIGRIS_OK, "output");
    for (uint32_t i = 0; i < 7u; i++)
        TEST_ASSERT(output[i] == input[4u*i] + input[4u*i+1u] + input[4u*i+2u] + input[4u*i+3u], "exact sums");
    for (uint32_t i = 96u; i < sizeof(fast); i++) TEST_ASSERT_EQ(fast[i], 0xA5, "arena canary");
    tile->original_height = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "wrong leading extent");
    tile->original_height = 7;
    tile->num_tiles = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "wrong leading count");
    tile->num_tiles = 3;
    tile->tile_height = 0;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "empty leading band");
    tile->tile_height = 8;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "oversized leading band");
    tile->tile_height = 3;
    tile->halo = 1;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "leading halo");
    tile->halo = 0;
    tile->receptive_field = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "leading receptive field");
    tile->receptive_field = 1;
    tile->flags = 1;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "leading flags");
    tile->flags = 0;
    tile->tile_width = 1;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "leading width");
    tile->tile_width = 0;
    buf[TILED_ATTR_DATA_OFF] = 0;
    dims[4] = 1; dims[6] = 4; t[1].size_bytes = 16u;
    TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "cannot reduce the leading band axis");
}


static void test_gelu_attribute_contract(void)
{
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;
    build_transpose_plan(buf);
    tigris_op_t *op = (tigris_op_t *)(buf + TRANSPOSE_OPS_OFF);
    op->op_type = TIGRIS_OP_GELU;
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TRANSPOSE_TENSORS_OFF);
    tensors[1].shape_off = tensors[0].shape_off;
    tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + TRANSPOSE_ATTRS_OFF + 4u);
    attr->type = TIGRIS_OP_ATTR_GELU_APPROXIMATE;
    attr->data_len = 1;
    for (uint8_t mode = 0; mode < 2; mode++) {
        buf[TRANSPOSE_ATTR_DATA_OFF] = mode;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "GELU options load");
    }
    for (uint16_t i = 0; i < 2; i++) {
        tensors[i].dtype = 3;
        tensors[i].size_bytes /= 4u;
        tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
    }
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "int8 GELU requires quantization");
    for (uint16_t i = 0; i < 2; i++) {
        tensors[i].dtype = 1;
        tensors[i].size_bytes *= 4u;
    }
    buf[TRANSPOSE_ATTR_DATA_OFF] = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION, "invalid GELU mode");
    buf[TRANSPOSE_ATTR_DATA_OFF] = 0;
    attr->data_len = 2;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION, "invalid GELU payload length");
    attr->data_len = 1;
    op->op_type = TIGRIS_OP_ERF;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION, "GELU options belong to GELU");
    op->op_type = TIGRIS_OP_GELU;
    plan_drop_section(buf, 6u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR, "GELU requires options");
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
        plan_drop_section(buf, 6u);
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       k == 0u ? TIGRIS_OK : TIGRIS_ERR_BAD_SECTION,
                       "LeakyRelu defaults alpha but L2Normalization requires epsilon");
    }
}

static void test_activation_semantic_contract(void)
{
    _Alignas(4) uint8_t buf[ELEMENTWISE_PLAN_SIZE + 20u];
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
                header->file_size = attrs_off + 20u;
                dir[quantized ? 7 : 6] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, attrs_off};
                memset(buf + attrs_off, 0, 16u);
                plan_finish(buf);
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

#define PRELU_PLAN_SIZE 448u
#define PRELU_ATTRS_OFF 428u
#define PRELU_WEIGHTS_OFF 328u
#define PRELU_QUANT_OFF 348u

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
    plan_finish(buf);
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
        plan_drop_section(buf, 8u);
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "PRelu refuses absent operand placement metadata");
    }
    build_semantic_plan(buf, TIGRIS_OP_PRELU, 2);
    TEST_ASSERT_EQ(tigris_plan_load(buf, SEMANTIC_PLAN_SIZE, &plan), TIGRIS_ERR_BAD_OPERATOR,
                   "PRelu refuses two dynamic operands");
}

static void test_prelu_tiling_contract(void)
{
    _Alignas(4) uint8_t buf[PRELU_PLAN_SIZE + 60u];
    tigris_plan_t plan;
    for (uint8_t rank = 2; rank <= 4u; rank++) {
        build_prelu_contract_plan(buf, 0);
        /* Room for one more directory entry, then a tile section ahead of the index pool. */
        memmove(buf + SEMANTIC_TENSORS_OFF + 24u, buf + SEMANTIC_TENSORS_OFF + 16u,
                PRELU_PLAN_SIZE - SEMANTIC_TENSORS_OFF - 16u);
        tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
        for (int i = 0; i < 10; i++) dir[i].offset += 8u;
        const uint32_t tile_off = SEMANTIC_INDEX_OFF + 24u;
        memmove(buf + tile_off + sizeof(tigris_tile_plan_v11_t), buf + tile_off,
                PRELU_PLAN_SIZE + 8u - tile_off);
        for (int i = 0; i < 10; i++)
            if (dir[i].offset >= tile_off) dir[i].offset += sizeof(tigris_tile_plan_v11_t);
        memmove(&dir[4], &dir[3], 7u * sizeof(dir[0]));
        dir[3] = (tigris_section_entry_t){TIGRIS_SEC_TILE_PLANS, tile_off};
        tigris_file_header_t *header = (tigris_file_header_t *)buf;
        header->file_size = sizeof(buf);
        header->num_tile_plans = 1;
        plan_finish(buf);
        tigris_stage_t *stage = (tigris_stage_t *)(buf + SEMANTIC_STAGES_OFF + 24u);
        stage->tile_plan_idx = 0;
        stage->inputs_off = 3; stage->inputs_count = 1;
        stage->outputs_off = 4; stage->outputs_count = 1;
        tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + SEMANTIC_TENSORS_OFF + 24u);
        int32_t *dims = (int32_t *)(buf + SEMANTIC_SHAPES_OFF + 72u);
        const int32_t shapes[3][4] = {{4, 2, 1, 1}, {1, 4, 2, 1}, {1, 2, 2, 2}};
        for (int t = 0; t < 2; t++) {
            tensors[t].ndim = rank;
            memcpy(dims + 4 * t, shapes[rank - 2u], 4u * sizeof(int32_t));
        }
        /* One row at a time: a matrix's rows at rank 2, the height otherwise. */
        uint32_t rows = rank == 4u ? 2u : 4u;
        tigris_tile_plan_v11_t tile;
        memset(&tile, 0, sizeof(tile));
        tile.choices.tileable = 1;
        tile.choices.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
        /* A matrix bands its input rows; a height stripe states its output rows. */
        if (rank == 2u)
            tile.choices.input_band_extent = 1u;
        else
            tile.choices.output_band_extent = 1u;
        tile.choices.tiled_peak_bytes = 128u;
        tile.facts.tile_extent = 1u;
        tile.facts.num_tiles = rows;
        tile.facts.original_extent = rows;
        tile.facts.band_axis = rank == 2u ? 0u : 1u;
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
            /* One row at a time: a matrix's rows at rank 2, the height otherwise. */
            uint32_t rows = (uint32_t)(rank == 2u ? dims[0] : dims[1]);
            if (rank == 2u)
                tiled_record(buf)->choices.input_band_extent = 1u;
            else
                tiled_record(buf)->choices.output_band_extent = 1u;
            tiled_state_facts(buf, 1u, rows, rows, 0u, rank == 2u ? 0u : 1u);
            if (kinds[k] == TIGRIS_OP_L2_NORMALIZATION) {
                tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + sizeof(tigris_file_header_t));
                dir[7] = (tigris_section_entry_t){TIGRIS_SEC_OP_ATTRIBUTES, TILED_ATTRS_OFF};
                plan_finish(buf);
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
    tiled_record(buf)->choices.output_band_extent = 1u;
    tiled_state_facts(buf, 1u, 2u, 2u, 0u, 1u);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "L2Pool supports image height tiling");
}

static void test_resize_scale_guards(void)
{
    _Alignas(4) uint8_t buf[TILED_PLAN_SIZE + 4u];
    tigris_plan_t plan;
    build_tiled_stage_plan(buf, TIGRIS_OP_TRANSPOSE, 4);
    ((tigris_file_header_t *)buf)->file_size = sizeof(buf);
    plan_finish(buf);
    tigris_op_t *op = (tigris_op_t *)(buf + TILED_OPS_OFF);
    op->op_type = TIGRIS_OP_RESIZE_LINEAR;
    int32_t shapes[8] = {1, 10, 7, 4, 1, 7, 4, 4};
    memcpy(buf + TILED_SHAPES_OFF, shapes, sizeof(shapes));
    tigris_tensor_t *tensors = (tigris_tensor_t *)(buf + TILED_TENSORS_OFF);
    tensors[0].size_bytes = 10u * 7u * 4u * 4u;
    tensors[1].size_bytes = 7u * 4u * 4u * 4u;
    /* One output row at a time, each reading the two source rows around it. */
    tiled_record(buf)->choices.output_band_extent = 1u;
    tiled_state_facts(buf, 1u, 7u, 7u, 1u, 1u);
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
#define CONVT_PLAN_SIZE    296u
#define CONVT_TENSORS_OFF  128u
#define CONVT_OPS_OFF      160u
#define CONVT_STAGES_OFF   200u
#define CONVT_INDEX_OFF    228u
#define CONVT_SHAPES_OFF   240u
#define CONVT_STRINGS_OFF  272u
#define CONVT_WEIGHTS_OFF  276u

static void build_conv_transpose_plan(uint8_t *buf)
{
    plan_builder_t builder = plan_builder_begin(buf, CONVT_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_weights = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, CONVT_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, CONVT_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, CONVT_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, CONVT_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, CONVT_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, CONVT_STRINGS_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_WEIGHTS, CONVT_WEIGHTS_OFF);
    plan_finish(buf);

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
#define CONVTQ_PLAN_SIZE    348u
#define CONVTQ_TENSORS_OFF  136u
#define CONVTQ_OPS_OFF      168u
#define CONVTQ_STAGES_OFF   206u
#define CONVTQ_INDEX_OFF    234u
#define CONVTQ_SHAPES_OFF   244u
#define CONVTQ_STRINGS_OFF  276u
#define CONVTQ_WEIGHTS_OFF  280u
#define CONVTQ_QUANT_OFF    300u

static void build_conv_transpose_quant_plan(uint8_t *buf, uint16_t num_channels)
{
    plan_builder_t builder = plan_builder_begin(buf, CONVTQ_PLAN_SIZE, TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *hdr = builder.header;
    hdr->num_tensors = 2;
    hdr->num_ops = 1;
    hdr->num_stages = 1;
    hdr->num_weights = 1;
    hdr->num_quant_params = 1;
    hdr->model_io_off = 2;
    hdr->num_model_inputs = 1;
    hdr->num_model_outputs = 1;

    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, CONVTQ_TENSORS_OFF);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, CONVTQ_OPS_OFF);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, CONVTQ_STAGES_OFF);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, CONVTQ_INDEX_OFF);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, CONVTQ_SHAPES_OFF);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, CONVTQ_STRINGS_OFF);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_WEIGHTS, CONVTQ_WEIGHTS_OFF);
    plan_builder_section(&builder, 7u, TIGRIS_SEC_QUANT_PARAMS, CONVTQ_QUANT_OFF);
    plan_finish(buf);

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

/* Up to two chained operators t0 -> t1 -> t2 as one stage, held in memory. */
typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stage;
    uint16_t pool[6];
    int32_t shapes[15];
    tigris_plan_t plan;
} stripe_case_t;

static int stripe_contract_case(uint8_t rank, int alone, uint8_t first, uint8_t second,
                                uint16_t chain_len, uint8_t axis)
{
    static const int32_t rank3[3][3] = {{1, 8, 4}, {1, 8, 4}, {1, 8, 4}};
    static const int32_t rank4[3][4] = {{1, 8, 8, 4}, {1, 8, 8, 4}, {1, 16, 16, 4}};
    stripe_case_t c;
    memset(&c, 0, sizeof(c));
    c.header.version = TIGRIS_SCHEMA_VERSION;
    c.header.num_tensors = 3;
    c.header.num_ops = 2;
    for (uint16_t t = 0; t < 3u; t++) {
        c.tensors[t].ndim = rank;
        c.tensors[t].shape_off = (uint16_t)(t * rank);
        c.tensors[t].dtype = 1;
        for (uint8_t d = 0; d < rank; d++)
            c.shapes[t * rank + d] = rank == 3u ? rank3[t][d] :
                                     rank == 4u ? rank4[t][d] : 2;
    }
    /* pool: op0 in/out, op1 in/out, stage ops */
    c.pool[0] = 0; c.pool[1] = 1; c.pool[2] = 1; c.pool[3] = 2;
    c.pool[4] = 0; c.pool[5] = 1;
    for (uint16_t j = 0; j < 2u; j++) {
        c.ops[j].op_type = j == 0u ? first : second;
        c.ops[j].num_inputs = 1;
        c.ops[j].num_outputs = 1;
        c.ops[j].inputs_off = (uint16_t)(2u * j);
        c.ops[j].outputs_off = (uint16_t)(2u * j + 1u);
        c.ops[j].weight_idx = TIGRIS_NO_WEIGHT;
        c.ops[j].bias_idx = TIGRIS_NO_WEIGHT;
    }
    c.stage.ops_off = alone ? 5u : 4u;
    c.stage.ops_count = alone ? 1u : 2u;
    c.stage.tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    c.stage.chain_len = chain_len;
    c.stage.chain_id = chain_len != 0u ? 0u : TIGRIS_NO_CHAIN;
    c.plan.header = &c.header;
    c.plan.tensors = c.tensors;
    c.plan.ops = c.ops;
    c.plan.stages = &c.stage;
    c.plan.index_pool = c.pool;
    c.plan.shape_pool = c.shapes;
    return tigris_stage_stripe_contract(&c.plan, &c.stage, rank, axis);
}

static void test_stage_stripe_contract_axes(void)
{
    printf("  test_stage_stripe_contract_axes...\n");
    const uint8_t height = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    const uint8_t hw = TIGRIS_TILE_AXIS_HW;
    struct {
        uint8_t rank;
        int alone;
        uint8_t first, second;
        uint16_t chain_len;
        int height, hw;
        const char *what;
    } cases[] = {
        {4, 1, 0, TIGRIS_OP_CONV_TRANSPOSE, 0, 0, 1, "ConvTranspose only along HW"},
        {4, 0, TIGRIS_OP_RELU, TIGRIS_OP_CONV_TRANSPOSE, 0, 0, 1, "pointwise and ConvTranspose only along HW"},
        {4, 0, TIGRIS_OP_CONV, TIGRIS_OP_CONV_TRANSPOSE, 0, 0, 0, "Conv with ConvTranspose holds two spatial ops"},
        {4, 0, TIGRIS_OP_CONV, TIGRIS_OP_RELU, 0, 1, 1, "Conv with pointwise"},
        {4, 0, TIGRIS_OP_CONV, TIGRIS_OP_RELU, 2, 1, 1, "Conv with pointwise in a chain"},
        {4, 0, TIGRIS_OP_CONV, TIGRIS_OP_MAX_POOL, 0, 0, 0, "two spatial ops"},
        {4, 0, TIGRIS_OP_CONV, TIGRIS_OP_GLOBAL_AVG, 0, 0, 0, "a global reduction is no height op"},
        {4, 1, 0, TIGRIS_OP_RESIZE, 0, 1, 1, "Resize alone"},
        {4, 0, TIGRIS_OP_RELU, TIGRIS_OP_RESIZE, 2, 0, 0, "Resize in a chain"},
        {3, 0, TIGRIS_OP_CONV1D, TIGRIS_OP_RELU, 0, 1, 1, "Conv1D with unary pointwise"},
        {3, 0, TIGRIS_OP_CONV1D, TIGRIS_OP_ADD, 0, 0, 0, "Conv1D with binary pointwise"},
        {3, 0, TIGRIS_OP_ADD, TIGRIS_OP_RELU, 0, 1, 1, "binary pointwise without Conv1D"},
        {3, 0, TIGRIS_OP_CONV1D, TIGRIS_OP_CONV1D, 0, 0, 0, "two Conv1D"},
        {3, 0, TIGRIS_OP_CONV1D, TIGRIS_OP_RELU, 2, 0, 0, "rank 3 in a chain"},
        {3, 0, TIGRIS_OP_CONV, TIGRIS_OP_RELU, 0, 0, 0, "rank-4 spatial op at rank 3"},
        {5, 0, TIGRIS_OP_RELU, TIGRIS_OP_RELU, 0, 0, 0, "rank 5"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        TEST_ASSERT_EQ(stripe_contract_case(cases[i].rank, cases[i].alone, cases[i].first, cases[i].second,
                                            cases[i].chain_len, height),
                       cases[i].height, cases[i].what);
        TEST_ASSERT_EQ(stripe_contract_case(cases[i].rank, cases[i].alone, cases[i].first, cases[i].second,
                                            cases[i].chain_len, hw),
                       cases[i].hw, cases[i].what);
    }
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

static void test_fixture(const uint8_t *buf, uint32_t buf_len)
{
    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(err, TIGRIS_OK, "load succeeds");

    if (err != TIGRIS_OK) {
        fprintf(stderr, "  Load error: %s\n", tigris_error_str(err));
        return;
    }

    /* Header checks */
    TEST_ASSERT(memcmp(plan.header->magic, "TGRS", 4) == 0, "magic");
    TEST_ASSERT(plan.header->version >= TIGRIS_SCHEMA_VERSION_MIN &&
                plan.header->version <= TIGRIS_SCHEMA_VERSION,
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
        /* Rank zero is a scalar: the empty shape product is one element. */
        uint64_t elements = 1u;
        const int32_t *shape = tigris_tensor_shape(&plan, t);
        for (uint8_t d = 0; d < t->ndim; d++) {
            TEST_ASSERT(shape[d] > 0, "shape dim positive");
            elements *= (uint32_t)shape[d];
        }
        TEST_ASSERT_EQ(t->size_bytes, elements * tigris_dtype_size(t->dtype),
                       "tensor size matches shape and dtype");
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
}

static void test_fixture_file(const char *path)
{
    printf("  test_fixture_file(%s)...\n", path);
    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    TEST_ASSERT(buf != NULL, "fixture is readable");
    if (!buf)
        return;
    test_fixture(buf, buf_len);
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
                            "schema-v6-interface-dtype.tgrs");
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

/* The oldest supported schema validates its built-in operators as the newest
 * does: a record that no kernel can execute is refused, and so is an opcode
 * outside the built-in table. */
static void test_oldest_schema_validates_operators(void)
{
    printf("  test_oldest_schema_validates_operators...\n");
    char path[512];
    int path_len = snprintf(path, sizeof(path), "%s/%s", TIGRIS_SCHEMA_COMPAT_DIR,
                            "schema-v4-transpose.tgrs");
    TEST_ASSERT(path_len > 0 && (size_t)path_len < sizeof(path), "fixture path fits");
    uint32_t len = 0;
    uint8_t *buf = load_file(path, &len);
    TEST_ASSERT(buf != NULL, "v4 fixture is readable");
    if (!buf)
        return;

    tigris_plan_t plan;
    TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK, "v4 fixture loads");
    TEST_ASSERT_EQ(plan.header->version, TIGRIS_SCHEMA_VERSION_MIN, "fixture has the oldest schema");
    tigris_op_t *relu = NULL;
    for (uint16_t i = 0; i < plan.header->num_ops; i++) {
        if (plan.ops[i].op_type == TIGRIS_OP_RELU)
            relu = (tigris_op_t *)&plan.ops[i];
    }
    TEST_ASSERT(relu != NULL, "v4 fixture has a Relu");
    if (relu) {
        relu->fused_act = TIGRIS_ACT_RELU6 + 1u;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "v4 Relu with an activation no kernel applies is refused");
        relu->fused_act = TIGRIS_ACT_NONE;

        relu->op_type = 200;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_ERR_BAD_OPERATOR,
                       "v4 opcode outside the built-in table is refused");
        relu->op_type = TIGRIS_OP_RELU;
        TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK,
                       "restored v4 fixture loads");
    }
    free(buf);
}

/* Schemas 2 and 3 leave the meaning of opcodes outside their table open, so
 * operator validation cannot check them, and this MatMul with operands of 2x3
 * and 1x6 would read past its arena. The loader refuses both schemas, and the
 * same plan at the oldest supported schema fails operator validation. */
static void test_schemas_before_4_are_refused(void)
{
    printf("  test_schemas_before_4_are_refused...\n");
    _Alignas(4) uint8_t buf[TRANSPOSE_PLAN_SIZE];
    tigris_plan_t plan;

    build_transpose_plan(buf);
    plan_drop_section(buf, 6u);
    tigris_op_t *op = (tigris_op_t *)(buf + TRANSPOSE_OPS_OFF);
    op->op_type = TIGRIS_OP_MATMUL;
    op->num_inputs = 2;
    int32_t *shapes = (int32_t *)(buf + TRANSPOSE_SHAPES_OFF);
    shapes[2] = 1;
    shapes[3] = 6;
    for (uint32_t version = 2u; version <= 3u; version++) {
        plan_set_version(buf, version);
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                       TIGRIS_ERR_BAD_VERSION, "a schema 2 or 3 plan is refused");
    }
    plan_set_version(buf, TIGRIS_SCHEMA_VERSION_MIN);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan),
                   TIGRIS_ERR_BAD_OPERATOR,
                   "the MatMul is refused at the oldest supported schema");
}

static void test_schema_compatibility_fixtures(void)
{
    /*
     * Immutable compiler-emitted artifacts, selected to exercise the feature
     * introduced by each supported schema:
     *   v4 Transpose:tigris 208b322cab7f97c9960c63a8075944374fdfff2c
     *   v5 Conv1D:   explicit serialized length axis
     *   v6 QDQ Conv: a boundary tensor declaring its float model interface
     *   v7 MatMul:   tensors recording whether they are stored in model order
     *   v8 LayerNorm:the typed attribute kinds
     *   v9 QDQ Add:  a quantized sum stating the multipliers that scale its
     *                two operands and its result
     *   v10 state:   a variable kept across invocations in the state buffer
     *   v11 QDQ Conv:the fast-arena figure, below the budget of a compressed
     *                tiled stage, and a tile record that states its tiling
     */
    static const struct {
        const char *filename;
        uint32_t version;
        uint16_t quant_params;
        uint16_t op_attributes;
        uint16_t tile_plans;
        uint8_t tile_axis;
    } fixtures[] = {
        {"schema-v4-transpose.tgrs", TIGRIS_SCHEMA_VERSION_V4, 0, 1, 0, 0},
        {"schema-v5-conv1d-axis.tgrs", TIGRIS_SCHEMA_VERSION_V5, 0, 0, 1,
         TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH},
        {"schema-v6-interface-dtype.tgrs", TIGRIS_SCHEMA_VERSION_V6, 3, 0, 0, 0},
        {"schema-v7-tensor-layout.tgrs", TIGRIS_SCHEMA_VERSION_V7, 0, 2, 0, 0},
        {"schema-v8-layer-norm.tgrs", TIGRIS_SCHEMA_VERSION_V8, 0, 3, 0, 0},
        {"schema-v9-binary-requant.tgrs", TIGRIS_SCHEMA_VERSION_V9, 3, 1, 0, 0},
        {"schema-v10-state.tgrs", TIGRIS_SCHEMA_VERSION_V10, 0, 3, 0, 0},
        {"schema-v11-scheduled-peak.tgrs", TIGRIS_SCHEMA_VERSION_V11, 2, 0, 1,
         TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH},
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

        test_fixture_file(path);
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
            if (fixtures[i].version == TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK) {
                /* 233 bytes for the tiled stage, 256 for the weights; seven one-row bands, halo 2. */
                TEST_ASSERT_EQ(plan.scheduled_peak, 233u, "v11 states its scheduled peak");
                TEST_ASSERT(plan.scheduled_peak < plan.header->budget,
                            "v11 scheduled peak below the budget");
                TEST_ASSERT_EQ(tigris_fast_arena_required(&plan),
                               plan.scheduled_peak + tigris_weight_decompression_overhead(&plan),
                               "v11 requirement is the scheduled peak and the weights");
                const tigris_tile_facts_t *facts = &plan.tile_plans_v11[0].facts;
                tigris_tile_facts_t derived;
                TEST_ASSERT(facts->tile_extent == 1u && facts->num_tiles == 7u &&
                            facts->original_extent == 7u && facts->halo == 2u &&
                            facts->band_axis == 1u, "v11 tile record states its tiling");
                TEST_ASSERT_EQ(tigris_stage_tile_facts(&plan, &plan.stages[0], &derived), 1,
                               "v11 tiling derived");
                TEST_ASSERT(memcmp(&derived, facts, sizeof(derived)) == 0,
                            "v11 stated tiling is the derived one");
            }
            if (fixtures[i].version == TIGRIS_SCHEMA_VERSION_TILE_AXIS) {
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

/* The attribute section ends at 1280; the memory record follows. */
#define MOVEMENT_PLAN_SIZE 1284u

static void build_movement_plan(uint8_t *buf, const movement_golden_t *gold)
{
    plan_builder_t builder = plan_builder_begin(buf, MOVEMENT_PLAN_SIZE,
                                                TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *h = builder.header;
    h->num_tensors = 3;
    h->num_ops = h->num_stages = 1;
    h->num_weights = gold->index_count != 0u ? 1 : 0;
    h->num_quant_params = gold->dtype == 3 ? 3 : 0;
    h->num_model_inputs = gold->update_count != 0u ? 2 : 1;
    h->num_model_outputs = 1;
    h->model_io_off = 4;
    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, 144u);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, 192u);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, 232u);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, 260u);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, 288u);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, 360u);
    uint8_t section = 6;
    if (h->num_weights != 0u)
        plan_builder_section(&builder, section++, TIGRIS_SEC_WEIGHTS, 368u);
    if (h->num_quant_params != 0u)
        plan_builder_section(&builder, section++, TIGRIS_SEC_QUANT_PARAMS, 1024u);
    plan_builder_section(&builder, section, TIGRIS_SEC_OP_ATTRIBUTES, 1152u);
    plan_finish(buf);
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

static void test_runtime_index_contract(void)
{
    uint8_t buf[MOVEMENT_PLAN_SIZE];
    tigris_plan_t plan;
    for (size_t g = 0u; g < sizeof(movement_goldens) / sizeof(movement_goldens[0]); g++) {
        movement_golden_t gold = movement_goldens[g];
        int update = gold.op == TIGRIS_OP_DYNAMIC_UPDATE_SLICE;
        if (gold.index_count == 0u && !update) continue;
        if (update) {
            gold.index_count = gold.update_count * tigris_dtype_size(gold.dtype) / 4u;
            gold.indices = (const int32_t *)gold.update;
            gold.update_count = 0u;
        }
        build_movement_plan(buf, &gold);
        tigris_file_header_t *h = (tigris_file_header_t *)buf;
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + 144);
        tigris_op_t *op = (tigris_op_t *)(buf + 192);
        uint16_t *io = (uint16_t *)(buf + 260);
        int32_t *shape = (int32_t *)(buf + 288);
        tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + 1156);
        h->num_model_inputs = 2u;
        io[5] = 1u; io[6] = 2u;
        t[1].flags = TIGRIS_TENSOR_MODEL_INPUT;
        t[1].dtype = 6u;
        t[1].iface_dtype = 7u;
        t[1].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        op->num_inputs = 2u;
        if (update) {
            t[1].ndim = 1u;
            shape[6] = gold.rank;
            t[1].size_bytes = (uint32_t)gold.rank * 4u;
            attr->data_len = (uint8_t)(gold.rank * 4u);
            memcpy(buf + 1164, gold.metadata + gold.rank, attr->data_len);
        } else {
            uint8_t head = gold.op == TIGRIS_OP_GATHER_ND ? 1u : 3u;
            t[1].ndim = (uint8_t)gold.metadata[head - 1u];
            memcpy(shape + 6, gold.metadata + head, (size_t)t[1].ndim * 4u);
            t[1].size_bytes = gold.index_count * 4u;
            op->weight_idx = TIGRIS_NO_WEIGHT;
        }
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK, "runtime index shape and dtype load");
        op->num_inputs = 1u;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "missing runtime index refused");
        op->num_inputs = 2u;
        t[1].dtype = 1u; t[1].iface_dtype = 0u;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "float index refused");
        t[1].dtype = 6u;
        if (t[1].ndim == 0u) t[1].ndim = 1u;
        shape[6]++;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "index shape mismatch refused");
    }
}

static void test_constant_gather_contract(void)
{
    uint8_t buf[MOVEMENT_PLAN_SIZE];
    tigris_plan_t plan;
    for (size_t g = 0u; g < sizeof(movement_goldens) / sizeof(movement_goldens[0]); g++) {
        const movement_golden_t *gold = &movement_goldens[g];
        if (gold->op != TIGRIS_OP_GATHER && gold->op != TIGRIS_OP_GATHER_ND &&
            gold->op != TIGRIS_OP_EMBEDDING_LOOKUP) continue;
        build_movement_plan(buf, gold);
        tigris_tensor_t *t = (tigris_tensor_t *)(buf + 144);
        tigris_op_t *op = (tigris_op_t *)(buf + 192);
        uint8_t irank = (uint8_t)(gold->op == TIGRIS_OP_GATHER_ND ? gold->metadata[0] : gold->metadata[2]);
        uint8_t head = gold->op == TIGRIS_OP_GATHER_ND ? 1u : 3u;
        t[0].dtype = 6u; t[0].ndim = irank;
        t[0].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        t[0].size_bytes = gold->index_count * 4u;
        memcpy(buf + 288, gold->metadata + head, (size_t)irank * 4u);
        tigris_weight_entry_t *weight = (tigris_weight_entry_t *)(buf + 368);
        weight->size_bytes = gold->input_count * tigris_dtype_size(gold->dtype);
        memcpy(buf + 368 + sizeof(*weight), gold->input, weight->size_bytes);
        uint16_t count = 2u;
        memcpy(buf + 1152, &count, 2u);
        tigris_op_attribute_t *attr = (tigris_op_attribute_t *)(buf + 1156);
        attr[1] = (tigris_op_attribute_t){0u, TIGRIS_OP_ATTR_TABLE_SHAPE, (uint8_t)(gold->rank * 4u), (uint32_t)gold->metadata_count * 4u};
        memcpy(buf + 1172, gold->metadata, (size_t)gold->metadata_count * 4u);
        memcpy(buf + 1172 + attr[1].data_offset, gold->shapes, attr[1].data_len);
        tigris_error_t error = tigris_plan_load(buf, sizeof(buf), &plan);
        TEST_ASSERT_EQ(error, TIGRIS_OK, "constant Gather table loads");
        if (error != TIGRIS_OK) continue;
        uint32_t original_offset = attr[1].data_offset;
        attr[1].data_offset = MOVEMENT_PLAN_SIZE;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                       "future attribute offset is checked before another payload reads it");
        attr[1].data_offset = original_offset;
        TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                       "restored table plan loads");
        uint8_t output[1024] = {0};
        tigris_mem_t mem;
        memset(&mem, 0, sizeof(mem));
        void *pointers[3] = {(void *)(uintptr_t)gold->indices, NULL, output};
        mem.tensor_ptrs = pointers;
        mem.num_tensors = 3u;
        TEST_ASSERT_EQ(tigris_movement_execute(&plan, &plan.ops[0], 0u, &mem), 0, "constant table executes");
        TEST_ASSERT(memcmp(output, gold->output, gold->output_count * tigris_dtype_size(gold->dtype)) == 0,
                    "constant table matches Micro values");
        weight->size_bytes--;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "table byte count mismatch refused");
        weight->size_bytes++;
        attr[1].data_len = 0u;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "empty table shape refused");
        attr[1].data_len = (uint8_t)(gold->rank * 4u);
        int32_t zero = 0;
        memcpy(buf + 1172 + attr[1].data_offset, &zero, 4u);
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "zero table extent refused");
        op->num_inputs = 0u;
        TEST_ASSERT(tigris_plan_load(buf, sizeof(buf), &plan) != TIGRIS_OK, "missing table indices refused");
    }
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
    plan_builder_t builder = plan_builder_begin(buf, BOOL_PLAN_SIZE,
                                                TIGRIS_SCHEMA_VERSION);
    tigris_file_header_t *h = builder.header;
    h->num_tensors = gold->inputs + 1u;
    h->num_ops = h->num_stages = 1;
    h->num_model_inputs = gold->inputs;
    h->num_model_outputs = 1;
    h->model_io_off = 40;
    h->num_quant_params = 4;
    plan_builder_section(&builder, 0u, TIGRIS_SEC_TENSORS, 160u);
    plan_builder_section(&builder, 1u, TIGRIS_SEC_OPS, 240u);
    plan_builder_section(&builder, 2u, TIGRIS_SEC_STAGES, 280u);
    plan_builder_section(&builder, 3u, TIGRIS_SEC_INDEX_POOL, 320u);
    plan_builder_section(&builder, 4u, TIGRIS_SEC_SHAPE_POOL, 448u);
    plan_builder_section(&builder, 5u, TIGRIS_SEC_STRINGS, 528u);
    plan_builder_section(&builder, 6u, TIGRIS_SEC_QUANT_PARAMS, 544u);
    const int compare_s8 = gold->op >= TIGRIS_OP_EQUAL && gold->op <= TIGRIS_OP_GREATER_EQUAL && gold->dtypes[0] == 3;
    if (compare_s8)
        plan_builder_section(&builder, 7u, TIGRIS_SEC_OP_ATTRIBUTES, 672u);
    plan_finish(buf);
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

static void test_scalar_fixture(void)
{
    printf("  test_scalar_fixture...\n");
    const bool_golden_t scalar = {
        .op = TIGRIS_OP_LOGICAL_NOT, .inputs = 1,
        .dtypes = {9, 0, 0, 9}, .sizes = {1, 0, 0, 1},
        .scales = {1.0f, 1.0f, 1.0f, 1.0f}
    };
    _Alignas(4) uint8_t buf[BOOL_PLAN_SIZE];
    build_bool_plan(buf, &scalar);
    memcpy(buf + 528, "scalar", sizeof("scalar"));
    test_fixture(buf, sizeof(buf));
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
            /* Three bytes of strings, then the memory record. */
            h->num_quant_params = 0; h->file_size = 535;
            dir[4].offset = 528;
            plan_drop_section(buf, 6u);
            plan_finish(buf);
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

static uint32_t state_section_offset(const uint8_t *buf)
{
    uint32_t dir;
    memcpy(&dir, buf + 12, sizeof(dir));
    for (;;) {
        uint32_t type, offset;
        memcpy(&type, buf + dir, sizeof(type));
        memcpy(&offset, buf + dir + 4u, sizeof(offset));
        if (type == 0u || type == TIGRIS_SEC_STATE)
            return type == 0u ? 0u : offset;
        dir += 8u;
    }
}

/* One stateful invocation of a plan: its inputs one after another in x, its
 * outputs one after another into out, each in its own encoding. */
static tigris_exec_error_t run_with_state_once(const tigris_plan_t *plan, void *state,
                                               size_t state_size, const void *x,
                                               uint32_t x_bytes, void *out, uint32_t out_bytes,
                                               uint32_t fast_limit)
{
    const uint32_t fast_size = fast_limit ? fast_limit : tigris_fast_arena_required(plan);
    uint32_t slow_size = 0u;
    for (uint16_t t = 0; t < plan->header->num_tensors; t++)
        slow_size += 2u * (plan->tensors[t].size_bytes + TIGRIS_TENSOR_ALIGN);
    uint8_t *fast_raw = malloc((size_t)fast_size + TIGRIS_TENSOR_ALIGN);
    uint8_t *slow_raw = malloc((size_t)slow_size + TIGRIS_TENSOR_ALIGN);
    void **pointers = calloc(plan->header->num_tensors, sizeof(*pointers));
    size_t workspace_size = tigris_executor_workspace_required(plan);
    void *workspace = malloc(workspace_size);
    tigris_exec_error_t result = TIGRIS_EXEC_ERR_MEM;
    if (fast_raw && slow_raw && pointers && workspace) {
        uint8_t *fast = fast_raw + (TIGRIS_TENSOR_ALIGN - ((uintptr_t)fast_raw % TIGRIS_TENSOR_ALIGN)) % TIGRIS_TENSOR_ALIGN;
        uint8_t *slow = slow_raw + (TIGRIS_TENSOR_ALIGN - ((uintptr_t)slow_raw % TIGRIS_TENSOR_ALIGN)) % TIGRIS_TENSOR_ALIGN;
        tigris_mem_t mem;
        tigris_kernel_fn dispatch = tigris_plan_data_dtype(plan) == 3u
            ? tigris_dispatch_kernel_s8 : tigris_dispatch_kernel;
        int ready = tigris_mem_init(&mem, pointers, plan->header->num_tensors, fast, fast_size,
                                    slow, slow_size) == TIGRIS_MEM_OK;
        uint32_t at = 0u;
        for (uint16_t i = 0; ready && i < plan->header->num_model_inputs; i++) {
            const uint16_t input = plan->model_inputs[i];
            const uint32_t bytes = plan->tensors[input].size_bytes;
            ready = at + bytes <= x_bytes &&
                    tigris_mem_alloc_slow(&mem, input, bytes) == TIGRIS_MEM_OK &&
                    tigris_input_write(plan, &mem, input, (const uint8_t *)x + at, bytes) == TIGRIS_OK;
            at += bytes;
        }
        if (ready && at == x_bytes) {
            result = tigris_run_with_state(plan, &mem, dispatch, NULL, NULL,
                                           workspace, workspace_size, state, state_size);
            at = 0u;
            for (uint16_t o = 0; result == TIGRIS_EXEC_OK && o < plan->header->num_model_outputs; o++) {
                const uint16_t output = plan->model_outputs[o];
                const uint32_t bytes = plan->tensors[output].size_bytes;
                if (at + bytes > out_bytes ||
                    tigris_output_read(plan, &mem, output, (uint8_t *)out + at, bytes) != TIGRIS_OK)
                    result = TIGRIS_EXEC_ERR_MEM;
                at += bytes;
            }
            if (result == TIGRIS_EXEC_OK && at != out_bytes)
                result = TIGRIS_EXEC_ERR_MEM;
        }
    }
    free(workspace);
    free(pointers);
    free(slow_raw);
    free(fast_raw);
    return result;
}

/* One invocation of the v10 fixture on input {1, 0.5}: out receives its
 * four outputs. */
static tigris_exec_error_t run_state_fixture(const tigris_plan_t *plan, void *state,
                                             size_t state_size, float *out)
{
    const float x[2] = {1.0f, 0.5f};
    return run_with_state_once(plan, state, state_size, x, sizeof(x), out, 4u * sizeof(float), 0u);
}

static void check_state_batch_bands(const tigris_plan_t *original, const state_golden_t *golden)
{
    tigris_plan_t plan = *original;
    tigris_file_header_t header = *plan.header;
    tigris_stage_t stages[8] = {0};
    tigris_op_t ops[8];
    uint16_t indices[128];
    tigris_tile_plan_v10_t tile = {0};
    if (header.num_ops > 8u) return;
    int found = 0;
    uint16_t at = 0u;
    for (uint16_t j = 0u; j < header.num_ops; j++) {
        ops[j] = original->ops[j];
        const tigris_op_t *op = &original->ops[j];
        TEST_ASSERT(at + op->num_inputs + op->num_outputs + 1u <= 128u, "batch test index capacity");
        if (at + op->num_inputs + op->num_outputs + 1u > 128u) return;
        ops[j].inputs_off = stages[j].inputs_off = at;
        stages[j].inputs_count = op->num_inputs;
        for (uint8_t k = 0u; k < op->num_inputs; k++) indices[at++] = tigris_op_inputs(original, op)[k];
        ops[j].outputs_off = stages[j].outputs_off = at;
        stages[j].outputs_count = op->num_outputs;
        for (uint8_t k = 0u; k < op->num_outputs; k++) indices[at++] = tigris_op_outputs(original, op)[k];
        stages[j].ops_off = at;
        indices[at++] = j;
        stages[j].ops_count = 1u;
        stages[j].chain_id = TIGRIS_NO_CHAIN;
        stages[j].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        if (op->op_type == TIGRIS_OP_LSTM || op->op_type == TIGRIS_OP_SVDF) {
            int32_t time_major = 0;
            if (op->op_type == TIGRIS_OP_LSTM) {
                const uint8_t *params = tigris_op_attribute_data(original, j, TIGRIS_OP_ATTR_LSTM, NULL);
                memcpy(&time_major, params, sizeof(time_major));
            }
            const tigris_tensor_t *input = &plan.tensors[tigris_op_inputs(original, op)[0]];
            int32_t batch = tigris_tensor_shape(original, input)[time_major ? 1u : 0u];
            if (batch <= 1) return;
            stages[j].tile_plan_idx = 0u;
            found = 1;
        }
    }
    if (!found) return;
    tile.tileable = 1u;
    tile.axis = TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    tile.input_band_extent = 1u;
    tile.tiled_peak_bytes = 256u;
    header.num_stages = header.num_ops;
    header.num_tile_plans = 1u;
    plan.header = &header;
    plan.stages = stages;
    plan.ops = ops;
    plan.index_pool = indices;
    plan.tile_plans = NULL;
    plan.tile_plans_v10 = &tile;
    size_t state_size = tigris_state_required(&plan);
    void *state = malloc(state_size);
    uint8_t output[256];
    TEST_ASSERT(state != NULL && golden->output_bytes <= sizeof(output), "batch test storage");
    if (!state || golden->output_bytes > sizeof(output)) { free(state); return; }
    TEST_ASSERT_EQ(tigris_state_init(&plan, state, state_size), TIGRIS_EXEC_OK, "banded state init");
    for (uint32_t run = 0u; run < golden->runs; run++) {
        TEST_ASSERT_EQ(run_with_state_once(&plan, state, state_size,
                       (const uint8_t *)golden->input + run * golden->input_bytes, golden->input_bytes,
                       output, golden->output_bytes, 256u), TIGRIS_EXEC_OK, "256-byte batch run");
        TEST_ASSERT(memcmp(output, (const uint8_t *)golden->output + run * golden->output_bytes,
                           golden->output_bytes) == 0, "batch state sequence matches TFLite Micro");
    }
    free(state);
}

/* A copy of `golden` with `bytes` written at `offset`, loaded. */
static tigris_error_t load_altered(const state_golden_t *golden, uint8_t *copy, size_t offset,
                                   const void *bytes, size_t length)
{
    tigris_plan_t plan;
    memcpy(copy, golden->plan, golden->plan_size);
    memcpy(copy + offset, bytes, length);
    return tigris_plan_load(copy, golden->plan_size, &plan);
}

static void test_state_goldens(void)
{
    printf("  test_state_goldens...\n");
    for (size_t g = 0; g < sizeof(state_goldens) / sizeof(state_goldens[0]); g++) {
        const state_golden_t *golden = &state_goldens[g];
        uint8_t *buf = aligned_alloc(64u, ((size_t)golden->plan_size + 63u) / 64u * 64u);
        uint8_t *copy = aligned_alloc(64u, ((size_t)golden->plan_size + 63u) / 64u * 64u);
        TEST_ASSERT(buf != NULL && copy != NULL, "plan buffers");
        if (!buf || !copy) {
            free(buf);
            free(copy);
            continue;
        }
        memcpy(buf, golden->plan, golden->plan_size);
        tigris_plan_t plan;
        TEST_ASSERT_EQ(tigris_plan_load(buf, golden->plan_size, &plan), TIGRIS_OK, "stateful plan loads");
        size_t state_size = tigris_state_required(&plan);
        /* A plan without state still runs through the stateful entry. */
        void *state = malloc(state_size > 0u ? state_size : 1u);
        uint8_t out[256];
        TEST_ASSERT(state != NULL && golden->output_bytes <= sizeof(out), "state and output buffers");
        if (state && golden->output_bytes <= sizeof(out)) {
            /* The state carries from run to run as TFLite Micro keeps it. */
            TEST_ASSERT_EQ(tigris_state_init(&plan, state, state_size), TIGRIS_EXEC_OK, "state init");
            for (uint32_t run = 0; run < golden->runs; run++) {
                const uint8_t *x = (const uint8_t *)golden->input + (size_t)run * golden->input_bytes;
                const uint8_t *y = (const uint8_t *)golden->output + (size_t)run * golden->output_bytes;
                TEST_ASSERT_EQ(run_with_state_once(&plan, state, state_size, x, golden->input_bytes,
                                                   out, golden->output_bytes, 0u),
                               TIGRIS_EXEC_OK, "stateful run");
                TEST_ASSERT(memcmp(out, y, golden->output_bytes) == 0, "stateful run matches TFLite Micro");
            }
        }
        uint16_t lstm = plan.header->num_ops;
        check_state_batch_bands(&plan, golden);
        for (uint16_t n = 0; n < plan.header->num_ops; n++)
            if (plan.ops[n].op_type == TIGRIS_OP_LSTM) lstm = n;
        if (lstm < plan.header->num_ops) {
            uint8_t len = 0u;
            const uint8_t *params = tigris_op_attribute_data(&plan, lstm, TIGRIS_OP_ATTR_LSTM, &len);
            const uint8_t params_len = len;
            const uint8_t *list = tigris_op_attribute_data(&plan, lstm, TIGRIS_OP_ATTR_CONSTANTS, &len);
            size_t at = (size_t)(params - buf), listed = (size_t)(list - buf);
            const int32_t two = 2;
            const float negative = -1.0f;
            uint16_t swapped[5];
            memcpy(swapped, list, sizeof(swapped));
            swapped[0] = swapped[4];
            TEST_ASSERT_EQ(load_altered(golden, copy, at, &two, 4u), TIGRIS_ERR_BAD_SECTION,
                           "lstm time_major is 0 or 1");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 4u, &negative, 4u), TIGRIS_ERR_BAD_SECTION,
                           "lstm cell clip is not negative");
            TEST_ASSERT_EQ(load_altered(golden, copy, listed, swapped, sizeof(swapped)),
                           TIGRIS_ERR_BAD_OPERATOR, "lstm constants match their shapes");
            if (params_len == 112u) {
                const int32_t wide = 200, left = 31;
                TEST_ASSERT_EQ(load_altered(golden, copy, at + 12u, &wide, 4u), TIGRIS_ERR_BAD_OPERATOR,
                               "lstm hidden zero point is int8");
                TEST_ASSERT_EQ(load_altered(golden, copy, at + 28u, &left, 4u), TIGRIS_ERR_BAD_OPERATOR,
                               "lstm gate shifts stay applicable");
            }
        }
        uint16_t looping = plan.header->num_ops;
        for (uint16_t n = 0; n < plan.header->num_ops; n++)
            if (plan.ops[n].op_type == TIGRIS_OP_WHILE) looping = n;
        if (looping < plan.header->num_ops) {
            uint8_t len = 0u;
            const uint8_t *graphs = tigris_op_attribute_data(&plan, looping, TIGRIS_OP_ATTR_SUBGRAPHS, &len);
            const size_t at = (size_t)(graphs - buf);
            uint16_t same[2];
            memcpy(&same[0], graphs, 2u);
            same[1] = same[0];
            uint16_t swapped[2];
            memcpy(&swapped[0], graphs + 2, 2u);
            memcpy(&swapped[1], graphs, 2u);
            TEST_ASSERT_EQ(load_altered(golden, copy, at, same, sizeof(same)), TIGRIS_ERR_BAD_OPERATOR,
                           "a While's condition and body are two graphs");
            TEST_ASSERT_EQ(load_altered(golden, copy, at, swapped, sizeof(swapped)), TIGRIS_ERR_BAD_OPERATOR,
                           "a While's condition gives one bool");
        }
        /* An If inside a branch runs graphs placed after its own, never its
         * own graph or one before it. */
        if (plan.num_subgraphs > 3u) {
            uint16_t found = plan.header->num_ops;
            for (uint16_t n = 0; n < plan.header->num_ops; n++)
                if (plan.ops[n].op_type == TIGRIS_OP_IF) found = n;  /* the last If is the inner one */
            uint8_t len = 0u;
            const uint8_t *runs = tigris_op_attribute_data(&plan, found, TIGRIS_OP_ATTR_SUBGRAPHS, &len);
            const uint16_t own = 1u, main_graph = 0u;
            TEST_ASSERT_EQ(load_altered(golden, copy, (size_t)(runs - buf), &own, 2u),
                           TIGRIS_ERR_BAD_OPERATOR, "an If cannot run its own graph");
            TEST_ASSERT_EQ(load_altered(golden, copy, (size_t)(runs - buf) + 2u, &main_graph, 2u),
                           TIGRIS_ERR_BAD_OPERATOR, "an If cannot run a graph placed before it");
        }
        /* The main graph's If: the first one placed. */
        if (looping < plan.header->num_ops) {
            uint8_t len = 0u;
            const uint8_t *list = tigris_op_attribute_data(&plan, looping, TIGRIS_OP_ATTR_CONSTANTS, &len);
            if (list != NULL) {
                const uint16_t none = TIGRIS_NO_WEIGHT;
                TEST_ASSERT_EQ(load_altered(golden, copy, (size_t)(list - buf), &none, 2u),
                               TIGRIS_ERR_BAD_OPERATOR, "a While's inputs supply the loop variables without constants");
            }
        }
        uint16_t counting = plan.header->num_ops;
        for (uint16_t n = 0; n < plan.header->num_ops; n++) {
            const tigris_op_t *candidate = &plan.ops[n];
            if (candidate->op_type == TIGRIS_OP_ADD &&
                plan.tensors[tigris_op_outputs(&plan, candidate)[0]].dtype == 6u) counting = n;
        }
        if (counting < plan.header->num_ops) {
            const tigris_op_t *op = &plan.ops[counting];
            const tigris_tensor_t *sum = &plan.tensors[tigris_op_outputs(&plan, op)[0]];
            const uint8_t float32 = 1u, relu = TIGRIS_ACT_RELU;
            TEST_ASSERT(load_altered(golden, copy, (size_t)((const uint8_t *)&sum->dtype - buf),
                                     &float32, 1u) != TIGRIS_OK,
                        "int32 operands and result go together");
            TEST_ASSERT_EQ(load_altered(golden, copy, (size_t)((const uint8_t *)&op->fused_act - buf),
                                        &relu, 1u), TIGRIS_ERR_BAD_OPERATOR,
                           "int32 arithmetic takes no activation");
        }
        uint16_t converting = plan.header->num_ops;
        for (uint16_t n = 0; n < plan.header->num_ops; n++)
            if (plan.ops[n].op_type == TIGRIS_OP_QUANTIZE) converting = n;
        if (converting < plan.header->num_ops) {
            const tigris_op_t *op = &plan.ops[converting];
            const tigris_tensor_t *real = &plan.tensors[tigris_op_inputs(&plan, op)[0]];
            const tigris_tensor_t *encoded = &plan.tensors[tigris_op_outputs(&plan, op)[0]];
            const uint16_t quantized = encoded->quant_param_idx, none = TIGRIS_NO_QUANT_PARAM;
            TEST_ASSERT(load_altered(golden, copy, (size_t)((const uint8_t *)&real->quant_param_idx - buf),
                                     &quantized, 2u) != TIGRIS_OK,
                        "a Quantize reads float32 without a quantization");
            TEST_ASSERT(load_altered(golden, copy, (size_t)((const uint8_t *)&encoded->quant_param_idx - buf),
                                     &none, 2u) != TIGRIS_OK,
                        "a Quantize writes a quantized int8 tensor");
        }
        uint16_t branching = plan.header->num_ops;
        for (uint16_t n = plan.header->num_ops; n-- > 0u;)
            if (plan.ops[n].op_type == TIGRIS_OP_IF) branching = n;
        if (branching < plan.header->num_ops) {
            uint8_t len = 0u;
            const uint8_t *branches = tigris_op_attribute_data(&plan, branching, TIGRIS_OP_ATTR_SUBGRAPHS, &len);
            const size_t at = (size_t)(branches - buf);
            const size_t section = (size_t)((const uint8_t *)plan.subgraphs - buf);
            const uint16_t main_graph = 0u, missing = 9u, one = 1u;
            const uint16_t shifted = (uint16_t)(plan.subgraphs[1].first_stage + 1u);
            const tigris_op_t *op = &plan.ops[branching];
            const size_t operand = (size_t)((const uint8_t *)&tigris_op_inputs(&plan, op)[1] - buf);
            const uint16_t borrowed = plan.index_pool[plan.subgraphs[1].inputs_off];
            TEST_ASSERT_EQ(load_altered(golden, copy, at, &main_graph, 2u), TIGRIS_ERR_BAD_OPERATOR,
                           "an If cannot run the main graph");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 2u, &missing, 2u), TIGRIS_ERR_BAD_OPERATOR,
                           "an If runs a graph the plan has");
            TEST_ASSERT_EQ(load_altered(golden, copy, section - 4u, &one, 2u), TIGRIS_ERR_BAD_SECTION,
                           "a plan with control flow has a subgraph besides the main graph");
            TEST_ASSERT_EQ(load_altered(golden, copy, section + sizeof(tigris_subgraph_t), &shifted, 2u),
                           TIGRIS_ERR_BAD_SECTION, "graphs cover consecutive stage ranges");
            TEST_ASSERT_EQ(load_altered(golden, copy, operand, &borrowed, 2u), TIGRIS_ERR_BAD_TENSOR,
                           "a subgraph's tensors belong to it alone");
        }
        uint16_t detecting = plan.header->num_ops;
        for (uint16_t n = 0; n < plan.header->num_ops; n++)
            if (plan.ops[n].op_type == TIGRIS_OP_DETECTION_POSTPROCESS) detecting = n;
        if (detecting < plan.header->num_ops) {
            uint8_t len = 0u;
            const uint8_t *params = tigris_op_attribute_data(&plan, detecting, TIGRIS_OP_ATTR_DETECTION, &len);
            const size_t at = (size_t)(params - buf);
            const tigris_op_t *op = &plan.ops[detecting];
            const tigris_tensor_t *work = &plan.tensors[tigris_op_outputs(&plan, op)[4]];
            const tigris_tensor_t *count = &plan.tensors[tigris_op_outputs(&plan, op)[3]];
            const uint32_t short_work = work->size_bytes - 4u;
            const int32_t short_elements = (int32_t)(short_work / 4u);
            const size_t work_shape = (size_t)((const uint8_t *)tigris_tensor_shape(&plan, work) - buf);
            const uint16_t many_classes = 9u, no_detections = 0u;
            const uint8_t three = 3u, int8 = 3u;
            const float wide = 1.5f, zero = 0.0f, nan = NAN;
            tigris_plan_t altered;
            memcpy(copy, golden->plan, golden->plan_size);
            memcpy(copy + work_shape, &short_elements, 4u);
            memcpy(copy + ((const uint8_t *)&work->size_bytes - buf), &short_work, 4u);
            TEST_ASSERT_EQ(tigris_plan_load(copy, golden->plan_size, &altered), TIGRIS_ERR_BAD_OPERATOR,
                           "detection working memory covers its scratch");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 4u, &many_classes, 2u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection classes fit the score columns");
            TEST_ASSERT_EQ(load_altered(golden, copy, at, &no_detections, 2u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection keeps at least one result");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 6u, &three, 1u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection suppression is fast or regular");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 12u, &wide, 4u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection IoU threshold is at most 1");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 8u, &nan, 4u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection score threshold is finite");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 16u, &zero, 4u), TIGRIS_ERR_BAD_OPERATOR,
                           "detection scales divide");
            TEST_ASSERT(load_altered(golden, copy, (size_t)((const uint8_t *)&count->dtype - buf),
                                     &int8, 1u) != TIGRIS_OK,
                        "detection results are float32");
        }
        if (tigris_plan_data_dtype(&plan) == 3u && plan.ops[0].op_type == TIGRIS_OP_SVDF) {
            uint8_t len = 0u;
            const uint8_t *params = tigris_op_attribute_data(&plan, 0u, TIGRIS_OP_ATTR_SVDF, &len);
            const uint8_t *list = tigris_op_attribute_data(&plan, 0u, TIGRIS_OP_ATTR_CONSTANTS, &len);
            size_t at = (size_t)(params - buf), listed = (size_t)(list - buf);
            const int32_t zero = 0, positive = 1;
            const uint16_t missing = 0xFFFEu;
            const uint16_t swapped[3] = {1u, 0u, 2u};
            TEST_ASSERT_EQ(load_altered(golden, copy, at, &zero, 4u), TIGRIS_ERR_BAD_SECTION,
                           "svdf rank must be positive");
            TEST_ASSERT_EQ(load_altered(golden, copy, at + 12u, &positive, 4u), TIGRIS_ERR_BAD_SECTION,
                           "svdf requantization shifts right only");
            TEST_ASSERT_EQ(load_altered(golden, copy, listed, &missing, 2u), TIGRIS_ERR_BAD_SECTION,
                           "svdf constants name weights the plan has");
            TEST_ASSERT_EQ(load_altered(golden, copy, listed, swapped, sizeof(swapped)),
                           TIGRIS_ERR_BAD_OPERATOR, "svdf constants match their shapes");
            const tigris_tensor_t *kept = &plan.tensors[tigris_op_outputs(&plan, &plan.ops[0])[1]];
            uint8_t flags = (uint8_t)(kept->flags & ~TIGRIS_TENSOR_STATE);
            TEST_ASSERT(load_altered(golden, copy, (size_t)((const uint8_t *)&kept->flags - buf),
                                     &flags, 1u) != TIGRIS_OK,
                        "int16 belongs to state tensors only");
        }
        free(state);
        free(copy);
        free(buf);
    }
}

/* A compressed stage reads every weight from its own decoded block, the
 * constants an operator lists included: one placed past the block is refused. */
static void test_constants_stay_in_their_weight_block(void)
{
    printf("  test_constants_stay_in_their_weight_block...\n");
    uint32_t checked = 0u;
    for (size_t g = 0; g < sizeof(state_goldens) / sizeof(state_goldens[0]); g++) {
        const state_golden_t *golden = &state_goldens[g];
        const size_t padded = ((size_t)golden->plan_size + 63u) / 64u * 64u;
        uint8_t *buf = aligned_alloc(64u, padded);
        uint8_t *copy = aligned_alloc(64u, padded);
        tigris_plan_t plan;
        memset(&plan, 0, sizeof(plan));
        TEST_ASSERT(buf != NULL && copy != NULL, "plan buffers");
        if (buf && copy) {
            memcpy(buf, golden->plan, golden->plan_size);
            TEST_ASSERT_EQ(tigris_plan_load(buf, golden->plan_size, &plan), TIGRIS_OK, "plan loads");
        }
        if (!buf || !copy || plan.weight_blocks == NULL) {
            free(buf);
            free(copy);
            continue;
        }
        uint32_t beyond = 0u;
        for (uint16_t b = 0u; b < plan.num_weight_blocks; b++)
            if (plan.weight_blocks[b].uncompressed_size > beyond)
                beyond = plan.weight_blocks[b].uncompressed_size;
        for (uint16_t n = 0u; n < plan.header->num_ops; n++) {
            uint8_t len = 0u;
            const uint8_t *list = tigris_op_attribute_data(&plan, n, TIGRIS_OP_ATTR_CONSTANTS, &len);
            for (uint8_t k = 0u; list != NULL && k < len / 2u; k++) {
                const uint16_t index = (uint16_t)((uint16_t)list[2u * k] |
                                                  (uint16_t)((uint16_t)list[2u * k + 1u] << 8));
                if (index == TIGRIS_NO_WEIGHT)
                    continue;
                const size_t at = (size_t)((const uint8_t *)&plan.weight_entries[index].offset - buf);
                TEST_ASSERT_EQ(load_altered(golden, copy, at, &beyond, sizeof(beyond)),
                               TIGRIS_ERR_BAD_SECTION, "a constant past its stage's weight block is refused");
                checked++;
            }
        }
        free(buf);
        free(copy);
    }
    TEST_ASSERT_EQ(checked, 15u, "the 3 Svdf and 12 Lstm constants of the compressed goldens are checked");
}

/* Two constants an operator lists never share bytes of the decoded block:
 * each one moved onto the lowest offset of its operator's list is refused. */
static void test_constants_do_not_overlap_in_their_weight_block(void)
{
    printf("  test_constants_do_not_overlap_in_their_weight_block...\n");
    uint32_t checked = 0u;
    for (size_t g = 0; g < sizeof(state_goldens) / sizeof(state_goldens[0]); g++) {
        const state_golden_t *golden = &state_goldens[g];
        const size_t padded = ((size_t)golden->plan_size + 63u) / 64u * 64u;
        uint8_t *buf = aligned_alloc(64u, padded);
        uint8_t *copy = aligned_alloc(64u, padded);
        tigris_plan_t plan;
        memset(&plan, 0, sizeof(plan));
        TEST_ASSERT(buf != NULL && copy != NULL, "plan buffers");
        if (buf && copy) {
            memcpy(buf, golden->plan, golden->plan_size);
            TEST_ASSERT_EQ(tigris_plan_load(buf, golden->plan_size, &plan), TIGRIS_OK, "plan loads");
        }
        if (!buf || !copy || plan.weight_blocks == NULL) {
            free(buf);
            free(copy);
            continue;
        }
        for (uint16_t n = 0u; n < plan.header->num_ops; n++) {
            uint8_t len = 0u;
            const uint8_t *list = tigris_op_attribute_data(&plan, n, TIGRIS_OP_ATTR_CONSTANTS, &len);
            uint16_t lowest = TIGRIS_NO_WEIGHT;
            for (uint8_t k = 0u; list != NULL && k < len / 2u; k++) {
                const uint16_t index = (uint16_t)((uint16_t)list[2u * k] |
                                                  (uint16_t)((uint16_t)list[2u * k + 1u] << 8));
                if (index != TIGRIS_NO_WEIGHT && plan.weight_entries[index].size_bytes != 0u &&
                    (lowest == TIGRIS_NO_WEIGHT ||
                     plan.weight_entries[index].offset < plan.weight_entries[lowest].offset))
                    lowest = index;
            }
            for (uint8_t k = 0u; list != NULL && k < len / 2u; k++) {
                const uint16_t index = (uint16_t)((uint16_t)list[2u * k] |
                                                  (uint16_t)((uint16_t)list[2u * k + 1u] << 8));
                if (index == TIGRIS_NO_WEIGHT || index == lowest ||
                    plan.weight_entries[index].size_bytes == 0u)
                    continue;
                const uint32_t onto = plan.weight_entries[lowest].offset;
                const size_t at = (size_t)((const uint8_t *)&plan.weight_entries[index].offset - buf);
                TEST_ASSERT_EQ(load_altered(golden, copy, at, &onto, sizeof(onto)),
                               TIGRIS_ERR_BAD_SECTION, "two constants on the same block bytes are refused");
                checked++;
            }
        }
        free(buf);
        free(copy);
    }
    TEST_ASSERT_EQ(checked, 13u, "2 Svdf and 11 Lstm constants of the compressed goldens are moved");
}

/* The staged plan with three 4-byte weights in two uncompressed blocks:
 * stage 0 reads weight 0 at offset 0 of its block, stage 1 reads weight 1
 * twice at offset 0 and weight 2 at offset 4 of its own. */
#define BLOCKED_PLAN_SIZE  772u
#define BLOCKED_BLOCKS_OFF 712u

static void build_blocked_plan(uint8_t *buf)
{
    memset(buf, 0, BLOCKED_PLAN_SIZE);
    build_staged_plan(buf);
    tigris_file_header_t *hdr = (tigris_file_header_t *)buf;
    hdr->file_size = BLOCKED_PLAN_SIZE;
    hdr->num_weights = 3;
    tigris_section_entry_t *dir = (tigris_section_entry_t *)(buf + hdr->section_dir_off);
    dir[7].type = TIGRIS_SEC_WEIGHT_BLOCKS;
    dir[7].offset = BLOCKED_BLOCKS_OFF;
    plan_finish(buf);

    const tigris_weight_entry_t weights[3] = {{0u, 0u, 4u}, {0u, 0u, 4u}, {0u, 4u, 4u}};
    memcpy(buf + STAGED_WEIGHTS_OFF, weights, sizeof(weights));
    const uint16_t header[2] = {2u, TIGRIS_COMPRESS_NONE};
    memcpy(buf + BLOCKED_BLOCKS_OFF, header, sizeof(header));
    const tigris_weight_block_t blocks[2] = {{0u, 0u, 1u, 0u, 0u, 4u, 4u},
                                             {1u, 1u, 2u, 0u, 4u, 8u, 8u}};
    memcpy(buf + BLOCKED_BLOCKS_OFF + sizeof(header), blocks, sizeof(blocks));

    tigris_op_t *ops = (tigris_op_t *)(buf + STAGED_OPS_OFF);
    for (uint16_t i = 4u; i < hdr->num_ops; i++)
        ops[i].weight_idx = i < 6u ? 1u : 2u;
}

static void test_weight_ranges_overlap_only_across_blocks(void)
{
    printf("  test_weight_ranges_overlap_only_across_blocks...\n");
    _Alignas(4) uint8_t buf[BLOCKED_PLAN_SIZE];
    tigris_weight_entry_t *weights = (tigris_weight_entry_t *)(buf + STAGED_WEIGHTS_OFF);
    tigris_plan_t plan;

    build_blocked_plan(buf);
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_OK,
                   "equal offsets in two blocks and one weight read twice load");

    build_blocked_plan(buf);
    weights[2].offset = 0u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "two weights at one offset of a block are refused");

    build_blocked_plan(buf);
    weights[2].offset = 2u;
    TEST_ASSERT_EQ(tigris_plan_load(buf, sizeof(buf), &plan), TIGRIS_ERR_BAD_SECTION,
                   "two weights sharing bytes of a block are refused");
}

static void test_state_fixture(void)
{
    printf("  test_state_fixture...\n");
    char path[512];
    snprintf(path, sizeof(path), "%s/schema-v10-state.tgrs", TIGRIS_SCHEMA_COMPAT_DIR);
    uint32_t len = 0;
    uint8_t *buf = load_file(path, &len);
    TEST_ASSERT(buf != NULL, "state fixture is readable");
    if (!buf)
        return;
    tigris_plan_t plan;
    TEST_ASSERT_EQ(tigris_plan_load(buf, len, &plan), TIGRIS_OK, "state plan loads");
    TEST_ASSERT_EQ(plan.num_state, 1, "one variable");
    TEST_ASSERT_EQ(tigris_state_required(&plan), 48, "state buffer size");

    /* The variable carries from run to run, a reset starts it over, and a
     * run without a state buffer is refused. */
    float state[12];
    float first[4], later[4], again[4];
    memset(state, 0xA5, sizeof(state));
    TEST_ASSERT_EQ(tigris_state_init(&plan, state, sizeof(state)), TIGRIS_EXEC_OK, "state init");
    TEST_ASSERT_EQ(tigris_state_init(&plan, state, sizeof(state) - 4u), TIGRIS_EXEC_ERR_STATE,
                   "state init refuses a short buffer");
    TEST_ASSERT_EQ(run_state_fixture(&plan, state, sizeof(state), first), TIGRIS_EXEC_OK, "first run");
    TEST_ASSERT_EQ(run_state_fixture(&plan, state, sizeof(state), later), TIGRIS_EXEC_OK, "second run");
    TEST_ASSERT(memcmp(first, later, sizeof(first)) != 0, "the variable carries into the next run");
    TEST_ASSERT_EQ(tigris_state_init(&plan, state, sizeof(state)), TIGRIS_EXEC_OK, "state reset");
    TEST_ASSERT_EQ(run_state_fixture(&plan, state, sizeof(state), again), TIGRIS_EXEC_OK, "run after reset");
    TEST_ASSERT(memcmp(first, again, sizeof(first)) == 0, "a reset run repeats the first");
    TEST_ASSERT_EQ(run_state_fixture(&plan, NULL, 0u, again), TIGRIS_EXEC_ERR_STATE,
                   "a stateful plan needs its state buffer");
    TEST_ASSERT_EQ(run_state_fixture(&plan, state, sizeof(state) - 4u, again), TIGRIS_EXEC_ERR_STATE,
                   "a short state buffer is refused");

    /* Malformed state sections. */
    uint8_t *copy = malloc(len);
    uint32_t section = state_section_offset(buf);
    TEST_ASSERT(copy != NULL && section != 0u, "state section found");
    if (copy && section) {
        tigris_state_entry_t entry;
        memcpy(&entry, buf + section + 8u, sizeof(entry));
        memcpy(copy, buf, len);
        copy[section + 2u] = 1u;
        TEST_ASSERT(tigris_plan_load(copy, len, &plan) != TIGRIS_OK, "reserved field must be zero");
        memcpy(copy, buf, len);
        entry.bytes += 4u;
        memcpy(copy + section + 8u, &entry, sizeof(entry));
        TEST_ASSERT(tigris_plan_load(copy, len, &plan) != TIGRIS_OK, "entry size must match the tensor");
        memcpy(copy, buf, len);
        uint32_t version = TIGRIS_SCHEMA_VERSION_V9;
        memcpy(copy + 4u, &version, sizeof(version));
        TEST_ASSERT_EQ(tigris_plan_load(copy, len, &plan), TIGRIS_ERR_BAD_SECTION,
                       "state needs schema 10");
        memcpy(copy, buf, len);
        uint32_t state_bytes = 44u;
        memcpy(copy + section + 4u, &state_bytes, sizeof(state_bytes));
        TEST_ASSERT_EQ(tigris_plan_load(copy, len, &plan), TIGRIS_ERR_BAD_SECTION,
                       "a variable must fit the state buffer");
    }
    free(copy);
    free(buf);
}

int main(void)
{
    test_movement_contract();
    test_constant_gather_contract();
    test_runtime_index_contract();
    test_bool_contract();
    test_scalar_fixture();
    printf("TiGrIS Plan Loader Tests\n\n");

    printf("Error handling tests:\n");
    test_null_args();
    test_too_small();
    test_bad_magic();
    test_bad_version();
    test_size_mismatch();
    test_error_strings();
    test_section_directory_guards();
    test_memory_section();
    test_pool_alignment_guards();
    test_quant_plan_tables();
    test_counted_sections_required();
    test_chain_limit_guards();
    test_chain_shape_guards();
    test_chain_graph_guards();
    test_chain_member_runs_from_its_head();
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
    test_compact_tile_record();
    test_tiled_reduction_contract();
    test_rank4_row_band_contract();
    test_reshape_band_contract();
    test_tile_facts_contract();
    test_stripe_runs_its_stated_band();
    test_hw_record_contract();
    test_band_kind_order();
    test_float_section();
    test_operator_semantic_guards();
    test_nonweighted_operator_semantics();
    test_elementwise_semantics();
    test_native_reduction_contract();
    test_arg_contract();
    test_reduce_all_contract();
    test_cumsum_loader_contract();
    test_native_reduction_tiling_contract();
    test_leading_band_contract();
    test_activation_attribute_contract();
    test_gelu_attribute_contract();
    test_activation_semantic_contract();
    test_prelu_loader_contract();
    test_prelu_tiling_contract();
    test_activation_tiling_contract();
    test_convolution_semantic_guards();
    test_conv_transpose_semantic_guards();
    test_stage_stripe_contract_axes();
    test_conv_transpose_quant_channels_guard();

    printf("\nSchema compatibility fixtures:\n");
    test_schema_compatibility_fixtures();
    test_band_kind_fixtures();
    test_state_fixture();
    test_state_goldens();
    test_constants_stay_in_their_weight_block();
    test_constants_do_not_overlap_in_their_weight_block();
    test_weight_ranges_overlap_only_across_blocks();
    test_oldest_schema_validates_operators();
    test_schemas_before_4_are_refused();
    test_quant_param_counts_are_reconciled();

    printf("\nStruct size tests:\n");
    test_struct_sizes();

#ifdef TIGRIS_TRACE
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_NORMAL)) != 0u,
                   1, "normal path checked against trace");
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_RESHAPE)) != 0u,
                   1, "reshape path checked against trace");
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_ROWS)) != 0u,
                   1, "rows path checked against trace");
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_CONTROL)) != 0u,
                   1, "control path checked against trace");
#endif

    return test_report();
}

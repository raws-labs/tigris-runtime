/**
 * @file test_weight_arena.c
 * @brief Focused compressed-weight fast-arena lifetime tests.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_mem.h"
#include "test_harness.h"

#define TEST_ALIGN_UP(x) \
    (((uint32_t)(x) + (TIGRIS_TENSOR_ALIGN - 1u)) & \
     ~(TIGRIS_TENSOR_ALIGN - 1u))

#define CHAIN_BLOCK0_SIZE 5u
#define CHAIN_BLOCK1_SIZE 37u
#define STANDALONE_BLOCK_SIZE 17u
#define TEST_FAST_STORAGE_SIZE 1024u
#define TEST_SLOW_SIZE 1024u

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stages[2];
    int32_t shapes[12];
    uint16_t indices[12];
    tigris_weight_block_t blocks[2];
    uint8_t weight_data[64];
    tigris_plan_t plan;
} plan_fixture_t;

typedef struct {
    tigris_mem_t mem;
    void *ptrs[3];
    _Alignas(TIGRIS_TENSOR_ALIGN)
        uint8_t fast_storage[TEST_FAST_STORAGE_SIZE + TIGRIS_TENSOR_ALIGN];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[TEST_SLOW_SIZE];
    uint32_t baseline_used;
    uint32_t baseline_reserved;
    uint32_t baseline_capacity;
} arena_fixture_t;

typedef struct {
    int calls;
    int fail_on_call;
} kernel_ctx_t;

#include "trace_assert.h"

static void build_chain_plan(plan_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));

    /* A scheduled peak of 0 understates the stages, so a run reaches the in-run memory checks. */
    fx->header.version = TIGRIS_SCHEMA_VERSION;
    fx->header.num_tensors = 3;
    fx->header.num_ops = 2;
    fx->header.num_stages = 2;
    fx->header.num_model_inputs = 1;
    fx->header.num_model_outputs = 1;
    fx->header.model_io_off = 10;

    for (uint16_t i = 0; i < 3; i++) {
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].size_bytes = 2;
        fx->tensors[i].quant_param_idx = TIGRIS_NO_QUANT_PARAM;
        fx->shapes[i * 4] = 1;
        fx->shapes[i * 4 + 1] = 2;
        fx->shapes[i * 4 + 2] = 1;
        fx->shapes[i * 4 + 3] = 1;
    }
    fx->tensors[0].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;

    fx->indices[0] = 0;  /* op 0 input */
    fx->indices[1] = 1;  /* op 0 output */
    fx->indices[2] = 1;  /* op 1 input */
    fx->indices[3] = 2;  /* op 1 output */
    fx->indices[4] = 0;  /* stage 0 op */
    fx->indices[5] = 0;  /* stage 0 input */
    fx->indices[6] = 1;  /* stage 0 output */
    fx->indices[7] = 1;  /* stage 1 op */
    fx->indices[8] = 1;  /* stage 1 input */
    fx->indices[9] = 2;  /* stage 1 output */
    fx->indices[10] = 0; /* model input */
    fx->indices[11] = 2; /* model output */

    for (uint16_t i = 0; i < 2; i++) {
        fx->ops[i].op_type = TIGRIS_OP_RELU;
        fx->ops[i].num_inputs = 1;
        fx->ops[i].num_outputs = 1;
        fx->ops[i].inputs_off = (uint16_t)(i * 2);
        fx->ops[i].outputs_off = (uint16_t)(i * 2 + 1);
        fx->ops[i].weight_idx = TIGRIS_NO_WEIGHT;
        fx->ops[i].bias_idx = TIGRIS_NO_WEIGHT;
    }

    fx->stages[0].ops_off = 4;
    fx->stages[0].ops_count = 1;
    fx->stages[0].inputs_off = 5;
    fx->stages[0].inputs_count = 1;
    fx->stages[0].outputs_off = 6;
    fx->stages[0].outputs_count = 1;
    fx->stages[0].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    fx->stages[0].chain_id = 0;
    fx->stages[0].chain_len = 2;
    fx->stages[0].chain_tile_h = 1;

    fx->stages[1].ops_off = 7;
    fx->stages[1].ops_count = 1;
    fx->stages[1].inputs_off = 8;
    fx->stages[1].inputs_count = 1;
    fx->stages[1].outputs_off = 9;
    fx->stages[1].outputs_count = 1;
    fx->stages[1].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    fx->stages[1].chain_id = 0;
    fx->stages[1].chain_len = 2;

    fx->blocks[0].stage_idx = 0;
    fx->blocks[0].blob_offset = 0;
    fx->blocks[0].compressed_size = CHAIN_BLOCK0_SIZE;
    fx->blocks[0].uncompressed_size = CHAIN_BLOCK0_SIZE;
    fx->blocks[1].stage_idx = 1;
    fx->blocks[1].blob_offset = CHAIN_BLOCK0_SIZE;
    fx->blocks[1].compressed_size = CHAIN_BLOCK1_SIZE;
    fx->blocks[1].uncompressed_size = CHAIN_BLOCK1_SIZE;
    memset(fx->weight_data, 0x5a, sizeof(fx->weight_data));

    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = fx->ops;
    fx->plan.stages = fx->stages;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
    fx->plan.weight_blocks = fx->blocks;
    fx->plan.weight_blocks_data = fx->weight_data;
    fx->plan.num_weight_blocks = 2;
    fx->plan.weight_compression = TIGRIS_COMPRESS_NONE;
    fx->plan.model_inputs = &fx->indices[10];
    fx->plan.model_outputs = &fx->indices[11];
}

static void make_standalone(plan_fixture_t *fx)
{
    build_chain_plan(fx);
    fx->header.num_tensors = 2;
    fx->header.num_ops = 1;
    fx->header.num_stages = 1;
    fx->indices[11] = 1;
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    fx->stages[0].chain_id = TIGRIS_NO_CHAIN;
    fx->stages[0].chain_len = 0;
    fx->stages[0].peak_bytes = 2u * TIGRIS_TENSOR_ALIGN;
    fx->blocks[0].compressed_size = STANDALONE_BLOCK_SIZE;
    fx->blocks[0].uncompressed_size = STANDALONE_BLOCK_SIZE;
    fx->plan.num_weight_blocks = 1;
}

static void make_two_standalone_stages(plan_fixture_t *fx)
{
    build_chain_plan(fx);
    for (uint16_t i = 0; i < 2; i++) {
        fx->stages[i].chain_id = TIGRIS_NO_CHAIN;
        fx->stages[i].chain_len = 0;
        fx->stages[i].peak_bytes = 2u * TIGRIS_TENSOR_ALIGN;
    }
}

/* One compressed stage whose op adds model inputs 0 and 1 into model output 2. */
static void make_two_input_standalone(plan_fixture_t *fx)
{
    make_standalone(fx);
    fx->header.version = TIGRIS_SCHEMA_VERSION_V11;
    fx->header.num_tensors = 3;
    fx->header.num_model_inputs = 2;
    fx->header.model_io_off = 7;
    fx->tensors[1].flags = TIGRIS_TENSOR_MODEL_INPUT;
    fx->tensors[2].flags = TIGRIS_TENSOR_MODEL_OUTPUT;
    const uint16_t indices[] = {0, 1, 2, 0, 0, 1, 2, 0, 1, 2};
    memcpy(fx->indices, indices, sizeof(indices));
    fx->ops[0].op_type = TIGRIS_OP_ADD;
    fx->ops[0].num_inputs = 2;
    fx->ops[0].inputs_off = 0;
    fx->ops[0].outputs_off = 2;
    fx->stages[0].ops_off = 3;
    fx->stages[0].inputs_off = 4;
    fx->stages[0].inputs_count = 2;
    fx->stages[0].outputs_off = 6;
    fx->stages[0].peak_bytes = 3u * TIGRIS_TENSOR_ALIGN;
    fx->plan.scheduled_peak = 3u * TIGRIS_TENSOR_ALIGN;
    fx->plan.model_inputs = &fx->indices[7];
    fx->plan.model_outputs = &fx->indices[9];
}

static int init_arena(arena_fixture_t *fx, uint32_t fast_capacity,
                      uint16_t num_tensors)
{
    memset(fx, 0, sizeof(*fx));
    if (fast_capacity > TEST_FAST_STORAGE_SIZE)
        return 0;

    tigris_mem_error_t err = tigris_mem_init(
        &fx->mem, fx->ptrs, num_tensors,
        fx->fast_storage + 1, fast_capacity,
        fx->slow, sizeof(fx->slow));
    if (err != TIGRIS_MEM_OK)
        return 0;

    /* Simulate caller-owned fast bytes above the reset reservation. */
    fx->baseline_reserved = fx->mem.fast_reserved;
    fx->baseline_capacity = fx->mem.fast_size;
    if (fx->mem.fast_used > fx->mem.fast_size ||
        TIGRIS_TENSOR_ALIGN > fx->mem.fast_size - fx->mem.fast_used)
        return 0;
    memset(fx->mem.fast_base + fx->mem.fast_used, 0xa5, TIGRIS_TENSOR_ALIGN);
    fx->mem.fast_used += TIGRIS_TENSOR_ALIGN;
    fx->baseline_used = fx->mem.fast_used;
    return 1;
}

static int fast_state_restored(const arena_fixture_t *fx)
{
    if (fx->mem.fast_used != fx->baseline_used ||
        fx->mem.fast_reserved != fx->baseline_reserved ||
        fx->mem.fast_size != fx->baseline_capacity)
        return 0;
    for (uint32_t i = fx->baseline_reserved; i < fx->baseline_used; i++) {
        if (fx->mem.fast_base[i] != 0xa5)
            return 0;
    }
    return 1;
}

static int allocate_input(arena_fixture_t *arena, uint16_t tensor_idx,
                          uint32_t size)
{
    if (tigris_mem_alloc_slow(&arena->mem, tensor_idx, size) != TIGRIS_MEM_OK)
        return 0;
    memset(arena->ptrs[tensor_idx], 1, size);
    return 1;
}

static int test_kernel(const tigris_plan_t *plan, const tigris_op_t *op,
                       uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    (void)op_index;
    kernel_ctx_t *ctx = (kernel_ctx_t *)user_ctx;
    ctx->calls++;
    if (ctx->fail_on_call == ctx->calls)
        return -1;

    const uint16_t *outputs = tigris_op_outputs(plan, op);
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        uint16_t tidx = outputs[i];
        uint8_t *dst = (uint8_t *)tigris_mem_tensor_ptr(mem, tidx);
        if (!dst)
            return -1;

        uint32_t bytes = plan->tensors[tidx].size_bytes;
        if (mem->tile.active && plan->tensors[tidx].ndim == 4) {
            const int32_t *shape =
                tigris_tensor_shape(plan, &plan->tensors[tidx]);
            uint32_t full_numel = (uint32_t)shape[0] * (uint32_t)shape[1] *
                                  (uint32_t)shape[2] * (uint32_t)shape[3];
            uint32_t elem = bytes / full_numel;
            bytes = (uint32_t)shape[0] * (uint32_t)mem->tile.out_h *
                    (uint32_t)shape[2] * (uint32_t)shape[3] * elem;
        }
        memset(dst, 0x3c, bytes);
    }
    return 0;
}

static void test_overhead_sizing(void)
{
    printf("  test_overhead_sizing...\n");
    plan_fixture_t fx;
    build_chain_plan(&fx);

    uint32_t expected_chain = TEST_ALIGN_UP(CHAIN_BLOCK0_SIZE) +
                              TEST_ALIGN_UP(CHAIN_BLOCK1_SIZE);
    TEST_ASSERT_EQ(tigris_weight_decompression_overhead(&fx.plan),
                   expected_chain, "chain overhead is simultaneous sum");

    make_two_standalone_stages(&fx);
    TEST_ASSERT_EQ(tigris_weight_decompression_overhead(&fx.plan),
                   TEST_ALIGN_UP(CHAIN_BLOCK1_SIZE),
                   "standalone stages need the largest block, not their sum");

    make_standalone(&fx);
    TEST_ASSERT_EQ(tigris_weight_decompression_overhead(&fx.plan),
                   TEST_ALIGN_UP(STANDALONE_BLOCK_SIZE),
                   "standalone overhead is one aligned block");

    fx.blocks[0].uncompressed_size = UINT32_MAX;
    TEST_ASSERT_EQ(tigris_weight_decompression_overhead(&fx.plan), UINT32_MAX,
                   "unrepresentable aligned overhead saturates");
}

static void test_required_arena_sizing(void)
{
    printf("  test_required_arena_sizing...\n");
    plan_fixture_t fx;
    /* Before schema 11 the activation figure is the budget. */
    build_chain_plan(&fx);
    fx.header.version = TIGRIS_SCHEMA_VERSION_V10;

    fx.header.budget = 4096;
    uint32_t overhead = tigris_weight_decompression_overhead(&fx.plan);
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), 4096u + overhead,
                   "required arena combines activation budget and weights");

    fx.plan.num_weight_blocks = 0;
    fx.plan.weight_blocks = NULL;
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), 4096,
                   "uncompressed required arena equals activation budget");

    TEST_ASSERT_EQ(tigris_fast_arena_required(NULL), UINT32_MAX,
                   "null plan has no representable requirement");
    fx.plan.header = NULL;
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), UINT32_MAX,
                   "plan without header has no representable requirement");

    build_chain_plan(&fx);
    fx.header.version = TIGRIS_SCHEMA_VERSION_V10;
    fx.header.budget = UINT32_MAX;
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), UINT32_MAX,
                   "activation plus weight overflow saturates");

    build_chain_plan(&fx);
    fx.header.budget = 4096;
    fx.plan.scheduled_peak = 1024;
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), 1024u + overhead,
                   "schema 11 requirement is the scheduled peak and the weights");
    fx.plan.scheduled_peak = UINT32_MAX;
    TEST_ASSERT_EQ(tigris_fast_arena_required(&fx.plan), UINT32_MAX,
                   "scheduled peak plus weight overflow saturates");
}

static void test_chain_success_and_repeat(void)
{
    printf("  test_chain_success_and_repeat...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    kernel_ctx_t ctx = {0, -1};
    build_chain_plan(&plan_fx);

    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                        overhead + 3u * TIGRIS_TENSOR_ALIGN;
    TEST_ASSERT(init_arena(&arena, capacity, 3), "initialise unaligned arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate first input");

    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_OK, "first chain run succeeds at exact overhead");
    TEST_ASSERT_EQ(ctx.calls, 4, "two stages across two tiles");
    TEST_ASSERT(fast_state_restored(&arena), "chain restores caller fast state");

    ctx.calls = 0;
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate repeated input");
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_OK, "repeated chain run succeeds");
    TEST_ASSERT_EQ(ctx.calls, 4, "repeated run executes same work");
    TEST_ASSERT(fast_state_restored(&arena), "repeated run does not accumulate");
}

static void test_standalone_stages_reclaim_between_groups(void)
{
    printf("  test_standalone_stages_reclaim_between_groups...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    kernel_ctx_t ctx = {0, -1};
    make_two_standalone_stages(&plan_fx);

    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                        overhead + 2u * TIGRIS_TENSOR_ALIGN;
    TEST_ASSERT(init_arena(&arena, capacity, 3),
                "initialise exact standalone-stage arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate two-stage input");
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_OK, "standalone blocks are reclaimed per stage");
    TEST_ASSERT_EQ(ctx.calls, 2, "both standalone stages execute");
    TEST_ASSERT(fast_state_restored(&arena),
                "two standalone stages restore caller state");
}

static void test_chain_oom_restores_state(void)
{
    printf("  test_chain_oom_restores_state...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    kernel_ctx_t ctx = {0, -1};
    build_chain_plan(&plan_fx);

    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                        overhead - 1u;
    TEST_ASSERT(init_arena(&arena, capacity, 3), "initialise constrained arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate constrained input");
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_MEM, "chain weight OOM is reported");
    TEST_ASSERT_EQ(ctx.calls, 0, "OOM occurs before kernels");
    TEST_ASSERT(fast_state_restored(&arena), "chain OOM restores caller state");
}

static void test_chain_rejects_unsafe_operator(void)
{
    printf("  test_chain_rejects_unsafe_operator...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    kernel_ctx_t ctx = {0, -1};
    build_chain_plan(&plan_fx);
    plan_fx.ops[0].op_type = TIGRIS_OP_RESIZE;

    TEST_ASSERT(init_arena(&arena, TEST_FAST_STORAGE_SIZE, 3),
                "initialise unsafe-chain arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate unsafe-chain input");
    TEST_ASSERT_EQ(tigris_run(
                       &plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_TILE,
                   "shape-changing op is rejected from height chain");
    TEST_ASSERT_EQ(ctx.calls, 0, "unsafe chain fails before kernel");
    TEST_ASSERT(fast_state_restored(&arena),
                "unsafe chain rejection restores caller state");
}

static void test_standalone_success_and_errors(void)
{
    printf("  test_standalone_success_and_errors...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    kernel_ctx_t ctx = {0, -1};
    make_standalone(&plan_fx);

    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                        overhead + 2u * TIGRIS_TENSOR_ALIGN;

    TEST_ASSERT(init_arena(&arena, capacity, 2), "initialise standalone arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate standalone input");
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_OK, "standalone compressed stage succeeds");
    TEST_ASSERT(fast_state_restored(&arena), "standalone success restores state");

    TEST_ASSERT(init_arena(&arena, capacity, 2), "reinitialise kernel-error arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate kernel-error input");
    ctx.calls = 0;
    ctx.fail_on_call = 1;
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_KERNEL, "standalone kernel error propagates");
    TEST_ASSERT(fast_state_restored(&arena), "kernel error restores state");

    TEST_ASSERT(init_arena(&arena, capacity, 2), "reinitialise decode-error arena");
    TEST_ASSERT(allocate_input(&arena, 0, 2), "allocate decode-error input");
    plan_fx.plan.weight_compression = TIGRIS_COMPRESS_LZ4;
    plan_fx.blocks[0].compressed_size = 1;
    ctx.calls = 0;
    ctx.fail_on_call = -1;
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, test_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_KERNEL, "decode error propagates");
    TEST_ASSERT_EQ(ctx.calls, 0, "decode error occurs before kernel");
    TEST_ASSERT(fast_state_restored(&arena), "decode error restores state");
}

typedef struct {
    const uint8_t *weights;
    const uint8_t *input;
    uint8_t first;
} placement_ctx_t;

static int placement_kernel(const tigris_plan_t *plan, const tigris_op_t *op,
                            uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    placement_ctx_t *ctx = (placement_ctx_t *)user_ctx;
    (void)op_index;
    ctx->weights = (const uint8_t *)plan->weight_blob;
    ctx->input = (const uint8_t *)mem->tensor_ptrs[tigris_op_inputs(plan, op)[0]];
    ctx->first = ctx->input ? ctx->input[0] : 0u;
    memset(mem->tensor_ptrs[tigris_op_outputs(plan, op)[0]], 0x3c, 2);
    return 0;
}

/* An arena with caller bytes above its floor and exactly `capacity` bytes, no slow memory. */
static int init_in_place_arena(arena_fixture_t *arena, uint32_t capacity, uint16_t num_tensors)
{
    if (!init_arena(arena, capacity, num_tensors) ||
        tigris_mem_init(&arena->mem, arena->ptrs, num_tensors, arena->fast_storage + 1, capacity,
                        NULL, 0) != TIGRIS_MEM_OK)
        return 0;
    memset(arena->mem.fast_base + arena->mem.fast_used, 0xa5, TIGRIS_TENSOR_ALIGN);
    arena->mem.fast_used += TIGRIS_TENSOR_ALIGN;
    arena->baseline_used = arena->mem.fast_used;
    arena->baseline_reserved = arena->mem.fast_reserved;
    arena->baseline_capacity = arena->mem.fast_size;
    return 1;
}

static void test_standalone_inputs_in_fast(void)
{
    printf("  test_standalone_inputs_in_fast...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    placement_ctx_t ctx = {NULL, NULL, 0u};
    tigris_exec_stats_t stats;
    make_standalone(&plan_fx);
    plan_fx.header.version = TIGRIS_SCHEMA_VERSION_V11;
    plan_fx.plan.scheduled_peak = 2u * TIGRIS_TENSOR_ALIGN;

    /* Alignment padding, caller bytes, then exactly the plan's requirement. */
    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t required = tigris_fast_arena_required(&plan_fx.plan);
    TEST_ASSERT_EQ(required, overhead + 2u * TIGRIS_TENSOR_ALIGN, "requirement of the plan");
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN + required;

    /* Two runs on one manager: each places its input the same way. */
    TEST_ASSERT(init_in_place_arena(&arena, capacity, 2), "in-place arena");
    for (uint8_t run = 1u; run <= 2u; run++) {
        TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, 0), TIGRIS_OK,
                       "input placed above the caller bytes");
        uint8_t *input = (uint8_t *)arena.ptrs[0];
        TEST_ASSERT(input == arena.mem.fast_base + arena.baseline_used + overhead,
                    "the input leaves the weights room below it");
        memset(input, run, 2);
        memset(&ctx, 0, sizeof(ctx));

        TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, placement_kernel, &ctx, &stats),
                       TIGRIS_EXEC_OK, "weights go below an input read in place");
        TEST_ASSERT(ctx.weights == arena.mem.fast_base + arena.baseline_used,
                    "the weights fill the room below the input");
        TEST_ASSERT(ctx.input == input, "the input is read where it was placed");
        TEST_ASSERT_EQ(ctx.first, run, "the input keeps its data");
        TEST_ASSERT_EQ(stats.loads_bytes, 0u, "input read in place");
        TEST_ASSERT_EQ(stats.spills_bytes, 0u, "output kept in fast");
        TEST_ASSERT_EQ(stats.slow_peak, 0u, "no slow memory used");
        TEST_ASSERT_EQ(stats.weight_bytes, STANDALONE_BLOCK_SIZE, "weights decompressed");
        TEST_ASSERT(arena.ptrs[1] != NULL &&
                    (uint8_t *)arena.ptrs[1] >= arena.mem.fast_base &&
                    (uint8_t *)arena.ptrs[1] < arena.mem.fast_base + arena.mem.fast_size,
                    "output stays in the fast arena");
        TEST_ASSERT_EQ(((uint8_t *)arena.ptrs[1])[1], 0x3c, "output written");
        TEST_ASSERT(arena.ptrs[0] == NULL, "the run consumes its input");
        TEST_ASSERT(fast_state_restored(&arena), "caller bytes below the input survive");
    }

    /* No room for the weights below the input, or for the input above them. */
    TEST_ASSERT(init_in_place_arena(&arena, (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                                    overhead - 1u, 2), "arena without the weights' room");
    TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, 0), TIGRIS_ERR_MEM,
                   "no room for the weights");
    TEST_ASSERT(arena.ptrs[0] == NULL && fast_state_restored(&arena),
                "a placement without the weights' room changes nothing");
    TEST_ASSERT(init_in_place_arena(&arena, (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN +
                                    overhead + 1u, 2), "arena with only the weights' room");
    TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, 0), TIGRIS_ERR_MEM,
                   "no room for the input above the weights");
    TEST_ASSERT(arena.ptrs[0] == NULL && fast_state_restored(&arena),
                "a placement without the input's room changes nothing");

    /* One byte short of the requirement: refused before anything runs. */
    TEST_ASSERT(init_in_place_arena(&arena, capacity - 1u, 2), "short in-place arena");
    TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, 0), TIGRIS_OK,
                   "input placed in the short arena");
    uint32_t used = arena.mem.fast_used;
    void *placed = arena.ptrs[0];
    memset(&ctx, 0, sizeof(ctx));
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, placement_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_MEM, "a fast arena below the requirement is refused");
    TEST_ASSERT(ctx.weights == NULL, "no stage ran");
    TEST_ASSERT_EQ(arena.mem.fast_used, used, "the refused run leaves the arena as it was");
    TEST_ASSERT(arena.ptrs[0] == placed, "the refused run keeps the input");

    /* An input on the floor has no room below it, and its copy does not fit the exact requirement. */
    uint32_t exact = capacity - TIGRIS_TENSOR_ALIGN;
    TEST_ASSERT(init_arena(&arena, exact, 2) &&
                tigris_mem_init(&arena.mem, arena.ptrs, 2, arena.fast_storage + 1, exact,
                                NULL, 0) == TIGRIS_MEM_OK, "arena without caller bytes");
    TEST_ASSERT_EQ(tigris_mem_alloc_fast(&arena.mem, 0, 2), TIGRIS_MEM_OK,
                   "input placed on the floor");
    memset(arena.ptrs[0], 0x55, 2);
    used = arena.mem.fast_used;
    memset(&ctx, 0, sizeof(ctx));
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, placement_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_MEM, "no room below the input: refused");
    TEST_ASSERT(ctx.weights == NULL, "no weights decompressed over the input");
    TEST_ASSERT_EQ(((uint8_t *)arena.ptrs[0])[0], 0x55, "the input is untouched");
    TEST_ASSERT_EQ(arena.mem.fast_used, used, "the arena is as it was");

    /* A tensor in the room below the inputs keeps the run from taking it. */
    TEST_ASSERT(init_in_place_arena(&arena, capacity, 2), "in-place arena");
    TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, 0), TIGRIS_OK,
                   "input placed");
    arena.ptrs[1] = arena.mem.fast_base + arena.baseline_used;
    memset(arena.ptrs[1], 0x77, 2);
    memset(&ctx, 0, sizeof(ctx));
    TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, placement_kernel, &ctx, NULL),
                   TIGRIS_EXEC_ERR_MEM, "a tensor below the input: refused, not overwritten");
    TEST_ASSERT_EQ(((uint8_t *)arena.ptrs[1])[0], 0x77, "the tensor below the input survives");
}

typedef struct {
    const uint8_t *weights;
    const uint8_t *inputs[2];
    uint8_t first[2];
} two_input_ctx_t;

static int two_input_kernel(const tigris_plan_t *plan, const tigris_op_t *op,
                            uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    two_input_ctx_t *ctx = (two_input_ctx_t *)user_ctx;
    (void)op_index;
    ctx->weights = (const uint8_t *)plan->weight_blob;
    for (uint8_t i = 0u; i < 2u; i++) {
        ctx->inputs[i] = (const uint8_t *)mem->tensor_ptrs[tigris_op_inputs(plan, op)[i]];
        ctx->first[i] = ctx->inputs[i] ? ctx->inputs[i][0] : 0u;
    }
    memset(mem->tensor_ptrs[tigris_op_outputs(plan, op)[0]], 0x3c, 2);
    return 0;
}

static void test_two_inputs_in_fast(void)
{
    printf("  test_two_inputs_in_fast...\n");
    plan_fixture_t plan_fx;
    arena_fixture_t arena;
    tigris_exec_stats_t stats;
    make_two_input_standalone(&plan_fx);
    uint32_t overhead = tigris_weight_decompression_overhead(&plan_fx.plan);
    uint32_t required = tigris_fast_arena_required(&plan_fx.plan);
    TEST_ASSERT_EQ(required, overhead + 3u * TIGRIS_TENSOR_ALIGN, "requirement of the plan");
    uint32_t capacity = (TIGRIS_TENSOR_ALIGN - 1u) + TIGRIS_TENSOR_ALIGN + required;

    /* Either order: the first input placed leaves the one room, the second sits on it. */
    const uint16_t orders[2][2] = {{0u, 1u}, {1u, 0u}};
    for (uint8_t order = 0u; order < 2u; order++) {
        two_input_ctx_t ctx;
        memset(&ctx, 0, sizeof(ctx));
        TEST_ASSERT(init_in_place_arena(&arena, capacity, 3), "in-place arena");
        uint16_t first = orders[order][0], second = orders[order][1];
        TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, first), TIGRIS_OK,
                       "first input placed");
        TEST_ASSERT_EQ(tigris_input_alloc(&plan_fx.plan, &arena.mem, second), TIGRIS_OK,
                       "second input placed");
        uint8_t *low = (uint8_t *)arena.ptrs[first], *high = (uint8_t *)arena.ptrs[second];
        TEST_ASSERT(low == arena.mem.fast_base + arena.baseline_used + overhead,
                    "the first input leaves exactly one room below it");
        TEST_ASSERT(high == low + TIGRIS_TENSOR_ALIGN, "the second input follows the first");
        TEST_ASSERT_EQ(arena.mem.fast_used, arena.baseline_used + overhead +
                       2u * TIGRIS_TENSOR_ALIGN, "both inputs sit together at the top");
        memset(arena.ptrs[0], 0x11, 2);
        memset(arena.ptrs[1], 0x22, 2);
        uint8_t *placed[2] = {(uint8_t *)arena.ptrs[0], (uint8_t *)arena.ptrs[1]};

        TEST_ASSERT_EQ(tigris_run(&plan_fx.plan, &arena.mem, two_input_kernel, &ctx, &stats),
                       TIGRIS_EXEC_OK, "both inputs read in place without a slow arena");
        TEST_ASSERT(ctx.weights == arena.mem.fast_base + arena.baseline_used,
                    "the weights fill the room below the inputs");
        TEST_ASSERT(ctx.inputs[0] == placed[0] && ctx.inputs[1] == placed[1],
                    "the inputs are read where they were placed");
        TEST_ASSERT(ctx.first[0] == 0x11 && ctx.first[1] == 0x22, "the inputs keep their data");
        TEST_ASSERT_EQ(stats.loads_bytes, 0u, "inputs read in place");
        TEST_ASSERT_EQ(stats.slow_peak, 0u, "no slow memory used");
        TEST_ASSERT_EQ(stats.stages_normal, 1u, "the stage ran");
        TEST_ASSERT(arena.ptrs[2] != NULL && ((uint8_t *)arena.ptrs[2])[1] == 0x3c,
                    "output written");
        TEST_ASSERT(fast_state_restored(&arena), "caller bytes below the room survive");
    }
}

int main(void)
{
    printf("TiGrIS Compressed Weight Arena Tests\n\n");
    test_overhead_sizing();
    test_required_arena_sizing();
    test_chain_success_and_repeat();
    test_standalone_stages_reclaim_between_groups();
    test_chain_oom_restores_state();
    test_chain_rejects_unsafe_operator();
    test_standalone_success_and_errors();
    test_standalone_inputs_in_fast();
    test_two_inputs_in_fast();

#ifdef TIGRIS_TRACE
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_CHAIN)) != 0u,
                   1, "chain path checked against trace");
    TEST_ASSERT_EQ((trace_paths_seen & (1u << TIGRIS_TRACE_PATH_NORMAL)) != 0u,
                   1, "normal path checked against trace");
#endif

    return test_report();
}

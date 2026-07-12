/**
 * @file test_weight_arena.c
 * @brief Focused compressed-weight fast-arena lifetime tests.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_mem.h"

#define TEST_ALIGN_UP(x) \
    (((uint32_t)(x) + (TIGRIS_TENSOR_ALIGN - 1u)) & \
     ~(TIGRIS_TENSOR_ALIGN - 1u))

#define CHAIN_BLOCK0_SIZE 5u
#define CHAIN_BLOCK1_SIZE 37u
#define STANDALONE_BLOCK_SIZE 17u
#define TEST_FAST_STORAGE_SIZE 1024u
#define TEST_SLOW_SIZE 1024u

static int tests_run;
static int tests_passed;
static int tests_failed;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", \
                __func__, __LINE__, msg); \
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
} arena_fixture_t;

typedef struct {
    int calls;
    int fail_on_call;
} kernel_ctx_t;

static void build_chain_plan(plan_fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));

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
    }
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
        fx->mem.fast_reserved != fx->baseline_reserved)
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

int main(void)
{
    printf("TiGrIS Compressed Weight Arena Tests\n\n");
    test_overhead_sizing();
    test_chain_success_and_repeat();
    test_standalone_stages_reclaim_between_groups();
    test_chain_oom_restores_state();
    test_standalone_success_and_errors();

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);
    return tests_failed ? 1 : 0;
}

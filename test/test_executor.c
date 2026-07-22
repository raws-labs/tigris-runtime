/**
 * @file test_executor.c
 * @brief Native POSIX tests for the memory manager and stage executor.
 *
 * Usage: ./test_executor [fixture1.tgrs] [fixture2.tgrs] ...
 *
 * If no arguments given, runs built-in memory manager tests only.
 * When fixture files are provided, loads each and runs executor tests.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"

/* Alignment helper */

#define ALIGN_UP(x) (((x) + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u))

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

/* File loader helper */

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

/* Memory manager tests (no fixtures needed) */

static void test_mem_init(void)
{
    printf("  test_mem_init...\n");
    tigris_mem_t mem;
    void *ptrs[4];
    /* Use aligned buffers so initial offset is 0 */
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];

    tigris_mem_error_t err = tigris_mem_init(&mem, ptrs, 4, fast, 256, slow, 256);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "init ok");
    TEST_ASSERT_EQ(mem.num_tensors, 4, "num_tensors");
    TEST_ASSERT_EQ(mem.fast_used, 0, "fast_used=0");
    TEST_ASSERT_EQ(mem.slow_used, 0, "slow_used=0");
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(ptrs[i] == NULL, "ptrs zeroed");
    }
}

static void test_mem_init_null(void)
{
    printf("  test_mem_init_null...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t buf[64];

    TEST_ASSERT_EQ(tigris_mem_init(NULL, ptrs, 2, buf, 64, buf, 64),
                   TIGRIS_MEM_ERR_NULL, "null mem");
    TEST_ASSERT_EQ(tigris_mem_init(&mem, NULL, 2, buf, 64, buf, 64),
                   TIGRIS_MEM_ERR_NULL, "null ptrs");
    TEST_ASSERT_EQ(tigris_mem_init(&mem, ptrs, 2, NULL, 64, buf, 64),
                   TIGRIS_MEM_ERR_NULL, "null fast");
    TEST_ASSERT_EQ(tigris_mem_init(&mem, ptrs, 2, buf, 64, NULL, 64),
                   TIGRIS_MEM_ERR_NULL, "null slow");
}

static void test_mem_init_unalignable_buffer(void)
{
    printf("  test_mem_init_unalignable_buffer...\n");
    tigris_mem_t mem;
    void *ptrs[1];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[TIGRIS_TENSOR_ALIGN + 1];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[TIGRIS_TENSOR_ALIGN + 1];

#if TIGRIS_TENSOR_ALIGN > 1
    TEST_ASSERT_EQ(
        tigris_mem_init(&mem, ptrs, 1, fast + 1, 1, slow, sizeof(slow)),
        TIGRIS_MEM_ERR_OOM, "fast buffer cannot reach alignment");
    TEST_ASSERT_EQ(
        tigris_mem_init(&mem, ptrs, 1, fast, sizeof(fast), slow + 1, 1),
        TIGRIS_MEM_ERR_OOM, "slow buffer cannot reach alignment");
#else
    TEST_ASSERT_EQ(
        tigris_mem_init(&mem, ptrs, 1, fast + 1, 1, slow, sizeof(slow)),
        TIGRIS_MEM_OK, "alignment-one fast buffer remains usable");
    TEST_ASSERT_EQ(
        tigris_mem_init(&mem, ptrs, 1, fast, sizeof(fast), slow + 1, 1),
        TIGRIS_MEM_OK, "alignment-one slow buffer remains usable");
#endif
}

static void test_mem_alloc_null(void)
{
    printf("  test_mem_alloc_null...\n");
    tigris_mem_t invalid;
    memset(&invalid, 0, sizeof(invalid));
    invalid.num_tensors = 1;

    TEST_ASSERT_EQ(tigris_mem_alloc_fast(NULL, 0, 1), TIGRIS_MEM_ERR_NULL,
                   "fast null mem");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(NULL, 0, 1), TIGRIS_MEM_ERR_NULL,
                   "slow null mem");
    TEST_ASSERT_EQ(tigris_mem_alloc_fast(&invalid, 0, 1), TIGRIS_MEM_ERR_NULL,
                   "fast uninitialised arenas");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&invalid, 0, 1), TIGRIS_MEM_ERR_NULL,
                   "slow uninitialised arenas");
}

static void test_mem_alloc_fast(void)
{
    printf("  test_mem_alloc_fast...\n");
    tigris_mem_t mem;
    void *ptrs[4];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];
    tigris_mem_init(&mem, ptrs, 4, fast, 256, slow, 256);

    tigris_mem_error_t err = tigris_mem_alloc_fast(&mem, 0, 64);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "alloc ok");
    TEST_ASSERT(ptrs[0] == fast, "ptr points to fast base");
    TEST_ASSERT_EQ(mem.fast_used, ALIGN_UP(64), "fast_used after 64B alloc");

    err = tigris_mem_alloc_fast(&mem, 1, 32);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "second alloc ok");
    TEST_ASSERT(ptrs[1] == fast + ALIGN_UP(64), "ptr points to fast+aligned(64)");
    TEST_ASSERT_EQ(mem.fast_used, ALIGN_UP(64) + ALIGN_UP(32), "fast_used after 64+32B allocs");
}

static void test_mem_alloc_fast_oom(void)
{
    printf("  test_mem_alloc_fast_oom...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[64];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    tigris_mem_init(&mem, ptrs, 2, fast, 64, slow, 64);

    TEST_ASSERT_EQ(tigris_mem_alloc_fast(&mem, 0, 65), TIGRIS_MEM_ERR_OOM, "oom");
}

static void test_mem_alloc_slow(void)
{
    printf("  test_mem_alloc_slow...\n");
    tigris_mem_t mem;
    void *ptrs[4];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];
    tigris_mem_init(&mem, ptrs, 4, fast, 256, slow, 256);

    tigris_mem_error_t err = tigris_mem_alloc_slow(&mem, 2, 100);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "alloc ok");
    TEST_ASSERT(ptrs[2] == slow, "ptr points to slow base");
    TEST_ASSERT_EQ(mem.slow_used, ALIGN_UP(100), "slow_used after 100B alloc");
}

static void test_mem_alloc_slow_oom(void)
{
    printf("  test_mem_alloc_slow_oom...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[64];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    tigris_mem_init(&mem, ptrs, 2, fast, 64, slow, 64);

    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 0, 65), TIGRIS_MEM_ERR_OOM, "oom");
}

static void test_mem_alloc_alignment_overflow(void)
{
    printf("  test_mem_alloc_alignment_overflow...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[64];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    tigris_mem_init(&mem, ptrs, 2, fast, sizeof(fast), slow, sizeof(slow));

    TEST_ASSERT_EQ(tigris_mem_alloc_fast(&mem, 0, UINT32_MAX),
                   TIGRIS_MEM_ERR_OOM, "fast alignment overflow");
    TEST_ASSERT_EQ(mem.fast_used, 0, "fast offset unchanged after overflow");
    TEST_ASSERT(ptrs[0] == NULL, "fast pointer unchanged after overflow");

    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 1, UINT32_MAX),
                   TIGRIS_MEM_ERR_OOM, "slow alignment overflow");
    TEST_ASSERT_EQ(mem.slow_used, 0, "slow offset unchanged after overflow");
    TEST_ASSERT(ptrs[1] == NULL, "slow pointer unchanged after overflow");
}

static void test_mem_bad_index(void)
{
    printf("  test_mem_bad_index...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[64];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    tigris_mem_init(&mem, ptrs, 2, fast, 64, slow, 64);

    TEST_ASSERT_EQ(tigris_mem_alloc_fast(&mem, 2, 8), TIGRIS_MEM_ERR_BAD_INDEX,
                   "fast bad idx");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 99, 8), TIGRIS_MEM_ERR_BAD_INDEX,
                   "slow bad idx");
    TEST_ASSERT_EQ(tigris_mem_load(&mem, 5, 8), TIGRIS_MEM_ERR_BAD_INDEX,
                   "load bad idx");
    TEST_ASSERT_EQ(tigris_mem_spill(&mem, 5, 8), TIGRIS_MEM_ERR_BAD_INDEX,
                   "spill bad idx");
}

static void test_mem_reset(void)
{
    printf("  test_mem_reset...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[128];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[128];
    tigris_mem_init(&mem, ptrs, 2, fast, 128, slow, 128);

    tigris_mem_alloc_fast(&mem, 0, 64);
    TEST_ASSERT_EQ(mem.fast_used, 64, "used before reset");

    tigris_mem_reset_fast(&mem);
    TEST_ASSERT_EQ(mem.fast_used, 0, "used after reset");
    /* tensor_ptrs are NOT reset - that's the caller's responsibility */
}

static void test_mem_load(void)
{
    printf("  test_mem_load...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];
    tigris_mem_init(&mem, ptrs, 2, fast, 256, slow, 256);

    /* Alloc in slow and write test data */
    tigris_mem_alloc_slow(&mem, 0, 16);
    memset(ptrs[0], 0xAB, 16);

    /* Load: should copy slow -> fast */
    tigris_mem_error_t err = tigris_mem_load(&mem, 0, 16);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "load ok");
    TEST_ASSERT(ptrs[0] >= (void *)fast && ptrs[0] < (void *)(fast + 256),
                "ptr now in fast");

    /* Verify data was copied */
    uint8_t *data = (uint8_t *)ptrs[0];
    int match = 1;
    for (int i = 0; i < 16; i++) {
        if (data[i] != 0xAB) { match = 0; break; }
    }
    TEST_ASSERT(match, "data matches after load");
}

static void test_mem_load_not_set(void)
{
    printf("  test_mem_load_not_set...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[64];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    tigris_mem_init(&mem, ptrs, 2, fast, 64, slow, 64);

    TEST_ASSERT_EQ(tigris_mem_load(&mem, 0, 16), TIGRIS_MEM_ERR_NOT_SET,
                   "load null src");
}

static void test_mem_spill(void)
{
    printf("  test_mem_spill...\n");
    tigris_mem_t mem;
    void *ptrs[2];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];
    tigris_mem_init(&mem, ptrs, 2, fast, 256, slow, 256);

    /* Alloc in fast and write test data */
    tigris_mem_alloc_fast(&mem, 1, 32);
    memset(ptrs[1], 0xCD, 32);

    /* Spill: should copy fast -> slow */
    tigris_mem_error_t err = tigris_mem_spill(&mem, 1, 32);
    TEST_ASSERT_EQ(err, TIGRIS_MEM_OK, "spill ok");
    TEST_ASSERT(ptrs[1] >= (void *)slow && ptrs[1] < (void *)(slow + 256),
                "ptr now in slow");

    /* Verify data was copied */
    uint8_t *data = (uint8_t *)ptrs[1];
    int match = 1;
    for (int i = 0; i < 32; i++) {
        if (data[i] != 0xCD) { match = 0; break; }
    }
    TEST_ASSERT(match, "data matches after spill");
}

static void test_mem_alignment(void)
{
    printf("  test_mem_alignment...\n");
    tigris_mem_t mem;
    void *ptrs[4];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[256];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[256];
    tigris_mem_init(&mem, ptrs, 4, fast, 256, slow, 256);

    /* Alloc odd sizes - pointers should be TIGRIS_TENSOR_ALIGN-byte aligned */
    tigris_mem_alloc_fast(&mem, 0, 1);
    TEST_ASSERT_EQ(mem.fast_used, ALIGN_UP(1), "1 byte aligned up");

    tigris_mem_alloc_fast(&mem, 1, 7);
    TEST_ASSERT_EQ(mem.fast_used, ALIGN_UP(1) + ALIGN_UP(7), "1+7 bytes aligned up");

    tigris_mem_alloc_slow(&mem, 2, 3);
    TEST_ASSERT_EQ(mem.slow_used, ALIGN_UP(3), "3 bytes aligned up");

    tigris_mem_alloc_slow(&mem, 3, 13);
    TEST_ASSERT_EQ(mem.slow_used, ALIGN_UP(3) + ALIGN_UP(13), "3+13 bytes aligned up");

    /* All pointers are TIGRIS_TENSOR_ALIGN-byte aligned */
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(((uintptr_t)ptrs[i] % TIGRIS_TENSOR_ALIGN) == 0, "pointer aligned");
    }
}

static void test_mem_error_strings(void)
{
    printf("  test_mem_error_strings...\n");
    TEST_ASSERT(strlen(tigris_mem_error_str(TIGRIS_MEM_OK)) > 0, "OK string");
    TEST_ASSERT(strlen(tigris_mem_error_str(TIGRIS_MEM_ERR_OOM)) > 0, "OOM string");
    TEST_ASSERT(strlen(tigris_mem_error_str(TIGRIS_MEM_ERR_NOT_SET)) > 0, "NOT_SET string");
    TEST_ASSERT(strlen(tigris_mem_error_str((tigris_mem_error_t)999)) > 0, "unknown string");
}

/* Executor tests (require fixtures) */

/* Stub kernel: memsets each output tensor to 0x01, increments counter */
typedef struct {
    int call_count;
    int fail_on_call;  /* -1 = never fail */
} stub_ctx_t;

static int stub_kernel(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    (void)op_index;
    stub_ctx_t *ctx = (stub_ctx_t *)user_ctx;
    ctx->call_count++;

    if (ctx->fail_on_call >= 0 && ctx->call_count == ctx->fail_on_call)
        return -1;

    const uint16_t *outs = tigris_op_outputs(plan, op);
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        uint16_t tidx = outs[i];
        void *ptr = tigris_mem_tensor_ptr(mem, tidx);
        if (ptr) {
            memset(ptr, 0x01, plan->tensors[tidx].size_bytes);
        }
    }
    return 0;
}

static void test_exec_null_args(void)
{
    printf("  test_exec_null_args...\n");
    tigris_plan_t plan;
    tigris_mem_t mem;
    memset(&plan, 0, sizeof(plan));
    memset(&mem, 0, sizeof(mem));

    TEST_ASSERT_EQ(tigris_run(NULL, &mem, stub_kernel, NULL, NULL),
                   TIGRIS_EXEC_ERR_NULL, "null plan");
    TEST_ASSERT_EQ(tigris_run(&plan, NULL, stub_kernel, NULL, NULL),
                   TIGRIS_EXEC_ERR_NULL, "null mem");
    TEST_ASSERT_EQ(tigris_run(&plan, &mem, NULL, NULL, NULL),
                   TIGRIS_EXEC_ERR_NULL, "null kernel");
    TEST_ASSERT_EQ(tigris_run_with_workspace(
                       &plan, &mem, stub_kernel, NULL, NULL, NULL),
                   TIGRIS_EXEC_ERR_WORKSPACE, "null explicit workspace");
}

static void test_exec_workspace_contract(void)
{
    printf("  test_exec_workspace_contract...\n");
    tigris_executor_workspace_t workspace;
    tigris_plan_t empty_plan;
    memset(&empty_plan, 0, sizeof(empty_plan));

    TEST_ASSERT_EQ(tigris_executor_workspace_size(), sizeof(workspace),
                   "workspace query matches public type");
    TEST_ASSERT(sizeof(workspace) == TIGRIS_EXECUTOR_WORKSPACE_BYTES,
                "workspace has configured byte size");
    TEST_ASSERT(((uintptr_t)&workspace % _Alignof(void *)) == 0,
                "workspace is pointer aligned");
    TEST_ASSERT_EQ(tigris_executor_workspace_required(NULL), 0,
                   "null plan has no workspace requirement");
    TEST_ASSERT_EQ(tigris_executor_workspace_required(&empty_plan), 0,
                   "incomplete plan has no workspace requirement");
    TEST_ASSERT(strlen(tigris_exec_error_str(TIGRIS_EXEC_ERR_WORKSPACE)) > 0,
                "workspace error string");
}

static void test_exec_workspace_chain_sizing(void)
{
    printf("  test_exec_workspace_chain_sizing...\n");
    tigris_file_header_t header;
    tigris_op_t ops[3];
    tigris_stage_t stages[2];
    tigris_plan_t plan;
    const uint16_t indices[] = {
        0, 1,       /* stage 0 ops */
        0, 1,       /* stage 0 inputs */
        2,          /* stage 0 output */
        2,          /* stage 1 op */
        2,          /* stage 1 input */
        2, 3, 4,    /* stage 1 outputs */
    };

    memset(&header, 0, sizeof(header));
    memset(ops, 0, sizeof(ops));
    memset(stages, 0, sizeof(stages));
    memset(&plan, 0, sizeof(plan));
    header.num_tensors = 5;
    header.num_ops = 3;
    header.num_stages = 2;
    ops[0].op_type = TIGRIS_OP_CONV;
    ops[1].op_type = TIGRIS_OP_DEPTHWISE;
    ops[2].op_type = TIGRIS_OP_CONV;
    stages[0].ops_off = 0;
    stages[0].ops_count = 2;
    stages[0].inputs_off = 2;
    stages[0].inputs_count = 2;
    stages[0].outputs_off = 4;
    stages[0].outputs_count = 1;
    stages[0].chain_id = 0;
    stages[0].chain_len = 2;
    stages[1].ops_off = 5;
    stages[1].ops_count = 1;
    stages[1].inputs_off = 6;
    stages[1].inputs_count = 1;
    stages[1].outputs_off = 7;
    stages[1].outputs_count = 3;
    stages[1].chain_id = 0;
    stages[1].chain_len = 2;
    plan.header = &header;
    plan.ops = ops;
    plan.stages = stages;
    plan.index_pool = indices;

    TEST_ASSERT_EQ(
        tigris_executor_workspace_required(&plan),
        TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(5, 2, 3, 2, 2),
        "query includes actual chain and spatial capacities");
}

/** Helper: compute sum of all non-constant tensor sizes for slow buffer sizing. */
static uint32_t total_tensor_bytes(const tigris_plan_t *plan)
{
    uint32_t total = 0;
    for (uint16_t i = 0; i < plan->header->num_tensors; i++) {
        if (!(plan->tensors[i].flags & TIGRIS_TENSOR_CONSTANT))
            total += plan->tensors[i].size_bytes;
    }
    return total;
}

/**
 * Run executor on a fixture: allocate model inputs in slow, fill with 0xAA,
 * run with stub kernel (fills outputs with 0x01), verify output is in slow.
 */
static void test_exec_fixture(const char *path, int expected_kernel_calls)
{
    printf("  test_exec_fixture(%s, expect %d kernel calls)...\n",
           path, expected_kernel_calls);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load fixture\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t lerr = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(lerr, TIGRIS_OK, "load plan");
    if (lerr != TIGRIS_OK) { free(buf); return; }

    /* Allocate buffers */
    uint16_t nt = plan.header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    uint32_t fast_size = total_tensor_bytes(&plan);
    uint8_t *fast_buf = (uint8_t *)malloc(fast_size);
    /* Slow: enough for all activation tensors (2x for load+spill headroom) */
    uint32_t slow_size = total_tensor_bytes(&plan) * 2;
    uint8_t *slow_buf = (uint8_t *)malloc(slow_size);

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, ptrs, nt,
                                              fast_buf, fast_size,
                                              slow_buf, slow_size);
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init");

    /* Allocate model inputs in slow and fill with test data */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "alloc model input");
        memset(ptrs[tidx], 0xAA, plan.tensors[tidx].size_bytes);
    }

    /* Run executor */
    stub_ctx_t ctx = { .call_count = 0, .fail_on_call = -1 };
    tigris_executor_workspace_t workspace;
    tigris_exec_error_t eerr = tigris_run_with_workspace(
        &plan, &mem, stub_kernel, &ctx, NULL, &workspace);
    TEST_ASSERT_EQ(eerr, TIGRIS_EXEC_OK, "exec ok");
    TEST_ASSERT_EQ(ctx.call_count, expected_kernel_calls, "kernel call count");

    /* Model outputs should be in slow buffer and filled with 0x01 */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        void *ptr = ptrs[tidx];
        TEST_ASSERT(ptr != NULL, "output ptr not null");
        TEST_ASSERT(ptr >= (void *)slow_buf && ptr < (void *)(slow_buf + slow_size),
                    "output in slow buffer");
        if (ptr) {
            uint8_t *data = (uint8_t *)ptr;
            int match = 1;
            for (uint32_t b = 0; b < plan.tensors[tidx].size_bytes; b++) {
                if (data[b] != 0x01) { match = 0; break; }
            }
            TEST_ASSERT(match, "output data is 0x01");
        }
    }

    free(ptrs);
    free(fast_buf);
    free(slow_buf);
    free(buf);
}

static void test_exec_kernel_error(const char *path)
{
    printf("  test_exec_kernel_error(%s)...\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load fixture\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t lerr = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(lerr, TIGRIS_OK, "load plan");
    if (lerr != TIGRIS_OK) { free(buf); return; }

    uint16_t nt = plan.header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    uint32_t fast_size = total_tensor_bytes(&plan);
    uint8_t *fast_buf = (uint8_t *)malloc(fast_size);
    uint32_t slow_size = total_tensor_bytes(&plan) * 2;
    uint8_t *slow_buf = (uint8_t *)malloc(slow_size);

    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, nt, fast_buf, fast_size, slow_buf, slow_size);

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        memset(ptrs[tidx], 0xAA, plan.tensors[tidx].size_bytes);
    }

    /* Kernel fails on 2nd call */
    stub_ctx_t ctx = { .call_count = 0, .fail_on_call = 2 };
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, stub_kernel, &ctx, NULL);
    TEST_ASSERT_EQ(eerr, TIGRIS_EXEC_ERR_KERNEL, "kernel error propagated");
    TEST_ASSERT_EQ(ctx.call_count, 2, "stopped on 2nd call");

    free(ptrs);
    free(fast_buf);
    free(slow_buf);
    free(buf);
}

static void test_exec_error_strings(void)
{
    printf("  test_exec_error_strings...\n");
    TEST_ASSERT(strlen(tigris_exec_error_str(TIGRIS_EXEC_OK)) > 0, "OK string");
    TEST_ASSERT(strlen(tigris_exec_error_str(TIGRIS_EXEC_ERR_KERNEL)) > 0, "kernel string");
    TEST_ASSERT(strlen(tigris_exec_error_str(TIGRIS_EXEC_ERR_NO_STAGES)) > 0, "no_stages string");
    TEST_ASSERT(strlen(tigris_exec_error_str((tigris_exec_error_t)999)) > 0, "unknown string");
}

typedef struct {
    int preserved;
} compaction_ctx_t;

static int compaction_kernel(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index,
    tigris_mem_t *mem, void *user_ctx)
{
    compaction_ctx_t *ctx = (compaction_ctx_t *)user_ctx;
    const uint16_t *inputs = tigris_op_inputs(plan, op);
    const uint16_t *outputs = tigris_op_outputs(plan, op);
    uint8_t *output = (uint8_t *)mem->tensor_ptrs[outputs[0]];

    if (op_index == 0) {
        memset(output, 0x5a, plan->tensors[outputs[0]].size_bytes);
    } else {
        const uint8_t *input = (const uint8_t *)mem->tensor_ptrs[inputs[0]];
        ctx->preserved = input && input[0] == 0x5a && input[31] == 0x5a;
        memset(output, 0x6b, plan->tensors[outputs[0]].size_bytes);
    }
    return 0;
}

static void test_exec_compaction_preserves_data(void)
{
    printf("  test_exec_compaction_preserves_data...\n");
    tigris_file_header_t header;
    tigris_tensor_t tensors[3];
    tigris_op_t ops[2];
    tigris_stage_t stage;
    const uint16_t indices[] = {0, 1, 1, 2, 0, 1, 0, 2};
    tigris_plan_t plan;
    void *ptrs[3];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[96];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[128];
    tigris_mem_t mem;
    uint8_t workspace[
        TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(3, 1, 1, 0, 0)];
    tigris_exec_stats_t stats;
    compaction_ctx_t ctx = {0};

    memset(&header, 0, sizeof(header));
    memset(tensors, 0, sizeof(tensors));
    memset(ops, 0, sizeof(ops));
    memset(&stage, 0, sizeof(stage));
    memset(&plan, 0, sizeof(plan));

    header.num_tensors = 3;
    header.num_ops = 2;
    header.num_stages = 1;
    header.num_model_inputs = 1;
    header.num_model_outputs = 1;
    tensors[0].size_bytes = 32;
    tensors[1].size_bytes = 32;
    tensors[2].size_bytes = 48;
    for (size_t i = 0; i < 3; i++)
        tensors[i].ndim = 1;

    ops[0].num_inputs = 1;
    ops[0].num_outputs = 1;
    ops[0].inputs_off = 0;
    ops[0].outputs_off = 1;
    ops[1].num_inputs = 1;
    ops[1].num_outputs = 1;
    ops[1].inputs_off = 2;
    ops[1].outputs_off = 3;

    stage.ops_off = 4;
    stage.ops_count = 2;
    stage.inputs_off = 6;
    stage.inputs_count = 1;
    stage.outputs_off = 7;
    stage.outputs_count = 1;
    stage.tile_plan_idx = TIGRIS_NO_TILE_PLAN;
    stage.chain_id = TIGRIS_NO_CHAIN;

    plan.header = &header;
    plan.tensors = tensors;
    plan.ops = ops;
    plan.stages = &stage;
    plan.index_pool = indices;
    plan.model_inputs = &indices[6];
    plan.model_outputs = &indices[7];

    size_t workspace_required = tigris_executor_workspace_required(&plan);
    TEST_ASSERT_EQ(workspace_required, sizeof(workspace),
                   "simple plan gets exact compile-time workspace size");
    TEST_ASSERT(workspace_required < tigris_executor_workspace_size(),
                "simple plan avoids generic workspace reservation");

    TEST_ASSERT_EQ(tigris_mem_init(
                       &mem, ptrs, 3, fast, sizeof(fast), slow, sizeof(slow)),
                   TIGRIS_MEM_OK, "compaction memory init");
    TEST_ASSERT_EQ(tigris_mem_alloc_slow(&mem, 0, tensors[0].size_bytes),
                   TIGRIS_MEM_OK, "compaction input allocation");
    memset(mem.tensor_ptrs[0], 0x11, tensors[0].size_bytes);

    TEST_ASSERT_EQ(tigris_run_with_workspace_buffer(
                       &plan, &mem, compaction_kernel, &ctx, &stats,
                       workspace, workspace_required - 1),
                   TIGRIS_EXEC_ERR_WORKSPACE,
                   "one-byte-short workspace is rejected");
    TEST_ASSERT_EQ(tigris_run_with_workspace_buffer(
                       &plan, &mem, compaction_kernel, &ctx, &stats,
                       workspace, sizeof(workspace)),
                   TIGRIS_EXEC_OK, "compaction execution");
    TEST_ASSERT_EQ(stats.compactions, 1, "fast arena compacted once");
    TEST_ASSERT(ctx.preserved, "live tensor data survives compaction");
    TEST_ASSERT(mem.tensor_ptrs[2] != NULL, "compacted output remains live");
    TEST_ASSERT(((uint8_t *)mem.tensor_ptrs[2])[47] == 0x6b,
                "output survives slow-arena compaction");
}

/* Main */

int main(int argc, char *argv[])
{
    printf("TiGrIS Memory + Executor Tests\n\n");

    printf("Memory manager tests:\n");
    test_mem_init();
    test_mem_init_null();
    test_mem_init_unalignable_buffer();
    test_mem_alloc_null();
    test_mem_alloc_fast();
    test_mem_alloc_fast_oom();
    test_mem_alloc_slow();
    test_mem_alloc_slow_oom();
    test_mem_alloc_alignment_overflow();
    test_mem_bad_index();
    test_mem_reset();
    test_mem_load();
    test_mem_load_not_set();
    test_mem_spill();
    test_mem_alignment();
    test_mem_error_strings();

    printf("\nExecutor error tests:\n");
    test_exec_null_args();
    test_exec_workspace_contract();
    test_exec_workspace_chain_sizing();
    test_exec_compaction_preserves_data();
    test_exec_error_strings();

    if (argc > 1) {
        printf("\nExecutor fixture tests:\n");

        /* linear_3op: 1 stage, 3 ops -> 3 kernel calls */
        test_exec_fixture(argv[1], 3);

        if (argc > 2) {
            /* conv_relu_chain: 1 stage, 2 ops -> 2 kernel calls (Relu fused into Conv) */
            test_exec_fixture(argv[2], 2);

            /* Kernel error test: needs >=2 ops, conv_relu_chain has 2 */
            test_exec_kernel_error(argv[2]);
        }
    } else {
        printf("\n(No fixture files provided - pass .tgrs paths as arguments)\n");
    }

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

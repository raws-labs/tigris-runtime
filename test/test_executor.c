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
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, stub_kernel, &ctx, NULL);
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

/* Main */

int main(int argc, char *argv[])
{
    printf("TiGrIS Memory + Executor Tests\n\n");

    printf("Memory manager tests:\n");
    test_mem_init();
    test_mem_init_null();
    test_mem_alloc_fast();
    test_mem_alloc_fast_oom();
    test_mem_alloc_slow();
    test_mem_alloc_slow_oom();
    test_mem_bad_index();
    test_mem_reset();
    test_mem_load();
    test_mem_load_not_set();
    test_mem_spill();
    test_mem_alignment();
    test_mem_error_strings();

    printf("\nExecutor error tests:\n");
    test_exec_null_args();
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

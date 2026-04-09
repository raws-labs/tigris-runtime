/**
 * @file test_e2e.c
 * @brief E2E POSIX simulation - full MobileNetV2 inference on host.
 *
 * Usage: ./test_e2e <path_to.tgrs>
 *
 * Loads a compiled plan, allocates buffers, fills inputs with 1.0f,
 * runs inference with real float32 kernels, prints top-5 output values.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"

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

/* Timing + metrics */

static double elapsed_ms(struct timespec *start, struct timespec *end)
{
    double s  = (double)(end->tv_sec  - start->tv_sec)  * 1e3;
    double ns = (double)(end->tv_nsec - start->tv_nsec) / 1e6;
    return s + ns;
}

typedef struct {
    double  time_ms;
    uint32_t fast_size;
    uint32_t slow_size;
    double  max_err;
    double  mean_err;
    int     ok;
} e2e_result_t;

static e2e_result_t result_nontiled;
static e2e_result_t result_tiled;

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

/* Helpers */

/** Total non-constant tensor bytes (for slow buffer sizing). */
static uint32_t total_tensor_bytes(const tigris_plan_t *plan)
{
    uint32_t total = 0;
    for (uint16_t i = 0; i < plan->header->num_tensors; i++) {
        if (!(plan->tensors[i].flags & TIGRIS_TENSOR_CONSTANT))
            total += plan->tensors[i].size_bytes;
    }
    return total;
}

/* Reference comparison */

static void compare_reference(const float *output, uint32_t num_floats,
                              const char *ref_path, e2e_result_t *result)
{
    printf("\n  Comparing against reference: %s\n", ref_path);

    uint32_t ref_len = 0;
    uint8_t *ref_buf = load_file(ref_path, &ref_len);
    if (!ref_buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load reference file\n");
        return;
    }

    uint32_t ref_floats = ref_len / sizeof(float);
    TEST_ASSERT_EQ(ref_floats, num_floats, "reference float count matches output");
    if (ref_floats != num_floats) {
        fprintf(stderr, "  Reference has %u floats, output has %u\n",
                ref_floats, num_floats);
        free(ref_buf);
        return;
    }

    const float *ref = (const float *)ref_buf;
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    int max_err_idx = 0;

    for (uint32_t i = 0; i < num_floats; i++) {
        double err = fabs((double)output[i] - (double)ref[i]);
        sum_abs_err += err;
        if (err > max_abs_err) {
            max_abs_err = err;
            max_err_idx = (int)i;
        }
    }

    double mean_abs_err = sum_abs_err / (double)num_floats;

    printf("  Max absolute error:  %.6e (idx %d: got %.6f, ref %.6f)\n",
           max_abs_err, max_err_idx,
           (double)output[max_err_idx], (double)ref[max_err_idx]);
    printf("  Mean absolute error: %.6e\n", mean_abs_err);

    if (result) {
        result->max_err  = max_abs_err;
        result->mean_err = mean_abs_err;
    }

    TEST_ASSERT(max_abs_err < 1e-3, "max absolute error < 1e-3");

    if (max_abs_err < 1e-3)
        printf("  [PASS] Reference comparison\n");
    else
        fprintf(stderr, "  [FAIL] Reference comparison: max error %.6e >= 1e-3\n",
                max_abs_err);

    free(ref_buf);
}

/* E2E test */

static void test_e2e(const char *path, const char *ref_path)
{
    printf("  Loading plan: %s\n", path);

    /* Load file */
    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load plan file\n");
        return;
    }
    printf("  File size: %u bytes (%.1f MB)\n", buf_len, buf_len / (1024.0 * 1024.0));

    /* Parse plan */
    tigris_plan_t plan;
    tigris_error_t lerr = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(lerr, TIGRIS_OK, "plan load");
    if (lerr != TIGRIS_OK) { free(buf); return; }

    printf("  Model: %s\n", tigris_model_name(&plan));
    printf("  Tensors: %u, Ops: %u, Stages: %u, Weights: %u\n",
           plan.header->num_tensors, plan.header->num_ops,
           plan.header->num_stages, plan.header->num_weights);
    printf("  Budget: %u bytes, Peak: %u bytes\n",
           plan.header->budget, plan.header->peak);

    /* Allocate memory */
    uint16_t nt = plan.header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    TEST_ASSERT(ptrs != NULL, "alloc tensor ptrs");
    if (!ptrs) { free(buf); return; }

    uint32_t fast_size = plan.header->peak;
    if (fast_size < total_tensor_bytes(&plan))
        fast_size = total_tensor_bytes(&plan);
    fast_size += tigris_weight_decompression_overhead(&plan);
    uint8_t *fast_buf = (uint8_t *)malloc(fast_size);

    uint32_t slow_size = total_tensor_bytes(&plan) * 2;
    uint8_t *slow_buf = (uint8_t *)malloc(slow_size);

    TEST_ASSERT(fast_buf != NULL, "alloc fast buffer");
    TEST_ASSERT(slow_buf != NULL, "alloc slow buffer");
    if (!fast_buf || !slow_buf) {
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    printf("  Fast buffer: %u bytes (%.1f MB)\n", fast_size, fast_size / (1024.0 * 1024.0));
    printf("  Slow buffer: %u bytes (%.1f MB)\n", slow_size, slow_size / (1024.0 * 1024.0));

    /* Init memory manager */
    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, ptrs, nt,
                                              fast_buf, fast_size,
                                              slow_buf, slow_size);
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init");

    /* Allocate model inputs in slow and fill with 1.0f */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "alloc model input");

        /* Fill with 1.0f for deterministic, non-zero input */
        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);
        float *data = (float *)ptrs[tidx];
        for (uint32_t j = 0; j < num_floats; j++)
            data[j] = 1.0f;

        printf("  Input tensor %u: %u bytes (%.1f KB)\n",
               tidx, plan.tensors[tidx].size_bytes,
               plan.tensors[tidx].size_bytes / 1024.0);
    }

    /* Run inference */
    printf("\n  Running inference (%u stages, %u ops)...\n",
           plan.header->num_stages, plan.header->num_ops);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = elapsed_ms(&t0, &t1);

    TEST_ASSERT_EQ(eerr, TIGRIS_EXEC_OK, "inference completed");

    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "  Inference failed: %s\n", tigris_exec_error_str(eerr));
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    result_nontiled.time_ms   = ms;
    result_nontiled.fast_size = fast_size;
    result_nontiled.slow_size = slow_size;
    result_nontiled.ok        = 1;

    printf("  Inference complete! (%.1f ms)\n\n", ms);

    /* Read and validate output */
    uint32_t total_output_floats = 0;
    const float *first_output = NULL;

    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        float *output = (float *)ptrs[tidx];
        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);

        TEST_ASSERT(output != NULL, "output ptr not null");
        if (!output) continue;

        if (i == 0) first_output = output;
        total_output_floats += num_floats;

        printf("  Output tensor %u (%s): %u floats\n",
               tidx, tigris_tensor_name(&plan, &plan.tensors[tidx]), num_floats);

        /* Validate: no NaN/Inf */
        int has_nan = 0, has_inf = 0;
        double total_sum = 0.0;
        for (uint32_t j = 0; j < num_floats; j++) {
            if (isnan(output[j])) has_nan = 1;
            if (isinf(output[j])) has_inf = 1;
            total_sum += (double)output[j];
        }
        TEST_ASSERT(!has_nan, "no NaN in output");
        TEST_ASSERT(!has_inf, "no Inf in output");
        TEST_ASSERT(isfinite(total_sum), "output sum is finite");

        printf("  Output sum: %.6f\n", total_sum);

        /* Print top-5 values and indices */
        printf("  Top-5 outputs:\n");
        int top5_idx[5] = {-1, -1, -1, -1, -1};
        float top5_val[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};

        for (uint32_t j = 0; j < num_floats; j++) {
            /* Find insertion point in top-5 */
            for (int k = 0; k < 5; k++) {
                if (output[j] > top5_val[k]) {
                    /* Shift down */
                    for (int m = 4; m > k; m--) {
                        top5_val[m] = top5_val[m - 1];
                        top5_idx[m] = top5_idx[m - 1];
                    }
                    top5_val[k] = output[j];
                    top5_idx[k] = (int)j;
                    break;
                }
            }
        }

        for (int k = 0; k < 5 && top5_idx[k] >= 0; k++) {
            printf("    [%d] idx=%d  val=%.6f\n", k, top5_idx[k], (double)top5_val[k]);
        }
    }

    /* Optional reference comparison */
    if (ref_path && first_output) {
        compare_reference(first_output, total_output_floats, ref_path, &result_nontiled);
    }

    free(ptrs);
    free(fast_buf);
    free(slow_buf);
    free(buf);
}

/**
 * Compute minimum fast buffer size for stages that cannot be height-tiled
 * (i.e. stages containing non-4D tensors like GlobalAvgPool->FC).
 */
static uint32_t min_untileable_fast(const tigris_plan_t *plan)
{
    uint32_t max_needed = 0;
    for (uint16_t s = 0; s < plan->header->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        const uint16_t *sin  = tigris_stage_inputs(plan, stage);
        const uint16_t *sout = tigris_stage_outputs(plan, stage);
        const uint16_t *sops = tigris_stage_ops(plan, stage);

        /* Check if all tensors (inputs + all op outputs) are 4D */
        int all_4d = 1;
        uint32_t total = 0;
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            total += plan->tensors[sin[i]].size_bytes;
            if (plan->tensors[sin[i]].ndim != 4) all_4d = 0;
        }
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            const tigris_op_t *op = &plan->ops[sops[j]];
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                total += plan->tensors[outs[k]].size_bytes;
                if (plan->tensors[outs[k]].ndim != 4) all_4d = 0;
            }
        }
        (void)sout;

        if (!all_4d && total > max_needed)
            max_needed = total;
    }
    return max_needed;
}

/* E2E test - tiled mode */

static void test_e2e_tiled(const char *path, const char *ref_path)
{
    printf("  [TILED] Loading plan: %s\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load plan file\n");
        return;
    }

    tigris_plan_t plan;
    tigris_error_t lerr = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(lerr, TIGRIS_OK, "plan load (tiled)");
    if (lerr != TIGRIS_OK) { free(buf); return; }

    printf("  [TILED] Model: %s\n", tigris_model_name(&plan));
    printf("  [TILED] Budget: %u bytes (%.1f KB)\n",
           plan.header->budget, plan.header->budget / 1024.0);

    uint16_t nt = plan.header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    TEST_ASSERT(ptrs != NULL, "alloc tensor ptrs (tiled)");
    if (!ptrs) { free(buf); return; }

    /* fast_size = max(budget, min needed for untileable stages) + weight overhead */
    uint32_t untile_min = min_untileable_fast(&plan);
    uint32_t fast_size = plan.header->budget;
    if (untile_min > fast_size) {
        printf("  [TILED] Note: untileable stage needs %u bytes (%.1f KB), "
               "raising fast from budget %u\n",
               untile_min, untile_min / 1024.0, fast_size);
        fast_size = untile_min;
    }
    fast_size += tigris_weight_decompression_overhead(&plan);
    uint8_t *fast_buf = (uint8_t *)malloc(fast_size);

    uint32_t slow_size = total_tensor_bytes(&plan) * 2;
    uint8_t *slow_buf = (uint8_t *)malloc(slow_size);

    TEST_ASSERT(fast_buf != NULL, "alloc fast buffer (tiled)");
    TEST_ASSERT(slow_buf != NULL, "alloc slow buffer (tiled)");
    if (!fast_buf || !slow_buf) {
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    printf("  [TILED] Fast buffer: %u bytes (%.1f KB)\n",
           fast_size, fast_size / 1024.0);
    printf("  [TILED] Slow buffer: %u bytes (%.1f MB)\n",
           slow_size, slow_size / (1024.0 * 1024.0));

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, ptrs, nt,
                                              fast_buf, fast_size,
                                              slow_buf, slow_size);
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init (tiled)");

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "alloc model input (tiled)");

        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);
        float *data = (float *)ptrs[tidx];
        for (uint32_t j = 0; j < num_floats; j++)
            data[j] = 1.0f;
    }

    printf("\n  [TILED] Running tiled inference (%u stages, %u ops)...\n",
           plan.header->num_stages, plan.header->num_ops);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = elapsed_ms(&t0, &t1);

    TEST_ASSERT_EQ(eerr, TIGRIS_EXEC_OK, "tiled inference completed");

    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "  [TILED] Inference failed: %s\n", tigris_exec_error_str(eerr));
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    result_tiled.time_ms   = ms;
    result_tiled.fast_size = fast_size;
    result_tiled.slow_size = slow_size;
    result_tiled.ok        = 1;

    printf("  [TILED] Inference complete! (%.1f ms)\n\n", ms);

    /* Read output and compare against reference */
    uint32_t total_output_floats = 0;
    const float *first_output = NULL;

    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        float *output = (float *)ptrs[tidx];
        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);

        TEST_ASSERT(output != NULL, "output ptr not null (tiled)");
        if (!output) continue;

        if (i == 0) first_output = output;
        total_output_floats += num_floats;

        int has_nan = 0, has_inf = 0;
        for (uint32_t j = 0; j < num_floats; j++) {
            if (isnan(output[j])) has_nan = 1;
            if (isinf(output[j])) has_inf = 1;
        }
        TEST_ASSERT(!has_nan, "no NaN in tiled output");
        TEST_ASSERT(!has_inf, "no Inf in tiled output");
    }

    if (ref_path && first_output) {
        printf("  [TILED] ");
        compare_reference(first_output, total_output_floats, ref_path, &result_tiled);
    }

    free(ptrs);
    free(fast_buf);
    free(slow_buf);
    free(buf);
}

/* Comparison summary */

static void print_comparison(void)
{
    if (!result_nontiled.ok || !result_tiled.ok) {
        printf("\n  (comparison skipped - one or both runs failed)\n");
        return;
    }

    printf("\n--- Comparison: Non-tiled vs Tiled ---\n\n");
    printf("  %-24s %12s %12s %8s\n", "", "Non-tiled", "Tiled", "Ratio");
    printf("  %-24s %12s %12s %8s\n",
           "", "", "", "");
    printf("  %-24s %9.1f ms %9.1f ms %7.2fx\n",
           "Inference time",
           result_nontiled.time_ms, result_tiled.time_ms,
           result_tiled.time_ms / result_nontiled.time_ms);
    printf("  %-24s %9.1f KB %9.1f KB %7.2fx\n",
           "Fast (SRAM) buffer",
           result_nontiled.fast_size / 1024.0, result_tiled.fast_size / 1024.0,
           (double)result_nontiled.fast_size / (double)result_tiled.fast_size);
    printf("  %-24s %9.1f MB %9.1f MB\n",
           "Slow (PSRAM) buffer",
           result_nontiled.slow_size / (1024.0 * 1024.0),
           result_tiled.slow_size / (1024.0 * 1024.0));
    printf("  %-24s %11.2e %11.2e\n",
           "Max abs error vs ref",
           result_nontiled.max_err, result_tiled.max_err);
    printf("  %-24s %11.2e %11.2e\n",
           "Mean abs error vs ref",
           result_nontiled.mean_err, result_tiled.mean_err);

    double sram_saving = 1.0 - (double)result_tiled.fast_size /
                                (double)result_nontiled.fast_size;
    printf("\n  SRAM reduction: %.1f%% (%.1f MB -> %.1f KB)\n",
           sram_saving * 100.0,
           result_nontiled.fast_size / (1024.0 * 1024.0),
           result_tiled.fast_size / 1024.0);
    printf("  Tiling overhead: %.2fx execution time\n",
           result_tiled.time_ms / result_nontiled.time_ms);
    printf("  Accuracy: %s (max err diff = %.2e)\n",
           fabs(result_tiled.max_err - result_nontiled.max_err) < 1e-6
               ? "IDENTICAL" : "within tolerance",
           fabs(result_tiled.max_err - result_nontiled.max_err));
}

/* E2E test - compressed plan */

/**
 * Run a compressed plan and compare its output against the reference.
 * The decompressed weights should produce bit-identical results.
 */
static void test_e2e_compressed(const char *path, const char *ref_path)
{
    printf("  [LZ4] Loading compressed plan: %s\n", path);

    uint32_t buf_len = 0;
    uint8_t *buf = load_file(path, &buf_len);
    if (!buf) {
        tests_run++;
        tests_failed++;
        fprintf(stderr, "  FAIL: could not load compressed plan file\n");
        return;
    }
    printf("  [LZ4] File size: %u bytes (%.1f KB)\n", buf_len, buf_len / 1024.0);

    tigris_plan_t plan;
    tigris_error_t lerr = tigris_plan_load(buf, buf_len, &plan);
    TEST_ASSERT_EQ(lerr, TIGRIS_OK, "compressed plan load");
    if (lerr != TIGRIS_OK) { free(buf); return; }

    printf("  [LZ4] Model: %s\n", tigris_model_name(&plan));
    printf("  [LZ4] Weight blocks: %u, compression: %u\n",
           plan.num_weight_blocks, plan.weight_compression);
    TEST_ASSERT(plan.num_weight_blocks > 0, "has weight blocks");
    TEST_ASSERT(plan.weight_blob == NULL, "no XIP weight blob");

    uint16_t nt = plan.header->num_tensors;
    void **ptrs = (void **)calloc(nt, sizeof(void *));
    TEST_ASSERT(ptrs != NULL, "alloc tensor ptrs (lz4)");
    if (!ptrs) { free(buf); return; }

    uint32_t fast_size = plan.header->peak;
    if (fast_size < total_tensor_bytes(&plan))
        fast_size = total_tensor_bytes(&plan);
    /* Extra space for decompressed weights in fast arena */
    uint32_t max_uncomp = 0;
    for (uint16_t i = 0; i < plan.num_weight_blocks; i++) {
        if (plan.weight_blocks[i].uncompressed_size > max_uncomp)
            max_uncomp = plan.weight_blocks[i].uncompressed_size;
    }
    fast_size += max_uncomp + 16;  /* +16 for alignment */

    uint8_t *fast_buf = (uint8_t *)malloc(fast_size);
    uint32_t slow_size = total_tensor_bytes(&plan) * 2;
    uint8_t *slow_buf = (uint8_t *)malloc(slow_size);

    TEST_ASSERT(fast_buf != NULL, "alloc fast buffer (lz4)");
    TEST_ASSERT(slow_buf != NULL, "alloc slow buffer (lz4)");
    if (!fast_buf || !slow_buf) {
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    printf("  [LZ4] Fast buffer: %u bytes (%.1f KB)\n",
           fast_size, fast_size / 1024.0);

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, ptrs, nt,
                                              fast_buf, fast_size,
                                              slow_buf, slow_size);
    TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "mem init (lz4)");

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        TEST_ASSERT_EQ(merr, TIGRIS_MEM_OK, "alloc model input (lz4)");

        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);
        float *data = (float *)ptrs[tidx];
        for (uint32_t j = 0; j < num_floats; j++)
            data[j] = 1.0f;
    }

    printf("  [LZ4] Running inference...\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = elapsed_ms(&t0, &t1);

    TEST_ASSERT_EQ(eerr, TIGRIS_EXEC_OK, "compressed inference completed");

    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "  [LZ4] Inference failed: %s\n", tigris_exec_error_str(eerr));
        free(ptrs); free(fast_buf); free(slow_buf); free(buf);
        return;
    }

    printf("  [LZ4] Inference complete! (%.1f ms)\n", ms);

    /* Compare against reference */
    uint32_t total_output_floats = 0;
    const float *first_output = NULL;

    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        float *output = (float *)ptrs[tidx];
        uint32_t num_floats = plan.tensors[tidx].size_bytes / sizeof(float);

        TEST_ASSERT(output != NULL, "output ptr not null (lz4)");
        if (!output) continue;

        if (i == 0) first_output = output;
        total_output_floats += num_floats;

        int has_nan = 0, has_inf = 0;
        for (uint32_t j = 0; j < num_floats; j++) {
            if (isnan(output[j])) has_nan = 1;
            if (isinf(output[j])) has_inf = 1;
        }
        TEST_ASSERT(!has_nan, "no NaN in lz4 output");
        TEST_ASSERT(!has_inf, "no Inf in lz4 output");
    }

    if (ref_path && first_output) {
        printf("  [LZ4] ");
        compare_reference(first_output, total_output_floats, ref_path, NULL);
    }

    free(ptrs);
    free(fast_buf);
    free(slow_buf);
    free(buf);
}

/* Main */

int main(int argc, char *argv[])
{
    printf("TiGrIS E2E POSIX Simulation\n\n");

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <plan.tgrs> [reference.bin] [--lz4 <compressed.lz4.tgrs>]\n", argv[0]);
        return 1;
    }

    const char *ref_path = NULL;
    const char *lz4_path = NULL;

    /* Parse args: plan.tgrs [reference.bin] [--lz4 compressed.lz4.tgrs] */
    int i = 2;
    while (i < argc) {
        if (strcmp(argv[i], "--lz4") == 0 && i + 1 < argc) {
            lz4_path = argv[i + 1];
            i += 2;
        } else if (!ref_path) {
            ref_path = argv[i];
            i++;
        } else {
            i++;
        }
    }

    test_e2e(argv[1], ref_path);

    printf("\n--- Tiled E2E ---\n\n");
    test_e2e_tiled(argv[1], ref_path);

    if (ref_path)
        print_comparison();

    if (lz4_path) {
        printf("\n--- Compressed (LZ4) E2E ---\n\n");
        test_e2e_compressed(lz4_path, ref_path);
    }

    printf("\nResults: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}

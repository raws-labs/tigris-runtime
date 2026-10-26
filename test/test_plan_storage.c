/**
 * @file test_plan_storage.c
 * @brief Loader and executor storage follows the loaded plan, not the build
 *        limits. Built once with the default limits and once with every limit
 *        at 65535: each stateful, control-flow and chained golden loads and
 *        runs to TFLite Micro's outputs in a thread with a small stack, and
 *        needs the workspace its own figures give.
 */

#define _POSIX_C_SOURCE 200112L

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "state_golden.h"
#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"
#include "tigris_binary_operands.h"
#include "test_harness.h"

/* Under half the 262 KB that loader and executor frames sized by limits of 65535 would take. */
#define STORAGE_TEST_STACK (128u * 1024u)

/* How deep control flow nests below graph g, by recursion over the graphs it runs. */
static uint16_t graph_height(const tigris_plan_t *plan, uint16_t g)
{
    const tigris_subgraph_t *graph = &plan->subgraphs[g];
    uint16_t height = 0u;
    for (uint16_t s = graph->first_stage; s < graph->first_stage + graph->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        if (stage->ops_count != 1u) continue;
        const uint16_t op = tigris_stage_ops(plan, stage)[0];
        if (plan->ops[op].op_type != TIGRIS_OP_IF && plan->ops[op].op_type != TIGRIS_OP_WHILE)
            continue;
        const uint8_t *runs = tigris_op_attribute_data(plan, op, TIGRIS_OP_ATTR_SUBGRAPHS, NULL);
        for (uint8_t b = 0u; b < 2u; b++) {
            const uint16_t child = (uint16_t)(runs[2u * b] | (runs[2u * b + 1u] << 8));
            const uint16_t below = (uint16_t)(graph_height(plan, child) + 1u);
            if (below > height) height = below;
        }
    }
    return height;
}

/* The workspace for the plan's own figures; *longest receives the longest chain. */
static size_t plan_workspace(const tigris_plan_t *plan, uint16_t depth, uint16_t *longest)
{
    uint16_t inputs = 0u, outputs = 0u, chain = 0u;
    uint32_t chain_spatial = 0u;
    for (uint16_t s = 0; s < plan->header->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        if (stage->inputs_count > inputs) inputs = stage->inputs_count;
        if (stage->outputs_count > outputs) outputs = stage->outputs_count;
        if (stage->chain_len < 2u || stage->chain_id != s) continue;
        if (stage->chain_len > chain) chain = stage->chain_len;
        uint32_t spatial = 0u;
        for (uint16_t c = s; c < s + stage->chain_len; c++) {
            const uint16_t *ops = tigris_stage_ops(plan, &plan->stages[c]);
            for (uint16_t j = 0; j < plan->stages[c].ops_count; j++)
                spatial += tigris_height_spatial_op(plan->ops[ops[j]].op_type) ? 1u : 0u;
        }
        if (spatial > chain_spatial) chain_spatial = spatial;
    }
    *longest = chain;
    return TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(
        plan->header->num_tensors, inputs, outputs, chain, chain_spatial, depth);
}

/* One run on input x into out, with exactly the plan's workspace. */
static tigris_exec_error_t run_once(const tigris_plan_t *plan, void *state, size_t state_size,
                                    const uint8_t *x, uint32_t x_bytes,
                                    uint8_t *out, uint32_t out_bytes)
{
    const uint32_t fast_size = tigris_fast_arena_required(plan);
    uint32_t slow_size = 0u;
    for (uint16_t t = 0; t < plan->header->num_tensors; t++)
        slow_size += 2u * (plan->tensors[t].size_bytes + TIGRIS_TENSOR_ALIGN);
    uint8_t *fast_raw = malloc((size_t)fast_size + TIGRIS_TENSOR_ALIGN);
    uint8_t *slow_raw = malloc((size_t)slow_size + TIGRIS_TENSOR_ALIGN);
    void **pointers = calloc(plan->header->num_tensors, sizeof(*pointers));
    const size_t workspace_size = tigris_executor_workspace_required(plan);
    void *workspace = malloc(workspace_size);
    tigris_exec_error_t result = TIGRIS_EXEC_ERR_MEM;
    if (fast_raw && slow_raw && pointers && workspace) {
        uint8_t *fast = fast_raw + (TIGRIS_TENSOR_ALIGN - (uintptr_t)fast_raw % TIGRIS_TENSOR_ALIGN) %
                                       TIGRIS_TENSOR_ALIGN;
        uint8_t *slow = slow_raw + (TIGRIS_TENSOR_ALIGN - (uintptr_t)slow_raw % TIGRIS_TENSOR_ALIGN) %
                                       TIGRIS_TENSOR_ALIGN;
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
                    tigris_input_write(plan, &mem, input, x + at, bytes) == TIGRIS_OK;
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
                    tigris_output_read(plan, &mem, output, out + at, bytes) != TIGRIS_OK)
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

typedef struct {
    const state_golden_t *golden;
    uint16_t depth;
    uint16_t chain;
} golden_job_t;

/* Load and run one golden; runs on the small-stack thread. */
static void *run_golden(void *arg)
{
    golden_job_t *job = arg;
    const state_golden_t *golden = job->golden;
    uint8_t *buf = aligned_alloc(64u, ((size_t)golden->plan_size + 63u) / 64u * 64u);
    uint8_t *out = malloc(golden->output_bytes);
    void *state = NULL;
    tigris_plan_t plan;
    TEST_ASSERT(buf != NULL && out != NULL, "golden buffers");
    if (!buf || !out) goto done;
    memcpy(buf, golden->plan, golden->plan_size);
    TEST_ASSERT_EQ(tigris_plan_load(buf, golden->plan_size, &plan), TIGRIS_OK, "golden loads");
    if (plan.header == NULL) goto done;

    job->depth = plan.num_subgraphs ? graph_height(&plan, 0u) : 0u;
    TEST_ASSERT_EQ(plan.subgraph_depth, job->depth, "the loader states the plan's nesting depth");
    TEST_ASSERT_EQ(tigris_executor_workspace_required(&plan),
                   plan_workspace(&plan, job->depth, &job->chain),
                   "the workspace follows the plan's own figures");

    const size_t state_size = tigris_state_required(&plan);
    state = malloc(state_size > 0u ? state_size : 1u);
    TEST_ASSERT(state != NULL, "state buffer");
    if (!state) goto done;
    TEST_ASSERT_EQ(tigris_state_init(&plan, state, state_size), TIGRIS_EXEC_OK, "state init");
    for (uint32_t run = 0; run < golden->runs; run++) {
        const uint8_t *x = (const uint8_t *)golden->input + (size_t)run * golden->input_bytes;
        const uint8_t *y = (const uint8_t *)golden->output + (size_t)run * golden->output_bytes;
        TEST_ASSERT_EQ(run_once(&plan, state, state_size, x, golden->input_bytes,
                                out, golden->output_bytes), TIGRIS_EXEC_OK, "golden run");
        TEST_ASSERT(memcmp(out, y, golden->output_bytes) == 0, "golden run matches TFLite Micro");
    }
done:
    free(state);
    free(out);
    free(buf);
    return NULL;
}

static void test_storage_follows_plan(void)
{
    printf("  test_storage_follows_plan...\n");
    uint16_t deepest = 0u, longest = 0u;
    pthread_attr_t attr;
    TEST_ASSERT_EQ(pthread_attr_init(&attr), 0, "thread attributes");
    TEST_ASSERT_EQ(pthread_attr_setstacksize(&attr, STORAGE_TEST_STACK), 0, "small thread stack");
    for (size_t g = 0; g < sizeof(state_goldens) / sizeof(state_goldens[0]); g++) {
        golden_job_t job = {&state_goldens[g], 0u, 0u};
        pthread_t thread;
        int started = pthread_create(&thread, &attr, run_golden, &job);
        TEST_ASSERT_EQ(started, 0, "golden thread starts");
        if (started == 0) pthread_join(thread, NULL);
        if (job.depth > deepest) deepest = job.depth;
        if (job.chain > longest) longest = job.chain;
    }
    pthread_attr_destroy(&attr);
    TEST_ASSERT_EQ(deepest, 2, "a golden nests control flow two deep");
    TEST_ASSERT_EQ(longest, 3, "a golden runs a chain of three stages");
}

int main(void)
{
    printf("test_plan_storage (TIGRIS_MAX_TENSORS=%u)\n", (unsigned)TIGRIS_MAX_TENSORS);
    test_storage_follows_plan();
    return test_report();
}

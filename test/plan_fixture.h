/**
 * @file plan_fixture.h
 * @brief In-memory plans for executor tests: the wiring of a linear operator
 *        chain, and one run of a plan with one model input and one output.
 *
 * A test that uses trace_assert.h includes it first, so that run_fixture's
 * runs are checked against the trace in trace builds.
 */

#ifndef TIGRIS_TEST_PLAN_FIXTURE_H
#define TIGRIS_TEST_PLAN_FIXTURE_H

#include <stdint.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_mem.h"

/* Index pool entries of a chain: per op its input and output, per stage its
 * ops, input and output, then the model input and output. */
#define PLAN_FIXTURE_POOL(num_ops) (5u * (num_ops) + 2u)
#define PLAN_FIXTURE_POOL_ONE_STAGE(num_ops) (3u * (num_ops) + 4u)

#define PLAN_FIXTURE_MAX_TENSORS 8u

/*
 * Zeroes the header, the num_ops + 1 tensors, the ops, the stages, the pool
 * and the plan, then wires op i to read tensor i and write tensor i + 1, with
 * one stage per op, or all ops in one stage when one_stage is set. Tensor 0
 * is the model input and tensor num_ops the model output. Stages are
 * standalone and untiled. Everything else stays 0 for the caller to set:
 * version, tensor sizes, shapes, dtypes, flags and quant indices, op types and
 * weights, tile records and attributes.
 */
static inline void plan_fixture_chain(
    tigris_plan_t *plan, tigris_file_header_t *header, tigris_tensor_t *tensors,
    tigris_op_t *ops, tigris_stage_t *stages, uint16_t *pool,
    const int32_t *shapes, uint16_t num_ops, int one_stage)
{
    const uint16_t num_stages = one_stage ? 1u : num_ops;
    const uint16_t pool_size = (uint16_t)(one_stage
        ? PLAN_FIXTURE_POOL_ONE_STAGE(num_ops) : PLAN_FIXTURE_POOL(num_ops));
    uint16_t next = (uint16_t)(2u * num_ops);

    memset(header, 0, sizeof(*header));
    memset(tensors, 0, sizeof(*tensors) * (num_ops + 1u));
    memset(ops, 0, sizeof(*ops) * num_ops);
    memset(stages, 0, sizeof(*stages) * num_stages);
    memset(pool, 0, sizeof(*pool) * pool_size);
    memset(plan, 0, sizeof(*plan));

    header->num_tensors = (uint16_t)(num_ops + 1u);
    header->num_ops = num_ops;
    header->num_stages = num_stages;
    header->num_model_inputs = 1;
    header->num_model_outputs = 1;

    for (uint16_t i = 0; i < num_ops; i++) {
        ops[i].num_inputs = 1;
        ops[i].num_outputs = 1;
        ops[i].inputs_off = (uint16_t)(2u * i);
        ops[i].outputs_off = (uint16_t)(2u * i + 1u);
        pool[2u * i] = i;
        pool[2u * i + 1u] = (uint16_t)(i + 1u);
    }
    for (uint16_t s = 0; s < num_stages; s++) {
        const uint16_t count = one_stage ? num_ops : 1u;
        stages[s].ops_off = next;
        stages[s].ops_count = count;
        for (uint16_t i = 0; i < count; i++)
            pool[next++] = (uint16_t)(s + i);
        stages[s].inputs_off = next;
        stages[s].inputs_count = 1;
        pool[next++] = s;
        stages[s].outputs_off = next;
        stages[s].outputs_count = 1;
        pool[next++] = (uint16_t)(s + count);
        stages[s].tile_plan_idx = TIGRIS_NO_TILE_PLAN;
        stages[s].chain_id = TIGRIS_NO_CHAIN;
    }
    header->model_io_off = next;
    pool[next] = 0;
    pool[next + 1u] = num_ops;

    plan->header = header;
    plan->tensors = tensors;
    plan->ops = ops;
    plan->stages = stages;
    plan->index_pool = pool;
    plan->shape_pool = shapes;
    plan->model_inputs = &pool[next];
    plan->model_outputs = &pool[next + 1u];
}

/*
 * Runs the plan with the model input copied into slow memory and copies the
 * model output out. The arenas and the workspace belong to the caller.
 * *tiled receives the number of tiled stages.
 */
static inline tigris_exec_error_t run_fixture(
    const tigris_plan_t *plan, tigris_kernel_fn kernel, void *user_ctx,
    uint8_t *fast, uint32_t fast_size, uint8_t *slow, uint32_t slow_size,
    uint8_t *workspace, size_t workspace_size,
    const void *input, void *output, uint16_t *tiled)
{
    const uint16_t in = plan->model_inputs[0];
    const uint16_t out = plan->model_outputs[0];
    tigris_exec_stats_t stats;
    tigris_mem_t mem;
    void *ptrs[PLAN_FIXTURE_MAX_TENSORS];

    memset(&stats, 0, sizeof(stats));
    memset(slow, 0, slow_size);

    if (plan->header->num_tensors > PLAN_FIXTURE_MAX_TENSORS ||
        tigris_mem_init(&mem, ptrs, plan->header->num_tensors, fast, fast_size,
                        slow, slow_size) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    if (tigris_mem_alloc_slow(&mem, in, plan->tensors[in].size_bytes)
            != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    memcpy(mem.tensor_ptrs[in], input, plan->tensors[in].size_bytes);

    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        plan, &mem, kernel, user_ctx, &stats, workspace, workspace_size);
    if (err != TIGRIS_EXEC_OK)
        return err;

    memcpy(output, mem.tensor_ptrs[out], plan->tensors[out].size_bytes);
    *tiled = stats.stages_tiled;
    return TIGRIS_EXEC_OK;
}

#endif /* TIGRIS_TEST_PLAN_FIXTURE_H */

#ifndef TIGRIS_TEST_TRACE_ASSERT_H
#define TIGRIS_TEST_TRACE_ASSERT_H

#ifdef TIGRIS_TRACE
#include "tigris_trace.h"
#include "tigris_executor.h"

static uint32_t trace_paths_seen;

typedef struct {
    const tigris_mem_t *mem;
    const tigris_plan_t *plan;
    uint16_t stages[64];
    unsigned depth;
    uint32_t loads, spills, weights, copies, tiles, events, begins, ends;
    uint32_t fast_peak, slow_peak;
} trace_observer_t;

static void trace_observe(const tigris_trace_event_t *event, void *ctx)
{
    trace_observer_t *seen = ctx;
    seen->events++;
    TEST_ASSERT_EQ(event->kind >= TIGRIS_TRACE_STAGE_BEGIN &&
                   event->kind <= TIGRIS_TRACE_COPY, 1, "trace event kind");
    TEST_ASSERT_EQ(event->fast_used, seen->mem->fast_used, "trace fast occupancy");
    TEST_ASSERT_EQ(event->slow_used, seen->mem->slow_used, "trace slow occupancy");
    if (event->fast_used > seen->fast_peak) seen->fast_peak = event->fast_used;
    if (event->slow_used > seen->slow_peak) seen->slow_peak = event->slow_used;
    if (event->kind == TIGRIS_TRACE_STAGE_BEGIN) {
        TEST_ASSERT_EQ(event->path <= TIGRIS_TRACE_PATH_CONTROL, 1, "trace path");
        trace_paths_seen |= 1u << event->path;
        TEST_ASSERT_EQ(seen->depth < 64u, 1, "trace nesting fits observer");
        if (seen->depth < 64u) seen->stages[seen->depth++] = event->stage;
        seen->begins++;
    }
    if (event->kind == TIGRIS_TRACE_STAGE_END) {
        TEST_ASSERT_EQ(seen->depth > 0u, 1, "stage end follows begin");
        if (seen->depth > 0u) {
            TEST_ASSERT_EQ(event->stage, seen->stages[--seen->depth], "stage end matches begin");
        }
        seen->ends++;
    }
    if (event->kind == TIGRIS_TRACE_OP) {
        TEST_ASSERT_EQ(event->stage < seen->plan->header->num_stages, 1, "OP stage exists");
        if (event->stage < seen->plan->header->num_stages) {
            const tigris_stage_t *stage = &seen->plan->stages[event->stage];
            const uint16_t *ops = tigris_stage_ops(seen->plan, stage);
            int found = 0;
            for (uint16_t i = 0; i < stage->ops_count; i++)
                if (ops[i] == event->op) found = 1;
            TEST_ASSERT_EQ(found, 1, "OP belongs to reported stage");
        }
    }
    if (event->kind == TIGRIS_TRACE_LOAD) seen->loads += event->bytes;
    if (event->kind == TIGRIS_TRACE_SPILL) seen->spills += event->bytes;
    if (event->kind == TIGRIS_TRACE_WEIGHTS) seen->weights += event->bytes;
    if (event->kind == TIGRIS_TRACE_COPY) seen->copies += event->bytes;
    if (event->kind == TIGRIS_TRACE_TILE_BEGIN) {
        seen->tiles++;
        TEST_ASSERT_EQ(seen->depth > 0u, 1, "tile follows stage begin");
        if (seen->depth > 0u)
            TEST_ASSERT_EQ(event->stage, seen->stages[seen->depth - 1u], "tile belongs to active stage");
        TEST_ASSERT_EQ(event->row1 > event->row0, 1, "tile has input rows");
    }
    if (event->kind == TIGRIS_TRACE_ALLOC || event->kind == TIGRIS_TRACE_LOAD ||
        event->kind == TIGRIS_TRACE_SPILL || event->kind == TIGRIS_TRACE_MOVE ||
        event->kind == TIGRIS_TRACE_WEIGHTS) {
        uint32_t capacity = event->pool == TIGRIS_TRACE_POOL_FAST
            ? seen->mem->fast_size : seen->mem->slow_size;
        TEST_ASSERT_EQ(event->offset <= capacity &&
                       event->bytes <= capacity - event->offset, 1,
                       "trace destination is within its arena");
    }
}

static inline tigris_exec_error_t trace_checked_run(int api,
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_kernel_fn kernel,
    void *ctx, tigris_exec_stats_t *stats, void *workspace, size_t workspace_size,
    void *state, size_t state_size)
{
    tigris_exec_stats_t local = {0};
    trace_observer_t seen = {0};
    seen.mem = mem;
    seen.plan = plan;
    if (mem && plan && kernel) {
        seen.fast_peak = mem->fast_used;
        seen.slow_peak = mem->slow_used;
        tigris_mem_set_trace(mem, trace_observe, &seen);
    }
    tigris_exec_stats_t *actual = stats ? stats : &local;
    tigris_exec_error_t result;
    if (api == 0)
        result = tigris_run(plan, mem, kernel, ctx, actual);
    else if (api == 1)
        result = tigris_run_with_workspace(plan, mem, kernel, ctx, actual, workspace);
    else if (api == 2)
        result = tigris_run_with_workspace_buffer(plan, mem, kernel, ctx, actual,
                                                 workspace, workspace_size);
    else
        result = tigris_run_with_state(plan, mem, kernel, ctx, actual,
                                      workspace, workspace_size, state, state_size);
    if (mem && plan && kernel) tigris_mem_set_trace(mem, NULL, NULL);
    if (seen.events) {
        TEST_ASSERT_EQ(seen.loads, actual->loads_bytes, "LOAD sum equals stats");
        TEST_ASSERT_EQ(seen.spills, actual->spills_bytes, "SPILL sum equals stats");
        TEST_ASSERT_EQ(seen.weights, actual->weight_bytes, "WEIGHTS sum equals stats");
        TEST_ASSERT_EQ(seen.copies, actual->copy_bytes, "COPY sum equals stats");
        TEST_ASSERT_EQ(seen.tiles, actual->total_tiles, "tile events equal stats");
        TEST_ASSERT_EQ(seen.fast_peak, mem->fast_peak, "events capture fast peak");
        TEST_ASSERT_EQ(seen.slow_peak, actual->slow_peak, "events capture slow peak");
        TEST_ASSERT_EQ(mem->slow_peak, actual->slow_peak, "slow allocator peak equals stats");
        if (result == TIGRIS_EXEC_OK) {
            TEST_ASSERT_EQ(seen.begins, seen.ends, "completed stage events balance");
            uint32_t state_bytes = 0u;
            for (uint16_t i = 0; i < plan->num_state; i++) {
                state_bytes += plan->state_entries[i].bytes;
                if (plan->state_entries[i].output != UINT16_MAX)
                    state_bytes += plan->state_entries[i].bytes;
            }
            TEST_ASSERT_EQ(seen.copies >= state_bytes, 1, "state transfers counted");
            if (plan->num_subgraphs == 0u) {
                uint32_t weights = 0u;
                for (uint16_t i = 0; i < plan->num_weight_blocks; i++)
                    if (plan->weight_blocks[i].compressed_size > 0u)
                        weights += plan->weight_blocks[i].uncompressed_size;
                TEST_ASSERT_EQ(seen.weights, weights, "all weight blocks counted");
            }
        }
    }
    return result;
}

#define tigris_run(p, m, k, c, s) \
    trace_checked_run(0, (p), (m), (k), (c), (s), NULL, 0, NULL, 0)
#define tigris_run_with_workspace(p, m, k, c, s, w) \
    trace_checked_run(1, (p), (m), (k), (c), (s), (w), 0, NULL, 0)
#define tigris_run_with_workspace_buffer(p, m, k, c, s, w, n) \
    trace_checked_run(2, (p), (m), (k), (c), (s), (w), (n), NULL, 0)
#define tigris_run_with_state(p, m, k, c, s, w, n, state, size) \
    trace_checked_run(3, (p), (m), (k), (c), (s), (w), (n), (state), (size))
#endif
#endif

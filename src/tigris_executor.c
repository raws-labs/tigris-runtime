/**
 * @file tigris_executor.c
 * @brief Stage loop: load -> execute -> spill -> reset.
 */

#include "tigris_executor.h"
#include "tigris_lz4.h"

#include <string.h>

/* Platform hooks - override in platform header or compiler flags */
#ifndef TIGRIS_PLATFORM_FEED_WDT
#  ifdef ESP_PLATFORM
#    include "freertos/FreeRTOS.h"
#    include "freertos/task.h"
#    if CONFIG_ESP_TASK_WDT_EN
#      include "esp_task_wdt.h"
#      define TIGRIS_PLATFORM_FEED_WDT() do { esp_task_wdt_reset(); vTaskDelay(1); } while(0)
#    else
#      define TIGRIS_PLATFORM_FEED_WDT() vTaskDelay(1)
#    endif
#  else
#    define TIGRIS_PLATFORM_FEED_WDT() ((void)0)
#  endif
#endif

#ifndef TIGRIS_PLATFORM_DBG
#  ifdef CONFIG_IDF_TARGET_ESP32S3
#    define TIGRIS_PLATFORM_DBG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#  else
#    define TIGRIS_PLATFORM_DBG(fmt, ...) ((void)0)
#  endif
#endif

#define TILE_ALIGN_UP(x) (((x) + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u))

#ifdef TIGRIS_COUNT_KERNEL_ROWS
/* Test-only: geometry of the most recently executed chain, so a test can
 * assert tile-0 / interior / partial-last coverage. Never in shipping builds. */
int32_t g_tigris_chain_tile_h = 0;
int32_t g_tigris_chain_num_tiles = 0;
/* Test-only: number of (interior tile, stage) pairs where a stage that could
 * otherwise roll was forced to full-compute by a per-stage boundary clamp
 * (sp_tile_pt/pb != 0) while the tile is neither tile 0 nor the last tile. A
 * nonzero value proves the interior-clamp fallback path was exercised. */
int32_t g_tigris_interior_clamps = 0;
#endif

typedef struct {
    int32_t full_in_h;
    int32_t full_in_w;
    int32_t full_out_h;
    int32_t full_out_w;
    int32_t stride_h;
    int32_t eff_kh;
    int32_t orig_pad_top;
    int32_t sp_count;
} chain_stage_info_t;

typedef struct {
    void **input_ptrs;
    void **output_ptrs;
    const tigris_stage_t **stages;
    uint8_t **weight_bases;
    chain_stage_info_t *info;
    int32_t *s_out;
    int32_t *s_in;
    int32_t *in_starts;
    int32_t *in_ends;
    int32_t *out_starts;
    int32_t *out_ends;
    int32_t *sp_op_indices;
    int32_t *sp_strides;
    int32_t *sp_eff_khs;
    int32_t *sp_pad_tops;
    int32_t *sp_full_in_hs;
    int32_t *sp_full_in_ws;
    int32_t *sp_full_out_ws;
    int32_t *sp_tile_in_h;
    int32_t *sp_tile_out_h;
    int32_t *sp_tile_pt;
    int32_t *sp_tile_pb;
    int32_t *sp_tile_out_gs;   /* this tile: spatial-op output global row start */
    int32_t *sp_tile_out_ge;   /* this tile: spatial-op output global row end */
    int32_t *roll_prev_gs;     /* per op-output tensor: previous tile out start */
    int32_t *roll_prev_ge;     /* per op-output tensor: previous tile out end */
    uint16_t *last_use_op;
    uint16_t *last_consumer;
    uint16_t tensor_capacity;
    uint16_t input_capacity;
    uint16_t output_capacity;
    uint16_t chain_capacity;
    uint16_t spatial_capacity;
} executor_workspace_impl_t;

typedef struct {
    uint16_t tensors;
    uint16_t inputs;
    uint16_t outputs;
    uint16_t chain_stages;
    uint16_t spatial_ops;
} executor_workspace_limits_t;

typedef char tigris_executor_compat_workspace_too_small[
    TIGRIS_EXECUTOR_WORKSPACE_BYTES >=
        TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(
            TIGRIS_MAX_TENSORS, TIGRIS_MAX_STAGE_INPUTS,
            TIGRIS_MAX_STAGE_OUTPUTS, TIGRIS_MAX_CHAIN_STAGES,
            TIGRIS_MAX_SPATIAL_OPS_PER_STAGE)
        ? 1 : -1];

static uintptr_t align_address(uintptr_t address, size_t alignment)
{
    return (address + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
}

static int is_height_spatial_op(uint8_t type)
{
    return type == TIGRIS_OP_CONV ||
           type == TIGRIS_OP_DEPTHWISE ||
           type == TIGRIS_OP_MAX_POOL ||
           type == TIGRIS_OP_AVG_POOL;
}

static int is_height_tiling_op(uint8_t type)
{
    return is_height_spatial_op(type) ||
           type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH ||
           type == TIGRIS_OP_ADD ||
           type == TIGRIS_OP_MUL ||
           type == TIGRIS_OP_CONCAT;
}

static int is_axis1_unary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH;
}

static int is_axis1_binary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_ADD || type == TIGRIS_OP_MUL;
}

static int workspace_limits(
    const tigris_plan_t *plan, executor_workspace_limits_t *limits)
{
    if (!plan || !limits || !plan->header || !plan->stages ||
        !plan->ops || !plan->index_pool)
        return 0;

    memset(limits, 0, sizeof(*limits));
    limits->tensors = plan->header->num_tensors;

    for (uint16_t s = 0; s < plan->header->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];
        if (stage->inputs_count > limits->inputs)
            limits->inputs = stage->inputs_count;
        if (stage->outputs_count > limits->outputs)
            limits->outputs = stage->outputs_count;

        if (stage->chain_len >= 2 && stage->chain_id == s) {
            uint16_t chain_end;
            if (stage->chain_len > UINT16_MAX - s)
                return 0;
            chain_end = (uint16_t)(s + stage->chain_len);
            if (chain_end > plan->header->num_stages)
                return 0;
            if (stage->chain_len > limits->chain_stages)
                limits->chain_stages = stage->chain_len;

            for (uint16_t c = s; c < chain_end; c++) {
                const tigris_stage_t *chain_stage = &plan->stages[c];
                const uint16_t *ops = tigris_stage_ops(plan, chain_stage);
                uint16_t spatial = 0;
                for (uint16_t j = 0; j < chain_stage->ops_count; j++) {
                    if (ops[j] >= plan->header->num_ops)
                        return 0;
                    uint8_t type = plan->ops[ops[j]].op_type;
                    if (is_height_spatial_op(type))
                        spatial++;
                }
                if (spatial > limits->spatial_ops)
                    limits->spatial_ops = spatial;
            }
        }
    }
    return 1;
}

static size_t workspace_bytes_for_limits(
    const executor_workspace_limits_t *limits)
{
    size_t total = TIGRIS_EXECUTOR_POINTER_ALIGNMENT - 1u;
    size_t count =
        (size_t)limits->inputs + limits->outputs +
        2u * (size_t)limits->chain_stages;
    if (count > (SIZE_MAX - total) / sizeof(void *))
        return 0;
    total += count * sizeof(void *);

    if (total > SIZE_MAX - (TIGRIS_EXECUTOR_INT32_ALIGNMENT - 1u))
        return 0;
    total += TIGRIS_EXECUTOR_INT32_ALIGNMENT - 1u;
    count = 14u + 13u * (size_t)limits->spatial_ops;
    if (limits->chain_stages != 0 &&
        count > SIZE_MAX / limits->chain_stages)
        return 0;
    count *= limits->chain_stages;
    /* Plus two int32 line-buffer arrays indexed by op-output tensor. */
    if (2u * (size_t)limits->tensors > SIZE_MAX - count)
        return 0;
    count += 2u * (size_t)limits->tensors;
    if (count > (SIZE_MAX - total) / sizeof(int32_t))
        return 0;
    total += count * sizeof(int32_t);

    if (total > SIZE_MAX - (TIGRIS_EXECUTOR_UINT16_ALIGNMENT - 1u))
        return 0;
    total += TIGRIS_EXECUTOR_UINT16_ALIGNMENT - 1u;
    count = 2u * (size_t)limits->tensors;
    if (count > (SIZE_MAX - total) / sizeof(uint16_t))
        return 0;
    return total + count * sizeof(uint16_t);
}

static int workspace_layout(
    void *storage, size_t storage_size,
    const executor_workspace_limits_t *limits,
    executor_workspace_impl_t *workspace)
{
    size_t required = workspace_bytes_for_limits(limits);
    if (!storage || !workspace || required == 0 ||
        storage_size < required ||
        storage_size > UINTPTR_MAX - (uintptr_t)storage)
        return 0;

    uintptr_t cursor = align_address(
        (uintptr_t)storage, TIGRIS_EXECUTOR_POINTER_ALIGNMENT);
    workspace->input_ptrs = (void **)cursor;
    cursor += (size_t)limits->inputs * sizeof(void *);
    workspace->output_ptrs = (void **)cursor;
    cursor += (size_t)limits->outputs * sizeof(void *);
    workspace->stages = (const tigris_stage_t **)cursor;
    cursor += (size_t)limits->chain_stages * sizeof(void *);
    workspace->weight_bases = (uint8_t **)cursor;
    cursor += (size_t)limits->chain_stages * sizeof(void *);

    cursor = align_address(cursor, TIGRIS_EXECUTOR_INT32_ALIGNMENT);
    workspace->info = (chain_stage_info_t *)cursor;
    cursor += (size_t)limits->chain_stages * sizeof(chain_stage_info_t);
#define TIGRIS_LAYOUT_CHAIN_FIELD(field)                                      \
    do {                                                                       \
        workspace->field = (int32_t *)cursor;                                 \
        cursor += (size_t)limits->chain_stages * sizeof(int32_t);              \
    } while (0)
    TIGRIS_LAYOUT_CHAIN_FIELD(s_out);
    TIGRIS_LAYOUT_CHAIN_FIELD(s_in);
    TIGRIS_LAYOUT_CHAIN_FIELD(in_starts);
    TIGRIS_LAYOUT_CHAIN_FIELD(in_ends);
    TIGRIS_LAYOUT_CHAIN_FIELD(out_starts);
    TIGRIS_LAYOUT_CHAIN_FIELD(out_ends);
#undef TIGRIS_LAYOUT_CHAIN_FIELD

#define TIGRIS_LAYOUT_SPATIAL_FIELD(field)                                     \
    do {                                                                       \
        workspace->field = (int32_t *)cursor;                                 \
        cursor += (size_t)limits->chain_stages *                               \
                  (size_t)limits->spatial_ops * sizeof(int32_t);               \
    } while (0)
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_op_indices);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_strides);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_eff_khs);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_pad_tops);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_full_in_hs);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_full_in_ws);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_full_out_ws);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_in_h);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_out_h);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_pt);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_pb);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_out_gs);
    TIGRIS_LAYOUT_SPATIAL_FIELD(sp_tile_out_ge);
#undef TIGRIS_LAYOUT_SPATIAL_FIELD

    /* Line-buffer roll bookkeeping, indexed by op-output tensor. */
#define TIGRIS_LAYOUT_TENSOR_I32_FIELD(field)                                  \
    do {                                                                       \
        workspace->field = (int32_t *)cursor;                                 \
        cursor += (size_t)limits->tensors * sizeof(int32_t);                   \
    } while (0)
    TIGRIS_LAYOUT_TENSOR_I32_FIELD(roll_prev_gs);
    TIGRIS_LAYOUT_TENSOR_I32_FIELD(roll_prev_ge);
#undef TIGRIS_LAYOUT_TENSOR_I32_FIELD

    cursor = align_address(cursor, TIGRIS_EXECUTOR_UINT16_ALIGNMENT);
    workspace->last_use_op = (uint16_t *)cursor;
    cursor += (size_t)limits->tensors * sizeof(uint16_t);
    workspace->last_consumer = (uint16_t *)cursor;
    cursor += (size_t)limits->tensors * sizeof(uint16_t);

    workspace->tensor_capacity = limits->tensors;
    workspace->input_capacity = limits->inputs;
    workspace->output_capacity = limits->outputs;
    workspace->chain_capacity = limits->chain_stages;
    workspace->spatial_capacity = limits->spatial_ops;

    return cursor <= (uintptr_t)storage + storage_size;
}

static size_t spatial_index(
    const executor_workspace_impl_t *workspace, uint16_t chain, uint16_t op)
{
    return (size_t)chain * workspace->spatial_capacity + op;
}

/* Memory compaction */

/**
 * Compact the fast pool: move alive tensors to the front, reclaim
 * space from dead ones.  Similar to compact_slow but for fast arena.
 */
static uint32_t compact_pool(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint8_t *base, uint32_t first, uint32_t size)
{
    uintptr_t cursor = (uintptr_t)(base + first);
    uintptr_t end = (uintptr_t)(base + size);
    uint32_t wp = first;

    /* Select the next live allocation by address on each pass.  This is O(n²)
     * in the configured tensor limit, but compaction is an exceptional OOM
     * recovery path and no longer needs a max-tensor index array on the stack. */
    while (cursor < end) {
        uintptr_t next = end;
        uint16_t tidx = UINT16_MAX;
        for (uint16_t i = 0; i < mem->num_tensors; i++) {
            uintptr_t p = (uintptr_t)mem->tensor_ptrs[i];
            if (p >= cursor && p < next && p < end) {
                next = p;
                tidx = i;
            }
        }
        if (tidx == UINT16_MAX)
            break;

        uint32_t sz = TILE_ALIGN_UP(plan->tensors[tidx].size_bytes);
        uint8_t *src = (uint8_t *)next;
        uint8_t *dst = base + wp;
        if (dst != src)
            memmove(dst, src, sz);

        /* Preserve in-place aliases that point at the same allocation. */
        for (uint16_t i = 0; i < mem->num_tensors; i++) {
            if (mem->tensor_ptrs[i] == src)
                mem->tensor_ptrs[i] = dst;
        }
        wp += sz;
        cursor = next + sz;
    }
    return wp;
}

static void compact_fast(tigris_mem_t *mem, const tigris_plan_t *plan)
{
    mem->fast_used = compact_pool(
        mem, plan, mem->fast_base, mem->fast_reserved, mem->fast_size);
}

/**
 * Compact the slow pool: move alive tensors to the front, reclaim
 * space from dead ones.  Tensors must be processed in address order
 * so memmove never reads from already-overwritten memory.
 */
static void compact_slow(tigris_mem_t *mem, const tigris_plan_t *plan)
{
    mem->slow_used = compact_pool(mem, plan, mem->slow_base, 0, mem->slow_size);
}

/* Helpers */

/**
 * Find the first axis-1 spatial op (Conv/DWConv/Pool/Conv1D) in a stage.
 * Returns its index in plan->ops, or -1 if none.
 */
static int find_spatial_op(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t t = plan->ops[sops[j]].op_type;
        if (is_height_spatial_op(t) || t == TIGRIS_OP_CONV1D)
            return (int)sops[j];
    }
    return -1;
}

/**
 * Return whether every op in a standalone stage implements the selected
 * axis-1 stripe contract used by exec_stage_tiled.
 *
 * This is intentionally an audited allow-list.  An operator being executable
 * as a full tensor does not imply that its kernel consumes the tile geometry
 * correctly.  In particular, global reductions and shape-changing Resize
 * must stay on the normal path until they have dedicated range propagation.
 */
static int stage_supports_axis1_tiling(
    const tigris_plan_t *plan, const tigris_stage_t *stage, uint8_t rank)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    int spatial_count = 0;

    if (rank == 3) {
        int binary_count = 0;
        if (stage->chain_len != 0)
            return 0;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint8_t type = plan->ops[sops[j]].op_type;
            if (type == TIGRIS_OP_CONV1D) {
                spatial_count++;
            } else if (is_axis1_binary_pointwise_op(type)) {
                binary_count++;
            } else if (!is_axis1_unary_pointwise_op(type)) {
                return 0;
            }
        }
        return spatial_count <= 1 &&
               !(spatial_count == 1 && binary_count > 0);
    }
    if (rank != 4)
        return 0;

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t type = plan->ops[sops[j]].op_type;
        if (!is_height_tiling_op(type))
            return 0;
        if (is_height_spatial_op(type))
            spatial_count++;
    }
    return spatial_count <= 1;
}

static uint8_t stage_tile_axis(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (!plan || !plan->header || !stage || !plan->tile_plans ||
        stage->tile_plan_idx == TIGRIS_NO_TILE_PLAN ||
        stage->tile_plan_idx >= plan->header->num_tile_plans)
        return TIGRIS_TILE_AXIS_NONE;

    const tigris_tile_plan_t *tile =
        &plan->tile_plans[stage->tile_plan_idx];
    if (!tile->tileable)
        return TIGRIS_TILE_AXIS_NONE;
    return plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS
        ? tile->axis
        : TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
}

/* Non-tiled stage execution */

static tigris_exec_error_t exec_stage_normal(
    const tigris_plan_t *plan,
    const tigris_stage_t *stage,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    executor_workspace_impl_t *workspace)
{
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    void **saved_slow = workspace->input_ptrs;

    /* Pre-compute last_use_op[t] = index within sops[] of last op that reads t.
     * After that op completes, t's fast memory can be reclaimed (if not a stage output). */
    uint16_t num_t = plan->header->num_tensors;
    uint16_t *last_use_op = workspace->last_use_op;
    for (uint16_t i = 0; i < num_t; i++)
        last_use_op[i] = UINT16_MAX;  /* not used in this stage */

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = sops[j];
        const tigris_op_t *op = &plan->ops[op_idx];
        const uint16_t *ins = tigris_op_inputs(plan, op);
        for (uint8_t k = 0; k < op->num_inputs; k++) {
            uint16_t tidx = ins[k];
            if (tidx < num_t)
                last_use_op[tidx] = j;  /* update to latest op that reads this tensor */
        }
    }

    /* Mark stage outputs as never-free (they survive the stage) */
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        if (tidx < num_t)
            last_use_op[tidx] = UINT16_MAX;
    }

    /* Load stage inputs */
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint16_t tidx = sin[i];
        saved_slow[i] = mem->tensor_ptrs[tidx];

        tigris_mem_error_t merr = tigris_mem_load(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        if (stats) stats->loads_bytes += plan->tensors[tidx].size_bytes;
    }

    /* Execute ops with intra-stage reclamation */
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = sops[j];
        const tigris_op_t *op = &plan->ops[op_idx];

        /* Allocate outputs (with compaction retry on OOM, slow fallback) */
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint16_t tidx = outs[k];
            if (!mem->tensor_ptrs[tidx]) {
                tigris_mem_error_t merr = tigris_mem_alloc_fast(
                    mem, tidx, plan->tensors[tidx].size_bytes);
                if (merr != TIGRIS_MEM_OK) {
                    /* Try compacting fast memory and retry */
                    compact_fast(mem, plan);
                    if (stats) stats->compactions++;
                    merr = tigris_mem_alloc_fast(
                        mem, tidx, plan->tensors[tidx].size_bytes);
                    if (merr != TIGRIS_MEM_OK) {
                        /* Overflow to slow (PSRAM) as last resort */
                        merr = tigris_mem_alloc_slow(
                            mem, tidx, plan->tensors[tidx].size_bytes);
                        if (merr != TIGRIS_MEM_OK)
                            return TIGRIS_EXEC_ERR_MEM;
                        if (stats) {
                            stats->slow_overflow_count++;
                            stats->slow_overflow_bytes +=
                                plan->tensors[tidx].size_bytes;
                        }
                    }
                }
            }
        }

        int kret = kernel(plan, op, op_idx, mem, user_ctx);
        if (kret != 0)
            return TIGRIS_EXEC_ERR_KERNEL;

        /* Free tensors whose last use was this op */
        const uint16_t *ins = tigris_op_inputs(plan, op);
        for (uint8_t k = 0; k < op->num_inputs; k++) {
            uint16_t tidx = ins[k];
            if (tidx < num_t && last_use_op[tidx] == j && mem->tensor_ptrs[tidx])
                mem->tensor_ptrs[tidx] = NULL;
        }
    }

    /* Spill stage outputs to slow */
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        tigris_mem_error_t merr = tigris_mem_spill(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        if (stats) stats->spills_bytes += plan->tensors[tidx].size_bytes;
    }

    tigris_mem_reset_fast(mem);

    /* Restore stage input pointers to slow */
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint16_t tidx = sin[i];
        mem->tensor_ptrs[tidx] = saved_slow[i];
    }

    return TIGRIS_EXEC_OK;
}

/* Tiled stage execution */

static int axis1_allocation_bytes(
    const tigris_plan_t *plan, uint16_t tensor_idx,
    int32_t extent, uint32_t *bytes)
{
    const tigris_tensor_t *tensor = &plan->tensors[tensor_idx];
    const int32_t *shape = tigris_tensor_shape(plan, tensor);
    if (!bytes || (tensor->ndim != 3 && tensor->ndim != 4) ||
        shape[1] <= 0 || extent <= 0)
        return 0;
    if (extent > shape[1])
        extent = shape[1];

    uint64_t raw =
        ((uint64_t)tensor->size_bytes / (uint32_t)shape[1]) *
        (uint32_t)extent;
    uint64_t aligned =
        (raw + (TIGRIS_TENSOR_ALIGN - 1u)) &
        ~(uint64_t)(TIGRIS_TENSOR_ALIGN - 1u);
    if (aligned > UINT32_MAX)
        return 0;
    *bytes = (uint32_t)aligned;
    return 1;
}

static int stage_axis1_fast_bytes(
    const tigris_plan_t *plan, const tigris_stage_t *stage,
    int spatial_op_idx, int32_t output_extent,
    int32_t stride, int32_t effective_kernel, uint32_t *required)
{
    const uint16_t *stage_inputs = tigris_stage_inputs(plan, stage);
    const uint16_t *stage_ops = tigris_stage_ops(plan, stage);
    int64_t input_extent_wide = spatial_op_idx >= 0
        ? ((int64_t)output_extent - 1) * stride + effective_kernel
        : output_extent;
    if (input_extent_wide <= 0 || input_extent_wide > INT32_MAX)
        return 0;
    int32_t input_extent = (int32_t)input_extent_wide;
    int32_t current_extent = input_extent;
    uint64_t total = 0;

    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint32_t bytes;
        if (!axis1_allocation_bytes(
                plan, stage_inputs[i], input_extent, &bytes))
            return 0;
        total += bytes;
    }

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = stage_ops[j];
        const tigris_op_t *op = &plan->ops[op_idx];
        int is_spatial = (int)op_idx == spatial_op_idx;
        int32_t allocation_extent =
            is_spatial ? output_extent : current_extent;
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint32_t bytes;
            if (!axis1_allocation_bytes(
                    plan, outputs[k], allocation_extent, &bytes))
                return 0;
            total += bytes;
        }
        if (is_spatial)
            current_extent = output_extent;
    }

    if (total > UINT32_MAX)
        return 0;
    *required = (uint32_t)total;
    return 1;
}

static tigris_exec_error_t exec_stage_tiled(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    const uint16_t           *last_consumer,
    uint16_t                  current_stage,
    executor_workspace_impl_t *workspace)
{
    /* 1. Pre-allocate full output tensors in slow (or reuse input for in-place) */
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    void **out_slow_bases = workspace->output_ptrs;
    int in_place = 0;  /* set if output reuses input's slow memory */

    /* Spatial op + image dims, computed up front: the in-place halo guard below
     * and the tile geometry both need them. */
    int sp_op_idx = find_spatial_op(plan, stage);
    const tigris_spatial_attrs_t *sp_attrs =
        (sp_op_idx >= 0) ? &plan->ops[sp_op_idx].spatial : NULL;
    uint8_t rank = plan->tensors[sin[0]].ndim;
    const int32_t *in_shape = tigris_tensor_shape(plan, &plan->tensors[sin[0]]);
    int32_t full_in_h = in_shape[1];
    int32_t full_in_w = rank == 4 ? in_shape[2] : 1;
    const int32_t *out_shape = tigris_tensor_shape(plan, &plan->tensors[sout[0]]);
    int32_t full_out_h = out_shape[1];
    int32_t full_out_w = rank == 4 ? out_shape[2] : 1;
    int32_t orig_pad_top = sp_attrs ? sp_attrs->pad_top : 0;
    int32_t stride = sp_attrs ? sp_attrs->stride_h : 1;
    int32_t dh = (sp_attrs && sp_attrs->dilation_h) ? sp_attrs->dilation_h : 1;
    int32_t eff_kh = sp_attrs ? ((sp_attrs->kernel_h - 1) * dh + 1) : 1;

    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        tigris_mem_error_t merr = tigris_mem_alloc_slow(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {
            /* Allocation failed - try in-place if:
             *   - Single input/output
             *   - Input dies after this stage (last_consumer == current_stage)
             *   - Input and output have same size
             *   - No cross-tile halo: with eff_kh > stride a kernel reads input
             *     rows beyond this tile, so writing the output into the shared
             *     buffer would clobber rows the next tile still needs.
             */
            if (stage->inputs_count == 1 && stage->outputs_count == 1 &&
                last_consumer && last_consumer[sin[0]] == current_stage &&
                plan->tensors[sin[0]].size_bytes == plan->tensors[tidx].size_bytes &&
                eff_kh <= stride)
            {
                /* Reuse input's slow memory for output */
                out_slow_bases[i] = mem->tensor_ptrs[sin[0]];
                mem->tensor_ptrs[tidx] = out_slow_bases[i];
                in_place = 1;
            } else {
                return TIGRIS_EXEC_ERR_MEM;
            }
        } else {
            out_slow_bases[i] = mem->tensor_ptrs[tidx];
        }
    }

    /* 2. Save slow ptrs for stage inputs */
    void **in_slow_bases = workspace->input_ptrs;
    for (uint16_t i = 0; i < stage->inputs_count; i++)
        in_slow_bases[i] = mem->tensor_ptrs[sin[i]];

    /* (Spatial op + image dims computed up front, above.) */

    const uint16_t *sops = tigris_stage_ops(plan, stage);
    uint32_t available = mem->fast_size - mem->fast_reserved;
    int32_t low = 1;
    int32_t high = full_out_h;
    int32_t out_tile_h = 0;
    while (low <= high) {
        int32_t candidate = low + (high - low) / 2;
        uint32_t required;
        if (!stage_axis1_fast_bytes(
                plan, stage, sp_op_idx, candidate,
                stride, eff_kh, &required))
            return TIGRIS_EXEC_ERR_TILE;
        if (required <= available) {
            out_tile_h = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    if (out_tile_h == 0)
        return TIGRIS_EXEC_ERR_TILE;

    int32_t num_tiles = 1 + (full_out_h - 1) / out_tile_h;
    int32_t out_row_cursor = 0;

    /* 3. Tile loop - iterate over output row ranges */
    for (int32_t tile = 0; tile < num_tiles; tile++) {
        /* a. Output bounds for this tile */
        int32_t out_start = tile * out_tile_h;
        int32_t out_end = full_out_h - out_start < out_tile_h
            ? full_out_h
            : out_start + out_tile_h;
        int32_t tile_out_h = out_end - out_start;

        /* b. Back-compute required input rows */
        int32_t in_start, in_end;
        int32_t eff_pad_top = 0;
        int32_t eff_pad_bottom = 0;

        if (sp_attrs) {
            /* First input row needed: out_start * stride - pad_top */
            in_start = out_start * stride - orig_pad_top;
            /* Last input row needed: (out_end - 1) * stride - pad_top + eff_kh - 1 */
            in_end = (out_end - 1) * stride - orig_pad_top + eff_kh;

            /* Compute effective padding from clamping */
            if (in_start < 0) {
                eff_pad_top = (int32_t)(-in_start);
                in_start = 0;
            }
            if (in_end > full_in_h) {
                eff_pad_bottom = in_end - full_in_h;
                in_end = full_in_h;
            }
        } else {
            /* Pointwise-only: 1:1 mapping */
            in_start = out_start;
            in_end   = out_end;
        }

        int32_t tile_in_h = in_end - in_start;

        /* c. Activate tiling. The per-op tile context (in/out h+w, pad) is set
         * inside the execute loop below: ops before the spatial op run at the
         * input tile dims, the spatial op reduces them, ops after run at the
         * output dims. */
        mem->tile.active = 1;

        /* d. LOAD: load tile for each stage input */
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            uint16_t tidx = sin[i];
            mem->tensor_ptrs[tidx] = in_slow_bases[i]; /* restore before load */
            tigris_mem_error_t merr = tigris_mem_load_tile(
                mem, plan, tidx, in_start, in_end);
            if (merr != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
        }

        /* e. EXECUTE: track the data height/width flowing through the stage. It
         * starts at the input tile size; the (single) spatial op reduces it to
         * the output tile size; pointwise ops preserve it. Mirrors
         * exec_chain_tiled so a pointwise op placed BEFORE the spatial op (e.g. a
         * residual Add feeding a strided conv) is sized at the input height, not
         * under-allocated/under-computed at the output height (which made the
         * following spatial op read past the buffer). */
        int32_t cur_h = tile_in_h;
        int32_t cur_w = full_in_w;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx = sops[j];
            const tigris_op_t *op = &plan->ops[op_idx];
            int is_spatial = ((int)op_idx == sp_op_idx);

            if (is_spatial) {
                mem->tile.in_h       = cur_h;          /* = tile_in_h */
                mem->tile.out_h      = tile_out_h;
                mem->tile.in_w       = full_in_w;
                mem->tile.out_w      = full_out_w;
                mem->tile.pad_top    = eff_pad_top;
                mem->tile.pad_bottom = eff_pad_bottom;
            } else {
                /* Pointwise op: height/width preserved, no padding. */
                mem->tile.in_h       = cur_h;
                mem->tile.out_h      = cur_h;
                mem->tile.in_w       = cur_w;
                mem->tile.out_w      = cur_w;
                mem->tile.pad_top    = 0;
                mem->tile.pad_bottom = 0;
            }

            int32_t alloc_h = is_spatial ? tile_out_h : cur_h;

            /* Alloc tile-sized outputs in fast (NHWC, alloc_h rows) */
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                uint16_t tidx = outs[k];
                const tigris_tensor_t *t = &plan->tensors[tidx];
                const int32_t *shape = tigris_tensor_shape(plan, t);
                uint32_t width = t->ndim == 4 ? (uint32_t)shape[2] : 1u;
                uint32_t channels = (uint32_t)shape[t->ndim - 1];
                uint32_t elem_size = t->size_bytes /
                    ((uint32_t)shape[0] * (uint32_t)shape[1] *
                     width * channels);
                uint32_t tile_bytes = (uint32_t)shape[0] * (uint32_t)alloc_h *
                    width * channels * elem_size;
                tigris_mem_error_t merr = tigris_mem_alloc_fast(
                    mem, tidx, tile_bytes);
                if (merr != TIGRIS_MEM_OK)
                    return TIGRIS_EXEC_ERR_MEM;
            }

            int kret = kernel(plan, op, op_idx, mem, user_ctx);
            if (kret != 0)
                return TIGRIS_EXEC_ERR_KERNEL;

            if (is_spatial) {
                cur_h = tile_out_h;
                cur_w = full_out_w;
            }
        }

        /* f. SPILL: spill tile for each stage output */
        for (uint16_t i = 0; i < stage->outputs_count; i++) {
            uint16_t tidx = sout[i];
            tigris_mem_error_t merr = tigris_mem_spill_tile(
                mem, plan, tidx, out_slow_bases[i],
                out_row_cursor, out_row_cursor + tile_out_h);
            if (merr != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
        }

        out_row_cursor += tile_out_h;

        /* g. Reset fast arena for next tile */
        tigris_mem_reset_fast(mem);

        TIGRIS_PLATFORM_FEED_WDT();
    }

    /* 4. Verify all output rows covered */
    if (out_row_cursor != full_out_h) {
        return TIGRIS_EXEC_ERR_TILE;
    }

    /* 5. Clear tile context, restore input ptrs to slow (unless in-place),
     *    and NULL intermediate op outputs so compact_fast won't find stale
     *    fast-arena pointers with full-size metadata. */
    memset(&mem->tile, 0, sizeof(mem->tile));
    if (!in_place) {
        for (uint16_t i = 0; i < stage->inputs_count; i++)
            mem->tensor_ptrs[sin[i]] = in_slow_bases[i];
    } else {
        /* In-place: input was overwritten, clear its pointer */
        for (uint16_t i = 0; i < stage->inputs_count; i++)
            mem->tensor_ptrs[sin[i]] = NULL;
    }

    /* NULL intermediate op outputs (stage outputs already point to slow
     * via spill_tile, so skip those) */
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint16_t tidx = outs[k];
            /* Skip stage outputs - already spilled to slow */
            int is_stage_out = 0;
            for (uint16_t s = 0; s < stage->outputs_count; s++) {
                if (tidx == sout[s]) { is_stage_out = 1; break; }
            }
            if (!is_stage_out)
                mem->tensor_ptrs[tidx] = NULL;
        }
    }

    return TIGRIS_EXEC_OK;
}

/* Weight block helpers */

static int align_tensor_size(uint32_t value, uint32_t *aligned)
{
    const uint32_t mask = TIGRIS_TENSOR_ALIGN - 1u;
    if (!aligned || value > UINT32_MAX - mask)
        return 0;
    *aligned = (value + mask) & ~mask;
    return 1;
}

/** Find the weight block for a given stage, or NULL if none. */
static const tigris_weight_block_t *find_weight_block(
    const tigris_plan_t *plan, uint16_t stage_idx)
{
    for (uint16_t i = 0; i < plan->num_weight_blocks; i++) {
        if (plan->weight_blocks[i].stage_idx == stage_idx)
            return &plan->weight_blocks[i];
    }
    return NULL;
}

/* Chained tiled execution */

/**
 * Execute a chain of stages with tile-through streaming.
 *
 * Only the chain's first input and last output touch slow memory.
 * All intermediate tensors between chain stages live exclusively in
 * fast as tile-sized buffers.
 *
 * @param plan       Parsed plan (possibly with weight_blob redirected).
 * @param first_idx  Index of the first stage in the chain.
 * @param num_chain  Number of stages in the chain.
 * @param mem        Memory manager.
 * @param kernel     Kernel dispatch function.
 * @param user_ctx   User context for kernels.
 */
static tigris_exec_error_t exec_chain_tiled(
    const tigris_plan_t *plan,
    uint16_t             first_idx,
    uint16_t             num_chain,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    executor_workspace_impl_t *workspace)
{
    if (num_chain < 2 || num_chain > workspace->chain_capacity)
        return TIGRIS_EXEC_ERR_TILE;

    const tigris_stage_t **stages = workspace->stages;
    chain_stage_info_t *info = workspace->info;

    for (uint16_t c = 0; c < num_chain; c++) {
        stages[c] = &plan->stages[first_idx + c];

        /* Get full tensor dimensions (NHWC) */
        const uint16_t *sin  = tigris_stage_inputs(plan, stages[c]);
        const uint16_t *sout = tigris_stage_outputs(plan, stages[c]);
        const int32_t *ish = tigris_tensor_shape(plan, &plan->tensors[sin[0]]);
        const int32_t *osh = tigris_tensor_shape(plan, &plan->tensors[sout[0]]);
        info[c].full_in_h  = ish[1];
        info[c].full_in_w  = ish[2];
        info[c].full_out_h = osh[1];
        info[c].full_out_w = osh[2];

        /* Compose receptive fields of ALL spatial ops in this stage.
         * When a stage has multiple spatial ops (including pool), the composed
         * parameters determine how many stage-input rows are needed to produce
         * a given number of stage-output rows. */
        int32_t comp_stride = 1, comp_eff_kh = 1, comp_pad_top = 0;
        int sp_count = 0;

        const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
        for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
            uint8_t t = plan->ops[sops[j]].op_type;
            if (!is_height_tiling_op(t))
                return TIGRIS_EXEC_ERR_TILE;
            if (is_height_spatial_op(t)) {
                const tigris_spatial_attrs_t *sp = &plan->ops[sops[j]].spatial;
                int32_t sh = sp->stride_h;
                int32_t dh = sp->dilation_h ? sp->dilation_h : 1;
                int32_t ekh = (sp->kernel_h - 1) * dh + 1;
                int32_t pt = sp->pad_top;

                /* Get this spatial op's input tensor dims */
                const uint16_t *op_ins = tigris_op_inputs(plan, &plan->ops[sops[j]]);
                const int32_t *op_ish = tigris_tensor_shape(plan, &plan->tensors[op_ins[0]]);
                const uint16_t *op_outs = tigris_op_outputs(plan, &plan->ops[sops[j]]);
                const int32_t *op_osh = tigris_tensor_shape(plan, &plan->tensors[op_outs[0]]);

                if (sp_count >= workspace->spatial_capacity)
                    return TIGRIS_EXEC_ERR_WORKSPACE;
                size_t spi = spatial_index(
                    workspace, c, (uint16_t)sp_count);
                workspace->sp_op_indices[spi] = (int32_t)sops[j];
                workspace->sp_strides[spi] = sh;
                workspace->sp_eff_khs[spi] = ekh;
                workspace->sp_pad_tops[spi] = pt;
                workspace->sp_full_in_hs[spi] = op_ish[1];
                workspace->sp_full_in_ws[spi] = op_ish[2];
                workspace->sp_full_out_ws[spi] = op_osh[2];

                /* Compose: eff_kh_new = eff_kh + (ekh - 1) * stride
                 *          pad_top_new = pad_top + pt * stride
                 *          stride_new  = stride * sh              */
                comp_eff_kh = comp_eff_kh + (ekh - 1) * comp_stride;
                comp_pad_top = comp_pad_top + pt * comp_stride;
                comp_stride = comp_stride * sh;

                sp_count++;
            }
        }

        info[c].sp_count     = sp_count;
        info[c].stride_h     = comp_stride;
        info[c].eff_kh       = comp_eff_kh;
        info[c].orig_pad_top = comp_pad_top;
    }

    /* 0. Decompress all chain stages' weight blocks into fast prefix */
    uint8_t **weight_bases = workspace->weight_bases;
    memset(weight_bases, 0, num_chain * sizeof(*weight_bases));

    int32_t chain_tile_h_override = 0;

    if (plan->weight_blocks && plan->num_weight_blocks > 0) {
        for (uint16_t c = 0; c < num_chain; c++) {
            const tigris_weight_block_t *wb =
                find_weight_block(plan, first_idx + c);
            if (!wb || wb->compressed_size == 0) continue;

            uint8_t *scratch = mem->fast_base + mem->fast_used;
            const uint8_t *csrc =
                plan->weight_blocks_data + wb->blob_offset;
            uint32_t needed;

            if (!align_tensor_size(wb->uncompressed_size, &needed) ||
                mem->fast_used > mem->fast_size ||
                needed > mem->fast_size - mem->fast_used) {
                TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                return TIGRIS_EXEC_ERR_MEM;
            }

            if (plan->weight_compression == TIGRIS_COMPRESS_LZ4) {
                int32_t dec = tigris_lz4_decompress(
                    csrc, wb->compressed_size,
                    scratch, wb->uncompressed_size);
                if (dec < 0 || (uint32_t)dec != wb->uncompressed_size)
                    return TIGRIS_EXEC_ERR_KERNEL;
            } else {
                memcpy(scratch, csrc, wb->uncompressed_size);
            }

            weight_bases[c] = scratch;
            mem->fast_used += needed;
            tigris_mem_note_fast_peak(mem);
        }
        mem->fast_reserved = mem->fast_used;
    }

    /* 0b. Validate chain_tile_h against effective fast budget.
     *     Decompressed weights may have reduced available space. */
    {
        int32_t cth = stages[0]->chain_tile_h;
        if (cth <= 0) cth = 1;
        int32_t last_oh = info[num_chain - 1].full_out_h;
        if (cth > last_oh) cth = last_oh;

        uint32_t eff_fast = mem->fast_size - mem->fast_reserved;

        /* Binary search: find largest tile_h that fits in eff_fast */
        int32_t lo = 1, hi = cth, best = 0;
        while (lo <= hi) {
            int32_t mid = (lo + hi) / 2;

            /* Back-propagate heights for this tile_h */
            int32_t *s_out = workspace->s_out;
            int32_t *s_in = workspace->s_in;
            s_out[num_chain - 1] = mid;
            for (int c = num_chain - 1; c >= 0; c--) {
                if (c < num_chain - 1)
                    s_out[c] = s_in[c + 1];
                int32_t ek = info[c].eff_kh;
                int32_t st = info[c].stride_h;
                int32_t overlap = ek - st;
                s_in[c] = s_out[c] * st + (overlap > 0 ? overlap : 0);
            }

            /* Compute total tile buffer cost */
            uint32_t total = 0;

            /* First stage input tiles - ALL inputs (with alignment) */
            for (uint16_t ii = 0; ii < stages[0]->inputs_count; ii++) {
                uint16_t tidx = tigris_stage_inputs(plan, stages[0])[ii];
                const tigris_tensor_t *t = &plan->tensors[tidx];
                const int32_t *sh = tigris_tensor_shape(plan, t);
                if (t->ndim == 4 && sh[0] && sh[1] && sh[2] && sh[3]) {
                    uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] * (uint32_t)sh[2] * (uint32_t)sh[3];
                    uint32_t elem = t->size_bytes / numel;
                    total += TILE_ALIGN_UP((uint32_t)sh[0] * (uint32_t)s_in[0] *
                             (uint32_t)sh[2] * (uint32_t)sh[3] * elem);
                }
            }

            /* All op output tiles - use correct per-op intermediate height.
             * Forward-compute heights through spatial ops in each stage.
             * TILE_ALIGN_UP each allocation to match runtime allocator. */
            for (uint16_t c = 0; c < num_chain && total <= eff_fast; c++) {
                int32_t cur_h = s_in[c];
                int sp_j = 0;
                const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
                for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
                    const tigris_op_t *op = &plan->ops[sops[j]];
                    uint8_t ot = op->op_type;

                    /* Spatial ops reduce height */
                    if (is_height_spatial_op(ot) &&
                        sp_j < info[c].sp_count) {
                        size_t spi = spatial_index(
                            workspace, c, (uint16_t)sp_j);
                        int32_t ekh = workspace->sp_eff_khs[spi];
                        int32_t sh  = workspace->sp_strides[spi];
                        int32_t pt  = workspace->sp_pad_tops[spi];
                        /* Worst case (with full padding at boundary) */
                        int32_t oh = (cur_h + pt - ekh) / sh + 1;
                        cur_h = oh;
                        sp_j++;
                    }

                    const uint16_t *outs = tigris_op_outputs(plan, op);
                    for (uint8_t k = 0; k < op->num_outputs; k++) {
                        const tigris_tensor_t *t = &plan->tensors[outs[k]];
                        if (t->ndim != 4) continue;
                        const int32_t *shp = tigris_tensor_shape(plan, t);
                        if (!shp[0] || !shp[1] || !shp[2] || !shp[3]) continue;
                        uint32_t elem = t->size_bytes /
                            (uint32_t)(shp[0] * shp[1] * shp[2] * shp[3]);
                        total += TILE_ALIGN_UP((uint32_t)shp[0] * (uint32_t)cur_h *
                                 (uint32_t)shp[2] * (uint32_t)shp[3] * elem);
                    }
                }
            }

            if (total <= eff_fast) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        if (best <= 0) {
            TIGRIS_PLATFORM_DBG("chain tile_h=0 cth=%ld oh=%ld eff=%lu res=%lu\n",
                (long)cth, (long)last_oh,
                (unsigned long)eff_fast, (unsigned long)mem->fast_reserved);
            return TIGRIS_EXEC_ERR_MEM;
        }

        /* Override chain_tile_h with the validated value */
        chain_tile_h_override = best;
    }

    /* 1. Pre-allocate full output(s) of LAST stage in slow.
     *    Last stage may have multiple outputs consumed by later stages. */
    const tigris_stage_t *last_stage = stages[num_chain - 1];
    const uint16_t *last_sout = tigris_stage_outputs(plan, last_stage);
    void **out_slow_bases = workspace->output_ptrs;
    memset(out_slow_bases, 0,
           workspace->output_capacity * sizeof(*out_slow_bases));

    for (uint16_t i = 0; i < last_stage->outputs_count; i++) {
        uint16_t tidx = last_sout[i];
        tigris_mem_error_t merr = tigris_mem_alloc_slow(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {
            TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
            return TIGRIS_EXEC_ERR_MEM;
        }
        out_slow_bases[i] = mem->tensor_ptrs[tidx];
    }

    /* 2. Save slow ptrs for first stage's inputs */
    const tigris_stage_t *first_stage = stages[0];
    const uint16_t *first_sin = tigris_stage_inputs(plan, first_stage);
    void **in_slow_bases = workspace->input_ptrs;
    for (uint16_t i = 0; i < first_stage->inputs_count; i++)
        in_slow_bases[i] = mem->tensor_ptrs[first_sin[i]];

    /* Line-buffered chain flag from the compiler (recomputing chain head).
     * When set, each interior tile ROLLS every intermediate stage's overlap
     * output rows to the front of a persistent buffer and computes only the
     * new rows, instead of recomputing the full back-propagated range per tile.
     * Gated so an unflagged chain runs the exact recompute path byte-for-byte.
     *
     * Preconditions the roll relies on (documented, not just asserted):
     *  - chain_tile_h is constant across the run (a single value below);
     *  - the fast arena is single-owner across the TIGRIS_PLATFORM_FEED_WDT()
     *    yield between tiles, because the roll now depends on the persistent
     *    buffers' BYTES surviving from one tile to the next, not merely on the
     *    bump allocator handing back the same address;
     *  - backend weight-decompression scratch is carved once, before the loop.
     * Roll is only applied to rank-4, batch-1 op outputs (height tiling). */
    int line_buffered =
        (first_stage->_reserved1 & TIGRIS_STAGE_FLAG_LINE_BUFFERED) != 0;

    /* 3. Get chain tile height - use validated override (accounts for
     *    decompressed weights reducing available fast space) */
    int32_t chain_tile_h = chain_tile_h_override;
    int32_t last_full_out_h = info[num_chain - 1].full_out_h;

    int32_t num_tiles = (last_full_out_h + chain_tile_h - 1) / chain_tile_h;
    int32_t out_row_cursor = 0;

#ifdef TIGRIS_COUNT_KERNEL_ROWS
    g_tigris_chain_tile_h = chain_tile_h;
    g_tigris_chain_num_tiles = num_tiles;
    g_tigris_interior_clamps = 0;
#endif

    /* Persistent fast bases for the last stage's spilled outputs. The spill
     * repoints those tensors at slow each tile; the roll re-establishes them
     * here so the next tile's kernels write back into the persistent buffer. */
    void *roll_last_fast[TIGRIS_MAX_STAGE_OUTPUTS];

    /* 3b. Line-buffered pre-allocation: allocate every chain op-output buffer
     *     ONCE at its maximum (interior) tile height, so the buffers keep fixed
     *     addresses and surviving content across tiles. The first-input load
     *     and last-output spill continue to cycle above these persistent
     *     buffers. Sizes mirror the budget-validation forward pass exactly, so
     *     the reserved footprint equals the already-validated tile working set. */
    if (line_buffered) {
        /* Load-bearing bound: roll_last_fast is sized TIGRIS_MAX_STAGE_OUTPUTS
         * and later indexed by [0, last_stage->outputs_count), so reject a plan
         * whose last stage has more outputs before recording any base below. */
        if (last_stage->outputs_count > TIGRIS_MAX_STAGE_OUTPUTS)
            return TIGRIS_EXEC_ERR_WORKSPACE;

        /* Interior (unclamped) back-propagated heights for chain_tile_h. */
        int32_t *s_out = workspace->s_out;
        int32_t *s_in = workspace->s_in;
        s_out[num_chain - 1] = chain_tile_h;
        for (int c = num_chain - 1; c >= 0; c--) {
            if (c < num_chain - 1)
                s_out[c] = s_in[c + 1];
            int32_t overlap = info[c].eff_kh - info[c].stride_h;
            s_in[c] = s_out[c] * info[c].stride_h +
                      (overlap > 0 ? overlap : 0);
        }

        for (uint16_t c = 0; c < num_chain; c++) {
            int32_t cur_h = s_in[c];
            int sp_j = 0;
            const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
            for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
                const tigris_op_t *op = &plan->ops[sops[j]];
                uint8_t ot = op->op_type;
                if (is_height_spatial_op(ot) && sp_j < info[c].sp_count) {
                    size_t spi = spatial_index(workspace, c, (uint16_t)sp_j);
                    int32_t ekh = workspace->sp_eff_khs[spi];
                    int32_t sh  = workspace->sp_strides[spi];
                    int32_t pt  = workspace->sp_pad_tops[spi];
                    cur_h = (cur_h + pt - ekh) / sh + 1;
                    sp_j++;
                }
                const uint16_t *outs = tigris_op_outputs(plan, op);
                for (uint8_t k = 0; k < op->num_outputs; k++) {
                    uint16_t tidx = outs[k];
                    const tigris_tensor_t *t = &plan->tensors[tidx];
                    uint32_t bytes;
                    if (t->ndim == 4) {
                        const int32_t *sh = tigris_tensor_shape(plan, t);
                        uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] *
                                         (uint32_t)sh[2] * (uint32_t)sh[3];
                        uint32_t elem = t->size_bytes / numel;
                        bytes = (uint32_t)sh[0] * (uint32_t)cur_h *
                                (uint32_t)sh[2] * (uint32_t)sh[3] * elem;
                    } else {
                        bytes = t->size_bytes;
                    }
                    tigris_mem_error_t merr =
                        tigris_mem_alloc_fast(mem, tidx, bytes);
                    if (merr != TIGRIS_MEM_OK) {
                        TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n",
                            __LINE__,
                            (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                            (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                        return TIGRIS_EXEC_ERR_MEM;
                    }
                }
            }
        }

        /* Record last-stage output persistent bases for the post-spill restore. */
        for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++)
            roll_last_fast[oi] = mem->tensor_ptrs[last_sout[oi]];

        /* These buffers must survive tigris_mem_reset_fast between tiles. */
        mem->fast_reserved = mem->fast_used;

        /* Nothing rolled yet: clear the previous-tile bookkeeping. */
        for (uint16_t i = 0; i < mem->num_tensors; i++) {
            workspace->roll_prev_gs[i] = 0;
            workspace->roll_prev_ge[i] = 0;
        }
    }

    /* 4. Tile loop - iterate over last stage's output tiles */
    for (int32_t tile = 0; tile < num_tiles; tile++) {
        /* a. Compute last stage output range */
        int32_t out_start = tile * chain_tile_h;
        int32_t out_end = out_start + chain_tile_h;
        if (out_end > last_full_out_h)
            out_end = last_full_out_h;

        /* b. Back-propagate ranges through chain: last->first.
         *    Uses COMPOSED receptive fields so stages with multiple
         *    spatial ops get enough input rows for all of them. */
        int32_t *in_starts = workspace->in_starts;
        int32_t *in_ends = workspace->in_ends;
        int32_t *out_starts = workspace->out_starts;
        int32_t *out_ends = workspace->out_ends;

        {
            int32_t os = out_start, oe = out_end;
            for (int c = num_chain - 1; c >= 0; c--) {
                out_starts[c] = os;
                out_ends[c]   = oe;

                /* Back-propagate through ALL spatial ops in reverse order
                 * to get per-op tile info and the stage's input range. */
                int32_t cur_os = os, cur_oe = oe;
                for (int j = info[c].sp_count - 1; j >= 0; j--) {
                    size_t spi = spatial_index(
                        workspace, (uint16_t)c, (uint16_t)j);
                    workspace->sp_tile_out_h[spi] = cur_oe - cur_os;
                    /* Global output row range of this spatial op, before it is
                     * back-propagated to the input range below. Consumed by the
                     * line-buffer roll to size the per-op overlap. */
                    workspace->sp_tile_out_gs[spi] = cur_os;
                    workspace->sp_tile_out_ge[spi] = cur_oe;

                    int32_t sh  = workspace->sp_strides[spi];
                    int32_t ekh = workspace->sp_eff_khs[spi];
                    int32_t pt  = workspace->sp_pad_tops[spi];
                    int32_t fih = workspace->sp_full_in_hs[spi];

                    int32_t is_ = cur_os * sh - pt;
                    int32_t ie_ = (cur_oe - 1) * sh - pt + ekh;

                    int32_t ept = 0, epb = 0;
                    if (is_ < 0) { ept = -is_; is_ = 0; }
                    if (ie_ > fih) { epb = ie_ - fih; ie_ = fih; }

                    workspace->sp_tile_pt[spi] = ept;
                    workspace->sp_tile_pb[spi] = epb;
                    workspace->sp_tile_in_h[spi] = ie_ - is_;

                    cur_os = is_;
                    cur_oe = ie_;
                }

                /* If no spatial ops, passthrough */
                if (info[c].sp_count == 0) {
                    cur_os = os;
                    cur_oe = oe;
                }

                in_starts[c] = cur_os;
                in_ends[c]   = cur_oe;

                /* Previous stage's output range = this stage's input range */
                if (c > 0) { os = cur_os; oe = cur_oe; }
            }
        }

        /* c. Load first stage's input tile from slow -> fast */
        for (uint16_t i = 0; i < first_stage->inputs_count; i++) {
            uint16_t tidx = first_sin[i];
            mem->tensor_ptrs[tidx] = in_slow_bases[i];
            tigris_mem_error_t merr = tigris_mem_load_tile(
                mem, plan, tidx, in_starts[0], in_ends[0]);
            if (merr != TIGRIS_MEM_OK) {
                TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                return TIGRIS_EXEC_ERR_MEM;
            }
        }

        /* d. Run each stage in the chain */
        for (uint16_t c = 0; c < num_chain; c++) {
            const tigris_stage_t *st = stages[c];
            int32_t tile_in_h  = in_ends[c] - in_starts[c];
            int32_t tile_out_h = out_ends[c] - out_starts[c];

            /* Resolve per-stage weight blob (compressed plan support) */
            tigris_plan_t cplan_copy;
            const tigris_plan_t *cplan = plan;
            if (weight_bases[c]) {
                cplan_copy = *plan;
                cplan_copy.weight_blob = weight_bases[c];
                cplan = &cplan_copy;
            }

            /* Set initial tile context - applies to ops before the
             * first spatial op (if any) or the entire stage. */
            mem->tile.active     = 1;
            mem->tile.in_h       = tile_in_h;
            mem->tile.out_h      = tile_out_h;
            mem->tile.in_w       = info[c].full_in_w;
            mem->tile.out_w      = info[c].full_out_w;
            if (info[c].sp_count > 0) {
                size_t spi = spatial_index(workspace, c, 0);
                mem->tile.pad_top    = workspace->sp_tile_pt[spi];
                mem->tile.pad_bottom = workspace->sp_tile_pb[spi];
            } else {
                mem->tile.pad_top    = 0;
                mem->tile.pad_bottom = 0;
            }

            /* Track current data height flowing through the stage.
             * Starts at tile_in_h, decreases after each spatial op. */
            int32_t cur_data_h = tile_in_h;
            int sp_j = 0;  /* next spatial op index within this stage */

            /* Line-buffer roll gating for this stage on this tile. A stage may
             * roll only on interior tiles where no spatial op was boundary
             * clamped (top pad exists only in tile 0's head, bottom pad only in
             * the last tile's tail) and where it contains no Concat (Concat does
             * not honor the new-row offset contract). Otherwise every op in the
             * stage full-computes, which also re-establishes its boundary rows. */
            int stage_rollable = 0;
            if (line_buffered && tile > 0) {
                stage_rollable = 1;
                for (int spx = 0; spx < info[c].sp_count; spx++) {
                    size_t spi2 = spatial_index(workspace, c, (uint16_t)spx);
                    if (workspace->sp_tile_pt[spi2] != 0 ||
                        workspace->sp_tile_pb[spi2] != 0) {
                        stage_rollable = 0;
#ifdef TIGRIS_COUNT_KERNEL_ROWS
                        /* Count clamps on genuinely interior tiles (not tile 0,
                         * not the last tile) so a test can assert the mixed
                         * roll/full-compute fallback actually ran. */
                        if (tile + 1 < num_tiles)
                            g_tigris_interior_clamps++;
#endif
                        break;
                    }
                }
                if (stage_rollable) {
                    const uint16_t *csops = tigris_stage_ops(plan, st);
                    for (uint16_t jj = 0; jj < st->ops_count; jj++) {
                        if (plan->ops[csops[jj]].op_type == TIGRIS_OP_CONCAT) {
                            stage_rollable = 0;
                            break;
                        }
                    }
                }
            }

            /* Global row range of the tensor currently flowing through the
             * stage: the stage input range before the first spatial op, then
             * each spatial op's output range. Drives the per-op roll overlap. */
            int32_t run_gs = in_starts[c];
            int32_t run_ge = in_ends[c];

            /* Alloc tile-sized op outputs in fast, run kernels */
            const uint16_t *sops = tigris_stage_ops(cplan, st);
            for (uint16_t j = 0; j < st->ops_count; j++) {
                uint16_t op_idx = sops[j];
                const tigris_op_t *op = &cplan->ops[op_idx];

                /* Detect spatial op and set per-op tile context */
                size_t spi = spatial_index(
                    workspace, c, (uint16_t)sp_j);
                int is_spatial = (sp_j < info[c].sp_count &&
                    (int32_t)op_idx == workspace->sp_op_indices[spi]);
                if (is_spatial) {
                    mem->tile.in_h       = workspace->sp_tile_in_h[spi];
                    mem->tile.out_h      = workspace->sp_tile_out_h[spi];
                    mem->tile.pad_top    = workspace->sp_tile_pt[spi];
                    mem->tile.pad_bottom = workspace->sp_tile_pb[spi];
                    mem->tile.in_w       = workspace->sp_full_in_ws[spi];
                    mem->tile.out_w      = workspace->sp_full_out_ws[spi];
                }

                /* Spatial op outputs have fewer rows than input (stride).
                 * Non-spatial (pointwise) ops preserve height. */
                int32_t alloc_h = is_spatial
                    ? workspace->sp_tile_out_h[spi] : cur_data_h;

                /* Allocate op output at correct tile height. Line-buffered
                 * chains pre-allocated every op-output buffer once (persistent
                 * across tiles), so skip the per-tile alloc and reuse those. */
                const uint16_t *outs = tigris_op_outputs(cplan, op);
                for (uint8_t k = 0; !line_buffered && k < op->num_outputs; k++) {
                    uint16_t tidx = outs[k];
                    const tigris_tensor_t *t = &cplan->tensors[tidx];
                    if (t->ndim == 4) {
                        const int32_t *sh = tigris_tensor_shape(cplan, t);
                        uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] * (uint32_t)sh[2] * (uint32_t)sh[3];
                        uint32_t elem_size = t->size_bytes / numel;
                        uint32_t tile_bytes = (uint32_t)sh[0] *
                            (uint32_t)alloc_h *
                            (uint32_t)sh[2] * (uint32_t)sh[3] * elem_size;
                        tigris_mem_error_t merr = tigris_mem_alloc_fast(
                            mem, tidx, tile_bytes);
                        if (merr != TIGRIS_MEM_OK) {
                            TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                            return TIGRIS_EXEC_ERR_MEM;
                        }
                    } else {
                        /* Non-4D tensor: alloc full size */
                        if (!mem->tensor_ptrs[tidx]) {
                            tigris_mem_error_t merr = tigris_mem_alloc_fast(
                                mem, tidx, t->size_bytes);
                            if (merr != TIGRIS_MEM_OK) {
                                TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                                return TIGRIS_EXEC_ERR_MEM;
                            }
                        }
                    }
                }

                /* Line-buffer roll: this op's output global row range. Spatial
                 * ops take their back-propagated output range; pointwise ops
                 * preserve the range currently flowing through the stage. */
                int32_t op_gs = is_spatial
                    ? workspace->sp_tile_out_gs[spi] : run_gs;
                int32_t op_ge = is_spatial
                    ? workspace->sp_tile_out_ge[spi] : run_ge;

                /* Full row count this op would compute without rolling; restored
                 * after the kernel so a rolled op does not leak its narrowed
                 * out_h into the next (e.g. a following pointwise op). */
                int32_t op_full_out_h = mem->tile.out_h;

                /* Default: full compute (offsets 0, byte-identical to the
                 * recompute path). Roll only a rank-4, batch-1, single-output
                 * op whose reconstructed height matches the tile context, so a
                 * mismatch (e.g. a pointwise op ahead of the spatial op) safely
                 * falls back to full compute. */
                mem->tile.out_row_start = 0;
                mem->tile.in_row_start  = 0;
                if (line_buffered && stage_rollable && op->num_outputs == 1) {
                    uint16_t otid = outs[0];
                    const tigris_tensor_t *ot = &cplan->tensors[otid];
                    int32_t existing_out_h = op_full_out_h;
                    int32_t this_h = op_ge - op_gs;
                    const int32_t *osh = tigris_tensor_shape(cplan, ot);
                    if (ot->ndim == 4 && osh[0] == 1 && this_h == existing_out_h) {
                        int32_t prev_gs = workspace->roll_prev_gs[otid];
                        int32_t prev_ge = workspace->roll_prev_ge[otid];
                        int32_t prev_h  = prev_ge - prev_gs;
                        int32_t overlap = prev_ge - op_gs;
                        if (overlap < 0) overlap = 0;
                        if (overlap > this_h) overlap = this_h;
                        if (overlap > prev_h) overlap = prev_h;
                        if (overlap > 0) {
                            /* Per-row byte stride is width*channels*elem, i.e.
                             * size_bytes / (N*H); independent of tile height. */
                            uint32_t row_bytes = ot->size_bytes /
                                ((uint32_t)osh[0] * (uint32_t)osh[1]);
                            int32_t src_row = op_gs - prev_gs; /* = prev_h - overlap */
                            uint8_t *base = (uint8_t *)mem->tensor_ptrs[otid];
                            /* Carry the overlap rows to the front, then compute
                             * only the new rows after them. */
                            memmove(base,
                                    base + (size_t)src_row * row_bytes,
                                    (size_t)overlap * row_bytes);
                            mem->tile.out_row_start = overlap;
                            /* Spatial kernels reconstruct the global input row
                             * from the output row, so they read the full input
                             * buffer at offset 0. Height-preserving ops compute
                             * Y[i] = f(X[i]) over the literal offsets, and their
                             * input shares the same global base row as their
                             * output, so their input must skip the same overlap
                             * the output does. */
                            mem->tile.in_row_start  = is_spatial ? 0 : overlap;
                            mem->tile.out_h = existing_out_h - overlap;
                        }
                    }
                }

                /* Record this op's output range for the next tile's roll.
                 * Done every op, every tile (including full-computed ops), so
                 * the buffer always describes its current-tile content. These
                 * ranges are read only on the line-buffered roll path, so skip
                 * the writes entirely on the hot unflagged path. */
                if (line_buffered && op->num_outputs == 1) {
                    workspace->roll_prev_gs[outs[0]] = op_gs;
                    workspace->roll_prev_ge[outs[0]] = op_ge;
                }

                int kret = kernel(cplan, op, op_idx, mem, user_ctx);
                if (kret != 0)
                    return TIGRIS_EXEC_ERR_KERNEL;

                /* Undo any roll narrowing so the next op sees this op's full
                 * height and offsets 0 (a following pointwise op inherits this
                 * out_h; a following spatial op overwrites it below). */
                mem->tile.out_h        = op_full_out_h;
                mem->tile.out_row_start = 0;
                mem->tile.in_row_start  = 0;

                /* After spatial op: advance to next, update data height,
                 * running range, and tile context for pointwise followers. */
                if (sp_j < info[c].sp_count &&
                    (int32_t)op_idx == workspace->sp_op_indices[spi]) {
                    cur_data_h = workspace->sp_tile_out_h[spi];
                    mem->tile.in_h = cur_data_h;
                    mem->tile.in_w = workspace->sp_full_out_ws[spi];
                    run_gs = workspace->sp_tile_out_gs[spi];
                    run_ge = workspace->sp_tile_out_ge[spi];
                    sp_j++;
                    /* Pre-set pad for next spatial op (if any) */
                    if (sp_j < info[c].sp_count) {
                        spi = spatial_index(
                            workspace, c, (uint16_t)sp_j);
                        mem->tile.pad_top = workspace->sp_tile_pt[spi];
                        mem->tile.pad_bottom = workspace->sp_tile_pb[spi];
                    }
                }
            }
        }

        /* e. Spill last stage's output tile to slow */
        for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++) {
            uint16_t tidx = last_sout[oi];
            if (!out_slow_bases[oi]) continue;
            tigris_mem_error_t merr = tigris_mem_spill_tile(
                mem, plan, tidx, out_slow_bases[oi],
                out_row_cursor, out_row_cursor + (out_ends[num_chain - 1] - out_starts[num_chain - 1]));
            if (merr != TIGRIS_MEM_OK) {
                TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                return TIGRIS_EXEC_ERR_MEM;
            }
            /* spill_tile repointed this tensor at slow; for a line-buffered
             * chain, restore its persistent fast base so the next tile's last
             * stage writes back into the surviving buffer. After the final
             * tile leave it at slow, where the model output lives. */
            if (line_buffered && tile + 1 < num_tiles)
                mem->tensor_ptrs[tidx] = roll_last_fast[oi];
        }

        out_row_cursor += out_ends[num_chain - 1] - out_starts[num_chain - 1];

        /* f. Reset fast arena for next tile */
        tigris_mem_reset_fast(mem);

        TIGRIS_PLATFORM_FEED_WDT();
    }

    /* 5. Verify all output rows covered */
    if (out_row_cursor != last_full_out_h) {
        return TIGRIS_EXEC_ERR_TILE;
    }

    /* 6. Clear tile context, restore first stage's input ptrs to slow,
     *    and NULL all intermediate op-output tensor pointers.
     *
     *    After chain execution, intermediate tensors (op outputs from all
     *    chain stages) still point into the fast arena even though it's been
     *    reset.  If left dangling, compact_fast() will find them, use their
     *    full plan sizes (not tile sizes), and push fast_used past the arena. */
    memset(&mem->tile, 0, sizeof(mem->tile));
    for (uint16_t i = 0; i < first_stage->inputs_count; i++)
        mem->tensor_ptrs[first_sin[i]] = in_slow_bases[i];

    for (uint16_t c = 0; c < num_chain; c++) {
        const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
        for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
            const tigris_op_t *op = &plan->ops[sops[j]];
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                uint16_t tidx = outs[k];
                /* Skip the last stage's outputs - already spilled to slow */
                if (c == num_chain - 1) {
                    int is_output = 0;
                    for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++) {
                        if (tidx == last_sout[oi]) { is_output = 1; break; }
                    }
                    if (is_output) continue;
                }
                mem->tensor_ptrs[tidx] = NULL;
            }
        }
    }

    return TIGRIS_EXEC_OK;
}

/* Public API */

tigris_exec_error_t tigris_run_with_workspace_buffer(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace_storage,
    size_t               workspace_size)
{
    if (!plan || !mem || !kernel)
        return TIGRIS_EXEC_ERR_NULL;
    if (!plan->header)
        return TIGRIS_EXEC_ERR_NULL;
    if (!workspace_storage)
        return TIGRIS_EXEC_ERR_WORKSPACE;

    if (plan->header->num_stages == 0 || !plan->stages)
        return TIGRIS_EXEC_ERR_NO_STAGES;

    executor_workspace_limits_t limits;
    executor_workspace_impl_t workspace_view;
    if (!workspace_limits(plan, &limits) ||
        !workspace_layout(
            workspace_storage, workspace_size, &limits, &workspace_view))
        return TIGRIS_EXEC_ERR_WORKSPACE;
    executor_workspace_impl_t *workspace = &workspace_view;

    /* Preserve the caller's fast-arena state across all temporary stage and
     * chain weight reservations.  During the run, fast_used is also the reset
     * floor so caller-owned bytes above fast_reserved cannot be overwritten. */
    const uint32_t caller_fast_used = mem->fast_used;
    const uint32_t caller_fast_reserved = mem->fast_reserved;
    if (caller_fast_used > mem->fast_size ||
        caller_fast_reserved > caller_fast_used)
        return TIGRIS_EXEC_ERR_MEM;
    mem->fast_reserved = caller_fast_used;

    tigris_exec_error_t result = TIGRIS_EXEC_OK;

    /* Zero stats if provided */
    if (stats) memset(stats, 0, sizeof(*stats));

    /* Pre-compute last_consumer[t] = last stage that reads tensor t.
     * After that stage completes, t's slow memory can be reclaimed. */
    uint16_t num_t = plan->header->num_tensors;
    uint16_t *last_consumer = workspace->last_consumer;
    for (uint16_t i = 0; i < num_t; i++)
        last_consumer[i] = 0;
    for (uint16_t s = 0; s < plan->header->num_stages; s++) {
        const tigris_stage_t *st = &plan->stages[s];
        const uint16_t *sin = tigris_stage_inputs(plan, st);
        for (uint16_t i = 0; i < st->inputs_count; i++) {
            if (sin[i] < num_t && s > last_consumer[sin[i]])
                last_consumer[sin[i]] = s;
        }
    }
    /* Model outputs are never freed */
    for (uint8_t i = 0; i < plan->header->num_model_outputs; i++) {
        if (plan->model_outputs[i] < num_t)
            last_consumer[plan->model_outputs[i]] = UINT16_MAX;
    }

    for (uint16_t s = 0; s < plan->header->num_stages; s++) {
        const tigris_stage_t *stage = &plan->stages[s];

        /* Per-stage weight decompression (skip for chain heads) */
        tigris_plan_t stage_plan;
        const tigris_plan_t *run_plan = plan;

        /* Chain stages decompress internally in exec_chain_tiled */
        if (plan->weight_blocks && plan->num_weight_blocks > 0
            && !(stage->chain_len >= 2 && stage->chain_id == s)) {
            const tigris_weight_block_t *wb = find_weight_block(plan, s);
            if (wb && wb->compressed_size > 0) {
                /* Decompress into fast arena prefix */
                uint8_t *scratch = mem->fast_base + mem->fast_used;
                const uint8_t *csrc = plan->weight_blocks_data + wb->blob_offset;
                uint32_t needed;

                if (!align_tensor_size(wb->uncompressed_size, &needed) ||
                    mem->fast_used > mem->fast_size ||
                    needed > mem->fast_size - mem->fast_used) {
                    result = TIGRIS_EXEC_ERR_MEM;
                    goto restore_fast_arena;
                }

                if (plan->weight_compression == TIGRIS_COMPRESS_LZ4) {
                    int32_t dec = tigris_lz4_decompress(
                        csrc, wb->compressed_size,
                        scratch, wb->uncompressed_size);
                    if (dec < 0 || (uint32_t)dec != wb->uncompressed_size) {
                        result = TIGRIS_EXEC_ERR_KERNEL;
                        goto restore_fast_arena;
                    }
                } else {
                    memcpy(scratch, csrc, wb->uncompressed_size);
                }

                mem->fast_used += needed;
                mem->fast_reserved = mem->fast_used;  /* lock weights in arena */
                tigris_mem_note_fast_peak(mem);

                stage_plan = *plan;  /* shallow copy */
                stage_plan.weight_blob = scratch;
                run_plan = &stage_plan;
            }
        }

        /* Decide: chain, tile, or normal */

        if (stage->chain_len >= 2 && stage->chain_id == s) {
            /* First stage of a chain - execute entire chain */
            tigris_exec_error_t err = exec_chain_tiled(
                run_plan, s, stage->chain_len, mem, kernel, user_ctx,
                workspace);
            if (err != TIGRIS_EXEC_OK) {
                TIGRIS_PLATFORM_DBG("S%u chain(%u) ERR=%d fast=%lu/%lu slow=%lu/%lu\n",
                    s, stage->chain_len, (int)err,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                result = err;
                goto restore_fast_arena;
            }
            if (stats) stats->stages_chain += stage->chain_len;
            /* Skip remaining chain stages (they were executed above) */
            s += stage->chain_len - 1;
        } else if (stage->chain_len >= 2) {
            /* Non-head chain stage - should have been skipped; just in case */
        } else {
            const uint16_t *sin  = tigris_stage_inputs(run_plan, stage);
            const uint16_t *sout = tigris_stage_outputs(run_plan, stage);
            uint32_t total_io = 0;
            uint8_t rank = 0;
            int common_tile_rank = 1;
            for (uint16_t i = 0; i < stage->inputs_count; i++) {
                total_io += run_plan->tensors[sin[i]].size_bytes;
                uint8_t tensor_rank = run_plan->tensors[sin[i]].ndim;
                if (rank == 0) rank = tensor_rank;
                if (tensor_rank != rank) common_tile_rank = 0;
            }
            for (uint16_t i = 0; i < stage->outputs_count; i++) {
                total_io += run_plan->tensors[sout[i]].size_bytes;
                uint8_t tensor_rank = run_plan->tensors[sout[i]].ndim;
                if (rank == 0) rank = tensor_rank;
                if (tensor_rank != rank) common_tile_rank = 0;
            }
            if (rank != 3 && rank != 4) common_tile_rank = 0;

            /* Only tile stages whose complete operator sequence implements
             * the selected axis-1 stripe contract. A serialized tile plan is
             * a compiler promise: fail closed if an older or malformed plan
             * attaches it to a stage the runtime cannot tile safely. Without
             * a tile plan, unsupported stages retain ordinary full-tensor
             * execution and may use slow overflow. */
            uint8_t tile_axis = stage_tile_axis(run_plan, stage);
            int axis1_tileable =
                common_tile_rank &&
                stage_supports_axis1_tiling(run_plan, stage, rank);
            int needs_tiling =
                common_tile_rank &&
                total_io > (mem->fast_size - mem->fast_reserved);

            if (needs_tiling &&
                stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN &&
                (stage->tile_plan_idx >= run_plan->header->num_tile_plans ||
                 !run_plan->tile_plans ||
                 !run_plan->tile_plans[stage->tile_plan_idx].tileable ||
                 tile_axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH ||
                 !axis1_tileable)) {
                result = TIGRIS_EXEC_ERR_TILE;
                goto restore_fast_arena;
            }

            /* Rank-4 schema-v2-v4 plans retain the historical automatic
             * height path even when they carry no standalone tile record.
             * Rank-3 length tiling is schema-v5-only and requires the explicit
             * axis promise. */
            int may_tile =
                axis1_tileable &&
                (rank == 4 ||
                 (run_plan->header->version >=
                      TIGRIS_SCHEMA_VERSION_TILE_AXIS &&
                  tile_axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH));

            if (needs_tiling && may_tile) {
                tigris_exec_error_t err = exec_stage_tiled(
                    run_plan, stage, mem, kernel, user_ctx,
                    last_consumer, s, workspace);
                if (err != TIGRIS_EXEC_OK) {
                    TIGRIS_PLATFORM_DBG("S%u tiled ERR=%d io=%lu fast=%lu/%lu slow=%lu/%lu\n",
                        s, (int)err, (unsigned long)total_io,
                        (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                        (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                    result = err;
                    goto restore_fast_arena;
                }
                if (stats) stats->stages_tiled++;
            } else {
                tigris_exec_error_t err = exec_stage_normal(
                    run_plan, stage, mem, kernel, user_ctx, stats, workspace);
                if (err != TIGRIS_EXEC_OK) {
                    TIGRIS_PLATFORM_DBG("S%u normal ERR=%d io=%lu fast=%lu/%lu slow=%lu/%lu\n",
                        s, (int)err, (unsigned long)total_io,
                        (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                        (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                    result = err;
                    goto restore_fast_arena;
                }
                if (stats) stats->stages_normal++;
            }
        }

        /* Track slow high-water mark */
        if (stats && mem->slow_used > stats->slow_peak)
            stats->slow_peak = mem->slow_used;

        /* Free dead tensors from slow and compact */
        int need_compact = 0;
        for (uint16_t t = 0; t < num_t; t++) {
            if (last_consumer[t] <= s && last_consumer[t] != UINT16_MAX) {
                uint8_t *p = (uint8_t *)mem->tensor_ptrs[t];
                if (p && p >= mem->slow_base &&
                    p < mem->slow_base + mem->slow_size) {
                    mem->tensor_ptrs[t] = NULL;
                    need_compact = 1;
                }
            }
        }
        if (need_compact)
            compact_slow(mem, plan);

        /* Release stage/chain-local fast allocations before the next group,
         * retaining the caller's in-run reset floor. */
        mem->fast_used = caller_fast_used;
        mem->fast_reserved = caller_fast_used;

        TIGRIS_PLATFORM_FEED_WDT();
    }

restore_fast_arena:
    mem->fast_used = caller_fast_used;
    mem->fast_reserved = caller_fast_reserved;
    return result;
}

size_t tigris_executor_workspace_required(const tigris_plan_t *plan)
{
    executor_workspace_limits_t limits;
    if (!workspace_limits(plan, &limits))
        return 0;
    return workspace_bytes_for_limits(&limits);
}

tigris_exec_error_t tigris_run_with_workspace(
    const tigris_plan_t          *plan,
    tigris_mem_t                 *mem,
    tigris_kernel_fn              kernel,
    void                         *user_ctx,
    tigris_exec_stats_t          *stats,
    tigris_executor_workspace_t  *workspace)
{
    if (!workspace)
        return TIGRIS_EXEC_ERR_WORKSPACE;
    return tigris_run_with_workspace_buffer(
        plan, mem, kernel, user_ctx, stats,
        workspace->bytes, sizeof(workspace->bytes));
}

size_t tigris_executor_workspace_size(void)
{
    return sizeof(tigris_executor_workspace_t);
}

const char *tigris_exec_error_str(tigris_exec_error_t err)
{
    switch (err) {
        case TIGRIS_EXEC_OK:             return "OK";
        case TIGRIS_EXEC_ERR_NULL:       return "null pointer argument";
        case TIGRIS_EXEC_ERR_MEM:        return "memory allocation failed";
        case TIGRIS_EXEC_ERR_KERNEL:     return "kernel callback error";
        case TIGRIS_EXEC_ERR_NO_STAGES:  return "plan has no stages";
        case TIGRIS_EXEC_ERR_TILE:       return "tiled execution error";
        case TIGRIS_EXEC_ERR_WORKSPACE:  return "executor workspace unavailable";
        default:                         return "unknown error";
    }
}

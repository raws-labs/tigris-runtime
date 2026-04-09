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

/*
 * Executor limits - validated at plan load time (tigris_plan_load).
 * Exceeding these limits returns TIGRIS_ERR_PLAN_LIMITS.
 */
#define MAX_STAGE_INPUTS      16
#define MAX_STAGE_OUTPUTS     16
#define MAX_CHAIN_STAGES      16
#define MAX_SPATIAL_PER_STAGE  8
#define TIGRIS_MAX_TENSORS   512  /* max tensor index for alive/last_use arrays */

#define TILE_ALIGN_UP(x) (((x) + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u))

/* Memory compaction */

/**
 * Compact the fast pool: move alive tensors to the front, reclaim
 * space from dead ones.  Similar to compact_slow but for fast arena.
 */
static void compact_fast(tigris_mem_t *mem, const tigris_plan_t *plan)
{
    /* Collect alive fast-pool tensors, sorted by current address. */
    uint16_t alive[TIGRIS_MAX_TENSORS];
    uint16_t n = 0;

    for (uint16_t i = 0; i < mem->num_tensors && n < TIGRIS_MAX_TENSORS; i++) {
        uint8_t *p = (uint8_t *)mem->tensor_ptrs[i];
        if (p && p >= mem->fast_base + mem->fast_reserved &&
            p < mem->fast_base + mem->fast_size)
            alive[n++] = i;
    }

    if (n == 0) { mem->fast_used = mem->fast_reserved; return; }

    /* Insertion sort by address (n is small) */
    for (uint16_t i = 1; i < n; i++) {
        uint16_t key = alive[i];
        uint8_t *kp = (uint8_t *)mem->tensor_ptrs[key];
        int j = (int)i - 1;
        while (j >= 0 && (uint8_t *)mem->tensor_ptrs[alive[j]] > kp) {
            alive[j + 1] = alive[j];
            j--;
        }
        alive[j + 1] = key;
    }

    /* Slide each alive tensor toward the front (after reserved area) */
    uint32_t wp = mem->fast_reserved;
    for (uint16_t i = 0; i < n; i++) {
        uint16_t tidx = alive[i];
        uint32_t sz = (plan->tensors[tidx].size_bytes + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u);
        uint8_t *src = (uint8_t *)mem->tensor_ptrs[tidx];
        uint8_t *dst = mem->fast_base + wp;
        if (dst != src)
            memmove(dst, src, sz);
        mem->tensor_ptrs[tidx] = dst;
        wp += sz;
    }
    mem->fast_used = wp;
}

/**
 * Compact the slow pool: move alive tensors to the front, reclaim
 * space from dead ones.  Tensors must be processed in address order
 * so memmove never reads from already-overwritten memory.
 */
static void compact_slow(tigris_mem_t *mem, const tigris_plan_t *plan)
{
    /* Collect alive slow-pool tensors, sorted by current address. */
    uint16_t alive[TIGRIS_MAX_TENSORS];
    uint16_t n = 0;

    for (uint16_t i = 0; i < mem->num_tensors && n < TIGRIS_MAX_TENSORS; i++) {
        uint8_t *p = (uint8_t *)mem->tensor_ptrs[i];
        if (p && p >= mem->slow_base && p < mem->slow_base + mem->slow_size)
            alive[n++] = i;
    }

    if (n == 0) { mem->slow_used = 0; return; }

    /* Insertion sort by address (n is small) */
    for (uint16_t i = 1; i < n; i++) {
        uint16_t key = alive[i];
        uint8_t *kp = (uint8_t *)mem->tensor_ptrs[key];
        int j = (int)i - 1;
        while (j >= 0 && (uint8_t *)mem->tensor_ptrs[alive[j]] > kp) {
            alive[j + 1] = alive[j];
            j--;
        }
        alive[j + 1] = key;
    }

    /* Slide each alive tensor toward the front */
    uint32_t wp = 0;
    for (uint16_t i = 0; i < n; i++) {
        uint16_t tidx = alive[i];
        uint32_t sz = (plan->tensors[tidx].size_bytes + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u);
        uint8_t *src = (uint8_t *)mem->tensor_ptrs[tidx];
        uint8_t *dst = mem->slow_base + wp;
        if (dst != src)
            memmove(dst, src, sz);
        mem->tensor_ptrs[tidx] = dst;
        wp += sz;
    }
    mem->slow_used = wp;
}

/* Helpers */

/**
 * Find the first spatial op (Conv/DWConv/MaxPool) in a stage.
 * Returns its index in plan->ops, or -1 if none.
 */
static int find_spatial_op(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t t = plan->ops[sops[j]].op_type;
        if (t == TIGRIS_OP_CONV || t == TIGRIS_OP_DEPTHWISE ||
            t == TIGRIS_OP_MAX_POOL)
            return (int)sops[j];
    }
    return -1;
}

/**
 * Count total spatial ops (Conv/DWConv/MaxPool) in a stage.
 * exec_stage_tiled only handles stages with at most 1 spatial op.
 */
static int count_spatial_ops(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    int count = 0;
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint8_t t = plan->ops[sops[j]].op_type;
        if (t == TIGRIS_OP_CONV || t == TIGRIS_OP_DEPTHWISE ||
            t == TIGRIS_OP_MAX_POOL)
            count++;
    }
    return count;
}

/* Non-tiled stage execution */

static tigris_exec_error_t exec_stage_normal(
    const tigris_plan_t *plan,
    const tigris_stage_t *stage,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats)
{
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    void *saved_slow[MAX_STAGE_INPUTS];

    /* Pre-compute last_use_op[t] = index within sops[] of last op that reads t.
     * After that op completes, t's fast memory can be reclaimed (if not a stage output). */
    uint16_t num_t = plan->header->num_tensors;
    uint16_t last_use_op[TIGRIS_MAX_TENSORS];
    if (num_t > TIGRIS_MAX_TENSORS) num_t = TIGRIS_MAX_TENSORS;
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
    for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++) {
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
    for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++) {
        uint16_t tidx = sin[i];
        mem->tensor_ptrs[tidx] = saved_slow[i];
    }

    return TIGRIS_EXEC_OK;
}

/* Tiled stage execution */

static tigris_exec_error_t exec_stage_tiled(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    const uint16_t           *last_consumer,
    uint16_t                  current_stage)
{
    /* 1. Pre-allocate full output tensors in slow (or reuse input for in-place) */
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    void *out_slow_bases[MAX_STAGE_OUTPUTS];
    int in_place = 0;  /* set if output reuses input's slow memory */

    for (uint16_t i = 0; i < stage->outputs_count && i < MAX_STAGE_OUTPUTS; i++) {
        uint16_t tidx = sout[i];
        tigris_mem_error_t merr = tigris_mem_alloc_slow(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {
            /* Allocation failed - try in-place if:
             *   - Single input/output
             *   - Input dies after this stage (last_consumer == current_stage)
             *   - Input and output have same size
             */
            if (stage->inputs_count == 1 && stage->outputs_count == 1 &&
                last_consumer && last_consumer[sin[0]] == current_stage &&
                plan->tensors[sin[0]].size_bytes == plan->tensors[tidx].size_bytes)
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
    void *in_slow_bases[MAX_STAGE_INPUTS];
    for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++)
        in_slow_bases[i] = mem->tensor_ptrs[sin[i]];

    /* Find the spatial op (if any) for computing output H per tile */
    int sp_op_idx = find_spatial_op(plan, stage);
    const tigris_spatial_attrs_t *sp_attrs = NULL;
    if (sp_op_idx >= 0)
        sp_attrs = &plan->ops[sp_op_idx].spatial;

    /* Get input/output image dimensions for the first input/output (NHWC) */
    const int32_t *in_shape = tigris_tensor_shape(plan, &plan->tensors[sin[0]]);
    int32_t full_in_h = in_shape[1];
    int32_t full_in_w = in_shape[2];

    const int32_t *out_shape = tigris_tensor_shape(plan, &plan->tensors[sout[0]]);
    int32_t full_out_h = out_shape[1];
    int32_t full_out_w = out_shape[2];

    int32_t orig_pad_top = sp_attrs ? sp_attrs->pad_top : 0;
    int32_t stride = sp_attrs ? sp_attrs->stride_h : 1;
    int32_t dh = (sp_attrs && sp_attrs->dilation_h) ? sp_attrs->dilation_h : 1;
    int32_t eff_kh = sp_attrs ? ((sp_attrs->kernel_h - 1) * dh + 1) : 1;

    /* Compute the per-output-row memory cost including intermediates.
     *
     * At peak, all stage inputs, all intermediate op outputs, and all
     * stage outputs are alive in fast. For each output row:
     *   - Stage inputs cost stride rows each (spatial op) or 1 row (pointwise)
     *   - Intermediates and stage outputs cost 1 row each
     *   - Fixed overhead from the convolution kernel window (eff_kh - stride)
     *     applied only to stage inputs.
     *
     * We enumerate all unique tensors referenced by ops in this stage.
     */
    uint32_t in_row_cost = 0;   /* input bytes per output row (proportional) */
    uint32_t in_fixed_cost = 0; /* input bytes from kernel overlap (one-time per tile) */
    for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++) {
        const tigris_tensor_t *t = &plan->tensors[sin[i]];
        const int32_t *sh = tigris_tensor_shape(plan, t);
        /* NHWC: row_bytes = W * C * elem_size, where one row = one H-position */
        uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] * (uint32_t)sh[2] * (uint32_t)sh[3];
        uint32_t elem_size = t->size_bytes / numel;
        uint32_t row_bytes = (uint32_t)sh[2] * (uint32_t)sh[3] * elem_size;
        uint32_t n = (uint32_t)sh[0];
        in_row_cost += n * row_bytes * (uint32_t)stride;
        in_fixed_cost += n * row_bytes * (uint32_t)(eff_kh - stride);
    }

    /* Collect all op output tensors (intermediates + stage outputs).
     * All of these are tile-sized (out_h rows) in fast simultaneously at peak. */
    uint32_t all_out_row_cost = 0;
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            const tigris_tensor_t *t = &plan->tensors[outs[k]];
            if (t->ndim != 4) continue;
            const int32_t *sh = tigris_tensor_shape(plan, t);
            /* NHWC: row_bytes = W * C * elem_size */
            uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] * (uint32_t)sh[2] * (uint32_t)sh[3];
            uint32_t elem_size = t->size_bytes / numel;
            uint32_t row_bytes = (uint32_t)sh[2] * (uint32_t)sh[3] * elem_size;
            uint32_t n = (uint32_t)sh[0];
            all_out_row_cost += n * row_bytes;
        }
    }

    uint32_t cost_per_row = in_row_cost + all_out_row_cost;
    int32_t out_tile_h;
    if (cost_per_row == 0) {
        out_tile_h = full_out_h;
    } else {
        int32_t avail = (int32_t)(mem->fast_size - mem->fast_reserved)
                      - (int32_t)in_fixed_cost;
        if (avail <= 0)
            return TIGRIS_EXEC_ERR_TILE;
        out_tile_h = avail / (int32_t)cost_per_row;
        if (out_tile_h < 1) out_tile_h = 1;
        if (out_tile_h > full_out_h) out_tile_h = full_out_h;
    }

    int32_t num_tiles = (full_out_h + out_tile_h - 1) / out_tile_h;
    int32_t out_row_cursor = 0;

    /* 3. Tile loop - iterate over output row ranges */
    for (int32_t tile = 0; tile < num_tiles; tile++) {
        /* a. Output bounds for this tile */
        int32_t out_start = tile * out_tile_h;
        int32_t out_end   = out_start + out_tile_h;
        if (out_end > full_out_h)
            out_end = full_out_h;
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

        /* c. Set tile context */
        mem->tile.active     = 1;
        mem->tile.pad_top    = (uint8_t)eff_pad_top;
        mem->tile.pad_bottom = (uint8_t)eff_pad_bottom;
        mem->tile.in_h       = tile_in_h;
        mem->tile.out_h      = tile_out_h;
        mem->tile.in_w       = full_in_w;
        mem->tile.out_w      = full_out_w;

        /* d. LOAD: load tile for each stage input */
        for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++) {
            uint16_t tidx = sin[i];
            mem->tensor_ptrs[tidx] = in_slow_bases[i]; /* restore before load */
            tigris_mem_error_t merr = tigris_mem_load_tile(
                mem, plan, tidx, in_start, in_end);
            if (merr != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
        }

        /* e. EXECUTE: alloc tile-sized outputs in fast, run kernels */
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx = sops[j];
            const tigris_op_t *op = &plan->ops[op_idx];

            /* Alloc tile-sized outputs in fast (NHWC) */
            const uint16_t *outs = tigris_op_outputs(plan, op);
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                uint16_t tidx = outs[k];
                const tigris_tensor_t *t = &plan->tensors[tidx];
                const int32_t *shape = tigris_tensor_shape(plan, t);
                uint32_t elem_size = t->size_bytes /
                    (uint32_t)(shape[0] * shape[1] * shape[2] * shape[3]);
                /* NHWC: N * tile_out_h * W * C */
                uint32_t tile_bytes = (uint32_t)shape[0] * (uint32_t)tile_out_h *
                    (uint32_t)shape[2] * (uint32_t)shape[3] * elem_size;
                tigris_mem_error_t merr = tigris_mem_alloc_fast(
                    mem, tidx, tile_bytes);
                if (merr != TIGRIS_MEM_OK)
                    return TIGRIS_EXEC_ERR_MEM;
            }

            int kret = kernel(plan, op, op_idx, mem, user_ctx);
            if (kret != 0)
                return TIGRIS_EXEC_ERR_KERNEL;

            /* After spatial op, update context for pointwise followers */
            if ((int)op_idx == sp_op_idx) {
                mem->tile.in_h = tile_out_h;
                mem->tile.in_w = full_out_w;
            }
        }

        /* f. SPILL: spill tile for each stage output */
        for (uint16_t i = 0; i < stage->outputs_count && i < MAX_STAGE_OUTPUTS; i++) {
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
        for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++)
            mem->tensor_ptrs[sin[i]] = in_slow_bases[i];
    } else {
        /* In-place: input was overwritten, clear its pointer */
        for (uint16_t i = 0; i < stage->inputs_count && i < MAX_STAGE_INPUTS; i++)
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

#define ALIGN_TENSOR(x) (((x) + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u))

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

typedef struct {
    int32_t full_in_h;
    int32_t full_in_w;
    int32_t full_out_h;
    int32_t full_out_w;
    /* Composed receptive field for inter-stage back-propagation */
    int32_t stride_h;        /* product of all spatial strides */
    int32_t eff_kh;          /* composed effective kernel height */
    int32_t orig_pad_top;    /* composed top padding */
    /* Per-spatial-op tracking within the stage */
    int     sp_count;
    int     sp_op_indices[MAX_SPATIAL_PER_STAGE];
    int32_t sp_strides[MAX_SPATIAL_PER_STAGE];
    int32_t sp_eff_khs[MAX_SPATIAL_PER_STAGE];
    int32_t sp_pad_tops[MAX_SPATIAL_PER_STAGE];    /* original op padding */
    int32_t sp_full_in_hs[MAX_SPATIAL_PER_STAGE];  /* full input H of each spatial op */
    int32_t sp_full_in_ws[MAX_SPATIAL_PER_STAGE];  /* full input W */
    int32_t sp_full_out_ws[MAX_SPATIAL_PER_STAGE];  /* full output W */
} chain_stage_info_t;

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
    void                *user_ctx)
{
    if (num_chain < 2 || num_chain > MAX_CHAIN_STAGES)
        return TIGRIS_EXEC_ERR_TILE;

    const tigris_stage_t *stages[MAX_CHAIN_STAGES];
    chain_stage_info_t info[MAX_CHAIN_STAGES];

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
         * When a stage has multiple spatial ops (e.g. Conv S2 -> DW S1 -> PW),
         * the composed parameters determine how many stage-input rows are
         * needed to produce a given number of stage-output rows. */
        int32_t comp_stride = 1, comp_eff_kh = 1, comp_pad_top = 0;
        int sp_count = 0;

        const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
        for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
            uint8_t t = plan->ops[sops[j]].op_type;
            if (t == TIGRIS_OP_CONV || t == TIGRIS_OP_DEPTHWISE) {
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

                if (sp_count < MAX_SPATIAL_PER_STAGE) {
                    info[c].sp_op_indices[sp_count] = (int)sops[j];
                    info[c].sp_strides[sp_count] = sh;
                    info[c].sp_eff_khs[sp_count] = ekh;
                    info[c].sp_pad_tops[sp_count] = pt;
                    info[c].sp_full_in_hs[sp_count] = op_ish[1];
                    info[c].sp_full_in_ws[sp_count] = op_ish[2];
                    info[c].sp_full_out_ws[sp_count] = op_osh[2];
                }

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
    uint8_t *weight_bases[MAX_CHAIN_STAGES];
    memset(weight_bases, 0, sizeof(weight_bases));

    int32_t chain_tile_h_override = 0;

    if (plan->weight_blocks && plan->num_weight_blocks > 0) {
        for (uint16_t c = 0; c < num_chain; c++) {
            const tigris_weight_block_t *wb =
                find_weight_block(plan, first_idx + c);
            if (!wb || wb->compressed_size == 0) continue;

            uint8_t *scratch = mem->fast_base + mem->fast_used;
            const uint8_t *csrc =
                plan->weight_blocks_data + wb->blob_offset;
            uint32_t needed = ALIGN_TENSOR(wb->uncompressed_size);

            if (mem->fast_used + needed > mem->fast_size) {
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
            int32_t s_out[MAX_CHAIN_STAGES], s_in[MAX_CHAIN_STAGES];
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
                    if ((ot == TIGRIS_OP_CONV || ot == TIGRIS_OP_DEPTHWISE) &&
                        sp_j < info[c].sp_count) {
                        int32_t ekh = info[c].sp_eff_khs[sp_j];
                        int32_t sh  = info[c].sp_strides[sp_j];
                        int32_t pt  = info[c].sp_pad_tops[sp_j];
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
    void *out_slow_bases[MAX_STAGE_OUTPUTS];
    memset(out_slow_bases, 0, sizeof(out_slow_bases));

    for (uint16_t i = 0; i < last_stage->outputs_count && i < MAX_STAGE_OUTPUTS; i++) {
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
    void *in_slow_bases[MAX_STAGE_INPUTS];
    for (uint16_t i = 0; i < first_stage->inputs_count && i < MAX_STAGE_INPUTS; i++)
        in_slow_bases[i] = mem->tensor_ptrs[first_sin[i]];

    /* 3. Get chain tile height - use validated override (accounts for
     *    decompressed weights reducing available fast space) */
    int32_t chain_tile_h = chain_tile_h_override;
    int32_t last_full_out_h = info[num_chain - 1].full_out_h;

    int32_t num_tiles = (last_full_out_h + chain_tile_h - 1) / chain_tile_h;
    int32_t out_row_cursor = 0;

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
        int32_t in_starts[MAX_CHAIN_STAGES], in_ends[MAX_CHAIN_STAGES];
        int32_t out_starts[MAX_CHAIN_STAGES], out_ends[MAX_CHAIN_STAGES];

        /* Per-spatial-op tile info (computed per tile) */
        int32_t sp_tile_in_h[MAX_CHAIN_STAGES][MAX_SPATIAL_PER_STAGE];
        int32_t sp_tile_out_h[MAX_CHAIN_STAGES][MAX_SPATIAL_PER_STAGE];
        int32_t sp_tile_pt[MAX_CHAIN_STAGES][MAX_SPATIAL_PER_STAGE];
        int32_t sp_tile_pb[MAX_CHAIN_STAGES][MAX_SPATIAL_PER_STAGE];

        {
            int32_t os = out_start, oe = out_end;
            for (int c = num_chain - 1; c >= 0; c--) {
                out_starts[c] = os;
                out_ends[c]   = oe;

                /* Back-propagate through ALL spatial ops in reverse order
                 * to get per-op tile info and the stage's input range. */
                int32_t cur_os = os, cur_oe = oe;
                for (int j = info[c].sp_count - 1; j >= 0; j--) {
                    sp_tile_out_h[c][j] = cur_oe - cur_os;

                    int32_t sh  = info[c].sp_strides[j];
                    int32_t ekh = info[c].sp_eff_khs[j];
                    int32_t pt  = info[c].sp_pad_tops[j];
                    int32_t fih = info[c].sp_full_in_hs[j];

                    int32_t is_ = cur_os * sh - pt;
                    int32_t ie_ = (cur_oe - 1) * sh - pt + ekh;

                    int32_t ept = 0, epb = 0;
                    if (is_ < 0) { ept = -is_; is_ = 0; }
                    if (ie_ > fih) { epb = ie_ - fih; ie_ = fih; }

                    sp_tile_pt[c][j] = ept;
                    sp_tile_pb[c][j] = epb;
                    sp_tile_in_h[c][j] = ie_ - is_;

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
        for (uint16_t i = 0; i < first_stage->inputs_count && i < MAX_STAGE_INPUTS; i++) {
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
                mem->tile.pad_top    = (uint8_t)sp_tile_pt[c][0];
                mem->tile.pad_bottom = (uint8_t)sp_tile_pb[c][0];
            } else {
                mem->tile.pad_top    = 0;
                mem->tile.pad_bottom = 0;
            }

            /* Track current data height flowing through the stage.
             * Starts at tile_in_h, decreases after each spatial op. */
            int32_t cur_data_h = tile_in_h;
            int sp_j = 0;  /* next spatial op index within this stage */

            /* Alloc tile-sized op outputs in fast, run kernels */
            const uint16_t *sops = tigris_stage_ops(cplan, st);
            for (uint16_t j = 0; j < st->ops_count; j++) {
                uint16_t op_idx = sops[j];
                const tigris_op_t *op = &cplan->ops[op_idx];

                /* Detect spatial op and set per-op tile context */
                int is_spatial = (sp_j < info[c].sp_count &&
                    (int)op_idx == info[c].sp_op_indices[sp_j]);
                if (is_spatial) {
                    mem->tile.in_h       = sp_tile_in_h[c][sp_j];
                    mem->tile.out_h      = sp_tile_out_h[c][sp_j];
                    mem->tile.pad_top    = (uint8_t)sp_tile_pt[c][sp_j];
                    mem->tile.pad_bottom = (uint8_t)sp_tile_pb[c][sp_j];
                    mem->tile.in_w       = info[c].sp_full_in_ws[sp_j];
                    mem->tile.out_w      = info[c].sp_full_out_ws[sp_j];
                }

                /* Spatial op outputs have fewer rows than input (stride).
                 * Non-spatial (pointwise) ops preserve height. */
                int32_t alloc_h = is_spatial
                    ? sp_tile_out_h[c][sp_j] : cur_data_h;

                /* Allocate op output at correct tile height */
                const uint16_t *outs = tigris_op_outputs(cplan, op);
                for (uint8_t k = 0; k < op->num_outputs; k++) {
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

                int kret = kernel(cplan, op, op_idx, mem, user_ctx);
                if (kret != 0)
                    return TIGRIS_EXEC_ERR_KERNEL;

                /* After spatial op: advance to next, update data height
                 * and tile context for pointwise followers. */
                if (sp_j < info[c].sp_count &&
                    (int)op_idx == info[c].sp_op_indices[sp_j]) {
                    cur_data_h = sp_tile_out_h[c][sp_j];
                    mem->tile.in_h = cur_data_h;
                    mem->tile.in_w = info[c].sp_full_out_ws[sp_j];
                    sp_j++;
                    /* Pre-set pad for next spatial op (if any) */
                    if (sp_j < info[c].sp_count) {
                        mem->tile.pad_top = (uint8_t)sp_tile_pt[c][sp_j];
                        mem->tile.pad_bottom = (uint8_t)sp_tile_pb[c][sp_j];
                    }
                }
            }
        }

        /* e. Spill last stage's output tile to slow */
        for (uint16_t oi = 0; oi < last_stage->outputs_count && oi < MAX_STAGE_OUTPUTS; oi++) {
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
    for (uint16_t i = 0; i < first_stage->inputs_count && i < MAX_STAGE_INPUTS; i++)
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

tigris_exec_error_t tigris_run(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats)
{
    if (!plan || !mem || !kernel)
        return TIGRIS_EXEC_ERR_NULL;

    if (plan->header->num_stages == 0 || !plan->stages)
        return TIGRIS_EXEC_ERR_NO_STAGES;

    /* Zero stats if provided */
    if (stats) memset(stats, 0, sizeof(*stats));

    /* Pre-compute last_consumer[t] = last stage that reads tensor t.
     * After that stage completes, t's slow memory can be reclaimed. */
    uint16_t num_t = plan->header->num_tensors;
    uint16_t last_consumer[TIGRIS_MAX_TENSORS];
    if (num_t > TIGRIS_MAX_TENSORS) num_t = TIGRIS_MAX_TENSORS;
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
                uint32_t needed = ALIGN_TENSOR(wb->uncompressed_size);

                if (mem->fast_used + needed > mem->fast_size)
                    return TIGRIS_EXEC_ERR_MEM;

                if (plan->weight_compression == TIGRIS_COMPRESS_LZ4) {
                    int32_t dec = tigris_lz4_decompress(
                        csrc, wb->compressed_size,
                        scratch, wb->uncompressed_size);
                    if (dec < 0 || (uint32_t)dec != wb->uncompressed_size)
                        return TIGRIS_EXEC_ERR_KERNEL;
                } else {
                    memcpy(scratch, csrc, wb->uncompressed_size);
                }

                mem->fast_used += needed;
                mem->fast_reserved = mem->fast_used;  /* lock weights in arena */

                stage_plan = *plan;  /* shallow copy */
                stage_plan.weight_blob = scratch;
                run_plan = &stage_plan;
            }
        }

        /* Decide: chain, tile, or normal */

        if (stage->chain_len >= 2 && stage->chain_id == s) {
            /* First stage of a chain - execute entire chain */
            tigris_exec_error_t err = exec_chain_tiled(
                run_plan, s, stage->chain_len, mem, kernel, user_ctx);
            if (err != TIGRIS_EXEC_OK) {
                TIGRIS_PLATFORM_DBG("S%u chain(%u) ERR=%d fast=%lu/%lu slow=%lu/%lu\n",
                    s, stage->chain_len, (int)err,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                return err;
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
            int all_4d = 1;
            for (uint16_t i = 0; i < stage->inputs_count; i++) {
                total_io += run_plan->tensors[sin[i]].size_bytes;
                if (run_plan->tensors[sin[i]].ndim != 4) all_4d = 0;
            }
            for (uint16_t i = 0; i < stage->outputs_count; i++) {
                total_io += run_plan->tensors[sout[i]].size_bytes;
                if (run_plan->tensors[sout[i]].ndim != 4) all_4d = 0;
            }

            /* Only tile stages with at most 1 spatial op (Conv/DWConv/MaxPool).
             * Multi-spatial stages (e.g. full backbone with stride changes,
             * MaxPool, Resize) can't be tiled by exec_stage_tiled which
             * assumes a single spatial transition.  Run those as normal -
             * exec_stage_normal handles OOM via compaction + slow overflow. */
            int sp_count = count_spatial_ops(run_plan, stage);

            if (all_4d && sp_count <= 1 &&
                total_io > (mem->fast_size - mem->fast_reserved)) {
                tigris_exec_error_t err = exec_stage_tiled(
                    run_plan, stage, mem, kernel, user_ctx,
                    last_consumer, s);
                if (err != TIGRIS_EXEC_OK) {
                    TIGRIS_PLATFORM_DBG("S%u tiled ERR=%d io=%lu fast=%lu/%lu slow=%lu/%lu\n",
                        s, (int)err, (unsigned long)total_io,
                        (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                        (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                    return err;
                }
                if (stats) stats->stages_tiled++;
            } else {
                tigris_exec_error_t err = exec_stage_normal(
                    run_plan, stage, mem, kernel, user_ctx, stats);
                if (err != TIGRIS_EXEC_OK) {
                    TIGRIS_PLATFORM_DBG("S%u normal ERR=%d io=%lu fast=%lu/%lu slow=%lu/%lu\n",
                        s, (int)err, (unsigned long)total_io,
                        (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                        (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                    return err;
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

        /* Unlock decompressed weights after stage */
        mem->fast_reserved = 0;

        TIGRIS_PLATFORM_FEED_WDT();
    }

    return TIGRIS_EXEC_OK;
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
        default:                         return "unknown error";
    }
}

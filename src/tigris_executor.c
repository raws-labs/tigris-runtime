/**
 * @file tigris_executor.c
 * @brief Stage loop: load -> execute -> spill -> reset.
 */

#include "tigris_executor.h"
#include "tigris_loader.h"
#include "tigris_binary_operands.h"
#include "tigris_io_in_fast.h"
#include "tigris_lz4.h"
#include "tigris_kernel_window.h"

#include <string.h>

#ifndef TIGRIS_PLATFORM_DBG
#  ifdef CONFIG_IDF_TARGET_ESP32S3
#    include <stdio.h>
#    define TIGRIS_PLATFORM_DBG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#  else
#    define TIGRIS_PLATFORM_DBG(fmt, ...) ((void)0)
#  endif
#endif

/* A stage runs on exactly one executor path. Keeping the chained executor
 * out of line stops the compiler from folding its locals into the dispatcher
 * frame alongside the other executors, so each execution path stays within
 * the per-frame stack budget. It runs once per chain (the tile loop is inside
 * it), so the extra call is negligible. */
#if defined(__GNUC__) || defined(__clang__)
#  define TIGRIS_NOINLINE __attribute__((noinline))
#else
#  define TIGRIS_NOINLINE
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
    int32_t sp_first;   /* this stage's first entry in the chain's spatial-op arrays */
} chain_stage_info_t;

typedef char tigris_chain_stage_info_is_int32s[
    sizeof(chain_stage_info_t) == 9u * sizeof(int32_t) ? 1 : -1];

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
    int32_t *out_gs;           /* per op-output tensor: global row start held */
    int32_t *out_ge;           /* per op-output tensor: global row end held */
    int32_t *out_covered;      /* per stage output: rows a tiled stage or chain has written */
    void **roll_last_fast;     /* per chain last-stage output: its persistent fast buffer */
    uint16_t *last_use_op;
    uint16_t *last_consumer;
    /* Control flow: the depth, the running graph's first and end stage and the
     * stage to continue at, then per running control-flow operator a frame of
     * its stage, the range it was entered from, and the graph it runs. */
    uint16_t *frames;
    uint16_t tensor_capacity;
    uint16_t input_capacity;
    uint16_t output_capacity;
    uint16_t chain_capacity;
    uint32_t spatial_capacity;
    uint16_t depth_capacity;
} executor_workspace_impl_t;

typedef struct {
    uint16_t tensors;
    uint16_t inputs;
    uint16_t outputs;
    uint16_t chain_stages;
    uint32_t chain_spatial_ops;
    uint16_t subgraph_depth;
} executor_workspace_limits_t;

typedef char tigris_executor_compat_workspace_too_small[
    TIGRIS_EXECUTOR_WORKSPACE_BYTES >=
        TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(
            TIGRIS_MAX_TENSORS, TIGRIS_MAX_STAGE_INPUTS,
            TIGRIS_MAX_STAGE_OUTPUTS, TIGRIS_MAX_CHAIN_STAGES,
            TIGRIS_EXECUTOR_BUILD_CHAIN_SPATIAL_OPS,
            TIGRIS_EXECUTOR_BUILD_SUBGRAPH_DEPTH)
        ? 1 : -1];

static uintptr_t align_address(uintptr_t address, size_t alignment)
{
    return (address + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
}

/* Floor division for signed numerators (b > 0). Bare C `/` truncates toward
 * zero, which rounds negative numerators the wrong way; the ConvTranspose
 * inverse-rectangle back-computation needs a true floor for negative
 * numerators, so it must go through this helper, never `/`. */
static int32_t floor_div(int32_t a, int32_t b)
{
    return (a >= 0) ? (a / b) : -(((-a) + b - 1) / b);
}

static int is_height_tiling_op(uint8_t type)
{
    return tigris_height_spatial_op(type) ||
           tigris_height_upsample_op(type) ||
           type == TIGRIS_OP_CONV_TRANSPOSE ||
           type == TIGRIS_OP_RELU ||
           type == TIGRIS_OP_RELU6 ||
           type == TIGRIS_OP_SIGMOID ||
           type == TIGRIS_OP_TANH ||
           type == TIGRIS_OP_SOFTMAX ||
           type == TIGRIS_OP_LEAKY_RELU ||
           type == TIGRIS_OP_ELU ||
           type == TIGRIS_OP_LOG_SOFTMAX ||
           type == TIGRIS_OP_L2_NORMALIZATION ||
           type == TIGRIS_OP_ERF || type == TIGRIS_OP_GELU ||
           type == TIGRIS_OP_HARDSWISH ||
           type == TIGRIS_OP_ABS ||
           type == TIGRIS_OP_RSQRT ||
           type == TIGRIS_OP_NEG || type == TIGRIS_OP_LOGICAL_NOT || type == TIGRIS_OP_CAST ||
           type == TIGRIS_OP_QUANTIZE || type == TIGRIS_OP_DEQUANTIZE ||
           type == TIGRIS_OP_EXP ||
           type == TIGRIS_OP_LOG ||
           type == TIGRIS_OP_SQRT ||
           type == TIGRIS_OP_SQUARE ||
           type == TIGRIS_OP_FLOOR ||
           type == TIGRIS_OP_CEIL ||
           type == TIGRIS_OP_ROUND ||
           type == TIGRIS_OP_SIN ||
           type == TIGRIS_OP_COS ||
           type == TIGRIS_OP_LAYER_NORM ||
           type == TIGRIS_OP_ADD ||
           type == TIGRIS_OP_SUB ||
           type == TIGRIS_OP_DIV ||
           type == TIGRIS_OP_SQUARED_DIFFERENCE ||
           type == TIGRIS_OP_MAXIMUM ||
           type == TIGRIS_OP_MINIMUM ||
           type == TIGRIS_OP_FLOOR_DIV ||
           type == TIGRIS_OP_FLOOR_MOD ||
           type == TIGRIS_OP_PRELU ||
           (type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_LOGICAL_OR) ||
           type == TIGRIS_OP_SELECT_V2 || type == TIGRIS_OP_ADD_N ||
           type == TIGRIS_OP_MUL ||
           type == TIGRIS_OP_CONCAT;
}

static int is_axis1_binary_pointwise_op(uint8_t type)
{
    return type == TIGRIS_OP_ADD || type == TIGRIS_OP_SUB ||
           type == TIGRIS_OP_DIV ||
           type == TIGRIS_OP_SQUARED_DIFFERENCE ||
           type == TIGRIS_OP_MAXIMUM ||
           type == TIGRIS_OP_MINIMUM ||
           type == TIGRIS_OP_FLOOR_DIV ||
           type == TIGRIS_OP_FLOOR_MOD ||
           type == TIGRIS_OP_PRELU ||
           (type >= TIGRIS_OP_EQUAL && type <= TIGRIS_OP_LOGICAL_OR) ||
           type == TIGRIS_OP_SELECT_V2 || type == TIGRIS_OP_ADD_N ||
           type == TIGRIS_OP_MUL;
}

static int workspace_limits(
    const tigris_plan_t *plan, executor_workspace_limits_t *limits)
{
    if (!plan || !limits || !plan->header || !plan->stages ||
        !plan->ops || !plan->index_pool)
        return 0;

    memset(limits, 0, sizeof(*limits));
    limits->tensors = plan->header->num_tensors;
    limits->subgraph_depth = plan->subgraph_depth;

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

            uint32_t spatial = 0;
            for (uint16_t c = s; c < chain_end; c++) {
                const tigris_stage_t *chain_stage = &plan->stages[c];
                const uint16_t *ops = tigris_stage_ops(plan, chain_stage);
                for (uint16_t j = 0; j < chain_stage->ops_count; j++) {
                    if (ops[j] >= plan->header->num_ops)
                        return 0;
                    uint8_t type = plan->ops[ops[j]].op_type;
                    if (tigris_height_spatial_op(type))
                        spatial++;
                }
            }
            if (spatial > limits->chain_spatial_ops)
                limits->chain_spatial_ops = spatial;
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
        (limits->chain_stages != 0u ? (size_t)limits->outputs : 0u) +
        2u * (size_t)limits->chain_stages;
    if (count > (SIZE_MAX - total) / sizeof(void *))
        return 0;
    total += count * sizeof(void *);

    if (total > SIZE_MAX - (TIGRIS_EXECUTOR_INT32_ALIGNMENT - 1u))
        return 0;
    total += TIGRIS_EXECUTOR_INT32_ALIGNMENT - 1u;
    /* Chain stages, their spatial operators, two line-buffer arrays indexed by
     * op-output tensor, and the rows written per stage output. */
    count = 15u * (size_t)limits->chain_stages +
            2u * (size_t)limits->tensors + limits->outputs;
    if (limits->chain_spatial_ops > (SIZE_MAX - count) / 13u)
        return 0;
    count += 13u * (size_t)limits->chain_spatial_ops;
    if (count > (SIZE_MAX - total) / sizeof(int32_t))
        return 0;
    total += count * sizeof(int32_t);

    if (total > SIZE_MAX - (TIGRIS_EXECUTOR_UINT16_ALIGNMENT - 1u))
        return 0;
    total += TIGRIS_EXECUTOR_UINT16_ALIGNMENT - 1u;
    /* Two uint16 arrays indexed by tensor, then the control-flow frames. */
    count = 2u * (size_t)limits->tensors + 4u + 4u * (size_t)limits->subgraph_depth;
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
    workspace->roll_last_fast = (void **)cursor;
    if (limits->chain_stages != 0u)
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
        cursor += (size_t)limits->chain_spatial_ops * sizeof(int32_t);         \
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

    /* The global row range each op-output buffer currently holds, indexed by
     * tensor. Read by the spill to place the rows an op actually wrote, and by
     * the line-buffer roll as the previous tile's range. */
#define TIGRIS_LAYOUT_TENSOR_I32_FIELD(field)                                  \
    do {                                                                       \
        workspace->field = (int32_t *)cursor;                                 \
        cursor += (size_t)limits->tensors * sizeof(int32_t);                   \
    } while (0)
    TIGRIS_LAYOUT_TENSOR_I32_FIELD(out_gs);
    TIGRIS_LAYOUT_TENSOR_I32_FIELD(out_ge);
#undef TIGRIS_LAYOUT_TENSOR_I32_FIELD
    workspace->out_covered = (int32_t *)cursor;
    cursor += (size_t)limits->outputs * sizeof(int32_t);

    cursor = align_address(cursor, TIGRIS_EXECUTOR_UINT16_ALIGNMENT);
    workspace->last_use_op = (uint16_t *)cursor;
    cursor += (size_t)limits->tensors * sizeof(uint16_t);
    workspace->last_consumer = (uint16_t *)cursor;
    cursor += (size_t)limits->tensors * sizeof(uint16_t);
    workspace->frames = (uint16_t *)cursor;
    cursor += (4u + 4u * (size_t)limits->subgraph_depth) * sizeof(uint16_t);

    workspace->tensor_capacity = limits->tensors;
    workspace->input_capacity = limits->inputs;
    workspace->output_capacity = limits->outputs;
    workspace->chain_capacity = limits->chain_stages;
    workspace->spatial_capacity = limits->chain_spatial_ops;
    workspace->depth_capacity = limits->subgraph_depth;

    return cursor <= (uintptr_t)storage + storage_size;
}

static size_t spatial_index(
    const executor_workspace_impl_t *workspace, uint16_t chain, uint16_t op)
{
    return (size_t)workspace->info[chain].sp_first + op;
}

#ifdef TIGRIS_TRACE
static void trace_stage_begin(tigris_mem_t *mem, uint16_t stage, tigris_trace_path_t path)
{
    mem->trace_stage = stage;
    mem->trace_path = (uint8_t)path;
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_STAGE_BEGIN, TIGRIS_TRACE_POOL_NONE,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16, 0u, 0u, 0u, -1, -1, -1, -1);
}

static void trace_stage_end(tigris_mem_t *mem, uint16_t stage)
{
    mem->trace_stage = stage;
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_STAGE_END, TIGRIS_TRACE_POOL_NONE,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16, 0u, 0u, 0u, -1, -1, -1, -1);
}

static TIGRIS_NOINLINE int traced_kernel(tigris_kernel_fn kernel,
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t index,
    tigris_mem_t *mem, void *ctx)
{
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_OP, TIGRIS_TRACE_POOL_NONE,
        TIGRIS_TRACE_NONE_U16, index, 0u, 0u, 0u, -1, -1, -1, -1);
    return kernel(plan, op, index, mem, ctx);
}
#define TRACE_STAGE_BEGIN(...) trace_stage_begin(__VA_ARGS__)
#define TRACE_STAGE_END(...) trace_stage_end(__VA_ARGS__)
#define TRACED_KERNEL(...) traced_kernel(__VA_ARGS__)
#else
#define TRACE_STAGE_BEGIN(...) ((void)0)
#define TRACE_STAGE_END(...) ((void)0)
#define TRACED_KERNEL(fn, plan, op, index, mem, ctx) ((fn)((plan), (op), (index), (mem), (ctx)))
#endif

/* Memory compaction */

/**
 * Compact the fast or the slow pool: move alive tensors, in address order,
 * to offset `first` and up, reclaim space from dead ones, and return the new
 * used offset.
 */
static uint32_t compact_pool(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint8_t *base, uint32_t first, uint32_t size)
{
    uintptr_t cursor = (uintptr_t)(base + first);
    uintptr_t end = (uintptr_t)(base + size);
    uint32_t wp = first;

    /* Select the next live allocation by address on each pass.  This is O(n^2)
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
        if (dst != src) {
            memmove(dst, src, sz);
            TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_MOVE,
                base == mem->fast_base ? TIGRIS_TRACE_POOL_FAST : TIGRIS_TRACE_POOL_SLOW,
                tidx, TIGRIS_TRACE_NONE_U16, wp, TIGRIS_TRACE_OFFSET(src, base),
                sz, -1, -1, -1, -1);
        }

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
    if (mem->stats) mem->stats->compactions++;
    mem->fast_used = compact_pool(
        mem, plan, mem->fast_base, mem->fast_reserved, mem->fast_size);
}

/**
 * Compact the slow pool from its first aligned offset, where tigris_mem_init
 * starts it, so slow tensors keep TIGRIS_TENSOR_ALIGN for an unaligned
 * buffer.  Tensors must be processed in address order so memmove never
 * reads from already-overwritten memory.
 */
static void compact_slow(tigris_mem_t *mem, const tigris_plan_t *plan)
{
    const uint32_t mask = TIGRIS_TENSOR_ALIGN - 1u;
    const uint32_t first =
        (uint32_t)((TIGRIS_TENSOR_ALIGN - ((uintptr_t)mem->slow_base & mask)) & mask);
    if (mem->stats) mem->stats->compactions++;
    mem->slow_used = compact_pool(mem, plan, mem->slow_base, first, mem->slow_size);
}

/* Helpers */

/* Rows of a tiled tensor: axis 1 of the stored NHWC/NLC shape. */
static int32_t tensor_row_count(const tigris_plan_t *plan, uint16_t tidx)
{
    const tigris_tensor_t *t = &plan->tensors[tidx];
    if (t->ndim < 2u)
        return 0;
    return tigris_tensor_shape(plan, t)[1];
}

static uint8_t stage_tile_axis(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    return tile.tileable ? tile.axis : TIGRIS_TILE_AXIS_NONE;
}

/* Non-tiled stage execution */

/**
 * Whether an operator gives a tensor a new shape without moving a byte.
 *
 * A reshape produces the same bytes in the same order, and only the shape the
 * consumer reads differs, in two cases. Two tensors that each state their own
 * axis order hold their elements in the order the shape lists them, so
 * regrouping the shape moves nothing. Two spatial tensors hold their channel
 * axis last and their spatial axes in order ahead of it, so regrouping only
 * the spatial axes moves nothing either: that is what an unfold is, and it is
 * how a vision transformer turns a feature map into patches. There the batch
 * and the channel extent have to survive, because they are the ends of the
 * stored order and a reshape that changes either is moving bytes.
 *
 * The output can then share the input's buffer instead of being allocated and
 * copied into. Mirrors pure_reinterpretations in the compiler, which counts
 * the two as one allocation when it decides whether a stage fits.
 */
static int states_its_own_order(const tigris_tensor_t *t)
{
    return (t->flags & TIGRIS_TENSOR_LINEAR) != 0u || t->ndim <= 2u;
}

static int is_pure_reinterpretation(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    if (op->op_type != TIGRIS_OP_RESHAPE && op->op_type != TIGRIS_OP_FLATTEN)
        return 0;
    if (op->num_inputs != 1u || op->num_outputs != 1u)
        return 0;
    const tigris_tensor_t *in =
        &plan->tensors[tigris_op_inputs(plan, op)[0]];
    const tigris_tensor_t *out =
        &plan->tensors[tigris_op_outputs(plan, op)[0]];
    if (in->size_bytes != out->size_bytes)
        return 0;
    /* The bytes are only the same bytes if they mean the same thing. A plan
     * may give the two tensors different quantization, and then the reshape
     * has to re-encode rather than pass through. */
    const tigris_quant_param_t *in_qp = tigris_tensor_quant(plan, in);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, out);
    if (in_qp != out_qp) {
        if (!in_qp || !out_qp)
            return 0;
        if (in_qp->scale != out_qp->scale ||
            in_qp->zero_point != out_qp->zero_point)
            return 0;
    }
    int in_own = states_its_own_order(in);
    int out_own = states_its_own_order(out);
    if (in_own && out_own)
        return 1;
    if (in_own || out_own)
        return 0;
    if (plan->shape_pool == NULL || in->ndim == 0u || out->ndim == 0u)
        return 0;
    const int32_t *a = tigris_tensor_shape(plan, in);
    const int32_t *b = tigris_tensor_shape(plan, out);
    return a[0] == b[0] && a[in->ndim - 1u] == b[out->ndim - 1u];
}

/* Keep a roomy caller arena within the serialized stage working set. */
static uint32_t stage_fast_limit(const tigris_plan_t *plan,
                                 const tigris_stage_t *stage, uint32_t available)
{
    uint32_t peak = stage->peak_bytes;
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    if (tile.tileable)
        peak = tile.tiled_peak_bytes;
    /* Hand-built and older plans may omit the working-set estimate. */
    return peak != 0u && peak < available ? peak : available;
}

static uint32_t stage_fast_capacity(const tigris_plan_t *plan,
                                   const tigris_stage_t *stage,
                                   const tigris_mem_t *mem)
{
    return stage_fast_limit(plan, stage, mem->fast_size - mem->fast_reserved);
}

/* Drop a dead tensor's buffer. Another tensor may hold the same buffer
 * through an in-place reshape, an op output or a stage input the reshape
 * read; its pointer would read to compaction as a live allocation, so the
 * whole group goes at once. Stage inputs get their slow pointers back after
 * the stage. */
static void release_stage_tensor(const tigris_plan_t *plan,
                                 const tigris_stage_t *stage,
                                 tigris_mem_t *mem, uint16_t tidx)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    uint16_t num_t = plan->header->num_tensors;
    void *dead = mem->tensor_ptrs[tidx];

    if (!dead)
        return;
    mem->tensor_ptrs[tidx] = NULL;
    for (uint16_t q = 0; q < stage->ops_count; q++) {
        const tigris_op_t *other = &plan->ops[sops[q]];
        const uint16_t *oouts = tigris_op_outputs(plan, other);
        for (uint8_t r = 0; r < other->num_outputs; r++) {
            if (oouts[r] < num_t && mem->tensor_ptrs[oouts[r]] == dead)
                mem->tensor_ptrs[oouts[r]] = NULL;
        }
    }
    for (uint16_t q = 0; q < stage->inputs_count; q++) {
        if (sin[q] < num_t && mem->tensor_ptrs[sin[q]] == dead)
            mem->tensor_ptrs[sin[q]] = NULL;
    }
}

static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_normal_impl(
    const tigris_plan_t *plan,
    const tigris_stage_t *stage,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    executor_workspace_impl_t *workspace,
    int                  outputs_in_fast)
{
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    void **saved_slow = workspace->input_ptrs;
    void **kept = workspace->output_ptrs;

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

    /* An output no op of the stage reads dies where it is written, as the
     * compiler's lifetimes have it; held to the end of the stage it takes
     * fast memory the schedule gave to later outputs. */
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            if (outs[k] < num_t && last_use_op[outs[k]] == UINT16_MAX)
                last_use_op[outs[k]] = j;
        }
    }

    /* Mark stage outputs as never-free (they survive the stage) */
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        if (tidx < num_t)
            last_use_op[tidx] = UINT16_MAX;
    }

    /* A reshape group shares one buffer, so every member gets the group's last read. */
    for (uint16_t j = stage->ops_count; j-- > 0; ) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        uint16_t produced;
        uint16_t source;
        uint16_t group;
        if (!is_pure_reinterpretation(plan, op))
            continue;
        produced = tigris_op_outputs(plan, op)[0];
        source = tigris_op_inputs(plan, op)[0];
        if (produced >= num_t || source >= num_t)
            continue;
        if (last_use_op[produced] == UINT16_MAX ||
            last_use_op[source] == UINT16_MAX) {
            group = UINT16_MAX;
        } else {
            group = last_use_op[produced] > last_use_op[source]
                  ? last_use_op[produced] : last_use_op[source];
        }
        last_use_op[produced] = group;
        last_use_op[source] = group;
    }
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        uint16_t produced;
        uint16_t source;
        if (!is_pure_reinterpretation(plan, op))
            continue;
        produced = tigris_op_outputs(plan, op)[0];
        source = tigris_op_inputs(plan, op)[0];
        if (produced < num_t && source < num_t)
            last_use_op[produced] = last_use_op[source];
    }

    /* Load stage inputs; one already in this stage's fast region is read in place. */
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint16_t tidx = sin[i];
        const uint8_t *held = (const uint8_t *)mem->tensor_ptrs[tidx];
        if (held >= mem->fast_base + mem->fast_reserved &&
            held < mem->fast_base + mem->fast_used) {
            saved_slow[i] = NULL;
            continue;
        }
        saved_slow[i] = mem->tensor_ptrs[tidx];

        tigris_mem_error_t merr = tigris_mem_load(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
    }

    /* Execute ops with intra-stage reclamation */
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = sops[j];
        const tigris_op_t *op = &plan->ops[op_idx];

        /* A reshape that moves no byte shares its input's buffer, so there is
         * nothing to allocate and nothing for the kernel to copy. The
         * compiler counts the two as one allocation when it decides whether
         * the stage fits, so not doing this would need more memory than it
         * was promised. */
        const uint16_t *outs = tigris_op_outputs(plan, op);
        if (is_pure_reinterpretation(plan, op)) {
            uint16_t src = tigris_op_inputs(plan, op)[0];
            if (!mem->tensor_ptrs[outs[0]] && mem->tensor_ptrs[src]) {
                mem->tensor_ptrs[outs[0]] = mem->tensor_ptrs[src];
                if (last_use_op[outs[0]] == j)
                    release_stage_tensor(plan, stage, mem, outs[0]);
                continue;
            }
        }

        /* Allocate outputs (with compaction retry on OOM, slow fallback) */
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint16_t tidx = outs[k];
            if (!mem->tensor_ptrs[tidx]) {
                tigris_mem_error_t merr = tigris_mem_alloc_fast(
                    mem, tidx, plan->tensors[tidx].size_bytes);
                if (merr != TIGRIS_MEM_OK) {
                    /* Try compacting fast memory and retry */
                    compact_fast(mem, plan);
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

        int kret = TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx);
        if (kret != 0)
            return TIGRIS_EXEC_ERR_KERNEL;

        /* Free tensors whose last use was this op */
        const uint16_t *ins = tigris_op_inputs(plan, op);
        for (uint8_t k = 0; k < op->num_inputs; k++) {
            if (ins[k] < num_t && last_use_op[ins[k]] == j)
                release_stage_tensor(plan, stage, mem, ins[k]);
        }
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            if (outs[k] < num_t && last_use_op[outs[k]] == j)
                release_stage_tensor(plan, stage, mem, outs[k]);
        }
    }

    /* Spill stage outputs to slow, unless the run keeps them in fast */
    for (uint16_t i = 0; !outputs_in_fast && i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        uintptr_t ptr = (uintptr_t)mem->tensor_ptrs[tidx];
        uintptr_t slow = (uintptr_t)mem->slow_base;
        if (ptr && ptr >= slow && ptr - slow < mem->slow_used)
            continue;
        tigris_mem_error_t merr = tigris_mem_spill(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
    }

    /* Dropping the fast arena invalidates every pointer into it. A tensor
     * whose buffer an in-place reshape shares is not freed by its own last
     * use, and one nothing reads again is never freed at all; either way the
     * stale pointer would look to the next stage's compaction like a live
     * allocation sitting where that stage has already put something else. */
    {
        const uint8_t *low = mem->fast_base + mem->fast_reserved;
        const uint8_t *high = mem->fast_base + mem->fast_size;
        for (uint16_t i = 0; outputs_in_fast && i < stage->outputs_count; i++)
            kept[i] = mem->tensor_ptrs[sout[i]];
        for (uint16_t i = 0; i < mem->num_tensors; i++) {
            const uint8_t *p = (const uint8_t *)mem->tensor_ptrs[i];
            if (p >= low && p < high)
                mem->tensor_ptrs[i] = NULL;
        }
        for (uint16_t i = 0; outputs_in_fast && i < stage->outputs_count; i++)
            mem->tensor_ptrs[sout[i]] = kept[i];
    }

    tigris_mem_reset_fast(mem);

    /* Restore stage input pointers to slow */
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint16_t tidx = sin[i];
        mem->tensor_ptrs[tidx] = saved_slow[i];
    }

    return TIGRIS_EXEC_OK;
}

static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_normal(
    const tigris_plan_t *plan,
    const tigris_stage_t *stage,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    executor_workspace_impl_t *workspace,
    int                  outputs_in_fast)
{
    uint32_t capacity = mem->fast_size;
    mem->fast_size = mem->fast_reserved + stage_fast_capacity(plan, stage, mem);
    tigris_exec_error_t result = exec_stage_normal_impl(
        plan, stage, mem, kernel, user_ctx, stats, workspace, outputs_in_fast);
    mem->fast_size = capacity;
    return result;
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
    int64_t input_extent_wide;
    if (spatial_op_idx >= 0 && tigris_height_upsample_op(plan->ops[spatial_op_idx].op_type)) {
        input_extent_wide = tigris_resize_max_rows(plan, &plan->ops[spatial_op_idx], output_extent);
    } else {
        input_extent_wide = spatial_op_idx >= 0
            ? ((int64_t)output_extent - 1) * stride + effective_kernel
            : output_extent;
    }
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

/* Output rows per band of a height stripe. From schema 11 the record states
 * the band, and a band whose working set exceeds the stage's fast capacity
 * fails closed; older plans take the largest band that fits. */
static tigris_exec_error_t stripe_band_rows(
    const tigris_plan_t *plan, const tigris_stage_t *stage, const tigris_mem_t *mem,
    int spatial_op_idx, int32_t output_extent, int32_t stride, int32_t effective_kernel,
    int32_t *rows)
{
    uint32_t available = stage_fast_capacity(plan, stage, mem);
    uint32_t required;
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS) {
        uint32_t band = tigris_stage_tiles(plan, stage).output_band_extent;
        if (band == 0u || output_extent <= 0 || band > (uint32_t)output_extent ||
            !stage_axis1_fast_bytes(plan, stage, spatial_op_idx, (int32_t)band,
                                    stride, effective_kernel, &required) ||
            required > available)
            return TIGRIS_EXEC_ERR_TILE;
        *rows = (int32_t)band;
        return TIGRIS_EXEC_OK;
    }
    int32_t low = 1;
    int32_t high = output_extent;
    int32_t best = 0;
    while (low <= high) {
        int32_t candidate = low + (high - low) / 2;
        if (!stage_axis1_fast_bytes(
                plan, stage, spatial_op_idx, candidate,
                stride, effective_kernel, &required))
            return TIGRIS_EXEC_ERR_TILE;
        if (required <= available) {
            best = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    if (best == 0)
        return TIGRIS_EXEC_ERR_TILE;
    *rows = best;
    return TIGRIS_EXEC_OK;
}

static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    const uint16_t           *last_consumer,
    uint16_t                  current_stage,
    executor_workspace_impl_t *workspace,
    tigris_exec_stats_t *stats)
{
    /* 1. Pre-allocate full output tensors in slow (or reuse input for in-place) */
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    void **out_slow_bases = workspace->output_ptrs;
    int in_place = 0;  /* set if output reuses input's slow memory */

    /* Spatial op + image dims, computed up front: the in-place halo guard below
     * and the tile geometry both need them. */
    int sp_op_idx = tigris_stage_spatial_op(plan, stage);
    const tigris_spatial_attrs_t *sp_attrs =
        (sp_op_idx >= 0) ? &plan->ops[sp_op_idx].spatial : NULL;

    /* Fail-closed: ConvTranspose expands its output, so the 1D shrink
     * back-computation (in = out*stride - pad) below is invalid for it. Only
     * the 2D executor (exec_stage_tiled_2d) inverts its rectangle. A plan that
     * routes a ConvTranspose stage here (a 1D axis-1 tile plan) is malformed. */
    if (sp_op_idx >= 0 &&
        plan->ops[sp_op_idx].op_type == TIGRIS_OP_CONV_TRANSPOSE)
        return TIGRIS_EXEC_ERR_TILE;

    /* The band runs over the first input that has rows to band. A rank-4
     * input one row high is a per-channel operand broadcast over every band
     * (see the LOAD step), so it does not set the extent while another does. */
    uint16_t band_input = sin[0];
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        const tigris_tensor_t *candidate = &plan->tensors[sin[i]];
        if (candidate->ndim != 4u ||
            tigris_tensor_shape(plan, candidate)[1] != 1) {
            band_input = sin[i];
            break;
        }
    }
    uint8_t rank = plan->tensors[band_input].ndim;
    const int32_t *in_shape = tigris_tensor_shape(plan, &plan->tensors[band_input]);
    int32_t full_in_h = in_shape[1];
    int32_t full_in_w = rank == 4 ? in_shape[2] : 1;
    const int32_t *out_shape = tigris_tensor_shape(plan, &plan->tensors[sout[0]]);
    int32_t full_out_h = out_shape[1];
    int32_t full_out_w = rank == 4 ? out_shape[2] : 1;
    int32_t orig_pad_top = sp_attrs ? sp_attrs->pad_top : 0;
    int32_t stride = sp_attrs ? sp_attrs->stride_h : 1;
    int32_t dh = (sp_attrs && sp_attrs->dilation_h) ? sp_attrs->dilation_h : 1;
    int32_t eff_kh = sp_attrs ? ((sp_attrs->kernel_h - 1) * dh + 1) : 1;
    int upsamples = (sp_op_idx >= 0) &&
        tigris_height_upsample_op(plan->ops[sp_op_idx].op_type);
    if (upsamples) {
        if ((sp_attrs->stride_h != 0u && (int64_t)full_in_h * sp_attrs->stride_h != full_out_h) ||
            (sp_attrs->stride_w != 0u && (int64_t)full_in_w * sp_attrs->stride_w != full_out_w))
            return TIGRIS_EXEC_ERR_TILE;
        orig_pad_top = 0;
        stride = 0;
        eff_kh = 1;
    }

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
    int32_t out_tile_h = 0;
    tigris_exec_error_t band_error = stripe_band_rows(
        plan, stage, mem, sp_op_idx, full_out_h, stride, eff_kh, &out_tile_h);
    if (band_error != TIGRIS_EXEC_OK)
        return band_error;

    int32_t num_tiles = 1 + (full_out_h - 1) / out_tile_h;

    /* Rows of each stage output written so far, so the tile loop can prove it
     * leaves no gap. */
    int32_t *out_covered = workspace->out_covered;
    if (stage->outputs_count > workspace->output_capacity)
        return TIGRIS_EXEC_ERR_WORKSPACE;
    for (uint16_t i = 0; i < stage->outputs_count; i++)
        out_covered[i] = 0;

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

        if (upsamples) {
            int source_rows = tigris_resize_source_rows(
                plan, &plan->ops[sp_op_idx], out_start, out_end, &in_start);
            if (source_rows <= 0) return TIGRIS_EXEC_ERR_TILE;
            in_end = in_start + source_rows;
        } else if (sp_attrs) {
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

        if (stats) stats->total_tiles++;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(tile), 0u, 0u, 0u,
            in_start, in_end, -1, -1);
        int32_t tile_in_h = in_end - in_start;

        /* c. Activate tiling. The per-op tile context (in/out h+w, pad) is set
         * inside the execute loop below: ops before the spatial op run at the
         * input tile dims, the spatial op reduces them, ops after run at the
         * output dims. */
        mem->tile.active = 1;
        mem->tile.out_row_origin = out_start;
        mem->tile.in_row_origin = in_start;

        /* d. LOAD: load tile for each stage input. A rank-4 input one row
         * high is a per-channel operand broadcast over every row, so each band
         * takes it whole; the loader admits one only in this executor. */
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            uint16_t tidx = sin[i];
            const tigris_tensor_t *in_tensor = &plan->tensors[tidx];
            int broadcast = in_tensor->ndim == 4u && full_in_h > 1 &&
                            tigris_tensor_shape(plan, in_tensor)[1] == 1;
            mem->tensor_ptrs[tidx] = in_slow_bases[i]; /* restore before load */
            tigris_mem_error_t merr = tigris_mem_load_tile(
                mem, plan, tidx, broadcast ? 0 : in_start,
                broadcast ? 1 : in_end);
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
        /* Global rows of the tensor flowing through the stage: the input range
         * ahead of the spatial op, its output range after. */
        int32_t cur_gs = in_start;
        int32_t cur_ge = in_end;
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx = sops[j];
            const tigris_op_t *op = &plan->ops[op_idx];
            int is_spatial = ((int)op_idx == sp_op_idx);

            if (is_spatial) {
                if (eff_pad_top < 0 || eff_pad_top > UINT16_MAX ||
                    eff_pad_bottom < 0 || eff_pad_bottom > UINT16_MAX)
                    return TIGRIS_EXEC_ERR_TILE;
                mem->tile.in_h       = cur_h;          /* = tile_in_h */
                mem->tile.out_h      = tile_out_h;
                mem->tile.in_w       = full_in_w;
                mem->tile.out_w      = full_out_w;
                mem->tile.pad_top    = (uint16_t)eff_pad_top;
                mem->tile.pad_bottom = (uint16_t)eff_pad_bottom;
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

            int kret = TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx);
            if (kret != 0)
                return TIGRIS_EXEC_ERR_KERNEL;

            if (is_spatial) {
                cur_h = tile_out_h;
                cur_w = full_out_w;
                cur_gs = out_start;
                cur_ge = out_end;
            }
            /* Record the rows this op's output buffer now holds, so the spill
             * places a stage output by its own range. */
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                workspace->out_gs[outs[k]] = cur_gs;
                workspace->out_ge[outs[k]] = cur_ge;
            }
        }

        /* f. SPILL: spill tile for each stage output, at the row range the op
         * that produced it wrote. An output produced ahead of the spatial op
         * carries the tile's INPUT range, not the stage's output range. */
        for (uint16_t i = 0; i < stage->outputs_count; i++) {
            uint16_t tidx = sout[i];
            int32_t gs = workspace->out_gs[tidx];
            int32_t ge = workspace->out_ge[tidx];
            tigris_mem_error_t merr;
            if (gs > out_covered[i])
                return TIGRIS_EXEC_ERR_TILE;
            merr = tigris_mem_spill_tile(
                mem, plan, tidx, out_slow_bases[i], gs, ge);
            if (merr != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
            if (ge > out_covered[i])
                out_covered[i] = ge;
        }

        /* g. Reset fast arena for next tile */
        tigris_mem_reset_fast(mem);
    }

    /* 4. Verify every output row was covered */
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        if (out_covered[i] != tensor_row_count(plan, sout[i]))
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

/* 2D (HW axis) standalone tiled execution */

/**
 * Aligned fast-arena byte cost of one packed [N, h_extent, w_extent, C] tile
 * of a rank-4 NHWC tensor, matching tigris_mem_alloc_fast's per-allocation
 * alignment. Returns 0 on malformed metadata.
 */
static int tile2d_allocation_bytes(
    const tigris_plan_t *plan, uint16_t tensor_idx,
    int32_t h_extent, int32_t w_extent, uint32_t *bytes)
{
    const tigris_tensor_t *tensor = &plan->tensors[tensor_idx];
    const int32_t *shape = tigris_tensor_shape(plan, tensor);
    if (!bytes || tensor->ndim != 4 ||
        shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 ||
        h_extent <= 0 || w_extent <= 0)
        return 0;

    uint64_t elems = (uint64_t)shape[0] * (uint32_t)shape[1] *
                     (uint32_t)shape[2] * (uint32_t)shape[3];
    if (elems == 0)
        return 0;
    uint32_t elem_size = tensor->size_bytes / (uint32_t)elems;

    uint64_t raw = (uint64_t)shape[0] * (uint32_t)h_extent *
                   (uint32_t)w_extent * (uint32_t)shape[3] * elem_size;
    uint64_t aligned =
        (raw + (TIGRIS_TENSOR_ALIGN - 1u)) &
        ~(uint64_t)(TIGRIS_TENSOR_ALIGN - 1u);
    if (aligned > UINT32_MAX)
        return 0;
    *bytes = (uint32_t)aligned;
    return 1;
}

/**
 * Largest fast working set a single 2D (HW) tile can require: one packed input
 * tile per stage input plus every op's packed output tile, all resident at
 * once (the arena is reset only between tiles). Extents use the interior
 * (full out_tile_h x out_tile_w) tile, which back-computes the widest input
 * halo, so the total upper-bounds every partial edge and corner tile.
 */
static int stage_2d_fast_bytes(
    const tigris_plan_t *plan, const tigris_stage_t *stage,
    int sp_op_idx, int32_t out_tile_h, int32_t out_tile_w,
    int32_t stride_h, int32_t stride_w, int32_t eff_kh, int32_t eff_kw,
    int32_t full_in_h, int32_t full_in_w, uint32_t *required)
{
    const uint16_t *stage_inputs = tigris_stage_inputs(plan, stage);
    const uint16_t *stage_ops = tigris_stage_ops(plan, stage);

    int is_conv_transpose = sp_op_idx >= 0 &&
        plan->ops[sp_op_idx].op_type == TIGRIS_OP_CONV_TRANSPOSE;
    int32_t in_tile_h;
    int32_t in_tile_w;
    if (is_conv_transpose) {
        /* ConvTranspose inverts to a SMALLER input tile. Proven upper bound on
         * the widest realized input span: ceil((out_tile + eff_k)/stride) plus
         * 2 rows/cols of slack. Keeps this pre-flight check a safe upper bound
         * over every edge and corner tile. */
        in_tile_h = (out_tile_h + eff_kh + stride_h - 1) / stride_h + 2;
        in_tile_w = (out_tile_w + eff_kw + stride_w - 1) / stride_w + 2;
    } else {
        in_tile_h = sp_op_idx >= 0
            ? (out_tile_h - 1) * stride_h + eff_kh : out_tile_h;
        in_tile_w = sp_op_idx >= 0
            ? (out_tile_w - 1) * stride_w + eff_kw : out_tile_w;
    }
    if (in_tile_h <= 0 || in_tile_w <= 0)
        return 0;
    if (in_tile_h > full_in_h) in_tile_h = full_in_h;
    if (in_tile_w > full_in_w) in_tile_w = full_in_w;

    uint64_t total = 0;
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint32_t b;
        if (!tile2d_allocation_bytes(plan, stage_inputs[i],
                                     in_tile_h, in_tile_w, &b))
            return 0;
        total += b;
    }

    int32_t cur_h = in_tile_h;
    int32_t cur_w = in_tile_w;
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = stage_ops[j];
        const tigris_op_t *op = &plan->ops[op_idx];
        int is_spatial = ((int)op_idx == sp_op_idx);
        int32_t ah = is_spatial ? out_tile_h : cur_h;
        int32_t aw = is_spatial ? out_tile_w : cur_w;
        const uint16_t *outputs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint32_t b;
            if (!tile2d_allocation_bytes(plan, outputs[k], ah, aw, &b))
                return 0;
            total += b;
        }
        if (is_spatial) {
            cur_h = out_tile_h;
            cur_w = out_tile_w;
        }
    }

    if (total > UINT32_MAX)
        return 0;
    *required = (uint32_t)total;
    return 1;
}

/**
 * Execute a standalone rank-4 stage tiled along both height and width.
 *
 * Drives an (tile_row, tile_col) grid over the output, back-computing each
 * tile's input rectangle and effective four-sided padding, then reusing the
 * strided 2D load/spill primitives and the width-aware kernels. The
 * full output tensor is assembled in slow, byte-identical to the whole-op
 * result. Marked TIGRIS_NOINLINE to keep the inlined dispatcher within its
 * per-frame stack budget.
 */
static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled_2d(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    executor_workspace_impl_t *workspace,
    tigris_exec_stats_t *stats)
{
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    const uint16_t *sin  = tigris_stage_inputs(plan, stage);
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    void **out_slow_bases = workspace->output_ptrs;
    void **in_slow_bases  = workspace->input_ptrs;

    if (stage->inputs_count == 0 || stage->outputs_count == 0)
        return TIGRIS_EXEC_ERR_TILE;

    /* 2D tiling is NHWC-only; the width axis has no meaning at rank 3. */
    if (plan->tensors[sin[0]].ndim != 4 || plan->tensors[sout[0]].ndim != 4)
        return TIGRIS_EXEC_ERR_TILE;

    /* Spatial op + image geometry (height and width). */
    int sp_op_idx = tigris_stage_spatial_op(plan, stage);
    const tigris_spatial_attrs_t *sp =
        (sp_op_idx >= 0) ? &plan->ops[sp_op_idx].spatial : NULL;

    const int32_t *in_shape = tigris_tensor_shape(plan, &plan->tensors[sin[0]]);
    const int32_t *out_shape = tigris_tensor_shape(plan, &plan->tensors[sout[0]]);
    int32_t IH = in_shape[1];
    int32_t IW = in_shape[2];
    int32_t OH = out_shape[1];
    int32_t OW = out_shape[2];

    int32_t SH = sp ? sp->stride_h : 1;
    int32_t SW = sp ? sp->stride_w : 1;
    int32_t PT = sp ? sp->pad_top : 0;
    int32_t PL = sp ? sp->pad_left : 0;
    int32_t dh = (sp && sp->dilation_h) ? sp->dilation_h : 1;
    int32_t dw = (sp && sp->dilation_w) ? sp->dilation_w : 1;
    int32_t eff_kh = sp ? ((sp->kernel_h - 1) * dh + 1) : 1;
    int32_t eff_kw = sp ? ((sp->kernel_w - 1) * dw + 1) : 1;

    tigris_tile_plan_v10_t tp = tigris_stage_tiles(plan, stage);
    int32_t th = (int32_t)tp.output_band_extent;
    int32_t tw = (int32_t)tp.output_width_extent;
    if (th <= 0 || tw <= 0 || OH <= 0 || OW <= 0)
        return TIGRIS_EXEC_ERR_TILE;

    /* Shape validation (fail-closed): the emitted (th, tw) working set must
     * fit the real fast arena. The 2D solver models only a proportional
     * activation peak and ignores resident scratch/weight bytes, so the
     * runtime is the authority on whether the shape actually fits. */
    int32_t val_out_h = th < OH ? th : OH;
    int32_t val_out_w = tw < OW ? tw : OW;
    uint32_t required = 0;
    if (!stage_2d_fast_bytes(plan, stage, sp_op_idx, val_out_h, val_out_w,
                             SH, SW, eff_kh, eff_kw, IH, IW, &required))
        return TIGRIS_EXEC_ERR_TILE;
    if (mem->fast_reserved > mem->fast_size ||
        required > mem->fast_size - mem->fast_reserved)
        return TIGRIS_EXEC_ERR_TILE;

    /* Pre-allocate full output tensors in slow. */
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        tigris_mem_error_t merr = tigris_mem_alloc_slow(
            mem, tidx, plan->tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        out_slow_bases[i] = mem->tensor_ptrs[tidx];
    }

    /* Save slow ptrs for stage inputs. */
    for (uint16_t i = 0; i < stage->inputs_count; i++)
        in_slow_bases[i] = mem->tensor_ptrs[sin[i]];

    int32_t num_tiles_h = 1 + (OH - 1) / th;
    int32_t num_tiles_w = 1 + (OW - 1) / tw;

    /* Tile grid: iterate output (row, col) blocks. */
    for (int32_t tr = 0; tr < num_tiles_h; tr++) {
        for (int32_t tc = 0; tc < num_tiles_w; tc++) {
            int32_t oh0 = tr * th;
            int32_t oh1 = oh0 + th; if (oh1 > OH) oh1 = OH;
            int32_t ow0 = tc * tw;
            int32_t ow1 = ow0 + tw; if (ow1 > OW) ow1 = OW;
            int32_t tile_out_h = oh1 - oh0;
            int32_t tile_out_w = ow1 - ow0;

            /* Back-compute the input rectangle and effective four-sided pads.
             * pad_bottom/pad_right are set for completeness; the kernels read
             * the bottom/right edges implicitly via the ih<IH / iw<IW bound
             * checks, exactly as the height path leaves pad_bottom implicit. */
            int32_t ih0, ih1, iw0, iw1;
            int32_t pt = 0, pb = 0, pl = 0, pr = 0;
            if (sp &&
                plan->ops[sp_op_idx].op_type == TIGRIS_OP_CONV_TRANSPOSE) {
                /* ConvTranspose expands: from an output tile back-compute the
                 * SMALLER input rectangle via the gather relation
                 * ih = floor((oh + PT - kh)/SH), and fold the tile's global
                 * offset into effective pads so the kernel runs in local coords
                 * (out_row_start/in_row_start stay 0, set above). floor_div is
                 * mandatory: the numerators go negative near the top/left edge.
                 * pad_bottom/pad_right are implicit via the ih<IH / iw<IW
                 * clamps, matching the conv arm below. */
                ih0 = floor_div(oh0 + PT - (eff_kh - 1), SH);
                ih1 = floor_div(oh1 - 1 + PT, SH) + 1;
                iw0 = floor_div(ow0 + PL - (eff_kw - 1), SW);
                iw1 = floor_div(ow1 - 1 + PL, SW) + 1;
                if (ih0 < 0) ih0 = 0;
                if (ih1 > IH) ih1 = IH;
                if (iw0 < 0) iw0 = 0;
                if (iw1 > IW) iw1 = IW;
                pt = oh0 + PT - ih0 * SH;   /* PT_eff, in [0, SH+eff_kh-2] */
                pl = ow0 + PL - iw0 * SW;   /* PL_eff */
                pb = 0;
                pr = 0;
            } else if (sp) {
                ih0 = oh0 * SH - PT;
                ih1 = (oh1 - 1) * SH - PT + eff_kh;
                iw0 = ow0 * SW - PL;
                iw1 = (ow1 - 1) * SW - PL + eff_kw;
                if (ih0 < 0) { pt = -ih0; ih0 = 0; }
                if (ih1 > IH) { pb = ih1 - IH; ih1 = IH; }
                if (iw0 < 0) { pl = -iw0; iw0 = 0; }
                if (iw1 > IW) { pr = iw1 - IW; iw1 = IW; }
            } else {
                ih0 = oh0; ih1 = oh1; iw0 = ow0; iw1 = ow1;
            }
            if (ih1 <= ih0 || iw1 <= iw0)
                return TIGRIS_EXEC_ERR_TILE;
            if (stats) stats->total_tiles++;
            TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
                TIGRIS_TRACE_NONE_U16, (uint16_t)(tr * num_tiles_w + tc), 0u, 0u, 0u,
                ih0, ih1, iw0, iw1);
            int32_t tile_in_h = ih1 - ih0;
            int32_t tile_in_w = iw1 - iw0;

            /* Activate a 2D (HW) tile. */
            mem->tile.active = 1;
            mem->tile.width_tiled = 1;
            mem->tile.out_row_start = 0;
            mem->tile.in_row_start = 0;

            /* LOAD: pack the input sub-rectangle for each stage input. */
            for (uint16_t i = 0; i < stage->inputs_count; i++) {
                uint16_t tidx = sin[i];
                mem->tensor_ptrs[tidx] = in_slow_bases[i]; /* restore before load */
                tigris_mem_error_t merr = tigris_mem_load_tile_2d(
                    mem, plan, tidx, ih0, ih1, iw0, iw1);
                if (merr != TIGRIS_MEM_OK)
                    return TIGRIS_EXEC_ERR_MEM;
            }

            /* EXECUTE: track the tile's height/width through the op sequence.
             * It starts at the input tile size; the single spatial op reduces
             * it to the output tile size; pointwise ops preserve it. */
            int32_t cur_h = tile_in_h;
            int32_t cur_w = tile_in_w;
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                uint16_t op_idx = sops[j];
                const tigris_op_t *op = &plan->ops[op_idx];
                int is_spatial = ((int)op_idx == sp_op_idx);

                if (is_spatial) {
                    mem->tile.in_h       = cur_h;
                    mem->tile.out_h      = tile_out_h;
                    mem->tile.in_w       = cur_w;
                    mem->tile.out_w      = tile_out_w;
                    mem->tile.pad_top    = (uint16_t)pt;
                    mem->tile.pad_bottom = (uint16_t)pb;
                    mem->tile.pad_left   = (uint16_t)pl;
                    mem->tile.pad_right  = (uint16_t)pr;
                } else {
                    /* Pointwise op: height/width preserved, no padding. */
                    mem->tile.in_h       = cur_h;
                    mem->tile.out_h      = cur_h;
                    mem->tile.in_w       = cur_w;
                    mem->tile.out_w      = cur_w;
                    mem->tile.pad_top    = 0;
                    mem->tile.pad_bottom = 0;
                    mem->tile.pad_left   = 0;
                    mem->tile.pad_right  = 0;
                }

                int32_t alloc_h = is_spatial ? tile_out_h : cur_h;
                int32_t alloc_w = is_spatial ? tile_out_w : cur_w;

                /* Alloc tile-sized outputs in fast (NHWC, alloc_h x alloc_w). */
                const uint16_t *outs = tigris_op_outputs(plan, op);
                for (uint8_t k = 0; k < op->num_outputs; k++) {
                    uint16_t tidx = outs[k];
                    const tigris_tensor_t *t = &plan->tensors[tidx];
                    const int32_t *shape = tigris_tensor_shape(plan, t);
                    uint32_t full_w = t->ndim == 4 ? (uint32_t)shape[2] : 1u;
                    uint32_t channels = (uint32_t)shape[t->ndim - 1];
                    uint32_t elem_size = t->size_bytes /
                        ((uint32_t)shape[0] * (uint32_t)shape[1] *
                         full_w * channels);
                    uint32_t tile_bytes = (uint32_t)shape[0] * (uint32_t)alloc_h *
                        (uint32_t)alloc_w * channels * elem_size;
                    tigris_mem_error_t merr = tigris_mem_alloc_fast(
                        mem, tidx, tile_bytes);
                    if (merr != TIGRIS_MEM_OK)
                        return TIGRIS_EXEC_ERR_MEM;
                }

                int kret = TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx);
                if (kret != 0)
                    return TIGRIS_EXEC_ERR_KERNEL;

                if (is_spatial) {
                    cur_h = tile_out_h;
                    cur_w = tile_out_w;
                }
            }

            /* SPILL: write each packed output core back to its slow rectangle. */
            for (uint16_t i = 0; i < stage->outputs_count; i++) {
                uint16_t tidx = sout[i];
                tigris_mem_error_t merr = tigris_mem_spill_tile_2d(
                    mem, plan, tidx, out_slow_bases[i], oh0, oh1, ow0, ow1);
                if (merr != TIGRIS_MEM_OK)
                    return TIGRIS_EXEC_ERR_MEM;
            }

            /* Reset fast arena for next tile. */
            tigris_mem_reset_fast(mem);
        }
    }

    /* Clear tile context, restore input ptrs to slow, and NULL intermediate op
     * outputs so compact_fast won't find stale fast-arena pointers with
     * full-size metadata. Stage outputs already point to slow via spill. */
    memset(&mem->tile, 0, sizeof(mem->tile));
    for (uint16_t i = 0; i < stage->inputs_count; i++)
        mem->tensor_ptrs[sin[i]] = in_slow_bases[i];

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        const uint16_t *outs = tigris_op_outputs(plan, op);
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint16_t tidx = outs[k];
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

/* Adds one tensor as the arena allocates it, aligned; saturates on overflow. */
static uint32_t add_allocation_bytes(uint32_t total, uint32_t size)
{
    uint32_t aligned = 0u;
    uint32_t sum = UINT32_MAX;
    if (align_tensor_size(size, &aligned) && aligned <= UINT32_MAX - total)
        sum = total + aligned;
    return sum;
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

/* Input-driven tiled execution */

/**
 * Execute a standalone global reduction by walking bands of its input.
 *
 * Each band loads a range of input rows into the fast arena and calls the
 * kernel with a running accumulator, one value per (batch, channel), held
 * below the arena's reset point so it survives between bands. The kernel
 * writes the output tensor on the last band only, at which point the whole
 * reduction has been folded in, so the result is the byte the untiled kernel
 * produces. Marked TIGRIS_NOINLINE to keep the inlined dispatcher within its
 * per-frame stack budget.
 */
static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled_by_input(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    tigris_exec_stats_t *stats)
{
    uint16_t in_idx = tigris_stage_inputs(plan, stage)[0];
    uint16_t out_idx = tigris_stage_outputs(plan, stage)[0];
    uint16_t op_idx = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[op_idx];

    const int32_t *in_shape =
        tigris_tensor_shape(plan, &plan->tensors[in_idx]);
    int32_t full_in_h = in_shape[1];
    int32_t full_in_w = in_shape[2];

    void *in_slow_base = mem->tensor_ptrs[in_idx];
    if (!in_slow_base)
        return TIGRIS_EXEC_ERR_MEM;

    /* The assembled output lives in slow and is written on the last band. */
    if (tigris_mem_alloc_slow(mem, out_idx,
                              plan->tensors[out_idx].size_bytes)
            != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    void *out_slow_base = mem->tensor_ptrs[out_idx];

    /* One partial per (batch, channel): int32 for the int8 kernels, float
     * for the float32 ones, four bytes either way. Raising fast_reserved
     * over it keeps the per-band reset from reclaiming it. */
    uint64_t acc_elems = (uint64_t)in_shape[0] * (uint32_t)in_shape[3];
    uint32_t acc_bytes;
    if (acc_elems == 0 || acc_elems > UINT32_MAX / 4u ||
        !align_tensor_size((uint32_t)acc_elems * 4u, &acc_bytes))
        return TIGRIS_EXEC_ERR_TILE;
    if (mem->fast_used > mem->fast_size ||
        acc_bytes > mem->fast_size - mem->fast_used)
        return TIGRIS_EXEC_ERR_MEM;

    const uint32_t entry_reserved = mem->fast_reserved;
    uint32_t available = stage_fast_capacity(plan, stage, mem);
    if (acc_bytes > available)
        return TIGRIS_EXEC_ERR_MEM;
    available -= acc_bytes;
    void *acc = mem->fast_base + mem->fast_used;
    mem->fast_used += acc_bytes;
    tigris_mem_note_fast_peak(mem);
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_ALLOC, TIGRIS_TRACE_POOL_FAST,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16, mem->fast_used - acc_bytes,
        0u, acc_bytes, -1, -1, -1, -1);
    mem->fast_reserved = mem->fast_used;

    /* Largest band whose rows plus the output tensor fit what is left. */
    uint32_t out_bytes;
    int32_t band = 0;
    if (tile2d_allocation_bytes(plan, out_idx, 1, 1, &out_bytes)) {
        int32_t low = 1;
        int32_t high = full_in_h;
        while (low <= high) {
            int32_t candidate = low + (high - low) / 2;
            uint32_t in_bytes;
            if (!tile2d_allocation_bytes(plan, in_idx, candidate,
                                         full_in_w, &in_bytes)) {
                band = 0;
                break;
            }
            if ((uint64_t)in_bytes + out_bytes <= available) {
                band = candidate;
                low = candidate + 1;
            } else {
                high = candidate - 1;
            }
        }
    }
    if (band == 0) {
        mem->fast_reserved = entry_reserved;
        tigris_mem_reset_fast(mem);
        return TIGRIS_EXEC_ERR_TILE;
    }

    tigris_exec_error_t result = TIGRIS_EXEC_OK;
    for (int32_t start = 0; start < full_in_h; start += band) {
        int32_t end = full_in_h - start < band ? full_in_h : start + band;

        if (stats) stats->total_tiles++;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(start / band), 0u, 0u, 0u,
            start, end, -1, -1);
        mem->tile.active = 1;
        mem->tile.in_h = end - start;
        mem->tile.out_h = 1;
        mem->tile.in_w = full_in_w;
        mem->tile.out_w = 1;
        mem->tile.reduce_acc = acc;
        mem->tile.reduce_first = (start == 0) ? 1u : 0u;
        mem->tile.reduce_last = (end == full_in_h) ? 1u : 0u;

        mem->tensor_ptrs[in_idx] = in_slow_base;
        if (tigris_mem_load_tile(mem, plan, in_idx, start, end)
                != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }
        if (tigris_mem_alloc_fast(mem, out_idx,
                                  plan->tensors[out_idx].size_bytes)
                != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }

        if (TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx) != 0) {
            result = TIGRIS_EXEC_ERR_KERNEL;
            break;
        }

        if (mem->tile.reduce_last &&
            tigris_mem_spill_tile(mem, plan, out_idx, out_slow_base, 0, 1)
                != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }

        tigris_mem_reset_fast(mem);
    }

    memset(&mem->tile, 0, sizeof(mem->tile));
    mem->fast_reserved = entry_reserved;
    tigris_mem_reset_fast(mem);
    mem->tensor_ptrs[in_idx] = in_slow_base;
    mem->tensor_ptrs[out_idx] = out_slow_base;
    return result;
}

/**
 * Execute a standalone layout conversion a band at a time.
 *
 * The band runs along the longer of the two permuted axes, so the fast arena
 * holds one slice of each side and never a whole plane. That axis is a row
 * range on one side and a column range on the other, so exactly one of the
 * two transfers is strided: gathering the input columns when the input is the
 * short-rowed side, scattering the output columns when it is not.
 * Marked TIGRIS_NOINLINE to keep the inlined dispatcher within its per-frame
 * stack budget.
 */
static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled_transpose(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    tigris_exec_stats_t *stats)
{
    uint16_t in_idx = tigris_stage_inputs(plan, stage)[0];
    uint16_t out_idx = tigris_stage_outputs(plan, stage)[0];
    uint16_t op_idx = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[op_idx];

    const tigris_tensor_t *in = &plan->tensors[in_idx];
    const tigris_tensor_t *out = &plan->tensors[out_idx];
    int32_t batch = 0;  /* the axes ahead of the swapped pair */
    int32_t rows = 0;   /* becomes the output's column count */
    int32_t cols = 0;   /* becomes the output's row count */
    if (!tigris_transpose_extents(plan, op_idx, in_idx, &batch, &rows, &cols))
        return TIGRIS_EXEC_ERR_TILE;

    uint64_t positions = (uint64_t)batch * (uint32_t)rows * (uint32_t)cols;
    if (positions == 0u || in->size_bytes % positions != 0u)
        return TIGRIS_EXEC_ERR_TILE;
    /* What one position of the transposed matrix holds: the dtype, times the
     * axes behind the swapped pair, which travel with it. The kernel works in
     * dtype units, so it is told the block separately. */
    uint32_t elem_size = (uint32_t)(in->size_bytes / positions);
    const int32_t *full_shape = tigris_tensor_shape(plan, in);
    uint64_t total = 1;
    for (uint8_t axis = 0; axis < in->ndim; axis++)
        total *= (uint32_t)full_shape[axis];
    if (total == 0u || in->size_bytes % total != 0u)
        return TIGRIS_EXEC_ERR_TILE;
    uint32_t dtype_size = (uint32_t)(in->size_bytes / total);
    if (dtype_size == 0u || elem_size % dtype_size != 0u)
        return TIGRIS_EXEC_ERR_TILE;
    int32_t block = (int32_t)(elem_size / dtype_size);

    /* Band the longer axis: it gives the most bands and the smallest slice,
     * since a slice holds batch * (the other axis) * band elements. Which one
     * that is belongs to whoever sized the band, so a plan that states it is
     * obeyed and only an older one is re-derived here. */
    tigris_tile_plan_v10_t conversion_plan = tigris_stage_tiles(plan, stage);
    int band_on_cols;
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_BAND_ORIENTATION &&
        conversion_plan.tileable)
        band_on_cols = (conversion_plan.flags &
                        TIGRIS_TILE_FLAG_BAND_ON_COLUMNS) != 0u;
    else
        band_on_cols = cols >= rows;
    int32_t banded = band_on_cols ? cols : rows;
    int32_t other = band_on_cols ? rows : cols;

    void *in_slow_base = mem->tensor_ptrs[in_idx];
    if (!in_slow_base)
        return TIGRIS_EXEC_ERR_MEM;

    if (tigris_mem_alloc_slow(mem, out_idx, out->size_bytes) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    void *out_slow_base = mem->tensor_ptrs[out_idx];

    /* Largest band of output rows whose gathered input rectangle and packed
     * output tile fit together. Both hold the same element count, aligned
     * separately because they are two arena allocations. */
    uint32_t available = stage_fast_capacity(plan, stage, mem);
    int32_t band = 0;
    int32_t low = 1;
    int32_t high = banded;
    while (low <= high) {
        int32_t candidate = low + (high - low) / 2;
        uint64_t tile_elems =
            (uint64_t)batch * (uint32_t)other * (uint32_t)candidate;
        if (tile_elems > UINT32_MAX / elem_size) {
            high = candidate - 1;
            continue;
        }
        uint32_t one;
        if (!align_tensor_size((uint32_t)tile_elems * elem_size, &one))
            return TIGRIS_EXEC_ERR_TILE;
        if ((uint64_t)one * 2u <= available) {
            band = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    if (band == 0)
        return TIGRIS_EXEC_ERR_TILE;

    /* Both sides are the same [batch, rows, cols] matrix in memory whatever
     * the tensor's rank, so the two transfers are expressed on it directly
     * rather than through the rank-4 rectangle helpers, which slice the width
     * axis and cannot express a range of channels. */
    tigris_exec_error_t result = TIGRIS_EXEC_OK;
    for (int32_t start = 0; start < banded; start += band) {
        int32_t end = banded - start < band ? banded : start + band;
        int32_t width = end - start;
        uint32_t tile_bytes = (uint32_t)((uint64_t)batch * (uint32_t)other *
                                         (uint32_t)width * elem_size);
        uint32_t strip = (uint32_t)width * elem_size;

        if (stats) stats->total_tiles++;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(start / band), 0u, 0u, 0u,
            band_on_cols ? 0 : start, band_on_cols ? rows : end, band_on_cols ? start : 0, band_on_cols ? end : cols);
        mem->tile.active = 1;
        mem->tile.transposed_tile = 1;
        mem->tile.transpose_outer = batch;
        mem->tile.transpose_block = block;
        mem->tile.in_h = band_on_cols ? other : width;
        mem->tile.in_w = band_on_cols ? width : other;
        mem->tile.out_h = band_on_cols ? width : other;
        mem->tile.out_w = band_on_cols ? other : width;

        /* LOAD: a column range when the band runs along the input's columns,
         * a contiguous row range otherwise. */
        const uint8_t *src = (const uint8_t *)in_slow_base;
        if (tigris_mem_alloc_fast(mem, in_idx, tile_bytes) != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }
        uint8_t *in_tile = (uint8_t *)mem->tensor_ptrs[in_idx];
        if (band_on_cols) {
            for (int32_t n = 0; n < batch; n++) {
                for (int32_t r = 0; r < other; r++) {
                    memcpy(in_tile + ((size_t)n * (size_t)other + (size_t)r) *
                                         strip,
                           src + (((size_t)n * (size_t)other + (size_t)r) *
                                      (size_t)banded + (size_t)start) *
                                     elem_size,
                           strip);
                }
            }
        } else {
            size_t rows_bytes = (size_t)width * (size_t)other * elem_size;
            for (int32_t n = 0; n < batch; n++) {
                memcpy(in_tile + (size_t)n * rows_bytes,
                       src + ((size_t)n * (size_t)banded + (size_t)start) *
                                 (size_t)other * elem_size,
                       rows_bytes);
            }
        }

        if (stats) stats->loads_bytes += tile_bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_LOAD, TIGRIS_TRACE_POOL_FAST,
            in_idx, TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(in_tile, mem->fast_base),
            TIGRIS_TRACE_OFFSET(src, mem->slow_base) +
                (uint32_t)start * (band_on_cols ? 1u : (uint32_t)other) * elem_size,
            tile_bytes, band_on_cols ? 0 : start, band_on_cols ? rows : end,
            band_on_cols ? start : 0, band_on_cols ? end : cols);
        if (tigris_mem_alloc_fast(mem, out_idx, tile_bytes) != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }

        if (TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx) != 0) {
            result = TIGRIS_EXEC_ERR_KERNEL;
            break;
        }

        /* SPILL: the mirror of the load, on the other side of the transpose. */
        const uint8_t *out_tile = (const uint8_t *)mem->tensor_ptrs[out_idx];
        uint8_t *dst = (uint8_t *)out_slow_base;
        if (band_on_cols) {
            size_t rows_bytes = (size_t)width * (size_t)other * elem_size;
            for (int32_t n = 0; n < batch; n++) {
                memcpy(dst + ((size_t)n * (size_t)banded + (size_t)start) *
                                 (size_t)other * elem_size,
                       out_tile + (size_t)n * rows_bytes,
                       rows_bytes);
            }
        } else {
            for (int32_t n = 0; n < batch; n++) {
                for (int32_t r = 0; r < other; r++) {
                    memcpy(dst + (((size_t)n * (size_t)other + (size_t)r) *
                                      (size_t)banded + (size_t)start) *
                                     elem_size,
                           out_tile + ((size_t)n * (size_t)other + (size_t)r) *
                                          strip,
                           strip);
                }
            }
        }

        if (stats) stats->spills_bytes += tile_bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_SPILL, TIGRIS_TRACE_POOL_SLOW,
            out_idx, TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(dst, mem->slow_base) +
                (uint32_t)start * (band_on_cols ? (uint32_t)other : 1u) * elem_size,
            TIGRIS_TRACE_OFFSET(out_tile, mem->fast_base), tile_bytes,
            band_on_cols ? start : 0, band_on_cols ? end : cols,
            band_on_cols ? 0 : start, band_on_cols ? rows : end);
        tigris_mem_reset_fast(mem);
    }

    memset(&mem->tile, 0, sizeof(mem->tile));
    tigris_mem_reset_fast(mem);
    mem->tensor_ptrs[in_idx] = in_slow_base;
    mem->tensor_ptrs[out_idx] = out_slow_base;
    return result;
}

static int tensor_band_view(const tigris_plan_t *plan, const tigris_tensor_t *tensor,
                            int leading, int32_t *batch, int32_t *rows, int32_t *cols)
{
    if (!leading) return tigris_tensor_row_view(plan, tensor, batch, rows, cols);
    uint8_t axis = tigris_band_axis(plan, tensor, leading);
    const int32_t *shape = tigris_tensor_shape(plan, tensor);
    *batch = 1;
    for (uint8_t i = 0u; i < axis; i++) *batch *= shape[i];
    *rows = tigris_tensor_shape(plan, tensor)[axis];
    *cols = (int32_t)(tensor->size_bytes / tigris_dtype_size(tensor->dtype) / (uint32_t)*rows / (uint32_t)*batch);
    return 1;
}


/**
 * Gather a band of rows out of a whole [batch, rows, cols] tensor, and the
 * mirror that writes one back.
 *
 * The rectangle helpers in tigris_mem slice a tensor as NHWC, which puts the
 * band on the wrong axis once the leading axes are batch rather than spatial,
 * so the row band states its extents and copies directly. The band cuts the
 * second to last axis, so every batch contributes one contiguous run.
 */
static void gather_row_band(
    uint8_t *band_buf, const uint8_t *whole, int32_t batch, int32_t rows,
    int32_t start, int32_t band, uint32_t row_bytes,
    tigris_mem_t *mem, uint16_t tensor)
{
    size_t whole_stride = (size_t)rows * (size_t)row_bytes;
    size_t band_stride = (size_t)band * (size_t)row_bytes;
    size_t offset = (size_t)start * (size_t)row_bytes;
    for (int32_t n = 0; n < batch; n++)
        memcpy(band_buf + (size_t)n * band_stride,
               whole + (size_t)n * whole_stride + offset, band_stride);
    if (mem->stats) mem->stats->loads_bytes += (uint32_t)((size_t)batch * band_stride);
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_LOAD, TIGRIS_TRACE_POOL_FAST,
        tensor, TIGRIS_TRACE_NONE_U16,
        TIGRIS_TRACE_OFFSET(band_buf, mem->fast_base),
        TIGRIS_TRACE_OFFSET(whole, mem->slow_base) + (uint32_t)offset,
        (uint32_t)((size_t)batch * band_stride), start, start + band, -1, -1);
#ifndef TIGRIS_TRACE
    (void)tensor;
#endif
}

static void scatter_row_band(
    uint8_t *whole, const uint8_t *band_buf, int32_t batch, int32_t rows,
    int32_t start, int32_t band, uint32_t row_bytes,
    tigris_mem_t *mem, uint16_t tensor)
{
    size_t whole_stride = (size_t)rows * (size_t)row_bytes;
    size_t band_stride = (size_t)band * (size_t)row_bytes;
    size_t offset = (size_t)start * (size_t)row_bytes;
    for (int32_t n = 0; n < batch; n++)
        memcpy(whole + (size_t)n * whole_stride + offset,
               band_buf + (size_t)n * band_stride, band_stride);
    if (mem->stats) mem->stats->spills_bytes += (uint32_t)((size_t)batch * band_stride);
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_SPILL, TIGRIS_TRACE_POOL_SLOW,
        tensor, TIGRIS_TRACE_NONE_U16,
        TIGRIS_TRACE_OFFSET(whole, mem->slow_base) + (uint32_t)offset,
        TIGRIS_TRACE_OFFSET(band_buf, mem->fast_base),
        (uint32_t)((size_t)batch * band_stride), start, start + band, -1, -1);
#ifndef TIGRIS_TRACE
    (void)tensor;
#endif
}

static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_grouped_reduction(
    const tigris_plan_t *plan, const tigris_stage_t *stage, tigris_mem_t *mem,
    tigris_kernel_fn kernel, void *user_ctx, tigris_exec_stats_t *stats)
{
    uint16_t input = tigris_stage_inputs(plan, stage)[0];
    uint16_t output = tigris_stage_outputs(plan, stage)[0];
    uint16_t index = tigris_stage_ops(plan, stage)[0];
    const tigris_op_t *op = &plan->ops[index];
    const int32_t *shape = tigris_tensor_shape(plan, &plan->tensors[input]);
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    int32_t channels = shape[3];
    int32_t spatial = shape[1] * shape[2];
    int32_t group = (int32_t)tile.input_channel_group;
    int32_t interval = (int32_t)tile.input_spatial_interval;
    uint32_t element = tigris_dtype_size(plan->tensors[input].dtype);
    void *source = mem->tensor_ptrs[input];
    if (!source || group <= 0 || interval <= 0)
        return TIGRIS_EXEC_ERR_TILE;
    if (tigris_mem_alloc_slow(mem, output, plan->tensors[output].size_bytes) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    void *destination = mem->tensor_ptrs[output];
    uint32_t acc_bytes;
    if (!align_tensor_size((uint32_t)group * 4u, &acc_bytes) ||
        mem->fast_used > mem->fast_size || acc_bytes > mem->fast_size - mem->fast_used ||
        acc_bytes > stage_fast_capacity(plan, stage, mem))
        return TIGRIS_EXEC_ERR_MEM;
    uint32_t reserved = mem->fast_reserved;
    void *accumulator = mem->fast_base + mem->fast_used;
    mem->fast_used += acc_bytes;
    mem->fast_reserved = mem->fast_used;
    tigris_mem_note_fast_peak(mem);
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_ALLOC, TIGRIS_TRACE_POOL_FAST,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16, mem->fast_used - acc_bytes,
        0u, acc_bytes, -1, -1, -1, -1);
    tigris_exec_error_t result = TIGRIS_EXEC_OK;
    for (int32_t batch = 0; batch < shape[0] && result == TIGRIS_EXEC_OK; batch++) {
        for (int32_t channel = 0; channel < channels && result == TIGRIS_EXEC_OK; ) {
            int32_t count = channels - channel < group ? channels - channel : group;
            for (int32_t start = 0; start < spatial; ) {
                int32_t positions = spatial - start < interval ? spatial - start : interval;
                uint32_t bytes = (uint32_t)positions * (uint32_t)count * element;
                if (stats) stats->total_tiles++;
                TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
                    TIGRIS_TRACE_NONE_U16, (uint16_t)(start / interval), 0u, 0u, 0u,
                    start, start + positions, channel, channel + count);
                if (tigris_mem_alloc_fast(mem, input, bytes) != TIGRIS_MEM_OK ||
                    tigris_mem_alloc_fast(mem, output, (uint32_t)count * element) != TIGRIS_MEM_OK) {
                    result = TIGRIS_EXEC_ERR_MEM;
                    break;
                }
                const uint8_t *first = (const uint8_t *)source +
                    ((size_t)batch * (size_t)spatial + (size_t)start) * (size_t)channels * element;
                gather_row_band((uint8_t *)mem->tensor_ptrs[input], first, positions,
                                channels, channel, count, element, mem, input);
                mem->tile.active = 1u;
                mem->tile.in_h = 1;
                mem->tile.in_w = 1;
                mem->tile.out_h = 1;
                mem->tile.out_w = 1;
                mem->tile.reduce_channels = count;
                mem->tile.reduce_positions = positions;
                mem->tile.reduce_acc = accumulator;
                mem->tile.reduce_first = start == 0 ? 1u : 0u;
                mem->tile.reduce_last = spatial - start <= interval ? 1u : 0u;
                if (TRACED_KERNEL(kernel, plan, op, index, mem, user_ctx) != 0) {
                    result = TIGRIS_EXEC_ERR_KERNEL;
                    break;
                }
                if (mem->tile.reduce_last)
                    scatter_row_band((uint8_t *)destination + (size_t)batch * (size_t)channels * element,
                        (const uint8_t *)mem->tensor_ptrs[output], 1, channels, channel, count, element, mem, output);
                tigris_mem_reset_fast(mem);
                start += positions;
            }
            channel += count;
        }
    }
    memset(&mem->tile, 0, sizeof(mem->tile));
    mem->fast_reserved = reserved;
    tigris_mem_reset_fast(mem);
    mem->tensor_ptrs[input] = source;
    mem->tensor_ptrs[output] = destination;
    return result;
}

/**
 * One band of a row-tiled stage: load the band of each stage input, run the
 * ops at the band's row count, spill the band of each stage output. Rows are
 * independent, so the mapping is 1:1 on every tensor and there is no halo.
 */
static tigris_exec_error_t run_row_band(
    const tigris_plan_t  *plan,
    const tigris_stage_t *stage,
    tigris_mem_t         *mem,
    tigris_kernel_fn      kernel,
    void                 *user_ctx,
    void                **in_slow_bases,
    void                **out_slow_bases,
    int32_t               rows,
    int                   leading,
    int32_t               start,
    int32_t               end)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);

    int32_t band = end - start;
    mem->tile.active = 1;
    mem->tile.row_tiled = leading ? (uint8_t)leading : 1u;
    mem->tile.in_h = band;
    mem->tile.out_h = band;
    mem->tile.out_row_origin = start;
    mem->tile.in_w = 1;
    mem->tile.out_w = 1;

    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        uint16_t tidx = sin[i];
        const tigris_tensor_t *t = &plan->tensors[tidx];
        int32_t b;
        int32_t r;
        int32_t c;
        if (tigris_whole_band_operand(plan, stage, tidx)) {
            mem->tensor_ptrs[tidx] = in_slow_bases[i];
            continue;
        }
        if (!tensor_band_view(plan, t, leading, &b, &r, &c))
            return TIGRIS_EXEC_ERR_TILE;
        if (r != rows) {
            /* An operand the band does not cut is read whole, straight out of
             * the arena it already lives in. Nothing to copy and nothing to
             * allocate. */
            mem->tensor_ptrs[tidx] = in_slow_bases[i];
            continue;
        }
        uint32_t row_bytes = t->size_bytes / (uint32_t)(b * r);
        uint32_t band_bytes = (uint32_t)b * (uint32_t)band * row_bytes;
        const uint8_t *whole = (const uint8_t *)in_slow_bases[i];
        if (tigris_mem_alloc_fast(mem, tidx, band_bytes) != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        gather_row_band((uint8_t *)mem->tensor_ptrs[tidx], whole, b, r,
                        start, band, row_bytes, mem, tidx);
    }

    for (uint16_t j = 0; j < stage->ops_count; j++) {
        uint16_t op_idx = sops[j];
        const tigris_op_t *op = &plan->ops[op_idx];
        for (uint8_t k = 0u; k < op->num_outputs; k++) {
            uint16_t tidx = tigris_op_outputs(plan, op)[k];
            const tigris_tensor_t *t = &plan->tensors[tidx];
            uint32_t band_bytes =
                (uint32_t)((uint64_t)(t->size_bytes / (uint32_t)rows) * (uint32_t)band);
            if (tigris_mem_alloc_fast(mem, tidx, band_bytes) != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
        }
        if (TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx) != 0)
            return TIGRIS_EXEC_ERR_KERNEL;
    }

    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        uint16_t tidx = sout[i];
        const tigris_tensor_t *t = &plan->tensors[tidx];
        int32_t b;
        int32_t r;
        int32_t c;
        if (!tensor_band_view(plan, t, leading, &b, &r, &c) || r != rows)
            return TIGRIS_EXEC_ERR_TILE;
        uint32_t row_bytes = t->size_bytes / (uint32_t)(b * r);
        scatter_row_band((uint8_t *)out_slow_bases[i],
                         (const uint8_t *)mem->tensor_ptrs[tidx],
                         b, r, start, band, row_bytes, mem, tidx);
        mem->tensor_ptrs[tidx] = out_slow_bases[i];
    }
    return TIGRIS_EXEC_OK;
}

/**
 * Whether a row-banded stage may write its output over its input.
 *
 * A band is gathered out of the input's slow buffer into fast before any op
 * runs and scattered back to the SAME rows afterwards, so sharing one buffer
 * only ever overwrites rows the band already consumed. That holds while the
 * mapping is 1:1 in both directions, which is what an identical row view plus
 * an identical size says, and while nothing after this stage reads the input.
 * A model input stays out of it: the buffer belongs to the caller, who may
 * still hold the values for a second run.
 */
static int row_band_output_may_alias_input(
    const tigris_plan_t  *plan,
    const tigris_stage_t *stage,
    const uint16_t       *last_consumer,
    uint16_t              current_stage)
{
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    int32_t ib, ir, ic, ob, orr, oc;

    if (stage->inputs_count != 1u || stage->outputs_count != 1u)
        return 0;
    if (!last_consumer || last_consumer[sin[0]] != current_stage)
        return 0;
    if (plan->tensors[sin[0]].size_bytes != plan->tensors[sout[0]].size_bytes)
        return 0;
    if (!tigris_tensor_row_view(plan, &plan->tensors[sin[0]], &ib, &ir, &ic) ||
        !tigris_tensor_row_view(plan, &plan->tensors[sout[0]], &ob, &orr, &oc))
        return 0;
    if (ib != ob || ir != orr || ic != oc)
        return 0;
    for (uint8_t i = 0; i < plan->header->num_model_inputs; i++) {
        if (plan->model_inputs[i] == sin[0])
            return 0;
    }
    return 1;
}

/**
 * Execute a rank-2 matrix pipeline a band of rows at a time.
 *
 * Rows are independent, so a band needs no halo and the mapping is 1:1 on
 * every tensor: load the band of each stage input, run the ops at the band's
 * row count, spill the band of each stage output. Marked TIGRIS_NOINLINE to
 * keep the inlined dispatcher within its per-frame stack budget.
 */
static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled_rows(
    const tigris_plan_t      *plan,
    const tigris_stage_t     *stage,
    tigris_mem_t             *mem,
    tigris_kernel_fn          kernel,
    void                     *user_ctx,
    executor_workspace_impl_t *workspace,
    const uint16_t           *last_consumer,
    uint16_t                  current_stage,
    int32_t                   rows,
    int                       leading,
    tigris_exec_stats_t *stats)
{
    const uint16_t *sops = tigris_stage_ops(plan, stage);
    const uint16_t *sin = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    void **out_slow_bases = workspace->output_ptrs;
    void **in_slow_bases = workspace->input_ptrs;

    int aliased = !leading && row_band_output_may_alias_input(
        plan, stage, last_consumer, current_stage);

    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        in_slow_bases[i] = mem->tensor_ptrs[sin[i]];
        if (!in_slow_bases[i])
            return TIGRIS_EXEC_ERR_MEM;
    }
    if (aliased) {
        /* One buffer carries both tensors, so the pool never holds the pair
         * at once. The compiler's slow-pool estimate applies the same rule. */
        out_slow_bases[0] = in_slow_bases[0];
        mem->tensor_ptrs[sout[0]] = out_slow_bases[0];
    } else {
        for (uint16_t i = 0; i < stage->outputs_count; i++) {
            if (tigris_mem_alloc_slow(mem, sout[i],
                                      plan->tensors[sout[i]].size_bytes)
                    != TIGRIS_MEM_OK)
                return TIGRIS_EXEC_ERR_MEM;
            out_slow_bases[i] = mem->tensor_ptrs[sout[i]];
        }
    }

    /* Largest band whose stage inputs and every op output fit together. Each
     * is its own aligned allocation, so the alignment is paid per tensor
     * rather than once on the total, and the search recomputes it per
     * candidate rather than scaling one figure. */
    uint32_t available = stage_fast_capacity(plan, stage, mem);
    int32_t band = 0;
    int32_t low = 1;
    int32_t high = rows;
    while (low <= high) {
        int32_t candidate = low + (high - low) / 2;
        /* Each tensor is its own aligned allocation, so the alignment is paid
         * per tensor rather than once on the total. */
        uint64_t total = 0;
        for (uint16_t i = 0; i < stage->inputs_count; i++) {
            const tigris_tensor_t *t = &plan->tensors[sin[i]];
            uint32_t bytes;
            int32_t b, r, c;
            int whole = tigris_whole_band_operand(plan, stage, sin[i]) ||
                (tensor_band_view(plan, t, leading, &b, &r, &c) && r != rows);
            uint32_t size = whole ? t->size_bytes :
                (uint32_t)((uint64_t)(t->size_bytes / (uint32_t)rows) * (uint32_t)candidate);
            if (!align_tensor_size(
                    size, &bytes))
                return TIGRIS_EXEC_ERR_TILE;
            total += bytes;
        }
        for (uint16_t j = 0; j < stage->ops_count; j++) {
            const tigris_op_t *op = &plan->ops[sops[j]];
            for (uint8_t k = 0u; k < op->num_outputs; k++) {
                const tigris_tensor_t *t = &plan->tensors[tigris_op_outputs(plan, op)[k]];
                uint32_t bytes;
                if (!align_tensor_size(
                        (uint32_t)((uint64_t)(t->size_bytes / (uint32_t)rows) *
                                   (uint32_t)candidate), &bytes))
                    return TIGRIS_EXEC_ERR_TILE;
                total += bytes;
            }
        }
        if (total <= available) {
            band = candidate;
            low = candidate + 1;
        } else {
            high = candidate - 1;
        }
    }
    if (band == 0)
        return TIGRIS_EXEC_ERR_TILE;

    tigris_exec_error_t result = TIGRIS_EXEC_OK;
    for (int32_t start = 0; start < rows; start += band) {
        int32_t end = rows - start < band ? rows : start + band;
        if (stats) stats->total_tiles++;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(start / band), 0u, 0u, 0u,
            start, end, -1, -1);
        result = run_row_band(plan, stage, mem, kernel, user_ctx,
                              in_slow_bases, out_slow_bases, rows, leading, start, end);
        if (result != TIGRIS_EXEC_OK)
            break;
        tigris_mem_reset_fast(mem);
    }

    memset(&mem->tile, 0, sizeof(mem->tile));
    tigris_mem_reset_fast(mem);
    for (uint16_t i = 0; i < stage->inputs_count; i++)
        mem->tensor_ptrs[sin[i]] = in_slow_bases[i];
    for (uint16_t i = 0; i < stage->outputs_count; i++)
        mem->tensor_ptrs[sout[i]] = out_slow_bases[i];
    /* The shared buffer now belongs to the output. Dropping the input's
     * pointer keeps the caller's dead-tensor sweep from freeing it. */
    if (aliased)
        mem->tensor_ptrs[sin[0]] = NULL;
    /* Intermediates lived only inside a band. */
    for (uint16_t j = 0; j < stage->ops_count; j++) {
        const tigris_op_t *op = &plan->ops[sops[j]];
        for (uint8_t k = 0; k < op->num_outputs; k++) {
            uint16_t tidx = tigris_op_outputs(plan, op)[k];
            int is_stage_out = 0;
            for (uint16_t i = 0; i < stage->outputs_count; i++) {
                if (tidx == sout[i]) { is_stage_out = 1; break; }
            }
            if (!is_stage_out)
                mem->tensor_ptrs[tidx] = NULL;
        }
    }
    return result;
}

/* The If or While a stage holds alone, or NULL. */
static const tigris_op_t *stage_control(const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (plan->num_subgraphs == 0u || stage->ops_count != 1u)
        return NULL;
    const tigris_op_t *op = &plan->ops[tigris_stage_ops(plan, stage)[0]];
    return (op->op_type == TIGRIS_OP_IF || op->op_type == TIGRIS_OP_WHILE) ? op : NULL;
}

/* The graph in slot 0 or 1 of a control-flow operator's subgraph attribute. */
static uint16_t control_graph(const tigris_plan_t *plan, uint16_t op_index, uint8_t slot)
{
    const uint8_t *graphs = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_SUBGRAPHS, NULL);
    if (!graphs)
        return 0u;
    return (uint16_t)((uint16_t)graphs[2u * slot] | (uint16_t)((uint16_t)graphs[2u * slot + 1u] << 8));
}

/* Copy tensors `src` into graph g's inputs, placed in slow memory. */
static tigris_exec_error_t copy_into_graph(const tigris_plan_t *plan, tigris_mem_t *mem,
                                           uint16_t g, const uint16_t *src)
{
    const tigris_subgraph_t *graph = &plan->subgraphs[g];
    for (uint16_t i = 0; i < graph->inputs_count; i++) {
        const uint16_t dst = plan->index_pool[graph->inputs_off + i];
        if (!mem->tensor_ptrs[src[i]] ||
            tigris_mem_alloc_slow(mem, dst, plan->tensors[dst].size_bytes) != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        memcpy(mem->tensor_ptrs[dst], mem->tensor_ptrs[src[i]], plan->tensors[dst].size_bytes);
        if (mem->stats) mem->stats->copy_bytes += plan->tensors[dst].size_bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_COPY, TIGRIS_TRACE_POOL_SLOW,
            dst, TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[dst], mem->slow_base),
            TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[src[i]], mem->slow_base),
            plan->tensors[dst].size_bytes, -1, -1, -1, -1);
    }
    return TIGRIS_EXEC_OK;
}

/* Copy graph g's outputs into `dst`, placing a destination not yet placed, or
 * none when `dst` is NULL; then release g's inputs and outputs, which no stage
 * of it reads any more. */
static tigris_exec_error_t copy_out_of_graph(const tigris_plan_t *plan, tigris_mem_t *mem,
                                             uint16_t g, const uint16_t *dst)
{
    const tigris_subgraph_t *graph = &plan->subgraphs[g];
    for (uint16_t i = 0; dst != NULL && i < graph->outputs_count; i++) {
        const uint16_t src = plan->index_pool[graph->outputs_off + i];
        if (!mem->tensor_ptrs[src] ||
            (!mem->tensor_ptrs[dst[i]] &&
             tigris_mem_alloc_slow(mem, dst[i], plan->tensors[dst[i]].size_bytes) != TIGRIS_MEM_OK))
            return TIGRIS_EXEC_ERR_MEM;
        memcpy(mem->tensor_ptrs[dst[i]], mem->tensor_ptrs[src], plan->tensors[dst[i]].size_bytes);
        if (mem->stats) mem->stats->copy_bytes += plan->tensors[dst[i]].size_bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_COPY, TIGRIS_TRACE_POOL_SLOW,
            dst[i], TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[dst[i]], mem->slow_base),
            TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[src], mem->slow_base),
            plan->tensors[dst[i]].size_bytes, -1, -1, -1, -1);
    }
    for (uint16_t i = 0; i < graph->inputs_count; i++)
        mem->tensor_ptrs[plan->index_pool[graph->inputs_off + i]] = NULL;
    for (uint16_t i = 0; i < graph->outputs_count; i++)
        mem->tensor_ptrs[plan->index_pool[graph->outputs_off + i]] = NULL;
    compact_slow(mem, plan);
    return TIGRIS_EXEC_OK;
}

/* Start a control-flow operator: an If enters the branch its condition
 * selects; a While copies its operands into its results and enters its
 * condition. *graph_out is the graph to run. */
static TIGRIS_NOINLINE tigris_exec_error_t start_control(
    const tigris_plan_t *plan, tigris_mem_t *mem, const tigris_op_t *op,
    uint16_t op_index, uint16_t *graph_out)
{
    const uint16_t *ins = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);
    if (op->op_type == TIGRIS_OP_IF) {
        const uint8_t *condition = (const uint8_t *)mem->tensor_ptrs[ins[0]];
        if (!condition)
            return TIGRIS_EXEC_ERR_MEM;
        *graph_out = control_graph(plan, op_index, (*condition != 0u) ? 0u : 1u);
        return copy_into_graph(plan, mem, *graph_out, ins + 1);
    }
    /* Each loop variable starts from the next input, or from its constant. */
    uint8_t next = 0u;
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        const void *initial = tigris_op_constant(plan, op_index, i);
        if (initial == NULL && next < op->num_inputs) {
            initial = mem->tensor_ptrs[ins[next]];
            next++;
        }
        if (initial == NULL ||
            tigris_mem_alloc_slow(mem, outs[i], plan->tensors[outs[i]].size_bytes) != TIGRIS_MEM_OK)
            return TIGRIS_EXEC_ERR_MEM;
        memcpy(mem->tensor_ptrs[outs[i]], initial, plan->tensors[outs[i]].size_bytes);
        if (mem->stats) mem->stats->copy_bytes += plan->tensors[outs[i]].size_bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_COPY, TIGRIS_TRACE_POOL_SLOW,
            outs[i], TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[outs[i]], mem->slow_base),
            tigris_op_constant(plan, op_index, i) != NULL ? 0u :
                TIGRIS_TRACE_OFFSET(initial, mem->slow_base),
            plan->tensors[outs[i]].size_bytes, -1, -1, -1, -1);
    }
    *graph_out = control_graph(plan, op_index, 0u);
    return copy_into_graph(plan, mem, *graph_out, outs);
}

/* After graph *graph finishes: an If copies its branch's results out and is
 * done; a While's condition either ends the loop or enters the body on the
 * current results, and its body updates the results and re-enters the
 * condition. *done is set when the operator has finished. */
static TIGRIS_NOINLINE tigris_exec_error_t step_control(
    const tigris_plan_t *plan, tigris_mem_t *mem, const tigris_op_t *op,
    uint16_t op_index, uint16_t *graph, int *done)
{
    const uint16_t *outs = tigris_op_outputs(plan, op);
    *done = 0;
    if (op->op_type == TIGRIS_OP_IF) {
        *done = 1;
        return copy_out_of_graph(plan, mem, *graph, outs);
    }
    const uint16_t condition = control_graph(plan, op_index, 0u);
    if (*graph == condition) {
        const tigris_subgraph_t *graph_desc = &plan->subgraphs[condition];
        const uint8_t *verdict =
            (const uint8_t *)mem->tensor_ptrs[plan->index_pool[graph_desc->outputs_off]];
        if (!verdict)
            return TIGRIS_EXEC_ERR_MEM;
        const int again = *verdict != 0u;
        tigris_exec_error_t err = copy_out_of_graph(plan, mem, condition, NULL);
        if (err != TIGRIS_EXEC_OK || !again) {
            *done = 1;
            return err;
        }
        *graph = control_graph(plan, op_index, 1u);
        return copy_into_graph(plan, mem, *graph, outs);
    }
    tigris_exec_error_t err = copy_out_of_graph(plan, mem, *graph, outs);
    if (err != TIGRIS_EXEC_OK)
        return err;
    *graph = condition;
    return copy_into_graph(plan, mem, condition, outs);
}

static TIGRIS_NOINLINE void free_dead_tensors(const tigris_plan_t *plan, tigris_mem_t *mem,
                                              const uint16_t *last_consumer,
                                              uint16_t first, uint16_t s);

/* Start the control-flow operator at stage s: push a frame of its stage, the
 * range it is entered from and the graph it runs, and continue (flow[3]) one
 * before that graph's first stage, which the stage loop advances past. */
static TIGRIS_NOINLINE tigris_exec_error_t enter_control(
    const tigris_plan_t *plan, tigris_mem_t *mem, uint16_t *flow, uint16_t depth_capacity,
    uint16_t s)
{
    if (flow[0] >= depth_capacity)
        return TIGRIS_EXEC_ERR_WORKSPACE;
    uint16_t *frame = &flow[4u + 4u * flow[0]];
    const uint16_t op_index = tigris_stage_ops(plan, &plan->stages[s])[0];
    tigris_exec_error_t err = start_control(plan, mem, &plan->ops[op_index], op_index, &frame[3]);
    if (err != TIGRIS_EXEC_OK)
        return err;
    frame[0] = s;
    frame[1] = flow[1];
    frame[2] = flow[2];
    flow[0]++;
    flow[1] = plan->subgraphs[frame[3]].first_stage;
    flow[2] = (uint16_t)(flow[1] + plan->subgraphs[frame[3]].num_stages);
    flow[3] = (uint16_t)(flow[1] - 1u);
    return TIGRIS_EXEC_OK;
}

/* The running graph finished at stage s: step the innermost control-flow
 * operator into its next graph, or, when it is done, return to the graph that
 * ran it after its stage, as many levels as finish together. */
static TIGRIS_NOINLINE tigris_exec_error_t unwind_control(
    const tigris_plan_t *plan, tigris_mem_t *mem, const uint16_t *last_consumer,
    uint16_t *flow, uint16_t s)
{
    flow[3] = s;
    while (flow[0] > 0u && (uint16_t)(flow[3] + 1u) == flow[2]) {
        uint16_t *frame = &flow[4u * flow[0]];
        const uint16_t op_index = tigris_stage_ops(plan, &plan->stages[frame[0]])[0];
        int done = 0;
#ifdef TIGRIS_TRACE
        mem->trace_stage = frame[0];
#endif
        tigris_exec_error_t err = step_control(plan, mem, &plan->ops[op_index], op_index,
                                               &frame[3], &done);
        if (err != TIGRIS_EXEC_OK)
            return err;
        if (!done) {
            flow[1] = plan->subgraphs[frame[3]].first_stage;
            flow[2] = (uint16_t)(flow[1] + plan->subgraphs[frame[3]].num_stages);
            flow[3] = (uint16_t)(flow[1] - 1u);
            return TIGRIS_EXEC_OK;
        }
        flow[3] = frame[0];
        flow[1] = frame[1];
        flow[2] = frame[2];
        flow[0]--;
        free_dead_tensors(plan, mem, last_consumer, flow[1], flow[3]);
        TRACE_STAGE_END(mem, flow[3]);
    }
    return TIGRIS_EXEC_OK;
}

/* Release the slow tensors a stage in [first, s] read last, then compact. A
 * graph frees only its own tensors: those last read in its own stage range. */
static TIGRIS_NOINLINE void free_dead_tensors(const tigris_plan_t *plan, tigris_mem_t *mem,
                                              const uint16_t *last_consumer,
                                              uint16_t first, uint16_t s)
{
    int need_compact = 0;
    for (uint16_t t = 0; t < plan->header->num_tensors; t++) {
        if (last_consumer[t] >= first && last_consumer[t] <= s &&
            last_consumer[t] != UINT16_MAX) {
            uintptr_t p = (uintptr_t)mem->tensor_ptrs[t];
            uintptr_t slow = (uintptr_t)mem->slow_base;
            if (p && p >= slow && p - slow < mem->slow_size) {
                mem->tensor_ptrs[t] = NULL;
                need_compact = 1;
            }
        }
    }
    if (need_compact)
        compact_slow(mem, plan);
}

static TIGRIS_NOINLINE tigris_exec_error_t exec_stage_tiled_reshape(
    const tigris_plan_t *plan, const tigris_stage_t *stage, tigris_mem_t *mem,
    const uint16_t *last_consumer, uint16_t current_stage,
    tigris_exec_stats_t *stats)
{
    tigris_reshape_band_t view;
    tigris_tile_plan_v10_t tile = tigris_stage_tiles(plan, stage);
    if (!tigris_reshape_band(plan, stage, &view))
        return TIGRIS_EXEC_ERR_TILE;
    if (plan->header->version < TIGRIS_SCHEMA_VERSION_STATE) {
        if (!tigris_reshape_tile_valid(tigris_stage_tile_plan(plan, stage), &view))
            return TIGRIS_EXEC_ERR_TILE;
    } else if (tile.input_band_extent == 0u || tile.input_band_extent > (uint32_t)view.input_rows ||
               ((int64_t)tile.input_band_extent * view.input_width) % view.output_width != 0) {
        return TIGRIS_EXEC_ERR_TILE;
    }
    int32_t band_rows = (int32_t)tile.input_band_extent;
    uint16_t src = tigris_stage_inputs(plan, stage)[0], dst = tigris_stage_outputs(plan, stage)[0];
    void *input = mem->tensor_ptrs[src];
    if (!input) return TIGRIS_EXEC_ERR_MEM;
    int alias = row_band_output_may_alias_input(plan, stage, last_consumer, current_stage);
    if (alias) mem->tensor_ptrs[dst] = input;
    else if (tigris_mem_alloc_slow(mem, dst, plan->tensors[dst].size_bytes) != TIGRIS_MEM_OK)
        return TIGRIS_EXEC_ERR_MEM;
    void *output = mem->tensor_ptrs[dst];
    uint32_t element = tigris_dtype_size(plan->tensors[src].dtype);
    uint32_t input_width = (uint32_t)view.input_width * element;
    uint32_t output_width = (uint32_t)view.output_width * element;
    tigris_exec_error_t result = TIGRIS_EXEC_OK;
    for (int32_t start = 0; start < view.input_rows; start += band_rows) {
        int32_t count = view.input_rows - start;
        if (count > band_rows) count = band_rows;
        int32_t target_start = (int32_t)((int64_t)start * view.input_width / view.output_width);
        int32_t target_count = (int32_t)((int64_t)count * view.input_width / view.output_width);
        if (stats) stats->total_tiles++;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(start / band_rows), 0u, 0u, 0u,
            start, start + count, -1, -1);
        uint32_t bytes = (uint32_t)view.blocks * (uint32_t)count * input_width;
        if (tigris_mem_alloc_fast(mem, src, bytes) != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            break;
        }
        void *band = mem->tensor_ptrs[src];
        gather_row_band((uint8_t *)band, (const uint8_t *)input, view.blocks,
                        view.input_rows, start, count, input_width, mem, src);
        /* Both intervals name the same bytes; the output aliases this band. */
        scatter_row_band((uint8_t *)output, (const uint8_t *)band, view.blocks,
                         view.output_rows, target_start, target_count, output_width, mem, dst);
        tigris_mem_reset_fast(mem);
    }
    mem->tensor_ptrs[src] = alias ? NULL : input;
    mem->tensor_ptrs[dst] = output;
    tigris_mem_reset_fast(mem);
    return result;
}

/* Chained tiled execution */

static TIGRIS_NOINLINE tigris_exec_error_t chain_kernel(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_idx,
    tigris_mem_t *mem, tigris_kernel_fn kernel, void *user_ctx,
    const executor_workspace_impl_t *workspace, int32_t gs, int32_t ge)
{
    int binary = op->num_inputs == 2 &&
        is_axis1_binary_pointwise_op(op->op_type);
    if (!binary)
        return TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx) != 0
            ? TIGRIS_EXEC_ERR_KERNEL : TIGRIS_EXEC_OK;

    const uint16_t *ins = tigris_op_inputs(plan, op);
    void *bases[2] = { mem->tensor_ptrs[ins[0]], mem->tensor_ptrs[ins[1]] };
    size_t offsets[2] = { 0, 0 };
    for (unsigned k = 0; k < 2; k++) {
        uint16_t tidx = ins[k];
        const tigris_tensor_t *t = &plan->tensors[tidx];
        const int32_t *sh = tigris_tensor_shape(plan, t);
        if (t->ndim != 4 || (t->flags & TIGRIS_TENSOR_LINEAR) ||
            sh[0] != 1 || sh[1] <= 1 ||
            workspace->out_ge[tidx] <= workspace->out_gs[tidx])
            continue;
        if (!bases[k] || gs < workspace->out_gs[tidx] ||
            ge > workspace->out_ge[tidx]) {
            return TIGRIS_EXEC_ERR_TILE;
        }
        offsets[k] = (size_t)(gs - workspace->out_gs[tidx]) *
            (t->size_bytes / (uint32_t)sh[1]);
    }
    /* A residual operand can retain halo rows that the main path consumed. */
    for (unsigned k = 0; k < 2; k++) {
        if (offsets[k])
            mem->tensor_ptrs[ins[k]] = (uint8_t *)bases[k] + offsets[k];
    }
    int result = TRACED_KERNEL(kernel, plan, op, op_idx, mem, user_ctx);
    for (unsigned k = 0; k < 2; k++)
        mem->tensor_ptrs[ins[k]] = bases[k];
    return result != 0 ? TIGRIS_EXEC_ERR_KERNEL : TIGRIS_EXEC_OK;
}

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
static TIGRIS_NOINLINE tigris_exec_error_t exec_chain_tiled(
    const tigris_plan_t *plan,
    uint16_t             first_idx,
    uint16_t             num_chain,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    executor_workspace_impl_t *workspace,
    tigris_exec_stats_t *stats)
{
    TRACE_STAGE_BEGIN(mem, first_idx, TIGRIS_TRACE_PATH_CHAIN);
    if (num_chain < 2 || num_chain > workspace->chain_capacity)
        return TIGRIS_EXEC_ERR_TILE;

    const tigris_stage_t **stages = workspace->stages;
    chain_stage_info_t *info = workspace->info;
    uint32_t sp_total = 0;

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
        info[c].sp_first = (int32_t)sp_total;

        const uint16_t *sops = tigris_stage_ops(plan, stages[c]);
        for (uint16_t j = 0; j < stages[c]->ops_count; j++) {
            uint8_t t = plan->ops[sops[j]].op_type;
            if (!is_height_tiling_op(t))
                return TIGRIS_EXEC_ERR_TILE;
            /* Fail closed on a chained ConvTranspose. is_height_tiling_op now
             * admits it (the standalone 2D route needs that), but the chain
             * composer only models shrinking spatial ops via
             * tigris_height_spatial_op; a ConvTranspose would be skipped and run
             * with shape-preserving pointwise geometry, producing wrong output
             * or an OOB write instead of expanding. This mirrors the 1D guard
             * in exec_stage_tiled. The guard sits in the pre-execution
             * enumeration pass, so it covers every op across all chained stages
             * before any tile geometry is composed or executed. The compiler
             * never chains an (untileable) ConvTranspose; this defends against
             * forged or malformed plans. */
            if (t == TIGRIS_OP_CONV_TRANSPOSE || tigris_height_upsample_op(t))
                return TIGRIS_EXEC_ERR_TILE;
            if (tigris_height_spatial_op(t)) {
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

                if (sp_total >= workspace->spatial_capacity)
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
                sp_total++;
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
#ifdef TIGRIS_TRACE
            mem->trace_stage = (uint16_t)(first_idx + c);
#endif

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
            if (stats) stats->weight_bytes += wb->uncompressed_size;
            TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_ALLOC, TIGRIS_TRACE_POOL_FAST,
                TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16,
                TIGRIS_TRACE_OFFSET(scratch, mem->fast_base), 0u, needed, -1, -1, -1, -1);
            TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_WEIGHTS, TIGRIS_TRACE_POOL_FAST,
                TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16,
                TIGRIS_TRACE_OFFSET(scratch, mem->fast_base), wb->compressed_size,
                wb->uncompressed_size, -1, -1, -1, -1);
        }
        mem->fast_reserved = mem->fast_used;
    }
#ifdef TIGRIS_TRACE
    mem->trace_stage = first_idx;
#endif

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
                if (sh[0] && sh[1] && sh[2] && sh[3]) {
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
                    if (tigris_height_spatial_op(ot) &&
                        sp_j < info[c].sp_count) {
                        size_t spi = spatial_index(
                            workspace, c, (uint16_t)sp_j);
                        int32_t ekh = workspace->sp_eff_khs[spi];
                        int32_t sh  = workspace->sp_strides[spi];
                        int32_t overlap = ekh > sh ? ekh - sh : 0;
                        int32_t oh = (cur_h - overlap) / sh;
                        cur_h = oh;
                        sp_j++;
                    }

                    const uint16_t *outs = tigris_op_outputs(plan, op);
                    for (uint8_t k = 0; k < op->num_outputs; k++) {
                        const tigris_tensor_t *t = &plan->tensors[outs[k]];
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
     *  - the fast arena is single-owner between tiles, because the roll
     *    depends on the persistent buffers' BYTES surviving from one tile to
     *    the next, not merely on the bump allocator handing back the same
     *    address;
     *  - backend weight-decompression scratch is carved once, before the loop.
     * Roll is only applied to rank-4, batch-1 op outputs (height tiling). */
    int line_buffered =
        (first_stage->_reserved1 & TIGRIS_STAGE_FLAG_LINE_BUFFERED) != 0;

    /* 3. Get chain tile height - use validated override (accounts for
     *    decompressed weights reducing available fast space) */
    int32_t chain_tile_h = chain_tile_h_override;
    int32_t last_full_out_h = info[num_chain - 1].full_out_h;

    int32_t num_tiles = (last_full_out_h + chain_tile_h - 1) / chain_tile_h;

    /* Rows of each last-stage output written so far, so the tile loop can
     * prove it leaves no gap. */
    int32_t *out_covered = workspace->out_covered;
    if (last_stage->outputs_count > workspace->output_capacity)
        return TIGRIS_EXEC_ERR_WORKSPACE;
    for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++)
        out_covered[oi] = 0;

#ifdef TIGRIS_COUNT_KERNEL_ROWS
    g_tigris_chain_tile_h = chain_tile_h;
    g_tigris_chain_num_tiles = num_tiles;
    g_tigris_interior_clamps = 0;
#endif

    /* Persistent fast bases for the last stage's spilled outputs. The spill
     * repoints those tensors at slow each tile; the roll re-establishes them
     * here so the next tile's kernels write back into the persistent buffer. */
    void **roll_last_fast = workspace->roll_last_fast;

    /* 3b. Line-buffered pre-allocation: allocate every chain op-output buffer
     *     ONCE at its maximum (interior) tile height, so the buffers keep fixed
     *     addresses and surviving content across tiles. The first-input load
     *     and last-output spill continue to cycle above these persistent
     *     buffers. Sizes mirror the budget-validation forward pass, so the
     *     reserved footprint equals the already-validated tile working set. */
    if (line_buffered) {
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
                if (tigris_height_spatial_op(ot) && sp_j < info[c].sp_count) {
                    size_t spi = spatial_index(workspace, c, (uint16_t)sp_j);
                    int32_t ekh = workspace->sp_eff_khs[spi];
                    int32_t sh  = workspace->sp_strides[spi];
                    int32_t overlap = ekh > sh ? ekh - sh : 0;
                    cur_h = (cur_h - overlap) / sh;
                    sp_j++;
                }
                const uint16_t *outs = tigris_op_outputs(plan, op);
                for (uint8_t k = 0; k < op->num_outputs; k++) {
                    uint16_t tidx = outs[k];
                    const tigris_tensor_t *t = &plan->tensors[tidx];
                    const int32_t *sh = tigris_tensor_shape(plan, t);
                    uint32_t numel = (uint32_t)sh[0] * (uint32_t)sh[1] *
                                     (uint32_t)sh[2] * (uint32_t)sh[3];
                    uint32_t elem = t->size_bytes / numel;
                    uint32_t bytes = (uint32_t)sh[0] * (uint32_t)cur_h *
                                     (uint32_t)sh[2] * (uint32_t)sh[3] * elem;
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
    }

    /* Row ranges belong to this chain, including when buffers are recomputed. */
    for (uint16_t i = 0; i < mem->num_tensors; i++) {
        workspace->out_gs[i] = 0;
        workspace->out_ge[i] = 0;
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

        if (stats) stats->total_tiles++;
#ifdef TIGRIS_TRACE
        mem->trace_stage = first_idx;
#endif
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_TILE_BEGIN, TIGRIS_TRACE_POOL_NONE,
            TIGRIS_TRACE_NONE_U16, (uint16_t)(tile), 0u, 0u, 0u,
            in_starts[0], in_ends[0], -1, -1);
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
            workspace->out_gs[tidx] = in_starts[0];
            workspace->out_ge[tidx] = in_ends[0];
        }

        /* d. Run each stage in the chain */
        for (uint16_t c = 0; c < num_chain; c++) {
            const tigris_stage_t *st = stages[c];
#ifdef TIGRIS_TRACE
            mem->trace_stage = (uint16_t)(first_idx + c);
#endif
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
                if (workspace->sp_tile_pt[spi] < 0 ||
                    workspace->sp_tile_pt[spi] > UINT16_MAX ||
                    workspace->sp_tile_pb[spi] < 0 ||
                    workspace->sp_tile_pb[spi] > UINT16_MAX)
                    return TIGRIS_EXEC_ERR_TILE;
                mem->tile.pad_top    = (uint16_t)workspace->sp_tile_pt[spi];
                mem->tile.pad_bottom = (uint16_t)workspace->sp_tile_pb[spi];
            } else {
                mem->tile.pad_top    = 0;
                mem->tile.pad_bottom = 0;
            }

            /* Track the data height and width flowing through the stage.
             * Starts at the stage's input geometry and takes each spatial
             * op's output geometry after it runs. */
            int32_t cur_data_h = tile_in_h;
            int32_t cur_data_w = info[c].full_in_w;
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
                    if (workspace->sp_tile_pt[spi] < 0 ||
                        workspace->sp_tile_pt[spi] > UINT16_MAX ||
                        workspace->sp_tile_pb[spi] < 0 ||
                        workspace->sp_tile_pb[spi] > UINT16_MAX)
                        return TIGRIS_EXEC_ERR_TILE;
                    mem->tile.in_h       = workspace->sp_tile_in_h[spi];
                    mem->tile.out_h      = workspace->sp_tile_out_h[spi];
                    mem->tile.pad_top    = (uint16_t)workspace->sp_tile_pt[spi];
                    mem->tile.pad_bottom = (uint16_t)workspace->sp_tile_pb[spi];
                    mem->tile.in_w       = workspace->sp_full_in_ws[spi];
                    mem->tile.out_w      = workspace->sp_full_out_ws[spi];
                } else {
                    /* Pointwise op: height and width preserved, no padding.
                     * Stating it per op rather than once before the loop is
                     * what exec_stage_tiled does, and the difference is not
                     * cosmetic: an op ahead of the spatial one was handed the
                     * stage's OUTPUT height, so it computed only the rows the
                     * stage finally emits and left the halo rows of its own
                     * output untouched. The spatial op then read those as
                     * data. */
                    mem->tile.in_h       = cur_data_h;
                    mem->tile.out_h      = cur_data_h;
                    mem->tile.in_w       = cur_data_w;
                    mem->tile.out_w      = cur_data_w;
                    mem->tile.pad_top    = 0;
                    mem->tile.pad_bottom = 0;
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
                 * recompute path). Roll only a batch-1, single-output op
                 * whose reconstructed height matches the tile context, so a
                 * mismatch (e.g. a pointwise op ahead of the spatial op) safely
                 * falls back to full compute. */
                mem->tile.out_row_start = 0;
                mem->tile.in_row_start  = 0;
                int fully_rolled = 0;
                if (line_buffered && stage_rollable && op->num_outputs == 1) {
                    uint16_t otid = outs[0];
                    const tigris_tensor_t *ot = &cplan->tensors[otid];
                    int32_t existing_out_h = op_full_out_h;
                    int32_t this_h = op_ge - op_gs;
                    const int32_t *osh = tigris_tensor_shape(cplan, ot);
                    if (osh[0] == 1 && this_h == existing_out_h) {
                        int32_t prev_gs = workspace->out_gs[otid];
                        int32_t prev_ge = workspace->out_ge[otid];
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
                            TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_MOVE, TIGRIS_TRACE_POOL_FAST,
                                otid, TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_OFFSET(base, mem->fast_base),
                                TIGRIS_TRACE_OFFSET(base, mem->fast_base) + (uint32_t)src_row * row_bytes,
                                (uint32_t)overlap * row_bytes, op_gs, op_gs + overlap, -1, -1);
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
                            fully_rolled = mem->tile.out_h == 0;
                        }
                    }
                }

                /* Record the range this op's output buffer now holds. Done
                 * every op, every tile (including full-computed ops), so the
                 * bookkeeping always describes current content: the spill
                 * places a stage output by it, and the next tile's roll reads
                 * it as the previous range. */
                for (uint8_t k = 0; k < op->num_outputs; k++) {
                    workspace->out_gs[outs[k]] = op_gs;
                    workspace->out_ge[outs[k]] = op_ge;
                }

                /* A fully carried band has no new rows to compute. */
                if (!fully_rolled) {
                    tigris_exec_error_t kernel_error = chain_kernel(
                        cplan, op, op_idx, mem, kernel, user_ctx,
                        workspace, op_gs, op_ge);
                    if (kernel_error != TIGRIS_EXEC_OK)
                        return kernel_error;
                }

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
                    cur_data_w = workspace->sp_full_out_ws[spi];
                    mem->tile.in_h = cur_data_h;
                    mem->tile.in_w = cur_data_w;
                    run_gs = workspace->sp_tile_out_gs[spi];
                    run_ge = workspace->sp_tile_out_ge[spi];
                    sp_j++;
                    /* Pre-set pad for next spatial op (if any) */
                    if (sp_j < info[c].sp_count) {
                        spi = spatial_index(
                            workspace, c, (uint16_t)sp_j);
                        if (workspace->sp_tile_pt[spi] < 0 ||
                            workspace->sp_tile_pt[spi] > UINT16_MAX ||
                            workspace->sp_tile_pb[spi] < 0 ||
                            workspace->sp_tile_pb[spi] > UINT16_MAX)
                            return TIGRIS_EXEC_ERR_TILE;
                        mem->tile.pad_top = (uint16_t)workspace->sp_tile_pt[spi];
                        mem->tile.pad_bottom = (uint16_t)workspace->sp_tile_pb[spi];
                    }
                }
            }
        }

        /* e. Spill the last stage's output tiles to slow, each at the row
         * range the op that produced it wrote. A stage may emit more than one
         * tensor, and an output produced ahead of the stage's last spatial op
         * carries that op's INPUT range: wider than the stage's output range
         * and starting on a different row. Writing them all at the stage's
         * output range puts those rows in the wrong place. */
        for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++) {
            uint16_t tidx = last_sout[oi];
            int32_t gs = workspace->out_gs[tidx];
            int32_t ge = workspace->out_ge[tidx];
            tigris_mem_error_t merr;
            if (!out_slow_bases[oi]) continue;
            /* Tiles must leave no row unwritten: this tile has to start at or
             * before the first row no tile has covered yet. */
            if (gs > out_covered[oi])
                return TIGRIS_EXEC_ERR_TILE;
            merr = tigris_mem_spill_tile(
                mem, plan, tidx, out_slow_bases[oi], gs, ge);
            if (merr != TIGRIS_MEM_OK) {
                TIGRIS_PLATFORM_DBG("MEM@%d f=%lu/%lu s=%lu/%lu\n", __LINE__,
                    (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                    (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                return TIGRIS_EXEC_ERR_MEM;
            }
            if (ge > out_covered[oi])
                out_covered[oi] = ge;
            /* spill_tile repointed this tensor at slow; for a line-buffered
             * chain, restore its persistent fast base so the next tile's last
             * stage writes back into the surviving buffer. After the final
             * tile leave it at slow, where the model output lives. */
            if (line_buffered && tile + 1 < num_tiles)
                mem->tensor_ptrs[tidx] = roll_last_fast[oi];
        }

        /* f. Reset fast arena for next tile */
        tigris_mem_reset_fast(mem);
    }

    /* 5. Verify every output row was covered */
    for (uint16_t oi = 0; oi < last_stage->outputs_count; oi++) {
        if (!out_slow_bases[oi]) continue;
        if (out_covered[oi] != tensor_row_count(plan, last_sout[oi]))
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
    return tigris_run_with_state(plan, mem, kernel, user_ctx, stats,
                                 workspace_storage, workspace_size, NULL, 0u);
}

size_t tigris_state_required(const tigris_plan_t *plan)
{
    return (plan != NULL) ? (size_t)plan->state_bytes : 0u;
}

tigris_exec_error_t tigris_state_init(const tigris_plan_t *plan, void *state, size_t state_size)
{
    if (plan == NULL)
        return TIGRIS_EXEC_ERR_NULL;
    if (plan->num_state == 0u)
        return TIGRIS_EXEC_OK;
    if (state == NULL || state_size < plan->state_bytes)
        return TIGRIS_EXEC_ERR_STATE;
    for (uint16_t i = 0; i < plan->num_state; i++) {
        const tigris_state_entry_t *entry = &plan->state_entries[i];
        memcpy((uint8_t *)state + entry->offset, plan->state_section + entry->initial,
               entry->bytes);
    }
    return TIGRIS_EXEC_OK;
}

/* `gap` is the free room an in-place run left at the floor below its inputs, 0 otherwise. */
static TIGRIS_NOINLINE tigris_exec_error_t prepare_stage_weights(
    const tigris_plan_t *plan, uint16_t stage, tigris_mem_t *mem, uint32_t gap,
    tigris_plan_t *prepared, const tigris_plan_t **run_plan,
    tigris_exec_stats_t *stats)
{
    const tigris_weight_block_t *wb = find_weight_block(plan, stage);
    if (!wb || wb->compressed_size == 0u)
        return TIGRIS_EXEC_OK;
    uint32_t needed;
    if (!align_tensor_size(wb->uncompressed_size, &needed) ||
        mem->fast_used > mem->fast_size ||
        (gap == 0u && needed > mem->fast_size - mem->fast_used) ||
        (gap != 0u && needed > gap))
        return TIGRIS_EXEC_ERR_MEM;
    uint8_t *scratch = mem->fast_base + (gap != 0u ? mem->fast_reserved : mem->fast_used);
    const uint8_t *source = plan->weight_blocks_data + wb->blob_offset;
    if (plan->weight_compression == TIGRIS_COMPRESS_LZ4) {
        int32_t decoded = tigris_lz4_decompress(source, wb->compressed_size,
                                             scratch, wb->uncompressed_size);
        if (decoded < 0 || (uint32_t)decoded != wb->uncompressed_size)
            return TIGRIS_EXEC_ERR_KERNEL;
    } else {
        memcpy(scratch, source, wb->uncompressed_size);
    }
    if (gap != 0u) {
        mem->fast_reserved += gap;
    } else {
        mem->fast_used += needed;
        mem->fast_reserved = mem->fast_used;
    }
    tigris_mem_note_fast_peak(mem);
    if (stats) stats->weight_bytes += wb->uncompressed_size;
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_ALLOC, TIGRIS_TRACE_POOL_FAST,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16,
        TIGRIS_TRACE_OFFSET(scratch, mem->fast_base), 0u, needed, -1, -1, -1, -1);
    TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_WEIGHTS, TIGRIS_TRACE_POOL_FAST,
        TIGRIS_TRACE_NONE_U16, TIGRIS_TRACE_NONE_U16,
        TIGRIS_TRACE_OFFSET(scratch, mem->fast_base), wb->compressed_size,
        wb->uncompressed_size, -1, -1, -1, -1);
    *prepared = *plan;
    prepared->weight_blob = scratch;
    *run_plan = prepared;
    return TIGRIS_EXEC_OK;
}

static uint32_t stage_dispatch_capacity(const tigris_plan_t *plan,
    const tigris_stage_t *stage, const tigris_mem_t *mem)
{
    uint32_t available = mem->fast_size - mem->fast_reserved;
    const tigris_weight_block_t *wb = find_weight_block(plan, (uint16_t)(stage - plan->stages));
    if (wb && wb->compressed_size > 0u) {
        uint32_t needed;
        if (!align_tensor_size(wb->uncompressed_size, &needed) || needed > available)
            return 0u;
        available -= needed;
    }
    return stage_fast_limit(plan, stage, available);
}

/* How a standalone stage runs: banded by its kind, or whole when arm is TIGRIS_BAND_KIND_NONE. */
typedef struct {
    tigris_band_kind_t arm;
    tigris_stage_band_t band;
    uint32_t total_io;
} stage_route_t;

static TIGRIS_NOINLINE tigris_exec_error_t route_stage(
    const tigris_plan_t *plan, const tigris_stage_t *stage, const tigris_mem_t *mem,
    int io_in_fast, stage_route_t *route)
{
    tigris_band_kind_t kind = tigris_stage_band_kind(plan, stage, &route->band);
    route->arm = kind;
    route->total_io = 0u;
    /* A Reshape or leading band is banded whatever the arena holds. */
    if (kind == TIGRIS_BAND_KIND_RESHAPE || kind == TIGRIS_BAND_KIND_LEADING)
        return TIGRIS_EXEC_OK;

    const uint16_t *sin  = tigris_stage_inputs(plan, stage);
    const uint16_t *sout = tigris_stage_outputs(plan, stage);
    uint32_t total_io = 0;
    uint8_t rank = 0;
    int common_tile_rank = 1;
    for (uint16_t i = 0; i < stage->inputs_count; i++) {
        total_io = add_allocation_bytes(total_io, plan->tensors[sin[i]].size_bytes);
        uint8_t tensor_rank = plan->tensors[sin[i]].ndim;
        if (rank == 0) rank = tensor_rank;
        if (tensor_rank != rank) common_tile_rank = 0;
    }
    for (uint16_t i = 0; i < stage->outputs_count; i++) {
        total_io = add_allocation_bytes(total_io, plan->tensors[sout[i]].size_bytes);
        uint8_t tensor_rank = plan->tensors[sout[i]].ndim;
        if (rank == 0) rank = tensor_rank;
        if (tensor_rank != rank) common_tile_rank = 0;
    }
    if (rank != 3 && rank != 4) common_tile_rank = 0;
    route->total_io = total_io;

    uint8_t tile_axis = stage_tile_axis(plan, stage);
    /* A row band, a banded transpose and a global reduction by input band on
     * their own terms, outside the stripe contract. */
    int own_band = kind == TIGRIS_BAND_KIND_ROWS || kind == TIGRIS_BAND_KIND_TRANSPOSE ||
                   kind == TIGRIS_BAND_KIND_BY_INPUT;
    /* The 2D contract is the stripe contract with a lone ConvTranspose
     * admitted, which exec_stage_tiled refuses on its own. */
    int axis1_tileable = own_band || (common_tile_rank &&
        tigris_stage_stripe_contract(plan, stage, rank, TIGRIS_TILE_AXIS_HW));
    /* A stage that reads its inputs in place runs whole. */
    int needs_tiling = !io_in_fast &&
        (common_tile_rank || kind == TIGRIS_BAND_KIND_ROWS || kind == TIGRIS_BAND_KIND_TRANSPOSE) &&
        total_io > stage_dispatch_capacity(plan, stage, mem);

    /* A serialized tile plan is a compiler promise: fail closed if an older
     * or malformed plan attaches it to a stage the runtime cannot tile
     * safely. Without a tile plan, unsupported stages retain ordinary
     * full-tensor execution and may use slow overflow. WIDTH and any other
     * axis value remain fail-closed. */
    if (needs_tiling && stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN &&
        ((tile_axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH && tile_axis != TIGRIS_TILE_AXIS_HW) ||
         (own_band && tile_axis != TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH) || !axis1_tileable))
        return TIGRIS_EXEC_ERR_TILE;

    int banded;
    if (own_band) {
        banded = needs_tiling && tile_axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    } else if (kind == TIGRIS_BAND_KIND_HW) {
        banded = needs_tiling && axis1_tileable;
    } else if (plan->header->version >= TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK) {
        /* A stage without a height record runs whole, within the stated peak. */
        banded = needs_tiling && axis1_tileable && tile_axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH;
    } else {
        /* Below schema 11 a rank-4 stage without a tile record also takes the
         * height stripe. Rank-3 length tiling requires the schema-v5 axis. */
        banded = needs_tiling && axis1_tileable &&
            (rank == 4 ||
             (plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS &&
              tile_axis == TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH));
    }
    if (!banded)
        route->arm = TIGRIS_BAND_KIND_NONE;
    return TIGRIS_EXEC_OK;
}

/* In place: inputs together at the top, above a room for the weights that holds no tensor. */
static int fast_floor(const tigris_plan_t *plan, const tigris_mem_t *mem,
                      uint32_t *floor, uint32_t *gap)
{
    uint32_t low = mem->fast_used;
    uint32_t total = 0u;
    uint32_t reserve = (plan->header->num_model_inputs > 0u)
        ? tigris_weight_decompression_overhead(plan) : 0u;
    int in_place = tigris_plan_io_in_fast(plan);
    uintptr_t base = (uintptr_t)mem->fast_base;
    for (uint8_t i = 0u; (in_place != 0) && (i < plan->header->num_model_inputs); i++) {
        uint16_t tidx = plan->model_inputs[i];
        uintptr_t p = (uintptr_t)mem->tensor_ptrs[tidx];
        uint32_t size = TILE_ALIGN_UP(plan->tensors[tidx].size_bytes);
        if (p < base + mem->fast_reserved || p >= base + mem->fast_used ||
            size > mem->fast_used - (uint32_t)(p - base)) {
            in_place = 0;
        }
        for (uint8_t j = 0u; (in_place != 0) && (j < i); j++) {
            uintptr_t q = (uintptr_t)mem->tensor_ptrs[plan->model_inputs[j]];
            uint32_t other = TILE_ALIGN_UP(plan->tensors[plan->model_inputs[j]].size_bytes);
            if (p < q + other && q < p + size)
                in_place = 0;
        }
        if ((in_place != 0) && ((uint32_t)(p - base) < low))
            low = (uint32_t)(p - base);
        total += size;
    }
    if ((in_place != 0) &&
        ((total != mem->fast_used - low) || (reserve > low - mem->fast_reserved)))
        in_place = 0;
    for (uint16_t t = 0u; (in_place != 0) && (t < mem->num_tensors); t++) {
        uintptr_t p = (uintptr_t)mem->tensor_ptrs[t];
        if ((p >= base + low - reserve) && (p < base + low))
            in_place = 0;
    }
    *gap = (in_place != 0) ? reserve : 0u;
    *floor = (in_place != 0) ? (low - reserve) : mem->fast_used;
    return in_place;
}

static int lists_tensor(const uint16_t *list, uint8_t count, uint16_t tidx)
{
    int found = 0;
    for (uint8_t i = 0u; i < count; i++) {
        if (list[i] == tidx)
            found = 1;
    }
    return found;
}

/* Pointers into the freed fast arena are stale; keep only what the run owes the caller. */
static void drop_free_fast_pointers(const tigris_plan_t *plan, tigris_mem_t *mem,
                                    uint32_t floor, int keep_inputs, int keep_outputs)
{
    uintptr_t low = (uintptr_t)mem->fast_base + floor;
    uintptr_t high = (uintptr_t)mem->fast_base + mem->fast_size;
    for (uint16_t i = 0u; i < mem->num_tensors; i++) {
        uintptr_t p = (uintptr_t)mem->tensor_ptrs[i];
        int kept = ((keep_inputs != 0) &&
                    lists_tensor(plan->model_inputs, plan->header->num_model_inputs, i)) ||
                   ((keep_outputs != 0) &&
                    lists_tensor(plan->model_outputs, plan->header->num_model_outputs, i));
        if ((p >= low) && (p < high) && (kept == 0))
            mem->tensor_ptrs[i] = NULL;
    }
}

tigris_exec_error_t tigris_run_with_state(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace_storage,
    size_t               workspace_size,
    void                *state,
    size_t               state_size)
{
    /* Stats start at zero, also for a run refused before its first stage. */
    if (stats) memset(stats, 0, sizeof(*stats));
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
     * floor so caller-owned bytes above fast_reserved cannot be overwritten,
     * except model inputs the stage reads in place, which the run consumes,
     * and the weight room below them. */
    const uint32_t caller_fast_used = mem->fast_used;
    const uint32_t caller_fast_reserved = mem->fast_reserved;
    if (caller_fast_used > mem->fast_size ||
        caller_fast_reserved > caller_fast_used)
        return TIGRIS_EXEC_ERR_MEM;
    uint32_t run_floor = caller_fast_used;
    uint32_t weight_gap = 0u;
    const int io_in_fast = fast_floor(plan, mem, &run_floor, &weight_gap);
    /* A plan that states its fast requirement is refused below it, before anything runs. */
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK) {
        uint32_t required = tigris_fast_arena_required(plan);
        if (required == UINT32_MAX || mem->fast_size - run_floor < required)
            return TIGRIS_EXEC_ERR_MEM;
    }
    drop_free_fast_pointers(plan, mem, run_floor, io_in_fast, 0);
    mem->fast_reserved = run_floor;

    tigris_exec_error_t result = TIGRIS_EXEC_OK;

    mem->stats = stats;
    mem->fast_peak = mem->fast_used;
    mem->slow_peak = mem->slow_used;
#ifdef TIGRIS_TRACE
    mem->trace_stage = TIGRIS_TRACE_NONE_U16;
#endif

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
    /* Model outputs are never freed. Subgraph outputs are freed by copy_out_of_graph. */
    for (uint8_t i = 0; i < plan->header->num_model_outputs; i++) {
        if (plan->model_outputs[i] < num_t)
            last_consumer[plan->model_outputs[i]] = UINT16_MAX;
    }
    for (uint16_t g = 1; g < plan->num_subgraphs; g++) {
        const tigris_subgraph_t *graph = &plan->subgraphs[g];
        for (uint16_t i = 0; i < graph->outputs_count; i++) {
            const uint16_t t = plan->index_pool[graph->outputs_off + i];
            if (t < num_t)
                last_consumer[t] = UINT16_MAX;
        }
    }

    /* Each variable enters as its value from the last run, placed in slow
     * memory as the caller places model inputs; its value on exit is kept
     * until it is stored back. */
    if (plan->num_state > 0u &&
        (state == NULL || state_size < plan->state_bytes)) {
        result = TIGRIS_EXEC_ERR_STATE;
        goto restore_fast_arena;
    }
    for (uint16_t i = 0; i < plan->num_state; i++) {
        const tigris_state_entry_t *entry = &plan->state_entries[i];
        if (tigris_mem_alloc_slow(mem, entry->input, entry->bytes) != TIGRIS_MEM_OK) {
            result = TIGRIS_EXEC_ERR_MEM;
            goto restore_fast_arena;
        }
        memcpy(mem->tensor_ptrs[entry->input], (const uint8_t *)state + entry->offset,
               entry->bytes);
        if (stats) stats->copy_bytes += entry->bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_COPY, TIGRIS_TRACE_POOL_SLOW,
            entry->input, TIGRIS_TRACE_NONE_U16,
            TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[entry->input], mem->slow_base),
            entry->offset, entry->bytes, -1, -1, -1, -1);
        if (entry->output != 0xFFFFu)
            last_consumer[entry->output] = UINT16_MAX;
    }

    /* The graph running and its control-flow frames live in the workspace:
     * flow[0] depth, flow[1] first stage, flow[2] end stage, flow[3] the
     * stage to continue at. */
    uint16_t *flow = workspace->frames;
    flow[0] = 0u;
    flow[1] = 0u;
    flow[2] = plan->num_subgraphs ? plan->subgraphs[0].num_stages : plan->header->num_stages;
    for (uint16_t s = 0; s < flow[2]; s++) {
        const tigris_stage_t *stage = &plan->stages[s];

        if (stage_control(plan, stage)) {
            TRACE_STAGE_BEGIN(mem, s, TIGRIS_TRACE_PATH_CONTROL);
            tigris_exec_error_t err = enter_control(plan, mem, flow, workspace->depth_capacity, s);
            if (err != TIGRIS_EXEC_OK) {
                result = err;
                goto restore_fast_arena;
            }
            s = flow[3];
            continue;
        }

        /* Per-stage weight decompression (skip for chain heads) */
        tigris_plan_t stage_plan;
        const tigris_plan_t *run_plan = plan;

        /* A chain runs from its head; any other stage on the path its band
         * kind, its record and the arena decide. */
        if (stage->chain_len >= 2 && stage->chain_id == s) {
            /* First stage of a chain - execute entire chain */
            tigris_exec_error_t err = exec_chain_tiled(
                run_plan, s, stage->chain_len, mem, kernel, user_ctx,
                workspace, stats);
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
            /* A chain member runs from its head; the loader keeps chains inside one graph. */
            result = TIGRIS_EXEC_ERR_TILE;
            goto restore_fast_arena;
        } else {
            /* The arm an ESP32-S3 error print names; a Reshape or leading band prints none. */
            static const char *const names[TIGRIS_BAND_KIND_STRIPE + 1] = {
                [TIGRIS_BAND_KIND_NONE] = "normal", [TIGRIS_BAND_KIND_ROWS] = "rows",
                [TIGRIS_BAND_KIND_TRANSPOSE] = "convert", [TIGRIS_BAND_KIND_BY_INPUT] = "reduce",
                [TIGRIS_BAND_KIND_HW] = "tiled2d", [TIGRIS_BAND_KIND_STRIPE] = "tiled",
            };
#ifdef TIGRIS_TRACE
            /* The path each arm traces: a leading band traces as rows. */
            static const tigris_trace_path_t paths[TIGRIS_BAND_KIND_STRIPE + 1] = {
                [TIGRIS_BAND_KIND_NONE] = TIGRIS_TRACE_PATH_NORMAL,
                [TIGRIS_BAND_KIND_RESHAPE] = TIGRIS_TRACE_PATH_RESHAPE,
                [TIGRIS_BAND_KIND_LEADING] = TIGRIS_TRACE_PATH_ROWS,
                [TIGRIS_BAND_KIND_ROWS] = TIGRIS_TRACE_PATH_ROWS,
                [TIGRIS_BAND_KIND_TRANSPOSE] = TIGRIS_TRACE_PATH_TRANSPOSE,
                [TIGRIS_BAND_KIND_BY_INPUT] = TIGRIS_TRACE_PATH_BY_INPUT,
                [TIGRIS_BAND_KIND_HW] = TIGRIS_TRACE_PATH_TILED_2D,
                [TIGRIS_BAND_KIND_STRIPE] = TIGRIS_TRACE_PATH_TILED,
            };
#endif
            stage_route_t route;
            result = route_stage(run_plan, stage, mem, io_in_fast, &route);
            if (result != TIGRIS_EXEC_OK)
                goto restore_fast_arena;
            TRACE_STAGE_BEGIN(mem, s, paths[route.arm]);
            result = prepare_stage_weights(plan, s, mem, weight_gap, &stage_plan, &run_plan, stats);
            if (result != TIGRIS_EXEC_OK)
                goto restore_fast_arena;
            switch (route.arm) {
            case TIGRIS_BAND_KIND_RESHAPE:
                result = exec_stage_tiled_reshape(run_plan, stage, mem, last_consumer, s, stats);
                break;
            case TIGRIS_BAND_KIND_LEADING:
                result = exec_stage_tiled_rows(
                    run_plan, stage, mem, kernel, user_ctx, workspace, last_consumer, s,
                    route.band.rows, route.band.mode, stats);
                break;
            case TIGRIS_BAND_KIND_ROWS:
                result = exec_stage_tiled_rows(
                    run_plan, stage, mem, kernel, user_ctx, workspace, last_consumer, s,
                    route.band.rows, 0, stats);
                break;
            case TIGRIS_BAND_KIND_TRANSPOSE:
                result = exec_stage_tiled_transpose(run_plan, stage, mem, kernel, user_ctx, stats);
                break;
            case TIGRIS_BAND_KIND_BY_INPUT:
                result = tigris_stage_tiles(run_plan, stage).input_channel_group != 0u
                    ? exec_stage_grouped_reduction(run_plan, stage, mem, kernel, user_ctx, stats)
                    : exec_stage_tiled_by_input(run_plan, stage, mem, kernel, user_ctx, stats);
                break;
            case TIGRIS_BAND_KIND_HW:
                result = exec_stage_tiled_2d(run_plan, stage, mem, kernel, user_ctx, workspace, stats);
                break;
            case TIGRIS_BAND_KIND_STRIPE:
                result = exec_stage_tiled(
                    run_plan, stage, mem, kernel, user_ctx, last_consumer, s, workspace, stats);
                break;
            default:
                result = exec_stage_normal(
                    run_plan, stage, mem, kernel, user_ctx, stats, workspace, io_in_fast);
                break;
            }
            if (result != TIGRIS_EXEC_OK) {
                if (names[route.arm] != NULL) {
                    TIGRIS_PLATFORM_DBG("S%u %s ERR=%d io=%lu fast=%lu/%lu slow=%lu/%lu\n",
                        s, names[route.arm], (int)result, (unsigned long)route.total_io,
                        (unsigned long)mem->fast_used, (unsigned long)mem->fast_size,
                        (unsigned long)mem->slow_used, (unsigned long)mem->slow_size);
                }
                goto restore_fast_arena;
            }
            if (stats) {
                if (route.arm == TIGRIS_BAND_KIND_NONE)
                    stats->stages_normal++;
                else
                    stats->stages_tiled++;
            }
        }

        /* Free dead tensors from slow and compact */
        free_dead_tensors(plan, mem, last_consumer, flow[1], s);

        /* Release stage/chain-local fast allocations before the next group,
         * retaining the caller's in-run reset floor. */
        mem->fast_reserved = run_floor;
        tigris_mem_reset_fast(mem);
        TRACE_STAGE_END(mem, stage->chain_len >= 2u ?
            (uint16_t)(s + 1u - stage->chain_len) : s);

        /* A finished subgraph steps its control-flow operator: into the next
         * subgraph it runs, or, once the operator is done, back to the main
         * graph after it. */
        if (flow[0] > 0u && (uint16_t)(s + 1u) == flow[2]) {
            tigris_exec_error_t err = unwind_control(plan, mem, last_consumer, flow, s);
            if (err != TIGRIS_EXEC_OK) {
                result = err;
                goto restore_fast_arena;
            }
            s = flow[3];
        }
    }

#ifdef TIGRIS_TRACE
    mem->trace_stage = TIGRIS_TRACE_NONE_U16;
#endif
    for (uint16_t i = 0; i < plan->num_state; i++) {
        const tigris_state_entry_t *entry = &plan->state_entries[i];
        if (entry->output == 0xFFFFu)
            continue;
        if (mem->tensor_ptrs[entry->output] == NULL) {
            result = TIGRIS_EXEC_ERR_MEM;
            goto restore_fast_arena;
        }
        memcpy((uint8_t *)state + entry->offset, mem->tensor_ptrs[entry->output], entry->bytes);
        if (stats) stats->copy_bytes += entry->bytes;
        TIGRIS_TRACE_EMIT(mem, TIGRIS_TRACE_COPY, TIGRIS_TRACE_POOL_NONE,
            entry->output, TIGRIS_TRACE_NONE_U16, entry->offset,
            TIGRIS_TRACE_OFFSET(mem->tensor_ptrs[entry->output], mem->slow_base),
            entry->bytes, -1, -1, -1, -1);
    }

restore_fast_arena:
    if (stats) stats->slow_peak = mem->slow_peak;
    mem->stats = NULL;
    drop_free_fast_pointers(plan, mem, run_floor, 0,
                            io_in_fast && result == TIGRIS_EXEC_OK);
    mem->fast_reserved = run_floor;
    tigris_mem_reset_fast(mem);
    mem->fast_reserved = caller_fast_reserved;
#ifdef TIGRIS_TRACE
    mem->trace_stage = TIGRIS_TRACE_NONE_U16;
#endif
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
    if (!workspace) {
        if (stats) memset(stats, 0, sizeof(*stats));
        return TIGRIS_EXEC_ERR_WORKSPACE;
    }
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
        case TIGRIS_EXEC_ERR_MEM:        return "arena too small for the plan";
        case TIGRIS_EXEC_ERR_KERNEL:     return "kernel callback error";
        case TIGRIS_EXEC_ERR_NO_STAGES:  return "plan has no stages";
        case TIGRIS_EXEC_ERR_TILE:       return "tiled execution error";
        case TIGRIS_EXEC_ERR_WORKSPACE:  return "executor workspace unavailable";
        case TIGRIS_EXEC_ERR_STATE:      return "state buffer missing or too small";
        default:                         return "unknown error";
    }
}

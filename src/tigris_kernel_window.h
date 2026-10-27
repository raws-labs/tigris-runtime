/**
 * @file tigris_kernel_window.h
 * @brief Element counts, window and resampling geometry, float attribute
 * reads, GELU and the dispatch of dtype-generic operators, shared by the
 * float and int8 kernels.
 *
 * Internal to src/. Not installed and not part of the public API.
 */
#ifndef TIGRIS_KERNEL_WINDOW_H
#define TIGRIS_KERNEL_WINDOW_H

#include "tigris.h"
#include "tigris_mem.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

float tigris_resize_explicit_scale(const tigris_plan_t *plan, const tigris_op_t *op, int axis);
float tigris_resize_scale(int input, int output, int convention);
void tigris_resize_position(int index, int input, int output, int convention, float explicit_scale,
                            float *position, int *lower, int *upper);
int tigris_resize_nearest(int index, int input, int output, int convention, float explicit_scale);
int32_t tigris_resize_scale_s8(int input, int output, int convention, float explicit_scale);
int tigris_resize_position_s8(int index, int input, int convention, int32_t scale,
                              int32_t *position, int32_t *lower, int32_t *upper);
int tigris_resize_source_rows(const tigris_plan_t *plan, const tigris_op_t *op,
                              int first, int end, int32_t *source_first);
int tigris_resize_max_rows(const tigris_plan_t *plan, const tigris_op_t *op, int rows);

/** Writes `count` elements read from `src` to `dst`, re-encoded as the kernel needs. */
typedef void (*tigris_pixel_copy_fn)(void *dst, const void *src, int count, const void *ctx);

/** Nearest-neighbor Resize of an NHWC tensor of `element`-byte values. Each
 * output pixel takes the channels of one input pixel, by memcpy, or through
 * `copy` when it is not NULL. Returns 0, or -1 for anything it cannot place. */
int tigris_resize_nearest_execute(const tigris_plan_t *plan, const tigris_op_t *op,
                                  tigris_mem_t *mem, uint32_t element,
                                  tigris_pixel_copy_fn copy, const void *ctx);

/** The extents [outer, reduced, inner] of a rank-3 input around the op's
 * single axis, with the band axis at the tile's input rows. Returns 0, or -1
 * for another rank, a missing or out-of-range axis, a band the op cannot run
 * on, an extent that is not positive, or an outer or inner count above
 * INT32_MAX.
 *
 * The plan names a serialized axis, so the operator reads the same whichever
 * order a tensor states its own axes in: the token axis of a sequence and the
 * channel axis of a feature map are both just one axis of [outer, reduced,
 * inner]. Collapsing the token axis is what a mean-pooled sequence head does. */
int tigris_rank3_axis_extents(
    const tigris_plan_t *plan, const tigris_op_t *op, uint16_t op_index, const tigris_mem_t *mem,
    int32_t *outer, int32_t *reduced, int32_t *inner);

/** Operators whose kernels take the element size from the tensors, reached
 * from the default case of both dispatchers. Returns -1 for any other op. */
int tigris_dispatch_generic(const tigris_plan_t *plan, const tigris_op_t *op,
                            uint16_t op_index, tigris_mem_t *mem);

/** Total number of elements in a tensor. */
uint32_t tigris_tensor_numel(const tigris_plan_t *plan, uint16_t tidx);

/** Tile-aware element count: uses tile dimensions when tiling active. */
uint32_t tigris_tile_numel(const tigris_plan_t *plan, uint16_t tidx, const tigris_mem_t *mem);

/** Per-row element count for pointwise row-offset pointer arithmetic:
 * the same width*channels term tigris_tile_numel multiplies by out_h. */
uint32_t tigris_tile_row_elems(const tigris_plan_t *plan, uint16_t tidx,
                               const tigris_mem_t *mem);

/** Element offsets of the current sub-range of a pointwise (height-preserving)
 * kernel's input and output: in_row_start and out_row_start rows. Both are 0
 * when tiling is inactive. */
void tigris_pointwise_row_offsets(const tigris_plan_t *plan, uint16_t tidx,
                                  const tigris_mem_t *mem, size_t *in_offset, size_t *out_offset);

/** Concat along one stored axis: each input contributes one contiguous run
 * per position, a run being its extent along the axis times every axis after
 * it, and the positions are the product of the axes ahead of it. Along the
 * last axis, an image's channels or a sequence's features, the run is the
 * input's own extent and a tile only shortens the axes ahead; any other axis
 * is cut whole and never tiled. */
typedef struct {
    uint8_t axis;
    int positions;
    int inner;      /* elements per index of the axis */
    int total;      /* output elements per position */
} tigris_concat_geometry_t;

/** Fills `g` for `op` under the current tile. Returns 0, or -1 for an axis
 * it cannot concatenate along. */
int tigris_concat_geometry(const tigris_plan_t *plan, const tigris_op_t *op,
                           const tigris_mem_t *mem, tigris_concat_geometry_t *g);

/** Reads a float attribute of the op into `value`. Returns 1 when present, 0
 * when absent (value untouched), or -1 for a payload that is not one float. */
int tigris_float_attribute(const tigris_plan_t *plan, uint16_t op_index,
                           uint8_t type, float *value);

/** GELU of x: the tanh approximation when `approximate` is not 0, else the
 * exact form through erfc. */
static inline float tigris_gelu(float x, uint8_t approximate)
{
    float value;
    if (approximate != 0u) {
        value = 0.5f * x * (1.0f + tanhf(
            0.7978845608028654f * (x + 0.044715f * x * x * x)));
    } else {
        value = 0.5f * x * erfcf(-0.7071067811865475f * x);
    }
    return value;
}

/** Geometry of a window operator on NHWC tensors (convolution, transposed
 * convolution, depthwise convolution, pooling): the shapes, the op's kernel,
 * stride, top and left pads and dilation (0 read as 1). While a tile is
 * active its heights, widths, top pad and row starts replace them, and its
 * left pad only on a 2D (HW) tile. The right and bottom edges stay implicit
 * in the input bounds. */
typedef struct {
    int32_t n;
    int32_t in_h;
    int32_t in_w;
    int32_t in_c;
    int32_t out_h;
    int32_t out_w;
    int32_t out_c;
    uint8_t kernel_h;
    uint8_t kernel_w;
    uint8_t stride_h;
    uint8_t stride_w;
    uint16_t pad_top;
    uint16_t pad_left;
    uint16_t dilation_h;
    uint16_t dilation_w;
    int32_t out_row_start;
    int32_t in_row_start;
} tigris_window_t;

/** Fills `w` for `op` under the current tile. */
static inline void tigris_window(const tigris_plan_t *plan, const tigris_op_t *op,
                                 const tigris_mem_t *mem, tigris_window_t *w)
{
    const int32_t *x = tigris_tensor_shape(plan, &plan->tensors[tigris_op_inputs(plan, op)[0]]);
    const int32_t *y = tigris_tensor_shape(plan, &plan->tensors[tigris_op_outputs(plan, op)[0]]);
    w->n = x[0];
    w->in_h = x[1];
    w->in_w = x[2];
    w->in_c = x[3];
    w->out_h = y[1];
    w->out_w = y[2];
    w->out_c = y[3];
    w->kernel_h = op->spatial.kernel_h;
    w->kernel_w = op->spatial.kernel_w;
    w->stride_h = op->spatial.stride_h;
    w->stride_w = op->spatial.stride_w;
    w->pad_top = op->spatial.pad_top;
    w->pad_left = op->spatial.pad_left;
    w->dilation_h = op->spatial.dilation_h;
    w->dilation_w = op->spatial.dilation_w;
    if (w->dilation_h == 0u) {
        w->dilation_h = 1u;
    }
    if (w->dilation_w == 0u) {
        w->dilation_w = 1u;
    }
    w->out_row_start = 0;
    w->in_row_start = 0;
    if (mem->tile.active != 0u) {
        w->in_h = mem->tile.in_h;
        w->out_h = mem->tile.out_h;
        w->in_w = mem->tile.in_w;
        w->out_w = mem->tile.out_w;
        w->pad_top = mem->tile.pad_top;
        if (mem->tile.width_tiled != 0u) {
            w->pad_left = mem->tile.pad_left;
        }
        w->out_row_start = mem->tile.out_row_start;
        w->in_row_start = mem->tile.in_row_start;
    }
}

/** The half-open range of tap indices whose input coordinate is inside the
 * tensor.
 *
 * The coordinate is base + k * dilation and is strictly increasing in k, so
 * the taps that land inside form one interval and the inner loops need no
 * bounds test. Computing the interval per output row and per output column
 * also stops it being recomputed for every channel, which the
 * channel-innermost loop order made the convolution kernels do.
 */
static inline void tap_range(
    int base, int extent, int taps, int dilation, int *first, int *last)
{
    int lo = 0;
    int hi = taps;
    int span = extent - 1 - base;
    if (base < 0) {
        lo = (-base + dilation - 1) / dilation;
        if (lo > taps)
            lo = taps;
    }
    if (span < 0) {
        hi = 0;
    } else {
        int bound = span / dilation + 1;
        if (bound < hi)
            hi = bound;
    }
    if (hi < lo)
        hi = lo;
    *first = lo;
    *last = hi;
}

#endif /* TIGRIS_KERNEL_WINDOW_H */

/**
 * @file tigris_kernel_window.h
 * @brief The sliding-window tap range shared by the float and int8 kernels.
 *
 * Internal to src/. Not installed and not part of the public API.
 */
#ifndef TIGRIS_KERNEL_WINDOW_H
#define TIGRIS_KERNEL_WINDOW_H

#include "tigris.h"

float tigris_resize_explicit_scale(const tigris_plan_t *plan, const tigris_op_t *op, int axis);
float tigris_resize_scale(int input, int output, int convention);
void tigris_resize_position(int index, int input, int output, int convention, float explicit_scale,
                            float *position, int *lower, int *upper);
int tigris_resize_nearest(int index, int input, int output, int convention, float explicit_scale);
int32_t tigris_resize_scale_s8(int input, int output, int convention, float explicit_scale);
int tigris_resize_position_s8(int index, int input, int convention, int32_t scale,
                              int32_t *position, int32_t *lower, int32_t *upper);
int tigris_resize_source_rows(const tigris_plan_t *plan, const tigris_op_t *op,
                              int first, int end, int *source_first);
int tigris_resize_max_rows(const tigris_plan_t *plan, const tigris_op_t *op, int rows);

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

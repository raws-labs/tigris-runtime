/**
 * @file tigris_kernel_window.h
 * @brief The sliding-window tap range shared by the float and int8 kernels.
 *
 * Internal to src/. Not installed and not part of the public API.
 */
#ifndef TIGRIS_KERNEL_WINDOW_H
#define TIGRIS_KERNEL_WINDOW_H

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

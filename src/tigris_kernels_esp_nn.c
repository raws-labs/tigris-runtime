/**
 * @file tigris_kernels_esp_nn.c
 * @brief ESP-NN accelerated int8 kernel adapter for TiGrIS runtime.
 *
 * Thin adapter mapping TiGrIS plan structures to ESP-NN API calls.
 * Unsupported ops fall back to the reference tigris_dispatch_kernel_s8().
 *
 * ESP-NN API reference: https://github.com/espressif/esp-nn
 * Data layout: NHWC (matches ESP-NN convention).
 *
 * ESP-NN calling convention (matches TFLM):
 *   - filter_dims.channels = 0 (not actual channel count)
 *   - dilation = {0, 0} encodes "no dilation" (NOT {1, 1})
 *   - Depthwise weights: [KH, KW, C] layout (HWC)
 *
 * CRITICAL: must #include "sdkconfig.h" before <esp_nn.h> so that
 * CONFIG_NN_OPTIMIZED routes to the S3 SIMD path (not ANSI C fallback).
 *
 * Memory: tigris_esp_nn_prepare() obtains platform-managed aligned workspace.
 * No workspace allocation occurs inside tigris_run().
 */

#ifdef TIGRIS_HAS_ESP_NN

#include "tigris_kernels_esp_nn.h"
#include "tigris_accel_policy.h"
#include "tigris_kernels_s8.h"

#include "sdkconfig.h"   /* CONFIG_NN_OPTIMIZED, CONFIG_IDF_TARGET_ESP32S3 */
#include <esp_nn.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef __XTENSA__
#include "esp_heap_caps.h"
#endif

/* Preparation-time platform-managed workspace */
/*
 * ESP-IDF allocations prefer PSRAM, with an optional internal-SRAM Conv
 * scratch for bandwidth. The buffers are independent of mem->fast_size and
 * live in process-lifetime adapter state.
 *
 * Each region is 16-byte aligned (ESP-NN SIMD requires this).
 * The scratch region gets +16 bytes slack because ESP-NN internally
 * sub-allocates with 16-byte alignment (see esp_nn_conv_esp32s3.c).
 *
 * Since ops execute sequentially, the same scratch buffer is shared
 * between conv and dw_conv (set via both esp_nn_set_*_scratch_buf).
 */
static void *s_scratch_buf  = NULL;   /* PSRAM fallback scratch (16-align) */
static int   s_scratch_size = 0;      /* allocated PSRAM scratch bytes */
static void *s_sram_scratch = NULL;   /* SRAM scratch (leftover from heap) */
static int   s_sram_scratch_size = 0; /* SRAM scratch bytes */
#ifdef __XTENSA__
static int s_conv_sram = 0, s_conv_psram = 0, s_conv_fallback = 0;
#endif
static void *s_dw_out_buf   = NULL;   /* DW output bounce (16-align) */
static void *s_pad_buf      = NULL;   /* asymmetric padding pre-pad */
static int   s_pad_size     = 0;
static int32_t *s_quant_buf = NULL;   /* scalar-to-per-channel expansion */
static uint32_t s_quant_channels = 0;

/* Helpers */

static inline int32_t get_mult(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off + ch];
}

static inline int32_t get_shft(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp, int ch)
{
    uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
    return plan->quant_data[page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off + ch];
}

static int esp_per_channel_quant(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp,
    uint32_t channels, quant_data_t *quant)
{
    if (!plan || !quant || channels == 0)
        return -1;

    if (qp && qp->num_channels == channels) {
        uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2
            ? 0u : qp->_pad * TIGRIS_QUANT_PAGE_ELEMS;
        quant->mult = (int32_t *)&plan->quant_data[
            page + qp->multiplier_off];
        quant->shift = (int32_t *)&plan->quant_data[page + qp->shift_off];
        return 0;
    }

    if ((qp && qp->num_channels != 1u) || !s_quant_buf ||
        channels > s_quant_channels)
        return -1;

    int32_t *mult = s_quant_buf;
    int32_t *shift = s_quant_buf + s_quant_channels;
    for (uint32_t channel = 0; channel < channels; channel++) {
        mult[channel] = qp ? get_mult(plan, qp, 0) : 1;
        shift[channel] = qp ? get_shft(plan, qp, 0) : 0;
    }
    quant->mult = mult;
    quant->shift = shift;
    return 0;
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void esp_nn_workspace_free(void *ptr)
{
#ifdef __XTENSA__
    heap_caps_free(ptr);
#else
    free(ptr);
#endif
}

void tigris_esp_nn_deinit(void)
{
    /* ESP-NN stores these pointers globally too; clear them before freeing
     * adapter-owned memory so a later dispatch cannot observe stale storage. */
    esp_nn_set_conv_scratch_buf(NULL);
    esp_nn_set_depthwise_conv_scratch_buf(NULL);
    esp_nn_workspace_free(s_scratch_buf);
    esp_nn_workspace_free(s_sram_scratch);
    esp_nn_workspace_free(s_dw_out_buf);
    esp_nn_workspace_free(s_pad_buf);
    esp_nn_workspace_free(s_quant_buf);
    s_scratch_buf = NULL;
    s_scratch_size = 0;
    s_sram_scratch = NULL;
    s_sram_scratch_size = 0;
    s_dw_out_buf = NULL;
    s_pad_buf = NULL;
    s_pad_size = 0;
    s_quant_buf = NULL;
    s_quant_channels = 0;
#ifdef __XTENSA__
    s_conv_sram = s_conv_psram = s_conv_fallback = 0;
#endif
}

/* tigris_esp_nn_prepare */

int tigris_esp_nn_prepare(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem)
{
    if (!plan || !mem) return -1;

    /* The adapter owns its platform workspace. Replacing it here makes a
     * repeat prepare deterministic and avoids leaking PSRAM/SRAM reservations. */
    tigris_esp_nn_deinit();

    int max_scratch = 0;
    int max_dw_out  = 0;
    int max_pad     = 0;
    int max_quant_channels = 0;

    for (uint16_t i = 0; i < plan->header->num_ops; i++) {
        const tigris_op_t *op = &plan->ops[i];

        /* ESP-NN encodes only unit dilation. Dilated Conv/Depthwise execute in
         * s8_ref, so do not size or allocate vendor scratch for them. */
        if (tigris_accel_pre_route(TIGRIS_ACCEL_ESP_NN, plan, op, 0) ==
            TIGRIS_ACCEL_ROUTE_S8_REF)
            continue;

        if (op->op_type == TIGRIS_OP_CONV) {
            const uint16_t *ins  = tigris_op_inputs(plan, op);
            const uint16_t *outs = tigris_op_outputs(plan, op);
            const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
            const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

            int IH = x_shape[1], IW = x_shape[2], IC = x_shape[3];
            int OH = y_shape[1], OW = y_shape[2], OC = y_shape[3];
            int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

            int pt = op->spatial.pad_top,  pb = op->spatial.pad_bottom;
            int pl = op->spatial.pad_left, pr = op->spatial.pad_right;
            int asymmetric = (pt != pb) || (pl != pr);

            int conv_IH = IH, conv_IW = IW;
            int pad_h = pt, pad_w = pl;

            if (asymmetric) {
                uint32_t required;
                if (!tigris_accel_esp_pad_workspace_fits(
                        IH, IW, IC, (uint16_t)pt, (uint16_t)pb,
                        (uint16_t)pl, (uint16_t)pr,
                        UINT32_MAX, &required) ||
                    required > (uint32_t)(INT_MAX - 15))
                    return -1;
                int PH = IH + pt + pb;
                int PW = IW + pl + pr;
                max_pad = MAX(max_pad, (int)required);
                conv_IH = PH;
                conv_IW = PW;
                pad_h = 0;
                pad_w = 0;
            }

            data_dims_t id = { .width = conv_IW, .height = conv_IH,
                               .channels = IC, .extra = 1 };
            data_dims_t fd = { .width = KW, .height = KH,
                               .channels = 0, .extra = 0 };
            data_dims_t od = { .width = OW, .height = OH,
                               .channels = OC, .extra = 1 };
            conv_params_t cp = {
                .stride  = { .width = op->spatial.stride_w,
                             .height = op->spatial.stride_h },
                .padding = { .width = pad_w, .height = pad_h },
                .dilation = { .width = 0, .height = 0 },
            };

            int s = esp_nn_get_conv_scratch_size(&id, &fd, &od, &cp);
#ifdef __XTENSA__
            if (s > 100000 || i < 5)
                printf("  Op%u Conv %dx%dx%d->%dx%dx%d k%dx%d: scratch=%d\n",
                       i, IH,IW,IC, OH,OW,OC, KH,KW, s);
#endif
            max_scratch = MAX(max_scratch, s);
            max_quant_channels = MAX(max_quant_channels, OC);

        } else if (op->op_type == TIGRIS_OP_DEPTHWISE) {
            const uint16_t *ins  = tigris_op_inputs(plan, op);
            const uint16_t *outs = tigris_op_outputs(plan, op);
            const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
            const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

            int IH = x_shape[1], IW = x_shape[2], C = x_shape[3];
            int OH = y_shape[1], OW = y_shape[2];
            int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

            /* When tiling is active, the executor tiles along H.
             * Cap OH at the budget-derived tile height to avoid
             * oversizing the bounce buffer for the full tensor. */
            int tile_oh = OH;
            if (plan->header->num_stages > 1 && OH > 1) {
                /* Rough estimate: one row of output = OW * C bytes.
                 * Budget can hold at most budget / (2 * OW * C) output rows
                 * (factor of 2: input + output tile both in arena). */
                int row_cost = OW * C * 2;
                if (row_cost > 0) {
                    int max_h = (int)(mem->fast_size / (uint32_t)row_cost);
                    if (max_h < 1) max_h = 1;
                    if (max_h < tile_oh) tile_oh = max_h;
                }
            }

            data_dims_t id = { .width = IW, .height = IH,
                               .channels = C, .extra = 1 };
            data_dims_t fd = { .width = KW, .height = KH,
                               .channels = 0, .extra = 0 };
            data_dims_t od = { .width = OW, .height = tile_oh,
                               .channels = C, .extra = 1 };
            dw_conv_params_t dp = {
                .ch_mult  = 1,
                .stride   = { .width = op->spatial.stride_w,
                              .height = op->spatial.stride_h },
                .padding  = { .width = op->spatial.pad_left,
                              .height = op->spatial.pad_top },
                .dilation = { .width = 0, .height = 0 },
            };

            int s = esp_nn_get_depthwise_conv_scratch_size(&id, &fd, &od, &dp);
            max_scratch = MAX(max_scratch, s);
            max_quant_channels = MAX(max_quant_channels, C);

            /* DW output bounce buffer: ee.vst.128.xp needs 16-byte alignment.
             * Size for tile height, not full tensor height. */
            max_dw_out = MAX(max_dw_out, tile_oh * OW * C);
        }
    }

    /* Round each sub-region up to 16-byte multiples so that
     * consecutive regions maintain 16-byte alignment. */
    uint32_t dw_aligned  = (uint32_t)((max_dw_out  + 15) & ~15);
    uint32_t pad_aligned = (uint32_t)((max_pad     + 15) & ~15);
    if (max_quant_channels < 0 ||
        (uint32_t)max_quant_channels >
            (UINT32_MAX - 15u) / (2u * sizeof(int32_t)))
        goto fail;
    uint32_t quant_aligned = (
        (uint32_t)max_quant_channels * 2u * sizeof(int32_t) + 15u) & ~15u;

    /* Allocate all ESP-NN buffers from heap (PSRAM) to avoid stealing
     * from the tensor arena, which the plan compiler sized for tensors. */
#ifdef __XTENSA__
    #define ESP_NN_ALLOC(sz) \
        heap_caps_aligned_alloc(16, (sz), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
#else
    #define ESP_NN_ALLOC(sz) aligned_alloc(16, (sz))
#endif

    if (dw_aligned > 0) {
        s_dw_out_buf = ESP_NN_ALLOC(dw_aligned);
        if (!s_dw_out_buf) goto fail;
    }
    if (pad_aligned > 0) {
        s_pad_buf  = ESP_NN_ALLOC(pad_aligned);
        /* Allocation failure is non-fatal: adapt_conv2d verifies this exact
         * workspace before changing vendor padding and falls back to s8_ref. */
        s_pad_size = s_pad_buf ? (int)pad_aligned : 0;
    }
    if (quant_aligned > 0) {
        s_quant_buf = ESP_NN_ALLOC(quant_aligned);
        if (!s_quant_buf) goto fail;
        s_quant_channels = quant_aligned / (2u * sizeof(int32_t));
    }
    if (max_scratch > 0) {
        uint32_t scr_bytes = (uint32_t)((max_scratch + 16 + 15) & ~15);
        s_scratch_buf = ESP_NN_ALLOC(scr_bytes);
        s_scratch_size = s_scratch_buf ? (int)scr_bytes : 0;
    }

    /* Try to allocate SRAM scratch from leftover internal RAM.
     * SRAM scratch gives ~10x bandwidth vs PSRAM for SIMD kernels.
     * This uses whatever SRAM is left AFTER the arena allocation,
     * so it doesn't break the compiler's budget. */
#ifdef __XTENSA__
    if (max_scratch > 0) {
        uint32_t sram_free = heap_caps_get_largest_free_block(
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        /* Leave 8K headroom for stack/system */
        uint32_t sram_avail = (sram_free > 8192) ? sram_free - 8192 : 0;
        uint32_t sram_want = (uint32_t)max_scratch;
        if (sram_want > sram_avail) sram_want = sram_avail;
        sram_want = (sram_want + 15) & ~15u;
        if (sram_want >= 1024) {
            s_sram_scratch = heap_caps_aligned_alloc(
                16, sram_want, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            if (s_sram_scratch)
                s_sram_scratch_size = (int)sram_want;
        }
    }
#endif

    if (s_scratch_buf) {
        esp_nn_set_conv_scratch_buf(s_scratch_buf);
        esp_nn_set_depthwise_conv_scratch_buf(s_scratch_buf);
    } else if (s_sram_scratch) {
        esp_nn_set_conv_scratch_buf(s_sram_scratch);
        esp_nn_set_depthwise_conv_scratch_buf(s_sram_scratch);
    }

#ifdef __XTENSA__
    printf("  ESP-NN: psram=%d sram=%d max_needed=%d dw=%d pad=%d\n",
           s_scratch_size, s_sram_scratch_size, max_scratch,
           (int)dw_aligned, s_pad_size);
    s_conv_sram = s_conv_psram = s_conv_fallback = 0;
#endif

#undef ESP_NN_ALLOC
    return 0;

fail:
#undef ESP_NN_ALLOC
    tigris_esp_nn_deinit();
    return -1;
}

/* Asymmetric padding helper */
/*
 * ESP-NN internally pads inputs symmetrically (pad_top == pad_bottom,
 * pad_left == pad_right).  When the plan specifies asymmetric padding
 * (e.g. pad_top=4, pad_bottom=5), we must pre-pad the input ourselves
 * and pass the padded buffer to ESP-NN with padding=0.
 *
 * Buffer was pre-allocated in tigris_esp_nn_prepare().
 */
static const int8_t *asymmetric_pad_input(
    const int8_t *input, int IH, int IW, int IC,
    int pt, int pb, int pl, int pr, int8_t fill_val,
    int *out_h, int *out_w)
{
    uint32_t needed;
    if (!s_pad_buf || s_pad_size <= 0 ||
        !tigris_accel_esp_pad_workspace_fits(
            IH, IW, IC, (uint16_t)pt, (uint16_t)pb,
            (uint16_t)pl, (uint16_t)pr,
            (uint32_t)s_pad_size, &needed))
        return NULL;

    int PH = IH + pt + pb;
    int PW = IW + pl + pr;

    memset(s_pad_buf, fill_val, needed);
    int8_t *dst = (int8_t *)s_pad_buf;
    for (int h = 0; h < IH; h++) {
        memcpy(dst + ((h + pt) * PW + pl) * IC,
               input + (h * IW) * IC,
               (size_t)(IW * IC));
    }
    *out_h = PH;
    *out_w = PW;
    return (const int8_t *)dst;
}

/* Conv2D */

static int adapt_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int IH = x_shape[1], IW = x_shape[2], IC = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2], OC = y_shape[3];
    int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

    /* Use tile dimensions when tiling is active */
    int pt = op->spatial.pad_top,    pb = op->spatial.pad_bottom;
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
        pt = mem->tile.pad_top;
        pb = mem->tile.pad_bottom;
    }

    /* Line-buffer roll: fold out_row_start into the output pointer and
     * (out_row_start*stride - in_row_start) into the input pointer, matching
     * kern_conv2d_s8 and the CMSIS-NN adapter. Rolled tiles are interior
     * (pt==pb==0), so the asymmetric-pad path below is not taken. */
    if (mem->tile.active &&
        (mem->tile.out_row_start != 0 || mem->tile.in_row_start != 0)) {
        int delta = mem->tile.out_row_start * op->spatial.stride_h
                  - mem->tile.in_row_start;
        X  += delta * IW * IC;
        Y  += mem->tile.out_row_start * OW * OC;
        IH -= delta;
    }

    int32_t in_offset = in_qp ? -in_qp->zero_point : 0;

    /* Handle asymmetric padding */
    int pl = op->spatial.pad_left,   pr = op->spatial.pad_right;
    int asymmetric = (pt != pb) || (pl != pr);

    const int8_t *conv_input = X;
    int conv_IH = IH, conv_IW = IW;
    int pad_h = pt, pad_w = pl;

    if (asymmetric) {
        const int8_t *padded = asymmetric_pad_input(
            X, IH, IW, IC, pt, pb, pl, pr,
            (int8_t)(-in_offset), &conv_IH, &conv_IW);
        if (!padded) {
#ifdef __XTENSA__
            s_conv_fallback++;
#endif
            return -99;
        }
        conv_input = padded;
        pad_h = 0;
        pad_w = 0;
    }

    data_dims_t input_dims  = { .width = conv_IW, .height = conv_IH,
                                .channels = IC, .extra = 1 };
    data_dims_t filter_dims = { .width = KW, .height = KH, .channels = 0, .extra = 0 };
    data_dims_t output_dims = { .width = OW, .height = OH, .channels = OC, .extra = 1 };

    quant_data_t quant;
    if (OC <= 0 || esp_per_channel_quant(
            plan, out_qp, (uint32_t)OC, &quant) != 0)
        return -99;

    conv_params_t params = {
        .in_offset  = in_offset,
        .out_offset = out_qp ? out_qp->zero_point : 0,
        .stride     = { .width = op->spatial.stride_w,  .height = op->spatial.stride_h },
        .padding    = { .width = pad_w, .height = pad_h },
        .dilation   = { .width = 0, .height = 0 },
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    /* Two-tier scratch: prefer SRAM (fast), fall back to PSRAM, then s8_ref. */
    int needed = esp_nn_get_conv_scratch_size(&input_dims, &filter_dims,
                                              &output_dims, &params);
    if (needed <= s_sram_scratch_size && s_sram_scratch) {
        esp_nn_set_conv_scratch_buf(s_sram_scratch);
#ifdef __XTENSA__
        s_conv_sram++;
#endif
    } else if (needed <= s_scratch_size && s_scratch_buf) {
        esp_nn_set_conv_scratch_buf(s_scratch_buf);
#ifdef __XTENSA__
        s_conv_psram++;
#endif
    } else {
#ifdef __XTENSA__
        s_conv_fallback++;
#endif
        return -99;  /* fallback to s8_ref */
    }

    esp_nn_conv_s8(&input_dims, conv_input, &filter_dims, W, B,
                   &output_dims, Y, &params, &quant);
    return 0;
}

/* Depthwise Conv2D */

static int adapt_depthwise_conv2d(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int IH = x_shape[1], IW = x_shape[2], C = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2];
    int KH = op->spatial.kernel_h, KW = op->spatial.kernel_w;

    /* Use tile dimensions when tiling is active */
    if (mem->tile.active) {
        IH = mem->tile.in_h;
        OH = mem->tile.out_h;
    }

    /* Line-buffer roll: fold out_row_start into the output pointer and
     * (out_row_start*stride - in_row_start) into the input pointer, matching
     * kern_conv2d_s8 and the CMSIS-NN adapter. */
    if (mem->tile.active &&
        (mem->tile.out_row_start != 0 || mem->tile.in_row_start != 0)) {
        int delta = mem->tile.out_row_start * op->spatial.stride_h
                  - mem->tile.in_row_start;
        X  += delta * IW * C;
        Y  += mem->tile.out_row_start * OW * C;
        IH -= delta;
    }

    data_dims_t input_dims  = { .width = IW, .height = IH, .channels = C, .extra = 1 };
    data_dims_t filter_dims = { .width = KW, .height = KH, .channels = 0, .extra = 0 };
    data_dims_t output_dims = { .width = OW, .height = OH, .channels = C, .extra = 1 };

    quant_data_t quant;
    if (C <= 0 || esp_per_channel_quant(
            plan, out_qp, (uint32_t)C, &quant) != 0)
        return -99;

    /* Depthwise weights are already in [KH, KW, C] (HWC) layout from
     * the compiler - matches ESP-NN's expected layout. */

    int32_t in_offset = in_qp  ? -in_qp->zero_point  : 0;
    int32_t out_offset = out_qp ? out_qp->zero_point : 0;

    dw_conv_params_t params = {
        .in_offset  = in_offset,
        .out_offset = out_offset,
        .ch_mult    = 1,
        .stride     = { .width = op->spatial.stride_w,  .height = op->spatial.stride_h },
        .padding    = { .width = op->spatial.pad_left,   .height = mem->tile.active ? mem->tile.pad_top : op->spatial.pad_top },
        .dilation   = { .width = 0, .height = 0 },
        .activation = { .min = op->act_min, .max = op->act_max },
    };

    /* Check scratch fits */
    if (!s_scratch_buf || !s_dw_out_buf)
        return -99;

    /* DW output -> 16-byte aligned bounce buffer, copy back to arena */
    int y_bytes = OH * OW * C;
    int8_t *dw_Y = (int8_t *)s_dw_out_buf;

    esp_nn_depthwise_conv_s8(&input_dims, X, &filter_dims, W, B,
                             &output_dims, dw_Y, &params, &quant);

    memcpy(Y, dw_Y, (size_t)y_bytes);
    return 0;
}

/* Fully Connected */

static int adapt_fully_connected(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t  *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t        *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);
    const int8_t  *W = (const int8_t *)tigris_op_weight(plan, op);
    const int32_t *B = (const int32_t *)tigris_op_bias(plan, op);

    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);
    const tigris_tensor_t *y_tensor = &plan->tensors[outs[0]];

    const tigris_quant_param_t *in_qp  = tigris_tensor_quant(plan, &plan->tensors[ins[0]]);
    const tigris_quant_param_t *out_qp = tigris_tensor_quant(plan, &plan->tensors[outs[0]]);

    int OC = (y_tensor->ndim >= 2) ? y_shape[1] : y_shape[0];

    const tigris_tensor_t *x_tensor = &plan->tensors[ins[0]];
    const int32_t *x_shape = tigris_tensor_shape(plan, x_tensor);
    uint32_t x_numel = 1;
    for (uint8_t i = 0; i < x_tensor->ndim; i++)
        x_numel *= (uint32_t)x_shape[i];
    int N = (y_tensor->ndim >= 2) ? y_shape[0] : 1;
    uint16_t row_len = (uint16_t)(x_numel / (uint32_t)N);

    int32_t in_offset  = in_qp  ? -in_qp->zero_point  : 0;
    int32_t out_offset = out_qp ? out_qp->zero_point : 0;

    /* With TIGRIS_TENSOR_ALIGN >= 8 on Xtensa, FC input is already
     * 8-byte aligned as required by ESP-NN S3 SIMD (ee.vld.l.64.ip). */

    if (out_qp && out_qp->num_channels > 1) {
        uint32_t page = plan->header->version == TIGRIS_SCHEMA_VERSION_V2
            ? 0u : out_qp->_pad * TIGRIS_QUANT_PAGE_ELEMS;
        const int32_t *mult_arr = &plan->quant_data[
            page + out_qp->multiplier_off];
        const int32_t *shift_arr = &plan->quant_data[
            page + out_qp->shift_off];
        for (int n = 0; n < N; n++) {
            esp_nn_fully_connected_per_ch_s8(
                X + n * row_len, in_offset, row_len,
                W, 0, B,
                Y + n * OC, (uint16_t)OC,
                out_offset, shift_arr, mult_arr,
                op->act_min, op->act_max);
        }
    } else {
        int32_t m = out_qp ? get_mult(plan, out_qp, 0) : 1;
        int32_t s = out_qp ? get_shft(plan, out_qp, 0) : 0;
        for (int n = 0; n < N; n++) {
            esp_nn_fully_connected_s8(
                X + n * row_len, in_offset, row_len,
                W, 0, B,
                Y + n * OC, (uint16_t)OC,
                out_offset, s, m,
                op->act_min, op->act_max);
        }
    }
    return 0;
}

/* Average Pool */

static int adapt_avg_pool(
    const tigris_plan_t *plan, const tigris_op_t *op, tigris_mem_t *mem)
{
    const uint16_t *ins  = tigris_op_inputs(plan, op);
    const uint16_t *outs = tigris_op_outputs(plan, op);

    const int8_t *X = (const int8_t *)tigris_mem_tensor_ptr(mem, ins[0]);
    int8_t       *Y = (int8_t *)tigris_mem_tensor_ptr(mem, outs[0]);

    const int32_t *x_shape = tigris_tensor_shape(plan, &plan->tensors[ins[0]]);
    const int32_t *y_shape = tigris_tensor_shape(plan, &plan->tensors[outs[0]]);

    int IH = x_shape[1], IW = x_shape[2], C = x_shape[3];
    int OH = y_shape[1], OW = y_shape[2];

    int N = x_shape[0];
    for (int n = 0; n < N; n++) {
        esp_nn_avg_pool_s8(
            X + n * IH * IW * C,
            (uint16_t)IW, (uint16_t)IH,
            Y + n * OH * OW * C,
            (uint16_t)OW, (uint16_t)OH,
            (uint16_t)op->spatial.stride_w, (uint16_t)op->spatial.stride_h,
            (uint16_t)op->spatial.kernel_w, (uint16_t)op->spatial.kernel_h,
            (uint16_t)op->spatial.pad_left, (uint16_t)op->spatial.pad_top,
            op->act_min, op->act_max,
            (uint16_t)C);
    }
    return 0;
}

/* Stats */

void tigris_esp_nn_print_conv_stats(void)
{
#ifdef __XTENSA__
    printf("Conv dispatch: sram=%d psram=%d fallback=%d "
           "(sram_scratch=%d psram_scratch=%d)\n",
           s_conv_sram, s_conv_psram, s_conv_fallback,
           s_sram_scratch_size, s_scratch_size);
#endif
}

/* Dispatch */

int tigris_dispatch_kernel_esp_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx)
{
    /* Pre-route dilated Conv/Depthwise and pooling variants whose tiling or
     * quantization cannot be represented by the ESP-NN adapters. This runs
     * before entering an adapter, where those semantics would be lost. */
    int handled = 0;
    int route_rc = tigris_accel_try_s8_ref(
        TIGRIS_ACCEL_ESP_NN, plan, op, op_index, mem, user_ctx, &handled);
    if (handled) {
#ifdef __XTENSA__
        if (op->op_type == TIGRIS_OP_CONV)
            s_conv_fallback++;
#endif
        return route_rc;
    }

    int rc;
    switch ((tigris_op_type_t)op->op_type) {
    case TIGRIS_OP_CONV:        rc = adapt_conv2d(plan, op, mem); break;
    case TIGRIS_OP_DEPTHWISE:   rc = adapt_depthwise_conv2d(plan, op, mem); break;
    case TIGRIS_OP_FULLY_CONN:  rc = adapt_fully_connected(plan, op, mem); break;
    case TIGRIS_OP_AVG_POOL:    rc = adapt_avg_pool(plan, op, mem); break;
    /* Global avg pool: s8_ref uses float rescale (in_scale/out_scale)
     * that esp_nn_avg_pool_s8 doesn't support - fall back. */
    case TIGRIS_OP_GLOBAL_AVG:  rc = -99; break;
    default:
        rc = -99; break;
    }

    /* Fallback to s8_ref for unsupported ops */
    if (rc == -99)
        return tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);
    return rc;
}

#else
/* ISO C forbids empty translation units */
typedef int tigris_kernels_esp_nn_unused_;
#endif /* TIGRIS_HAS_ESP_NN */

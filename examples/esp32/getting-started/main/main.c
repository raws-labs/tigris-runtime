/*
 * TiGrIS ESP-IDF getting-started example: a U-Net that does NOT fit.
 *
 * Runs a real 256x256 int8 U-Net (encoder-decoder segmentation, 17 ops) on an
 * ESP32-S3. Its largest single activation is ~1.19 MiB and its naive peak is
 * ~2.38 MiB, so an arena-based runtime (e.g. TFLite Micro) OOMs against internal
 * SRAM. TiGrIS 2D-tiles the stages and spills the long-lived skips to PSRAM, so
 * the working set fits a 232 KiB fast arena. The plan is embedded in the app; no
 * partition flashing or extra steps are needed.
 *
 * Expected serial output ends with `SELF_CHECK: PASS` and `TIGRIS_DONE`.
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "hal/wdt_hal.h"
#include "soc/timer_group_reg.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/rtc_cntl_reg.h"

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#ifdef TIGRIS_HAS_ESP_NN
#include "tigris_kernels_esp_nn.h"
#endif

static const char *TAG = "tigris-example";

/* The compiled plan and the golden output, embedded by the build (see the
 * EMBED_FILES entries in main/CMakeLists.txt). */
extern const uint8_t unet_tgrs_start[] asm("_binary_unet_tgrs_start");
extern const uint8_t unet_tgrs_end[]   asm("_binary_unet_tgrs_end");
extern const uint8_t unet_ref_start[]  asm("_binary_unet_ref_bin_start");
extern const uint8_t unet_ref_end[]    asm("_binary_unet_ref_bin_end");

/* The int8 kernels track the ORT float oracle (unet_ref.bin) to within a few
 * LSB from accumulated requant rounding, so the self-check is statistical, not
 * bit-exact: the mean abs difference must stay well under 1 LSB and no single
 * element may drift far. (Reference host s8_ref run: mean 0.52, max 3.) */
#define SELF_CHECK_MAX_ABS 10
#define SELF_CHECK_MEAN    1.0f

/* -- Watchdog suppression --------------------------------------------------
 * The s8_ref decoder stages are slow, so keep every watchdog fed/disabled from
 * a high-priority background task (some are re-armed by FreeRTOS/bootloader). */
static void feed_and_disable_all_wdts(void) {
    wdt_hal_context_t ctx;
    for (int tg = 0; tg < 2; tg++) {
        REG_WRITE(TIMG_WDTWPROTECT_REG(tg), TIMG_WDT_WKEY_VALUE);
        REG_WRITE(TIMG_WDTFEED_REG(tg), 1);
        REG_WRITE(TIMG_WDTWPROTECT_REG(tg), 0);
        wdt_hal_init(&ctx, tg == 0 ? WDT_MWDT0 : WDT_MWDT1, 0, false);
        wdt_hal_write_protect_disable(&ctx);
        wdt_hal_set_flashboot_en(&ctx, false);
        wdt_hal_disable(&ctx);
        wdt_hal_write_protect_enable(&ctx);
    }
    wdt_hal_init(&ctx, WDT_RWDT, 0, false);
    wdt_hal_write_protect_disable(&ctx);
    wdt_hal_set_flashboot_en(&ctx, false);
    wdt_hal_disable(&ctx);
    wdt_hal_write_protect_enable(&ctx);
    REG_WRITE(RTC_CNTL_SWD_WPROTECT_REG, RTC_CNTL_SWD_WKEY_VALUE);
    SET_PERI_REG_MASK(RTC_CNTL_SWD_CONF_REG, RTC_CNTL_SWD_AUTO_FEED_EN);
    REG_WRITE(RTC_CNTL_SWD_WPROTECT_REG, 0);
}

static void wdt_killer_task(void *arg) {
    (void)arg;
    for (;;) { feed_and_disable_all_wdts(); vTaskDelay(pdMS_TO_TICKS(500)); }
}

/* Called by the runtime between tiles/stages on long runs. */
void tigris_feed_wdt(void) { feed_and_disable_all_wdts(); }

/* Largest single activation in the plan — the tensor an arena runtime must hold
 * whole, and the reason TFLM OOMs where TiGrIS tiles. */
static uint32_t largest_tensor_bytes(const tigris_plan_t *plan) {
    uint32_t mx = 0;
    for (uint16_t i = 0; i < plan->header->num_tensors; i++)
        if (plan->tensors[i].size_bytes > mx) mx = plan->tensors[i].size_bytes;
    return mx;
}

void app_main(void) {
    feed_and_disable_all_wdts();
    xTaskCreatePinnedToCore(wdt_killer_task, "wdt_kill", 2048, NULL,
                            configMAX_PRIORITIES - 1, NULL, 1);

    printf("\n=== TiGrIS getting-started: U-Net that does not fit ===\n\n");

    /* 1. Load the embedded plan. */
    uint32_t plan_size = (uint32_t)(unet_tgrs_end - unet_tgrs_start);
    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(unet_tgrs_start, plan_size, &plan);
    if (perr != TIGRIS_OK) {
        ESP_LOGE(TAG, "plan load failed: %s", tigris_error_str(perr));
        return;
    }

    uint32_t budget      = plan.header->budget;
    uint32_t largest     = largest_tensor_bytes(&plan);
    uint8_t  in_dtype    = plan.tensors[plan.model_inputs[0]].dtype;
    int      is_quant    = (in_dtype == 3);

    printf("Model:            %s\n", tigris_model_name(&plan));
    printf("Ops / stages:     %u / %u\n", plan.header->num_ops,
           plan.header->num_stages);
    printf("Plan (flash):     %.1f KiB\n", plan_size / 1024.0f);
    printf("Fast arena:       %.0f KiB\n", budget / 1024.0f);
    printf("Largest tensor:   %.2f MiB  <- an arena runtime must hold this whole\n",
           largest / (1024.0f * 1024.0f));
    printf("                  (TFLite Micro OOMs on it against a 256 KiB arena;\n");
    printf("                   TiGrIS tiles it and spills skips to PSRAM)\n\n");

    /* 2. Two-tier arena: fast = internal SRAM (the budget), slow = PSRAM. */
#if !CONFIG_SPIRAM
    ESP_LOGE(TAG, "this example needs an ESP32-S3 with PSRAM (enable CONFIG_SPIRAM)");
    return;
#endif
    uint32_t fast_size = budget + tigris_weight_decompression_overhead(&plan);
    void *fast_buf = heap_caps_malloc(fast_size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint32_t slow_size = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    if (slow_size > 2 * 1024 * 1024) slow_size -= 2 * 1024 * 1024; /* ESP-NN scratch */
    void *slow_buf = heap_caps_malloc(slow_size, MALLOC_CAP_SPIRAM);
    void **tensor_ptrs = heap_caps_calloc(plan.header->num_tensors, sizeof(void *),
                                          MALLOC_CAP_SPIRAM);
    size_t ws_size = tigris_executor_workspace_required(&plan);
    void *ws = ws_size ? heap_caps_malloc(ws_size,
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
                       : NULL;
    if (!fast_buf || !slow_buf || !tensor_ptrs || !ws) {
        ESP_LOGE(TAG, "arena allocation failed");
        return;
    }

    tigris_mem_t mem;
    if (tigris_mem_init(&mem, tensor_ptrs, plan.header->num_tensors,
                        fast_buf, fast_size, slow_buf, slow_size) != TIGRIS_MEM_OK) {
        ESP_LOGE(TAG, "mem init failed");
        return;
    }

    tigris_kernel_fn dispatch = tigris_dispatch_kernel_s8;
#if defined(TIGRIS_HAS_ESP_NN)
    dispatch = tigris_dispatch_kernel_esp_nn;   /* accelerate the encoder convs */
    if (tigris_esp_nn_prepare(&plan, &mem) != 0) {
        ESP_LOGE(TAG, "esp_nn_prepare failed (arena too small for scratch)");
        return;
    }
#endif
    printf("Fast (SRAM): %.0f KiB   Slow (PSRAM): %.0f KiB   Workspace: %u B\n\n",
           mem.fast_size / 1024.0f, slow_size / 1024.0f, (unsigned)ws_size);

    /* 3. Reset the arena (esp_nn_prepare reduced mem.fast_size in place), fill
     *    the input (int8 1 — the value the reference was generated with), and
     *    run once. The s8_ref decoder stages make this take ~30 s. */
    tigris_mem_init(&mem, mem.tensor_ptrs, mem.num_tensors,
                    mem.fast_base, mem.fast_size, mem.slow_base, mem.slow_size);
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t t = plan.model_inputs[i];
        tigris_mem_alloc_slow(&mem, t, plan.tensors[t].size_bytes);
        memset(mem.tensor_ptrs[t], is_quant ? 1 : 0, plan.tensors[t].size_bytes);
    }
    printf("Running inference (the s8_ref decoder takes ~30 s)...\n\n");
    tigris_exec_stats_t stats = {0};
    int64_t t0 = esp_timer_get_time();
    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &plan, &mem, dispatch, NULL, &stats, ws, ws_size);
    int64_t latency_us = esp_timer_get_time() - t0;
    if (err != TIGRIS_EXEC_OK) {
        ESP_LOGE(TAG, "inference failed: %s", tigris_exec_error_str(err));
        return;
    }

    /* 4. Self-check the int8 output against the embedded golden reference. */
    uint16_t out_t = plan.model_outputs[0];
    const int8_t *out = (const int8_t *)mem.tensor_ptrs[out_t];
    uint32_t out_n = plan.tensors[out_t].size_bytes;
    uint32_t ref_n = (uint32_t)(unet_ref_end - unet_ref_start);
    const int8_t *ref = (const int8_t *)unet_ref_start;

    int pass = (out_n == ref_n);
    int max_abs = 0, degenerate = 1;
    int64_t sum_abs = 0;
    uint32_t n_over2 = 0;
    if (pass) {
        for (uint32_t i = 0; i < out_n; i++) {
            int d = (int)out[i] - (int)ref[i];
            if (d < 0) d = -d;
            sum_abs += d;
            if (d > max_abs) max_abs = d;
            if (d > 2) n_over2++;
            if (i && out[i] != out[0]) degenerate = 0;
        }
    }
    float mean_abs = pass ? (float)((double)sum_abs / (double)out_n) : 999.0f;
    pass = pass && !degenerate && max_abs <= SELF_CHECK_MAX_ABS
                && mean_abs < SELF_CHECK_MEAN;

    /* 5. Report. */
    printf("Inference latency:  %.1f ms\n", latency_us / 1000.0f);
    printf("Stages tiled:       %u of %u   (total tiles %u, chains %u)\n",
           (unsigned)stats.stages_tiled, plan.header->num_stages,
           (unsigned)stats.total_tiles, (unsigned)stats.stages_chain);
    printf("Peak fast arena:    %.0f KiB of a %.2f MiB naive peak\n",
           mem.fast_size / 1024.0f,
           /* naive peak is not stored in the plan; the headline figure comes
            * from `tigris analyze` and is documented in the README */ 2.38f);
    printf("Output vs golden:   %u int8 values, mean|diff|=%.3f, max|diff|=%d, "
           "%.3f%% over 2 LSB\n",
           (unsigned)out_n, mean_abs, max_abs,
           100.0f * (float)n_over2 / (float)out_n);
    printf("\nSELF_CHECK: %s\n", pass ? "PASS" : "FAIL");
    printf("TIGRIS_DONE\n");

    heap_caps_free(ws);
    heap_caps_free(tensor_ptrs);
    heap_caps_free(slow_buf);
    heap_caps_free(fast_buf);
}

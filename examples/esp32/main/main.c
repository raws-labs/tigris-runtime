/*
 * TiGrIS ESP32 Runtime - application entry point.
 *
 * Loads a .tgrs plan from flash, runs inference, prints results.
 *
 * Build & flash:
 *   idf.py set-target esp32
 *   idf.py build
 *   idf.py flash
 *   scripts/flash_plan.sh ../tigris-runtime/test/fixtures/conv_relu_chain.tgrs
 *   idf.py monitor
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
/* ESP32-S3 watchdog management using HAL API */
#include "hal/wdt_hal.h"
#include "soc/timer_group_reg.h"

/* Progress callback - called from executor between tiles/stages */
static int progress_count = 0;

void tigris_feed_wdt(void) {
    progress_count++;
    if (progress_count % 100 == 0) {
        printf("Progress: %d tiles/stages\n", progress_count);
    }
}

static void disable_all_wdts(void) {
    /* Disable MWDT0 including flashboot mode (the one causing TG0WDT_SYS_RST).
     * Flashboot mode is independent and can trigger a reset even when the
     * WDT enable bit (bit 31) is 0. It's auto-enabled at boot with ~60s timeout. */
    wdt_hal_context_t mwdt0_ctx;
    wdt_hal_init(&mwdt0_ctx, WDT_MWDT0, 0, false);
    wdt_hal_write_protect_disable(&mwdt0_ctx);
    wdt_hal_set_flashboot_en(&mwdt0_ctx, false);
    wdt_hal_disable(&mwdt0_ctx);
    wdt_hal_write_protect_enable(&mwdt0_ctx);

    /* Disable MWDT1 including flashboot mode */
    wdt_hal_context_t mwdt1_ctx;
    wdt_hal_init(&mwdt1_ctx, WDT_MWDT1, 0, false);
    wdt_hal_write_protect_disable(&mwdt1_ctx);
    wdt_hal_set_flashboot_en(&mwdt1_ctx, false);
    wdt_hal_disable(&mwdt1_ctx);
    wdt_hal_write_protect_enable(&mwdt1_ctx);

    /* Disable RWDT including flashboot mode */
    wdt_hal_context_t rwdt_ctx;
    wdt_hal_init(&rwdt_ctx, WDT_RWDT, 0, false);
    wdt_hal_write_protect_disable(&rwdt_ctx);
    wdt_hal_set_flashboot_en(&rwdt_ctx, false);
    wdt_hal_disable(&rwdt_ctx);
    wdt_hal_write_protect_enable(&rwdt_ctx);

    printf("All WDTs + flashboot disabled\n");
}

#ifdef TIGRIS_PROFILING
#include "xtensa/core-macros.h"
#include "esp_cpu.h"
#endif

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#ifdef TIGRIS_HAS_ESP_NN
#include "tigris_kernels_esp_nn.h"
#endif

static const char *TAG = "tigris";

void app_main(void)
{
    /* Disable all watchdogs for long-running inference */
    disable_all_wdts();

    printf("\nTiGrIS ESP32 Runtime (schema v%d)\n\n",
           TIGRIS_SCHEMA_VERSION);

    /* 1. Find and memory-map the "plan" partition */

    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "plan");
    if (!part) {
        ESP_LOGE(TAG, "partition 'plan' not found");
        return;
    }
    ESP_LOGI(TAG, "plan partition: offset=0x%lx size=%lu",
             (unsigned long)part->address, (unsigned long)part->size);

    const void *mapped_ptr = NULL;
    esp_partition_mmap_handle_t mmap_handle;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return;
    }

    /* 2. Load the plan */

    /* Read the actual plan size from the file header (the partition is
       larger than the plan - the loader requires an exact size match). */
    const tigris_file_header_t *raw_hdr = (const tigris_file_header_t *)mapped_ptr;
    uint32_t plan_size = raw_hdr->file_size;
    if (plan_size < sizeof(tigris_file_header_t) || plan_size > part->size) {
        ESP_LOGE(TAG, "bad plan file_size: %lu (partition: %lu)",
                 (unsigned long)plan_size, (unsigned long)part->size);
        esp_partition_munmap(mmap_handle);
        return;
    }

    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(
        (const uint8_t *)mapped_ptr, plan_size, &plan);
    if (perr != TIGRIS_OK) {
        ESP_LOGE(TAG, "plan load failed: %s", tigris_error_str(perr));
        esp_partition_munmap(mmap_handle);
        return;
    }

    const char *model_name = tigris_model_name(&plan);
    printf("Model:    %s\n", model_name);
    printf("Tensors:  %u\n", plan.header->num_tensors);
    printf("Ops:      %u\n", plan.header->num_ops);
    printf("Stages:   %u\n", plan.header->num_stages);
    printf("Weights:  %u\n", plan.header->num_weights);
    printf("Budget:   %lu bytes\n", (unsigned long)plan.header->budget);
    printf("Peak:     %lu bytes\n", (unsigned long)plan.header->peak);

    /* 3. Compute buffer sizes */

    uint32_t fast_size = plan.header->budget;
    if (fast_size == 0)
        fast_size = 64 * 1024;  /* fallback: 64 KB */
    fast_size += tigris_weight_decompression_overhead(&plan);

    /* Slow buffer: use all available PSRAM (or internal if no PSRAM) */
    uint32_t slow_size;
#if CONFIG_SPIRAM
    slow_size = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    if (slow_size > 64 * 1024)
        slow_size -= 16 * 1024;  /* leave headroom for tensor_ptrs + misc */
#else
    slow_size = heap_caps_get_largest_free_block(
                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (slow_size > 32 * 1024)
        slow_size -= 16 * 1024;
#endif
    if (slow_size < 64 * 1024)
        slow_size = 64 * 1024;

    printf("\nMemory allocation:\n");
    printf("  Fast (SRAM):  %lu bytes\n", (unsigned long)fast_size);
    printf("  Slow:         %lu bytes\n", (unsigned long)slow_size);

    /* 4. Allocate buffers */

    printf("  Internal free: %lu bytes\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#if CONFIG_SPIRAM
    printf("  SPIRAM free:   %lu bytes\n",
           (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#endif

    void *fast_buf = heap_caps_malloc(fast_size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!fast_buf) {
        ESP_LOGE(TAG, "failed to alloc %lu bytes from internal SRAM",
                 (unsigned long)fast_size);
        esp_partition_munmap(mmap_handle);
        return;
    }

#if CONFIG_SPIRAM
    void *slow_buf = heap_caps_malloc(slow_size, MALLOC_CAP_SPIRAM);
    if (!slow_buf) {
        ESP_LOGE(TAG, "failed to alloc %lu bytes from PSRAM",
                 (unsigned long)slow_size);
        heap_caps_free(fast_buf);
        esp_partition_munmap(mmap_handle);
        return;
    }

    /* Tensor pointer table - can live in PSRAM */
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
                                          MALLOC_CAP_SPIRAM);
#else
    void *slow_buf = heap_caps_malloc(slow_size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!slow_buf) {
        ESP_LOGE(TAG, "failed to alloc %lu bytes from internal SRAM",
                 (unsigned long)slow_size);
        heap_caps_free(fast_buf);
        esp_partition_munmap(mmap_handle);
        return;
    }

    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
    if (!tensor_ptrs) {
        ESP_LOGE(TAG, "failed to alloc tensor_ptrs");
        heap_caps_free(slow_buf);
        heap_caps_free(fast_buf);
        esp_partition_munmap(mmap_handle);
        return;
    }

    /* 5. Init memory manager */

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, num_t,
        fast_buf, fast_size,
        slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) {
        ESP_LOGE(TAG, "mem init failed: %s", tigris_mem_error_str(merr));
        goto cleanup;
    }

    /* 6. Detect quantized model */

    /* dtype 3 = INT8, dtype 1 = FLOAT32 */
    uint8_t input_dtype = plan.tensors[plan.model_inputs[0]].dtype;
    int is_quantized = (input_dtype == 3);
    printf("\nModel type: %s\n", is_quantized ? "int8 quantized" : "float32");

    /* 7. Allocate model inputs in slow, fill with test data */

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        merr = tigris_mem_alloc_slow(&mem, tidx, sz);
        if (merr != TIGRIS_MEM_OK) {
            ESP_LOGE(TAG, "alloc input %u failed: %s", tidx,
                     tigris_mem_error_str(merr));
            goto cleanup;
        }
        if (is_quantized) {
            /* Fill int8 input with deterministic test data */
            int8_t *data = (int8_t *)mem.tensor_ptrs[tidx];
            for (uint32_t j = 0; j < sz; j++)
                data[j] = 1;
        } else {
            /* Fill float32 input with deterministic test data: 0.5 */
            float *data = (float *)mem.tensor_ptrs[tidx];
            uint32_t n_floats = sz / sizeof(float);
            for (uint32_t j = 0; j < n_floats; j++)
                data[j] = 0.5f;
        }

        const int32_t *shape = tigris_tensor_shape(&plan, &plan.tensors[tidx]);
        uint8_t ndim = plan.tensors[tidx].ndim;
        printf("\nInput[%u] '%s': ", tidx,
               tigris_tensor_name(&plan, &plan.tensors[tidx]));
        for (uint8_t d = 0; d < ndim; d++)
            printf("%s%d", d ? "x" : "", (int)shape[d]);
        printf(" (%lu bytes)\n", (unsigned long)sz);
    }

    /* 8. Run inference */

    /* Select dispatch function based on model type */
    tigris_kernel_fn dispatch;
    if (is_quantized) {
#ifdef TIGRIS_HAS_ESP_NN
        dispatch = tigris_dispatch_kernel_esp_nn;
        printf("\nUsing ESP-NN accelerated int8 kernels\n");
#else
        dispatch = tigris_dispatch_kernel_s8;
        printf("\nUsing reference int8 kernels\n");
#endif
    } else {
        dispatch = tigris_dispatch_kernel;
    }

    printf("Running inference...\n");

    tigris_exec_stats_t exec_stats;
    int64_t t0 = esp_timer_get_time();

    tigris_exec_error_t eerr = tigris_run(
        &plan, &mem, dispatch, NULL, &exec_stats);

    int64_t t1 = esp_timer_get_time();
    float elapsed_ms = (float)(t1 - t0) / 1000.0f;

    if (eerr != TIGRIS_EXEC_OK) {
        ESP_LOGE(TAG, "inference failed: %s", tigris_exec_error_str(eerr));
        goto cleanup;
    }

    /* 9. Inference report */

    printf("\n");
    printf("\n");
    printf("  TiGrIS Inference Report\n");
    printf("\n");

    printf("\n  [Hardware]\n");
    printf("    SoC:             ESP32-S3 (Xtensa LX7, 240 MHz)\n");
    printf("    SRAM:            512 KB (384 KB usable)\n");
#if CONFIG_SPIRAM
    printf("    PSRAM:           8 MB (octal SPI)\n");
#endif
    printf("    Flash:           16 MB (quad SPI)\n");

    printf("\n  [Model]\n");
    printf("    Name:            %s\n", model_name);
    printf("    Type:            %s\n",
           is_quantized ? "int8 quantized" : "float32");
    printf("    Ops:             %u\n", plan.header->num_ops);
    printf("    Stages:          %u\n", plan.header->num_stages);
    printf("    Weights:         %u\n", plan.header->num_weights);
    printf("    Plan size:       %lu KB\n",
           (unsigned long)(plan_size / 1024));

    printf("\n  [Memory]\n");
    printf("    Fast (SRAM):     %lu KB\n",
           (unsigned long)(fast_size / 1024));
    printf("    Slow (PSRAM):    %lu KB\n",
           (unsigned long)(slow_size / 1024));
    printf("    Slow peak:       %lu KB\n",
           (unsigned long)(exec_stats.slow_peak / 1024));
    printf("    Plan budget:     %lu KB\n",
           (unsigned long)(plan.header->budget / 1024));
    printf("    Plan peak:       %lu KB\n",
           (unsigned long)(plan.header->peak / 1024));

    printf("\n  [Execution]\n");
    printf("    Latency:         %.1f ms\n", elapsed_ms);
    printf("    Stages normal:   %u\n", exec_stats.stages_normal);
    printf("    Stages tiled:    %u\n", exec_stats.stages_tiled);
    printf("    Stages chain:    %u\n", exec_stats.stages_chain);
    printf("    Compactions:     %lu\n",
           (unsigned long)exec_stats.compactions);
    printf("    Loads  (slow->fast):  %lu KB\n",
           (unsigned long)(exec_stats.loads_bytes / 1024));
    printf("    Spills (fast->slow):  %lu KB\n",
           (unsigned long)(exec_stats.spills_bytes / 1024));
    if (exec_stats.slow_overflow_count > 0) {
        printf("    Overflow to slow:     %lu KB (%lu allocs)\n",
               (unsigned long)(exec_stats.slow_overflow_bytes / 1024),
               (unsigned long)exec_stats.slow_overflow_count);
    }

    printf("\n  [Heap after inference]\n");
    printf("    Internal free:   %lu KB\n",
           (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
#if CONFIG_SPIRAM
    printf("    PSRAM free:      %lu KB\n",
           (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
#endif

    printf("\n  [Output]\n");
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        const tigris_tensor_t *t = &plan.tensors[tidx];
        const int32_t *shape = tigris_tensor_shape(&plan, t);
        uint8_t ndim = t->ndim;

        printf("    Output[%u] '%s': ", tidx,
               tigris_tensor_name(&plan, t));
        for (uint8_t d = 0; d < ndim; d++)
            printf("%s%d", d ? "x" : "", (int)shape[d]);
        printf("\n");

        void *out_ptr = mem.tensor_ptrs[tidx];
        if (!out_ptr) {
            printf("      (null)\n");
            continue;
        }

        if (is_quantized) {
            int8_t *out = (int8_t *)out_ptr;
            uint32_t n = t->size_bytes;
            uint32_t print_n = n < 10 ? n : 10;
            printf("      First %" PRIu32 ":", print_n);
            for (uint32_t j = 0; j < print_n; j++)
                printf(" %d", (int)out[j]);
            if (n > print_n)
                printf(" ...");
            printf("\n");
        } else {
            float *out = (float *)out_ptr;
            uint32_t n = t->size_bytes / sizeof(float);
            uint32_t print_n = n < 10 ? n : 10;
            printf("      First %" PRIu32 ":", print_n);
            for (uint32_t j = 0; j < print_n; j++)
                printf(" %.4f", out[j]);
            if (n > print_n)
                printf(" ...");
            printf("\n");
        }
    }

    printf("\n");

cleanup:
    heap_caps_free(tensor_ptrs);
    heap_caps_free(slow_buf);
    heap_caps_free(fast_buf);
    esp_partition_munmap(mmap_handle);
}

/*
 * Canonical POSIX integration for an int8 TiGrIS plan.
 *
 * The example uses host allocation to obtain caller-owned plan and arena
 * buffers. The TiGrIS loader and executor do not own or grow those buffers.
 */
#define _POSIX_C_SOURCE 200112L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

#define EXAMPLE_SLOW_BYTES (512u * 1024u)
#define ONNX_DTYPE_INT8 3u

static int plan_has_int8_io(const tigris_plan_t *plan)
{
    uint8_t i;

    if (plan->header->num_model_inputs == 0 ||
        plan->header->num_model_outputs == 0 ||
        plan->num_quant_params == 0)
        return 0;

    for (i = 0; i < plan->header->num_model_inputs; ++i) {
        uint16_t idx = plan->model_inputs[i];
        if (plan->tensors[idx].dtype != ONNX_DTYPE_INT8)
            return 0;
    }
    for (i = 0; i < plan->header->num_model_outputs; ++i) {
        uint16_t idx = plan->model_outputs[i];
        if (plan->tensors[idx].dtype != ONNX_DTYPE_INT8)
            return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    static tigris_executor_workspace_t executor_workspace;
    FILE *file = NULL;
    void *plan_storage = NULL;
    void *fast_buf = NULL;
    void *slow_buf = NULL;
    void **tensor_ptrs = NULL;
    int result = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s model.tgrs\n", argv[0]);
        return 2;
    }

    file = fopen(argv[1], "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "could not open or seek plan\n");
        goto cleanup;
    }

    {
        long end = ftell(file);
        uint32_t file_size;
        size_t alignment = TIGRIS_TENSOR_ALIGN;
        tigris_plan_t plan;
        tigris_mem_t mem;
        tigris_error_t load_err;
        tigris_mem_error_t mem_err;
        uint32_t fast_size;
        uint32_t slow_size = EXAMPLE_SLOW_BYTES;
        uint8_t i;

        if (end <= 0 || (unsigned long)end > UINT32_MAX ||
            fseek(file, 0, SEEK_SET) != 0) {
            fprintf(stderr, "invalid plan size\n");
            goto cleanup;
        }
        file_size = (uint32_t)end;

        if (alignment < sizeof(void *))
            alignment = sizeof(void *);
        if (posix_memalign(&plan_storage, alignment, file_size) != 0) {
            fprintf(stderr, "could not allocate aligned plan buffer\n");
            goto cleanup;
        }
        if (fread(plan_storage, 1, file_size, file) != file_size) {
            fprintf(stderr, "could not read complete plan\n");
            goto cleanup;
        }
        if (fclose(file) != 0) {
            file = NULL;
            fprintf(stderr, "could not close plan file\n");
            goto cleanup;
        }
        file = NULL;

        load_err = tigris_plan_load(
            (const uint8_t *)plan_storage, file_size, &plan);
        if (load_err != TIGRIS_OK) {
            fprintf(stderr, "load failed: %s\n", tigris_error_str(load_err));
            goto cleanup;
        }
        if (!plan_has_int8_io(&plan)) {
            fprintf(stderr, "this example requires an int8 plan\n");
            goto cleanup;
        }

        fast_size = tigris_fast_arena_required(&plan);
        if (fast_size == UINT32_MAX) {
            fprintf(stderr, "invalid fast-buffer requirement\n");
            goto cleanup;
        }
        if (fast_size == 0 ||
            posix_memalign(&fast_buf, alignment, fast_size) != 0 ||
            posix_memalign(&slow_buf, alignment, slow_size) != 0) {
            fprintf(stderr, "could not allocate aligned arenas\n");
            goto cleanup;
        }

        tensor_ptrs = calloc(plan.header->num_tensors, sizeof(void *));
        if (tensor_ptrs == NULL) {
            fprintf(stderr, "could not allocate tensor pointer table\n");
            goto cleanup;
        }

        mem_err = tigris_mem_init(
            &mem, tensor_ptrs, plan.header->num_tensors,
            fast_buf, fast_size, slow_buf, slow_size);
        if (mem_err != TIGRIS_MEM_OK) {
            fprintf(stderr, "memory init failed: %s\n",
                    tigris_mem_error_str(mem_err));
            goto cleanup;
        }

        for (i = 0; i < plan.header->num_model_inputs; ++i) {
            uint16_t idx = plan.model_inputs[i];
            mem_err = tigris_mem_alloc_slow(
                &mem, idx, plan.tensors[idx].size_bytes);
            if (mem_err != TIGRIS_MEM_OK) {
                fprintf(stderr, "input allocation failed: %s\n",
                        tigris_mem_error_str(mem_err));
                goto cleanup;
            }
            memset(mem.tensor_ptrs[idx], 1, plan.tensors[idx].size_bytes);
        }

        {
            tigris_exec_stats_t stats;
            tigris_exec_error_t exec_err = tigris_run_with_workspace(
                &plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats,
                &executor_workspace);
            uint16_t out_idx;
            uint32_t out_bytes;
            float *output = NULL;
            uint32_t count;
            uint32_t show;
            uint32_t j;

            if (exec_err != TIGRIS_EXEC_OK) {
                fprintf(stderr, "inference failed: %s\n",
                        tigris_exec_error_str(exec_err));
                goto cleanup;
            }

            /* Read the output in the dtype the model declares, which the
             * runtime converts to from whatever the plan executes on. */
            out_idx = plan.model_outputs[0];
            out_bytes = tigris_iface_bytes(&plan, out_idx);
            if (out_bytes == 0) {
                fprintf(stderr, "inference produced no first output\n");
                goto cleanup;
            }
            output = (float *)malloc(out_bytes);
            if (output == NULL) {
                fprintf(stderr, "output allocation failed\n");
                goto cleanup;
            }
            if (tigris_output_read(&plan, &mem, out_idx, output, out_bytes)
                != TIGRIS_OK) {
                fprintf(stderr, "output conversion failed\n");
                free(output);
                goto cleanup;
            }

            count = out_bytes / (uint32_t)sizeof(float);
            show = count;
            if (show > 5)
                show = 5;
            printf("Output:");
            for (j = 0; j < show; ++j)
                printf(" %.6g", (double)output[j]);
            printf("\nFast arena peak: %lu bytes\n",
                   (unsigned long)mem.fast_peak);
            free(output);
        }
    }

    result = 0;

cleanup:
    if (file != NULL && fclose(file) != 0)
        result = 1;
    free(tensor_ptrs);
    free(slow_buf);
    free(fast_buf);
    free(plan_storage);
    return result;
}

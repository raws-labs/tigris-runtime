/*
 * Host-only compiler/runtime contract bridge.
 *
 * Usage:
 *   tigris_contract_runner plan.tgrs inputs.bin outputs.bin
 *
 * inputs.bin and outputs.bin concatenate model tensors in model-I/O order.
 * The companion compiler gate owns ONNX Runtime comparison and malformed-plan
 * expectations; this executable deliberately proves only the public runtime
 * load/allocate/run/output path.
 */
#define _POSIX_C_SOURCE 200112L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

static int load_file_aligned(
    const char *path, uint8_t **out_data, uint32_t *out_size)
{
    FILE *file;
    long end;
    uint8_t *data = NULL;
    size_t alignment = TIGRIS_TENSOR_ALIGN;

    if (!path || !out_data || !out_size)
        return -1;
    *out_data = NULL;
    *out_size = 0;

    file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0)
        goto fail;
    end = ftell(file);
    if (end <= 0 || (unsigned long)end > UINT32_MAX ||
        fseek(file, 0, SEEK_SET) != 0)
        goto fail;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);
    if (posix_memalign((void **)&data, alignment, (size_t)end) != 0)
        goto fail;
    if (fread(data, 1, (size_t)end, file) != (size_t)end)
        goto fail;
    if (fclose(file) != 0) {
        file = NULL;
        goto fail;
    }

    *out_data = data;
    *out_size = (uint32_t)end;
    return 0;

fail:
    if (file)
        fclose(file);
    free(data);
    return -1;
}

static int allocate_aligned(size_t alignment, uint32_t size, void **out)
{
    if (!out || size == 0)
        return -1;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);
    return posix_memalign(out, alignment, size) == 0 ? 0 : -1;
}

static int choose_dispatch(
    const tigris_plan_t *plan, tigris_kernel_fn *out_dispatch)
{
    uint8_t dtype;

    if (!plan || !plan->header || !plan->tensors || !out_dispatch ||
        plan->header->num_model_inputs == 0 ||
        plan->header->num_model_outputs == 0)
        return -1;

    dtype = plan->tensors[plan->model_inputs[0]].dtype;
    for (uint16_t i = 0; i < plan->header->num_tensors; ++i) {
        if ((plan->tensors[i].flags & TIGRIS_TENSOR_CONSTANT) == 0 &&
            plan->tensors[i].dtype != dtype)
            return -1;
    }

    if (dtype == 1) {
        *out_dispatch = tigris_dispatch_kernel;
        return 0;
    }
    if (dtype == 3) {
        *out_dispatch = tigris_dispatch_kernel_s8;
        return 0;
    }
    return -1;
}

static int slow_requirement(
    const tigris_plan_t *plan, uint32_t *out_size)
{
    uint64_t total = 0;

    if (!plan || !plan->header || !plan->tensors || !out_size)
        return -1;
    for (uint16_t i = 0; i < plan->header->num_tensors; ++i) {
        total += plan->tensors[i].size_bytes;
        if (total > UINT32_MAX / 2u)
            return -1;
    }
    total *= 2u;
    if (total < 4096u)
        total = 4096u;
    *out_size = (uint32_t)total;
    return 0;
}

static int load_inputs(
    const char *path, const tigris_plan_t *plan, tigris_mem_t *mem)
{
    FILE *file;
    int trailing;

    if (!path || !plan || !plan->header || !mem)
        return -1;
    file = fopen(path, "rb");
    if (!file)
        return -1;

    for (uint8_t i = 0; i < plan->header->num_model_inputs; ++i) {
        uint16_t index = plan->model_inputs[i];
        uint32_t size;
        if (index >= plan->header->num_tensors)
            goto fail;
        size = plan->tensors[index].size_bytes;
        if (tigris_mem_alloc_slow(mem, index, size) != TIGRIS_MEM_OK ||
            fread(mem->tensor_ptrs[index], 1, size, file) != size)
            goto fail;
    }
    trailing = fgetc(file);
    if (trailing != EOF)
        goto fail;
    if (fclose(file) != 0)
        return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

static int write_outputs(
    const char *path, const tigris_plan_t *plan, const tigris_mem_t *mem)
{
    FILE *file;

    if (!path || !plan || !plan->header || !mem)
        return -1;
    file = fopen(path, "wb");
    if (!file)
        return -1;

    for (uint8_t i = 0; i < plan->header->num_model_outputs; ++i) {
        uint16_t index = plan->model_outputs[i];
        uint32_t size;
        void *data;
        if (index >= plan->header->num_tensors)
            goto fail;
        size = plan->tensors[index].size_bytes;
        data = tigris_mem_tensor_ptr(mem, index);
        if (!data || fwrite(data, 1, size, file) != size)
            goto fail;
    }
    if (fclose(file) != 0)
        return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

int main(int argc, char **argv)
{
    uint8_t *plan_data = NULL;
    uint32_t plan_size = 0;
    void **tensor_ptrs = NULL;
    void *fast_buf = NULL;
    void *slow_buf = NULL;
    uint32_t fast_size;
    uint32_t slow_size;
    uint32_t overhead;
    tigris_plan_t plan;
    tigris_mem_t mem;
    tigris_kernel_fn dispatch = NULL;
    int result = 1;

    if (argc != 4) {
        fprintf(stderr, "usage: %s plan.tgrs inputs.bin outputs.bin\n", argv[0]);
        return 2;
    }
    if (load_file_aligned(argv[1], &plan_data, &plan_size) != 0) {
        fprintf(stderr, "could not read plan: %s\n", argv[1]);
        goto cleanup;
    }
    {
        tigris_error_t error = tigris_plan_load(plan_data, plan_size, &plan);
        if (error != TIGRIS_OK) {
            fprintf(stderr, "plan load failed: %s\n", tigris_error_str(error));
            goto cleanup;
        }
    }
    if (choose_dispatch(&plan, &dispatch) != 0) {
        fprintf(stderr, "plan has unsupported or mixed contract dtype\n");
        goto cleanup;
    }
    overhead = tigris_weight_decompression_overhead(&plan);
    if (plan.header->budget == 0 || overhead == UINT32_MAX ||
        plan.header->budget > UINT32_MAX - overhead ||
        slow_requirement(&plan, &slow_size) != 0) {
        fprintf(stderr, "plan has unrepresentable arena requirements\n");
        goto cleanup;
    }
    fast_size = plan.header->budget + overhead;
    tensor_ptrs = calloc(plan.header->num_tensors, sizeof(*tensor_ptrs));
    if (!tensor_ptrs ||
        allocate_aligned(TIGRIS_TENSOR_ALIGN, fast_size, &fast_buf) != 0 ||
        allocate_aligned(TIGRIS_TENSOR_ALIGN, slow_size, &slow_buf) != 0) {
        fprintf(stderr, "could not allocate contract arenas\n");
        goto cleanup;
    }
    if (tigris_mem_init(
            &mem, tensor_ptrs, plan.header->num_tensors,
            fast_buf, fast_size, slow_buf, slow_size) != TIGRIS_MEM_OK) {
        fprintf(stderr, "memory initialization failed\n");
        goto cleanup;
    }
    if (load_inputs(argv[2], &plan, &mem) != 0) {
        fprintf(stderr, "input layout does not match plan model inputs\n");
        goto cleanup;
    }
    {
        tigris_exec_error_t error =
            tigris_run(&plan, &mem, dispatch, NULL, NULL);
        if (error != TIGRIS_EXEC_OK) {
            fprintf(stderr, "inference failed: %s\n",
                    tigris_exec_error_str(error));
            goto cleanup;
        }
    }
    if (write_outputs(argv[3], &plan, &mem) != 0) {
        fprintf(stderr, "could not write model outputs\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    free(slow_buf);
    free(fast_buf);
    free(tensor_ptrs);
    free(plan_data);
    return result;
}

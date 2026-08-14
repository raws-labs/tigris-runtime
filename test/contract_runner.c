/*
 * Host-only compiler/runtime contract bridge.
 *
 * Usage:
 *   tigris_contract_runner plan.tgrs inputs.bin outputs.bin [activation-limit]
 *
 * inputs.bin and outputs.bin concatenate model tensors in model-I/O order.
 * The companion compiler gate owns ONNX Runtime comparison and malformed-plan
 * expectations; this executable deliberately proves only the public runtime
 * load/allocate/run/output path.
 */
#define _POSIX_C_SOURCE 200112L

#include <inttypes.h>
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

#ifdef TIGRIS_COUNT_KERNEL_ROWS
/* Test-only kernel row counters, defined in the runtime sources when they are
 * compiled with TIGRIS_COUNT_KERNEL_ROWS. The shipping library and the default
 * contract runner are built without this flag, so these are never linked there.
 * The rows-instrumented runner uses them to report how many kernel output rows
 * an execution computed, which the cross-repo gate diffs between the
 * line-buffered roll and the recompute path. */
extern unsigned long g_tigris_kernel_rows;
extern int32_t g_tigris_chain_tile_h;
extern int32_t g_tigris_chain_num_tiles;
extern int32_t g_tigris_interior_clamps;
#endif

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
    uint32_t required_size;
    uint32_t activation_limit;
    uint32_t slow_size;
    uint32_t overhead;
    tigris_plan_t plan;
    tigris_mem_t mem;
    tigris_kernel_fn dispatch = NULL;
    int result = 1;
    int no_linebuffer = 0;
    char *pos_argv[5];
    int pos_argc = 1;
    int arg_index;

    /* Separate the optional --no-linebuffer switch from the positional
     * arguments so it may appear anywhere on the command line. Everything
     * else keeps its existing positional meaning, so a call without the
     * switch behaves exactly as before. */
    pos_argv[0] = argv[0];
    for (arg_index = 1; arg_index < argc; ++arg_index) {
        if (strcmp(argv[arg_index], "--no-linebuffer") == 0) {
            no_linebuffer = 1;
        } else if (pos_argc < (int)(sizeof(pos_argv) / sizeof(pos_argv[0]))) {
            pos_argv[pos_argc++] = argv[arg_index];
        } else {
            fprintf(stderr, "too many arguments\n");
            return 2;
        }
    }

    if (pos_argc != 4 && pos_argc != 5) {
        fprintf(stderr,
                "usage: %s plan.tgrs inputs.bin outputs.bin [activation-limit] "
                "[--no-linebuffer]\n",
                pos_argv[0]);
        return 2;
    }
    if (load_file_aligned(pos_argv[1], &plan_data, &plan_size) != 0) {
        fprintf(stderr, "could not read plan: %s\n", pos_argv[1]);
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
    required_size = tigris_fast_arena_required(&plan);
    if (plan.header->budget == 0 || overhead == UINT32_MAX ||
        required_size == UINT32_MAX ||
        slow_requirement(&plan, &slow_size) != 0) {
        fprintf(stderr, "plan has unrepresentable arena requirements\n");
        goto cleanup;
    }
    activation_limit = plan.header->budget;
    if (pos_argc == 5) {
        char *end = NULL;
        unsigned long parsed = strtoul(pos_argv[4], &end, 10);
        if (!pos_argv[4][0] || !end || *end != '\0' || parsed == 0 ||
            parsed > plan.header->budget || parsed > UINT32_MAX) {
            fprintf(stderr, "invalid activation limit: %s\n", pos_argv[4]);
            goto cleanup;
        }
        activation_limit = (uint32_t)parsed;
    }
    if (activation_limit > UINT32_MAX - overhead) {
        fprintf(stderr, "activation limit plus reserve is unrepresentable\n");
        goto cleanup;
    }
    fast_size = activation_limit + overhead;
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
    if (load_inputs(pos_argv[2], &plan, &mem) != 0) {
        fprintf(stderr, "input layout does not match plan model inputs\n");
        goto cleanup;
    }
    if (no_linebuffer) {
        /* Force the recompute path: clear the line-buffered flag on every
         * loaded stage. The loader is zero-copy, so plan.stages aliases the
         * loaded buffer; locate each _reserved1 byte inside plan_data and
         * clear the flag bit in place, exactly as the executor line-buffer
         * ctest does. A chain without the flag executes the unchanged
         * recompute path, which is what the gate measures against. */
        for (uint16_t s = 0; s < plan.header->num_stages; ++s) {
            const uint8_t *field = (const uint8_t *)&plan.stages[s]._reserved1;
            size_t off = (size_t)(field - plan_data);
            if (off + sizeof(uint16_t) <= plan_size)
                plan_data[off] = (uint8_t)(plan_data[off] &
                    ~(uint8_t)(TIGRIS_STAGE_FLAG_LINE_BUFFERED & 0xFFu));
        }
    }
#ifdef TIGRIS_COUNT_KERNEL_ROWS
    g_tigris_kernel_rows = 0;
#endif
    {
        tigris_exec_error_t error =
            tigris_run(&plan, &mem, dispatch, NULL, NULL);
        if (error != TIGRIS_EXEC_OK) {
            fprintf(stderr, "inference failed: %s\n",
                    tigris_exec_error_str(error));
            goto cleanup;
        }
    }
    if (write_outputs(pos_argv[3], &plan, &mem) != 0) {
        fprintf(stderr, "could not write model outputs\n");
        goto cleanup;
    }
    printf(
        "TIGRIS_CONTRACT_MEMORY budget=%" PRIu32 " activation_limit=%" PRIu32
        " reserve=%" PRIu32 " required=%" PRIu32 " allocated=%" PRIu32
        " peak=%" PRIu32 "\n",
        plan.header->budget, activation_limit, overhead, required_size,
        fast_size, mem.fast_peak);
#ifdef TIGRIS_COUNT_KERNEL_ROWS
    printf(
        "TIGRIS_CONTRACT_ROWS kernel_rows=%lu chain_tiles=%" PRId32
        " chain_tile_h=%" PRId32 " interior_clamps=%" PRId32 "\n",
        g_tigris_kernel_rows, g_tigris_chain_num_tiles,
        g_tigris_chain_tile_h, g_tigris_interior_clamps);
#endif
    result = 0;

cleanup:
    free(slow_buf);
    free(fast_buf);
    free(tensor_ptrs);
    free(plan_data);
    return result;
}

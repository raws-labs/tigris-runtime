/**
 * Quick s8 kernel e2e test - loads a plan, fills input with deterministic
 * pattern, runs with s8_ref kernel, prints output values.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"

static uint8_t *load_file(const char *path, uint32_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, (size_t)len, f) != len) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_len = (uint32_t)len;
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <plan.tgrs> [budget_bytes]\n", argv[0]);
        return 1;
    }

    uint32_t budget = argc >= 3 ? (uint32_t)atoi(argv[2]) : 262144;

    uint32_t fsize;
    uint8_t *fbuf = load_file(argv[1], &fsize);
    if (!fbuf) return 1;

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(fbuf, fsize, &plan);
    if (err != TIGRIS_OK) {
        fprintf(stderr, "Plan load failed: %s\n", tigris_error_str(err));
        free(fbuf);
        return 1;
    }

    printf("Model: %s\n", tigris_model_name(&plan));
    printf("Ops: %u, Tensors: %u\n", plan.header->num_ops, plan.header->num_tensors);

    uint16_t nt = plan.header->num_tensors;
    uint32_t fast_size = budget + tigris_weight_decompression_overhead(&plan);
    uint32_t slow_size = 4 * 1024 * 1024;

    void *fast_buf = malloc(fast_size);
    void *slow_buf = malloc(slow_size);
    void **ptrs = calloc(nt, sizeof(void *));
    if (!fast_buf || !slow_buf || !ptrs) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }

    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, nt, fast_buf, fast_size, slow_buf, slow_size);

    /* Fill input with deterministic pattern */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        tigris_mem_alloc_slow(&mem, tidx, sz);
        int8_t *data = (int8_t *)ptrs[tidx];
        for (uint32_t j = 0; j < sz; j++)
            data[j] = (int8_t)(((j * 73 + 17) % 201) - 100);
        printf("Input[%u]: %d %d %d %d %d (sz=%u)\n",
               tidx, data[0], data[1], data[2], data[3], data[4], sz);
    }

    /* Run with s8_ref kernel */
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats);
    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "Inference failed: %s\n", tigris_exec_error_str(eerr));
        return 1;
    }

    /* Print output */
    uint16_t last_op_idx = plan.header->num_ops - 1;
    const tigris_op_t *last_op = &plan.ops[last_op_idx];
    const uint16_t *last_outs = tigris_op_outputs(&plan, last_op);
    uint16_t otidx = last_outs[0];
    int8_t *out = (int8_t *)ptrs[otidx];
    uint32_t n = plan.tensors[otidx].size_bytes;
    printf("Output[%u] (%u bytes):", otidx, n);
    for (uint32_t j = 0; j < n; j++)
        printf(" %d", (int)out[j]);
    printf("\n");

    free(ptrs);
    free(slow_buf);
    free(fast_buf);
    free(fbuf);
    return 0;
}

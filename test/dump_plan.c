/**
 * Dump plan structure - stages, ops, chain info - for debugging tiling.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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

static const char *op_type_name(uint8_t t)
{
    switch (t) {
    case 1: return "CONV";
    case 2: return "DEPTHWISE";
    case 3: return "FULLY_CONN";
    case 4: return "RELU";
    case 5: return "RELU6";
    case 6: return "ADD";
    case 7: return "GLOBAL_AVG";
    case 8: return "RESHAPE";
    case 9: return "FLATTEN";
    default: return "???";
    }
}

/* Simple checksum of a buffer */
static uint32_t checksum(const int8_t *data, uint32_t n)
{
    uint32_t sum = 0;
    for (uint32_t i = 0; i < n; i++)
        sum += (uint32_t)(uint8_t)data[i] * (i + 1);
    return sum;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <plan.tgrs> [budget_bytes]\n", argv[0]);
        return 1;
    }

    uint32_t budget = argc >= 3 ? (uint32_t)atoi(argv[2]) : 0;

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

    if (budget == 0) budget = plan.header->budget;

    printf("Plan: %s\n", tigris_model_name(&plan));
    printf("Budget: %u, Ops: %u, Tensors: %u, Stages: %u\n",
           plan.header->budget, plan.header->num_ops,
           plan.header->num_tensors, plan.header->num_stages);
    printf("Model inputs: ");
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++)
        printf("%u ", plan.model_inputs[i]);
    printf("\nModel outputs: ");
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++)
        printf("%u ", plan.model_outputs[i]);
    printf("\n\n");

    /* Dump stages */
    for (uint16_t s = 0; s < plan.header->num_stages; s++) {
        const tigris_stage_t *st = &plan.stages[s];
        printf("Stage %u: ops=%u, chain_id=%u, chain_len=%u, chain_tile_h=%u\n",
               s, st->ops_count, st->chain_id, st->chain_len, st->chain_tile_h);

        /* Stage inputs */
        const uint16_t *sin = tigris_stage_inputs(&plan, st);
        printf("  Inputs: ");
        for (uint16_t i = 0; i < st->inputs_count; i++) {
            uint16_t tidx = sin[i];
            const tigris_tensor_t *t = &plan.tensors[tidx];
            const int32_t *sh = tigris_tensor_shape(&plan, t);
            printf("T%u[", tidx);
            for (uint8_t d = 0; d < t->ndim; d++)
                printf("%s%d", d ? "," : "", sh[d]);
            printf("]=%uB ", t->size_bytes);
        }
        printf("\n");

        /* Stage outputs */
        const uint16_t *sout = tigris_stage_outputs(&plan, st);
        printf("  Outputs: ");
        for (uint16_t i = 0; i < st->outputs_count; i++) {
            uint16_t tidx = sout[i];
            const tigris_tensor_t *t = &plan.tensors[tidx];
            const int32_t *sh = tigris_tensor_shape(&plan, t);
            printf("T%u[", tidx);
            for (uint8_t d = 0; d < t->ndim; d++)
                printf("%s%d", d ? "," : "", sh[d]);
            printf("]=%uB ", t->size_bytes);
        }
        printf("\n");

        /* Ops in this stage */
        const uint16_t *sops = tigris_stage_ops(&plan, st);
        for (uint16_t j = 0; j < st->ops_count; j++) {
            uint16_t op_idx = sops[j];
            const tigris_op_t *op = &plan.ops[op_idx];
            printf("  Op %u: %s", op_idx, op_type_name(op->op_type));

            if (op->op_type == 1 || op->op_type == 2) {
                printf(" K%dx%d S%dx%d P[%d,%d,%d,%d]",
                       op->spatial.kernel_h, op->spatial.kernel_w,
                       op->spatial.stride_h, op->spatial.stride_w,
                       op->spatial.pad_top, op->spatial.pad_bottom,
                       op->spatial.pad_left, op->spatial.pad_right);
            }
            printf(" act[%d,%d]", op->act_min, op->act_max);

            /* Op inputs */
            const uint16_t *ins = tigris_op_inputs(&plan, op);
            printf(" in=");
            for (uint8_t k = 0; k < op->num_inputs; k++) {
                const int32_t *sh = tigris_tensor_shape(&plan, &plan.tensors[ins[k]]);
                printf("T%u[%d,%d,%d,%d] ", ins[k],
                       plan.tensors[ins[k]].ndim >= 1 ? sh[0] : 0,
                       plan.tensors[ins[k]].ndim >= 2 ? sh[1] : 0,
                       plan.tensors[ins[k]].ndim >= 3 ? sh[2] : 0,
                       plan.tensors[ins[k]].ndim >= 4 ? sh[3] : 0);
            }

            /* Op outputs */
            const uint16_t *outs = tigris_op_outputs(&plan, op);
            printf("out=");
            for (uint8_t k = 0; k < op->num_outputs; k++) {
                const int32_t *sh = tigris_tensor_shape(&plan, &plan.tensors[outs[k]]);
                printf("T%u[%d,%d,%d,%d] ", outs[k],
                       plan.tensors[outs[k]].ndim >= 1 ? sh[0] : 0,
                       plan.tensors[outs[k]].ndim >= 2 ? sh[1] : 0,
                       plan.tensors[outs[k]].ndim >= 3 ? sh[2] : 0,
                       plan.tensors[outs[k]].ndim >= 4 ? sh[3] : 0);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Now run inference and print per-op checksums */
    printf("Running inference (budget=%u)\n\n", budget);

    uint16_t nt = plan.header->num_tensors;
    uint32_t fast_size = budget + tigris_weight_decompression_overhead(&plan);
    uint32_t slow_size = 16 * 1024 * 1024;

    void *fast_buf = malloc(fast_size);
    void *slow_buf = malloc(slow_size);
    void **ptrs = calloc(nt, sizeof(void *));
    if (!fast_buf || !slow_buf || !ptrs) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }

    tigris_mem_t mem;
    tigris_mem_init(&mem, ptrs, nt, fast_buf, fast_size, slow_buf, slow_size);

    /* Fill input with memset 1 (matching benchmark) */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        tigris_mem_alloc_slow(&mem, tidx, sz);
        memset(ptrs[tidx], 1, sz);
    }

    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats);
    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "Inference failed: %s\n", tigris_exec_error_str(eerr));
        return 1;
    }

    printf("Stats: normal=%u tiled=%u chain=%u\n",
           stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    /* Print output */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        if (!ptrs[tidx]) { printf("Output T%u: NULL\n", tidx); continue; }
        int8_t *out = (int8_t *)ptrs[tidx];
        uint32_t n = plan.tensors[tidx].size_bytes;
        printf("Output T%u (%u bytes):", tidx, n);
        for (uint32_t j = 0; j < n && j < 20; j++)
            printf(" %d", (int)out[j]);
        printf(" csum=%u\n", checksum(out, n));
    }

    free(ptrs);
    free(slow_buf);
    free(fast_buf);
    free(fbuf);
    return 0;
}

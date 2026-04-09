/**
 * @file test_yolo_s8.c
 * @brief Host-side int8 inference test for YOLOv5n with per-op logging.
 *
 * Usage: ./test_yolo_s8 <plan.tgrs> <input.bin> <budget>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

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

static double elapsed_ms(struct timespec *t0, struct timespec *t1)
{
    return (t1->tv_sec - t0->tv_sec) * 1e3 + (t1->tv_nsec - t0->tv_nsec) / 1e6;
}

static const char *op_name(uint8_t t)
{
    static const char *names[] = {
        [0]="UNK", [1]="Conv", [2]="DWConv", [3]="Relu", [4]="Relu6",
        [5]="MaxPool", [6]="AvgPool", [7]="Add", [8]="Mul", [9]="FC",
        [10]="Softmax", [11]="Clip", [12]="Sigmoid", [13]="Concat",
        [14]="Pad", [15]="GAvgPool", [16]="Flatten", [17]="Reshape",
        [18]="Sub", [19]="Div", [20]="Tanh", [21]="LRelu",
        [22]="BN", [23]="IN", [24]="ConvT", [25]="MatMul",
        [26]="ReduceMean", [27]="Squeeze", [28]="Unsqueeze",
        [29]="Transpose", [30]="Resize",
    };
    if (t < sizeof(names) / sizeof(names[0]) && names[t])
        return names[t];
    return "???";
}

/* Logging dispatch that wraps s8_ref */
static int logging_dispatch(
    const tigris_plan_t *plan, const tigris_op_t *op,
    uint16_t op_index, tigris_mem_t *mem, void *ctx)
{
    (void)ctx;

    int ret = tigris_dispatch_kernel_s8(plan, op, op_index, mem, NULL);

    /* Log first output tensor stats */
    const uint16_t *outs = tigris_op_outputs(plan, op);
    for (uint8_t i = 0; i < op->num_outputs; i++) {
        uint16_t tidx = outs[i];
        const int8_t *data = (const int8_t *)mem->tensor_ptrs[tidx];
        uint32_t sz = plan->tensors[tidx].size_bytes;

        if (!data || sz == 0) continue;

        int8_t mn = 127, mx = -128;
        int64_t sum = 0;
        for (uint32_t j = 0; j < sz; j++) {
            if (data[j] < mn) mn = data[j];
            if (data[j] > mx) mx = data[j];
            sum += data[j];
        }

        printf("Op%03u %-8s -> T%03u [%4d,%4d] mean=%6.1f | %d %d %d %d %d\n",
               op_index, op_name(op->op_type), tidx,
               (int)mn, (int)mx, (double)sum / sz,
               (int)data[0], (int)data[1], (int)data[2], (int)data[3],
               sz > 4 ? (int)data[4] : 0);
    }

    return ret;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <plan.tgrs> <input.bin> <budget>\n", argv[0]);
        return 1;
    }

    const char *plan_path = argv[1];
    const char *input_path = argv[2];
    uint32_t budget = (uint32_t)atoi(argv[3]);

    uint32_t fsize;
    uint8_t *fbuf = load_file(plan_path, &fsize);
    if (!fbuf) return 1;

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(fbuf, fsize, &plan);
    if (err != TIGRIS_OK) {
        fprintf(stderr, "Plan load: %s\n", tigris_error_str(err));
        return 1;
    }

    printf("Model: %s | Ops: %u | Stages: %u\n",
           tigris_model_name(&plan), plan.header->num_ops,
           plan.header->num_stages);

    uint32_t input_size;
    uint8_t *input_data = load_file(input_path, &input_size);
    if (!input_data) return 1;

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

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        tigris_mem_alloc_slow(&mem, tidx, sz);
        uint32_t copy_sz = sz < input_size ? sz : input_size;
        memcpy(ptrs[tidx], input_data, copy_sz);
    }

    printf("\n--- Per-op output (s8_ref) ---\n");
    struct timespec t0, t1;
    tigris_exec_stats_t stats;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, logging_dispatch, NULL, &stats);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "Inference failed: %s\n", tigris_exec_error_str(eerr));
        return 1;
    }

    printf("\nDone! %.1f ms\n", elapsed_ms(&t0, &t1));

    /* Write output tensors */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t tidx = plan.model_outputs[i];
        int8_t *out = (int8_t *)ptrs[tidx];
        uint32_t sz = plan.tensors[tidx].size_bytes;

        char fname[64];
        snprintf(fname, sizeof(fname), "out%u.bin", i);
        FILE *f = fopen(fname, "wb");
        if (f) { fwrite(out, 1, sz, f); fclose(f); }

        int8_t mn = 127, mx = -128;
        int64_t sum = 0;
        for (uint32_t j = 0; j < sz; j++) {
            if (out[j] < mn) mn = out[j];
            if (out[j] > mx) mx = out[j];
            sum += out[j];
        }
        printf("Output %u (T%u): %u bytes [%d,%d] mean=%.1f -> %s\n",
               i, tidx, sz, mn, mx, (double)sum / sz, fname);
    }

    free(input_data);
    free(ptrs);
    free(slow_buf);
    free(fast_buf);
    free(fbuf);
    return 0;
}

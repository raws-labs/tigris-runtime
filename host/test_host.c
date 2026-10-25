#include "tigris_host.h"
#include "tigris_trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "failed at line %d: %s\n", __LINE__, #condition); exit(1); \
} } while (0)

/* A short buffer reports the full event count without overflowing; the first
 * event places the input in input_pool. A buffer of that size holds every
 * event, and their bytes equal the run's counters. */
static void trace_checks(tigris_host_t *host, const void *const *inputs, const uint32_t *in_size,
                         void *const *outputs, const uint32_t *out_size, uint8_t input_pool)
{
    tigris_trace_event_t guard[2];
    tigris_trace_event_t *events;
    uint32_t count;
    uint64_t sums[4] = {0u, 0u, 0u, 0u};
    memset(guard, 0xA5, sizeof(guard));
    tigris_host_trace_buffer(host, guard, 1u);
    CHECK(tigris_host_run(host, inputs, in_size, 1, outputs, out_size, 1) == NULL);
    count = tigris_host_trace_count(host);
    CHECK(count >= 2u && guard[0].kind == TIGRIS_TRACE_ALLOC && guard[0].pool == input_pool);
    CHECK(guard[1].kind == 0xA5u);
    events = calloc(count, sizeof(*events));
    CHECK(events != NULL);
    tigris_host_trace_buffer(host, events, count);
    CHECK(tigris_host_run(host, inputs, in_size, 1, outputs, out_size, 1) == NULL);
    CHECK(tigris_host_trace_count(host) == count);
    for (uint32_t i = 0; i < count; ++i) {
        switch (events[i].kind) {
        case TIGRIS_TRACE_LOAD: sums[0] += events[i].bytes; break;
        case TIGRIS_TRACE_SPILL: sums[1] += events[i].bytes; break;
        case TIGRIS_TRACE_WEIGHTS: sums[2] += events[i].bytes; break;
        case TIGRIS_TRACE_COPY: sums[3] += events[i].bytes; break;
        default: break;
        }
    }
    CHECK(sums[0] == tigris_host_metric(host, TIGRIS_HOST_LOAD_BYTES));
    CHECK(sums[1] == tigris_host_metric(host, TIGRIS_HOST_SPILL_BYTES));
    CHECK(sums[2] == tigris_host_metric(host, TIGRIS_HOST_WEIGHT_BYTES));
    CHECK(sums[3] == tigris_host_metric(host, TIGRIS_HOST_COPY_BYTES));
    tigris_host_trace_buffer(host, NULL, 0u);
    CHECK(tigris_host_run(host, inputs, in_size, 1, outputs, out_size, 1) == NULL);
    CHECK(tigris_host_trace_count(host) == 0u);
    free(events);
}

/* input_pool: fast for a plan that reads its inputs in place, slow otherwise. */
static void fixture(const char *name, uint8_t input_pool)
{
    char path[1024];
    FILE *file;
    long size;
    void *data;
    tigris_host_t *host = NULL;
    tigris_host_t *tight = NULL;
    const char *error;
    uint32_t in_size, out_size;
    void *input, *output;
    const void *inputs[1];
    void *outputs[1];
    CHECK(snprintf(path, sizeof(path), "%s/%s", TIGRIS_SCHEMA_COMPAT_DIR, name) > 0);
#if defined(_MSC_VER)
    CHECK(fopen_s(&file, path, "rb") == 0);
#else
    file = fopen(path, "rb");
#endif
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    CHECK(size > 0 && (uint64_t)size <= UINT32_MAX);
    rewind(file);
    data = malloc((size_t)size);
    CHECK(data != NULL);
    CHECK(fread(data, 1, (size_t)size, file) == (size_t)size);
    CHECK(fclose(file) == 0);
    error = tigris_host_create(data, (uint32_t)size, 0, &host);
    if (error) fprintf(stderr, "%s: %s\n", name, error);
    CHECK(error == NULL && host != NULL);
    CHECK(tigris_host_create(data, (uint32_t)size, 1, &tight) == NULL && tight != NULL);
    memset(data, 0, (size_t)size);
    free(data);
    CHECK(tigris_host_tensor_count(host, 0) == 1u);
    CHECK(tigris_host_tensor_count(host, 1) == 1u);
    CHECK(tigris_host_tensor_name(host, 0, 0) != NULL);
    CHECK(tigris_host_tensor_name(host, 0, 1) == NULL);
    CHECK(tigris_host_tensor_rank(host, 0, 0) > 0u);
    CHECK(tigris_host_tensor_dim(host, 0, 0, 0) > 0);
    in_size = tigris_host_tensor_bytes(host, 0, 0);
    out_size = tigris_host_tensor_bytes(host, 1, 0);
    CHECK(in_size > 0u && out_size > 0u);
    input = calloc(1, in_size);
    output = malloc(out_size);
    CHECK(input && output);
    inputs[0] = input;
    outputs[0] = output;
    if (strcmp(name, "schema-v4-transpose.tgrs") == 0) {
        CHECK(in_size == 24u && out_size == 24u);
        for (int i = 0; i < 6; ++i) ((float *)input)[i] = (float)(i - 3);
    }
    if (strcmp(name, "schema-v6-interface-dtype.tgrs") == 0) {
        CHECK(in_size == 64u && out_size == 64u);
        for (int i = 0; i < 16; ++i) ((float *)input)[i] = (float)(2 * i - 16) * 0.25f;
    }
    /* A one-byte slow arena holds no input that has to go there. */
    error = tigris_host_run(tight, inputs, &in_size, 1, outputs, &out_size, 1);
    if (input_pool == TIGRIS_TRACE_POOL_SLOW)
        CHECK(error != NULL && strcmp(error, "arena cannot hold the tensor") == 0);
    else
        CHECK(error == NULL);
    tigris_host_destroy(tight);
    for (int run = 0; run < 3; ++run) {
        error = tigris_host_run(host, inputs, &in_size, 1, outputs, &out_size, 1);
        if (error) fprintf(stderr, "%s: %s\n", name, error);
        CHECK(error == NULL);
        /* Relu, then a Transpose that keeps the element order. */
        if (strcmp(name, "schema-v4-transpose.tgrs") == 0)
            for (int i = 0; i < 6; ++i)
                CHECK(((float *)output)[i] == (float)(i < 3 ? 0 : i - 3));
        /* A 1x1 int8 Conv of weight 2 at scale 0.25 between scale 0.25
         * boundaries halves each even step of the input grid. */
        if (strcmp(name, "schema-v6-interface-dtype.tgrs") == 0)
            for (int i = 0; i < 16; ++i)
                CHECK(((float *)output)[i] == (float)(i - 8) * 0.25f);
        CHECK(tigris_host_metric(host, TIGRIS_HOST_FAST_PEAK) <=
              tigris_host_metric(host, TIGRIS_HOST_FAST_CAPACITY));
    }
    trace_checks(host, inputs, &in_size, outputs, &out_size, input_pool);
    in_size--;
    CHECK(tigris_host_run(host, inputs, &in_size, 1, outputs, &out_size, 1) != NULL);
    in_size++;
    CHECK(tigris_host_run(host, inputs, &in_size, 0, outputs, &out_size, 1) != NULL);
    CHECK(tigris_host_run(host, inputs, &in_size, 1, outputs, &out_size, 1) == NULL);
    free(input);
    free(output);
    tigris_host_destroy(host);
}

int main(void)
{
    tigris_host_t *host = NULL;
    CHECK(tigris_host_abi() == 2u);
    CHECK(strlen(tigris_host_version()) > 0u);
    CHECK(tigris_host_create("bad", 3, 0, &host) != NULL && host == NULL);
    CHECK(tigris_host_tensor_count(NULL, 0) == 0u);
    tigris_host_destroy(NULL);
    fixture("schema-v4-transpose.tgrs", TIGRIS_TRACE_POOL_SLOW);
    fixture("schema-v6-interface-dtype.tgrs", TIGRIS_TRACE_POOL_FAST);
    fixture("schema-v8-layer-norm.tgrs", TIGRIS_TRACE_POOL_FAST);
    puts("Host library checks passed");
    return 0;
}

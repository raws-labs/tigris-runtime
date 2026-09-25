#include "tigris_host.h"

#include <stdlib.h>
#include <string.h>

#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"

struct tigris_host {
    tigris_plan_t plan;
    tigris_mem_t mem;
    tigris_kernel_fn dispatch;
    tigris_exec_stats_t stats;
    void *plan_allocation;
    void *fast_allocation;
    void *slow_allocation;
    void *workspace;
    size_t workspace_size;
};

static void *allocate_aligned(uint32_t size, void **allocation)
{
    size_t alignment = TIGRIS_TENSOR_ALIGN;
    uintptr_t address;
    if (size == 0u || (uint64_t)size + alignment - 1u > SIZE_MAX)
        return NULL;
    *allocation = malloc((size_t)size + alignment - 1u);
    if (!*allocation)
        return NULL;
    address = (uintptr_t)*allocation;
    return (void *)((address + alignment - 1u) & ~(uintptr_t)(alignment - 1u));
}

uint32_t tigris_host_abi(void) { return 1u; }
const char *tigris_host_version(void) { return TIGRIS_HOST_VERSION; }

void tigris_host_destroy(tigris_host_t *host)
{
    if (host) {
        free(host->workspace);
        free(host->mem.tensor_ptrs);
        free(host->slow_allocation);
        free(host->fast_allocation);
        free(host->plan_allocation);
        free(host);
    }
}

const char *tigris_host_create(
    const void *data, uint32_t size, uint32_t slow_capacity, tigris_host_t **out)
{
    tigris_host_t *host;
    uint8_t *plan_data;
    void *fast;
    void *slow;
    void **pointers;
    uint8_t dtype;
    uint32_t fast_capacity;
    uint64_t slow_required = 0;
    tigris_error_t loaded;
    const char *error = "Host allocation failed";
    if (!out)
        return "Missing session destination";
    *out = NULL;
    if (!data || size == 0u)
        return "Missing plan data";
    host = calloc(1, sizeof(*host));
    if (!host)
        return error;
    plan_data = allocate_aligned(size, &host->plan_allocation);
    if (!plan_data)
        goto fail;
    memcpy(plan_data, data, size);
    loaded = tigris_plan_load(plan_data, size, &host->plan);
    if (loaded != TIGRIS_OK) {
        error = tigris_error_str(loaded);
        goto fail;
    }
    if (host->plan.header->num_model_inputs == 0u ||
        host->plan.header->num_model_outputs == 0u ||
        host->plan.header->num_stages == 0u || host->plan.header->budget == 0u) {
        error = "Host execution requires model inputs, outputs, stages, and a fast budget";
        goto fail;
    }
    dtype = host->plan.tensors[host->plan.model_inputs[0]].dtype;
    if (dtype != 1u && dtype != 3u) {
        error = "Host execution supports float32 and int8 plans";
        goto fail;
    }
    for (uint16_t i = 0; i < host->plan.header->num_tensors; ++i) {
        const tigris_tensor_t *tensor = &host->plan.tensors[i];
        if (!(tensor->flags & TIGRIS_TENSOR_CONSTANT) && tensor->dtype != dtype) {
            error = "Mixed execution dtypes are not supported";
            goto fail;
        }
        slow_required += (uint64_t)tensor->size_bytes + TIGRIS_TENSOR_ALIGN;
    }
    /* Keep room for live tensors and a second copy during arena movement. */
    slow_required *= 2u;
    if (slow_capacity == 0u) {
        if (slow_required > UINT32_MAX) {
            error = "Host slow capacity exceeds the runtime limit";
            goto fail;
        }
        slow_capacity = (uint32_t)slow_required;
    }
    fast_capacity = tigris_fast_arena_required(&host->plan);
    host->workspace_size = tigris_executor_workspace_required(&host->plan);
    if (fast_capacity == UINT32_MAX || host->workspace_size == 0u) {
        error = "Unrepresentable host memory requirements";
        goto fail;
    }
    fast = allocate_aligned(fast_capacity, &host->fast_allocation);
    slow = allocate_aligned(slow_capacity, &host->slow_allocation);
    host->workspace = malloc(host->workspace_size);
    pointers = calloc(host->plan.header->num_tensors, sizeof(*pointers));
    host->mem.tensor_ptrs = pointers;
    if (!fast || !slow || !host->workspace || !pointers)
        goto fail;
    if (tigris_mem_init(&host->mem, pointers, host->plan.header->num_tensors,
                        fast, fast_capacity, slow, slow_capacity) != TIGRIS_MEM_OK) {
        error = "Host arena initialization failed";
        goto fail;
    }
    host->dispatch = dtype == 1u ? tigris_dispatch_kernel : tigris_dispatch_kernel_s8;
    *out = host;
    return NULL;
fail:
    tigris_host_destroy(host);
    return error;
}

uint32_t tigris_host_tensor_count(const tigris_host_t *host, int output)
{
    if (!host || (output != 0 && output != 1))
        return 0u;
    return output ? host->plan.header->num_model_outputs : host->plan.header->num_model_inputs;
}

static const tigris_tensor_t *tensor_at(const tigris_host_t *host, int output, uint32_t index)
{
    uint16_t id;
    if (index >= tigris_host_tensor_count(host, output))
        return NULL;
    id = output ? host->plan.model_outputs[index] : host->plan.model_inputs[index];
    return &host->plan.tensors[id];
}

const char *tigris_host_tensor_name(const tigris_host_t *host, int output, uint32_t index)
{
    const tigris_tensor_t *tensor = tensor_at(host, output, index);
    return tensor ? tigris_tensor_name(&host->plan, tensor) : NULL;
}

uint32_t tigris_host_tensor_bytes(const tigris_host_t *host, int output, uint32_t index)
{
    if (!tensor_at(host, output, index))
        return 0u;
    return tigris_iface_bytes(&host->plan, output ? host->plan.model_outputs[index] : host->plan.model_inputs[index]);
}

uint32_t tigris_host_tensor_dtype(const tigris_host_t *host, int output, uint32_t index)
{
    const tigris_tensor_t *tensor = tensor_at(host, output, index);
    return tensor ? (tensor->iface_dtype ? tensor->iface_dtype : tensor->dtype) : 0u;
}

uint32_t tigris_host_tensor_rank(const tigris_host_t *host, int output, uint32_t index)
{
    const tigris_tensor_t *tensor = tensor_at(host, output, index);
    return tensor ? tensor->ndim : 0u;
}

int32_t tigris_host_tensor_dim(const tigris_host_t *host, int output, uint32_t index, uint32_t axis)
{
    const tigris_tensor_t *tensor = tensor_at(host, output, index);
    return tensor && axis < tensor->ndim ? tigris_tensor_shape(&host->plan, tensor)[axis] : 0;
}

const char *tigris_host_run(
    tigris_host_t *host,
    const void *const *inputs, const uint32_t *input_sizes, uint32_t input_count,
    void *const *outputs, const uint32_t *output_sizes, uint32_t output_count)
{
    tigris_exec_error_t executed;
    if (!host || !inputs || !input_sizes || !outputs || !output_sizes)
        return "Missing session or tensor buffers";
    if (input_count != tigris_host_tensor_count(host, 0) ||
        output_count != tigris_host_tensor_count(host, 1))
        return "Tensor count does not match model interface";
    for (uint32_t i = 0; i < input_count; ++i)
        if (!inputs[i] || input_sizes[i] != tigris_host_tensor_bytes(host, 0, i))
            return "Input byte count does not match model interface";
    for (uint32_t i = 0; i < output_count; ++i)
        if (!outputs[i] || output_sizes[i] != tigris_host_tensor_bytes(host, 1, i))
            return "Output byte count does not match model interface";
    memset(&host->stats, 0, sizeof(host->stats));
    if (tigris_mem_init(&host->mem, host->mem.tensor_ptrs, host->plan.header->num_tensors,
                       host->mem.fast_base, host->mem.fast_size,
                       host->mem.slow_base, host->mem.slow_size) != TIGRIS_MEM_OK)
        return "Host arena reset failed";
    for (uint32_t i = 0; i < input_count; ++i) {
        uint16_t index = host->plan.model_inputs[i];
        tigris_error_t written;
        if (tigris_mem_alloc_slow(&host->mem, index, host->plan.tensors[index].size_bytes) != TIGRIS_MEM_OK)
            return "Host slow arena cannot hold model inputs";
        written = tigris_input_write(&host->plan, &host->mem, index, inputs[i], input_sizes[i]);
        if (written != TIGRIS_OK)
            return tigris_error_str(written);
    }
    executed = tigris_run_with_workspace_buffer(&host->plan, &host->mem, host->dispatch,
                                                NULL, &host->stats, host->workspace, host->workspace_size);
    if (executed != TIGRIS_EXEC_OK)
        return tigris_exec_error_str(executed);
    for (uint32_t i = 0; i < output_count; ++i) {
        tigris_error_t read = tigris_output_read(&host->plan, &host->mem,
                                               host->plan.model_outputs[i], outputs[i], output_sizes[i]);
        if (read != TIGRIS_OK)
            return tigris_error_str(read);
    }
    return NULL;
}

uint64_t tigris_host_metric(const tigris_host_t *host, uint32_t metric)
{
    if (!host)
        return 0u;
    switch (metric) {
    case TIGRIS_HOST_FAST_CAPACITY: return host->mem.fast_size;
    case TIGRIS_HOST_SLOW_CAPACITY: return host->mem.slow_size;
    case TIGRIS_HOST_FAST_PEAK: return host->mem.fast_peak;
    case TIGRIS_HOST_SLOW_PEAK: return host->stats.slow_peak;
    case TIGRIS_HOST_WORKSPACE_BYTES: return host->workspace_size;
    default: return 0u;
    }
}

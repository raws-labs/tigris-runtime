/**
 * @file tigris_mem.c
 * @brief Arena allocator and tensor pointer table - zero malloc.
 */

#include "tigris_mem.h"

#include <string.h>

/* Helpers */

#define ALIGN_UP(x) (((x) + (TIGRIS_TENSOR_ALIGN - 1u)) & ~(TIGRIS_TENSOR_ALIGN - 1u))

/* Public API */

tigris_mem_error_t tigris_mem_init(
    tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t num_tensors,
    void *fast_buf, uint32_t fast_size,
    void *slow_buf, uint32_t slow_size)
{
    if (!mem || !tensor_ptrs || !fast_buf || !slow_buf)
        return TIGRIS_MEM_ERR_NULL;

    mem->tensor_ptrs = tensor_ptrs;
    mem->num_tensors = num_tensors;
    mem->fast_base   = (uint8_t *)fast_buf;
    mem->fast_size   = fast_size;
    mem->slow_base   = (uint8_t *)slow_buf;
    mem->slow_size   = slow_size;

    /* Align initial bump offset so first allocation respects TIGRIS_TENSOR_ALIGN,
     * even if the caller-provided buffer base isn't aligned. */
    uint32_t mask = TIGRIS_TENSOR_ALIGN - 1u;
    uint32_t fast_adj = (TIGRIS_TENSOR_ALIGN - ((uintptr_t)fast_buf & mask)) & mask;
    uint32_t slow_adj = (TIGRIS_TENSOR_ALIGN - ((uintptr_t)slow_buf & mask)) & mask;
    mem->fast_used     = fast_adj;
    mem->fast_reserved = fast_adj;
    mem->fast_peak     = fast_adj;
    mem->slow_used     = slow_adj;

    memset(tensor_ptrs, 0, (size_t)num_tensors * sizeof(void *));
    memset(&mem->tile, 0, sizeof(mem->tile));
    return TIGRIS_MEM_OK;
}

static tigris_mem_error_t alloc_bump(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes,
    uint8_t *base, uint32_t *used, uint32_t capacity)
{
    if (!mem) return TIGRIS_MEM_ERR_NULL;
    if (tensor_idx >= mem->num_tensors) return TIGRIS_MEM_ERR_BAD_INDEX;

    uint32_t aligned = ALIGN_UP(size_bytes);
    if (*used > capacity || aligned > capacity - *used)
        return TIGRIS_MEM_ERR_OOM;

    mem->tensor_ptrs[tensor_idx] = base + *used;
    *used += aligned;
    if (used == &mem->fast_used)
        tigris_mem_note_fast_peak(mem);
    return TIGRIS_MEM_OK;
}

tigris_mem_error_t tigris_mem_alloc_fast(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes)
{
    return alloc_bump(mem, tensor_idx, size_bytes,
                      mem->fast_base, &mem->fast_used, mem->fast_size);
}

tigris_mem_error_t tigris_mem_alloc_slow(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes)
{
    return alloc_bump(mem, tensor_idx, size_bytes,
                      mem->slow_base, &mem->slow_used, mem->slow_size);
}

tigris_mem_error_t tigris_mem_load(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes)
{
    if (!mem) return TIGRIS_MEM_ERR_NULL;
    if (tensor_idx >= mem->num_tensors) return TIGRIS_MEM_ERR_BAD_INDEX;

    void *src = mem->tensor_ptrs[tensor_idx];
    if (!src) return TIGRIS_MEM_ERR_NOT_SET;

    tigris_mem_error_t err = tigris_mem_alloc_fast(mem, tensor_idx, size_bytes);
    if (err != TIGRIS_MEM_OK) return err;

    memcpy(mem->tensor_ptrs[tensor_idx], src, size_bytes);
    return TIGRIS_MEM_OK;
}

tigris_mem_error_t tigris_mem_spill(
    tigris_mem_t *mem, uint16_t tensor_idx, uint32_t size_bytes)
{
    if (!mem) return TIGRIS_MEM_ERR_NULL;
    if (tensor_idx >= mem->num_tensors) return TIGRIS_MEM_ERR_BAD_INDEX;

    void *src = mem->tensor_ptrs[tensor_idx];
    if (!src) return TIGRIS_MEM_ERR_NOT_SET;

    tigris_mem_error_t err = tigris_mem_alloc_slow(mem, tensor_idx, size_bytes);
    if (err != TIGRIS_MEM_OK) return err;

    memcpy(mem->tensor_ptrs[tensor_idx], src, size_bytes);
    return TIGRIS_MEM_OK;
}

tigris_mem_error_t tigris_mem_load_tile(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tensor_idx, int32_t h_start, int32_t h_end)
{
    if (!mem || !plan) return TIGRIS_MEM_ERR_NULL;
    if (tensor_idx >= mem->num_tensors) return TIGRIS_MEM_ERR_BAD_INDEX;

    const tigris_tensor_t *t = &plan->tensors[tensor_idx];
    const int32_t *shape = tigris_tensor_shape(plan, t);

    /* NHWC layout: [N, H, W, C] */
    int32_t N = shape[0];
    int32_t H = shape[1];
    int32_t W = shape[2];
    int32_t C = shape[3];

    int32_t tile_h = h_end - h_start;
    uint32_t elem_size = t->size_bytes / (uint32_t)(N * H * W * C);
    uint32_t row_bytes = (uint32_t)W * (uint32_t)C * elem_size;
    uint32_t tile_rows_bytes = (uint32_t)tile_h * row_bytes;
    uint32_t tile_bytes = (uint32_t)N * tile_rows_bytes;

    /* Source is in slow */
    uint8_t *slow_ptr = (uint8_t *)mem->tensor_ptrs[tensor_idx];
    if (!slow_ptr) return TIGRIS_MEM_ERR_NOT_SET;

    /* Alloc tile in fast */
    tigris_mem_error_t err = tigris_mem_alloc_fast(mem, tensor_idx, tile_bytes);
    if (err != TIGRIS_MEM_OK) return err;

    uint8_t *fast_ptr = (uint8_t *)mem->tensor_ptrs[tensor_idx];

    /* NHWC: rows [h_start, h_end) are contiguous per batch */
    uint32_t full_batch_bytes = (uint32_t)H * row_bytes;
    uint32_t src_row_off = (uint32_t)h_start * row_bytes;

    for (int32_t n = 0; n < N; n++) {
        memcpy(fast_ptr + (uint32_t)n * tile_rows_bytes,
               slow_ptr + (uint32_t)n * full_batch_bytes + src_row_off,
               tile_rows_bytes);
    }

    return TIGRIS_MEM_OK;
}

tigris_mem_error_t tigris_mem_spill_tile(
    tigris_mem_t *mem, const tigris_plan_t *plan,
    uint16_t tensor_idx, void *slow_base, int32_t h_start, int32_t h_end)
{
    if (!mem || !plan || !slow_base) return TIGRIS_MEM_ERR_NULL;
    if (tensor_idx >= mem->num_tensors) return TIGRIS_MEM_ERR_BAD_INDEX;

    const tigris_tensor_t *t = &plan->tensors[tensor_idx];
    const int32_t *shape = tigris_tensor_shape(plan, t);

    /* NHWC layout: [N, H, W, C] */
    int32_t N = shape[0];
    int32_t H = shape[1];
    int32_t W = shape[2];
    int32_t C = shape[3];
    int32_t tile_h = h_end - h_start;
    uint32_t elem_size = t->size_bytes / (uint32_t)(N * H * W * C);
    uint32_t row_bytes = (uint32_t)W * (uint32_t)C * elem_size;

    uint8_t *fast_ptr = (uint8_t *)mem->tensor_ptrs[tensor_idx];
    if (!fast_ptr) return TIGRIS_MEM_ERR_NOT_SET;

    uint8_t *slow_ptr = (uint8_t *)slow_base;
    uint32_t full_batch_bytes = (uint32_t)H * row_bytes;
    uint32_t tile_rows_bytes = (uint32_t)tile_h * row_bytes;
    uint32_t dst_row_off = (uint32_t)h_start * row_bytes;

    for (int32_t n = 0; n < N; n++) {
        memcpy(slow_ptr + (uint32_t)n * full_batch_bytes + dst_row_off,
               fast_ptr + (uint32_t)n * tile_rows_bytes,
               tile_rows_bytes);
    }

    /* Restore pointer to slow base */
    mem->tensor_ptrs[tensor_idx] = slow_base;
    return TIGRIS_MEM_OK;
}

void tigris_mem_reset_fast(tigris_mem_t *mem)
{
    if (mem)
        mem->fast_used = mem->fast_reserved;
}

const char *tigris_mem_error_str(tigris_mem_error_t err)
{
    switch (err) {
        case TIGRIS_MEM_OK:            return "OK";
        case TIGRIS_MEM_ERR_NULL:      return "null pointer argument";
        case TIGRIS_MEM_ERR_OOM:       return "out of memory";
        case TIGRIS_MEM_ERR_BAD_INDEX: return "tensor index out of range";
        case TIGRIS_MEM_ERR_NOT_SET:   return "tensor pointer not set";
        default:                       return "unknown error";
    }
}

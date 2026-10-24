/* Model interface conversion. See include/tigris_iface.h. */
#include "tigris_iface.h"
#include "tigris_io_in_fast.h"

#include <math.h>
#include <string.h>

/* ONNX TensorProto.DataType values this runtime can present at a boundary. */
#define IFACE_DTYPE_FLOAT32 1
#define IFACE_DTYPE_UINT8   2
#define IFACE_DTYPE_INT8    3
#define IFACE_DTYPE_INT32   6
#define IFACE_DTYPE_INT64   7

/* The tensor's declared dtype, which is its own dtype when none is recorded. */
static uint8_t declared_dtype(const tigris_tensor_t *tensor)
{
    return (tensor->iface_dtype != 0u) ? tensor->iface_dtype : tensor->dtype;
}

static const tigris_tensor_t *boundary_tensor(
    const tigris_plan_t *plan, uint16_t tensor_idx, uint8_t flag)
{
    if (!plan || !plan->header || !plan->tensors ||
        tensor_idx >= plan->header->num_tensors)
        return NULL;
    const tigris_tensor_t *tensor = &plan->tensors[tensor_idx];
    if ((tensor->flags & flag) == 0u)
        return NULL;
    return tensor;
}

uint32_t tigris_iface_bytes(const tigris_plan_t *plan, uint16_t tensor_idx)
{
    const tigris_tensor_t *tensor = boundary_tensor(
        plan, tensor_idx,
        (uint8_t)(TIGRIS_TENSOR_MODEL_INPUT | TIGRIS_TENSOR_MODEL_OUTPUT));
    if (!tensor)
        return 0u;
    uint32_t stored = tigris_dtype_size(tensor->dtype);
    uint32_t declared = tigris_dtype_size(declared_dtype(tensor));
    if (stored == 0u || declared == 0u || tensor->size_bytes / stored > UINT32_MAX / declared)
        return 0u;
    return (tensor->size_bytes / stored) * declared;
}

/* Half away from zero, matching the rounding the compiler quantizes with. */
int8_t tigris_quantize_value(float value, const tigris_quant_param_t *quant)
{
    /* roundf rounds half away from zero exactly, as std::round does; adding
     * 0.5 first would round values just below one half up. The level is
     * clamped as a float, which a value beyond int32 would overflow. */
    float level = roundf(value / quant->scale) + (float)quant->zero_point;
    if (!(level >= -128.0f)) level = -128.0f;
    if (level > 127.0f) level = 127.0f;
    return (int8_t)level;
}


/* Common preamble: resolve the boundary tensor, its element count, and the
 * quantization that relates the declared dtype to the stored one. */
static tigris_error_t resolve(
    const tigris_plan_t *plan, uint16_t tensor_idx, uint8_t flag,
    uint32_t buf_bytes, const tigris_tensor_t **out_tensor,
    uint32_t *out_elements, const tigris_quant_param_t **out_quant)
{
    const tigris_tensor_t *tensor = boundary_tensor(plan, tensor_idx, flag);
    if (!tensor)
        return TIGRIS_ERR_BAD_TENSOR;

    uint8_t declared = declared_dtype(tensor);
    uint32_t stored_size = tigris_dtype_size(tensor->dtype);
    uint32_t declared_size = tigris_dtype_size(declared);
    if (stored_size == 0u || declared_size == 0u)
        return TIGRIS_ERR_BAD_INTERFACE;

    uint32_t elements = tensor->size_bytes / stored_size;
    if (elements > UINT32_MAX / declared_size || buf_bytes != elements * declared_size)
        return TIGRIS_ERR_BAD_SIZE;

    const tigris_quant_param_t *quant = tigris_tensor_quant(plan, tensor);
    if (declared != tensor->dtype &&
        !(tensor->dtype == IFACE_DTYPE_INT32 && declared == IFACE_DTYPE_INT64)) {
        /* The conversions this runtime performs are from a float or a uint8
         * interface to a quantized int8 plan tensor. uint8 v and int8 v - 128
         * are the same real value because the compiler moved the zero point
         * by the same 128, so that one is exact. Anything else is refused
         * rather than approximated. */
        if ((declared != IFACE_DTYPE_FLOAT32 &&
             declared != IFACE_DTYPE_UINT8) ||
            tensor->dtype != IFACE_DTYPE_INT8 || !quant ||
            !isfinite(quant->scale) || quant->scale <= 0.0f)
            return TIGRIS_ERR_BAD_INTERFACE;
    }

    *out_tensor = tensor;
    *out_elements = elements;
    *out_quant = quant;
    return TIGRIS_OK;
}

static tigris_error_t placement_error(tigris_mem_error_t placed)
{
    tigris_error_t err;
    if (placed == TIGRIS_MEM_OK) {
        err = TIGRIS_OK;
    } else if (placed == TIGRIS_MEM_ERR_OOM) {
        err = TIGRIS_ERR_MEM;
    } else {
        err = TIGRIS_ERR_NULL;
    }
    return err;
}

/* Whether another model input already has a buffer. */
static int other_input_placed(
    const tigris_plan_t *plan, const tigris_mem_t *mem, uint16_t tensor_idx)
{
    int found = 0;
    for (uint8_t i = 0u; i < plan->header->num_model_inputs; i++) {
        uint16_t other = plan->model_inputs[i];
        if ((other != tensor_idx) && (other < mem->num_tensors) &&
            (mem->tensor_ptrs[other] != NULL)) {
            found = 1;
        }
    }
    return found;
}

tigris_error_t tigris_input_alloc(
    const tigris_plan_t *plan, tigris_mem_t *mem, uint16_t tensor_idx)
{
    tigris_error_t err;
    const tigris_tensor_t *tensor =
        boundary_tensor(plan, tensor_idx, TIGRIS_TENSOR_MODEL_INPUT);
    if ((plan == NULL) || (mem == NULL) || (mem->tensor_ptrs == NULL)) {
        err = TIGRIS_ERR_NULL;
    } else if (tensor == NULL) {
        err = TIGRIS_ERR_BAD_TENSOR;
    } else if (tensor_idx >= mem->num_tensors) {
        err = TIGRIS_ERR_TENSOR_SLOTS;
    } else if (mem->tensor_ptrs[tensor_idx] != NULL) {
        err = TIGRIS_OK;
    } else if (tigris_plan_io_in_fast(plan) == 0) {
        err = placement_error(tigris_mem_alloc_slow(mem, tensor_idx, tensor->size_bytes));
    } else {
        /* The first input leaves the stage's decompressed weights room below it. */
        uint32_t reserve = (other_input_placed(plan, mem, tensor_idx) != 0)
            ? 0u : tigris_weight_decompression_overhead(plan);
        uint32_t used = mem->fast_used;
        if ((reserve == UINT32_MAX) || (used > mem->fast_size) ||
            (reserve > (mem->fast_size - used))) {
            err = TIGRIS_ERR_MEM;
        } else {
            mem->fast_used = used + reserve;
            err = placement_error(tigris_mem_alloc_fast(mem, tensor_idx, tensor->size_bytes));
            if (err != TIGRIS_OK) {
                mem->fast_used = used;
            }
        }
    }
    return err;
}

tigris_error_t tigris_input_write(
    const tigris_plan_t *plan, tigris_mem_t *mem, uint16_t tensor_idx,
    const void *src, uint32_t src_bytes)
{
    if (!mem || !src)
        return TIGRIS_ERR_NULL;

    const tigris_tensor_t *tensor = NULL;
    const tigris_quant_param_t *quant = NULL;
    uint32_t elements = 0u;
    tigris_error_t err = resolve(
        plan, tensor_idx, TIGRIS_TENSOR_MODEL_INPUT, src_bytes,
        &tensor, &elements, &quant);
    if (err != TIGRIS_OK)
        return err;

    err = tigris_input_alloc(plan, mem, tensor_idx);
    if (err != TIGRIS_OK)
        return err;
    void *dst = mem->tensor_ptrs[tensor_idx];

    if (tensor->dtype == 9u) {
        const uint8_t *values = (const uint8_t *)src;
        for (uint32_t i = 0u; i < elements; i++)
            if (values[i] > 1u) return TIGRIS_ERR_BAD_INTERFACE;
    }
    if (declared_dtype(tensor) == tensor->dtype) {
        memcpy(dst, src, src_bytes);
        return TIGRIS_OK;
    }

    if (tensor->dtype == IFACE_DTYPE_INT32) {
        const int64_t *wide = (const int64_t *)src;
        int32_t *indices = (int32_t *)dst;
        for (uint32_t i = 0u; i < elements; i++) {
            int64_t value = wide[i];
            if (value < INT32_MIN || value > INT32_MAX) return TIGRIS_ERR_BAD_INTERFACE;
            indices[i] = (int32_t)value;
        }
        return TIGRIS_OK;
    }

    if (declared_dtype(tensor) == IFACE_DTYPE_UINT8) {
        const uint8_t *u = (const uint8_t *)src;
        int8_t *s = (int8_t *)dst;
        for (uint32_t i = 0u; i < elements; i++)
            s[i] = (int8_t)((int32_t)u[i] - 128);
        return TIGRIS_OK;
    }

    const float *in = (const float *)src;
    int8_t *out = (int8_t *)dst;
    for (uint32_t i = 0u; i < elements; i++)
        out[i] = tigris_quantize_value(in[i], quant);
    return TIGRIS_OK;
}

tigris_error_t tigris_output_read(
    const tigris_plan_t *plan, const tigris_mem_t *mem, uint16_t tensor_idx,
    void *dst, uint32_t dst_bytes)
{
    if (!mem || !dst)
        return TIGRIS_ERR_NULL;

    const tigris_tensor_t *tensor = NULL;
    const tigris_quant_param_t *quant = NULL;
    uint32_t elements = 0u;
    tigris_error_t err = resolve(
        plan, tensor_idx, TIGRIS_TENSOR_MODEL_OUTPUT, dst_bytes,
        &tensor, &elements, &quant);
    if (err != TIGRIS_OK)
        return err;

    const void *src = tigris_mem_tensor_ptr(mem, tensor_idx);
    if (!src)
        return TIGRIS_ERR_NULL;

    if (declared_dtype(tensor) == tensor->dtype) {
        memcpy(dst, src, dst_bytes);
        return TIGRIS_OK;
    }

    if (tensor->dtype == IFACE_DTYPE_INT32) {
        const int32_t *indices = (const int32_t *)src;
        int64_t *wide = (int64_t *)dst;
        for (uint32_t i = 0u; i < elements; i++)
            wide[i] = indices[i];
        return TIGRIS_OK;
    }
    const int8_t *in = (const int8_t *)src;
    if (declared_dtype(tensor) == IFACE_DTYPE_UINT8) {
        uint8_t *u = (uint8_t *)dst;
        for (uint32_t i = 0u; i < elements; i++)
            u[i] = (uint8_t)((int32_t)in[i] + 128);
        return TIGRIS_OK;
    }

    float *out = (float *)dst;
    for (uint32_t i = 0u; i < elements; i++)
        out[i] = ((float)in[i] - (float)quant->zero_point) * quant->scale;
    return TIGRIS_OK;
}

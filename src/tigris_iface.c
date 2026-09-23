/* Model interface conversion. See include/tigris_iface.h. */
#include "tigris_iface.h"

#include <math.h>
#include <string.h>

/* ONNX TensorProto.DataType values this runtime can present at a boundary. */
#define IFACE_DTYPE_FLOAT32 1
#define IFACE_DTYPE_UINT8   2
#define IFACE_DTYPE_INT8    3

static uint32_t dtype_size(uint8_t dtype)
{
    if (dtype == IFACE_DTYPE_FLOAT32) return (uint32_t)sizeof(float);
    if (dtype == IFACE_DTYPE_INT8) return (uint32_t)sizeof(int8_t);
    if (dtype == IFACE_DTYPE_UINT8) return (uint32_t)sizeof(uint8_t);
    return 0u;
}

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
    uint32_t stored = dtype_size(tensor->dtype);
    uint32_t declared = dtype_size(declared_dtype(tensor));
    if (stored == 0u || declared == 0u)
        return 0u;
    return (tensor->size_bytes / stored) * declared;
}

/* Half away from zero, matching the rounding the compiler quantizes with. */
static int32_t round_half_away(float value)
{
    return (value >= 0.0f) ? (int32_t)floorf(value + 0.5f)
                           : (int32_t)ceilf(value - 0.5f);
}

static int8_t clamp_s8(int32_t value)
{
    if (value < -128) return (int8_t)-128;
    if (value > 127) return (int8_t)127;
    return (int8_t)value;
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
    uint32_t stored_size = dtype_size(tensor->dtype);
    uint32_t declared_size = dtype_size(declared);
    if (stored_size == 0u || declared_size == 0u)
        return TIGRIS_ERR_BAD_INTERFACE;

    uint32_t elements = tensor->size_bytes / stored_size;
    if (buf_bytes != elements * declared_size)
        return TIGRIS_ERR_BAD_SIZE;

    const tigris_quant_param_t *quant = tigris_tensor_quant(plan, tensor);
    if (declared != tensor->dtype) {
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

    void *dst = tigris_mem_tensor_ptr(mem, tensor_idx);
    if (!dst)
        return TIGRIS_ERR_NULL;

    if (declared_dtype(tensor) == tensor->dtype) {
        memcpy(dst, src, src_bytes);
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
        out[i] = clamp_s8(round_half_away(in[i] / quant->scale) +
                          quant->zero_point);
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

/**
 * Model interface conversion.
 *
 * A plan executes on one dtype, which is not always the dtype the model file
 * declares for its inputs and outputs: an ONNX graph states float32 at the
 * boundary and quantizes inside itself, and the compiler folds that
 * quantization into the boundary tensor. These calls move data across that
 * boundary, so an application hands over and reads back exactly what the model
 * declares. ``tensor->iface_dtype`` names the declared dtype; zero means the
 * plan already stores what the caller supplies.
 */
#ifndef TIGRIS_IFACE_H
#define TIGRIS_IFACE_H

#include <stdint.h>

#include "tigris.h"
#include "tigris_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Bytes the caller supplies for a model input, or receives for a model output,
 * in the dtype the model declares.
 *
 * @param plan        Loaded plan.
 * @param tensor_idx  Index of a model input or output tensor.
 * @return Byte count, or 0 if the tensor is not a model boundary tensor.
 */
uint32_t tigris_iface_bytes(const tigris_plan_t *plan, uint16_t tensor_idx);

/**
 * Convert the caller's data into a model input tensor.
 *
 * @param plan        Loaded plan.
 * @param mem         Memory manager with the input tensor allocated.
 * @param tensor_idx  Index of a model input tensor.
 * @param src         Caller data in the declared dtype.
 * @param src_bytes   Size of src, which must equal tigris_iface_bytes().
 * @return TIGRIS_OK, or a negative error code.
 */
tigris_error_t tigris_input_write(
    const tigris_plan_t *plan, tigris_mem_t *mem, uint16_t tensor_idx,
    const void *src, uint32_t src_bytes);

/**
 * Convert a model output tensor into the caller's buffer.
 *
 * @param plan        Loaded plan.
 * @param mem         Memory manager after a run.
 * @param tensor_idx  Index of a model output tensor.
 * @param dst         Caller buffer receiving the declared dtype.
 * @param dst_bytes   Size of dst, which must equal tigris_iface_bytes().
 * @return TIGRIS_OK, or a negative error code.
 */
tigris_error_t tigris_output_read(
    const tigris_plan_t *plan, const tigris_mem_t *mem, uint16_t tensor_idx,
    void *dst, uint32_t dst_bytes);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_IFACE_H */

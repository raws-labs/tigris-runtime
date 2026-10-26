/**
 * Model interface conversion.
 *
 * A plan executes on one dtype, which is not always the dtype the model file
 * declares for its inputs and outputs: an ONNX graph states float32 or uint8 at
 * the boundary and quantizes inside itself, and the compiler folds that
 * quantization into the boundary tensor, which is int8. These calls move data
 * across that
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
 * Place a model input where the run reads it.
 *
 * A plan reads its inputs in place when it has schema 5 or later and a single
 * stage, without a tile plan, chain, control flow or state, that reads every
 * model input and writes every model output. Such a plan's input goes in the
 * fast arena, and the first of its inputs leaves
 * tigris_weight_decompression_overhead() bytes free directly below it, where
 * the run decompresses the stage's weights. Place every input of such a plan
 * this way, one after another, so they sit together at the top of the fast
 * arena; the run then reads them there, consumes them, and leaves the model
 * outputs in the fast arena. Any other plan's input goes in the slow buffer.
 * An input that already has a buffer keeps it.
 *
 * @param plan        Loaded plan.
 * @param mem         Memory manager.
 * @param tensor_idx  Index of a model input tensor.
 * @return TIGRIS_OK, TIGRIS_ERR_NULL when plan or mem is NULL or mem has no
 *         pointer table, TIGRIS_ERR_BAD_TENSOR when tensor_idx is not a model
 *         input, TIGRIS_ERR_TENSOR_SLOTS when mem has no slot for it, or
 *         TIGRIS_ERR_MEM when the arena cannot hold it. A failed call leaves
 *         mem unchanged.
 */
tigris_error_t tigris_input_alloc(
    const tigris_plan_t *plan, tigris_mem_t *mem, uint16_t tensor_idx);

/**
 * Convert the caller's data into a model input tensor.
 * Int64 index inputs must fit int32; narrowing overflow returns BAD_INTERFACE.
 *
 * An input with no buffer yet is placed with tigris_input_alloc() first. A
 * buffer the caller allocated beforehand is used as it is.
 *
 * @param plan        Loaded plan.
 * @param mem         Memory manager.
 * @param tensor_idx  Index of a model input tensor.
 * @param src         Caller data in the declared dtype.
 * @param src_bytes   Size of src, which must equal tigris_iface_bytes().
 * @return TIGRIS_OK, an error of tigris_input_alloc(), or another negative
 *         error code.
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

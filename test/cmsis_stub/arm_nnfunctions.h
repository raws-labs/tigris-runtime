/* CMSIS-NN types and declarations the adapter uses, with the upstream
 * signatures, for host builds. Declarations only. */
#ifndef TIGRIS_TEST_ARM_NNFUNCTIONS_H
#define TIGRIS_TEST_ARM_NNFUNCTIONS_H

#include <stdint.h>

typedef enum {
    ARM_CMSIS_NN_SUCCESS = 0
} arm_cmsis_nn_status;

typedef struct { int32_t w, h; } cmsis_nn_tile;
typedef struct { int32_t min, max; } cmsis_nn_activation;
typedef struct { int32_t n, h, w, c; } cmsis_nn_dims;
typedef struct { void *buf; int32_t size; } cmsis_nn_context;

typedef struct {
    int32_t input_offset;
    int32_t output_offset;
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile dilation;
    cmsis_nn_activation activation;
} cmsis_nn_conv_params;

typedef struct {
    int32_t input_offset;
    int32_t output_offset;
    int32_t ch_mult;
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile dilation;
    cmsis_nn_activation activation;
} cmsis_nn_dw_conv_params;

typedef struct {
    int32_t input_offset;
    int32_t filter_offset;
    int32_t output_offset;
    cmsis_nn_activation activation;
} cmsis_nn_fc_params;

typedef struct {
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_activation activation;
} cmsis_nn_pool_params;

typedef struct { int32_t *multiplier; int32_t *shift; }
    cmsis_nn_per_channel_quant_params;
typedef struct { int32_t multiplier; int32_t shift; }
    cmsis_nn_per_tensor_quant_params;

int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
    const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims, const cmsis_nn_dims *output_dims);
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *dw_conv_params,
    const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims, const cmsis_nn_dims *output_dims);
int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);
int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_conv_params *conv_params, const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims, const int8_t *input_data, const cmsis_nn_dims *filter_dims,
    const int8_t *filter_data, const cmsis_nn_dims *bias_dims, const int32_t *bias_data,
    const cmsis_nn_dims *output_dims, int8_t *output_data);
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_dw_conv_params *dw_conv_params, const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims, const int8_t *input_data, const cmsis_nn_dims *filter_dims,
    const int8_t *filter_data, const cmsis_nn_dims *bias_dims, const int32_t *bias_data,
    const cmsis_nn_dims *output_dims, int8_t *output_data);
arm_cmsis_nn_status arm_fully_connected_per_channel_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_fc_params *fc_params, const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims, const int8_t *input_data, const cmsis_nn_dims *filter_dims,
    const int8_t *filter_data, const cmsis_nn_dims *bias_dims, const int32_t *bias_data,
    const cmsis_nn_dims *output_dims, int8_t *output_data);
arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_fc_params *fc_params, const cmsis_nn_per_tensor_quant_params *quant_params,
    const cmsis_nn_dims *input_dims, const int8_t *input_data, const cmsis_nn_dims *filter_dims,
    const int8_t *filter_data, const cmsis_nn_dims *bias_dims, const int32_t *bias_data,
    const cmsis_nn_dims *output_dims, int8_t *output_data);
arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx, const cmsis_nn_pool_params *pool_params,
    const cmsis_nn_dims *input_dims, const int8_t *input_data, const cmsis_nn_dims *filter_dims,
    const cmsis_nn_dims *output_dims, int8_t *output_data);
arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *input_1_vect, const int8_t *input_2_vect,
    const int32_t input_1_offset, const int32_t input_2_offset, int8_t *output, const int32_t out_offset,
    const int32_t out_mult, const int32_t out_shift, const int32_t out_activation_min,
    const int32_t out_activation_max, const int32_t block_size);
arm_cmsis_nn_status arm_maximum_s8(const cmsis_nn_context *ctx, const int8_t *input_1_data,
    const cmsis_nn_dims *input_1_dims, const int8_t *input_2_data, const cmsis_nn_dims *input_2_dims,
    int8_t *output_data, const cmsis_nn_dims *output_dims);
arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx, const int8_t *input_1_data,
    const cmsis_nn_dims *input_1_dims, const int8_t *input_2_data, const cmsis_nn_dims *input_2_dims,
    int8_t *output_data, const cmsis_nn_dims *output_dims);
arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf, const int32_t vector_cols,
    const int32_t vector_rows, const int8_t *vector_data, const int32_t lhs_offset,
    const int32_t rhs_offset, const int32_t *bias_data);

#endif

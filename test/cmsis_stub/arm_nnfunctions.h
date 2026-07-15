/* Minimal CMSIS-NN API surface for host compile coverage only. */
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

int32_t arm_convolve_wrapper_s8_get_buffer_size(
    const cmsis_nn_conv_params *, ...);
int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(
    const cmsis_nn_dw_conv_params *, ...);
int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *);
int32_t arm_avgpool_s8_get_buffer_size(int32_t, int32_t);

arm_cmsis_nn_status arm_convolve_wrapper_s8(
    const cmsis_nn_context *, ...);
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(
    const cmsis_nn_context *, ...);
arm_cmsis_nn_status arm_fully_connected_per_channel_s8(
    const cmsis_nn_context *, ...);
arm_cmsis_nn_status arm_fully_connected_s8(
    const cmsis_nn_context *, ...);
arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *, ...);
void arm_vector_sum_s8(int32_t *, ...);

#endif

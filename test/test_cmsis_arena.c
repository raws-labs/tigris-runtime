/** Host tests for the CMSIS-NN adapter: arena sizing and the arguments of every vendor call. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "tigris_kernels_cmsis_nn.h"
#include "tigris_kernels_s8.h"
#include "test_harness.h"

/* Vendor stubs. Each one copies the arguments it receives into a record. */

typedef enum {
    CALL_NONE, CALL_CONV, CALL_DEPTHWISE, CALL_FC, CALL_FC_PER_CHANNEL, CALL_AVGPOOL,
    CALL_MUL, CALL_MAXIMUM, CALL_MINIMUM
} vendor_fn_t;

#define RECORDED_CHANNELS 16

typedef struct {
    vendor_fn_t fn;
    int calls;
    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv;
    cmsis_nn_dw_conv_params dw;
    cmsis_nn_fc_params fc;
    cmsis_nn_pool_params pool;
    cmsis_nn_per_tensor_quant_params tensor_quant;
    const int32_t *multiplier;
    const int32_t *shift;
    int32_t multipliers[RECORDED_CHANNELS];
    int32_t shifts[RECORDED_CHANNELS];
    cmsis_nn_dims input, input2, filter, bias, output;
    const int8_t *input_data, *input2_data, *filter_data;
    const int32_t *bias_data;
    int8_t *output_data;
    int32_t input_1_offset, input_2_offset, out_offset, out_mult, out_shift;
    int32_t out_activation_min, out_activation_max, block_size;
} vendor_call_t;

typedef struct {
    vendor_fn_t fn;
    int calls;
    cmsis_nn_conv_params conv;
    cmsis_nn_dw_conv_params dw;
    cmsis_nn_dims input, filter, output;
    int32_t dst_width, src_channels;
} size_query_t;

typedef struct {
    int calls;
    int32_t *buf;
    int32_t cols, rows;
    const int8_t *data;
    int32_t lhs_offset, rhs_offset;
    const int32_t *bias;
} vector_sum_t;

static vendor_call_t call;
static size_query_t query;
static vector_sum_t vsum;
static int32_t query_bytes = 33;

static void reset_records(void)
{
    memset(&call, 0, sizeof(call));
    memset(&query, 0, sizeof(query));
    memset(&vsum, 0, sizeof(vsum));
}

static void record_operands(vendor_fn_t fn, const cmsis_nn_context *ctx,
    const cmsis_nn_dims *input, const int8_t *input_data,
    const cmsis_nn_dims *filter, const int8_t *filter_data,
    const cmsis_nn_dims *bias, const int32_t *bias_data,
    const cmsis_nn_dims *output, int8_t *output_data)
{
    call.fn = fn;
    call.calls++;
    call.ctx = *ctx;
    call.input = *input;
    call.input_data = input_data;
    call.filter = *filter;
    call.filter_data = filter_data;
    if (bias)
        call.bias = *bias;
    call.bias_data = bias_data;
    call.output = *output;
    call.output_data = output_data;
}

static void record_channels(const cmsis_nn_per_channel_quant_params *quant, int32_t channels)
{
    call.multiplier = quant->multiplier;
    call.shift = quant->shift;
    for (int32_t c = 0; c < channels && c < RECORDED_CHANNELS; c++) {
        call.multipliers[c] = quant->multiplier[c];
        call.shifts[c] = quant->shift[c];
    }
}

static void record_query(vendor_fn_t fn, const cmsis_nn_dims *input,
                         const cmsis_nn_dims *filter, const cmsis_nn_dims *output)
{
    query.fn = fn;
    query.calls++;
    if (input)
        query.input = *input;
    query.filter = *filter;
    if (output)
        query.output = *output;
}

int32_t arm_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *params,
    const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims, const cmsis_nn_dims *output_dims)
{
    record_query(CALL_CONV, input_dims, filter_dims, output_dims);
    query.conv = *params;
    return query_bytes;
}

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(const cmsis_nn_dw_conv_params *params,
    const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims, const cmsis_nn_dims *output_dims)
{
    record_query(CALL_DEPTHWISE, input_dims, filter_dims, output_dims);
    query.dw = *params;
    return query_bytes;
}

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims)
{
    record_query(CALL_FC, NULL, filter_dims, NULL);
    return query_bytes;
}

int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src)
{
    query.fn = CALL_AVGPOOL;
    query.calls++;
    query.dst_width = dim_dst_width;
    query.src_channels = ch_src;
    return query_bytes;
}

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx, const cmsis_nn_conv_params *params,
    const cmsis_nn_per_channel_quant_params *quant, const cmsis_nn_dims *input_dims, const int8_t *input,
    const cmsis_nn_dims *filter_dims, const int8_t *filter, const cmsis_nn_dims *bias_dims, const int32_t *bias,
    const cmsis_nn_dims *output_dims, int8_t *output)
{
    record_operands(CALL_CONV, ctx, input_dims, input, filter_dims, filter, bias_dims, bias, output_dims, output);
    call.conv = *params;
    record_channels(quant, output_dims->c);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_dw_conv_params *params, const cmsis_nn_per_channel_quant_params *quant,
    const cmsis_nn_dims *input_dims, const int8_t *input, const cmsis_nn_dims *filter_dims, const int8_t *filter,
    const cmsis_nn_dims *bias_dims, const int32_t *bias, const cmsis_nn_dims *output_dims, int8_t *output)
{
    record_operands(CALL_DEPTHWISE, ctx, input_dims, input, filter_dims, filter, bias_dims, bias, output_dims,
                    output);
    call.dw = *params;
    record_channels(quant, output_dims->c);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_fully_connected_per_channel_s8(const cmsis_nn_context *ctx,
    const cmsis_nn_fc_params *params, const cmsis_nn_per_channel_quant_params *quant,
    const cmsis_nn_dims *input_dims, const int8_t *input, const cmsis_nn_dims *filter_dims, const int8_t *filter,
    const cmsis_nn_dims *bias_dims, const int32_t *bias, const cmsis_nn_dims *output_dims, int8_t *output)
{
    record_operands(CALL_FC_PER_CHANNEL, ctx, input_dims, input, filter_dims, filter, bias_dims, bias,
                    output_dims, output);
    call.fc = *params;
    record_channels(quant, output_dims->c);
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx, const cmsis_nn_fc_params *params,
    const cmsis_nn_per_tensor_quant_params *quant, const cmsis_nn_dims *input_dims, const int8_t *input,
    const cmsis_nn_dims *filter_dims, const int8_t *filter, const cmsis_nn_dims *bias_dims, const int32_t *bias,
    const cmsis_nn_dims *output_dims, int8_t *output)
{
    record_operands(CALL_FC, ctx, input_dims, input, filter_dims, filter, bias_dims, bias, output_dims, output);
    call.fc = *params;
    call.tensor_quant = *quant;
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_avgpool_s8(const cmsis_nn_context *ctx, const cmsis_nn_pool_params *params,
    const cmsis_nn_dims *input_dims, const int8_t *input, const cmsis_nn_dims *filter_dims,
    const cmsis_nn_dims *output_dims, int8_t *output)
{
    record_operands(CALL_AVGPOOL, ctx, input_dims, input, filter_dims, NULL, NULL, NULL, output_dims, output);
    call.pool = *params;
    return ARM_CMSIS_NN_SUCCESS;
}

/* The binary stubs also compute their result, so the adapter output can be
 * compared with the s8 reference kernel. */
arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *a, const int8_t *b, const int32_t a_offset,
    const int32_t b_offset, int8_t *y, const int32_t y_offset, const int32_t mult, const int32_t shift,
    const int32_t act_min, const int32_t act_max, const int32_t count)
{
    call.fn = CALL_MUL;
    call.calls++;
    call.input_data = a;
    call.input2_data = b;
    call.output_data = y;
    call.input_1_offset = a_offset;
    call.input_2_offset = b_offset;
    call.out_offset = y_offset;
    call.out_mult = mult;
    call.out_shift = shift;
    call.out_activation_min = act_min;
    call.out_activation_max = act_max;
    call.block_size = count;
    for (int32_t i = 0; i < count; i++) {
        int32_t product = ((int32_t)a[i] + a_offset) * ((int32_t)b[i] + b_offset);
        product *= 1 << (shift > 0 ? shift : 0);
        int32_t value = (int32_t)(((int64_t)product * mult + ((int64_t)1 << 30)) >> 31);
        int right = shift < 0 ? -shift : 0;
        int32_t mask = (1 << right) - 1;
        value = (value >> right) + ((value & mask) > ((mask >> 1) + (value < 0)));
        value += y_offset;
        y[i] = (int8_t)(value < act_min ? act_min : value > act_max ? act_max : value);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

static void record_extremum(vendor_fn_t fn, const cmsis_nn_context *ctx, const int8_t *a,
    const cmsis_nn_dims *ad, const int8_t *b, const cmsis_nn_dims *bd, int8_t *y, const cmsis_nn_dims *yd)
{
    call.fn = fn;
    call.calls++;
    call.ctx = *ctx;
    call.input_data = a;
    call.input = *ad;
    call.input2_data = b;
    call.input2 = *bd;
    call.output_data = y;
    call.output = *yd;
}

arm_cmsis_nn_status arm_maximum_s8(const cmsis_nn_context *ctx, const int8_t *a, const cmsis_nn_dims *ad,
    const int8_t *b, const cmsis_nn_dims *bd, int8_t *y, const cmsis_nn_dims *yd)
{
    record_extremum(CALL_MAXIMUM, ctx, a, ad, b, bd, y, yd);
    for (int32_t i = 0; i < yd->c; i++) y[i] = a[i] > b[i] ? a[i] : b[i];
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx, const int8_t *a, const cmsis_nn_dims *ad,
    const int8_t *b, const cmsis_nn_dims *bd, int8_t *y, const cmsis_nn_dims *yd)
{
    record_extremum(CALL_MINIMUM, ctx, a, ad, b, bd, y, yd);
    for (int32_t i = 0; i < yd->c; i++) y[i] = a[i] < b[i] ? a[i] : b[i];
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf, const int32_t vector_cols,
    const int32_t vector_rows, const int8_t *vector_data, const int32_t lhs_offset, const int32_t rhs_offset,
    const int32_t *bias_data)
{
    vsum.calls++;
    vsum.buf = vector_sum_buf;
    vsum.cols = vector_cols;
    vsum.rows = vector_rows;
    vsum.data = vector_data;
    vsum.lhs_offset = lhs_offset;
    vsum.rhs_offset = rhs_offset;
    vsum.bias = bias_data;
    return ARM_CMSIS_NN_SUCCESS;
}

/* One-operator plan fixture. */

/* Quant arrays sit on page 1; page 0 holds other values at the same offsets. */
#define QUANT_PAGE 1u
#define QUANT_BASE (QUANT_PAGE * TIGRIS_QUANT_PAGE_ELEMS)
#define OUT_QUANT_OFF 8u
#define BIAS_OFF 256u

static int32_t quant_data[TIGRIS_QUANT_PAGE_ELEMS + 64u];
static _Alignas(16) uint8_t weight_blob[512];
static _Alignas(TIGRIS_TENSOR_ALIGN) int8_t input_buf[512];
static _Alignas(TIGRIS_TENSOR_ALIGN) int8_t output_buf[512];
static _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast_arena[1024];

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    uint16_t indices[2];
    int32_t shapes[8];
    tigris_quant_param_t quant[2];
    tigris_weight_entry_t weights[2];
    tigris_op_attribute_t attribute;
    uint8_t attribute_data[1];
    tigris_plan_t plan;
} fixture_t;

static void build_fixture(fixture_t *fx)
{
    memset(fx, 0, sizeof(*fx));
    fx->header.budget = 100;
    fx->header.num_ops = 1;
    fx->header.num_tensors = 2;
    fx->op.op_type = TIGRIS_OP_CONV;
    fx->op.num_inputs = 1;
    fx->op.num_outputs = 1;
    fx->op.inputs_off = 0;
    fx->op.outputs_off = 1;
    fx->op.spatial.kernel_h = 3;
    fx->op.spatial.kernel_w = 3;
    fx->op.spatial.stride_h = 1;
    fx->op.spatial.stride_w = 1;
    fx->op.spatial.dilation_h = 1;
    fx->op.spatial.dilation_w = 1;
    fx->op.weight_idx = TIGRIS_NO_WEIGHT;
    fx->op.bias_idx = TIGRIS_NO_WEIGHT;
    fx->indices[0] = 0;
    fx->indices[1] = 1;
    for (uint16_t i = 0; i < 2; i++) {
        fx->tensors[i].shape_off = (uint16_t)(i * 4);
        fx->tensors[i].ndim = 4;
        fx->tensors[i].dtype = 3;
        fx->tensors[i].size_bytes = 64;
        fx->shapes[i * 4] = 1;
        fx->shapes[i * 4 + 1] = 4;
        fx->shapes[i * 4 + 2] = 4;
        fx->shapes[i * 4 + 3] = 4;
    }
    fx->plan.header = &fx->header;
    fx->plan.tensors = fx->tensors;
    fx->plan.ops = &fx->op;
    fx->plan.index_pool = fx->indices;
    fx->plan.shape_pool = fx->shapes;
}

static void set_shape(fixture_t *fx, uint16_t tensor, uint8_t ndim, const int32_t *dims)
{
    uint32_t bytes = 1;
    for (uint8_t i = 0; i < ndim; i++) {
        fx->shapes[tensor * 4u + i] = dims[i];
        bytes *= (uint32_t)dims[i];
    }
    fx->tensors[tensor].ndim = ndim;
    fx->tensors[tensor].size_bytes = bytes;
}

static int32_t out_multiplier(uint32_t channel)
{
    return 1073741824 + (int32_t)channel * 4096;
}

static int32_t out_shift(uint32_t channel)
{
    return -1 - (int32_t)channel;
}

static void attach_quant(fixture_t *fx, int32_t in_zp, int32_t out_zp, uint16_t out_channels)
{
    memset(quant_data, 0, sizeof(quant_data));
    quant_data[QUANT_BASE] = 1500000000;
    quant_data[QUANT_BASE + 1u] = -9;
    for (uint32_t c = 0; c < out_channels; c++) {
        quant_data[QUANT_BASE + OUT_QUANT_OFF + c] = out_multiplier(c);
        quant_data[QUANT_BASE + OUT_QUANT_OFF + out_channels + c] = out_shift(c);
        quant_data[OUT_QUANT_OFF + c] = 7;
        quant_data[OUT_QUANT_OFF + out_channels + c] = 7;
    }
    fx->quant[0].scale = 0.5f;
    fx->quant[0].zero_point = in_zp;
    fx->quant[0].num_channels = 1;
    fx->quant[0].multiplier_off = 0;
    fx->quant[0].shift_off = 1;
    fx->quant[0]._pad = QUANT_PAGE;
    fx->quant[1].scale = 0.25f;
    fx->quant[1].zero_point = out_zp;
    fx->quant[1].num_channels = out_channels;
    fx->quant[1].multiplier_off = OUT_QUANT_OFF;
    fx->quant[1].shift_off = (uint16_t)(OUT_QUANT_OFF + out_channels);
    fx->quant[1]._pad = QUANT_PAGE;
    fx->tensors[0].quant_param_idx = 0;
    fx->tensors[1].quant_param_idx = 1;
    fx->header.num_quant_params = 2;
    fx->plan.quant_params = fx->quant;
    fx->plan.quant_data = quant_data;
    fx->plan.num_quant_params = 2;
}

static void attach_weights(fixture_t *fx)
{
    fx->weights[0].offset = 0;
    fx->weights[0].size_bytes = BIAS_OFF;
    fx->weights[1].offset = BIAS_OFF;
    fx->weights[1].size_bytes = 64;
    fx->op.weight_idx = 0;
    fx->op.bias_idx = 1;
    fx->header.num_weights = 2;
    fx->plan.weight_entries = fx->weights;
    fx->plan.weight_blob = weight_blob;
}

/* Conv [1,7,6,3] -> [1,OH,6,8]: kernel 3x2, stride 2x1, pads 2/2 top/bottom
 * and 1/0 left/right, H dilation 2 (OH 4), else OH 5; W dilation 0. Dilation 0
 * means 1. */
static void build_conv(fixture_t *fx, uint16_t dilation_h, uint16_t out_quant_channels)
{
    static const int32_t x[4] = {1, 7, 6, 3};
    const int32_t y[4] = {1, dilation_h == 2u ? 4 : 5, 6, 8};
    build_fixture(fx);
    set_shape(fx, 0, 4, x);
    set_shape(fx, 1, 4, y);
    fx->op.spatial.kernel_h = 3;
    fx->op.spatial.kernel_w = 2;
    fx->op.spatial.stride_h = 2;
    fx->op.spatial.stride_w = 1;
    fx->op.spatial.pad_top = 2;
    fx->op.spatial.pad_bottom = 2;
    fx->op.spatial.pad_left = 1;
    fx->op.spatial.pad_right = 0;
    fx->op.spatial.dilation_h = dilation_h;
    fx->op.spatial.dilation_w = 0;
    fx->op.spatial.group = 1;
    fx->op.fused_act = TIGRIS_ACT_RELU;
    fx->op.act_min = -20;
    fx->op.act_max = 100;
    attach_weights(fx);
    if (out_quant_channels)
        attach_quant(fx, -5, 7, out_quant_channels);
}

/* Depthwise [1,7,6,4] -> [1,4,6,12] (multiplier 3): kernel 3x1, stride 2x1,
 * pads 1/1 top/bottom; H dilation 0, W dilation as given. Dilation 0 means 1. */
static void build_depthwise(fixture_t *fx, uint16_t dilation_w, uint16_t out_quant_channels)
{
    static const int32_t x[4] = {1, 7, 6, 4};
    static const int32_t y[4] = {1, 4, 6, 12};
    build_fixture(fx);
    fx->op.op_type = TIGRIS_OP_DEPTHWISE;
    set_shape(fx, 0, 4, x);
    set_shape(fx, 1, 4, y);
    fx->op.spatial.kernel_h = 3;
    fx->op.spatial.kernel_w = 1;
    fx->op.spatial.stride_h = 2;
    fx->op.spatial.stride_w = 1;
    fx->op.spatial.pad_top = 1;
    fx->op.spatial.pad_bottom = 1;
    fx->op.spatial.dilation_h = 0;
    fx->op.spatial.dilation_w = dilation_w;
    fx->op.spatial.group = 4;
    fx->op.fused_act = TIGRIS_ACT_RELU6;
    fx->op.act_min = 3;
    fx->op.act_max = 60;
    attach_weights(fx);
    attach_quant(fx, 11, -9, out_quant_channels);
}

/* Fully connected [2,3,4] -> [2,5], or [12] -> [5] when flat. */
static void build_fc(fixture_t *fx, int flat, uint16_t out_quant_channels)
{
    static const int32_t x[3] = {2, 3, 4};
    static const int32_t y[2] = {2, 5};
    static const int32_t x_flat[1] = {12};
    static const int32_t y_flat[1] = {5};
    build_fixture(fx);
    fx->op.op_type = TIGRIS_OP_FULLY_CONN;
    memset(&fx->op.spatial, 0, sizeof(fx->op.spatial));
    if (flat) {
        set_shape(fx, 0, 1, x_flat);
        set_shape(fx, 1, 1, y_flat);
    } else {
        set_shape(fx, 0, 3, x);
        set_shape(fx, 1, 2, y);
    }
    fx->op.act_min = -128;
    fx->op.act_max = 90;
    attach_weights(fx);
    attach_quant(fx, -5, 7, out_quant_channels);
}

/* Average pool [1,6,5,4] -> [1,3,5,4]: kernel 2x3, stride 2x1, pads 0 top
 * and 1/1 left/right. A global pool reduces to [1,1,1,4]. Input and output
 * share their quantization. */
static void build_pool(fixture_t *fx, int global)
{
    static const int32_t x[4] = {1, 6, 5, 4};
    static const int32_t y[4] = {1, 3, 5, 4};
    static const int32_t y_global[4] = {1, 1, 1, 4};
    build_fixture(fx);
    memset(&fx->op.spatial, 0, sizeof(fx->op.spatial));
    set_shape(fx, 0, 4, x);
    if (global) {
        fx->op.op_type = TIGRIS_OP_GLOBAL_AVG;
        set_shape(fx, 1, 4, y_global);
    } else {
        fx->op.op_type = TIGRIS_OP_AVG_POOL;
        set_shape(fx, 1, 4, y);
        fx->op.spatial.kernel_h = 2;
        fx->op.spatial.kernel_w = 3;
        fx->op.spatial.stride_h = 2;
        fx->op.spatial.stride_w = 1;
        fx->op.spatial.pad_left = 1;
        fx->op.spatial.pad_right = 1;
    }
    fx->op.act_min = -128;
    fx->op.act_max = 127;
    attach_quant(fx, -3, -3, 1);
    fx->quant[1].scale = fx->quant[0].scale;
}

static void add_pool_rounding(fixture_t *fx)
{
    fx->attribute.op_index = 0;
    fx->attribute.type = TIGRIS_OP_ATTR_POOL_ROUNDING;
    fx->attribute.data_len = 1;
    fx->attribute.data_offset = 0;
    fx->attribute_data[0] = TIGRIS_POOL_ROUNDING_AVERAGE;
    fx->plan.op_attributes = &fx->attribute;
    fx->plan.op_attribute_data = fx->attribute_data;
    fx->plan.num_op_attributes = 1;
}

static int prepare(fixture_t *fx, tigris_mem_t *mem, void **ptrs)
{
    if (tigris_mem_init(mem, ptrs, 2, fast_arena, sizeof(fast_arena), fast_arena, 0) != TIGRIS_MEM_OK)
        return -1;
    ptrs[0] = input_buf;
    ptrs[1] = output_buf;
    reset_records();
    return tigris_cmsis_nn_prepare(&fx->plan, mem);
}

static int dispatch(fixture_t *fx, tigris_mem_t *mem)
{
    reset_records();
    return tigris_dispatch_kernel_cmsis_nn(&fx->plan, &fx->op, 0, mem, NULL);
}

static int dims_are(cmsis_nn_dims d, int32_t n, int32_t h, int32_t w, int32_t c)
{
    return d.n == n && d.h == h && d.w == w && d.c == c;
}

static int tile_is(cmsis_nn_tile t, int32_t w, int32_t h)
{
    return t.w == w && t.h == h;
}

/* The vendor scratch lies inside the workspace carved from the top of the fast arena. */
static int scratch_in_workspace(const tigris_mem_t *mem, cmsis_nn_context ctx, int32_t size)
{
    const uint8_t *buf = (const uint8_t *)ctx.buf;
    return ctx.size == size && buf && ((uintptr_t)buf % 16u) == 0u &&
           buf >= mem->fast_base + mem->fast_size &&
           buf + size <= fast_arena + sizeof(fast_arena);
}

static int channels_are_plan_arrays(uint32_t channels)
{
    if (call.multiplier != &quant_data[QUANT_BASE + OUT_QUANT_OFF] ||
        call.shift != &quant_data[QUANT_BASE + OUT_QUANT_OFF + channels])
        return 0;
    for (uint32_t c = 0; c < channels; c++)
        if (call.multipliers[c] != out_multiplier(c) || call.shifts[c] != out_shift(c))
            return 0;
    return 1;
}

static int channels_are(uint32_t channels, int32_t multiplier, int32_t shift)
{
    for (uint32_t c = 0; c < channels; c++)
        if (call.multipliers[c] != multiplier || call.shifts[c] != shift)
            return 0;
    return 1;
}

/* Tests */

static void test_queries_and_prepare(void)
{
    fixture_t fx;
    build_fixture(&fx);

    TEST_ASSERT(tigris_cmsis_nn_scratch_required(NULL) == UINT32_MAX,
                "null plan fails closed");
    TEST_ASSERT(tigris_cmsis_nn_scratch_required(&fx.plan) == 80,
                "workspace includes aligned vendor and quant-expansion scratch");
    TEST_ASSERT(tigris_cmsis_nn_fast_arena_required(&fx.plan) == 192,
                "total preserves aligned core capacity plus all workspace");

    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[192];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    void *ptrs[2];
    tigris_mem_t mem;
    TEST_ASSERT(tigris_mem_init(
                    &mem, ptrs, 2, fast, sizeof(fast), slow, sizeof(slow)) ==
                    TIGRIS_MEM_OK,
                "memory manager initializes");
    TEST_ASSERT(tigris_cmsis_nn_prepare(&fx.plan, &mem) == 0,
                "exact total capacity prepares");
    TEST_ASSERT(mem.fast_size == 112,
                "scratch carve leaves at least the full core requirement");
    TEST_ASSERT(tigris_cmsis_nn_prepare(&fx.plan, &mem) == 0,
                "repeated prepare is idempotent");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0 && mem.fast_size == sizeof(fast),
                "deinit restores original capacity");

    TEST_ASSERT(tigris_mem_init(
                    &mem, ptrs, 2, fast, 191, slow, sizeof(slow)) == TIGRIS_MEM_OK,
                "undersized arena still initializes");
    TEST_ASSERT(tigris_cmsis_nn_prepare(&fx.plan, &mem) == -1,
                "one byte below total requirement fails before carving");

    query_bytes = -1;
    TEST_ASSERT(tigris_cmsis_nn_scratch_required(&fx.plan) == UINT32_MAX,
                "negative vendor result fails closed");
    query_bytes = 33;

    fx.header.budget = UINT32_MAX;
    TEST_ASSERT(tigris_cmsis_nn_fast_arena_required(&fx.plan) == UINT32_MAX,
                "unrepresentable total fails closed");
}

static void test_conv_arguments(void)
{
    fixture_t fx;
    void *ptrs[2];
    tigris_mem_t mem;

    build_conv(&fx, 2, 8);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0, "conv prepares");
    TEST_ASSERT(query.fn == CALL_CONV &&
                dims_are(query.input, 1, 7, 6, 3) && dims_are(query.filter, 8, 3, 2, 3) &&
                dims_are(query.output, 1, 4, 6, 8) && tile_is(query.conv.stride, 1, 2) &&
                tile_is(query.conv.padding, 1, 2) && tile_is(query.conv.dilation, 1, 2),
                "conv sizing uses the tensor shapes and the operator geometry");
    TEST_ASSERT(dispatch(&fx, &mem) == 0, "conv adapter runs");
    TEST_ASSERT(call.fn == CALL_CONV && call.calls == 1, "conv makes one vendor call");
    TEST_ASSERT(call.conv.input_offset == 5 && call.conv.output_offset == 7,
                "conv offsets are the negated input and the output zero points");
    TEST_ASSERT(tile_is(call.conv.stride, 1, 2) && tile_is(call.conv.padding, 1, 2) &&
                tile_is(call.conv.dilation, 1, 2),
                "conv stride, leading pads and dilation (0 means 1) per axis");
    TEST_ASSERT(call.conv.activation.min == -20 && call.conv.activation.max == 100,
                "conv activation range is the operator's");
    TEST_ASSERT(dims_are(call.input, 1, 7, 6, 3) && dims_are(call.filter, 8, 3, 2, 3) &&
                dims_are(call.bias, 1, 1, 1, 8) && dims_are(call.output, 1, 4, 6, 8),
                "conv dims: filter [OC, KH, KW, IC], bias [OC]");
    TEST_ASSERT(call.input_data == input_buf && call.output_data == output_buf &&
                call.filter_data == (const int8_t *)weight_blob &&
                call.bias_data == (const int32_t *)(const void *)(weight_blob + BIAS_OFF),
                "conv data pointers are the tensors and the operator's weights");
    TEST_ASSERT(channels_are_plan_arrays(8), "per-channel conv quantization points into the plan");
    TEST_ASSERT(scratch_in_workspace(&mem, call.ctx, 33), "conv scratch is the prepared workspace");
    TEST_ASSERT(query.calls == 1 && dims_are(query.input, 1, 7, 6, 3) &&
                dims_are(query.filter, 8, 3, 2, 3) && dims_are(query.output, 1, 4, 6, 8) &&
                tile_is(query.conv.padding, 1, 2) && tile_is(query.conv.dilation, 1, 2),
                "conv dispatch sizes the call it makes");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "conv releases workspace");

    build_conv(&fx, 0, 8);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && tile_is(query.conv.dilation, 1, 1),
                "conv sizing maps dilation 0 to 1 on both axes");
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_CONV && dims_are(call.output, 1, 5, 6, 8) &&
                tile_is(call.conv.dilation, 1, 1) && tile_is(query.conv.dilation, 1, 1),
                "conv dilation 0 means 1 on both axes");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "undilated conv releases workspace");

    build_conv(&fx, 2, 1);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "per-tensor conv runs");
    TEST_ASSERT(call.multiplier != &quant_data[QUANT_BASE + OUT_QUANT_OFF] &&
                channels_are(8, out_multiplier(0), out_shift(0)),
                "per-tensor conv quantization is repeated for every output channel");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "per-tensor conv releases workspace");

    build_conv(&fx, 2, 0);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "unquantized conv runs");
    TEST_ASSERT(call.conv.input_offset == 0 && call.conv.output_offset == 0 &&
                channels_are(8, 1, 0),
                "unquantized conv uses zero offsets and identity requantization");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "unquantized conv releases workspace");
}

static void test_conv_tiles(void)
{
    fixture_t fx;
    void *ptrs[2];
    tigris_mem_t mem;

    /* Height tile: output rows 2..3 read input rows 2..6. */
    build_conv(&fx, 1, 8);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0, "tiled conv prepares");
    TEST_ASSERT(dims_are(query.input, 1, 7, 6, 3) && dims_are(query.output, 1, 5, 6, 8) &&
                tile_is(query.conv.padding, 1, 2),
                "tiled conv sizing uses the whole tensors");
    mem.tile.active = 1;
    mem.tile.in_h = 5;
    mem.tile.out_h = 2;
    mem.tile.pad_top = 0;
    mem.tile.pad_bottom = 0;
    mem.tile.in_w = 6;
    mem.tile.out_w = 6;
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_CONV, "height tile runs on the adapter");
    TEST_ASSERT(dims_are(call.input, 1, 5, 6, 3) && dims_are(call.output, 1, 2, 6, 8) &&
                tile_is(call.conv.padding, 1, 0) && tile_is(call.conv.dilation, 1, 1),
                "height tile passes its rows and its top pad, keeps the width");
    TEST_ASSERT(call.input_data == input_buf && call.output_data == output_buf,
                "height tile starts at the tile buffers");
    TEST_ASSERT(dims_are(query.input, 1, 5, 6, 3) && dims_are(query.output, 1, 2, 6, 8) &&
                tile_is(query.conv.padding, 1, 0),
                "height tile dispatch sizes the tile");

    /* 2D tile: columns 3..5 of the same rows read input columns 2..5. */
    mem.tile.width_tiled = 1;
    mem.tile.in_w = 4;
    mem.tile.out_w = 3;
    mem.tile.pad_left = 0;
    mem.tile.pad_right = 0;
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_CONV, "2D tile runs on the adapter");
    TEST_ASSERT(dims_are(call.input, 1, 5, 4, 3) && dims_are(call.output, 1, 2, 3, 8) &&
                tile_is(call.conv.padding, 0, 0),
                "2D tile passes its columns and its left pad");

    /* Line-buffer roll: 1 output row and 1 input row are already in place. */
    mem.tile.width_tiled = 0;
    mem.tile.in_w = 6;
    mem.tile.out_w = 6;
    mem.tile.pad_left = 0;
    mem.tile.out_row_start = 1;
    mem.tile.in_row_start = 1;
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_CONV, "rolled tile runs on the adapter");
    TEST_ASSERT(call.input_data == input_buf + 1 * 6 * 3 && call.output_data == output_buf + 1 * 6 * 8,
                "rolled tile skips (out_row_start * stride - in_row_start) input rows and out_row_start output rows");
    TEST_ASSERT(dims_are(call.input, 1, 4, 6, 3) && dims_are(call.output, 1, 2, 6, 8),
                "rolled tile drops the skipped input rows");

    mem.tile.width_tiled = 1;
    mem.tile.in_w = 4;
    mem.tile.out_w = 3;
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_CONV, "rolled 2D tile runs on the adapter");
    TEST_ASSERT(call.input_data == input_buf + 1 * 4 * 3 && call.output_data == output_buf + 1 * 3 * 8,
                "rolled 2D tile steps rows of the tile width");
    TEST_ASSERT(dims_are(call.input, 1, 4, 4, 3) && dims_are(call.output, 1, 2, 3, 8),
                "rolled 2D tile dims");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "tiled conv releases workspace");
}

static void test_depthwise_arguments(void)
{
    fixture_t fx;
    void *ptrs[2];
    tigris_mem_t mem;

    build_depthwise(&fx, 3, 1);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0, "depthwise prepares");
    TEST_ASSERT(query.fn == CALL_DEPTHWISE && query.dw.ch_mult == 3 &&
                dims_are(query.input, 1, 7, 6, 4) && dims_are(query.filter, 1, 3, 1, 12) &&
                dims_are(query.output, 1, 4, 6, 12) && tile_is(query.dw.stride, 1, 2) &&
                tile_is(query.dw.padding, 0, 1) && tile_is(query.dw.dilation, 3, 1),
                "depthwise sizing uses the channel multiplier and the operator geometry");
    TEST_ASSERT(dispatch(&fx, &mem) == 0, "depthwise adapter runs");
    TEST_ASSERT(call.fn == CALL_DEPTHWISE && call.calls == 1, "depthwise makes one vendor call");
    TEST_ASSERT(call.dw.input_offset == -11 && call.dw.output_offset == -9 && call.dw.ch_mult == 3,
                "depthwise offsets and channel multiplier");
    TEST_ASSERT(tile_is(call.dw.stride, 1, 2) && tile_is(call.dw.padding, 0, 1) &&
                tile_is(call.dw.dilation, 3, 1),
                "depthwise stride, leading pads and dilation (0 means 1) per axis");
    TEST_ASSERT(call.dw.activation.min == 3 && call.dw.activation.max == 60,
                "depthwise activation range is the operator's");
    TEST_ASSERT(dims_are(call.input, 1, 7, 6, 4) && dims_are(call.filter, 1, 3, 1, 12) &&
                dims_are(call.bias, 1, 1, 1, 12) && dims_are(call.output, 1, 4, 6, 12),
                "vendor receives input and output channels independently");
    TEST_ASSERT(call.input_data == input_buf && call.output_data == output_buf &&
                call.filter_data == (const int8_t *)weight_blob &&
                call.bias_data == (const int32_t *)(const void *)(weight_blob + BIAS_OFF),
                "depthwise data pointers are the tensors and the operator's weights");
    TEST_ASSERT(channels_are(12, out_multiplier(0), out_shift(0)),
                "scalar quantization covers every output channel");
    TEST_ASSERT(scratch_in_workspace(&mem, call.ctx, 33), "depthwise scratch is the prepared workspace");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "depthwise releases workspace");

    build_depthwise(&fx, 0, 1);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && tile_is(query.dw.dilation, 1, 1),
                "depthwise sizing maps dilation 0 to 1 on both axes");
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_DEPTHWISE &&
                tile_is(call.dw.dilation, 1, 1) && tile_is(query.dw.dilation, 1, 1),
                "depthwise dilation 0 means 1 on both axes");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "undilated depthwise releases workspace");

    build_depthwise(&fx, 1, 12);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "per-channel depthwise runs");
    TEST_ASSERT(channels_are_plan_arrays(12), "per-channel depthwise quantization points into the plan");

    /* Rolled 2D tile at the left edge: 2 input rows skipped (1 output row at
     * stride 2), 4 input columns and a left pad of 1 give 5 output columns. */
    mem.tile.active = 1;
    mem.tile.in_h = 5;
    mem.tile.out_h = 2;
    mem.tile.pad_top = 0;
    mem.tile.width_tiled = 1;
    mem.tile.in_w = 4;
    mem.tile.out_w = 5;
    mem.tile.pad_left = 1;
    mem.tile.out_row_start = 1;
    mem.tile.in_row_start = 0;
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.fn == CALL_DEPTHWISE, "rolled depthwise tile runs on the adapter");
    TEST_ASSERT(call.input_data == input_buf + 2 * 4 * 4 && call.output_data == output_buf + 1 * 5 * 12,
                "rolled depthwise tile steps input rows of in_w and output rows of out_w");
    TEST_ASSERT(dims_are(call.input, 1, 3, 4, 4) && dims_are(call.output, 1, 2, 5, 12) &&
                tile_is(call.dw.padding, 1, 0),
                "rolled depthwise tile dims and pads");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "per-channel depthwise releases workspace");
}

static void test_fully_connected_arguments(void)
{
    fixture_t fx;
    void *ptrs[2];
    tigris_mem_t mem;

    build_fc(&fx, 0, 1);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0, "fully connected prepares");
    TEST_ASSERT(query.fn == CALL_FC && dims_are(query.filter, 12, 1, 1, 5),
                "fully connected sizing uses [IC, OC]");
    TEST_ASSERT(dispatch(&fx, &mem) == 0, "fully connected adapter runs");
    TEST_ASSERT(call.fn == CALL_FC && call.calls == 1, "per-tensor fully connected uses the per-tensor kernel");
    TEST_ASSERT(call.fc.input_offset == 5 && call.fc.filter_offset == 0 && call.fc.output_offset == 7 &&
                call.fc.activation.min == -128 && call.fc.activation.max == 90,
                "fully connected offsets and activation range");
    TEST_ASSERT(call.tensor_quant.multiplier == out_multiplier(0) && call.tensor_quant.shift == out_shift(0),
                "per-tensor fully connected quantization is the output's");
    TEST_ASSERT(dims_are(call.input, 2, 1, 1, 12) && dims_are(call.filter, 12, 1, 1, 5) &&
                dims_are(call.bias, 1, 1, 1, 5) && dims_are(call.output, 2, 1, 1, 5),
                "fully connected dims: rows from the output, depth from the input size");
    TEST_ASSERT(call.input_data == input_buf && call.output_data == output_buf &&
                call.filter_data == (const int8_t *)weight_blob &&
                call.bias_data == (const int32_t *)(const void *)(weight_blob + BIAS_OFF),
                "fully connected data pointers");
    TEST_ASSERT(scratch_in_workspace(&mem, call.ctx, 33), "fully connected scratch is the prepared workspace");
    TEST_ASSERT(vsum.calls == 1 && vsum.buf == (int32_t *)call.ctx.buf && vsum.cols == 12 &&
                vsum.rows == 5 && vsum.data == (const int8_t *)weight_blob &&
                vsum.lhs_offset == 5 && vsum.rhs_offset == 0 && vsum.bias == NULL,
                "kernel sums are computed into the scratch with the input offset");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "fully connected releases workspace");

    build_fc(&fx, 0, 5);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "per-channel fully connected runs");
    TEST_ASSERT(call.fn == CALL_FC_PER_CHANNEL && call.calls == 1,
                "per-channel fully connected uses the per-channel kernel");
    TEST_ASSERT(call.fc.input_offset == 5 && call.fc.output_offset == 7 &&
                dims_are(call.filter, 12, 1, 1, 5) && dims_are(call.output, 2, 1, 1, 5),
                "per-channel fully connected parameters");
    TEST_ASSERT(channels_are_plan_arrays(5), "per-channel fully connected quantization points into the plan");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "per-channel fully connected releases workspace");

    query_bytes = 0;
    build_fc(&fx, 1, 1);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "flat fully connected runs");
    TEST_ASSERT(dims_are(call.input, 1, 1, 1, 12) && dims_are(call.filter, 12, 1, 1, 5) &&
                dims_are(call.output, 1, 1, 1, 5),
                "a rank-1 output is one row");
    TEST_ASSERT(call.ctx.buf != NULL && call.ctx.size == 0 && vsum.calls == 0,
                "no kernel sums without vendor scratch");
    query_bytes = 33;
}

static void test_pool_arguments(void)
{
    fixture_t fx;
    void *ptrs[2];
    tigris_mem_t mem;

    build_pool(&fx, 0);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0, "average pool prepares");
    TEST_ASSERT(query.fn == CALL_AVGPOOL && query.dst_width == 5 && query.src_channels == 4,
                "average pool sizing uses output width and channels");
    TEST_ASSERT(dispatch(&fx, &mem) == 0, "average pool adapter runs");
    TEST_ASSERT(call.fn == CALL_AVGPOOL && call.calls == 1, "average pool makes one vendor call");
    TEST_ASSERT(tile_is(call.pool.stride, 1, 2) && tile_is(call.pool.padding, 1, 0) &&
                call.pool.activation.min == -128 && call.pool.activation.max == 127,
                "average pool stride, leading pads and activation range");
    TEST_ASSERT(dims_are(call.input, 1, 6, 5, 4) && dims_are(call.filter, 1, 2, 3, 1) &&
                dims_are(call.output, 1, 3, 5, 4),
                "average pool dims: filter [1, KH, KW, 1]");
    TEST_ASSERT(call.input_data == input_buf && call.output_data == output_buf,
                "average pool data pointers");
    TEST_ASSERT(scratch_in_workspace(&mem, call.ctx, 33), "average pool scratch is the prepared workspace");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "average pool releases workspace");

    build_pool(&fx, 1);
    add_pool_rounding(&fx);
    TEST_ASSERT(prepare(&fx, &mem, ptrs) == 0 && dispatch(&fx, &mem) == 0, "rounded global pool runs");
    TEST_ASSERT(call.fn == CALL_AVGPOOL && tile_is(call.pool.stride, 1, 1) && tile_is(call.pool.padding, 0, 0) &&
                dims_are(call.filter, 1, 6, 5, 1) && dims_are(call.output, 1, 1, 1, 4),
                "global pool is one window over the whole input");
    TEST_ASSERT(query.dst_width == 1 && query.src_channels == 4, "global pool dispatch sizes one column");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == 0, "rounded global pool releases workspace");

    build_pool(&fx, 1);
    memset(input_buf, 0, sizeof(input_buf));
    TEST_ASSERT(tigris_cmsis_nn_scratch_required(&fx.plan) == 0 && prepare(&fx, &mem, ptrs) == 0 &&
                query.calls == 0 && mem.fast_size == sizeof(fast_arena),
                "a global mean without pool rounding reserves no vendor workspace");
    TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.calls == 0,
                "a global mean without pool rounding runs on the reference kernel");
    TEST_ASSERT(tigris_cmsis_nn_deinit(&mem) == -1, "a global mean leaves no reservation to release");

    /* arm_avgpool_s8 keeps the input quantization, so a rescaling pool runs on
     * the reference kernel and prepare reserves no workspace for it. */
    for (int rescale = 0; rescale < 2; rescale++) {
        build_pool(&fx, 0);
        if (rescale)
            fx.quant[1].scale = 0.375f;
        else
            fx.quant[1].zero_point = -2;
        TEST_ASSERT(tigris_cmsis_nn_scratch_required(&fx.plan) == 0 && prepare(&fx, &mem, ptrs) == 0 &&
                    query.calls == 0 && mem.fast_size == sizeof(fast_arena),
                    "a rescaling average pool reserves no vendor workspace");
        TEST_ASSERT(dispatch(&fx, &mem) == 0 && call.calls == 0 && query.calls == 0,
                    "a rescaling average pool makes no vendor call");
    }
}

static void test_binary_dispatch(void)
{
    tigris_file_header_t header = { .num_tensors = 3, .num_ops = 1 };
    tigris_tensor_t tensors[3] = {0};
    tigris_quant_param_t quant[3] = {0};
    uint16_t indices[3] = {0, 1, 2};
    int32_t shapes[12] = {1, 4, 4, 4, 1, 4, 4, 4, 1, 4, 4, 4};
    tigris_op_t op = { .num_inputs = 2, .num_outputs = 1, .outputs_off = 2,
                       .weight_idx = TIGRIS_NO_WEIGHT, .bias_idx = TIGRIS_NO_WEIGHT };
    tigris_plan_t plan = { .header = &header, .tensors = tensors, .ops = &op, .index_pool = indices,
                          .shape_pool = shapes, .quant_params = quant, .num_quant_params = 3 };
    for (uint16_t i = 0; i < 3; i++) {
        tensors[i].dtype = 3; tensors[i].ndim = 4; tensors[i].shape_off = i * 4u;
        tensors[i].size_bytes = 64; tensors[i].quant_param_idx = i;
        quant[i].scale = 0.125f; quant[i].zero_point = -17; quant[i].num_channels = 1;
    }
    int8_t a[64], b[64], reference[64], output[64];
    for (int i = 0; i < 64; i++) { a[i] = (int8_t)(i * 4 - 128); b[i] = (int8_t)(127 - i * 3); }
    void *ptrs[3] = {a, b, reference};
    tigris_mem_t mem = { .tensor_ptrs = ptrs, .num_tensors = 3 };
    const uint8_t kinds[] = {TIGRIS_OP_MUL, TIGRIS_OP_MAXIMUM, TIGRIS_OP_MINIMUM};
    const vendor_fn_t fns[] = {CALL_MUL, CALL_MAXIMUM, CALL_MINIMUM};
    for (unsigned k = 0; k < sizeof(kinds); k++) {
        op.op_type = kinds[k];
        for (int tiled = 0; tiled < 2; tiled++) {
            mem.tile.active = (uint8_t)tiled;
            mem.tile.out_h = 1; mem.tile.out_w = 4;
            memset(output, 99, sizeof(output));
            ptrs[2] = reference;
            TEST_ASSERT(tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL) == 0, "binary reference runs");
            ptrs[2] = output;
            reset_records();
            TEST_ASSERT(tigris_dispatch_kernel_cmsis_nn(&plan, &op, 0, &mem, NULL) == 0, "binary adapter runs");
            int count = tiled ? 16 : 64;
            TEST_ASSERT(call.fn == fns[k] && call.calls == 1, "binary makes one vendor call");
            TEST_ASSERT(call.input_data == a && call.input2_data == b && call.output_data == output,
                        "binary data pointers are the operands");
            if (kinds[k] == TIGRIS_OP_MUL) {
                TEST_ASSERT(call.block_size == count, "vendor receives only the output band");
                TEST_ASSERT(call.input_1_offset == 17 && call.input_2_offset == 17 && call.out_offset == -17 &&
                            call.out_mult == 1073741824 && call.out_shift == -2 &&
                            call.out_activation_min == -128 && call.out_activation_max == 127,
                            "multiply offsets, requantization (scale 0.125) and full int8 range");
            } else {
                TEST_ASSERT(dims_are(call.input, 1, 1, 1, count) && dims_are(call.input2, 1, 1, 1, count) &&
                            dims_are(call.output, 1, 1, 1, count),
                            "vendor receives only the output band");
                TEST_ASSERT(call.ctx.buf == NULL && call.ctx.size == 0, "extremum needs no scratch");
            }
            TEST_ASSERT(memcmp(reference, output, (size_t)count) == 0, "binary output matches reference");
            if (tiled) TEST_ASSERT(output[count] == 99, "binary band does not overwrite its neighbor");
        }
    }
}

int main(void)
{
    test_queries_and_prepare();
    test_conv_arguments();
    test_conv_tiles();
    test_depthwise_arguments();
    test_fully_connected_arguments();
    test_pool_arguments();
    test_binary_dispatch();
    return test_report();
}

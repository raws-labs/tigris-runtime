/** Host behavioral tests for the CMSIS-NN arena sizing contract. */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "tigris_kernels_cmsis_nn.h"
#include "tigris_kernels_s8.h"

static int tests_run;
static int tests_failed;
static int32_t conv_scratch = 33;
static int binary_calls;
static int32_t binary_count;
static int32_t dw_multiplier, dw_input_channels, dw_output_channels;

#define CHECK(expr, msg) do { \
    tests_run++; \
    if (!(expr)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, msg); \
        tests_failed++; \
    } \
} while (0)

int32_t arm_convolve_wrapper_s8_get_buffer_size(
    const cmsis_nn_conv_params *params, ...)
{
    (void)params;
    return conv_scratch;
}

int32_t arm_depthwise_conv_wrapper_s8_get_buffer_size(
    const cmsis_nn_dw_conv_params *params, ...)
{
    va_list args;
    va_start(args, params);
    const cmsis_nn_dims *input = va_arg(args, const cmsis_nn_dims *);
    (void)va_arg(args, const cmsis_nn_dims *);
    const cmsis_nn_dims *output = va_arg(args, const cmsis_nn_dims *);
    dw_multiplier = params->ch_mult;
    dw_input_channels = input->c;
    dw_output_channels = output->c;
    va_end(args);
    return 0;
}

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *dims)
{
    (void)dims;
    return 0;
}

int32_t arm_avgpool_s8_get_buffer_size(int32_t width, int32_t channels)
{
    (void)width;
    (void)channels;
    return 0;
}

#define SUCCESS_STUB(name) \
    arm_cmsis_nn_status name(const cmsis_nn_context *ctx, ...) \
    { \
        (void)ctx; \
        return ARM_CMSIS_NN_SUCCESS; \
    }

SUCCESS_STUB(arm_convolve_wrapper_s8)
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx, ...)
{
    va_list args;
    va_start(args, ctx);
    const cmsis_nn_dw_conv_params *params = va_arg(args, const cmsis_nn_dw_conv_params *);
    const cmsis_nn_per_channel_quant_params *quant = va_arg(args, const cmsis_nn_per_channel_quant_params *);
    const cmsis_nn_dims *input = va_arg(args, const cmsis_nn_dims *);
    (void)va_arg(args, const int8_t *);
    const cmsis_nn_dims *filter = va_arg(args, const cmsis_nn_dims *);
    (void)va_arg(args, const int8_t *);
    const cmsis_nn_dims *bias = va_arg(args, const cmsis_nn_dims *);
    (void)va_arg(args, const int32_t *);
    const cmsis_nn_dims *output = va_arg(args, const cmsis_nn_dims *);
    dw_multiplier = params->ch_mult;
    dw_input_channels = input->c;
    dw_output_channels = output->c;
    CHECK(filter->c == output->c && bias->c == output->c, "depthwise metadata uses output channels");
    CHECK(quant->multiplier[output->c - 1] == quant->multiplier[0], "scalar quantization covers every output channel");
    va_end(args);
    return ARM_CMSIS_NN_SUCCESS;
}
SUCCESS_STUB(arm_fully_connected_per_channel_s8)
SUCCESS_STUB(arm_fully_connected_s8)
SUCCESS_STUB(arm_avgpool_s8)

arm_cmsis_nn_status arm_elementwise_mul_s8(const int8_t *a, const int8_t *b, int32_t ao, int32_t bo,
    int8_t *y, int32_t yo, int32_t mult, int32_t shift, int32_t lo, int32_t hi, int32_t count)
{
    binary_calls++;
    binary_count = count;
    for (int32_t i = 0; i < count; i++) {
        int32_t product = ((int32_t)a[i] + ao) * ((int32_t)b[i] + bo);
        product *= 1 << (shift > 0 ? shift : 0);
        int32_t value = (int32_t)(((int64_t)product * mult + ((int64_t)1 << 30)) >> 31);
        int right = shift < 0 ? -shift : 0;
        int32_t mask = (1 << right) - 1;
        value = (value >> right) + ((value & mask) > ((mask >> 1) + (value < 0)));
        value += yo;
        y[i] = (int8_t)(value < lo ? lo : value > hi ? hi : value);
    }
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_maximum_s8(const cmsis_nn_context *ctx, const int8_t *a, const cmsis_nn_dims *ad,
    const int8_t *b, const cmsis_nn_dims *bd, int8_t *y, const cmsis_nn_dims *yd)
{
    (void)ctx; (void)ad; (void)bd;
    binary_calls++;
    binary_count = yd->c;
    for (int32_t i = 0; i < yd->c; i++) y[i] = a[i] > b[i] ? a[i] : b[i];
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_minimum_s8(const cmsis_nn_context *ctx, const int8_t *a, const cmsis_nn_dims *ad,
    const int8_t *b, const cmsis_nn_dims *bd, int8_t *y, const cmsis_nn_dims *yd)
{
    (void)ctx; (void)ad; (void)bd;
    binary_calls++;
    binary_count = yd->c;
    for (int32_t i = 0; i < yd->c; i++) y[i] = a[i] < b[i] ? a[i] : b[i];
    return ARM_CMSIS_NN_SUCCESS;
}

arm_cmsis_nn_status arm_vector_sum_s8(int32_t *vector_sum_buf, const int32_t vector_cols,
                                      const int32_t vector_rows, const int8_t *vector_data,
                                      const int32_t lhs_offset, const int32_t rhs_offset,
                                      const int32_t *bias_data)
{
    (void)vector_sum_buf; (void)vector_cols; (void)vector_rows; (void)vector_data;
    (void)lhs_offset; (void)rhs_offset; (void)bias_data;
    return ARM_CMSIS_NN_SUCCESS;
}

typedef struct {
    tigris_file_header_t header;
    tigris_tensor_t tensors[2];
    tigris_op_t op;
    uint16_t indices[2];
    int32_t shapes[8];
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

static void test_queries_and_prepare(void)
{
    fixture_t fx;
    build_fixture(&fx);

    CHECK(tigris_cmsis_nn_scratch_required(NULL) == UINT32_MAX,
          "null plan fails closed");
    CHECK(tigris_cmsis_nn_scratch_required(&fx.plan) == 80,
          "workspace includes aligned vendor and quant-expansion scratch");
    CHECK(tigris_cmsis_nn_fast_arena_required(&fx.plan) == 192,
          "total preserves aligned core capacity plus all workspace");

    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[192];
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[64];
    void *ptrs[2];
    tigris_mem_t mem;
    CHECK(tigris_mem_init(
              &mem, ptrs, 2, fast, sizeof(fast), slow, sizeof(slow)) ==
              TIGRIS_MEM_OK,
          "memory manager initializes");
    CHECK(tigris_cmsis_nn_prepare(&fx.plan, &mem) == 0,
          "exact total capacity prepares");
    CHECK(mem.fast_size == 112,
          "scratch carve leaves at least the full core requirement");
    CHECK(tigris_cmsis_nn_prepare(&fx.plan, &mem) == 0,
          "repeated prepare is idempotent");
    CHECK(tigris_cmsis_nn_deinit(&mem) == 0 && mem.fast_size == sizeof(fast),
          "deinit restores original capacity");

    CHECK(tigris_mem_init(
              &mem, ptrs, 2, fast, 191, slow, sizeof(slow)) == TIGRIS_MEM_OK,
          "undersized arena still initializes");
    CHECK(tigris_cmsis_nn_prepare(&fx.plan, &mem) == -1,
          "one byte below total requirement fails before carving");

    conv_scratch = -1;
    CHECK(tigris_cmsis_nn_scratch_required(&fx.plan) == UINT32_MAX,
          "negative vendor result fails closed");
    conv_scratch = 33;

    fx.header.budget = UINT32_MAX;
    CHECK(tigris_cmsis_nn_fast_arena_required(&fx.plan) == UINT32_MAX,
          "unrepresentable total fails closed");
}

static void test_depthwise_output_channels(void)
{
    fixture_t fx;
    build_fixture(&fx);
    fx.op.op_type = TIGRIS_OP_DEPTHWISE;
    fx.op.weight_idx = TIGRIS_NO_WEIGHT;
    fx.op.bias_idx = TIGRIS_NO_WEIGHT;
    fx.shapes[7] = 12;
    fx.tensors[1].size_bytes = 192;
    _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[512];
    void *ptrs[2];
    tigris_mem_t mem;
    CHECK(tigris_mem_init(&mem, ptrs, 2, fast, sizeof(fast), fast, 0) == TIGRIS_MEM_OK, "depthwise memory");
    CHECK(tigris_cmsis_nn_prepare(&fx.plan, &mem) == 0, "depthwise prepares");
    CHECK(dw_multiplier == 3 && dw_input_channels == 4 && dw_output_channels == 12,
          "scratch sizing uses the channel multiplier");
    ptrs[0] = fast;
    ptrs[1] = fast + 64;
    CHECK(tigris_dispatch_kernel_cmsis_nn(&fx.plan, &fx.op, 0, &mem, NULL) == 0, "depthwise adapter runs");
    CHECK(dw_multiplier == 3 && dw_input_channels == 4 && dw_output_channels == 12,
          "vendor receives input and output channels independently");
    CHECK(tigris_cmsis_nn_deinit(&mem) == 0, "depthwise releases workspace");
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
    for (unsigned k = 0; k < sizeof(kinds); k++) {
        op.op_type = kinds[k];
        for (int tiled = 0; tiled < 2; tiled++) {
            mem.tile.active = (uint8_t)tiled;
            mem.tile.out_h = 1; mem.tile.out_w = 4;
            memset(output, 99, sizeof(output));
            ptrs[2] = reference;
            CHECK(tigris_dispatch_kernel_s8(&plan, &op, 0, &mem, NULL) == 0, "binary reference runs");
            ptrs[2] = output;
            binary_calls = 0;
            CHECK(tigris_dispatch_kernel_cmsis_nn(&plan, &op, 0, &mem, NULL) == 0, "binary adapter runs");
            int count = tiled ? 16 : 64;
            CHECK(binary_calls == 1 && binary_count == count, "vendor receives only the output band");
            CHECK(memcmp(reference, output, (size_t)count) == 0, "binary output matches reference");
            if (tiled) CHECK(output[count] == 99, "binary band does not overwrite its neighbor");
        }
    }
}

int main(void)
{
    test_queries_and_prepare();
    test_depthwise_output_channels();
    test_binary_dispatch();
    printf("CMSIS arena tests: %d checks, %d failed\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}

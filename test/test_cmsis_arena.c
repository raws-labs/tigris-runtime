/** Host behavioral tests for the CMSIS-NN arena sizing contract. */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include "tigris_kernels_cmsis_nn.h"

static int tests_run;
static int tests_failed;
static int32_t conv_scratch = 33;

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
    (void)params;
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
SUCCESS_STUB(arm_depthwise_conv_wrapper_s8)
SUCCESS_STUB(arm_fully_connected_per_channel_s8)
SUCCESS_STUB(arm_fully_connected_s8)
SUCCESS_STUB(arm_avgpool_s8)

void arm_vector_sum_s8(int32_t *output, ...)
{
    (void)output;
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

int main(void)
{
    test_queries_and_prepare();
    printf("CMSIS arena tests: %d checks, %d failed\n", tests_run, tests_failed);
    return tests_failed ? 1 : 0;
}

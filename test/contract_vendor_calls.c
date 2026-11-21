/*
 * Call counters for the vendor functions the accelerated adapters call, and
 * for the adapters' calls into the s8 reference dispatcher.
 *
 * The accelerated contract runners link with -Wl,--wrap=<symbol> for every
 * symbol in VENDOR_CALLS, for tigris_dispatch_kernel_s8 and for
 * tigris_accel_try_s8_ref, so each call from another object file reaches
 * __wrap_<symbol> here, which counts it and calls __real_<symbol>. The build checks that the list covers every arm_* and
 * esp_nn_* symbol the runtime objects reference. A vendor function calling a
 * wrapped symbol of its own library is counted too, for example
 * arm_avgpool_s8 calling arm_avgpool_s8_get_buffer_size. ESP-NN's public
 * names are macros; ESP_NN_VARIANT names the _opt or _ansi symbol they
 * resolve to under the CONFIG_NN_OPTIMIZED setting of the build.
 *
 * The routing policy calls tigris_dispatch_kernel_s8 inside its own object
 * file, so only an adapter's call is wrapped: the adapter took the operator
 * and then handed it to s8. tigris_accel_try_s8_ref is wrapped to count the
 * operators the policy handed to s8 only because a tile was active.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_nnfunctions.h"
#include <esp_nn.h>

#include "tigris.h"
#include "tigris_accel_policy.h"
#include "tigris_kernels_s8.h"

/* Called by test/contract_runner.c. */
void tigris_contract_vendor_begin_run(const char *dispatch, const void *workspace,
                                      uint32_t workspace_size);
void tigris_contract_vendor_print(void);

#if defined(CONFIG_NN_OPTIMIZED)
#define ESP_NN_VARIANT(name) name##_opt
#else
#define ESP_NN_VARIANT(name) name##_ansi
#endif

static void note_ctx(const cmsis_nn_context *ctx);
static void note_quant(const int32_t *multiplier, int32_t channels);

/* X(return type, symbol, parameters, arguments, return keyword or empty,
 *   statement run on the arguments before the call) */
#define VENDOR_CALLS(X)                                                        \
    X(arm_cmsis_nn_status, arm_avgpool_s8,                                     \
      (const cmsis_nn_context *ctx, const cmsis_nn_pool_params *pool_params,   \
       const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
       const cmsis_nn_dims *filter_dims, const cmsis_nn_dims *output_dims,    \
       int8_t *output_data),                                                   \
      (ctx, pool_params, input_dims, input_data, filter_dims, output_dims,     \
       output_data), return, note_ctx(ctx))                                    \
    X(int32_t, arm_avgpool_s8_get_buffer_size,                                 \
      (const int dim_dst_width, const int ch_src), (dim_dst_width, ch_src),    \
      return, (void)0)                                                         \
    X(arm_cmsis_nn_status, arm_convolve_wrapper_s8,                            \
      (const cmsis_nn_context *ctx, const cmsis_nn_conv_params *conv_params,   \
       const cmsis_nn_per_channel_quant_params *quant_params,                 \
       const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
       const cmsis_nn_dims *filter_dims, const int8_t *filter_data,           \
       const cmsis_nn_dims *bias_dims, const int32_t *bias_data,              \
       const cmsis_nn_dims *output_dims, int8_t *output_data),                \
      (ctx, conv_params, quant_params, input_dims, input_data, filter_dims,    \
       filter_data, bias_dims, bias_data, output_dims, output_data), return,   \
      (note_ctx(ctx), note_quant(quant_params->multiplier, output_dims->c)))   \
    X(int32_t, arm_convolve_wrapper_s8_get_buffer_size,                        \
      (const cmsis_nn_conv_params *conv_params,                                \
       const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims,     \
       const cmsis_nn_dims *output_dims),                                      \
      (conv_params, input_dims, filter_dims, output_dims), return, (void)0)    \
    X(arm_cmsis_nn_status, arm_depthwise_conv_wrapper_s8,                      \
      (const cmsis_nn_context *ctx,                                            \
       const cmsis_nn_dw_conv_params *dw_conv_params,                          \
       const cmsis_nn_per_channel_quant_params *quant_params,                 \
       const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
       const cmsis_nn_dims *filter_dims, const int8_t *filter_data,           \
       const cmsis_nn_dims *bias_dims, const int32_t *bias_data,              \
       const cmsis_nn_dims *output_dims, int8_t *output_data),                \
      (ctx, dw_conv_params, quant_params, input_dims, input_data,              \
       filter_dims, filter_data, bias_dims, bias_data, output_dims,            \
       output_data), return,                                                   \
      (note_ctx(ctx), note_quant(quant_params->multiplier, output_dims->c)))   \
    X(int32_t, arm_depthwise_conv_wrapper_s8_get_buffer_size,                  \
      (const cmsis_nn_dw_conv_params *dw_conv_params,                          \
       const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims,     \
       const cmsis_nn_dims *output_dims),                                      \
      (dw_conv_params, input_dims, filter_dims, output_dims), return, (void)0) \
    X(arm_cmsis_nn_status, arm_elementwise_mul_s8,                             \
      (const int8_t *input_1_vect, const int8_t *input_2_vect,                 \
       const int32_t input_1_offset, const int32_t input_2_offset,            \
       int8_t *output, const int32_t out_offset, const int32_t out_mult,      \
       const int32_t out_shift, const int32_t out_activation_min,             \
       const int32_t out_activation_max, const int32_t block_size),           \
      (input_1_vect, input_2_vect, input_1_offset, input_2_offset, output,     \
       out_offset, out_mult, out_shift, out_activation_min,                    \
       out_activation_max, block_size), return, (void)0)                       \
    X(arm_cmsis_nn_status, arm_fully_connected_per_channel_s8,                 \
      (const cmsis_nn_context *ctx, const cmsis_nn_fc_params *fc_params,       \
       const cmsis_nn_per_channel_quant_params *quant_params,                 \
       const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
       const cmsis_nn_dims *filter_dims, const int8_t *filter_data,           \
       const cmsis_nn_dims *bias_dims, const int32_t *bias_data,              \
       const cmsis_nn_dims *output_dims, int8_t *output_data),                \
      (ctx, fc_params, quant_params, input_dims, input_data, filter_dims,      \
       filter_data, bias_dims, bias_data, output_dims, output_data), return,   \
      (note_ctx(ctx), note_quant(quant_params->multiplier, output_dims->c)))   \
    X(arm_cmsis_nn_status, arm_fully_connected_s8,                             \
      (const cmsis_nn_context *ctx, const cmsis_nn_fc_params *fc_params,       \
       const cmsis_nn_per_tensor_quant_params *quant_params,                  \
       const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
       const cmsis_nn_dims *filter_dims, const int8_t *filter_data,           \
       const cmsis_nn_dims *bias_dims, const int32_t *bias_data,              \
       const cmsis_nn_dims *output_dims, int8_t *output_data),                \
      (ctx, fc_params, quant_params, input_dims, input_data, filter_dims,      \
       filter_data, bias_dims, bias_data, output_dims, output_data), return,   \
      note_ctx(ctx))                                                           \
    X(int32_t, arm_fully_connected_s8_get_buffer_size,                         \
      (const cmsis_nn_dims *filter_dims), (filter_dims), return, (void)0)      \
    X(arm_cmsis_nn_status, arm_maximum_s8,                                     \
      (const cmsis_nn_context *ctx, const int8_t *input_1_data,                \
       const cmsis_nn_dims *input_1_dims, const int8_t *input_2_data,         \
       const cmsis_nn_dims *input_2_dims, int8_t *output_data,                \
       const cmsis_nn_dims *output_dims),                                      \
      (ctx, input_1_data, input_1_dims, input_2_data, input_2_dims,            \
       output_data, output_dims), return, note_ctx(ctx))                       \
    X(arm_cmsis_nn_status, arm_minimum_s8,                                     \
      (const cmsis_nn_context *ctx, const int8_t *input_1_data,                \
       const cmsis_nn_dims *input_1_dims, const int8_t *input_2_data,         \
       const cmsis_nn_dims *input_2_dims, int8_t *output_data,                \
       const cmsis_nn_dims *output_dims),                                      \
      (ctx, input_1_data, input_1_dims, input_2_data, input_2_dims,            \
       output_data, output_dims), return, note_ctx(ctx))                       \
    X(arm_cmsis_nn_status, arm_vector_sum_s8,                                  \
      (int32_t *vector_sum_buf, const int32_t vector_cols,                     \
       const int32_t vector_rows, const int8_t *vector_data,                  \
       const int32_t lhs_offset, const int32_t rhs_offset,                    \
       const int32_t *bias_data),                                              \
      (vector_sum_buf, vector_cols, vector_rows, vector_data, lhs_offset,      \
       rhs_offset, bias_data), return, (void)0)                                \
    X(void, esp_nn_avg_pool_s8_ansi,                                           \
      (const int8_t *input, const uint16_t input_wd, const uint16_t input_ht,  \
       int8_t *output, const uint16_t output_wd, const uint16_t output_ht,    \
       const uint16_t stride_wd, const uint16_t stride_ht,                    \
       const uint16_t filter_wd, const uint16_t filter_ht,                    \
       const uint16_t pad_wd, const uint16_t pad_ht,                          \
       const int32_t activation_min, const int32_t activation_max,            \
       const uint16_t channels),                                               \
      (input, input_wd, input_ht, output, output_wd, output_ht, stride_wd,     \
       stride_ht, filter_wd, filter_ht, pad_wd, pad_ht, activation_min,        \
       activation_max, channels), , (void)0)                                   \
    X(void, ESP_NN_VARIANT(esp_nn_conv_s8),                                    \
      (const data_dims_t *input_dims, const int8_t *input_data,                \
       const data_dims_t *filter_dims, const int8_t *filter_data,             \
       const int32_t *bias, const data_dims_t *output_dims, int8_t *out_data, \
       const conv_params_t *conv_params, const quant_data_t *quant_data),     \
      (input_dims, input_data, filter_dims, filter_data, bias, output_dims,    \
       out_data, conv_params, quant_data), , (void)0)                          \
    X(void, ESP_NN_VARIANT(esp_nn_depthwise_conv_s8),                          \
      (const data_dims_t *input_dims, const int8_t *input_data,                \
       const data_dims_t *filter_dims, const int8_t *filter_data,             \
       const int32_t *bias, const data_dims_t *output_dims, int8_t *out_data, \
       const dw_conv_params_t *conv_params, const quant_data_t *quant_data),  \
      (input_dims, input_data, filter_dims, filter_data, bias, output_dims,    \
       out_data, conv_params, quant_data), , (void)0)                          \
    X(void, esp_nn_fully_connected_per_ch_s8_ansi,                             \
      (const int8_t *input_data, const int32_t input_offset,                   \
       const uint16_t row_len, const int8_t *filter_data,                     \
       const int32_t filter_offset, const int32_t *bias, int8_t *out_data,    \
       const uint16_t out_channels, const int32_t out_offset,                 \
       const int32_t *out_shift, const int32_t *out_mult,                     \
       const int32_t activation_min, const int32_t activation_max),           \
      (input_data, input_offset, row_len, filter_data, filter_offset, bias,    \
       out_data, out_channels, out_offset, out_shift, out_mult,                \
       activation_min, activation_max), , (void)0)                             \
    X(void, esp_nn_fully_connected_s8_ansi,                                    \
      (const int8_t *input_data, const int32_t input_offset,                   \
       const uint16_t row_len, const int8_t *filter_data,                     \
       const int32_t filter_offset, const int32_t *bias, int8_t *out_data,    \
       const uint16_t out_channels, const int32_t out_offset,                 \
       const int32_t out_shift, const int32_t out_mult,                       \
       const int32_t activation_min, const int32_t activation_max),           \
      (input_data, input_offset, row_len, filter_data, filter_offset, bias,    \
       out_data, out_channels, out_offset, out_shift, out_mult,                \
       activation_min, activation_max), , (void)0)                             \
    X(int, ESP_NN_VARIANT(esp_nn_get_conv_scratch_size),                       \
      (const data_dims_t *input_dims, const data_dims_t *filter_dims,          \
       const data_dims_t *output_dims, const conv_params_t *conv_params),     \
      (input_dims, filter_dims, output_dims, conv_params), return, (void)0)    \
    X(int, ESP_NN_VARIANT(esp_nn_get_depthwise_conv_scratch_size),             \
      (const data_dims_t *input_dims, const data_dims_t *filter_dims,          \
       const data_dims_t *output_dims, const dw_conv_params_t *conv_params),  \
      (input_dims, filter_dims, output_dims, conv_params), return, (void)0)    \
    X(void, esp_nn_mul_elementwise_s8_ansi,                                    \
      (const int8_t *input1_data, const int8_t *input2_data,                   \
       const int32_t input1_offset, const int32_t input2_offset,              \
       int8_t *output, const int32_t out_offset, const int32_t out_mult,      \
       const int32_t out_shift, const int32_t activation_min,                 \
       const int32_t activation_max, const int32_t size),                     \
      (input1_data, input2_data, input1_offset, input2_offset, output,         \
       out_offset, out_mult, out_shift, activation_min, activation_max, size), \
      , (void)0)                                                               \
    X(void, ESP_NN_VARIANT(esp_nn_set_conv_scratch_buf), (const void *buf),    \
      (buf), , (void)0)                                                        \
    X(void, ESP_NN_VARIANT(esp_nn_set_depthwise_conv_scratch_buf),             \
      (const void *buf), (buf), , (void)0)

/* Defines the counter and the wrapper, and checks that the parameters match
 * the vendor's own declaration of the symbol. The outer macro expands
 * ESP_NN_VARIANT before the inner one pastes the name. */
#define DEFINE_WRAPPER(...) DEFINE_WRAPPER_(__VA_ARGS__)
#define DEFINE_WRAPPER_(ret, sym, params, args, return_kw, note)              \
    static unsigned long count_##sym;                                          \
    ret __real_##sym params;                                                   \
    ret __wrap_##sym params;                                                   \
    _Static_assert(__builtin_types_compatible_p(__typeof__(sym),               \
                                                __typeof__(__real_##sym)),     \
                   #sym " does not match its vendor declaration");             \
    ret __wrap_##sym params                                                    \
    {                                                                          \
        ++count_##sym;                                                         \
        note;                                                                  \
        return_kw __real_##sym args;                                           \
    }

VENDOR_CALLS(DEFINE_WRAPPER)

typedef struct {
    const char *name;
    unsigned long *count;
} vendor_call_t;

#define TABLE_ENTRY(...) TABLE_ENTRY_(__VA_ARGS__)
#define TABLE_ENTRY_(ret, sym, params, args, return_kw, note) { #sym, &count_##sym },

static const vendor_call_t s_vendor_calls[] = { VENDOR_CALLS(TABLE_ENTRY) };

#define VENDOR_CALL_COUNT (sizeof(s_vendor_calls) / sizeof(s_vendor_calls[0]))

/* Calls made before the run: preparation, including the deinitialization it
 * starts with. */
static unsigned long s_prepare_counts[VENDOR_CALL_COUNT];

/* Largest scratch a CMSIS-NN call was given, in bytes, the most output
 * channels a call read expanded quantization for from the workspace, and the
 * calls of operators preparation sizes scratch for that the policy sent to
 * s8 because a tile was active, though it gives them the adapter untiled. */
static int32_t s_scratch_requested;
static int32_t s_quant_channels;
static unsigned long s_tile_routed;
static const uint8_t *s_workspace;
static uint32_t s_workspace_size;

static void note_ctx(const cmsis_nn_context *ctx)
{
    if (ctx && ctx->size > s_scratch_requested)
        s_scratch_requested = ctx->size;
}

static void note_quant(const int32_t *multiplier, int32_t channels)
{
    const uint8_t *at = (const uint8_t *)multiplier;
    if (s_workspace && at >= s_workspace && at < s_workspace + s_workspace_size &&
        channels > s_quant_channels)
        s_quant_channels = channels;
}

/* Operator types each dispatcher's adapter switch takes. An s8 call from the
 * adapter for one of them is a fallback; for any other type it is the
 * adapter's default route. */
typedef struct {
    uint8_t op_type;
    const char *name;
    int sized;  /* preparation sizes vendor scratch for it */
} adapter_op_t;

static const adapter_op_t s_cmsis_nn_ops[] = {
    { TIGRIS_OP_CONV, "conv", 1 },
    { TIGRIS_OP_DEPTHWISE, "depthwise", 1 },
    { TIGRIS_OP_FULLY_CONN, "fully_connected", 1 },
    { TIGRIS_OP_AVG_POOL, "average_pool", 1 },
    { TIGRIS_OP_GLOBAL_AVG, "global_average_pool", 1 },
    { TIGRIS_OP_MUL, "mul", 0 },
    { TIGRIS_OP_MAXIMUM, "maximum", 0 },
    { TIGRIS_OP_MINIMUM, "minimum", 0 },
};

static const adapter_op_t s_esp_nn_ops[] = {
    { TIGRIS_OP_CONV, "conv", 1 },
    { TIGRIS_OP_DEPTHWISE, "depthwise", 1 },
    { TIGRIS_OP_FULLY_CONN, "fully_connected", 0 },
    { TIGRIS_OP_AVG_POOL, "average_pool", 0 },
    { TIGRIS_OP_MUL, "mul", 0 },
};

#define MAX_ADAPTER_OPS (sizeof(s_cmsis_nn_ops) / sizeof(s_cmsis_nn_ops[0]))

static const adapter_op_t *s_adapter_ops;
static size_t s_adapter_op_count;
static unsigned long s_fallbacks[MAX_ADAPTER_OPS];

int __real_tigris_dispatch_kernel_s8(const tigris_plan_t *plan, const tigris_op_t *op,
                                     uint16_t op_index, tigris_mem_t *mem, void *user_ctx);
int __wrap_tigris_dispatch_kernel_s8(const tigris_plan_t *plan, const tigris_op_t *op,
                                     uint16_t op_index, tigris_mem_t *mem, void *user_ctx);
_Static_assert(__builtin_types_compatible_p(__typeof__(tigris_dispatch_kernel_s8),
                                            __typeof__(__real_tigris_dispatch_kernel_s8)),
               "tigris_dispatch_kernel_s8 does not match its declaration");

int __wrap_tigris_dispatch_kernel_s8(const tigris_plan_t *plan, const tigris_op_t *op,
                                     uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    for (size_t i = 0; op && i < s_adapter_op_count; ++i)
        if (s_adapter_ops[i].op_type == op->op_type)
            ++s_fallbacks[i];
    return __real_tigris_dispatch_kernel_s8(plan, op, op_index, mem, user_ctx);
}

int __real_tigris_accel_try_s8_ref(tigris_accel_backend_t backend, const tigris_plan_t *plan,
                                    const tigris_op_t *op, uint16_t op_index, tigris_mem_t *mem,
                                    void *user_ctx, int *handled);
int __wrap_tigris_accel_try_s8_ref(tigris_accel_backend_t backend, const tigris_plan_t *plan,
                                    const tigris_op_t *op, uint16_t op_index, tigris_mem_t *mem,
                                    void *user_ctx, int *handled);
_Static_assert(__builtin_types_compatible_p(__typeof__(tigris_accel_try_s8_ref),
                                            __typeof__(__real_tigris_accel_try_s8_ref)),
               "tigris_accel_try_s8_ref does not match its declaration");

int __wrap_tigris_accel_try_s8_ref(tigris_accel_backend_t backend, const tigris_plan_t *plan,
                                    const tigris_op_t *op, uint16_t op_index, tigris_mem_t *mem,
                                    void *user_ctx, int *handled)
{
    int result = __real_tigris_accel_try_s8_ref(backend, plan, op, op_index, mem, user_ctx,
                                                handled);
    if (*handled && mem->tile.active &&
        tigris_accel_pre_route(backend, plan, op, op_index, 0) == TIGRIS_ACCEL_ROUTE_ADAPTER) {
        for (size_t i = 0; i < s_adapter_op_count; ++i)
            if (s_adapter_ops[i].op_type == op->op_type && s_adapter_ops[i].sized)
                ++s_tile_routed;
    }
    return result;
}

void tigris_contract_vendor_begin_run(const char *dispatch, const void *workspace,
                                      uint32_t workspace_size)
{
    for (size_t i = 0; i < VENDOR_CALL_COUNT; ++i) {
        s_prepare_counts[i] = *s_vendor_calls[i].count;
        *s_vendor_calls[i].count = 0;
    }
    s_scratch_requested = 0;
    s_quant_channels = 0;
    s_tile_routed = 0;
    s_workspace = workspace;
    s_workspace_size = workspace_size;
    memset(s_fallbacks, 0, sizeof(s_fallbacks));
    s_adapter_ops = NULL;
    s_adapter_op_count = 0;
    if (strcmp(dispatch, "cmsis-nn") == 0) {
        s_adapter_ops = s_cmsis_nn_ops;
        s_adapter_op_count = sizeof(s_cmsis_nn_ops) / sizeof(s_cmsis_nn_ops[0]);
    } else if (strcmp(dispatch, "esp-nn") == 0) {
        s_adapter_ops = s_esp_nn_ops;
        s_adapter_op_count = sizeof(s_esp_nn_ops) / sizeof(s_esp_nn_ops[0]);
    }
}

static int compare_names(const void *a, const void *b)
{
    const vendor_call_t *const *left = a;
    const vendor_call_t *const *right = b;
    return strcmp((*left)->name, (*right)->name);
}

static void print_counts(const char *tag, const vendor_call_t *const *sorted, int prepare)
{
    fputs(tag, stdout);
    /* Every wrapped symbol is listed, uncalled ones with 0, so the line also
     * states the wrapped set. */
    for (size_t i = 0; i < VENDOR_CALL_COUNT; ++i) {
        size_t index = (size_t)(sorted[i] - s_vendor_calls);
        printf(" %s=%lu", sorted[i]->name,
               prepare ? s_prepare_counts[index] : *sorted[i]->count);
    }
    fputc('\n', stdout);
}

void tigris_contract_vendor_print(void)
{
    const vendor_call_t *sorted[VENDOR_CALL_COUNT];
    for (size_t i = 0; i < VENDOR_CALL_COUNT; ++i)
        sorted[i] = &s_vendor_calls[i];
    qsort(sorted, VENDOR_CALL_COUNT, sizeof(sorted[0]), compare_names);
    print_counts("TIGRIS_CONTRACT_VENDOR_PREPARE", sorted, 1);
    print_counts("TIGRIS_CONTRACT_VENDOR", sorted, 0);
    fputs("TIGRIS_CONTRACT_FALLBACK", stdout);
    for (size_t i = 0; i < s_adapter_op_count; ++i)
        printf(" %s=%lu", s_adapter_ops[i].name, s_fallbacks[i]);
    fputc('\n', stdout);
    printf("TIGRIS_CONTRACT_SCRATCH requested=%ld quant_channels=%ld tile_routed=%lu\n",
           (long)s_scratch_requested, (long)s_quant_channels, s_tile_routed);
}

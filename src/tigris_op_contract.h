/* Generated from the compiler's operator table. Do not edit. */
#ifndef TIGRIS_OP_CONTRACT_H
#define TIGRIS_OP_CONTRACT_H

typedef struct {
    uint8_t inputs[3];
    uint8_t output;
} dtype_signature_t;
typedef tigris_error_t (*tigris_op_validator_t)(const tigris_plan_t *,
                                                const tigris_op_t *, uint16_t);
typedef struct {
    dtype_signature_t dtype;
    uint8_t min_inputs;
    uint8_t max_inputs;
    uint8_t min_outputs;
    uint8_t max_outputs;
    uint32_t allowed_attributes;
    uint32_t required_attributes;
    uint16_t flags;
    uint8_t executor;
    uint8_t linked_f32;
    uint8_t linked_s8;
    uint8_t linked_i32;
    tigris_op_validator_t validate;
} tigris_op_contract_t;
typedef struct {
    uint8_t min_len;
    uint8_t max_len;
    uint8_t step;
    uint8_t exact_len;
} tigris_attribute_contract_t;

#define TIGRIS_CONTRACT_LEGACY 0x0001u
#define TIGRIS_CONTRACT_BINARY 0x0002u
#define TIGRIS_CONTRACT_AXIS_UNARY 0x0004u
#define TIGRIS_CONTRACT_AXIS_BINARY 0x0008u
#define TIGRIS_CONTRACT_ROW 0x0010u
#define TIGRIS_CONTRACT_HEIGHT 0x0020u
#define TIGRIS_CONTRACT_SPATIAL 0x0040u
#define TIGRIS_CONTRACT_RESAMPLE 0x0080u
#define TIGRIS_CONTRACT_WHOLE_OPERAND 0x0100u
#define TIGRIS_CONTRACT_FUSED 0x0200u
#define TIGRIS_CONTRACT_FUSED_FLOAT 0x0400u

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert((uint32_t)TIGRIS_OP_CONV == 1u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_DEPTHWISE == 2u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RELU == 3u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RELU6 == 4u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MAX_POOL == 5u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_AVG_POOL == 6u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ADD == 7u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MUL == 8u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_FULLY_CONN == 9u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SOFTMAX == 10u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CLIP == 11u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SIGMOID == 12u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CONCAT == 13u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_PAD == 14u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GLOBAL_AVG == 15u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_FLATTEN == 16u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RESHAPE == 17u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SUB == 18u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_DIV == 19u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_TANH == 20u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LEAKY_RELU == 21u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_BATCH_NORM == 22u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_INST_NORM == 23u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CONV_TRANSPOSE == 24u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MATMUL == 25u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REDUCE_MEAN == 26u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SQUEEZE == 27u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_UNSQUEEZE == 28u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_TRANSPOSE == 29u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RESIZE == 30u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GLOBAL_MAX == 31u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CONV1D == 32u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LAYER_NORM == 33u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ERF == 34u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SPLIT == 35u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RESIZE_LINEAR == 36u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_HARDSWISH == 37u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ABS == 38u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_RSQRT == 39u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SQUARED_DIFFERENCE == 40u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MAXIMUM == 41u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MINIMUM == 42u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_NEG == 43u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_EXP == 44u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LOG == 45u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SQRT == 46u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SQUARE == 47u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_FLOOR == 48u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CEIL == 49u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ROUND == 50u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SIN == 51u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_COS == 52u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_FLOOR_DIV == 53u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_FLOOR_MOD == 54u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_PRELU == 55u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ELU == 56u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LOG_SOFTMAX == 57u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_L2_NORMALIZATION == 58u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_L2_POOL == 59u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REDUCE_MAX == 60u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REDUCE_MIN == 61u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REDUCE_SUM == 62u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CUMSUM == 63u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ARG_MAX == 64u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ARG_MIN == 65u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GATHER == 66u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GATHER_ND == 67u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_STRIDED_SLICE == 68u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_MIRROR_PAD == 69u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REVERSE_V2 == 70u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_EMBEDDING_LOOKUP == 71u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_DYNAMIC_UPDATE_SLICE == 72u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_EQUAL == 73u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LESS == 74u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LESS_EQUAL == 75u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GREATER == 76u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GREATER_EQUAL == 77u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LOGICAL_AND == 78u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LOGICAL_OR == 79u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LOGICAL_NOT == 80u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SELECT_V2 == 81u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_CAST == 82u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_ADD_N == 83u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_REDUCE_ALL == 84u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_SVDF == 85u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_LSTM == 86u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_IF == 87u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_WHILE == 88u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_DETECTION_POSTPROCESS == 89u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_QUANTIZE == 90u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_DEQUANTIZE == 91u, "operator opcode");
_Static_assert((uint32_t)TIGRIS_OP_GELU == 92u, "operator opcode");
#endif

static tigris_error_t validate_contract_abs(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_add(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_arg_max(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_concat(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_conv(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_conv1d(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_conv_transpose(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_cumsum(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_detection_postprocess(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_div(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_equal(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_fully_conn(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_gather(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_gelu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_global_max(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_hardswish(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_if(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_layer_norm(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_leaky_relu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_lstm(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_matmul(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_max_pool(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_pad(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_quantize(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_relu(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_reshape(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_resize_linear(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_split(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_svdf(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);
static tigris_error_t validate_contract_transpose(const tigris_plan_t *plan,
                                         const tigris_op_t *op, uint16_t op_index);

static const tigris_op_contract_t *tigris_op_contract(uint8_t opcode)
{
    static const tigris_op_contract_t rows[(uint32_t)TIGRIS_OP_GELU + 1u] = {
        [TIGRIS_OP_CONV] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_SPATIAL | TIGRIS_CONTRACT_FUSED, 0u, TIGRIS_KERNEL_F32_CONV, TIGRIS_KERNEL_S8_CONV, TIGRIS_KERNEL_I32, validate_contract_conv},
        [TIGRIS_OP_DEPTHWISE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_SPATIAL | TIGRIS_CONTRACT_FUSED, 0u, TIGRIS_KERNEL_F32_DEPTHWISE, TIGRIS_KERNEL_S8_DEPTHWISE, TIGRIS_KERNEL_I32, validate_contract_conv},
        [TIGRIS_OP_RELU] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_RELU, TIGRIS_KERNEL_S8_RELU, TIGRIS_KERNEL_I32, validate_contract_relu},
        [TIGRIS_OP_RELU6] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_RELU6, TIGRIS_KERNEL_S8_RELU6, TIGRIS_KERNEL_I32, validate_contract_relu},
        [TIGRIS_OP_MAX_POOL] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_SPATIAL, 0u, TIGRIS_KERNEL_F32_MAX_POOL, TIGRIS_KERNEL_S8_MAX_POOL, TIGRIS_KERNEL_I32, validate_contract_max_pool},
        [TIGRIS_OP_AVG_POOL] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_SPATIAL, 0u, TIGRIS_KERNEL_F32_AVG_POOL, TIGRIS_KERNEL_S8_AVG_POOL, TIGRIS_KERNEL_I32, validate_contract_max_pool},
        [TIGRIS_OP_ADD] = {{{253u, 253u, 253u}, 253u}, 1u, 2u, 1u, 1u, 0x00000280u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT | TIGRIS_CONTRACT_FUSED, 0u, TIGRIS_KERNEL_F32_ADD, TIGRIS_KERNEL_S8_ADD, TIGRIS_KERNEL_I32, validate_contract_add},
        [TIGRIS_OP_MUL] = {{{253u, 253u, 253u}, 253u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_MUL, TIGRIS_KERNEL_S8_MUL, TIGRIS_KERNEL_I32, validate_contract_add},
        [TIGRIS_OP_FULLY_CONN] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_WHOLE_OPERAND | TIGRIS_CONTRACT_FUSED, 0u, TIGRIS_KERNEL_F32_FULLY_CONN, TIGRIS_KERNEL_S8_FULLY_CONN, TIGRIS_KERNEL_I32, validate_contract_fully_conn},
        [TIGRIS_OP_SOFTMAX] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SOFTMAX, TIGRIS_KERNEL_S8_SOFTMAX, TIGRIS_KERNEL_I32, validate_contract_relu},
        [TIGRIS_OP_CLIP] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000010u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_CLIP, TIGRIS_KERNEL_S8_CLIP, TIGRIS_KERNEL_I32, NULL},
        [TIGRIS_OP_SIGMOID] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SIGMOID, TIGRIS_KERNEL_S8_SIGMOID, TIGRIS_KERNEL_I32, validate_contract_relu},
        [TIGRIS_OP_CONCAT] = {{{0u, 0u, 0u}, 0u}, 1u, 255u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_CONCAT, TIGRIS_KERNEL_S8_CONCAT, TIGRIS_KERNEL_I32, validate_contract_concat},
        [TIGRIS_OP_PAD] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000020u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_PAD, TIGRIS_KERNEL_S8_PAD, TIGRIS_KERNEL_I32, validate_contract_pad},
        [TIGRIS_OP_GLOBAL_AVG] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000100u, 0x00000000u, TIGRIS_CONTRACT_LEGACY, 0u, TIGRIS_KERNEL_F32_GLOBAL_AVG, TIGRIS_KERNEL_S8_GLOBAL_AVG, TIGRIS_KERNEL_I32, validate_contract_global_max},
        [TIGRIS_OP_FLATTEN] = {{{255u, 255u, 255u}, 255u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_ROW, 0u, TIGRIS_KERNEL_F32_FLATTEN, TIGRIS_KERNEL_S8_FLATTEN, TIGRIS_KERNEL_I32, validate_contract_reshape},
        [TIGRIS_OP_RESHAPE] = {{{255u, 255u, 255u}, 255u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_ROW, 0u, TIGRIS_KERNEL_F32_RESHAPE, TIGRIS_KERNEL_S8_RESHAPE, TIGRIS_KERNEL_I32, validate_contract_reshape},
        [TIGRIS_OP_SUB] = {{{253u, 253u, 253u}, 253u}, 1u, 2u, 1u, 1u, 0x00000280u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SUB, TIGRIS_KERNEL_S8_SUB, TIGRIS_KERNEL_I32, validate_contract_add},
        [TIGRIS_OP_DIV] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_DIV, TIGRIS_KERNEL_S8_DIV, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_TANH] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_TANH, TIGRIS_KERNEL_S8_TANH, TIGRIS_KERNEL_I32, validate_contract_relu},
        [TIGRIS_OP_LEAKY_RELU] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000008u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LEAKY_RELU, TIGRIS_KERNEL_S8_LEAKY_RELU, TIGRIS_KERNEL_I32, validate_contract_leaky_relu},
        [TIGRIS_OP_BATCH_NORM] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_BATCH_NORM, TIGRIS_KERNEL_S8_BATCH_NORM, TIGRIS_KERNEL_I32, NULL},
        [TIGRIS_OP_INST_NORM] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_INST_NORM, TIGRIS_KERNEL_S8_INST_NORM, TIGRIS_KERNEL_I32, NULL},
        [TIGRIS_OP_CONV_TRANSPOSE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY, 0u, TIGRIS_KERNEL_F32_CONV_TRANSPOSE, TIGRIS_KERNEL_S8_CONV_TRANSPOSE, TIGRIS_KERNEL_I32, validate_contract_conv_transpose},
        [TIGRIS_OP_MATMUL] = {{{0u, 0u, 0u}, 0u}, 2u, 2u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_WHOLE_OPERAND, 0u, TIGRIS_KERNEL_F32_MATMUL, TIGRIS_KERNEL_S8_MATMUL, TIGRIS_KERNEL_I32, validate_contract_matmul},
        [TIGRIS_OP_REDUCE_MEAN] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, TIGRIS_CONTRACT_LEGACY, 0u, TIGRIS_KERNEL_F32_REDUCE_MEAN, TIGRIS_KERNEL_S8_REDUCE_MEAN, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_SQUEEZE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_SQUEEZE, TIGRIS_KERNEL_S8_SQUEEZE, TIGRIS_KERNEL_I32, NULL},
        [TIGRIS_OP_UNSQUEEZE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_UNSQUEEZE, TIGRIS_KERNEL_S8_UNSQUEEZE, TIGRIS_KERNEL_I32, NULL},
        [TIGRIS_OP_TRANSPOSE] = {{{255u, 255u, 255u}, 255u}, 1u, 1u, 1u, 1u, 0x00000002u, 0x00000002u, 0u, 0u, TIGRIS_KERNEL_F32_TRANSPOSE, TIGRIS_KERNEL_S8_TRANSPOSE, TIGRIS_KERNEL_I32, validate_contract_transpose},
        [TIGRIS_OP_RESIZE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000400u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_RESAMPLE, 0u, TIGRIS_KERNEL_F32_RESIZE, TIGRIS_KERNEL_S8_RESIZE, TIGRIS_KERNEL_I32, validate_contract_resize_linear},
        [TIGRIS_OP_GLOBAL_MAX] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY, 0u, TIGRIS_KERNEL_F32_GLOBAL_MAX, TIGRIS_KERNEL_S8_GLOBAL_MAX, TIGRIS_KERNEL_I32, validate_contract_global_max},
        [TIGRIS_OP_CONV1D] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_LEGACY | TIGRIS_CONTRACT_FUSED, 0u, TIGRIS_KERNEL_F32_CONV1D, TIGRIS_KERNEL_S8_CONV1D, TIGRIS_KERNEL_I32, validate_contract_conv1d},
        [TIGRIS_OP_LAYER_NORM] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000004u, 0x00000004u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LAYER_NORM, TIGRIS_KERNEL_S8_LAYER_NORM, TIGRIS_KERNEL_I32, validate_contract_layer_norm},
        [TIGRIS_OP_ERF] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_ERF, TIGRIS_KERNEL_S8_ERF, TIGRIS_KERNEL_I32, validate_contract_hardswish},
        [TIGRIS_OP_SPLIT] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 2u, 255u, 0x00000000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_SPLIT, TIGRIS_KERNEL_S8_SPLIT, TIGRIS_KERNEL_I32, validate_contract_split},
        [TIGRIS_OP_RESIZE_LINEAR] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000400u, 0x00000000u, TIGRIS_CONTRACT_RESAMPLE, 0u, TIGRIS_KERNEL_F32_RESIZE_LINEAR, TIGRIS_KERNEL_S8_RESIZE_LINEAR, TIGRIS_KERNEL_I32, validate_contract_resize_linear},
        [TIGRIS_OP_HARDSWISH] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_HARDSWISH, TIGRIS_KERNEL_S8_HARDSWISH, TIGRIS_KERNEL_I32, validate_contract_hardswish},
        [TIGRIS_OP_ABS] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_ABS, TIGRIS_KERNEL_S8_ABS, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_RSQRT] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_RSQRT, TIGRIS_KERNEL_S8_RSQRT, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_SQUARED_DIFFERENCE] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SQUARED_DIFFERENCE, TIGRIS_KERNEL_S8_SQUARED_DIFFERENCE, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_MAXIMUM] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_MAXIMUM, TIGRIS_KERNEL_S8_MAXIMUM, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_MINIMUM] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_MINIMUM, TIGRIS_KERNEL_S8_MINIMUM, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_NEG] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_NEG, TIGRIS_KERNEL_S8_NEG, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_EXP] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_EXP, TIGRIS_KERNEL_S8_EXP, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_LOG] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LOG, TIGRIS_KERNEL_S8_LOG, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_SQRT] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SQRT, TIGRIS_KERNEL_S8_SQRT, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_SQUARE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SQUARE, TIGRIS_KERNEL_S8_SQUARE, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_FLOOR] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_FLOOR, TIGRIS_KERNEL_S8_FLOOR, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_CEIL] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_CEIL, TIGRIS_KERNEL_S8_CEIL, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_ROUND] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_ROUND, TIGRIS_KERNEL_S8_ROUND, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_SIN] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SIN, TIGRIS_KERNEL_S8_SIN, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_COS] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_COS, TIGRIS_KERNEL_S8_COS, TIGRIS_KERNEL_I32, validate_contract_abs},
        [TIGRIS_OP_FLOOR_DIV] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_FLOOR_DIV, TIGRIS_KERNEL_S8_FLOOR_DIV, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_FLOOR_MOD] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_FLOOR_MOD, TIGRIS_KERNEL_S8_FLOOR_MOD, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_PRELU] = {{{0u, 0u, 0u}, 0u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_PRELU, TIGRIS_KERNEL_S8_PRELU, TIGRIS_KERNEL_I32, validate_contract_div},
        [TIGRIS_OP_ELU] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_ELU, TIGRIS_KERNEL_S8_ELU, TIGRIS_KERNEL_I32, validate_contract_leaky_relu},
        [TIGRIS_OP_LOG_SOFTMAX] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LOG_SOFTMAX, TIGRIS_KERNEL_S8_LOG_SOFTMAX, TIGRIS_KERNEL_I32, validate_contract_leaky_relu},
        [TIGRIS_OP_L2_NORMALIZATION] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000004u, 0x00000004u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_L2_NORMALIZATION, TIGRIS_KERNEL_S8_L2_NORMALIZATION, TIGRIS_KERNEL_I32, validate_contract_leaky_relu},
        [TIGRIS_OP_L2_POOL] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_SPATIAL, 0u, TIGRIS_KERNEL_F32_L2_POOL, TIGRIS_KERNEL_S8_L2_POOL, TIGRIS_KERNEL_I32, validate_contract_max_pool},
        [TIGRIS_OP_REDUCE_MAX] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_REDUCE_MAX, TIGRIS_KERNEL_S8_REDUCE_MAX, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_REDUCE_MIN] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_REDUCE_MIN, TIGRIS_KERNEL_S8_REDUCE_MIN, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_REDUCE_SUM] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_REDUCE_SUM, TIGRIS_KERNEL_S8_REDUCE_SUM, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_CUMSUM] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00000840u, 0x00000840u, 0u, 0u, TIGRIS_KERNEL_F32_CUMSUM, TIGRIS_KERNEL_S8_CUMSUM, TIGRIS_KERNEL_I32, validate_contract_cumsum},
        [TIGRIS_OP_ARG_MAX] = {{{0u, 0u, 0u}, 6u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_ARG_MAX, TIGRIS_KERNEL_S8_ARG_MAX, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_ARG_MIN] = {{{0u, 0u, 0u}, 6u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_ARG_MIN, TIGRIS_KERNEL_S8_ARG_MIN, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_GATHER] = {{{0u, 6u, 6u}, 0u}, 0u, 2u, 1u, 1u, 0x00101000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_GATHER, TIGRIS_KERNEL_S8_GATHER, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_GATHER_ND] = {{{0u, 6u, 6u}, 0u}, 0u, 2u, 1u, 1u, 0x00101000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_GATHER_ND, TIGRIS_KERNEL_S8_GATHER_ND, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_STRIDED_SLICE] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00001000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_STRIDED_SLICE, TIGRIS_KERNEL_S8_STRIDED_SLICE, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_MIRROR_PAD] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00001000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_MIRROR_PAD, TIGRIS_KERNEL_S8_MIRROR_PAD, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_REVERSE_V2] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00001000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_REVERSE_V2, TIGRIS_KERNEL_S8_REVERSE_V2, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_EMBEDDING_LOOKUP] = {{{0u, 6u, 6u}, 0u}, 0u, 2u, 1u, 1u, 0x00101000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_EMBEDDING_LOOKUP, TIGRIS_KERNEL_S8_EMBEDDING_LOOKUP, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_DYNAMIC_UPDATE_SLICE] = {{{0u, 0u, 6u}, 0u}, 1u, 3u, 1u, 1u, 0x00001000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_DYNAMIC_UPDATE_SLICE, TIGRIS_KERNEL_S8_DYNAMIC_UPDATE_SLICE, TIGRIS_KERNEL_I32, validate_contract_gather},
        [TIGRIS_OP_EQUAL] = {{{253u, 253u, 253u}, 9u}, 1u, 2u, 1u, 1u, 0x00002200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_EQUAL, TIGRIS_KERNEL_S8_EQUAL, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_LESS] = {{{253u, 253u, 253u}, 9u}, 1u, 2u, 1u, 1u, 0x00002200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LESS, TIGRIS_KERNEL_S8_LESS, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_LESS_EQUAL] = {{{253u, 253u, 253u}, 9u}, 1u, 2u, 1u, 1u, 0x00002200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LESS_EQUAL, TIGRIS_KERNEL_S8_LESS_EQUAL, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_GREATER] = {{{253u, 253u, 253u}, 9u}, 1u, 2u, 1u, 1u, 0x00002200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_GREATER, TIGRIS_KERNEL_S8_GREATER, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_GREATER_EQUAL] = {{{253u, 253u, 253u}, 9u}, 1u, 2u, 1u, 1u, 0x00002200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_GREATER_EQUAL, TIGRIS_KERNEL_S8_GREATER_EQUAL, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_LOGICAL_AND] = {{{9u, 9u, 9u}, 9u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LOGICAL_AND, TIGRIS_KERNEL_S8_LOGICAL_AND, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_LOGICAL_OR] = {{{9u, 9u, 9u}, 9u}, 1u, 2u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LOGICAL_OR, TIGRIS_KERNEL_S8_LOGICAL_OR, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_LOGICAL_NOT] = {{{9u, 9u, 9u}, 9u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_LOGICAL_NOT, TIGRIS_KERNEL_S8_LOGICAL_NOT, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_SELECT_V2] = {{{9u, 0u, 0u}, 0u}, 2u, 3u, 1u, 1u, 0x00000200u, 0x00000000u, TIGRIS_CONTRACT_BINARY | TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_SELECT_V2, TIGRIS_KERNEL_S8_SELECT_V2, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_CAST] = {{{252u, 252u, 252u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_CAST, TIGRIS_KERNEL_S8_CAST, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_ADD_N] = {{{0u, 0u, 0u}, 0u}, 1u, 255u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_BINARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_ADD_N, TIGRIS_KERNEL_S8_ADD_N, TIGRIS_KERNEL_I32, validate_contract_equal},
        [TIGRIS_OP_REDUCE_ALL] = {{{9u, 9u, 9u}, 9u}, 1u, 1u, 1u, 1u, 0x00000040u, 0x00000040u, 0u, 0u, TIGRIS_KERNEL_F32_REDUCE_ALL, TIGRIS_KERNEL_S8_REDUCE_ALL, TIGRIS_KERNEL_I32, validate_contract_arg_max},
        [TIGRIS_OP_SVDF] = {{{0u, 254u, 254u}, 254u}, 2u, 2u, 2u, 2u, 0x0000c000u, 0x00000000u, TIGRIS_CONTRACT_FUSED_FLOAT, 0u, TIGRIS_KERNEL_F32_SVDF, TIGRIS_KERNEL_S8_SVDF, TIGRIS_KERNEL_I32, validate_contract_svdf},
        [TIGRIS_OP_LSTM] = {{{0u, 254u, 254u}, 254u}, 3u, 3u, 3u, 3u, 0x00014000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_LSTM, TIGRIS_KERNEL_S8_LSTM, TIGRIS_KERNEL_I32, validate_contract_lstm},
        [TIGRIS_OP_IF] = {{{9u, 251u, 251u}, 251u}, 1u, 255u, 1u, 255u, 0x00020000u, 0x00000000u, 0u, 1u, 1u, 1u, 1u, validate_contract_if},
        [TIGRIS_OP_WHILE] = {{{251u, 251u, 251u}, 251u}, 1u, 255u, 1u, 255u, 0x00024000u, 0x00000000u, 0u, 1u, 1u, 1u, 1u, validate_contract_if},
        [TIGRIS_OP_DETECTION_POSTPROCESS] = {{{0u, 0u, 0u}, 1u}, 2u, 2u, 5u, 5u, 0x00044000u, 0x00000000u, 0u, 0u, TIGRIS_KERNEL_F32_DETECTION_POSTPROCESS, TIGRIS_KERNEL_S8_DETECTION_POSTPROCESS, TIGRIS_KERNEL_I32, validate_contract_detection_postprocess},
        [TIGRIS_OP_QUANTIZE] = {{{250u, 250u, 250u}, 0u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_QUANTIZE, TIGRIS_KERNEL_S8_QUANTIZE, TIGRIS_KERNEL_I32, validate_contract_quantize},
        [TIGRIS_OP_DEQUANTIZE] = {{{0u, 0u, 0u}, 1u}, 1u, 1u, 1u, 1u, 0x00000000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_DEQUANTIZE, TIGRIS_KERNEL_S8_DEQUANTIZE, TIGRIS_KERNEL_I32, validate_contract_quantize},
        [TIGRIS_OP_GELU] = {{{0u, 0u, 0u}, 0u}, 1u, 1u, 1u, 1u, 0x00080000u, 0x00000000u, TIGRIS_CONTRACT_AXIS_UNARY | TIGRIS_CONTRACT_ROW | TIGRIS_CONTRACT_HEIGHT, 0u, TIGRIS_KERNEL_F32_GELU, TIGRIS_KERNEL_S8_GELU, TIGRIS_KERNEL_I32, validate_contract_gelu},
    };
    const tigris_op_contract_t *result = NULL;
    if (opcode < sizeof(rows) / sizeof(rows[0]))
        result = &rows[opcode];
    return result;
}

static const tigris_attribute_contract_t *tigris_attribute_contract(uint8_t kind)
{
    static const tigris_attribute_contract_t rows[21u] = {
        [1] = {0u, 255u, 1u, 0u},
        [2] = {4u, 4u, 1u, 4u},
        [3] = {4u, 4u, 1u, 4u},
        [4] = {8u, 8u, 1u, 8u},
        [5] = {8u, 248u, 8u, 0u},
        [6] = {1u, 255u, 1u, 0u},
        [7] = {24u, 24u, 1u, 24u},
        [8] = {1u, 1u, 1u, 1u},
        [9] = {4u, 252u, 4u, 0u},
        [10] = {8u, 8u, 1u, 8u},
        [11] = {2u, 2u, 1u, 2u},
        [12] = {0u, 255u, 1u, 0u},
        [13] = {20u, 20u, 1u, 20u},
        [14] = {2u, 254u, 2u, 0u},
        [15] = {4u, 24u, 1u, 0u},
        [16] = {8u, 112u, 1u, 0u},
        [17] = {4u, 4u, 1u, 4u},
        [18] = {32u, 32u, 1u, 32u},
        [19] = {1u, 1u, 1u, 1u},
        [20] = {4u, 252u, 4u, 0u},
    };
    const tigris_attribute_contract_t *result = NULL;
    if (kind < sizeof(rows) / sizeof(rows[0]))
        result = &rows[kind];
    return result;
}
#endif

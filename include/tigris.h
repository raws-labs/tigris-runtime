/**
 * @file tigris.h
 * @brief TiGrIS binary plan format - shared struct definitions.
 *
 * This header defines the binary wire format produced by the TiGrIS Python
 * toolchain (tigris compile). It is target-agnostic: any C runtime (ESP32,
 * POSIX, etc.) includes this header to parse .tgrs files.
 *
 * All structs are packed, little-endian, fixed-size. The plan file is
 * designed for zero-copy loading: the C loader returns pointers directly
 * into the loaded buffer.
 */

#ifndef TIGRIS_H
#define TIGRIS_H

#include <stdint.h>
#include <stddef.h>

#include "tigris_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime release; scripts/check_version_sources.py holds it to the CMake project version. */
#define TIGRIS_RUNTIME_VERSION_MAJOR 0
#define TIGRIS_RUNTIME_VERSION_MINOR 12
#define TIGRIS_RUNTIME_VERSION_PATCH 1

/* Keep in sync with tigris/src/tigris/emitters/binary/defs.py */

/* Format constants */

#define TIGRIS_MAGIC           0x53524754  /* "TGRS" in little-endian */
#define TIGRIS_MAGIC_BYTES     "TGRS"
#define TIGRIS_SCHEMA_VERSION_V11 11
#define TIGRIS_SCHEMA_VERSION_V10 10
#define TIGRIS_SCHEMA_VERSION_V9 9
#define TIGRIS_SCHEMA_VERSION_V8 8
#define TIGRIS_SCHEMA_VERSION_V7 7
#define TIGRIS_SCHEMA_VERSION_V6 6
#define TIGRIS_SCHEMA_VERSION_V5 5
#define TIGRIS_SCHEMA_VERSION TIGRIS_SCHEMA_VERSION_V11
#define TIGRIS_SCHEMA_VERSION_V4 4
#define TIGRIS_SCHEMA_VERSION_MIN TIGRIS_SCHEMA_VERSION_V4
#define TIGRIS_SCHEMA_VERSION_OP_ATTRIBUTES TIGRIS_SCHEMA_VERSION_V4
#define TIGRIS_SCHEMA_VERSION_TILE_AXIS TIGRIS_SCHEMA_VERSION_V5
#define TIGRIS_SCHEMA_VERSION_STAGE_TABLE_AUTHORITY TIGRIS_SCHEMA_VERSION_V5
/* A model input or output carries the dtype the model file declares for it,
 * so the runtime can present the interface the model states rather than the
 * encoding the plan executes on. */
#define TIGRIS_SCHEMA_VERSION_INTERFACE_DTYPE TIGRIS_SCHEMA_VERSION_V6
/* A tensor records whether its axes are stored in the model's own order, so a
 * boundary's element order is recoverable from the plan rather than inferred
 * from the shape of the graph that produced it. */
#define TIGRIS_SCHEMA_VERSION_TENSOR_LAYOUT TIGRIS_SCHEMA_VERSION_V7

/* A quantized sum states the multipliers that scale each operand and its own
 * result, so the kernel stops deriving them in double precision on every
 * call and a vendor kernel that takes them can be given them. */
#define TIGRIS_SCHEMA_VERSION_BINARY_REQUANT TIGRIS_SCHEMA_VERSION_V9

/* A conversion's tile plan states which of the two swapped axes its band runs
 * along, instead of both sides deriving it from the extents. */
#define TIGRIS_SCHEMA_VERSION_BAND_ORIENTATION TIGRIS_SCHEMA_VERSION_V9

/* A plan may keep variables across invocations in a caller-supplied state
 * buffer. Only a plan that does is written at this version. */
#define TIGRIS_SCHEMA_VERSION_STATE TIGRIS_SCHEMA_VERSION_V10

/* A plan states the fast-arena bytes its activations need, so a run refuses a smaller arena up front. */
#define TIGRIS_SCHEMA_VERSION_SCHEDULED_PEAK TIGRIS_SCHEMA_VERSION_V11

/* A tile record states its tiling, which the loader derives from the plan and checks. */
#define TIGRIS_SCHEMA_VERSION_TILE_FACTS TIGRIS_SCHEMA_VERSION_V11
#define TIGRIS_QUANT_PAGE_ELEMS 65536u

/* Section type IDs */
#define TIGRIS_SEC_TENSORS        1
#define TIGRIS_SEC_OPS            2
#define TIGRIS_SEC_STAGES         3
#define TIGRIS_SEC_TILE_PLANS     4
#define TIGRIS_SEC_INDEX_POOL     5
#define TIGRIS_SEC_SHAPE_POOL     6
#define TIGRIS_SEC_STRINGS        7
#define TIGRIS_SEC_WEIGHTS        8
#define TIGRIS_SEC_QUANT_PARAMS   9
#define TIGRIS_SEC_WEIGHT_BLOCKS  10
#define TIGRIS_SEC_OP_ATTRIBUTES  11
#define TIGRIS_SEC_STATE          12
#define TIGRIS_SEC_SUBGRAPHS      13
#define TIGRIS_SEC_MEMORY         14  /* tigris_memory_t, schema 11+ */
#define TIGRIS_SEC_MAX            15  /* one past the last valid section */

/* ONNX TensorProto.DataType values that a plan tensor or a model boundary
 * can carry. */
#define TIGRIS_DTYPE_FLOAT32 1u
#define TIGRIS_DTYPE_UINT8   2u
#define TIGRIS_DTYPE_INT8    3u
#define TIGRIS_DTYPE_INT16   5u
#define TIGRIS_DTYPE_INT32   6u
#define TIGRIS_DTYPE_INT64   7u
#define TIGRIS_DTYPE_BOOL    9u

/* Auxiliary tensor placement is constrained by operator dtype slots. */
static inline int tigris_dtype_requires_source(uint8_t dtype)
{
    return dtype == TIGRIS_DTYPE_INT32;
}

/* ONNX dtype element sizes; zero denotes an unsupported storage type. */
static inline uint32_t tigris_dtype_size(uint8_t dtype)
{
    switch (dtype) {
    case TIGRIS_DTYPE_FLOAT32: case TIGRIS_DTYPE_INT32: return 4u;
    case TIGRIS_DTYPE_UINT8: case TIGRIS_DTYPE_INT8: case TIGRIS_DTYPE_BOOL: return 1u;
    case TIGRIS_DTYPE_INT16: return 2u;
    case TIGRIS_DTYPE_INT64: return 8u;
    default: return 0u;
    }
}

/* Typed payloads stored in TIGRIS_SEC_OP_ATTRIBUTES. The loader refuses a kind
 * it does not know, so a plan carrying a newer kind fails to load on an older
 * runtime instead of running with different semantics; a kind can therefore be
 * added without a schema version. Payloads are little-endian and their length
 * is the record's data_len. */
#define TIGRIS_OP_ATTR_TRANSPOSE_PERM 1  /* uint8[rank], axis permutation */
#define TIGRIS_OP_ATTR_EPSILON        2  /* float32, normalization floor */
#define TIGRIS_OP_ATTR_ALPHA          3  /* float32, negative slope */
#define TIGRIS_OP_ATTR_CLIP_BOUNDS    4  /* float32[2], lower then upper */
#define TIGRIS_OP_ATTR_PADS           5  /* int32[2*rank], leading, trailing */
#define TIGRIS_OP_ATTR_AXES           6  /* uint8[n], axes a reduction takes */
#define TIGRIS_OP_ATTR_BINARY_REQUANT 7  /* int32[6], see below */
#define TIGRIS_OP_ATTR_POOL_ROUNDING  8  /* uint8[1], see below */
#define TIGRIS_OP_ATTR_CONSTANT_OPERAND 9 /* uint8[4], see below */
#define TIGRIS_OP_ATTR_RESIZE_SCALES  10 /* float32[2], output/input H/W scales */
#define TIGRIS_OP_ATTR_CUMSUM_OPTIONS 11 /* uint8[2], exclusive and reverse */
#define TIGRIS_OP_ATTR_MOVEMENT       12 /* int32 metadata, described below */
/* Gather/EmbeddingLookup: axis, batch_dims, index_rank, index_shape[index_rank].
 * GatherND: index_rank, index_shape[index_rank]. Indices are an int32 weight.
 * StridedSlice: (normalized start, exclusive end, step) per input axis,
 * optionally followed by a shrink-axis bitmask.
 * MirrorPad: symmetric (0/1), then (before, after) for each input axis.
 * ReverseV2: a nonempty contiguous axis bitmask.
 * DynamicUpdateSlice: clamped starts[rank], update_shape[rank]. The update is
 * the second data tensor or the operator weight, in the input's encoding. */
#define TIGRIS_OP_ATTR_COMPARISON_REQUANT 13 /* int32[5]: left shift, two multiplier/shift pairs */
#define TIGRIS_OP_ATTR_COMPARISON_REQUANT_LEN 20u
#define TIGRIS_OP_ATTR_CONSTANTS      14 /* uint16 weight index per constant operand */
/* In operand order, TIGRIS_NO_WEIGHT for an absent optional operand; an
 * operator carrying it has neither weight_idx nor bias_idx. */
#define TIGRIS_OP_ATTR_SVDF           15 /* int32 rank; int8 adds four more, below */
/* Svdf: rank; an int8 Svdf adds the state zero point, then the input-to-state
 * and state-to-output (multiplier, shift) pairs. */
#define TIGRIS_OP_ATTR_LSTM           16 /* int32 time_major (0/1), float32 cell clip */
#define TIGRIS_OP_ATTR_SUBGRAPHS      17 /* uint16 graphs a control-flow op runs (If: then, else;
                                          * While: condition, body) */
#define TIGRIS_OP_ATTR_DETECTION      18 /* DetectionPostProcess options, below */
#define TIGRIS_OP_ATTR_DETECTION_LEN  32u
/* DetectionPostProcess: uint16 max detections, detections per class and
 * classes, uint8 regular (1) or fast (0) suppression, a zero byte, then float32
 * score threshold, IoU threshold and the y, x, h, w scales. */
#define TIGRIS_OP_ATTR_GELU_APPROXIMATE 19 /* uint8 exact (0) or tanh approximation (1) */
#define TIGRIS_OP_ATTR_TABLE_SHAPE    20 /* int32 extents of a constant Gather data operand */
#define TIGRIS_OP_ATTR_MAX            20

/* Working memory DetectionPostProcess takes in its last output, for `boxes`
 * anchors, `detections` results and `candidates` scores kept across classes:
 * the detections, plus the detections per class under regular suppression. */
#define TIGRIS_DETECTION_SCRATCH_BYTES(boxes, detections, candidates)               \
    (4u * (9u * (size_t)(boxes) +                                                   \
           ((size_t)(boxes) > (size_t)(candidates) ? (size_t)(boxes) : (size_t)(candidates)) + \
           2u * (size_t)(candidates) + (size_t)(detections)) +                      \
     (size_t)(boxes))

/* A binary requant payload is three (multiplier, shift) pairs in Q0.31: the
 * first operand's, the second operand's, and the result's. */
#define TIGRIS_OP_ATTR_BINARY_REQUANT_LEN 24u

/* A binary or select operator may take one operand from its weight: one
 * value, one per channel, or one per element. This payload says which operand
 * is constant (byte 0: operand index, up to 2 for select; byte 1: zero) and,
 * for int8, its quantization (bytes 2-3: quant param index, little endian;
 * TIGRIS_NO_QUANT_PARAM for float or bool). Without it, only a float Add or Mul takes
 * a constant, as its second operand. */
#define TIGRIS_OP_ATTR_CONSTANT_OPERAND_LEN 4u

/* An int8 global average pool rounds like TFLite MEAN by default: the 1/HW
 * factor is folded into a fixed-point multiplier. With this attribute it rounds
 * like TFLite AVERAGE_POOL_2D instead: the integer sum is divided by HW with
 * ties away from zero. The only valid payload is this value. */
#define TIGRIS_POOL_ROUNDING_AVERAGE 1u

/* Compression types */
#define TIGRIS_COMPRESS_NONE  0
#define TIGRIS_COMPRESS_LZ4   1

/* Header flags */
#define TIGRIS_FLAG_XIP       0x00000001u  /* weights are execute-in-place from flash */

/* Fused activation types */
#define TIGRIS_ACT_NONE   0
#define TIGRIS_ACT_RELU   1
#define TIGRIS_ACT_RELU6  2

/* Sentinel: weight_idx/bias_idx == 0xFFFF means no weight/bias */
#define TIGRIS_NO_WEIGHT        0xFFFF

/* Sentinel: quant_param_idx == 0xFFFF means no quant param */
#define TIGRIS_NO_QUANT_PARAM   0xFFFF

/* Op type enum */

typedef enum {
    TIGRIS_OP_CONV              = 1,
    TIGRIS_OP_DEPTHWISE         = 2,
    TIGRIS_OP_RELU              = 3,
    TIGRIS_OP_RELU6             = 4,
    TIGRIS_OP_MAX_POOL          = 5,
    TIGRIS_OP_AVG_POOL          = 6,
    TIGRIS_OP_ADD               = 7,
    TIGRIS_OP_MUL               = 8,
    TIGRIS_OP_FULLY_CONN        = 9,
    TIGRIS_OP_SOFTMAX           = 10,
    TIGRIS_OP_CLIP              = 11,
    TIGRIS_OP_SIGMOID           = 12,
    TIGRIS_OP_CONCAT            = 13,
    TIGRIS_OP_PAD               = 14,
    TIGRIS_OP_GLOBAL_AVG        = 15,
    TIGRIS_OP_FLATTEN           = 16,
    TIGRIS_OP_RESHAPE           = 17,
    TIGRIS_OP_SUB               = 18,
    TIGRIS_OP_DIV               = 19,
    TIGRIS_OP_TANH              = 20,
    TIGRIS_OP_LEAKY_RELU        = 21,
    TIGRIS_OP_BATCH_NORM        = 22,
    TIGRIS_OP_INST_NORM         = 23,
    TIGRIS_OP_CONV_TRANSPOSE    = 24,
    TIGRIS_OP_MATMUL            = 25,
    TIGRIS_OP_REDUCE_MEAN       = 26,
    TIGRIS_OP_SQUEEZE           = 27,
    TIGRIS_OP_UNSQUEEZE         = 28,
    TIGRIS_OP_TRANSPOSE         = 29,
    TIGRIS_OP_RESIZE            = 30,
    TIGRIS_OP_GLOBAL_MAX        = 31,
    TIGRIS_OP_CONV1D            = 32,
    TIGRIS_OP_LAYER_NORM        = 33,
    TIGRIS_OP_ERF               = 34,
    TIGRIS_OP_SPLIT             = 35,
    TIGRIS_OP_RESIZE_LINEAR     = 36,  /* bilinear upsample */
    TIGRIS_OP_HARDSWISH         = 37,
    TIGRIS_OP_ABS               = 38,
    TIGRIS_OP_RSQRT             = 39,
    TIGRIS_OP_SQUARED_DIFFERENCE = 40,
    TIGRIS_OP_MAXIMUM           = 41,
    TIGRIS_OP_MINIMUM           = 42,
    TIGRIS_OP_NEG               = 43,
    TIGRIS_OP_EXP               = 44,
    TIGRIS_OP_LOG               = 45,
    TIGRIS_OP_SQRT              = 46,
    TIGRIS_OP_SQUARE            = 47,
    TIGRIS_OP_FLOOR             = 48,
    TIGRIS_OP_CEIL              = 49,
    TIGRIS_OP_ROUND             = 50,
    TIGRIS_OP_SIN               = 51,
    TIGRIS_OP_COS               = 52,
    TIGRIS_OP_FLOOR_DIV         = 53,
    TIGRIS_OP_FLOOR_MOD         = 54,
    TIGRIS_OP_PRELU              = 55,
    TIGRIS_OP_ELU                = 56,
    TIGRIS_OP_LOG_SOFTMAX        = 57,
    TIGRIS_OP_L2_NORMALIZATION   = 58,
    TIGRIS_OP_L2_POOL            = 59,
    TIGRIS_OP_REDUCE_MAX         = 60,
    TIGRIS_OP_REDUCE_MIN         = 61,
    TIGRIS_OP_REDUCE_SUM         = 62,
    TIGRIS_OP_CUMSUM             = 63,
    TIGRIS_OP_ARG_MAX            = 64,
    TIGRIS_OP_ARG_MIN            = 65,
    TIGRIS_OP_GATHER                   = 66,
    TIGRIS_OP_GATHER_ND                = 67,
    TIGRIS_OP_STRIDED_SLICE            = 68,
    TIGRIS_OP_MIRROR_PAD               = 69,
    TIGRIS_OP_REVERSE_V2               = 70,
    TIGRIS_OP_EMBEDDING_LOOKUP         = 71,
    TIGRIS_OP_DYNAMIC_UPDATE_SLICE     = 72,
    TIGRIS_OP_EQUAL                      = 73,
    TIGRIS_OP_LESS                       = 74,
    TIGRIS_OP_LESS_EQUAL                 = 75,
    TIGRIS_OP_GREATER                    = 76,
    TIGRIS_OP_GREATER_EQUAL              = 77,
    TIGRIS_OP_LOGICAL_AND                = 78,
    TIGRIS_OP_LOGICAL_OR                 = 79,
    TIGRIS_OP_LOGICAL_NOT                = 80,
    TIGRIS_OP_SELECT_V2                  = 81,
    TIGRIS_OP_CAST                       = 82,
    TIGRIS_OP_ADD_N                      = 83,
    TIGRIS_OP_REDUCE_ALL                 = 84,
    /* State in [batch, filters * memory], kept between runs; constants are
     * feature [filters, features], time [filters, memory] and bias [units]. */
    TIGRIS_OP_SVDF                       = 85,
    /* Sequence, hidden state and cell state in; hidden sequence and both
     * states out. Constants: input weights, recurrent weights and biases of
     * gates i, f, c, o, twelve in all. */
    TIGRIS_OP_LSTM                       = 86,
    /* A one-element bool condition, then operands; runs the then or the else
     * subgraph on copies of the operands and copies its results out. */
    TIGRIS_OP_IF                         = 87,
    /* Loop variables in and out; runs its condition subgraph, and while that
     * gives true, its body on the current values. */
    TIGRIS_OP_WHILE                      = 88,
    /* SSD box encodings [1, boxes, 4] and scores [1, boxes, classes] in the
     * plan's data dtype, anchors [boxes, 4] in float32 as its constant; writes
     * float32 boxes [1, detections, 4], classes and scores [1, detections],
     * the count [1] and its working memory, as TFLite Micro does. */
    TIGRIS_OP_DETECTION_POSTPROCESS      = 89,
    /* float32 to the int8 encoding of its output, as TFLite Micro's QUANTIZE:
     * a float32 division by the scale, rounded half away from zero. */
    TIGRIS_OP_QUANTIZE                   = 90,
    /* int8 to float32, as TFLite Micro's DEQUANTIZE: the scale times the
     * offset value, in double. Both carry float32 across the boundary of a
     * control-flow subgraph in an int8 plan. */
    TIGRIS_OP_DEQUANTIZE                 = 91,
    TIGRIS_OP_GELU                       = 92,
    TIGRIS_OP_UNKNOWN           = 255,
} tigris_op_type_t;

/* Tensor flags */

#define TIGRIS_TENSOR_CONSTANT      0x01
#define TIGRIS_TENSOR_MODEL_INPUT   0x02
#define TIGRIS_TENSOR_MODEL_OUTPUT  0x04
/* The tensor's axes are stored in the order the model states them, not
 * channels-last. Read it on a model input or output to know which order the
 * caller's data is in: without it, a boundary written by a terminal Transpose
 * is indistinguishable from any other. Plans below schema 7 never set it, and
 * their boundaries follow the older rule of channels-last unless the graph
 * ends in a Transpose. */
#define TIGRIS_TENSOR_LINEAR        0x08
/* A variable's value entering or leaving an invocation through the state
 * buffer; it is not part of the model's interface. */
#define TIGRIS_TENSOR_STATE         0x10

/* Error codes */

/** Error codes returned by tigris_plan_load() and the model interface calls. */
typedef enum {
    TIGRIS_OK               = 0,    /* Success */
    TIGRIS_ERR_NULL         = -1,   /* Null pointer argument */
    TIGRIS_ERR_TOO_SMALL    = -2,   /* Buffer smaller than file header */
    TIGRIS_ERR_BAD_MAGIC    = -3,   /* Magic bytes are not "TGRS" */
    TIGRIS_ERR_BAD_VERSION  = -4,   /* Unsupported plan schema version */
    TIGRIS_ERR_BAD_SIZE     = -5,   /* file_size field does not match buf_len */
    TIGRIS_ERR_BAD_SECTION  = -6,   /* Section offset out of bounds */
    TIGRIS_ERR_MISSING_SEC  = -7,   /* Required section not present */
    TIGRIS_ERR_ENDIAN       = -8,   /* Platform is not little-endian */
    TIGRIS_ERR_PLAN_LIMITS  = -9,   /* Plan exceeds compiled executor limits */
    TIGRIS_ERR_BAD_TENSOR   = -10,  /* Tensor metadata is not executable */
    TIGRIS_ERR_BAD_OPERATOR = -11,  /* Operator contract is not executable */
    TIGRIS_ERR_BAD_INTERFACE = -12, /* Declared model interface is not convertible */
    TIGRIS_ERR_KERNEL_DISABLED = -13, /* Build leaves out a kernel the plan uses */
    TIGRIS_ERR_MEM          = -14,  /* Arena cannot hold the tensor */
    TIGRIS_ERR_TENSOR_SLOTS = -15,  /* Memory manager has fewer tensor slots than the plan */
    TIGRIS_ERR_TILE_FACTS   = -16,  /* A tile record's stated tiling disagrees with the plan */
} tigris_error_t;

/* Sentinel: tile_plan_idx == 0xFFFF means no tile plan for this stage */
#define TIGRIS_NO_TILE_PLAN     0xFFFF

/* Serialized activation axes used by schema-v5 tile plans. */
#define TIGRIS_TILE_AXIS_NONE             0
#define TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH 1
#define TIGRIS_TILE_AXIS_WIDTH            2  /* reserved, not accepted */
#define TIGRIS_TILE_AXIS_HW               3

/* Sentinel: chain_id == 0xFFFF means stage is not part of a chain */
#define TIGRIS_NO_CHAIN         0xFFFF

/* Stage flags stored in the chain head stage's _reserved1 field. The
 * compiler sets LINE_BUFFERED on a recomputing chain whose shared boundary
 * rows can be kept in a line buffer; the executor then rolls those rows
 * instead of recomputing them per tile. */
#define TIGRIS_STAGE_FLAG_LINE_BUFFERED 0x0001u

/* Packed struct definitions */

#pragma pack(push, 1)

/**
 * File header - 48 bytes.
 */
typedef struct {
    uint8_t     magic[4];           /*  0: "TGRS" */
    uint32_t    version;            /*  4: TIGRIS_SCHEMA_VERSION */
    uint32_t    file_size;          /*  8: total file size in bytes */
    uint32_t    section_dir_off;    /* 12: byte offset of section directory */
    uint16_t    num_tensors;        /* 16 */
    uint16_t    num_ops;            /* 18 */
    uint16_t    num_stages;         /* 20 */
    uint16_t    num_tile_plans;     /* 22 */
    uint32_t    budget;             /* 24: memory budget in bytes */
    uint32_t    peak;               /* 28: peak activation memory */
    uint32_t    model_name_str;     /* 32: offset into string table */
    uint16_t    model_io_off;       /* 36: offset into index pool */
    uint8_t     num_model_inputs;   /* 38 */
    uint8_t     num_model_outputs;  /* 39 */
    uint16_t    num_weights;        /* 40 */
    uint16_t    num_quant_params;   /* 42: quant param count (0 = no quantization) */
    uint32_t    flags;              /* 44: header-level flags (currently 0) */
} tigris_file_header_t;             /* 48 bytes total */

/**
 * Section directory entry - 8 bytes.
 * Terminated by type == 0.
 */
typedef struct {
    uint32_t    type;               /* TIGRIS_SEC_* constant, 0 = sentinel */
    uint32_t    offset;             /* absolute byte offset in file */
} tigris_section_entry_t;

/**
 * Tensor descriptor - 16 bytes.
 */
typedef struct {
    uint32_t    name_str;           /*  0: offset into string table */
    uint32_t    size_bytes;         /*  4: total byte size */
    uint16_t    shape_off;          /*  8: offset into shape pool (element index) */
    uint8_t     ndim;               /* 10: number of dimensions */
    uint8_t     dtype;              /* 11: ONNX TensorProto.DataType enum */
    uint8_t     flags;              /* 12: TIGRIS_TENSOR_* flags */
    uint16_t    quant_param_idx;    /* 13-14: index into quant_params, 0xFFFF=none */
    uint8_t     iface_dtype;        /* 15: ONNX dtype the model declares for this
                                     *     model input/output, 0 = same as dtype */
} tigris_tensor_t;                  /* 16 bytes total */

/* The dtype the model file declares for a boundary tensor, which is its own
 * dtype when the plan records none. */
static inline uint8_t tigris_tensor_declared_dtype(const tigris_tensor_t *tensor)
{
    return (tensor->iface_dtype != 0u) ? tensor->iface_dtype : tensor->dtype;
}

/**
 * Spatial attributes - 18 bytes, embedded in tigris_op_t.
 * Zero for non-spatial ops (pointwise, reshape, etc.).
 */
typedef struct {
    uint8_t     kernel_h;
    uint8_t     kernel_w;
    uint8_t     stride_h;
    uint8_t     stride_w;
    uint16_t    pad_top;
    uint16_t    pad_bottom;
    uint16_t    pad_left;
    uint16_t    pad_right;
    uint16_t    dilation_h;
    uint16_t    dilation_w;
    uint16_t    group;              /* LE; 1 for normal conv */
} tigris_spatial_attrs_t;           /* 18 bytes */

/**
 * Operator descriptor - 38 bytes (layout retained by schema v3).
 */
typedef struct {
    uint32_t    name_str;            /*  0: offset into string table */
    uint8_t     op_type;             /*  4: tigris_op_type_t */
    uint8_t     num_inputs;          /*  5 */
    uint8_t     num_outputs;         /*  6 */
    uint8_t     stage;               /*  7: stage index (v4); low-byte hint (v5+) */
    uint16_t    inputs_off;          /*  8: offset into index pool */
    uint16_t    outputs_off;         /* 10: offset into index pool */
    tigris_spatial_attrs_t spatial;  /* 12-29: spatial attributes (18 bytes, v2) */
    uint16_t    weight_idx;          /* 30: index into weight_entries, 0xFFFF=none */
    uint16_t    bias_idx;            /* 32: index into weight_entries, 0xFFFF=none */
    uint8_t     fused_act;           /* 34: TIGRIS_ACT_* */
    int8_t      act_min;             /* 35: int8 lower bound */
    int8_t      act_max;             /* 36: int8 upper bound */
    uint8_t     _pad1;               /* 37 */
} tigris_op_t;                       /* 38 bytes total */

/**
 * Stage descriptor - 28 bytes.
 */
typedef struct {
    uint32_t    peak_bytes;         /*  0 */
    uint16_t    ops_off;            /*  4: offset into index pool */
    uint16_t    ops_count;          /*  6 */
    uint16_t    inputs_off;         /*  8 */
    uint16_t    inputs_count;       /* 10 */
    uint16_t    outputs_off;        /* 12 */
    uint16_t    outputs_count;      /* 14 */
    uint16_t    tile_plan_idx;      /* 16: index into tile_plan table, 0xFFFF=none */
    uint16_t    _pad0;              /* 18 */
    uint16_t    chain_id;           /* 20: first stage of chain, 0xFFFF=standalone */
    uint16_t    chain_len;          /* 22: number of stages in chain (0=not in chain) */
    uint16_t    chain_tile_h;       /* 24: output tile height for last stage (on head) */
    uint16_t    _reserved1;         /* 26: TIGRIS_STAGE_FLAG_* bits */
} tigris_stage_t;                   /* 28 bytes total */

/**
 * Tile plan through schema 9 - 24 bytes.
 */
typedef struct {
    uint8_t     tileable;           /*  0: 1=tileable, 0=not */
    uint8_t     axis;               /*  1: serialized activation axis (v5+) */
    uint16_t    tile_height;        /*  2 */
    uint16_t    num_tiles;          /*  4 */
    uint16_t    halo;               /*  6 */
    uint16_t    receptive_field;    /*  8 */
    uint16_t    original_height;    /* 10 */
    uint32_t    tiled_peak_bytes;   /* 12 */
    uint32_t    overhead_bytes;     /* 16 */
    uint16_t    tile_width;         /* 20: 2D tile width, 0 when the plan is 1D */
    uint16_t    flags;              /* 22: TIGRIS_TILE_FLAG_* (v9+), else 0 */
} tigris_tile_plan_t;               /* 24 bytes total */

/** Schema-10 scheduling choices, 32 bytes. Zero extents are derived. */
typedef struct {
    uint8_t tileable;                 /*  0 */
    uint8_t axis;                     /*  1 */
    uint16_t flags;                   /*  2 */
    uint32_t input_band_extent;       /*  4: input rows */
    uint32_t output_band_extent;      /*  8: output rows */
    uint32_t output_width_extent;     /* 12: output columns */
    uint32_t input_channel_group;     /* 16: input channels */
    uint32_t input_spatial_interval;  /* 20: positions within one input batch */
    uint32_t output_linear_segment;   /* 24: output elements */
    uint32_t tiled_peak_bytes;        /* 28: fast working-set ceiling */
} tigris_tile_plan_v10_t;

/**
 * What a schema 11 tile record states about its tiling, 16 bytes. The loader
 * derives each field from the plan and refuses a record that disagrees.
 */
typedef struct {
    uint32_t tile_extent;             /*  0: band length the count uses */
    uint32_t num_tiles;               /*  4: bands of tile_extent (times columns for HW) */
    uint32_t original_extent;         /*  8: extent the bands divide */
    uint16_t halo;                    /* 12: rows a band reads beyond its own */
    uint8_t band_axis;                /* 14: serialized axis the bands run along */
    uint8_t band;                     /* 15: band kind and mode, TIGRIS_TILE_BAND_* */
} tigris_tile_facts_t;

/** Schema-11 tile record, 48 bytes: the schema-10 choices and their facts. */
typedef struct {
    tigris_tile_plan_v10_t choices;   /*  0 */
    tigris_tile_facts_t facts;        /* 32 */
} tigris_tile_plan_v11_t;

/* The band byte of a schema 11 tile record states how the executor bands the
 * stage: a tigris_band_kind_t (tigris_loader.h) in the low four bits and, for
 * a leading band, its band mode in the high four, otherwise 0. */
#define TIGRIS_TILE_BAND_KIND(band) ((uint8_t)((band) & 0x0Fu))
#define TIGRIS_TILE_BAND_MODE(band) ((uint8_t)((band) >> 4))

/* Leading band modes: the axis of each tensor a leading band runs along. */
#define TIGRIS_BAND_LEADING 2u          /* the first axis whose extent is not 1 */
#define TIGRIS_BAND_BATCH 3u            /* axis 0 */
#define TIGRIS_BAND_TIME_MAJOR_BATCH 4u /* axis 1 of a rank-3 tensor, else axis 0 */
#define TIGRIS_BAND_TRAILING 5u         /* the last axis */
#define TIGRIS_BAND_AXIS_BASE 6u        /* plus the serialized axis */

/* Set when a layout conversion's band runs along the columns. The compiler
 * records the axis, so the two sides agree when the extents are equal, as in a
 * square attention block. */
#define TIGRIS_TILE_FLAG_BAND_ON_COLUMNS 0x0001u

/**
 * Weight entry - 12 bytes.
 * Describes one named weight blob within the weights section.
 */
typedef struct {
    uint32_t    name_str;           /*  0: offset into string table */
    uint32_t    offset;             /*  4: byte offset into weight blob */
    uint32_t    size_bytes;         /*  8: size of this weight in bytes */
} tigris_weight_entry_t;            /* 12 bytes total */

/**
 * Quantization parameter - 16 bytes.
 * Per-tensor or per-channel quantization metadata.
 * For per-channel, multiplier/shift arrays are in the quant data blob.
 */
typedef struct {
    float       scale;              /*  0: per-tensor scale (or first channel) */
    int32_t     zero_point;         /*  4: per-tensor zero point */
    uint16_t    num_channels;       /*  8: 1=per-tensor, >1=per-channel */
    uint16_t    multiplier_off;     /* 10: offset into quant data (int32 elements) */
    uint16_t    shift_off;          /* 12: offset into quant data (int32 elements) */
    uint16_t    _pad;               /* 14: quant-data page */
} tigris_quant_param_t;             /* 16 bytes total */

/**
 * Weight block entry - 20 bytes.
 * Describes one per-stage compressed weight block. Its weight indices need
 * not be consecutive, so first_weight_idx and num_weights do not name a
 * range; each weight entry locates its weight in the decoded block.
 */
typedef struct {
    uint16_t    stage_idx;          /*  0: stage index */
    uint16_t    first_weight_idx;   /*  2: lowest weight index in the block */
    uint16_t    num_weights;        /*  4: number of weights in the block */
    uint16_t    _pad0;              /*  6 */
    uint32_t    blob_offset;        /*  8: offset into compressed data area */
    uint32_t    compressed_size;    /* 12 */
    uint32_t    uncompressed_size;  /* 16 */
} tigris_weight_block_t;            /* 20 bytes total */

/** Typed variable-length operator attribute descriptor - 8 bytes. */
typedef struct {
    uint16_t    op_index;            /* 0: global operator index */
    uint8_t     type;                /* 2: TIGRIS_OP_ATTR_* */
    uint8_t     data_len;            /* 3: payload bytes */
    uint32_t    data_offset;         /* 4: offset into attribute data pool */
} tigris_op_attribute_t;

#pragma pack(pop)

/*
 * Compile-time wire-format contract.
 *
 * Use named negative-size typedefs rather than C11-only _Static_assert so the
 * public header remains usable from C99 and C++ translation units. A layout
 * drift therefore fails at compile time, before a mismatched runtime can load
 * a plan produced by the compiler.
 */
#define TIGRIS_LAYOUT_ASSERT(name, expr) \
    typedef char name[(expr) ? 1 : -1]

TIGRIS_LAYOUT_ASSERT(tigris_layout_file_header_size,
                     sizeof(tigris_file_header_t) == 48);
TIGRIS_LAYOUT_ASSERT(tigris_layout_section_entry_size,
                     sizeof(tigris_section_entry_t) == 8);
TIGRIS_LAYOUT_ASSERT(tigris_layout_tensor_size,
                     sizeof(tigris_tensor_t) == 16);
TIGRIS_LAYOUT_ASSERT(tigris_layout_spatial_attrs_size,
                     sizeof(tigris_spatial_attrs_t) == 18);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_size,
                     sizeof(tigris_op_t) == 38);
TIGRIS_LAYOUT_ASSERT(tigris_layout_stage_size,
                     sizeof(tigris_stage_t) == 28);
TIGRIS_LAYOUT_ASSERT(tigris_layout_tile_plan_v10_size,
                     sizeof(tigris_tile_plan_v10_t) == 32);
TIGRIS_LAYOUT_ASSERT(tigris_layout_tile_facts_size,
                     sizeof(tigris_tile_facts_t) == 16);
TIGRIS_LAYOUT_ASSERT(tigris_layout_tile_plan_v11_size,
                     sizeof(tigris_tile_plan_v11_t) == 48);
TIGRIS_LAYOUT_ASSERT(tigris_layout_tile_plan_size,
                     sizeof(tigris_tile_plan_t) == 24);
TIGRIS_LAYOUT_ASSERT(tigris_layout_weight_entry_size,
                     sizeof(tigris_weight_entry_t) == 12);
TIGRIS_LAYOUT_ASSERT(tigris_layout_quant_param_size,
                     sizeof(tigris_quant_param_t) == 16);
TIGRIS_LAYOUT_ASSERT(tigris_layout_weight_block_size,
                     sizeof(tigris_weight_block_t) == 20);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_attribute_size,
                     sizeof(tigris_op_attribute_t) == 8);

TIGRIS_LAYOUT_ASSERT(tigris_layout_spatial_pad_top_offset,
                     offsetof(tigris_spatial_attrs_t, pad_top) == 4);
TIGRIS_LAYOUT_ASSERT(tigris_layout_spatial_dilation_h_offset,
                     offsetof(tigris_spatial_attrs_t, dilation_h) == 12);
TIGRIS_LAYOUT_ASSERT(tigris_layout_spatial_group_offset,
                     offsetof(tigris_spatial_attrs_t, group) == 16);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_spatial_offset,
                     offsetof(tigris_op_t, spatial) == 12);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_weight_idx_offset,
                     offsetof(tigris_op_t, weight_idx) == 30);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_bias_idx_offset,
                     offsetof(tigris_op_t, bias_idx) == 32);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_fused_act_offset,
                     offsetof(tigris_op_t, fused_act) == 34);
TIGRIS_LAYOUT_ASSERT(tigris_layout_op_pad_offset,
                     offsetof(tigris_op_t, _pad1) == 37);

/**
 * State entry - 16 bytes, one per variable in TIGRIS_SEC_STATE. The section
 * starts with count(u16), a zero u16 and state_bytes(u32); the entries follow,
 * then the initial values they point at.
 */
typedef struct {
    uint16_t    input;              /*  0: tensor holding the value on entry */
    uint16_t    output;             /*  2: tensor holding the value on exit, 0xFFFF = unchanged */
    uint32_t    offset;             /*  4: byte offset in the state buffer */
    uint32_t    bytes;              /*  8: byte size of the value */
    uint32_t    initial;            /* 12: offset of the initial value in the section */
} tigris_state_entry_t;

TIGRIS_LAYOUT_ASSERT(tigris_layout_state_entry_size, sizeof(tigris_state_entry_t) == 16);

/**
 * Subgraph - 12 bytes, one per graph in TIGRIS_SEC_SUBGRAPHS, the main graph
 * first. The section starts with count(u16) and a zero u16. A graph is a range
 * of stages with its input and output tensors in the index pool; a subgraph's
 * tensors belong to it alone.
 */
typedef struct {
    uint16_t    first_stage;        /*  0 */
    uint16_t    num_stages;         /*  2 */
    uint16_t    inputs_off;         /*  4: index pool */
    uint16_t    inputs_count;       /*  6 */
    uint16_t    outputs_off;        /*  8: index pool */
    uint16_t    outputs_count;      /* 10 */
} tigris_subgraph_t;

TIGRIS_LAYOUT_ASSERT(tigris_layout_subgraph_size, sizeof(tigris_subgraph_t) == 12);

/**
 * Memory record, 4 bytes: the whole of TIGRIS_SEC_MEMORY, which every plan of
 * schema 11 or later carries and earlier plans do not.
 */
typedef struct {
    uint32_t    scheduled_peak;     /* 0: fast bytes the largest stage or chain holds, as checked; no decompressed weights */
} tigris_memory_t;

TIGRIS_LAYOUT_ASSERT(tigris_layout_memory_size, sizeof(tigris_memory_t) == 4);

#undef TIGRIS_LAYOUT_ASSERT

/* Parsed plan handle */

/**
 * Parsed plan: pointers into the loaded buffer. No allocations needed.
 * Populate with tigris_plan_load().
 */
typedef struct {
    const tigris_file_header_t  *header;
    const tigris_tensor_t       *tensors;       /* [num_tensors] */
    const tigris_op_t           *ops;           /* [num_ops] */
    const tigris_stage_t        *stages;        /* [num_stages] */
    const tigris_tile_plan_t    *tile_plans;    /* [num_tile_plans] */
    const uint16_t              *index_pool;
    const int32_t               *shape_pool;
    const char                  *strings;
    const tigris_weight_entry_t *weight_entries; /* [num_weights] or NULL */
    const uint8_t               *weight_blob;    /* contiguous weight data or NULL */

    /* Compressed weight blocks (optional) */
    const tigris_weight_block_t *weight_blocks;      /* [num_weight_blocks] or NULL */
    const uint8_t               *weight_blocks_data;  /* compressed blobs or NULL */
    uint16_t                     num_weight_blocks;
    uint16_t                     weight_compression;  /* TIGRIS_COMPRESS_* */

    /* Optional variable-length per-operator attributes. */
    const tigris_op_attribute_t *op_attributes;
    const uint8_t               *op_attribute_data;
    uint16_t                     num_op_attributes;

    /* Quantization (optional) */
    const tigris_quant_param_t  *quant_params;  /* [num_quant_params] or NULL */
    const int32_t               *quant_data;    /* multiplier/shift arrays or NULL */
    uint16_t                     num_quant_params;

    /* Variables kept across invocations (optional, schema v10+) */
    const tigris_state_entry_t  *state_entries;  /* [num_state] or NULL */
    const uint8_t               *state_section;  /* initial values are read from here */
    uint16_t                     num_state;
    uint32_t                     state_bytes;    /* size of the caller's state buffer */
    const tigris_subgraph_t     *subgraphs;      /* [num_subgraphs] or NULL */
    uint16_t                     num_subgraphs;  /* 0 without control flow */
    uint16_t                     subgraph_depth; /* most control-flow operators running one inside another */

    const tigris_tile_plan_v10_t *tile_plans_v10; /* schema 10 only, otherwise NULL */

    /* Convenience: model I/O resolved from header */
    const uint16_t              *model_inputs;  /* [num_model_inputs] */
    const uint16_t              *model_outputs; /* [num_model_outputs] */

    uint32_t                     scheduled_peak; /* schema 11+, see tigris_memory_t; 0 before */
    const tigris_tile_plan_v11_t *tile_plans_v11; /* schema 11+, otherwise NULL */
} tigris_plan_t;

/* Inline helper functions */

/**
 * Get the name of a tensor from the string table.
 *
 * @param plan  Loaded plan.
 * @param t     Tensor descriptor.
 * @return Null-terminated tensor name string.
 */
static inline const char *tigris_tensor_name(
    const tigris_plan_t *plan, const tigris_tensor_t *t)
{
    return plan->strings + t->name_str;
}

/** Return the payload for a typed attribute on one operator, or NULL. */
static inline const uint8_t *tigris_op_attribute_data(
    const tigris_plan_t *plan, uint16_t op_index, uint8_t type,
    uint8_t *out_len)
{
    if (out_len)
        *out_len = 0;
    if (!plan || !plan->op_attributes || !plan->op_attribute_data)
        return NULL;
    for (uint16_t i = 0; i < plan->num_op_attributes; i++) {
        const tigris_op_attribute_t *attr = &plan->op_attributes[i];
        if (attr->op_index == op_index && attr->type == type) {
            if (out_len)
                *out_len = attr->data_len;
            return plan->op_attribute_data + attr->data_offset;
        }
    }
    return NULL;
}

/**
 * Get the shape array for a tensor.
 *
 * @param plan  Loaded plan.
 * @param t     Tensor descriptor.
 * @return Pointer to t->ndim int32_t dimension values in shape_pool.
 */
static inline const int32_t *tigris_tensor_shape(
    const tigris_plan_t *plan, const tigris_tensor_t *t)
{
    return plan->shape_pool + t->shape_off;
}

/**
 * Get the name of an op from the string table.
 *
 * @param plan  Loaded plan.
 * @param op    Operator descriptor.
 * @return Null-terminated op name string.
 */
static inline const char *tigris_op_name(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    return plan->strings + op->name_str;
}

/**
 * Get the input tensor indices for an op.
 *
 * @param plan  Loaded plan.
 * @param op    Operator descriptor.
 * @return Pointer to op->num_inputs uint16_t tensor indices.
 */
static inline const uint16_t *tigris_op_inputs(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    return plan->index_pool + op->inputs_off;
}

/**
 * Get the output tensor indices for an op.
 *
 * @param plan  Loaded plan.
 * @param op    Operator descriptor.
 * @return Pointer to op->num_outputs uint16_t tensor indices.
 */
static inline const uint16_t *tigris_op_outputs(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    return plan->index_pool + op->outputs_off;
}

/**
 * Get the op indices for a stage.
 *
 * @param plan   Loaded plan.
 * @param stage  Stage descriptor.
 * @return Pointer to stage->ops_count uint16_t op indices.
 */
static inline const uint16_t *tigris_stage_ops(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    return plan->index_pool + stage->ops_off;
}

/**
 * Get the input tensor indices for a stage.
 *
 * @param plan   Loaded plan.
 * @param stage  Stage descriptor.
 * @return Pointer to stage->inputs_count uint16_t tensor indices.
 */
static inline const uint16_t *tigris_stage_inputs(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    return plan->index_pool + stage->inputs_off;
}

/**
 * Get the output tensor indices for a stage.
 *
 * @param plan   Loaded plan.
 * @param stage  Stage descriptor.
 * @return Pointer to stage->outputs_count uint16_t tensor indices.
 */
static inline const uint16_t *tigris_stage_outputs(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    return plan->index_pool + stage->outputs_off;
}

/**
 * Get a legacy tile record for a stage.
 *
 * @param plan   Loaded plan.
 * @param stage  Stage descriptor.
 * @return Legacy record or NULL. Use tigris_stage_tiles for all schemas.
 */
static inline const tigris_tile_plan_t *tigris_stage_tile_plan(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (!plan->tile_plans || stage->tile_plan_idx == TIGRIS_NO_TILE_PLAN)
        return NULL;
    return &plan->tile_plans[stage->tile_plan_idx];
}

/** Scheduling choices for any wire layout; no allocation or shared cache. */
static inline tigris_tile_plan_v10_t tigris_stage_tiles(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    tigris_tile_plan_v10_t result = {0};
    if (!plan || !plan->header || !stage ||
        stage->tile_plan_idx == TIGRIS_NO_TILE_PLAN ||
        stage->tile_plan_idx >= plan->header->num_tile_plans)
        return result;
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_FACTS) {
        if (plan->tile_plans_v11)
            return plan->tile_plans_v11[stage->tile_plan_idx].choices;
        return result;
    }
    if (plan->header->version >= TIGRIS_SCHEMA_VERSION_STATE) {
        if (plan->tile_plans_v10)
            return plan->tile_plans_v10[stage->tile_plan_idx];
        return result;
    }
    const tigris_tile_plan_t *old = tigris_stage_tile_plan(plan, stage);
    if (!old)
        return result;
    result.tileable = old->tileable;
    result.axis = (uint8_t)(plan->header->version >= TIGRIS_SCHEMA_VERSION_TILE_AXIS
        ? old->axis : (old->tileable ? TIGRIS_TILE_AXIS_HEIGHT_OR_LENGTH : TIGRIS_TILE_AXIS_NONE));
    result.flags = old->flags;
    result.tiled_peak_bytes = old->tiled_peak_bytes;
    result.output_width_extent = old->tile_width;
    int output_driven = 0;
    if (result.axis == TIGRIS_TILE_AXIS_HW)
        output_driven = 1;
    const uint16_t *ops = tigris_stage_ops(plan, stage);
    for (uint16_t i = 0; i < stage->ops_count; i++) {
        uint8_t type = plan->ops[ops[i]].op_type;
        if (type == TIGRIS_OP_RESIZE || type == TIGRIS_OP_RESIZE_LINEAR)
            output_driven = 1;
    }
    if (output_driven)
        result.output_band_extent = old->tile_height;
    else
        result.input_band_extent = old->tile_height;
    return result;
}

/**
 * Get the model name string from the plan header.
 *
 * @param plan  Loaded plan.
 * @return Null-terminated model name string.
 */
static inline const char *tigris_model_name(const tigris_plan_t *plan)
{
    return plan->strings + plan->header->model_name_str;
}

/**
 * Get the weight data pointer for a weight entry.
 *
 * @param plan  Loaded plan.
 * @param w     Weight entry descriptor.
 * @return Pointer to the raw weight bytes in the weight blob.
 */
static inline const void *tigris_weight_data(
    const tigris_plan_t *plan, const tigris_weight_entry_t *w)
{
    return plan->weight_blob + w->offset;
}

/**
 * Get the name of a weight entry from the string table.
 *
 * @param plan  Loaded plan.
 * @param w     Weight entry descriptor.
 * @return Null-terminated weight name string.
 */
static inline const char *tigris_weight_name(
    const tigris_plan_t *plan, const tigris_weight_entry_t *w)
{
    return plan->strings + w->name_str;
}

/**
 * Get the weight data for an op.
 *
 * @param plan  Loaded plan.
 * @param op    Operator descriptor.
 * @return Pointer to weight data, or NULL if the op has no weight.
 */
static inline const void *tigris_op_weight(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    if (op->weight_idx == TIGRIS_NO_WEIGHT || !plan->weight_entries)
        return NULL;
    return tigris_weight_data(plan, &plan->weight_entries[op->weight_idx]);
}

/**
 * Get the bias data for an op.
 *
 * @param plan  Loaded plan.
 * @param op    Operator descriptor.
 * @return Pointer to bias data, or NULL if the op has no bias.
 */
static inline const void *tigris_op_bias(
    const tigris_plan_t *plan, const tigris_op_t *op)
{
    if (op->bias_idx == TIGRIS_NO_WEIGHT || !plan->weight_entries)
        return NULL;
    return tigris_weight_data(plan, &plan->weight_entries[op->bias_idx]);
}

/**
 * Get the data of an operator's constant operand listed by
 * TIGRIS_OP_ATTR_CONSTANTS.
 *
 * @param plan      Loaded plan.
 * @param op_index  Operator index.
 * @param k         Position among the operator's constant operands.
 * @return Pointer to the constant's data, or NULL if it is absent.
 */
static inline const void *tigris_op_constant(
    const tigris_plan_t *plan, uint16_t op_index, uint8_t k)
{
    uint8_t len = 0u;
    const uint8_t *list = tigris_op_attribute_data(plan, op_index, TIGRIS_OP_ATTR_CONSTANTS, &len);
    if (!list || (uint32_t)k * 2u + 2u > len || !plan->weight_entries)
        return NULL;
    uint16_t index = (uint16_t)((uint16_t)list[(uint32_t)k * 2u] |
                                (uint16_t)((uint16_t)list[(uint32_t)k * 2u + 1u] << 8));
    if (index == TIGRIS_NO_WEIGHT || index >= plan->header->num_weights)
        return NULL;
    return tigris_weight_data(plan, &plan->weight_entries[index]);
}

/**
 * Get the quantization parameter for a tensor.
 *
 * @param plan  Loaded plan.
 * @param t     Tensor descriptor.
 * @return Quant param pointer, or NULL if the tensor is not quantized.
 */
static inline const tigris_quant_param_t *tigris_tensor_quant(
    const tigris_plan_t *plan, const tigris_tensor_t *t)
{
    if (t->quant_param_idx == TIGRIS_NO_QUANT_PARAM || !plan->quant_params)
        return NULL;
    return &plan->quant_params[t->quant_param_idx];
}

/**
 * Get the index in quant_data where a quant param's page starts.
 *
 * quant_data is split into pages of TIGRIS_QUANT_PAGE_ELEMS elements, and
 * multiplier_off and shift_off count from the start of the param's page.
 *
 * @param qp  Quant param.
 * @return Element index of the page start.
 */
static inline uint32_t tigris_quant_page_start(const tigris_quant_param_t *qp)
{
    return (uint32_t)qp->_pad * TIGRIS_QUANT_PAGE_ELEMS;
}

/**
 * Get the requantization multipliers of a quant param.
 *
 * @param plan  Loaded plan.
 * @param qp    Quant param of that plan.
 * @return Pointer to qp->num_channels Q0.31 multipliers in quant_data.
 */
static inline const int32_t *tigris_quant_multipliers(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp)
{
    return plan->quant_data + tigris_quant_page_start(qp) + qp->multiplier_off;
}

/**
 * Get the requantization shifts of a quant param.
 *
 * @param plan  Loaded plan.
 * @param qp    Quant param of that plan.
 * @return Pointer to qp->num_channels shifts in quant_data.
 */
static inline const int32_t *tigris_quant_shifts(
    const tigris_plan_t *plan, const tigris_quant_param_t *qp)
{
    return plan->quant_data + tigris_quant_page_start(qp) + qp->shift_off;
}

/**
 * Compute extra fast-arena bytes needed for weight decompression.
 *
 * Standalone stages decompress one block at a time. A tiled chain keeps every
 * stage's block resident simultaneously, so its requirement is the aligned
 * sum across the chain. The returned value is the maximum of those execution
 * groups, aligned to TIGRIS_TENSOR_ALIGN by the runtime implementation.
 *
 * @param plan  Loaded plan.
 * @return Required overhead in bytes, 0 for an uncompressed plan, or
 *         UINT32_MAX if the required aligned sum cannot be represented.
 */
uint32_t tigris_weight_decompression_overhead(const tigris_plan_t *plan);

/**
 * Compute the core fast-arena capacity required by a loaded plan.
 *
 * This is the activation figure plus the simultaneous compressed-weight
 * reservation returned by tigris_weight_decompression_overhead(). The
 * activation figure is the plan's scheduled peak from schema 11 on, and its
 * budget before. A run of a schema 11 plan refuses a fast arena below this
 * with TIGRIS_EXEC_ERR_MEM before it starts. The result assumes the arena base
 * satisfies TIGRIS_TENSOR_ALIGN. Leading padding for an unaligned base and
 * backend-specific scratch/workspace are not included.
 *
 * @param plan  Loaded plan.
 * @return Required core arena capacity in bytes, or UINT32_MAX if plan/header
 *         is NULL or the sum cannot be represented.
 */
uint32_t tigris_fast_arena_required(const tigris_plan_t *plan);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_H */

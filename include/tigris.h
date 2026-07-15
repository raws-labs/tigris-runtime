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

#ifdef __cplusplus
extern "C" {
#endif

/* Keep in sync with tigris/src/tigris/emitters/binary/defs.py */

/* Format constants */

#define TIGRIS_MAGIC           0x53524754  /* "TGRS" in little-endian */
#define TIGRIS_MAGIC_BYTES     "TGRS"
#define TIGRIS_SCHEMA_VERSION  4
#define TIGRIS_SCHEMA_VERSION_V3 3
#define TIGRIS_SCHEMA_VERSION_V2 2
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
#define TIGRIS_SEC_MAX            12  /* one past the last valid section */

/* Typed payloads stored in TIGRIS_SEC_OP_ATTRIBUTES. */
#define TIGRIS_OP_ATTR_TRANSPOSE_PERM 1

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
    TIGRIS_OP_UNKNOWN           = 255,
} tigris_op_type_t;

/* Tensor flags */

#define TIGRIS_TENSOR_CONSTANT      0x01
#define TIGRIS_TENSOR_MODEL_INPUT   0x02
#define TIGRIS_TENSOR_MODEL_OUTPUT  0x04

/* Error codes */

/** Loader error codes returned by tigris_plan_load(). */
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
} tigris_error_t;

/* Sentinel: tile_plan_idx == 0xFFFF means no tile plan for this stage */
#define TIGRIS_NO_TILE_PLAN     0xFFFF

/* Sentinel: chain_id == 0xFFFF means stage is not part of a chain */
#define TIGRIS_NO_CHAIN         0xFFFF

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
    uint8_t     _pad;               /* 15: reserved */
} tigris_tensor_t;                  /* 16 bytes total */

/**
 * Spatial attributes - 18 bytes, embedded in tigris_op_t.
 * Zero for non-spatial ops (pointwise, reshape, etc.). pad/dilation are u16
 * (were u8) so deep dilated convs (dilation/pad up to 65535) don't overflow.
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
    uint8_t     stage;               /*  7: stage assignment */
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
    uint16_t    _reserved1;         /* 26 */
} tigris_stage_t;                   /* 28 bytes total */

/**
 * Tile plan - 24 bytes.
 */
typedef struct {
    uint8_t     tileable;           /*  0: 1=tileable, 0=not */
    uint8_t     _pad0;              /*  1 */
    uint16_t    tile_height;        /*  2 */
    uint16_t    num_tiles;          /*  4 */
    uint16_t    halo;               /*  6 */
    uint16_t    receptive_field;    /*  8 */
    uint16_t    original_height;    /* 10 */
    uint32_t    tiled_peak_bytes;   /* 12 */
    uint32_t    overhead_bytes;     /* 16 */
    uint32_t    _reserved;          /* 20 */
} tigris_tile_plan_t;               /* 24 bytes total */

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
    uint16_t    _pad;               /* 14: v3 quant-data page (v2 reserved) */
} tigris_quant_param_t;             /* 16 bytes total */

/**
 * Weight block entry - 20 bytes.
 * Describes one per-stage compressed weight block.
 */
typedef struct {
    uint16_t    stage_idx;          /*  0: stage index */
    uint16_t    first_weight_idx;   /*  2: first global weight index */
    uint16_t    num_weights;        /*  4: count of weight entries in block */
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

    /* Convenience: model I/O resolved from header */
    const uint16_t              *model_inputs;  /* [num_model_inputs] */
    const uint16_t              *model_outputs; /* [num_model_outputs] */
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
 * Get the tile plan for a stage.
 *
 * @param plan   Loaded plan.
 * @param stage  Stage descriptor.
 * @return Tile plan pointer, or NULL if the stage has no tile plan.
 */
static inline const tigris_tile_plan_t *tigris_stage_tile_plan(
    const tigris_plan_t *plan, const tigris_stage_t *stage)
{
    if (stage->tile_plan_idx == TIGRIS_NO_TILE_PLAN)
        return NULL;
    return &plan->tile_plans[stage->tile_plan_idx];
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

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_H */

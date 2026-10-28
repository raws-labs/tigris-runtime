/**
 * @file tigris_band.h
 * @brief Band geometry shared by the loader, the executor and the kernels.
 *
 * Internal to src/. Not installed and not part of the public API. The
 * operator classes are defined in tigris_loader.c, which holds the operator
 * contract table; everything else is defined in tigris_band.c.
 */
#ifndef TIGRIS_BAND_H
#define TIGRIS_BAND_H

#include "tigris.h"

#include <stdint.h>

/* Tensor helpers the loader and the band rules share. */
static inline int tigris_tensor_shapes_equal(
    const tigris_plan_t *plan,
    const tigris_tensor_t *a,
    const tigris_tensor_t *b)
{
    if (a->ndim != b->ndim)
        return 0;
    const int32_t *a_shape = tigris_tensor_shape(plan, a);
    const int32_t *b_shape = tigris_tensor_shape(plan, b);
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a_shape[i] != b_shape[i])
            return 0;
    }
    return 1;
}

static inline uint32_t tigris_tensor_elements(const tigris_tensor_t *tensor)
{
    return tensor->size_bytes / ((tensor->dtype == 1 || tensor->dtype == 6) ? sizeof(int32_t) : 1u);
}

/* How a height band treats an operator, read from its contract row. */
typedef enum {
    TIGRIS_HEIGHT_OP_REFUSED = 0,   /* no height band runs it */
    TIGRIS_HEIGHT_OP_POINTWISE = 1, /* shape-preserving and halo-free */
    TIGRIS_HEIGHT_OP_SPATIAL = 2,   /* a convolution or pool, with stride and halo */
    TIGRIS_HEIGHT_OP_RESAMPLE = 3   /* a Resize, banded from the full extents on its own stage */
} tigris_height_op_t;

/* Operator classes, from the operator contract table. */
int tigris_binary_op(uint8_t type);
int tigris_axis1_unary_pointwise_op(uint8_t type);
int tigris_axis1_binary_pointwise_op(uint8_t type);
tigris_height_op_t tigris_height_op(uint8_t type);
int tigris_height_spatial_op(uint8_t type);
int tigris_height_upsample_op(uint8_t type);
int tigris_row_tiling_op(uint8_t type);
int tigris_whole_operand_op(uint8_t type);

typedef struct {
    int32_t blocks;
    int32_t input_rows;
    int32_t input_width;
    int32_t output_rows;
    int32_t output_width;
    uint8_t input_axis;     /* the input axis whose rows the band takes */
} tigris_reshape_band_t;

int tigris_reshape_band(const tigris_plan_t *plan, const tigris_stage_t *stage,
                        tigris_reshape_band_t *view);
int tigris_pure_reinterpretation(const tigris_plan_t *plan, const tigris_tensor_t *in,
                                 const tigris_tensor_t *out);
int tigris_reshape_tile_valid(const tigris_tile_plan_t *tile,
                              const tigris_reshape_band_t *view);

/* Band modes 0 and 1 are the height and row views; TIGRIS_BAND_LEADING and the
 * modes after it (tigris.h) are the leading bands. */
uint8_t tigris_band_axis(const tigris_plan_t *plan, const tigris_tensor_t *tensor, int mode);
int32_t tigris_stage_leading_band(const tigris_plan_t *plan, const tigris_stage_t *stage);

/* Whether a reduction or movement preserves the existing row-band geometry. */
int tigris_op_independent_band(const tigris_plan_t *plan, const tigris_op_t *op,
                                uint16_t op_index, int row_tiled);

/* The rules the executor bands a stage by, which the tile facts also derive from. */
int tigris_stage_spatial_op(const tigris_plan_t *plan, const tigris_stage_t *stage);
int tigris_stage_stripe_contract(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                 uint8_t rank, uint8_t axis);
/* The executor that holds a stage's band, which settles what moves it. */
typedef enum {
    TIGRIS_BAND_HELD_BY_CHAIN = 0,
    TIGRIS_BAND_HELD_BY_STRIPE = 1,
    TIGRIS_BAND_HELD_BY_HW_TILE = 2
} tigris_band_holder_t;

int tigris_stage_reads_its_band(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                tigris_band_holder_t holder);
tigris_error_t tigris_stage_stripe_banding(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                           tigris_band_holder_t holder);
uint16_t tigris_stage_band_input(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                 tigris_band_holder_t holder);
uint16_t tigris_stage_band_output(const tigris_plan_t *plan, const tigris_stage_t *stage);
int tigris_tensor_row_view(const tigris_plan_t *plan, const tigris_tensor_t *t,
                           int32_t *batch, int32_t *rows, int32_t *cols);
/* The row band rule; `loader` selects the loader's wider form. */
int tigris_stage_row_band(const tigris_plan_t *plan, const tigris_stage_t *stage, int loader,
                          int32_t *out_batch, int32_t *out_rows);
int tigris_stage_row_tiled(const tigris_plan_t *plan, const tigris_stage_t *stage,
                           int32_t *out_batch, int32_t *out_rows);
int tigris_whole_band_operand(const tigris_plan_t *plan, const tigris_stage_t *stage,
                              uint16_t tidx);
int tigris_stage_input_driven_reduction(const tigris_plan_t *plan, const tigris_stage_t *stage);
int tigris_transpose_extents(const tigris_plan_t *plan, uint16_t op_idx, uint16_t in_idx,
                             int32_t *outer, int32_t *rows, int32_t *cols);
int tigris_stage_tiled_transpose(const tigris_plan_t *plan, const tigris_stage_t *stage);

#endif /* TIGRIS_BAND_H */

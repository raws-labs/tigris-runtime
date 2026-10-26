/**
 * @file tigris_loader.h
 * @brief Plan loader API - parse a .tgrs buffer into a tigris_plan_t.
 */

#ifndef TIGRIS_LOADER_H
#define TIGRIS_LOADER_H

#include "tigris.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a binary plan from a buffer.
 *
 * Zero-copy, zero-alloc: all pointers in out_plan point directly into buf.
 * The caller must keep buf alive for the lifetime of out_plan.
 *
 * ALIGNMENT CONTRACT: buf must be aligned to 16 bytes when the plan's weights
 * are read in place (XIP / non-compressed) and consumed by the optimized
 * CMSIS-NN kernels. The compiler aligns every weight blob to 16 bytes
 * *relative to the file base*, whatever TIGRIS_TENSOR_ALIGN is, so an
 * unaligned buf makes the weight pointers unaligned and the opt kernels fault
 * (LDRD/SIMD). bin2c emits the embedded blob aligned(16); mmap/flash bases
 * satisfy this; a malloc'd host buffer should use aligned_alloc(16, ...). The
 * reference (s8/f32) kernels do not require this; only the opt backends do.
 *
 * Structural bounds, cross-references, tensor metadata and the built-in
 * operator semantics are validated for every supported schema.
 *
 * @param buf       Pointer to the loaded .tgrs file contents.
 * @param buf_len   Size of the buffer in bytes.
 * On every failure, out_plan is cleared and contains no pointers into buf.
 *
 * @param out_plan  Output: parsed plan handle, or an empty handle on failure.
 * @return TIGRIS_OK on success, negative error code on failure.
 */
tigris_error_t tigris_plan_load(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan);

/**
 * Load a plan and report what it asks of the build.
 *
 * Identical to tigris_plan_load, except that *out_required* is filled with
 * the limits the plan needs, so a caller can say which build would run it
 * rather than only that this one will not. On TIGRIS_ERR_PLAN_LIMITS it holds
 * everything read up to the limit this build cannot meet, which is the one to
 * raise; raising it and loading again reports the next, if there is one. Pass
 * NULL to ignore.
 */
tigris_error_t tigris_plan_load_ex(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan,
    tigris_plan_limits_t *out_required);

/** Data dtype of a loaded plan: int8 (3) when any activation is int8, else
 * float32 (1). Float32 tensors in an int8 plan are only those operator dtype
 * slots state, which loading checks. */
uint8_t tigris_plan_data_dtype(const tigris_plan_t *plan);

/** How the executor bands a standalone stage. */
typedef enum {
    TIGRIS_BAND_KIND_NONE = 0,      /**< no band kind */
    TIGRIS_BAND_KIND_RESHAPE = 1,   /**< a Reshape whose rows map onto the output's */
    TIGRIS_BAND_KIND_LEADING = 2,   /**< one operator banded along an axis it keeps independent */
    TIGRIS_BAND_KIND_ROWS = 3,      /**< a matrix pipeline banded along its rows */
    TIGRIS_BAND_KIND_TRANSPOSE = 4, /**< a transpose banded along an axis group it swaps */
    TIGRIS_BAND_KIND_BY_INPUT = 5,  /**< a global reduction walked along its input */
    TIGRIS_BAND_KIND_HW = 6,        /**< height and width bands of a 2D record */
    TIGRIS_BAND_KIND_STRIPE = 7     /**< the height or length stripe */
} tigris_band_kind_t;

/** The extent a band kind divides. */
typedef struct {
    int32_t batch;  /**< ROWS: the matrices each band repeats over, otherwise 0 */
    int32_t rows;   /**< RESHAPE, LEADING and ROWS: the rows the bands divide, otherwise 0 */
    int32_t mode;   /**< LEADING: the band mode, TIGRIS_BAND_LEADING or a later one, otherwise 0 */
} tigris_stage_band_t;

/**
 * How the executor bands a standalone stage. From schema 11 a tileable record
 * states the kind and a leading band's mode; the stated kind is returned when
 * the stage meets that kind's conditions, and TIGRIS_BAND_KIND_NONE, which
 * the loader refuses, when it does not. A stage without a tileable record has
 * no kind. Below schema 11 the kind is derived in the order the executor
 * decides: a Reshape band, then a leading band (both only under a tileable
 * record), a row band, a banded transpose, a global reduction by input, a 2D
 * record, and otherwise the stripe. Whether the stage is banded at all also
 * depends on its record and the arena, which the run checks. Chain members
 * and control-flow stages run on their own paths and are not classified here.
 *
 * @param band  Output, may be NULL: the extent the kind divides.
 * @return The kind, or TIGRIS_BAND_KIND_NONE for a NULL plan or stage, a
 *         schema 11 stage without a tileable record, or a stated kind the
 *         stage does not meet.
 */
tigris_band_kind_t tigris_stage_band_kind(const tigris_plan_t *plan, const tigris_stage_t *stage,
                                          tigris_stage_band_t *band);

/**
 * Derive what a stage's tile record states about its tiling (schema 11+),
 * by the rules the executor bands the stage with: the band axis, the extent
 * the bands divide, the band, the tile count, the halo, and the band byte of
 * the kind the stage is banded by (tigris_stage_band_kind). A height stripe
 * divides its output rows into the stated output band, and num_tiles is the
 * number of bands the executor runs. A layout conversion sizes its band in
 * the run, so its tile_extent is the record's, bounded by the extent. An
 * untileable record yields all zero.
 * The loader returns TIGRIS_ERR_TILE_FACTS for a record that disagrees.
 *
 * @return 1 when derived, 0 when the stage's tiling has no such description.
 */
int tigris_stage_tile_facts(const tigris_plan_t *plan, const tigris_stage_t *stage,
                            tigris_tile_facts_t *out);

/** The limits this build was compiled to allow. */
void tigris_build_limits(tigris_plan_limits_t *out);

/**
 * Return a human-readable string for a tigris_error_t code.
 *
 * @param err  Error code.
 * @return Static string describing the error.
 */
const char *tigris_error_str(tigris_error_t err);

#ifdef __cplusplus
}
#endif

#endif /* TIGRIS_LOADER_H */

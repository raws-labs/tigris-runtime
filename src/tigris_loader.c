/**
 * @file tigris_loader.c
 * @brief Zero-alloc binary plan parser.
 *
 * Validates the file header, walks the section directory, and sets
 * pointers into the buffer. Never allocates memory.
 */

#include "tigris_loader.h"

#include <string.h>

/* Helpers */

/* Keep these in sync with the executor's fixed-size working arrays. */
#define LOADER_MAX_STAGE_INPUTS       16u
#define LOADER_MAX_STAGE_OUTPUTS      16u
#define LOADER_MAX_CHAIN_STAGES       16u
#define LOADER_MAX_CHAIN_SPATIAL_OPS   8u
#define LOADER_MAX_TENSORS           512u

/** Check that [offset, offset+size) fits within buf_len. */
static inline int bounds_ok(uint32_t offset, uint32_t size, uint32_t buf_len)
{
    return (offset <= buf_len) && (size <= buf_len - offset);
}

/** Check an element range without overflowing offset + count. */
static inline int elements_ok(uint32_t offset, uint32_t count, uint32_t total)
{
    return offset <= total && count <= total - offset;
}

/** Check that an offset names a terminated string inside the string section. */
static int string_ok(const char *strings, uint32_t strings_len, uint32_t offset)
{
    if (!strings || offset >= strings_len)
        return 0;
    return memchr(strings + offset, '\0', strings_len - offset) != NULL;
}

/* Public API */

tigris_error_t tigris_plan_load(
    const uint8_t *buf, uint32_t buf_len, tigris_plan_t *out_plan)
{
    if (!buf || !out_plan)
        return TIGRIS_ERR_NULL;

    /* Verify little-endian platform (plan format assumes LE) */
    {
        const uint32_t endian_test = 1;
        if (*(const uint8_t *)&endian_test != 1)
            return TIGRIS_ERR_ENDIAN;
    }

    memset(out_plan, 0, sizeof(*out_plan));

    /* Header validation */

    if (buf_len < sizeof(tigris_file_header_t))
        return TIGRIS_ERR_TOO_SMALL;

    const tigris_file_header_t *hdr = (const tigris_file_header_t *)buf;

    if (memcmp(hdr->magic, TIGRIS_MAGIC_BYTES, 4) != 0)
        return TIGRIS_ERR_BAD_MAGIC;

    if (hdr->version != TIGRIS_SCHEMA_VERSION &&
        hdr->version != TIGRIS_SCHEMA_VERSION_V2)
        return TIGRIS_ERR_BAD_VERSION;

    if (hdr->file_size != buf_len)
        return TIGRIS_ERR_BAD_SIZE;

    out_plan->header = hdr;

    /* Section directory */

    uint32_t sec_off = hdr->section_dir_off;
    if (!bounds_ok(sec_off, sizeof(tigris_section_entry_t), buf_len))
        return TIGRIS_ERR_BAD_SECTION;

    /* Track which sections we found via offsets into buf (0 = not found) */
    uint32_t section_offsets[TIGRIS_SEC_MAX];
    uint32_t section_ends[TIGRIS_SEC_MAX];
    uint8_t section_seen[TIGRIS_SEC_MAX];
    memset(section_offsets, 0, sizeof(section_offsets));
    memset(section_ends, 0, sizeof(section_ends));
    memset(section_seen, 0, sizeof(section_seen));

    int found_sentinel = 0;
    uint32_t sec_cursor = sec_off;
    uint32_t previous_type = 0;
    uint32_t previous_offset = 0;
    while (bounds_ok(sec_cursor, sizeof(tigris_section_entry_t), buf_len)) {
        const tigris_section_entry_t *sec =
            (const tigris_section_entry_t *)(buf + sec_cursor);
        if (sec->type == 0)
        {
            found_sentinel = 1;
            break;
        }

        if (sec->type >= TIGRIS_SEC_MAX)
            return TIGRIS_ERR_BAD_SECTION;

        if (section_seen[sec->type])
            return TIGRIS_ERR_BAD_SECTION;

        /* The writer emits an ordered directory and non-overlapping section
         * starts. Empty sections may share an offset with the next section. */
        if (sec->type <= previous_type || sec->offset < previous_offset ||
            sec->offset > buf_len)
            return TIGRIS_ERR_BAD_SECTION;

        section_seen[sec->type] = 1;
        section_offsets[sec->type] = sec->offset;
        previous_type = sec->type;
        previous_offset = sec->offset;
        sec_cursor += sizeof(tigris_section_entry_t);
    }

    if (!found_sentinel)
        return TIGRIS_ERR_BAD_SECTION;

    /* No section may overlap the parsed directory. Derive each section's
     * exclusive end from the next distinct section start. */
    uint32_t dir_end = sec_cursor + sizeof(tigris_section_entry_t);
    for (uint32_t i = 1; i < TIGRIS_SEC_MAX; i++) {
        if (!section_seen[i])
            continue;
        if (section_offsets[i] < dir_end)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t end = buf_len;
        for (uint32_t j = 1; j < TIGRIS_SEC_MAX; j++) {
            if (section_seen[j] && section_offsets[j] > section_offsets[i] &&
                section_offsets[j] < end)
                end = section_offsets[j];
        }
        section_ends[i] = end;
    }

    /* Required sections */

    if (!section_offsets[TIGRIS_SEC_TENSORS])     return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_OPS])         return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_INDEX_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_SHAPE_POOL])  return TIGRIS_ERR_MISSING_SEC;
    if (!section_offsets[TIGRIS_SEC_STRINGS])     return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_stages && !section_offsets[TIGRIS_SEC_STAGES])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_tile_plans && !section_offsets[TIGRIS_SEC_TILE_PLANS])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_weights && !section_offsets[TIGRIS_SEC_WEIGHTS])
        return TIGRIS_ERR_MISSING_SEC;
    if (hdr->num_quant_params && !section_offsets[TIGRIS_SEC_QUANT_PARAMS])
        return TIGRIS_ERR_MISSING_SEC;

    /* Set pointers */

    /* Tensors */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_TENSORS];
        uint32_t need = (uint32_t)hdr->num_tensors * sizeof(tigris_tensor_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_TENSORS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->tensors = (const tigris_tensor_t *)(buf + off);
    }

    /* Ops */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_OPS];
        uint32_t need = (uint32_t)hdr->num_ops * sizeof(tigris_op_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_OPS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->ops = (const tigris_op_t *)(buf + off);
    }

    /* Stages (optional - 0 stages is valid for un-partitioned graphs) */
    if (section_offsets[TIGRIS_SEC_STAGES] && hdr->num_stages > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_STAGES];
        uint32_t need = (uint32_t)hdr->num_stages * sizeof(tigris_stage_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_STAGES]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->stages = (const tigris_stage_t *)(buf + off);
    }

    /* Tile plans (optional) */
    if (section_offsets[TIGRIS_SEC_TILE_PLANS] && hdr->num_tile_plans > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_TILE_PLANS];
        uint32_t need = (uint32_t)hdr->num_tile_plans * sizeof(tigris_tile_plan_t);
        if (!bounds_ok(off, need, section_ends[TIGRIS_SEC_TILE_PLANS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->tile_plans = (const tigris_tile_plan_t *)(buf + off);
    }

    /* Weights (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHTS] && hdr->num_weights > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights * sizeof(tigris_weight_entry_t);
        if (!bounds_ok(off, entries_size, section_ends[TIGRIS_SEC_WEIGHTS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->weight_entries = (const tigris_weight_entry_t *)(buf + off);
        /* weight_blob set below only if no compressed blocks */
    }

    /* Weight blocks - compressed per-stage weight data (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS]) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        if (!bounds_ok(off, 4, section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t num_blocks, compression;
        memcpy(&num_blocks, buf + off, 2);
        memcpy(&compression, buf + off + 2, 2);
        out_plan->num_weight_blocks = num_blocks;
        out_plan->weight_compression = compression;
        uint32_t entries_size = (uint32_t)num_blocks * sizeof(tigris_weight_block_t);
        if (!bounds_ok(off + 4, entries_size, section_ends[TIGRIS_SEC_WEIGHT_BLOCKS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->weight_blocks = (const tigris_weight_block_t *)(buf + off + 4);
        out_plan->weight_blocks_data = buf + off + 4 + entries_size;
    }

    /* Set weight_blob for XIP only when no compressed blocks */
    if (out_plan->weight_entries && !out_plan->weight_blocks) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights * sizeof(tigris_weight_entry_t);
        out_plan->weight_blob = buf + off + entries_size;
    }

    /* Quant params (optional) */
    if (section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t off = section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        /* First 4 bytes: uint16_t num_quant_params + quant-data length (v2)
         * or page count (v3). */
        if (!bounds_ok(off, 4, section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t nqp, qd_field;
        memcpy(&nqp, buf + off, 2);
        memcpy(&qd_field, buf + off + 2, 2);
        out_plan->num_quant_params = nqp;
        uint32_t entries_size = (uint32_t)nqp * sizeof(tigris_quant_param_t);
        uint32_t data_size;
        if (hdr->version == TIGRIS_SCHEMA_VERSION_V2)
            data_size = (uint32_t)qd_field * sizeof(int32_t);
        else {
            if (section_ends[TIGRIS_SEC_QUANT_PARAMS] < off + 4u + entries_size)
                return TIGRIS_ERR_BAD_SECTION;
            data_size = section_ends[TIGRIS_SEC_QUANT_PARAMS] - off - 4u - entries_size;
            if ((data_size % sizeof(int32_t)) != 0)
                return TIGRIS_ERR_BAD_SECTION;
            uint32_t physical_pages = (data_size / sizeof(int32_t) +
                                       TIGRIS_QUANT_PAGE_ELEMS - 1u) /
                                      TIGRIS_QUANT_PAGE_ELEMS;
            /* The next aligned section can contribute up to 15 padding bytes
             * to this section span. It may therefore extend an otherwise
             * page-aligned pool into one physical page without changing the
             * semantic v3 page count recorded by the compiler. */
            if (qd_field == 0 || physical_pages < qd_field ||
                physical_pages > (uint32_t)qd_field + 1u)
                return TIGRIS_ERR_BAD_SECTION;
        }
        if (!bounds_ok(off + 4, entries_size + data_size,
                       section_ends[TIGRIS_SEC_QUANT_PARAMS]))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->quant_params = (const tigris_quant_param_t *)(buf + off + 4);
        if (data_size > 0)
            out_plan->quant_data = (const int32_t *)(buf + off + 4 + entries_size);
    }

    /* Index pool */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_INDEX_POOL];
        out_plan->index_pool = (const uint16_t *)(buf + off);
    }

    /* Shape pool */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_SHAPE_POOL];
        out_plan->shape_pool = (const int32_t *)(buf + off);
    }

    /* String table */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_STRINGS];
        out_plan->strings = (const char *)(buf + off);
    }

    /* Validate every table span and cross-reference before exposing the plan
     * to the executor. Section boundaries above ensure these counts cannot
     * borrow bytes from the next table. */
    uint32_t index_bytes = section_ends[TIGRIS_SEC_INDEX_POOL] -
                           section_offsets[TIGRIS_SEC_INDEX_POOL];
    uint32_t shape_bytes = section_ends[TIGRIS_SEC_SHAPE_POOL] -
                           section_offsets[TIGRIS_SEC_SHAPE_POOL];
    uint32_t strings_len = section_ends[TIGRIS_SEC_STRINGS] -
                           section_offsets[TIGRIS_SEC_STRINGS];
    if (((index_bytes % sizeof(uint16_t)) != 0 &&
         (hdr->num_ops || hdr->num_stages || hdr->num_model_inputs ||
          hdr->num_model_outputs)) ||
        ((shape_bytes % sizeof(int32_t)) != 0 && hdr->num_tensors) ||
        strings_len == 0)
        return TIGRIS_ERR_BAD_SECTION;
    uint32_t index_count = index_bytes / sizeof(uint16_t);
    uint32_t shape_count = shape_bytes / sizeof(int32_t);

    if (!string_ok(out_plan->strings, strings_len, hdr->model_name_str))
        return TIGRIS_ERR_BAD_SECTION;

    uint32_t model_io_count = (uint32_t)hdr->num_model_inputs +
                              (uint32_t)hdr->num_model_outputs;
    if (!elements_ok(hdr->model_io_off, model_io_count, index_count))
        return TIGRIS_ERR_BAD_SECTION;
    out_plan->model_inputs  = out_plan->index_pool + hdr->model_io_off;
    out_plan->model_outputs = out_plan->model_inputs + hdr->num_model_inputs;
    for (uint32_t i = 0; i < model_io_count; i++) {
        if (out_plan->model_inputs[i] >= hdr->num_tensors)
            return TIGRIS_ERR_BAD_SECTION;
    }

    if (hdr->num_tensors > LOADER_MAX_TENSORS)
        return TIGRIS_ERR_PLAN_LIMITS;

    for (uint16_t i = 0; i < hdr->num_tensors; i++) {
        const tigris_tensor_t *tensor = &out_plan->tensors[i];
        if (!string_ok(out_plan->strings, strings_len, tensor->name_str) ||
            !elements_ok(tensor->shape_off, tensor->ndim, shape_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (tensor->quant_param_idx != TIGRIS_NO_QUANT_PARAM &&
            tensor->quant_param_idx >= hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
        uint32_t elements = 1;
        for (uint8_t dim = 0; dim < tensor->ndim; dim++) {
            int32_t value = out_plan->shape_pool[tensor->shape_off + dim];
            if (value <= 0 || (uint32_t)value > UINT32_MAX / elements)
                return TIGRIS_ERR_BAD_SECTION;
            elements *= (uint32_t)value;
        }
        uint32_t element_size;
        if (tensor->dtype == 1)
            element_size = sizeof(float);
        else if (tensor->dtype == 3)
            element_size = sizeof(int8_t);
        else
            return TIGRIS_ERR_BAD_SECTION;
        if (elements > UINT32_MAX / element_size ||
            tensor->size_bytes != elements * element_size)
            return TIGRIS_ERR_BAD_SECTION;
    }

    for (uint16_t i = 0; i < hdr->num_ops; i++) {
        const tigris_op_t *op = &out_plan->ops[i];
        if (!string_ok(out_plan->strings, strings_len, op->name_str) ||
            !elements_ok(op->inputs_off, op->num_inputs, index_count) ||
            !elements_ok(op->outputs_off, op->num_outputs, index_count))
            return TIGRIS_ERR_BAD_SECTION;
        if (hdr->num_stages && op->stage >= hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        if ((op->weight_idx != TIGRIS_NO_WEIGHT &&
             op->weight_idx >= hdr->num_weights) ||
            (op->bias_idx != TIGRIS_NO_WEIGHT &&
             op->bias_idx >= hdr->num_weights))
            return TIGRIS_ERR_BAD_SECTION;
        for (uint8_t j = 0; j < op->num_inputs; j++) {
            if (out_plan->index_pool[op->inputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
        for (uint8_t j = 0; j < op->num_outputs; j++) {
            if (out_plan->index_pool[op->outputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (out_plan->weight_entries && !out_plan->weight_blocks) {
        uint32_t weights_off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights *
                                sizeof(tigris_weight_entry_t);
        uint32_t blob_len = section_ends[TIGRIS_SEC_WEIGHTS] -
                            weights_off - entries_size;
        for (uint16_t i = 0; i < hdr->num_weights; i++) {
            const tigris_weight_entry_t *weight = &out_plan->weight_entries[i];
            if (!string_ok(out_plan->strings, strings_len, weight->name_str) ||
                !bounds_ok(weight->offset, weight->size_bytes, blob_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (section_offsets[TIGRIS_SEC_QUANT_PARAMS]) {
        uint32_t qp_off = section_offsets[TIGRIS_SEC_QUANT_PARAMS];
        uint16_t nqp, qd_field;
        memcpy(&nqp, buf + qp_off, sizeof(nqp));
        memcpy(&qd_field, buf + qp_off + sizeof(nqp), sizeof(qd_field));
        uint32_t qd_len = hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? qd_field :
            (section_ends[TIGRIS_SEC_QUANT_PARAMS] - qp_off - 4u -
             (uint32_t)nqp * sizeof(tigris_quant_param_t)) / sizeof(int32_t);
        if (nqp != hdr->num_quant_params)
            return TIGRIS_ERR_BAD_SECTION;
        for (uint16_t i = 0; i < nqp; i++) {
            const tigris_quant_param_t *qp = &out_plan->quant_params[i];
            uint32_t page = hdr->version == TIGRIS_SCHEMA_VERSION_V2 ? 0u : qp->_pad;
            uint32_t moff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->multiplier_off;
            uint32_t soff = page * TIGRIS_QUANT_PAGE_ELEMS + qp->shift_off;
            if (qp->num_channels == 0 ||
                (hdr->version != TIGRIS_SCHEMA_VERSION_V2 &&
                 (page >= qd_field ||
                  (uint32_t)qp->multiplier_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS ||
                  (uint32_t)qp->shift_off + qp->num_channels > TIGRIS_QUANT_PAGE_ELEMS)) ||
                !elements_ok(moff, qp->num_channels, qd_len) ||
                !elements_ok(soff, qp->num_channels, qd_len))
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    if (out_plan->weight_blocks) {
        uint32_t wb_off = section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        uint32_t block_entries = (uint32_t)out_plan->num_weight_blocks *
                                 sizeof(tigris_weight_block_t);
        uint32_t blobs_off = wb_off + 4u + block_entries;
        uint32_t blobs_len = section_ends[TIGRIS_SEC_WEIGHT_BLOCKS] - blobs_off;
        if (out_plan->weight_compression != TIGRIS_COMPRESS_NONE &&
            out_plan->weight_compression != TIGRIS_COMPRESS_LZ4)
            return TIGRIS_ERR_BAD_SECTION;
        if (out_plan->num_weight_blocks == 0 || hdr->num_weights == 0 ||
            out_plan->num_weight_blocks > hdr->num_stages)
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t previous_stage = TIGRIS_NO_CHAIN;
        for (uint16_t i = 0; i < out_plan->num_weight_blocks; i++) {
            const tigris_weight_block_t *block = &out_plan->weight_blocks[i];
            if (block->stage_idx >= hdr->num_stages ||
                (i > 0 && block->stage_idx <= previous_stage) ||
                !elements_ok(block->first_weight_idx, block->num_weights,
                             hdr->num_weights) ||
                block->num_weights == 0 || block->compressed_size == 0 ||
                block->uncompressed_size == 0 ||
                !bounds_ok(block->blob_offset, block->compressed_size, blobs_len) ||
                (out_plan->weight_compression == TIGRIS_COMPRESS_NONE &&
                 block->compressed_size != block->uncompressed_size))
                return TIGRIS_ERR_BAD_SECTION;
            previous_stage = block->stage_idx;
        }
    }

    /* Plan limits validation */

    uint16_t next_scheduled_op = 0;
    for (uint16_t i = 0; i < hdr->num_stages; i++) {
        const tigris_stage_t *stage = &out_plan->stages[i];
        if (!elements_ok(stage->ops_off, stage->ops_count, index_count) ||
            !elements_ok(stage->inputs_off, stage->inputs_count, index_count) ||
            !elements_ok(stage->outputs_off, stage->outputs_count, index_count) ||
            (stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN &&
             stage->tile_plan_idx >= hdr->num_tile_plans) ||
            (stage->chain_len == 0 && stage->chain_id != TIGRIS_NO_CHAIN) ||
            (stage->chain_len > 0 && stage->chain_id == TIGRIS_NO_CHAIN))
            return TIGRIS_ERR_BAD_SECTION;
        if (stage->inputs_count > LOADER_MAX_STAGE_INPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->outputs_count > LOADER_MAX_STAGE_OUTPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->chain_len > LOADER_MAX_CHAIN_STAGES)
            return TIGRIS_ERR_PLAN_LIMITS;

        /* A chain is represented redundantly on every member.  The executor
         * only starts it at its declared head and skips the other members, so
         * accepting an incomplete or displaced chain silently drops work. */
        if (stage->chain_len > 0) {
            if (stage->chain_len < 2 || stage->chain_id > i ||
                stage->chain_len > hdr->num_stages - stage->chain_id)
                return TIGRIS_ERR_BAD_SECTION;
            const tigris_stage_t *head = &out_plan->stages[stage->chain_id];
            if (head->chain_id != stage->chain_id ||
                head->chain_len != stage->chain_len ||
                head->chain_tile_h == 0 ||
                stage->tile_plan_idx != TIGRIS_NO_TILE_PLAN ||
                (i != stage->chain_id && stage->chain_tile_h != 0))
                return TIGRIS_ERR_BAD_SECTION;
            if (i == stage->chain_id) {
                for (uint16_t member = i + 1u;
                     member < i + stage->chain_len; member++) {
                    const tigris_stage_t *next = &out_plan->stages[member];
                    if (next->chain_id != i ||
                        next->chain_len != stage->chain_len ||
                        next->chain_tile_h != 0 ||
                        next->tile_plan_idx != TIGRIS_NO_TILE_PLAN)
                        return TIGRIS_ERR_BAD_SECTION;
                }
            }
        } else if (stage->chain_tile_h != 0) {
            return TIGRIS_ERR_BAD_SECTION;
        }

        for (uint16_t j = 0; j < stage->ops_count; j++) {
            uint16_t op_idx = out_plan->index_pool[stage->ops_off + j];
            if (op_idx >= hdr->num_ops)
                return TIGRIS_ERR_BAD_SECTION;
            /* The binary plan contract is a forward, partitioned schedule.
             * Enforcing it here prevents a malformed stage list from silently
             * omitting work or running the same operation more than once. */
            if (op_idx != next_scheduled_op)
                return TIGRIS_ERR_BAD_SECTION;
            next_scheduled_op++;
        }
        for (uint16_t j = 0; j < stage->inputs_count; j++) {
            if (out_plan->index_pool[stage->inputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }
        for (uint16_t j = 0; j < stage->outputs_count; j++) {
            if (out_plan->index_pool[stage->outputs_off + j] >= hdr->num_tensors)
                return TIGRIS_ERR_BAD_SECTION;
        }

        if (stage->chain_len >= 2) {
            if (stage->chain_id == TIGRIS_NO_CHAIN ||
                stage->chain_id >= hdr->num_stages ||
                stage->chain_len > hdr->num_stages - stage->chain_id)
                return TIGRIS_ERR_BAD_SECTION;

            /* Chain execution stores spatial-op metadata in a fixed [8] array.
             * Validate only the stage-op slice needed for that limit here; full
             * plan cross-reference validation belongs to the hardening pass. */
            uint32_t pool_off = section_offsets[TIGRIS_SEC_INDEX_POOL];
            uint32_t prefix = (uint32_t)stage->ops_off * sizeof(uint16_t);
            uint32_t bytes = (uint32_t)stage->ops_count * sizeof(uint16_t);
            if (!bounds_ok(pool_off, prefix, buf_len) ||
                !bounds_ok(pool_off + prefix, bytes, buf_len))
                return TIGRIS_ERR_BAD_SECTION;

            const uint8_t *op_indices = buf + pool_off + prefix;
            uint32_t spatial_count = 0;
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                uint16_t op_idx;
                memcpy(&op_idx, op_indices + (uint32_t)j * sizeof(op_idx),
                       sizeof(op_idx));
                if (op_idx >= hdr->num_ops)
                    return TIGRIS_ERR_BAD_SECTION;

                uint8_t op_type = out_plan->ops[op_idx].op_type;
                if (op_type == TIGRIS_OP_CONV || op_type == TIGRIS_OP_DEPTHWISE) {
                    spatial_count++;
                    if (spatial_count > LOADER_MAX_CHAIN_SPATIAL_OPS)
                        return TIGRIS_ERR_PLAN_LIMITS;
                }
            }
        }
    }

    if (next_scheduled_op != hdr->num_ops)
        return TIGRIS_ERR_BAD_SECTION;

    /* A compressed stage redirects weight_entry offsets into its own decoded
     * block.  Check that every referenced weight/bias is present in that
     * block before any executor or kernel can dereference the offset. */
    if (out_plan->weight_blocks) {
        uint16_t block_cursor = 0;
        for (uint16_t s = 0; s < hdr->num_stages; s++) {
            const tigris_weight_block_t *block = NULL;
            if (block_cursor < out_plan->num_weight_blocks &&
                out_plan->weight_blocks[block_cursor].stage_idx == s) {
                block = &out_plan->weight_blocks[block_cursor++];
            }

            const tigris_stage_t *stage = &out_plan->stages[s];
            for (uint16_t j = 0; j < stage->ops_count; j++) {
                const tigris_op_t *op =
                    &out_plan->ops[out_plan->index_pool[stage->ops_off + j]];
                uint16_t refs[2] = {op->weight_idx, op->bias_idx};
                for (uint8_t r = 0; r < 2; r++) {
                    if (refs[r] == TIGRIS_NO_WEIGHT)
                        continue;
                    if (!block ||
                        !bounds_ok(out_plan->weight_entries[refs[r]].offset,
                                   out_plan->weight_entries[refs[r]].size_bytes,
                                   block->uncompressed_size))
                        return TIGRIS_ERR_BAD_SECTION;
                }
            }
        }
    }

    return TIGRIS_OK;
}

const char *tigris_error_str(tigris_error_t err)
{
    switch (err) {
        case TIGRIS_OK:             return "OK";
        case TIGRIS_ERR_NULL:       return "null pointer argument";
        case TIGRIS_ERR_TOO_SMALL:  return "buffer too small for header";
        case TIGRIS_ERR_BAD_MAGIC:  return "bad magic (expected TGRS)";
        case TIGRIS_ERR_BAD_VERSION:return "unsupported plan version";
        case TIGRIS_ERR_BAD_SIZE:   return "file_size field mismatch";
        case TIGRIS_ERR_BAD_SECTION:return "section offset out of bounds";
        case TIGRIS_ERR_MISSING_SEC:return "required section missing";
        case TIGRIS_ERR_ENDIAN:     return "platform is not little-endian";
        case TIGRIS_ERR_PLAN_LIMITS:return "plan exceeds executor limits";
        default:                    return "unknown error";
    }
}

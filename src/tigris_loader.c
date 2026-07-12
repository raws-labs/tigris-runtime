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

    if (hdr->version != TIGRIS_SCHEMA_VERSION)
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
    uint8_t section_seen[TIGRIS_SEC_MAX];
    memset(section_offsets, 0, sizeof(section_offsets));
    memset(section_seen, 0, sizeof(section_seen));

    int found_sentinel = 0;
    uint32_t sec_cursor = sec_off;
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

        if (sec->offset > buf_len)
            return TIGRIS_ERR_BAD_SECTION;

        section_seen[sec->type] = 1;
        section_offsets[sec->type] = sec->offset;
        sec_cursor += sizeof(tigris_section_entry_t);
    }

    if (!found_sentinel)
        return TIGRIS_ERR_BAD_SECTION;

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
        if (!bounds_ok(off, need, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->tensors = (const tigris_tensor_t *)(buf + off);
    }

    /* Ops */
    {
        uint32_t off = section_offsets[TIGRIS_SEC_OPS];
        uint32_t need = (uint32_t)hdr->num_ops * sizeof(tigris_op_t);
        if (!bounds_ok(off, need, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->ops = (const tigris_op_t *)(buf + off);
    }

    /* Stages (optional - 0 stages is valid for un-partitioned graphs) */
    if (section_offsets[TIGRIS_SEC_STAGES] && hdr->num_stages > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_STAGES];
        uint32_t need = (uint32_t)hdr->num_stages * sizeof(tigris_stage_t);
        if (!bounds_ok(off, need, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->stages = (const tigris_stage_t *)(buf + off);
    }

    /* Tile plans (optional) */
    if (section_offsets[TIGRIS_SEC_TILE_PLANS] && hdr->num_tile_plans > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_TILE_PLANS];
        uint32_t need = (uint32_t)hdr->num_tile_plans * sizeof(tigris_tile_plan_t);
        if (!bounds_ok(off, need, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->tile_plans = (const tigris_tile_plan_t *)(buf + off);
    }

    /* Weights (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHTS] && hdr->num_weights > 0) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHTS];
        uint32_t entries_size = (uint32_t)hdr->num_weights * sizeof(tigris_weight_entry_t);
        if (!bounds_ok(off, entries_size, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->weight_entries = (const tigris_weight_entry_t *)(buf + off);
        /* weight_blob set below only if no compressed blocks */
    }

    /* Weight blocks - compressed per-stage weight data (optional) */
    if (section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS]) {
        uint32_t off = section_offsets[TIGRIS_SEC_WEIGHT_BLOCKS];
        if (!bounds_ok(off, 4, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t num_blocks, compression;
        memcpy(&num_blocks, buf + off, 2);
        memcpy(&compression, buf + off + 2, 2);
        out_plan->num_weight_blocks = num_blocks;
        out_plan->weight_compression = compression;
        uint32_t entries_size = (uint32_t)num_blocks * sizeof(tigris_weight_block_t);
        if (!bounds_ok(off + 4, entries_size, buf_len))
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
        /* First 4 bytes: uint16_t num_quant_params + uint16_t quant_data_len */
        if (!bounds_ok(off, 4, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        uint16_t nqp, qd_len;
        memcpy(&nqp, buf + off, 2);
        memcpy(&qd_len, buf + off + 2, 2);
        out_plan->num_quant_params = nqp;
        uint32_t entries_size = (uint32_t)nqp * sizeof(tigris_quant_param_t);
        uint32_t data_size = (uint32_t)qd_len * sizeof(int32_t);
        if (!bounds_ok(off + 4, entries_size + data_size, buf_len))
            return TIGRIS_ERR_BAD_SECTION;
        out_plan->quant_params = (const tigris_quant_param_t *)(buf + off + 4);
        if (qd_len > 0)
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

    /* Model I/O convenience pointers */

    out_plan->model_inputs  = out_plan->index_pool + hdr->model_io_off;
    out_plan->model_outputs = out_plan->model_inputs + hdr->num_model_inputs;

    /* Shape pool bounds validation */

    {
        /* Compute shape pool section size: gap to the next section after it */
        uint32_t sp_off = section_offsets[TIGRIS_SEC_SHAPE_POOL];
        uint32_t sp_end = buf_len;  /* default: extends to end of file */
        for (uint32_t i = 1; i < TIGRIS_SEC_MAX; i++) {
            if (section_offsets[i] > sp_off && section_offsets[i] < sp_end)
                sp_end = section_offsets[i];
        }
        /* Also consider the section directory itself as a boundary */
        if (hdr->section_dir_off > sp_off && hdr->section_dir_off < sp_end)
            sp_end = hdr->section_dir_off;

        uint32_t shape_pool_count = (sp_end - sp_off) / sizeof(int32_t);
        for (uint16_t i = 0; i < hdr->num_tensors; i++) {
            const tigris_tensor_t *t = &out_plan->tensors[i];
            if ((uint32_t)t->shape_off + t->ndim > shape_pool_count)
                return TIGRIS_ERR_BAD_SECTION;
        }
    }

    /* Plan limits validation */

    if (hdr->num_tensors > LOADER_MAX_TENSORS)
        return TIGRIS_ERR_PLAN_LIMITS;

    for (uint16_t i = 0; i < hdr->num_stages; i++) {
        const tigris_stage_t *stage = &out_plan->stages[i];
        if (stage->inputs_count > LOADER_MAX_STAGE_INPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->outputs_count > LOADER_MAX_STAGE_OUTPUTS)
            return TIGRIS_ERR_PLAN_LIMITS;
        if (stage->chain_len > LOADER_MAX_CHAIN_STAGES)
            return TIGRIS_ERR_PLAN_LIMITS;

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

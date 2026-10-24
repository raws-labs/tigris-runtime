/**
 * @file tigris_io_in_fast.h
 * @brief Which plans keep their model inputs and outputs in the fast arena.
 *
 * Internal to src/. Not installed and not part of the public API. The input
 * interface places an input where the executor reads it, so both use this
 * rule rather than a copy on each side. The compiler sizes the slow arena by
 * the same rule.
 */
#ifndef TIGRIS_IO_IN_FAST_H
#define TIGRIS_IO_IN_FAST_H

#include "tigris.h"

/**
 * Whether the plan reads its model inputs in place: one stage, no tile plan,
 * no chain, no control flow, no state, every model input read by that stage
 * and every model output written by it, schema 5 or later. Such a stage reads
 * model inputs where the caller placed them in the fast arena and leaves its
 * outputs there, so the run needs slow memory only for what the stage cannot
 * fit in the fast arena.
 */
static inline int tigris_plan_io_in_fast(const tigris_plan_t *plan)
{
    int fits = 0;
    if ((plan != NULL) && (plan->header != NULL) && (plan->stages != NULL) &&
        (plan->index_pool != NULL) &&
        (plan->header->version >= (uint32_t)TIGRIS_SCHEMA_VERSION_TILE_AXIS) &&
        (plan->header->num_stages == 1u) && (plan->num_subgraphs == 0u) &&
        (plan->num_state == 0u)) {
        const tigris_stage_t *stage = &plan->stages[0];
        if ((stage->chain_len < 2u) &&
            (stage->tile_plan_idx == (uint16_t)TIGRIS_NO_TILE_PLAN)) {
            const uint16_t *sin = tigris_stage_inputs(plan, stage);
            const uint16_t *sout = tigris_stage_outputs(plan, stage);
            fits = 1;
            for (uint8_t i = 0u; (fits != 0) && (i < plan->header->num_model_inputs); i++) {
                int found = 0;
                for (uint16_t j = 0u; j < stage->inputs_count; j++) {
                    if (sin[j] == plan->model_inputs[i]) {
                        found = 1;
                    }
                }
                fits = found;
            }
            for (uint8_t i = 0u; (fits != 0) && (i < plan->header->num_model_outputs); i++) {
                int found = 0;
                for (uint16_t j = 0u; j < stage->outputs_count; j++) {
                    if (sout[j] == plan->model_outputs[i]) {
                        found = 1;
                    }
                }
                fits = found;
            }
        }
    }
    return fits;
}

#endif /* TIGRIS_IO_IN_FAST_H */

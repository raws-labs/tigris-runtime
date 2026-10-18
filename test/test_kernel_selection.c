/* Kernel selection: built with test/kernel_selection_config.h, the runtime links
 * only the int8 and float32 Relu kernels, runs plans that use them bit for bit
 * as the full build does, and refuses at load a plan that needs Abs. Built
 * without the header, every plan loads and runs. */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "kernel_selection_plans.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failures++; } \
    else { printf("  ok: %s\n", msg); } \
} while (0)

typedef struct {
    const char *name;
    const uint8_t *plan;
    uint32_t plan_size;
    const uint8_t *input;
    uint32_t input_size;
    const uint8_t *golden;
    uint32_t golden_size;
    int selected;          /* the selected build links its kernel */
} selection_case_t;

#define CASE(sym, selected) { #sym, sym##_plan, sizeof(sym##_plan), sym##_input, \
    sizeof(sym##_input), sym##_golden, sizeof(sym##_golden), selected }

static const selection_case_t cases[] = {
    CASE(relu_s8, 1),
    CASE(relu_f32, 1),
    CASE(abs_s8, 0),
    CASE(abs_f32, 0),
};

static _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t plan_copy[1024];
static _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t fast[1024];
static _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t slow[4096];
static _Alignas(TIGRIS_TENSOR_ALIGN) uint8_t workspace[16384];
static uint8_t output[1024];

static void run_case(const selection_case_t *c)
{
    char msg[160];
    tigris_plan_t plan;
    memcpy(plan_copy, c->plan, c->plan_size);
    tigris_error_t lerr = tigris_plan_load(plan_copy, c->plan_size, &plan);
#ifdef TIGRIS_OP_CONFIG
    if (!c->selected) {
        snprintf(msg, sizeof(msg), "%s: refused at load without its kernel", c->name);
        CHECK(lerr == TIGRIS_ERR_KERNEL_DISABLED, msg);
        return;
    }
#endif
    snprintf(msg, sizeof(msg), "%s: loads", c->name);
    CHECK(lerr == TIGRIS_OK, msg);
    if (lerr != TIGRIS_OK) return;

    void *ptrs[8] = {NULL};
    tigris_mem_t mem;
    tigris_exec_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    CHECK(plan.header->num_tensors <= 8u, "tensor table fits");
    CHECK(tigris_mem_init(&mem, ptrs, plan.header->num_tensors, fast, sizeof(fast),
                          slow, sizeof(slow)) == TIGRIS_MEM_OK, "arena initializes");
    uint32_t offset = 0u;
    for (uint16_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t idx = plan.model_inputs[i];
        uint32_t n = tigris_iface_bytes(&plan, idx);
        CHECK(tigris_mem_alloc_slow(&mem, idx, plan.tensors[idx].size_bytes) == TIGRIS_MEM_OK,
              "input placed");
        CHECK(tigris_input_write(&plan, &mem, idx, c->input + offset, n) == TIGRIS_OK,
              "input written");
        offset += n;
    }
    CHECK(offset == c->input_size, "inputs consumed");
    tigris_kernel_fn dispatch = tigris_plan_data_dtype(&plan) == 1u
        ? tigris_dispatch_kernel : tigris_dispatch_kernel_s8;
    tigris_exec_error_t err = tigris_run_with_workspace_buffer(
        &plan, &mem, dispatch, NULL, &stats, workspace, sizeof(workspace));
    snprintf(msg, sizeof(msg), "%s: runs", c->name);
    CHECK(err == TIGRIS_EXEC_OK, msg);
    offset = 0u;
    for (uint16_t i = 0; i < plan.header->num_model_outputs; i++) {
        uint16_t idx = plan.model_outputs[i];
        uint32_t n = tigris_iface_bytes(&plan, idx);
        CHECK(offset + n <= sizeof(output) &&
              tigris_output_read(&plan, &mem, idx, output + offset, n) == TIGRIS_OK,
              "output read");
        offset += n;
    }
    snprintf(msg, sizeof(msg), "%s: output matches the reference bit for bit", c->name);
    CHECK(offset == c->golden_size && memcmp(output, c->golden, offset) == 0, msg);
}

int main(void)
{
#ifdef TIGRIS_OP_CONFIG
    printf("TiGrIS kernel selection tests (selected build)\n");
#else
    printf("TiGrIS kernel selection tests (full build)\n");
#endif
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        run_case(&cases[i]);
    printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

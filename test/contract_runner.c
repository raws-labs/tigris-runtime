/*
 * Host-only compiler/runtime contract bridge.
 *
 * Usage:
 *   tigris_contract_runner plan.tgrs inputs.bin outputs.bin [activation-limit]
 *       [--no-linebuffer] [--workspace-limits=T,I,O,C,S,D] [--slow-arena=BYTES]
 *       [--inputs-in-slow] [--dispatch=reference|cmsis-nn|esp-nn]
 *
 * inputs.bin and outputs.bin concatenate model tensors in model-I/O order.
 * --dispatch=reference, the default, picks the float32 or s8 reference kernels
 * by plan dtype. cmsis-nn and esp-nn exist only in a runner built with
 * TIGRIS_HAS_CMSIS_NN and TIGRIS_HAS_ESP_NN, and run int8 plans. Guard bytes
 * follow each arena, and on cmsis-nn also the activation region below the
 * vendor scratch; a changed guard byte fails the run with a line starting
 * TIGRIS_CONTRACT_GUARD, and under AddressSanitizer any access to a guard
 * fails it. A runner built with TIGRIS_CONTRACT_VENDOR_CALLS prints, after a
 * run on an accelerated dispatch, the calls each wrapped vendor function
 * received during preparation (TIGRIS_CONTRACT_VENDOR_PREPARE) and during the
 * run (TIGRIS_CONTRACT_VENDOR), 0 included; per operator type the adapter
 * takes, its calls into the s8 dispatcher (TIGRIS_CONTRACT_FALLBACK); and the
 * largest scratch a CMSIS-NN call was given with the most output channels a
 * call read expanded quantization for from the workspace
 * (TIGRIS_CONTRACT_SCRATCH).
 * The companion compiler gate owns ONNX Runtime comparison and malformed-plan
 * expectations; this executable deliberately proves only the public runtime
 * load/allocate/run/output path.
 */
#define _POSIX_C_SOURCE 200112L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#ifdef TIGRIS_HAS_CMSIS_NN
#include "tigris_kernels_cmsis_nn.h"
#endif
#ifdef TIGRIS_HAS_ESP_NN
#include "tigris_kernels_esp_nn.h"
#endif

#ifdef TIGRIS_CONTRACT_VENDOR_CALLS
/* Vendor call counters, defined in test/contract_vendor_calls.c. */
void tigris_contract_vendor_begin_run(const char *dispatch, const void *workspace,
                                      uint32_t workspace_size);
void tigris_contract_vendor_print(void);
#endif

#if defined(__SANITIZE_ADDRESS__)
#define CONTRACT_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CONTRACT_ASAN 1
#endif
#endif
#ifdef CONTRACT_ASAN
#include <sanitizer/asan_interface.h>
#endif

#ifdef TIGRIS_COUNT_KERNEL_ROWS
/* Test-only kernel row counters, defined in the runtime sources when they are
 * compiled with TIGRIS_COUNT_KERNEL_ROWS. The shipping library and the default
 * contract runner are built without this flag, so these are never linked there.
 * The rows-instrumented runner uses them to report how many kernel output rows
 * an execution computed, which the cross-repo gate diffs between the
 * line-buffered roll and the recompute path. */
extern unsigned long g_tigris_kernel_rows;
extern int32_t g_tigris_chain_tile_h;
extern int32_t g_tigris_chain_num_tiles;
extern int32_t g_tigris_interior_clamps;
#endif

typedef enum {
    DISPATCH_REFERENCE,
    DISPATCH_CMSIS_NN,
    DISPATCH_ESP_NN,
} dispatch_kind_t;

/* Bytes after each arena that no allocation may reach. */
#define GUARD_BYTES 64u

static uint8_t guard_byte(uint32_t index)
{
    return (uint8_t)(0xA5u ^ (index * 37u));
}

/* Fills a guard. Under AddressSanitizer the guard is also poisoned, so a read
 * of it fails the run as well as a write. */
static void guard_fill(uint8_t *guard, uint32_t length)
{
    for (uint32_t i = 0; i < length; ++i)
        guard[i] = guard_byte(i);
#ifdef CONTRACT_ASAN
    ASAN_POISON_MEMORY_REGION(guard, length);
#endif
}

/* Prints the guard line and returns 1 when a guard byte changed. */
static int guard_changed(uint8_t *guard, uint32_t length, const char *arena)
{
    uint32_t changed = 0;
#ifdef CONTRACT_ASAN
    ASAN_UNPOISON_MEMORY_REGION(guard, length);
#endif
    uint32_t first = 0;
    for (uint32_t i = 0; i < length; ++i) {
        if (guard[i] != guard_byte(i)) {
            if (changed == 0u)
                first = i;
            ++changed;
        }
    }
    if (changed == 0u)
        return 0;
    printf("TIGRIS_CONTRACT_GUARD arena=%s first=%" PRIu32 " changed=%" PRIu32 "\n",
           arena, first, changed);
    return 1;
}

static int load_file_aligned(
    const char *path, uint8_t **out_data, uint32_t *out_size)
{
    FILE *file;
    long end;
    uint8_t *data = NULL;
    size_t alignment = TIGRIS_TENSOR_ALIGN;

    if (!path || !out_data || !out_size)
        return -1;
    *out_data = NULL;
    *out_size = 0;

    file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0)
        goto fail;
    end = ftell(file);
    if (end <= 0 || (unsigned long)end > UINT32_MAX ||
        fseek(file, 0, SEEK_SET) != 0)
        goto fail;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);
#ifdef TIGRIS_HAS_ESP_NN
    /* ESP-NN reads filters in place from a 16-byte aligned plan. */
    if (alignment < 16u)
        alignment = 16u;
#endif
    if (posix_memalign((void **)&data, alignment, (size_t)end) != 0)
        goto fail;
    if (fread(data, 1, (size_t)end, file) != (size_t)end)
        goto fail;
    if (fclose(file) != 0) {
        file = NULL;
        goto fail;
    }

    *out_data = data;
    *out_size = (uint32_t)end;
    return 0;

fail:
    if (file)
        fclose(file);
    free(data);
    return -1;
}

static int allocate_aligned(size_t alignment, uint32_t size, void **out)
{
    if (!out || size == 0)
        return -1;
    if (alignment < sizeof(void *))
        alignment = sizeof(void *);
    return posix_memalign(out, alignment, size) == 0 ? 0 : -1;
}

static int parse_dispatch(const char *name, dispatch_kind_t *out_kind)
{
    if (strcmp(name, "reference") == 0) {
        *out_kind = DISPATCH_REFERENCE;
        return 0;
    }
#ifdef TIGRIS_HAS_CMSIS_NN
    if (strcmp(name, "cmsis-nn") == 0) {
        *out_kind = DISPATCH_CMSIS_NN;
        return 0;
    }
#endif
#ifdef TIGRIS_HAS_ESP_NN
    if (strcmp(name, "esp-nn") == 0) {
        *out_kind = DISPATCH_ESP_NN;
        return 0;
    }
#endif
    return -1;
}

static const char *dispatch_name(dispatch_kind_t kind)
{
    switch (kind) {
    case DISPATCH_REFERENCE:
        return "reference";
    case DISPATCH_CMSIS_NN:
        return "cmsis-nn";
    case DISPATCH_ESP_NN:
        return "esp-nn";
    }
    return "unknown";
}

static int choose_dispatch(
    const tigris_plan_t *plan, dispatch_kind_t kind,
    tigris_kernel_fn *out_dispatch)
{
    uint8_t dtype;

    if (!plan || !plan->header || !plan->tensors || !out_dispatch ||
        plan->header->num_model_inputs == 0 ||
        plan->header->num_model_outputs == 0)
        return -1;

    dtype = tigris_plan_data_dtype(plan);
    if (kind != DISPATCH_REFERENCE) {
        /* The accelerated dispatchers run int8 plans. */
        if (dtype != 3)
            return -1;
#ifdef TIGRIS_HAS_CMSIS_NN
        if (kind == DISPATCH_CMSIS_NN) {
            *out_dispatch = tigris_dispatch_kernel_cmsis_nn;
            return 0;
        }
#endif
#ifdef TIGRIS_HAS_ESP_NN
        if (kind == DISPATCH_ESP_NN) {
            *out_dispatch = tigris_dispatch_kernel_esp_nn;
            return 0;
        }
#endif
        return -1;
    }
    if (dtype == 1) {
        *out_dispatch = tigris_dispatch_kernel;
        return 0;
    }
    if (dtype == 3) {
        *out_dispatch = tigris_dispatch_kernel_s8;
        return 0;
    }
    return -1;
}

static int slow_requirement(
    const tigris_plan_t *plan, uint32_t *out_size)
{
    uint64_t total = 0;

    if (!plan || !plan->header || !plan->tensors || !out_size)
        return -1;
    for (uint16_t i = 0; i < plan->header->num_tensors; ++i) {
        total += plan->tensors[i].size_bytes;
        if (total > UINT32_MAX / 2u)
            return -1;
    }
    total *= 2u;
    if (total < 4096u)
        total = 4096u;
    *out_size = (uint32_t)total;
    return 0;
}

/* Reads the model inputs from path and reports on stderr why it could not. */
static int load_inputs(
    const char *path, const tigris_plan_t *plan, tigris_mem_t *mem, int in_slow)
{
    FILE *file;
    int trailing;

    if (!path || !plan || !plan->header || !mem)
        return -1;
    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "could not read inputs: %s\n", path);
        return -1;
    }

    for (uint8_t i = 0; i < plan->header->num_model_inputs; ++i) {
        uint16_t index = plan->model_inputs[i];
        uint32_t iface_size;
        void *staging;
        tigris_error_t written;
        if (index >= plan->header->num_tensors)
            goto layout;
        /* The file holds the dtype the model declares, which is not always the
         * dtype the plan executes on. The write places the tensor where the
         * run reads it. */
        iface_size = tigris_iface_bytes(plan, index);
        if (iface_size == 0u)
            goto layout;
        if (in_slow) {
            tigris_mem_error_t allocated =
                tigris_mem_alloc_slow(mem, index, plan->tensors[index].size_bytes);
            if (allocated != TIGRIS_MEM_OK) {
                fprintf(stderr, "input allocation failed: %s\n",
                        tigris_mem_error_str(allocated));
                goto fail;
            }
        }
        staging = malloc(iface_size);
        if (!staging) {
            fprintf(stderr, "could not allocate input buffer\n");
            goto fail;
        }
        if (fread(staging, 1, iface_size, file) != iface_size) {
            free(staging);
            goto layout;
        }
        written = tigris_input_write(plan, mem, index, staging, iface_size);
        free(staging);
        if (written != TIGRIS_OK) {
            fprintf(stderr, "input write failed: %s\n", tigris_error_str(written));
            goto fail;
        }
    }
    trailing = fgetc(file);
    if (trailing != EOF)
        goto layout;
    if (fclose(file) != 0) {
        fprintf(stderr, "could not read inputs: %s\n", path);
        return -1;
    }
    return 0;

layout:
    fprintf(stderr, "input layout does not match plan model inputs\n");
fail:
    fclose(file);
    return -1;
}

static int write_outputs(
    const char *path, const tigris_plan_t *plan, const tigris_mem_t *mem)
{
    FILE *file;

    if (!path || !plan || !plan->header || !mem)
        return -1;
    file = fopen(path, "wb");
    if (!file)
        return -1;

    for (uint8_t i = 0; i < plan->header->num_model_outputs; ++i) {
        uint16_t index = plan->model_outputs[i];
        uint32_t iface_size;
        void *staging;
        if (index >= plan->header->num_tensors)
            goto fail;
        iface_size = tigris_iface_bytes(plan, index);
        if (iface_size == 0u)
            goto fail;
        staging = malloc(iface_size);
        if (!staging)
            goto fail;
        if (tigris_output_read(plan, mem, index, staging, iface_size) !=
                TIGRIS_OK ||
            fwrite(staging, 1, iface_size, file) != iface_size) {
            free(staging);
            goto fail;
        }
        free(staging);
    }
    if (fclose(file) != 0)
        return -1;
    return 0;

fail:
    fclose(file);
    return -1;
}

int main(int argc, char **argv)
{
    uint8_t *plan_data = NULL;
    uint32_t plan_size = 0;
    void **tensor_ptrs = NULL;
    uint8_t *fast_buf = NULL;
    tigris_exec_stats_t stats;
    uint8_t *slow_buf = NULL;
    uint32_t fast_size;
    uint32_t core_size;
    uint32_t activation_size;
    uint32_t activation_guard = 0;
    uint32_t backend_scratch = 0;
    dispatch_kind_t dispatch_kind = DISPATCH_REFERENCE;
    int prepared = 0;
    int guards_ok;
    uint32_t required_size;
    uint32_t activation_limit;
    uint32_t slow_size;
    uint32_t overhead;
    tigris_plan_t plan;
    tigris_mem_t mem;
    tigris_kernel_fn dispatch = NULL;
    int result = 1;
    int no_linebuffer = 0;
    /* Workspace limits as generated code states them, or all zero to run
     * with the executor's own global workspace. */
    unsigned long limits[6] = {0, 0, 0, 0, 0, 0};
    int have_limits = 0;
    /* Exact slow arena size, 0 for none. */
    unsigned long slow_arena = 0;
    int have_slow_arena = 0;
    /* Allocate inputs in slow first, so the run copies them into fast. */
    int inputs_in_slow = 0;
    void *workspace = NULL;
    void *state = NULL;
    char *pos_argv[5];
    int pos_argc = 1;
    int arg_index;

    /* Separate the optional --no-linebuffer switch from the positional
     * arguments so it may appear anywhere on the command line. Everything
     * else keeps its existing positional meaning, so a call without the
     * switch behaves exactly as before. */
    pos_argv[0] = argv[0];
    for (arg_index = 1; arg_index < argc; ++arg_index) {
        if (strcmp(argv[arg_index], "--no-linebuffer") == 0) {
            no_linebuffer = 1;
        } else if (strncmp(argv[arg_index], "--workspace-limits=", 19) == 0) {
            if (sscanf(argv[arg_index] + 19, "%lu,%lu,%lu,%lu,%lu,%lu", &limits[0],
                       &limits[1], &limits[2], &limits[3], &limits[4], &limits[5]) != 6) {
                fprintf(stderr, "invalid workspace limits: %s\n", argv[arg_index]);
                return 2;
            }
            have_limits = 1;
        } else if (strncmp(argv[arg_index], "--slow-arena=", 13) == 0) {
            char *end = NULL;
            slow_arena = strtoul(argv[arg_index] + 13, &end, 10);
            if (!argv[arg_index][13] || !end || *end != '\0' || slow_arena > UINT32_MAX) {
                fprintf(stderr, "invalid slow arena: %s\n", argv[arg_index]);
                return 2;
            }
            have_slow_arena = 1;
        } else if (strcmp(argv[arg_index], "--inputs-in-slow") == 0) {
            inputs_in_slow = 1;
        } else if (strncmp(argv[arg_index], "--dispatch=", 11) == 0) {
            if (parse_dispatch(argv[arg_index] + 11, &dispatch_kind) != 0) {
                fprintf(stderr, "dispatch not available in this runner: %s\n",
                        argv[arg_index] + 11);
                return 2;
            }
        } else if (pos_argc < (int)(sizeof(pos_argv) / sizeof(pos_argv[0]))) {
            pos_argv[pos_argc++] = argv[arg_index];
        } else {
            fprintf(stderr, "too many arguments\n");
            return 2;
        }
    }

    if (pos_argc != 4 && pos_argc != 5) {
        fprintf(stderr,
                "usage: %s plan.tgrs inputs.bin outputs.bin [activation-limit] "
                "[--no-linebuffer] [--workspace-limits=T,I,O,C,S,D] [--slow-arena=BYTES] "
                "[--inputs-in-slow] [--dispatch=reference|cmsis-nn|esp-nn]\n",
                pos_argv[0]);
        return 2;
    }
    if (load_file_aligned(pos_argv[1], &plan_data, &plan_size) != 0) {
        fprintf(stderr, "could not read plan: %s\n", pos_argv[1]);
        goto cleanup;
    }
    {
        tigris_error_t error = tigris_plan_load(plan_data, plan_size, &plan);
        if (error != TIGRIS_OK) {
            fprintf(stderr, "plan load failed: %s\n", tigris_error_str(error));
            goto cleanup;
        }
    }
    if (choose_dispatch(&plan, dispatch_kind, &dispatch) != 0) {
        fprintf(stderr, "plan has unsupported or mixed contract dtype for the "
                "%s dispatch\n", dispatch_name(dispatch_kind));
        goto cleanup;
    }
    overhead = tigris_weight_decompression_overhead(&plan);
    required_size = tigris_fast_arena_required(&plan);
    if (plan.header->budget == 0 || overhead == UINT32_MAX ||
        required_size == UINT32_MAX ||
        slow_requirement(&plan, &slow_size) != 0) {
        fprintf(stderr, "plan has unrepresentable arena requirements\n");
        goto cleanup;
    }
    activation_limit = plan.header->budget;
    if (pos_argc == 5) {
        char *end = NULL;
        unsigned long parsed = strtoul(pos_argv[4], &end, 10);
        if (!pos_argv[4][0] || !end || *end != '\0' || parsed == 0 ||
            parsed > plan.header->budget || parsed > UINT32_MAX) {
            fprintf(stderr, "invalid activation limit: %s\n", pos_argv[4]);
            goto cleanup;
        }
        activation_limit = (uint32_t)parsed;
    }
    if (activation_limit > UINT32_MAX - overhead) {
        fprintf(stderr, "activation limit plus reserve is unrepresentable\n");
        goto cleanup;
    }
    fast_size = activation_limit + overhead;
    core_size = fast_size;
    activation_size = fast_size;
#ifdef TIGRIS_HAS_CMSIS_NN
    if (dispatch_kind == DISPATCH_CMSIS_NN) {
        /* The scratch goes on top of the 16-aligned core arena, as
         * tigris_cmsis_nn_fast_arena_required() sizes it. */
        backend_scratch = tigris_cmsis_nn_scratch_required(&plan);
        if (backend_scratch == UINT32_MAX || fast_size > UINT32_MAX - 15u ||
            (backend_scratch > 0u &&
             backend_scratch > UINT32_MAX - ((fast_size + 15u) & ~15u))) {
            fprintf(stderr, "plan has unrepresentable CMSIS-NN scratch\n");
            goto cleanup;
        }
        if (backend_scratch > 0u) {
            activation_size = (fast_size + 15u) & ~15u;
            fast_size = activation_size + backend_scratch;
            activation_guard = GUARD_BYTES;
        }
    }
#endif
    if (have_slow_arena)
        slow_size = (uint32_t)slow_arena;
    /* Fast buffer layout, offsets from its base:
     *   [0, core_size)  activation region the executor allocates in, exactly
     *       the activation limit plus the reserve, as on the reference dispatch
     *   [core_size, activation_size + activation_guard)  guard
     *       "fast_activation": the padding to the 16-aligned core arena plus
     *       the inner guard
     *   then backend_scratch bytes of CMSIS-NN workspace, then guard "fast".
     * activation_guard is 0 without a CMSIS-NN workspace, so the "fast" guard
     * directly follows the activation region. tigris_cmsis_nn_prepare() carves
     * the workspace from the top of the arena it is given, so the runner
     * passes the 16-aligned region plus the inner guard and, once the
     * workspace is placed above that guard, lowers mem.fast_size to
     * core_size. The executor sees the region a reference run sees, and the
     * vendor the scratch it would have without the guard;
     * tigris_cmsis_nn_deinit() restores the size the arena was prepared with. */
    tensor_ptrs = calloc(plan.header->num_tensors, sizeof(*tensor_ptrs));
    if (!tensor_ptrs || fast_size > UINT32_MAX - 2u * GUARD_BYTES ||
        slow_size > UINT32_MAX - GUARD_BYTES ||
        allocate_aligned(TIGRIS_TENSOR_ALIGN,
                         fast_size + activation_guard + GUARD_BYTES,
                         (void **)&fast_buf) != 0 ||
        (slow_size > 0u &&
         allocate_aligned(TIGRIS_TENSOR_ALIGN, slow_size + GUARD_BYTES,
                          (void **)&slow_buf) != 0)) {
        fprintf(stderr, "could not allocate contract arenas\n");
        goto cleanup;
    }
    if (activation_guard)
        guard_fill(fast_buf + core_size, activation_size - core_size + activation_guard);
    guard_fill(fast_buf + activation_guard + fast_size, GUARD_BYTES);
    if (slow_buf)
        guard_fill(slow_buf + slow_size, GUARD_BYTES);
    if (tigris_mem_init(
            &mem, tensor_ptrs, plan.header->num_tensors,
            fast_buf, fast_size + activation_guard, slow_buf, slow_size) != TIGRIS_MEM_OK) {
        fprintf(stderr, "memory initialization failed\n");
        goto cleanup;
    }
#ifdef TIGRIS_HAS_CMSIS_NN
    if (dispatch_kind == DISPATCH_CMSIS_NN) {
        /* Preparation would accept an arena short of its requirement by at
         * most the inner guard, so the runner refuses that arena itself. */
        uint32_t cmsis_required = tigris_cmsis_nn_fast_arena_required(&plan);
        if ((activation_guard &&
             (cmsis_required == UINT32_MAX || cmsis_required > fast_size)) ||
            tigris_cmsis_nn_prepare(&plan, &mem) != 0) {
            fprintf(stderr, "CMSIS-NN preparation failed\n");
            goto cleanup;
        }
        prepared = 1;
        if (mem.fast_size != activation_size + activation_guard) {
            fprintf(stderr, "CMSIS-NN workspace is not where the guard layout "
                    "expects it\n");
            goto cleanup;
        }
        mem.fast_size = core_size;
    }
#endif
#ifdef TIGRIS_HAS_ESP_NN
    if (dispatch_kind == DISPATCH_ESP_NN) {
        /* ESP-NN takes its workspace from the heap, not the fast arena. */
        if (tigris_esp_nn_prepare(&plan, &mem) != 0) {
            fprintf(stderr, "ESP-NN preparation failed\n");
            goto cleanup;
        }
        prepared = 1;
    }
#endif
    if (load_inputs(pos_argv[2], &plan, &mem, inputs_in_slow) != 0)
        goto cleanup;
    if (no_linebuffer) {
        /* Force the recompute path: clear the line-buffered flag on every
         * loaded stage. The loader is zero-copy, so plan.stages aliases the
         * loaded buffer; locate each _reserved1 byte inside plan_data and
         * clear the flag bit in place, exactly as the executor line-buffer
         * ctest does. A chain without the flag executes the unchanged
         * recompute path, which is what the gate measures against. */
        for (uint16_t s = 0; s < plan.header->num_stages; ++s) {
            const uint8_t *field = (const uint8_t *)&plan.stages[s]._reserved1;
            size_t off = (size_t)(field - plan_data);
            if (off + sizeof(uint16_t) <= plan_size)
                plan_data[off] = (uint8_t)(plan_data[off] &
                    ~(uint8_t)(TIGRIS_STAGE_FLAG_LINE_BUFFERED & 0xFFu));
        }
    }
#ifdef TIGRIS_COUNT_KERNEL_ROWS
    g_tigris_kernel_rows = 0;
#endif
#ifdef TIGRIS_CONTRACT_VENDOR_CALLS
    if (activation_guard)
        tigris_contract_vendor_begin_run(dispatch_name(dispatch_kind),
                                         fast_buf + activation_size + activation_guard,
                                         backend_scratch);
    else
        tigris_contract_vendor_begin_run(dispatch_name(dispatch_kind), NULL, 0);
#endif
    memset(&stats, 0, sizeof(stats));
    {
        tigris_exec_error_t error;
        if (tigris_state_required(&plan) > 0u) {
            /* A plan that keeps state runs once from its initial state, as a
             * fresh session does. */
            size_t workspace_size = tigris_executor_workspace_required(&plan);
            workspace = malloc(workspace_size);
            state = malloc(tigris_state_required(&plan));
            if (!workspace || !state ||
                tigris_state_init(&plan, state, tigris_state_required(&plan)) != TIGRIS_EXEC_OK) {
                fprintf(stderr, "could not prepare the state buffer\n");
                goto cleanup;
            }
            error = tigris_run_with_state(&plan, &mem, dispatch, NULL, &stats, workspace,
                                          workspace_size, state, tigris_state_required(&plan));
        } else if (have_limits) {
            /* Exactly what generated code reserves for these limits. */
            size_t workspace_size = TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS(
                limits[0], limits[1], limits[2], limits[3], limits[4], limits[5]);
            workspace = malloc(workspace_size);
            if (!workspace) {
                fprintf(stderr, "could not allocate the executor workspace\n");
                goto cleanup;
            }
            error = tigris_run_with_workspace_buffer(
                &plan, &mem, dispatch, NULL, &stats, workspace, workspace_size);
        } else {
            error = tigris_run(&plan, &mem, dispatch, NULL, &stats);
        }
        guards_ok = !(activation_guard &&
                      guard_changed(fast_buf + core_size,
                                    activation_size - core_size + activation_guard,
                                    "fast_activation"));
        if (guard_changed(fast_buf + activation_guard + fast_size, GUARD_BYTES, "fast"))
            guards_ok = 0;
        if (slow_buf && guard_changed(slow_buf + slow_size, GUARD_BYTES, "slow"))
            guards_ok = 0;
        if (!guards_ok)
            goto cleanup;
        if (error != TIGRIS_EXEC_OK) {
            fprintf(stderr, "inference failed: %s after %u stages "
                    "(fast peak %u of %u)\n",
                    tigris_exec_error_str(error),
                    (unsigned)(stats.stages_normal + stats.stages_tiled +
                               stats.stages_chain),
                    (unsigned)mem.fast_peak, (unsigned)mem.fast_size);
            goto cleanup;
        }
    }
    if (write_outputs(pos_argv[3], &plan, &mem) != 0) {
        fprintf(stderr, "could not write model outputs\n");
        goto cleanup;
    }
    printf(
        "TIGRIS_CONTRACT_MEMORY budget=%" PRIu32 " activation_limit=%" PRIu32
        " reserve=%" PRIu32 " required=%" PRIu32 " allocated=%" PRIu32
        " peak=%" PRIu32 " slow_peak=%" PRIu32 "\n",
        plan.header->budget, activation_limit, overhead, required_size,
        fast_size, mem.fast_peak, stats.slow_peak);
    printf("TIGRIS_CONTRACT_DISPATCH name=%s scratch=%" PRIu32 "\n",
           dispatch_name(dispatch_kind), backend_scratch);
#ifdef TIGRIS_CONTRACT_VENDOR_CALLS
    if (dispatch_kind != DISPATCH_REFERENCE)
        tigris_contract_vendor_print();
#endif
#ifdef TIGRIS_COUNT_KERNEL_ROWS
    printf(
        "TIGRIS_CONTRACT_ROWS kernel_rows=%lu chain_tiles=%" PRId32
        " chain_tile_h=%" PRId32 " interior_clamps=%" PRId32 "\n",
        g_tigris_kernel_rows, g_tigris_chain_num_tiles,
        g_tigris_chain_tile_h, g_tigris_interior_clamps);
#endif
    result = 0;

cleanup:
#ifdef TIGRIS_HAS_CMSIS_NN
    if (prepared && dispatch_kind == DISPATCH_CMSIS_NN)
        (void)tigris_cmsis_nn_deinit(&mem);
#endif
#ifdef TIGRIS_HAS_ESP_NN
    if (prepared && dispatch_kind == DISPATCH_ESP_NN)
        tigris_esp_nn_deinit();
#endif
    (void)prepared;
    free(state);
    free(workspace);
    free(slow_buf);
    free(fast_buf);
    free(tensor_ptrs);
    free(plan_data);
    return result;
}

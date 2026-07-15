# Stack usage contract

TiGrIS runtime stack use is independent of model tensor and channel counts.
The loader rejects plans beyond the configured structural limits, the executor
keeps limit-sized bookkeeping in an explicit workspace, and accelerated
backends prepare channel-sized quantization storage before inference. Runtime
sources are compiled with `-Wvla`, so a variable-length array is a build error.

## Executor workspace

For new integrations, allocate one workspace outside the task stack and use
the re-entrant entry point:

```c
static tigris_executor_workspace_t executor_workspace;

tigris_exec_error_t err = tigris_run_with_workspace(
    &plan, &mem, dispatch, user_ctx, &stats, &executor_workspace);
```

The default configuration uses 9,152 bytes on a 64-bit target and 8,896 bytes
on the supported 32-bit embedded targets. `sizeof(tigris_executor_workspace_t)`
and `tigris_executor_workspace_size()` are authoritative for a particular
build. The storage may instead be heap-backed or task-local, but putting it on
the task stack adds its full size to that task's requirement.

The source-compatible `tigris_run()` wrapper owns one static workspace. It is
not re-entrant and must not be used concurrently. Define
`TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0` when building the runtime to omit
that compatibility workspace; callers must then use
`tigris_run_with_workspace()`.

## Compile-time limits

Override these definitions consistently for every runtime translation unit:

| Definition | Default | Bounds |
|---|---:|---:|
| `TIGRIS_MAX_TENSORS` | 512 | 1–65,535 |
| `TIGRIS_MAX_STAGE_INPUTS` | 16 | 1–65,535 |
| `TIGRIS_MAX_STAGE_OUTPUTS` | 16 | 1–65,535 |
| `TIGRIS_MAX_CHAIN_STAGES` | 16 | 2–65,535 |
| `TIGRIS_MAX_SPATIAL_OPS_PER_STAGE` | 8 | 1 or greater |

Lower values reduce executor workspace, while higher values accept larger
plans and enlarge it. `tigris_plan_load()` returns `TIGRIS_ERR_PLAN_LIMITS`
before execution when a plan exceeds the selected limits. An internal
compile-time assertion verifies that `TIGRIS_EXECUTOR_WORKSPACE_BYTES` can
hold the implementation's bounded state.

## Accelerated backends

`tigris_cmsis_nn_prepare()` reserves both vendor scratch and any required
scalar-to-per-channel quantization expansion from the top of the fast arena.
Use `tigris_cmsis_nn_fast_arena_required()` when sizing that arena.

`tigris_esp_nn_prepare()` allocates the same quantization expansion alongside
the ESP-NN scratch, depthwise-output, and padding buffers. Neither adapter
allocates channel-sized arrays during inference. Both adapters use one global
prepared context, so preparation and dispatch are not concurrently re-entrant.

## Measured frames and CI budget

The measurements below were captured on 2026-07-15 from Release/`-Os` builds
with GCC `-fstack-usage`. They are maximum individual runtime function frames,
not complete application task-stack requirements.

| Target and toolchain | Largest runtime frame | Executor frame | Accelerated adapter frame |
|---|---:|---:|---:|
| x86-64, GCC 13.3.0 | 768 B | 768 B | CMSIS stub: 320 B |
| Cortex-M4, Arm GNU 13.2.1 | 504 B | 384 B | CMSIS stub: 240 B |
| ESP32-S3, Xtensa GNU 14.2.0, ESP-IDF 5.4 | 528 B | 416 B | ESP-NN: 272 B |

The host and Cortex-M CI jobs fail if any shipped runtime function exceeds
1,024 bytes or GCC reports an unbounded dynamic frame. The check can be run
locally with:

```bash
cmake -S . -B build-stack \
  -DCMAKE_BUILD_TYPE=Release \
  -DTIGRIS_STACK_USAGE=ON
cmake --build build-stack --target tigris_runtime cmsis_adapter_compile
python3 scripts/check_stack_usage.py --max-frame 1024 build-stack
```

Compiler `.su` output excludes caller frames, vendor-library callees, C library
internals, RTOS context, interrupt nesting, and instrumentation. A task stack
must cover the complete call chain plus target-specific margin. Measure the
final firmware's task high-water mark under its largest model and worst
interrupt load before choosing the production stack size.

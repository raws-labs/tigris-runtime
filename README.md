# TiGrIS Runtime

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/docs)

Portable C runtime for [TiGrIS](https://github.com/raws-labs/tigris). It
loads compatible `.tgrs` plans and executes them against caller-owned memory
arenas. The reference runtime has no dependency beyond the C library and does
not call a general-purpose allocator during inference.

## What it does

A plan records the operator order, tensor metadata, stage boundaries, and tile
strategy selected by the compiler. At runtime, TiGrIS:

- validates the plan version and structural limits before exposing its tables;
- assigns activation addresses from caller-provided fast and slow arenas;
- executes normal, spatially tiled, and streamed-chain stages;
- reads uncompressed weights in place or decompresses stage blocks into the
  fast arena; and
- reports allocation, tiling, and kernel failures to the caller.

The schedule and activation bound are compiled, but addresses are assigned by
bounded bump allocation, reset, and compaction during execution. A finite
arena can still be exhausted. The loader validates plan-version compatibility
before exposing plan tables.

## Kernel backends

| Backend | Model dtype | Target | Dispatch |
|---|---|---|---|
| `reference` | float32 | Any C target | `tigris_dispatch_kernel` |
| `s8_ref` | int8 | Any C target | `tigris_dispatch_kernel_s8` |
| `esp-nn` | int8 | ESP32 family | `tigris_dispatch_kernel_esp_nn` |
| `cmsis-nn` | int8 | Cortex-M family | `tigris_dispatch_kernel_cmsis_nn` |

Accelerated adapters fall back to `s8_ref` for explicitly supported
non-native variants. They are not interchangeable with arbitrary plans:
generate a harness with `tigris codegen --backend ...` so the compiler can
validate the selected dtype/operator route.

ESP-NN and CMSIS-NN also require a successful preparation call after
`tigris_mem_init()` and before inference. ESP-NN preparation obtains
platform-managed workspace; CMSIS-NN reserves vendor and quantization workspace
from the top of the fast buffer and reduces `mem.fast_size`. ESP preparation may be repeated safely to
replace its workspace and `tigris_esp_nn_deinit()` releases it after inference.
CMSIS preparation is idempotent for the same arena when its existing scratch
is sufficient; call `tigris_cmsis_nn_deinit()` before changing arena or plan.
Use `tigris_cmsis_nn_fast_arena_required()` to size that arena before
initialization; it preserves the full core activation/weight capacity below the
exact workspace requirement reported by the linked CMSIS-NN library.

## Memory contract

The caller owns:

- a fast arena, normally SRAM;
- a slow arena, normally PSRAM or another writable RAM region; and
- one `void *` entry per tensor.

Generated integrations reserve a plan-sized executor buffer automatically; no
workspace limit tuning is required. Manual integrations can query
`tigris_executor_workspace_required()` after loading the plan and pass that many
caller-owned bytes to `tigris_run_with_workspace_buffer()`. The fixed
`tigris_executor_workspace_t` and `tigris_run_with_workspace()` remain available
for source compatibility. `tigris_run()` uses one process-global fixed
workspace and is not safe for concurrent inference. Arena sizing, stack
provisioning, and re-entrant execution are documented in the
[runtime integration guide](https://tigris-ml.dev/docs/runtime/integration/).

The plan's `budget` is the modeled activation requirement. For an arena whose
base satisfies `TIGRIS_TENSOR_ALIGN`, `tigris_fast_arena_required()` returns
the core capacity needed for that budget plus any simultaneous compressed
weights. Account for base-alignment padding when using an unaligned buffer and
provision backend workspace separately.
Optimized Cortex-M builds require the plan base and tensors to satisfy
`TIGRIS_TENSOR_ALIGN` (16 bytes with DSP enabled).

For CMSIS-NN, “separately” is enforceable rather than an estimate:
`tigris_cmsis_nn_scratch_required()` queries the linked vendor library and
`tigris_cmsis_nn_fast_arena_required()` combines that result with the core
requirement. Generated standalone CMSIS harnesses reserve 4 KiB by default,
validate it at startup, and report the exact required total when a larger model
needs `TIGRIS_CMSIS_NN_SCRATCH_BYTES` overridden at build time.

The compiler's current cost model aligns each activation to 32 bytes, which is
conservative for the supported host, Cortex-M, and ESP targets. `mem.fast_peak`
is the observed allocator high-water mark, not the scheduled minimum: a roomy
arena may defer compaction and therefore report a larger value. Contract tests
run with the compiler's scheduled limit to prove that compaction, tiling, and
weight reservation still execute within that bound.

The core executor performs no unbounded heap fallback. Normal stages may spill
to the supplied slow arena; if neither bounded arena can satisfy an operation,
the executor returns an error. `mem.fast_peak` records the measured core
fast-arena high-water mark.

## Build and test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest registers only self-contained tests. Fixture-driven executables are
built but require plans generated by the compiler. Generate them with:

```bash
pip install tigris-ml
tigris gen-fixtures model.onnx -o test/fixtures/ -m 256K
./test/run_all.sh test/fixtures
```

The top-level build also compiles the canonical POSIX int8 integration example:

```bash
./build/tigris_posix_example model.tgrs
```

See [examples/posix/main.c](examples/posix/main.c) for checked loader, arena,
input, execution, and output handling. Production integrations normally use
the harness emitted by `tigris codegen`.

## Install for CMake consumers

Install the runtime to a prefix:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/tigris-runtime
```

An external project can then consume the installed headers and library without
depending on the TiGrIS source tree:

```cmake
find_package(tigris_runtime 0.5 REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE tigris::runtime)
```

Pass the installation prefix through `CMAKE_PREFIX_PATH` when it is outside a
standard system location.

## ESP32 (ESP-IDF)

The ESP-IDF example lives in `examples/esp32`:

```bash
cd examples/esp32
idf.py set-target esp32s3
idf.py build
idf.py flash
```

The example's portable int8 path is the default. With the `espressif/esp-nn`
managed component available, pass `-DTIGRIS_ENABLE_ESP_NN=ON` to build the
ESP-NN adapter.

The plan should live in a memory-mapped flash partition. Allocate the fast
arena from internal SRAM and the slow arena from PSRAM when available. When
using ESP-NN, call `tigris_esp_nn_prepare()` and check its return value before
`tigris_run()`, then call `tigris_esp_nn_deinit()` after the final inference.

## Further reading

- [Getting started](https://tigris-ml.dev/docs)
- [Compiler and plan compatibility data](https://github.com/raws-labs/tigris/blob/main/compatibility.json)
- [Runtime integration](https://tigris-ml.dev/docs/runtime/integration)
- [Runtime API](https://tigris-ml.dev/docs/runtime/api-reference)
- [Memory model](https://tigris-ml.dev/docs/architecture/memory-model)

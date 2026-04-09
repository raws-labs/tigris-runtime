# TiGrIS Runtime

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/docs)

Portable C99 runtime for [TiGrIS](https://github.com/raws-labs/tigris). Executes `.tgrs` inference plans on embedded devices with no dynamic allocation and no dependencies beyond libc.

## What it does

A `.tgrs` plan is an ordered list of operator calls with pre-computed tile parameters, arena offsets, and weight pointers. The runtime is small because it does not need to make any decisions: the compiler already made them.

- Loads a `.tgrs` plan from flash (memory-mapped, no copy)
- Walks the plan stage by stage, using a bump allocator over an arena you provide
- Handles tiled execution (spatial strips, chain streaming, temporal partitioning) transparently
- Pluggable kernel backends: pick one at link time
- About 5 KB of code on a typical Cortex-M build, zero heap allocation at runtime

The plan format is stable and versioned; the runtime refuses to load plans it does not understand.

## Kernel backends

Backends are swapped at link time. The compiler does not care which one you pick: plans are the same across all of them.

| Backend | Target | Source |
|---------|--------|--------|
| `reference` | Any C99 | `tigris_kernels.c` |
| `s8_ref` | Any C99, int8 | `tigris_kernels_s8.c` |
| `esp-nn` | ESP32 family (SIMD on S3/P4, C fallbacks elsewhere) | `tigris_kernels_esp_nn.c` |
| `cmsis-nn` | Cortex-M family (DSP on M4/M7, Helium on M55, plain C elsewhere) | `tigris_kernels_cmsis_nn.c` |

You can also register your own kernel function and pass it to `tigris_run`; the runtime does not assume any particular kernel library.

## Memory model

The runtime works against a memory region you own. You pass in an SRAM buffer (always required) and optionally a PSRAM buffer; the plan decides which stages live where. Weights are either copied into the arena or read in place from flash (XIP), depending on how the plan was compiled.

```
+--------- SRAM arena ---------+  +------- PSRAM (optional) -------+
| activations (bump-allocated) |  | spill buffers, large weights   |
+------------------------------+  +--------------------------------+
                |
                +--- weights: copied here, or read in place from flash
```

No malloc, no free, no hidden allocations. The arena sizes are fixed at compile time and validated when the plan is loaded.

## Build and test

```bash
cmake -B build && cmake --build build
./build/test_kernels
./build/test_e2e test/fixtures/model 262144
```

Generate test fixtures from an ONNX model using the compiler:

```bash
pip install tigris-ml
tigris gen-fixtures model.onnx -o test/fixtures/ -m 256K
```

## ESP32 (ESP-IDF)

The runtime ships as a standard ESP-IDF component. An example project lives under `examples/esp32`:

```bash
cd examples/esp32
idf.py set-target esp32s3
idf.py build && idf.py flash
```

## Integration

The runtime is a single static library. Link it, pick a kernel backend, and call:

```c
#include "tigris.h"

tigris_plan_t plan;
tigris_load(&plan, plan_data, plan_size);

tigris_mem_t mem;
tigris_mem_init(&mem, sram_buf, sram_size, psram_buf, psram_size);

tigris_run(&plan, &mem, input_data, output_ptrs, kernel_func, NULL);
```

For most users this is wrapped by the C harness that `tigris codegen` produces. You only touch these APIs directly when you embed the runtime in a larger system with its own memory layout.

## Further reading

- [Getting started](https://tigris-ml.dev/docs): installation, first compile, deploying to ESP32
- [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris): design, benchmarks, how tiling works
- [Runtime API](https://tigris-ml.dev/docs/runtime/api-reference): header reference and memory model

# POSIX example

Canonical host integration for an int8 `.tgrs` plan: load a plan file, allocate
caller-owned arenas, run one inference with the portable int8 kernels, and print
the output in the dtype the model declares. Use it as the reference for wiring
the loader, the memory manager, the interface conversion, and the executor
together before porting to a device.

## Build and run

The example is built with the runtime:

```bash
cmake -S . -B build && cmake --build build
./build/tigris_posix_example model.tgrs
```

Generate a plan with the compiler (`tigris compile model.onnx -m 256K -o
model.tgrs`), or use the frozen int8 plan
`test/schema_compat/schema-v9-binary-requant.tgrs`, which CTest runs as
`posix_example`.

The plan must execute on int8; the example reports and exits otherwise. Its
boundary is a separate question: `tigris_output_read()` returns the output in
the dtype the model file declares, float32, uint8 or int8, and the example
prints up to five values of the first output in that dtype. Arenas are host
allocations, so the slow tier is sized generously and is not a device budget.
The frozen plan runs as one stage, so its inputs and output stay in the fast
arena and the slow tier goes unused.

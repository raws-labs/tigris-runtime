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
model.tgrs`), or use one of the fixtures under `test/fixtures/`.

The plan must execute on int8; the example reports and exits otherwise. Its
boundary is a separate question: `tigris_output_read()` returns the output in
the dtype the model file declares, which for a quantized ONNX graph is float32,
so the example prints floats whatever the plan stores. Arenas are host
allocations, so the slow tier is sized generously and is not a device budget.

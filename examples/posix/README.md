# POSIX example

Canonical host integration for an int8 `.tgrs` plan: load a plan file, allocate
caller-owned arenas, run one inference with the portable int8 kernels, and print
the output. Use it as the reference for wiring the loader, the memory manager,
and the executor together before porting to a device.

## Build and run

The example is built with the runtime:

```bash
cmake -S . -B build && cmake --build build
./build/tigris_posix_example model.tgrs
```

Generate a plan with the compiler (`tigris compile model.onnx -m 256K -o
model.tgrs`), or use one of the fixtures under `test/fixtures/`.

The plan must have int8 model inputs and outputs; the example reports and exits
otherwise. Arenas are host allocations, so the slow tier is sized generously and
is not a device budget.

"""Record reduction outputs from the pinned TFLite Micro reference kernels."""

from importlib.metadata import version
from pathlib import Path

from generate_elementwise_goldens import (
    REFERENCE_VERSION, flatbuffers, model_bytes, np, runtime, schema,
)


KINDS = {"REDUCE_MAX": 60, "REDUCE_MIN": 61, "SUM": 62, "CUMSUM": 63}


def reduction_model(kind, shape, axis, keep, exclusive, reverse, quantized, scales, zeros):
    dtype = schema.TensorType.INT8 if quantized else schema.TensorType.FLOAT32
    base = model_bytes(kind, int(np.prod(shape)), scales, zeros, 2, dtype)
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(base, 0))
    graph = model.subgraphs[0]
    graph.inputs = np.array([0], np.int32)
    target = list(shape)
    if kind != "CUMSUM":
        if keep:
            target[axis] = 1
        else:
            target.pop(axis)
    graph.tensors[0].shape = np.array(shape, np.int32)
    graph.tensors[2].shape = np.array(target, np.int32)
    graph.tensors[1].shape = np.array([1], np.int32)
    graph.tensors[1].type = schema.TensorType.INT32
    graph.tensors[1].quantization = None
    graph.tensors[1].buffer = 1
    buffer = schema.BufferT()
    buffer.data = np.array([axis - len(shape)], dtype="<i4").view(np.uint8)
    model.buffers.append(buffer)
    op = graph.operators[0]
    options = "CumsumOptions" if kind == "CUMSUM" else "ReducerOptions"
    op.builtinOptionsType = getattr(schema.BuiltinOptions, options)
    op.builtinOptions = getattr(schema, options + "T")()
    if kind == "CUMSUM":
        op.builtinOptions.exclusive = exclusive
        op.builtinOptions.reverse = reverse
    else:
        op.builtinOptions.keepDims = keep
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output()), target


def cases():
    for kind in KINDS:
        for quantized in (False, True):
            for config in range(3):
                shape = (2, 5, 7)
                scales = [(0.125, 1.0, 0.25), (0.2, 1.0, 0.3), (0.03125, 1.0, 0.011)][config]
                zeros = [(0, 0, 0), (-17, 0, 31), (127, 0, -128)][config]
                if kind in {"REDUCE_MAX", "REDUCE_MIN"}:
                    scales, zeros = (scales[0],) * 3, (zeros[0],) * 3
                raw = (np.arange(np.prod(shape), dtype=np.int32) * 73 + 19) % 256 - 128
                data = raw.astype(np.int8) if quantized else raw.astype(np.float32) * np.float32(scales[0])
                if not quantized:
                    data[:4] = [0, -0.0, 1e-7, -1e-7]
                for axis in range(3):
                    modes = [(True, x, r) for x in (False, True) for r in (False, True)] if kind == "CUMSUM" else [(k, False, False) for k in (False, True)]
                    for keep, exclusive, reverse in modes:
                        model, target = reduction_model(kind, shape, axis, keep, exclusive, reverse, quantized, scales, zeros)
                        interpreter = runtime.Interpreter.from_bytes(model, arena_size=65536)
                        interpreter.set_input(data.reshape(shape), 0)
                        interpreter.invoke()
                        expected = np.array(interpreter.get_output(0), copy=True).reshape(-1)
                        yield kind, quantized, shape, target, scales, zeros, axis, exclusive, reverse, data, expected


def main():
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")
    lines = [f"/* TFLite Micro {REFERENCE_VERSION}; test/generate_reduction_goldens.py. */",
             "#ifndef TIGRIS_REDUCTION_GOLDEN_H", "#define TIGRIS_REDUCTION_GOLDEN_H", "",
             "typedef struct {", "    uint8_t op, dtype, output_rank, axis, exclusive, reverse;",
             "    int32_t shapes[6];", "    float scales[2];", "    int32_t zeros[2];",
             "    uint32_t input_count, output_count;", "    const void *input, *output;",
             "} reduction_golden_t;", ""]
    records = []
    counts = {False: 0, True: 0}
    for number, case in enumerate(cases()):
        kind, quantized, shape, target, scales, zeros, axis, exclusive, reverse, data, expected = case
        counts[quantized] += len(expected)
        for label, array in (("x", data), ("y", expected)):
            ctype = "int8_t" if quantized else "float"
            lines.append(f"static const {ctype} reduction_{number}_{label}[] = {{")
            for offset in range(0, len(array), 12):
                values = [str(int(v)) if quantized else f"{float(v):.9e}f" for v in array[offset:offset + 12]]
                lines.append("    " + ", ".join(values) + ",")
            lines.append("};")
        dims = ", ".join(map(str, (*shape, *target, *([1] * (3 - len(target))))))
        records.append(f"    {{{KINDS[kind]}, {3 if quantized else 1}, {len(target)}, {axis}, {int(exclusive)}, {int(reverse)}, "
                       f"{{{dims}}}, {{{scales[0]:.9e}f, {scales[2]:.9e}f}}, {{{zeros[0]}, {zeros[2]}}}, "
                       f"{len(data)}, {len(expected)}, reduction_{number}_x, reduction_{number}_y}},")
    lines += ["static const reduction_golden_t reduction_goldens[] = {", *records, "};", "#endif", ""]
    Path(__file__).with_name("reduction_golden.h").write_text("\n".join(lines), encoding="ascii")
    print(f"Recorded {len(records)} cases: {counts[False]} float32 and {counts[True]} int8 outputs")


if __name__ == "__main__":
    main()

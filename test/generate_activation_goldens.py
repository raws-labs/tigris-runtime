"""Record activation outputs from the pinned TFLite Micro reference kernels."""

from importlib.metadata import version
from pathlib import Path

from generate_elementwise_goldens import (
    REFERENCE_VERSION, flatbuffers, model_bytes, np, runtime, schema,
)


KINDS = {"LEAKY_RELU": 21, "PRELU": 55, "ELU": 56, "LOG_SOFTMAX": 57,
         "L2_NORMALIZATION": 58, "L2_POOL_2D": 59}


def activation_model(kind, shape, target, quantized, scales, zeros, alpha, slope):
    dtype = schema.TensorType.INT8 if quantized else schema.TensorType.FLOAT32
    base = model_bytes(kind, int(np.prod(shape)), scales, zeros, 2 if alpha is not None else 1, dtype)
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(base, 0))
    graph = model.subgraphs[0]
    graph.inputs = np.array([0], np.int32)
    graph.tensors[0].shape = np.array(shape, np.int32)
    graph.tensors[-1].shape = np.array(target, np.int32)
    if alpha is not None:
        graph.tensors[1].shape = np.array(alpha.shape, np.int32)
        graph.tensors[1].buffer = 1
        buffer = schema.BufferT()
        buffer.data = alpha.reshape(-1).view(np.uint8)
        model.buffers.append(buffer)
    options = {"LEAKY_RELU": "LeakyReluOptions", "L2_NORMALIZATION": "L2NormOptions",
               "L2_POOL_2D": "Pool2DOptions", "LOG_SOFTMAX": "LogSoftmaxOptions"}.get(kind)
    if options:
        op = graph.operators[0]
        op.builtinOptionsType = getattr(schema.BuiltinOptions, options)
        op.builtinOptions = getattr(schema, options + "T")()
        if kind == "LEAKY_RELU":
            op.builtinOptions.alpha = slope
        if kind == "L2_POOL_2D":
            op.builtinOptions.padding = schema.Padding.SAME
            op.builtinOptions.strideH = op.builtinOptions.strideW = 2
            op.builtinOptions.filterHeight = op.builtinOptions.filterWidth = 3
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def cases():
    for kind in KINDS:
        for quantized in (False, True):
            if quantized and kind == "L2_POOL_2D":
                continue
            for config in range(4):
                shape = (2, 3, 4, (1, 4, 16, 64)[config])
                target = shape
                if kind == "L2_POOL_2D":
                    shape, target = (2, 5, 7, 3), (2, 3, 4, 3)
                scales = [(0.125, 0.25, 0.125), (0.03125, 0.09375, 0.0625),
                          (0.2, 0.07, 0.3), (0.003, 0.021, 0.011)][config]
                zeros = [(0, 0, 0), (-17, 31, -63), (19, -27, 11), (127, -128, 17)][config]
                if kind == "LOG_SOFTMAX":
                    scales, zeros = (*scales[:2], 1 / 16), (*zeros[:2], 127)
                if kind == "L2_NORMALIZATION":
                    scales, zeros = (*scales[:2], 1 / 128), (*zeros[:2], 0)
                raw = (np.arange(np.prod(shape), dtype=np.int32) * 73 + 19) % 256 - 128
                data = raw.astype(np.int8) if quantized else raw.astype(np.float32) * np.float32(scales[0])
                data[:shape[-1]] = zeros[0] if quantized else 0
                if not quantized and kind == "L2_NORMALIZATION":
                    data[shape[-1]:2 * shape[-1]] = np.float32(1e-9)
                alpha = None
                slope = (0.01, -0.5, 0.0, 1.75)[config]
                if kind == "PRELU":
                    alpha_shape = [(1, 1, 1, 1), (1, 1, 1, shape[-1]),
                                   (1, 3, 1, shape[-1]), shape][config]
                    raw_alpha = (np.arange(np.prod(alpha_shape), dtype=np.int32) * 31) % 256 - 128
                    alpha = (raw_alpha.astype(np.int8) if quantized else
                             raw_alpha.astype(np.float32) * np.float32(0.03125)).reshape(alpha_shape)
                interpreter = runtime.Interpreter.from_bytes(
                    activation_model(kind, shape, target, quantized, scales, zeros, alpha, slope),
                    arena_size=131072)
                interpreter.set_input(data.reshape(shape), 0)
                interpreter.invoke()
                expected = np.array(interpreter.get_output(0), copy=True).reshape(-1)
                yield kind, quantized, shape, target, scales, zeros, alpha, slope, data, expected


def main():
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")
    lines = [f"/* TFLite Micro {REFERENCE_VERSION}; test/generate_activation_goldens.py. */",
             "#ifndef TIGRIS_ACTIVATION_GOLDEN_H", "#define TIGRIS_ACTIVATION_GOLDEN_H", "",
             "typedef struct {", "    uint8_t op, dtype;", "    int32_t shapes[8];",
             "    float scales[3];", "    int32_t zeros[3];", "    float slope;",
             "    uint32_t input_count, output_count, alpha_count;",
             "    int32_t alpha_shape[4];", "    const void *input, *output, *alpha;",
             "} activation_golden_t;", ""]
    records = []
    counts = {False: 0, True: 0}
    for number, case in enumerate(cases()):
        kind, quantized, shape, target, scales, zeros, alpha, slope, data, expected = case
        counts[quantized] += len(expected)
        for label, array in (("x", data), ("y", expected), ("alpha", alpha)):
            if array is None:
                continue
            array = array.reshape(-1)
            ctype = "int8_t" if quantized else "float"
            lines.append(f"static const {ctype} activation_{number}_{label}[] = {{")
            for offset in range(0, len(array), 12):
                values = [str(int(v)) if quantized else f"{float(v):.9e}f" for v in array[offset:offset + 12]]
                lines.append("    " + ", ".join(values) + ",")
            lines.append("};")
        dims = ", ".join(map(str, (*shape, *target)))
        quant_scales = ", ".join(f"{value:.9e}f" for value in scales)
        quant_zeros = ", ".join(map(str, zeros))
        alpha_dims = ", ".join(map(str, alpha.shape if alpha is not None else (1,) * 4))
        alpha_name = f"activation_{number}_alpha" if alpha is not None else "NULL"
        records.append(f"    {{{KINDS[kind]}, {3 if quantized else 1}, {{{dims}}}, "
                       f"{{{quant_scales}}}, {{{quant_zeros}}}, {slope:.9e}f, "
                       f"{len(data)}, {len(expected)}, {alpha.size if alpha is not None else 0}, "
                       f"{{{alpha_dims}}}, activation_{number}_x, activation_{number}_y, {alpha_name}}},")
    lines += ["static const activation_golden_t activation_goldens[] = {", *records, "};", "#endif", ""]
    Path(__file__).with_name("activation_golden.h").write_text("\n".join(lines), encoding="ascii")
    print(f"Recorded {len(records)} cases: {counts[False]} float32 and {counts[True]} int8 outputs")


if __name__ == "__main__":
    main()

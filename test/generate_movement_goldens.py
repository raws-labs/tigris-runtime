"""Record data movement outputs from the pinned TFLite Micro kernels."""

from importlib.metadata import version
from pathlib import Path

from generate_elementwise_goldens import REFERENCE_VERSION, flatbuffers, np, runtime, schema

KINDS = {"GATHER": 66, "GATHER_ND": 67, "STRIDED_SLICE": 68, "MIRROR_PAD": 69,
         "REVERSE_V2": 70, "EMBEDDING_LOOKUP": 71, "DYNAMIC_UPDATE_SLICE": 72}


def configurations():
    for axis in range(3):
        shape = (2, 3, 4)
        yield "GATHER", shape, [np.array([[shape[axis] - 1, 0], [0, shape[axis] - 1]], np.int32)], {"axis": axis, "batchDims": 0}
    yield "GATHER", (2, 3, 4), [np.array([[2, 0], [1, 2]], np.int32)], {"axis": 1, "batchDims": 1}
    yield "GATHER", (2, 3, 4), [np.array(2, np.int32)], {"axis": 2, "batchDims": 0}
    yield "GATHER_ND", (2, 3, 4), [np.array([[1, 2], [0, 0], [1, 0]], np.int32)], {}
    yield "GATHER_ND", (2, 3, 4), [np.array([1, 2, 3], np.int32)], {}
    yield "GATHER_ND", (2, 3, 4), [np.array([[[1, 2], [0, 0]], [[0, 2], [1, 0]]], np.int32)], {}
    for negative in (False, True):
        shape = (2, 5, 7)
        starts = [1, 4, 6] if negative else [0, 0, 1]
        ends = [0, 0, 0] if negative else [2, 5, 7]
        steps = [-1, -2, -2] if negative else [1, 2, 3]
        yield "STRIDED_SLICE", shape, [np.array(x, np.int32) for x in (starts, ends, steps)], {"endMask": 7 if negative else 0}
    yield "STRIDED_SLICE", (2, 5, 7), [np.array(x, np.int32) for x in ([0, -1, 6], [2, 0, 0], [1, 1, -2])], {"endMask": 4, "shrinkAxisMask": 2}
    for symmetric in (False, True):
        yield "MIRROR_PAD", (2, 3, 4), [np.array([[1, 1], [2, 1], [0, 2]], np.int32)], {"mode": int(symmetric)}
    yield "REVERSE_V2", (2, 3, 4), [np.array([1, -1], np.int32)], {}
    yield "REVERSE_V2", (1, 2, 1, 2, 1, 3), [np.array([4, 5], np.int32)], {}
    yield "EMBEDDING_LOOKUP", (5, 7), [np.array([4, 0, 2, 4], np.int32)], {}
    for starts in ([99, -4, 1], [0, 1, 2]):
        yield "DYNAMIC_UPDATE_SLICE", (2, 3, 4), [np.array(starts, np.int64)], {}


def reference(kind, shape, constants, options, quantized, dynamic=False):
    raw = (np.arange(np.prod(shape), dtype=np.int32) * 73 + 19) % 256 - 128
    data = raw.astype(np.int8) if quantized else raw.astype(np.float32) * np.float32(0.125)
    data = data.reshape(shape)
    update = None
    if kind == "GATHER":
        axis, batch = options["axis"], options["batchDims"]
        idx = constants[0]
        target = (*shape[:axis], *idx.shape[batch:], *shape[axis + 1:])
        metadata = [axis, batch, idx.ndim, *idx.shape]
    elif kind == "GATHER_ND":
        idx = constants[0]
        target = (*idx.shape[:-1], *shape[idx.shape[-1]:])
        metadata = [idx.ndim, *idx.shape]
    elif kind == "STRIDED_SLICE":
        starts, ends, steps = constants
        spans = [slice(int(s), None if options["endMask"] & (1 << a) else int(e), int(t)).indices(d)
                 for a, (s, e, t, d) in enumerate(zip(starts, ends, steps, shape))]
        shrink = options.get("shrinkAxisMask", 0)
        spans = [(span[0], span[0] + 1, 1) if shrink & (1 << a) else span for a, span in enumerate(spans)]
        target = tuple(len(range(*span)) for a, span in enumerate(spans) if not shrink & (1 << a))
        metadata = [v for span in spans for v in span] + ([shrink] if shrink else [])
    elif kind == "MIRROR_PAD":
        target = tuple(d + int(sum(p)) for d, p in zip(shape, constants[0]))
        metadata = [options["mode"], *constants[0].reshape(-1).tolist()]
    elif kind == "REVERSE_V2":
        target = shape
        metadata = [sum(1 << (int(a) % len(shape)) for a in constants[0])]
    elif kind == "EMBEDDING_LOOKUP":
        target = (constants[0].size, *shape[1:])
        metadata = [0, 0, 1, constants[0].size]
    else:
        update = np.arange(4, dtype=data.dtype).reshape(1, 2, 2)
        target = shape
        clamped = np.clip(constants[0], 0, np.array(shape) - update.shape)
        metadata = [*clamped.tolist(), *update.shape]
    model = schema.ModelT()
    model.version = 3
    code = schema.OperatorCodeT()
    code.builtinCode = getattr(schema.BuiltinOperator, kind)
    code.deprecatedBuiltinCode = min(code.builtinCode, 127)
    code.version = 1
    model.operatorCodes = [code]
    model.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.tensors = []
    graph.inputs = []

    def tensor(array, constant=False):
        item = schema.TensorT()
        item.name = f"tensor_{len(graph.tensors)}".encode()
        item.shape = np.array(array.shape, np.int32)
        item.type = {np.dtype("int32"): schema.TensorType.INT32,
                     np.dtype("int64"): schema.TensorType.INT64,
                     np.dtype("int8"): schema.TensorType.INT8,
                     np.dtype("float32"): schema.TensorType.FLOAT32}[array.dtype]
        item.buffer = 0
        if array.dtype == np.int8:
            item.quantization = schema.QuantizationParametersT()
            item.quantization.scale = np.array([0.125], np.float32)
            item.quantization.zeroPoint = np.array([-17], np.int64)
        if constant:
            item.buffer = len(model.buffers)
            buf = schema.BufferT()
            buf.data = np.frombuffer(array.tobytes(), np.uint8)
            model.buffers.append(buf)
        graph.tensors.append(item)
        return len(graph.tensors) - 1

    x = tensor(data)
    graph.inputs = [x]
    args = [x]
    if update is not None:
        args.append(tensor(update))
        graph.inputs.append(args[-1])
    for value in constants:
        args.append(tensor(value, not dynamic))
        if dynamic:
            graph.inputs.append(args[-1])
    if kind == "EMBEDDING_LOOKUP":
        args.reverse()
    out = tensor(np.zeros(target, data.dtype))
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = np.array(args, np.int32)
    op.outputs = np.array([out], np.int32)
    option_class = {"GATHER": "GatherOptions", "STRIDED_SLICE": "StridedSliceOptions",
                    "MIRROR_PAD": "MirrorPadOptions"}.get(kind)
    if option_class:
        op.builtinOptionsType = getattr(schema.BuiltinOptions, option_class)
        op.builtinOptions = getattr(schema, option_class + "T")()
        for key, value in options.items():
            setattr(op.builtinOptions, key, value)
    graph.outputs = np.array([out], np.int32)
    graph.inputs = np.array(graph.inputs, np.int32)
    graph.operators = [op]
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    interpreter = runtime.Interpreter.from_bytes(bytes(builder.Output()), arena_size=65536)
    interpreter.set_input(data, 0)
    if update is not None:
        interpreter.set_input(update, 1)
    if dynamic:
        for i, value in enumerate(constants, 2 if update is not None else 1):
            interpreter.set_input(value, i)
    interpreter.invoke()
    result = np.array(interpreter.get_output(0), copy=True)
    assert result.shape == tuple(target), (kind, result.shape, target)
    if not dynamic and kind in {"GATHER", "GATHER_ND", "EMBEDDING_LOOKUP", "DYNAMIC_UPDATE_SLICE"}:
        runtime_result = reference(kind, shape, constants, options, quantized, dynamic=True)[2]
        np.testing.assert_array_equal(runtime_result, result)
    indices = constants[0] if kind in {"GATHER", "GATHER_ND", "EMBEDDING_LOOKUP"} else np.array([], np.int32)
    return data, update, result, metadata, indices


def main():
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")
    lines = [f"/* TFLite Micro {REFERENCE_VERSION}; test/generate_movement_goldens.py. */",
             "#ifndef TIGRIS_MOVEMENT_GOLDEN_H", "#define TIGRIS_MOVEMENT_GOLDEN_H", "",
             "typedef struct {", "    uint8_t op, dtype, rank, output_rank, update_rank, metadata_count;",
             "    int32_t shapes[18], metadata[19];", "    uint32_t input_count, output_count, update_count, index_count;",
             "    const void *input, *update, *output;", "    const int32_t *indices;", "} movement_golden_t;", ""]
    records = []
    outputs = 0
    for config in configurations():
        for quantized in (False, True):
            kind, shape, constants, options = config
            data, update, result, metadata, indices = reference(*config, quantized)
            number = len(records)
            pointers = []
            for label, array in (("x", data), ("u", update), ("y", result), ("indices", indices)):
                if array is None or array.size == 0:
                    pointers.append("NULL")
                    continue
                pointers.append(f"movement_{number}_{label}")
                integer = label == "indices" or quantized
                ctype = "int32_t" if label == "indices" else "int8_t" if quantized else "float"
                lines.append(f"static const {ctype} {pointers[-1]}[] = {{")
                array = array.reshape(-1)
                for start in range(0, array.size, 12):
                    values = [str(int(v)) if integer else f"{float(v):.9e}f" for v in array[start:start + 12]]
                    lines.append("    " + ", ".join(values) + ",")
                lines.append("};")
            ushape = () if update is None else update.shape
            dims = [v for dims in (shape, ushape, result.shape) for v in (*dims, *([1] * (6 - len(dims))))]
            fmt = lambda values: ", ".join(map(str, values))
            records.append(f"    {{{KINDS[kind]}, {3 if quantized else 1}, {len(shape)}, {result.ndim}, {len(ushape)}, {len(metadata)}, "
                           f"{{{fmt(dims)}}}, {{{fmt(metadata)}}}, {data.size}, {result.size}, {0 if update is None else update.size}, {indices.size}, "
                           f"{fmt(pointers)}}},")
            outputs += result.size
    lines += ["static const movement_golden_t movement_goldens[] = {", *records, "};", "#endif", ""]
    Path(__file__).with_name("movement_golden.h").write_text("\n".join(lines), encoding="ascii")
    print(f"Recorded {len(records)} cases with {outputs} output elements")


if __name__ == "__main__":
    main()

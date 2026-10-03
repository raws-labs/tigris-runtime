"""Record comparison, logical, selection, cast and sum reference outputs."""

from importlib.metadata import version
import math
from pathlib import Path

from generate_elementwise_goldens import REFERENCE_VERSION, flatbuffers, np, runtime, schema

KINDS = dict(zip(("EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL",
                  "LOGICAL_AND", "LOGICAL_OR", "LOGICAL_NOT", "SELECT_V2", "CAST", "ADD_N"), range(73, 84)))
TYPES = {np.dtype("float32"): (schema.TensorType.FLOAT32, 1),
         np.dtype("int8"): (schema.TensorType.INT8, 3),
         np.dtype("bool"): (schema.TensorType.BOOL, 9)}


def model_bytes(kind, arrays, target, dtype, scales, zeros):
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
    for i, array in enumerate([*arrays, np.zeros(target, dtype)]):
        slot = i if i < len(arrays) else 3
        tensor = schema.TensorT()
        tensor.name = f"tensor_{i}".encode()
        tensor.shape = np.array(array.shape, np.int32)
        tensor.type = TYPES[array.dtype][0]
        tensor.buffer = 0
        if array.dtype == np.int8:
            tensor.quantization = schema.QuantizationParametersT()
            tensor.quantization.scale = np.array([scales[slot]], np.float32)
            tensor.quantization.zeroPoint = np.array([zeros[slot]], np.int64)
        graph.tensors.append(tensor)
    op = schema.OperatorT()
    op.inputs = np.arange(len(arrays), dtype=np.int32)
    op.outputs = np.array([len(arrays)], np.int32)
    if kind == "CAST":
        op.builtinOptionsType = schema.BuiltinOptions.CastOptions
        op.builtinOptions = schema.CastOptionsT()
        op.builtinOptions.inDataType = TYPES[arrays[0].dtype][0]
        op.builtinOptions.outDataType = TYPES[np.dtype(dtype)][0]
    graph.inputs, graph.outputs, graph.operators = op.inputs, op.outputs, [op]
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def multiplier(scale):
    fraction, shift = math.frexp(float(np.float32(scale)))
    value = math.floor(fraction * (1 << 31) + 0.5)
    if value == 1 << 31:
        value //= 2
        shift += 1
    return (0, 0) if shift < -31 else (value, shift)


def cases():
    for kind in KINDS:
        for quantized in (False, True):
            if quantized and kind.startswith("LOGICAL"):
                continue
            for variant in (0, 1, 2, 4, *([3] if quantized and kind in list(KINDS)[:5] else [])):
                count = 3 if kind in {"SELECT_V2", "ADD_N"} else 1 if kind in {"CAST", "LOGICAL_NOT"} else 2
                shape = () if variant == 4 else (1, 4, 8, 2)
                shapes = [shape] * count
                if variant == 1 and count > 1 and kind != "ADD_N":
                    shapes[-1] = (1, 1, 1, 2)
                if variant == 2 and count > 1 and kind != "ADD_N":
                    shapes[0] = (1, 4, 1, 2)
                    shapes[-1] = (1, 1, 8, 1)
                    if kind == "SELECT_V2":
                        shape = (2, 2, 3, 4, 2)
                        shapes = [(2, 1, 3, 1, 2), (1, 2, 1, 4, 1), (1, 1, 1, 1, 2)]
                scales = [0.125, 0.25, 0.125, 0.25]
                zeros = [-17, 31, -17, -63]
                if variant == 2:
                    scales = [0.003, 0.07, 0.003, 0.1]
                if variant == 3:
                    scales = [2**-40, 0.9999999403953552, 0.125, 0.25]
                if kind == "SELECT_V2":
                    scales, zeros = [0.125] * 4, [-17] * 4
                if kind == "ADD_N":
                    scales[:3], zeros[:3] = [scales[0]] * 3, [zeros[0]] * 3
                if kind == "CAST":
                    scales[3], zeros[3] = 1.0, 0
                arrays = []
                for i, dims in enumerate(shapes):
                    raw = (np.arange(np.prod(dims, dtype=int), dtype=np.int32) * (73 + i * 8) + 19) % 256 - 128
                    boolean = kind.startswith("LOGICAL") or kind == "CAST" or kind == "SELECT_V2" and i == 0
                    array = raw > 0 if boolean else raw.astype(np.int8) if quantized else raw.astype(np.float32) * np.float32(0.125)
                    if not boolean and not quantized and kind not in {"ADD_N", "SELECT_V2"}:
                        array[:8] = np.resize([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 0.125], min(8, array.size))
                    arrays.append(array.reshape(dims))
                output_dtype = np.bool_ if kind.startswith("LOGICAL") or kind in list(KINDS)[:5] else np.int8 if quantized else np.float32
                interpreter = runtime.Interpreter.from_bytes(model_bytes(kind, arrays, shape, output_dtype, scales, zeros), arena_size=65536)
                for i, array in enumerate(arrays):
                    interpreter.set_input(array, i)
                interpreter.invoke()
                result = np.array(interpreter.get_output(0), copy=True)
                requant = [8, *multiplier(scales[0]), *multiplier(scales[1])]
                yield kind, arrays, result, scales, zeros, requant


def main():
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")
    lines = [f"/* TFLite Micro {REFERENCE_VERSION}; test/generate_bool_goldens.py. */",
             "#ifndef TIGRIS_BOOL_GOLDEN_H", "#define TIGRIS_BOOL_GOLDEN_H",
             "typedef struct {", "    uint8_t op, inputs, ranks[4], dtypes[4];",
             "    int32_t shapes[20], zeros[4], requant[5];", "    float scales[4];",
             "    uint32_t sizes[4];", "    const uint8_t *data[4];", "} bool_golden_t;"]
    records = []
    for number, (kind, inputs, output, scales, zeros, requant) in enumerate(cases()):
        arrays = [*inputs, *([None] * (3 - len(inputs))), output]
        names, ranks, dtypes, shapes, sizes = [], [], [], [], []
        for i, array in enumerate(arrays):
            names.append(f"bool_{number}_{i}" if array is not None else "NULL")
            ranks.append(array.ndim if array is not None else 1)
            dtypes.append(TYPES[array.dtype][1] if array is not None else 9)
            shapes.extend([*array.shape, *([1] * (5 - array.ndim))] if array is not None else [1] * 5)
            sizes.append(array.nbytes if array is not None else 0)
            if array is not None:
                lines.append(f"static const uint8_t {names[-1]}[] = {{")
                data = array.tobytes()
                for start in range(0, len(data), 24):
                    lines.append("    " + ", ".join(str(v) for v in data[start:start + 24]) + ",")
                lines.append("};")
        fmt = lambda values: "{" + ", ".join(map(str, values)) + "}"
        records.append("    {" + f"{KINDS[kind]}, {len(inputs)}, " + ", ".join(fmt(v) for v in (ranks, dtypes, shapes, zeros, requant,
                        [f"{float(np.float32(s)):.9e}f" for s in scales], sizes, names)) + "},")
    lines += ["static const bool_golden_t bool_goldens[] = {", *records, "};", "#endif", ""]
    Path(__file__).with_name("bool_golden.h").write_text("\n".join(lines), encoding="ascii")
    print(f"Recorded {len(records)} reference cases")


if __name__ == "__main__":
    main()

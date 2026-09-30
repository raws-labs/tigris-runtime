"""Record resize outputs and source bands from the pinned TFLite Micro wheel."""

from importlib.metadata import version
from pathlib import Path
import math

from generate_elementwise_goldens import (
    REFERENCE_VERSION, flatbuffers, model_bytes, np, runtime, schema,
)


def resize_model(shape, target, linear, convention, quantized):
    dtype = schema.TensorType.INT8 if quantized else schema.TensorType.FLOAT32
    base = model_bytes("DIV", 16, (0.125,) * 3, (-17,) * 3, 2, dtype)
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(base, 0))
    kind = "RESIZE_BILINEAR" if linear else "RESIZE_NEAREST_NEIGHBOR"
    model.operatorCodes[0].builtinCode = getattr(schema.BuiltinOperator, kind)
    model.operatorCodes[0].deprecatedBuiltinCode = getattr(schema.BuiltinOperator, kind)
    graph = model.subgraphs[0]
    graph.inputs = np.array([0], dtype=np.int32)
    graph.tensors[0].shape = np.array(shape, np.int32)
    graph.tensors[1].shape = np.array([2], np.int32)
    graph.tensors[1].type = schema.TensorType.INT32
    graph.tensors[1].buffer = 1
    graph.tensors[1].quantization = None
    graph.tensors[2].shape = np.array(target, np.int32)
    size = schema.BufferT()
    size.data = np.array(target[1:3], dtype="<i4").view(np.uint8)
    model.buffers.append(size)
    op = graph.operators[0]
    name = "ResizeBilinearOptions" if linear else "ResizeNearestNeighborOptions"
    op.builtinOptionsType = getattr(schema.BuiltinOptions, name)
    op.builtinOptions = getattr(schema, name + "T")()
    op.builtinOptions.alignCorners = convention == 2
    op.builtinOptions.halfPixelCenters = convention == (0 if linear else 3)
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def bands(ih, oh, linear, convention, quantized):
    align = convention == 2 and oh > 1
    ni, no = (ih - 1, oh - 1) if align else (ih, oh)
    scale = np.float32(ni) / np.float32(no)
    scale_10 = (1024 * ni + no // 2) // no
    starts, ends = [], []
    for row in range(oh):
        if linear and quantized:
            pos = row * scale_10 + (scale_10 // 2 - 512 if convention == 0 else 0)
            lo, hi = max(0, math.trunc(pos / 1024)), min(ih - 1, math.trunc((pos + 1023) / 1024))
        elif linear:
            pos = (np.float32(row + 0.5) * scale - np.float32(0.5) if convention == 0
                   else np.float32(row) * scale)
            lo, hi = max(0, math.floor(pos)), min(ih - 1, math.ceil(pos))
        else:
            pos = np.float32(row + (0.5 if convention == 3 else 0.0)) * scale
            lo = hi = min(ih - 1, math.floor(float(pos) + (0.5 if convention == 2 else 0.0)))
        if not 0 <= lo <= hi < ih:
            raise ValueError((ih, oh, row, convention, lo, hi))
        starts.append(lo)
        ends.append(hi + 1)
    return starts, ends


def cases():
    shapes = [(3, 4, 5, 7), (7, 9, 3, 4), (3, 7, 8, 2),
              (1, 4, 5, 1), (5, 7, 1, 1), (2, 3, 17, 9)]
    for linear, conventions in ((False, (1, 2, 3)), (True, (0, 1, 2))):
        for convention in conventions:
            for ih, iw, oh, ow in shapes:
                shape, target = (2, ih, iw, 2), (2, oh, ow, 2)
                for quantized in (False, True):
                    raw = ((np.arange(np.prod(shape), dtype=np.int32) * 73 + 19) % 256 - 128)
                    data = raw.astype(np.int8) if quantized else raw.astype(np.float32) * np.float32(0.125)
                    interpreter = runtime.Interpreter.from_bytes(
                        resize_model(shape, target, linear, convention, quantized), arena_size=65536)
                    interpreter.set_input(data.reshape(shape), 0)
                    interpreter.invoke()
                    expected = np.array(interpreter.get_output(0), copy=True).reshape(-1)
                    yield linear, convention, quantized, shape, target, data, expected, bands(
                        ih, oh, linear, convention, quantized)


def main():
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")
    lines = [f"/* TFLite Micro {REFERENCE_VERSION}; test/generate_resize_goldens.py. */",
             "#ifndef TIGRIS_RESIZE_GOLDEN_H", "#define TIGRIS_RESIZE_GOLDEN_H", "",
             "typedef struct {", "    uint8_t linear, convention, dtype;",
             "    int32_t shapes[8];", "    uint32_t input_count, output_count;",
             "    const void *input, *output;", "    uint16_t starts[17], ends[17];",
             "} resize_golden_t;", ""]
    records = []
    counts = {False: 0, True: 0}
    for number, case in enumerate(cases()):
        linear, convention, quantized, shape, target, data, expected, (starts, ends) = case
        counts[quantized] += len(expected)
        for label, array in (("x", data), ("y", expected)):
            ctype = "int8_t" if quantized else "float"
            lines.append(f"static const {ctype} resize_{number}_{label}[] = {{")
            for offset in range(0, len(array), 12):
                values = [str(int(v)) if quantized else f"{float(v):.9e}f" for v in array[offset:offset + 12]]
                lines.append("    " + ", ".join(values) + ",")
            lines.append("};")
        dims = ", ".join(map(str, (*shape, *target)))
        first, end = ", ".join(map(str, starts)), ", ".join(map(str, ends))
        records.append(f"    {{{int(linear)}, {convention}, {3 if quantized else 1}, {{{dims}}}, "
                       f"{len(data)}, {len(expected)}, resize_{number}_x, resize_{number}_y, "
                       f"{{{first}}}, {{{end}}}}},")
    lines += ["static const resize_golden_t resize_goldens[] = {", *records, "};", "#endif", ""]
    Path(__file__).with_name("resize_golden.h").write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {len(records)} vectors: {counts[True]} int8 and {counts[False]} float outputs")


if __name__ == "__main__":
    main()

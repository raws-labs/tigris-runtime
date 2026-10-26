"""Write int8 reference vectors using the pinned TFLite Micro wheel."""

from importlib.metadata import version
import math
from pathlib import Path

import flatbuffers
import numpy as np
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.tensorflow.lite.python import schema_py_generated as schema


REFERENCE_VERSION = "0.dev20260925222358"
OPERATORS = ("ABS", "RSQRT", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM", "DIV")
FLOAT_OPERATORS = (*OPERATORS, "NEG", "EXP", "LOG", "SQRT", "SQUARE",
                   "FLOOR", "CEIL", "ROUND", "SIN", "COS", "FLOOR_DIV", "FLOOR_MOD")


def check_reference():
    """Refuse to record with any TFLite Micro but the pinned one."""
    if version("tflite-micro") != REFERENCE_VERSION:
        raise RuntimeError(f"Requires tflite-micro=={REFERENCE_VERSION}")


def preamble(script, guard, *notes):
    """A header's first lines: its source, any notes, and its include guard."""
    return [f"/* TFLite Micro {REFERENCE_VERSION}; test/{script}. */", *notes,
            f"#ifndef {guard}", f"#define {guard}"]


def operator_model(operation):
    """A model with one subgraph and one operator code, `operation`."""
    model = schema.ModelT()
    model.version = 3
    code = schema.OperatorCodeT()
    code.builtinCode = getattr(schema.BuiltinOperator, operation)
    code.deprecatedBuiltinCode = min(code.builtinCode, 127)
    code.version = 1
    model.operatorCodes = [code]
    model.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.tensors = []
    model.subgraphs = [graph]
    return model, graph


def serialize(model):
    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def c_array(ctype, name, values, per_row, integer=True):
    """A C array definition, per_row values to a line."""
    lines = [f"static const {ctype} {name}[] = {{"]
    for start in range(0, len(values), per_row):
        row = values[start:start + per_row]
        lines.append("    " + ", ".join(str(int(v)) if integer else f"{float(v):.9e}f"
                                        for v in row) + ",")
    return lines + ["};"]


def model_bytes(operation, count, scales, zeros, inputs, dtype=schema.TensorType.INT8):
    model, graph = operator_model(operation)
    graph.name = operation.encode()
    for index in range(inputs + 1):
        slot = index if index < inputs else 2
        tensor = schema.TensorT()
        tensor.name = f"tensor_{index}".encode()
        tensor.shape = np.array([1, count // 16, 16, 1], dtype=np.int32)
        tensor.type = dtype
        tensor.buffer = 0
        if dtype == schema.TensorType.INT8:
            tensor.quantization = schema.QuantizationParametersT()
            tensor.quantization.scale = np.array([scales[slot]], dtype=np.float32)
            tensor.quantization.zeroPoint = np.array([zeros[slot]], dtype=np.int64)
        graph.tensors.append(tensor)
    op = schema.OperatorT()
    op.opcodeIndex = 0
    op.inputs = np.arange(inputs, dtype=np.int32)
    op.outputs = np.array([inputs], dtype=np.int32)
    if operation == "DIV":
        op.builtinOptionsType = schema.BuiltinOptions.DivOptions
        op.builtinOptions = schema.DivOptionsT()
        op.builtinOptions.fusedActivationFunction = schema.ActivationFunctionType.NONE
    graph.inputs = op.inputs
    graph.outputs = op.outputs
    graph.operators = [op]
    return serialize(model)


def high_mul(a, b):
    if a == b == -(1 << 31):
        return (1 << 31) - 1
    product = a * b
    nudged = product + ((1 << 30) if product >= 0 else 1 - (1 << 30))
    return (1 if nudged >= 0 else -1) * (abs(nudged) // (1 << 31))


def rescale(value, shift):
    return min((1 << 31) - 1, max(-(1 << 31), value * (1 << shift)))


def div_reference(a, b, scales, zeros):
    a, b = int(a) - zeros[0], int(b) - zeros[1]
    if b == 0:
        raise ValueError("DIV denominator is zero")
    if b < 0:
        a, b = -a, -b
    leading = 32 - b.bit_length()
    normalized = (b << leading) - (1 << 31)
    half = (normalized + (1 << 31)) // 2
    inverse = 1515870810 + high_mul(half, -1010580540)
    for _ in range(3):
        error = (1 << 29) - high_mul(half, inverse)
        inverse += rescale(high_mul(inverse, error), 2)
    inverse = rescale(inverse, 1)
    headroom = 31 - (a if a >= 0 else ~a).bit_length()
    quotient = high_mul(a * (1 << headroom), inverse)
    sa, sb, sy = map(np.float32, scales)
    fraction, shift = math.frexp(float(sa / (sb * sy)))
    multiplier = math.floor(fraction * (1 << 31) + 0.5)
    if multiplier == 1 << 31:
        multiplier //= 2
        shift += 1
    if shift < -31:
        multiplier, shift = 0, 0
    exponent = 31 - leading + headroom - shift
    if exponent < 0:
        raise ValueError("DIV reference requires a nonnegative rounding exponent")
    value = high_mul(quotient, multiplier)
    mask = (1 << exponent) - 1
    remainder = value & mask
    threshold = (mask >> 1) + (value < 0)
    rounded = (value >> exponent) + (remainder > threshold)
    return min(127, max(-128, rounded + zeros[2])), exponent


def cases():
    for operation in OPERATORS:
        for config in range(6 if operation == "DIV" else 3):
            scales = [(0.125, 0.25, 0.125), (0.03125, 0.09375, 0.0625),
                      (0.2, 0.07, 0.3), (2.0**-32, 1.0, 1.0),
                      (2.0**-40, 1.0, 1.0), (2.0**35, 1.0, 1.0)][config]
            zeros = [(0, 0, 0), (-17, 31, -63), (19, -27, 11),
                     (0, 0, 0), (0, 0, 0), (0, 0, 0)][config]
            if operation in {"MAXIMUM", "MINIMUM"}:
                scales = (scales[0],) * 3
                zeros = (zeros[0],) * 3
            inputs = 1 if operation in {"ABS", "RSQRT"} else 2
            count = 256 if inputs == 1 else 768
            if config >= 3:
                count = 64
            index = np.arange(count, dtype=np.int32)
            first = index % 256 - 128
            second = ((index * 73 + 19) % 256) - 128
            if inputs == 2 and config < 3:
                second[256:512] = np.resize(np.array([-128, -1, 0, 1, 127]), 256)
                second[512:] = first[512:]
            if operation == "RSQRT":
                first = zeros[0] + index % (128 - zeros[0])
            if operation == "DIV":
                if config >= 3:
                    first = np.resize(np.array([0, 1, -1, 2, -2, 127, -128, 19]), count)
                    second = np.repeat(np.array([1, -1, 2, -2, 19, -19, 127, -128]), 8)
                if config == 5:
                    first = np.resize(np.array([0, 1, -1, 1]), count)
                    second = np.repeat(np.array([127, -127, -128, 127]), 16)
                second[second == zeros[1]] = zeros[1] + 1
            arrays = [first.astype(np.int8)]
            if inputs == 2:
                arrays.append(second.astype(np.int8))
            interpreter = runtime.Interpreter.from_bytes(
                model_bytes(operation, count, scales, zeros, inputs), arena_size=65536)
            for input_index, array in enumerate(arrays):
                interpreter.set_input(array.reshape(1, count // 16, 16, 1), input_index)
            interpreter.invoke()
            expected = np.array(interpreter.get_output(0), copy=True).reshape(-1)
            wide = []
            if operation == "DIV":
                for index, (a, b) in enumerate(zip(*arrays)):
                    calculated, exponent = div_reference(a, b, scales, zeros)
                    if exponent >= 32:
                        expected[index] = calculated
                        wide.append((index, int(a), int(b), exponent))
                    elif calculated != expected[index]:
                        raise AssertionError((config, index, int(a), int(b), calculated,
                                              int(expected[index]), exponent))
                print(f"DIV config {config}: {len(wide)} wide rounding entries: {wide}")
            yield operation, config, scales, zeros, arrays, expected, wide


def main():
    check_reference()
    lines = [*preamble("generate_elementwise_goldens.py", "TIGRIS_ELEMENTWISE_S8_GOLDEN_H",
                       "/* Listed DIV entries use wide mask/remainder/threshold rounding because",
                       " * the reference's int32 rounding shift is >= 32; all other bytes are from the wheel. */"),
             "",
             "typedef struct {", "    uint8_t operation;", "    float scales[3];",
             "    int32_t zeros[3];", "    uint32_t count;",
             "    const int8_t *input_a;", "    const int8_t *input_b;",
             "    const int8_t *output;", "} elementwise_golden_t;", ""]
    records = []
    for operation, config, scales, zeros, arrays, expected, wide in cases():
        name = f"golden_{operation.lower()}_{config}"
        lines += c_array("int8_t", name + "_a", arrays[0], 16) + [""]
        if len(arrays) == 2:
            lines += c_array("int8_t", name + "_b", arrays[1], 16) + [""]
        if wide:
            lines.append("/* Wide DIV entries: index, raw numerator, raw denominator, exponent.")
            lines += [f" * {index}, {a}, {b}, {exponent}" for index, a, b, exponent in wide]
            lines.append(" */")
        lines += c_array("int8_t", name + "_y", expected, 16) + [""]
        c_scales = ", ".join(f"{float(np.float32(v)):.9e}f" for v in scales)
        c_zeros = ", ".join(str(v) for v in zeros)
        records.append(f"    {{TIGRIS_OP_{operation}, {{{c_scales}}}, {{{c_zeros}}}, "
                       f"{len(expected)}u, {name}_a, "
                       f"{name + '_b' if len(arrays) == 2 else 'NULL'}, {name}_y}},")
    lines += ["static const elementwise_golden_t elementwise_goldens[] = {",
              *records, "};", "", "#endif", ""]
    destination = Path(__file__).with_name("elementwise_s8_golden.h")
    destination.write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {len(records)} reference cases to {destination.name}")
    write_float_cases()


def write_float_cases():
    lines = [*preamble("generate_elementwise_goldens.py", "TIGRIS_ELEMENTWISE_F32_GOLDEN_H"), "",
             "typedef struct {", "    uint8_t operation;", "    uint8_t inputs;",
             "    float a[16];", "    float b[16];", "    float y[16];",
             "} elementwise_float_golden_t;", "",
             "static const elementwise_float_golden_t elementwise_float_goldens[] = {"]
    for operation in FLOAT_OPERATORS:
        inputs = 2 if operation in {"SQUARED_DIFFERENCE", "DIV", "MAXIMUM", "MINIMUM",
                                   "FLOOR_DIV", "FLOOR_MOD"} else 1
        a = np.array([-4.5, -3.5, -2.5, -1.5, -0.5, -0.0, 0.0, 0.5,
                      1.5, 2.5, 3.5, 4.5, -7.25, 7.25, -0.125, 0.125], dtype=np.float32)
        if operation in {"RSQRT", "SQRT", "LOG"}:
            a = np.abs(a) + np.float32(0.25)
        b = np.resize(np.array([2, -2, 3, -3, 0.25, -0.25, 1, -1], dtype=np.float32), 16)
        interpreter = runtime.Interpreter.from_bytes(
            model_bytes(operation, 16, (), (), inputs, schema.TensorType.FLOAT32),
            arena_size=65536)
        interpreter.set_input(a.reshape(1, 1, 16, 1), 0)
        if inputs == 2:
            interpreter.set_input(b.reshape(1, 1, 16, 1), 1)
        interpreter.invoke()
        expected = np.array(interpreter.get_output(0), copy=True).reshape(-1)
        lines.append(f"    {{TIGRIS_OP_{operation}, {inputs},")
        for values in (a, b, expected):
            lines.append("        {" + ", ".join(f"{float(v):.9e}f" for v in values) + "},")
        lines.append("    },")
    lines += ["};", "", "#endif", ""]
    destination = Path(__file__).with_name("elementwise_f32_golden.h")
    destination.write_text("\n".join(lines), encoding="ascii")
    print(f"Wrote {len(FLOAT_OPERATORS)} float reference cases to {destination.name}")


if __name__ == "__main__":
    main()

"""Embed compiled SVDF plans with TFLite Micro's outputs over consecutive runs.

Each argument is NAME=PLAN.tgrs:GOLDEN.npz, a plan compiled from one of the
compiler's TFLite corpus cases and that case's recorded inputs and outputs:

    python test/generate_svdf_goldens.py float_svdf=float_svdf.tgrs:float_svdf.npz ...
"""

import sys
from pathlib import Path

import numpy as np

C_TYPES = {np.dtype("float32"): "float", np.dtype("int8"): "int8_t"}


def array(name, values):
    values = np.asarray(values).reshape(-1)
    kind = C_TYPES[values.dtype]
    text = ", ".join(f"{v:.9e}f" if kind == "float" else str(int(v)) for v in values)
    return f"static const {kind} {name}[] = {{{text}}};\n"


def main(arguments):
    body, entries = [], []
    for argument in arguments:
        name, paths = argument.split("=", 1)
        plan_path, golden_path = paths.split(":", 1)
        plan = Path(plan_path).read_bytes()
        golden = np.load(golden_path)
        x, y = golden["input_0"], golden["output_0"]
        body.append(f"static const uint8_t {name}_plan[] = {{{', '.join(map(str, plan))}}};\n")
        body.append(array(f"{name}_input", x))
        body.append(array(f"{name}_output", y))
        entries.append(f"    {{{name}_plan, sizeof({name}_plan), {name}_input, {name}_output, "
                       f"{x[0].nbytes}u, {y[0].nbytes}u, {len(x)}u}},\n")
    print("/* test/generate_svdf_goldens.py; outputs recorded from TFLite Micro. */")
    print("#ifndef TIGRIS_SVDF_GOLDEN_H\n#define TIGRIS_SVDF_GOLDEN_H\n")
    print("typedef struct {\n    const uint8_t *plan;\n    uint32_t plan_size;\n"
          "    const void *input, *output;\n    uint32_t input_bytes, output_bytes, runs;\n"
          "} svdf_golden_t;\n")
    print("".join(body))
    print("static const svdf_golden_t svdf_goldens[] = {\n" + "".join(entries) + "};\n")
    print("#endif")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Embed compiled plans with TFLite Micro's outputs over consecutive runs.

Each argument is NAME=PLAN.tgrs:GOLDEN.npz, a plan compiled from one of the
compiler's TFLite corpus cases and that case's recorded inputs and outputs.
Per run, the model inputs and outputs are stored one after another in plan
order, as bytes:

    python test/generate_state_goldens.py float_svdf=float_svdf.tgrs:float_svdf.npz ...
"""

import sys
from pathlib import Path

import numpy as np



def array(name, runs):
    """Per run, the arrays' bytes one after another."""
    data = b"".join(np.ascontiguousarray(a[run]).tobytes() for run in range(len(runs[0])) for a in runs)
    return f"static const uint8_t {name}[] = {{{', '.join(map(str, data))}}};\n"


def main(arguments):
    body, entries = [], []
    for argument in arguments:
        name, paths = argument.split("=", 1)
        plan_path, golden_path = paths.split(":", 1)
        plan = Path(plan_path).read_bytes()
        golden = np.load(golden_path)
        order = sorted(golden.files, key=lambda k: int(k.rsplit("_", 1)[1]))
        xs = [golden[k] for k in order if k.startswith("input_")]
        ys = [golden[k] for k in order if k.startswith("output_")]
        body.append(f"static const uint8_t {name}_plan[] = {{{', '.join(map(str, plan))}}};\n")
        body.append(array(f"{name}_input", xs))
        body.append(array(f"{name}_output", ys))
        entries.append(f"    {{{name}_plan, sizeof({name}_plan), {name}_input, {name}_output, "
                       f"{sum(x[0].nbytes for x in xs)}u, {sum(y[0].nbytes for y in ys)}u, "
                       f"{len(xs[0])}u}},\n")
    print("/* test/generate_state_goldens.py; outputs recorded from TFLite Micro. */")
    print("#ifndef TIGRIS_STATE_GOLDEN_H\n#define TIGRIS_STATE_GOLDEN_H\n")
    print("typedef struct {\n    const uint8_t *plan;\n    uint32_t plan_size;\n"
          "    const void *input, *output;\n    uint32_t input_bytes, output_bytes, runs;\n"
          "} state_golden_t;\n")
    print("".join(body))
    print("static const state_golden_t state_goldens[] = {\n" + "".join(entries) + "};\n")
    print("#endif")


if __name__ == "__main__":
    main(sys.argv[1:])

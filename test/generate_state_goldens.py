"""Embed compiled plans with TFLite Micro's outputs over consecutive runs.

Each argument is NAME=PLAN.tgrs:GOLDEN.npz, a plan compiled from one of the
compiler's TFLite corpus cases and that case's recorded inputs and outputs.
Per run, the model inputs and outputs are stored one after another in plan
order, as bytes:

    python test/generate_state_goldens.py float_svdf=float_svdf.tgrs:float_svdf.npz ...

The plans are frozen compiler outputs at schemas 9, 10 and 11 and are not
recompiled: the schema 9 and 10 plans are the tests of those released formats
for LSTM, SVDF, detection, If and While. RECIPES names, per plan, the compiler
revision and flags that emit it, and the header lists them.

The recorded outputs are glibc 2.39's. With --cr, each argument is
NAME=GOLDEN.npz:GOLDEN_cr.npz, the case's recording and its recording on a
correctly rounded libm, and the header holds the outputs of the cases where
the two differ (test/libm_rounding.h):

    python test/generate_state_goldens.py --cr float_lstm=float_lstm.npz:float_lstm_cr.npz ...
"""

import sys
from pathlib import Path

import numpy as np

RECIPES = {
    **dict.fromkeys(("float_svdf", "float_svdf_rank2_relu", "svdf", "svdf_batch"),
                    "tigris develop 91b90aa, -m 256K"),
    **dict.fromkeys(("float_lstm", "float_lstm_time_major_clip"),
                    "tigris develop c4f824e, -m 256K"),
    **dict.fromkeys(("lstm", "lstm_time_major_clip"), "tigris develop a895a48, -m 256K"),
    **dict.fromkeys(("float_if", "float_while", "float_if_if"), "tigris develop 6bb9378, -m 256K"),
    **dict.fromkeys(("float_detection_fast", "float_detection_regular", "detection_fast",
                     "detection_regular"), "tigris develop b7b5aeb, -m 256K"),
    **dict.fromkeys(("float_if_int32", "select_int32_condition"),
                    "tigris develop ae986a0, -m 256K"),
    **dict.fromkeys(("if_conv", "while_conv"), "tigris develop e86f70b, -m 256K"),
    **dict.fromkeys(("float_if_through", "float_while_through", "float_while_if_through",
                     "float_while_counter", "float_while_countdown", "gather_index_arithmetic"),
                    "tigris v0.12.1, -m 256K --keep-names"),
    "float_while_through_tiled": "tigris v0.12.1, -m 256 --keep-names",
    **dict.fromkeys(("float_svdf_lz4", "lstm_lz4"), "tigris v0.13.0, -m 256K -c lz4"),
    "if_conv_chained": "tigris v0.13.0, -m 256",
}
# The libm functions a float case calls, as named in libm_rounding.h.
LIBM_CALLS = dict.fromkeys(("float_lstm", "float_lstm_time_major_clip", "float_if_through"), "LIBM_TANHF")


def array(name, runs):
    """Per run, the arrays' bytes one after another."""
    data = b"".join(np.ascontiguousarray(a[run]).tobytes() for run in range(len(runs[0])) for a in runs)
    return f"static const uint8_t {name}[] = {{{', '.join(map(str, data))}}};\n"


def main(arguments):
    body, entries, recipes = [], [], []
    for argument in arguments:
        name, paths = argument.split("=", 1)
        recipes.append(f" *   {name:<28}{RECIPES[name]}\n")
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
    print("/* test/generate_state_goldens.py; outputs recorded from TFLite Micro.\n"
          " * Frozen plans and their origin (compiler revision, flags):\n"
          + "".join(recipes) + " */")
    print("#ifndef TIGRIS_STATE_GOLDEN_H\n#define TIGRIS_STATE_GOLDEN_H\n")
    print("typedef struct {\n    const uint8_t *plan;\n    uint32_t plan_size;\n"
          "    const void *input, *output;\n    uint32_t input_bytes, output_bytes, runs;\n"
          "} state_golden_t;\n")
    print("".join(body))
    print("static const state_golden_t state_goldens[] = {\n" + "".join(entries) + "};\n")
    print("#endif")


def main_correctly_rounded(arguments):
    body, entries = [], []
    for argument in arguments:
        name, paths = argument.split("=", 1)
        legacy, correct = (np.load(path) for path in paths.split(":", 1))
        if sorted(legacy.files) != sorted(correct.files) or any(
                not np.array_equal(legacy[k], correct[k]) for k in legacy.files if k.startswith("input_")):
            raise ValueError(f"{name}: the two recordings differ in their inputs")
        order = sorted((k for k in legacy.files if k.startswith("output_")), key=lambda k: int(k[7:]))
        if all(legacy[k].tobytes() == correct[k].tobytes() for k in order):
            continue
        if name not in LIBM_CALLS:
            raise ValueError(f"{name}: outputs differ but the case calls none of the probed functions")
        body.append(array(f"{name}_output_cr", [correct[k] for k in order]))
        entries.append(f"    {{{name}_output, {LIBM_CALLS[name]}, {name}_output_cr}},\n")
    print("/* test/generate_state_goldens.py --cr; outputs recorded from TFLite Micro on a\n"
          " * correctly rounded libm, for the cases where they differ from state_golden.h. */")
    print("#ifndef TIGRIS_STATE_GOLDEN_CR_H\n#define TIGRIS_STATE_GOLDEN_CR_H\n")
    print("".join(body))
    print("static const libm_golden_t state_goldens_cr[] = {\n" + "".join(entries) + "};\n")
    print("#endif")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cr"]:
        main_correctly_rounded(sys.argv[2:])
    else:
        main(sys.argv[1:])

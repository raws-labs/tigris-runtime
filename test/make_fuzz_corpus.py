#!/usr/bin/env python3
"""Create deterministic loader-fuzz seeds for every cheap rejection path."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


PLAN_SIZE = 100
SECTION_DIR_OFFSET = 48
SCHEMA_VERSION = 2
CURRENT_SCHEMA_VERSION = 4
REQUIRED_SECTIONS = (1, 2, 5, 6, 7)

UNARY_PLAN_SIZE = 238
UNARY_TENSORS_OFFSET = 112
UNARY_OPS_OFFSET = 144
UNARY_STAGES_OFFSET = 184
UNARY_INDEX_OFFSET = 212
UNARY_SHAPES_OFFSET = 228
UNARY_STRINGS_OFFSET = 236


def valid_plan() -> bytearray:
    """Return the smallest plan accepted by the loader."""
    plan = bytearray(PLAN_SIZE)
    struct.pack_into(
        "<4sIII",
        plan,
        0,
        b"TGRS",
        SCHEMA_VERSION,
        PLAN_SIZE,
        SECTION_DIR_OFFSET,
    )
    for index, section_type in enumerate(REQUIRED_SECTIONS):
        struct.pack_into(
            "<II",
            plan,
            SECTION_DIR_OFFSET + index * 8,
            section_type,
            PLAN_SIZE - 1,
        )
    return plan


def valid_unary_plan() -> bytearray:
    """Return a complete schema-v4 Relu plan with executable semantics."""
    plan = bytearray(UNARY_PLAN_SIZE)
    struct.pack_into(
        "<4sIII",
        plan,
        0,
        b"TGRS",
        CURRENT_SCHEMA_VERSION,
        UNARY_PLAN_SIZE,
        SECTION_DIR_OFFSET,
    )
    struct.pack_into("<HHH", plan, 16, 2, 1, 1)  # tensors, ops, stages
    struct.pack_into("<HBB", plan, 36, 2, 1, 1)  # model I/O pool offset/counts

    sections = (
        (1, UNARY_TENSORS_OFFSET),
        (2, UNARY_OPS_OFFSET),
        (3, UNARY_STAGES_OFFSET),
        (5, UNARY_INDEX_OFFSET),
        (6, UNARY_SHAPES_OFFSET),
        (7, UNARY_STRINGS_OFFSET),
    )
    for index, (section_type, offset) in enumerate(sections):
        struct.pack_into(
            "<II", plan, SECTION_DIR_OFFSET + index * 8, section_type, offset
        )

    # Two [2, 3] float tensors, one public input and one public output.
    for index, flag in enumerate((2, 4)):
        offset = UNARY_TENSORS_OFFSET + index * 16
        struct.pack_into("<IIHBBB", plan, offset, 0, 24, 0, 2, 1, flag)
        struct.pack_into("<H", plan, offset + 13, 0xFFFF)

    # Relu(input=0, output=1), with no weight or bias references.
    struct.pack_into("<IBBBBHH", plan, UNARY_OPS_OFFSET, 0, 3, 1, 1, 0, 0, 1)
    struct.pack_into("<HH", plan, UNARY_OPS_OFFSET + 30, 0xFFFF, 0xFFFF)

    # One standalone stage containing the op and its boundary tensors.
    struct.pack_into(
        "<HHHHHHH",
        plan,
        UNARY_STAGES_OFFSET + 4,
        4,
        1,
        5,
        1,
        6,
        1,
        0xFFFF,
    )
    struct.pack_into("<H", plan, UNARY_STAGES_OFFSET + 20, 0xFFFF)
    struct.pack_into("<7H", plan, UNARY_INDEX_OFFSET, 0, 1, 0, 1, 0, 0, 1)
    struct.pack_into("<2i", plan, UNARY_SHAPES_OFFSET, 2, 3)
    plan[UNARY_STRINGS_OFFSET:] = b"x\0"
    return plan


def write(path: Path, data: bytearray) -> None:
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = valid_plan()
    write(args.output / "minimal-valid.tgrs", baseline)

    bad_magic = baseline.copy()
    bad_magic[:4] = b"FAIL"
    write(args.output / "bad-magic.tgrs", bad_magic)

    bad_version = baseline.copy()
    struct.pack_into("<I", bad_version, 4, 0xFFFFFFFF)
    write(args.output / "bad-version.tgrs", bad_version)

    bad_file_size = baseline.copy()
    struct.pack_into("<I", bad_file_size, 8, PLAN_SIZE - 1)
    write(args.output / "bad-file-size.tgrs", bad_file_size)

    duplicate_section = baseline.copy()
    struct.pack_into("<I", duplicate_section, SECTION_DIR_OFFSET + 8, 1)
    write(args.output / "duplicate-section.tgrs", duplicate_section)

    directory_overlap = baseline.copy()
    struct.pack_into("<I", directory_overlap, SECTION_DIR_OFFSET + 4, 48)
    write(args.output / "directory-overlap.tgrs", directory_overlap)

    invalid_string = baseline.copy()
    struct.pack_into("<I", invalid_string, 32, 1)
    write(args.output / "invalid-model-string.tgrs", invalid_string)

    invalid_io = baseline.copy()
    invalid_io[38] = 1
    write(args.output / "invalid-model-io.tgrs", invalid_io)

    unary = valid_unary_plan()
    write(args.output / "semantic-valid-relu.tgrs", unary)

    unknown_op = unary.copy()
    unknown_op[UNARY_OPS_OFFSET + 4] = 0xFF
    write(args.output / "semantic-unknown-op.tgrs", unknown_op)

    undispatched_op = unary.copy()
    undispatched_op[UNARY_OPS_OFFSET + 4] = 11  # Clip
    write(args.output / "semantic-undispatched-op.tgrs", undispatched_op)

    empty_input = unary.copy()
    empty_input[UNARY_OPS_OFFSET + 5] = 0
    write(args.output / "semantic-empty-input.tgrs", empty_input)

    mixed_dtype = unary.copy()
    struct.pack_into("<I", mixed_dtype, UNARY_TENSORS_OFFSET + 16 + 4, 6)
    mixed_dtype[UNARY_TENSORS_OFFSET + 16 + 11] = 3
    write(args.output / "semantic-mixed-dtype.tgrs", mixed_dtype)

    wrong_output_flag = unary.copy()
    wrong_output_flag[UNARY_TENSORS_OFFSET + 16 + 12] = 0
    write(args.output / "semantic-output-flag.tgrs", wrong_output_flag)

    invalid_fusion = unary.copy()
    invalid_fusion[UNARY_OPS_OFFSET + 34] = 1
    write(args.output / "semantic-invalid-fusion.tgrs", invalid_fusion)

    print(f"wrote {len(list(args.output.glob('*.tgrs')))} deterministic seeds")


if __name__ == "__main__":
    main()

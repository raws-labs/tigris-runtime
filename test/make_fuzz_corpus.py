#!/usr/bin/env python3
"""Create deterministic loader-fuzz seeds for every cheap rejection path."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


PLAN_SIZE = 100
SECTION_DIR_OFFSET = 48
SCHEMA_VERSION = 2
REQUIRED_SECTIONS = (1, 2, 5, 6, 7)


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

    print(f"wrote {len(list(args.output.glob('*.tgrs')))} deterministic seeds")


if __name__ == "__main__":
    main()

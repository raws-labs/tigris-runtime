#!/usr/bin/env python3
"""Write the smallest valid schema-v2 plan as a libFuzzer seed."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


PLAN_SIZE = 100
SECTION_DIR_OFFSET = 48
SCHEMA_VERSION = 2
REQUIRED_SECTIONS = (1, 2, 5, 6, 7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

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
    plan[PLAN_SIZE - 1] = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(plan)


if __name__ == "__main__":
    main()

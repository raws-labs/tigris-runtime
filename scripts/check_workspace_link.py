#!/usr/bin/env python3
"""Reject an unused compatibility executor workspace in a linked binary."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--nm", default="nm", help="nm-compatible executable")
    args = parser.parse_args()

    result = subprocess.run(
        [args.nm, "--defined-only", str(args.binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    symbols = result.stdout
    if "tigris_explicit_executor_workspace" not in symbols:
        parser.error("explicit workspace symbol is missing from smoke binary")
    if "tigris_default_executor_workspace" in symbols:
        parser.error("unused compatibility workspace was linked into smoke binary")

    print("Explicit-workspace binary contains no compatibility workspace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Enforce bounded stack frames for TiGrIS runtime sources.

Consumes GCC ``-fstack-usage`` files below one or more build directories.
Test/example frames are intentionally ignored: this gate covers code shipped
in the runtime library and optional accelerated adapters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


RECORD = re.compile(
    r"^(?P<source>.*):(?P<line>\d+):(?P<column>\d+):"
    r"(?P<function>[^\t]+)\t(?P<bytes>\d+)\t(?P<kind>.+)$"
)


@dataclass(frozen=True)
class Usage:
    source: str
    line: int
    function: str
    bytes: int
    kind: str
    report: Path


def is_runtime_source(source: str) -> bool:
    normalized = source.replace("\\", "/")
    return normalized.startswith("src/") or "/src/" in normalized


def read_reports(roots: list[Path]) -> tuple[list[Usage], list[str]]:
    usage: list[Usage] = []
    errors: list[str] = []
    reports = sorted(
        report for root in roots for report in root.rglob("*.su")
    )
    if not reports:
        return usage, ["no .su reports found (compile with TIGRIS_STACK_USAGE=ON)"]

    for report in reports:
        for number, raw in enumerate(
            report.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = RECORD.match(raw)
            if not match:
                errors.append(f"{report}:{number}: malformed stack-usage record")
                continue
            source = match.group("source")
            if not is_runtime_source(source):
                continue
            usage.append(
                Usage(
                    source=source,
                    line=int(match.group("line")),
                    function=match.group("function"),
                    bytes=int(match.group("bytes")),
                    kind=match.group("kind"),
                    report=report,
                )
            )
    if not usage:
        errors.append(".su reports contained no runtime src/ records")
    return usage, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", nargs="+", type=Path, help="build directories containing .su files"
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        required=True,
        help="largest permitted compiler-reported runtime frame in bytes",
    )
    args = parser.parse_args()
    if args.max_frame < 1:
        parser.error("--max-frame must be positive")

    usage, errors = read_reports(args.roots)
    failures = list(errors)
    for item in usage:
        qualifiers = {part.strip() for part in item.kind.split(",")}
        if "dynamic" in qualifiers and "bounded" not in qualifiers:
            failures.append(
                f"{item.source}:{item.line}: {item.function}: "
                f"unbounded dynamic frame ({item.kind})"
            )
        if item.bytes > args.max_frame:
            failures.append(
                f"{item.source}:{item.line}: {item.function}: {item.bytes} bytes "
                f"exceeds {args.max_frame}-byte frame budget"
            )

    # Duplicate source records can occur when multiple targets compile the same
    # file. Preserve the largest instance for a concise, deterministic report.
    largest: dict[tuple[str, str], Usage] = {}
    for item in usage:
        key = (item.source, item.function)
        if key not in largest or item.bytes > largest[key].bytes:
            largest[key] = item

    print(f"Runtime stack frames (limit {args.max_frame} bytes):")
    for item in sorted(largest.values(), key=lambda value: (-value.bytes, value.function)):
        print(f"{item.bytes:5d}  {item.kind:15s}  {item.function}")

    if failures:
        print("\nStack-usage check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"Checked {len(usage)} runtime stack-usage records; all within budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

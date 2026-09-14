#!/usr/bin/env python3
"""Hold the version sources to one number.

Three files state this library's version and nothing compared them, so they
drifted: idf_component.yml sat a release behind on develop because the bump was
made on a release branch and never merged back, and the CMake project version
had not moved in three releases. A stale component version is the expensive
one, because the registry refuses a version it already has and CONTRIBUTING
says that is fixed with a new patch release rather than a re-tag.

Run with a tag to check the release itself:

    python3 scripts/check_version_sources.py --expect 0.9.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = (
    ("CMakeLists.txt", r"project\(tigris_runtime VERSION ([0-9]+\.[0-9]+\.[0-9]+)"),
    ("idf_component.yml", r'^version:\s*"([0-9]+\.[0-9]+\.[0-9]+)"'),
)
# The packaged-consumer test asks for a major.minor floor, which must be this
# release's own, or a consumer requesting the current version is refused.
CONSUMER = ("test/package_consumer/CMakeLists.txt",
            r"find_package\(tigris_runtime ([0-9]+\.[0-9]+)")


def _read(relative: str, pattern: str) -> tuple[str, str | None]:
    text = (ROOT / relative).read_text()
    found = re.search(pattern, text, re.MULTILINE)
    return relative, found.group(1) if found else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", help="version the release tag states")
    args = parser.parse_args()

    versions = dict(_read(name, pattern) for name, pattern in SOURCES)
    errors = [f"{name}: no version found" for name, v in versions.items() if v is None]
    distinct = {v for v in versions.values() if v is not None}
    if len(distinct) > 1:
        errors.append("version sources disagree: " + ", ".join(
            f"{name}={version}" for name, version in sorted(versions.items())))

    consumer_name, consumer_floor = _read(*CONSUMER)
    if consumer_floor is None:
        errors.append(f"{consumer_name}: no find_package version found")
    elif distinct:
        expected_floor = ".".join(next(iter(distinct)).split(".")[:2])
        if consumer_floor != expected_floor:
            errors.append(
                f"{consumer_name}: asks for {consumer_floor}, "
                f"this release is {expected_floor}")

    if args.expect and distinct and args.expect not in distinct:
        errors.append(
            f"version sources state {next(iter(distinct))}, tag states {args.expect}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"Version sources agree: {next(iter(distinct))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

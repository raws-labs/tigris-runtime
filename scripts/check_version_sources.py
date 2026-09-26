#!/usr/bin/env python3
"""Hold the version sources to one number.

Three files state this library's version and nothing compared them, so they
drifted: idf_component.yml sat a release behind on develop because the bump was
made on a release branch and never merged back, and the CMake project version
had not moved in three releases. A stale component version is the expensive
one, because the registry refuses a version it already has and CONTRIBUTING
says that is fixed with a new patch release rather than a re-tag.

Run with a tag to check the release itself, or set a new release's version:

    python3 scripts/check_version_sources.py --expect 0.9.0
    python3 scripts/check_version_sources.py --set 0.12.0
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

# The shipped example depends on this component by a caret constraint, so it
# names the version too. It is inside the published component, which makes a
# stale constraint there a released artifact pointing at an older self.
EXAMPLE = ("examples/esp32/getting-started/main/idf_component.yml",
           r'raws-labs/tigris-runtime:\s*\n\s*version:\s*"\^([0-9]+\.[0-9]+\.[0-9]+)"')

# The README shows consumers the find_package call, so it states the floor too.
README = ("README.md", r"find_package\(tigris_runtime ([0-9]+\.[0-9]+) REQUIRED")


def _read(relative: str, pattern: str) -> tuple[str, str | None]:
    text = (ROOT / relative).read_text()
    found = re.search(pattern, text, re.MULTILINE)
    return relative, found.group(1) if found else None


def _set(relative: str, pattern: str, value: str) -> None:
    path = ROOT / relative
    text = path.read_text()
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        raise ValueError(f"{relative}: no version found")
    path.write_text(text[:found.start(1)] + value + text[found.end(1):])


def set_version(version: str) -> None:
    """Write VERSION into every source, and its major.minor into the floors."""
    floor = ".".join(version.split(".")[:2])
    for name, pattern in (*SOURCES, EXAMPLE):
        _set(name, pattern, version)
    for name, pattern in (CONSUMER, README):
        _set(name, pattern, floor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", help="version the release tag states")
    parser.add_argument("--set", metavar="VERSION",
                        help="write VERSION into every source, then check them")
    args = parser.parse_args()
    if args.set:
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", args.set):
            parser.error("--set takes a version X.Y.Z")
        set_version(args.set)
        args.expect = args.set

    versions = dict(_read(name, pattern) for name, pattern in SOURCES)
    errors = [f"{name}: no version found" for name, v in versions.items() if v is None]
    distinct = {v for v in versions.values() if v is not None}
    if len(distinct) > 1:
        errors.append("version sources disagree: " + ", ".join(
            f"{name}={version}" for name, version in sorted(versions.items())))

    for name, pattern in (CONSUMER, README):
        found_name, floor = _read(name, pattern)
        if floor is None:
            errors.append(f"{found_name}: no find_package version found")
        elif distinct:
            expected_floor = ".".join(next(iter(distinct)).split(".")[:2])
            if floor != expected_floor:
                errors.append(
                    f"{found_name}: asks for {floor}, "
                    f"this release is {expected_floor}")

    example_name, example_version = _read(*EXAMPLE)
    if example_version is None:
        errors.append(f"{example_name}: no component constraint found")
    elif distinct and example_version != next(iter(distinct)):
        errors.append(
            f"{example_name}: depends on ^{example_version}, "
            f"this release is {next(iter(distinct))}")

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

#!/usr/bin/env python3
"""Enforce deterministic GCC line and branch-outcome coverage floors."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# These floors describe the safety/control surface exercised by the native
# suite. Backend adapters have separate route and target tests and are not
# folded into these host-only numbers.
MINIMUMS = {
    "tigris_loader.c": (82.0, 61.0),
    "tigris_executor.c": (90.0, 69.0),
    "tigris_mem.c": (94.0, 71.0),
    "tigris_lz4.c": (85.0, 66.0),
    "tigris_kernels.c": (83.0, 66.0),
    "tigris_kernels_s8.c": (78.0, 59.0),
}


@dataclass(frozen=True)
class Coverage:
    lines_hit: int
    lines_total: int
    branches_hit: int
    branches_total: int

    @property
    def line_percent(self) -> float:
        return 100.0 * self.lines_hit / self.lines_total

    @property
    def branch_percent(self) -> float:
        return 100.0 * self.branches_hit / self.branches_total


def measure(document: object, source_name: str) -> Coverage:
    if not isinstance(document, dict) or not isinstance(document.get("files"), list):
        raise ValueError("gcov JSON has no files list")
    matches = [
        source for source in document["files"]
        if isinstance(source, dict)
        and Path(str(source.get("file", ""))).name == source_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one gcov record for {source_name}, found {len(matches)}")
    lines = matches[0].get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"gcov record for {source_name} has no lines")

    line_counts: list[int] = []
    branch_counts: list[int] = []
    for line in lines:
        if not isinstance(line, dict):
            raise ValueError(f"gcov line record for {source_name} is not an object")
        count = line.get("count")
        branches = line.get("branches")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"gcov line count for {source_name} is invalid")
        if not isinstance(branches, list):
            raise ValueError(f"gcov branch list for {source_name} is invalid")
        line_counts.append(count)
        for branch in branches:
            branch_count = branch.get("count") if isinstance(branch, dict) else None
            if not isinstance(branch_count, int) or isinstance(branch_count, bool):
                raise ValueError(f"gcov branch count for {source_name} is invalid")
            branch_counts.append(branch_count)

    if not branch_counts:
        raise ValueError(f"gcov record for {source_name} has no branches")
    return Coverage(
        lines_hit=sum(count > 0 for count in line_counts),
        lines_total=len(line_counts),
        branches_hit=sum(count > 0 for count in branch_counts),
        branches_total=len(branch_counts),
    )


def check_minimums(
        results: dict[str, Coverage],
        minimums: dict[str, tuple[float, float]],
) -> list[str]:
    errors: list[str] = []
    for source, (minimum_lines, minimum_branches) in minimums.items():
        coverage = results.get(source)
        if coverage is None:
            errors.append(f"missing coverage for {source}")
            continue
        if coverage.line_percent + 1e-9 < minimum_lines:
            errors.append(
                f"{source} line coverage {coverage.line_percent:.2f}% is below "
                f"{minimum_lines:.2f}%")
        if coverage.branch_percent + 1e-9 < minimum_branches:
            errors.append(
                f"{source} branch-outcome coverage "
                f"{coverage.branch_percent:.2f}% is below "
                f"{minimum_branches:.2f}%")
    return errors


def collect(build_dir: Path, gcov: str) -> dict[str, Coverage]:
    object_dir = build_dir / "CMakeFiles/tigris_runtime.dir/src"
    results: dict[str, Coverage] = {}
    with tempfile.TemporaryDirectory(prefix="tigris-gcov-") as directory:
        output_dir = Path(directory)
        for source in MINIMUMS:
            notes = object_dir / f"{source}.gcno"
            data = object_dir / f"{source}.gcda"
            if not notes.is_file() or not data.is_file():
                raise ValueError(
                    f"missing coverage data for {source}; build and run tests first")
            completed = subprocess.run(
                [gcov, "--json-format", "--branch-probabilities", str(notes)],
                cwd=output_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise ValueError(
                    f"gcov failed for {source}: "
                    f"{completed.stdout}{completed.stderr}".strip())
            generated = output_dir / f"{source}.gcov.json.gz"
            if not generated.is_file():
                raise ValueError(f"gcov produced no JSON for {source}")
            with gzip.open(generated, "rt", encoding="utf-8") as stream:
                document = json.load(stream)
            results[source] = measure(document, source)
            generated.unlink()
    return results


def self_test() -> None:
    document = {
        "files": [{
            "file": "/source/example.c",
            "lines": [
                {"count": 1, "branches": [{"count": 1}, {"count": 0}]},
                {"count": 0, "branches": []},
            ],
        }],
    }
    coverage = measure(document, "example.c")
    if coverage != Coverage(1, 2, 1, 2):
        raise AssertionError(f"unexpected synthetic coverage: {coverage}")
    if check_minimums({"example.c": coverage}, {"example.c": (50.0, 50.0)}):
        raise AssertionError("exact coverage floors should pass")
    errors = check_minimums(
        {"example.c": coverage}, {"example.c": (50.01, 50.01)})
    if len(errors) != 2:
        raise AssertionError(f"threshold mutations were not rejected: {errors}")
    malformed = {"files": [{"file": "example.c", "lines": []}]}
    try:
        measure(malformed, "example.c")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed gcov input was accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", nargs="?", type=Path)
    parser.add_argument("--gcov", default="gcov")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("Coverage checker mutation self-test passed.")
    if args.build_dir is None:
        if args.self_test:
            return
        parser.error("build_dir is required unless --self-test is used")

    try:
        results = collect(args.build_dir.resolve(), args.gcov)
        errors = check_minimums(results, MINIMUMS)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    for source, coverage in results.items():
        minimum_lines, minimum_branches = MINIMUMS[source]
        print(
            f"{source}: lines {coverage.line_percent:.2f}% "
            f"({coverage.lines_hit}/{coverage.lines_total}, min {minimum_lines:.2f}%); "
            f"branch outcomes {coverage.branch_percent:.2f}% "
            f"({coverage.branches_hit}/{coverage.branches_total}, "
            f"min {minimum_branches:.2f}%)")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Runtime coverage floors passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run pinned Cppcheck checks and enforce the reviewed set of deviations.

General correctness diagnostics must be absent. MISRA findings are advisory
on a codebase this size, so what is reviewed is which rule each file is
allowed to deviate from, not how many times it does. Editing a function that
already deviates from a rule therefore changes nothing here; a new rule, or an
old rule in a new file, is a decision and fails until it is recorded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


CPPCHECK_VERSION = "2.21.1"
SOURCES = (
    "src/tigris_loader.c",
    "src/tigris_iface.c",
    "src/tigris_mem.c",
    "src/tigris_executor.c",
    "src/tigris_executor_compat.c",
    "src/tigris_kernels.c",
    "src/tigris_kernels_s8.c",
    "src/tigris_lz4.c",
)
ACCEPTED_PATH = Path(__file__).with_name("static_analysis_accepted.txt")


@dataclass(frozen=True)
class Diagnostic:
    identifier: str
    severity: str
    file: str
    message: str
    source_line: str


def normalized_source_line(root: Path, file: str, line: int) -> str:
    path = (root / file).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"diagnostic path escapes repository: {file}") from exc
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read diagnostic source {file}: {exc}") from exc
    if line < 1 or line > len(lines):
        raise ValueError(f"diagnostic line {file}:{line} is out of range")
    return " ".join(lines[line - 1].split())


def parse_report(path: Path, root: Path) -> tuple[str, list[Diagnostic]]:
    try:
        document = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    if document.tag != "results" or document.get("version") != "2":
        raise ValueError(f"{path} is not Cppcheck XML version 2")
    tool = document.find("cppcheck")
    version = tool.get("version") if tool is not None else None
    if not version:
        raise ValueError(f"{path} does not identify the Cppcheck version")

    diagnostics: list[Diagnostic] = []
    errors = document.find("errors")
    if errors is None:
        raise ValueError(f"{path} has no errors element")
    for error in errors.findall("error"):
        identifier = error.get("id")
        severity = error.get("severity")
        message = error.get("msg")
        location = error.find("location")
        if not identifier or not severity or message is None or location is None:
            raise ValueError(f"{path} contains an incomplete diagnostic")
        file = location.get("file")
        line_text = location.get("line")
        if not file or not line_text:
            raise ValueError(f"{path} contains a diagnostic without a location")
        try:
            line = int(line_text)
        except ValueError as exc:
            raise ValueError(
                f"{path} contains an invalid line number: {line_text}"
            ) from exc
        diagnostics.append(
            Diagnostic(
                identifier=identifier,
                severity=severity,
                file=file,
                message=message,
                source_line=normalized_source_line(root, file, line),
            )
        )
    return version, diagnostics


def deviations(diagnostics: list[Diagnostic]) -> set[tuple[str, str]]:
    """The (rule, file) pairs a scan found, which is what is reviewed."""
    return {(item.identifier, item.file) for item in diagnostics}


ACCEPTED_HEADER = """\
# Cppcheck deviations this code base is allowed to carry, one per line as
# "<rule> <file>". Regenerate with:
#
#     python3 scripts/check_static_analysis.py --cppcheck <bin> --update-accepted
#
# Adding a line is a decision: it says this file may deviate from this rule.
# Removing the last occurrence of a pair leaves its line stale, which the gate
# reports without failing, because a stale line cannot hide a new defect
# anywhere except in the file it already names.
#
# General correctness diagnostics are never listed here. There are none, and
# the gate fails if one appears.
"""


def read_accepted(path: Path) -> set[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    accepted: set[tuple[str, str]] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected '<rule> <file>'")
        accepted.add((fields[0], fields[1]))
    return accepted


def write_accepted(path: Path, pairs: set[tuple[str, str]]) -> None:
    body = "".join(f"{rule} {file}\n" for rule, file in sorted(pairs))
    path.write_text(ACCEPTED_HEADER + body, encoding="utf-8")


def run_cppcheck(cppcheck: Path, root: Path, output: Path, mode: str) -> None:
    common = [
        str(cppcheck),
        "--std=c11",
        "--platform=unix64",
        "--max-configs=1",
        "--xml",
        "--xml-version=2",
        f"--output-file={output}",
        "-Iinclude",
    ]
    if mode == "correctness":
        options = ["--enable=warning,performance,portability"]
    elif mode == "misra":
        options = ["--enable=style", "--addon=misra"]
    else:
        raise ValueError(f"unknown analysis mode: {mode}")
    try:
        completed = subprocess.run(
            common + options + list(SOURCES),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Cppcheck {mode} scan failed to run: {exc}") from exc
    if completed.returncode != 0:
        detail = f"{completed.stdout}\n{completed.stderr}".strip()
        raise ValueError(
            f"Cppcheck {mode} scan exited with {completed.returncode}: {detail}"
        )
    if not output.is_file():
        raise ValueError(f"Cppcheck {mode} scan produced no XML report")


def self_test() -> None:
    finding = Diagnostic(
        identifier="misra-c2012-15.6",
        severity="style",
        file="src/example.c",
        message="misra violation",
        source_line="if (ready)",
    )
    accepted = deviations([finding])
    if deviations([finding, finding]) - accepted:
        raise AssertionError("a second occurrence of an accepted rule was flagged")

    elsewhere = Diagnostic(
        identifier=finding.identifier,
        severity=finding.severity,
        file="src/other.c",
        message=finding.message,
        source_line=finding.source_line,
    )
    if not deviations([elsewhere]) - accepted:
        raise AssertionError("the same rule in a new file was accepted")

    another_rule = Diagnostic(
        identifier="misra-c2012-15.7",
        severity=finding.severity,
        file=finding.file,
        message=finding.message,
        source_line=finding.source_line,
    )
    if not deviations([another_rule]) - accepted:
        raise AssertionError("a new rule in an accepted file was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cppcheck", type=Path, help=f"Cppcheck {CPPCHECK_VERSION} executable"
    )
    parser.add_argument("--accepted", type=Path, default=ACCEPTED_PATH)
    parser.add_argument("--update-accepted", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("Static-analysis checker mutation self-test passed.")
    if args.cppcheck is None:
        if args.self_test:
            return 0
        parser.error("--cppcheck is required unless only --self-test is used")

    root = Path(__file__).resolve().parents[1]
    cppcheck = args.cppcheck.resolve()
    try:
        completed = subprocess.run(
            [str(cppcheck), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        reported_version = completed.stdout.strip()
        if completed.returncode != 0 or reported_version != (
            f"Cppcheck {CPPCHECK_VERSION}"
        ):
            raise ValueError(
                f"expected Cppcheck {CPPCHECK_VERSION}, got "
                f"{reported_version or 'no version'}"
            )

        with tempfile.TemporaryDirectory(prefix="tigris-static-analysis-") as temp:
            directory = Path(temp)
            reports: dict[str, list[Diagnostic]] = {}
            for mode in ("correctness", "misra"):
                report = directory / f"{mode}.xml"
                run_cppcheck(cppcheck, root, report, mode)
                version, diagnostics = parse_report(report, root)
                if version != CPPCHECK_VERSION:
                    raise ValueError(
                        f"{mode} report identifies Cppcheck {version}, expected "
                        f"{CPPCHECK_VERSION}"
                    )
                reports[mode] = diagnostics

        correctness = reports["correctness"]
        misra = reports["misra"]
        found = deviations(misra)
        if args.update_accepted:
            if correctness:
                raise ValueError(
                    "refusing to record deviations while general correctness "
                    "diagnostics remain; fix them first"
                )
            write_accepted(args.accepted, found)
            print(f"Recorded {len(found)} reviewed deviations in {args.accepted}.")
            return 0

        accepted = read_accepted(args.accepted)
        errors = [
            f"correctness finding in {item.file}: {item.identifier} "
            f"{item.message}"
            for item in correctness
        ]
        errors.extend(
            f"{rule} is not recorded for {file}"
            for rule, file in sorted(found - accepted)
        )
        stale = sorted(accepted - found)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    misra_rules = sum(
        item.identifier.startswith("misra-c2012-") for item in misra
    )
    config_errors = sum(item.identifier == "misra-config" for item in misra)
    print(
        f"Cppcheck {CPPCHECK_VERSION}: {len(correctness)} correctness findings; "
        f"{misra_rules} MISRA C:2012 findings across {len(found)} reviewed "
        f"rule/file pairs; {len(misra) - misra_rules - config_errors} "
        f"additional style findings; {config_errors} reviewed analyzer "
        "configuration limitations."
    )
    for rule, file in stale:
        print(f"note: {rule} no longer occurs in {file}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(
            "Each line is a deviation this code base does not yet accept. "
            "Fix it, or record it with --update-accepted and commit the file."
        )
        return 1
    print("Static analysis found only reviewed deviations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

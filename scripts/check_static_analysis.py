#!/usr/bin/env python3
"""Run pinned Cppcheck checks and enforce the reviewed diagnostic snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


CPPCHECK_VERSION = "2.21.1"
SOURCES = (
    "src/tigris_loader.c",
    "src/tigris_mem.c",
    "src/tigris_executor.c",
    "src/tigris_executor_compat.c",
    "src/tigris_kernels.c",
    "src/tigris_kernels_s8.c",
    "src/tigris_lz4.c",
)
BASELINE_PATH = Path(__file__).with_name("static_analysis_baseline.json")


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


def fingerprint(diagnostics: list[Diagnostic]) -> str:
    records = sorted(
        (
            item.identifier,
            item.severity,
            item.file,
            item.message,
            item.source_line,
        )
        for item in diagnostics
    )
    encoded = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot(diagnostics: list[Diagnostic]) -> dict[str, object]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in diagnostics:
        counts[item.file][item.identifier] += 1
    return {
        "total": len(diagnostics),
        "misra_total": sum(
            item.identifier.startswith("misra-c2012-") for item in diagnostics
        ),
        "config_errors": sum(
            item.identifier == "misra-config" for item in diagnostics
        ),
        "fingerprint_sha256": fingerprint(diagnostics),
        "counts": {
            file: dict(sorted(identifiers.items()))
            for file, identifiers in sorted(counts.items())
        },
    }


def compare_snapshot(
    actual: dict[str, object], expected: object, label: str
) -> list[str]:
    if not isinstance(expected, dict):
        return [f"baseline {label} entry is not an object"]
    errors: list[str] = []
    for key in (
        "total",
        "misra_total",
        "config_errors",
        "fingerprint_sha256",
        "counts",
    ):
        if actual.get(key) != expected.get(key):
            errors.append(f"{label} {key} differs from the reviewed baseline")
    return errors


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


def read_baseline(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read baseline {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError(f"{path} is not a static-analysis baseline version 1")
    if document.get("cppcheck_version") != CPPCHECK_VERSION:
        raise ValueError(
            f"baseline expects Cppcheck {document.get('cppcheck_version')}, "
            f"checker expects {CPPCHECK_VERSION}"
        )
    if document.get("sources") != list(SOURCES):
        raise ValueError("baseline source scope differs from the checker")
    return document


def make_baseline(
    correctness: list[Diagnostic], misra: list[Diagnostic]
) -> dict[str, object]:
    if correctness:
        raise ValueError(
            "refusing to baseline general correctness diagnostics; fix them first"
        )
    return {
        "format_version": 1,
        "cppcheck_version": CPPCHECK_VERSION,
        "sources": list(SOURCES),
        "correctness": snapshot(correctness),
        "misra": snapshot(misra),
    }


def self_test() -> None:
    clean = snapshot([])
    if clean["total"] != 0 or clean["fingerprint_sha256"] != fingerprint([]):
        raise AssertionError("empty diagnostic snapshot is inconsistent")

    finding = Diagnostic(
        identifier="misra-c2012-15.6",
        severity="style",
        file="src/example.c",
        message="misra violation",
        source_line="if (ready)",
    )
    expected = snapshot([finding])
    if compare_snapshot(expected, expected, "synthetic"):
        raise AssertionError("identical diagnostic snapshots did not match")

    added = snapshot([finding, finding])
    if not compare_snapshot(added, expected, "synthetic"):
        raise AssertionError("an added diagnostic was accepted")
    moved = snapshot([
        Diagnostic(
            identifier=finding.identifier,
            severity=finding.severity,
            file=finding.file,
            message=finding.message,
            source_line="if (different)",
        )
    ])
    errors = compare_snapshot(moved, expected, "synthetic")
    if not any("fingerprint" in error for error in errors):
        raise AssertionError("a changed diagnostic location was accepted")
    try:
        make_baseline([finding], [])
    except ValueError:
        pass
    else:
        raise AssertionError("a correctness diagnostic was allowed into a baseline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cppcheck", type=Path, help=f"Cppcheck {CPPCHECK_VERSION} executable"
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--update-baseline", action="store_true")
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

        actual = make_baseline(reports["correctness"], reports["misra"])
        if args.update_baseline:
            args.baseline.write_text(
                json.dumps(actual, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Wrote reviewed baseline candidate to {args.baseline}.")
            return 0

        expected = read_baseline(args.baseline)
        errors = compare_snapshot(
            actual["correctness"], expected.get("correctness"), "correctness"
        )
        errors.extend(
            compare_snapshot(actual["misra"], expected.get("misra"), "MISRA")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    misra = actual["misra"]
    print(
        f"Cppcheck {CPPCHECK_VERSION}: 0 correctness findings; "
        f"{misra['misra_total']} MISRA C:2012 findings; "
        f"{misra['total'] - misra['misra_total'] - misra['config_errors']} "
        f"additional style findings; {misra['config_errors']} reviewed analyzer "
        "configuration limitations."
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(
            "Review the diagnostics, then regenerate and commit the baseline "
            "only for an intentional change."
        )
        return 1
    print("Static-analysis findings match the reviewed baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

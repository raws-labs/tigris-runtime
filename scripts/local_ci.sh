#!/usr/bin/env bash
#
# Run the three tigris-runtime gate checks locally, the same way Runtime CI
# runs them, so a tiling/roll branch fails on the developer's machine instead
# of only in CI on push:
#
#   1. stack usage        (scripts/check_stack_usage.py + check_workspace_link.py)
#   2. static analysis    (scripts/check_static_analysis.py, pinned Cppcheck 2.21.1)
#   3. coverage floors    (scripts/check_coverage.py, pinned gcc-13/gcov-13)
#
# Usage:
#   scripts/local_ci.sh                 run every check the local toolchain allows
#   scripts/local_ci.sh --strict        treat a skipped check as a failure
#   scripts/local_ci.sh --build-cppcheck  build+cache pinned Cppcheck, then run
#
# Install as a git pre-push hook (from the repo root):
#   ln -s ../../scripts/local_ci.sh .git/hooks/pre-push
#
# Environment:
#   TIGRIS_CPPCHECK   path to an existing Cppcheck 2.21.1 binary; skips the
#                     one-time source build entirely.
#
# A check is SKIPPED (not failed) when the toolchain it is calibrated against
# is missing, because the coverage floors and MISRA baseline only mean anything
# against the pinned tools. Skips are reported loudly; --strict turns them into
# failures so a pre-push gate can insist the full suite ran.

set -euo pipefail

CPPCHECK_VERSION="2.21.1"
CPPCHECK_COMMIT="904cfdcf774c44b17db789c8a212e2f1c69fc833"
STACK_MAX_FRAME=1024
COVERAGE_CC="gcc-13"
COVERAGE_GCOV="gcov-13"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STRICT=0
BUILD_CPPCHECK=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
    --build-cppcheck) BUILD_CPPCHECK=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

PASS=(); FAIL=(); SKIP=()
pass() { PASS+=("$1"); printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { FAIL+=("$1"); printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
skip() { SKIP+=("$1"); printf '  \033[33mSKIP\033[0m  %s (%s)\n' "$1" "$2"; }

echo "== checker self-tests =="
python3 scripts/check_static_analysis.py --self-test
python3 scripts/check_coverage.py --self-test

echo "== stack usage =="
run_stack() {
  cmake -S . -B build-stack -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc \
        -DTIGRIS_STACK_USAGE=ON >/dev/null
  cmake --build build-stack \
        --target tigris_runtime cmsis_adapter_compile explicit_workspace_link_smoke \
        --parallel >/dev/null
  python3 scripts/check_stack_usage.py --max-frame "$STACK_MAX_FRAME" build-stack
  python3 scripts/check_workspace_link.py build-stack/explicit_workspace_link_smoke
}
if have gcc && have cmake; then
  if run_stack; then pass "stack usage"; else fail "stack usage"; fi
else
  skip "stack usage" "gcc or cmake not found"
fi

echo "== static analysis / MISRA =="
resolve_cppcheck() {
  local cached="$ROOT/.cache/cppcheck-build/bin/cppcheck"
  if [ -n "${TIGRIS_CPPCHECK:-}" ] \
     && [ "$("$TIGRIS_CPPCHECK" --version 2>/dev/null)" = "Cppcheck $CPPCHECK_VERSION" ]; then
    echo "$TIGRIS_CPPCHECK"; return 0
  fi
  if [ -x "$cached" ] \
     && [ "$("$cached" --version 2>/dev/null)" = "Cppcheck $CPPCHECK_VERSION" ]; then
    echo "$cached"; return 0
  fi
  if [ "$BUILD_CPPCHECK" = 1 ] && have git && have cmake; then
    echo "  building pinned Cppcheck $CPPCHECK_VERSION (one-time, cached in .cache/) ..." >&2
    rm -rf "$ROOT/.cache/cppcheck-src"
    git init -q "$ROOT/.cache/cppcheck-src"
    git -C "$ROOT/.cache/cppcheck-src" remote add origin \
        https://github.com/cppcheck-opensource/cppcheck.git
    git -C "$ROOT/.cache/cppcheck-src" fetch -q --depth=1 origin "$CPPCHECK_COMMIT"
    git -C "$ROOT/.cache/cppcheck-src" checkout -q --detach FETCH_HEAD
    cmake -S "$ROOT/.cache/cppcheck-src" -B "$ROOT/.cache/cppcheck-build" \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_TESTING=OFF >/dev/null
    cmake --build "$ROOT/.cache/cppcheck-build" --target cppcheck --parallel 2 >/dev/null
    [ -x "$cached" ] && echo "$cached" && return 0
  fi
  echo ""; return 0
}
CPPCHECK_BIN="$(resolve_cppcheck)"
if [ -n "$CPPCHECK_BIN" ]; then
  if python3 scripts/check_static_analysis.py --cppcheck "$CPPCHECK_BIN"; then
    pass "static analysis"; else fail "static analysis"; fi
else
  skip "static analysis" "no Cppcheck $CPPCHECK_VERSION; set TIGRIS_CPPCHECK or pass --build-cppcheck"
fi

echo "== coverage floors =="
run_coverage() {
  cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_C_COMPILER="$COVERAGE_CC" \
        -DCMAKE_C_FLAGS="-O0 -g --coverage" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" >/dev/null
  cmake --build build-coverage --parallel >/dev/null
  ctest --test-dir build-coverage --output-on-failure >/dev/null
  python3 scripts/check_coverage.py --gcov "$COVERAGE_GCOV" build-coverage
}
if have "$COVERAGE_CC" && have "$COVERAGE_GCOV" && have cmake; then
  if run_coverage; then pass "coverage floors"; else fail "coverage floors"; fi
else
  skip "coverage floors" "$COVERAGE_CC/$COVERAGE_GCOV not found (floors are calibrated to them)"
fi

echo
echo "== summary =="
printf 'passed: %s   failed: %s   skipped: %s\n' "${#PASS[@]}" "${#FAIL[@]}" "${#SKIP[@]}"
if [ "${#FAIL[@]}" -gt 0 ]; then
  exit 1
fi
if [ "$STRICT" = 1 ] && [ "${#SKIP[@]}" -gt 0 ]; then
  echo "strict mode: a skipped check counts as a failure" >&2
  exit 1
fi
exit 0

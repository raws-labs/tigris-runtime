#!/usr/bin/env bash
# TiGrIS s8 E2E consistency tests.
#
# Builds the runtime, runs test_s8_e2e on every model/budget combination,
# and verifies that tiled outputs are byte-identical to the untiled baseline.
#
# Usage: bash test/run_all.sh [models_dir]
#   models_dir defaults to ../tigris-bench/models/output (relative to repo root)
#
# Exit code: 0 if all pass, 1 if any fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_DIR/build"
BINARY="$BUILD_DIR/test_s8_e2e"
MODELS_DIR="${1:-$(dirname "$REPO_DIR")/tigris-bench/models/output}"

passed=0
failed=0

# Extract the Output[...] line from test_s8_e2e stdout.
get_output() {
    local plan="$1" budget="$2"
    "$BINARY" "$plan" "$budget" 2>/dev/null | grep '^Output\['
}

# Run one model family: baseline + budget variants.
# Args: model_name baseline_plan budget1:plan1 budget2:plan2 ...
check_model() {
    local name="$1"; shift
    local baseline_plan="$1"; shift

    printf "\n--- %s ---\n" "$name"

    if [ ! -f "$baseline_plan" ]; then
        printf "  baseline: SKIP (file not found: %s)\n" "$baseline_plan"
        return
    fi

    local baseline
    baseline=$(get_output "$baseline_plan" 262144) || true
    if [ -z "$baseline" ]; then
        printf "  baseline: FAIL (inference error)\n"
        failed=$((failed + 1))
        return
    fi
    printf "  baseline (default) : %s\n" "$baseline"

    for spec in "$@"; do
        local label="${spec%%:*}"
        local plan="${spec#*:}"
        local budget="${label%K}"
        budget=$((budget * 1024))

        if [ ! -f "$plan" ]; then
            printf "  %-20s: SKIP (file not found)\n" "$label"
            continue
        fi

        local result
        result=$(get_output "$plan" "$budget") || true
        if [ -z "$result" ]; then
            printf "  %-20s: FAIL (inference error)\n" "$label"
            failed=$((failed + 1))
        elif [ "$result" = "$baseline" ]; then
            printf "  %-20s: PASS\n" "$label"
            passed=$((passed + 1))
        else
            printf "  %-20s: FAIL (output mismatch)\n" "$label"
            printf "    expected: %s\n" "$baseline"
            printf "    got:      %s\n" "$result"
            failed=$((failed + 1))
        fi
    done
}

# Main

printf "TiGrIS s8 E2E Consistency Tests\n"

# Build
printf "Building runtime...\n"
cmake -B "$BUILD_DIR" -S "$REPO_DIR" -DCMAKE_BUILD_TYPE=Release \
    > /dev/null 2>&1
cmake --build "$BUILD_DIR" --parallel > /dev/null 2>&1

if [ ! -x "$BINARY" ]; then
    printf "FATAL: test_s8_e2e not found at %s\n" "$BINARY"
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    printf "FATAL: models directory not found at %s\n" "$MODELS_DIR"
    exit 1
fi

M="$MODELS_DIR"

check_model "ds_cnn_i8" \
    "$M/ds_cnn_i8.tgrs" \
    "128K:$M/ds_cnn_i8_128k.tgrs" \
    "64K:$M/ds_cnn_i8_64k.tgrs" \
    "32K:$M/ds_cnn_i8_32k.tgrs" \
    "16K:$M/ds_cnn_i8_16k.tgrs"

check_model "mobilenet_v1_i8" \
    "$M/mobilenet_v1_i8.tgrs" \
    "128K:$M/mobilenet_v1_i8_128k.tgrs" \
    "64K:$M/mobilenet_v1_i8_64k.tgrs" \
    "32K:$M/mobilenet_v1_i8_32k.tgrs"

# Summary
total=$((passed + failed))
printf "\nResults: %d passed, %d failed, %d total\n" \
    "$passed" "$failed" "$total"

[ "$failed" -eq 0 ]

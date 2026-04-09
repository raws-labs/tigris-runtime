#!/bin/bash
# Flash a .tgrs file to the ESP32 "plan" partition.
# Auto-detects target from sdkconfig and reads the offset from the
# corresponding partition table CSV.
#
# Usage: scripts/flash_plan.sh <model.tgrs> [port]
#
# Example:
#   scripts/flash_plan.sh ../tigris-runtime/test/fixtures/conv_relu_chain.tgrs

set -euo pipefail

PLAN_FILE="${1:?Usage: scripts/flash_plan.sh <model.tgrs> [port]}"
PORT="${2:-/dev/ttyUSB0}"

# Find project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-detect target from sdkconfig
if [ ! -f "$PROJECT_DIR/sdkconfig" ]; then
    echo "Error: sdkconfig not found. Run 'idf.py set-target' first." >&2
    exit 1
fi

TARGET=$(grep '^CONFIG_IDF_TARGET=' "$PROJECT_DIR/sdkconfig" | cut -d'"' -f2)
if [ -z "$TARGET" ]; then
    echo "Error: could not detect IDF_TARGET from sdkconfig" >&2
    exit 1
fi

# Find the correct partition table
PARTITIONS="$PROJECT_DIR/partitions_${TARGET}.csv"
if [ ! -f "$PARTITIONS" ]; then
    echo "Error: partition table not found: $PARTITIONS" >&2
    exit 1
fi

# Parse plan partition offset from CSV
OFFSET=$(grep '^plan,' "$PARTITIONS" | tr -d ' ' | cut -d',' -f4)
if [ -z "$OFFSET" ]; then
    echo "Error: 'plan' partition not found in $PARTITIONS" >&2
    exit 1
fi

echo "Target:  $TARGET"
echo "Plan:    $PLAN_FILE"
echo "Port:    $PORT"
echo "Offset:  $OFFSET (from partitions_${TARGET}.csv)"
echo ""
python -m esptool --port "$PORT" write_flash "$OFFSET" "$PLAN_FILE"

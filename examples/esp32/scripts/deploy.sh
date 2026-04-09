#!/bin/bash
# Full deploy: build firmware, flash firmware + plan, open monitor.
#
# Usage: scripts/deploy.sh <model.tgrs> [port]
#
# Example:
#   scripts/deploy.sh ../tigris-runtime/test/fixtures/conv_relu_chain.tgrs

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

PLAN_FILE="${1:?Usage: scripts/deploy.sh <model.tgrs> [port]}"
PORT="${2:-/dev/ttyUSB0}"

echo "Building firmware"
idf.py build

echo "Flashing firmware"
idf.py -p "$PORT" flash

echo "Flashing plan: $PLAN_FILE"
"$SCRIPT_DIR/flash_plan.sh" "$PLAN_FILE" "$PORT"

echo "Opening monitor (Ctrl+] to exit)"
idf.py -p "$PORT" monitor

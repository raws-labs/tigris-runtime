#!/bin/bash
# Flash the TiGrIS ESP32 firmware (app only, not the plan).
#
# Usage: scripts/flash.sh [port]

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-/dev/ttyUSB0}"

idf.py -p "$PORT" flash

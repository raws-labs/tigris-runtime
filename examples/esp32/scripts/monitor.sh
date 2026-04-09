#!/bin/bash
# Open the serial monitor. Ctrl+] to exit.
#
# Usage: scripts/monitor.sh [port]

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-/dev/ttyUSB0}"

idf.py -p "$PORT" monitor

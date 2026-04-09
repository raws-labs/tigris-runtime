#!/bin/bash
# Build the TiGrIS ESP32 firmware.
#
# Usage: scripts/build.sh

set -euo pipefail
cd "$(dirname "$0")/.."

idf.py build

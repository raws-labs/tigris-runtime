#!/bin/bash
# Full clean - removes build dir and sdkconfig. Use after changing target or
# when the build is in a bad state.
#
# Usage: scripts/clean.sh

set -euo pipefail
cd "$(dirname "$0")/.."

idf.py fullclean

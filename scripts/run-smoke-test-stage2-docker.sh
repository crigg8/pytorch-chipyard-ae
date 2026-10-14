#!/usr/bin/env bash
# Run the three FPGA tests using the ELFs produced by run-smoke-test-stage1.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "${SCRIPT_DIR}/smoke-stage2-launch.py" "$@"

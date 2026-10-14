#!/usr/bin/env bash
# Public Stage 2 entrypoint: FireMarshal and FireSim execute in system Docker.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export STAGE2_ARTIFACT_DIR="${STAGE2_ARTIFACT_DIR:-${SCRIPT_DIR}/../examples}"
export STAGE2_SMOKE_ARTIFACT_DIR="${STAGE2_SMOKE_ARTIFACT_DIR:-${SCRIPT_DIR}/../results/smoke-test/artifacts/pytorch-chipyard}"
exec bash "${SCRIPT_DIR}/run-stage2-local-docker.sh" "$@"

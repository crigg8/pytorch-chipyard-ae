#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
IMAGE="${PYTORCH_CHIPYARD_STAGE1_IMAGE:-pytorch-chipyard:stage1}"
COMPILER_SOURCE="${REPO_ROOT}/triton_chipyard/backend/compiler.py"
COMPILER_TARGET="/opt/pytorch-chipyard/triton/python/triton/backends/triton_chipyard/compiler.py"
ABLATION_SOURCE="${REPO_ROOT}/scripts/run-alias-first-cnn.sh"
ABLATION_TARGET="/opt/pytorch-chipyard/scripts/run-alias-first-cnn.sh"

[[ -f "${COMPILER_SOURCE}" ]] || {
  echo "missing compiler source: ${COMPILER_SOURCE}" >&2
  exit 1
}

echo "[alias-first-off] compiling all four CNNs with custom libraries disabled"
sudo docker run --rm \
  --mount "type=bind,src=${REPO_ROOT}/examples,dst=/opt/pytorch-chipyard/examples" \
  --mount "type=bind,src=${COMPILER_SOURCE},dst=${COMPILER_TARGET},readonly" \
  --mount "type=bind,src=${REPO_ROOT}/scripts/stage1.sh,dst=/opt/pytorch-chipyard/scripts/stage1.sh,readonly" \
  --mount "type=bind,src=${ABLATION_SOURCE},dst=${ABLATION_TARGET},readonly" \
  --mount "type=volume,src=pytorch-chipyard-triton-cache,dst=/tmp/triton-chipyard-cache" \
  --env PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY= \
  --env TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG= \
  --env TRITON_CHIPYARD_ENABLE_ALIAS_FIRST=0 \
  "${IMAGE}" \
  bash scripts/run-alias-first-cnn.sh --mode=off

sudo chown -R "$(id -u):$(id -g)" \
  "${REPO_ROOT}/examples/artifact-alexnet/gemmini-alias-first-off" \
  "${REPO_ROOT}/examples/artifact-mobilenetv2/gemmini-alias-first-off" \
  "${REPO_ROOT}/examples/artifact-resnet50/gemmini-alias-first-off" \
  "${REPO_ROOT}/examples/artifact-squeezenet/gemmini-alias-first-off"

echo "[alias-first-off] artifacts ready"

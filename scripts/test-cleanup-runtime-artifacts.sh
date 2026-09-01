#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-chipyard-cleanup-test.XXXXXX")"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

TEST_HOME="${TEST_ROOT}/home/tester"
TEST_REPO="${TEST_HOME}/pytorch-chipyard"
TEST_STATE="${TEST_HOME}/.local/share/pytorch-chipyard"
TEST_FM="${TEST_HOME}/firemarshal"
TEST_FS="${TEST_HOME}/firesim"

mkdir -p \
  "${TEST_REPO}/scripts/figures/results-workload/already-durable" \
  "${TEST_REPO}/examples/.logs" \
  "${TEST_REPO}/examples/artifact-model" \
  "${TEST_REPO}/results/table4/stage1-run/artifacts/kernel/trial-1/gemmini" \
  "${TEST_REPO}/results/smoke-test/run/firesim-results/smoke" \
  "${TEST_STATE}/figure-results/image-a" \
  "${TEST_STATE}/figure-results/legacy" \
  "${TEST_STATE}/figure-results/orphan" \
  "${TEST_STATE}/firemarshal/images/firechip/image-a" \
  "${TEST_STATE}/firemarshal/images/firechip/image-incomplete" \
  "${TEST_STATE}/firemarshal/images/firechip/br-base" \
  "${TEST_STATE}/firemarshal/images/firechip/br.1234" \
  "${TEST_STATE}/firemarshal/workloads/overlay-image-a" \
  "${TEST_STATE}/firesim/deploy/results-workload/raw0" \
  "${TEST_STATE}/firesim/deploy/results-workload/other0" \
  "${TEST_STATE}/firesim/deploy/workloads" \
  "${TEST_STATE}/firesim-runs/run-a" \
  "${TEST_STATE}/firesim-runtime" \
  "${TEST_FM}/boards/default/distros/br/buildroot/output" \
  "${TEST_FM}/boards/default/distros/br/buildroot/dl" \
  "${TEST_FS}/deploy"

cp -- "${SOURCE_DIR}/env.sh" "${TEST_REPO}/scripts/env.sh"
cp -- "${SOURCE_DIR}/stage2-common.sh" "${TEST_REPO}/scripts/stage2-common.sh"
cp -- "${SOURCE_DIR}/cleanup-runtime-artifacts.sh" \
  "${TEST_REPO}/scripts/cleanup-runtime-artifacts.sh"

printf '%s\n' old-model >"${TEST_REPO}/scripts/figures/results-workload/already-durable/model.log"
printf '%s\n' old-autotune >"${TEST_REPO}/scripts/figures/results-workload/already-durable/autotune.log"
printf '%s\n' disposable >"${TEST_REPO}/scripts/figures/results-workload/already-durable/uartlog"
printf '%s\n' model >"${TEST_STATE}/figure-results/image-a/model.log"
printf '%s\n' autotune >"${TEST_STATE}/figure-results/image-a/autotune.log"
touch "${TEST_STATE}/figure-results/image-a/.completed"
printf '%s\n' model >"${TEST_STATE}/figure-results/legacy/model.log"
printf '%s\n' autotune >"${TEST_STATE}/figure-results/legacy/autotune.log"
touch "${TEST_STATE}/figure-results/legacy/.completed"
printf '%s\n' orphan-model >"${TEST_STATE}/figure-results/orphan/model.log"
printf '%s\n' raw-model >"${TEST_STATE}/firesim/deploy/results-workload/raw0/model.log"
printf '%s\n' raw-autotune >"${TEST_STATE}/firesim/deploy/results-workload/raw0/autotune.log"
printf '%s\n' other-model >"${TEST_STATE}/firesim/deploy/results-workload/other0/model.log"
printf '%s\n' other-autotune >"${TEST_STATE}/firesim/deploy/results-workload/other0/autotune.log"
printf '%s\n' image >"${TEST_STATE}/firemarshal/images/firechip/image-a/image-a.img"
printf '%s\n' incomplete >"${TEST_STATE}/firemarshal/images/firechip/image-incomplete/image-incomplete.img"
printf '%s\n' base >"${TEST_STATE}/firemarshal/images/firechip/br-base/br-base.img"
printf '%s\n' base-cache >"${TEST_STATE}/firemarshal/images/firechip/br.1234/br.1234.img"
printf '%s\n' workload >"${TEST_STATE}/firemarshal/workloads/image-a.json"
printf '%s\n' deploy >"${TEST_STATE}/firesim/deploy/workloads/image-a.json"
printf '%s\n' artifact >"${TEST_REPO}/examples/artifact-model/blob.bin"
printf '%s\n' artifact-model-log >"${TEST_REPO}/examples/artifact-model/model.log"
printf '%s\n' object >"${TEST_REPO}/examples/artifact-model/kernel.obj"
printf '%s\n' runner >"${TEST_REPO}/results/table4/stage1-run/artifacts/kernel/trial-1/gemmini/runner.cpp"
printf '%s\n' spec >"${TEST_REPO}/results/table4/stage1-run/artifacts/kernel/trial-1/gemmini/model_spec.json"
printf '%s\n' input >"${TEST_REPO}/results/table4/stage1-run/artifacts/kernel/trial-1/gemmini/input.bin"
printf '%s\n' weights >"${TEST_REPO}/results/table4/stage1-run/artifacts/kernel/trial-1/gemmini/weights.bin"
printf '%s\n' summary >"${TEST_REPO}/results/table4/stage1-run/raw.csv"
ln -s stage1-run "${TEST_REPO}/results/table4/stage1-latest"
printf '%s\n' smoke-model >"${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/model.log"
printf '%s\n' smoke-autotune >"${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/autotune.log"
printf '%s\n' diagnostic >"${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/uartlog"
printf '%s\n' summary >"${TEST_REPO}/results/smoke-test/summary.csv"

HOME="${TEST_HOME}" \
PYTORCH_CHIPYARD_ACCOUNT_ENV= \
TABLE4_ACCOUNT_ENV= \
CHIPYARD_DIR="${TEST_HOME}/chipyard" \
FIREMARSHAL_DIR="${TEST_FM}" \
FIRESIM_DIR="${TEST_FS}" \
FIRESIM_DEPLOY_DIR="${TEST_FS}/deploy" \
PYTORCH_CHIPYARD_STATE_DIR="${TEST_STATE}" \
FIRESIM_WORKLOAD_DIR="${TEST_STATE}/firesim/deploy/workloads" \
PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR="${TEST_STATE}/firesim/deploy/results-workload" \
PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${TEST_REPO}/results/current" \
FIREMARSHAL_IMAGE_DIR="${TEST_STATE}/firemarshal/images/firechip" \
PYTORCH_CHIPYARD_WORKLOAD_DIR="${TEST_STATE}/firemarshal/workloads" \
FIRESIM_RUNS_DIR="${TEST_STATE}/firesim-runs" \
PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${TEST_STATE}/firesim-runtime" \
bash "${TEST_REPO}/scripts/cleanup-runtime-artifacts.sh" \
  --workload image-a --workload image-incomplete --workload raw --final >/dev/null

HOME="${TEST_HOME}" \
PYTORCH_CHIPYARD_ACCOUNT_ENV= \
TABLE4_ACCOUNT_ENV= \
CHIPYARD_DIR="${TEST_HOME}/chipyard" \
FIREMARSHAL_DIR="${TEST_FM}" \
FIRESIM_DIR="${TEST_FS}" \
FIRESIM_DEPLOY_DIR="${TEST_FS}/deploy" \
PYTORCH_CHIPYARD_STATE_DIR="${TEST_STATE}" \
FIRESIM_WORKLOAD_DIR="${TEST_STATE}/firesim/deploy/workloads" \
PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR="${TEST_STATE}/firesim/deploy/results-workload" \
PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${TEST_REPO}/results/current" \
FIREMARSHAL_IMAGE_DIR="${TEST_STATE}/firemarshal/images/firechip" \
PYTORCH_CHIPYARD_WORKLOAD_DIR="${TEST_STATE}/firemarshal/workloads" \
FIRESIM_RUNS_DIR="${TEST_STATE}/firesim-runs" \
PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${TEST_STATE}/firesim-runtime" \
bash "${TEST_REPO}/scripts/cleanup-runtime-artifacts.sh" --all >/dev/null

test -s "${TEST_REPO}/scripts/figures/results-workload/legacy/model.log"
test -s "${TEST_REPO}/scripts/figures/results-workload/legacy/autotune.log"
test -f "${TEST_REPO}/scripts/figures/results-workload/legacy/.completed"
test -s "${TEST_REPO}/scripts/figures/results-workload/orphan/model.log"
test -s "${TEST_REPO}/results/current/raw/model.log"
test -s "${TEST_REPO}/results/current/raw/autotune.log"
test -s "${TEST_REPO}/scripts/figures/results-workload/other/model.log"
test -s "${TEST_REPO}/scripts/figures/results-workload/other/autotune.log"
test ! -e "${TEST_REPO}/scripts/figures/results-workload/already-durable/uartlog"
test -f "${TEST_REPO}/scripts/figures/results-workload/already-durable/.completed"
test ! -e "${TEST_STATE}/firemarshal/images/firechip/image-a"
test ! -e "${TEST_STATE}/firemarshal/images/firechip/image-incomplete"
test -s "${TEST_STATE}/firemarshal/images/firechip/br-base/br-base.img"
test -s "${TEST_STATE}/firemarshal/images/firechip/br.1234/br.1234.img"
test -s "${TEST_REPO}/examples/artifact-model/blob.bin"
test -s "${TEST_REPO}/examples/artifact-model/kernel.obj"
test -L "${TEST_REPO}/results/table4/stage1-latest"
test -s "${TEST_REPO}/results/table4/stage1-latest/raw.csv"
test -s "${TEST_REPO}/results/table4/stage1-latest/artifacts/kernel/trial-1/gemmini/model_spec.json"
test -s "${TEST_REPO}/results/table4/stage1-latest/artifacts/kernel/trial-1/gemmini/input.bin"
test -s "${TEST_REPO}/results/table4/stage1-latest/artifacts/kernel/trial-1/gemmini/weights.bin"
test -s "${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/model.log"
test -s "${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/autotune.log"
test -s "${TEST_REPO}/results/smoke-test/run/firesim-results/smoke/uartlog"
test -s "${TEST_REPO}/results/smoke-test/summary.csv"

printf 'CLEANUP_RUNTIME_ARTIFACTS_TEST=PASS\n'

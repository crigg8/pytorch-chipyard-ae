#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage2-common.sh"

log() { printf '[smoke] %s\n' "$*"; }
pass() { printf '[smoke][PASS] %s\n' "$*"; }
die() { printf '[smoke][error] %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-smoke-test.sh [options]

Run the bounded host smoke tests:
  1. Compile and execute one 32x32x32 GEMM with TVM-Gemmini/Verilator.
  2. Build and execute the same GEMM with PyTorch-Chipyard on FireSim using
     RVV 4 cores, Gemmini 4 cores, and scalar Rocket 16 cores.

Options:
  --artifact-root=PATH  Docker Stage 1 artifact root. Default:
                        results/smoke-test/artifacts/pytorch-chipyard
  --output-dir=PATH     Per-run durable FireSim logs. Must be inside this
                        pytorch-chipyard checkout. Default:
                        results/smoke-test/runs/default (resumable)
  --skip-tvm           Skip the TVM-Gemmini/Verilator check.
  --skip-firesim       Skip the three PyTorch-Chipyard/FireSim checks.
  -h, --help           Show this help.

Environment:
  TABLE4_TVM_AE_ROOT       Prepared TVM-Gemmini tree.
  TABLE4_TVM_BUILD_DIR     Reusable LLVM-enabled TVM build directory.
  TABLE4_VERILATOR_BIN     Required prebuilt Verilator simulator. This script
                           never builds or modifies the shared simulator.
  PYTORCH_CHIPYARD_ACCOUNT_ENV
                           Optional host environment file.
  PYTORCH_CHIPYARD_SMOKE_FIRESIM_HW_CONFIG_OVERRIDES
                           Smoke-specific workload=hardware overrides.
EOF
}

run_logged() {
  local log_path="$1"
  shift
  local rc
  mkdir -p "$(dirname -- "${log_path}")"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

require_nonempty() {
  local path="$1"
  [[ -s "${path}" ]] || die "expected non-empty smoke-test output: ${path}"
}

artifact_root="${REPO_ROOT}/results/smoke-test/artifacts/pytorch-chipyard"
output_dir=""
skip_tvm=0
skip_firesim=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-root=*) artifact_root="${1#*=}"; shift ;;
    --artifact-root)
      [[ "$#" -ge 2 ]] || die "--artifact-root requires a value"
      artifact_root="$2"
      shift 2
      ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir)
      [[ "$#" -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --skip-tvm) skip_tvm=1; shift ;;
    --skip-firesim) skip_firesim=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1'; pass --help for usage" ;;
  esac
done

if [[ "${skip_tvm}" -eq 1 && "${skip_firesim}" -eq 1 ]]; then
  die "--skip-tvm and --skip-firesim cannot be used together"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ -z "${output_dir}" ]]; then
  output_dir="${REPO_ROOT}/results/smoke-test/runs/default"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${PWD}/${output_dir}"
fi
if [[ "${artifact_root}" != /* ]]; then
  artifact_root="${PWD}/${artifact_root}"
fi
mkdir -p "${output_dir}/logs" "${output_dir}/setup"
output_dir="$(cd -- "${output_dir}" >/dev/null 2>&1 && pwd -P)"
case "${output_dir}" in
  "${REPO_ROOT}"/*) ;;
  *) die "--output-dir must be inside ${REPO_ROOT}: ${output_dir}" ;;
esac

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
  log "loaded host environment: ${account_env}"
fi
set +u
source "${SCRIPT_DIR}/env.sh"
set -u

smoke_workload_root=""
smoke_collected_root=""
smoke_runtime_root=""
smoke_cleanup_workloads=()

smoke_cleanup_on_exit() {
  local status=$?
  local cleanup_status=0
  local workload_dir entry

  local workload
  local -a cleanup_args=(--all --final)

  trap - EXIT
  for workload in "${smoke_cleanup_workloads[@]}"; do
    cleanup_args+=(--workload "${workload}")
  done
  if [[ -n "${smoke_workload_root}" ]]; then
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${smoke_workload_root}" \
      PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${smoke_collected_root}" \
      PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${smoke_runtime_root}" \
      bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" "${cleanup_args[@]}" || cleanup_status=$?
  else
    bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" "${cleanup_args[@]}" || cleanup_status=$?
  fi

  if [[ -d "${output_dir}/firesim-results" ]]; then
    while IFS= read -r -d '' workload_dir; do
      while IFS= read -r -d '' entry; do
        case "$(basename -- "${entry}")" in
          model.log | autotune.log | .completed) ;;
          *) rm -rf -- "${entry}" ;;
        esac
      done < <(find "${workload_dir}" -mindepth 1 -maxdepth 1 -print0)
    done < <(find "${output_dir}/firesim-results" -mindepth 1 -maxdepth 1 -type d -print0)
  fi
  if [[ "${status}" -eq 0 ]]; then
    rm -rf -- "${output_dir}/logs" "${output_dir}/setup" "${output_dir}/artifacts"
    rm -rf -- "${REPO_ROOT}/results/smoke-test/setup"
    # Preserve Stage 1 runner/input/weights for a reproducible resume. Only
    # kernel objects and completed Stage 2 ELF derivatives are disposable.
    find "${artifact_root}" -type f \
      \( -name '*.obj' -o -name 'model.elf' -o -name 'model-*core.elf' -o \
         -name '.model-*core.elf-fingerprint' \) -delete 2>/dev/null || true
  else
    printf '[smoke][warn] preserving failure diagnostics under %s\n' \
      "${output_dir}" >&2
  fi

  if [[ "${cleanup_status}" -ne 0 ]]; then
    printf '[smoke][warn] final runtime cleanup failed with status %s\n' \
      "${cleanup_status}" >&2
    [[ "${status}" -ne 0 ]] || status="${cleanup_status}"
  fi
  exit "${status}"
}
trap smoke_cleanup_on_exit EXIT

run_tvm_smoke() {
  local tvm_root tvm_build_dir simulator target_header
  local project elf compile_log verilator_log

  tvm_root="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
  require_file "${tvm_root}/scripts/env.sh"
  [[ -n "${CHIPYARD_DIR:-}" ]] || die "CHIPYARD_DIR is required for the Verilator smoke test"

  simulator="${TABLE4_VERILATOR_BIN:-}"
  target_header="${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"

  [[ -n "${simulator}" ]] || \
    die "TABLE4_VERILATOR_BIN is required; ask the shared simulator owner to build it first"
  [[ -x "${simulator}" ]] || \
    die "TABLE4_VERILATOR_BIN is not executable: ${simulator}"
  require_file "${target_header}"
  grep -Eq '^#define DIM 16$' "${target_header}" || \
    die "${target_header} is not the DIM=16 target paired with TABLE4_VERILATOR_BIN"
  grep -Eq '^typedef int8_t elem_t;$' "${target_header}" || \
    die "${target_header} is not the INT8 target paired with TABLE4_VERILATOR_BIN"

  tvm_build_dir="${TABLE4_TVM_BUILD_DIR:-${REPO_ROOT}/results/smoke-test/setup/tvm-build-llvm}"
  log "preparing the reusable LLVM-enabled TVM build"
  run_logged "${output_dir}/logs/tvm-prepare.log" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/prepare-tvm.sh" \
      --build-dir "${tvm_build_dir}" \
    || die "TVM preparation failed; see ${output_dir}/logs/tvm-prepare.log"

  project="${output_dir}/artifacts/tvm-gemmini/smoke-gemm-32/generated-project"
  compile_log="${output_dir}/logs/tvm-smoke-gemm-compile.log"
  log "compiling the TVM-Gemmini 32x32x32 GEMM"
  run_logged "${compile_log}" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" \
      TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
      TABLE4_VERILATOR_BIN="${simulator}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/compile-kernel.sh" \
      --kernel smoke_gemm_32 --output-dir "${project}" \
    || die "TVM-Gemmini smoke compile failed; see ${compile_log}"

  elf="${project}/src/build/dense-baremetal"
  require_file "${elf}"
  verilator_log="${output_dir}/logs/tvm-smoke-gemm-verilator.log"
  log "executing the TVM-Gemmini 32x32x32 GEMM on Verilator"
  run_logged "${verilator_log}" \
    env TABLE4_TVM_AE_ROOT="${tvm_root}" \
      TABLE4_TVM_BUILD_DIR="${tvm_build_dir}" \
      TABLE4_VERILATOR_BIN="${simulator}" \
    bash "${SCRIPT_DIR}/tvm-gemmini-table4/run-verilator.sh" --elf "${elf}" \
    || die "TVM-Gemmini Verilator smoke test failed; see ${verilator_log}"

  grep -Fq 'KERNEL_SHAPE=32x32x32' "${verilator_log}" || \
    die "32x32x32 runtime marker missing from ${verilator_log}"
  grep -Fq 'KERNEL_EXECUTION=PASS' "${verilator_log}" || \
    die "TVM runtime did not report PASS in ${verilator_log}"
  grep -Fq 'VERILATOR_STATUS=PASS' "${verilator_log}" || \
    die "Verilator wrapper did not report PASS in ${verilator_log}"
  pass "TVM-Gemmini Verilator shape=32x32x32 ELF=${elf} log=${verilator_log}"
}

run_firesim_smoke() {
  local workload_root collected_root firesim_log_root runtime_root
  local smoke_hw_overrides
  local backend core artifact workload result_dir label output_file
  local pending_workloads=()
  local backends=(rvv gemmini scalar)
  local cores=(4 4 16)
  local workloads=(
    smoke-gemm-rvv-4core
    smoke-gemm-gemmini-4core
    smoke-gemm-scalar-16core
  )
  local labels=("RVV 4-core" "Gemmini 4-core" "scalar 16-core")

  [[ -n "${CHIPYARD_DIR:-}" ]] || die "CHIPYARD_DIR is required for the FireSim smoke tests"
  [[ -d "${artifact_root}" ]] || \
    die "smoke artifacts not found at ${artifact_root}; run scripts/run-smoke-test-stage1.sh in Docker first"

  for index in "${!backends[@]}"; do
    backend="${backends[${index}]}"
    artifact="${artifact_root}/${backend}"
    require_file "${artifact}/model_spec.json"
    require_file "${artifact}/input.bin"
    require_file "${artifact}/weights.bin"
  done

  workload_root="${output_dir}/setup/firemarshal-workloads"
  collected_root="${output_dir}/firesim-results"
  firesim_log_root="${output_dir}/logs/firesim"
  runtime_root="${output_dir}/setup/firesim-runtime"
  smoke_workload_root="${workload_root}"
  smoke_collected_root="${collected_root}"
  smoke_runtime_root="${runtime_root}"
  smoke_cleanup_workloads=("${workloads[@]}")
  smoke_hw_overrides="${PYTORCH_CHIPYARD_SMOKE_FIRESIM_HW_CONFIG_OVERRIDES:-smoke-gemm-rvv-4core=alveo_u250_firesim_minv128d64_rocket_4core_no_nic}"
  mkdir -p "${workload_root}" "${collected_root}" "${firesim_log_root}" "${runtime_root}"

  for index in "${!workloads[@]}"; do
    workload="${workloads[${index}]}"
    if pc_stage2_result_complete "${collected_root}" "${workload}"; then
      log "resume: skipping completed smoke workload ${workload}"
      env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
        PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
        PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${runtime_root}" \
        bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" --workload "${workload}"
      rm -f -- \
        "${artifact_root}/${backends[${index}]}/model-${cores[${index}]}core.elf" \
        "${artifact_root}/${backends[${index}]}/.model-${cores[${index}]}core.elf-fingerprint" \
        "${artifact_root}/${backends[${index}]}/model.elf"
    else
      pending_workloads+=("${workload}")
    fi
  done

  if [[ "${#pending_workloads[@]}" -gt 0 ]]; then
    preflight_args=(--preflight-only)
    for workload in "${pending_workloads[@]}"; do
      preflight_args+=(--workload "${workload}")
    done
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
      PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${runtime_root}" \
      PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_OVERRIDES="${smoke_hw_overrides}" \
      bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${preflight_args[@]}"
  fi

  built_image_count=0
  for index in "${!workloads[@]}"; do
    workload="${workloads[${index}]}"
    if pc_stage2_result_complete "${collected_root}" "${workload}"; then
      continue
    fi
    backend="${backends[${index}]}"
    core="${cores[${index}]}"
    artifact="${artifact_root}/${backend}"

    log "${workload}: building its ELF and deleting kernel .obj files"
    run_logged "${output_dir}/logs/${workload}-elf.log" \
      bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
        --artifact-dir "${artifact}" --core "${core}" \
      || die "${workload}: ELF construction failed"
    require_file "${artifact}/model-${core}core.elf"

    log "${workload}: packaging only this smoke workload"
    run_logged "${output_dir}/logs/${workload}-package.log" \
      env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
        PYTORCH_CHIPYARD_OMP_FIRST_CPU_SMOKE_GEMM_RVV_4CORE=0 \
      bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" \
        --no-clean --artifact-dir "${artifact}" --core "${core}" \
      || die "${workload}: packaging failed"

    image_args=(--workload "${workload}")
    [[ "${built_image_count}" -eq 0 ]] || image_args+=(--reuse-shared-kernel-build)
    log "${workload}: building one FireMarshal image"
    run_logged "${output_dir}/logs/${workload}-image.log" \
      env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      bash "${SCRIPT_DIR}/build-firemarshal-images.sh" "${image_args[@]}" \
      || die "${workload}: FireMarshal image build failed"
    built_image_count=$((built_image_count + 1))

    log "${workload}: running FireSim"
    run_logged "${output_dir}/logs/${workload}-firesim.log" \
      env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
        PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
        PYTORCH_CHIPYARD_LOG_DIR="${firesim_log_root}" \
        PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${runtime_root}" \
        PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=1 \
        PYTORCH_CHIPYARD_DEFER_FINAL_CLEANUP=1 \
        PYTORCH_CHIPYARD_FIRESIM_HW_CONFIG_OVERRIDES="${smoke_hw_overrides}" \
      bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
        --workload "${workload}" --rvv-panic-retries=0 \
      || die "${workload}: FireSim smoke test failed"

    pc_stage2_result_complete "${collected_root}" "${workload}" || \
      die "${workload}: FireSim returned without both durable logs"
    env PYTORCH_CHIPYARD_WORKLOAD_DIR="${workload_root}" \
      PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${collected_root}" \
      PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${runtime_root}" \
      bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" --workload "${workload}"
    rm -f -- \
      "${artifact}/model-${core}core.elf" \
      "${artifact}/.model-${core}core.elf-fingerprint" \
      "${artifact}/model.elf"

  done

  for index in "${!workloads[@]}"; do
    workload="${workloads[${index}]}"
    label="${labels[${index}]}"
    result_dir="${collected_root}/${workload}"
    for output_file in model.log autotune.log; do
      require_nonempty "${result_dir}/${output_file}"
    done
    pass "PyTorch-Chipyard ${label} shape=32x32x32 results=${result_dir}"
  done
}

log "output: ${output_dir}"
if [[ "${skip_tvm}" -eq 0 ]]; then
  run_tvm_smoke
fi
if [[ "${skip_firesim}" -eq 0 ]]; then
  run_firesim_smoke
fi

mkdir -p "${REPO_ROOT}/results/smoke-test"
ln -sfn "${output_dir}" \
  "${REPO_ROOT}/results/smoke-test/latest"
printf 'SMOKE_TEST_RESULTS=%s\n' "${output_dir}"
printf 'SMOKE_TEST_STATUS=PASS\n'

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run-table4-kernel-compile.sh --kernel=ID --artifact-dir=PATH' \
    '' \
    'Compile one built-in Table 4 kernel with the normal bounded' \
    'TorchInductor autotuning set. Gemmini max autotune is forced off.'
}

kernel=""
artifact_dir=""
force_recompile=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --kernel=*) kernel="${1#*=}"; shift ;;
    --kernel) [[ "$#" -ge 2 ]] || pc_usage_error "--kernel requires a value"; kernel="$2"; shift 2 ;;
    --artifact-dir=*) artifact_dir="${1#*=}"; shift ;;
    --artifact-dir) [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"; artifact_dir="$2"; shift 2 ;;
    --force-recompile) force_recompile=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) pc_usage_error "unknown argument '$1'" ;;
  esac
done

[[ -n "${kernel}" ]] || pc_usage_error "--kernel is required"
[[ -n "${artifact_dir}" ]] || pc_usage_error "--artifact-dir is required"

pc_prepare_environment
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0
pc_write_artifact_workload_hint "${artifact_dir}" "table4-kernels/${kernel}/gemmini"
pc_write_artifact_build_plan "${artifact_dir}" gemmini 4
if [[ "${force_recompile}" -eq 1 ]]; then
  rm -f -- "$(pc_compile_stamp_path "${artifact_dir}")"
fi
pc_run_compile_once gemmini "${artifact_dir}" "table4-${kernel}" \
  "${REPO_ROOT}/examples/table4-kernel.py" --kernel "${kernel}"

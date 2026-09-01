#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_custom_linalg_matmul.sh [--artifact-dir=PATH]

Generate the linalg.matmul -> libtriton_chipyard.a Gemmini 2-core artifact.
EOF
}

artifact_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"
      artifact_dir="$2"
      shift 2
      ;;
    --artifact-dir=*)
      artifact_dir="${1#--artifact-dir=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      pc_usage_error "unknown argument '$1'"
      ;;
  esac
done

pc_prepare_environment

backend=gemmini
suffix="custom-linalg-matmul/gemmini"
storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
output_dir="${artifact_dir:-${PC_REPO_ROOT}/examples/${storage_suffix}}"
library_path="${TRITON_CHIPYARD_EXTERN_CALL_LIBRARY:-${PC_REPO_ROOT}/examples/custom-linalg-matmul/libtriton_chipyard.a}"
pc_require_file "${library_path}"
export TRITON_CHIPYARD_EXTERN_CALL_LIBRARY="$(cd -- "$(dirname -- "${library_path}")" && pwd -P)/$(basename -- "${library_path}")"
unset TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG

pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
pc_write_artifact_build_plan "${output_dir}" "${backend}" 2
pc_run_compile_once \
  "${backend}" \
  "${output_dir}" \
  "custom-linalg-matmul" \
  "${PC_REPO_ROOT}/examples/custom-linalg-matmul/model.py"

pc_log "done; build the 2-core ELF with scripts/build-chipyard-elves.sh"

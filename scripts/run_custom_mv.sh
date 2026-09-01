#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_custom_mv.sh [--artifact-dir=PATH]

Generate the aten.mv libpytorch_chipyard.a test artifact for Gemmini 2-core.
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
suffix="custom-mv/gemmini"
storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
output_dir="${artifact_dir:-${PC_REPO_ROOT}/examples/${storage_suffix}}"
library_path="${PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY:-${PC_REPO_ROOT}/examples/custom-mv/libpytorch_chipyard.a}"
pc_require_file "${library_path}"
export PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY="$(cd -- "$(dirname -- "${library_path}")" && pwd -P)/$(basename -- "${library_path}")"

pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
pc_write_artifact_build_plan "${output_dir}" "${backend}" 2
pc_run_compile_once \
  "${backend}" \
  "${output_dir}" \
  "custom-mv" \
  "${PC_REPO_ROOT}/examples/custom-mv/model.py"

pc_log "done; build the 2-core ELF with scripts/build-chipyard-elves.sh"

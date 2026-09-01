#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-smoke-test-stage1.sh [--artifact-root=PATH]

Compile a 32x32x32 GEMM for the RVV, Gemmini, and scalar
PyTorch-Chipyard backends. The generated plans select RVV 4 cores,
Gemmini 4 cores, and scalar 16 cores for the host smoke test.

Options:
  --artifact-root=PATH  Artifact root. Default:
                        results/smoke-test/artifacts/pytorch-chipyard
  -h, --help            Show this help.
EOF
}

artifact_root="${REPO_ROOT}/results/smoke-test/artifacts/pytorch-chipyard"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-root=*) artifact_root="${1#*=}"; shift ;;
    --artifact-root)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-root requires a value"
      artifact_root="$2"
      shift 2
      ;;
    -h | --help) usage; exit 0 ;;
    *) pc_usage_error "unknown argument '$1'" ;;
  esac
done

if [[ "${artifact_root}" != /* ]]; then
  artifact_root="${PWD}/${artifact_root}"
fi
mkdir -p "${artifact_root}"
artifact_root="$(cd -- "${artifact_root}" >/dev/null 2>&1 && pwd -P)"
case "${artifact_root}" in
  "${REPO_ROOT}"/*) ;;
  *) pc_usage_error "--artifact-root must be inside ${REPO_ROOT}: ${artifact_root}" ;;
esac

smoke_stage1_cleanup_on_exit() {
  local status=$?
  trap - EXIT
  if [[ "${status}" -ne 0 ]]; then
    rm -rf -- "${artifact_root}"
  fi
  exit "${status}"
}
trap smoke_stage1_cleanup_on_exit EXIT

pc_prepare_environment
export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0

backends=(rvv gemmini scalar)
cores=(4 4 16)
for index in "${!backends[@]}"; do
  backend="${backends[${index}]}"
  core="${cores[${index}]}"
  output_dir="${artifact_root}/${backend}"
  pc_write_artifact_workload_hint "${output_dir}" "smoke-gemm/${backend}"
  pc_write_artifact_build_plan "${output_dir}" "${backend}" "${core}"
  pc_run_compile_once \
    "${backend}" \
    "${output_dir}" \
    "smoke-gemm-${backend}" \
    "${REPO_ROOT}/examples/smoke-gemm.py"
  pc_require_artifacts "${output_dir}"
  printf '[smoke-stage1][PASS] backend=%s cores=%s artifact=%s\n' \
    "${backend}" "${core}" "${output_dir}"
done

# The Docker container may run as a remapped root user. Stage 2 is run by the
# host AE account and must be able to read and update these bind-mounted files.
chmod -R a+rwX "${artifact_root}"

printf 'SMOKE_STAGE1_ARTIFACT_ROOT=%s\n' "${artifact_root}"
printf 'SMOKE_STAGE1_STATUS=PASS\n'

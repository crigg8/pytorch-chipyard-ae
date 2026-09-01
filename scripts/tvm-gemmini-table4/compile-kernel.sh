#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
fi
TVM_AE_ROOT="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
account_chipyard_dir="${CHIPYARD_DIR:-}"
source "${TVM_AE_ROOT}/scripts/env.sh"
if [[ -n "${account_chipyard_dir}" ]]; then
  export CHIPYARD_DIR="${account_chipyard_dir}"
  export FIRESIM_DIR="${CHIPYARD_DIR}/sims/firesim"
  export DTC_BIN="${CHIPYARD_DIR}/.conda-env/bin/dtc"
  export PK_BIN="${CHIPYARD_DIR}/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk"
  export LD_LIBRARY_PATH="${CHIPYARD_DIR}/.conda-env/riscv-tools/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${TABLE4_TVM_BUILD_DIR:-}" ]]; then
  TVM_BUILD_DIR="${TABLE4_TVM_BUILD_DIR}"
  export TVM_BUILD_DIR
fi
export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}"
export LD_LIBRARY_PATH="${TVM_BUILD_DIR}:${LD_LIBRARY_PATH:-}"
export TABLE4_TVM_GEMMINI_INCLUDE="${TABLE4_TVM_GEMMINI_INCLUDE:-${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include}"
export TABLE4_TVM_CHIPYARD_COMMIT="$(git -C "${CHIPYARD_DIR}" rev-parse HEAD)"

kernel=""
output_dir=""
compile_only=0
force=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --kernel=*) kernel="${1#*=}"; shift ;;
    --kernel) [[ "$#" -ge 2 ]] || exit 2; kernel="$2"; shift 2 ;;
    --output-dir=*) output_dir="${1#*=}"; shift ;;
    --output-dir) [[ "$#" -ge 2 ]] || exit 2; output_dir="$2"; shift 2 ;;
    --compile-only) compile_only=1; shift ;;
    --force) force=1; shift ;;
    -h | --help)
      printf '%s\n' \
        'Usage: compile-kernel.sh --kernel=ID --output-dir=PATH [--compile-only]'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${kernel}" && -n "${output_dir}" ]] || exit 2
[[ -f "${TABLE4_TVM_GEMMINI_INCLUDE}/gemmini_params.h" ]] || {
  printf 'Chipyard Gemmini headers not found: %s\n' "${TABLE4_TVM_GEMMINI_INCLUDE}" >&2
  exit 1
}
[[ -f "${TVM_BUILD_DIR}/libtvm.so" ]] || {
  printf 'LLVM-enabled TVM build not found: %s/libtvm.so\n' "${TVM_BUILD_DIR}" >&2
  printf 'Run scripts/tvm-gemmini-table4/prepare-tvm.sh first.\n' >&2
  exit 1
}
args=(--kernel "${kernel}" --output-dir "${output_dir}")
[[ "${compile_only}" -eq 0 ]] || args+=(--compile-only)
[[ "${force}" -eq 0 ]] || args+=(--force)
exec "${TVM_ENV}/bin/python" "${SCRIPT_DIR}/compile-kernel.py" "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

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

elf=""
config="${TABLE4_VERILATOR_CONFIG:-OriginalGemminiRocketConfig}"
simulator="${TABLE4_VERILATOR_BIN:-}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --elf=*) elf="${1#*=}"; shift ;;
    --elf) [[ "$#" -ge 2 ]] || exit 2; elf="$2"; shift 2 ;;
    --config=*) config="${1#*=}"; shift ;;
    --config) [[ "$#" -ge 2 ]] || exit 2; config="$2"; shift 2 ;;
    --simulator=*) simulator="${1#*=}"; shift ;;
    --simulator) [[ "$#" -ge 2 ]] || exit 2; simulator="$2"; shift 2 ;;
    -h | --help)
      printf '%s\n' 'Usage: run-verilator.sh --elf=PATH [--simulator=PATH]'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n "${elf}" && -f "${elf}" ]] || { printf 'bare-metal ELF not found: %s\n' "${elf}" >&2; exit 1; }
if [[ -z "${simulator}" ]]; then
  printf 'TABLE4_VERILATOR_BIN is required for config %s; alternatively pass --simulator.\n' \
    "${config}" >&2
  exit 1
fi
[[ -x "${simulator}" ]] || {
  printf 'Verilator simulator not executable: %s\n' "${simulator}" >&2
  printf 'Ask the shared simulator owner to rebuild it.\n' >&2
  exit 1
}

project_src="$(dirname -- "$(dirname -- "${elf}")")"
project_header="${project_src}/include/gemmini_params.h"
target_include="${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include"
target_header="${target_include}/gemmini_params.h"
[[ -f "${project_header}" ]] || { printf 'Gemmini parameter header not found: %s\n' "${project_header}" >&2; exit 1; }
[[ -f "${target_header}" ]] || { printf 'Verilator target header not found: %s\n' "${target_header}" >&2; exit 1; }
grep -Eq '^#define DIM 16$' "${project_header}" || { printf 'expected DIM=16 in %s\n' "${project_header}" >&2; exit 1; }
grep -Eq '^typedef int8_t elem_t;$' "${project_header}" || { printf 'expected int8 elem_t in %s\n' "${project_header}" >&2; exit 1; }
grep -Eq '^typedef int32_t acc_t;$' "${project_header}" || { printf 'expected int32 acc_t in %s\n' "${project_header}" >&2; exit 1; }
cmp -s "${project_header}" "${target_header}" || {
  printf 'compiled Gemmini parameters do not match the Verilator target: %s vs %s\n' \
    "${project_header}" "${target_header}" >&2
  printf 'Recompile the TVM kernel after building the Verilator target.\n' >&2
  exit 1
}
for target_api_header in "${target_include}"/*.h; do
  project_api_header="${project_src}/include/$(basename -- "${target_api_header}")"
  [[ -f "${project_api_header}" ]] || {
    printf 'compiled project is missing target header: %s\n' "${project_api_header}" >&2
    exit 1
  }
  cmp -s "${project_api_header}" "${target_api_header}" || {
    printf 'compiled Gemmini API header does not match the Verilator target: %s\n' \
      "$(basename -- "${target_api_header}")" >&2
    printf 'Recompile the TVM kernel after building the Verilator target.\n' >&2
    exit 1
  }
done

printf 'SIMULATOR=%s\n' "${simulator}"
printf 'CHIPYARD_COMMIT=%s\n' "$(git -C "${CHIPYARD_DIR}" rev-parse HEAD)"
printf 'GEMMINI_PARAMS_SHA256=%s\n' "$(sha256sum "${target_header}" | awk '{print $1}')"

tmp_log="$(mktemp)"
cleanup() { rm -f -- "${tmp_log}"; }
trap cleanup EXIT
elf_abs="$(cd -- "$(dirname -- "${elf}")" >/dev/null 2>&1 && pwd -P)/$(basename -- "${elf}")"
set +e
(
  cd "$(dirname -- "${simulator}")"
  "${simulator}" "${elf_abs}"
) 2>&1 | tee "${tmp_log}"
sim_rc=${PIPESTATUS[0]}
set -e
[[ "${sim_rc}" -eq 0 ]] || exit "${sim_rc}"
grep -q '^KERNEL_EXECUTION=PASS' "${tmp_log}" || {
  printf 'kernel execution marker was not PASS\n' >&2
  exit 1
}
printf 'VERILATOR_STATUS=PASS\n'

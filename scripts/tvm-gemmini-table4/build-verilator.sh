#!/usr/bin/env bash
set -euo pipefail

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
fi

config="${TABLE4_VERILATOR_CONFIG:-OriginalGemminiRocketConfig}"
jobs="${TABLE4_VERILATOR_BUILD_JOBS:-8}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config=*) config="${1#*=}"; shift ;;
    --config) [[ "$#" -ge 2 ]] || exit 2; config="$2"; shift 2 ;;
    -j) [[ "$#" -ge 2 ]] || exit 2; jobs="$2"; shift 2 ;;
    -h | --help)
      printf '%s\n' \
        'Usage: build-verilator.sh [--config=CONFIG] [-j N]' \
        'Default: OriginalGemminiRocketConfig (INT8, DIM=16, one Rocket core).' \
        'Environment: TABLE4_FIRTOOL_BIN may select an explicit firtool binary.' \
        'TABLE4_VERILATOR_RISCV may select the Chipyard host fesvr installation.'
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid job count: %s\n' "${jobs}" >&2; exit 2; }
[[ -n "${CHIPYARD_DIR:-}" ]] || {
  printf 'CHIPYARD_DIR is required to build the shared Verilator simulator.\n' >&2
  exit 1
}
export FIRESIM_DIR="${FIRESIM_DIR:-${CHIPYARD_DIR}/sims/firesim}"
export DTC_BIN="${DTC_BIN:-${CHIPYARD_DIR}/.conda-env/bin/dtc}"
export PK_BIN="${PK_BIN:-${CHIPYARD_DIR}/.conda-env/riscv-tools/riscv64-unknown-elf/bin/pk}"
sim_dir="${CHIPYARD_DIR}/sims/verilator"
[[ -f "${sim_dir}/Makefile" ]] || { printf 'Chipyard Verilator directory not found: %s\n' "${sim_dir}" >&2; exit 1; }

firtool_bin="${TABLE4_FIRTOOL_BIN:-}"
if [[ -z "${firtool_bin}" ]]; then
  firtool_bin="${CHIPYARD_DIR}/.conda-env/riscv-tools/bin/firtool"
  if [[ ! -x "${firtool_bin}" ]]; then
    firtool_bin="$(command -v firtool || true)"
  fi
fi
[[ -n "${firtool_bin}" && -x "${firtool_bin}" ]] || {
  printf 'firtool not found; expected %s or set TABLE4_FIRTOOL_BIN\n' \
    "${CHIPYARD_DIR}/.conda-env/riscv-tools/bin/firtool" >&2
  exit 1
}
export PATH="$(dirname -- "${firtool_bin}"):${PATH}"

# tvm-gemmini-ae intentionally exports its own older RISC-V compiler for TVM
# and bare-metal kernel construction. Chipyard's Verilator makefiles also use
# RISCV, but there it means the matching host-side Spike/FESVR installation.
# Keep this override local to this subprocess so the two toolchains cannot leak
# into one another.
verilator_riscv="${TABLE4_VERILATOR_RISCV:-${CHIPYARD_DIR}/.conda-env/riscv-tools}"
for required_path in \
  "${verilator_riscv}/include/fesvr/htif.h" \
  "${verilator_riscv}/include/fesvr/tsi.h" \
  "${verilator_riscv}/include/fesvr/memif.h" \
  "${verilator_riscv}/lib/libfesvr.a"; do
  [[ -r "${required_path}" ]] || {
    printf 'Chipyard Verilator dependency not found: %s\n' "${required_path}" >&2
    printf 'Set TABLE4_VERILATOR_RISCV to a matching Spike/FESVR installation.\n' >&2
    exit 1
  }
done
if [[ ! -r "${verilator_riscv}/lib/libriscv.so" && \
      ! -r "${verilator_riscv}/lib/libriscv.a" ]]; then
  printf 'Chipyard Verilator dependency not found: %s/lib/libriscv.{so,a}\n' \
    "${verilator_riscv}" >&2
  exit 1
fi
export RISCV="${verilator_riscv}"
export PATH="${CHIPYARD_DIR}/.conda-env/bin:${RISCV}/bin:${PATH}"
export LD_LIBRARY_PATH="${RISCV}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${RISCV}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${RISCV}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"

# Verilator records RISCV include/library paths in VTestDriver.mk. A failed run
# may therefore retain a syntactically valid makefile with TVM's incompatible
# toolchain paths. Remove only that generated makefile so the normal Chipyard
# rule recreates the model directory with the selected FESVR installation.
generated_model_dir="${sim_dir}/generated-src/chipyard.harness.TestHarness.${config}/chipyard.harness.TestHarness.${config}"
generated_model_mk="${generated_model_dir}/VTestDriver.mk"
if [[ -f "${generated_model_mk}" ]] && \
   ! grep -Fq -- "-I${RISCV}/include" "${generated_model_mk}"; then
  rm -f -- "${generated_model_mk}"
  printf '[tvm-verilator] invalidated stale model makefile: %s\n' \
    "${generated_model_mk}"
fi

target_include="${CHIPYARD_DIR}/generators/gemmini/software/gemmini-rocc-tests/include"
target_header="${target_include}/gemmini_params.h"
shared_header="${TABLE4_FIRESIM_GEMMINI_HEADER:-${CHIPYARD_DIR}/sims/firesim/deploy/results-build/gemmini_params.h}"
shared_header_backup=""
shared_header_existed=0
shared_header_stamp_before=""

restore_shared_header() {
  local rc="$?"
  trap - EXIT
  if [[ "${shared_header_existed}" -eq 1 ]]; then
    if cp -f -- "${shared_header_backup}" "${shared_header}"; then
      rm -f -- "${shared_header_backup}"
      printf '[tvm-verilator] restored shared FireSim header: %s\n' "${shared_header}"
    else
      printf 'failed to restore shared FireSim header; backup retained at %s\n' \
        "${shared_header_backup}" >&2
      rc=1
    fi
  elif [[ -e "${shared_header}" ]]; then
    if ! rm -f -- "${shared_header}"; then
      printf 'failed to remove Table 4 generated shared header: %s\n' "${shared_header}" >&2
      rc=1
    fi
  fi
  exit "${rc}"
}

if [[ -e "${shared_header}" ]]; then
  [[ -f "${shared_header}" && -r "${shared_header}" && -w "${shared_header}" ]] || {
    printf 'shared FireSim Gemmini header must be a readable/writable file: %s\n' \
      "${shared_header}" >&2
    exit 1
  }
  shared_header_backup="$(mktemp)"
  if ! cp -- "${shared_header}" "${shared_header_backup}"; then
    rm -f -- "${shared_header_backup}"
    printf 'failed to back up shared FireSim header: %s\n' "${shared_header}" >&2
    exit 1
  fi
  shared_header_existed=1
  shared_header_stamp_before="$(stat -c '%y:%s' "${shared_header}")"
elif [[ -d "$(dirname -- "${shared_header}")" && ! -w "$(dirname -- "${shared_header}")" ]]; then
  printf 'shared FireSim results directory is not writable: %s\n' \
    "$(dirname -- "${shared_header}")" >&2
  exit 1
fi
trap restore_shared_header EXIT

printf '[tvm-verilator] building CONFIG=%s with %s jobs\n' "${config}" "${jobs}"
printf '[tvm-verilator] firtool=%s\n' "${firtool_bin}"
printf '[tvm-verilator] RISCV=%s\n' "${RISCV}"
make -C "${sim_dir}" -j"${jobs}" CONFIG="${config}"
simulator="${sim_dir}/simulator-chipyard.harness-${config}"
[[ -x "${simulator}" ]] || { printf 'simulator was not produced: %s\n' "${simulator}" >&2; exit 1; }

# The author Chipyard tree redirects Gemmini's generated header into FireSim's
# results-build directory. If elaboration updated it, copy that generated header
# to the standard include tree used by TVM, then let the EXIT trap restore the
# pre-existing FireSim/RVV header. Upstream Chipyard writes target_header
# directly, in which case shared_header remains unchanged and no copy is needed.
if [[ -f "${shared_header}" ]]; then
  shared_header_stamp_after="$(stat -c '%y:%s' "${shared_header}")"
  if [[ "${shared_header_existed}" -eq 0 || "${shared_header_stamp_after}" != "${shared_header_stamp_before}" ]]; then
    [[ -d "${target_include}" ]] || { printf 'Gemmini include directory not found: %s\n' "${target_include}" >&2; exit 1; }
    cp -f -- "${shared_header}" "${target_header}"
    printf '[tvm-verilator] synchronized generated target header: %s\n' "${target_header}"
  fi
fi
[[ -f "${target_header}" ]] || { printf 'Verilator target header not found: %s\n' "${target_header}" >&2; exit 1; }
if [[ "${config}" == "OriginalGemminiRocketConfig" ]]; then
  grep -Eq '^#define DIM 16$' "${target_header}" || { printf 'expected DIM=16 in %s\n' "${target_header}" >&2; exit 1; }
  grep -Eq '^typedef int8_t elem_t;$' "${target_header}" || { printf 'expected int8 elem_t in %s\n' "${target_header}" >&2; exit 1; }
  grep -Eq '^typedef int32_t acc_t;$' "${target_header}" || { printf 'expected int32 acc_t in %s\n' "${target_header}" >&2; exit 1; }
fi
printf '[tvm-verilator] simulator=%s\n' "${simulator}"
printf '[tvm-verilator] chipyard-commit=%s\n' "$(git -C "${CHIPYARD_DIR}" rev-parse HEAD)"
printf '[tvm-verilator] gemmini-params=%s\n' \
  "$(sha256sum "${target_header}" | awk '{print $1}')"

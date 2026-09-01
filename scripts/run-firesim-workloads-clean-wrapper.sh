#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for account checkouts whose FireSim runner contains
# account-specific scheduling fixes. Install that runner as
# run-firesim-workloads.account-impl.sh and this file as
# run-firesim-workloads.sh instead of overwriting the account-specific code.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
IMPL="${PYTORCH_CHIPYARD_FIRESIM_RUNNER_IMPL:-${SCRIPT_DIR}/run-firesim-workloads.account-impl.sh}"

[[ -f "${IMPL}" ]] || {
  printf '[firesim-clean-wrapper][error] runner implementation not found: %s\n' "${IMPL}" >&2
  exit 1
}

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
fi
set +u
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/stage2-common.sh"
set -u

preflight_only=0
skip_cleanup=0
resume_mode=0
workloads=()
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--workload" ]]; then
    IFS=, read -r -a parsed <<<"${argument}"
    workloads+=("${parsed[@]}")
    previous=""
    continue
  fi
  case "${argument}" in
    --preflight-only) preflight_only=1 ;;
    -h | --help) skip_cleanup=1 ;;
    --resume) resume_mode=1 ;;
    --workload) previous="--workload" ;;
    --workload=*)
      IFS=, read -r -a parsed <<<"${argument#*=}"
      workloads+=("${parsed[@]}")
      ;;
  esac
done

# Older account runners treated output.bin/run.log as durable completion
# requirements. Supply temporary compatibility markers so --resume honors the
# new two-log policy; the EXIT cleanup removes these markers again.
if [[ "${resume_mode}" -eq 1 && "${preflight_only}" -eq 0 ]]; then
  result_dir="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
  if [[ -d "${result_dir}" ]]; then
    while IFS= read -r -d '' model_log; do
      workload_dir="$(dirname -- "${model_log}")"
      workload="$(basename -- "${workload_dir}")"
      pc_stage2_result_complete "${result_dir}" "${workload}" || continue
      touch "${workload_dir}/.completed"
      printf '%s\n' '[runner] model_ret=0' >"${workload_dir}/run.log"
      case "${workload}" in
        gpt2-* | gpt-neo-* | opt-* | pythia-* | *tok*) ;;
        *) printf '%s\n' 'transient completion marker' >"${workload_dir}/output.bin" ;;
      esac
    done < <(find "${result_dir}" -mindepth 2 -maxdepth 2 -type f -name model.log -print0)
  fi
fi

cleanup_on_exit() {
  local status=$?
  local cleanup_status=0
  local wall_marker workload
  local -a cleanup_args=()

  trap - EXIT
  if [[ "${preflight_only}" -eq 1 || "${skip_cleanup}" -eq 1 ]]; then
    exit "${status}"
  fi

  wall_marker="$(grep -aRhE '^TABLE4_FIRESIM_WALL_S=[0-9]+([.][0-9]+)?$' \
    "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}" 2>/dev/null | tail -n 1 || true)"
  [[ -z "${wall_marker}" ]] || printf '%s\n' "${wall_marker}"

  if [[ "${#workloads[@]}" -eq 0 ]]; then
    cleanup_args+=(--all)
  else
    for workload in "${workloads[@]}"; do
      [[ -n "${workload}" ]] && cleanup_args+=(--workload "${workload}")
    done
  fi
  [[ "${#cleanup_args[@]}" -gt 0 ]] || cleanup_args+=(--all)
  if [[ "${PYTORCH_CHIPYARD_DEFER_FINAL_CLEANUP:-0}" != "1" ]]; then
    cleanup_args+=(--final)
  fi
  bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" "${cleanup_args[@]}" || cleanup_status=$?
  if [[ "${cleanup_status}" -ne 0 ]]; then
    printf '[firesim-clean-wrapper][warn] runtime cleanup failed with status %s\n' \
      "${cleanup_status}" >&2
    [[ "${status}" -ne 0 ]] || status="${cleanup_status}"
  fi
  exit "${status}"
}
trap cleanup_on_exit EXIT

export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=0
bash "${IMPL}" "$@"

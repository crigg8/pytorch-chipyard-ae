#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage2-common.sh"

log() {
  printf '[runtime-cleanup] %s\n' "$*"
}

warn() {
  printf '[runtime-cleanup][warn] %s\n' "$*" >&2
}

die() {
  printf '[runtime-cleanup][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup-runtime-artifacts.sh [options]

Migrate every discoverable model.log/autotune.log pair into the current
pytorch-chipyard checkout, verify the copies, and remove disposable runtime
state. At least one of --all or --workload is required.

Options:
  --workload=NAME  Remove runtime packages/images for NAME. Repeatable.
  --all            Remove runtime derivatives for every workload that has a
                   verified model.log/autotune.log pair.
  --final          Also remove shared transient build products. Stage 1
                   compiler artifacts and repository results are preserved.
  --dry-run        Print removals without changing files. Log migration still
                   reports what would be copied but does not write.
  -h, --help       Show this help.

Durable outputs:
  scripts/figures/results-workload/<workload>/model.log
  scripts/figures/results-workload/<workload>/autotune.log
  scripts/figures/results-workload/<workload>/.completed, when present

Reusable inputs:
  Stage 1 compiler artifacts under examples/artifact-* and results/table4.
  FireMarshal base images such as br-base and br.<build-id> are preserved.
EOF
}

cleanup_all=0
final_cleanup=0
dry_run=0
workloads=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --workload)
      [[ "$#" -ge 2 ]] || die "--workload requires a value"
      workloads+=("$2")
      shift 2
      ;;
    --workload=*)
      workloads+=("${1#*=}")
      shift
      ;;
    --all)
      cleanup_all=1
      shift
      ;;
    --final)
      final_cleanup=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1'"
      ;;
  esac
done

[[ "${cleanup_all}" -eq 1 || "${#workloads[@]}" -gt 0 ]] || \
  die "pass --all or at least one --workload"

for workload in "${workloads[@]}"; do
  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "invalid workload '${workload}'"
done

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
fi
set +u
source "${SCRIPT_DIR}/env.sh"
set -u

RESULT_ROOT="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
CANONICAL_ROOT="${WORKSPACE}/scripts/figures/results-workload"
STATE_ROOT="${PYTORCH_CHIPYARD_STATE_DIR:-${HOME}/.local/share/pytorch-chipyard}"

case "${RESULT_ROOT}" in
  "${WORKSPACE}"/*) ;;
  *) die "durable result root must be inside ${WORKSPACE}: ${RESULT_ROOT}" ;;
esac

if [[ "${dry_run}" -eq 0 ]]; then
  mkdir -p "${RESULT_ROOT}" "${CANONICAL_ROOT}"
fi

workload_selected() {
  local candidate="$1"
  local workload

  [[ "${cleanup_all}" -eq 1 ]] && return 0
  for workload in "${workloads[@]}"; do
    [[ "${candidate}" == "${workload}" ]] && return 0
  done
  return 1
}

copy_single_log() {
  local workload="$1"
  local log_name="$2"
  local source="$3"
  local output_root="$4"
  local selection_required="${5:-1}"
  local dest

  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || return 0
  [[ "${log_name}" == "model.log" || "${log_name}" == "autotune.log" ]] || \
    die "invalid durable log name '${log_name}'"
  [[ -s "${source}" ]] || return 0
  if [[ "${selection_required}" -eq 1 ]]; then
    workload_selected "${workload}" || return 0
  fi

  dest="${output_root}/${workload}/${log_name}"
  if [[ "$(readlink -f -- "${source}")" == "$(readlink -m -- "${dest}")" ]]; then
    return 0
  fi
  if [[ -s "${dest}" && ! "${source}" -nt "${dest}" ]]; then
    return 0
  fi
  if [[ "${dry_run}" -eq 1 ]]; then
    log "would preserve ${source} -> ${dest}"
    return 0
  fi

  mkdir -p "$(dirname -- "${dest}")"
  cp -p -f -- "${source}" "${dest}"
  cmp -s -- "${source}" "${dest}" || die "${log_name} verification failed for ${workload}"
  log "preserved ${workload}/${log_name} in ${output_root}"
}

copy_log_pair() {
  local workload="$1"
  local model_src="$2"
  local autotune_src="$3"
  local completed_src="${4:-}"
  local output_root="${5:-${RESULT_ROOT}}"
  local selection_required="${6:-1}"
  local dest model_dest

  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    warn "skipping invalid inferred workload '${workload}' from ${model_src}"
    return 0
  }
  [[ -s "${model_src}" && -s "${autotune_src}" ]] || return 0
  if [[ "${selection_required}" -eq 1 ]]; then
    workload_selected "${workload}" || return 0
  fi

  dest="${output_root}/${workload}"
  model_dest="${dest}/model.log"

  if [[ "$(readlink -f -- "${model_src}")" == "$(readlink -m -- "${model_dest}")" ]]; then
    return 0
  fi

  if [[ "${dry_run}" -eq 1 ]]; then
    log "would preserve ${model_src} and ${autotune_src} -> ${dest}"
    return 0
  fi

  copy_single_log "${workload}" model.log "${model_src}" "${output_root}" 0
  copy_single_log "${workload}" autotune.log "${autotune_src}" "${output_root}" 0
  touch "${dest}/.completed"
  log "preserved ${workload} model.log/autotune.log in ${dest}"
}

migrate_collected_tree() {
  local root="$1"
  local output_root="${2:-${RESULT_ROOT}}"
  local model_src parent workload

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' model_src; do
    parent="$(dirname -- "${model_src}")"
    workload="$(basename -- "${parent}")"
    copy_log_pair \
      "${workload}" \
      "${model_src}" \
      "${parent}/autotune.log" \
      "${parent}/.completed" \
      "${output_root}" \
      0
  done < <(find "${root}" -mindepth 2 -maxdepth 2 -type f -name model.log -print0 2>/dev/null)
}

migrate_raw_tree() {
  local root="$1"
  local model_src parent job workload completed_src output_root

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' model_src; do
    parent="$(dirname -- "${model_src}")"
    job="$(basename -- "${parent}")"
    workload="${job%0}"
    completed_src="${parent}/.completed"
    if workload_selected "${workload}"; then
      output_root="${RESULT_ROOT}"
    else
      output_root="${CANONICAL_ROOT}"
    fi
    copy_log_pair \
      "${workload}" \
      "${model_src}" \
      "${parent}/autotune.log" \
      "${completed_src}" \
      "${output_root}" \
      0
  done < <(find "${root}" -type f -name model.log -print0 2>/dev/null)
}

migrate_flat_log_tree() {
  local root="$1"
  local output_root="${2:-${RESULT_ROOT}}"
  local model_src base workload autotune_src

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' model_src; do
    base="$(basename -- "${model_src}")"
    workload="${base%-model.log}"
    autotune_src="${root}/${workload}-autotune.log"
    copy_log_pair "${workload}" "${model_src}" "${autotune_src}" "" "${output_root}" 0
  done < <(find "${root}" -maxdepth 1 -type f -name '*-model.log' -print0 2>/dev/null)
}

migrate_individual_collected_tree() {
  local root="$1"
  local output_root="$2"
  local source parent workload log_name

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' source; do
    parent="$(dirname -- "${source}")"
    workload="$(basename -- "${parent}")"
    log_name="$(basename -- "${source}")"
    copy_single_log "${workload}" "${log_name}" "${source}" "${output_root}" 0
  done < <(find "${root}" -mindepth 2 -maxdepth 2 -type f \
    \( -name model.log -o -name autotune.log \) -print0 2>/dev/null)
}

migrate_individual_raw_tree() {
  local root="$1"
  local source parent job workload output_root log_name

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' source; do
    parent="$(dirname -- "${source}")"
    job="$(basename -- "${parent}")"
    workload="${job%0}"
    log_name="$(basename -- "${source}")"
    if workload_selected "${workload}"; then
      output_root="${RESULT_ROOT}"
    else
      output_root="${CANONICAL_ROOT}"
    fi
    copy_single_log "${workload}" "${log_name}" "${source}" "${output_root}" 0
  done < <(find "${root}" -type f \
    \( -name model.log -o -name autotune.log \) -print0 2>/dev/null)
}

migrate_individual_flat_tree() {
  local root="$1"
  local output_root="$2"
  local source base workload log_name

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' source; do
    base="$(basename -- "${source}")"
    case "${base}" in
      *-model.log) workload="${base%-model.log}"; log_name=model.log ;;
      *-autotune.log) workload="${base%-autotune.log}"; log_name=autotune.log ;;
      *) continue ;;
    esac
    copy_single_log "${workload}" "${log_name}" "${source}" "${output_root}" 0
  done < <(find "${root}" -maxdepth 1 -type f \
    \( -name '*-model.log' -o -name '*-autotune.log' \) -print0 2>/dev/null)
}

migrate_named_logs_from_tree() {
  local root="$1"
  local prefix="$2"
  local source relative parent workload log_name

  [[ -d "${root}" ]] || return 0
  while IFS= read -r -d '' source; do
    relative="${source#"${root}"/}"
    parent="$(dirname -- "${relative}")"
    [[ "${parent}" != "." ]] || parent="root"
    workload="${prefix}-${parent//\//-}"
    workload="${workload//[^A-Za-z0-9._-]/-}"
    log_name="$(basename -- "${source}")"
    copy_single_log "${workload}" "${log_name}" "${source}" "${CANONICAL_ROOT}" 0
  done < <(find "${root}" -type f \
    \( -name model.log -o -name autotune.log \) -print0 2>/dev/null)
}

remove_path() {
  local target="$1"
  local resolved

  [[ -e "${target}" || -L "${target}" ]] || return 0
  resolved="$(readlink -m -- "${target}")"
  case "${resolved}" in
    / | "${HOME}" | "${WORKSPACE}" | "${RESULT_ROOT}" | "${RESULT_ROOT}"/* | \
    "${CANONICAL_ROOT}" | "${CANONICAL_ROOT}"/*)
      die "refusing unsafe cleanup target '${target}'"
      ;;
    "${HOME}"/*) ;;
    *)
      warn "skipping cleanup outside ${HOME}: ${target}"
      return 0
      ;;
  esac

  if [[ "${dry_run}" -eq 1 ]]; then
    log "would remove ${target}"
  else
    rm -rf -- "${target}"
    log "removed ${target}"
  fi
}

clean_contents() {
  local target="$1"
  local resolved entry

  # Several FireMarshal paths (notably res-dir and wlutil/generated) must
  # exist before FireMarshal/doit initializes.  "Clean contents" therefore
  # also means "leave an empty directory behind", including when an older
  # cleanup already removed the directory itself.
  resolved="$(readlink -m -- "${target}")"
  case "${resolved}" in
    / | "${HOME}" | "${WORKSPACE}" | "${RESULT_ROOT}" | "${RESULT_ROOT}"/*)
      die "refusing unsafe cleanup directory '${target}'"
      ;;
    "${HOME}"/*) ;;
    *)
      warn "skipping cleanup outside ${HOME}: ${target}"
      return 0
      ;;
  esac

  if [[ "${dry_run}" -eq 1 ]]; then
    [[ -d "${target}" ]] || log "would create required directory ${target}"
  else
    mkdir -p -- "${target}"
  fi

  while IFS= read -r -d '' entry; do
    remove_path "${entry}"
  done < <(find -H "${target}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)
}

durable_result_complete() {
  local root="$1"
  local workload="$2"
  pc_stage2_result_complete "${root}" "${workload}"
}

completed_cleanup_workloads=()

append_completed_cleanup_workload() {
  local candidate="$1"
  local existing

  [[ "${candidate}" =~ ^[A-Za-z0-9._-]+$ ]] || return 0
  for existing in "${completed_cleanup_workloads[@]}"; do
    [[ "${existing}" == "${candidate}" ]] && return 0
  done
  completed_cleanup_workloads+=("${candidate}")
}

select_completed_cleanup_workloads() {
  local root model_log workload

  if [[ "${cleanup_all}" -eq 1 ]]; then
    for root in "${RESULT_ROOT}" "${CANONICAL_ROOT}"; do
      [[ -d "${root}" ]] || continue
      while IFS= read -r -d '' model_log; do
        workload="$(basename -- "$(dirname -- "${model_log}")")"
        durable_result_complete "${root}" "${workload}" || continue
        append_completed_cleanup_workload "${workload}"
      done < <(find "${root}" -mindepth 2 -maxdepth 2 -type f \
        -name model.log -print0 2>/dev/null)
    done
  fi

  # Explicit --workload targets are always runtime-cleanup targets. This is
  # needed on a failed run as well: preserve any logs first, then discard the
  # partial image/package so the next Stage 2 invocation recreates it cleanly.
  for workload in "${workloads[@]}"; do
    append_completed_cleanup_workload "${workload}"
  done
}

mark_and_prune_completed_result() {
  local root="$1"
  local workload="$2"
  local result_dir="${root}/${workload}"
  local entry

  durable_result_complete "${root}" "${workload}" || return 0
  if [[ "${dry_run}" -eq 1 ]]; then
    log "would mark completed result ${result_dir}"
  else
    touch "${result_dir}/.completed"
  fi

  while IFS= read -r -d '' entry; do
    case "$(basename -- "${entry}")" in
      model.log | autotune.log | .completed) ;;
      *)
        if [[ "${dry_run}" -eq 1 ]]; then
          log "would remove non-durable result artifact ${entry}"
        else
          rm -rf -- "${entry}"
          log "removed non-durable result artifact ${entry}"
        fi
        ;;
    esac
  done < <(find "${result_dir}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)
}

clean_completed_workload_runtime() {
  local workload="$1"

  remove_path "${FIREMARSHAL_IMAGE_DIR}/${workload}"
  remove_path "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload}"
  remove_path "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/${workload}.json"
  remove_path "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/.pytorch-chipyard-${workload}-linux-config"
  remove_path "${FIRESIM_WORKLOAD_DIR}/${workload}.json"
  remove_path "${FIRESIM_WORKLOAD_DIR}/${workload}"
  remove_path "${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR}/config_runtime_${workload}.yaml"
  remove_path "${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR}/.${workload}.run-start"

  remove_path "${STATE_ROOT}/figure-results/${workload}"
  remove_path "${STATE_ROOT}/firesim/deploy/results-workload/${workload}"
  remove_path "${STATE_ROOT}/firesim/deploy/results-workload/${workload}0"
  remove_path "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}/${workload}"
  remove_path "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}/${workload}0"
  remove_path "${WORKSPACE}/examples/.logs/${workload}-model.log"
  remove_path "${WORKSPACE}/examples/.logs/${workload}-autotune.log"
  remove_path "${STATE_ROOT}/logs/${workload}-model.log"
  remove_path "${STATE_ROOT}/logs/${workload}-autotune.log"
}

# Migrate legacy per-account results and raw FireSim outputs before removing
# any transient state. New runs already write directly to RESULT_ROOT.
migrate_collected_tree "${STATE_ROOT}/figure-results" "${CANONICAL_ROOT}"
migrate_raw_tree "${STATE_ROOT}/firesim/deploy/results-workload"
migrate_raw_tree "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}"
migrate_flat_log_tree "${WORKSPACE}/examples/.logs" "${CANONICAL_ROOT}"
migrate_individual_collected_tree "${STATE_ROOT}/figure-results" "${CANONICAL_ROOT}"
migrate_individual_raw_tree "${STATE_ROOT}/firesim/deploy/results-workload"
migrate_individual_raw_tree "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}"
migrate_individual_flat_tree "${WORKSPACE}/examples/.logs" "${CANONICAL_ROOT}"
migrate_named_logs_from_tree "${STATE_ROOT}/logs" state-log
migrate_named_logs_from_tree "${STATE_ROOT}/firesim/deploy/logs" firesim-deploy-log
migrate_named_logs_from_tree "${FIRESIM_RUNS_DIR}" firesim-run
migrate_named_logs_from_tree "${MARSHAL_LOG_DIR}" firemarshal-log
migrate_named_logs_from_tree "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" packaged-workload
select_completed_cleanup_workloads
if [[ "${#completed_cleanup_workloads[@]}" -eq 0 ]]; then
  log "no completed workloads eligible for runtime cleanup"
fi
for workload in "${completed_cleanup_workloads[@]}"; do
  mark_and_prune_completed_result "${RESULT_ROOT}" "${workload}"
  if [[ "${CANONICAL_ROOT}" != "${RESULT_ROOT}" ]]; then
    mark_and_prune_completed_result "${CANONICAL_ROOT}" "${workload}"
  fi
  clean_completed_workload_runtime "${workload}"
done

if [[ "${#completed_cleanup_workloads[@]}" -gt 0 ]]; then
  clean_contents "${FIRESIM_RUNS_DIR}"
fi

if [[ "${final_cleanup}" -eq 1 && "${#completed_cleanup_workloads[@]}" -gt 0 ]]; then
  clean_contents "${PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR}"
  clean_contents "${MARSHAL_MOUNT_DIR}"
  clean_contents "${LIBGUESTFS_CACHEDIR}"
  clean_contents "${LIBGUESTFS_TMPDIR}"

  remove_path "${FIREMARSHAL_DIR}/boards/default/distros/br/buildroot/output"
  remove_path "${FIREMARSHAL_DIR}/boards/default/distros/br/buildroot/dl"
  # wlutil requires res-dir to exist during initialize(). Empty it without
  # deleting the directory itself. Also clear the old source-local location
  # when upgrading an existing account environment.
  clean_contents "${MARSHAL_RES_DIR}"
  clean_contents "${FIREMARSHAL_DIR}/runOutput"
  # FireMarshal/doit stores marshaldb.dat under wlutil/generated and assumes
  # the parent already exists.  Purge the database/cache, not its directory.
  clean_contents "${FIREMARSHAL_DIR}/wlutil/generated"
  remove_path "${FIRESIM_DEPLOY_DIR}/results-build"
  remove_path "${FIRESIM_DEPLOY_DIR}/firesim-staging"
  remove_path "${FIRESIM_DEPLOY_DIR}/.Xil"
  remove_path "${WORKSPACE}/.buddy-rvv-store-fence"

  linux_dir="${FIREMARSHAL_DIR}/boards/default/linux"
  opensbi_dir="${FIREMARSHAL_DIR}/boards/default/firmware/opensbi"
  if [[ -f "${linux_dir}/Makefile" ]]; then
    if [[ "${dry_run}" -eq 1 ]]; then
      log "would run make -C ${linux_dir} ARCH=riscv mrproper"
    else
      make -C "${linux_dir}" ARCH=riscv mrproper >/dev/null
      log "cleaned shared Linux build products"
    fi
  fi
  if [[ -f "${opensbi_dir}/Makefile" ]]; then
    if [[ "${dry_run}" -eq 1 ]]; then
      log "would run make -C ${opensbi_dir} clean"
    else
      make -C "${opensbi_dir}" clean >/dev/null
      log "cleaned shared OpenSBI build products"
    fi
  fi
fi

log "durable logs: ${RESULT_ROOT}"

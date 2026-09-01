#!/usr/bin/env bash

# Shared Stage 2 completion contract.  Runtime markers are useful metadata, but
# the two durable logs are the authoritative indication that a workload no
# longer needs to run.  This also lets an older run resume after a checkout was
# replaced and its .completed marker was lost.
pc_stage2_validate_workload_name() {
  local workload="$1"
  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]]
}

pc_stage2_result_dir() {
  local root="$1"
  local workload="$2"
  printf '%s/%s\n' "${root%/}" "${workload}"
}

pc_stage2_result_complete() {
  local root="$1"
  local workload="$2"
  local result_dir

  pc_stage2_validate_workload_name "${workload}" || return 1
  result_dir="$(pc_stage2_result_dir "${root}" "${workload}")"
  [[ -s "${result_dir}/model.log" && -s "${result_dir}/autotune.log" ]]
}

pc_stage2_mark_result_complete() {
  local root="$1"
  local workload="$2"
  local result_dir

  pc_stage2_result_complete "${root}" "${workload}" || return 1
  result_dir="$(pc_stage2_result_dir "${root}" "${workload}")"
  touch "${result_dir}/.completed"
}

pc_stage2_prune_result_dir() {
  local root="$1"
  local workload="$2"
  local result_dir entry

  pc_stage2_result_complete "${root}" "${workload}" || return 1
  result_dir="$(pc_stage2_result_dir "${root}" "${workload}")"
  pc_stage2_mark_result_complete "${root}" "${workload}"
  while IFS= read -r -d '' entry; do
    case "$(basename -- "${entry}")" in
      model.log | autotune.log | .completed) ;;
      *) rm -rf -- "${entry}" ;;
    esac
  done < <(find "${result_dir}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null)
}

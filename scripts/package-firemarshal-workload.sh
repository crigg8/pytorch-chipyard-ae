#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

log() {
  printf '[artifact-package] %s\n' "$*"
}

warn() {
  printf '[artifact-package][warn] %s\n' "$*" >&2
}

die() {
  printf '[artifact-package][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/package-firemarshal-workload.sh [options]

Default:
  Scan artifact-* directories under examples/ for generated Stage 1 artifacts
  and package every model-<N>core.elf into FireMarshal and FireSim workload
  files. LLM runners use --no-output; other runners retain output.bin.

Generated files:
  $PYTORCH_CHIPYARD_WORKLOAD_DIR/<workload>.json
  $PYTORCH_CHIPYARD_WORKLOAD_DIR/overlay-<workload>/
  $FIRESIM_WORKLOAD_DIR/<workload>.json

Options:
  --artifact-dir=PATH  Package ELF files from this artifact directory only.
                       May be repeated.
  --core=LIST          Package only the selected core counts, for example 1 or 2,4.
                       Default: package every ELF in each selected artifact.
  --no-clean            Keep previously generated workload JSONs/overlays.
  --no-chipyard-check   Allow file generation without initialized FireMarshal/FireSim.
  -h, --help            Show this help.

Environment:
  PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE  FireMarshal rootfs-size. Default: 8GiB.
  PYTORCH_CHIPYARD_WORKLOAD_DIR             FireMarshal workload output directory.
  FIRESIM_WORKLOAD_DIR                      FireSim deploy workload output directory.
  PYTORCH_CHIPYARD_OMP_FIRST_CPU_<WORKLOAD>
                                            Override the first OpenMP CPU for one
                                            workload. Hyphens become underscores.
  PYTORCH_CHIPYARD_OMP_PLACES_<WORKLOAD>
                                            Override the complete OpenMP place list,
                                            for example {2},{3},{4},{5}. Must be used
                                            with GOMP_CPU_AFFINITY below.
  PYTORCH_CHIPYARD_GOMP_CPU_AFFINITY_<WORKLOAD>
                                            Override the matching CPU list, for
                                            example "2 3 4 5".
  PYTORCH_CHIPYARD_OMP_WAIT_POLICY_<WORKLOAD>
                                            Override ACTIVE/PASSIVE for one workload.
  PYTORCH_CHIPYARD_GOMP_SPINCOUNT_<WORKLOAD>
                                            Override a count/INFINITE for one workload.
EOF
}

abs_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "directory not found: ${path}"
  (cd -- "${path}" >/dev/null 2>&1 && pwd -P)
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

validate_name() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "invalid ${label} '${value}'; use only letters, numbers, '.', '_', and '-'"
}

validate_size() {
  local value="$1"
  [[ "${value}" =~ ^[A-Za-z0-9._+-]+$ ]] || die "invalid rootfs size: ${value}"
}

env_suffix_for_workload() {
  local workload="$1"
  local suffix

  suffix="${workload^^}"
  suffix="${suffix//[^A-Z0-9_]/_}"
  printf '%s\n' "${suffix}"
}

shell_quote() {
  printf '%q' "$1"
}

artifact_storage_rel() {
  local logical_rel="$1"
  local first rest

  logical_rel="${logical_rel#/}"
  first="${logical_rel%%/*}"
  if [[ "${logical_rel}" == "${first}" ]]; then
    printf 'artifact-%s\n' "${first}"
  else
    rest="${logical_rel#*/}"
    printf 'artifact-%s/%s\n' "${first}" "${rest}"
  fi
}

normalize_artifact_rel() {
  local rel="$1"
  local first rest

  rel="${rel#./}"
  first="${rel%%/*}"
  if [[ "${first}" == artifact-* ]]; then
    first="${first#artifact-}"
  fi

  if [[ "${rel}" == */* ]]; then
    rest="${rel#*/}"
    printf '%s/%s\n' "${first}" "${rest}"
  else
    printf '%s\n' "${first}"
  fi
}

existing_artifact_dir_for_rel() {
  local logical_rel="$1"
  local storage_dir legacy_dir

  storage_dir="${ARTIFACT_ROOT}/$(artifact_storage_rel "${logical_rel}")"
  if [[ -d "${storage_dir}" ]]; then
    printf '%s\n' "${storage_dir}"
    return 0
  fi

  legacy_dir="${ARTIFACT_ROOT}/${logical_rel}"
  if [[ -d "${legacy_dir}" ]]; then
    printf '%s\n' "${legacy_dir}"
    return 0
  fi

  return 1
}

infer_core_from_elf() {
  local elf_name="$1"
  if [[ "${elf_name}" =~ ([0-9]+)core\.elf$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  fi
}

select_input_bin() {
  local artifact_dir="$1"
  local matches=()
  local candidate

  if [[ -f "${artifact_dir}/input.bin" ]]; then
    printf '%s\n' "${artifact_dir}/input.bin"
    return 0
  fi

  for candidate in "${artifact_dir}"/*input*.bin; do
    [[ -f "${candidate}" ]] || continue
    matches+=("${candidate}")
  done

  case "${#matches[@]}" in
    0)
      return 1
      ;;
    1)
      printf '%s\n' "${matches[0]}"
      return 0
      ;;
    *)
      printf '[artifact-package][error] multiple input bin candidates under %s:\n' "${artifact_dir}" >&2
      printf '  %s\n' "${matches[@]}" >&2
      return 2
      ;;
  esac
}

workload_kind_for() {
  local artifact_dir="$1"
  local rel first_component

  rel="$(artifact_rel_path "${artifact_dir}")"
  first_component="${rel%%/*}"

  case "${first_component}" in
    gpt2 | gpt-neo | opt | pythia)
      printf '%s\n' llm
      ;;
    *)
      printf '%s\n' cnn
      ;;
  esac
}

guest_root_for() {
  local kind="$1"
  case "${kind}" in
    llm) printf '%s\n' /root/llm ;;
    *) printf '%s\n' /root/cnn ;;
  esac
}

artifact_rel_path() {
  local artifact_dir="$1"
  local rel hint_path

  hint_path="${artifact_dir}/.pytorch-chipyard-workload-rel"
  if [[ -f "${hint_path}" ]]; then
    rel="$(head -n 1 "${hint_path}")"
    [[ "${rel}" =~ ^[A-Za-z0-9._/-]+$ ]] || die "invalid workload hint in ${hint_path}: ${rel}"
    normalize_artifact_rel "${rel}"
    return
  fi

  case "${artifact_dir}" in
    "${ARTIFACT_ROOT}/"*)
      rel="${artifact_dir#"${ARTIFACT_ROOT}/"}"
      ;;
    "${WORKSPACE}/examples/"*)
      rel="${artifact_dir#"${WORKSPACE}/examples/"}"
      ;;
    *)
      rel="$(basename "${artifact_dir}")"
      ;;
  esac

  normalize_artifact_rel "${rel}"
}

derive_workload_name() {
  local artifact_dir="$1"
  local core="$2"
  local rel

  rel="$(artifact_rel_path "${artifact_dir}")"

  rel="${rel//\//-}"
  rel="${rel//_/-}"
  rel="$(printf '%s\n' "${rel}" | sed -E 's/(^|-)seq([0-9]+)(-|$)/\1\2tok\3/g')"
  rel="${rel//--/-}"

  if [[ "${rel}" =~ [0-9]+core$ ]]; then
    printf '%s\n' "${rel}"
  else
    printf '%s-%score\n' "${rel}" "${core}"
  fi
}

derive_packaged_workload_name() {
  local artifact_dir="$1"
  local core="$2"
  local rel model attention seq_len

  rel="$(artifact_rel_path "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    attention="${BASH_REMATCH[2]}"
    seq_len="${BASH_REMATCH[3]}"
    if [[ "${core}" == "1" ]]; then
      printf '%s-boom-gemmini-%s-%stok-%score\n' "${model}" "${attention}" "${seq_len}" "${core}"
    else
      printf '%s-rocket-gemmini-%s-%stok-%score\n' "${model}" "${attention}" "${seq_len}" "${core}"
    fi
    return
  fi

  derive_workload_name "${artifact_dir}" "${core}"
}

is_shadowed_elf() {
  local artifact_dir="$1"
  local core="$2"
  local rel model host attention seq_len canonical_dir

  rel="$(artifact_rel_path "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini$ ]]; then
    model="${BASH_REMATCH[1]}"
    if canonical_dir="$(existing_artifact_dir_for_rel "${model}/gemmini/sdpa/seq256")"; then
      warn "skipping shadowed SDPA artifact ${artifact_dir}/model-${core}core.elf; using ${canonical_dir}"
      return 0
    fi
    return
  fi

  if [[ "${rel}" =~ ^(opt|pythia)/(rocket|boom)-gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    host="${BASH_REMATCH[2]}"
    attention="${BASH_REMATCH[3]}"
    seq_len="${BASH_REMATCH[4]}"
    if canonical_dir="$(existing_artifact_dir_for_rel "${model}/gemmini/${attention}/seq${seq_len}")"; then
      warn "skipping shadowed ${host}-gemmini artifact ${artifact_dir}/model-${core}core.elf; using ${canonical_dir}"
      return 0
    fi
  fi

  return 1
}

elf_is_in_build_plan() {
  local artifact_dir="$1"
  local core="$2"
  local core_file="${artifact_dir}/.pytorch-chipyard-build-cores"

  [[ -f "${core_file}" ]] || return 0
  grep -Fxq -- "${core}" "${core_file}"
}

is_boom_flex_kernel_workload() {
  local workload="$1"

  case "${workload}" in
    opt-boom-gemmini-flash-*tok-1core | opt-boom-gemmini-window-*tok-1core)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

require_flex_kernel_only_elf() {
  local elf_path="$1"

  if ! LC_ALL=C grep -aFq -- '--kernel-only=flex_attention' "${elf_path}"; then
    die "${elf_path} does not support BOOM FlexAttention kernel-only execution; rerun Stage 1 flex-attention generation, then rebuild the ELF"
  fi
}

elf_has_openmp_symbols() {
  local elf_path="$1"

  LC_ALL=C readelf -Ws "${elf_path}" 2>/dev/null | awk '
    NF >= 8 {
      name = $8
      if (name ~ /^(GOMP_|omp_)/ || name ~ /_omp_fn/) found = 1
    }
    END { exit(found ? 0 : 1) }
  '
}

require_no_openmp_elf() {
  local elf_path="$1"

  command -v readelf >/dev/null 2>&1 || \
    die "readelf is required to verify that BOOM FlexAttention does not link OpenMP"
  if elf_has_openmp_symbols "${elf_path}"; then
    die "${elf_path} still contains OpenMP code; rebuild it with PYTORCH_CHIPYARD_SPIKE_EXECUTABLE=1"
  fi
  if LC_ALL=C readelf -d "${elf_path}" 2>/dev/null | grep -Fqi 'libgomp'; then
    die "${elf_path} still links libgomp; rebuild it with PYTORCH_CHIPYARD_SPIKE_EXECUTABLE=1"
  fi
}

clean_generated_workloads() {
  [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" ]] || return 0

  find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 \
    -type f \
    -name '.pytorch-chipyard-*-linux-config' \
    -delete
  find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    ! -name '*base.json' \
    -delete
  find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 \
    -type d \
    -name 'overlay-*' \
    -exec rm -rf {} +
}

discover_artifact_elf_paths() {
  local root="$1"
  local artifact_dirs=()
  local dir

  if [[ "$(basename "${root}")" == artifact-* ]]; then
    find "${root}" -type f -name 'model-*core.elf' -print0 | sort -z
    return
  fi

  while IFS= read -r -d '' dir; do
    artifact_dirs+=("${dir}")
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'artifact-*' -print0 | sort -z)

  if [[ "${#artifact_dirs[@]}" -gt 0 ]]; then
    find "${artifact_dirs[@]}" -type f -name 'model-*core.elf' -print0 | sort -z
  else
    find "${root}" -type f -name 'model-*core.elf' -print0 | sort -z
  fi
}

omp_places_for() {
  local core="$1"
  local first_cpu="${2:-0}"
  local places=""
  local i

  for ((i = 0; i < core; i++)); do
    places+="${places:+,}{$((first_cpu + i))}"
  done
  printf '%s\n' "${places}"
}

omp_cpu_affinity_for() {
  local core="$1"
  local first_cpu="${2:-0}"
  local affinity=""
  local i

  for ((i = 0; i < core; i++)); do
    affinity+="${affinity:+ }$((first_cpu + i))"
  done
  printf '%s\n' "${affinity}"
}

write_workload() {
  local artifact_dir="$1"
  local elf_path="$2"
  local rootfs_size="$3"
  local workload_name="$4"

  local elf_base core kind guest_root guest_root_rel
  local input_path input_base weights_path elf_sha256 input_sha256 weights_sha256
  local workload_dir deploy_workload_dir overlay_dir guest_dir runner_path hook_path
  local workload_json deploy_json workload_config_path linux_config_path linux_json_entry
  local firesim_workload_input_dir common_bootbinary common_rootfs
  local places cpu_affinity model_command
  local omp_first_cpu omp_places_override omp_cpu_affinity_override
  local omp_wait_policy omp_display_affinity gomp_spincount
  local workload_env_suffix omp_first_cpu_var omp_places_var gomp_cpu_affinity_var
  local omp_wait_policy_var gomp_spincount_var omp_places_env gomp_cpu_affinity_env
  local expect_output_bin runner_check_files output_entries kernel_only=0
  local runner_runtime_env runner_runtime_log runner_runtime_tuning=""
  local rvv_hz100_profile=0 rvv_worker_cpus="" rvv_housekeeping_cpus=""
  local rvv_housekeeping_mask="" rvv_expected_affinity=""

  elf_base="$(basename "${elf_path}")"
  core="$(infer_core_from_elf "${elf_base}")"
  [[ -n "${core}" ]] || die "could not infer core count from ${elf_path}"
  [[ "${core}" =~ ^[0-9]+$ && "${core}" != "0" ]] || die "invalid core count in ${elf_path}"

  input_path="$(select_input_bin "${artifact_dir}")" || \
    die "could not find input.bin or a unique *input*.bin under ${artifact_dir}"
  input_base="$(basename "${input_path}")"

  weights_path="${artifact_dir}/weights.bin"
  require_file "${weights_path}"

  elf_sha256="$(sha256sum "${elf_path}" | awk '{print $1}')"
  input_sha256="$(sha256sum "${input_path}" | awk '{print $1}')"
  weights_sha256="$(sha256sum "${weights_path}" | awk '{print $1}')"

  kind="$(workload_kind_for "${artifact_dir}")"
  guest_root="$(guest_root_for "${kind}")"
  guest_root_rel="${guest_root#/}"
  validate_name "workload name" "${workload_name}"

  if [[ -n "${PACKAGED_WORKLOADS[${workload_name}]:-}" ]]; then
    die "duplicate workload name '${workload_name}' from ${artifact_dir} and ${PACKAGED_WORKLOADS[${workload_name}]}"
  fi
  PACKAGED_WORKLOADS["${workload_name}"]="${artifact_dir}"

  workload_dir="${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
  deploy_workload_dir="${FIRESIM_WORKLOAD_DIR}"
  overlay_dir="${workload_dir}/overlay-${workload_name}"
  guest_dir="${overlay_dir}/${guest_root_rel}/${workload_name}"
  runner_path="${overlay_dir}/${guest_root_rel}/run_${workload_name}.sh"
  hook_path="${overlay_dir}/firemarshal.sh"
  workload_json="${workload_dir}/${workload_name}.json"
  deploy_json="${deploy_workload_dir}/${workload_name}.json"
  workload_config_path="${guest_dir}/workload-config.txt"
  linux_config_path="${workload_dir}/.pytorch-chipyard-${workload_name}-linux-config"
  linux_json_entry=""

  mkdir -p "${guest_dir}" "${deploy_workload_dir}" "$(dirname "${runner_path}")"
  firesim_workload_input_dir="${deploy_workload_dir}/${workload_name}"
  common_bootbinary="$(realpath -m --relative-to="${firesim_workload_input_dir}" \
    "${FIREMARSHAL_IMAGE_DIR}/${workload_name}/${workload_name}-bin")"
  common_rootfs="$(realpath -m --relative-to="${firesim_workload_input_dir}" \
    "${FIREMARSHAL_IMAGE_DIR}/${workload_name}/${workload_name}.img")"

  if is_boom_flex_kernel_workload "${workload_name}"; then
    kernel_only=1
    require_flex_kernel_only_elf "${elf_path}"
    require_no_openmp_elf "${elf_path}"

    cat >"${linux_config_path}" <<'EOF'
CONFIG_CMDLINE="maxcpus=1 console=ttySIF0,3686400 earlycon"
EOF
    printf -v linux_json_entry \
      '  "linux": {"config": "%s"},\n' "$(basename "${linux_config_path}")"
  fi

  cp -f "${elf_path}" "${guest_dir}/${elf_base}"
  cp -f "${input_path}" "${guest_dir}/${input_base}"
  if [[ "${kernel_only}" -eq 1 ]]; then
    : >"${guest_dir}/weights.bin"
  else
    cp -f "${weights_path}" "${guest_dir}/weights.bin"
  fi
  chmod +x "${guest_dir}/${elf_base}" 2>/dev/null || true

  if [[ "${kernel_only}" -ne 1 ]]; then
  omp_first_cpu=0
  omp_places_override=""
  omp_cpu_affinity_override=""
  omp_wait_policy=PASSIVE
  omp_display_affinity=FALSE
  gomp_spincount=0
  if [[ "${workload_name}" == *-rvv-* ]]; then
    omp_wait_policy=ACTIVE
    gomp_spincount=INFINITE

    case "${workload_name}" in
      resnet50-rvv-2core)
        omp_first_cpu=0
        ;;
      mobilenetv2-rvv-2core)
        omp_first_cpu=2
        ;;
      squeezenet-rvv-4core)
        omp_first_cpu=0
        omp_display_affinity=TRUE
        omp_wait_policy=PASSIVE
        gomp_spincount=0
        ;;
      mobilenetv2-rvv-4core)
        omp_places_override="{0},{1},{6},{7}"
        omp_cpu_affinity_override="0 1 6 7"
        omp_display_affinity=TRUE
        ;;
      *)
        if [[ "${core}" -eq 4 ]]; then
          omp_first_cpu=4
          omp_display_affinity=TRUE
        fi
        ;;
    esac
  fi

  workload_env_suffix="$(env_suffix_for_workload "${workload_name}")"
  omp_first_cpu_var="PYTORCH_CHIPYARD_OMP_FIRST_CPU_${workload_env_suffix}"
  omp_places_var="PYTORCH_CHIPYARD_OMP_PLACES_${workload_env_suffix}"
  gomp_cpu_affinity_var="PYTORCH_CHIPYARD_GOMP_CPU_AFFINITY_${workload_env_suffix}"
  omp_wait_policy_var="PYTORCH_CHIPYARD_OMP_WAIT_POLICY_${workload_env_suffix}"
  gomp_spincount_var="PYTORCH_CHIPYARD_GOMP_SPINCOUNT_${workload_env_suffix}"

  if [[ -n "${!omp_first_cpu_var:-}" ]]; then
    omp_first_cpu="${!omp_first_cpu_var}"
  fi
  if [[ -n "${!omp_wait_policy_var:-}" ]]; then
    omp_wait_policy="${!omp_wait_policy_var}"
  fi
  if [[ -n "${!gomp_spincount_var:-}" ]]; then
    gomp_spincount="${!gomp_spincount_var}"
  fi

  omp_places_env="${!omp_places_var:-}"
  gomp_cpu_affinity_env="${!gomp_cpu_affinity_var:-}"
  if [[ -n "${omp_places_env}" || -n "${gomp_cpu_affinity_env}" ]]; then
    [[ -n "${omp_places_env}" && -n "${gomp_cpu_affinity_env}" ]] || \
      die "${omp_places_var} and ${gomp_cpu_affinity_var} must be set together"
    [[ "${omp_places_env}" =~ ^\{[0-9]+\}(,\{[0-9]+\})*$ ]] || \
      die "invalid ${omp_places_var}='${omp_places_env}'; expected {N},{N},..."
    [[ "${gomp_cpu_affinity_env}" =~ ^[0-9]+([[:space:]]+[0-9]+)*$ ]] || \
      die "invalid ${gomp_cpu_affinity_var}='${gomp_cpu_affinity_env}'; expected space-separated CPU numbers"
    omp_places_override="${omp_places_env}"
    omp_cpu_affinity_override="${gomp_cpu_affinity_env}"
  fi

  [[ "${omp_first_cpu}" =~ ^[0-9]+$ ]] || \
    die "invalid ${omp_first_cpu_var}='${omp_first_cpu}'; expected a non-negative integer"
  [[ "${omp_wait_policy}" == "ACTIVE" || "${omp_wait_policy}" == "PASSIVE" ]] || \
    die "invalid ${omp_wait_policy_var}='${omp_wait_policy}'; expected ACTIVE or PASSIVE"
  [[ "${gomp_spincount}" == "INFINITE" || "${gomp_spincount}" =~ ^[0-9]+$ ]] || \
    die "invalid ${gomp_spincount_var}='${gomp_spincount}'; expected a non-negative integer or INFINITE"

  if [[ -n "${omp_places_override}" ]]; then
    places="${omp_places_override}"
    cpu_affinity="${omp_cpu_affinity_override}"
  else
    places="$(omp_places_for "${core}" "${omp_first_cpu}")"
    cpu_affinity="$(omp_cpu_affinity_for "${core}" "${omp_first_cpu}")"
  fi
  fi
  model_command="$(shell_quote "./${elf_base}") $(shell_quote "${input_base}") weights.bin output.bin"

  case "${workload_name}" in
    alexnet-rvv-4core | resnet50-rvv-4core)
      rvv_hz100_profile=1
      rvv_worker_cpus="4-7"
      rvv_housekeeping_cpus="0-3"
      rvv_housekeeping_mask="0f"
      rvv_expected_affinity="4 5 6 7"
      ;;
    mobilenetv2-rvv-4core)
      rvv_hz100_profile=1
      rvv_worker_cpus="0-1,6-7"
      rvv_housekeeping_cpus="2-5"
      rvv_housekeeping_mask="3c"
      rvv_expected_affinity="0 1 6 7"
      ;;
    resnet50-rvv-2core)
      rvv_hz100_profile=1
      rvv_worker_cpus="0-1"
      rvv_housekeeping_cpus="2-3"
      rvv_housekeeping_mask="0c"
      rvv_expected_affinity="0 1"
      ;;
  esac

  if [[ "${rvv_hz100_profile}" -eq 1 ]]; then
    [[ "${cpu_affinity}" == "${rvv_expected_affinity}" ]] || \
      die "${workload_name} requires GOMP_CPU_AFFINITY='${rvv_expected_affinity}' for its validated HZ=100 profile; got '${cpu_affinity}'"

    cat >"${linux_config_path}" <<EOF
CONFIG_HZ_100=y
# CONFIG_HZ_250 is not set
CONFIG_NO_HZ_FULL=y
# CONFIG_NO_HZ_IDLE is not set
CONFIG_RCU_NOCB_CPU=y
# CONFIG_SOFTLOCKUP_DETECTOR is not set
CONFIG_CMDLINE_FORCE=y
CONFIG_CMDLINE="nohz_full=${rvv_worker_cpus} rcu_nocbs=${rvv_worker_cpus} rcu_nocb_poll irqaffinity=${rvv_housekeeping_cpus} isolcpus=domain,managed_irq,${rvv_worker_cpus} workqueue.unbound_cpus=${rvv_housekeeping_cpus} watchdog_thresh=0 skew_tick=1 console=ttyS0 console=ttySIF0,3686400 earlycon"
EOF
    printf -v linux_json_entry \
      '  "linux": {"config": "%s"},\n' "$(basename "${linux_config_path}")"

    model_command="taskset -c ${rvv_worker_cpus} chrt -f 1 ${model_command}"
    runner_runtime_tuning="$(cat <<EOF
echo "[runner] rvv_stable_profile=hz100-nohz-full-fifo-v1"
echo "[runner] cmdline=\$(cat /proc/cmdline)"
echo "[runner] nohz_full=\$(cat /sys/devices/system/cpu/nohz_full 2>/dev/null || true)"
sysctl -w kernel.sched_rt_runtime_us=-1 || true
sysctl -w kernel.watchdog=0 || true
sysctl -w kernel.timer_migration=1 || true
sysctl -w vm.stat_interval=120 || true
sysctl -w vm.dirty_writeback_centisecs=0 || true
[ -w /sys/devices/virtual/workqueue/cpumask ] && echo ${rvv_housekeeping_mask} > /sys/devices/virtual/workqueue/cpumask || true
for irq_dir in /proc/irq/[0-9]*; do
  [ -w "\${irq_dir}/smp_affinity_list" ] || continue
  echo ${rvv_housekeeping_cpus} >"\${irq_dir}/smp_affinity_list" 2>/dev/null || true
done
sync || true
dd if=$(shell_quote "${input_base}") of=/dev/null bs=1M 2>/dev/null || true
dd if=weights.bin of=/dev/null bs=4M 2>/dev/null || true
echo "[runner] scheduling and IRQ tuning complete"
echo "[runner] launch=${model_command}"
EOF
)"
  fi

  if [[ "${kernel_only}" -eq 1 ]]; then
    cat >"${workload_config_path}" <<EOF
workload=${workload_name}
artifact_dir=${artifact_dir}
elf_source=${elf_path}
elf_sha256=${elf_sha256}
input_source=${input_path}
input_sha256=${input_sha256}
weights_source=${weights_path}
weights_sha256=${weights_sha256}
rootfs_size=${rootfs_size}
openmp=disabled
execution=flex_attention_kernel_only
linux_maxcpus=1
EOF
  else
    cat >"${workload_config_path}" <<EOF
workload=${workload_name}
artifact_dir=${artifact_dir}
elf_source=${elf_path}
elf_sha256=${elf_sha256}
input_source=${input_path}
input_sha256=${input_sha256}
weights_source=${weights_path}
weights_sha256=${weights_sha256}
rootfs_size=${rootfs_size}
OMP_NUM_THREADS=${core}
OMP_THREAD_LIMIT=${core}
OMP_STACKSIZE=64M
OMP_DYNAMIC=false
OMP_MAX_ACTIVE_LEVELS=1
OMP_NESTED=false
OMP_PROC_BIND=close
OMP_PLACES=${places}
OMP_SCHEDULE=static
OMP_WAIT_POLICY=${omp_wait_policy}
OMP_DISPLAY_ENV=FALSE
OMP_DISPLAY_AFFINITY=${omp_display_affinity}
GOMP_CPU_AFFINITY=${cpu_affinity}
GOMP_SPINCOUNT=${gomp_spincount}
MALLOC_ARENA_MAX=1
EOF
    if [[ "${rvv_hz100_profile}" -eq 1 ]]; then
      cat >>"${workload_config_path}" <<EOF
rvv_stable_profile=hz100-nohz-full-fifo-v1
rvv_worker_cpus=${rvv_worker_cpus}
rvv_housekeeping_cpus=${rvv_housekeeping_cpus}
linux_hz=100
linux_nohz_full=true
scheduler_policy=SCHED_FIFO
scheduler_priority=1
EOF
    fi
  fi

  expect_output_bin=1
  if [[ "${kernel_only}" -eq 1 ]]; then
    model_command+=" --kernel-only=flex_attention --no-output"
    expect_output_bin=0
  elif [[ "${kind}" == "llm" ]]; then
    model_command+=" --no-output"
    expect_output_bin=0
  fi

  if [[ "${expect_output_bin}" -eq 1 ]]; then
    runner_check_files="output.bin run.log model.log autotune.log workload-config.txt"
    output_entries="    \"${guest_root}/${workload_name}/output.bin\",
    \"${guest_root}/${workload_name}/run.log\",
    \"${guest_root}/${workload_name}/model.log\",
    \"${guest_root}/${workload_name}/autotune.log\",
    \"${guest_root}/${workload_name}/workload-config.txt\""
  else
    runner_check_files="run.log model.log autotune.log workload-config.txt"
    output_entries="    \"${guest_root}/${workload_name}/run.log\",
    \"${guest_root}/${workload_name}/model.log\",
    \"${guest_root}/${workload_name}/autotune.log\",
    \"${guest_root}/${workload_name}/workload-config.txt\""
  fi

  if [[ "${kernel_only}" -eq 1 ]]; then
    runner_runtime_env=""
    runner_runtime_log='echo "[runner] openmp=disabled"'
  else
    runner_runtime_env="$(cat <<EOF
export OMP_NUM_THREADS=${core}
export OMP_THREAD_LIMIT=${core}
export OMP_STACKSIZE=64M
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=false
export OMP_PROC_BIND=close
export OMP_PLACES="${places}"
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=${omp_wait_policy}
export OMP_DISPLAY_ENV=FALSE
export OMP_DISPLAY_AFFINITY=${omp_display_affinity}
export GOMP_CPU_AFFINITY="${cpu_affinity}"
export GOMP_SPINCOUNT=${gomp_spincount}
export MALLOC_ARENA_MAX=1
EOF
)"
    runner_runtime_log='echo "[runner] OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT} OMP_STACKSIZE=${OMP_STACKSIZE} OMP_DYNAMIC=${OMP_DYNAMIC} OMP_MAX_ACTIVE_LEVELS=${OMP_MAX_ACTIVE_LEVELS} OMP_NESTED=${OMP_NESTED} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES} OMP_SCHEDULE=${OMP_SCHEDULE} OMP_WAIT_POLICY=${OMP_WAIT_POLICY} GOMP_CPU_AFFINITY=${GOMP_CPU_AFFINITY} GOMP_SPINCOUNT=${GOMP_SPINCOUNT} MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"'
  fi

  cat >"${runner_path}" <<EOF
#!/bin/bash
set -u
set -o pipefail

ulimit -s unlimited
cd ${guest_root}/${workload_name}

if [ -b /dev/iceblk ]; then
  mount -o remount,rw,noatime / 2>/dev/null || \\
    mount -o remount,rw,noatime /dev/iceblk / 2>/dev/null || true
else
  mount -o remount,rw,noatime / 2>/dev/null || true
fi

${runner_runtime_env}

chmod +x ./${elf_base} 2>/dev/null || true
rm -f output.bin run.log model.log autotune.log

exec >./run.log 2>&1

#jseo: Apply the validated RVV scheduling setup before launching the model.
${runner_runtime_tuning}

echo "[runner] start ${workload_name}"
${runner_runtime_log}
echo "[runner] elf_sha256=${elf_sha256}"
${model_command}
rc=\$?
echo "[runner] model_ret=\${rc}"
for log_file in ${runner_check_files}; do
  if [ -f "\${log_file}" ]; then
    echo "[runner] captured \${log_file}"
  else
    echo "[runner] missing \${log_file}"
  fi
done
sync || true
sleep 5
exit "\${rc}"
EOF
  chmod +x "${runner_path}"

  cat >"${hook_path}" <<EOF
#!/bin/sh
bash ${guest_root}/run_${workload_name}.sh
EOF
  chmod +x "${hook_path}"

  cat >"${workload_json}" <<EOF
{
  "name": "${workload_name}",
  "workdir": ".",
  "base": "br-base.json",
  "rootfs-size": "${rootfs_size}",
${linux_json_entry}  "overlay": "overlay-${workload_name}",
  "command": "bash ${guest_root}/run_${workload_name}.sh",
  "outputs": [
${output_entries}
  ]
}
EOF

  cat >"${deploy_json}" <<EOF
{
  "benchmark_name": "${workload_name}",
  "common_simulation_outputs": [
    "uartlog"
  ],
  "common_bootbinary": "${common_bootbinary}",
  "common_rootfs": "${common_rootfs}",
  "common_outputs": [
${output_entries}
  ]
}
EOF

  log "packaged ${workload_name}"
}

artifact_root="${WORKSPACE}/examples"
artifact_dir_args=()
selected_cores=()
check_chipyard=1
clean_workloads=1
rootfs_size="${PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE:-8GiB}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      [[ "$#" -ge 2 ]] || die "--artifact-dir requires a value"
      artifact_dir_args+=("$2")
      shift 2
      ;;
    --artifact-dir=*)
      artifact_dir_args+=("${1#--artifact-dir=}")
      shift
      ;;
    --core)
      [[ "$#" -ge 2 ]] || die "--core requires a value"
      IFS=, read -r -a requested_cores <<<"$2"
      selected_cores+=("${requested_cores[@]}")
      shift 2
      ;;
    --core=*)
      IFS=, read -r -a requested_cores <<<"${1#--core=}"
      selected_cores+=("${requested_cores[@]}")
      shift
      ;;
    --no-clean)
      clean_workloads=0
      shift
      ;;
    --no-chipyard-check)
      check_chipyard=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1'; pass --help for usage"
      ;;
  esac
done

for selected_core in "${selected_cores[@]}"; do
  [[ "${selected_core}" =~ ^[0-9]+$ && "${selected_core}" != "0" ]] || \
    die "invalid core count '${selected_core}'"
done

core_is_selected() {
  local core="$1"
  local selected

  [[ "${#selected_cores[@]}" -gt 0 ]] || return 0
  for selected in "${selected_cores[@]}"; do
    [[ "${selected}" == "${core}" ]] && return 0
  done
  return 1
}

artifact_root="$(abs_dir "${artifact_root}")"
ARTIFACT_ROOT="${artifact_root}"
validate_size "${rootfs_size}"

artifact_dirs=()
for artifact_dir in "${artifact_dir_args[@]}"; do
  artifact_dirs+=("$(abs_dir "${artifact_dir}")")
done

if [[ "${check_chipyard}" -eq 1 ]]; then
  [[ -f "${FIREMARSHAL_DIR}/marshal" ]] || \
    die "FireMarshal is not initialized at ${FIREMARSHAL_DIR}; check CHIPYARD_DIR or FIREMARSHAL_DIR"
  mkdir -p "${FIRESIM_DEPLOY_DIR}" "${FIRESIM_WORKLOAD_DIR}"
fi

shopt -s nullglob

elf_paths=()
if [[ "${#artifact_dirs[@]}" -gt 0 ]]; then
  while IFS= read -r -d '' elf_path; do
    elf_paths+=("${elf_path}")
  done < <(find "${artifact_dirs[@]}" -maxdepth 1 -type f -name 'model-*core.elf' -print0 | sort -z)
else
  while IFS= read -r -d '' elf_path; do
    elf_paths+=("${elf_path}")
  done < <(discover_artifact_elf_paths "${artifact_root}")
fi

if [[ "${#elf_paths[@]}" -eq 0 ]]; then
  die "no model-*core.elf build outputs found under ${artifact_root}; Stage 2 cannot package workloads until Stage 1 artifacts are present and ELF generation succeeds"
fi

log "artifact root: ${artifact_root}"
log "FireMarshal workload dir: ${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
log "FireSim workload dir: ${FIRESIM_WORKLOAD_DIR}"

package_artifact_dirs=()
package_elf_paths=()
package_workload_names=()
declare -A PLANNED_WORKLOADS=()

package_cleanup_on_exit() {
  local status=$?
  local workload
  local -a cleanup_args=()

  trap - EXIT
  if [[ "${status}" -ne 0 ]]; then
    for workload in "${package_workload_names[@]}"; do
      cleanup_args+=(--workload "${workload}")
    done
    [[ "${#cleanup_args[@]}" -gt 0 ]] || cleanup_args+=(--all)
    bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" "${cleanup_args[@]}" --final || \
      warn "failure cleanup did not complete"
  fi
  exit "${status}"
}
trap package_cleanup_on_exit EXIT

for elf_path in "${elf_paths[@]}"; do
  artifact_dir="$(dirname "${elf_path}")"
  elf_base="$(basename "${elf_path}")"
  core="$(infer_core_from_elf "${elf_base}")"
  [[ -n "${core}" ]] || die "could not infer core count from ${elf_path}"
  if ! core_is_selected "${core}"; then
    continue
  fi
  if ! elf_is_in_build_plan "${artifact_dir}" "${core}"; then
    warn "skipping stale ${elf_path}; core ${core} is not in ${artifact_dir}/.pytorch-chipyard-build-cores"
    continue
  fi
  if is_shadowed_elf "${artifact_dir}" "${core}"; then
    continue
  fi
  workload_name="$(derive_packaged_workload_name "${artifact_dir}" "${core}")"
  validate_name "workload name" "${workload_name}"
  require_file "${elf_path}"
  if ! select_input_bin "${artifact_dir}" >/dev/null; then
    die "could not find input.bin or a unique *input*.bin under ${artifact_dir}; Stage 2 cannot package ${workload_name}"
  fi
  require_file "${artifact_dir}/weights.bin"
  if [[ -n "${PLANNED_WORKLOADS[${workload_name}]:-}" ]]; then
    die "duplicate workload name '${workload_name}' from ${artifact_dir} and ${PLANNED_WORKLOADS[${workload_name}]}"
  fi
  PLANNED_WORKLOADS["${workload_name}"]="${artifact_dir}"
  if is_boom_flex_kernel_workload "${workload_name}"; then
    require_flex_kernel_only_elf "${elf_path}"
    require_no_openmp_elf "${elf_path}"
  fi
  package_artifact_dirs+=("${artifact_dir}")
  package_elf_paths+=("${elf_path}")
  package_workload_names+=("${workload_name}")
done

if [[ "${#package_elf_paths[@]}" -eq 0 ]]; then
  die "found ${#elf_paths[@]} ELF file(s), but none match the selected core/build plan; existing packaged workloads were not changed"
fi

log "validated package inputs for ${#package_elf_paths[@]} workload(s)"
mkdir -p "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" "${FIRESIM_WORKLOAD_DIR}"
if [[ "${clean_workloads}" -eq 1 ]]; then
  log "cleaning generated FireMarshal workload JSONs/overlays"
  clean_generated_workloads
fi

declare -A PACKAGED_WORKLOADS=()
packaged_count=0
for ((package_index = 0; package_index < ${#package_elf_paths[@]}; package_index++)); do
  write_workload \
    "${package_artifact_dirs[package_index]}" \
    "${package_elf_paths[package_index]}" \
    "${rootfs_size}" \
    "${package_workload_names[package_index]}"
  packaged_count=$((packaged_count + 1))
done

[[ "${packaged_count}" -eq "${#package_elf_paths[@]}" ]] || \
  die "internal packaging count mismatch: planned ${#package_elf_paths[@]}, generated ${packaged_count}"
for workload_name in "${package_workload_names[@]}"; do
  require_file "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/${workload_name}.json"
  [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload_name}" ]] || \
    die "generated workload overlay not found: ${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload_name}"
  require_file "${FIRESIM_WORKLOAD_DIR}/${workload_name}.json"
done

log "done: packaged ${packaged_count} workload(s)"
log "next: bash ${SCRIPT_DIR}/build-firemarshal-images.sh"

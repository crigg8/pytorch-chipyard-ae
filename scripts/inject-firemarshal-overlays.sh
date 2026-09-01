#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/env.sh"

log() {
  printf '[firemarshal-inject] %s\n' "$*"
}

warn() {
  printf '[firemarshal-inject][warn] %s\n' "$*" >&2
}

die() {
  printf '[firemarshal-inject][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/inject-firemarshal-overlays.sh [options]

Replace the BOOM FlexAttention ELF, workload-config, and runner in existing
FireMarshal images without rebuilding the base image. The image must be
offline. Files are written after changing to the guest directory because the
server's debugfs 1.45.5 does not reliably create long absolute guest paths.

Options:
  --boom-flex       Inject all OPT BOOM flash/window sequence lengths.
  --workload=NAME  Inject one BOOM FlexAttention workload. May be repeated or
                   comma-separated.
  -h, --help       Show this help.
EOF
}

validate_workload() {
  local workload="$1"

  [[ "${workload}" =~ ^opt-boom-gemmini-(flash|window)-(256|512|768|1024)tok-1core$ ]] || \
    die "unsupported workload '${workload}'; expected an OPT BOOM FlexAttention 1-core workload"
}

append_workload_arg() {
  local value="$1"
  local item
  local -a items=()

  IFS=, read -r -a items <<<"${value}"
  for item in "${items[@]}"; do
    [[ -n "${item}" ]] || continue
    validate_workload "${item}"
    requested_workloads+=("${item}")
  done
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "required file not found: ${path}"
}

run_fsck() {
  local image="$1"
  local status=0

  set +e
  "${E2FSCK}" -pf "${image}"
  status=$?
  set -e
  if ((status <= 1)); then
    return 0
  fi

  warn "e2fsck -pf returned ${status}; retrying repair with -fy: ${image}"
  set +e
  "${E2FSCK}" -fy "${image}"
  status=$?
  set -e
  ((status <= 1)) || die "e2fsck failed with status ${status}: ${image}"
}

verify_image_file() {
  local image="$1"
  local source_file="$2"
  local guest_file="$3"
  local host_sha guest_sha

  host_sha="$(sha256sum "${source_file}" | awk '{print $1}')"
  guest_sha="$(
    "${DEBUGFS}" -R "cat ${guest_file}" "${image}" 2>/dev/null |
      sha256sum |
      awk '{print $1}'
  )"
  [[ "${host_sha}" == "${guest_sha}" ]] || \
    die "image hash mismatch for ${guest_file}: host=${host_sha} guest=${guest_sha}"
  log "verified ${guest_file} sha256=${host_sha}"
}

require_validated_boom_overlay() {
  local elf="$1"
  local runner="$2"

  grep -aFq -- '--kernel-only=flex_attention' "${elf}" || \
    die "kernel-only option is missing from ${elf}"
  LC_ALL=C readelf -Ws "${elf}" 2>/dev/null | awk '
    NF >= 8 {
      name = $8
      if (name ~ /^(GOMP_|omp_)/ || name ~ /_omp_fn/) found = 1
    }
    END { exit(found ? 0 : 1) }
  ' || die "OpenMP symbols are missing from ${elf}"
  LC_ALL=C readelf -d "${elf}" 2>/dev/null | awk '
    /\(NEEDED\)/ && /libgomp/ { found = 1 }
    END { exit(found ? 0 : 1) }
  ' || die "libgomp is missing from ${elf}"

  grep -Fq -- '--kernel-only=flex_attention' "${runner}" || \
    die "kernel-only command is missing from ${runner}"
  grep -Fq -- 'export OMP_NUM_THREADS=1' "${runner}" || \
    die "OMP_NUM_THREADS=1 is missing from ${runner}"
  grep -Fq -- 'export OMP_THREAD_LIMIT=1' "${runner}" || \
    die "OMP_THREAD_LIMIT=1 is missing from ${runner}"
  grep -Fq -- 'export OMP_PLACES="{0}"' "${runner}" || \
    die "OMP_PLACES={0} is missing from ${runner}"
  grep -Fq -- 'export OMP_SCHEDULE=static' "${runner}" || \
    die "OMP_SCHEDULE=static is missing from ${runner}"
  grep -Fq -- 'export OMP_WAIT_POLICY=PASSIVE' "${runner}" || \
    die "OMP_WAIT_POLICY=PASSIVE is missing from ${runner}"
  grep -Fq -- 'export GOMP_CPU_AFFINITY="0"' "${runner}" || \
    die "GOMP_CPU_AFFINITY=0 is missing from ${runner}"
  grep -Fq -- 'export GOMP_SPINCOUNT=0' "${runner}" || \
    die "GOMP_SPINCOUNT=0 is missing from ${runner}"
}

inject_workload() {
  local workload="$1"
  local overlay image elf config runner batch_output

  overlay="${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload}/root/llm"
  image="${FIREMARSHAL_IMAGE_DIR}/${workload}/${workload}.img"
  elf="${overlay}/${workload}/model-1core.elf"
  config="${overlay}/${workload}/workload-config.txt"
  runner="${overlay}/run_${workload}.sh"

  require_file "${image}"
  require_file "${elf}"
  require_file "${config}"
  require_file "${runner}"
  require_validated_boom_overlay "${elf}" "${runner}"

  log "checking filesystem: ${image}"
  run_fsck "${image}"

  batch_output="$(
    {
      printf 'cd /root/llm/%s\n' "${workload}"
      printf 'rm model-1core.elf\n'
      printf 'rm workload-config.txt\n'
      printf 'write %s model-1core.elf\n' "${elf}"
      printf 'write %s workload-config.txt\n' "${config}"
      printf 'cd /root/llm\n'
      printf 'rm run_%s.sh\n' "${workload}"
      printf 'write %s run_%s.sh\n' "${runner}" "${workload}"
    } | "${DEBUGFS}" -w -f - "${image}" 2>&1
  )" || die "debugfs failed for ${image}"
  printf '%s\n' "${batch_output}"

  run_fsck "${image}"
  verify_image_file "${image}" "${elf}" "/root/llm/${workload}/model-1core.elf"
  verify_image_file "${image}" "${config}" "/root/llm/${workload}/workload-config.txt"
  verify_image_file "${image}" "${runner}" "/root/llm/run_${workload}.sh"
  touch "${image}"
  log "IMAGE_OK ${workload}"
}

requested_workloads=()
inject_all_boom_flex=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --boom-flex)
      inject_all_boom_flex=1
      shift
      ;;
    --workload)
      [[ "$#" -ge 2 ]] || die "--workload requires a value"
      append_workload_arg "$2"
      shift 2
      ;;
    --workload=*)
      append_workload_arg "${1#--workload=}"
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

if [[ "${inject_all_boom_flex}" -eq 1 ]]; then
  for attention in flash window; do
    for seq in 256 512 768 1024; do
      requested_workloads+=("opt-boom-gemmini-${attention}-${seq}tok-1core")
    done
  done
fi

[[ "${#requested_workloads[@]}" -gt 0 ]] || die "select --boom-flex or at least one --workload"
[[ -n "${PYTORCH_CHIPYARD_WORKLOAD_DIR:-}" ]] || die "PYTORCH_CHIPYARD_WORKLOAD_DIR is not set"
[[ -n "${FIREMARSHAL_IMAGE_DIR:-}" ]] || die "FIREMARSHAL_IMAGE_DIR is not set"
DEBUGFS="$(command -v debugfs)" || die "debugfs is not installed"
E2FSCK="$(command -v e2fsck)" || die "e2fsck is not installed"
command -v readelf >/dev/null 2>&1 || die "readelf is not installed"

if pgrep -f '[F]ireSim-xilinx_alveo_u250|[f]iresim.*runworkload' >/dev/null; then
  die "FireSim is still running; stop it before modifying FireMarshal images"
fi

declare -A seen=()
for workload in "${requested_workloads[@]}"; do
  [[ -z "${seen[${workload}]:-}" ]] || continue
  seen["${workload}"]=1
  inject_workload "${workload}"
done

log "done"

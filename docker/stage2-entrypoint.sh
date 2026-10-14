#!/usr/bin/env bash
set -euo pipefail

die() { printf '[stage2-container] %s\n' "$*" >&2; exit 1; }
if [[ "$#" -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'EOF'
Stage2: prebuilt ELF -> FireMarshal disk image -> FireSim run -> results.
Use scripts/run-stage2-docker.sh with system Docker.
Options: --workload=LIST, --experiment=NAME, --resume, --rerun-completed,
         --preflight-only, --rvv-panic-retries=N, --resume-from=NAME.
Table 4, TVM, Verilator, model compilation and nodisk are not supported.
EOF
  exit 0
fi
[[ "${STAGE2_HOST_LOCKED:-}" == "1" ]] || die 'use the host launcher to acquire the global execution lock'
: "${STAGE2_RUN_DIR:?}" "${FIRESIM_SSH_HOST:?}" "${FIRESIM_SSH_USER:?}"
: "${STAGE2_IMAGE_ID:?}" "${FIRESIM_DB:?}" "${STAGE2_HOST_MACHINE_ID:?}"
[[ "${STAGE2_RUN_DIR}" == /* && "${STAGE2_RUN_DIR}" != / ]] || die 'invalid run directory'

args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --run) shift ;;
    --workload | --experiment | --resume-from | --rvv-panic-retries)
      [[ "$#" -ge 2 && "$2" != --* ]] || die "missing value for $1"
      args+=("$1=$2"); shift 2 ;;
    --workload=* | --experiment=* | --resume-from=* | --rvv-panic-retries=* | \
    --resume | --rerun-completed | --preflight-only)
      args+=("$1"); shift ;;
    *) die "unsupported Stage2 container option: $1" ;;
  esac
done
for argument in "${args[@]}"; do
  [[ "${argument}" != --experiment=table-4 ]] || die 'Table 4 is excluded from this image'
done

export PYTORCH_CHIPYARD_STAGE2_CONTAINER=1
export USER="${FIRESIM_SSH_USER}" LOGNAME="${FIRESIM_SSH_USER}"
export CHIPYARD_DIR=/opt/chipyard
export FIRESIM_DIR="${CHIPYARD_DIR}/sims/firesim"
export FIREMARSHAL_DIR="${CHIPYARD_DIR}/software/firemarshal"
export FIRESIM_DEPLOY_DIR="${STAGE2_RUN_DIR}/firesim/deploy"
export MARSHAL_FIRESIM_DIR="${STAGE2_RUN_DIR}/firesim"
export MARSHAL_IMAGE_DIR="${STAGE2_RUN_DIR}/images"
export FIREMARSHAL_IMAGE_DIR="${MARSHAL_IMAGE_DIR}/firechip"
export PYTORCH_CHIPYARD_WORKLOAD_DIR="${STAGE2_RUN_DIR}/workloads"
export FIRESIM_WORKLOAD_DIR="${FIRESIM_DEPLOY_DIR}/workloads"
export FIRESIM_HWDB_PATH="${FIRESIM_HWDB_PATH:-/opt/stage2-metadata/config_hwdb.yaml}"
export FIRESIM_BUILD_RECIPES_PATH="${FIRESIM_DIR}/deploy/config_build_recipes.yaml"
export PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR="${FIRESIM_DEPLOY_DIR}/results-workload"
export PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${STAGE2_RUN_DIR}/results"
export PYTORCH_CHIPYARD_LOG_DIR="${STAGE2_RUN_DIR}/logs"
export PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${STAGE2_RUN_DIR}/runtime"
export FIRESIM_RUNS_DIR="${STAGE2_RUN_DIR}/runfarm"
export PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR="${STAGE2_RUN_DIR}/tmp"
export TMPDIR="${STAGE2_RUN_DIR}/tmp"
export MARSHAL_MOUNT_DIR="${STAGE2_RUN_DIR}/unused-mount"
export LIBGUESTFS_CACHEDIR="${STAGE2_RUN_DIR}/guestfs-cache"
export LIBGUESTFS_TMPDIR="${STAGE2_RUN_DIR}/tmp"
export MARSHAL_LOG_DIR="${STAGE2_RUN_DIR}/marshal-logs"
export MARSHAL_RES_DIR="${STAGE2_RUN_DIR}/marshal-run-output"
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST=stage2-fpga
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC=one_fpgas_spec
export PYTORCH_CHIPYARD_FPGA_DB="${FIRESIM_DB}"
export PYTORCH_CHIPYARD_ACCOUNT_ENV=/dev/null
export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=0
export PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR=1
export PYTORCH_CHIPYARD_RVV_PANIC_RETRIES=0
export FIREMARSHAL_CONFIG_PATH="${STAGE2_RUN_DIR}/marshal-config.yaml"
export STAGE2_SSH_CONFIG=/root/.ssh/config
export LIBGUESTFS_BACKEND=direct LIBGUESTFS_BACKEND_SETTINGS=force_tcg
export LIBGUESTFS_PATH=/opt/guestfs-appliance
# Bound parallelism also works on this server's cgroup-v1 rootless daemon.
# Source archives can contain IDs outside the rootless UID/GID map (e.g.
# less-643 uses 197611). Set Buildroot's source-only option as a make override;
# a TAR_OPTIONS environment setting is discarded by Buildroot.
export MAKEFLAGS="-j${STAGE2_BUILD_JOBS:-8} BR2_TAR_OPTIONS=--no-same-owner"

if [[ "${STAGE2_EXECUTION:-ssh}" == local ]]; then
  # Early input/identity errors still leave diagnostic files owned by the user.
  trap 'chown -hR "${STAGE2_OUTPUT_UID:?}:${STAGE2_OUTPUT_GID:?}" -- "${STAGE2_RUN_DIR}" || true' EXIT
fi
mkdir -p "${FIRESIM_DEPLOY_DIR}" "${FIRESIM_WORKLOAD_DIR}" "${MARSHAL_IMAGE_DIR}" \
  "${FIREMARSHAL_IMAGE_DIR}" "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" "${TMPDIR}" \
  "${MARSHAL_MOUNT_DIR}" "${LIBGUESTFS_CACHEDIR}" "${MARSHAL_LOG_DIR}" \
  "${MARSHAL_RES_DIR}" "${PYTORCH_CHIPYARD_LOG_DIR}" \
  "${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR}" "${FIRESIM_RUNS_DIR}"

# Marshal's user config overrides the copied server-specific configuration.
cat >"${FIREMARSHAL_CONFIG_PATH}" <<EOF
board-dir: ${FIREMARSHAL_DIR}/boards/firechip
image-dir: ${MARSHAL_IMAGE_DIR}
mount-dir: ${MARSHAL_MOUNT_DIR}
firesim-dir: ${MARSHAL_FIRESIM_DIR}
log-dir: ${MARSHAL_LOG_DIR}
res-dir: ${MARSHAL_RES_DIR}
jlevel: ${STAGE2_BUILD_JOBS:-8}
EOF
cp "${FIREMARSHAL_CONFIG_PATH}" "${FIREMARSHAL_DIR}/marshal-config.yaml"
mkdir -p /root/.ssh
cp /run/stage2/known_hosts /root/.ssh/known_hosts
chmod 600 /root/.ssh/known_hosts
cat >"${STAGE2_SSH_CONFIG}" <<EOF
Host stage2-fpga
    HostName ${FIRESIM_SSH_HOST}
    User ${FIRESIM_SSH_USER}
    Port ${FIRESIM_SSH_PORT:-22}
    BatchMode yes
    StrictHostKeyChecking yes
    UserKnownHostsFile /root/.ssh/known_hosts
    ForwardAgent no
    ConnectTimeout 15
EOF
chmod 600 "${STAGE2_SSH_CONFIG}"

set +u
source "${CHIPYARD_DIR}/env.sh"
set -u
# Keep appliance utilities on the Ubuntu ABI, outside Conda's library PATH.
export LIBGUESTFS_HV=/usr/bin/qemu-system-x86_64
cd /opt/pytorch-chipyard
python=/usr/bin/python3
"${python}" /opt/stage2-support/runtime.py begin -- "${args[@]}"
child=""
finish() {
  local status=$?
  trap - EXIT
  "${python}" /opt/stage2-support/runtime.py finish "${status}" || true
  if [[ "${STAGE2_EXECUTION:-ssh}" == local ]]; then
    chown -hR "${STAGE2_OUTPUT_UID:?}:${STAGE2_OUTPUT_GID:?}" -- "${STAGE2_RUN_DIR}" || true
  fi
  exit "${status}"
}
interrupt() {
  trap - INT TERM
  if [[ -n "${child}" ]]; then
    kill -TERM -- "-${child}" 2>/dev/null || true
    wait "${child}" 2>/dev/null || true
  fi
  exit 130
}
trap finish EXIT
trap interrupt INT TERM
"${python}" /opt/stage2-support/runtime.py preflight
setsid bash scripts/run-stage2.sh --skip-elves --skip-table4 "${args[@]}" &
child=$!
wait "${child}"

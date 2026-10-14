#!/usr/bin/env bash
# Link full-evaluation ELFs in Stage 1 before mounting them read-only in Stage 2.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
die() { printf '[stage2-inputs][error] %s\n' "$*" >&2; exit 1; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'EOF'
Usage: prepare-stage2-inputs.sh
Link the completed Stage 1 examples using the Stage 1 image and GCC 12.2.0.
No FireMarshal build or FPGA execution is performed.

Environment:
  STAGE2_ARTIFACT_DIR     Default: <checkout>/examples
  STAGE1_IMAGE           Default: pytorch-chipyard:stage1
  DOCKER_HOST/DOCKER_CONTEXT select the Stage 1 rootless Docker engine.
EOF
  exit 0
fi
[[ "$#" -eq 0 ]] || die "unknown argument: $1"
artifacts="$(realpath -e -- "${STAGE2_ARTIFACT_DIR:-${REPO_ROOT}/examples}")"
[[ -d "${artifacts}" && "${artifacts}" != / ]] || die 'artifact directory must contain completed Stage 1 outputs'
[[ "${artifacts}" != *,* ]] || die 'bind mount paths cannot contain commas'
security="$(docker info --format '{{json .SecurityOptions}}')"
[[ "${security}" == *name=rootless* ]] || die 'select the Stage 1 rootless Docker engine'
image="$(docker image inspect --format '{{.Id}}' "${STAGE1_IMAGE:-pytorch-chipyard:stage1}")"
docker run --rm --init --pull=never -i \
  --mount "type=bind,src=${artifacts},dst=/opt/pytorch-chipyard/examples" \
  "${image}" bash -s <<'CONTAINER'
set -Eeuo pipefail
cd /opt/pytorch-chipyard
compiler=/opt/riscv-gcc-12.2.0/riscv-tools/bin/riscv64-unknown-linux-gnu-g++
[[ -x "$compiler" && "$("$compiler" -dumpfullversion)" == 12.2.0 ]] || {
  echo 'Rebuild the Stage 1 image to include RISC-V GCC 12.2.0' >&2
  exit 1
}
bash scripts/build-chipyard-elves.sh \
  --chipyard-env=/opt/riscv-gcc-12.2.0/env.sh \
  --riscv-gxx="$compiler"
CONTAINER
printf 'STAGE2_ARTIFACT_DIR=%s\nSTAGE2_INPUTS_STATUS=PASS\n' "${artifacts}"

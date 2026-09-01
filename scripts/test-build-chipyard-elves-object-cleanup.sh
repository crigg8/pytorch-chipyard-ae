#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-chipyard-elf-object-test.XXXXXX")"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

LLVM_DIR="${TEST_ROOT}/llvm-project"
CHIPYARD_ENV="${TEST_ROOT}/chipyard-env.sh"
PARENT="${TEST_ROOT}/examples/artifact-test/gemmini"
CHILD="${PARENT}/alias-first-on"

mkdir -p "${LLVM_DIR}" "${PARENT}/kernels/parent" "${CHILD}/kernels/child"
: >"${CHIPYARD_ENV}"

make_artifact() {
  local artifact_dir="$1"
  local object_rel="$2"

  printf '%s\n' gemmini >"${artifact_dir}/.pytorch-chipyard-backend"
  printf '%s\n' 4 >"${artifact_dir}/.pytorch-chipyard-build-cores"
  printf '%s\n' spec >"${artifact_dir}/model_spec.json"
  printf '%s\n' input >"${artifact_dir}/input.bin"
  printf '%s\n' weights >"${artifact_dir}/weights.bin"
  printf '%s\n' runner >"${artifact_dir}/runner.cpp"
  printf '%s\n' object >"${artifact_dir}/${object_rel}"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -euo pipefail'
    printf 'test -s %q\n' "${artifact_dir}/${object_rel}"
    printf 'printf %q > %q\n' elf "${artifact_dir}/model.elf"
  } >"${artifact_dir}/build.sh"
}

make_artifact "${PARENT}" kernels/parent/parent.obj
make_artifact "${CHILD}" kernels/child/child.obj

LLVM_PROJECT_PATH="${LLVM_DIR}" \
CHIPYARD_ENV_PATH="${CHIPYARD_ENV}" \
PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR= \
PYTORCH_CHIPYARD_RISCV_GXX= \
bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
  --artifact-dir="${PARENT}" \
  --artifact-dir="${CHILD}" >/dev/null

test -s "${PARENT}/model-4core.elf"
test -s "${CHILD}/model-4core.elf"
test ! -e "${PARENT}/kernels/parent/parent.obj"
test ! -e "${CHILD}/kernels/child/child.obj"
test -s "${PARENT}/runner.cpp"
test -s "${PARENT}/input.bin"
test -s "${PARENT}/weights.bin"
test -s "${PARENT}/model_spec.json"

printf 'BUILD_CHIPYARD_ELVES_OBJECT_CLEANUP_TEST=PASS\n'

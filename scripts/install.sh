#!/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
export TRITON_PLUGIN_DIRS="$WORKSPACE/triton_chipyard"
export LLVM_BUILD_DIR="$WORKSPACE/llvm-project/build"
export LLVM_BINARY_DIR="$LLVM_BUILD_DIR/bin"

TRITON_BUILD_ROOT="$WORKSPACE/triton/build"
TRITON_CMAKE_BUILD_DIR="$TRITON_BUILD_ROOT/cmake.linux-x86_64-cpython-3.12"
export TRITON_BUILD_DIR="$TRITON_CMAKE_BUILD_DIR"
export TRITON_CHIPYARD_OPT_PATH="$TRITON_CMAKE_BUILD_DIR/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt"

BUDDY_BUILD_DIR="$WORKSPACE/buddy-mlir/build"
KEEP_BUILD_ARTIFACTS="${PYTORCH_CHIPYARD_KEEP_BUILD_ARTIFACTS:-0}"
MAX_JOBS="${MAX_JOBS:-16}"
export PIP_NO_CACHE_DIR=1

BLUE="\033[0;34m"; GREEN="\033[0;32m"; RED="\033[0;31m"; NC="\033[0m"
info(){ echo -e "${BLUE}[INFO]${NC} $*"; }
ok(){ echo -e "${GREEN}[OK]${NC} $*"; }
die(){ echo -e "${RED}[ERR]${NC} $*" >&2; exit 1; }

case "$KEEP_BUILD_ARTIFACTS" in
  0 | 1) ;;
  *) die "PYTORCH_CHIPYARD_KEEP_BUILD_ARTIFACTS must be 0 or 1" ;;
esac

disk_report() {
  local label="$1"
  local path
  info "disk usage: $label"
  df -h "$WORKSPACE"
  for path in \
    "$LLVM_BUILD_DIR" \
    "$TRITON_BUILD_ROOT" \
    "$BUDDY_BUILD_DIR" \
    "${HOME:?}/.triton" \
    "${HOME:?}/.cache/pip"; do
    if [[ -e "$path" ]]; then
      du -sh "$path"
    fi
  done
}

strip_elf() {
  local path="$1"
  [[ -f "$path" ]] || die "cannot strip missing file: $path"
  file "$path" | grep -q 'ELF' || die "expected an ELF file: $path"
  if [[ "$KEEP_BUILD_ARTIFACTS" == "0" ]]; then
    strip --strip-unneeded "$path"
  else
    info "retaining symbols because PYTORCH_CHIPYARD_KEEP_BUILD_ARTIFACTS=1: $path"
  fi
}

assert_no_deleted_build_dependencies() {
  local path="$1"
  local output
  [[ -f "$path" ]] || die "cannot inspect missing runtime file: $path"
  output="$(ldd "$path" 2>&1 || true)"
  printf '%s\n' "$output"
  if grep -Fq 'not found' <<<"$output"; then
    die "unresolved shared library dependency: $path"
  fi
  if [[ "$KEEP_BUILD_ARTIFACTS" == "0" ]]; then
    if grep -Fq "$LLVM_BUILD_DIR" <<<"$output" || \
       grep -Fq "$TRITON_BUILD_ROOT" <<<"$output" || \
       grep -Fq "$BUDDY_BUILD_DIR" <<<"$output"; then
      die "runtime file depends on a build directory scheduled for deletion: $path"
    fi
  fi
}

cleanup_download_caches() {
  rm -rf "${HOME:?}/.triton" "${HOME:?}/.cache/pip"
}

disk_report "before dependency installation"

############################################################
## Conda and Python dependencies
############################################################

conda create -n pytorch-chipyard python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch-chipyard
conda install -n pytorch-chipyard conda-forge::conda-lock=1.4.0 -y
conda install -n pytorch-chipyard -c conda-forge \
  gcc_linux-64=13 gxx_linux-64=13 zlib libzlib -y

# Conda packages are all installed before the native compiler builds start.
conda clean -afy

python -m pip install --no-cache-dir matplotlib pandas
python -m pip install --no-cache-dir torch==2.8.0
python -m pip install --no-cache-dir torchvision==0.23.0
python -m pip uninstall -y triton

# Triton-Chipyard build dependencies.
python -m pip install --no-cache-dir ninja cmake wheel pytest pybind11 setuptools

python -m pip install --no-cache-dir -r "$WORKSPACE/triton/python/requirements.txt"
python -m pip install --no-cache-dir -r "$WORKSPACE/buddy-mlir/requirements.txt"
cleanup_download_caches
disk_report "after dependency installation and cache cleanup"

############################################################
## Patch the installed PyTorch before native compiler builds
############################################################

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE="${PATCHED_INDUCTOR_DIR:-}"
if [[ -z "$SOURCE" ]]; then
  if [[ -d "$WORKSPACE/pytorch/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/pytorch/torch/_inductor"
  elif [[ -d "$WORKSPACE/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/torch/_inductor"
  elif [[ -d "$WORKSPACE/LLAPI-PyTorch/torch/_inductor" ]]; then
    SOURCE="$WORKSPACE/LLAPI-PyTorch/torch/_inductor"
  else
    die "patched _inductor not found. Set PATCHED_INDUCTOR_DIR or place it under ./pytorch/torch/_inductor"
  fi
fi
[[ -d "$SOURCE" ]] || die "source not a dir: $SOURCE"

TORCH_DIR="$("$PYTHON_BIN" - <<'PY'
import os
import sys
import torch

version = torch.__version__
if not str(version).startswith("2.8.0"):
    print(f"ERROR: expected torch 2.8.0.*, got {version}", file=sys.stderr)
    sys.exit(2)
print(os.path.dirname(torch.__file__))
PY
)" || die "cannot locate torch (or version mismatch)"

TARGET="$TORCH_DIR/_inductor"
[[ -w "$TORCH_DIR" ]] || die "no write permission: $TORCH_DIR (wrong env?)"

info "python : $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
info "source : $SOURCE"
info "target : $TARGET"
rm -rf "$TARGET"
mkdir -p "$TARGET"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$SOURCE/" "$TARGET/"
else
  cp -a "$SOURCE/." "$TARGET/"
fi

"$PYTHON_BIN" - <<'PY'
import torch
import torch._inductor

print("torch version  :", torch.__version__)
print("torch._inductor:", torch._inductor.__file__)
PY
ok "patched torch/_inductor"

############################################################
## Build LLVM
############################################################

disk_report "before LLVM build"
cmake -G Ninja \
  -S "$WORKSPACE/llvm-project/llvm" \
  -B "$LLVM_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU;RISCV" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=ON \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DMLIR_INCLUDE_DOCS=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
cmake --build "$LLVM_BUILD_DIR" --parallel "$MAX_JOBS"
disk_report "after LLVM build"

############################################################
## Build and compact Triton + Triton-Chipyard
############################################################

export TRITON_BUILD_PROTON=OFF
export TRITON_APPEND_CMAKE_ARGS="${TRITON_APPEND_CMAKE_ARGS:-} -DTRITON_BUILD_UT=OFF"

disk_report "before Triton build"
pushd "$WORKSPACE/triton" >/dev/null
LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
  LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
  LLVM_SYSPATH="$LLVM_BUILD_DIR" \
  python -m pip install --no-cache-dir --no-build-isolation .
popd >/dev/null

mapfile -t TRITON_RUNTIME_PATHS < <("$PYTHON_BIN" - <<'PY'
import importlib
import inspect
from pathlib import Path

import triton
import triton.backends as triton_backends
from triton._C import libtriton

if "triton_chipyard" not in triton_backends.backends:
    raise SystemExit("Triton-Chipyard backend was not registered")

package_dir = Path(triton.__file__).resolve().parent
compiler_class = triton_backends.backends["triton_chipyard"].compiler
compiler_module = importlib.import_module(compiler_class.__module__)
compiler_path = Path(inspect.getfile(compiler_module)).resolve()
if package_dir not in compiler_path.parents:
    raise SystemExit(
        f"Triton-Chipyard backend is not installed under site-packages: {compiler_path}"
    )

print(package_dir)
print(Path(libtriton.__file__).resolve())
print(compiler_path)
PY
)
[[ "${#TRITON_RUNTIME_PATHS[@]}" -eq 3 ]] || die "failed to locate installed Triton runtime files"
TRITON_PACKAGE_DIR="${TRITON_RUNTIME_PATHS[0]}"
LIBTRITON_PATH="${TRITON_RUNTIME_PATHS[1]}"
TRITON_CHIPYARD_COMPILER_PATH="${TRITON_RUNTIME_PATHS[2]}"

[[ -f "$TRITON_CHIPYARD_OPT_PATH" ]] || \
  die "missing triton-chipyard-opt after Triton build: $TRITON_CHIPYARD_OPT_PATH"
info "Triton package             : $TRITON_PACKAGE_DIR"
info "Triton native library      : $LIBTRITON_PATH"
info "Triton-Chipyard compiler   : $TRITON_CHIPYARD_COMPILER_PATH"

strip_elf "$LIBTRITON_PATH"
strip_elf "$TRITON_CHIPYARD_OPT_PATH"
assert_no_deleted_build_dependencies "$LIBTRITON_PATH"
assert_no_deleted_build_dependencies "$TRITON_CHIPYARD_OPT_PATH"
"$TRITON_CHIPYARD_OPT_PATH" --help >/dev/null

if [[ "$KEEP_BUILD_ARTIFACTS" == "0" ]]; then
  triton_stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-chipyard-triton.XXXXXX")"
  mv "$TRITON_CHIPYARD_OPT_PATH" "$triton_stage_dir/triton-chipyard-opt"
  rm -rf "$TRITON_BUILD_ROOT"
  mkdir -p "$(dirname -- "$TRITON_CHIPYARD_OPT_PATH")"
  mv "$triton_stage_dir/triton-chipyard-opt" "$TRITON_CHIPYARD_OPT_PATH"
  rmdir "$triton_stage_dir"

  # These CUDA payloads are downloaded by Triton's unmodified setup.py. They
  # are not used by the Chipyard-only compiler image.
  for nvidia_backend_dir in \
    "$WORKSPACE/triton/third_party/nvidia/backend" \
    "$TRITON_PACKAGE_DIR/backends/nvidia"; do
    rm -rf \
      "$nvidia_backend_dir/bin" \
      "$nvidia_backend_dir/include" \
      "$nvidia_backend_dir/lib/cudart" \
      "$nvidia_backend_dir/lib/cupti"
  done

  # bdist_wheel creates source-tree links and metadata while collecting the
  # external backend. The installed wheel no longer needs these copies.
  [[ ! -L "$WORKSPACE/triton/python/triton/backends/triton_chipyard" ]] || \
    rm -f "$WORKSPACE/triton/python/triton/backends/triton_chipyard"
  [[ ! -L "$WORKSPACE/triton/python/triton/tools/extra/triton-chipyard-opt" ]] || \
    rm -f "$WORKSPACE/triton/python/triton/tools/extra/triton-chipyard-opt"
  rm -f "$WORKSPACE/triton/compile_commands.json"
  rm -rf \
    "$WORKSPACE/triton/triton.egg-info" \
    "$WORKSPACE/triton/python/triton.egg-info" \
    "$TRITON_PACKAGE_DIR/tools/extra/triton-chipyard-opt"
  cleanup_download_caches
fi

"$PYTHON_BIN" - <<'PY'
import triton
import triton.backends as triton_backends

assert "triton_chipyard" in triton_backends.backends
print("triton version:", triton.__version__)
print("triton backends:", sorted(triton_backends.backends))
PY
disk_report "after Triton compaction"

############################################################
## Configure and build only the Buddy tools used at runtime
############################################################

disk_report "before Buddy configure"
cmake -G Ninja \
  -S "$WORKSPACE/buddy-mlir" \
  -B "$BUDDY_BUILD_DIR" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DLLVM_MAIN_SRC_DIR="$WORKSPACE/llvm-project/llvm" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=OFF \
  -DBUDDY_EXAMPLES=OFF \
  -DBUDDY_TRANSFORMER_EXAMPLES=OFF \
  -DBUDDY_ENABLE_E2E_TESTS=OFF \
  -DPython3_EXECUTABLE="$(command -v python3)"

if [[ "$KEEP_BUILD_ARTIFACTS" == "0" ]]; then
  if grep -Eiq 'liblld|/tools/lld|/cmake/lld' "$BUDDY_BUILD_DIR/build.ninja"; then
    die "Buddy build graph unexpectedly references LLD; refusing to prune required artifacts"
  else
    info "Buddy build graph does not reference LLD; pruning Triton-only LLD artifacts"
    rm -rf \
      "$LLVM_BUILD_DIR/tools/lld" \
      "$LLVM_BUILD_DIR/lib/cmake/lld" \
      "$LLVM_BUILD_DIR/include/lld"
    find "$LLVM_BUILD_DIR/lib" -maxdepth 1 -type f -name 'liblld*.a' -delete
  fi
fi

cmake --build "$BUDDY_BUILD_DIR" \
  --target buddy-opt buddy-translate buddy-llc \
  --parallel "$MAX_JOBS"
disk_report "after Buddy tool build"

BUDDY_RUNTIME_TOOLS=(buddy-opt buddy-translate buddy-llc)
for tool in "${BUDDY_RUNTIME_TOOLS[@]}"; do
  tool_path="$BUDDY_BUILD_DIR/bin/$tool"
  strip_elf "$tool_path"
  assert_no_deleted_build_dependencies "$tool_path"
  "$tool_path" --version >/dev/null
done

if [[ "$KEEP_BUILD_ARTIFACTS" == "0" ]]; then
  buddy_stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-chipyard-buddy.XXXXXX")"
  for tool in "${BUDDY_RUNTIME_TOOLS[@]}"; do
    mv "$BUDDY_BUILD_DIR/bin/$tool" "$buddy_stage_dir/$tool"
  done
  rm -rf "$BUDDY_BUILD_DIR"
  mkdir -p "$BUDDY_BUILD_DIR/bin"
  for tool in "${BUDDY_RUNTIME_TOOLS[@]}"; do
    mv "$buddy_stage_dir/$tool" "$BUDDY_BUILD_DIR/bin/$tool"
  done
  rmdir "$buddy_stage_dir"

  # Buddy is the last consumer of the LLVM build tree. LLVM source remains for
  # Stage 1 runtime compilation of MLIR ExecutionEngine support sources.
  rm -rf "$LLVM_BUILD_DIR"
fi

for tool in "${BUDDY_RUNTIME_TOOLS[@]}"; do
  "$BUDDY_BUILD_DIR/bin/$tool" --version >/dev/null
done
"$TRITON_CHIPYARD_OPT_PATH" --help >/dev/null
cleanup_download_caches
conda clean -afy
disk_report "final compact compiler image state"
ok "installed compact PyTorch-Chipyard compiler stack"

#!/usr/bin/env bash
# Root workspace
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
# Buddy-MLIR
export BUDDY_BINARY_DIR="$WORKSPACE/buddy-mlir/build/bin"
# triton-chipyard-opt path
export TRITON_CHIPYARD_OPT_PATH="$WORKSPACE/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt"
# LLVM project
export LLVM_PROJECT_PATH="$WORKSPACE/llvm-project"
# Verilator: triton-chipyard passes simulation path, only does compilation
export CHIPYARD_SIM_VERILATOR_PATH="" 

# TorchInductor settings
# export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
# export TORCHINDUCTOR_MAX_AUTOTUNE=1 
# export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON 
# export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON 
# export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1

# Gemmini Usage env vars
# export TRITON_CHIPYARD_USE_GEMMINI=1
# export TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE=0 # if set, it tries multiple gemmini's tilings, which consumes very long simulation time
# FP32 Gemmini Configs example
# export TRITON_CHIPYARD_GEMMINI_ADDR_LEN="32"
# export TRITON_CHIPYARD_GEMMINI_DIM="8"
# export TRITON_CHIPYARD_GEMMINI_BANK_ROWS="2048"
# export TRITON_CHIPYARD_GEMMINI_ACC_ROWS="2048"
# export TRITON_CHIPYARD_GEMMINI_ELEM_T="f32"
# export TRITON_CHIPYARD_GEMMINI_ACC_T="f32"
# export TRITON_CHIPYARD_RISCV_MARCH="rv64imafdc"
# export TRITON_CHIPYARD_RISCV_MABI="lp64d"

# RVV Usage env vars: please uncomment below lines, comment above gemmini settings, re 'source env.sh'
# export TRITON_CHIPYARD_USE_RVV=1
# export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
# export TRITON_CHIPYARD_RISCV_MABI=lp64d
# export TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64

# Local Chipyard/FireSim host paths.
#
# For an external host setup, export CHIPYARD_DIR before sourcing this file.
# The repository no longer carries a local chipyard/ checkout, so there is no
# $WORKSPACE/chipyard fallback. Override derived variables only when the local
# FireMarshal/FireSim layout differs from the standard Chipyard tree.
export CHIPYARD_DIR="${CHIPYARD_DIR:-}"

_pytorch_chipyard_env_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/env.sh}"
_pytorch_chipyard_firemarshal_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/software/firemarshal}"
_pytorch_chipyard_firesim_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/sims/firesim}"

export CHIPYARD_ENV_PATH="${CHIPYARD_ENV_PATH:-${_pytorch_chipyard_env_default}}"
export FIREMARSHAL_DIR="${FIREMARSHAL_DIR:-${_pytorch_chipyard_firemarshal_default}}"
export FIREMARSHAL_CONFIG_PATH="${FIREMARSHAL_CONFIG_PATH:-${FIREMARSHAL_DIR:+${FIREMARSHAL_DIR}/marshal-config.yaml}}"
#jseo: Keep FireMarshal's real image output and the consumer path in sync.
export MARSHAL_IMAGE_DIR="${MARSHAL_IMAGE_DIR:-${FIREMARSHAL_DIR:+${FIREMARSHAL_DIR}/images}}"
export FIREMARSHAL_IMAGE_DIR="${FIREMARSHAL_IMAGE_DIR:-${MARSHAL_IMAGE_DIR:+${MARSHAL_IMAGE_DIR}/firechip}}"
export FIRESIM_DIR="${FIRESIM_DIR:-${_pytorch_chipyard_firesim_default}}"
export FIRESIM_DEPLOY_DIR="${FIRESIM_DEPLOY_DIR:-${FIRESIM_DIR:+${FIRESIM_DIR}/deploy}}"
export FIRESIM_HWDB_PATH="${FIRESIM_HWDB_PATH:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/config_hwdb.yaml}}"
export FIRESIM_BUILD_RECIPES_PATH="${FIRESIM_BUILD_RECIPES_PATH:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/config_build_recipes.yaml}}"
export FIRESIM_WORKLOAD_DIR="${FIRESIM_WORKLOAD_DIR:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/workloads}}"
export PYTORCH_CHIPYARD_WORKLOAD_DIR="${PYTORCH_CHIPYARD_WORKLOAD_DIR:-${FIREMARSHAL_DIR:+${FIREMARSHAL_DIR}/custom_application/pytorch-chipyard-workloads}}"
export PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR="${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/results-workload}}"
export PYTORCH_CHIPYARD_FPGA_DB="${PYTORCH_CHIPYARD_FPGA_DB:-/opt/firesim-db.json}"
export PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR="${PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR:-$WORKSPACE/.firemarshal/tmp}"
export FIRESIM_RUNS_DIR="${FIRESIM_RUNS_DIR:-$WORKSPACE/FIRESIM_RUNS_DIR}"
export PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR:-$WORKSPACE/.firesim-runtime}"
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST="${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST:-localhost}"
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC="${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC:-one_fpgas_spec}"
export PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR:-$WORKSPACE/scripts/figures/results-workload}"
export PYTORCH_CHIPYARD_LOG_DIR="${PYTORCH_CHIPYARD_LOG_DIR:-$WORKSPACE/examples/.logs}"
export PYTORCH_CHIPYARD_CONDA_ENV="${PYTORCH_CHIPYARD_CONDA_ENV:-pytorch-chipyard}"

# Optional RISC-V compiler override for model.elf builds. Set this when the
# Chipyard checkout used for FPGA setup should stay fixed, but model.elf should
# be built with a separate riscv64-unknown-linux-gnu-g++ toolchain.
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR="${PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR:-}"
export PYTORCH_CHIPYARD_RISCV_GXX="${PYTORCH_CHIPYARD_RISCV_GXX:-}"

# FireMarshal reads MARSHAL_* environment variables as config overrides.
#jseo: Install workload JSONs into the selected per-user deploy tree.
export MARSHAL_FIRESIM_DIR="${MARSHAL_FIRESIM_DIR:-${FIRESIM_DEPLOY_DIR:+$(dirname -- "${FIRESIM_DEPLOY_DIR}")}}"
export MARSHAL_MOUNT_DIR="${MARSHAL_MOUNT_DIR:-$WORKSPACE/.firemarshal/disk-mount}"

# Writable paths for FireMarshal's unprivileged guestmount fallback.
export LIBGUESTFS_CACHEDIR="${LIBGUESTFS_CACHEDIR:-$WORKSPACE/.firemarshal/libguestfs-cache}"
export LIBGUESTFS_TMPDIR="${LIBGUESTFS_TMPDIR:-$WORKSPACE/.firemarshal/libguestfs-tmp}"

#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export WORKSPACE="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
export PYTORCH_CHIPYARD_BITSTREAM_DIR="${PYTORCH_CHIPYARD_BITSTREAM_DIR:-$WORKSPACE/bitstreams}"
export BUDDY_BINARY_DIR="$WORKSPACE/buddy-mlir/build/bin"
export TRITON_CHIPYARD_OPT_PATH="$WORKSPACE/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_chipyard/tools/triton-chipyard-opt/triton-chipyard-opt"
export LLVM_PROJECT_PATH="$WORKSPACE/llvm-project"
export CHIPYARD_SIM_VERILATOR_PATH="" 
# export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
# export TORCHINDUCTOR_MAX_AUTOTUNE=1 
# export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON 
# export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=TRITON 
# export TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1
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
# export TRITON_CHIPYARD_USE_RVV=1
# export TRITON_CHIPYARD_RISCV_MARCH=rv64imafdcv_zicsr_zifencei_zvl128b
# export TRITON_CHIPYARD_RISCV_MABI=lp64d
# export TRITON_CHIPYARD_RISCV_VARCH=vlen:128,elen:64
export CHIPYARD_DIR="${CHIPYARD_DIR:-}"
export PYTORCH_CHIPYARD_STATE_DIR="${PYTORCH_CHIPYARD_STATE_DIR:-$HOME/.local/share/pytorch-chipyard}"
_pytorch_chipyard_env_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/env.sh}"
_pytorch_chipyard_firemarshal_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/software/firemarshal}"
_pytorch_chipyard_firesim_default="${CHIPYARD_DIR:+${CHIPYARD_DIR}/sims/firesim}"
export CHIPYARD_ENV_PATH="${CHIPYARD_ENV_PATH:-${_pytorch_chipyard_env_default}}"
export FIREMARSHAL_DIR="${FIREMARSHAL_DIR:-${_pytorch_chipyard_firemarshal_default}}"
export FIREMARSHAL_CONFIG_PATH="${FIREMARSHAL_CONFIG_PATH:-${FIREMARSHAL_DIR:+${FIREMARSHAL_DIR}/marshal-config.yaml}}"
export MARSHAL_IMAGE_DIR="${MARSHAL_IMAGE_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/images}"
export FIREMARSHAL_IMAGE_DIR="${FIREMARSHAL_IMAGE_DIR:-${MARSHAL_IMAGE_DIR:+${MARSHAL_IMAGE_DIR}/firechip}}"
export FIRESIM_DIR="${FIRESIM_DIR:-${_pytorch_chipyard_firesim_default}}"
export FIRESIM_DEPLOY_DIR="${FIRESIM_DEPLOY_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firesim/deploy}"
# Leave this empty to materialize the bundled artifact HWDB at runtime. Set an
# explicit path only when intentionally using an external FireSim HWDB.
export FIRESIM_HWDB_PATH="${FIRESIM_HWDB_PATH:-}"
# Build recipes and run-farm recipes are immutable inputs from the shared
# FireSim source. Only deploy outputs belong in the per-account state tree.
export FIRESIM_BUILD_RECIPES_PATH="${FIRESIM_BUILD_RECIPES_PATH:-${FIRESIM_DIR:+${FIRESIM_DIR}/deploy/config_build_recipes.yaml}}"
export FIRESIM_WORKLOAD_DIR="${FIRESIM_WORKLOAD_DIR:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/workloads}}"
export PYTORCH_CHIPYARD_WORKLOAD_DIR="${PYTORCH_CHIPYARD_WORKLOAD_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/workloads}"
export PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR="${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR:-${FIRESIM_DEPLOY_DIR:+${FIRESIM_DEPLOY_DIR}/results-workload}}"
export PYTORCH_CHIPYARD_FPGA_DB="${PYTORCH_CHIPYARD_FPGA_DB:-/opt/firesim-db.json}"
export PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR="${PYTORCH_CHIPYARD_FIREMARSHAL_TMP_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/tmp}"
export MARSHAL_LOG_DIR="${MARSHAL_LOG_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/logs}"
export MARSHAL_RES_DIR="${MARSHAL_RES_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/run-output}"
export FIRESIM_RUNS_DIR="${FIRESIM_RUNS_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firesim-runs}"
export PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR="${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firesim-runtime}"
export PYTORCH_CHIPYARD_GENERATED_HWDB_PATH="${PYTORCH_CHIPYARD_GENERATED_HWDB_PATH:-$PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR/config_hwdb.yaml}"
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST="${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_HOST:-localhost}"
export PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC="${PYTORCH_CHIPYARD_FIRESIM_RUN_FARM_SPEC:-one_fpgas_spec}"
_pytorch_chipyard_results_default="$WORKSPACE/scripts/figures/results-workload"
_pytorch_chipyard_results_requested="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR:-${_pytorch_chipyard_results_default}}"
_pytorch_chipyard_results_requested="$(readlink -m -- "${_pytorch_chipyard_results_requested}")"
case "${_pytorch_chipyard_results_requested}" in
  "$WORKSPACE"/*)
    export PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${_pytorch_chipyard_results_requested}"
    ;;
  *)
    # Result logs are durable outputs. Never allow a per-account host file to
    # redirect them into disposable ~/.local FireSim state.
    export PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR="${_pytorch_chipyard_results_default}"
    ;;
esac
_pytorch_chipyard_log_default="$WORKSPACE/examples/.logs"
_pytorch_chipyard_log_requested="${PYTORCH_CHIPYARD_LOG_DIR:-${_pytorch_chipyard_log_default}}"
_pytorch_chipyard_log_requested="$(readlink -m -- "${_pytorch_chipyard_log_requested}")"
case "${_pytorch_chipyard_log_requested}" in
  "$WORKSPACE"/*)
    export PYTORCH_CHIPYARD_LOG_DIR="${_pytorch_chipyard_log_requested}"
    ;;
  *)
    export PYTORCH_CHIPYARD_LOG_DIR="${_pytorch_chipyard_log_default}"
    ;;
esac
export PYTORCH_CHIPYARD_CONDA_ENV="${PYTORCH_CHIPYARD_CONDA_ENV:-pytorch-chipyard}"
export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR="${PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR:-}"
export PYTORCH_CHIPYARD_RISCV_GXX="${PYTORCH_CHIPYARD_RISCV_GXX:-}"
export MARSHAL_FIRESIM_DIR="${MARSHAL_FIRESIM_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firesim}"
export MARSHAL_MOUNT_DIR="${MARSHAL_MOUNT_DIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/disk-mount}"
export LIBGUESTFS_CACHEDIR="${LIBGUESTFS_CACHEDIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/libguestfs-cache}"
export LIBGUESTFS_TMPDIR="${LIBGUESTFS_TMPDIR:-$PYTORCH_CHIPYARD_STATE_DIR/firemarshal/libguestfs-tmp}"

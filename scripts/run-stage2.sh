#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage2-common.sh"

log() {
  printf '[stage2] %s\n' "$*"
}

die() {
  printf '[stage2][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-stage2.sh [options]

Run the Stage 2 host workflow in order:
  1. Build model-<N>core.elf files from Stage 1 artifacts.
  2. Package FireMarshal and FireSim workload files.
  3. Build/install one pending FireMarshal image at a time.
  4. Run that FireSim workload, collect results, and remove its image.
  5. Complete Table 4's sampled-kernel FireSim and Verilator measurements
     when running the complete, non-selective workflow.

By default, workloads with durable model.log and autotune.log files are
skipped. Table 4 also requires its PASS row containing compile/runtime timing.

Options:
  --experiment=NAME     Run one resumable paper experiment unit. NAME is one of:
                          figures-6-7-8-9-table5
                          figure-10
                          figure-11
                          figure-13
                          table-4
                          simple
                        Completed ordinary workloads are detected under the
                        collected results directory and skipped automatically.
  --riscv-toolchain-dir=PATH
                        RISC-V toolchain root or bin dir containing
                        riscv64-unknown-linux-gnu-g++.
  --riscv-gxx=PATH      Exact riscv64-unknown-linux-gnu-g++ path.
  --workload=LIST       Run selected FireSim workload(s). May be repeated.
  --only-alias-first    Build the Figure 9 ablation artifacts and limit
                        image generation and FireSim execution to 4 CNN
                        Gemmini 4-core off workloads and 8 LLM seq=256 SDPA
                        Gemmini 4-core on/off workloads.
  --only-alias-first-cnn-off
                        Build and run only the four CNN Gemmini 4-core
                        alias-first off workloads.
  --resume, --resume-firesim
                        Keep collected results and rebuild only missing steps
                        for incomplete workloads, one workload at a time.
  --resume-rebuild-images
                        Rebuild images only for workloads without complete
                        outputs, then run only those FireSim workloads.
  --resume-from=NAME    Keep collected results and restart incomplete workload
                        processing at NAME.
  --rerun-completed     Run completed FireSim workloads again instead of
                        applying the default completion filter.
  --rvv-panic-retries=N|unlimited
                        Retry an RVV workload after a detected guest kernel
                        panic. Default: 0; unlimited must be explicit.
  --skip-elves          Skip model-<N>core.elf generation.
  --skip-package        Skip FireMarshal workload/package generation.
  --skip-images         Use existing FireMarshal images instead of building
                        them. A missing selected image is a fatal error.
  --skip-firesim        Skip FireSim execution/collection.
  --skip-table4         Skip the sampled-kernel Table 4 workflow.
  --only-table4         Skip ordinary Stage 2 workloads and complete only the
                        Table 4 run started by Docker Stage 1.
  --table4-kernels=LIST Limit Table 4 to selected built-in kernel IDs.
                        Default: all three.
  --table4-repeats=N    Must match the Docker Stage 1 trial count. Default: 1.
  -h, --help            Show this help.

Environment:
  The script automatically sources PYTORCH_CHIPYARD_ACCOUNT_ENV (or the legacy
  TABLE4_ACCOUNT_ENV) when it exists. It defaults to ~/.ae-env.sh.
  CHIPYARD_DIR must point to the local Chipyard checkout. scripts/env.sh
  derives CHIPYARD_ENV_PATH as $CHIPYARD_DIR/env.sh.
  Existing model.log/autotune.log pairs in the repository are always preserved.
  FireMarshal images and other runtime artifacts are always removed on exit.
  PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR must remain 1 for Stage 2 so copied
  rootfs images cannot accumulate between workloads.
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1 selects the quick partial Figure
  6/8/9/11/13 experiment. Prefer scripts/simple-stage2.sh to set it.
  TABLE4_TVM_AE_ROOT points to the prepared TVM-Gemmini tree. It defaults to
  ~/tvm-gemmini-ae.
  TABLE4_VERILATOR_BIN must point to a prebuilt executable for Table 4. Stage 2
  never builds or modifies the Verilator simulator.
EOF
}

chipyard_env_arg=""
riscv_toolchain_dir_arg=""
riscv_gxx_arg=""
workload_args=()
experiment_arg=""
experiment_name=""
experiment_selected=0
experiment_artifact_dirs=()
experiment_workloads=()
experiment_pending_workloads=()
only_alias_first=0
only_alias_first_cnn_off=0
resume_firesim=0
rerun_completed=0
rebuild_pending_images=0
resume_from_arg=""
rvv_panic_retries_arg=""
skip_elves=0
skip_package=0
skip_images=0
skip_firesim=0
skip_table4=0
only_table4=0
table4_kernels="squeezenet_fire2_squeeze,resnet50_classifier,mobilenetv2_classifier"
table4_repeats="${TABLE4_REPEATS:-1}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --chipyard-env)
      [[ "$#" -ge 2 ]] || die "--chipyard-env requires a value"
      log "using deprecated --chipyard-env; prefer CHIPYARD_DIR with optional --riscv-toolchain-dir"
      chipyard_env_arg="$2"
      shift 2
      ;;
    --chipyard-env=*)
      log "using deprecated --chipyard-env; prefer CHIPYARD_DIR with optional --riscv-toolchain-dir"
      chipyard_env_arg="${1#--chipyard-env=}"
      shift
      ;;
    --riscv-toolchain-dir)
      [[ "$#" -ge 2 ]] || die "--riscv-toolchain-dir requires a value"
      riscv_toolchain_dir_arg="$2"
      shift 2
      ;;
    --riscv-toolchain-dir=*)
      riscv_toolchain_dir_arg="${1#--riscv-toolchain-dir=}"
      shift
      ;;
    --riscv-gxx)
      [[ "$#" -ge 2 ]] || die "--riscv-gxx requires a value"
      riscv_gxx_arg="$2"
      shift 2
      ;;
    --riscv-gxx=*)
      riscv_gxx_arg="${1#--riscv-gxx=}"
      shift
      ;;
    --workload)
      [[ "$#" -ge 2 ]] || die "--workload requires a value"
      workload_args+=("--workload=$2")
      shift 2
      ;;
    --workload=*)
      workload_args+=("$1")
      shift
      ;;
    --experiment)
      [[ "$#" -ge 2 ]] || die "--experiment requires a value"
      experiment_arg="$2"
      shift 2
      ;;
    --experiment=*)
      experiment_arg="${1#--experiment=}"
      shift
      ;;
    --only-alias-first | --only-alias-first-ablation)
      only_alias_first=1
      shift
      ;;
    --only-alias-first-cnn-off)
      only_alias_first_cnn_off=1
      shift
      ;;
    --resume | --resume-firesim)
      resume_firesim=1
      skip_images=0
      shift
      ;;
    --resume-rebuild-images)
      resume_firesim=1
      rebuild_pending_images=1
      skip_images=0
      shift
      ;;
    --resume-from)
      [[ "$#" -ge 2 ]] || die "--resume-from requires a value"
      resume_from_arg="$2"
      skip_images=0
      shift 2
      ;;
    --resume-from=*)
      resume_from_arg="${1#--resume-from=}"
      skip_images=0
      shift
      ;;
    --rerun-completed)
      rerun_completed=1
      shift
      ;;
    --rvv-panic-retries)
      [[ "$#" -ge 2 ]] || die "--rvv-panic-retries requires a value"
      rvv_panic_retries_arg="$2"
      shift 2
      ;;
    --rvv-panic-retries=*)
      rvv_panic_retries_arg="${1#--rvv-panic-retries=}"
      shift
      ;;
    --skip-elves)
      skip_elves=1
      shift
      ;;
    --skip-package)
      skip_package=1
      shift
      ;;
    --skip-images)
      skip_images=1
      shift
      ;;
    --skip-firesim)
      skip_firesim=1
      shift
      ;;
    --skip-table4)
      skip_table4=1
      shift
      ;;
    --only-table4)
      only_table4=1
      shift
      ;;
    --table4-kernels=*)
      table4_kernels="${1#*=}"
      shift
      ;;
    --table4-kernels)
      [[ "$#" -ge 2 ]] || die "--table4-kernels requires a value"
      table4_kernels="$2"
      shift 2
      ;;
    --table4-repeats=*)
      table4_repeats="${1#*=}"
      shift
      ;;
    --table4-repeats)
      [[ "$#" -ge 2 ]] || die "--table4-repeats requires a value"
      table4_repeats="$2"
      shift 2
      ;;
    --skip-plot)
      log "ignoring deprecated --skip-plot; plotting now lives in scripts/run-plot.sh"
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

if [[ "${rerun_completed}" -eq 1 && "${resume_firesim}" -eq 1 ]]; then
  die "--rerun-completed cannot be combined with --resume or --resume-rebuild-images"
fi

if [[ -n "${experiment_arg}" ]]; then
  if [[ "${#workload_args[@]}" -gt 0 || "${only_alias_first}" -eq 1 || \
        "${only_alias_first_cnn_off}" -eq 1 || "${only_table4}" -eq 1 || \
        "${resume_firesim}" -eq 1 || -n "${resume_from_arg}" ]]; then
    die "--experiment cannot be combined with another workload, resume, or only-selection option"
  fi

  case "${experiment_arg}" in
    figures-6-7-8-9-table5 | figures-7-8-9-table5 | fig6-7-8-9-table5 | fig7-8-9-table5 | fig7-9-table5)
      experiment_name="figures-6-7-8-9-table5"
      ;;
    figure-10 | fig10)
      experiment_name="figure-10"
      ;;
    figure-11 | fig11)
      experiment_name="figure-11"
      ;;
    figure-13 | fig13)
      experiment_name="figure-13"
      ;;
    table-4 | table4)
      experiment_name="table-4"
      ;;
    simple | simple-stage2)
      experiment_name="simple"
      export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
      ;;
    *)
      die "unknown experiment '${experiment_arg}'; expected figures-6-7-8-9-table5, figure-10, figure-11, figure-13, table-4, or simple"
      ;;
  esac

  if [[ "${experiment_name}" == "table-4" ]]; then
    only_table4=1
  else
    experiment_selected=1
    if [[ "${rerun_completed}" -eq 0 ]]; then
      resume_firesim=1
    fi
    skip_table4=1

    case "${experiment_name}" in
      figures-6-7-8-9-table5)
        for model in alexnet mobilenetv2 resnet50 squeezenet; do
          for backend in gemmini rvv scalar; do
            experiment_artifact_dirs+=(
              "${REPO_ROOT}/examples/artifact-${model}/${backend}"
            )
            case "${backend}" in
              gemmini | rvv) cores=(2 4) ;;
              scalar) cores=(4 8 16) ;;
            esac
            for core in "${cores[@]}"; do
              experiment_workloads+=("${model}-${backend}-${core}core")
            done
          done

          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini-alias-first-off"
          )
          experiment_workloads+=(
            "${model}-gemmini-alias-first-off-4core"
          )
        done

        for model in gpt2 gpt-neo opt pythia; do
          for mode in on off; do
            experiment_artifact_dirs+=(
              "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256/alias-first-${mode}"
            )
            experiment_workloads+=(
              "${model}-gemmini-sdpa-256tok-alias-first-${mode}-4core"
            )
          done
        done

        for model in gpt2 gpt-neo; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini"
          )
          for core in 2 4; do
            experiment_workloads+=("${model}-gemmini-${core}core")
          done
        done
        for model in opt pythia; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256"
          )
          for core in 2 4; do
            experiment_workloads+=(
              "${model}-rocket-gemmini-sdpa-256tok-${core}core"
            )
          done
        done
        ;;
      figure-10)
        # Figure 10 compares im2col against the same four-core direct Gemmini
        # runs already used by Figures 7--9.
        for model in alexnet mobilenetv2 resnet50 squeezenet; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-${model}/gemmini"
            "${REPO_ROOT}/examples/artifact-${model}/gemmini-im2col"
          )
          experiment_workloads+=(
            "${model}-gemmini-4core"
            "${model}-gemmini-im2col-4core"
          )
        done
        ;;
      figure-11)
        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-gemmini-max-autotune/gemmini"
        )
        experiment_workloads+=("gemmini-max-autotune-gemmini-4core")
        ;;
      figure-13)
        # Figure 13 reuses the OPT/Pythia seq=256 SDPA runs from Table 5 and
        # adds the remaining SDPA, FlashAttention, and windowed-attention runs.
        for model in opt pythia; do
          for attention in sdpa flash window; do
            for seq_len in 256 512 768 1024; do
              experiment_artifact_dirs+=(
                "${REPO_ROOT}/examples/artifact-${model}/gemmini/${attention}/seq${seq_len}"
              )
              experiment_workloads+=(
                "${model}-rocket-gemmini-${attention}-${seq_len}tok-4core"
              )
              if [[ "${model}" == "opt" && "${attention}" != "sdpa" ]]; then
                experiment_workloads+=(
                  "${model}-boom-gemmini-${attention}-${seq_len}tok-1core"
                )
              fi
            done
          done
        done
        ;;
      simple)
        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-squeezenet/scalar"
          "${REPO_ROOT}/examples/artifact-squeezenet/rvv"
          "${REPO_ROOT}/examples/artifact-squeezenet/gemmini"
          "${REPO_ROOT}/examples/artifact-squeezenet/gemmini-alias-first-off"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256/alias-first-on"
          "${REPO_ROOT}/examples/artifact-opt/gemmini/sdpa/seq256/alias-first-off"
        )
        experiment_workloads+=(
          "squeezenet-scalar-4core"
          "squeezenet-rvv-2core"
          "squeezenet-gemmini-2core"
          "squeezenet-gemmini-4core"
          "squeezenet-gemmini-alias-first-off-4core"
          "opt-rocket-gemmini-sdpa-256tok-2core"
          "opt-rocket-gemmini-sdpa-256tok-4core"
          "opt-gemmini-sdpa-256tok-alias-first-on-4core"
          "opt-gemmini-sdpa-256tok-alias-first-off-4core"
        )

        for attention in flash window; do
          experiment_artifact_dirs+=(
            "${REPO_ROOT}/examples/artifact-opt/gemmini/${attention}/seq256"
          )
          experiment_workloads+=(
            "opt-rocket-gemmini-${attention}-256tok-4core"
            "opt-boom-gemmini-${attention}-256tok-1core"
          )
        done

        experiment_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-gemmini-max-autotune-simple/gemmini"
        )
        experiment_workloads+=(
          "gemmini-max-autotune-simple-gemmini-4core"
        )
        ;;
    esac
    log "selected ${experiment_name}: ${#experiment_workloads[@]} workload(s)"
  fi
fi

alias_first_workloads=()
alias_first_artifact_dirs=()
alias_first_selected=0
if [[ "${only_alias_first}" -eq 1 && "${only_alias_first_cnn_off}" -eq 1 ]]; then
  die "--only-alias-first and --only-alias-first-cnn-off are mutually exclusive"
fi
if [[ "${only_table4}" -eq 1 && "${skip_table4}" -eq 1 ]]; then
  die "--only-table4 cannot be combined with --skip-table4"
fi
if [[ "${only_table4}" -eq 1 && \
      ( "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 || \
        "${#workload_args[@]}" -gt 0 || "${resume_firesim}" -eq 1 || \
        -n "${resume_from_arg}" ) ]]; then
  die "--only-table4 cannot be combined with workload or ordinary FireSim selection options"
fi
[[ "${table4_repeats}" =~ ^[1-9][0-9]*$ ]] || \
  die "--table4-repeats must be a positive integer"
if [[ "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ]]; then
  alias_first_selected=1
  [[ "${#workload_args[@]}" -eq 0 ]] || \
    die "an alias-first only option cannot be combined with --workload"
  if [[ "${only_alias_first}" -eq 1 ]]; then
    for model in gpt2 gpt-neo opt pythia; do
      for mode in on off; do
        alias_first_workloads+=(
          "${model}-gemmini-sdpa-256tok-alias-first-${mode}-4core"
        )
        alias_first_artifact_dirs+=(
          "${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256/alias-first-${mode}"
        )
      done
    done
  fi
  for model in alexnet mobilenetv2 resnet50 squeezenet; do
    alias_first_workloads+=(
      "${model}-gemmini-alias-first-off-4core"
    )
    alias_first_artifact_dirs+=(
      "${REPO_ROOT}/examples/artifact-${model}/gemmini-alias-first-off"
    )
  done
  for workload in "${alias_first_workloads[@]}"; do
    workload_args+=("--workload=${workload}")
  done
  log "selected Figure 9 alias-first ablation: ${#alias_first_workloads[@]} workloads"
fi

# Table 4 is a complete cross-toolchain experiment, not per-workload
# post-processing. Do not unexpectedly launch it after a selective Stage 2 run
# or when the caller explicitly disabled FireSim.
if [[ "${#workload_args[@]}" -gt 0 || \
      ( "${skip_firesim}" -eq 1 && "${only_table4}" -eq 0 ) ]]; then
  if [[ "${skip_table4}" -eq 0 ]]; then
    log "skipping Table 4 after a selective or --skip-firesim Stage 2 run"
  fi
  skip_table4=1
fi

if [[ "${only_table4}" -eq 1 ]]; then
  skip_elves=1
  skip_package=1
  skip_images=1
  skip_firesim=1
fi

# Ordinary and selectively scoped Stage 2 runs are resumable by default.
# The FireSim runner performs the authoritative per-workload result check.
if [[ "${skip_firesim}" -eq 0 && "${rerun_completed}" -eq 0 ]]; then
  resume_firesim=1
fi

if [[ -n "${chipyard_env_arg}" ]]; then
  export CHIPYARD_ENV_PATH="${chipyard_env_arg}"
fi
if [[ -n "${riscv_toolchain_dir_arg}" ]]; then
  export PYTORCH_CHIPYARD_RISCV_TOOLCHAIN_DIR="${riscv_toolchain_dir_arg}"
fi
if [[ -n "${riscv_gxx_arg}" ]]; then
  export PYTORCH_CHIPYARD_RISCV_GXX="${riscv_gxx_arg}"
fi

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
  log "loaded host environment: ${account_env}"
fi

set +u
source "${SCRIPT_DIR}/env.sh"
set -u

if [[ "${skip_table4}" -eq 0 ]]; then
  [[ -n "${TABLE4_VERILATOR_BIN:-}" ]] || \
    die "TABLE4_VERILATOR_BIN is required when Table 4 is enabled"
  [[ -x "${TABLE4_VERILATOR_BIN}" ]] || \
    die "TABLE4_VERILATOR_BIN is not executable: ${TABLE4_VERILATOR_BIN}"
fi

materialize_account_state_dir() {
  local path="$1" label="$2" old_target=""
  case "${path}" in
    "${PYTORCH_CHIPYARD_STATE_DIR}" | "${PYTORCH_CHIPYARD_STATE_DIR}"/*) ;;
    *) die "${label} must be under PYTORCH_CHIPYARD_STATE_DIR: ${path}" ;;
  esac
  if [[ -L "${path}" ]]; then
    old_target="$(readlink -f -- "${path}" 2>/dev/null || true)"
    rm -f -- "${path}"
    log "replaced legacy ${label} symlink with account-local state directory: ${path}"
    case "${old_target}" in
      "${REPO_ROOT}/.firesim-runtime" | "${REPO_ROOT}/FIRESIM_RUNS_DIR")
        rm -rf -- "${old_target}"
        log "removed legacy checkout-local runtime target: ${old_target}"
        ;;
    esac
  fi
  [[ ! -e "${path}" || -d "${path}" ]] || \
    die "${label} is not a directory: ${path}"
  mkdir -p "${path}"
}

materialize_account_state_dir "${FIRESIM_RUNS_DIR}" FIRESIM_RUNS_DIR
materialize_account_state_dir \
  "${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR}" \
  PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR

mkdir -p \
  "${PYTORCH_CHIPYARD_STATE_DIR}" \
  "${FIRESIM_DEPLOY_DIR}" \
  "${FIRESIM_WORKLOAD_DIR}" \
  "${PYTORCH_CHIPYARD_RESULTS_WORKLOAD_DIR}" \
  "${FIRESIM_RUNS_DIR}" \
  "${PYTORCH_CHIPYARD_FIRESIM_RUNTIME_DIR}" \
  "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
  "${FIREMARSHAL_IMAGE_DIR}"

stage2_active_runtime_workloads=()

stage2_cleanup_on_exit() {
  local status=$?
  local cleanup_status=0

  trap - EXIT
  local workload
  local -a cleanup_args=(--all --final)

  for workload in "${stage2_active_runtime_workloads[@]}"; do
    cleanup_args+=(--workload "${workload}")
  done
  bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" "${cleanup_args[@]}" || cleanup_status=$?
  if [[ "${cleanup_status}" -ne 0 ]]; then
    printf '[stage2][warn] final runtime cleanup failed with status %s\n' \
      "${cleanup_status}" >&2
    [[ "${status}" -ne 0 ]] || status="${cleanup_status}"
  fi
  exit "${status}"
}
trap stage2_cleanup_on_exit EXIT

if [[ "${skip_firesim}" -eq 0 ]]; then
  stage2_clean_firesim_runs="${PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR:-1}"
  [[ "${stage2_clean_firesim_runs}" == "0" || "${stage2_clean_firesim_runs}" == "1" ]] || \
    die "PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR must be 0 or 1"
  [[ "${stage2_clean_firesim_runs}" == "1" ]] || \
    die "Stage 2 requires PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR=1; disabling it accumulates 8GiB FireSim rootfs copies"
  export PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR=1
fi

if [[ "${experiment_name}" == "simple" ]]; then
  export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
fi

if [[ "${skip_firesim}" -eq 0 && "${resume_firesim}" -eq 1 ]]; then
  log "FireSim auto-resume enabled; completed workloads under ${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR} will be skipped"
fi

cd "${REPO_ROOT}"

collected_result_complete() {
  local workload="$1"
  pc_stage2_result_complete \
    "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}" "${workload}"
}

declare -A STAGE2_ARTIFACT_BY_WORKLOAD=()
declare -A STAGE2_CORE_BY_WORKLOAD=()

require_installed_firemarshal_image() {
  local workload="$1"
  local image_dir="${FIREMARSHAL_IMAGE_DIR}/${workload}"

  [[ -s "${image_dir}/${workload}.img" ]] || \
    die "missing FireMarshal image for ${workload}: ${image_dir}/${workload}.img; rerun without --skip-images"
  [[ -s "${image_dir}/${workload}-bin" ]] || \
    die "missing FireMarshal boot binary for ${workload}: ${image_dir}/${workload}-bin; rerun without --skip-images"
}

reusable_completed_workload() {
  local workload="$1"

  if collected_result_complete "${workload}"; then
    printf '%s\n' "${workload}"
    return 0
  fi
  if [[ "${workload}" == "gemmini-max-autotune-simple-gemmini-4core" ]] && \
      collected_result_complete "gemmini-max-autotune-gemmini-4core"; then
    printf '%s\n' "gemmini-max-autotune-gemmini-4core"
    return 0
  fi
  return 1
}

artifact_dir_for_workload() {
  local workload="$1"

  if [[ "${workload}" =~ ^(alexnet|mobilenetv2|resnet50|squeezenet)-(gemmini|rvv|scalar)-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(alexnet|mobilenetv2|resnet50|squeezenet)-gemmini-(alias-first-off|im2col)-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini-${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(gpt2|gpt-neo)-gemmini-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini"
  elif [[ "${workload}" =~ ^(gpt2|gpt-neo|opt|pythia)-gemmini-sdpa-256tok-alias-first-(on|off)-4core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini/sdpa/seq256/alias-first-${BASH_REMATCH[2]}"
  elif [[ "${workload}" =~ ^(opt|pythia)-(rocket|boom)-gemmini-(sdpa|flash|window)-([0-9]+)tok-[0-9]+core$ ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-${BASH_REMATCH[1]}/gemmini/${BASH_REMATCH[3]}/seq${BASH_REMATCH[4]}"
  elif [[ "${workload}" == "gemmini-max-autotune-gemmini-4core" ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-gemmini-max-autotune/gemmini"
  elif [[ "${workload}" == "gemmini-max-autotune-simple-gemmini-4core" ]]; then
    printf '%s\n' "${REPO_ROOT}/examples/artifact-gemmini-max-autotune-simple/gemmini"
  else
    die "no artifact mapping for experiment workload ${workload}"
  fi
}

remove_completed_workload_elf() {
  local workload="$1"
  local artifact_dir core
  local removed=0
  local path

  collected_result_complete "${workload}" || return 0
  if [[ ! "${workload}" =~ -([0-9]+)core$ ]]; then
    log "${workload}: cannot infer core count; preserving ELF"
    return 0
  fi
  core="${BASH_REMATCH[1]}"
  artifact_dir="${STAGE2_ARTIFACT_BY_WORKLOAD[${workload}]:-}"
  if [[ -z "${artifact_dir}" ]]; then
    artifact_dir="$(artifact_dir_for_workload "${workload}")"
  fi
  [[ -d "${artifact_dir}" ]] || return 0

  for path in \
    "${artifact_dir}/model-${core}core.elf" \
    "${artifact_dir}/.model-${core}core.elf-fingerprint" \
    "${artifact_dir}/model.elf"; do
    if [[ -e "${path}" ]]; then
      rm -f -- "${path}"
      removed=$((removed + 1))
    fi
  done
  if [[ "${removed}" -gt 0 ]]; then
    log "${workload}: removed ${removed} completed-workload ELF derivative(s)"
  fi
}

if [[ "${experiment_selected}" -eq 1 ]]; then
  completed_workloads=0
  for workload in "${experiment_workloads[@]}"; do
    if [[ "${rerun_completed}" -eq 0 ]] && \
        reused_workload="$(reusable_completed_workload "${workload}")"; then
      completed_workloads=$((completed_workloads + 1))
      remove_completed_workload_elf "${reused_workload}"
      if [[ "${reused_workload}" == "${workload}" ]]; then
        log "${experiment_name}: skipping completed workload ${workload}"
      else
        log "${experiment_name}: reusing ${reused_workload} for ${workload}"
      fi
    else
      experiment_pending_workloads+=("${workload}")
      workload_args+=("--workload=${workload}")
    fi
  done

  log "${experiment_name}: ${completed_workloads} complete, ${#experiment_pending_workloads[@]} pending"
  if [[ "${#experiment_pending_workloads[@]}" -eq 0 ]]; then
    skip_elves=1
    skip_package=1
    skip_images=1
    skip_firesim=1
    log "${experiment_name}: every required workload is already complete"
  fi
fi

if [[ "${skip_table4}" -eq 0 ]]; then
  table4_script="${SCRIPT_DIR}/run_table4.sh"
  table4_results_tool="${SCRIPT_DIR}/table4_results.py"
  table4_tvm_ae_root="${TABLE4_TVM_AE_ROOT:-${HOME}/tvm-gemmini-ae}"
  table4_stage1_output="${TABLE4_STAGE1_OUTPUT_DIR:-${REPO_ROOT}/results/table4/stage1-latest}"
  [[ -f "${table4_script}" ]] || die "missing Table 4 runner: ${table4_script}"
  [[ -f "${table4_results_tool}" ]] || die "missing Table 4 results tool: ${table4_results_tool}"
  [[ -f "${table4_tvm_ae_root}/scripts/env.sh" ]] || \
    die "missing TVM-Gemmini environment: ${table4_tvm_ae_root}/scripts/env.sh; set TABLE4_TVM_AE_ROOT or pass --skip-table4"
  [[ -s "${table4_stage1_output}/raw.csv" ]] || \
    die "missing Docker Stage 1 Table 4 results: ${table4_stage1_output}/raw.csv; rerun Stage 1 without --skip-table4"
  [[ -w "${table4_stage1_output}" && -w "${table4_stage1_output}/raw.csv" ]] || \
    die "Docker Stage 1 Table 4 results are not writable by $(id -un): ${table4_stage1_output}; repair bind-mount ownership or permissions before Stage 2"
  IFS=',' read -r -a table4_kernel_list <<<"${table4_kernels}"
  for table4_kernel in "${table4_kernel_list[@]}"; do
    table4_kernel="${table4_kernel//[[:space:]]/}"
    [[ -n "${table4_kernel}" ]] || continue
    for ((table4_trial = 1; table4_trial <= table4_repeats; table4_trial++)); do
      table4_kernel_spec="${table4_stage1_output}/artifacts/pytorch-chipyard/${table4_kernel}/trial-${table4_trial}/gemmini/model_spec.json"
      [[ -s "${table4_kernel_spec}" ]] || \
        die "missing Docker Stage 1 Table 4 artifact: ${table4_kernel_spec}"
      table4_artifact_dir="$(dirname -- "${table4_kernel_spec}")"
      [[ -w "${table4_artifact_dir}" ]] || \
        die "Docker Stage 1 artifact directory is not writable by $(id -un): ${table4_artifact_dir}"
      for table4_blob in input.bin weights.bin; do
        [[ -r "${table4_artifact_dir}/${table4_blob}" ]] || \
          die "Docker Stage 1 artifact is not readable by $(id -un): ${table4_artifact_dir}/${table4_blob}; repair bind-mount ownership or permissions before Stage 2"
      done
    done
  done
fi

stage2_workload_core() {
  local workload="$1"
  [[ "${workload}" =~ -([0-9]+)core$ ]] || \
    die "cannot infer core count from workload ${workload}"
  printf '%s\n' "${BASH_REMATCH[1]}"
}

stage2_artifact_rel() {
  local artifact_dir="$1"
  local rel first rest hint="${artifact_dir}/.pytorch-chipyard-workload-rel"

  if [[ -s "${hint}" ]]; then
    rel="$(head -n 1 "${hint}")"
  else
    rel="${artifact_dir#"${REPO_ROOT}/examples/"}"
  fi
  rel="${rel#./}"
  first="${rel%%/*}"
  first="${first#artifact-}"
  if [[ "${rel}" == */* ]]; then
    rest="${rel#*/}"
    printf '%s/%s\n' "${first}" "${rest}"
  else
    printf '%s\n' "${first}"
  fi
}

stage2_derive_workload_name() {
  local artifact_dir="$1" core="$2"
  local rel model attention seq_len

  rel="$(stage2_artifact_rel "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    attention="${BASH_REMATCH[2]}"
    seq_len="${BASH_REMATCH[3]}"
    if [[ "${core}" == 1 ]]; then
      printf '%s-boom-gemmini-%s-%stok-%score\n' \
        "${model}" "${attention}" "${seq_len}" "${core}"
    else
      printf '%s-rocket-gemmini-%s-%stok-%score\n' \
        "${model}" "${attention}" "${seq_len}" "${core}"
    fi
    return
  fi

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

stage2_artifact_backend() {
  local artifact_dir="$1" rel part backend_file
  backend_file="${artifact_dir}/.pytorch-chipyard-backend"
  if [[ -s "${backend_file}" ]]; then
    head -n 1 "${backend_file}"
    return
  fi
  rel="$(stage2_artifact_rel "${artifact_dir}")"
  IFS=/ read -r -a rel_parts <<<"${rel}"
  for part in "${rel_parts[@]}"; do
    part="${part%-im2col}"
    case "${part}" in
      gemmini | rvv | scalar) printf '%s\n' "${part}"; return ;;
    esac
  done
  die "cannot infer backend for Stage 1 artifact ${artifact_dir}"
}

stage2_artifact_cores() {
  local artifact_dir="$1" backend core_file core
  core_file="${artifact_dir}/.pytorch-chipyard-build-cores"
  if [[ -s "${core_file}" ]]; then
    while IFS= read -r core; do
      [[ "${core}" =~ ^[1-9][0-9]*$ ]] || continue
      printf '%s\n' "${core}"
    done <"${core_file}"
    return
  fi
  backend="$(stage2_artifact_backend "${artifact_dir}")"
  case "${backend}" in
    gemmini | rvv) printf '%s\n' 2 4 ;;
    scalar) printf '%s\n' 4 8 16 ;;
    *) die "invalid backend '${backend}' in ${artifact_dir}" ;;
  esac
}

stage2_artifact_is_shadowed() {
  local artifact_dir="$1" rel model host attention seq_len canonical
  rel="$(stage2_artifact_rel "${artifact_dir}")"
  if [[ "${rel}" =~ ^(opt|pythia)/gemmini$ ]]; then
    model="${BASH_REMATCH[1]}"
    canonical="${REPO_ROOT}/examples/artifact-${model}/gemmini/sdpa/seq256"
    [[ -f "${canonical}/build.sh" ]]
    return
  fi
  if [[ "${rel}" =~ ^(opt|pythia)/(rocket|boom)-gemmini/(sdpa|flash|window)/seq([0-9]+)$ ]]; then
    model="${BASH_REMATCH[1]}"
    host="${BASH_REMATCH[2]}"
    attention="${BASH_REMATCH[3]}"
    seq_len="${BASH_REMATCH[4]}"
    canonical="${REPO_ROOT}/examples/artifact-${model}/gemmini/${attention}/seq${seq_len}"
    [[ -f "${canonical}/build.sh" ]] && {
      log "skipping shadowed ${host} artifact ${artifact_dir}"
      return 0
    }
  fi
  return 1
}

stage2_plan_workloads=()
stage2_append_plan() {
  local workload="$1" artifact_dir="$2" core="$3" existing
  pc_stage2_validate_workload_name "${workload}" || \
    die "invalid planned workload ${workload}"
  [[ -d "${artifact_dir}" ]] || die "Stage 1 artifact directory not found: ${artifact_dir}"
  [[ -s "${artifact_dir}/build.sh" ]] || die "Stage 1 build script not found: ${artifact_dir}/build.sh"
  for existing in "${stage2_plan_workloads[@]}"; do
    if [[ "${existing}" == "${workload}" ]]; then
      [[ "${STAGE2_ARTIFACT_BY_WORKLOAD[${workload}]}" == "${artifact_dir}" ]] || \
        die "duplicate workload ${workload} from two Stage 1 artifacts"
      return
    fi
  done
  stage2_plan_workloads+=("${workload}")
  STAGE2_ARTIFACT_BY_WORKLOAD["${workload}"]="${artifact_dir}"
  STAGE2_CORE_BY_WORKLOAD["${workload}"]="${core}"
}

stage2_add_active_runtime_workload() {
  local workload="$1" existing
  for existing in "${stage2_active_runtime_workloads[@]}"; do
    [[ "${existing}" == "${workload}" ]] && return
  done
  stage2_active_runtime_workloads+=("${workload}")
}

stage2_require_packaged_workload() {
  local workload="$1"
  [[ -s "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/${workload}.json" ]] || \
    die "missing FireMarshal package for ${workload}; rerun without --skip-package"
  [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload}" ]] || \
    die "missing FireMarshal overlay for ${workload}; rerun without --skip-package"
  [[ -s "${FIRESIM_WORKLOAD_DIR}/${workload}.json" ]] || \
    die "missing FireSim workload JSON for ${workload}; rerun without --skip-package"
}

# First migrate old result locations and reclaim leftovers for workloads that
# are already complete. This occurs before any new ELF or image is created.
bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" --all

if [[ "${only_table4}" -eq 0 && \
      ! ( "${experiment_selected}" -eq 1 && \
          "${#experiment_pending_workloads[@]}" -eq 0 ) ]]; then
  requested_workloads=()
  for arg in "${workload_args[@]}"; do
    IFS=, read -r -a requested_chunk <<<"${arg#--workload=}"
    requested_workloads+=("${requested_chunk[@]}")
  done

  if [[ "${#requested_workloads[@]}" -gt 0 ]]; then
    for workload in "${requested_workloads[@]}"; do
      core="$(stage2_workload_core "${workload}")"
      artifact_dir="$(artifact_dir_for_workload "${workload}")"
      stage2_append_plan "${workload}" "${artifact_dir}" "${core}"
    done
  else
    while IFS= read -r -d '' build_script; do
      artifact_dir="$(dirname -- "${build_script}")"
      if stage2_artifact_is_shadowed "${artifact_dir}"; then
        continue
      fi
      while IFS= read -r core; do
        workload="$(stage2_derive_workload_name "${artifact_dir}" "${core}")"
        stage2_append_plan "${workload}" "${artifact_dir}" "${core}"
      done < <(stage2_artifact_cores "${artifact_dir}")
    done < <(find "${REPO_ROOT}/examples" -type f -name build.sh -print0 | sort -z)
  fi

  [[ "${#stage2_plan_workloads[@]}" -gt 0 ]] || \
    die "no Stage 1 workload artifacts were selected or found"

  stage2_firesim_workloads=()
  found_resume_from=0
  [[ -n "${resume_from_arg}" ]] || found_resume_from=1
  completed_count=0
  for workload in "${stage2_plan_workloads[@]}"; do
    if [[ "${found_resume_from}" -eq 0 ]]; then
      if [[ "${workload}" == "${resume_from_arg}" ]]; then
        found_resume_from=1
      else
        continue
      fi
    fi
    if [[ "${rerun_completed}" -eq 0 ]] && collected_result_complete "${workload}"; then
      completed_count=$((completed_count + 1))
      log "resume: skipping completed workload ${workload}"
      remove_completed_workload_elf "${workload}"
      continue
    fi
    stage2_firesim_workloads+=("${workload}")
  done
  [[ "${found_resume_from}" -eq 1 ]] || \
    die "resume workload '${resume_from_arg}' was not found in the Stage 1 workload plan"
  log "workload plan: ${completed_count} complete, ${#stage2_firesim_workloads[@]} pending"

  if [[ "${skip_firesim}" -eq 0 && "${#stage2_firesim_workloads[@]}" -gt 0 ]]; then
    preflight_args=(--preflight-only)
    for workload in "${stage2_firesim_workloads[@]}"; do
      preflight_args+=(--workload="${workload}")
    done
    log "validating pending FireSim hardware artifacts"
    bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${preflight_args[@]}"
  fi

  build_elves_common_args=()
  [[ -z "${chipyard_env_arg}" ]] || build_elves_common_args+=(--chipyard-env="${chipyard_env_arg}")
  [[ -z "${riscv_toolchain_dir_arg}" ]] || build_elves_common_args+=(--riscv-toolchain-dir="${riscv_toolchain_dir_arg}")
  [[ -z "${riscv_gxx_arg}" ]] || build_elves_common_args+=(--riscv-gxx="${riscv_gxx_arg}")

  declare -A STAGE2_ELVES_READY_BY_ARTIFACT=()
  built_image_count=0
  for workload in "${stage2_firesim_workloads[@]}"; do
    artifact_dir="${STAGE2_ARTIFACT_BY_WORKLOAD[${workload}]}"
    core="${STAGE2_CORE_BY_WORKLOAD[${workload}]}"

    if [[ -z "${STAGE2_ELVES_READY_BY_ARTIFACT[${artifact_dir}]:-}" ]]; then
      artifact_cores=()
      # Build the complete Stage 1-declared core set before deleting .obj files.
      # A partial run may select only one core today, but a later experiment
      # must still be able to reuse the other ELF without regenerating Stage 1.
      while IFS= read -r declared_core; do
        declared_workload="$(stage2_derive_workload_name "${artifact_dir}" "${declared_core}")"
        if [[ "${rerun_completed}" -eq 0 ]] && \
            collected_result_complete "${declared_workload}"; then
          continue
        fi
        artifact_cores+=("${declared_core}")
      done < <(stage2_artifact_cores "${artifact_dir}")
      for grouped_workload in "${stage2_firesim_workloads[@]}"; do
        [[ "${STAGE2_ARTIFACT_BY_WORKLOAD[${grouped_workload}]}" == "${artifact_dir}" ]] || continue
        grouped_core="${STAGE2_CORE_BY_WORKLOAD[${grouped_workload}]}"
        for existing_core in "${artifact_cores[@]}"; do
          [[ "${existing_core}" == "${grouped_core}" ]] && continue 2
        done
        artifact_cores+=("${grouped_core}")
      done
      cores_csv="$(IFS=,; printf '%s' "${artifact_cores[*]}")"
      if [[ "${skip_elves}" -eq 0 ]]; then
        log "${artifact_dir}: building pending core ELF set ${cores_csv}; kernel .obj files are removed afterward"
        bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
          "${build_elves_common_args[@]}" \
          --artifact-dir="${artifact_dir}" --core="${cores_csv}"
      fi
      for grouped_core in "${artifact_cores[@]}"; do
        [[ -s "${artifact_dir}/model-${grouped_core}core.elf" ]] || \
          die "missing ${artifact_dir}/model-${grouped_core}core.elf; rerun Stage 1 if kernel .obj files were removed before all required ELFs were built"
      done
      STAGE2_ELVES_READY_BY_ARTIFACT["${artifact_dir}"]=1
    fi

    stage2_add_active_runtime_workload "${workload}"
    if [[ "${skip_package}" -eq 0 ]]; then
      log "${workload}: generating only its FireMarshal overlay and YAML/JSON inputs"
      bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" \
        --no-clean --artifact-dir="${artifact_dir}" --core="${core}"
    fi
    stage2_require_packaged_workload "${workload}"

    if [[ "${skip_images}" -eq 0 ]]; then
      image_args=(--workload="${workload}")
      if [[ "${built_image_count}" -gt 0 ]]; then
        image_args+=(--reuse-shared-kernel-build)
      fi
      log "${workload}: building and installing one FireMarshal image"
      bash "${SCRIPT_DIR}/build-firemarshal-images.sh" "${image_args[@]}"
      built_image_count=$((built_image_count + 1))
    fi

    if [[ "${skip_firesim}" -eq 0 ]]; then
      require_installed_firemarshal_image "${workload}"
      firesim_args=(--workload="${workload}")
      [[ -z "${rvv_panic_retries_arg}" ]] || \
        firesim_args+=(--rvv-panic-retries="${rvv_panic_retries_arg}")
      log "${workload}: running FireSim"
      PYTORCH_CHIPYARD_DEFER_FINAL_CLEANUP=1 \
        bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${firesim_args[@]}"
      collected_result_complete "${workload}" || \
        die "FireSim returned success without durable model.log/autotune.log for ${workload}"

      # Reclaim every workload derivative immediately. The Stage 1 runner,
      # weights, input, model spec, and source remain reusable.
      bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" --workload="${workload}"
      remove_completed_workload_elf "${workload}"
      log "${workload}: durable logs verified; ELF/image/package/runtime removed"
    else
      # Diagnostic image/package-only runs produce no durable completion. Drop
      # runtime derivatives but preserve the ELF so a later FireSim run resumes.
      bash "${SCRIPT_DIR}/cleanup-runtime-artifacts.sh" --workload="${workload}"
    fi
  done

  if [[ "${skip_firesim}" -eq 0 ]]; then
    printf '[stage2][PASS] FireSim results=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
  fi
fi

if [[ "${skip_table4}" -eq 0 ]]; then
  log "completing Table 4 from Docker Stage 1 compile measurements"
  bash "${SCRIPT_DIR}/run_table4.sh" \
    --resume \
    --kernels="${table4_kernels}" \
    --repeats="${table4_repeats}" \
    --output-dir="${table4_stage1_output}"
  printf '[stage2][PASS] Table 4 results=%s\n' "${table4_stage1_output}"
fi

log "done; run bash scripts/run-plot.sh to generate figures"
if [[ -n "${experiment_name}" ]]; then
  printf 'STAGE2_EXPERIMENT=%s\n' "${experiment_name}"
fi
printf 'STAGE2_RESULTS_DIR=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
printf 'STAGE2_STATUS=PASS\n'

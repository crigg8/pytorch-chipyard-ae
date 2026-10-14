#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

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

By default, collected FireSim workloads with a valid .completed marker and
required result logs are skipped. Table 4 also resumes passed measurements.

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
                        Skip ELF/package stages, keep collected results, and
                        build/run/clean one incomplete FireSim workload at a time.
  --resume-rebuild-images
                        Skip ELF/package stages, deep-clean Linux/OpenSBI,
                        rebuild images only for workloads without complete
                        outputs, then run only those FireSim workloads.
  --resume-from=NAME    Skip ELF/package stages, keep collected results, and
                        restart per-workload image build/FireSim at NAME.
  --rerun-completed     Run completed FireSim workloads again instead of
                        applying the default completion filter.
  --rvv-panic-retries=N|unlimited
                        Retry an RVV workload after a detected guest kernel
                        panic. Default: 0; unlimited must be explicit.
  --skip-elves          Skip model-<N>core.elf generation.
  --preflight-only      Package inputs and validate hardware references, then
                        exit without building images or accessing the FPGA.
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
  TABLE4_ACCOUNT_ENV) when it exists. It defaults to $HOME/.ae-env.sh.
  CHIPYARD_DIR must point to the local Chipyard checkout. scripts/env.sh
  derives CHIPYARD_ENV_PATH as $CHIPYARD_DIR/env.sh.
  PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS defaults to 0 so existing collected
  results and logs are preserved. Set it to 1 to clean them before FireSim runs.
  PYTORCH_CHIPYARD_KEEP_FIREMARSHAL_IMAGES=1 preserves successfully run images.
  The default removes each image after its result has been collected.
  PYTORCH_CHIPYARD_CLEAN_FIRESIM_RUNS_DIR must remain 1 for Stage 2 so copied
  rootfs images cannot accumulate between workloads.
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1 selects the quick partial Figure
  6/8/9/11/13 experiment. Prefer scripts/simple-stage2.sh to set it.
  TABLE4_TVM_AE_ROOT points to the prepared TVM-Gemmini AE tree. It defaults to
  $HOME/tvm-gemmini-ae on the author review server.
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
experiment_pending_artifact_dirs=()
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
preflight_only=0
skip_package=0
skip_images=0
skip_firesim=0
skip_table4=0
only_table4=0
table4_kernels="squeezenet_fire2_squeeze,resnet50_classifier,mobilenetv2_classifier"
table4_repeats="${TABLE4_REPEATS:-1}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --preflight-only)
      preflight_only=1
      skip_elves=1
      skip_table4=1
      shift
      ;;
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
      skip_elves=1
      skip_package=1
      skip_images=0
      shift
      ;;
    --resume-rebuild-images)
      resume_firesim=1
      rebuild_pending_images=1
      skip_elves=1
      skip_package=1
      skip_images=0
      shift
      ;;
    --resume-from)
      [[ "$#" -ge 2 ]] || die "--resume-from requires a value"
      resume_from_arg="$2"
      skip_elves=1
      skip_package=1
      skip_images=0
      shift 2
      ;;
    --resume-from=*)
      resume_from_arg="${1#--resume-from=}"
      skip_elves=1
      skip_package=1
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

# Stage 2 runs may be invoked one workload at a time. Preserve results collected
# by earlier invocations unless cleanup is explicitly requested by the caller.
# Experiment units are always resumable, so their collected results must remain
# available for the per-workload completion check below.
if [[ "${experiment_selected}" -eq 1 ]]; then
  export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS=0
else
  export PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS="${PYTORCH_CHIPYARD_CLEAN_COLLECTED_RESULTS:-0}"
fi

account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
if [[ "${PYTORCH_CHIPYARD_STAGE2_CONTAINER:-0}" == "1" ]]; then
  [[ "${only_table4}" -eq 0 && "${experiment_arg}" != table-4 ]] || die 'Table 4 is excluded from the Stage2 container'
  skip_elves=1
  skip_table4=1
elif [[ -f "${account_env}" ]]; then
  set +u
  source "${account_env}"
  set -u
  log "loaded account environment: ${account_env}"
fi

set +u
source "${SCRIPT_DIR}/env.sh"
set -u

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

should_collect_output_bin() {
  local workload="$1"

  case "${workload}" in
    gpt2-* | gpt-neo-* | opt-* | pythia-* | *tok*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

count_stage2_files() {
  local root="$1"
  local pattern="$2"
  local count=0
  local path

  [[ -d "${root}" ]] || {
    printf '0\n'
    return
  }
  while IFS= read -r -d '' path; do
    count=$((count + 1))
  done < <(find "${root}" -type f -name "${pattern}" -print0)
  printf '%s\n' "${count}"
}

validate_packaged_workloads() {
  local workload_count=0
  local workload_json workload

  [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" ]] || \
    die "FireMarshal workload directory was not generated: ${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
  while IFS= read -r -d '' workload_json; do
    workload="$(basename -- "${workload_json}" .json)"
    [[ -s "${workload_json}" ]] || die "generated workload JSON is empty: ${workload_json}"
    [[ -d "${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload}" ]] || \
      die "generated workload overlay is missing: ${PYTORCH_CHIPYARD_WORKLOAD_DIR}/overlay-${workload}"
    [[ -s "${FIRESIM_WORKLOAD_DIR}/${workload}.json" ]] || \
      die "generated FireSim workload JSON is missing or empty: ${FIRESIM_WORKLOAD_DIR}/${workload}.json"
    workload_count=$((workload_count + 1))
  done < <(find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" \
    -maxdepth 1 -type f -name '*.json' ! -name '*base.json' -print0 | sort -z)

  [[ "${workload_count}" -gt 0 ]] || \
    die "workload packaging produced no usable workloads under ${PYTORCH_CHIPYARD_WORKLOAD_DIR}"
  printf '%s\n' "${workload_count}"
}

collected_result_complete() {
  local workload="$1"
  local result_dir="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}/${workload}"

  [[ -f "${result_dir}/.completed" ]] || return 1
  [[ -s "${result_dir}/model.log" ]] || return 1
  [[ -s "${result_dir}/autotune.log" ]] || return 1
  if should_collect_output_bin "${workload}"; then
    [[ -s "${result_dir}/output.bin" ]] || return 1
  fi
  if [[ -f "${result_dir}/run.log" ]]; then
    grep -Fq '[runner] model_ret=0' "${result_dir}/run.log" || return 1
  fi
}

append_unique_workload() {
  local workload="$1"
  local existing

  [[ -n "${workload}" ]] || return 0
  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "invalid workload '${workload}'"
  for existing in "${stage2_firesim_workloads[@]}"; do
    [[ "${existing}" == "${workload}" ]] && return 0
  done
  stage2_firesim_workloads+=("${workload}")
}

load_stage2_firesim_workloads() {
  local arg payload workload workload_json
  local requested=()

  stage2_firesim_workloads=()
  if [[ "${#workload_args[@]}" -gt 0 ]]; then
    for arg in "${workload_args[@]}"; do
      payload="${arg#--workload=}"
      IFS=',' read -r -a requested <<<"${payload}"
      for workload in "${requested[@]}"; do
        append_unique_workload "${workload}"
      done
    done
  else
    while IFS= read -r -d '' workload_json; do
      append_unique_workload "$(basename -- "${workload_json}" .json)"
    done < <(find "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" -maxdepth 1 -type f \
      -name '*.json' ! -name '*base.json' -print0 | sort -z)
  fi

  [[ "${#stage2_firesim_workloads[@]}" -gt 0 ]] || \
    die "no FireSim workloads selected or packaged"
}

filter_stage2_firesim_workloads() {
  local workload
  local found_resume_from=0
  local filtered=()

  if [[ -z "${resume_from_arg}" ]]; then
    found_resume_from=1
  fi

  for workload in "${stage2_firesim_workloads[@]}"; do
    if [[ "${found_resume_from}" -eq 0 ]]; then
      if [[ "${workload}" == "${resume_from_arg}" ]]; then
        found_resume_from=1
      else
        continue
      fi
    fi

    if [[ "${resume_firesim}" -eq 1 ]] && collected_result_complete "${workload}"; then
      log "resume: skipping completed workload ${workload}"
      continue
    fi
    filtered+=("${workload}")
  done

  [[ "${found_resume_from}" -eq 1 ]] || \
    die "resume workload '${resume_from_arg}' was not found in the selected/discovered workload list"
  stage2_firesim_workloads=("${filtered[@]}")
}

require_installed_firemarshal_image() {
  local workload="$1"
  local image_dir="${FIREMARSHAL_IMAGE_DIR}/${workload}"

  [[ -s "${image_dir}/${workload}.img" ]] || \
    die "missing FireMarshal image for ${workload}: ${image_dir}/${workload}.img; rerun without --skip-images"
  [[ -s "${image_dir}/${workload}-bin" ]] || \
    die "missing FireMarshal boot binary for ${workload}: ${image_dir}/${workload}-bin; rerun without --skip-images"
}

remove_installed_firemarshal_image() {
  local workload="$1"
  local keep_images="${PYTORCH_CHIPYARD_KEEP_FIREMARSHAL_IMAGES:-0}"
  local image_dir

  [[ "${keep_images}" == "0" || "${keep_images}" == "1" ]] || \
    die "PYTORCH_CHIPYARD_KEEP_FIREMARSHAL_IMAGES must be 0 or 1"
  if [[ "${keep_images}" == "1" ]]; then
    log "preserving FireMarshal image for ${workload}"
    return 0
  fi

  [[ -n "${FIREMARSHAL_IMAGE_DIR}" && "${FIREMARSHAL_IMAGE_DIR}" == /* && \
     "${FIREMARSHAL_IMAGE_DIR}" != "/" ]] || \
    die "refusing unsafe FireMarshal image cleanup root '${FIREMARSHAL_IMAGE_DIR}'"
  [[ "${workload}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "refusing cleanup for invalid workload '${workload}'"
  image_dir="${FIREMARSHAL_IMAGE_DIR}/${workload}"
  [[ "$(dirname -- "${image_dir}")" == "${FIREMARSHAL_IMAGE_DIR}" ]] || \
    die "refusing unsafe FireMarshal image cleanup target '${image_dir}'"

  if [[ -e "${image_dir}" ]]; then
    rm -rf -- "${image_dir}"
    log "removed completed FireMarshal image: ${image_dir}"
  fi
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

append_pending_artifact_once() {
  local artifact_dir="$1"
  local existing

  for existing in "${experiment_pending_artifact_dirs[@]}"; do
    [[ "${existing}" == "${artifact_dir}" ]] && return 0
  done
  experiment_pending_artifact_dirs+=("${artifact_dir}")
}

if [[ "${experiment_selected}" -eq 1 ]]; then
  completed_workloads=0
  for workload in "${experiment_workloads[@]}"; do
    if [[ "${rerun_completed}" -eq 0 ]] && \
        reused_workload="$(reusable_completed_workload "${workload}")"; then
      completed_workloads=$((completed_workloads + 1))
      if [[ "${reused_workload}" == "${workload}" ]]; then
        log "${experiment_name}: skipping completed workload ${workload}"
      else
        log "${experiment_name}: reusing ${reused_workload} for ${workload}"
      fi
    else
      experiment_pending_workloads+=("${workload}")
      append_pending_artifact_once "$(artifact_dir_for_workload "${workload}")"
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
    die "missing TVM-Gemmini AE environment: ${table4_tvm_ae_root}/scripts/env.sh; set TABLE4_TVM_AE_ROOT or pass --skip-table4"
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

build_elves_args=()
if [[ -n "${chipyard_env_arg}" ]]; then
  build_elves_args+=(--chipyard-env="${chipyard_env_arg}")
fi
if [[ -n "${riscv_toolchain_dir_arg}" ]]; then
  build_elves_args+=(--riscv-toolchain-dir="${riscv_toolchain_dir_arg}")
fi
if [[ -n "${riscv_gxx_arg}" ]]; then
  build_elves_args+=(--riscv-gxx="${riscv_gxx_arg}")
fi
if [[ "${experiment_selected}" -eq 1 ]]; then
  for artifact_dir in "${experiment_pending_artifact_dirs[@]}"; do
    build_elves_args+=(--artifact-dir="${artifact_dir}")
  done
elif [[ "${alias_first_selected}" -eq 1 ]]; then
  for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
    build_elves_args+=(--artifact-dir="${artifact_dir}")
  done
fi

# Validate selected bitstream and driver archives before expensive ELF/image work.
if [[ "${skip_firesim}" -eq 0 && "${#workload_args[@]}" -gt 0 ]]; then
  log "validating selected FireSim hardware artifacts"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
    --preflight-only \
    "${workload_args[@]}"
fi

if [[ "${skip_elves}" -eq 0 ]]; then
  log "building ELFs from examples"
  bash "${SCRIPT_DIR}/build-chipyard-elves.sh" \
    "${build_elves_args[@]}"
  elf_count="$(count_stage2_files "${REPO_ROOT}/examples" 'model-*core.elf')"
  [[ "${elf_count}" -gt 0 ]] || \
    die "ELF generation produced no model-<N>core.elf files; Stage 1 build materials are missing or incomplete"
  printf '[stage2][PASS] ELF generation root=%s count=%s\n' \
    "${REPO_ROOT}/examples" "${elf_count}"
fi

if [[ "${skip_package}" -eq 0 ]]; then
  log "packaging FireMarshal/FireSim workloads from examples"
  package_args=()
  if [[ "${experiment_selected}" -eq 1 ]]; then
    for artifact_dir in "${experiment_pending_artifact_dirs[@]}"; do
      package_args+=(--artifact-dir="${artifact_dir}")
    done
  elif [[ "${alias_first_selected}" -eq 1 ]]; then
    for artifact_dir in "${alias_first_artifact_dirs[@]}"; do
      package_args+=(--artifact-dir="${artifact_dir}")
    done
  fi
  bash "${SCRIPT_DIR}/package-firemarshal-workload.sh" "${package_args[@]}"
  packaged_workload_count="$(validate_packaged_workloads)"
  printf '[stage2][PASS] FireMarshal workload directory=%s count=%s\n' \
    "${PYTORCH_CHIPYARD_WORKLOAD_DIR}" "${packaged_workload_count}"
fi

# A non-selective run can only discover its workload set after packaging.
if [[ "${skip_firesim}" -eq 0 && "${#workload_args[@]}" -eq 0 ]]; then
  firesim_preflight_args=(--preflight-only)
  if [[ "${resume_firesim}" -eq 1 ]]; then
    firesim_preflight_args+=(--resume)
  fi
  if [[ -n "${resume_from_arg}" ]]; then
    firesim_preflight_args+=(--resume-from="${resume_from_arg}")
  fi
  log "validating discovered FireSim hardware artifacts"
  bash "${SCRIPT_DIR}/run-firesim-workloads.sh" \
    "${firesim_preflight_args[@]}"
fi

if [[ "${preflight_only}" -eq 1 ]]; then
  log "preflight complete; no FireMarshal image was built and no FPGA was accessed"
  printf 'STAGE2_PREFLIGHT_STATUS=PASS\n'
  exit 0
fi

if [[ "${skip_firesim}" -eq 0 ]]; then
  load_stage2_firesim_workloads
  filter_stage2_firesim_workloads
  if [[ "${#stage2_firesim_workloads[@]}" -eq 0 ]]; then
    log "resume: no pending FireSim workloads"
  else
    log "running ${#stage2_firesim_workloads[@]} FireSim workload(s) with one image at a time"
  fi

  for workload in "${stage2_firesim_workloads[@]}"; do
    image_built=0
    if [[ "${skip_images}" -eq 0 ]]; then
      log "${workload}: building and installing its FireMarshal image"
      bash "${SCRIPT_DIR}/build-firemarshal-images.sh" --workload="${workload}"
      image_built=1
    fi
    require_installed_firemarshal_image "${workload}"

    firesim_args=(--workload="${workload}")
    if [[ -n "${rvv_panic_retries_arg}" ]]; then
      firesim_args+=(--rvv-panic-retries="${rvv_panic_retries_arg}")
    fi
    log "${workload}: running FireSim"
    bash "${SCRIPT_DIR}/run-firesim-workloads.sh" "${firesim_args[@]}"

    # A failed FireSim invocation exits above and deliberately leaves the image
    # in place for diagnosis/retry. Only a successfully collected workload is
    # eligible for ephemeral image cleanup.
    if [[ "${image_built}" -eq 1 ]]; then
      remove_installed_firemarshal_image "${workload}"
    fi
  done
  printf '[stage2][PASS] FireSim results=%s\n' "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
elif [[ "${skip_images}" -eq 0 ]]; then
  # Building images without running FireSim is an explicit diagnostic mode.
  # Retain each completed image because there is no successful run after which
  # it can safely be removed. The builder resets shared state between images.
  log "building and installing FireMarshal images without FireSim execution"
  image_args=()
  if [[ "${rebuild_pending_images}" -eq 1 || "${resume_firesim}" -eq 1 ]]; then
    image_args+=(--pending-only)
  fi
  if [[ "${experiment_selected}" -eq 1 ]]; then
    for workload in "${experiment_pending_workloads[@]}"; do
      image_args+=(--workload="${workload}")
    done
  elif [[ "${alias_first_selected}" -eq 1 ]]; then
    for workload in "${alias_first_workloads[@]}"; do
      image_args+=(--workload="${workload}")
    done
  fi
  bash "${SCRIPT_DIR}/build-firemarshal-images.sh" "${image_args[@]}"
  printf '[stage2][PASS] FireMarshal images=%s FireSim workloads=%s\n' \
    "${FIREMARSHAL_IMAGE_DIR}" "${FIRESIM_WORKLOAD_DIR}"
fi

if [[ "${skip_table4}" -eq 0 ]]; then
  log "completing Table 4 from Docker Stage 1 compile measurements"
  bash "${SCRIPT_DIR}/run_table4.sh" \
    --resume \
    --build-verilator \
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

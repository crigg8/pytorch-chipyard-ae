#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

log() {
  printf '[stage1] %s\n' "$*"
}

die() {
  printf '[stage1][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-stage1.sh [options]

Run the Stage 1 compiler workflow and write artifacts under examples/.

Options:
  --skip-cnn                Skip default CNN workloads.
  --skip-alias-first        Skip the alias-first ablation workloads.
  --skip-im2col             Skip im2col CNN workloads.
  --skip-sdpa               Skip default SDPA LLM workloads.
  --skip-flex-attn          Skip flash/window attention LLM workloads.
  --skip-gemmini-autotune   Skip gemmini-max-autotune.
  --skip-table4             Skip Docker-side Table 4 kernel compile timing.
  --only-table4             Run only the Docker-side Table 4 kernel compile
                            measurements and save artifacts for Stage 2.
  --table4-kernels=LIST     Limit Table 4 to selected built-in kernel IDs.
                            Default: all three.
  --table4-repeats=N        Table 4 compile trials. Default: 1.
  --only-alias-first        Compile only the Figure 9 alias-first ablation:
                            CNN alias-first off plus LLM alias-first on/off,
                            all using Gemmini and 4 cores.
  --only-alias-first-cnn-off
                            Compile only the four CNN alias-first off cases.
  -h, --help                Show this help.
EOF
}

skip_cnn=0
skip_alias_first=0
skip_im2col=0
skip_sdpa=0
skip_flex_attn=0
skip_gemmini_autotune=0
skip_table4=0
only_table4=0
table4_kernels="squeezenet_fire2_squeeze,resnet50_classifier,mobilenetv2_classifier"
table4_repeats="${TABLE4_REPEATS:-1}"
only_alias_first=0
only_alias_first_cnn_off=0
skip_alias_first_requested=0
simple_stage2_enabled=0
case "${PYTORCH_CHIPYARD_SIMPLE_STAGE2:-0}" in
  1 | true | TRUE | yes | YES | on | ON)
    simple_stage2_enabled=1
    export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
    ;;
  0 | false | FALSE | no | NO | off | OFF | "")
    export PYTORCH_CHIPYARD_SIMPLE_STAGE2=0
    ;;
  *)
    die "PYTORCH_CHIPYARD_SIMPLE_STAGE2 must be 0 or 1"
    ;;
esac

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-cnn)
      skip_cnn=1
      shift
      ;;
    --skip-alias-first | --skip-alias-first-ablation)
      skip_alias_first=1
      skip_alias_first_requested=1
      shift
      ;;
    --skip-im2col)
      skip_im2col=1
      shift
      ;;
    --skip-sdpa)
      skip_sdpa=1
      shift
      ;;
    --skip-flex-attn)
      skip_flex_attn=1
      shift
      ;;
    --skip-gemmini-autotune)
      skip_gemmini_autotune=1
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
    --only-alias-first | --only-alias-first-ablation)
      only_alias_first=1
      shift
      ;;
    --only-alias-first-cnn-off)
      only_alias_first_cnn_off=1
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

if [[ "${only_alias_first}" -eq 1 && "${only_alias_first_cnn_off}" -eq 1 ]]; then
  die "--only-alias-first and --only-alias-first-cnn-off are mutually exclusive"
fi
if [[ "${only_table4}" -eq 1 && \
      ( "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ) ]]; then
  die "--only-table4 cannot be combined with an alias-first only option"
fi
if [[ "${only_table4}" -eq 1 && "${skip_table4}" -eq 1 ]]; then
  die "--only-table4 cannot be combined with --skip-table4"
fi

[[ "${table4_repeats}" =~ ^[1-9][0-9]*$ ]] || \
  die "--table4-repeats must be a positive integer"

if [[ "${only_table4}" -eq 1 ]]; then
  skip_cnn=1
  skip_alias_first=1
  skip_im2col=1
  skip_sdpa=1
  skip_flex_attn=1
  skip_gemmini_autotune=1
fi

if [[ "${only_alias_first}" -eq 1 || "${only_alias_first_cnn_off}" -eq 1 ]]; then
  [[ "${skip_alias_first_requested}" -eq 0 ]] || \
    die "an alias-first only option cannot be combined with --skip-alias-first"
  skip_cnn=1
  skip_im2col=1
  skip_sdpa=1
  skip_flex_attn=1
  skip_gemmini_autotune=1
  skip_table4=1
fi

if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
  [[ "${only_table4}" -eq 0 && "${only_alias_first}" -eq 0 && \
      "${only_alias_first_cnn_off}" -eq 0 ]] || \
    die "PYTORCH_CHIPYARD_SIMPLE_STAGE2 cannot be combined with an --only-* option"
  skip_im2col=1
  skip_table4=1
  log "simple Stage 2 artifact mode enabled"
fi

cd "${REPO_ROOT}"
mkdir -p examples

if [[ "${skip_cnn}" -eq 0 ]]; then
  log "running default CNN workloads"
  if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/run-cnn.sh" --backend=gemmini,rvv,scalar --model=squeezenet
  else
    bash "${SCRIPT_DIR}/run-cnn.sh"
  fi
  printf '[stage1][PASS] CNN compiler artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_alias_first}" -eq 0 ]]; then
  log "running CNN Gemmini 4-core alias-first off ablation"
  if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/run-alias-first-cnn.sh" --mode=off --model=squeezenet
  else
    bash "${SCRIPT_DIR}/run-alias-first-cnn.sh" --mode=off
  fi

  if [[ "${only_alias_first_cnn_off}" -eq 0 ]]; then
    log "running LLM SDPA seq=256 Gemmini 4-core alias-first on/off ablation"
    if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
      bash "${SCRIPT_DIR}/run-alias-first-sdpa.sh" --model=opt
    else
      bash "${SCRIPT_DIR}/run-alias-first-sdpa.sh"
    fi
  fi
  printf '[stage1][PASS] Figure 9 alias-first artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_im2col}" -eq 0 ]]; then
  log "running im2col CNN workloads"
  bash "${SCRIPT_DIR}/run-im2col.sh"
  printf '[stage1][PASS] Figure 10 im2col artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_sdpa}" -eq 0 ]]; then
  log "running SDPA LLM workloads"
  if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/run-sdpa.sh" --model=opt
  else
    bash "${SCRIPT_DIR}/run-sdpa.sh"
  fi
  printf '[stage1][PASS] Figure 6(b) SDPA artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_flex_attn}" -eq 0 ]]; then
  log "running flash/window attention LLM workloads"
  if [[ "${simple_stage2_enabled}" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/run-flex-attn.sh" \
      --model=opt --attention=flash,window --host=default --seq-len=256
  else
    bash "${SCRIPT_DIR}/run-flex-attn.sh"
  fi
  printf '[stage1][PASS] Figure 13 FlexAttention artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_gemmini_autotune}" -eq 0 ]]; then
  log "running gemmini-max-autotune workload"
  bash "${SCRIPT_DIR}/run_gemmini_autotune.sh"
  printf '[stage1][PASS] Figure 11 Gemmini autotuning artifacts=%s\n' "${REPO_ROOT}/examples"
fi

if [[ "${skip_table4}" -eq 0 ]]; then
  table4_results_root="${TABLE4_RESULTS_ROOT:-${REPO_ROOT}/results/table4}"
  table4_run_id="$(date -u +%Y%m%dT%H%M%SZ)-stage1"
  table4_output_dir="${table4_results_root}/${table4_run_id}"
  log "measuring Table 4 PyTorch compile times inside the Stage 1 container"
  TABLE4_PYTORCH_COMPILE_CONTEXT=docker-stage1 \
    bash "${SCRIPT_DIR}/run_table4.sh" \
    --kernels="${table4_kernels}" \
    --toolchains=pytorch \
    --phases=compile \
    --repeats="${table4_repeats}" \
    --output-dir="${table4_output_dir}"
  # Docker root is mapped to nobody on root-squashed bind mounts, and Torch's
  # atomic weight-file write leaves weights.bin at mode 0600. Stage 2 runs as
  # the host AE account and must be able to read artifacts and append results.
  chmod -R a+rwX "${table4_output_dir}"
  mkdir -p "${table4_results_root}"
  ln -sfn "${table4_run_id}" "${table4_results_root}/stage1-latest"
  printf '[stage1][PASS] Table 4 compile results=%s\n' "${table4_output_dir}"
  printf '[stage1] TABLE4_STAGE1_LATEST=%s\n' "${table4_results_root}/stage1-latest"
fi

log "done; artifacts are under ${REPO_ROOT}/examples and ${REPO_ROOT}/results"
printf 'STAGE1_ARTIFACT_DIR=%s\n' "${REPO_ROOT}/examples"
printf 'STAGE1_RESULTS_DIR=%s\n' "${REPO_ROOT}/results"
printf 'STAGE1_STATUS=PASS\n'

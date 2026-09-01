#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-im2col.sh \
    --backend=[gemmini|rvv|scalar|default|comma-separated-list] \
    --model=[alexnet|mobilenetv2|resnet50|squeezenet|default|comma-separated-list] \
    --core=[4|default|comma-separated-list] \
    --artifact-dir=<path>

Examples:
  bash scripts/run-im2col.sh --backend=gemmini --model=resnet50

Options:
  --backend=LIST       Backend list. default expands to gemmini.
  --model=LIST         CNN model list. default expands to all CNN workloads.
  --core=LIST          Core-count list to record for the local ELF build step.
                       Default expands to 4.
  --artifact-dir=PATH  Output directory. For multiple combinations, PATH is
                       treated as a root and per-combination subdirectories are used.
                       This script only generates compiler artifacts. Build ELF
                       files later with scripts/build-chipyard-elves.sh on the
                       local Chipyard/FireSim host.
  --batch-size=N       Input batch size. Default: 1.
  --seed=N             PyTorch seed. Default: 0.
  -h, --help           Show this help.
EOF
}

backend_arg="default"
model_arg="default"
core_arg="default"
artifact_dir=""
batch_size="${CNN_BATCH_SIZE:-1}"
seed="${CNN_SEED:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --backend)
      [[ "$#" -ge 2 ]] || pc_usage_error "--backend requires a value"
      backend_arg="$2"
      shift 2
      ;;
    --backend=*)
      backend_arg="${1#--backend=}"
      shift
      ;;
    --model)
      [[ "$#" -ge 2 ]] || pc_usage_error "--model requires a value"
      model_arg="$2"
      shift 2
      ;;
    --model=*)
      model_arg="${1#--model=}"
      shift
      ;;
    --core | --cores)
      [[ "$#" -ge 2 ]] || pc_usage_error "$1 requires a value"
      core_arg="$2"
      shift 2
      ;;
    --core=* | --cores=*)
      core_arg="${1#*=}"
      shift
      ;;
    --artifact-dir)
      [[ "$#" -ge 2 ]] || pc_usage_error "--artifact-dir requires a value"
      artifact_dir="$2"
      shift 2
      ;;
    --artifact-dir=*)
      artifact_dir="${1#--artifact-dir=}"
      shift
      ;;
    --batch-size)
      [[ "$#" -ge 2 ]] || pc_usage_error "--batch-size requires a value"
      batch_size="$2"
      shift 2
      ;;
    --batch-size=*)
      batch_size="${1#--batch-size=}"
      shift
      ;;
    --seed)
      [[ "$#" -ge 2 ]] || pc_usage_error "--seed requires a value"
      seed="$2"
      shift 2
      ;;
    --seed=*)
      seed="${1#--seed=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      pc_usage_error "unknown argument '$1'"
      ;;
  esac
done

[[ "${batch_size}" =~ ^[0-9]+$ && "${batch_size}" != "0" ]] || pc_die "invalid --batch-size=${batch_size}"
[[ "${seed}" =~ ^-?[0-9]+$ ]] || pc_die "invalid --seed=${seed}"

raw_backends=()
raw_models=()
raw_cores=()
pc_split_csv "${backend_arg}" raw_backends gemmini
pc_split_csv "${model_arg}" raw_models alexnet mobilenetv2 resnet50 squeezenet
pc_split_csv "${core_arg}" raw_cores 4

backends=()
for backend in "${raw_backends[@]}"; do
  if ! normalized_backend="$(pc_normalize_backend "${backend}")"; then
    exit 1
  fi
  pc_append_unique backends "${normalized_backend}"
done

models=()
for model in "${raw_models[@]}"; do
  if ! normalized_model="$(pc_normalize_cnn_model "${model}")"; then
    exit 1
  fi
  pc_append_unique models "${normalized_model}"
done

cores=()
for core in "${raw_cores[@]}"; do
  [[ "${core}" =~ ^[0-9]+$ && "${core}" != "0" ]] || pc_die "invalid --core=${core}"
  pc_append_unique cores "${core}"
done

pc_prepare_environment

export TORCHINDUCTOR_IM2COL_MM=1

combo_count=$((${#backends[@]} * ${#models[@]}))
for model in "${models[@]}"; do
  for backend in "${backends[@]}"; do
    script_path="$(pc_cnn_script "${model}")"
    suffix="${model}/${backend}-im2col"
    storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
    default_dir="${PC_REPO_ROOT}/examples/${storage_suffix}"
    output_dir="$(pc_combo_artifact_dir "${artifact_dir}" "${combo_count}" "${default_dir}" "${storage_suffix}")"
    cache_key="im2col-${model}-${backend}"
    pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
    pc_write_artifact_build_plan "${output_dir}" "${backend}" "${cores[@]}"

    pc_run_compile_once "${backend}" "${output_dir}" "${cache_key}" "${script_path}" \
      --batch-size "${batch_size}" --seed "${seed}"
  done
done

pc_log "done; build ELF files on the local Chipyard host with scripts/build-chipyard-elves.sh"

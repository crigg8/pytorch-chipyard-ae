#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-sdpa.sh \
    --model=[gpt2|gpt-neo|opt|pythia|default|comma-separated-list] \
    --artifact-dir=<path>

Examples:
  bash scripts/run-sdpa.sh --model=opt
  bash scripts/run-sdpa.sh --model=opt,pythia

Options:
  --model=LIST         LLM model list. default expands to all SDPA workloads.
  --artifact-dir=PATH  Output directory. For multiple models, PATH is treated
                       as a root and per-model subdirectories are used.
                       This script only generates compiler artifacts. Build ELF
                       files later with scripts/build-chipyard-elves.sh on the
                       local Chipyard/FireSim host.
  --batch-size=N       Input batch size. Default: 1.
  --seed=N             PyTorch seed. Default: 0.
  -h, --help           Show this help.
EOF
}

model_arg="default"
artifact_dir=""
batch_size="${LLM_BATCH_SIZE:-1}"
seed="${LLM_SEED:-0}"
seq_len=256
backend=gemmini

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --model)
      [[ "$#" -ge 2 ]] || pc_usage_error "--model requires a value"
      model_arg="$2"
      shift 2
      ;;
    --model=*)
      model_arg="${1#--model=}"
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

raw_models=()
pc_split_csv "${model_arg}" raw_models gpt2 gpt-neo opt pythia

models=()
for model in "${raw_models[@]}"; do
  if ! normalized_model="$(pc_normalize_llm_model "${model}")"; then
    exit 1
  fi
  pc_append_unique models "${normalized_model}"
done

pc_prepare_environment

combo_count="${#models[@]}"
for model in "${models[@]}"; do
  script_path="$(pc_llm_script "${model}")"
  case "${model}" in
    opt | pythia)
      suffix="${model}/${backend}/sdpa/seq${seq_len}"
      ;;
    *)
      suffix="${model}/${backend}"
      ;;
  esac
  storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
  default_dir="${PC_REPO_ROOT}/examples/${storage_suffix}"
  output_dir="$(pc_combo_artifact_dir "${artifact_dir}" "${combo_count}" "${default_dir}" "${storage_suffix}")"
  cache_key="sdpa-${model}-${backend}-seq${seq_len}"
  compile_args=(--batch-size "${batch_size}" --seed "${seed}")
  pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
  pc_write_artifact_build_plan "${output_dir}" "${backend}" 2 4

  case "${model}" in
    opt | pythia)
      compile_args+=(--attn sdpa)
      ;;
  esac

  export LLM_TOKEN_LENGTH="${seq_len}"
  pc_run_compile_once "${backend}" "${output_dir}" "${cache_key}" "${script_path}" "${compile_args[@]}"
done

pc_log "done; build ELF files on the local Chipyard host with scripts/build-chipyard-elves.sh"

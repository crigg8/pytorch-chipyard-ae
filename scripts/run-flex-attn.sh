#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-flex-attn.sh \
    --model=[opt|pythia|default|comma-separated-list] \
    --attention=[sdpa|flash|window|default|comma-separated-list] \
    --host=[rocket|boom|default|comma-separated-list] \
    --seq-len=[256|512|768|1024|default|comma-separated-list] \
    --artifact-dir=<path>

Examples:
  bash scripts/run-flex-attn.sh --model=opt --attention=window --host=rocket --seq-len=1024
  bash scripts/run-flex-attn.sh --model=opt,pythia --attention=flash,window --host=rocket --seq-len=256,512,768,1024

Options:
  --model=LIST         Model list. default expands to opt,pythia.
  --attention=LIST     Attention list. default expands to sdpa,flash,window.
  --host=LIST          Host list. The paper default uses Rocket for every
                       model/mode and BOOM only for OPT flash/window.
  --seq-len=LIST       Sequence length list. default expands to 256,512,768,1024.
  --artifact-dir=PATH  Output directory. For multiple compile combinations, PATH is
                       treated as a root and per-combination subdirectories are used.
                       Host variants share the same compile artifact and only record
                       the core-specific ELF build plan for the local host step.
                       Build ELF files later with scripts/build-chipyard-elves.sh
                       on the local Chipyard/FireSim host.
  --batch-size=N       Input batch size. Default: 1.
  --seed=N             PyTorch seed. Default: 0.
  -h, --help           Show this help.
EOF
}

model_arg="default"
attention_arg="default"
host_arg="default"
seq_len_arg="default"
artifact_dir=""
batch_size="${LLM_BATCH_SIZE:-1}"
seed="${LLM_SEED:-0}"
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
    --attention)
      [[ "$#" -ge 2 ]] || pc_usage_error "--attention requires a value"
      attention_arg="$2"
      shift 2
      ;;
    --attention=*)
      attention_arg="${1#--attention=}"
      shift
      ;;
    --host)
      [[ "$#" -ge 2 ]] || pc_usage_error "--host requires a value"
      host_arg="$2"
      shift 2
      ;;
    --host=*)
      host_arg="${1#--host=}"
      shift
      ;;
    --seq-len)
      [[ "$#" -ge 2 ]] || pc_usage_error "--seq-len requires a value"
      seq_len_arg="$2"
      shift 2
      ;;
    --seq-len=*)
      seq_len_arg="${1#--seq-len=}"
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
raw_attentions=()
raw_hosts=()
raw_seq_lens=()
pc_split_csv "${model_arg}" raw_models opt pythia
pc_split_csv "${attention_arg}" raw_attentions sdpa flash window
pc_split_csv "${host_arg}" raw_hosts rocket boom
pc_split_csv "${seq_len_arg}" raw_seq_lens 256 512 768 1024

models=()
for model in "${raw_models[@]}"; do
  if ! normalized="$(pc_normalize_llm_model "${model}")"; then
    exit 1
  fi
  case "${normalized}" in
    opt | pythia) pc_append_unique models "${normalized}" ;;
    *) pc_die "FlexAttention experiments only support opt and pythia, got '${model}'" ;;
  esac
done

attentions=()
for attention in "${raw_attentions[@]}"; do
  if ! normalized_attention="$(pc_normalize_attention "${attention}")"; then
    exit 1
  fi
  pc_append_unique attentions "${normalized_attention}"
done

hosts=()
for host in "${raw_hosts[@]}"; do
  if ! normalized_host="$(pc_normalize_host "${host}")"; then
    exit 1
  fi
  pc_append_unique hosts "${normalized_host}"
done

seq_lens=()
for seq_len in "${raw_seq_lens[@]}"; do
  pc_validate_seq_len "${seq_len}"
  pc_append_unique seq_lens "${seq_len}"
done

pc_prepare_environment

combo_count=$((${#models[@]} * ${#attentions[@]} * ${#seq_lens[@]}))
host_arg_default=0
if [[ -z "$(pc_trim "${host_arg}")" || "$(pc_lower "$(pc_trim "${host_arg}")")" == "default" ]]; then
  host_arg_default=1
fi

for model in "${models[@]}"; do
  script_path="$(pc_llm_script "${model}")"
  for attention in "${attentions[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
      suffix="${model}/${backend}/${attention}/seq${seq_len}"
      storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
      default_dir="${PC_REPO_ROOT}/examples/${storage_suffix}"
      output_dir="$(pc_combo_artifact_dir "${artifact_dir}" "${combo_count}" "${default_dir}" "${storage_suffix}")"
      case "${attention}" in
        sdpa) cache_key="sdpa-${model}-${backend}-seq${seq_len}" ;;
        *) cache_key="flex-${model}-${attention}-seq${seq_len}" ;;
      esac

      built_cores=()
      if [[ "${host_arg_default}" -eq 1 && "${attention}" == "sdpa" && "${seq_len}" == "256" ]]; then
        pc_append_unique built_cores 2
      fi
      for host in "${hosts[@]}"; do
        if [[ "${host_arg_default}" -eq 1 && "${attention}" == "sdpa" && "${host}" == "boom" ]]; then
          continue
        fi
        if [[ "${host_arg_default}" -eq 1 && "${host}" == "boom" && "${model}" != "opt" ]]; then
          continue
        fi
        core="$(pc_core_for_host "${host}")"
        pc_append_unique built_cores "${core}"
      done

      export LLM_TOKEN_LENGTH="${seq_len}"
      pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
      pc_write_artifact_build_plan "${output_dir}" "${backend}" "${built_cores[@]}"
      pc_run_compile_once "${backend}" "${output_dir}" "${cache_key}" "${script_path}" \
        --batch-size "${batch_size}" --seed "${seed}" --attn "${attention}"
    done
  done
done

pc_log "done; build ELF files on the local Chipyard host with scripts/build-chipyard-elves.sh"

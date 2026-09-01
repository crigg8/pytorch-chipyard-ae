#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
source "${SCRIPT_DIR}/stage1.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-alias-first-cnn.sh \
    --mode=[on|off|default|comma-separated-list] \
    --model=[alexnet|mobilenetv2|resnet50|squeezenet|default|comma-separated-list] \
    --artifact-dir=<path>

Default:
  Compile all four CNNs with alias-first both enabled and disabled. Artifacts
  use the fp32 8x8 Gemmini backend and request only a 4-core ELF.

Options:
  --mode=LIST         Alias-first modes. Default: on,off.
  --model=LIST        CNN models. Default: all CNN models.
  --artifact-dir=PATH Output directory. With multiple combinations, PATH is a
                      root containing per-model and per-mode directories.
  --batch-size=N      Input batch size. Default: 1.
  --seed=N            PyTorch seed. Default: 0.
  -h, --help          Show this help.
EOF
}

mode_arg="default"
model_arg="default"
artifact_dir=""
batch_size="${CNN_BATCH_SIZE:-1}"
seed="${CNN_SEED:-0}"
backend=gemmini

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ "$#" -ge 2 ]] || pc_usage_error "--mode requires a value"
      mode_arg="$2"
      shift 2
      ;;
    --mode=*)
      mode_arg="${1#--mode=}"
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

[[ "${batch_size}" =~ ^[0-9]+$ && "${batch_size}" != "0" ]] || \
  pc_die "invalid --batch-size=${batch_size}"
[[ "${seed}" =~ ^-?[0-9]+$ ]] || pc_die "invalid --seed=${seed}"

raw_modes=()
raw_models=()
pc_split_csv "${mode_arg}" raw_modes on off
pc_split_csv "${model_arg}" raw_models alexnet mobilenetv2 resnet50 squeezenet

modes=()
for mode in "${raw_modes[@]}"; do
  case "$(pc_lower "${mode}")" in
    on) pc_append_unique modes on ;;
    off) pc_append_unique modes off ;;
    *) pc_die "unknown alias-first mode '${mode}'; expected on, off, or default" ;;
  esac
done

models=()
for model in "${raw_models[@]}"; do
  if ! normalized_model="$(pc_normalize_cnn_model "${model}")"; then
    exit 1
  fi
  pc_append_unique models "${normalized_model}"
done

pc_prepare_environment

# Keep the ablation focused on alias-first rather than optional custom lowering.
unset PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY
unset TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG
unset TRITON_CHIPYARD_EXTERN_CALL_LIBRARY

combo_count=$((${#models[@]} * ${#modes[@]}))
for model in "${models[@]}"; do
  script_path="$(pc_cnn_script "${model}")"
  for mode in "${modes[@]}"; do
    alias_first=0
    [[ "${mode}" == "on" ]] && alias_first=1

    suffix="${model}/${backend}-alias-first-${mode}"
    storage_suffix="$(pc_artifact_storage_suffix "${suffix}")"
    default_dir="${PC_REPO_ROOT}/examples/${storage_suffix}"
    output_dir="$(pc_combo_artifact_dir \
      "${artifact_dir}" "${combo_count}" "${default_dir}" "${storage_suffix}")"
    cache_key="cnn-${model}-${backend}-alias-first-${mode}"

    export TRITON_CHIPYARD_ENABLE_ALIAS_FIRST="${alias_first}"
    pc_write_artifact_workload_hint "${output_dir}" "${suffix}"
    pc_write_artifact_build_plan "${output_dir}" "${backend}" 4
    pc_run_compile_once "${backend}" "${output_dir}" "${cache_key}" "${script_path}" \
      --batch-size "${batch_size}" --seed "${seed}"
  done
done

pc_log "done; alias-first CNN ablation artifacts are under ${PC_REPO_ROOT}/examples"

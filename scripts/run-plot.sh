#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

log() {
  printf '[plot] %s\n' "$*"
}

die() {
  printf '[plot][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-plot.sh [options]

Generate paper figures from collected FireSim results.
Without --results-dir, use all logs in scripts/figures/results-workload.
Existing per-run Docker results are collected there automatically.
Figures are written to scripts/figures.

Options:
  --results-dir=PATH  Result directory. Passed to scripts/figure/plot_results.sh.
  --only-alias-first  Generate only the Figure 9 alias-first plot.
  -h, --help          Show this help.

Environment:
  PYTHON_BIN                                      Override Python executable.
  PYTORCH_CHIPYARD_CONDA_ENV                     Plot conda environment name.
  PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR   Default result directory.
  STAGE2_STATE_ROOT                            Stage 2 Docker state directory.
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1               Generate partial simple figures.
EOF
}

plot_args=()
results_dir_selected=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --results-dir)
      [[ "$#" -ge 2 ]] || die "--results-dir requires a value"
      plot_args+=("--results-dir" "$2")
      results_dir_selected=1
      shift 2
      ;;
    --results-dir=*)
      plot_args+=("$1")
      results_dir_selected=1
      shift
      ;;
    --only-alias-first)
      plot_args+=("$1")
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

if [[ "${results_dir_selected}" -eq 0 ]]; then
  # Apply the same path policy as plot_results.sh, including ignoring legacy
  # account settings that point into a different checkout.
  source "${SCRIPT_DIR}/env.sh"
  results_dir="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}"
  python3 "${REPO_ROOT}/docker/stage2/local_runtime.py" collect-results \
    "${STAGE2_STATE_ROOT:-$HOME/.local/share/pytorch-chipyard-stage2-local}" \
    "${results_dir}"
  plot_args+=(--results-dir "${results_dir}")
fi

cd "${REPO_ROOT}"
log "generating figures"
bash "${SCRIPT_DIR}/figure/plot_results.sh" "${plot_args[@]}"
log "done; figures are under ${SCRIPT_DIR}/figures"
printf 'FIGURE_OUTPUT_DIR=%s\n' "${SCRIPT_DIR}/figures"
printf 'FIGURES_STATUS=PASS\n'

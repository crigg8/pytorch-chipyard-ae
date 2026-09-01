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

Options:
  --results-dir=PATH  Result directory. Passed to scripts/figure/plot_results.sh.
  --only-alias-first  Generate only the Figure 9 alias-first plot.
  -h, --help          Show this help.

Environment:
  PYTHON_BIN                                      Override Python executable.
  PYTORCH_CHIPYARD_CONDA_ENV                     Plot conda environment name.
  PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR   Default result directory.
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1               Generate partial simple figures.
EOF
}

plot_args=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --results-dir)
      [[ "$#" -ge 2 ]] || die "--results-dir requires a value"
      plot_args+=("--results-dir" "$2")
      shift 2
      ;;
    --results-dir=*)
      plot_args+=("$1")
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

cd "${REPO_ROOT}"
log "generating figures"
bash "${SCRIPT_DIR}/figure/plot_results.sh" "${plot_args[@]}"
log "done; figures are under ${SCRIPT_DIR}/figures"
printf 'FIGURE_OUTPUT_DIR=%s\n' "${SCRIPT_DIR}/figures"
printf 'FIGURES_STATUS=PASS\n'

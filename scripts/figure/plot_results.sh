#!/usr/bin/env bash
set -euo pipefail

FIGURE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "${FIGURE_SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
source "${ROOT_DIR}/env.sh"

RESULTS_DIR="${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR:-$ROOT_DIR/figures/results-workload}"
FIGURE_DIR="${ROOT_DIR}/figures"
CSV_DIR="${ROOT_DIR}/.csv"
LOG_DIR=""

log() {
  printf '[plot-results] %s\n' "$*"
}

warn() {
  printf '[plot-results][warn] %s\n' "$*" >&2
}

die() {
  printf '[plot-results][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  bash scripts/figure/plot_results.sh [--results-dir=PATH] [--only-alias-first]

Default:
  Generate CSV inputs from FireSim results, run all paper figure scripts, and
  print the generated figure paths.

Options:
  --results-dir=PATH  Result directory. Default: ${RESULTS_DIR}
  --only-alias-first  Generate only the alias-first ablation plot.
  -h, --help          Show this help.

Environment:
  PYTHON_BIN                                      Override Python executable
  PYTORCH_CHIPYARD_CONDA_ENV                     Default: ${PYTORCH_CHIPYARD_CONDA_ENV}
  PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR   Default result directory
  PYTORCH_CHIPYARD_SIMPLE_STAGE2=1               Generate the partial simple set
EOF
}

only_alias_first=0
simple_stage2=0
case "${PYTORCH_CHIPYARD_SIMPLE_STAGE2:-0}" in
  1 | true | TRUE | yes | YES | on | ON) simple_stage2=1 ;;
  0 | false | FALSE | no | NO | off | OFF | "") ;;
  *) die "PYTORCH_CHIPYARD_SIMPLE_STAGE2 must be 0 or 1" ;;
esac

select_python_cmd() {
  local chipyard_python

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    if conda run -n "${PYTORCH_CHIPYARD_CONDA_ENV}" python - <<'PY' >/dev/null 2>&1
import matplotlib
import numpy
import pandas
PY
    then
      PYTHON_CMD=(conda run -n "${PYTORCH_CHIPYARD_CONDA_ENV}" python)
      return
    fi
  fi

  chipyard_python="${CHIPYARD_DIR:+${CHIPYARD_DIR}/.conda-env/bin/python3}"
  if [[ -x "${chipyard_python}" ]] && "${chipyard_python}" - <<'PY' >/dev/null 2>&1
import matplotlib
import numpy
import pandas
PY
  then
    PYTHON_CMD=("${chipyard_python}")
    return
  fi

  PYTHON_CMD=(python3)
}

require_plot_python() {
  if ! "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
import matplotlib
import numpy
import pandas
PY
  then
    die "Python plot dependencies are missing. Run scripts/install.sh, activate ${PYTORCH_CHIPYARD_CONDA_ENV}, or set PYTHON_BIN."
  fi
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --results-dir)
      [[ "$#" -ge 2 ]] || { warn "--results-dir requires a value"; exit 2; }
      RESULTS_DIR="$2"
      shift 2
      ;;
    --results-dir=*)
      RESULTS_DIR="${1#--results-dir=}"
      shift
      ;;
    --only-alias-first)
      only_alias_first=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      warn "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pytorch-chipyard-plot-logs.XXXXXX")"
export PYTORCH_CHIPYARD_LOG_DIR="${LOG_DIR}"
marker="$FIGURE_DIR/.plot_results_start"
plot_cleanup_on_exit() {
  local status=$?
  trap - EXIT
  rm -rf -- "${LOG_DIR}"
  rm -f -- "${marker}"
  exit "${status}"
}
trap plot_cleanup_on_exit EXIT

mkdir -p "$FIGURE_DIR" "$CSV_DIR"
: >"$marker"

PYTHON_CMD=()
select_python_cmd
require_plot_python
python_path="$("${PYTHON_CMD[@]}" -c 'import sys; print(sys.executable)')"

log "result input: $RESULTS_DIR"
log "csv output  : $CSV_DIR"
log "temporary compatibility logs: $LOG_DIR"
log "figure out  : $FIGURE_DIR"
log "python      : $python_path"

"${PYTHON_CMD[@]}" "$FIGURE_SCRIPT_DIR/generate_plot_inputs.py" --results-dir "$RESULTS_DIR"

if [[ "${only_alias_first}" -eq 1 ]]; then
  plot_scripts=(plot_alias_first_ablation.py)
  expected_outputs=(
    "Figure 9|Fig9.pdf"
  )
elif [[ "${simple_stage2}" -eq 1 ]]; then
  plot_scripts=(
    plot_cnn_absolute_cycles.py
    plot_sdpa_prefill_256.py
    plot_alias_first_ablation.py
    plot_mobilenet_squeezenet_attribution.py
    plot_gemmini_max_autotune.py
    plot_flex_prefill.py
    plot_flash_window_core_ratio.py
  )
  expected_outputs=(
    "Figure 6(a), partial|Fig6a.pdf"
    "Figure 6(b), partial|Fig6b.pdf"
    "Figure 8(a), partial|Fig8a.pdf"
    "Figure 9, partial|Fig9.pdf"
    "Figure 11, partial|Fig11.pdf"
    "Figure 13(a), partial|Fig13a.pdf"
    "Figure 13(c), partial|Fig13c.pdf"
  )
else
  plot_scripts=(
    plot_cnn_absolute_cycles.py
    plot_cnn_result.py
    plot_alias_first_ablation.py
    plot_im2col.py
    plot_sdpa_prefill_256.py
    plot_flex_prefill.py
    plot_flash_window_core_ratio.py
    plot_im2col_site_attribution.py
    plot_mobilenet_squeezenet_attribution.py
    plot_gemmini_max_autotune.py
  )
  expected_outputs=(
    "Figure 6(a)|Fig6a.pdf"
    "Figure 6(b)|Fig6b.pdf"
    "Figure 7(a)|Fig7a.pdf"
    "Figure 7(b)|Fig7b.pdf"
    "Figure 7(c)|Fig7c.pdf"
    "Figure 8(a)|Fig8a.pdf"
    "Figure 8(b)|Fig8b.pdf"
    "Figure 9|Fig9.pdf"
    "Figure 10(a)|Fig10a.pdf"
    "Figure 10(b)|Fig10b.pdf"
    "Figure 10(c)|Fig10c.pdf"
    "Figure 11|Fig11.pdf"
    "Figure 13(a)|Fig13a.pdf"
    "Figure 13(b)|Fig13b.pdf"
    "Figure 13(c)|Fig13c.pdf"
  )
fi

failed=0
for plot_script in "${plot_scripts[@]}"; do
  log "running ${plot_script}"
  if "${PYTHON_CMD[@]}" "$FIGURE_SCRIPT_DIR/$plot_script"; then
    log "finished ${plot_script}"
  else
    warn "failed ${plot_script}"
    failed=1
  fi
done

log "generated figure files:"
generated_count=0
while IFS= read -r figure_path; do
  generated_count=$((generated_count + 1))
  printf '[plot-results]   %s\n' "$figure_path"
done < <(find "$FIGURE_DIR" -maxdepth 1 -type f -name '*.pdf' -newer "$marker" | sort)

if [[ "$generated_count" -eq 0 ]]; then
  warn "no figure files had sufficient inputs"
fi

log "checking the paper figure manifest:"
skipped_count=0
for expected in "${expected_outputs[@]}"; do
  figure_label="${expected%%|*}"
  figure_path="${FIGURE_DIR}/${expected#*|}"
  if [[ ! -s "${figure_path}" || ! "${figure_path}" -nt "${marker}" ]]; then
    printf '[plot-results][SKIP] %s=insufficient logs\n' "${figure_label}"
    skipped_count=$((skipped_count + 1))
    continue
  fi
  printf '[plot-results][PASS] %s=%s\n' "${figure_label}" "${figure_path}"
done

if [[ "$failed" -ne 0 ]]; then
  warn "one or more figure scripts failed"
  exit 1
fi

printf 'PLOT_RESULTS_DIR=%s\n' "${FIGURE_DIR}"
if [[ "${generated_count}" -eq 0 ]]; then
  printf 'PLOT_RESULTS_STATUS=NO_DATA\n'
elif [[ "${skipped_count}" -gt 0 ]]; then
  printf 'PLOT_RESULTS_STATUS=PARTIAL\n'
else
  printf 'PLOT_RESULTS_STATUS=PASS\n'
fi

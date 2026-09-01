#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

log() {
  printf '[simple-stage2] %s\n' "$*"
}

die() {
  printf '[simple-stage2][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/simple-stage2.sh [run-stage2 options] [--skip-plot]

Run the quick functionality-evaluation slice:
  - SqueezeNet and OPT-125M for partial Figures 6(a,b), 8(a), and 9
  - Figure 11 blocking (64,64,512) only
  - OPT-125M sequence length 256 for partial Figures 13(a,c)

Completed shared workloads are reused by both simple and full Stage 2 runs.
The reduced Figure 11 workload is separate; an existing full Figure 11 result
can satisfy it, but the reduced result never marks the full experiment complete.
Pending workloads build, run, and remove one FireMarshal image at a time.

Options:
  --skip-plot  Run the Stage 2 workloads without generating partial figures.
  -h, --help   Show this help.

All other options are passed to scripts/run-stage2.sh. Do not pass
--experiment, workload-selection, or resume-selection options.
EOF
}

skip_plot=0
stage2_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-plot)
      skip_plot=1
      shift
      ;;
    --experiment | --experiment=* | --workload | --workload=* | \
    --only-alias-first | --only-alias-first-ablation | \
    --only-alias-first-cnn-off | --only-table4 | --resume | \
    --resume-firesim | --resume-rebuild-images | --resume-from | --resume-from=*)
      die "'$1' conflicts with the fixed simple experiment selection"
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      stage2_args+=("$1")
      shift
      ;;
  esac
done

export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1

log "running the reusable simple Stage 2 workload set one image at a time"
bash "${SCRIPT_DIR}/run-stage2.sh" --experiment=simple "${stage2_args[@]}"

if [[ "${skip_plot}" -eq 0 ]]; then
  log "generating partial Figures 6(a,b), 8(a), 9, 11, and 13(a,c)"
  bash "${SCRIPT_DIR}/run-plot.sh"
fi

printf 'SIMPLE_STAGE2_STATUS=PASS\n'

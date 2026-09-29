#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"

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
If the reduced Figure 11 artifact is missing, derive it from the existing full
Stage 1 artifact on the host. No Docker rebuild or Stage 1 recompile is needed.

Options:
  --skip-plot  Run the Stage 2 workloads without generating partial figures.
  -h, --help   Show this help.

All other options are passed to scripts/run-stage2.sh. Do not pass
--experiment, workload-selection, or resume-selection options.
EOF
}

skip_plot=0
rerun_completed=0
stage2_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --skip-plot)
      skip_plot=1
      shift
      ;;
    --rerun-completed)
      rerun_completed=1
      stage2_args+=("$1")
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

prepare_simple_autotune_artifact() (
  local full_dir="${REPO_ROOT}/examples/artifact-gemmini-max-autotune/gemmini"
  local simple_dir="${REPO_ROOT}/examples/artifact-gemmini-max-autotune-simple/gemmini"
  local account_env="${PYTORCH_CHIPYARD_ACCOUNT_ENV:-${TABLE4_ACCOUNT_ENV:-${HOME}/.ae-env.sh}}"
  local workload

  [[ ! -s "${simple_dir}/build.sh" ]] || return 0

  # Resolve completion exactly as Stage 2 does, without changing the caller's
  # environment. Completed full Figure 11 results also satisfy the simple run.
  set +u
  [[ ! -f "${account_env}" ]] || source "${account_env}"
  source "${SCRIPT_DIR}/env.sh"
  set -u
  source "${SCRIPT_DIR}/stage2-common.sh"
  if [[ "${rerun_completed}" -eq 0 ]]; then
    for workload in gemmini-max-autotune-simple-gemmini-4core gemmini-max-autotune-gemmini-4core; do
      if pc_stage2_result_complete "${PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR}" "${workload}"; then
        return 0
      fi
    done
  fi

  command -v python3 >/dev/null 2>&1 || die "python3 is required to prepare the simple Figure 11 artifact"
  python3 - "${full_dir}" "${simple_dir}" <<'PY'
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile


def fail(message):
    raise SystemExit(f"[simple-stage2][error] {message}")


source, destination = map(Path, sys.argv[1:])
for name in ("build.sh", "runner.cpp", "model_spec.json", "input.bin", "weights.bin"):
    if not (source / name).is_file():
        fail(f"missing full Stage 1 artifact: {source / name}; retain the existing examples/ directory when updating the checkout")
if destination.exists():
    fail(f"incomplete simple artifact already exists: {destination}")

# All kernels are already compiled. Keep their original indices and dispatch
# code, and reduce only the candidate table traversed by the runner. Renumbering
# candidates would break the generated per-candidate grid dispatch functions.
runner = (source / "runner.cpp").read_text()
spec = json.loads((source / "model_spec.json").read_text())
sites = spec.get("deferred_autotune_sites", [])
if len(sites) != 1 or not isinstance(sites[0], dict):
    fail("expected exactly one Figure 11 autotune site")
site = sites[0]
choices = site.get("choices", [])
choice_by_index = {choice["candidate_index"]: choice for choice in choices}
if len(choice_by_index) != len(choices):
    fail("duplicate Figure 11 candidate indices")

configs = {}
for match in re.finditer(
    r"static const ChipyardAutotuneConfigEntry "
    r"(kChipyardAutotuneSite0Candidate\d+Config)\[\] = \{\n(.*?)\n\};",
    runner, re.S,
):
    configs[match[1]] = dict(re.findall(r'\{"([^"\\]+)", "([^"\\]*)"\}', match[2]))

table_pattern = (
    r"(static const ChipyardAutotuneCandidate kChipyardAutotuneSite0Candidates\[\] = \{\n)"
    r"(.*?)(\n\};)"
)
tables = list(re.finditer(table_pattern, runner, re.S))
if len(tables) != 1:
    fail("unsupported Figure 11 runner candidate table")
table = tables[0]
selected_rows = []
selected_indices = []
baseline_indices = []
seen_indices = set()
for row in table[2].splitlines():
    entry = re.fullmatch(
        r'\s*\{static_cast<int32_t>\((\d+)\), "([^"\\]+)", .*?, '
        r'(kChipyardAutotuneSite0Candidate\d+Config)\},', row,
    )
    if entry is None or entry[3] not in configs:
        fail("unsupported Figure 11 runner candidate entry")
    index = int(entry[1])
    choice = choice_by_index.get(index)
    if choice is None or choice.get("kernel_name") != entry[2] or index in seen_indices:
        fail("Figure 11 runner and model_spec.json candidates disagree")
    seen_indices.add(index)
    config = configs[entry[3]]
    if tuple(config.get(key) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")) != ("64", "64", "512"):
        continue
    selected_rows.append(row)
    selected_indices.append(index)
    if all(config.get(key) == "0" for key in ("gemmini_tile_i", "gemmini_tile_j", "gemmini_tile_k")):
        baseline_indices.append(index)

if seen_indices != set(choice_by_index):
    fail("Figure 11 runner does not contain every model_spec.json candidate")
if len(selected_indices) != 81 or len(baseline_indices) != 1:
    fail(f"expected 81 blocking (64,64,512) candidates and one baseline, found {len(selected_indices)} and {len(baseline_indices)}")
baseline = baseline_indices[0]
baseline_kernel = choice_by_index[baseline]["kernel_name"]
runner = runner[:table.start(2)] + "\n".join(selected_rows) + runner[table.end(2):]

# Update the count AND default in the site descriptor. Everything referring to
# kernel indices (including the full symbol table) deliberately stays intact.
site_pattern = (
    r"(static const ChipyardAutotuneSitePlan kChipyardAutotuneSites\[\] = \{\n"
    r"\s*\{static_cast<std::size_t>\(0\), static_cast<int32_t>\()\d+"
    r'(\), static_cast<int32_t>\(1\), [^\n]*?, )"[^"\\]*", '
    r"static_cast<std::size_t>\(\d+\)(, kChipyardAutotuneSite0Candidates, [^\n]+\n\};)"
)
runner, count = re.subn(
    site_pattern,
    lambda match: f'{match[1]}{baseline}{match[2]}"{baseline_kernel}", static_cast<std::size_t>(81){match[3]}',
    runner,
)
if count != 1:
    fail("unsupported Figure 11 runner site descriptor or disabled candidate dispatch")
site["choices"] = [choice_by_index[index] for index in selected_indices]
site["default_choice_index"] = baseline
site["default_kernel_name"] = baseline_kernel

# The generated build script uses paths relative to SCRIPT_DIR. Verify those
# inputs now; Stage 2 may have deleted objects after a previous ELF build.
build_script = (source / "build.sh").read_text()
objects = re.findall(r'^KERNEL_OBJECTS\+=\("\$\{SCRIPT_DIR\}/([^"\n]+)"\)$', build_script, re.M)
if not objects:
    fail("full Figure 11 build.sh has no reusable staged kernel objects")
for relative in objects:
    if not (source / relative).is_file():
        fail(f"full Stage 1 kernel object is missing: {source / relative}")

# Independent copies are essential: Stage 2 deletes .obj files after linking.
# Publish atomically so an interrupted preparation cannot look complete.
destination.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
try:
    shutil.copytree(
        source, staging, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.elf", ".*elf-fingerprint", "output.bin", "model.log", "autotune.log"),
    )
    (staging / "runner.cpp").write_text(runner)
    (staging / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (staging / ".pytorch-chipyard-workload-rel").write_text("gemmini-max-autotune-simple/gemmini\n")
    (staging / ".pytorch-chipyard-backend").write_text("gemmini\n")
    (staging / ".pytorch-chipyard-build-cores").write_text("4\n")
    staging.rename(destination)
finally:
    if staging.exists():
        shutil.rmtree(staging)
print(f"[simple-stage2] prepared {destination}: 81 candidates from existing Stage 1 artifacts")
PY
)

export PYTORCH_CHIPYARD_SIMPLE_STAGE2=1
prepare_simple_autotune_artifact

log "running the reusable simple Stage 2 workload set one image at a time"
bash "${SCRIPT_DIR}/run-stage2.sh" --experiment=simple "${stage2_args[@]}"

if [[ "${skip_plot}" -eq 0 ]]; then
  log "generating partial Figures 6(a,b), 8(a), 9, 11, and 13(a,c)"
  bash "${SCRIPT_DIR}/run-plot.sh"
fi

printf 'SIMPLE_STAGE2_STATUS=PASS\n'

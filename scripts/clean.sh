#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
FIGURE_DIR="${REPO_ROOT}/scripts/figures"

usage() {
  cat <<'EOF'
Usage: bash scripts/clean.sh [--dry-run | --yes]

Permanently remove this entire PyTorch-Chipyard checkout except final figure
files under scripts/figures. The cleanup never follows paths outside the
checkout.

Options:
  --dry-run  Print the cleanup scope without deleting anything.
  --yes      Confirm the permanent cleanup.
  -h, --help Show this help.
EOF
}

die() {
  printf '[clean][error] %s\n' "$*" >&2
  exit 1
}

dry_run=0
confirmed=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --yes) confirmed=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${dry_run}" -eq 0 || "${confirmed}" -eq 0 ]] || \
  die "--dry-run and --yes cannot be used together"

# The script is intentionally destructive. Resolve its own checkout and require
# repository-specific markers before constructing any deletion target.
case "${REPO_ROOT}" in
  / | "${HOME}" | "$(dirname -- "${REPO_ROOT}")")
    die "refusing unsafe repository root: ${REPO_ROOT}"
    ;;
esac
[[ -f "${REPO_ROOT}/README.md" ]] || die "README.md not found under ${REPO_ROOT}"
grep -Fq '# PyTorch-Chipyard Artifact' "${REPO_ROOT}/README.md" || \
  die "repository marker not found in ${REPO_ROOT}/README.md"
[[ -d "${REPO_ROOT}/scripts/figure" ]] || die "plot scripts not found under ${REPO_ROOT}"
[[ -d "${REPO_ROOT}/examples" ]] || die "examples directory not found under ${REPO_ROOT}"
[[ -d "${FIGURE_DIR}" ]] || die "figure directory not found: ${FIGURE_DIR}"

figures=()
while IFS= read -r -d '' figure; do
  figures+=("${figure}")
done < <(
  find "${FIGURE_DIR}" -mindepth 1 -maxdepth 1 -type f \
    \( -name '*.pdf' -o -name '*.png' -o -name '*.svg' -o -name '*.eps' \) \
    -size +0c -print0 | sort -z
)

[[ "${#figures[@]}" -gt 0 ]] || \
  die "no non-empty final figures found under ${FIGURE_DIR}; run scripts/run-plot.sh first"

printf '[clean] repository: %s\n' "${REPO_ROOT}"
printf '[clean] preserving %s final figure(s):\n' "${#figures[@]}"
for figure in "${figures[@]}"; do
  printf '[clean]   %s\n' "${figure#${REPO_ROOT}/}"
done

if [[ "${dry_run}" -eq 1 ]]; then
  printf '[clean] dry run: everything else under %s would be permanently removed\n' \
    "${REPO_ROOT}"
  exit 0
fi
[[ "${confirmed}" -eq 1 ]] || \
  die "permanent cleanup requires --yes; inspect it first with --dry-run"

staging="$(mktemp -d "${REPO_ROOT}/.clean-figures.XXXXXX")"
deletion_started=0
cleanup_staging_on_exit() {
  local status=$?
  trap - EXIT
  if [[ "${deletion_started}" -eq 0 && -d "${staging}" ]]; then
    rm -rf -- "${staging}"
  elif [[ "${status}" -ne 0 && -d "${staging}" ]]; then
    printf '[clean][error] cleanup was interrupted; preserved figure recovery files at %s\n' \
      "${staging}" >&2
  fi
  exit "${status}"
}
trap cleanup_staging_on_exit EXIT

for figure in "${figures[@]}"; do
  staged="${staging}/$(basename -- "${figure}")"
  cp -p -- "${figure}" "${staged}"
  cmp -s -- "${figure}" "${staged}" || die "failed to stage ${figure}"
done

entries=()
while IFS= read -r -d '' entry; do
  [[ "${entry}" == "${staging}" ]] || entries+=("${entry}")
done < <(find "${REPO_ROOT}" -mindepth 1 -maxdepth 1 -print0)

deletion_started=1
for entry in "${entries[@]}"; do
  case "${entry}" in
    "${REPO_ROOT}"/*) rm -rf -- "${entry}" ;;
    *) die "refusing cleanup target outside repository: ${entry}" ;;
  esac
done

mkdir -p -- "${FIGURE_DIR}"
while IFS= read -r -d '' staged; do
  mv -- "${staged}" "${FIGURE_DIR}/"
done < <(find "${staging}" -mindepth 1 -maxdepth 1 -type f -print0)
rmdir -- "${staging}"

for figure in "${figures[@]}"; do
  restored="${FIGURE_DIR}/$(basename -- "${figure}")"
  [[ -s "${restored}" ]] || die "restored figure is missing or empty: ${restored}"
done

trap - EXIT
printf '[clean] complete; only final figures remain under %s\n' "${FIGURE_DIR}"

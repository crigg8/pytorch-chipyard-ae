#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
BITSTREAM_URL="https://github.com/crigg8/pytorch-chipyard-ae/releases/download/v1.0.0/bitstreams.zip"
DESTINATION="${REPO_ROOT}/bitstreams"

die() {
  printf '[download-bitstreams][error] %s\n' "$*" >&2
  exit 1
}

for command in curl unzip; do
  command -v "${command}" >/dev/null 2>&1 || die "required command not found: ${command}"
done

[[ ! -e "${DESTINATION}" ]] || \
  die "destination already exists: ${DESTINATION}"

staging="$(mktemp -d "${REPO_ROOT}/.download-bitstreams.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${staging}" in
    "${REPO_ROOT}"/.download-bitstreams.*) rm -rf -- "${staging}" ;;
    *) printf '[download-bitstreams][error] refusing unsafe staging cleanup: %s\n' \
         "${staging}" >&2 ;;
  esac
  exit "${status}"
}
trap cleanup EXIT

archive="${staging}/bitstreams.zip"
unpacked="${staging}/unpacked"

printf '[download-bitstreams] downloading %s\n' "${BITSTREAM_URL}"
curl --fail --location --retry 5 --retry-delay 2 \
  --output "${archive}" "${BITSTREAM_URL}"

unzip -tq "${archive}" >/dev/null || die "downloaded ZIP archive is invalid"

entry_count=0
while IFS= read -r entry; do
  [[ -n "${entry}" ]] || continue
  case "${entry}" in
    bitstreams | bitstreams/*) ;;
    *) die "ZIP contains an entry outside its top-level bitstreams directory: ${entry}" ;;
  esac
  case "/${entry}/" in
    */../*) die "ZIP contains an unsafe parent-directory entry: ${entry}" ;;
  esac
  entry_count=$((entry_count + 1))
done < <(unzip -Z1 "${archive}")
[[ "${entry_count}" -gt 0 ]] || die "ZIP archive is empty"

mkdir -p -- "${unpacked}"
unzip -q "${archive}" -d "${unpacked}"
extracted="${unpacked}/bitstreams"
[[ -d "${extracted}" ]] || die "ZIP did not extract a top-level bitstreams directory"

if find "${extracted}" -type l -print -quit | grep -q .; then
  die "extracted bitstreams directory contains a symbolic link"
fi

bundle_count=0
while IFS= read -r -d '' bundle_dir; do
  [[ -s "${bundle_dir}/driver-bundle.tar.gz" ]] || \
    die "missing driver-bundle.tar.gz under ${bundle_dir}"
  [[ -s "${bundle_dir}/firesim.tar.gz" ]] || \
    die "missing firesim.tar.gz under ${bundle_dir}"
  bundle_count=$((bundle_count + 1))
done < <(find "${extracted}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
[[ "${bundle_count}" -gt 0 ]] || die "no bitstream bundles were found in the ZIP"

unexpected="$(find "${extracted}" -mindepth 1 -maxdepth 1 ! -type d -print -quit)"
[[ -z "${unexpected}" ]] || die "unexpected file directly under bitstreams: ${unexpected}"

mv -- "${extracted}" "${DESTINATION}"
printf '[download-bitstreams] installed %s bundle(s) under %s\n' \
  "${bundle_count}" "${DESTINATION}"


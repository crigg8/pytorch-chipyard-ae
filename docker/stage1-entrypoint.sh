#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-pytorch-chipyard}"

cd /opt/pytorch-chipyard

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec "$@"

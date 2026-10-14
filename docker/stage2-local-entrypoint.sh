#!/usr/bin/env bash
set -euo pipefail

export STAGE2_EXECUTION=local
export FIRESIM_SSH_HOST=127.0.0.1 FIRESIM_SSH_USER=root FIRESIM_SSH_PORT=22
export FIRESIM_DB=/run/stage2/fpga-db.json

if [[ "${1:-}" == --check-tools ]]; then
  # No hardware access: usable with a rootless build engine as well.
  /usr/bin/python3 /opt/stage2-support/check-local-tools.py
  exit
fi
if [[ "$#" -eq 0 || "$1" == --help || "$1" == -h ]]; then
  printf '%s\n' 'Stage2 local: FireMarshal, FireSim manager and simulation driver run in this container.' \
    'Use scripts/run-stage2-local-docker.sh on a rootful Docker engine.' \
    'Host Vivado programs the FPGA over SSH; host XDMA provides PCIe access.' \
    '--check-tools checks bundled tools without FPGA access.'
  exit
fi
[[ "${STAGE2_HOST_LOCKED:-}" == 1 ]] || { echo 'Use the local host launcher' >&2; exit 1; }
: "${STAGE2_RUN_DIR:?}" "${STAGE2_OUTPUT_UID:?}" "${STAGE2_OUTPUT_GID:?}"
[[ "${STAGE2_OUTPUT_UID}" =~ ^[0-9]+$ && "${STAGE2_OUTPUT_GID}" =~ ^[0-9]+$ ]]

# Fabric runs the simulator through loopback SSH. A separate SSH configuration
# and agent socket are used exclusively for host FPGA programming.
mkdir -p /run/sshd /run/stage2 /root/.ssh
chmod 700 /root/.ssh
ssh-keygen -q -t rsa -b 3072 -N '' -f /root/.ssh/id_rsa
cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -q -t rsa -b 3072 -N '' -f /run/stage2/ssh_host_rsa_key
read -r keytype keydata _ </run/stage2/ssh_host_rsa_key.pub
printf 'stage2-fpga,127.0.0.1 %s %s\n' "${keytype}" "${keydata}" >/run/stage2/known_hosts
cat >/run/stage2/sshd_config <<'EOF'
Port 22
ListenAddress 127.0.0.1
HostKey /run/stage2/ssh_host_rsa_key
PidFile /run/stage2/sshd.pid
PermitRootLogin prohibit-password
PasswordAuthentication no
ChallengeResponseAuthentication no
UsePAM no
AuthorizedKeysFile /root/.ssh/authorized_keys
AllowUsers root
AllowTcpForwarding no
X11Forwarding no
Subsystem sftp internal-sftp
EOF
/usr/sbin/sshd -f /run/stage2/sshd_config -E /run/stage2/sshd.log

exec /usr/local/bin/pytorch-chipyard-stage2 "$@"

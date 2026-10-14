#!/usr/bin/env python3
"""Root-owned sudo entrypoint for one administrator-installed smoke profile."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_runtime import agent_socket

CONFIG = Path("/etc/pytorch-chipyard-stage2-local.json")
BUNDLE = Path("/usr/local/lib/pytorch-chipyard-stage2-local")
STATE = Path("/var/lib/pytorch-chipyard-stage2-local")


def root_owned(path):
    """Reject symlinks and writable path components before privileged use."""
    path = Path(path)
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("not a protected root-owned path: " + str(item))


def invocation(config, args, caller_uid):
    account = pwd.getpwnam(config["user"])
    if caller_uid not in (0, account.pw_uid):
        raise RuntimeError("this smoke profile is installed for " + account.pw_name)
    agent = agent_socket(args.ssh_agent, account.pw_uid)
    image = config["image_id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise RuntimeError("profile must pin an image ID")
    run_id = args.resume or args.run_id
    if run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise RuntimeError("invalid RUN_ID")
    # No inherited Docker options, input/HWDB paths, Python variables, or
    # caller-writable state. The container uses its baked native drivers.
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/root",
           "LANG": "C.UTF-8", "USER": "root", "LOGNAME": "root",
           "DOCKER_CONFIG": str(STATE / "docker-config"),
           "STAGE2_DOCKER_HOST": "unix:///var/run/docker.sock",
           "STAGE2_IMAGE": image, "STAGE2_STATE_ROOT": str(STATE),
           "STAGE2_SMOKE_ARTIFACT_DIR": config["smoke_inputs"],
           "FIRESIM_DB": config["fpga_db"],
           "STAGE2_BUILD_JOBS": "8", "STAGE2_PROGRAM_AGENT_UID": str(account.pw_uid),
           "FIRESIM_SSH_HOST": config["ssh_host"], "FIRESIM_SSH_USER": account.pw_name,
           "FIRESIM_VIVADO_DIR": config["vivado"]}
    if run_id:
        env["RUN_ID"] = run_id
    program = ('import runpy, sys; sys.path.insert(0, sys.argv[1]); '
               'script=sys.argv.pop(1)+"/local-launch.py"; '
               'runpy.run_path(script, run_name="__main__")')
    command = ["/usr/bin/python3", "-I", "-c", program, str(BUNDLE), "--smoke-test", "--ssh-agent", str(agent)]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.resume:
        command.append("--resume")
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-id")
    selection.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ssh-agent", required=True, help="agent socket owned by the configured account")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("run sudo /usr/local/sbin/run-stage2-local-smoke")
    root_owned(CONFIG)
    config = json.loads(CONFIG.read_text())
    for path in (BUNDLE, BUNDLE / "local-launch.py", BUNDLE / "local_runtime.py",
                 config["smoke_inputs"], config["fpga_db"], STATE, STATE / "runs", STATE / "docker-config"):
        root_owned(path)
    command, env = invocation(config, args, int(os.environ.get("SUDO_UID", "0")))
    os.umask(0o022)
    # SUDO_UID is deliberately absent: local-launch keeps checkpoints root-owned.
    raise SystemExit(subprocess.run(command, env=env).returncode)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print("[stage2-local-sudo][error] " + str(error), file=sys.stderr)
        sys.exit(1)

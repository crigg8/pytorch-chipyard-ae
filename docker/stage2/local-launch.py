#!/usr/bin/env python3
"""Launch the full FireSim userspace on the local system Docker engine."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
from urllib.parse import unquote, urlparse

from local_runtime import agent_socket, collect_results, device_numbers, selected_device

WORKLOADS = ("smoke-gemm-rvv-4core", "smoke-gemm-gemmini-4core", "smoke-gemm-scalar-16core")
PROJECT = Path(__file__).resolve().parents[2]


def prepare_hwdb(source, destination):
    """Mount only referenced archives, rewriting their URIs to container paths."""
    import yaml
    entries = yaml.safe_load(source.read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("HWDB must contain hardware configurations")
    mounts = {}
    for name, entry in entries.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(name)) or not isinstance(entry, dict):
            raise ValueError("invalid HWDB entry: " + str(name))
        if not entry.get("bitstream_tar"):
            raise ValueError("U250 HWDB needs bitstream_tar: " + name)
        for field in ("bitstream_tar", "driver_tar"):
            uri = entry.get(field)
            if not uri:
                continue
            parsed = urlparse(uri)
            if parsed.scheme not in ("", "file") or parsed.netloc:
                raise ValueError("download HWDB archives locally first: " + uri)
            path = Path(unquote(parsed.path))
            if not path.is_absolute():
                path = source.parent / path
            path = path.resolve(strict=True)
            if not path.is_file() or "," in str(path):
                raise ValueError("invalid HWDB archive: " + str(path))
            if path not in mounts:
                mounts[path] = "/opt/stage2-hardware/%03d/archive.tar.gz" % len(mounts)
            entry[field] = "file://" + mounts[path]
        if entry.get("custom_runtime_config"):
            raise ValueError("custom_runtime_config is not yet supported by the local launcher")
    destination.write_text(yaml.safe_dump(entries, sort_keys=False))
    return [(str(path), target) for path, target in mounts.items()]


def require_no_host_simulation(proc=Path("/proc")):
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            name = (directory / "comm").read_text().strip()
            args = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        manager = any(Path(os.fsdecode(arg)).name == "firesim" for arg in args[:3] if arg)
        programming = any(Path(os.fsdecode(arg)).name in ("firesim-fpga-util.py", "vivado", "vivado_lab")
                          for arg in args[:4] if arg)
        if name.startswith("FireSim-") or manager or programming:
            raise RuntimeError("another FireSim/FPGA programming process is active (PID " + directory.name + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--smoke-test", action="store_true")
    selection.add_argument("--workload", action="append")
    selection.add_argument("--experiment")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--resume", action="store_true", help="skip existing result logs (the default)")
    reuse.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--print-command", action="store_true", help="prepare and print the launch command; do not run Docker or access devices")
    parser.add_argument("--rvv-panic-retries", type=int, default=0)
    parser.add_argument("--ssh-agent", default=os.environ.get("SSH_AUTH_SOCK"),
                        help="agent socket for the host programming SSH connection")
    args = parser.parse_args()
    if args.experiment == "table-4" or args.rvv_panic_retries < 0:
        parser.error("Table 4 is excluded; retries must be nonnegative")
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    account = pwd.getpwuid(uid)
    gid = int(os.environ.get("SUDO_GID", account.pw_gid))
    home = Path(account.pw_dir)
    if not args.ssh_agent:
        parser.error("provide --ssh-agent PATH (or SSH_AUTH_SOCK) with the host login key loaded")
    agent = agent_socket(args.ssh_agent, int(os.environ.get("STAGE2_PROGRAM_AGENT_UID", uid)))
    program_host = os.environ.get("FIRESIM_SSH_HOST", "host.docker.internal")
    program_user = os.environ.get("FIRESIM_SSH_USER", account.pw_name)
    program_port = os.environ.get("FIRESIM_SSH_PORT", "22")
    vivado = os.environ.get("FIRESIM_VIVADO_DIR", "/home/hongjun/Vivado/2021.1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", program_host):
        parser.error("invalid programming SSH host")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", program_user):
        parser.error("invalid programming SSH user")
    if not program_port.isdecimal() or not 1 <= int(program_port) <= 65535:
        parser.error("invalid programming SSH port")
    if not Path(vivado).is_absolute():
        parser.error("FIRESIM_VIVADO_DIR must be an absolute host path")
    artifact = os.environ.get("STAGE2_ARTIFACT_DIR")
    if args.smoke_test:
        artifact = os.environ.get("STAGE2_SMOKE_ARTIFACT_DIR", str(home / "pytorch-chipyard/results/smoke-test/artifacts/pytorch-chipyard"))
    if not artifact:
        parser.error("set STAGE2_ARTIFACT_DIR to the completed Stage1 input directory")
    artifact = Path(artifact).resolve(strict=True)
    if not artifact.is_dir():
        parser.error("artifact input must be a directory")
    if args.smoke_test:
        for backend, cores in (("rvv", 4), ("gemmini", 4), ("scalar", 16)):
            path = artifact / backend
            for name in (f"model-{cores}core.elf", "input.bin", "weights.bin", "model_spec.json"):
                if not (path / name).is_file() or not (path / name).stat().st_size:
                    parser.error("missing smoke input: " + str(path / name))
            if (path / ".pytorch-chipyard-workload-rel").read_text().strip() != "smoke-gemm/" + backend:
                parser.error("incorrect smoke workload hint: " + str(path))
    db_value = os.environ.get("FIRESIM_DB", "")
    if not db_value or not Path(db_value).is_absolute():
        parser.error("set FIRESIM_DB to the absolute path of the physical FPGA database")
    db = Path(db_value).resolve(strict=True)
    selected_device(db)
    jobs = os.environ.get("STAGE2_BUILD_JOBS", "8")
    if not re.fullmatch(r"[1-9][0-9]*", jobs):
        parser.error("STAGE2_BUILD_JOBS must be positive")
    run_id = os.environ.get("RUN_ID")
    run_id = run_id or datetime.datetime.now(datetime.timezone.utc).strftime("local-%Y%m%dT%H%M%SZ-") + str(os.getpid())
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        parser.error("invalid RUN_ID")
    state = Path(os.environ.get("STAGE2_STATE_ROOT", str(home / ".local/share/pytorch-chipyard-stage2-local"))).resolve()
    run = state / "runs" / run_id
    # Match scripts/env.sh: legacy account settings must not redirect this
    # checkout's durable logs into a different checkout or disposable state.
    if (PROJECT / "scripts/env.sh").is_file():
        results = Path(os.environ.get("PYTORCH_CHIPYARD_FIGURE_RESULTS_WORKLOAD_DIR",
                                     str(PROJECT / "scripts/figures/results-workload"))).resolve()
        if PROJECT not in results.parents:
            results = PROJECT / "scripts/figures/results-workload"
    else:
        # The administrator-installed smoke bundle has no project checkout.
        results = state / "results"
    if "," in str(results):
        parser.error("result bind mount path contains a comma: " + str(results))
    if not re.fullmatch(r"/[A-Za-z0-9_./+-]+", str(run)):
        parser.error("run path contains unsupported FireSim shell characters")
    docker_env = dict(os.environ)
    docker_env.pop("DOCKER_CONTEXT", None)
    endpoint = os.environ.get("STAGE2_DOCKER_HOST", "unix:///var/run/docker.sock")
    if not endpoint.startswith("unix://"):
        parser.error("local execution requires a local Unix Docker socket")
    docker_env["DOCKER_HOST"] = endpoint
    # sudo normally drops DOCKER_HOST; pass the selected engine explicitly.
    docker = (["sudo", "-n", "/usr/bin/docker", "--host", endpoint]
              if os.environ.get("STAGE2_DOCKER_SUDO") == "1" else ["docker"])
    image = os.environ.get("STAGE2_IMAGE", "pytorch-chipyard:stage2")
    image_id = "<resolved-at-launch>"
    lock_fd = None
    if not args.print_command:
        security = subprocess.check_output(docker + ["info", "--format", "{{json .SecurityOptions}}"], env=docker_env, text=True)
        if "name=rootless" in security:
            parser.error("local FPGA control requires rootful system Docker")
        metadata = json.loads(subprocess.check_output(docker + ["image", "inspect", image], env=docker_env, text=True))[0]
        if metadata.get("Config", {}).get("Labels", {}).get("org.pytorch-chipyard.stage2.execution") != "full-container-u250-v1":
            parser.error("build/load the Stage2 container execution image")
        if metadata["Config"]["Labels"].get("org.pytorch-chipyard.stage2.programmer") != "host-vivado-ssh-v1":
            parser.error("build/load the Stage2 image with host Vivado programming")
        image_id = metadata["Id"]
        lock_fd = os.open("/var/lock/firesim-stage2.lock", os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_no_host_simulation()
        device_numbers(db)
    if run.exists():
        if run.is_symlink() or not (run / "run.json").is_file():
            parser.error("RUN_ID already exists without run.json; use a new ID")
    run.mkdir(parents=True, exist_ok=True)
    if not args.print_command:
        results.mkdir(parents=True, exist_ok=True)
        collect_results(state, results)
    for path in (state, state / "runs", run):
        if os.geteuid() == 0:
            os.chown(path, uid, gid)
    for path in (artifact, db):
        if "," in str(path):
            parser.error("bind mount path contains a comma: " + str(path))
    name = "stage2-local-" + str(uid) + "-" + run_id
    command = docker + ["run", "--rm", "--init", "--name", name, "--stop-timeout", "30",
               "--userns=host", "--shm-size=1g", "--cap-add=MKNOD",
               "--device-cgroup-rule=c *:* rwm", "--add-host=host.docker.internal:host-gateway"]

    def mount(source, target, readonly=True):
        command.extend(["--mount", "type=bind,src=" + str(source) + ",dst=" + str(target) + (",readonly" if readonly else "")])

    def environment(key, value):
        command.extend(["-e", key + "=" + str(value)])

    mount(artifact, "/opt/pytorch-chipyard/examples")
    mount(run, run, readonly=False)
    mount(results, results, readonly=False)
    mount(db, "/run/stage2/fpga-db.json")
    mount("/sys", "/sys")
    mount(agent, "/run/stage2/program-agent.sock")
    # Trust this physical host's public host keys, without copying login keys
    # or allowing a first-connection prompt inside an unattended run.
    public_keys = sorted(Path("/etc/ssh").glob("ssh_host_*_key.pub"))
    known_hosts = run / "program-known-hosts"
    lines = []
    for path in public_keys:
        fields = path.read_text().split()
        if len(fields) >= 2 and fields[0] != "ssh-dss":
            lines.append("stage2-programmer " + " ".join(fields[:2]))
    if not lines:
        parser.error("cannot read this host's SSH public host keys in /etc/ssh")
    known_hosts.write_text("\n".join(lines) + "\n")
    ssh_config = run / "program-ssh-config"
    ssh_config.write_text(
        "Host stage2-programmer\n"
        f"  HostName {program_host}\n  User {program_user}\n  Port {program_port}\n"
        "  HostKeyAlias stage2-programmer\n  BatchMode yes\n  StrictHostKeyChecking yes\n"
        "  UserKnownHostsFile /run/stage2/program-known-hosts\n  GlobalKnownHostsFile /dev/null\n"
        "  IdentityAgent /run/stage2/program-agent.sock\n  IdentityFile none\n"
        "  ForwardAgent no\n  ConnectTimeout 15\n  ServerAliveInterval 15\n  ServerAliveCountMax 3\n")
    ssh_config.chmod(0o600)
    mount(known_hosts, "/run/stage2/program-known-hosts")
    mount(ssh_config, "/run/stage2/program-ssh-config")
    hwdb = os.environ.get("FIRESIM_HWDB_PATH")
    if hwdb:
        prepared = run / "launch-hwdb.yaml"
        for source, target in prepare_hwdb(Path(hwdb).resolve(strict=True), prepared):
            mount(source, target)
        mount(prepared, "/run/stage2/config_hwdb.yaml")
        environment("FIRESIM_HWDB_PATH", "/run/stage2/config_hwdb.yaml")
    for key, value in {
        "STAGE2_RUN_DIR": run, "STAGE2_RESULTS_DIR": results, "STAGE2_HOST_LOCKED": 1,
        "STAGE2_HOST_MACHINE_ID": Path("/etc/machine-id").read_text().strip(),
        "STAGE2_IMAGE_ID": image_id, "STAGE2_BUILD_JOBS": jobs,
        "STAGE2_OUTPUT_UID": uid, "STAGE2_OUTPUT_GID": gid,
        "STAGE2_PROGRAM_SSH_CONFIG": "/run/stage2/program-ssh-config",
        "STAGE2_PROGRAM_SSH_HOST": program_host, "STAGE2_PROGRAM_SSH_USER": program_user,
        "STAGE2_PROGRAM_SSH_PORT": program_port, "STAGE2_PROGRAM_VIVADO_DIR": vivado,
        "STAGE2_PROGRAM_FPGA_DB": db,
        "PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE": os.environ.get("PYTORCH_CHIPYARD_FIREMARSHAL_ROOTFS_SIZE", "8GiB"),
        "PYTORCH_CHIPYARD_KEEP_FIREMARSHAL_IMAGES": os.environ.get("PYTORCH_CHIPYARD_KEEP_FIREMARSHAL_IMAGES", "0"),
    }.items():
        environment(key, value)
    if args.smoke_test:
        environment("PYTORCH_CHIPYARD_OMP_FIRST_CPU_SMOKE_GEMM_RVV_4CORE", 0)
    command.extend([image_id if not args.print_command else image, "--run"])
    if args.smoke_test:
        command.append("--workload=" + ",".join(WORKLOADS))
    elif args.workload:
        command.extend("--workload=" + name for name in args.workload)
    elif args.experiment:
        command.append("--experiment=" + args.experiment)
    # Completion filtering is already the workflow default. Do not forward the
    # old --resume flag, which skipped packaging in earlier container images.
    for key in ("rerun_completed", "preflight_only"):
        if getattr(args, key):
            command.append("--" + key.replace("_", "-"))
    command.append("--rvv-panic-retries=" + str(args.rvv_panic_retries))
    print("STAGE2_RUN_ID=" + run_id, flush=True)
    print("STAGE2_RESULTS_DIR=" + str(results), flush=True)
    (run / "launch-command.json").write_text(json.dumps(command, indent=2) + "\n")
    if args.print_command:
        print("DOCKER_HOST=" + shlex.quote(endpoint) + " " + shlex.join(command))
        return

    def stop(signum, frame):
        subprocess.run(docker + ["stop", "--time", "30", name], env=docker_env, check=False)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    status = subprocess.run(command, env=docker_env).returncode
    if status:
        raise SystemExit(status)
    if args.smoke_test and not args.preflight_only:
        for workload in WORKLOADS:
            for output in ("model.log", "autotune.log"):
                path = results / workload / output
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError("missing smoke result: " + str(path))
        print("SMOKE_TEST_STATUS=PASS")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("[stage2-local][error] " + str(error), file=sys.stderr)
        sys.exit(1)

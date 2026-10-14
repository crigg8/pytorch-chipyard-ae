#!/usr/bin/env python3
"""Administrator-only installation of a fixed image/input/FPGA smoke launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
BUNDLE = Path("/usr/local/lib/pytorch-chipyard-stage2-local")
INPUTS = Path("/usr/local/share/pytorch-chipyard-stage2-local/smoke-inputs")
STATE = Path("/var/lib/pytorch-chipyard-stage2-local")
ENTRY = Path("/usr/local/sbin/run-stage2-local-smoke")
CONFIG = Path("/etc/pytorch-chipyard-stage2-local.json")
SUDOERS = Path("/etc/sudoers.d/pytorch-chipyard-stage2-local")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def protected(path):
    for item in (path, *path.parents):
        if item.is_symlink():
            raise RuntimeError("installation path must not be a symlink: " + str(item))
        if item.exists():
            info = item.stat()
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise RuntimeError("installation path must be protected and root-owned: " + str(item))


def install_text(path, text, mode):
    protected(path)
    path.write_text(text)
    path.chmod(mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True, help="delivery.json next to the exported image")
    parser.add_argument("--smoke-inputs", type=Path, required=True)
    parser.add_argument("--user", default="ae")
    parser.add_argument("--firesim-db", type=Path, required=True, help="absolute path of the physical FPGA database")
    parser.add_argument("--ssh-host", default="host.docker.internal")
    parser.add_argument("--vivado", type=Path, default=Path("/home/hongjun/Vivado/2021.1"))
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("an administrator must run this installer as root")
    if not args.firesim_db.is_absolute():
        parser.error("--firesim-db must be an absolute path")
    protected(args.firesim_db)
    db = args.firesim_db.resolve(strict=True)
    if not db.is_file():
        parser.error("missing physical FPGA database: " + str(db))
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", args.user):
        parser.error("invalid account name")
    pwd.getpwnam(args.user)
    delivery = json.loads(args.delivery.read_text())
    for name in ("local-launch.py", "local_runtime.py", "privileged-smoke.py"):
        if sha(REPO / "docker/stage2" / name) != delivery["launcher_sha256"][name]:
            raise RuntimeError("launcher changed since delivery was prepared: " + name)
    image_id = delivery["image_id"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        parser.error("delivery must pin an image ID")
    archive = args.delivery.resolve().parent / delivery["archive"]
    print("Verifying delivered image archive", flush=True)
    if sha(archive) != delivery["archive_sha256"]:
        raise RuntimeError("image archive SHA256 mismatch")
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/root", "LANG": "C.UTF-8",
           "DOCKER_HOST": "unix:///var/run/docker.sock"}
    if subprocess.run(["/usr/bin/docker", "image", "inspect", image_id], env=env,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["/usr/bin/docker", "image", "load", "--input", str(archive)], env=env, check=True)
    metadata = json.loads(subprocess.check_output(["/usr/bin/docker", "image", "inspect", image_id], env=env))[0]
    if metadata["Config"].get("Labels", {}).get("org.pytorch-chipyard.stage2.execution") != "full-container-u250-v1":
        raise RuntimeError("wrong Stage2 execution mode")
    if metadata["Config"]["Labels"].get("org.pytorch-chipyard.stage2.programmer") != "host-vivado-ssh-v1":
        raise RuntimeError("the delivered image must use host Vivado over SSH")
    for path in (BUNDLE, INPUTS, STATE, STATE / "runs", STATE / "docker-config", ENTRY.parent):
        protected(path)
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
    for name in ("local-launch.py", "local_runtime.py", "privileged-smoke.py"):
        install_text(BUNDLE / name, (REPO / "docker/stage2" / name).read_text(), 0o644)
    for backend, cores in (("rvv", 4), ("gemmini", 4), ("scalar", 16)):
        source = args.smoke_inputs / backend
        destination = INPUTS / backend
        protected(destination)
        destination.mkdir(exist_ok=True, mode=0o755)
        required = [source / name for name in (f"model-{cores}core.elf", "input.bin", "weights.bin", "model_spec.json", ".pytorch-chipyard-workload-rel")]
        for path in required:
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise RuntimeError("missing or non-regular smoke input: " + str(path))
        for path in sorted(set(required + list(source.glob(".pytorch-chipyard-*")))):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("non-regular smoke metadata: " + str(path))
            target = destination / path.name
            protected(target)
            shutil.copyfile(path, target)
            target.chmod(0o644)
    install_text(CONFIG, json.dumps({"user": args.user, "image_id": image_id,
                 "smoke_inputs": str(INPUTS), "fpga_db": str(db),
                 "ssh_host": args.ssh_host, "vivado": str(args.vivado.resolve(strict=True))}, indent=2) + "\n", 0o644)
    install_text(ENTRY, '#!/bin/sh\nexec /usr/bin/python3 -I ' + str(BUNDLE / "privileged-smoke.py") + ' "$@"\n', 0o755)
    rule = args.user + " ALL=(root) NOPASSWD: NOSETENV: " + str(ENTRY) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", prefix="stage2-sudoers-", delete=True) as temporary:
        temporary.write(rule)
        temporary.flush()
        subprocess.run(["/usr/sbin/visudo", "-cf", temporary.name], env=env, check=True)
        install_text(SUDOERS, rule, 0o440)
    print("Installed fixed smoke profile for " + args.user, flush=True)
    print('Run: sudo ' + str(ENTRY) + ' --ssh-agent "$SSH_AUTH_SOCK" --preflight-only', flush=True)
    print('Then: sudo ' + str(ENTRY) + ' --ssh-agent "$SSH_AUTH_SOCK" --resume <printed-run-id>', flush=True)
    print("Results: " + str(STATE / "runs") + "/<run-id>/results", flush=True)


if __name__ == "__main__":
    main()

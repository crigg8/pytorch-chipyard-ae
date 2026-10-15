"""Hardware checks and result collection for FireSim running inside one container.

XDMA belongs to the host kernel. This module never loads/unloads modules and
never modifies host device permissions. Device nodes are created in the
container's private /dev after PCI re-enumeration, when their numbers can change.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse

CONTAINER_FPGA_DB = Path("/run/stage2/fpga-db.json")


def collect_results(state, destination):
    """Consolidate legacy per-run outputs without removing their originals.

    Keep the newest complete log pair and its matching files from the same run.
    Run-level status is irrelevant: a failed batch can contain finished jobs.
    """
    destination = Path(destination).resolve()
    candidates = {}

    def complete(directory):
        return all((directory / name).is_file() and not (directory / name).is_symlink()
                   and (directory / name).stat().st_size > 0
                   for name in ("model.log", "autotune.log"))

    def modified(directory):
        return (directory / "model.log").stat().st_mtime_ns

    for source in sorted(Path(state).glob("runs/*/results/*")):
        if not source.is_dir() or source.is_symlink():
            continue
        if not complete(source):
            continue
        previous = candidates.get(source.name)
        if previous is None or modified(source) > modified(previous):
            candidates[source.name] = source

    copied = 0
    for workload, source in candidates.items():
        target = destination / workload
        if source.resolve() == target.resolve():
            continue
        if target.is_symlink():
            raise RuntimeError("collected result must not be a symlink: " + str(target))
        if complete(target) and modified(target) >= modified(source):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and not path.is_symlink():
                shutil.copy2(path, target / path.name)
        copied += 1
    if copied:
        print(f"[stage2-results] collected {copied} workload(s) into {destination}", flush=True)
    return copied


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def agent_socket(path, owner):
    path = Path(path)
    if not path.is_absolute() or "," in str(path):
        raise RuntimeError("SSH agent must be an absolute socket path without commas")
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != owner:
        raise RuntimeError("SSH agent must be a socket owned by the launching account")
    return path


def selected_device(db=None):
    records = json.loads(Path(db or os.environ["FIRESIM_DB"]).read_text())
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("FPGA DB must contain exactly one card")
    record = records[0]
    bdf = record.get("bdf", "")
    if not re.fullmatch(r"[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
        raise RuntimeError("U250 FPGA DB requires a domain-0000 BDF such as 3f:00.0")
    if not record.get("uid"):
        raise RuntimeError("FPGA DB needs the JTAG uid used by Vivado")
    return record, "0000:" + bdf.lower()


def device_numbers(db=None, sys_root=Path("/sys")):
    record, bdf = selected_device(db)
    pci = sys_root / "bus/pci/devices" / bdf
    if (pci / "vendor").read_text().strip().lower() != "0x10ee":
        raise RuntimeError("selected PCI device is not Xilinx: " + bdf)
    if not (sys_root / "module/xdma").is_dir():
        raise RuntimeError("load XDMA on the host before launching the container")
    if not (pci / "driver").is_symlink() or (pci / "driver").resolve().name != "xdma":
        raise RuntimeError("selected PCI device is not bound to XDMA: " + bdf)
    numbers = {}
    for entry in (pci / "xdma").iterdir():
        if re.fullmatch(r"xdma\d+_(?:user|h2c_0|c2h_0)", entry.name):
            major, minor = map(int, (entry / "dev").read_text().strip().split(":"))
            numbers[entry.name] = os.makedev(major, minor)
    ids = {name.split("_", 1)[0] for name in numbers}
    if len(ids) != 1 or len(numbers) != 3:
        raise RuntimeError("selected card needs XDMA user, h2c_0, and c2h_0 devices")
    return record, bdf, numbers


def sync_devices(db=None, sys_root=Path("/sys"), dev_root=Path("/dev")):
    record, bdf, numbers = device_numbers(db, sys_root)
    for name, number in numbers.items():
        path = dev_root / name
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if stat.S_ISCHR(info.st_mode) and info.st_rdev == number:
                continue
            path.unlink()
        os.mknod(path, stat.S_IFCHR | 0o600, number)
    return record, bdf, numbers


def hardware_identity():
    import yaml
    path = Path(os.environ["FIRESIM_HWDB_PATH"])
    hwdb = yaml.safe_load(path.read_text())
    files = {}
    for entry in hwdb.values():
        for field in ("bitstream_tar", "driver_tar"):
            uri = entry.get(field)
            if uri:
                parsed = urlparse(uri)
                if parsed.scheme != "file" or parsed.netloc:
                    raise RuntimeError("local-container HWDB requires local file:// archives: " + uri)
                target = Path(unquote(parsed.path))
                files[str(target)] = sha(target)
    return {"hwdb_sha256": sha(path), "archives": files,
            "fpga_db_sha256": sha(os.environ["FIRESIM_DB"]),
            "host_programming": {key: os.environ.get(key) for key in
                ("STAGE2_HOST_MACHINE_ID", "STAGE2_PROGRAM_FPGA_DB", "STAGE2_PROGRAM_VIVADO_DIR",
                 "STAGE2_PROGRAM_SSH_USER", "STAGE2_PROGRAM_SSH_HOST", "STAGE2_PROGRAM_SSH_PORT")}}


def preflight():
    if os.geteuid() != 0:
        raise RuntimeError("full-container execution needs root in a rootful container")
    mapping = Path("/proc/self/uid_map").read_text().split()
    if mapping != ["0", "0", "4294967295"]:
        raise RuntimeError("rootless/user-remapped containers cannot access host XDMA devices")
    record, bdf, numbers = sync_devices()
    for name in numbers:
        mode = os.O_WRONLY if name.endswith("h2c_0") else os.O_RDONLY if name.endswith("c2h_0") else os.O_RDWR
        # Opening (without reads/writes/mmap) also checks the device cgroup;
        # os.access alone would miss a denied character-device major.
        fd = os.open("/dev/" + name, mode | os.O_NONBLOCK)
        os.close(fd)
    from host_programmer import preflight as host_preflight
    programmer = host_preflight()
    subprocess.run(["ssh", "-F", os.environ["STAGE2_SSH_CONFIG"], "stage2-fpga", "true"], check=True)
    return {"execution": "full-container-u250-v1", "fpga": record,
            "kernel": os.uname().release, "devices": sorted(numbers),
            "host_programming": programmer, "preflight": "passed"}


def copy_results(server, slotno):
    """The run farm is in this container: read raw disks with guestfish."""
    from guestfs_backend import Image
    job = server.get_job()
    destination = Path(server.get_local_job_results_dir_path())
    destination.mkdir(parents=True, exist_ok=True)
    slot = Path(server.get_host_instance().get_sim_dir()) / ("sim_slot_" + str(slotno))
    rootfs = server.get_rootfs_name()
    if rootfs is not None:
        if rootfs.endswith(".qcow2"):
            raise RuntimeError("local Stage2 supports raw rootfs images")
        with Image(slot / rootfs, readonly=True) as image:
            for output in job.outputs:
                image.copy_out(output, destination)
    for output in job.simoutputs:
        source = (slot / output).resolve()
        if slot.resolve() not in source.parents:
            raise RuntimeError("simulation output escapes its slot: " + output)
        if source.is_dir():
            shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, destination / source.name)
    server.write_job_complete_file()


def main():
    action = sys.argv[1]
    if action == "collect-results":
        if len(sys.argv) != 4:
            raise RuntimeError("usage: local_runtime.py collect-results STATE_ROOT RESULTS_DIR")
        collect_results(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    # These CLI helpers run through container SSH and sudo, which discard the
    # manager's environment. The launcher always mounts the selected DB here.
    db = CONTAINER_FPGA_DB
    if action == "devices":
        # Re-enumeration is asynchronous; retry device discovery, never modprobe.
        deadline = time.monotonic() + 15
        while True:
            try:
                sync_devices(db)
                return
            except (OSError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
    elif action == "pci":
        _, bdf, _ = device_numbers(db)
        if len(sys.argv) != 3 or sys.argv[2].lower() != bdf:
            raise RuntimeError("helper requested a different FPGA")
    else:
        raise RuntimeError("unknown local helper action: " + action)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only FPGA host inspection, sent over SSH by the Stage2 manager."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    config = json.loads(sys.argv[1])
    if Path("/etc/machine-id").read_text().strip() != config["machine_id"]:
        raise RuntimeError("SSH target differs from the host holding the Stage2 lock")
    db_path = Path(config["fpga_db"])
    if hashlib.sha256(db_path.read_bytes()).hexdigest() != config["fpga_db_sha256"]:
        raise RuntimeError("host and container FPGA databases differ")
    devices = json.loads(db_path.read_text())
    if len(devices) != 1:
        raise RuntimeError("select a single-card FPGA database")
    bdf = devices[0]["bdf"]
    pci = Path("/sys/bus/pci/devices") / (bdf if bdf.count(":") == 2 else "0000:" + bdf)
    if (pci / "vendor").read_text().strip().lower() != "0x10ee":
        raise RuntimeError(f"selected BDF is not a Xilinx device: {bdf}")
    if not (pci / "driver").is_symlink() or (pci / "driver").resolve().name != "xdma":
        raise RuntimeError(f"selected device is not bound to XDMA: {bdf}")
    if not Path("/sys/module/xdma").exists():
        raise RuntimeError("XDMA must already be loaded by the host administrator")
    vivado = Path(config["vivado"])
    if not os.access(vivado / "bin/vivado", os.X_OK):
        raise RuntimeError(f"Vivado is unavailable at {vivado}")
    for helper in config["helpers"]:
        host_file = Path("/usr/local/bin") / helper["name"]
        if not host_file.is_file() or hashlib.sha256(host_file.read_bytes()).hexdigest() != helper["sha256"]:
            raise RuntimeError(f"host helper differs from image: {host_file}")
        if not os.access(host_file, os.X_OK):
            raise RuntimeError(f"helper is not executable: {host_file}")
        subprocess.run(["sudo", "-n", "-l", str(host_file)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for executable in ["screen", "rsync", "bash", "flock", "lspci"]:
        if not __import__("shutil").which(executable):
            raise RuntimeError(f"missing host command: {executable}")
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            name = (entry / "comm").read_text().strip()
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # Without visibility, the absence of another simulation is unknown.
            raise RuntimeError("cannot inspect other host processes for the global execution guard")
        manager = any(Path(os.fsdecode(arg)).name == "firesim" for arg in command[:3] if arg)
        if name.startswith("FireSim-") or manager:
            active.append({"pid": entry.name, "name": name})
    if active:
        raise RuntimeError(f"another FireSim job is active; no processes were stopped: {active}")
    if not os.access(config["run_dir"], os.W_OK):
        raise RuntimeError("SSH user cannot write the shared run directory")
    print(json.dumps({"fpga": devices[0], "kernel": os.uname().release,
                      "xdma_poll_mode": Path("/sys/module/xdma/parameters/poll_mode").read_text().strip(),
                      "vivado": str(vivado), "host_preflight": "passed"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Stage2 host preflight failed: {error}", file=sys.stderr)
        sys.exit(1)

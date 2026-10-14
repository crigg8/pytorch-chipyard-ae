#!/usr/bin/env python3
"""Ephemeral host-side receiver. Only the existing FPGA helper needs sudo."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HELPER = Path("/usr/local/bin/firesim-fpga-util.py")


def validate(config):
    if Path("/etc/machine-id").read_text().strip() != config["machine_id"]:
        raise RuntimeError("programming SSH target is not the host holding the FPGA lock")
    database = Path(config["fpga_db"]).read_bytes()
    if hashlib.sha256(database).hexdigest() != config["fpga_db_sha256"]:
        raise RuntimeError("host and container FPGA databases differ")
    records = json.loads(database)
    if len(records) != 1 or records[0]["bdf"] != config["bdf"]:
        raise RuntimeError("programming request must select exactly the configured FPGA")
    pci = Path("/sys/bus/pci/devices") / ("0000:" + config["bdf"])
    if (pci / "vendor").read_text().strip() != "0x10ee":
        raise RuntimeError("selected PCI device is not Xilinx")
    if not Path("/sys/module/xdma").is_dir() or (pci / "driver").resolve().name != "xdma":
        raise RuntimeError("host XDMA must already be loaded and bound")
    vivado = Path(config["vivado"])
    for path in (vivado / "bin/vivado", vivado / "bin/hw_server", HELPER):
        if not os.access(path, os.X_OK):
            raise RuntimeError("missing host programming tool: " + str(path))
    helpers = {}
    for name in (HELPER.name, "pcielib.py", "util.py", "program_fpga.tcl"):
        path = HELPER.parent / name
        helpers[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    subprocess.run(["/usr/bin/sudo", "-n", "-l", str(HELPER)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"machine_id": config["machine_id"], "fpga": records[0], "vivado": str(vivado),
            "helpers": helpers, "programmer": "host-vivado-ssh-v1", "preflight": "passed"}


def receive(stream, destination, size, digest):
    if not isinstance(size, int) or not 0 < size <= 256 * 1024 * 1024:
        raise RuntimeError("bitstream size must be between 1 byte and 256 MiB")
    value = hashlib.sha256()
    remaining = size
    with destination.open("xb") as output:
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            if not block:
                raise RuntimeError("truncated bitstream transfer")
            output.write(block)
            value.update(block)
            remaining -= len(block)
        if stream.read(1) or value.hexdigest() != digest:
            raise RuntimeError("bitstream transfer SHA256/length mismatch")


def main():
    config = json.loads(sys.argv[1])
    identity = validate(config)
    if "bitstream_sha256" not in config:
        print(json.dumps(identity))
        return
    # Keep Vivado logs for diagnosis. Its sudo-created .Xil files can be owned
    # by root, so do not depend on the SSH user recursively deleting them.
    cache = Path.home() / ".cache/pytorch-chipyard-stage2-program"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = tempfile.mkdtemp(prefix="program-", dir=cache)
    bitstream = Path(temporary) / "firesim.bit"
    try:
        receive(sys.stdin.buffer, bitstream, config["bitstream_bytes"], config["bitstream_sha256"])
        vivado = Path(config["vivado"])
        command = ["/usr/bin/sudo", "-n", str(HELPER), "--bitstream", str(bitstream),
                   "--bdf", config["bdf"], "--fpga-db", config["fpga_db"],
                   "--vivado-bin", str(vivado / "bin/vivado"),
                   "--hw-server-bin", str(vivado / "bin/hw_server")]
        subprocess.run(command, cwd=temporary, stdin=subprocess.DEVNULL, check=True)
    finally:
        if bitstream.exists():
            bitstream.unlink()
        print("[stage2-host-program] host programming logs: " + temporary, flush=True)
    print("[stage2-host-program] FPGA programmed; XDMA module kept loaded", flush=True)


if __name__ == "__main__":
    main()

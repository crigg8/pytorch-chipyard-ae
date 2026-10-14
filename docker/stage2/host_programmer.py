"""Send a checked bitstream to host Vivado; the simulator stays in Docker."""
import json
import os
from pathlib import Path
import shlex
import subprocess

from local_runtime import selected_device, sha


def configuration():
    record, _ = selected_device()
    return {"machine_id": os.environ["STAGE2_HOST_MACHINE_ID"],
            "fpga_db": os.environ["STAGE2_PROGRAM_FPGA_DB"],
            "fpga_db_sha256": sha(os.environ["FIRESIM_DB"]),
            "bdf": record["bdf"], "vivado": os.environ["STAGE2_PROGRAM_VIVADO_DIR"]}


def receiver_source():
    return Path(__file__).with_name("host-program.py").read_text()


def request(config, bitstream=None):
    # Code is an SSH command argument; stdin is reserved for the binary payload.
    # -I ignores host/user Python configuration and needs no host Conda/Chipyard.
    script = receiver_source()
    remote = shlex.join(["/usr/bin/python3", "-I", "-c", script, json.dumps(config)])
    command = ["/usr/bin/ssh", "-F", os.environ["STAGE2_PROGRAM_SSH_CONFIG"],
               "stage2-programmer", remote]
    env = dict(os.environ)
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    if bitstream is None:
        response = subprocess.check_output(command, stdin=subprocess.DEVNULL, env=env, text=True, timeout=60)
        return json.loads(response)
    with Path(bitstream).open("rb") as payload:
        subprocess.run(command, stdin=payload, env=env, check=True, timeout=900)


def preflight():
    return request(configuration())


def program(bitstream, bdf):
    config = configuration()
    if config["bdf"].lower() != bdf.lower():
        raise RuntimeError("programming request selects a different FPGA")
    bitstream = Path(bitstream)
    config.update(bitstream_sha256=sha(bitstream), bitstream_bytes=bitstream.stat().st_size)
    print("[stage2-local] programming host FPGA " + bdf + " over SSH", flush=True)
    request(config, bitstream)

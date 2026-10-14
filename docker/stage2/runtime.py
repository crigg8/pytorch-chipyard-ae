#!/usr/bin/env python3
"""Stage2 metadata, read-only preflight, and guest ELF dependency checks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from guestfs_backend import Image

SUPPORT = Path(__file__).resolve().parent
META = Path("/opt/stage2-metadata")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def readelf(path, option):
    return subprocess.check_output(["/usr/bin/readelf", option, str(path)],
                                   text=True, env=dict(os.environ, LC_ALL="C"))


def elf_header(path, workload=""):
    header = readelf(path, "-h")
    for required in ["ELF64", "little endian", "RISC-V", "double-float ABI"]:
        if required not in header:
            raise RuntimeError(f"unsupported ELF ({required} required): {path}")
    attributes = readelf(path, "-A")
    arch = re.search(r'Tag_RISCV_arch:\s*"([^"]+)"', attributes)
    if arch:
        vector = any(re.fullmatch(r"v\d+p\d+", part) for part in arch.group(1).split("_"))
        if vector and workload and "-rvv-" not in workload:
            raise RuntimeError(f"vector ELF selected for a non-RVV workload: {workload}")
        widths = [int(value) for value in re.findall(r"zvl(\d+)b", arch.group(1))]
        if "-rvv-" in workload and widths and max(widths) > 128:
            raise RuntimeError(f"ELF requires VLEN > 128: {path}")
    return arch.group(1) if arch else None


def inputs():
    root = Path("/opt/pytorch-chipyard/examples")
    elves = sorted(root.rglob("model-*core.elf"))
    if not elves:
        raise RuntimeError("no model-*core.elf files; Stage1 must finish ELF compilation first")
    files = set()
    for elf in elves:
        elf_header(elf)
        files.add(elf)
        for name in ["input.bin", "weights.bin", "model_spec.json"]:
            path = elf.parent / name
            if not path.is_file():
                raise RuntimeError(f"missing Stage1 input: {path}")
            files.add(path)
        files.update(elf.parent.glob(".pytorch-chipyard-*"))
        json.loads((elf.parent / "model_spec.json").read_text())
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
             "sha256": sha(path)} for path in sorted(files) if path.is_file()]


def begin(args):
    state = Path(os.environ["STAGE2_RUN_DIR"])
    entries = inputs()
    identity = {"inputs": entries, "snapshot": (META / "context-id").read_text().strip(),
                "image_id": os.environ["STAGE2_IMAGE_ID"]}
    if os.environ.get("STAGE2_EXECUTION") == "local":
        from local_runtime import hardware_identity
        identity.update(execution="full-container-u250-v1", hardware=hardware_identity())
    path = state / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["identity"] != identity:
            raise RuntimeError("inputs or image changed; use a new RUN_ID to avoid stale completion markers")
        record = old
    else:
        record = {"identity": identity, "attempts": []}
    record["attempts"].append({
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arguments": args.arguments, "firesim_db": os.environ["STAGE2_PROGRAM_FPGA_DB"],
        "host": os.environ["FIRESIM_SSH_HOST"], "user": os.environ["FIRESIM_SSH_USER"],
        "status": "preparing"})
    write_json(path, record)


def finish(args):
    path = Path(os.environ["STAGE2_RUN_DIR"]) / "run.json"
    if not path.exists():
        return
    record = json.loads(path.read_text())
    status = "failed"
    if args.code == 0:
        status = "preflight-passed" if "--preflight-only" in record["attempts"][-1]["arguments"] else "completed"
    record["attempts"][-1].update(ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  exit_code=args.code, status=status)
    write_json(path, record)


def preflight(args):
    if os.environ.get("STAGE2_EXECUTION") == "local":
        from local_runtime import preflight as local_preflight
        write_json(Path(os.environ["STAGE2_RUN_DIR"]) / "host.json", local_preflight())
        return
    db = Path(os.environ["PYTORCH_CHIPYARD_FPGA_DB"])
    records = json.loads(db.read_text())
    if len(records) != 1 or not re.fullmatch(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", records[0]["bdf"]):
        raise RuntimeError("FPGA database must contain one valid PCI BDF")
    helpers = []
    platform = Path(os.environ["FIRESIM_DIR"]) / "platforms/xilinx_alveo_u250/scripts"
    common = Path(os.environ["FIRESIM_DIR"]) / "deploy/sudo-scripts"
    for name in ["firesim-load-xdma-module", "firesim-remove-xdma-module",
                 "firesim-chmod-xdma-perm", "firesim-change-pcie-perms",
                 "firesim-mount-with-uid-gid", "firesim-unmount", "firesim-fpga-util.py"]:
        path = (platform if name.endswith(".py") else common) / name
        helpers.append({"name": name, "sha256": sha(path)})
    config = {"machine_id": os.environ["STAGE2_HOST_MACHINE_ID"], "fpga_db": str(db),
              "fpga_db_sha256": sha(db), "helpers": helpers,
              "vivado": os.environ["FIRESIM_VIVADO_DIR"], "run_dir": os.environ["STAGE2_RUN_DIR"]}
    ssh = ["ssh", "-F", os.environ["STAGE2_SSH_CONFIG"], "stage2-fpga",
           shlex.join(["python3", "-", json.dumps(config)])]
    result = subprocess.check_output(ssh, input=(SUPPORT / "host-preflight.py").read_text(), text=True)
    write_json(Path(os.environ["STAGE2_RUN_DIR"]) / "host.json", json.loads(result))
    for artifact in json.loads((META / "hardware.json").read_text()):
        path = Path("/opt/firesim-artifacts") / artifact["hardware_config"] / artifact["file"]
        if sha(path) != artifact["sha256"]:
            raise RuntimeError(f"hardware bundle hash mismatch: {path}")


def needed(path):
    return re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", readelf(path, "-d"))


def versions_needed(path):
    text = readelf(path, "--version-info")
    if "Version needs section" not in text:
        return {}
    result = {}
    current = None
    for line in text.split("Version needs section", 1)[1].splitlines():
        file = re.search(r"File:\s*(\S+)", line)
        if file:
            current = file.group(1)
            result.setdefault(current, set())
        name = re.search(r"Name:\s*(\S+)", line)
        if current and name:
            result[current].add(name.group(1))
    return result


def check_image(args):
    workload = args.workload
    if not re.fullmatch(r"[A-Za-z0-9._-]+", workload):
        raise ValueError("invalid workload")
    root = Path(os.environ["PYTORCH_CHIPYARD_WORKLOAD_DIR"]) / ("overlay-" + workload)
    elves = sorted(root.rglob("*.elf"))
    if len(elves) != 1:
        raise RuntimeError(f"expected one packaged ELF for {workload}, found {len(elves)}")
    elf = elves[0]
    arch = elf_header(elf, workload)
    image = Path(os.environ["FIREMARSHAL_IMAGE_DIR"]) / workload / (workload + ".img")
    with Image(image, readonly=True) as filesystem, tempfile.TemporaryDirectory(prefix="stage2-abi-") as temporary:
        guest_elf = "/" + str(elf.relative_to(root))
        if filesystem.command("checksum", "sha256", guest_elf) != sha(elf):
            raise RuntimeError(f"rootfs ELF differs from packaged ELF: {workload}")
        for blob in root.rglob("*.bin"):
            guest_blob = "/" + str(blob.relative_to(root))
            if filesystem.command("checksum", "sha256", guest_blob) != sha(blob):
                raise RuntimeError(f"rootfs input differs from packaged input: {blob}")
        program = readelf(elf, "-l")
        interpreter = re.search(r"Requesting program interpreter:\s*([^\]]+)", program)
        checked = {}

        def fetch(guest):
            if filesystem.command("is-file", guest, "followsymlinks:true") != "true":
                raise RuntimeError(f"guest library/loader missing: {guest}")
            local = Path(temporary) / hashlib.sha256(guest.encode()).hexdigest()
            filesystem.command("download", guest, local)
            elf_header(local)
            return local

        if interpreter:
            fetch(interpreter.group(1).strip())
        queue = [elf]
        while queue:
            binary = queue.pop()
            required_versions = versions_needed(binary)
            for library in needed(binary):
                if library not in checked:
                    candidates = [directory + "/" + library for directory in
                                  ["/lib", "/usr/lib", "/lib64", "/usr/lib64", "/lib/riscv64-linux-gnu", "/usr/lib/riscv64-linux-gnu"]]
                    guest = next((candidate for candidate in candidates
                                  if filesystem.command("is-file", candidate, "followsymlinks:true") == "true"), None)
                    if guest is None:
                        raise RuntimeError(f"guest rootfs lacks {library}, required by {binary.name}")
                    local = fetch(guest)
                    definition = readelf(local, "--version-info").split("Version needs section", 1)[0]
                    checked[library] = set(re.findall(r"Name:\s*(\S+)", definition))
                    queue.append(local)
                missing = required_versions.get(library, set()) - checked[library]
                if missing:
                    raise RuntimeError(f"guest {library} lacks required symbol versions: {sorted(missing)}")
    dest = Path(os.environ["STAGE2_RUN_DIR"]) / "abi"
    dest.mkdir(exist_ok=True)
    write_json(dest / (workload + ".json"), {"elf_sha256": sha(elf), "riscv_arch": arch,
                                             "libraries": sorted(checked), "status": "passed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("begin")
    start.add_argument("arguments", nargs=argparse.REMAINDER)
    stop = sub.add_parser("finish")
    stop.add_argument("code", type=int)
    sub.add_parser("preflight")
    check = sub.add_parser("check-image")
    check.add_argument("workload")
    args = parser.parse_args()
    {"begin": begin, "finish": finish, "preflight": preflight, "check-image": check_image}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[stage2-container] {error}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Install the container workflow and record the source-built image inputs."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_container_workflow(workflow, project):
    for line in (workflow / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        source = workflow / name
        if source.parent != workflow or sha(source) != expected:
            raise RuntimeError("container workflow checksum mismatch: " + name)
        shutil.copy2(source, project / "scripts" / name)


def main():
    support = Path("/opt/stage2-support")
    project = Path("/opt/pytorch-chipyard")
    metadata = Path("/opt/stage2-metadata")
    metadata.mkdir()
    (project / "examples").mkdir()
    install_container_workflow(support / "workflow", project)
    hardware = []
    for path in sorted(Path("/opt/firesim-artifacts").glob("*/*.tar.gz")):
        hardware.append({"hardware_config": path.parent.name, "file": path.name,
                         "bytes": path.stat().st_size, "sha256": sha(path)})
    (metadata / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    manifest = {}
    for root in (support, project / "scripts"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                manifest[str(path.relative_to("/opt"))] = sha(path)
    manifest["hardware"] = hardware
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (metadata / "source-build.json").write_text(encoded)
    (metadata / "context-id").write_text(hashlib.sha256(encoded.encode()).hexdigest() + "\n")


if __name__ == "__main__":
    main()

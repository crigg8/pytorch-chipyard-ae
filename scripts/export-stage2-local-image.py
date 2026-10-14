#!/usr/bin/env python3
"""Export a Docker image with individually compressed layers for docker load.

Unlike gzip around a whole docker-save archive, this keeps the loader's
temporary layer files compressed. The image config and uncompressed layer
digests remain unchanged. This exporter accepts Docker's blobs/sha256 save
layout and requires pigz. The output is an ordinary, seekable .tar file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delivery_record(metadata, archive):
    return {
        "image_id": metadata["Id"],
        "archive": archive.name,
        "archive_sha256": sha(archive),
        "programmer": "host-vivado-ssh-v1",
        "launcher_sha256": {
            name: sha(REPO / "docker/stage2" / name)
            for name in ("local-launch.py", "local_runtime.py", "privileged-smoke.py")
        },
    }


def header(name, size):
    entry = tarfile.TarInfo(name)
    entry.size = size
    entry.mode = 0o644
    return entry.tobuf(format=tarfile.GNU_FORMAT)


def write_file(output, name, data):
    output.write(header(name, len(data)))
    output.write(data)
    output.write(b"\0" * (-len(data) % 512))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="pytorch-chipyard:stage2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delivery", type=Path,
                        help="write an installer manifest next to the image archive")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.delivery and (args.delivery.exists() or args.delivery.resolve() == args.output.resolve()
                          or args.delivery.resolve().parent != args.output.resolve().parent):
        parser.error("delivery must be a new file next to the archive")
    metadata = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    config_id = metadata["Id"].split(":", 1)[1]
    diff_ids = [value.split(":", 1)[1] for value in metadata["RootFS"]["Layers"]]
    expected = {"blobs/sha256/" + value: value for value in diff_ids}
    seen = set()
    config = None
    output = args.output.open("xb", buffering=0)
    saved = subprocess.Popen(["docker", "image", "save", metadata["Id"]], stdout=subprocess.PIPE)
    try:
        with output, tarfile.open(fileobj=saved.stdout, mode="r|") as archive:
            for entry in archive:
                if entry.name == "blobs/sha256/" + config_id:
                    config = archive.extractfile(entry).read()
                    continue
                digest = expected.get(entry.name)
                if digest is None:
                    continue
                print("Compressing layer " + digest[:12] + " (" + str(entry.size) + " bytes)", flush=True)
                name = "layers/" + digest + ".tar.gz"
                start = output.tell()
                output.write(header(name, 0))
                checksum = hashlib.sha256()
                compressor = subprocess.Popen(["pigz", "-1", "-p", "4", "-n"], stdin=subprocess.PIPE, stdout=output)
                try:
                    source = archive.extractfile(entry)
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        checksum.update(block)
                        compressor.stdin.write(block)
                    compressor.stdin.close()
                    if compressor.wait() != 0:
                        raise RuntimeError("layer compressor failed")
                finally:
                    if compressor.poll() is None:
                        compressor.terminate()
                        compressor.wait()
                if checksum.hexdigest() != digest:
                    raise RuntimeError("saved layer digest mismatch: " + digest)
                end = output.tell()
                size = end - start - 512
                output.seek(start)
                output.write(header(name, size))
                output.seek(end)
                output.write(b"\0" * (-size % 512))
                seen.add(digest)
            if saved.wait() != 0:
                raise RuntimeError("docker image save failed")
            if seen != set(diff_ids) or config is None:
                raise RuntimeError("incomplete archive or unsupported Docker save layout")
            if hashlib.sha256(config).hexdigest() != config_id:
                raise RuntimeError("saved image config digest mismatch")
            config_name = "config/" + config_id + ".json"
            write_file(output, config_name, config)
            manifest = [{"Config": config_name, "RepoTags": metadata.get("RepoTags") or [],
                         "Layers": ["layers/" + value + ".tar.gz" for value in diff_ids]}]
            write_file(output, "manifest.json", json.dumps(manifest).encode())
            output.write(b"\0" * 1024)
        print("STAGE2_EXPORTED_IMAGE_ID=" + metadata["Id"], flush=True)
        print("STAGE2_IMAGE_ARCHIVE=" + str(args.output), flush=True)
        if args.delivery:
            args.delivery.write_text(json.dumps(delivery_record(metadata, args.output), indent=2) + "\n")
            print("STAGE2_DELIVERY=" + str(args.delivery), flush=True)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    finally:
        saved.stdout.close()
        if saved.poll() is None:
            saved.terminate()
            saved.wait()


if __name__ == "__main__":
    main()

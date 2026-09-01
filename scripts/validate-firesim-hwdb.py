#!/usr/bin/env python3
"""Validate the local artifacts used by selected FireSim HWDB entries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
from typing import Any
from urllib.parse import unquote, urlparse

import yaml


class ValidationError(RuntimeError):
    pass


def local_path(uri: Any, field: str) -> Path | None:
    if uri is None:
        return None
    if not isinstance(uri, str) or not uri.strip():
        raise ValidationError(f"{field} must be a non-empty string or null")

    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValidationError(
                f"{field} uses unsupported file URI host '{parsed.netloc}'"
            )
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme:
        return None
    return Path(uri).expanduser()


def require_local_file(path: Path, field: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValidationError(f"{field} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValidationError(f"{field} is not a regular file: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValidationError(f"{field} is empty: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise ValidationError(f"{field} is not readable: {resolved}")
    return resolved


def platform_for(entry: dict[str, Any]) -> str | None:
    quintuplet = entry.get("deploy_quintuplet")
    if isinstance(quintuplet, dict):
        value = quintuplet.get("platform")
        return value if isinstance(value, str) and value else None

    override = entry.get("deploy_quintuplet_override")
    if isinstance(override, str) and override:
        return override.split("-", 1)[0]
    return None


def validate_bitstream_archive(path: Path, entry: dict[str, Any]) -> None:
    platform = platform_for(entry)
    expected_bit = f"{platform}/firesim.bit" if platform else None
    expected_metadata = f"{platform}/metadata" if platform else None

    try:
        with tarfile.open(path, "r:*") as archive:
            members = {
                member.name.removeprefix("./").rstrip("/"): member
                for member in archive.getmembers()
            }
            if expected_bit and expected_bit not in members:
                raise ValidationError(
                    f"bitstream_tar is missing {expected_bit}: {path}"
                )
            if not any(name.endswith("/firesim.bit") for name in members):
                raise ValidationError(f"bitstream_tar has no firesim.bit: {path}")

            quintuplet = entry.get("deploy_quintuplet")
            if expected_metadata and isinstance(quintuplet, dict):
                member = members.get(expected_metadata)
                if member is None:
                    raise ValidationError(
                        f"bitstream_tar is missing {expected_metadata}: {path}"
                    )
                metadata_file = archive.extractfile(member)
                if metadata_file is None:
                    raise ValidationError(
                        f"could not read {expected_metadata} from {path}"
                    )
                metadata = metadata_file.read(64 * 1024).decode(
                    "utf-8", errors="replace"
                )
                for key in ("target_config", "platform_config"):
                    expected = quintuplet.get(key)
                    if isinstance(expected, str) and expected not in metadata:
                        raise ValidationError(
                            f"bitstream metadata does not match {key}={expected}: {path}"
                        )
    except (tarfile.TarError, OSError) as error:
        raise ValidationError(f"could not read bitstream_tar {path}: {error}") from error


def validate_driver_archive(path: Path, entry: dict[str, Any]) -> None:
    platform = platform_for(entry)
    expected_driver = f"FireSim-{platform}" if platform else None
    try:
        with tarfile.open(path, "r:*") as archive:
            basenames = {Path(member.name).name for member in archive.getmembers()}
    except (tarfile.TarError, OSError) as error:
        raise ValidationError(f"could not read driver_tar {path}: {error}") from error

    if expected_driver and expected_driver not in basenames:
        raise ValidationError(
            f"driver_tar is missing {expected_driver}: {path}"
        )
    if not any(name.startswith("FireSim-") for name in basenames):
        raise ValidationError(f"driver_tar has no FireSim driver executable: {path}")


def validate_entry(name: str, entry: Any) -> str:
    if not isinstance(entry, dict):
        raise ValidationError(f"hardware config '{name}' is not a mapping")

    bitstream_uri = entry.get("bitstream_tar")
    if bitstream_uri is None:
        raise ValidationError(f"hardware config '{name}' has no bitstream_tar")
    bitstream = local_path(bitstream_uri, "bitstream_tar")
    if bitstream is None:
        bitstream_status = f"remote:{bitstream_uri}"
    else:
        bitstream = require_local_file(bitstream, "bitstream_tar")
        validate_bitstream_archive(bitstream, entry)
        bitstream_status = str(bitstream)

    driver_uri = entry.get("driver_tar")
    if driver_uri is None:
        if not entry.get("deploy_quintuplet") and not entry.get(
            "deploy_quintuplet_override"
        ):
            raise ValidationError(
                f"hardware config '{name}' cannot build a driver without a deploy quintuplet"
            )
        driver_status = "build-local"
    else:
        driver = local_path(driver_uri, "driver_tar")
        if driver is None:
            driver_status = f"prebuilt-remote:{driver_uri}"
        else:
            driver = require_local_file(driver, "driver_tar")
            validate_driver_archive(driver, entry)
            driver_status = f"prebuilt:{driver}"

    return f"bitstream={bitstream_status} driver={driver_status}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hwdb", required=True, type=Path)
    parser.add_argument("--config", required=True, action="append", dest="configs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.hwdb.open("r", encoding="utf-8") as stream:
            hwdb = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        print(f"[firesim-hwdb][FAIL] could not load {args.hwdb}: {error}")
        return 1

    if not isinstance(hwdb, dict):
        print(f"[firesim-hwdb][FAIL] HWDB is not a mapping: {args.hwdb}")
        return 1

    status = 0
    for name in dict.fromkeys(args.configs):
        if name not in hwdb:
            print(
                f"[firesim-hwdb][FAIL] hardware config '{name}' "
                f"not found in {args.hwdb}"
            )
            status = 1
            continue
        try:
            result = validate_entry(name, hwdb[name])
        except ValidationError as error:
            print(f"[firesim-hwdb][FAIL] {name}: {error}")
            status = 1
        else:
            print(f"[firesim-hwdb][PASS] {name}: {result}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

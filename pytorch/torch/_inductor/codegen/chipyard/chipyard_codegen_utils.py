from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from typing import Optional, Union

from ...codecache import cache_dir, get_path, write, write_atomic


def chipyard_dump_path() -> Optional[str]:
    dump_path = os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH", "").strip()
    if not dump_path:
        return None
    os.makedirs(dump_path, exist_ok=True)
    return dump_path


def chipyard_artifact_dir() -> str:
    return chipyard_dump_path() or cache_dir()


def chipyard_kernel_dump_dir() -> Optional[str]:
    dump_path = chipyard_dump_path()
    if dump_path is None:
        return None
    kernels_path = os.path.join(dump_path, "kernels")
    os.makedirs(kernels_path, exist_ok=True)
    return kernels_path


def _chipyard_kernel_subdir(source_parent: str) -> str:
    parent_name = os.path.basename(source_parent.rstrip(os.sep)) or "kernel"
    digest = hashlib.sha256(source_parent.encode("utf-8")).hexdigest()[:10]
    sanitized_name = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in parent_name
    )
    return f"{sanitized_name}_{digest}"


def chipyard_stage_kernel_artifacts(
    metadata_group: Mapping[str, str],
    *,
    staged_paths: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    normalized_group = {str(name): str(path) for name, path in metadata_group.items()}
    kernels_path = chipyard_kernel_dump_dir()
    if kernels_path is None:
        return normalized_group

    staged_paths = {} if staged_paths is None else staged_paths
    staged_group: dict[str, str] = {}

    for artifact_name, artifact_path in normalized_group.items():
        source_path = os.path.abspath(artifact_path)
        existing_path = staged_paths.get(source_path)
        if existing_path is not None:
            staged_group[artifact_name] = existing_path
            continue

        if not os.path.exists(source_path):
            staged_group[artifact_name] = artifact_path
            continue

        source_parent = os.path.dirname(source_path)
        dest_dir = os.path.join(kernels_path, _chipyard_kernel_subdir(source_parent))
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))

        if os.path.isdir(source_path):
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
        elif source_path != dest_path:
            shutil.copy2(source_path, dest_path)

        staged_paths[source_path] = dest_path
        staged_group[artifact_name] = dest_path

    return staged_group


def chipyard_write(
    content: Union[str, bytes],
    extension: str,
    *,
    extra: str = "",
    hash_type: str = "code",
    key: Optional[str] = None,
) -> tuple[str, str]:
    dump_path = chipyard_dump_path()
    if dump_path is not None:
        return write(
            content,
            extension,
            extra=extra,
            hash_type=hash_type,
            specified_dir=dump_path,
            key=key,
        )
    return write(content, extension, extra=extra, hash_type=hash_type, key=key)


def chipyard_get_path(basename: str, extension: str) -> tuple[str, str, str]:
    dump_path = chipyard_dump_path()
    if dump_path is not None:
        return get_path(basename, extension, specified_dir=dump_path)
    return get_path(basename, extension)


def chipyard_named_path(filename: str) -> Optional[str]:
    dump_path = chipyard_dump_path()
    if dump_path is None:
        return None
    os.makedirs(dump_path, exist_ok=True)
    return os.path.join(dump_path, filename)


def chipyard_named_write(content: Union[str, bytes], filename: str) -> str:
    named_path = chipyard_named_path(filename)
    if named_path is not None:
        write_atomic(named_path, content, make_dirs=True)
        return named_path
    extension = filename.rsplit(".", 1)[-1]
    return chipyard_write(content, extension)[1]

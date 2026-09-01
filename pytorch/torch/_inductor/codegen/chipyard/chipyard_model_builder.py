from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from typing import Any, Optional

from ...codecache import output_code_log, write_atomic
from .chipyard_codegen_utils import (
    chipyard_get_path,
    chipyard_named_path,
    chipyard_named_write,
    chipyard_stage_kernel_artifacts,
)


def _chipyard_model_build_enabled() -> bool:
    return os.environ.get("TORCHINDUCTOR_COMPILE_CHIPYARD_MODEL_RUNNER", "") == "1"


def _chipyard_stage_kernel_artifacts_enabled() -> bool:
    staged = os.environ.get("TORCHINDUCTOR_STAGE_CHIPYARD_KERNEL_ARTIFACTS", "").strip()
    if staged:
        return staged == "1"
    return _chipyard_model_build_enabled()


def _runner_build_metadata_group(
    metadata_group: Mapping[str, str],
) -> dict[str, str]:
    filtered_group: dict[str, str] = {}
    for artifact_name, artifact_path in metadata_group.items():
        artifact_name = str(artifact_name)
        artifact_path = str(artifact_path)
        if artifact_name.endswith((".o", ".obj", ".a")) or artifact_path.endswith(
            (".o", ".obj", ".a")
        ):
            filtered_group[artifact_name] = artifact_path
    return filtered_group


def _materialize_model_spec(model_spec: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    if isinstance(model_spec, Mapping):
        return copy.deepcopy(dict(model_spec))
    with open(os.fspath(model_spec)) as spec_file:
        return json.load(spec_file)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(inner) for inner in value]
    return str(value)


def _config_to_dict(config_obj: Any) -> dict[str, Any]:
    if config_obj is None:
        return {}
    try:
        from ...runtime.triton_heuristics import config_to_dict

        return _json_safe(config_to_dict(config_obj))
    except Exception:
        return {"repr": repr(config_obj)}


def _compiled_metadata_group(obj: Any) -> Optional[dict[str, str]]:
    metadata_group = getattr(obj, "metadata_group", None)
    if not isinstance(metadata_group, dict):
        return None
    return {str(name): str(path) for name, path in metadata_group.items()}


def _compiled_kernel_variants(compiled_kernel: Any) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for compile_result in getattr(compiled_kernel, "compile_results", None) or []:
        variant: dict[str, Any] = {}
        config_obj = getattr(compile_result, "config", None)
        if config_obj is not None:
            variant["config"] = repr(config_obj)
            variant["config_dict"] = _config_to_dict(config_obj)
        kernel_obj = getattr(compile_result, "kernel", None)
        for candidate in (compile_result, kernel_obj):
            if candidate is None:
                continue
            metadata_group = _compiled_metadata_group(candidate)
            if metadata_group is not None:
                variant["metadata_group"] = metadata_group
                break
        if kernel_obj is not None:
            for attr_name in ("cache_key", "kernel_hash", "hash"):
                value = getattr(kernel_obj, attr_name, None)
                if value is not None:
                    variant[attr_name] = str(value)
                    break
        if variant:
            variants.append(variant)
    return variants


def _compiled_selected_launch(compiled_kernel: Any) -> Optional[dict[str, Any]]:
    launchers = getattr(compiled_kernel, "launchers", None) or []
    selected_config = (
        getattr(launchers[0], "config", None) if len(launchers) == 1 else None
    )
    compile_results = getattr(compiled_kernel, "compile_results", None) or []
    selected_result = None
    selected_config_dict = _config_to_dict(selected_config)
    if selected_config is not None:
        for compile_result in compile_results:
            if (
                _config_to_dict(getattr(compile_result, "config", None))
                == selected_config_dict
            ):
                selected_result = compile_result
                break
    elif compile_results:
        selected_result = compile_results[0]
        selected_config = getattr(selected_result, "config", None)
        selected_config_dict = _config_to_dict(selected_config)

    metadata_group = None
    for candidate in (
        selected_result,
        (
            getattr(selected_result, "kernel", None)
            if selected_result is not None
            else None
        ),
    ):
        if candidate is None:
            continue
        metadata_group = _compiled_metadata_group(candidate)
        if metadata_group is not None:
            break

    inductor_meta = getattr(selected_result, "inductor_meta", None)
    if inductor_meta is None:
        inductor_meta = getattr(compiled_kernel, "inductor_meta", None)
    if selected_config is None and inductor_meta is None and metadata_group is None:
        return None

    selected_launch: dict[str, Any] = {}
    if selected_config is not None:
        selected_launch["config"] = repr(selected_config)
        selected_launch["config_dict"] = selected_config_dict
    if inductor_meta is not None:
        selected_launch["inductor_meta"] = _json_safe(inductor_meta)
    if metadata_group is not None:
        selected_launch["metadata_group"] = metadata_group
    return selected_launch


def _compiled_custom_library_entry(obj: Any) -> Optional[dict[str, str]]:
    metadata = getattr(obj, "metadata", None)
    if metadata is None:
        return None
    library_path = str(
        getattr(metadata, "triton_chipyard_custom_library_path", "") or ""
    )
    if not library_path:
        return None
    library_path = os.path.abspath(os.path.expanduser(library_path))
    return {
        "path": library_path,
        "sha256": str(
            getattr(metadata, "triton_chipyard_custom_library_sha256", "") or ""
        ),
        "basename": os.path.basename(library_path),
        "config_path": str(
            getattr(metadata, "triton_chipyard_custom_config_path", "") or ""
        ),
        "config_sha256": str(
            getattr(metadata, "triton_chipyard_custom_config_sha256", "") or ""
        ),
    }


def _custom_library_from_environment() -> Optional[dict[str, str]]:
    inline_registry = os.environ.get(
        "TRITON_CHIPYARD_EXTERN_CALL_REGISTRY", ""
    ).strip()
    config_path = ""
    if inline_registry:
        config_bytes = inline_registry.encode("utf-8")
    else:
        config_path = os.environ.get(
            "TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG", ""
        ).strip()
        if not config_path:
            return None
        config_path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.isfile(config_path):
            return None
        with open(config_path, "rb") as config_file:
            config_bytes = config_file.read()
    document = json.loads(config_bytes.decode("utf-8"))
    library_value = document.get("library") if isinstance(document, dict) else None
    if not isinstance(library_value, str) or not library_value:
        return None
    library_path = os.path.expanduser(library_value)
    if not os.path.isabs(library_path):
        if not config_path:
            return None
        library_path = os.path.join(os.path.dirname(config_path), library_path)
    library_path = os.path.abspath(library_path)
    if not os.path.isfile(library_path):
        return None
    with open(library_path, "rb") as library_file:
        library_sha256 = hashlib.sha256(library_file.read()).hexdigest()
    expected_sha256 = document.get("library_sha256", "")
    if expected_sha256 and expected_sha256 != library_sha256:
        return None
    return {
        "path": library_path,
        "sha256": library_sha256,
        "basename": os.path.basename(library_path),
        "config_path": config_path,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def _stage_kernel_metadata_groups(model_spec: dict[str, Any]) -> dict[str, Any]:
    if not _chipyard_stage_kernel_artifacts_enabled():
        return model_spec

    staged_paths: dict[str, str] = {}

    def stage_entry_metadata(kernel_entry: Any) -> None:
        if not isinstance(kernel_entry, dict):
            return

        selected_launch = kernel_entry.get("selected_launch")
        if isinstance(selected_launch, dict):
            metadata_group = selected_launch.get("metadata_group")
            if isinstance(metadata_group, dict):
                filtered_group = _runner_build_metadata_group(metadata_group)
                selected_launch["metadata_group"] = chipyard_stage_kernel_artifacts(
                    filtered_group,
                    staged_paths=staged_paths,
                )

        compiled_variants = kernel_entry.get("compiled_variants")
        if isinstance(compiled_variants, list):
            for variant in compiled_variants:
                if not isinstance(variant, dict):
                    continue
                metadata_group = variant.get("metadata_group")
                if isinstance(metadata_group, dict):
                    filtered_group = _runner_build_metadata_group(metadata_group)
                    variant["metadata_group"] = chipyard_stage_kernel_artifacts(
                        filtered_group,
                        staged_paths=staged_paths,
                    )

    kernels = model_spec.get("kernels")
    if isinstance(kernels, list):
        for kernel in kernels:
            stage_entry_metadata(kernel)

    deferred_autotune_sites = model_spec.get("deferred_autotune_sites")
    if isinstance(deferred_autotune_sites, list):
        for site in deferred_autotune_sites:
            if not isinstance(site, dict):
                continue
            choices = site.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    stage_entry_metadata(choice)

    return model_spec


def _stage_custom_op_libraries(model_spec: dict[str, Any]) -> dict[str, Any]:
    libraries = model_spec.get("custom_op_libraries", [])
    if not isinstance(libraries, list):
        model_spec["custom_op_libraries"] = []
        return model_spec
    if not _chipyard_stage_kernel_artifacts_enabled():
        return model_spec

    staged_paths: dict[str, str] = {}
    library_path_map: dict[str, str] = {}
    for index, entry in enumerate(libraries):
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        if not path:
            continue
        basename = str(entry.get("basename") or os.path.basename(path))
        staged = chipyard_stage_kernel_artifacts(
            {f"custom_op_library_{index}_{basename}": path},
            staged_paths=staged_paths,
        )
        staged_path = next(iter(staged.values()), path)
        entry["path"] = staged_path
        library_path_map[os.path.abspath(path)] = staged_path
    steps = model_spec.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict) or step.get("kind") != "custom_call":
                continue
            original_path = os.path.abspath(str(step.get("library_path", "")))
            if original_path in library_path_map:
                step["library_path"] = library_path_map[original_path]
    return model_spec


def _stage_triton_custom_libraries(model_spec: dict[str, Any]) -> dict[str, Any]:
    libraries = model_spec.get("triton_custom_libraries", [])
    if not isinstance(libraries, list):
        model_spec["triton_custom_libraries"] = []
        return model_spec
    if not _chipyard_stage_kernel_artifacts_enabled():
        return model_spec

    staged_paths: dict[str, str] = {}
    for index, entry in enumerate(libraries):
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        if not path:
            continue
        basename = str(entry.get("basename") or os.path.basename(path))
        staged = chipyard_stage_kernel_artifacts(
            {f"triton_custom_library_{index}_{basename}": path},
            staged_paths=staged_paths,
        )
        entry["path"] = next(iter(staged.values()), path)
    return model_spec


def _prune_kernel_metadata_groups(model_spec: dict[str, Any]) -> dict[str, Any]:
    def prune_entry_metadata(kernel_entry: Any) -> None:
        if not isinstance(kernel_entry, dict):
            return

        selected_launch = kernel_entry.get("selected_launch")
        if isinstance(selected_launch, dict):
            metadata_group = selected_launch.get("metadata_group")
            if isinstance(metadata_group, dict):
                selected_launch["metadata_group"] = _runner_build_metadata_group(
                    metadata_group
                )

        compiled_variants = kernel_entry.get("compiled_variants")
        if isinstance(compiled_variants, list):
            for variant in compiled_variants:
                if not isinstance(variant, dict):
                    continue
                metadata_group = variant.get("metadata_group")
                if isinstance(metadata_group, dict):
                    variant["metadata_group"] = _runner_build_metadata_group(
                        metadata_group
                    )

    kernels = model_spec.get("kernels")
    if isinstance(kernels, list):
        for kernel in kernels:
            prune_entry_metadata(kernel)

    deferred_autotune_sites = model_spec.get("deferred_autotune_sites")
    if isinstance(deferred_autotune_sites, list):
        for site in deferred_autotune_sites:
            if not isinstance(site, dict):
                continue
            choices = site.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    prune_entry_metadata(choice)

    return model_spec


def _flatten_kernel_objects(model_spec: dict[str, Any]) -> tuple[list[str], dict[str, list[str]], list[str]]:
    flattened: list[str] = []
    seen: set[str] = set()
    per_kernel: dict[str, list[str]] = {}
    missing: list[str] = []

    def collect_object_candidates(
        owner_name: str,
        kernel_entry: dict[str, Any],
    ) -> None:
        candidates_by_basename: dict[str, str] = {}
        metadata_groups: list[dict[str, str]] = []

        selected_launch = kernel_entry.get("selected_launch", {})
        if isinstance(selected_launch, dict):
            metadata_group = selected_launch.get("metadata_group")
            if isinstance(metadata_group, dict):
                metadata_groups.append(
                    {str(name): str(path) for name, path in metadata_group.items()}
                )

        if not metadata_groups:
            for variant in kernel_entry.get("compiled_variants", []):
                if not isinstance(variant, dict):
                    continue
                metadata_group = variant.get("metadata_group")
                if isinstance(metadata_group, dict):
                    metadata_groups.append(
                        {str(name): str(path) for name, path in metadata_group.items()}
                    )

        for metadata_group in metadata_groups:
            for artifact_name, artifact_path in sorted(metadata_group.items()):
                if artifact_name.endswith((".o", ".obj", ".a")) or artifact_path.endswith(
                    (".o", ".obj", ".a")
                ):
                    basename = os.path.basename(str(artifact_path))
                    candidates_by_basename.setdefault(basename, artifact_path)

        ordered_candidates = [
            candidates_by_basename[basename]
            for basename in sorted(candidates_by_basename)
        ]

        per_kernel[owner_name] = ordered_candidates
        if not ordered_candidates:
            missing.append(owner_name)
            return

        for candidate in ordered_candidates:
            if candidate in seen:
                continue
            flattened.append(candidate)
            seen.add(candidate)

    kernels = model_spec.get("kernels", [])
    if not isinstance(kernels, list):
        return flattened, per_kernel, missing

    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        kernel_name = str(kernel.get("kernel_name", "unknown_kernel"))
        collect_object_candidates(kernel_name, kernel)

    deferred_autotune_sites = model_spec.get("deferred_autotune_sites", [])
    if isinstance(deferred_autotune_sites, list):
        for site in deferred_autotune_sites:
            if not isinstance(site, dict):
                continue
            site_id = str(site.get("site_id", "unknown_site"))
            for choice in site.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                kernel_name = str(choice.get("kernel_name", "unknown_kernel"))
                owner_name = f"{site_id}:{kernel_name}"
                collect_object_candidates(owner_name, choice)

    return flattened, per_kernel, missing


def _model_output_path(
    *,
    runner_cpp_path: str,
    kernel_objects: list[str],
    triton_custom_libraries: list[Any],
    custom_op_libraries: list[Any],
) -> str:
    named_path = chipyard_named_path("model.elf")
    if named_path is not None:
        return named_path
    key_material = json.dumps(
        {
            "runner_cpp_path": runner_cpp_path,
            "kernel_objects": kernel_objects,
            "triton_custom_libraries": triton_custom_libraries,
            "custom_op_libraries": custom_op_libraries,
        },
        sort_keys=True,
    )
    key = "cmodel_" + hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    _basename, subdir, output_path = chipyard_get_path(key, "out")
    os.makedirs(subdir, exist_ok=True)
    return output_path


def _build_script_source(
    *,
    runner_cpp_path: str,
    output_path: str,
    kernel_objects: list[str],
    missing_kernels: list[str],
    triton_custom_libraries=(),
    custom_op_libraries=(),
) -> str:
    script_dir = os.path.dirname(runner_cpp_path)

    def render_kernel_object_path(path: str) -> str:
        abs_path = os.path.abspath(path)
        try:
            if os.path.commonpath([script_dir, abs_path]) == script_dir:
                rel_path = os.path.relpath(abs_path, start=script_dir)
                return f'${{SCRIPT_DIR}}/{rel_path}'
        except ValueError:
            pass
        return abs_path

    object_lines = "\n".join(
        f'KERNEL_OBJECTS+=("{render_kernel_object_path(path)}")'
        for path in kernel_objects
    )
    custom_library_lines = "\n".join(
        f'CUSTOM_OP_LIBRARIES+=("{render_kernel_object_path(path)}")'
        for path in custom_op_libraries
    )
    triton_custom_library_lines = "\n".join(
        f'TRITON_CUSTOM_LIBRARIES+=("{render_kernel_object_path(path)}")'
        for path in triton_custom_libraries
    )
    missing_text = ", ".join(missing_kernels)
    runner_cpp_name = os.path.basename(runner_cpp_path)
    output_name = os.path.basename(output_path)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'LLVM_PROJECT_PATH="${LLVM_PROJECT_PATH:?LLVM_PROJECT_PATH is not set}"',
        'CHIPYARD_ENV_PATH="${CHIPYARD_ENV_PATH:?CHIPYARD_ENV_PATH is not set}"',
        "# Conda activation hooks are not consistently nounset-safe.",
        "set +u",
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        'source "${CHIPYARD_ENV_PATH}"',
        "set -u",
        "",
        'MLIR_INCLUDE_DIR="${LLVM_PROJECT_PATH}/mlir/include"',
        'MLIR_LIB_DIR="${LLVM_PROJECT_PATH}/mlir/lib"',
        'LLVM_INCLUDE_DIR="${LLVM_PROJECT_PATH}/llvm/include"',
        'RISCV_MARCH="${TRITON_CHIPYARD_RISCV_MARCH:-rv64imafdc_zicsr_zifencei_zve32f_zvl64b}"',
        'RISCV_MABI="${TRITON_CHIPYARD_RISCV_MABI:-lp64d}"',
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        f'RUNNER_CPP="${{SCRIPT_DIR}}/{runner_cpp_name}"',
        f'OUTPUT_ELF="${{SCRIPT_DIR}}/{output_name}"',
        "mkdir -p \"$(dirname \"${OUTPUT_ELF}\")\"",
        'CHIPYARD_OMP_NUM_THREADS="${CHIPYARD_OMP_NUM_THREADS:-4}"',
        "CHIPYARD_LINK_FLAGS=(-lm -fopenmp)",
        'if [[ "${PYTORCH_CHIPYARD_SPIKE_EXECUTABLE:-0}" == "1" ]]; then',
        "  CHIPYARD_OMP_NUM_THREADS=1",
        "  CHIPYARD_LINK_FLAGS=(-static -lm)",
        "fi",
        "",
        "KERNEL_OBJECTS=()",
        object_lines if object_lines else "",
        "",
        "UNIQUE_KERNEL_OBJECTS=()",
        "declare -A SEEN_KERNEL_BASENAMES=()",
        'for kernel_object in "${KERNEL_OBJECTS[@]}"; do',
        '  kernel_basename="$(basename "${kernel_object}")"',
        '  if [[ -n "${SEEN_KERNEL_BASENAMES[${kernel_basename}]:-}" ]]; then',
        "    continue",
        "  fi",
        '  SEEN_KERNEL_BASENAMES["${kernel_basename}"]=1',
        '  UNIQUE_KERNEL_OBJECTS+=("${kernel_object}")',
        "done",
        'KERNEL_OBJECTS=("${UNIQUE_KERNEL_OBJECTS[@]}")',
        "",
        "TRITON_CUSTOM_LIBRARIES=()",
        triton_custom_library_lines if triton_custom_library_lines else "",
        "",
        "CUSTOM_OP_LIBRARIES=()",
        custom_library_lines if custom_library_lines else "",
        "",
    ]

    if missing_kernels:
        lines.extend(
            [
                f'echo "Missing kernel object(s) for: {missing_text}" >&2',
                "exit 1",
                "",
            ]
        )

    lines.extend(
        [
            'riscv64-unknown-linux-gnu-g++ \\',
            '  "${RUNNER_CPP}" \\',
            '  "${KERNEL_OBJECTS[@]}" \\',
            '  "${TRITON_CUSTOM_LIBRARIES[@]}" \\',
            '  "${CUSTOM_OP_LIBRARIES[@]}" \\',
            '  "${MLIR_LIB_DIR}/ExecutionEngine/CRunnerUtils.cpp" \\',
            '  "${MLIR_LIB_DIR}/ExecutionEngine/RunnerUtils.cpp" \\',
            '  "${MLIR_LIB_DIR}/ExecutionEngine/Float16bits.cpp" \\',
            '  "${CHIPYARD_LINK_FLAGS[@]}" \\',
            '  -Wl,--allow-multiple-definition \\',
            '  -I"${MLIR_INCLUDE_DIR}" -I"${LLVM_INCLUDE_DIR}" \\',
            '  -march="${RISCV_MARCH}" -mabi="${RISCV_MABI}" \\',
            '  -DCHIPYARD_OMP_NUM_THREADS="${CHIPYARD_OMP_NUM_THREADS}" \\',
            '  -O2 -std=c++17 -o "${OUTPUT_ELF}"',
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


@dataclasses.dataclass(frozen=True)
class ChipyardModelBuildArtifacts:
    build_script_path: str
    output_elf_path: str
    materialized_model_spec: dict[str, Any]
    built_output_path: Optional[str] = None


def emit_chipyard_model_build_artifacts(
    *,
    runner_cpp_path: str,
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
) -> ChipyardModelBuildArtifacts:
    materialized_spec = _materialize_model_spec(model_spec)
    materialized_spec = _prune_kernel_metadata_groups(materialized_spec)
    materialized_spec = _stage_kernel_metadata_groups(materialized_spec)
    materialized_spec = _stage_triton_custom_libraries(materialized_spec)
    materialized_spec = _stage_custom_op_libraries(materialized_spec)

    kernel_objects, _kernel_object_map, missing_kernels = _flatten_kernel_objects(
        materialized_spec
    )
    custom_op_libraries = [
        str(entry.get("path"))
        for entry in materialized_spec.get("custom_op_libraries", [])
        if isinstance(entry, dict) and entry.get("path")
    ]
    triton_custom_libraries = [
        str(entry.get("path"))
        for entry in materialized_spec.get("triton_custom_libraries", [])
        if isinstance(entry, dict) and entry.get("path")
    ]
    triton_custom_library_key_material = [
        {
            "path": str(entry.get("path")),
            "sha256": str(entry.get("sha256", "")),
            "config_sha256": str(entry.get("config_sha256", "")),
        }
        for entry in materialized_spec.get("triton_custom_libraries", [])
        if isinstance(entry, dict) and entry.get("path")
    ]
    custom_op_library_key_material = [
        {
            "path": str(entry.get("path")),
            "sha256": str(entry.get("sha256", "")),
        }
        for entry in materialized_spec.get("custom_op_libraries", [])
        if isinstance(entry, dict) and entry.get("path")
    ]
    output_elf_path = _model_output_path(
        runner_cpp_path=runner_cpp_path,
        kernel_objects=kernel_objects,
        triton_custom_libraries=triton_custom_library_key_material,
        custom_op_libraries=custom_op_library_key_material,
    )

    build_script_source = _build_script_source(
        runner_cpp_path=runner_cpp_path,
        output_path=output_elf_path,
        kernel_objects=kernel_objects,
        missing_kernels=missing_kernels,
        triton_custom_libraries=triton_custom_libraries,
        custom_op_libraries=custom_op_libraries,
    )
    build_script_path = chipyard_named_write(build_script_source, "build.sh")

    built_output_path: Optional[str] = None
    if _chipyard_model_build_enabled() and not missing_kernels:
        subprocess.check_call(["bash", build_script_path])
        if os.path.exists(output_elf_path):
            built_output_path = output_elf_path

    output_code_log.info("Chipyard model build script written to: %s", build_script_path)
    if built_output_path is not None:
        output_code_log.info("Chipyard model ELF written to: %s", built_output_path)

    return ChipyardModelBuildArtifacts(
        build_script_path=build_script_path,
        output_elf_path=output_elf_path,
        materialized_model_spec=materialized_spec,
        built_output_path=built_output_path,
    )


def finalize_chipyard_model_build_artifacts(
    compiled_namespace: Mapping[str, Any],
) -> Optional[ChipyardModelBuildArtifacts]:
    """Refresh model build inputs after async Triton compilation completes.

    Chipyard runner source is emitted while the Python wrapper itself is being
    generated.  At that point ``async_compile.wait(globals())`` has not run yet,
    so a freshly compiled kernel cannot contribute its object/archive metadata.
    The generated wrapper calls this function immediately after that wait.
    """

    model_spec_path = chipyard_named_path("model_spec.json")
    runner_cpp_path = chipyard_named_path("runner.cpp")
    if (
        model_spec_path is None
        or runner_cpp_path is None
        or not os.path.isfile(model_spec_path)
        or not os.path.isfile(runner_cpp_path)
    ):
        return None

    model_spec = _materialize_model_spec(model_spec_path)
    libraries_by_path: dict[str, dict[str, str]] = {}
    for raw_entry in model_spec.get("triton_custom_libraries", []):
        if isinstance(raw_entry, dict) and raw_entry.get("path"):
            entry = {str(key): str(value) for key, value in raw_entry.items()}
            libraries_by_path[os.path.abspath(entry["path"])] = entry

    unresolved_entries: list[dict[str, Any]] = []

    def populate_entry(entry: Any) -> None:
        if not isinstance(entry, dict):
            return
        kernel_name = str(entry.get("kernel_name", ""))
        compiled_kernel = compiled_namespace.get(kernel_name)
        if compiled_kernel is None:
            unresolved_entries.append(entry)
            return

        entry.pop("module_load_error", None)
        entry["compiled_object_type"] = type(compiled_kernel).__name__
        entry["compiled_kernel_source"] = "post_async_compile"
        variants = _compiled_kernel_variants(compiled_kernel)
        if variants:
            entry["compiled_variants"] = variants
        selected_launch = _compiled_selected_launch(compiled_kernel)
        if selected_launch is not None:
            entry["selected_launch"] = selected_launch

        candidates = [compiled_kernel]
        for compile_result in getattr(compiled_kernel, "compile_results", None) or []:
            candidates.extend([compile_result, getattr(compile_result, "kernel", None)])
        for candidate in candidates:
            if candidate is None:
                continue
            library_entry = _compiled_custom_library_entry(candidate)
            if library_entry is not None:
                libraries_by_path[os.path.abspath(library_entry["path"])] = (
                    library_entry
                )

    for kernel_entry in model_spec.get("kernels", []):
        populate_entry(kernel_entry)
    for site in model_spec.get("deferred_autotune_sites", []):
        if not isinstance(site, dict):
            continue
        for choice in site.get("choices", []):
            populate_entry(choice)

    # A single-kernel dump has one deterministic object name.  This fallback is
    # useful when a backend object exposes no metadata_group even after wait().
    direct_object = chipyard_named_path("log-ttshared.o")
    if (
        len(unresolved_entries) == 1
        and direct_object is not None
        and os.path.isfile(direct_object)
    ):
        unresolved_entry = unresolved_entries[0]
        metadata_group = {os.path.basename(direct_object): direct_object}
        unresolved_entry.pop("module_load_error", None)
        unresolved_entry["compiled_kernel_source"] = "post_async_compile_dump"
        unresolved_entry["compiled_variants"] = [{"metadata_group": metadata_group}]
        unresolved_entry["selected_launch"] = {"metadata_group": metadata_group}

    environment_library = _custom_library_from_environment()
    if environment_library is not None:
        libraries_by_path[os.path.abspath(environment_library["path"])] = (
            environment_library
        )
    model_spec["triton_custom_libraries"] = [
        libraries_by_path[path] for path in sorted(libraries_by_path)
    ]

    build_artifacts = emit_chipyard_model_build_artifacts(
        runner_cpp_path=runner_cpp_path,
        model_spec=model_spec,
    )
    serialized_spec = json.dumps(
        build_artifacts.materialized_model_spec,
        indent=2,
        sort_keys=True,
    )
    write_atomic(model_spec_path, serialized_spec, make_dirs=True)
    output_code_log.info(
        "Finalized Chipyard model build artifacts after async compilation: %s",
        model_spec_path,
    )
    return build_artifacts

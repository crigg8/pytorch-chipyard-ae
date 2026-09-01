from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import logging
import os
import re
import struct
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import sympy
import torch

from ... import config, ir
from ...codecache import PyCodeCache, cache_dir, output_code_log
from ...virtualized import V
from .chipyard_codegen_utils import (
    chipyard_artifact_dir,
    chipyard_get_path,
    chipyard_named_path,
    chipyard_named_write,
    chipyard_write,
)
from .chipyard_model_builder import emit_chipyard_model_build_artifacts
from .chipyard_run_emitter import emit_chipyard_run_artifacts
from .custom_ops import (
    chipyard_custom_op_library_path,
    get_chipyard_custom_op_registration,
    schema_order_values,
)
from ..wrapper import PythonWrapperCodegen


CHIPYARD_MAX_TENSOR_RANK = 5


def _chipyard_runner_codegen_enabled() -> bool:
    return os.environ.get("TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER", "") == "1"


class ChipyardWrapperCodegen(PythonWrapperCodegen):
    """
    Wrapper used for triton_chipyard CPU backend.

    Keep emitting the normal Python compiled-module wrapper for now, while also
    collecting a model-level execution plan that can later drive a RISC-V runner
    emitter.
    """

    def __init__(self) -> None:
        self.chipyard_plan: list[dict[str, Any]] = []
        self.chipyard_buffers: dict[str, dict[str, Any]] = {}
        self.chipyard_kernels: dict[str, dict[str, Any]] = {}
        self.chipyard_runner_blockers: list[str] = []
        self.chipyard_weights_blob_path: Optional[str] = None
        self.chipyard_weights_manifest_path: Optional[str] = None
        self.chipyard_model_build_script_path: Optional[str] = None
        self.chipyard_model_elf_path: Optional[str] = None
        self.chipyard_run_script_path: Optional[str] = None
        self.chipyard_triton_custom_libraries: dict[str, dict[str, str]] = {}
        self._chipyard_recorded_extern_alloc_ids: set[int] = set()
        self._chipyard_autotune_scope: Optional[dict[str, Any]] = None
        self._chipyard_launch_view_overrides: dict[int, dict[int, dict[str, Any]]] = {}
        super().__init__()

    def write_async_compile_wait(self) -> None:
        super().write_async_compile_wait()
        if not _chipyard_runner_codegen_enabled():
            return
        self.prefix.splice(
            """

            from torch._inductor.codegen.chipyard.chipyard_model_builder import finalize_chipyard_model_build_artifacts as _finalize_chipyard_model_build_artifacts
            _finalize_chipyard_model_build_artifacts(globals())
            del _finalize_chipyard_model_build_artifacts
            """
        )

    def _record_chipyard_step(self, kind: str, **payload: Any) -> None:
        self.chipyard_plan.append(
            {"index": len(self.chipyard_plan), "kind": kind, **payload}
        )

    def _block_chipyard_runner(self, reason: str) -> None:
        if reason not in self.chipyard_runner_blockers:
            self.chipyard_runner_blockers.append(reason)

    @staticmethod
    def _is_materialized_tensor(value: Any) -> bool:
        fake_tensor_type = getattr(torch._subclasses, "FakeTensor", ())
        return (
            isinstance(value, torch.Tensor)
            and value.device.type != "meta"
            and not isinstance(value, fake_tensor_type)
        )

    @staticmethod
    def _materialize_tensor(value: Any) -> Optional[torch.Tensor]:
        if ChipyardWrapperCodegen._is_materialized_tensor(value):
            return value
        return None

    @staticmethod
    def _all_graph_input_names() -> list[str]:
        return [str(name) for name in V.graph.graph_inputs.keys()]

    @staticmethod
    def _real_input_static_tensor_sources() -> list[torch.Tensor]:
        static_input_idxs = tuple(getattr(V.graph, "static_input_idxs", ()) or ())
        if not static_input_idxs:
            return []
        try:
            real_inputs = list(V.real_inputs or ())
        except Exception:
            return []
        if not real_inputs:
            return []

        tensors = [
            materialized
            for index, tensor in enumerate(real_inputs)
            if index in static_input_idxs
            and (materialized := ChipyardWrapperCodegen._materialize_tensor(tensor))
            is not None
        ]
        if len(tensors) == len(static_input_idxs):
            return tensors
        output_code_log.warning(
            "Chipyard static tensors from real_inputs incomplete: real_inputs=%s static_input_idxs=%s resolved=%s",
            len(real_inputs),
            static_input_idxs,
            len(tensors),
        )
        return []

    @staticmethod
    def _tracing_context_static_tensor_sources() -> list[torch.Tensor]:
        tracing_context = torch._guards.TracingContext.try_get()
        if tracing_context is None:
            return []

        for attr_name in ("params_flat", "params_flat_unwrap_subclasses"):
            tensors = [
                materialized
                for tensor in (getattr(tracing_context, attr_name, ()) or ())
                if (materialized := ChipyardWrapperCodegen._materialize_tensor(tensor))
                is not None
            ]
            if tensors:
                return tensors
        return []

    @staticmethod
    def _flattened_static_tensor_sources() -> list[torch.Tensor]:
        real_input_tensors = ChipyardWrapperCodegen._real_input_static_tensor_sources()
        if real_input_tensors:
            return real_input_tensors

        tracing_context_tensors = (
            ChipyardWrapperCodegen._tracing_context_static_tensor_sources()
        )
        if tracing_context_tensors:
            return tracing_context_tensors

        lifted_input_tensors = list(getattr(V.graph, "lifted_input_tensors", ()) or ())
        lifted_real_tensors = [
            materialized
            for tensor in lifted_input_tensors
            if (materialized := ChipyardWrapperCodegen._materialize_tensor(tensor))
            is not None
        ]
        if lifted_real_tensors:
            return lifted_real_tensors

        static_input_idxs = tuple(getattr(V.graph, "static_input_idxs", ()) or ())
        example_inputs = list(getattr(V.graph, "example_inputs", ()) or ())
        if example_inputs and static_input_idxs:
            example_tensors = [
                materialized
                for index, tensor in enumerate(example_inputs)
                if index in static_input_idxs
                and (
                    materialized := ChipyardWrapperCodegen._materialize_tensor(tensor)
                )
                is not None
            ]
            if len(example_tensors) == len(static_input_idxs):
                return example_tensors
            output_code_log.warning(
                "Chipyard static tensors from example_inputs incomplete: lifted_input_tensors=%s example_inputs=%s static_input_idxs=%s resolved=%s",
                len(lifted_input_tensors),
                len(example_inputs),
                static_input_idxs,
                len(example_tensors),
            )

        ordered_tensors: list[torch.Tensor] = []
        total_named_parameter_count = 0
        total_named_buffer_count = 0
        for tensor_dict_name in ("named_parameters", "named_buffers"):
            tensor_dict = getattr(V.graph, tensor_dict_name, {}) or {}
            if not isinstance(tensor_dict, dict):
                continue
            if tensor_dict_name == "named_parameters":
                total_named_parameter_count = len(tensor_dict)
            else:
                total_named_buffer_count = len(tensor_dict)
            for tensor in tensor_dict.values():
                materialized = ChipyardWrapperCodegen._materialize_tensor(tensor)
                if materialized is not None:
                    ordered_tensors.append(materialized)
        if ordered_tensors:
            return ordered_tensors

        if (
            static_input_idxs
            or total_named_parameter_count
            or total_named_buffer_count
            or len(lifted_input_tensors)
            or len(getattr(V.graph, "constants", {}) or {})
        ):
            output_code_log.warning(
                "Chipyard static tensors unresolved: named_parameters=%s named_buffers=%s graph_lifted=%s example_inputs=%s tracing_context=%s",
                total_named_parameter_count,
                total_named_buffer_count,
                len(lifted_real_tensors),
                len(example_inputs),
                bool(torch._guards.TracingContext.try_get()),
            )
        return []

    def _static_graph_input_indices(self) -> tuple[int, ...]:
        graph_input_count = len(self._all_graph_input_names())
        static_input_idxs = getattr(V.graph, "static_input_idxs", ())
        if static_input_idxs:
            return tuple(
                int(index)
                for index in static_input_idxs
                if 0 <= int(index) < graph_input_count
            )
        static_source_count = len(self._flattened_static_tensor_sources())
        if static_source_count:
            return tuple(range(min(static_source_count, graph_input_count)))
        return ()

    def _runtime_graph_input_names(self) -> list[str]:
        names = self._all_graph_input_names()
        static_input_indices = set(self._static_graph_input_indices())
        return [
            name for index, name in enumerate(names) if index not in static_input_indices
        ]

    def _lifted_graph_input_names(self) -> list[str]:
        names = self._all_graph_input_names()
        static_input_indices = set(self._static_graph_input_indices())
        return [name for index, name in enumerate(names) if index in static_input_indices]

    def _lifted_graph_input_tensors(self) -> list[tuple[str, torch.Tensor]]:
        graph_input_names = self._all_graph_input_names()
        static_input_indices = self._static_graph_input_indices()
        static_tensors = self._flattened_static_tensor_sources()
        lifted_tensors: list[tuple[str, torch.Tensor]] = []
        if static_tensors and len(static_tensors) != len(static_input_indices):
            output_code_log.warning(
                "Chipyard static input mismatch: %s static tensors for %s static graph inputs",
                len(static_tensors),
                len(static_input_indices),
            )
        for param, graph_input_index in zip(static_tensors, static_input_indices):
            if graph_input_index >= len(graph_input_names):
                break
            name = graph_input_names[graph_input_index]
            if self._is_materialized_tensor(param):
                lifted_tensors.append((name, param))
        return lifted_tensors

    @staticmethod
    def _serialize_shape(values: Any) -> list[str]:
        try:
            return [str(value) for value in values]
        except TypeError:
            return []

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(key): ChipyardWrapperCodegen._json_safe(inner)
                for key, inner in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [ChipyardWrapperCodegen._json_safe(inner) for inner in value]
        return str(value)

    @staticmethod
    def _safe_get_name(value: Any) -> Optional[str]:
        maybe_get_name = getattr(value, "maybe_get_name", None)
        if callable(maybe_get_name):
            try:
                name = maybe_get_name()
            except Exception:
                name = None
            if name is not None:
                return str(name)

        get_name = getattr(value, "get_name", None)
        if callable(get_name):
            try:
                name = get_name()
            except Exception:
                name = None
            if name is not None:
                return str(name)
        return None

    @staticmethod
    def _has_tensor_output(value: Any) -> bool:
        has_tensor_output = getattr(value, "has_tensor_output", None)
        if callable(has_tensor_output):
            try:
                return bool(has_tensor_output())
            except Exception:
                return False
        return False

    @classmethod
    def _serialize_shape_constant(cls, expr: Any) -> dict[str, Any]:
        hinted_value = cls._size_hint_int(expr, fallback=-1)
        if hinted_value >= 0:
            return {"kind": "scalar", "value": hinted_value, "expr": str(expr)}
        return {"kind": "symbol", "expr": str(expr)}

    @staticmethod
    def _cpp_identifier(value: Any) -> str:
        text = str(value)
        if text.isidentifier():
            return text
        return re.sub(r"[^0-9A-Za-z_]", "_", text)

    @staticmethod
    def _cpp_string_literal(value: Any) -> str:
        return json.dumps(str(value))

    @staticmethod
    def _cpp_bool(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _cpp_untyped_scalar_literal(value: Any) -> str:
        if isinstance(value, bool):
            return ChipyardWrapperCodegen._cpp_bool(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        return "0"

    @staticmethod
    def _align_up(value: int, alignment: int = 64) -> int:
        if alignment <= 1:
            return value
        remainder = value % alignment
        if remainder == 0:
            return value
        return value + alignment - remainder

    @staticmethod
    def _tensor_storage_key(tensor: torch.Tensor) -> tuple[int, int]:
        storage = tensor.untyped_storage()
        return (int(storage.data_ptr()), int(storage.nbytes()))

    @staticmethod
    def _tensor_required_storage_bounds(tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.numel() == 0:
            storage_offset = int(tensor.storage_offset())
            return storage_offset, storage_offset

        min_index = int(tensor.storage_offset())
        max_index = int(tensor.storage_offset())
        for size_value, stride_value in zip(tensor.shape, tensor.stride()):
            size_int = int(size_value)
            stride_int = int(stride_value)
            if size_int <= 0:
                return min_index, min_index
            delta = (size_int - 1) * stride_int
            if delta >= 0:
                max_index += delta
            else:
                min_index += delta
        return min_index, max_index

    @staticmethod
    def _chipyard_layout_offset(layout: Any) -> Any:
        offset = getattr(layout, "offset", None)
        if isinstance(layout, ir.NonOwningLayout):
            view = getattr(layout, "view", None)
            try:
                view_layout = view.get_layout() if view is not None else None
                view_offset = getattr(view_layout, "offset", None)
                if view_offset is not None:
                    offset = view_offset
            except Exception:
                pass
        return offset

    @staticmethod
    def _named_or_keyed_chipyard_path(
        filename: str,
        *,
        key: str,
        extension: str,
    ) -> str:
        named_path = chipyard_named_path(filename)
        if named_path is not None:
            return named_path
        _basename, subdir, generated_path = chipyard_get_path(key, extension)
        os.makedirs(subdir, exist_ok=True)
        return generated_path

    @staticmethod
    def _stream_write_zeros(
        blob_file: Any,
        nbytes: int,
    ) -> None:
        if nbytes <= 0:
            return
        chunk = b"\0" * min(nbytes, 1 << 20)
        remaining = nbytes
        while remaining > 0:
            write_size = min(len(chunk), remaining)
            blob_file.write(chunk[:write_size])
            remaining -= write_size

    @staticmethod
    def _stream_write_tensor_bytes(
        blob_file: Any,
        base_ptr: int,
        nbytes: int,
        *,
        chunk_bytes: int = 16 << 20,
    ) -> None:
        if nbytes <= 0:
            return
        remaining = nbytes
        offset = 0
        while remaining > 0:
            current_chunk = min(chunk_bytes, remaining)
            buffer_type = ctypes.c_char * current_chunk
            buffer_view = buffer_type.from_address(base_ptr + offset)
            blob_file.write(buffer_view)
            offset += current_chunk
            remaining -= current_chunk

    @staticmethod
    def _load_json_if_exists(path: Optional[str]) -> Optional[dict[str, Any]]:
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _cpp_scalar_type(signature_type: Optional[str]) -> str:
        if signature_type is None:
            return "int64_t"
        if signature_type.startswith("*"):
            return "void*"
        if signature_type == "constexpr":
            return "int64_t"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "uint32_t",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "f16",
            "bf16": "bf16",
            "fp32": "float",
            "f32": "float",
            "fp64": "double",
            "f64": "double",
        }.get(signature_type, "int64_t")

    @staticmethod
    def _chipyard_dtype_enum_name(signature_type: Optional[str]) -> str:
        if signature_type is None:
            return "I64"
        normalized = signature_type[1:] if signature_type.startswith("*") else signature_type
        return {
            "i1": "I1",
            "i8": "I8",
            "i16": "I16",
            "i32": "I32",
            "i64": "I64",
            "u1": "U1",
            "u8": "U8",
            "u16": "U16",
            "u32": "U32",
            "u64": "U64",
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
            "f32": "FP32",
            "fp64": "FP64",
            "f64": "FP64",
        }.get(normalized, "I64")

    @staticmethod
    def _chipyard_scalar_bits(signature_type: Optional[str], value: Any) -> int:
        normalized = signature_type or "i64"
        if normalized.startswith("*"):
            normalized = normalized[1:]
        if normalized in {"fp32", "f32"}:
            return int(struct.unpack("<I", struct.pack("<f", float(value)))[0])
        if normalized in {"fp64", "f64"}:
            return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])
        if normalized == "fp16":
            encoded = (
                torch.tensor([float(value)], dtype=torch.float16)
                .view(torch.uint16)
                .item()
            )
            return int(encoded)
        if normalized == "bf16":
            encoded = (
                torch.tensor([float(value)], dtype=torch.bfloat16)
                .view(torch.uint16)
                .item()
            )
            return int(encoded)
        if normalized in {"i1", "u1"}:
            return int(1 if bool(value) else 0)
        bit_width = {
            "i8": 8,
            "u8": 8,
            "i16": 16,
            "u16": 16,
            "i32": 32,
            "u32": 32,
            "i64": 64,
            "u64": 64,
        }.get(normalized, 64)
        mask = (1 << bit_width) - 1 if bit_width < 64 else ((1 << 64) - 1)
        return int(value) & mask

    @classmethod
    def _chipyard_scalar_bits_literal(
        cls,
        signature_type: Optional[str],
        value: Any,
    ) -> str:
        bits = cls._chipyard_scalar_bits(signature_type, value)
        return f"UINT64_C({bits})"

    @staticmethod
    def _size_hint_int(expr: Any, *, fallback: int = 0) -> int:
        try:
            return int(V.graph.sizevars.size_hint(expr, fallback=fallback))
        except Exception:
            return fallback

    @classmethod
    def _runtime_int_value(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, sympy.Basic):
            hinted = cls._size_hint_int(value, fallback=-1)
            return hinted if hinted >= 0 else None
        node = getattr(value, "node", None)
        expr = getattr(node, "expr", None)
        if isinstance(expr, sympy.Basic):
            hinted = cls._size_hint_int(expr, fallback=-1)
            return hinted if hinted >= 0 else None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _torch_dtype_from_string(dtype_name: Any) -> Optional[torch.dtype]:
        if isinstance(dtype_name, torch.dtype):
            return dtype_name
        if not isinstance(dtype_name, str):
            return None
        dtype = getattr(torch, dtype_name.split(".")[-1], None)
        return dtype if isinstance(dtype, torch.dtype) else None

    @classmethod
    def _torch_dtype_itemsize(cls, dtype_name: Any) -> int:
        dtype = cls._torch_dtype_from_string(dtype_name)
        if dtype is None:
            return 1
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is not None:
            return max(int(itemsize), 1)
        try:
            return max(int(torch.empty((), dtype=dtype).element_size()), 1)
        except Exception:
            return 1

    def _lookup_chipyard_buffer_value(self, name: str) -> Optional[Any]:
        return (
            V.graph.try_get_buffer(name)
            or V.graph.graph_inputs.get(name)
            or V.graph.constants.get(name)
        )

    def _record_chipyard_buffer_alias(
        self,
        alias_name: Any,
        source_name: Any,
        *,
        origin: str,
    ) -> None:
        alias = str(alias_name)
        source = str(source_name)
        if not alias or not source or alias == source:
            return

        source_root = self._resolve_chipyard_alias_name(source)
        if source_root == alias:
            return

        source_buffer = self._lookup_chipyard_buffer_value(source)
        if source_buffer is not None and source not in self.chipyard_buffers:
            self._record_buffer_metadata(source_buffer, origin=origin)

        entry = self.chipyard_buffers.setdefault(alias, {"name": alias})
        entry.setdefault("origins", [])
        if origin not in entry["origins"]:
            entry["origins"].append(origin)
        entry["alias_of"] = source

    def make_buffer_reuse(self, old: Any, new: Any, delete_old: bool) -> str:
        old_name = self._safe_get_name(old)
        new_name = self._safe_get_name(new)
        if old_name is not None and new_name is not None:
            self._record_buffer_metadata(old, origin="reuse_source")
            self._record_buffer_metadata(new, origin="reuse")
            self._record_chipyard_buffer_alias(new_name, old_name, origin="reuse")
        return super().make_buffer_reuse(old, new, delete_old)

    @classmethod
    def _shape_hints(cls, values: Any) -> list[int]:
        try:
            return [cls._size_hint_int(value) for value in values]
        except TypeError:
            return []

    @staticmethod
    def _config_to_dict(config_obj: Any) -> dict[str, Any]:
        if config_obj is None:
            return {}
        try:
            from ...runtime.triton_heuristics import config_to_dict

            return ChipyardWrapperCodegen._json_safe(config_to_dict(config_obj))
        except Exception:
            return {"repr": repr(config_obj)}

    @classmethod
    def _chipyard_config_log_entries(cls, config_dict: Any) -> list[tuple[str, str]]:
        if not isinstance(config_dict, dict):
            return []
        entries: list[tuple[str, str]] = []
        for key, value in sorted(config_dict.items(), key=lambda item: str(item[0])):
            safe_value = cls._json_safe(value)
            if isinstance(safe_value, (dict, list)):
                value_text = json.dumps(
                    safe_value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            else:
                value_text = str(safe_value)
            entries.append((str(key), value_text))
        return entries

    @staticmethod
    def _chipyard_normalized_kernel_name(kernel_name: Any) -> str:
        return re.sub(r"_\d+$", "", str(kernel_name))

    @classmethod
    def _chipyard_description_config_entries(
        cls, description: Any
    ) -> list[tuple[str, str]]:
        if not isinstance(description, str) or not description:
            return []

        parts: list[str] = []
        current: list[str] = []
        quote_char: Optional[str] = None
        bracket_depth = 0
        escaped = False
        for char in description:
            if escaped:
                current.append(char)
                escaped = False
                continue
            if char == "\\":
                current.append(char)
                escaped = True
                continue
            if quote_char is not None:
                current.append(char)
                if char == quote_char:
                    quote_char = None
                continue
            if char in ("'", '"'):
                current.append(char)
                quote_char = char
                continue
            if char in "([{":
                bracket_depth += 1
                current.append(char)
                continue
            if char in ")]}" and bracket_depth > 0:
                bracket_depth -= 1
                current.append(char)
                continue
            if char == "," and bracket_depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append("".join(current).strip())

        entries: list[tuple[str, str]] = []
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                entries.append((key, value))
        return entries

    @classmethod
    def _chipyard_choice_config_log_entries(
        cls, choice: dict[str, Any]
    ) -> list[tuple[str, str]]:
        entries_by_key: dict[str, str] = {}

        def add_entries(entries: list[tuple[str, str]]) -> None:
            for key, value in entries:
                key_text = str(key)
                if key_text and key_text not in entries_by_key:
                    entries_by_key[key_text] = str(value)

        selected_launch = choice.get("selected_launch")
        if isinstance(selected_launch, dict):
            add_entries(
                cls._chipyard_config_log_entries(
                    selected_launch.get("config_dict", {})
                )
            )
        add_entries(cls._chipyard_config_log_entries(choice.get("config_dict", {})))
        add_entries(cls._chipyard_description_config_entries(choice.get("description")))
        add_entries(cls._chipyard_config_log_entries(choice.get("log_info", {})))
        return sorted(entries_by_key.items(), key=lambda item: item[0])

    def _record_buffer_metadata(self, buffer: Any, *, origin: str) -> Optional[str]:
        if not self._has_tensor_output(buffer):
            return None

        name = self._safe_get_name(buffer)
        if name is None:
            return None
        entry = self.chipyard_buffers.setdefault(name, {"name": name})
        entry.setdefault("origins", [])
        if origin not in entry["origins"]:
            entry["origins"].append(origin)

        graph_outputs = set(V.graph.get_output_names())
        if name in V.graph.graph_inputs:
            role = "graph_input"
        elif name in V.graph.constants:
            role = "constant"
        elif name in graph_outputs:
            role = "graph_output"
        else:
            role = "intermediate"
        entry["role"] = role
        entry["is_output"] = name in graph_outputs

        if hasattr(buffer, "get_dtype"):
            try:
                dtype = buffer.get_dtype()
                entry["dtype"] = str(dtype)
                itemsize = getattr(dtype, "itemsize", None)
            except Exception:
                itemsize = None
        else:
            itemsize = None

        if hasattr(buffer, "get_size"):
            try:
                size = buffer.get_size()
                entry["size"] = self._serialize_shape(size)
                entry["size_hint"] = self._shape_hints(size)
            except Exception:
                pass

        if hasattr(buffer, "get_stride"):
            try:
                stride = buffer.get_stride()
                entry["stride"] = self._serialize_shape(stride)
                entry["stride_hint"] = self._shape_hints(stride)
            except Exception:
                pass

        if hasattr(buffer, "get_layout"):
            try:
                layout = buffer.get_layout()
                offset = self._chipyard_layout_offset(layout)
                if offset is not None:
                    entry["layout_offset_expr"] = str(offset)
                    entry["layout_offset_hint"] = self._size_hint_int(offset)
                if isinstance(layout, ir.NonOwningLayout):
                    alias_source = layout.view.unwrap_view()
                    if hasattr(alias_source, "get_name"):
                        alias_of = str(alias_source.get_name())
                        if alias_of and alias_of != name:
                            self._record_chipyard_buffer_alias(
                                name, alias_of, origin=origin
                            )
            except Exception:
                pass

        try:
            storage_size = V.graph.get_allocation_storage_size(buffer)
            entry["storage_size_expr"] = str(storage_size)
            if itemsize is not None:
                nbytes_expr = storage_size * itemsize
                entry["nbytes_expr"] = str(nbytes_expr)
                entry["nbytes_hint"] = int(
                    V.graph.sizevars.size_hint(nbytes_expr, fallback=0)
                )
        except Exception:
            pass

        get_aliases = getattr(buffer, "get_inputs_that_alias_output", None)
        if callable(get_aliases):
            try:
                alias_names = [str(alias) for alias in get_aliases() if alias]
            except Exception:
                alias_names = []
            if alias_names:
                alias_of = alias_names[0]
                if alias_of and alias_of != name:
                    self._record_chipyard_buffer_alias(name, alias_of, origin=origin)

        return name

    def _extract_chipyard_launch_view_override(
        self,
        value: Any,
    ) -> Optional[dict[str, Any]]:
        if value is None or isinstance(value, (bool, int, float, str, sympy.Basic)):
            return None
        if not self._has_tensor_output(value):
            return None
        try:
            size = value.get_size()
            stride = value.get_stride()
        except Exception:
            return None
        size_hint = self._shape_hints(size)
        stride_hint = self._shape_hints(stride)
        if (
            len(size_hint) != len(stride_hint)
            or len(size_hint) > CHIPYARD_MAX_TENSOR_RANK
        ):
            return None

        offset = 0
        try:
            layout = value.get_layout()
            offset = self._chipyard_layout_offset(layout)
        except Exception:
            pass
        try:
            dtype = value.get_dtype()
        except Exception:
            dtype = getattr(value, "dtype", None)

        override: dict[str, Any] = {
            "type": type(value).__name__,
            "size": self._serialize_shape(size),
            "size_hint": size_hint,
            "stride": self._serialize_shape(stride),
            "stride_hint": stride_hint,
            "offset": str(offset),
            "offset_hint": self._size_hint_int(offset),
        }
        view_name = self._safe_get_name(value)
        if view_name is not None:
            override["name"] = view_name
        if dtype is not None:
            override["dtype"] = str(dtype)
        return override

    def _record_chipyard_launch_view_overrides(
        self,
        step_index: int,
        raw_args: Any,
    ) -> None:
        if not isinstance(raw_args, (list, tuple)):
            return
        overrides: dict[int, dict[str, Any]] = {}
        for arg_index, raw_arg in enumerate(raw_args):
            override = self._extract_chipyard_launch_view_override(raw_arg)
            if override is not None:
                overrides[arg_index] = override
        if overrides:
            self._chipyard_launch_view_overrides[step_index] = overrides

    def _serialize_runtime_value(self, value: Any, *, origin: str) -> dict[str, Any]:
        if isinstance(value, (bool, int, float)):
            return {"kind": "scalar", "value": value}
        if value is None:
            return {"kind": "none"}
        if isinstance(value, ir.NoneAsConstantBuffer):
            return {"kind": "none"}
        if isinstance(value, ir.ShapeAsConstantBuffer):
            return self._serialize_shape_constant(value.expr)
        if isinstance(value, sympy.Basic):
            return self._serialize_shape_constant(value)
        if isinstance(value, (list, tuple)):
            return {
                "kind": "sequence",
                "container": type(value).__name__,
                "items": [
                    self._serialize_runtime_value(inner, origin=origin) for inner in value
                ],
            }
        if isinstance(value, torch._ops.OpOverload):
            return {"kind": "op_overload", "name": value.name()}
        name = self._safe_get_name(value)
        if name is not None:
            self._record_buffer_metadata(value, origin=origin)
            if name in V.graph.graph_inputs:
                if name in self._lifted_graph_input_names():
                    return {"kind": "constant", "name": name}
                return {"kind": "graph_input", "name": name}
            if name in V.graph.constants:
                return {"kind": "constant", "name": name}
            if V.graph.try_get_buffer(name) is not None:
                serialized = {"kind": "buffer", "name": name}
                alias_of = self.chipyard_buffers.get(name, {}).get("alias_of")
                if alias_of and str(alias_of) != name:
                    serialized["alias_of"] = str(alias_of)
                return serialized
        return {"kind": type(value).__name__, "expr": repr(value)}

    def _serialize_launch_arg(self, arg: Any) -> dict[str, Any]:
        if isinstance(arg, str):
            if arg in V.graph.graph_inputs:
                self._record_buffer_metadata(V.graph.graph_inputs[arg], origin="launch")
                if arg in self._lifted_graph_input_names():
                    return {"kind": "constant", "name": arg}
                return {"kind": "graph_input", "name": arg}
            if arg in V.graph.constants:
                buffer = V.graph.try_get_buffer(arg)
                if buffer is not None:
                    self._record_buffer_metadata(buffer, origin="launch")
                return {"kind": "constant", "name": arg}

            buffer = V.graph.try_get_buffer(arg)
            if buffer is not None:
                self._record_buffer_metadata(buffer, origin="launch")
                serialized = {"kind": "buffer", "name": arg}
                alias_candidates = []
                get_aliases = getattr(buffer, "get_inputs_that_alias_output", None)
                if callable(get_aliases):
                    try:
                        alias_candidates.extend(str(name) for name in get_aliases() if name)
                    except Exception:
                        pass
                inplace_buffers = getattr(getattr(V.kernel, "args", None), "inplace_buffers", None)
                if isinstance(inplace_buffers, dict):
                    inplaced = inplace_buffers.get(arg)
                    other_names = list(getattr(inplaced, "other_names", ()) or ())
                    alias_candidates.extend(
                        str(name)
                        for name in reversed(other_names)
                        if name and str(name) != str(arg)
                    )
                if (
                    hasattr(V.kernel, "inplace_update_buffers")
                    and arg in V.kernel.inplace_update_buffers
                ):
                    alias_candidates.append(str(V.kernel.inplace_update_buffers[arg]))
                for alias_of in alias_candidates:
                    if alias_of and alias_of != arg:
                        alias_buffer = self._lookup_chipyard_buffer_value(alias_of)
                        if alias_buffer is not None:
                            self._record_buffer_metadata(alias_buffer, origin="launch")
                        self._record_chipyard_buffer_alias(
                            arg, alias_of, origin="launch"
                        )
                        serialized["alias_of"] = alias_of
                        break
                return serialized

            return {"kind": "symbol", "expr": arg}

        if isinstance(arg, (bool, int, float)):
            return {"kind": "scalar", "value": arg}

        if isinstance(arg, sympy.Basic):
            hinted_value = self._size_hint_int(arg, fallback=-1)
            if hinted_value >= 0:
                return {"kind": "scalar", "value": hinted_value, "expr": str(arg)}
            return {"kind": "symbol", "expr": str(arg)}

        return {"kind": type(arg).__name__, "expr": str(arg)}

    def _record_extern_kernel_alloc_step(self, node: ir.ExternKernelAlloc) -> None:
        node_id = id(node)
        if node_id in self._chipyard_recorded_extern_alloc_ids:
            return
        self._chipyard_recorded_extern_alloc_ids.add(node_id)

        kernel_name = str(
            getattr(node, "python_kernel_name", None)
            or getattr(node, "cpp_kernel_name", None)
            or getattr(getattr(node, "op_overload", None), "name", lambda: "unknown")()
        )
        op_overload = getattr(node, "op_overload", None)
        custom_registration = get_chipyard_custom_op_registration(op_overload)
        output_node: Any = node
        if custom_registration is not None and isinstance(
            getattr(node, "layout", None), ir.MultiOutputLayout
        ):
            outputs = list(getattr(node, "outputs", ()) or ())
            if len(outputs) != 1 or not self._has_tensor_output(outputs[0]):
                self._block_chipyard_runner(
                    f"custom op {kernel_name} must have exactly one materialized "
                    "Tensor output"
                )
                return
            output_node = outputs[0]

        output_name = self._record_buffer_metadata(
            output_node, origin="extern_output"
        )
        if output_name is None:
            self._block_chipyard_runner(
                f"extern alloc {kernel_name} has no materialized output buffer metadata"
            )
            return

        if custom_registration is not None:
            try:
                raw_args, raw_kwargs = node.unflatten_args(
                    node.inputs, node.constant_args
                )
                ordered_values = schema_order_values(
                    custom_registration.custom_op._schema,
                    raw_args,
                    raw_kwargs,
                )
            except Exception as exc:
                self._block_chipyard_runner(
                    f"failed to reconstruct custom op arguments for {kernel_name}: {exc}"
                )
                return

            arguments: list[dict[str, Any]] = []
            for argument_index, (input_kind, value) in enumerate(
                zip(custom_registration.input_kinds, ordered_values)
            ):
                serialized_value = self._serialize_runtime_value(
                    value, origin="custom_call"
                )
                if input_kind == "tensor":
                    try:
                        dtype = value.get_dtype()
                    except Exception:
                        dtype = None
                    if dtype is None:
                        value_name = self._safe_get_name(value)
                        dtype = self.chipyard_buffers.get(
                            str(value_name), {}
                        ).get("dtype")
                    dtype_signature = self._chipyard_dtype_signature_name(dtype)
                    if dtype_signature is None:
                        self._block_chipyard_runner(
                            f"unsupported custom Tensor dtype for {kernel_name}: {dtype}"
                        )
                        return
                    signature_type = "*" + dtype_signature
                else:
                    signature_type = {
                        "bool": "i1",
                        "int": "i64",
                        "float": "fp64",
                    }[input_kind]
                arguments.append(
                    {
                        "index": argument_index,
                        "kind": input_kind,
                        "signature_type": signature_type,
                        "value": serialized_value,
                    }
                )

            output_entry = self.chipyard_buffers.get(output_name, {})
            output_dtype = output_entry.get("dtype")
            output_dtype_signature = self._chipyard_dtype_signature_name(output_dtype)
            if output_dtype_signature is None:
                self._block_chipyard_runner(
                    f"unsupported custom output dtype for {kernel_name}: {output_dtype}"
                )
                return
            output_signature_type = "*" + output_dtype_signature
            library_path = chipyard_custom_op_library_path(require_exists=True)
            assert library_path is not None
            self._record_chipyard_step(
                "custom_call",
                op_name=custom_registration.custom_op.name(),
                source_op=custom_registration.source_op.name(),
                symbol=custom_registration.symbol,
                library_path=library_path,
                arguments=arguments,
                output={
                    "kind": "buffer",
                    "name": output_name,
                    "signature_type": output_signature_type,
                },
            )
            return

        if op_overload is torch.ops.aten.avg_pool2d.default:
            self._block_chipyard_runner(
                "avg_pool2d extern fallback is disabled for Chipyard runner; "
                "expected avg_pool2d to lower to a Triton launch kernel"
            )
            return

        self._block_chipyard_runner(
            f"unsupported extern alloc op for Chipyard runner: {kernel_name}"
        )

    def _resolve_chipyard_alias_name(self, name: Any) -> str:
        current = str(name)
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            buffer_entry = self.chipyard_buffers.get(current)
            if not isinstance(buffer_entry, dict):
                break
            alias_of = buffer_entry.get("alias_of")
            if not alias_of:
                break
            alias_name = str(alias_of)
            if not alias_name or alias_name == current:
                break
            current = alias_name
        return current

    def _summarize_triton_meta(self, triton_meta: Any) -> dict[str, Any]:
        if not isinstance(triton_meta, dict):
            return {}

        constants = triton_meta.get("constants", {})
        signature = triton_meta.get("signature", {})
        configs = triton_meta.get("configs", ())
        raw_signature_names = triton_meta.get("signature_names")
        if isinstance(raw_signature_names, (list, tuple)):
            signature_names = [str(name) for name in raw_signature_names]
        else:
            signature_names = [str(name) for name in signature.keys()]
        summary = {
            "config_count": len(configs),
            "constant_names": sorted(str(name) for name in constants.keys()),
            "constants": {
                str(name): self._json_safe(value) for name, value in constants.items()
            },
            "signature": {str(name): str(dtype) for name, dtype in signature.items()},
            "signature_names": signature_names,
        }
        for key in (
            "chipyard_autotune_site_id",
            "chipyard_default_choice_name",
            "chipyard_default_kernel_name",
        ):
            value = triton_meta.get(key)
            if value is not None:
                summary[key] = str(value)
        return summary

    @staticmethod
    def _extract_kernel_path_from_metadata(metadata: Any) -> Optional[str]:
        if metadata is None:
            return None
        for raw_line in str(metadata).splitlines():
            line = raw_line.strip()
            if not line.startswith("#"):
                continue
            _, _, suffix = line.partition(":")
            if line.lower().startswith("# kernel path:"):
                path = suffix.strip()
                if path:
                    return path
        return None

    def _record_chipyard_kernel_definition(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        cpp_definition: Optional[str] = None,
    ) -> None:
        entry = self.chipyard_kernels.setdefault(kernel_name, {"kernel_name": kernel_name})
        entry["kernel_name"] = kernel_name
        if metadata:
            entry["metadata"] = metadata
            kernel_path = self._extract_kernel_path_from_metadata(metadata)
            if kernel_path:
                entry["kernel_path"] = kernel_path
                entry["module_cache_key"] = Path(kernel_path).stem
        if cpp_definition is not None:
            entry["cpp_definition"] = cpp_definition
        if kernel_body:
            entry.setdefault("kernel_body", kernel_body)

    def _record_chipyard_launch_step(
        self,
        kernel_name: str,
        call_args: Any,
        *,
        triton_meta: Optional[dict[str, Any]] = None,
        original_fxnode_name: Optional[str] = None,
        raw_keys: Any = None,
        raw_args: Any = None,
    ) -> None:
        step_index = len(self.chipyard_plan)
        serialized_args = [self._serialize_launch_arg(arg) for arg in call_args]
        step: dict[str, Any] = {
            "kernel_name": kernel_name,
            "call_args": serialized_args,
        }
        if original_fxnode_name:
            step["original_fxnode_name"] = str(original_fxnode_name)
        summarized_meta = self._summarize_triton_meta(triton_meta)
        if summarized_meta:
            step["triton_meta"] = summarized_meta
        _ = raw_keys
        self._record_chipyard_launch_view_overrides(step_index, raw_args)
        self._record_chipyard_step("launch", **step)

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ):
        self._record_chipyard_kernel_definition(
            kernel_name,
            kernel_body,
            metadata=metadata,
            cpp_definition=cpp_definition,
        )
        return super()._define_kernel_helper(
            kernel_name,
            kernel_body,
            metadata=metadata,
            gpu=gpu,
            cpp_definition=cpp_definition,
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        graph_name="",
        original_fxnode_name=None,
    ):
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "cpu" and triton:
            self._record_chipyard_launch_step(
                kernel_name,
                call_args,
                triton_meta=triton_meta,
                original_fxnode_name=original_fxnode_name,
                raw_keys=raw_keys,
                raw_args=raw_args,
            )
        elif device.type == "cpu" and not triton:
            self._block_chipyard_runner(
                f"unsupported non-triton cpu kernel launch for Chipyard runner: {kernel_name}"
            )
        return super()._generate_kernel_call_helper(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            arg_types=arg_types,
            raw_keys=raw_keys,
            raw_args=raw_args,
            triton_meta=triton_meta,
            graph_name=graph_name,
            original_fxnode_name=original_fxnode_name,
        )

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        if isinstance(extern_kernel, ir.ExternKernelAlloc):
            self._record_extern_kernel_alloc_step(extern_kernel)
        return super()._generate_extern_kernel_alloc_helper(extern_kernel, args)

    def generate_fallback_kernel(self, node: ir.FallbackKernel) -> None:
        # Record when the wrapper line is created.  With wrapper FX IR enabled,
        # ExternKernelAllocLine.codegen_fx() bypasses
        # _generate_extern_kernel_alloc_helper() during final emission.
        self._record_extern_kernel_alloc_step(node)
        return super().generate_fallback_kernel(node)

    def generate_extern_kernel_alloc(self, node: ir.ExternKernelAlloc):
        # Keep the same early hook for non-FallbackKernel extern allocations.
        self._record_extern_kernel_alloc_step(node)
        return super().generate_extern_kernel_alloc(node)

    def _extract_metadata_group(self, obj: Any) -> Optional[dict[str, str]]:
        metadata_group = getattr(obj, "metadata_group", None)
        if not isinstance(metadata_group, dict):
            return None
        return {str(name): str(path) for name, path in metadata_group.items()}

    def _record_triton_custom_library_metadata(self, obj: Any) -> None:
        metadata = getattr(obj, "metadata", None)
        if metadata is None:
            return
        library_path = str(
            getattr(metadata, "triton_chipyard_custom_library_path", "") or ""
        )
        if not library_path:
            return
        library_path = os.path.abspath(os.path.expanduser(library_path))
        entry = {
            "path": library_path,
            "sha256": str(
                getattr(
                    metadata,
                    "triton_chipyard_custom_library_sha256",
                    "",
                )
                or ""
            ),
            "basename": os.path.basename(library_path),
            "config_path": str(
                getattr(metadata, "triton_chipyard_custom_config_path", "") or ""
            ),
            "config_sha256": str(
                getattr(
                    metadata,
                    "triton_chipyard_custom_config_sha256",
                    "",
                )
                or ""
            ),
        }
        previous = self.chipyard_triton_custom_libraries.get(library_path)
        if previous is not None and previous != entry:
            self._block_chipyard_runner(
                "conflicting Triton custom library metadata for " + library_path
            )
            return
        self.chipyard_triton_custom_libraries[library_path] = entry

    def _extract_compile_variants(self, compiled_kernel: Any) -> list[dict[str, Any]]:
        compile_results = getattr(compiled_kernel, "compile_results", None)
        if not compile_results:
            return []

        variants: list[dict[str, Any]] = []
        for compile_result in compile_results:
            variant: dict[str, Any] = {}
            config_obj = getattr(compile_result, "config", None)
            if config_obj is not None:
                variant["config"] = repr(config_obj)
                variant["config_dict"] = self._config_to_dict(config_obj)

            kernel_obj = getattr(compile_result, "kernel", None)
            for candidate in (compile_result, kernel_obj):
                if candidate is None:
                    continue
                metadata_group = self._extract_metadata_group(candidate)
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

    def _extract_selected_launch(self, compiled_kernel: Any) -> Optional[dict[str, Any]]:
        selected_config = None
        launchers = getattr(compiled_kernel, "launchers", None) or []
        if len(launchers) == 1:
            selected_config = getattr(launchers[0], "config", None)

        compile_results = getattr(compiled_kernel, "compile_results", None) or []
        selected_result = None
        selected_config_dict = self._config_to_dict(selected_config)
        if selected_config is not None:
            for compile_result in compile_results:
                if self._config_to_dict(getattr(compile_result, "config", None)) == selected_config_dict:
                    selected_result = compile_result
                    break
        elif len(compile_results) == 1:
            selected_result = compile_results[0]
            selected_config = getattr(selected_result, "config", None)
            selected_config_dict = self._config_to_dict(selected_config)
        elif compile_results:
            # Pointwise-style CPU kernels can keep multiple compile results without a
            # single selected launcher. The generated build script already links the
            # first compiled variant in this case, so use that same variant as the
            # fallback source of grid metadata for runner generation.
            selected_result = compile_results[0]
            selected_config = getattr(selected_result, "config", None)
            selected_config_dict = self._config_to_dict(selected_config)

        metadata_group = None
        for candidate in (
            selected_result,
            getattr(selected_result, "kernel", None) if selected_result is not None else None,
        ):
            if candidate is None:
                continue
            metadata_group = self._extract_metadata_group(candidate)
            if metadata_group is not None:
                break

        inductor_meta = None
        if selected_result is not None:
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
            selected_launch["inductor_meta"] = self._json_safe(inductor_meta)
        if metadata_group is not None:
            selected_launch["metadata_group"] = metadata_group
        return selected_launch

    def _populate_chipyard_compiled_kernel_entry(
        self, entry: dict[str, Any], compiled_kernel: Any
    ) -> None:
        precompile = getattr(compiled_kernel, "precompile", None)
        if callable(precompile):
            compile_results = getattr(compiled_kernel, "compile_results", None) or []
            launchers = getattr(compiled_kernel, "launchers", None) or []
            if not compile_results or not launchers:
                try:
                    precompile()
                except Exception as exc:
                    entry["precompile_error"] = repr(exc)
        entry["compiled_object_type"] = type(compiled_kernel).__name__
        self._record_triton_custom_library_metadata(compiled_kernel)
        for compile_result in getattr(compiled_kernel, "compile_results", None) or []:
            self._record_triton_custom_library_metadata(compile_result)
            kernel_obj = getattr(compile_result, "kernel", None)
            if kernel_obj is not None:
                self._record_triton_custom_library_metadata(kernel_obj)
        triton_meta = getattr(compiled_kernel, "triton_meta", None)
        if isinstance(triton_meta, dict):
            entry["triton_meta"] = self._summarize_triton_meta(triton_meta)
        variants = self._extract_compile_variants(compiled_kernel)
        if variants:
            entry["compiled_variants"] = variants
        selected_launch = self._extract_selected_launch(compiled_kernel)
        if selected_launch is not None:
            entry["selected_launch"] = selected_launch

    def _load_compiled_kernel(
        self,
        *,
        kernel_name: str,
        module_path: str,
        module_cache_key: Optional[str] = None,
    ) -> Optional[Any]:
        if not kernel_name or not module_path or not os.path.exists(module_path):
            return None
        cache_key = module_cache_key or Path(module_path).stem
        try:
            mod = PyCodeCache.load_by_key_path(cache_key, module_path)
        except Exception:
            return None
        return getattr(mod, kernel_name, None)

    def _load_compiled_kernel_from_module(
        self, kernel_entry: dict[str, Any]
    ) -> Optional[Any]:
        kernel_name = str(kernel_entry.get("kernel_name", ""))
        kernel_path = kernel_entry.get("kernel_path")
        if not kernel_name or not isinstance(kernel_path, str):
            return None
        compiled_kernel = self._load_compiled_kernel(
            kernel_name=kernel_name,
            module_path=kernel_path,
            module_cache_key=Path(kernel_path).stem,
        )
        if compiled_kernel is None:
            kernel_entry["module_load_error"] = (
                f"missing attribute {kernel_name!r} in {kernel_path!r}"
            )
            return None
        kernel_entry["compiled_kernel_source"] = "kernel_module"
        return compiled_kernel

    def _finalize_kernel_artifacts(self) -> None:
        for kernel_name, kernel_entry in self.chipyard_kernels.items():
            compiled_kernel = None
            if self._chipyard_autotune_scope:
                compiled_kernel = self._chipyard_autotune_scope.get(kernel_name)
                if compiled_kernel is not None:
                    kernel_entry["compiled_kernel_source"] = "autotune_scope"
            if compiled_kernel is None:
                compiled_kernel = self._load_compiled_kernel_from_module(kernel_entry)
            if compiled_kernel is None:
                continue
            self._populate_chipyard_compiled_kernel_entry(kernel_entry, compiled_kernel)

    def _finalize_chipyard_deferred_autotune_sites(self) -> list[dict[str, Any]]:
        raw_sites = getattr(V.graph, "chipyard_deferred_autotune_sites", [])
        if not isinstance(raw_sites, list):
            return []

        finalized_sites: list[dict[str, Any]] = []
        for raw_site in raw_sites:
            if not isinstance(raw_site, dict):
                continue
            site_entry = dict(raw_site)
            raw_choices = raw_site.get("choices", [])
            finalized_choices: list[dict[str, Any]] = []
            default_kernel_name = str(raw_site.get("default_kernel_name", ""))
            raw_default_choice_index = raw_site.get("default_choice_index", 0)
            default_choice_index = (
                int(raw_default_choice_index)
                if isinstance(raw_default_choice_index, int)
                and not isinstance(raw_default_choice_index, bool)
                else 0
            )
            for choice_index, raw_choice in enumerate(raw_choices):
                if not isinstance(raw_choice, dict):
                    continue
                choice_entry = dict(raw_choice)
                raw_candidate_index = choice_entry.get("candidate_index")
                candidate_index = (
                    int(raw_candidate_index)
                    if isinstance(raw_candidate_index, int)
                    and not isinstance(raw_candidate_index, bool)
                    else choice_index
                )
                choice_entry["candidate_index"] = candidate_index
                raw_kernel_name = str(choice_entry.get("kernel_name", ""))
                kernel_name = raw_kernel_name
                module_path = choice_entry.get("module_path")
                module_cache_key = choice_entry.get("module_cache_key")
                dispatch_kind = str(choice_entry.get("dispatch_kind", ""))
                wrapper_kernel_name = str(choice_entry.get("wrapper_kernel_name", ""))
                wrapper_dispatch_enabled = False
                if wrapper_kernel_name:
                    wrapper_kernel_entry = self.chipyard_kernels.get(wrapper_kernel_name)
                    wrapper_kernel_path = (
                        wrapper_kernel_entry.get("kernel_path")
                        if isinstance(wrapper_kernel_entry, dict)
                        else None
                    )
                    if isinstance(wrapper_kernel_path, str) and wrapper_kernel_path:
                        choice_entry["raw_kernel_name"] = raw_kernel_name
                        choice_entry["kernel_name"] = wrapper_kernel_name
                        choice_entry["module_path"] = wrapper_kernel_path
                        choice_entry["module_cache_key"] = Path(
                            wrapper_kernel_path
                        ).stem
                        kernel_name = wrapper_kernel_name
                        module_path = wrapper_kernel_path
                        module_cache_key = choice_entry["module_cache_key"]
                        wrapper_dispatch_enabled = True
                choice_entry["dispatch_kind"] = (
                    dispatch_kind if wrapper_dispatch_enabled else ""
                )
                if (
                    kernel_name
                    and isinstance(module_path, str)
                    and isinstance(module_cache_key, str)
                ):
                    compiled_kernel = self._load_compiled_kernel(
                        kernel_name=kernel_name,
                        module_path=module_path,
                        module_cache_key=module_cache_key,
                    )
                    if compiled_kernel is not None:
                        self._populate_chipyard_compiled_kernel_entry(
                            choice_entry, compiled_kernel
                        )
                    else:
                        choice_entry["module_load_error"] = (
                            f"missing attribute {kernel_name!r} in {module_path!r}"
                        )
                choice_triton_meta = choice_entry.get("triton_meta")
                if isinstance(choice_triton_meta, dict):
                    choice_entry["signature_items"] = self._runtime_signature_items(
                        choice_triton_meta.get("signature"),
                        choice_triton_meta.get("signature_names"),
                    )
                if kernel_name and kernel_name == default_kernel_name:
                    default_choice_index = candidate_index
                finalized_choices.append(choice_entry)
            site_entry["choices"] = finalized_choices
            site_entry["default_choice_index"] = default_choice_index
            finalized_sites.append(site_entry)
        return finalized_sites

    @staticmethod
    def _cpp_comment(text: Any) -> str:
        return str(text).replace("/*", "/ *").replace("*/", "* /")

    @staticmethod
    def _cpp_string_literal(text: Any) -> str:
        return json.dumps(str(text), ensure_ascii=True)

    @staticmethod
    def _signature_type_from_runtime_value(value: Any) -> str:
        if isinstance(value, bool):
            return "i1"
        if isinstance(value, int):
            return "i64"
        if isinstance(value, float):
            return "fp32"
        return "i64"

    @staticmethod
    def _signature_type_from_torch_dtype_name(dtype_name: Any) -> str:
        normalized = str(dtype_name).split(".")[-1]
        return {
            "bool": "i1",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
            "float64": "fp64",
        }.get(normalized, "i64")

    @classmethod
    def _runtime_signature_items(
        cls,
        signature: Any,
        signature_names: Any,
    ) -> list[tuple[str, str]]:
        if not isinstance(signature, dict):
            return []
        ordered_names: list[str] = []
        if isinstance(signature_names, (list, tuple)):
            ordered_names.extend(str(name) for name in signature_names)
        else:
            ordered_names.extend(str(name) for name in signature.keys())
        seen: set[str] = set()
        items: list[tuple[str, str]] = []
        for raw_name in [*ordered_names, *map(str, signature.keys())]:
            name = str(raw_name)
            if name in seen or name not in signature:
                continue
            seen.add(name)
            signature_type = str(signature[name])
            if signature_type == "constexpr":
                continue
            items.append((name, signature_type))
        return items

    @classmethod
    def _kernel_decl_from_signature(
        cls, signature_items: list[tuple[str, str]]
    ) -> str:
        decl_items: list[str] = []
        for arg_index, (_arg_name, signature_type) in enumerate(signature_items):
            if signature_type.startswith("*"):
                decl_items.extend(
                    [
                        f"int64_t arg{arg_index}_rank",
                        f"void* arg{arg_index}_descriptor",
                    ]
                )
            else:
                decl_items.append(
                    f"{cls._cpp_scalar_type(signature_type)} arg{arg_index}"
                )
        decl_items.extend(
            [
                "int32_t gridX",
                "int32_t gridY",
                "int32_t gridZ",
                "int32_t pidX",
                "int32_t pidY",
                "int32_t pidZ",
            ]
        )
        return ", ".join(decl_items)

    @staticmethod
    def _extra_launcher_args(kernel_info: Any) -> list[str]:
        if not isinstance(kernel_info, dict):
            return []
        selected_launch = kernel_info.get("selected_launch")
        if not isinstance(selected_launch, dict):
            return []
        inductor_meta = selected_launch.get("inductor_meta")
        if not isinstance(inductor_meta, dict):
            return []
        raw_args = inductor_meta.get("extra_launcher_args")
        if not isinstance(raw_args, (list, tuple)):
            return []
        return [str(name) for name in raw_args]

    @staticmethod
    def _kernel_grid_expr(kernel_info: Any) -> Optional[Any]:
        if not isinstance(kernel_info, dict):
            return None
        selected_launch = kernel_info.get("selected_launch")
        if not isinstance(selected_launch, dict):
            return None
        inductor_meta = selected_launch.get("inductor_meta")
        if not isinstance(inductor_meta, dict):
            return None
        cfg = selected_launch.get("config_dict", {})
        if not isinstance(cfg, dict):
            cfg = {}
        try:
            from ...runtime.triton_heuristics import GridExpr

            return GridExpr.from_meta(inductor_meta, cfg, mode="cpp")
        except Exception:
            return None

    @classmethod
    def _concrete_grid_args(cls, kernel_info: Any) -> Optional[tuple[int, int, int]]:
        grid_expr = cls._kernel_grid_expr(kernel_info)
        if grid_expr is None:
            return None
        if getattr(grid_expr, "prefix", None):
            return None
        grid_values = (
            cls._runtime_int_value(getattr(grid_expr, "x_grid", None)),
            cls._runtime_int_value(getattr(grid_expr, "y_grid", None)),
            cls._runtime_int_value(getattr(grid_expr, "z_grid", None)),
        )
        if any(value is None for value in grid_values):
            return None
        return int(grid_values[0]), int(grid_values[1]), int(grid_values[2])

    @classmethod
    def _fixed_grid_extra_args(
        cls, kernel_info: Any
    ) -> Optional[tuple[int, int, int]]:
        if not isinstance(kernel_info, dict):
            return None
        selected_launch = kernel_info.get("selected_launch")
        if not isinstance(selected_launch, dict):
            return None
        inductor_meta = selected_launch.get("inductor_meta")
        if not isinstance(inductor_meta, dict):
            return None
        if str(inductor_meta.get("grid_type", "")) != "FixedGrid":
            return None
        raw_extra_args = kernel_info.get("extra_args")
        if not isinstance(raw_extra_args, (list, tuple)) or len(raw_extra_args) != 3:
            return None
        grid_values = tuple(
            cls._runtime_int_value(raw_extra_args[index]) for index in range(3)
        )
        if any(value is None for value in grid_values):
            return None
        return int(grid_values[0]), int(grid_values[1]), int(grid_values[2])

    @staticmethod
    def _materialize_grid_placeholders(
        grid_expr: Any,
        arg_index_by_name: dict[str, int],
        num_step_args: int,
    ) -> Optional[Any]:
        if grid_expr is None:
            return None

        def replace_names(expr: Any) -> Any:
            if isinstance(expr, int):
                return expr
            text = str(expr)
            for name, arg_index in sorted(
                arg_index_by_name.items(), key=lambda item: len(item[0]), reverse=True
            ):
                replacement = name if name.isidentifier() else f"arg{arg_index}"
                text = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                    replacement,
                    text,
                )
            return text

        return SimpleNamespace(
            prefix=[replace_names(line) for line in getattr(grid_expr, "prefix", [])],
            x_grid=replace_names(getattr(grid_expr, "x_grid", 1)),
            y_grid=replace_names(getattr(grid_expr, "y_grid", 1)),
            z_grid=replace_names(getattr(grid_expr, "z_grid", 1)),
        )

    @staticmethod
    def _chipyard_source_kind_literal(source_kind: str) -> str:
        return {
            "buffer": "ChipyardValueSourceKind::Buffer",
            "input": "ChipyardValueSourceKind::InputBlob",
            "weight": "ChipyardValueSourceKind::WeightBlob",
            "immediate": "ChipyardValueSourceKind::ImmediateScalar",
        }.get(source_kind, "ChipyardValueSourceKind::None")

    def _resolve_chipyard_value_source(
        self,
        serialized_arg: Any,
        *,
        buffer_index_map: dict[str, int],
        input_index_map: dict[str, int],
        weight_index_map: dict[str, int],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(serialized_arg, dict):
            return None
        kind = str(serialized_arg.get("kind", ""))
        name = str(serialized_arg.get("name", ""))
        if kind == "constant":
            weight_index = weight_index_map.get(name)
            buffer_entry = self.chipyard_buffers.get(name)
            if weight_index is None or not isinstance(buffer_entry, dict):
                return None
            return {
                "source_kind": "weight",
                "source_index": weight_index,
                "buffer_entry": buffer_entry,
                "view_buffer_entry": buffer_entry,
                "source_buffer_entry": buffer_entry,
                "source_name": name,
                "view_name": name,
            }
        if kind == "graph_input":
            input_index = input_index_map.get(name)
            buffer_entry = self.chipyard_buffers.get(name)
            if input_index is None or not isinstance(buffer_entry, dict):
                return None
            return {
                "source_kind": "input",
                "source_index": input_index,
                "buffer_entry": buffer_entry,
                "view_buffer_entry": buffer_entry,
                "source_buffer_entry": buffer_entry,
                "source_name": name,
                "view_name": name,
            }
        if kind != "buffer":
            return None

        buffer_entry = self.chipyard_buffers.get(name)
        if not isinstance(buffer_entry, dict):
            return None
        source_name = self._resolve_chipyard_alias_name(
            serialized_arg.get("alias_of", buffer_entry.get("alias_of", name))
        )
        source_buffer_entry = self.chipyard_buffers.get(source_name) or buffer_entry
        if not isinstance(source_buffer_entry, dict):
            return None
        source_role = str(source_buffer_entry.get("role", "intermediate"))
        if source_role == "constant":
            source_index = weight_index_map.get(source_name)
            source_kind = "weight"
        elif source_role == "graph_input":
            source_index = input_index_map.get(source_name)
            source_kind = "input"
        else:
            source_index = buffer_index_map.get(source_name)
            source_kind = "buffer"
        if source_index is None:
            return None
        return {
            "source_kind": source_kind,
            "source_index": source_index,
            "buffer_entry": buffer_entry,
            "view_buffer_entry": buffer_entry,
            "source_buffer_entry": source_buffer_entry,
            "source_name": source_name,
            "view_name": name,
        }

    def _build_chipyard_scalar_arg_plan(
        self,
        serialized_arg: Any,
        signature_type: Optional[str],
        *,
        buffer_index_map: dict[str, int],
        input_index_map: dict[str, int],
        weight_index_map: dict[str, int],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(serialized_arg, dict):
            return None
        kind = str(serialized_arg.get("kind", ""))
        normalized_type = signature_type
        if kind == "scalar":
            if normalized_type is None:
                normalized_type = self._signature_type_from_runtime_value(
                    serialized_arg.get("value")
                )
            return {
                "source_kind": "immediate",
                "source_index": -1,
                "offset_elems": 0,
                "dtype": normalized_type,
                "immediate_bits": self._chipyard_scalar_bits_literal(
                    normalized_type,
                    serialized_arg.get("value"),
                ),
                "is_immediate": True,
            }
        source_ref = self._resolve_chipyard_value_source(
            serialized_arg,
            buffer_index_map=buffer_index_map,
            input_index_map=input_index_map,
            weight_index_map=weight_index_map,
        )
        if source_ref is None:
            return None
        buffer_entry = source_ref["buffer_entry"]
        size_hint = buffer_entry.get("size_hint", [])
        if not isinstance(size_hint, list):
            return None
        numel_hint = 1
        for size_value in size_hint:
            numel_hint *= int(size_value)
        if numel_hint != 1:
            return None
        if normalized_type is None:
            normalized_type = self._signature_type_from_torch_dtype_name(
                buffer_entry.get("dtype")
            )
        return {
            "source_kind": source_ref["source_kind"],
            "source_index": int(source_ref["source_index"]),
            "offset_elems": int(buffer_entry.get("layout_offset_hint", 0)),
            "dtype": normalized_type,
            "immediate_bits": "UINT64_C(0)",
            "is_immediate": False,
        }

    def _chipyard_hint_int_value(self, value: Any, *, fallback: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return self._size_hint_int(value, fallback=fallback)

    def _chipyard_hint_int_list(self, values: Any) -> Optional[list[int]]:
        if not isinstance(values, (list, tuple)):
            return None
        hinted_values: list[int] = []
        for value in values:
            hinted_value = self._chipyard_hint_int_value(value, fallback=-1)
            if hinted_value < 0:
                return None
            hinted_values.append(hinted_value)
        return hinted_values

    def _normalize_chipyard_tensor_view_override(
        self,
        view_override: Any,
    ) -> Optional[tuple[list[int], list[int], int]]:
        if not isinstance(view_override, dict):
            return None
        size_values = view_override.get("size_hint")
        if not isinstance(size_values, (list, tuple)):
            size_values = view_override.get("size")
        stride_values = view_override.get("stride_hint")
        if not isinstance(stride_values, (list, tuple)):
            stride_values = view_override.get("stride")
        size_hint = self._chipyard_hint_int_list(size_values)
        stride_hint = self._chipyard_hint_int_list(stride_values)
        if size_hint is None or stride_hint is None:
            return None
        if (
            len(size_hint) != len(stride_hint)
            or len(size_hint) > CHIPYARD_MAX_TENSOR_RANK
        ):
            return None
        offset_value = view_override.get(
            "offset_hint",
            view_override.get(
                "layout_offset_hint",
                view_override.get("offset", 0),
            ),
        )
        offset_hint = self._chipyard_hint_int_value(offset_value, fallback=0)
        return size_hint, stride_hint, offset_hint

    def _chipyard_valid_tensor_view_override(
        self,
        view_override: Any,
    ) -> Optional[dict[str, Any]]:
        if self._normalize_chipyard_tensor_view_override(view_override) is None:
            return None
        return view_override if isinstance(view_override, dict) else None

    @staticmethod
    def _chipyard_runtime_arg_is_output_pointer(runtime_name: str) -> bool:
        return runtime_name.startswith("out_ptr") or runtime_name.startswith("in_out_ptr")

    @staticmethod
    def _chipyard_dtype_signature_name(dtype_name: Any) -> Optional[str]:
        if dtype_name is None:
            return None
        normalized = str(dtype_name).strip().lower()
        while normalized.startswith("*"):
            normalized = normalized[1:]
        normalized = normalized.split(".")[-1]
        return {
            "bool": "i1",
            "i1": "i1",
            "int8": "i8",
            "i8": "i8",
            "uint8": "u8",
            "u8": "u8",
            "int16": "i16",
            "i16": "i16",
            "uint16": "u16",
            "u16": "u16",
            "float16": "fp16",
            "half": "fp16",
            "fp16": "fp16",
            "f16": "fp16",
            "bfloat16": "bf16",
            "bf16": "bf16",
            "int32": "i32",
            "i32": "i32",
            "uint32": "u32",
            "u32": "u32",
            "float": "fp32",
            "float32": "fp32",
            "fp32": "fp32",
            "f32": "fp32",
            "int64": "i64",
            "i64": "i64",
            "uint64": "u64",
            "u64": "u64",
            "double": "fp64",
            "float64": "fp64",
            "fp64": "fp64",
            "f64": "fp64",
        }.get(normalized)

    @classmethod
    def _chipyard_dtype_nbytes(cls, dtype_name: Any) -> Optional[int]:
        signature_name = cls._chipyard_dtype_signature_name(dtype_name)
        if signature_name is None:
            return None
        return {
            "i1": 1,
            "i8": 1,
            "u8": 1,
            "i16": 2,
            "u16": 2,
            "fp16": 2,
            "bf16": 2,
            "i32": 4,
            "u32": 4,
            "fp32": 4,
            "i64": 8,
            "u64": 8,
            "fp64": 8,
        }.get(signature_name)

    @classmethod
    def _chipyard_view_dtype_matches_signature(
        cls,
        view_override: dict[str, Any],
        signature_type: str,
    ) -> bool:
        view_dtype = cls._chipyard_dtype_signature_name(view_override.get("dtype"))
        signature_dtype = cls._chipyard_dtype_signature_name(signature_type)
        if view_dtype is None or signature_dtype is None:
            return True
        return view_dtype == signature_dtype

    @staticmethod
    def _chipyard_int_field(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                return int(text)
        return None

    @classmethod
    def _chipyard_entry_nbytes(cls, entry: Any) -> Optional[int]:
        if not isinstance(entry, dict):
            return None
        for key in ("nbytes_hint", "input_nbytes", "weight_nbytes", "nbytes"):
            nbytes = cls._chipyard_int_field(entry.get(key))
            if nbytes is not None and nbytes > 0:
                return nbytes
        nbytes_expr = cls._chipyard_int_field(entry.get("nbytes_expr"))
        if nbytes_expr is not None and nbytes_expr > 0:
            return nbytes_expr
        return None

    @classmethod
    def _chipyard_source_capacity_elems(
        cls,
        source_ref: dict[str, Any],
        signature_type: str,
    ) -> Optional[int]:
        source_buffer_entry = source_ref.get("source_buffer_entry")
        nbytes = cls._chipyard_entry_nbytes(source_buffer_entry)
        if nbytes is None:
            nbytes = cls._chipyard_entry_nbytes(source_ref.get("buffer_entry"))
        elem_nbytes = cls._chipyard_dtype_nbytes(signature_type)
        if nbytes is None or elem_nbytes is None or elem_nbytes <= 0:
            return None
        return nbytes // elem_nbytes

    @staticmethod
    def _chipyard_storage_index_span(
        size_hint: list[int],
        stride_hint: list[int],
        offset_elems: int,
    ) -> Optional[tuple[int, int]]:
        if len(size_hint) != len(stride_hint):
            return None
        if offset_elems < 0:
            return None
        if any(int(size_value) < 0 for size_value in size_hint):
            return None
        if any(int(size_value) == 0 for size_value in size_hint):
            return (0, -1)
        min_index = int(offset_elems)
        max_index = int(offset_elems)
        for size_value, stride_value in zip(size_hint, stride_hint):
            span = max(int(size_value) - 1, 0) * int(stride_value)
            if span < 0:
                min_index += span
            else:
                max_index += span
        if min_index < 0:
            return None
        return min_index, max_index

    @staticmethod
    def _chipyard_required_storage_elems(
        size_hint: list[int],
        stride_hint: list[int],
        offset_elems: int,
    ) -> Optional[int]:
        storage_span = ChipyardWrapperCodegen._chipyard_storage_index_span(
            size_hint,
            stride_hint,
            offset_elems,
        )
        if storage_span is None:
            return None
        min_index, max_index = storage_span
        if max_index < min_index:
            return 0
        return max_index + 1

    def _chipyard_tensor_view_storage_span(
        self,
        view_override: Any,
    ) -> Optional[tuple[int, int]]:
        normalized_override = self._normalize_chipyard_tensor_view_override(
            view_override
        )
        if normalized_override is None:
            return None
        size_hint, stride_hint, offset_elems = normalized_override
        return self._chipyard_storage_index_span(
            size_hint,
            stride_hint,
            offset_elems,
        )

    def _chipyard_tensor_view_fits_source(
        self,
        view_override: dict[str, Any],
        source_ref: dict[str, Any],
        signature_type: str,
    ) -> bool:
        normalized_override = self._normalize_chipyard_tensor_view_override(view_override)
        if normalized_override is None:
            return False
        size_hint, stride_hint, offset_elems = normalized_override
        required_elems = self._chipyard_required_storage_elems(
            size_hint,
            stride_hint,
            offset_elems,
        )
        if required_elems is None:
            return False
        capacity_elems = self._chipyard_source_capacity_elems(
            source_ref,
            signature_type,
        )
        if capacity_elems is None:
            return True
        return required_elems <= capacity_elems

    def _chipyard_tensor_view_is_usable_for_source(
        self,
        view_override: Any,
        source_ref: dict[str, Any],
        signature_type: str,
    ) -> Optional[dict[str, Any]]:
        view_override = self._chipyard_valid_tensor_view_override(view_override)
        if view_override is None:
            return None
        if not self._chipyard_view_dtype_matches_signature(view_override, signature_type):
            return None
        if not self._chipyard_tensor_view_fits_source(
            view_override,
            source_ref,
            signature_type,
        ):
            return None
        return view_override

    def _chipyard_source_match_names(
        self,
        serialized_arg: Any,
        source_ref: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        raw_names: set[str] = set()
        if isinstance(serialized_arg, dict):
            for key in ("name", "alias_of"):
                value = serialized_arg.get(key)
                if value:
                    raw_names.add(str(value))
        for key in ("source_name", "view_name"):
            value = source_ref.get(key)
            if value:
                raw_names.add(str(value))
        for entry_key in ("buffer_entry", "view_buffer_entry", "source_buffer_entry"):
            entry = source_ref.get(entry_key)
            if not isinstance(entry, dict):
                continue
            for key in ("name", "alias_of"):
                value = entry.get(key)
                if value:
                    raw_names.add(str(value))
        resolved_names = {self._resolve_chipyard_alias_name(name) for name in raw_names}
        return raw_names, resolved_names

    def _chipyard_deferred_input_match_names(
        self,
        deferred_input: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        raw_names: set[str] = set()
        for key in ("name", "alias_of", "source_name", "view_name"):
            value = deferred_input.get(key)
            if value:
                raw_names.add(str(value))
        resolved_names = {self._resolve_chipyard_alias_name(name) for name in raw_names}
        return raw_names, resolved_names

    def _select_chipyard_deferred_site_view_override(
        self,
        *,
        runtime_name: str,
        signature_type: str,
        serialized_arg: Any,
        source_ref: dict[str, Any],
        deferred_site_inputs: list[dict[str, Any]],
        consumed_deferred_site_input_indices: set[int],
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        if self._chipyard_runtime_arg_is_output_pointer(runtime_name):
            return None, None
        source_raw_names, source_resolved_names = self._chipyard_source_match_names(
            serialized_arg,
            source_ref,
        )
        candidates: list[tuple[int, dict[str, Any], set[str], set[str]]] = []
        for candidate_index, deferred_input in enumerate(deferred_site_inputs):
            if candidate_index in consumed_deferred_site_input_indices:
                continue
            view_override = self._chipyard_tensor_view_is_usable_for_source(
                deferred_input,
                source_ref,
                signature_type,
            )
            if view_override is None:
                continue
            candidate_raw_names, candidate_resolved_names = (
                self._chipyard_deferred_input_match_names(deferred_input)
            )
            candidates.append(
                (
                    candidate_index,
                    view_override,
                    candidate_raw_names,
                    candidate_resolved_names,
                )
            )

        def pick_unique(
            matches: list[tuple[int, dict[str, Any], set[str], set[str]]],
            match_kind: str,
        ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
            if len(matches) == 1:
                consumed_deferred_site_input_indices.add(matches[0][0])
                return matches[0][1], None
            if len(matches) > 1:
                return (
                    None,
                    f"ambiguous {match_kind} deferred input view override for {runtime_name!r}",
                )
            return None, None

        view_override, error_reason = pick_unique(
            [
                candidate
                for candidate in candidates
                if candidate[2] and candidate[2].intersection(source_raw_names)
            ],
            "exact-name",
        )
        if view_override is not None or error_reason is not None:
            return view_override, error_reason

        view_override, error_reason = pick_unique(
            [
                candidate
                for candidate in candidates
                if candidate[3] and candidate[3].intersection(source_resolved_names)
            ],
            "alias-root",
        )
        if view_override is not None or error_reason is not None:
            return view_override, error_reason

        source_span = self._chipyard_tensor_view_storage_span(
            source_ref.get("buffer_entry")
        )
        span_matches: list[tuple[int, dict[str, Any], set[str], set[str]]] = []
        if source_span is not None:
            span_matches = [
                candidate
                for candidate in candidates
                if self._chipyard_tensor_view_storage_span(candidate[1])
                == source_span
            ]
        view_override, error_reason = pick_unique(
            span_matches,
            "exact-span",
        )
        if view_override is not None or error_reason is not None:
            return view_override, error_reason

        return None, None

    def _build_chipyard_tensor_arg_plan(
        self,
        serialized_arg: Any,
        signature_type: str,
        *,
        buffer_index_map: dict[str, int],
        input_index_map: dict[str, int],
        weight_index_map: dict[str, int],
        view_override: Optional[dict[str, Any]] = None,
        source_ref: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if source_ref is None:
            source_ref = self._resolve_chipyard_value_source(
                serialized_arg,
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
            )
        if source_ref is None:
            return None
        buffer_entry = source_ref["buffer_entry"]
        if view_override is not None:
            view_override = self._chipyard_tensor_view_is_usable_for_source(
                view_override,
                source_ref,
                signature_type,
            )
            if view_override is None:
                return None
            normalized_override = self._normalize_chipyard_tensor_view_override(
                view_override
            )
            assert normalized_override is not None
            size_hint, stride_hint, offset_elems = normalized_override
        else:
            size_hint = buffer_entry.get("size_hint", [])
            stride_hint = buffer_entry.get("stride_hint", [])
            if not isinstance(size_hint, list) or not isinstance(stride_hint, list):
                return None
            offset_elems = int(buffer_entry.get("layout_offset_hint", 0))
            if not self._chipyard_tensor_view_fits_source(
                {
                    "size_hint": size_hint,
                    "stride_hint": stride_hint,
                    "offset_hint": offset_elems,
                },
                source_ref,
                signature_type,
            ):
                return None
        rank = len(size_hint)
        if rank > CHIPYARD_MAX_TENSOR_RANK:
            return None
        padded_sizes = [int(value) for value in size_hint] + [1] * (
            CHIPYARD_MAX_TENSOR_RANK - rank
        )
        padded_strides = [int(value) for value in stride_hint] + [1] * (
            CHIPYARD_MAX_TENSOR_RANK - rank
        )
        return {
            "source_kind": source_ref["source_kind"],
            "source_index": int(source_ref["source_index"]),
            "offset_elems": int(offset_elems),
            "rank": rank,
            "dtype": signature_type[1:],
            "sizes": padded_sizes,
            "strides": padded_strides,
        }

    def _build_chipyard_custom_call_plan(
        self,
        *,
        step: dict[str, Any],
        buffer_index_map: dict[str, int],
        input_index_map: dict[str, int],
        weight_index_map: dict[str, int],
    ) -> dict[str, Any]:
        plan: dict[str, Any] = {
            "step_index": int(step.get("index", -1)),
            "symbol": str(step.get("symbol", "")),
            "op_name": str(step.get("op_name", "")),
            "runtime_args": [],
            "tensor_plans": [],
            "scalar_plans": [],
            "argument_signature_types": [],
            "supported": True,
            "error_reason": "",
        }
        arguments = step.get("arguments", [])
        if not isinstance(arguments, list):
            plan["supported"] = False
            plan["error_reason"] = "custom call arguments are not a list"
            return plan
        for argument in arguments:
            if not isinstance(argument, dict):
                plan["supported"] = False
                plan["error_reason"] = "custom call argument metadata is invalid"
                return plan
            argument_kind = str(argument.get("kind", ""))
            signature_type = str(argument.get("signature_type", ""))
            serialized_value = argument.get("value")
            plan["argument_signature_types"].append(signature_type)
            if argument_kind == "tensor":
                if not signature_type.startswith("*"):
                    plan["supported"] = False
                    plan["error_reason"] = "custom Tensor argument lacks pointer dtype"
                    return plan
                tensor_plan = self._build_chipyard_tensor_arg_plan(
                    serialized_value,
                    signature_type,
                    buffer_index_map=buffer_index_map,
                    input_index_map=input_index_map,
                    weight_index_map=weight_index_map,
                )
                if tensor_plan is None:
                    plan["supported"] = False
                    plan["error_reason"] = "unsupported custom Tensor argument binding"
                    return plan
                plan["runtime_args"].append(
                    {"kind": "tensor", "plan_index": len(plan["tensor_plans"])}
                )
                plan["tensor_plans"].append(tensor_plan)
                continue
            scalar_plan = self._build_chipyard_scalar_arg_plan(
                serialized_value,
                signature_type,
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
            )
            if scalar_plan is None:
                plan["supported"] = False
                plan["error_reason"] = "unsupported custom scalar argument binding"
                return plan
            plan["runtime_args"].append(
                {"kind": "scalar", "plan_index": len(plan["scalar_plans"])}
            )
            plan["scalar_plans"].append(scalar_plan)

        output = step.get("output")
        if not isinstance(output, dict):
            plan["supported"] = False
            plan["error_reason"] = "custom call output metadata is invalid"
            return plan
        output_signature_type = str(output.get("signature_type", ""))
        output_tensor_plan = self._build_chipyard_tensor_arg_plan(
            {"kind": output.get("kind", "buffer"), "name": output.get("name", "")},
            output_signature_type,
            buffer_index_map=buffer_index_map,
            input_index_map=input_index_map,
            weight_index_map=weight_index_map,
        )
        if output_tensor_plan is None or output_tensor_plan.get("source_kind") != "buffer":
            plan["supported"] = False
            plan["error_reason"] = "custom call output is not an allocated runner buffer"
            return plan
        plan["output_plan"] = output_tensor_plan
        plan["output_signature_type"] = output_signature_type
        plan["output_buffer_index"] = int(output_tensor_plan["source_index"])
        return plan

    def _emit_chipyard_custom_call_runtime_plans(
        self,
        *,
        runner_lines: list[str],
        custom_call_instances: list[dict[str, Any]],
    ) -> int:
        def tensor_plan_literal(plan: dict[str, Any]) -> str:
            return (
                "  {"
                f"{self._chipyard_source_kind_literal(str(plan.get('source_kind', 'none')))}, "
                f"static_cast<int32_t>({int(plan.get('source_index', -1))}), "
                f"static_cast<int32_t>({int(plan.get('offset_elems', 0))}), "
                f"static_cast<int32_t>({int(plan.get('rank', 0))}), "
                f"ChipyardDType::{self._chipyard_dtype_enum_name(plan.get('dtype'))}, "
                "{"
                + ", ".join(
                    f"static_cast<int64_t>({int(value)})"
                    for value in plan.get("sizes", [1] * CHIPYARD_MAX_TENSOR_RANK)
                )
                + "}, {"
                + ", ".join(
                    f"static_cast<int64_t>({int(value)})"
                    for value in plan.get("strides", [1] * CHIPYARD_MAX_TENSOR_RANK)
                )
                + "}}"
            )

        def scalar_plan_literal(plan: dict[str, Any]) -> str:
            return (
                "  {"
                f"{self._chipyard_source_kind_literal(str(plan.get('source_kind', 'none')))}, "
                f"static_cast<int32_t>({int(plan.get('source_index', -1))}), "
                f"static_cast<int32_t>({int(plan.get('offset_elems', 0))}), "
                f"ChipyardDType::{self._chipyard_dtype_enum_name(plan.get('dtype'))}, "
                f"{str(plan.get('immediate_bits', 'UINT64_C(0)'))}, "
                f"static_cast<int32_t>({1 if bool(plan.get('is_immediate', False)) else 0})"
                "}"
            )

        declarations: dict[str, list[str]] = {}
        for instance in custom_call_instances:
            symbol = str(instance["symbol"])
            decl_args: list[str] = []
            for argument_index, signature_type in enumerate(
                instance.get("argument_signature_types", [])
            ):
                if str(signature_type).startswith("*"):
                    decl_args.extend(
                        [
                            f"int64_t arg{argument_index}_rank",
                            f"void* arg{argument_index}_descriptor",
                        ]
                    )
                else:
                    decl_args.append(
                        f"{self._cpp_scalar_type(str(signature_type))} arg{argument_index}"
                    )
            decl_args.extend(["int64_t output_rank", "void* output_descriptor"])
            previous = declarations.get(symbol)
            if previous is not None and previous != decl_args:
                raise RuntimeError(
                    f"custom op symbol {symbol} was registered with incompatible ABIs"
                )
            declarations[symbol] = decl_args
        if declarations:
            runner_lines.extend(
                [
                    *[
                        f'extern "C" void {symbol}({", ".join(decl_args)});'
                        for symbol, decl_args in sorted(declarations.items())
                    ],
                    "",
                ]
            )

        for instance_index, instance in enumerate(custom_call_instances):
            tensor_plans = list(instance.get("tensor_plans", []))
            scalar_plans = list(instance.get("scalar_plans", []))
            output_plan = dict(instance["output_plan"])
            if tensor_plans:
                runner_lines.extend(
                    [
                        "static const ChipyardTensorArgPlan "
                        f"kChipyardCustomCall{instance_index}TensorArgs[] = {{",
                        *[f"{tensor_plan_literal(plan)}," for plan in tensor_plans],
                        "};",
                        "",
                    ]
                )
            if scalar_plans:
                runner_lines.extend(
                    [
                        "static const ChipyardScalarArgPlan "
                        f"kChipyardCustomCall{instance_index}ScalarArgs[] = {{",
                        *[f"{scalar_plan_literal(plan)}," for plan in scalar_plans],
                        "};",
                        "",
                    ]
                )
            runner_lines.extend(
                [
                    "static const ChipyardTensorArgPlan "
                    f"kChipyardCustomCall{instance_index}Output =",
                    f"{tensor_plan_literal(output_plan)};",
                    "",
                    f"static int dispatch_custom_call_{instance_index}(",
                    "    RunnerState& state, std::uint64_t* total_cycles) {",
                    "  if (ensure_buffer_allocated(",
                    f"          state.buffers, static_cast<int32_t>({int(instance['output_buffer_index'])})) != 0) {{",
                    "    return -1;",
                    "  }",
                ]
            )
            call_args: list[str] = []
            tensor_index = 0
            scalar_index = 0
            for argument_index, runtime_arg in enumerate(instance["runtime_args"]):
                if runtime_arg["kind"] == "tensor":
                    tensor_plan = tensor_plans[tensor_index]
                    signature_type = str(
                        instance["argument_signature_types"][argument_index]
                    )
                    elem_cpp_type = self._cpp_scalar_type(signature_type[1:])
                    rank = int(tensor_plan["rank"])
                    runner_lines.extend(
                        [
                            f"  StridedMemRefType<{elem_cpp_type}, {rank}> input_desc_{argument_index}{{}};",
                            f"  UnrankedMemRefType<{elem_cpp_type}> input_{argument_index}{{}};",
                            "  if (chipyard_make_ranked_tensor_arg<"
                            f"{elem_cpp_type}, {rank}>(state, "
                            f"kChipyardCustomCall{instance_index}TensorArgs[{tensor_index}], "
                            f"&input_desc_{argument_index}, &input_{argument_index}) != 0) return -1;",
                        ]
                    )
                    call_args.extend(
                        [
                            f"input_{argument_index}.rank",
                            f"input_{argument_index}.descriptor",
                        ]
                    )
                    tensor_index += 1
                else:
                    signature_type = str(
                        instance["argument_signature_types"][argument_index]
                    )
                    cpp_type = self._cpp_scalar_type(signature_type)
                    runner_lines.extend(
                        [
                            f"  {cpp_type} input_{argument_index};",
                            f"  if (chipyard_load_scalar_arg<{cpp_type}>(state, "
                            f"kChipyardCustomCall{instance_index}ScalarArgs[{scalar_index}], "
                            f"&input_{argument_index}) != 0) return -1;",
                        ]
                    )
                    call_args.append(f"input_{argument_index}")
                    scalar_index += 1

            output_signature_type = str(instance["output_signature_type"])
            output_cpp_type = self._cpp_scalar_type(output_signature_type[1:])
            output_rank = int(output_plan["rank"])
            call_args.extend(["output.rank", "output.descriptor"])
            runner_lines.extend(
                [
                    f"  StridedMemRefType<{output_cpp_type}, {output_rank}> output_desc{{}};",
                    f"  UnrankedMemRefType<{output_cpp_type}> output{{}};",
                    "  if (chipyard_make_ranked_tensor_arg<"
                    f"{output_cpp_type}, {output_rank}>(state, "
                    f"kChipyardCustomCall{instance_index}Output, &output_desc, &output) != 0) return -1;",
                    "  const std::uint64_t cycle_start = rdcycle();",
                    f"  {instance['symbol']}({', '.join(call_args)});",
                    "  const std::uint64_t cycle_end = rdcycle();",
                    "  if (total_cycles != nullptr) {",
                    "    *total_cycles += (cycle_end - cycle_start);",
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

        runner_lines.extend(
            [
                "static int execute_custom_call(",
                "    int32_t custom_call_index,",
                "    RunnerState& state,",
                "    std::uint64_t* total_cycles) {",
                "  switch (custom_call_index) {",
                *[
                    f"    case {instance_index}: return dispatch_custom_call_{instance_index}(state, total_cycles);"
                    for instance_index in range(len(custom_call_instances))
                ],
                "    default: return -1;",
                "  }",
                "}",
                "",
            ]
        )
        return len(custom_call_instances)

    def _build_chipyard_launch_instance_plan(
        self,
        *,
        step_index: int,
        kernel_name: str,
        kernel_info: dict[str, Any],
        step_args: list[Any],
        step_arg_names: list[str],
        step_arg_types: list[Optional[str]],
        signature_items: list[tuple[str, str]],
        site_id: str,
        buffer_index_map: dict[str, int],
        input_index_map: dict[str, int],
        weight_index_map: dict[str, int],
        view_overrides_by_arg_index: Optional[dict[int, dict[str, Any]]] = None,
        deferred_site_inputs: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        plan: dict[str, Any] = {
            "step_index": step_index,
            "kernel_name": kernel_name,
            "site_id": site_id,
            "signature_items": list(signature_items),
            "step_args": list(step_args),
            "step_arg_names": list(step_arg_names),
            "step_arg_types": list(step_arg_types),
            "tensor_plans": [],
            "scalar_plans": [],
            "runtime_args": [],
            "scalar_plan_index_by_arg_index": {},
            "supported": True,
            "error_reason": "",
        }
        step_arg_index_by_name: dict[str, int] = {}
        for arg_index, arg_name in enumerate(step_arg_names[: len(step_args)]):
            step_arg_index_by_name.setdefault(arg_name, arg_index)
        plan["step_arg_index_by_name"] = step_arg_index_by_name

        if view_overrides_by_arg_index is None:
            view_overrides_by_arg_index = {}
        if deferred_site_inputs is None:
            deferred_site_inputs = []

        pointer_source_refs_by_arg_index: dict[int, dict[str, Any]] = {}
        for runtime_name, signature_type in signature_items:
            if not signature_type.startswith("*"):
                continue
            arg_index = step_arg_index_by_name.get(runtime_name)
            if arg_index is None or arg_index >= len(step_args):
                continue
            source_ref = self._resolve_chipyard_value_source(
                step_args[arg_index],
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
            )
            if source_ref is None:
                continue
            pointer_source_refs_by_arg_index.setdefault(arg_index, source_ref)

        for arg_index, serialized_arg in enumerate(step_args):
            signature_type = (
                step_arg_types[arg_index] if arg_index < len(step_arg_types) else None
            )
            if signature_type is not None and signature_type.startswith("*"):
                continue
            scalar_plan = self._build_chipyard_scalar_arg_plan(
                serialized_arg,
                signature_type,
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
            )
            if scalar_plan is None:
                continue
            plan["scalar_plan_index_by_arg_index"][arg_index] = len(
                plan["scalar_plans"]
            )
            plan["scalar_plans"].append(scalar_plan)

        consumed_deferred_site_input_indices: set[int] = set()
        for runtime_name, signature_type in signature_items:
            arg_index = step_arg_index_by_name.get(runtime_name)
            if arg_index is None:
                plan["supported"] = False
                plan["error_reason"] = (
                    f"missing runtime arg binding for {runtime_name!r}"
                )
                return plan
            serialized_arg = step_args[arg_index]
            if signature_type.startswith("*"):
                source_ref = pointer_source_refs_by_arg_index.get(arg_index)
                if source_ref is None:
                    source_ref = self._resolve_chipyard_value_source(
                        serialized_arg,
                        buffer_index_map=buffer_index_map,
                        input_index_map=input_index_map,
                        weight_index_map=weight_index_map,
                    )
                if source_ref is None:
                    plan["supported"] = False
                    plan["error_reason"] = (
                        f"unsupported tensor launch binding for {runtime_name!r}"
                    )
                    return plan

                raw_view_override = self._chipyard_valid_tensor_view_override(
                    view_overrides_by_arg_index.get(arg_index)
                )
                view_override = None
                if raw_view_override is not None:
                    view_override = self._chipyard_tensor_view_is_usable_for_source(
                        raw_view_override,
                        source_ref,
                        signature_type,
                    )
                    if view_override is None:
                        plan["supported"] = False
                        plan["error_reason"] = (
                            f"raw tensor view override rejected for {runtime_name!r}"
                        )
                        return plan

                if view_override is None and deferred_site_inputs:
                    view_override, error_reason = (
                        self._select_chipyard_deferred_site_view_override(
                            runtime_name=runtime_name,
                            signature_type=signature_type,
                            serialized_arg=serialized_arg,
                            source_ref=source_ref,
                            deferred_site_inputs=deferred_site_inputs,
                            consumed_deferred_site_input_indices=consumed_deferred_site_input_indices,
                        )
                    )
                    if error_reason is not None:
                        plan["supported"] = False
                        plan["error_reason"] = (
                            f"unsupported tensor launch binding for {runtime_name!r}: {error_reason}"
                        )
                        return plan

                tensor_plan = self._build_chipyard_tensor_arg_plan(
                    serialized_arg,
                    signature_type,
                    buffer_index_map=buffer_index_map,
                    input_index_map=input_index_map,
                    weight_index_map=weight_index_map,
                    view_override=view_override,
                    source_ref=source_ref,
                )
                if tensor_plan is None:
                    plan["supported"] = False
                    plan["error_reason"] = (
                        f"unsupported tensor launch binding for {runtime_name!r}"
                    )
                    return plan
                plan["runtime_args"].append(
                    {
                        "kind": "tensor",
                        "plan_index": len(plan["tensor_plans"]),
                    }
                )
                plan["tensor_plans"].append(tensor_plan)
                continue
            scalar_plan_index = plan["scalar_plan_index_by_arg_index"].get(arg_index)
            if scalar_plan_index is None:
                scalar_plan = self._build_chipyard_scalar_arg_plan(
                    serialized_arg,
                    signature_type,
                    buffer_index_map=buffer_index_map,
                    input_index_map=input_index_map,
                    weight_index_map=weight_index_map,
                )
                if scalar_plan is None:
                    plan["supported"] = False
                    plan["error_reason"] = (
                        f"unsupported scalar launch binding for {runtime_name!r}"
                    )
                    return plan
                scalar_plan_index = len(plan["scalar_plans"])
                plan["scalar_plan_index_by_arg_index"][arg_index] = scalar_plan_index
                plan["scalar_plans"].append(scalar_plan)
            plan["runtime_args"].append(
                {
                    "kind": "scalar",
                    "plan_index": int(scalar_plan_index),
                }
            )
        return plan

    def _emit_chipyard_launch_runtime_plans(
        self,
        *,
        runner_lines: list[str],
        declared_kernels: dict[str, list[tuple[str, str]]],
        launch_instances: list[dict[str, Any]],
        launch_instance_index_by_step: dict[int, int],
        custom_call_index_by_step: dict[int, int],
        ordered_deferred_autotune_sites: list[dict[str, Any]],
        kernel_map: dict[str, dict[str, Any]],
        buffer_index_map: dict[str, int],
        emit_runner_debug_comments: bool,
    ) -> int:
        signature_index_by_types: dict[tuple[str, ...], int] = {}
        signature_types_by_index: list[tuple[str, ...]] = []
        kernel_symbol_plans: list[dict[str, Any]] = []
        kernel_index_by_name: dict[str, int] = {}
        kernel_indices_by_signature: dict[int, list[int]] = {}
        for kernel_name, signature_items in sorted(declared_kernels.items()):
            signature_types = tuple(signature_type for _, signature_type in signature_items)
            signature_index = signature_index_by_types.get(signature_types)
            if signature_index is None:
                signature_index = len(signature_types_by_index)
                signature_index_by_types[signature_types] = signature_index
                signature_types_by_index.append(signature_types)
            kernel_index = len(kernel_symbol_plans)
            kernel_index_by_name[kernel_name] = kernel_index
            kernel_symbol_plans.append(
                {
                    "kernel_name": kernel_name,
                    "signature_index": signature_index,
                    "signature_types": signature_types,
                }
            )
            kernel_indices_by_signature.setdefault(signature_index, []).append(
                kernel_index
            )

        for launch_instance in launch_instances:
            signature_types = tuple(
                signature_type
                for _runtime_name, signature_type in launch_instance.get(
                    "signature_items", []
                )
            )
            launch_instance["signature_index"] = signature_index_by_types.get(
                signature_types, -1
            )
            launch_instance["default_kernel_index"] = kernel_index_by_name.get(
                str(launch_instance.get("default_kernel_name", "")),
                -1,
            )
            if (
                bool(launch_instance.get("supported", True))
                and int(launch_instance["signature_index"]) < 0
            ):
                launch_instance["supported"] = False
                launch_instance["error_reason"] = (
                    f"missing signature dispatcher for {launch_instance.get('kernel_name', 'unknown_kernel')}"
                )
            if (
                bool(launch_instance.get("supported", True))
                and int(launch_instance["default_kernel_index"]) < 0
            ):
                launch_instance["supported"] = False
                launch_instance["error_reason"] = (
                    f"missing kernel symbol for {launch_instance.get('default_kernel_name', 'unknown_kernel')}"
                )

        for site in ordered_deferred_autotune_sites:
            site_candidate_entries: list[dict[str, Any]] = []
            dispatch_reasons: list[str] = []
            if bool(site.get("dispatch_reason", "")):
                dispatch_reasons.extend(
                    reason
                    for reason in str(site.get("dispatch_reason", "")).split("; ")
                    if reason
                )
            for choice_index, choice in enumerate(site.get("choices", [])):
                if not isinstance(choice, dict):
                    continue
                kernel_name = str(choice.get("kernel_name", ""))
                if not kernel_name:
                    continue
                raw_candidate_index = choice.get("candidate_index")
                candidate_index = (
                    int(raw_candidate_index)
                    if isinstance(raw_candidate_index, int)
                    and not isinstance(raw_candidate_index, bool)
                    else choice_index
                )
                kernel_index = kernel_index_by_name.get(kernel_name, -1)
                if kernel_index < 0:
                    dispatch_reasons.append(
                        f"candidate kernel {kernel_name} is not available in the runner symbol table"
                    )
                raw_kernel_name = str(
                    choice.get("raw_kernel_name")
                    or choice.get("choice_name")
                    or kernel_name
                )
                normalized_kernel_name = self._chipyard_normalized_kernel_name(
                    kernel_name
                )
                config_entries = self._chipyard_choice_config_log_entries(choice)
                site_candidate_entries.append(
                    {
                        "candidate_index": candidate_index,
                        "kernel_name": kernel_name,
                        "raw_kernel_name": raw_kernel_name,
                        "normalized_kernel_name": normalized_kernel_name,
                        "kernel_index": kernel_index,
                        "config_entries": config_entries,
                    }
                )
            site["candidate_entries"] = site_candidate_entries
            site["dispatch_supported"] = bool(
                site.get("dispatch_supported", False)
            ) and not dispatch_reasons
            site["dispatch_reason"] = (
                "; ".join(dict.fromkeys(dispatch_reasons)) if dispatch_reasons else ""
            )

        for instance_index, launch_instance in enumerate(launch_instances):
            tensor_plans = list(launch_instance.get("tensor_plans", []))
            scalar_plans = list(launch_instance.get("scalar_plans", []))
            runtime_args = list(launch_instance.get("runtime_args", []))
            if tensor_plans:
                runner_lines.extend(
                    [
                        "static const ChipyardTensorArgPlan "
                        f"kChipyardLaunchInstance{instance_index}TensorArgs[] = {{",
                        *[
                            "  {"
                            f"{self._chipyard_source_kind_literal(str(plan.get('source_kind', 'none')))}, "
                            f"static_cast<int32_t>({int(plan.get('source_index', -1))}), "
                            f"static_cast<int32_t>({int(plan.get('offset_elems', 0))}), "
                            f"static_cast<int32_t>({int(plan.get('rank', 0))}), "
                            f"ChipyardDType::{self._chipyard_dtype_enum_name(plan.get('dtype'))}, "
                            "{"
                            + ", ".join(
                                f"static_cast<int64_t>({int(value)})"
                                for value in plan.get("sizes", [1, 1, 1, 1])
                            )
                            + "}, {"
                            + ", ".join(
                                f"static_cast<int64_t>({int(value)})"
                                for value in plan.get("strides", [1, 1, 1, 1])
                            )
                            + "}"
                            "},"
                            for plan in tensor_plans
                        ],
                        "};",
                        "",
                    ]
                )
            if scalar_plans:
                runner_lines.extend(
                    [
                        "static const ChipyardScalarArgPlan "
                        f"kChipyardLaunchInstance{instance_index}ScalarArgs[] = {{",
                        *[
                            "  {"
                            f"{self._chipyard_source_kind_literal(str(plan.get('source_kind', 'none')))}, "
                            f"static_cast<int32_t>({int(plan.get('source_index', -1))}), "
                            f"static_cast<int32_t>({int(plan.get('offset_elems', 0))}), "
                            f"ChipyardDType::{self._chipyard_dtype_enum_name(plan.get('dtype'))}, "
                            f"{str(plan.get('immediate_bits', 'UINT64_C(0)'))}, "
                            f"static_cast<int32_t>({1 if bool(plan.get('is_immediate', False)) else 0})"
                            "},"
                            for plan in scalar_plans
                        ],
                        "};",
                        "",
                    ]
                )
            if runtime_args:
                runner_lines.extend(
                    [
                        "static const ChipyardRuntimeArgPlan "
                        f"kChipyardLaunchInstance{instance_index}RuntimeArgs[] = {{",
                        *[
                            "  {"
                            f"ChipyardRuntimeArgKind::{str(arg_plan.get('kind', 'scalar')).capitalize()}, "
                            f"static_cast<int32_t>({int(arg_plan.get('plan_index', -1))})"
                            "},"
                            for arg_plan in runtime_args
                        ],
                        "};",
                        "",
                    ]
                )

            grid_helper_name = f"grid_step_{int(launch_instance.get('step_index', instance_index))}"
            launch_instance["grid_helper_name"] = grid_helper_name
            runner_lines.extend(
                [
                    "static int "
                    f"{grid_helper_name}("
                    "RunnerState& state, "
                    "int32_t candidate_index, "
                    "uint32_t* grid_0, "
                    "uint32_t* grid_1, "
                    "uint32_t* grid_2) {",
                    "  if (grid_0 == nullptr || grid_1 == nullptr || grid_2 == nullptr) {",
                    "    return -1;",
                    "  }",
                ]
            )
            if not bool(launch_instance.get("supported", True)):
                runner_lines.extend(
                    [
                        f'  (void)state; (void)candidate_index; // {self._cpp_comment(launch_instance.get("error_reason", "unsupported launch instance"))}',
                        "  return -1;",
                        "}",
                        "",
                    ]
                )
                continue

            step_arg_index_by_name = dict(launch_instance.get("step_arg_index_by_name", {}))
            grid_sources: list[tuple[Optional[int], Optional[tuple[int, int, int]], Any]] = []
            default_kernel_name = str(launch_instance.get("kernel_name", ""))
            default_kernel_info = kernel_map.get(default_kernel_name, {})
            default_grid_expr = self._kernel_grid_expr(default_kernel_info)
            default_grid_args = self._concrete_grid_args(default_kernel_info)
            grid_sources.append((None, default_grid_args, default_grid_expr))
            site_index = int(launch_instance.get("autotune_site_index", -1))
            site_entry = (
                ordered_deferred_autotune_sites[site_index]
                if 0 <= site_index < len(ordered_deferred_autotune_sites)
                else None
            )
            if isinstance(site_entry, dict) and bool(site_entry.get("dispatch_supported", False)):
                for choice in site_entry.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    grid_sources.append(
                        (
                            int(choice.get("candidate_index", 0)),
                            self._concrete_grid_args(choice),
                            self._kernel_grid_expr(choice),
                        )
                    )

            used_arg_names: list[str] = []
            for arg_name in launch_instance.get("step_arg_names", [])[
                : len(launch_instance.get("step_args", []))
            ]:
                fragments: list[str] = []
                for _candidate_index, concrete_grid_args, grid_expr in grid_sources:
                    if concrete_grid_args is not None or grid_expr is None:
                        continue
                    fragments.extend(str(value) for value in getattr(grid_expr, "prefix", []))
                    fragments.extend(
                        [
                            str(getattr(grid_expr, "x_grid", 1)),
                            str(getattr(grid_expr, "y_grid", 1)),
                            str(getattr(grid_expr, "z_grid", 1)),
                        ]
                    )
                if fragments and any(
                    re.search(
                        rf"(?<![A-Za-z0-9_]){re.escape(str(arg_name))}(?![A-Za-z0-9_])",
                        fragment,
                    )
                    for fragment in fragments
                ):
                    used_arg_names.append(str(arg_name))

            grid_helper_failed = False
            for arg_name in used_arg_names:
                arg_index = step_arg_index_by_name.get(arg_name)
                if arg_index is None:
                    continue
                scalar_plan_index = launch_instance.get(
                    "scalar_plan_index_by_arg_index", {}
                ).get(arg_index)
                if scalar_plan_index is None:
                    runner_lines.extend(
                        [
                            f"  (void)state; (void)candidate_index; // missing scalar grid binding for {self._cpp_comment(arg_name)}",
                            "  return -1;",
                            "}",
                            "",
                        ]
                    )
                    grid_helper_failed = True
                    break
                scalar_plans = launch_instance.get("scalar_plans", [])
                scalar_plan = (
                    scalar_plans[scalar_plan_index]
                    if 0 <= scalar_plan_index < len(scalar_plans)
                    else {}
                )
                signature_type = None
                step_arg_types = launch_instance.get("step_arg_types", [])
                if arg_index < len(step_arg_types):
                    signature_type = step_arg_types[arg_index]
                if signature_type is None:
                    signature_type = scalar_plan.get("dtype")
                cpp_type = self._cpp_scalar_type(signature_type)
                runner_lines.extend(
                    [
                        f"  {cpp_type} arg{arg_index};",
                        "  if (chipyard_load_scalar_arg<"
                        f"{cpp_type}>("
                        f"state, kChipyardLaunchInstance{instance_index}ScalarArgs[{scalar_plan_index}], &arg{arg_index}) != 0) return -1;",
                    ]
                )
                if str(arg_name).isidentifier() and str(arg_name) != f"arg{arg_index}":
                    runner_lines.append(f"  const auto {arg_name} = arg{arg_index};")
            if grid_helper_failed:
                continue

            if emit_runner_debug_comments:
                runner_lines.append(
                    f"  // launch_instance: {int(launch_instance.get('step_index', instance_index))}"
                )

            if isinstance(site_entry, dict) and bool(site_entry.get("dispatch_supported", False)):
                runner_lines.append("  switch (candidate_index) {")
                for choice in site_entry.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    candidate_index = int(choice.get("candidate_index", 0))
                    candidate_fixed_grid_args = self._fixed_grid_extra_args(choice)
                    candidate_grid_args = self._concrete_grid_args(choice)
                    candidate_grid_expr = (
                        None
                        if candidate_fixed_grid_args is not None
                        or candidate_grid_args is not None
                        else self._materialize_grid_placeholders(
                            self._kernel_grid_expr(choice),
                            step_arg_index_by_name,
                            len(launch_instance.get("step_args", [])),
                        )
                    )
                    runner_lines.append(f"  case {candidate_index}: {{")
                    if candidate_fixed_grid_args is not None:
                        runner_lines.extend(
                            [
                                f"    *grid_0 = static_cast<uint32_t>({candidate_fixed_grid_args[0]});",
                                f"    *grid_1 = static_cast<uint32_t>({candidate_fixed_grid_args[1]});",
                                f"    *grid_2 = static_cast<uint32_t>({candidate_fixed_grid_args[2]});",
                            ]
                        )
                    elif candidate_grid_args is not None:
                        runner_lines.extend(
                            [
                                f"    *grid_0 = static_cast<uint32_t>({candidate_grid_args[0]});",
                                f"    *grid_1 = static_cast<uint32_t>({candidate_grid_args[1]});",
                                f"    *grid_2 = static_cast<uint32_t>({candidate_grid_args[2]});",
                            ]
                        )
                    elif candidate_grid_expr is None:
                        runner_lines.extend(
                            [
                                "    *grid_0 = 1;",
                                "    *grid_1 = 1;",
                                "    *grid_2 = 1;",
                            ]
                        )
                    else:
                        runner_lines.extend(
                            [
                                f"    {line}"
                                for line in getattr(candidate_grid_expr, "prefix", [])
                            ]
                        )
                        runner_lines.extend(
                            [
                                f"    *grid_0 = static_cast<uint32_t>({candidate_grid_expr.x_grid});",
                                f"    *grid_1 = static_cast<uint32_t>({candidate_grid_expr.y_grid});",
                                f"    *grid_2 = static_cast<uint32_t>({candidate_grid_expr.z_grid});",
                            ]
                        )
                    runner_lines.extend(["    return 0;", "  }"])
                runner_lines.extend(["  default:", "    return -1;", "  }", "}", ""])
                continue

            runner_lines.append("  (void)candidate_index;")
            default_grid_expr = self._materialize_grid_placeholders(
                default_grid_expr,
                step_arg_index_by_name,
                len(launch_instance.get("step_args", [])),
            )
            if default_grid_args is not None:
                runner_lines.extend(
                    [
                        f"  *grid_0 = static_cast<uint32_t>({default_grid_args[0]});",
                        f"  *grid_1 = static_cast<uint32_t>({default_grid_args[1]});",
                        f"  *grid_2 = static_cast<uint32_t>({default_grid_args[2]});",
                    ]
                )
            elif default_grid_expr is None:
                runner_lines.extend(["  *grid_0 = 1;", "  *grid_1 = 1;", "  *grid_2 = 1;"])
            else:
                runner_lines.extend(
                    [f"  {line}" for line in getattr(default_grid_expr, "prefix", [])]
                )
                runner_lines.extend(
                    [
                        f"  *grid_0 = static_cast<uint32_t>({default_grid_expr.x_grid});",
                        f"  *grid_1 = static_cast<uint32_t>({default_grid_expr.y_grid});",
                        f"  *grid_2 = static_cast<uint32_t>({default_grid_expr.z_grid});",
                    ]
                )
            runner_lines.extend(["  return 0;", "}", ""])

        if kernel_symbol_plans:
            runner_lines.extend(
                [
                    f"constexpr std::size_t kChipyardKernelSymbolCount = {len(kernel_symbol_plans)};",
                    "static const ChipyardKernelSymbolPlan kChipyardKernelSymbols[] = {",
                    *[
                        "  {"
                        f"\"{self._cpp_comment(kernel_symbol['kernel_name'])}\", "
                        f"static_cast<int32_t>({int(kernel_symbol['signature_index'])})"
                        "},"
                        for kernel_symbol in kernel_symbol_plans
                    ],
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "constexpr std::size_t kChipyardKernelSymbolCount = 0;",
                    "static const ChipyardKernelSymbolPlan* kChipyardKernelSymbols = nullptr;",
                    "",
                ]
            )

        for signature_index, signature_types in enumerate(signature_types_by_index):
            runner_lines.extend(
                [
                    "static int "
                    f"dispatch_signature_{signature_index}("
                    "RunnerState& state, "
                    "const ChipyardLaunchInstancePlan& instance, "
                    "int32_t kernel_index, "
                    "int32_t gridX, "
                    "int32_t gridY, "
                    "int32_t gridZ, "
                    "std::uint64_t* total_cycles) {",
                    f"  if (instance.runtime_arg_count != static_cast<std::size_t>({len(signature_types)})) return -1;",
                ]
            )
            call_locals: list[str] = []
            for arg_index, signature_type in enumerate(signature_types):
                if signature_type.startswith("*"):
                    elem_cpp_type = self._cpp_scalar_type(signature_type[1:])
                    runner_lines.extend(
                        [
                            f"  if (instance.runtime_args[{arg_index}].kind != ChipyardRuntimeArgKind::Tensor) return -1;",
                            f"  StridedMemRefType<{elem_cpp_type}, {CHIPYARD_MAX_TENSOR_RANK}> desc{arg_index}{{}};",
                            f"  UnrankedMemRefType<{elem_cpp_type}> arg{arg_index}{{}};",
                            f"  if (chipyard_make_tensor_arg<{elem_cpp_type}>(state, instance.tensor_args[instance.runtime_args[{arg_index}].plan_index], &desc{arg_index}, &arg{arg_index}) != 0) return -1;",
                        ]
                    )
                    call_locals.extend(
                        [f"arg{arg_index}.rank", f"arg{arg_index}.descriptor"]
                    )
                else:
                    cpp_type = self._cpp_scalar_type(signature_type)
                    runner_lines.extend(
                        [
                            f"  if (instance.runtime_args[{arg_index}].kind != ChipyardRuntimeArgKind::Scalar) return -1;",
                            f"  {cpp_type} arg{arg_index};",
                            f"  if (chipyard_load_scalar_arg<{cpp_type}>(state, instance.scalar_args[instance.runtime_args[{arg_index}].plan_index], &arg{arg_index}) != 0) return -1;",
                        ]
                    )
                    call_locals.append(f"arg{arg_index}")
            call_locals.extend(["gridX", "gridY", "gridZ", "pidX", "pidY", "pidZ"])
            runner_lines.extend(
                [
                    "  const std::uint64_t cycle_start = rdcycle();",
                    "  switch (kernel_index) {",
                ]
            )
            for kernel_index in kernel_indices_by_signature.get(signature_index, []):
                kernel_name = str(kernel_symbol_plans[kernel_index]["kernel_name"])
                runner_lines.extend(
                    [
                        f"    case {kernel_index}:",
                        "      #pragma omp parallel for collapse(3) num_threads(CHIPYARD_OMP_NUM_THREADS) schedule(static)",
                        "      for (int32_t pidX = 0; pidX < gridX; ++pidX) {",
                        "        for (int32_t pidY = 0; pidY < gridY; ++pidY) {",
                        "          for (int32_t pidZ = 0; pidZ < gridZ; ++pidZ) {",
                        f"            {kernel_name}({', '.join(call_locals)});"
                        if call_locals
                        else f"            {kernel_name}();",
                        "          }",
                        "        }",
                        "      }",
                        "      break;",
                    ]
                )
            runner_lines.extend(
                [
                    "    default:",
                    "      return -1;",
                    "  }",
                    "  const std::uint64_t cycle_end = rdcycle();",
                    "  if (total_cycles != nullptr) {",
                    "    *total_cycles += (cycle_end - cycle_start);",
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

        autotune_site_entries: list[str] = []
        if ordered_deferred_autotune_sites:
            for site_index, site in enumerate(ordered_deferred_autotune_sites):
                site_id = str(site.get("site_id", f"site_{site_index}"))
                site_name = str(site.get("name", ""))
                default_choice_index = int(site.get("default_choice_index", 0))
                default_kernel_name = str(
                    site.get("default_kernel_name")
                    or site.get("step_kernel_name")
                    or "unknown_kernel"
                )
                dispatch_supported = bool(site.get("dispatch_supported", False))
                dispatch_reason = str(site.get("dispatch_reason", ""))
                site_choices = list(site.get("candidate_entries", []))
                candidate_array_expr = "nullptr"
                if site_choices:
                    candidate_array_name = (
                        f"kChipyardAutotuneSite{site_index}Candidates"
                    )
                    config_array_exprs: dict[int, tuple[int, str]] = {}
                    for choice_offset, choice in enumerate(site_choices):
                        config_entries = choice.get("config_entries", [])
                        if not isinstance(config_entries, list) or not config_entries:
                            config_array_exprs[choice_offset] = (0, "nullptr")
                            continue
                        config_array_name = (
                            f"kChipyardAutotuneSite{site_index}Candidate"
                            f"{choice_offset}Config"
                        )
                        runner_lines.extend(
                            [
                                "static const ChipyardAutotuneConfigEntry "
                                f"{config_array_name}[] = {{",
                                *[
                                    "  {"
                                    f"{self._cpp_string_literal(key)}, "
                                    f"{self._cpp_string_literal(value)}"
                                    "},"
                                    for key, value in config_entries
                                ],
                                "};",
                                "",
                            ]
                        )
                        config_array_exprs[choice_offset] = (
                            len(config_entries),
                            config_array_name,
                        )
                    runner_lines.extend(
                        [
                            "static const ChipyardAutotuneCandidate "
                            f"{candidate_array_name}[] = {{",
                            *[
                                "  {"
                                f"static_cast<int32_t>({int(choice.get('candidate_index', -1))}), "
                                f"{self._cpp_string_literal(choice.get('kernel_name', 'unknown_kernel'))}, "
                                f"{self._cpp_string_literal(choice.get('raw_kernel_name', ''))}, "
                                f"{self._cpp_string_literal(choice.get('normalized_kernel_name', ''))}, "
                                f"static_cast<int32_t>({int(choice.get('kernel_index', -1))}), "
                                f"static_cast<std::size_t>({config_array_exprs.get(choice_offset, (0, 'nullptr'))[0]}), "
                                f"{config_array_exprs.get(choice_offset, (0, 'nullptr'))[1]}"
                                "},"
                                for choice_offset, choice in enumerate(site_choices)
                            ],
                            "};",
                            "",
                        ]
                    )
                    candidate_array_expr = candidate_array_name
                autotune_site_entries.append(
                    "  {"
                    f"static_cast<std::size_t>({site_index}), "
                    f"static_cast<int32_t>({default_choice_index}), "
                    f"static_cast<int32_t>({1 if dispatch_supported else 0}), "
                    f"{self._cpp_string_literal(site_id)}, "
                    f"{self._cpp_string_literal(site_name)}, "
                    f"{self._cpp_string_literal(dispatch_reason)}, "
                    f"{self._cpp_string_literal(default_kernel_name)}, "
                    f"static_cast<std::size_t>({len(site_choices)}), "
                    f"{candidate_array_expr}, "
                    f"static_cast<int32_t>({int(site.get('representative_launch_instance_index', -1))})"
                    "},"
                )
            runner_lines.extend(
                [
                    "static const ChipyardAutotuneSitePlan kChipyardAutotuneSites[] = {",
                    *autotune_site_entries,
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const ChipyardAutotuneSitePlan* kChipyardAutotuneSites = nullptr;",
                    "",
                ]
            )

        if launch_instances:
            launch_instance_entries: list[str] = []
            for instance_index, launch_instance in enumerate(launch_instances):
                tensor_expr = (
                    f"kChipyardLaunchInstance{instance_index}TensorArgs"
                    if launch_instance.get("tensor_plans")
                    else "nullptr"
                )
                scalar_expr = (
                    f"kChipyardLaunchInstance{instance_index}ScalarArgs"
                    if launch_instance.get("scalar_plans")
                    else "nullptr"
                )
                runtime_arg_expr = (
                    f"kChipyardLaunchInstance{instance_index}RuntimeArgs"
                    if launch_instance.get("runtime_args")
                    else "nullptr"
                )
                launch_instance_entries.append(
                    "  {"
                    f"static_cast<int32_t>({int(launch_instance.get('autotune_site_index', -1))}), "
                    f"static_cast<int32_t>({int(launch_instance.get('default_kernel_index', -1))}), "
                    f"static_cast<int32_t>({int(launch_instance.get('signature_index', -1))}), "
                    f"static_cast<int32_t>({1 if bool(launch_instance.get('supported', True)) else 0}), "
                    f"\"{self._cpp_comment(launch_instance.get('error_reason', ''))}\", "
                    f"static_cast<std::size_t>({len(launch_instance.get('runtime_args', []))}), "
                    f"{runtime_arg_expr}, "
                    f"static_cast<std::size_t>({len(launch_instance.get('tensor_plans', []))}), "
                    f"{tensor_expr}, "
                    f"static_cast<std::size_t>({len(launch_instance.get('scalar_plans', []))}), "
                    f"{scalar_expr}, "
                    f"&{launch_instance.get('grid_helper_name', f'grid_step_{instance_index}')}"
                    "},"
                )
            runner_lines.extend(
                [
                    "static const ChipyardLaunchInstancePlan kChipyardLaunchInstances[] = {",
                    *launch_instance_entries,
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const ChipyardLaunchInstancePlan* kChipyardLaunchInstances = nullptr;",
                    "",
                ]
            )

        step_plan_entries: list[str] = []
        for step in self.chipyard_plan:
            step_index = int(step["index"])
            step_kind = step.get("kind")
            if step_kind == "alloc":
                buffer_name = step.get("name")
                buffer_index = buffer_index_map.get(buffer_name)
                if buffer_index is None:
                    continue
                alias_source_name = self._resolve_chipyard_alias_name(buffer_name)
                if str(alias_source_name) != str(buffer_name):
                    continue
                step_plan_entries.append(
                    "  {ChipyardStepKind::Alloc, "
                    f"static_cast<int32_t>({buffer_index}), "
                    "static_cast<int32_t>(-1), "
                    "static_cast<int32_t>(-1)},"
                )
            elif step_kind == "free":
                buffer_name = step.get("name")
                buffer_index = buffer_index_map.get(buffer_name)
                if buffer_index is None:
                    continue
                buffer_entry = self.chipyard_buffers.get(str(buffer_name), {})
                buffer_role = str(buffer_entry.get("role", "intermediate"))
                if buffer_role in {"graph_input", "constant"}:
                    continue
                alias_source_name = self._resolve_chipyard_alias_name(buffer_name)
                if str(alias_source_name) != str(buffer_name):
                    continue
                step_plan_entries.append(
                    "  {ChipyardStepKind::Free, "
                    f"static_cast<int32_t>({buffer_index}), "
                    "static_cast<int32_t>(-1), "
                    "static_cast<int32_t>(-1)},"
                )
            elif step_kind == "launch":
                launch_instance_index = launch_instance_index_by_step.get(step_index, -1)
                step_plan_entries.append(
                    "  {ChipyardStepKind::Launch, "
                    "static_cast<int32_t>(-1), "
                    f"static_cast<int32_t>({launch_instance_index}), "
                    "static_cast<int32_t>(-1)"
                    "},"
                )
            elif step_kind == "custom_call":
                custom_call_index = custom_call_index_by_step.get(step_index, -1)
                step_plan_entries.append(
                    "  {ChipyardStepKind::CustomCall, "
                    "static_cast<int32_t>(-1), "
                    "static_cast<int32_t>(-1), "
                    f"static_cast<int32_t>({custom_call_index})"
                    "},"
                )

        runner_lines.extend(
            [
                f"constexpr std::size_t kChipyardExecutableStepCount = {len(step_plan_entries)};",
                "",
            ]
        )
        if step_plan_entries:
            runner_lines.extend(
                [
                    "static const ChipyardStepPlan kChipyardExecutableSteps[] = {",
                    *step_plan_entries,
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const ChipyardStepPlan* kChipyardExecutableSteps = nullptr;",
                    "",
                ]
            )
        return len(signature_types_by_index)

    def _emit_chipyard_runner_execution_helpers(
        self,
        *,
        runner_lines: list[str],
        signature_dispatch_count: int,
        custom_call_dispatch_count: int,
        execute_model_epilogue_lines: list[str],
    ) -> None:
        del custom_call_dispatch_count
        dispatch_cases = [
            f"    case {signature_index}: return dispatch_signature_{signature_index}("
            "state, instance, kernel_index, "
            "static_cast<int32_t>(grid_0), "
            "static_cast<int32_t>(grid_1), "
            "static_cast<int32_t>(grid_2), total_cycles);"
            for signature_index in range(signature_dispatch_count)
        ]
        helper_template = textwrap.dedent(
            r"""
            static const char* chipyard_safe_cstr(const char* value) {
              return value != nullptr ? value : "";
            }

            static bool chipyard_starts_with(const char* value, const char* prefix) {
              value = chipyard_safe_cstr(value);
              prefix = chipyard_safe_cstr(prefix);
              const std::size_t prefix_len = std::strlen(prefix);
              return std::strncmp(value, prefix, prefix_len) == 0;
            }

            static std::string chipyard_normalized_kernel_name(const char* kernel_name) {
              const char* name = chipyard_safe_cstr(kernel_name);
              std::size_t normalized_length = std::strlen(name);
              while (normalized_length > 0) {
                const char tail = name[normalized_length - 1];
                if (tail >= '0' && tail <= '9') {
                  --normalized_length;
                  continue;
                }
                if (tail == '_') {
                  --normalized_length;
                }
                break;
              }
              return std::string(name, normalized_length);
            }

            static const ChipyardAutotuneCandidate* chipyard_find_autotune_candidate(
                const ChipyardAutotuneSitePlan& site,
                int32_t candidate_index) {
              for (std::size_t candidate_offset = 0;
                   candidate_offset < site.candidate_count;
                   ++candidate_offset) {
                const auto& candidate = site.candidates[candidate_offset];
                if (candidate.candidate_index == candidate_index) {
                  return &candidate;
                }
              }
              return nullptr;
            }

            static const char* chipyard_autotune_kernel_name(
                const ChipyardAutotuneSitePlan& site,
                int32_t candidate_index) {
              if (site.dispatch_supported == 0) {
                return site.default_kernel_name;
              }
              const auto* candidate =
                  chipyard_find_autotune_candidate(site, candidate_index);
              if (candidate != nullptr) {
                return candidate->kernel_name;
              }
              return site.default_kernel_name;
            }

            static const char* chipyard_autotune_raw_kernel_name(
                const ChipyardAutotuneSitePlan& site,
                int32_t candidate_index) {
              const auto* candidate =
                  chipyard_find_autotune_candidate(site, candidate_index);
              if (candidate != nullptr && candidate->raw_kernel_name != nullptr &&
                  candidate->raw_kernel_name[0] != '\0') {
                return candidate->raw_kernel_name;
              }
              return site.default_kernel_name;
            }

            static bool chipyard_is_flex_attention_site(
                const ChipyardAutotuneSitePlan& site,
                const char* raw_kernel_name) {
              return std::strcmp(chipyard_safe_cstr(site.site_name), "flex_attention") == 0 ||
                  chipyard_starts_with(site.site_id, "flex_attention_") ||
                  chipyard_starts_with(raw_kernel_name, "triton_flex_attention_chipyard_") ||
                  chipyard_starts_with(
                      site.default_kernel_name, "triton_flex_attention_chipyard_");
            }

            static std::string chipyard_semantic_group_name(
                const ChipyardAutotuneSitePlan* site,
                const ChipyardAutotuneCandidate* candidate,
                const char* exact_kernel_name,
                std::string* semantic_source,
                std::string* normalized_kernel_name,
                std::string* raw_kernel_name,
                std::string* site_id) {
              const std::string exact_normalized =
                  chipyard_normalized_kernel_name(exact_kernel_name);
              if (normalized_kernel_name != nullptr) {
                *normalized_kernel_name = exact_normalized;
              }
              if (raw_kernel_name != nullptr) {
                raw_kernel_name->clear();
              }
              if (site_id != nullptr) {
                site_id->clear();
              }
              if (semantic_source != nullptr) {
                *semantic_source = "normalized kernel_name";
              }
              if (site == nullptr) {
                return exact_normalized;
              }

              const char* candidate_raw =
                  candidate != nullptr ? candidate->raw_kernel_name : nullptr;
              const char* candidate_normalized =
                  candidate != nullptr ? candidate->normalized_kernel_name : nullptr;
              if (raw_kernel_name != nullptr) {
                *raw_kernel_name = chipyard_safe_cstr(candidate_raw);
                if (raw_kernel_name->empty()) {
                  *raw_kernel_name = chipyard_safe_cstr(site->default_kernel_name);
                }
              }
              if (site_id != nullptr) {
                *site_id = chipyard_safe_cstr(site->site_id);
              }
              if (candidate_normalized != nullptr && candidate_normalized[0] != '\0' &&
                  normalized_kernel_name != nullptr) {
                *normalized_kernel_name = candidate_normalized;
              }

              if (chipyard_is_flex_attention_site(*site, candidate_raw)) {
                if (semantic_source != nullptr) {
                  *semantic_source = "autotune site/raw";
                }
                return "flex_attention";
              }

              const char* normalized =
                  candidate_normalized != nullptr && candidate_normalized[0] != '\0'
                      ? candidate_normalized
                      : exact_normalized.c_str();
              if (std::strcmp(chipyard_safe_cstr(normalized), "triton_tem_fused") == 0 &&
                  site->site_name != nullptr && site->site_name[0] != '\0') {
                if (semantic_source != nullptr) {
                  *semantic_source = "autotune site fallback";
                }
                return site->site_name;
              }

              if (semantic_source != nullptr) {
                *semantic_source = "normalized kernel_name";
              }
              return normalized;
            }

            static void chipyard_close_log_files(RunnerState& state) {
              if (state.model_log != nullptr) {
                std::fclose(state.model_log);
                state.model_log = nullptr;
              }
              if (state.autotune_log != nullptr) {
                std::fclose(state.autotune_log);
                state.autotune_log = nullptr;
              }
            }

            static int32_t chipyard_candidate_kernel_index(
                const ChipyardAutotuneSitePlan& site,
                int32_t candidate_index,
                int32_t fallback_kernel_index) {
              if (site.dispatch_supported == 0) {
                return fallback_kernel_index;
              }
              for (std::size_t candidate_offset = 0;
                   candidate_offset < site.candidate_count;
                   ++candidate_offset) {
                const auto& candidate = site.candidates[candidate_offset];
                if (candidate.candidate_index == candidate_index) {
                  return candidate.kernel_index >= 0 ? candidate.kernel_index : fallback_kernel_index;
                }
              }
              return fallback_kernel_index;
            }

            static int chipyard_resolve_grid(
                RunnerState& state,
                const ChipyardLaunchInstancePlan& instance,
                int32_t candidate_index,
                uint32_t* grid_0,
                uint32_t* grid_1,
                uint32_t* grid_2) {
              if (instance.grid_fn == nullptr || grid_0 == nullptr || grid_1 == nullptr ||
                  grid_2 == nullptr) {
                return -1;
              }
              *grid_0 = 1;
              *grid_1 = 1;
              *grid_2 = 1;
              return instance.grid_fn(state, candidate_index, grid_0, grid_1, grid_2);
            }

            template <typename T>
            static std::uint64_t chipyard_scalar_key_bits_from_value(T value) {
              std::uint64_t bits = 0;
              std::memcpy(&bits, &value, sizeof(T));
              return bits;
            }

            static int chipyard_scalar_arg_key_bits(
                RunnerState& state,
                const ChipyardScalarArgPlan& plan,
                std::uint64_t* bits_out) {
              if (bits_out == nullptr) {
                return -1;
              }
              switch (plan.dtype) {
                case ChipyardDType::I1: {
                  bool value = false;
                  if (chipyard_load_scalar_arg<bool>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::I8: {
                  std::int8_t value = 0;
                  if (chipyard_load_scalar_arg<std::int8_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::I16: {
                  std::int16_t value = 0;
                  if (chipyard_load_scalar_arg<std::int16_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::I32: {
                  std::int32_t value = 0;
                  if (chipyard_load_scalar_arg<std::int32_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::I64: {
                  std::int64_t value = 0;
                  if (chipyard_load_scalar_arg<std::int64_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::U1: {
                  bool value = false;
                  if (chipyard_load_scalar_arg<bool>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::U8: {
                  std::uint8_t value = 0;
                  if (chipyard_load_scalar_arg<std::uint8_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::U16: {
                  std::uint16_t value = 0;
                  if (chipyard_load_scalar_arg<std::uint16_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::U32: {
                  std::uint32_t value = 0;
                  if (chipyard_load_scalar_arg<std::uint32_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::U64: {
                  std::uint64_t value = 0;
                  if (chipyard_load_scalar_arg<std::uint64_t>(state, plan, &value) != 0) {
                    return -1;
                  }
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::FP16: {
                  f16 value{};
                  if (chipyard_load_scalar_arg<f16>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::BF16: {
                  bf16 value{};
                  if (chipyard_load_scalar_arg<bf16>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::FP32: {
                  float value = 0.0f;
                  if (chipyard_load_scalar_arg<float>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
                case ChipyardDType::FP64: {
                  double value = 0.0;
                  if (chipyard_load_scalar_arg<double>(state, plan, &value) != 0) return -1;
                  *bits_out = chipyard_scalar_key_bits_from_value(value);
                  return 0;
                }
              }
              return -1;
            }

            static void chipyard_append_normalized_kernel_name_key(
                std::string& key,
                const char* kernel_name) {
              key.append("K:");
              if (kernel_name == nullptr) {
                key.append("null|");
                return;
              }
              std::size_t normalized_length = std::strlen(kernel_name);
              while (normalized_length > 0) {
                const char tail = kernel_name[normalized_length - 1];
                if (tail >= '0' && tail <= '9') {
                  --normalized_length;
                  continue;
                }
                if (tail == '_') {
                  --normalized_length;
                }
                break;
              }
              key.append(kernel_name, normalized_length);
              key.push_back('|');
            }

            static void chipyard_append_tensor_arg_key(
                std::string& key,
                const ChipyardTensorArgPlan& plan) {
              key.append("T:");
              key.append(std::to_string(static_cast<int>(plan.source_kind)));
              key.push_back(':');
              key.append(std::to_string(static_cast<int>(plan.dtype)));
              key.push_back(':');
              key.append(std::to_string(plan.rank));
              key.push_back(':');
              for (int32_t dim = 0; dim < kChipyardMaxTensorRank; ++dim) {
                key.append(std::to_string(plan.sizes[dim]));
                key.push_back(',');
              }
              key.push_back(':');
              for (int32_t dim = 0; dim < kChipyardMaxTensorRank; ++dim) {
                key.append(std::to_string(plan.strides[dim]));
                key.push_back(',');
              }
              key.push_back('|');
            }

            static void chipyard_append_scalar_arg_key(
                std::string& key,
                ChipyardDType dtype,
                std::uint64_t bits) {
              key.append("S:");
              key.append(std::to_string(static_cast<int>(dtype)));
              key.push_back(':');
              key.append(std::to_string(bits));
              key.push_back('|');
            }

            static int chipyard_build_autotune_cache_key(
                std::string* key_out,
                RunnerState& state,
                const ChipyardAutotuneSitePlan& site,
                const ChipyardLaunchInstancePlan& instance) {
              if (key_out == nullptr || site.dispatch_supported == 0) {
                return -1;
              }

              std::string key;
              key.reserve(512);
              key.append("sig=");
              key.append(std::to_string(instance.signature_index));
              key.push_back('|');
              key.append("candidates=");
              key.append(std::to_string(site.candidate_count));
              key.push_back('|');
              for (std::size_t candidate_offset = 0;
                   candidate_offset < site.candidate_count;
                   ++candidate_offset) {
                const auto& candidate = site.candidates[candidate_offset];
                uint32_t grid_0 = 1;
                uint32_t grid_1 = 1;
                uint32_t grid_2 = 1;
                if (chipyard_resolve_grid(
                        state,
                        instance,
                        candidate.candidate_index,
                        &grid_0,
                        &grid_1,
                        &grid_2) != 0) {
                  return -1;
                }
                key.append("C:");
                key.append(std::to_string(candidate.candidate_index));
                key.push_back(':');
                chipyard_append_normalized_kernel_name_key(key, candidate.kernel_name);
                key.append(std::to_string(grid_0));
                key.push_back(',');
                key.append(std::to_string(grid_1));
                key.push_back(',');
                key.append(std::to_string(grid_2));
                key.push_back('|');
              }
              for (std::size_t arg_offset = 0;
                   arg_offset < instance.runtime_arg_count;
                   ++arg_offset) {
                const auto& runtime_arg = instance.runtime_args[arg_offset];
                if (runtime_arg.kind == ChipyardRuntimeArgKind::Tensor) {
                  if (runtime_arg.plan_index < 0 ||
                      static_cast<std::size_t>(runtime_arg.plan_index) >=
                          instance.tensor_arg_count) {
                    return -1;
                  }
                  chipyard_append_tensor_arg_key(
                      key, instance.tensor_args[runtime_arg.plan_index]);
                  continue;
                }
                if (runtime_arg.kind != ChipyardRuntimeArgKind::Scalar ||
                    runtime_arg.plan_index < 0 ||
                    static_cast<std::size_t>(runtime_arg.plan_index) >=
                        instance.scalar_arg_count) {
                  return -1;
                }
                const auto& scalar_plan = instance.scalar_args[runtime_arg.plan_index];
                std::uint64_t scalar_bits = 0;
                if (chipyard_scalar_arg_key_bits(state, scalar_plan, &scalar_bits) != 0) {
                  return -1;
                }
                chipyard_append_scalar_arg_key(key, scalar_plan.dtype, scalar_bits);
              }
              *key_out = key;
              return 0;
            }

            static int32_t chipyard_candidate_index_from_cache_entry(
                RunnerState& state,
                const ChipyardAutotuneSitePlan& site,
                const ChipyardLaunchInstancePlan& instance,
                const ChipyardAutotuneCacheEntry& cache_entry) {
              if (site.dispatch_supported == 0) {
                return -1;
              }
              for (std::size_t candidate_offset = 0;
                   candidate_offset < site.candidate_count;
                   ++candidate_offset) {
                const auto& candidate = site.candidates[candidate_offset];
                if (candidate.candidate_index != cache_entry.candidate_index) {
                  continue;
                }
                uint32_t grid_0 = 1;
                uint32_t grid_1 = 1;
                uint32_t grid_2 = 1;
                if (chipyard_resolve_grid(
                        state,
                        instance,
                        candidate.candidate_index,
                        &grid_0,
                        &grid_1,
                        &grid_2) != 0) {
                  continue;
                }
                if (grid_0 == cache_entry.grid_0 &&
                    grid_1 == cache_entry.grid_1 &&
                    grid_2 == cache_entry.grid_2) {
                  return candidate.candidate_index;
                }
              }
              return -1;
            }

            struct ChipyardKernelCycleStat {
              std::size_t step_index;
              std::string semantic_group;
              std::string semantic_source;
              std::string normalized_kernel_name;
              std::string exact_kernel_name;
              std::string site_id;
              std::string raw_kernel_name;
              std::uint64_t total_cycles;
              std::uint64_t min_cycles;
              std::uint64_t max_cycles;
              std::uint64_t samples;
            };

            struct ChipyardKernelGroupStat {
              std::string semantic_group;
              std::uint64_t total_cycles;
              std::uint64_t min_cycles;
              std::uint64_t max_cycles;
              std::uint64_t samples;
              std::uint64_t launch_count;
              std::vector<std::string> semantic_sources;
              std::vector<std::string> normalized_kernel_names;
              std::vector<std::string> exact_kernel_names;
              std::vector<std::string> site_ids;
              std::vector<std::string> raw_kernel_names;
            };

            static const char* chipyard_kernel_symbol_name(int32_t kernel_index) {
              if (kernel_index < 0 ||
                  static_cast<std::size_t>(kernel_index) >= kChipyardKernelSymbolCount) {
                return "";
              }
              return kChipyardKernelSymbols[kernel_index].kernel_name;
            }

            static void chipyard_add_unique_value(
                std::vector<std::string>& values,
                const std::string& value) {
              if (value.empty()) {
                return;
              }
              if (std::find(values.begin(), values.end(), value) == values.end()) {
                values.push_back(value);
              }
            }

            static void chipyard_record_kernel_cycles(
                std::vector<ChipyardKernelCycleStat>* stats,
                std::size_t step_index,
                const ChipyardAutotuneSitePlan* site,
                int32_t candidate_index,
                const char* exact_kernel_name,
                std::uint64_t cycles) {
              if (stats == nullptr) {
                return;
              }
              for (auto& stat : *stats) {
                if (stat.step_index == step_index) {
                  stat.total_cycles += cycles;
                  if (cycles < stat.min_cycles) {
                    stat.min_cycles = cycles;
                  }
                  if (cycles > stat.max_cycles) {
                    stat.max_cycles = cycles;
                  }
                  ++stat.samples;
                  return;
                }
              }
              const ChipyardAutotuneCandidate* candidate =
                  site != nullptr
                      ? chipyard_find_autotune_candidate(*site, candidate_index)
                      : nullptr;
              std::string semantic_source;
              std::string normalized_kernel_name;
              std::string raw_kernel_name;
              std::string site_id;
              const std::string semantic_group = chipyard_semantic_group_name(
                  site,
                  candidate,
                  exact_kernel_name,
                  &semantic_source,
                  &normalized_kernel_name,
                  &raw_kernel_name,
                  &site_id);
              stats->push_back(ChipyardKernelCycleStat{
                  step_index,
                  semantic_group,
                  semantic_source,
                  normalized_kernel_name,
                  chipyard_safe_cstr(exact_kernel_name),
                  site_id,
                  raw_kernel_name,
                  cycles,
                  cycles,
                  cycles,
                  static_cast<std::uint64_t>(1)});
            }

            static ChipyardKernelGroupStat* chipyard_find_kernel_group_stat(
                std::vector<ChipyardKernelGroupStat>& groups,
                const std::string& semantic_group) {
              for (auto& group : groups) {
                if (group.semantic_group == semantic_group) {
                  return &group;
                }
              }
              return nullptr;
            }

            static void chipyard_print_name_samples(
                std::FILE* log,
                const char* title,
                const std::vector<std::string>& values) {
              if (log == nullptr || values.empty()) {
                return;
              }
              std::fprintf(log, "%s:\n", chipyard_safe_cstr(title));
              const std::size_t sample_limit = values.size() < 3 ? values.size() : 3;
              for (std::size_t index = 0; index < sample_limit; ++index) {
                std::fprintf(log, "- %s\n", values[index].c_str());
              }
              if (values.size() > sample_limit) {
                std::fprintf(log, "- ... +%zu more\n", values.size() - sample_limit);
              }
            }

            static void chipyard_write_model_log(
                std::FILE* log,
                const std::vector<ChipyardKernelCycleStat>& stats,
                std::uint64_t full_model_cycles_sum,
                std::uint64_t full_model_cycles_min,
                std::uint64_t full_model_cycles_max,
                std::uint64_t full_model_cycles_samples,
                std::uint64_t model_kernel_cycles_sum) {
              if (log == nullptr) {
                return;
              }
              const double full_model_avg_cycles =
                  full_model_cycles_samples > 0
                      ? (static_cast<double>(full_model_cycles_sum) /
                         static_cast<double>(full_model_cycles_samples))
                      : 0.0;
              std::fprintf(log, "Model Launch Log\n\n");
              std::fprintf(log, "Avg Model cycle: %.2f\n", full_model_avg_cycles);
              std::fprintf(
                  log,
                  "Max Model cycle: %llu\n",
                  static_cast<unsigned long long>(full_model_cycles_max));
              std::fprintf(
                  log,
                  "Min Model cycle: %llu\n",
                  static_cast<unsigned long long>(full_model_cycles_min));
              std::fprintf(
                  log,
                  "Model samples: %llu\n\n",
                  static_cast<unsigned long long>(full_model_cycles_samples));

              std::vector<ChipyardKernelGroupStat> groups;
              for (const auto& stat : stats) {
                ChipyardKernelGroupStat* group =
                    chipyard_find_kernel_group_stat(groups, stat.semantic_group);
                if (group == nullptr) {
                  groups.push_back(ChipyardKernelGroupStat{
                      stat.semantic_group,
                      static_cast<std::uint64_t>(0),
                      static_cast<std::uint64_t>(0),
                      static_cast<std::uint64_t>(0),
                      static_cast<std::uint64_t>(0),
                      static_cast<std::uint64_t>(0),
                      {},
                      {},
                      {},
                      {},
                      {}});
                  group = &groups.back();
                }
                group->total_cycles += stat.total_cycles;
                if (group->samples == 0 || stat.min_cycles < group->min_cycles) {
                  group->min_cycles = stat.min_cycles;
                }
                if (group->samples == 0 || stat.max_cycles > group->max_cycles) {
                  group->max_cycles = stat.max_cycles;
                }
                group->samples += stat.samples;
                ++group->launch_count;
                chipyard_add_unique_value(group->semantic_sources, stat.semantic_source);
                chipyard_add_unique_value(
                    group->normalized_kernel_names, stat.normalized_kernel_name);
                chipyard_add_unique_value(group->exact_kernel_names, stat.exact_kernel_name);
                chipyard_add_unique_value(group->site_ids, stat.site_id);
                chipyard_add_unique_value(group->raw_kernel_names, stat.raw_kernel_name);
              }
              std::sort(
                  groups.begin(),
                  groups.end(),
                  [](const ChipyardKernelGroupStat& lhs,
                     const ChipyardKernelGroupStat& rhs) {
                    if (lhs.total_cycles != rhs.total_cycles) {
                      return lhs.total_cycles > rhs.total_cycles;
                    }
                    return lhs.semantic_group < rhs.semantic_group;
                  });

              std::fprintf(log, "Top 10 Hot kernels\n\n");
              const std::size_t top_count = groups.size() < 10 ? groups.size() : 10;
              for (std::size_t group_index = 0; group_index < top_count; ++group_index) {
                const auto& group = groups[group_index];
                const double avg_cycles =
                    group.samples > 0
                        ? (static_cast<double>(group.total_cycles) /
                           static_cast<double>(group.samples))
                        : 0.0;
                const double cycle_percent =
                    model_kernel_cycles_sum > 0
                        ? (100.0 * static_cast<double>(group.total_cycles) /
                           static_cast<double>(model_kernel_cycles_sum))
                        : 0.0;
                std::fprintf(
                    log,
                    "%zu. %s\n",
                    group_index + 1,
                    group.semantic_group.c_str());
                std::fprintf(
                    log,
                    "Launch count: %llu\n",
                    static_cast<unsigned long long>(group.launch_count));
                std::fprintf(
                    log,
                    "Samples: %llu\n",
                    static_cast<unsigned long long>(group.samples));
                std::fprintf(
                    log,
                    "Total launch cycle: %llu (%.2f%%)\n",
                    static_cast<unsigned long long>(group.total_cycles),
                    cycle_percent);
                std::fprintf(log, "Avg launch cycle: %.2f\n", avg_cycles);
                std::fprintf(
                    log,
                    "Min launch cycle: %llu\n",
                    static_cast<unsigned long long>(group.min_cycles));
                std::fprintf(
                    log,
                    "Max launch cycle: %llu\n",
                    static_cast<unsigned long long>(group.max_cycles));
                chipyard_print_name_samples(
                    log, "Semantic source", group.semantic_sources);
                chipyard_print_name_samples(
                    log, "Normalized kernel names", group.normalized_kernel_names);
                chipyard_print_name_samples(log, "Autotune sites", group.site_ids);
                chipyard_print_name_samples(
                    log, "Raw candidate kernel names", group.raw_kernel_names);
                chipyard_print_name_samples(
                    log, "Exact kernel names", group.exact_kernel_names);
                std::fprintf(log, "\n");
              }

              std::fprintf(log, "All Kernel Cycle Stats (execution order)\n\n");
              std::fprintf(log, "Count: %zu\n\n", stats.size());
              for (std::size_t stat_index = 0; stat_index < stats.size(); ++stat_index) {
                const auto& stat = stats[stat_index];
                const double avg_cycles =
                    stat.samples > 0
                        ? (static_cast<double>(stat.total_cycles) /
                           static_cast<double>(stat.samples))
                        : 0.0;
                std::fprintf(
                    log,
                    "%zu. %s\n",
                    stat_index + 1,
                    stat.exact_kernel_name.c_str());
                std::fprintf(
                    log,
                    "Samples: %llu\n",
                    static_cast<unsigned long long>(stat.samples));
                std::fprintf(
                    log,
                    "Total launch cycle: %llu\n",
                    static_cast<unsigned long long>(stat.total_cycles));
                std::fprintf(log, "Avg launch cycle: %.2f\n", avg_cycles);
                std::fprintf(
                    log,
                    "Min launch cycle: %llu\n",
                    static_cast<unsigned long long>(stat.min_cycles));
                std::fprintf(
                    log,
                    "Max launch cycle: %llu\n\n",
                    static_cast<unsigned long long>(stat.max_cycles));
              }
            }

            static void chipyard_log_autotune_candidate_cycles(
                const RunnerState& state,
                const ChipyardAutotuneSitePlan& site,
                int32_t candidate_index,
                uint32_t grid_0,
                uint32_t grid_1,
                uint32_t grid_2,
                std::uint64_t cycles,
                const char* source) {
              std::FILE* log = state.autotune_log;
              if (log == nullptr) {
                return;
              }
              const auto* logged_candidate =
                  chipyard_find_autotune_candidate(site, candidate_index);
              std::fprintf(log, "Autotune Candidate\n");
              std::fprintf(log, "Site: %zu\n", site.site_index);
              std::fprintf(log, "Site name: %s\n", chipyard_safe_cstr(site.site_name));
              std::fprintf(log, "Site id: %s\n", chipyard_safe_cstr(site.site_id));
              std::fprintf(log, "Candidate: %d\n", static_cast<int>(candidate_index));
              std::fprintf(
                  log,
                  "Kernel: %s\n",
                  chipyard_safe_cstr(chipyard_autotune_kernel_name(site, candidate_index)));
              std::fprintf(
                  log,
                  "Raw kernel: %s\n",
                  chipyard_safe_cstr(chipyard_autotune_raw_kernel_name(site, candidate_index)));
              if (logged_candidate != nullptr) {
                std::fprintf(
                    log,
                    "Normalized kernel: %s\n",
                    chipyard_safe_cstr(logged_candidate->normalized_kernel_name));
              }
              std::fprintf(
                  log,
                  "Grid: %u,%u,%u\n",
                  static_cast<unsigned>(grid_0),
                  static_cast<unsigned>(grid_1),
                  static_cast<unsigned>(grid_2));
              std::fprintf(
                  log,
                  "Cycles: %llu\n",
                  static_cast<unsigned long long>(cycles));
              std::fprintf(log, "Source: %s\n", chipyard_safe_cstr(source));
              const std::size_t config_count =
                  logged_candidate != nullptr ? logged_candidate->config_count : 0;
              std::fprintf(log, "Config count: %zu\n", config_count);
              if (logged_candidate != nullptr &&
                  logged_candidate->config_entries != nullptr) {
                for (std::size_t config_index = 0;
                     config_index < logged_candidate->config_count;
                     ++config_index) {
                  const auto& config_entry =
                      logged_candidate->config_entries[config_index];
                  std::fprintf(
                      log,
                      "  %s=%s\n",
                      chipyard_safe_cstr(config_entry.key),
                      chipyard_safe_cstr(config_entry.value));
                }
              }
              std::fprintf(log, "\n");
            }

            static void chipyard_log_autotune_site_result(
                const RunnerState& state,
                const ChipyardAutotuneSitePlan& site,
                int32_t chosen_index,
                std::uint64_t best_cycles,
                const char* reason) {
              std::FILE* log = state.autotune_log;
              if (log == nullptr) {
                return;
              }
              std::fprintf(log, "Autotune Site Result\n");
              std::fprintf(log, "Site: %zu\n", site.site_index);
              std::fprintf(log, "Site name: %s\n", chipyard_safe_cstr(site.site_name));
              std::fprintf(log, "Site id: %s\n", chipyard_safe_cstr(site.site_id));
              std::fprintf(
                  log,
                  "Dispatch supported: %d\n",
                  static_cast<int>(site.dispatch_supported));
              std::fprintf(log, "Best candidate: %d\n", static_cast<int>(chosen_index));
              std::fprintf(
                  log,
                  "Best kernel: %s\n",
                  chipyard_safe_cstr(chipyard_autotune_kernel_name(site, chosen_index)));
              std::fprintf(
                  log,
                  "Best raw kernel: %s\n",
                  chipyard_safe_cstr(chipyard_autotune_raw_kernel_name(site, chosen_index)));
              std::fprintf(
                  log,
                  "Best cycles: %llu\n",
                  static_cast<unsigned long long>(best_cycles));
              if (reason != nullptr && reason[0] != '\0') {
                std::fprintf(log, "Reason: %s\n", reason);
              }
              std::fprintf(log, "\n");
            }

            static int execute_launch_instance(
                RunnerState& state,
                const ChipyardLaunchInstancePlan& instance,
                int32_t kernel_index,
                int32_t grid_candidate_index,
                std::uint64_t* total_cycles) {
              if (instance.supported == 0) {
                return -1;
              }
              if (instance.grid_fn == nullptr || instance.signature_index < 0) {
                return -1;
              }
              uint32_t grid_0 = 1;
              uint32_t grid_1 = 1;
              uint32_t grid_2 = 1;
              if (chipyard_resolve_grid(
                      state,
                      instance,
                      grid_candidate_index,
                      &grid_0,
                      &grid_1,
                      &grid_2) != 0) {
                return -1;
              }
              if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) {
                return 0;
              }
              switch (instance.signature_index) {
            __DISPATCH_CASES__
                default:
                  return -1;
              }
            }

            static int chipyard_autotune_site(
                RunnerState& state,
                const ChipyardAutotuneSitePlan& site,
                std::uint64_t* autotune_cycles) {
              int32_t chosen_index = site.default_choice_index;
              bool have_best = false;
              std::uint64_t best_cycles = 0;
              if (site.representative_launch_instance_index < 0 ||
                  static_cast<std::size_t>(site.representative_launch_instance_index) >=
                      kChipyardLaunchInstanceCount) {
                return -1;
              }
              const auto& instance =
                  kChipyardLaunchInstances[site.representative_launch_instance_index];
              if (site.dispatch_supported != 0) {
                for (std::size_t candidate_offset = 0;
                     candidate_offset < site.candidate_count;
                     ++candidate_offset) {
                  const auto& candidate = site.candidates[candidate_offset];
                  const int32_t candidate_index = candidate.candidate_index;
                  const int32_t kernel_index = chipyard_candidate_kernel_index(
                      site, candidate_index, instance.default_kernel_index);
                  std::uint64_t candidate_cycles = 0;
                  if (execute_launch_instance(
                          state,
                          instance,
                          kernel_index,
                          candidate_index,
                          &candidate_cycles) != 0) {
                    return -1;
                  }
                  if (autotune_cycles != nullptr) {
                    *autotune_cycles += candidate_cycles;
                  }
                  uint32_t grid_0 = 1;
                  uint32_t grid_1 = 1;
                  uint32_t grid_2 = 1;
                  if (chipyard_resolve_grid(
                          state,
                          instance,
                          candidate_index,
                          &grid_0,
                          &grid_1,
                          &grid_2) == 0) {
                    chipyard_log_autotune_candidate_cycles(
                        state,
                        site,
                        candidate_index,
                        grid_0,
                        grid_1,
                        grid_2,
                        candidate_cycles,
                        "measure");
                  }
                  if (!have_best || candidate_cycles < best_cycles) {
                    chosen_index = candidate_index;
                    best_cycles = candidate_cycles;
                    have_best = true;
                  }
                }
                if (!have_best) {
                  return -1;
                }
                std::string autotune_cache_key;
                if (chipyard_build_autotune_cache_key(
                        &autotune_cache_key, state, site, instance) == 0) {
                  auto& cache_entry = state.autotune_cache[autotune_cache_key];
                  cache_entry.candidate_index = chosen_index;
                  cache_entry.grid_0 = 1;
                  cache_entry.grid_1 = 1;
                  cache_entry.grid_2 = 1;
                  cache_entry.best_cycles = best_cycles;
                  if (chipyard_resolve_grid(
                          state,
                          instance,
                          chosen_index,
                          &cache_entry.grid_0,
                          &cache_entry.grid_1,
                          &cache_entry.grid_2) != 0) {
                    state.autotune_cache.erase(autotune_cache_key);
                  }
                }
              } else {
                const int32_t kernel_index = chipyard_candidate_kernel_index(
                    site, chosen_index, instance.default_kernel_index);
                if (execute_launch_instance(
                        state, instance, kernel_index, chosen_index, &best_cycles) != 0) {
                  return -1;
                }
                if (autotune_cycles != nullptr) {
                  *autotune_cycles += best_cycles;
                }
              }
              if (site.site_index < state.best_kernel_indices.size()) {
                state.best_kernel_indices[site.site_index] = chosen_index;
              }
              chipyard_log_autotune_site_result(
                  state, site, chosen_index, best_cycles, site.dispatch_reason);
              return 0;
            }

            static int execute_chipyard_step(
                const ChipyardStepPlan& step,
                std::size_t step_index,
                RunnerState& state,
                bool allow_autotune,
                std::uint64_t* autotune_kernel_cycles,
                std::uint64_t* model_kernel_cycles,
                std::vector<ChipyardKernelCycleStat>* kernel_cycle_stats) {
              auto& buffers = state.buffers;
              switch (step.kind) {
                case ChipyardStepKind::Alloc:
                  return ensure_buffer_allocated(buffers, step.buffer_index);
                case ChipyardStepKind::Free:
                  if (step.buffer_index < 0) {
                    return -1;
                  }
                  if (static_cast<std::size_t>(step.buffer_index) >= buffers.size()) {
                    return -1;
                  }
                  std::free(buffers[step.buffer_index]);
                  buffers[step.buffer_index] = nullptr;
                  return 0;
                case ChipyardStepKind::Launch: {
                  if (step.launch_instance_index < 0 ||
                      static_cast<std::size_t>(step.launch_instance_index) >=
                          kChipyardLaunchInstanceCount) {
                    return -1;
                  }
                  const auto& instance =
                      kChipyardLaunchInstances[step.launch_instance_index];
                  if (instance.autotune_site_index >= 0) {
                    if (static_cast<std::size_t>(instance.autotune_site_index) >=
                        kChipyardAutotuneSiteCount) {
                      return -1;
                    }
                    const auto& site = kChipyardAutotuneSites[instance.autotune_site_index];
                    std::string autotune_cache_key;
                    int32_t candidate_index = site.default_choice_index;
                    if (static_cast<std::size_t>(instance.autotune_site_index) <
                        state.best_kernel_indices.size()) {
                      candidate_index =
                          state.best_kernel_indices[instance.autotune_site_index];
                    }
                    if (candidate_index < 0) {
                      if (!allow_autotune) {
                        candidate_index = site.default_choice_index;
                      } else {
                        if (chipyard_build_autotune_cache_key(
                                &autotune_cache_key, state, site, instance) == 0) {
                          auto cache_it = state.autotune_cache.find(autotune_cache_key);
                          if (cache_it != state.autotune_cache.end()) {
                            candidate_index = chipyard_candidate_index_from_cache_entry(
                                state, site, instance, cache_it->second);
                            if (candidate_index >= 0 &&
                                static_cast<std::size_t>(instance.autotune_site_index) <
                                    state.best_kernel_indices.size()) {
                              state.best_kernel_indices[instance.autotune_site_index] =
                                  candidate_index;
                              chipyard_log_autotune_candidate_cycles(
                                  state,
                                  site,
                                  candidate_index,
                                  cache_it->second.grid_0,
                                  cache_it->second.grid_1,
                                  cache_it->second.grid_2,
                                  cache_it->second.best_cycles,
                                  "cache_hit");
                              chipyard_log_autotune_site_result(
                                  state,
                                  site,
                                  candidate_index,
                                  cache_it->second.best_cycles,
                                  "cache_hit");
                            }
                          }
                        }
                        if (candidate_index < 0) {
                          return chipyard_autotune_site(
                              state, site, autotune_kernel_cycles);
                        }
                      }
                    }
                    const int32_t kernel_index = chipyard_candidate_kernel_index(
                        site, candidate_index, instance.default_kernel_index);
                    const std::uint64_t cycle_start =
                        model_kernel_cycles != nullptr ? *model_kernel_cycles : 0;
                    const int launch_status = execute_launch_instance(
                        state,
                        instance,
                        kernel_index,
                        candidate_index,
                        model_kernel_cycles);
                    if (launch_status != 0) {
                      return launch_status;
                    }
                    const std::uint64_t launch_cycles =
                        model_kernel_cycles != nullptr
                            ? (*model_kernel_cycles - cycle_start)
                            : 0;
                    chipyard_record_kernel_cycles(
                        kernel_cycle_stats,
                        step_index,
                        &site,
                        candidate_index,
                        chipyard_autotune_kernel_name(site, candidate_index),
                        launch_cycles);
                    return 0;
                  }
                  const std::uint64_t cycle_start =
                      model_kernel_cycles != nullptr ? *model_kernel_cycles : 0;
                  const int launch_status = execute_launch_instance(
                      state,
                      instance,
                      instance.default_kernel_index,
                      static_cast<int32_t>(0),
                      model_kernel_cycles);
                  if (launch_status != 0) {
                    return launch_status;
                  }
                  const std::uint64_t launch_cycles =
                      model_kernel_cycles != nullptr
                          ? (*model_kernel_cycles - cycle_start)
                          : 0;
                  chipyard_record_kernel_cycles(
                      kernel_cycle_stats,
                      step_index,
                      nullptr,
                      static_cast<int32_t>(0),
                      chipyard_kernel_symbol_name(instance.default_kernel_index),
                      launch_cycles);
                  return 0;
                }
                case ChipyardStepKind::CustomCall: {
                  if (step.custom_call_index < 0 ||
                      static_cast<std::size_t>(step.custom_call_index) >=
                          kChipyardCustomCallCount) {
                    return -1;
                  }
                  std::uint64_t* custom_call_cycles =
                      model_kernel_cycles != nullptr
                          ? model_kernel_cycles
                          : autotune_kernel_cycles;
                  return execute_custom_call(
                      step.custom_call_index, state, custom_call_cycles);
                }
              }
              return -1;
            }

            static void print_best_kernel_table(const RunnerState& state) {
              std::FILE* log = state.autotune_log;
              if (log == nullptr || kChipyardAutotuneSiteCount == 0) {
                return;
              }
              std::fprintf(log, "Best Kernel Table\n");
              std::fprintf(
                  log,
                  "Count: %zu\n",
                  static_cast<std::size_t>(kChipyardAutotuneSiteCount));
              for (std::size_t site_index = 0;
                   site_index < kChipyardAutotuneSiteCount;
                   ++site_index) {
                const auto& site = kChipyardAutotuneSites[site_index];
                int32_t chosen_index = site.default_choice_index;
                if (site_index < state.best_kernel_indices.size() &&
                    state.best_kernel_indices[site_index] >= 0) {
                  chosen_index = state.best_kernel_indices[site_index];
                }
                std::fprintf(log, "Best Kernel\n");
                std::fprintf(log, "Site: %zu\n", site_index);
                std::fprintf(log, "Site name: %s\n", chipyard_safe_cstr(site.site_name));
                std::fprintf(log, "Site id: %s\n", chipyard_safe_cstr(site.site_id));
                std::fprintf(
                    log,
                    "Candidate count: %d\n",
                    static_cast<int>(site.candidate_count));
                std::fprintf(
                    log,
                    "Default candidate: %d\n",
                    static_cast<int>(site.default_choice_index));
                std::fprintf(log, "Candidate: %d\n", static_cast<int>(chosen_index));
                std::fprintf(
                    log,
                    "Dispatch supported: %d\n",
                    static_cast<int>(site.dispatch_supported));
                std::fprintf(
                    log,
                    "Kernel: %s\n",
                    chipyard_safe_cstr(chipyard_autotune_kernel_name(site, chosen_index)));
                std::fprintf(
                    log,
                    "Raw kernel: %s\n\n",
                    chipyard_safe_cstr(chipyard_autotune_raw_kernel_name(site, chosen_index)));
              }
              std::fprintf(log, "\n");
            }

            static int execute_model_steps(
                RunnerState& state,
                const char* outputs_path,
                bool capture_outputs,
                bool allow_autotune,
                std::uint64_t* autotune_kernel_cycles,
                std::uint64_t* model_kernel_cycles,
                std::vector<ChipyardKernelCycleStat>* kernel_cycle_stats) {
              bool flex_attention_executed = false;
              for (std::size_t step_index = 0;
                   step_index < kChipyardExecutableStepCount;
                   ++step_index) {
                const auto& step = kChipyardExecutableSteps[step_index];
                if (state.options.flex_attention_kernel_only) {
                  if (step.kind == ChipyardStepKind::Free ||
                      step.kind == ChipyardStepKind::CustomCall) {
                    continue;
                  }
                  if (step.kind == ChipyardStepKind::Launch) {
                    if (step.launch_instance_index < 0 ||
                        static_cast<std::size_t>(step.launch_instance_index) >=
                            kChipyardLaunchInstanceCount) {
                      return -1;
                    }
                    const auto& instance =
                        kChipyardLaunchInstances[step.launch_instance_index];
                    if (instance.autotune_site_index < 0 ||
                        static_cast<std::size_t>(instance.autotune_site_index) >=
                            kChipyardAutotuneSiteCount) {
                      continue;
                    }
                    const auto& site =
                        kChipyardAutotuneSites[instance.autotune_site_index];
                    if (!chipyard_is_flex_attention_site(
                            site, site.default_kernel_name)) {
                      continue;
                    }
                    for (std::size_t arg_index = 0;
                         arg_index < instance.tensor_arg_count;
                         ++arg_index) {
                      const auto& arg = instance.tensor_args[arg_index];
                      if (arg.source_kind == ChipyardValueSourceKind::Buffer &&
                          ensure_buffer_allocated(
                              state.buffers, arg.source_index) != 0) {
                        return -1;
                      }
                      if (arg.source_kind == ChipyardValueSourceKind::Buffer &&
                          kChipyardBuffers[arg.source_index].nbytes != 0) {
                        std::memset(
                            state.buffers[arg.source_index],
                            0,
                            kChipyardBuffers[arg.source_index].nbytes);
                      }
                    }
                    for (std::size_t arg_index = 0;
                         arg_index < instance.scalar_arg_count;
                         ++arg_index) {
                      const auto& arg = instance.scalar_args[arg_index];
                      if (!arg.is_immediate &&
                          arg.source_kind == ChipyardValueSourceKind::Buffer &&
                          ensure_buffer_allocated(
                              state.buffers, arg.source_index) != 0) {
                        return -1;
                      }
                      if (!arg.is_immediate &&
                          arg.source_kind == ChipyardValueSourceKind::Buffer &&
                          kChipyardBuffers[arg.source_index].nbytes != 0) {
                        std::memset(
                            state.buffers[arg.source_index],
                            0,
                            kChipyardBuffers[arg.source_index].nbytes);
                      }
                    }
                  }
                }
                if (execute_chipyard_step(
                        step,
                        step_index,
                        state,
                        allow_autotune,
                        autotune_kernel_cycles,
                        model_kernel_cycles,
                        kernel_cycle_stats) != 0) {
                  return -1;
                }
                if (state.options.flex_attention_kernel_only &&
                    step.kind == ChipyardStepKind::Launch) {
                  flex_attention_executed = true;
                  break;
                }
              }
              if (state.options.flex_attention_kernel_only &&
                  !flex_attention_executed) {
                return -1;
              }
            __EXECUTE_MODEL_EPILOGUE__
              return 0;
            }
            """
        ).strip().splitlines()
        for line in helper_template:
            if line == "__DISPATCH_CASES__":
                runner_lines.extend(dispatch_cases)
            elif line == "__EXECUTE_MODEL_EPILOGUE__":
                runner_lines.extend(execute_model_epilogue_lines)
            else:
                runner_lines.append(line)

    def _plan_chipyard_weights_blob(self) -> dict[str, Any]:
        weight_entries: list[dict[str, Any]] = []
        planned_sources: list[dict[str, Any]] = []
        constant_sources: list[tuple[str, torch.Tensor]] = [
            (str(name), materialized)
            for name, data in sorted(V.graph.constants.items())
            if isinstance(data, torch.Tensor)
            and (materialized := self._materialize_tensor(data)) is not None
        ]
        lifted_sources = self._lifted_graph_input_tensors()
        if not constant_sources and not lifted_sources:
            if self._static_graph_input_indices() or len(getattr(V.graph, "constants", {}) or {}):
                output_code_log.warning(
                    "Chipyard weights blob skipped: constants=%s lifted=%s static_input_idxs=%s graph_inputs=%s",
                    len(constant_sources),
                    len(lifted_sources),
                    self._static_graph_input_indices(),
                    self._all_graph_input_names(),
                )
            self.chipyard_weights_blob_path = None
            self.chipyard_weights_manifest_path = None
            return {
                "entries": weight_entries,
                "planned_sources": planned_sources,
                "manifest": None,
            }

        resolved_sources: list[tuple[str, torch.Tensor]] = []
        storage_use_counts: dict[tuple[int, int], int] = {}
        for name, data in [*constant_sources, *lifted_sources]:
            tensor = data.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            storage_key = self._tensor_storage_key(tensor)
            storage_use_counts[storage_key] = storage_use_counts.get(storage_key, 0) + 1
            resolved_sources.append((name, tensor))

        offset = 0
        manifest_entries: list[dict[str, Any]] = []
        for name, tensor in resolved_sources:
            storage_nbytes = int(tensor.untyped_storage().nbytes())
            aligned_offset = self._align_up(offset)
            itemsize = max(int(tensor.element_size()), 1)
            required_storage_min, required_storage_max = self._tensor_required_storage_bounds(
                tensor
            )
            required_nbytes = (
                0
                if tensor.numel() == 0
                else max(required_storage_max - required_storage_min + 1, 0) * itemsize
            )
            packing_mode = "logical_span"
            packed_storage_offset = required_storage_min
            packed_nbytes = required_nbytes
            if storage_use_counts.get(self._tensor_storage_key(tensor), 0) > 1:
                packing_mode = "full_storage_alias"
                packed_storage_offset = 0
                packed_nbytes = storage_nbytes
            elif (
                packed_storage_offset < 0
                or packed_nbytes < 0
                or packed_storage_offset * itemsize + packed_nbytes > storage_nbytes
            ):
                packing_mode = "full_storage_fallback"
                packed_storage_offset = 0
                packed_nbytes = storage_nbytes

            storage_offset = int(tensor.storage_offset())
            layout_offset_hint = storage_offset - packed_storage_offset
            entry = {
                "name": name,
                "offset": aligned_offset,
                "nbytes": packed_nbytes,
                "storage_offset": storage_offset,
                "packed_storage_offset": packed_storage_offset,
                "layout_offset_hint": layout_offset_hint,
                "packing_mode": packing_mode,
            }
            weight_entries.append(entry)
            manifest_entries.append(
                {
                    **entry,
                    "dtype": str(tensor.dtype),
                    "size": [int(value) for value in tensor.shape],
                    "stride": [int(value) for value in tensor.stride()],
                    "storage_nbytes": storage_nbytes,
                    "required_nbytes": required_nbytes,
                }
            )
            planned_sources.append(
                {
                    **entry,
                    "tensor": tensor,
                }
            )
            offset = aligned_offset + packed_nbytes

            buffer_entry = self.chipyard_buffers.setdefault(name, {"name": name})
            buffer_entry["role"] = "constant"
            buffer_entry["is_output"] = False
            buffer_entry["dtype"] = str(tensor.dtype)
            buffer_entry["size"] = [str(value) for value in tensor.shape]
            buffer_entry["size_hint"] = [int(value) for value in tensor.shape]
            buffer_entry["stride"] = [str(value) for value in tensor.stride()]
            buffer_entry["stride_hint"] = [int(value) for value in tensor.stride()]
            buffer_entry["nbytes_hint"] = entry["nbytes"]
            buffer_entry["weight_offset"] = entry["offset"]
            buffer_entry["weight_nbytes"] = entry["nbytes"]
            buffer_entry["layout_offset_hint"] = entry["layout_offset_hint"]

        if not weight_entries:
            self.chipyard_weights_blob_path = None
            self.chipyard_weights_manifest_path = None
            return {
                "entries": [],
                "planned_sources": [],
                "manifest": None,
            }

        manifest_core = {
            "format_version": 1,
            "alignment": 64,
            "total_bytes": offset,
            "weights": manifest_entries,
        }
        manifest_key = "cweights_" + hashlib.sha256(
            json.dumps(
                manifest_core,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        weights_blob_path = self._named_or_keyed_chipyard_path(
            "weights.bin",
            key=manifest_key,
            extension="bin",
        )
        weights_manifest_path = self._named_or_keyed_chipyard_path(
            "weights.manifest.json",
            key=f"{manifest_key}_manifest",
            extension="json",
        )
        manifest = {
            **manifest_core,
            "weights_blob_path": weights_blob_path,
            "weights_manifest_path": weights_manifest_path,
        }
        self.chipyard_weights_blob_path = weights_blob_path
        self.chipyard_weights_manifest_path = weights_manifest_path
        return {
            "entries": weight_entries,
            "planned_sources": planned_sources,
            "manifest": manifest,
        }

    def _materialize_chipyard_weights_blob(
        self,
        weight_plan: dict[str, Any],
    ) -> list[dict[str, Any]]:
        weight_entries = list(weight_plan.get("entries", []))
        manifest = weight_plan.get("manifest")
        planned_sources = list(weight_plan.get("planned_sources", []))
        if not weight_entries or not isinstance(manifest, dict):
            self.chipyard_weights_blob_path = None
            self.chipyard_weights_manifest_path = None
            return weight_entries

        weights_blob_path = str(manifest.get("weights_blob_path", ""))
        weights_manifest_path = str(manifest.get("weights_manifest_path", ""))
        self.chipyard_weights_blob_path = weights_blob_path or None
        self.chipyard_weights_manifest_path = weights_manifest_path or None

        blob_temp_path: Optional[str] = None
        manifest_temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=chipyard_artifact_dir(),
                suffix=".weights.tmp",
                delete=False,
            ) as blob_file:
                blob_temp_path = blob_file.name
                current_offset = 0
                for planned_entry in planned_sources:
                    entry_offset = int(planned_entry["offset"])
                    pad_bytes = entry_offset - current_offset
                    if pad_bytes > 0:
                        self._stream_write_zeros(blob_file, pad_bytes)
                        current_offset += pad_bytes
                    entry_nbytes = int(planned_entry["nbytes"])
                    if entry_nbytes > 0:
                        tensor = planned_entry["tensor"]
                        storage_ptr = int(tensor.untyped_storage().data_ptr())
                        itemsize = max(int(tensor.element_size()), 1)
                        storage_ptr += int(planned_entry["packed_storage_offset"]) * itemsize
                        self._stream_write_tensor_bytes(
                            blob_file,
                            storage_ptr,
                            entry_nbytes,
                        )
                        current_offset += entry_nbytes
                if current_offset != int(manifest.get("total_bytes", current_offset)):
                    raise RuntimeError(
                        "Chipyard weights blob size mismatch while materializing "
                        f"{weights_blob_path}: expected {manifest.get('total_bytes')}, "
                        f"wrote {current_offset}"
                    )

            with tempfile.NamedTemporaryFile(
                dir=chipyard_artifact_dir(),
                suffix=".weights_manifest.tmp",
                mode="w",
                encoding="utf-8",
                delete=False,
            ) as manifest_file:
                manifest_temp_path = manifest_file.name
                json.dump(manifest, manifest_file, separators=(",", ":"))

            if weights_blob_path:
                os.makedirs(os.path.dirname(weights_blob_path), exist_ok=True)
                assert blob_temp_path is not None
                os.replace(blob_temp_path, weights_blob_path)
                blob_temp_path = None
            if weights_manifest_path:
                os.makedirs(os.path.dirname(weights_manifest_path), exist_ok=True)
                assert manifest_temp_path is not None
                os.replace(manifest_temp_path, weights_manifest_path)
                manifest_temp_path = None
            output_code_log.info(
                "Materialized Chipyard weights blob: %s",
                weights_blob_path,
            )
            return weight_entries
        finally:
            if blob_temp_path is not None and os.path.exists(blob_temp_path):
                os.unlink(blob_temp_path)
            if manifest_temp_path is not None and os.path.exists(manifest_temp_path):
                os.unlink(manifest_temp_path)

    def _build_chipyard_model_spec(self, is_inference: bool) -> dict[str, Any]:
        if not self.chipyard_plan:
            log_fn = (
                output_code_log.warning
                if self.chipyard_kernels or self.chipyard_buffers
                else output_code_log.debug
            )
            log_fn(
                "Chipyard runner emission skipped: empty execution plan "
                "(cpu_backend=%s, launcher_fn=%s, kernels=%d, buffers=%d)",
                config.cpu_backend,
                self.launcher_fn_name,
                len(self.chipyard_kernels),
                len(self.chipyard_buffers),
            )
            return {}

        if self.chipyard_runner_blockers:
            output_code_log.warning(
                "Chipyard runner emission skipped: unsupported wrapper steps detected: %s",
                self.chipyard_runner_blockers,
            )
            return {}

        self._finalize_kernel_artifacts()
        weight_plan = self._plan_chipyard_weights_blob()
        weights = list(weight_plan.get("entries", []))
        buffers = sorted(self.chipyard_buffers.values(), key=lambda item: item["name"])
        runtime_graph_input_names = self._runtime_graph_input_names()
        inputs: list[dict[str, Any]] = []
        input_offset = 0
        for name in runtime_graph_input_names:
            buffer_entry = self.chipyard_buffers.get(name)
            if buffer_entry is None:
                continue
            nbytes = int(buffer_entry.get("nbytes_hint", 0))
            if nbytes <= 0:
                continue
            input_entry = {"name": name, "offset": input_offset, "nbytes": nbytes}
            inputs.append(input_entry)
            input_offset = self._align_up(input_offset + nbytes)
            buffer_entry["input_offset"] = input_entry["offset"]
            buffer_entry["input_nbytes"] = input_entry["nbytes"]
        outputs: list[dict[str, Any]] = []
        output_offset = 0
        for node in V.graph.graph_outputs:
            if not self._has_tensor_output(node):
                continue
            name = self._safe_get_name(node)
            if name is None:
                continue
            buffer_entry = self.chipyard_buffers.get(name)
            if buffer_entry is None:
                self._record_buffer_metadata(node, origin="graph_output")
                buffer_entry = self.chipyard_buffers.get(name)
            if buffer_entry is None:
                continue
            nbytes = int(buffer_entry.get("nbytes_hint", 0))
            if nbytes <= 0:
                continue
            output_entry = {"name": name, "offset": output_offset, "nbytes": nbytes}
            outputs.append(output_entry)
            output_offset = self._align_up(output_offset + nbytes)
            buffer_entry["output_offset"] = output_entry["offset"]
            buffer_entry["output_nbytes"] = output_entry["nbytes"]
        kernels = sorted(
            self.chipyard_kernels.values(), key=lambda item: item["kernel_name"]
        )
        custom_op_libraries: list[dict[str, Any]] = []
        seen_custom_op_libraries: set[str] = set()
        for step in self.chipyard_plan:
            if step.get("kind") != "custom_call":
                continue
            library_path = os.path.abspath(str(step.get("library_path", "")))
            if not library_path or library_path in seen_custom_op_libraries:
                continue
            if not os.path.isfile(library_path):
                self._block_chipyard_runner(
                    f"custom op library disappeared before runner emission: {library_path}"
                )
                return {}
            with open(library_path, "rb") as library_file:
                library_sha256 = hashlib.sha256(library_file.read()).hexdigest()
            custom_op_libraries.append(
                {
                    "path": library_path,
                    "sha256": library_sha256,
                    "basename": os.path.basename(library_path),
                }
            )
            seen_custom_op_libraries.add(library_path)
        triton_custom_libraries: list[dict[str, str]] = []
        for library_path, raw_entry in sorted(
            self.chipyard_triton_custom_libraries.items()
        ):
            if not os.path.isfile(library_path):
                self._block_chipyard_runner(
                    "Triton custom library disappeared before runner emission: "
                    + library_path
                )
                return {}
            with open(library_path, "rb") as library_file:
                actual_sha256 = hashlib.sha256(library_file.read()).hexdigest()
            expected_sha256 = str(raw_entry.get("sha256", ""))
            if expected_sha256 and actual_sha256 != expected_sha256:
                self._block_chipyard_runner(
                    "Triton custom library changed during compilation: "
                    + library_path
                )
                return {}
            triton_custom_libraries.append(
                {**raw_entry, "path": library_path, "sha256": actual_sha256}
            )
        deferred_autotune_sites = self._finalize_chipyard_deferred_autotune_sites()
        return {
            "backend": "triton_chipyard",
            "is_inference": bool(is_inference),
            "launcher_fn": self.launcher_fn_name,
            "graph_input_order": runtime_graph_input_names,
            "buffers": buffers,
            "inputs": inputs,
            "total_input_bytes": input_offset,
            "outputs": outputs,
            "total_output_bytes": output_offset,
            "kernels": kernels,
            "triton_custom_libraries": triton_custom_libraries,
            "custom_op_libraries": custom_op_libraries,
            "deferred_autotune_sites": deferred_autotune_sites,
            "weights": weights,
            "weights_blob_path": self.chipyard_weights_blob_path,
            "weights_manifest_path": self.chipyard_weights_manifest_path,
            "steps": self.chipyard_plan,
            "_chipyard_weight_plan": weight_plan,
        }

    def _emit_chipyard_runner_artifacts(self, is_inference: bool) -> None:
        model_spec = self._build_chipyard_model_spec(is_inference)
        if not model_spec:
            return

        weight_plan = model_spec.pop("_chipyard_weight_plan", {})
        weights = model_spec["weights"]
        buffers = model_spec["buffers"]
        inputs = model_spec["inputs"]
        outputs = model_spec.get("outputs", [])
        kernels = model_spec["kernels"]
        deferred_autotune_sites = model_spec.get("deferred_autotune_sites", [])
        if not isinstance(deferred_autotune_sites, list):
            deferred_autotune_sites = []
        input_offset = int(model_spec["total_input_bytes"])
        total_output_bytes = int(model_spec.get("total_output_bytes", 0))
        kernel_map = {kernel["kernel_name"]: kernel for kernel in kernels}
        buffer_index_map = {
            buffer["name"]: index for index, buffer in enumerate(buffers)
        }
        input_index_map = {
            input_entry["name"]: index for index, input_entry in enumerate(inputs)
        }
        weight_index_map = {
            weight["name"]: index for index, weight in enumerate(weights)
        }
        custom_call_instances: list[dict[str, Any]] = []
        custom_call_index_by_step: dict[int, int] = {}
        for step in self.chipyard_plan:
            if step.get("kind") != "custom_call":
                continue
            custom_call_plan = self._build_chipyard_custom_call_plan(
                step=step,
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
            )
            if not bool(custom_call_plan.get("supported", False)):
                self._block_chipyard_runner(
                    "unsupported custom call "
                    f"{custom_call_plan.get('symbol', '<unknown>')}: "
                    f"{custom_call_plan.get('error_reason', 'unknown error')}"
                )
                return
            step_index = int(step["index"])
            custom_call_index_by_step[step_index] = len(custom_call_instances)
            custom_call_instances.append(custom_call_plan)
        kernel_signature_map: dict[str, list[tuple[str, str]]] = {}
        for step in self.chipyard_plan:
            if step.get("kind") != "launch":
                continue
            kernel_name = str(step.get("kernel_name", "unknown_kernel"))
            step_triton_meta = step.get("triton_meta", {})
            if not isinstance(step_triton_meta, dict):
                continue
            signature = step_triton_meta.get("signature")
            signature_names = step_triton_meta.get("signature_names")
            if kernel_name not in kernel_signature_map:
                kernel_signature_map[kernel_name] = self._runtime_signature_items(
                    signature, signature_names
                )

        deferred_autotune_site_map: dict[str, dict[str, Any]] = {}
        for site in deferred_autotune_sites:
            if not isinstance(site, dict):
                continue
            site_id = str(site.get("site_id", ""))
            if site_id:
                deferred_autotune_site_map[site_id] = site

        ordered_deferred_autotune_sites: list[dict[str, Any]] = []
        deferred_autotune_site_index_by_id: dict[str, int] = {}
        deferred_autotune_site_index_by_step: dict[int, int] = {}
        launch_instances: list[dict[str, Any]] = []
        launch_instance_index_by_step: dict[int, int] = {}
        for step in self.chipyard_plan:
            if step.get("kind") != "launch":
                continue
            step_index = int(step["index"])
            kernel_name = str(step.get("kernel_name", "unknown_kernel"))
            kernel_info = kernel_map.get(kernel_name, {})
            step_triton_meta = step.get("triton_meta", {})
            step_signature_map: dict[str, str] = {}
            step_signature_names: list[str] = []
            if isinstance(step_triton_meta, dict):
                raw_signature = step_triton_meta.get("signature")
                if isinstance(raw_signature, dict):
                    step_signature_map = {
                        str(name): str(dtype) for name, dtype in raw_signature.items()
                    }
                raw_signature_names = step_triton_meta.get("signature_names")
                if isinstance(raw_signature_names, (list, tuple)):
                    step_signature_names = [
                        str(name) for name in raw_signature_names
                    ]
                signature_items = self._runtime_signature_items(
                    raw_signature,
                    raw_signature_names,
                )
            else:
                signature_items = []
            if not signature_items:
                signature_items = kernel_signature_map.get(kernel_name, [])
            extra_launcher_args = self._extra_launcher_args(kernel_info)
            if not step_signature_names and step_signature_map:
                step_signature_names = list(step_signature_map.keys())
            if not step_signature_names:
                step_signature_names = [name for name, _ in signature_items]
            step_arg_names = step_signature_names + extra_launcher_args
            step_arg_types: list[Optional[str]] = [
                step_signature_map.get(name) for name in step_signature_names
            ] + [None] * len(extra_launcher_args)
            step_args = step.get("call_args", [])
            if not isinstance(step_args, list):
                step_args = []
            while len(step_arg_names) > len(step_args):
                trailing_type = step_arg_types[-1] if step_arg_types else None
                if trailing_type != "constexpr":
                    break
                step_arg_names.pop()
                step_arg_types.pop()

            site_id = ""
            site_index: Optional[int] = None
            if isinstance(step_triton_meta, dict):
                site_id = str(step_triton_meta.get("chipyard_autotune_site_id", ""))
            if site_id:
                raw_site_entry = deferred_autotune_site_map.get(site_id)
                if raw_site_entry is not None:
                    site_index = deferred_autotune_site_index_by_id.get(site_id)
                    if site_index is None:
                        site_copy = dict(raw_site_entry)
                        site_copy["representative_launch_instance_index"] = -1
                        site_copy["dispatch_supported"] = False
                        site_copy["dispatch_reason"] = ""
                        ordered_deferred_autotune_sites.append(site_copy)
                        site_index = len(ordered_deferred_autotune_sites) - 1
                        deferred_autotune_site_index_by_id[site_id] = site_index
                    deferred_autotune_site_index_by_step[step_index] = site_index

            deferred_site_inputs: list[dict[str, Any]] = []
            if site_index is not None:
                raw_site_inputs = ordered_deferred_autotune_sites[site_index].get(
                    "inputs", []
                )
                if isinstance(raw_site_inputs, list):
                    deferred_site_inputs = [
                        site_input
                        for site_input in raw_site_inputs
                        if isinstance(site_input, dict)
                    ]

            launch_instance = self._build_chipyard_launch_instance_plan(
                step_index=step_index,
                kernel_name=kernel_name,
                kernel_info=kernel_info,
                step_args=step_args,
                step_arg_names=step_arg_names,
                step_arg_types=step_arg_types,
                signature_items=signature_items,
                site_id=site_id,
                buffer_index_map=buffer_index_map,
                input_index_map=input_index_map,
                weight_index_map=weight_index_map,
                view_overrides_by_arg_index=self._chipyard_launch_view_overrides.get(
                    step_index, {}
                ),
                deferred_site_inputs=deferred_site_inputs,
            )
            launch_instance["autotune_site_index"] = (
                site_index if site_index is not None else -1
            )
            launch_instance["default_kernel_name"] = kernel_name
            launch_instances.append(launch_instance)
            launch_instance_index = len(launch_instances) - 1
            launch_instance_index_by_step[step_index] = launch_instance_index
            if site_index is not None:
                site_entry = ordered_deferred_autotune_sites[site_index]
                if int(site_entry.get("representative_launch_instance_index", -1)) < 0:
                    site_entry["representative_launch_instance_index"] = (
                        launch_instance_index
                    )
                    site_entry["step_kernel_name"] = kernel_name
                    site_entry["signature_items"] = list(signature_items)

        for site in ordered_deferred_autotune_sites:
            dispatch_reasons: list[str] = []
            representative_launch_instance_index = int(
                site.get("representative_launch_instance_index", -1)
            )
            representative_instance: Optional[dict[str, Any]] = None
            if 0 <= representative_launch_instance_index < len(launch_instances):
                representative_instance = launch_instances[
                    representative_launch_instance_index
                ]
            if representative_instance is None:
                dispatch_reasons.append(
                    "no representative launch instance recorded for autotune site"
                )
            elif not bool(representative_instance.get("supported", True)):
                dispatch_reasons.append(
                    str(
                        representative_instance.get(
                            "error_reason",
                            "representative launch instance is unsupported",
                        )
                    )
                )
            site_signature_items = (
                list(representative_instance.get("signature_items", []))
                if isinstance(representative_instance, dict)
                else []
            )
            site_choices = site.get("choices", [])
            if not isinstance(site_choices, list) or not site_choices:
                dispatch_reasons.append("no candidate kernels recorded")
            else:
                for choice in site_choices:
                    if not isinstance(choice, dict):
                        dispatch_reasons.append(
                            "candidate metadata entry is not a dict"
                        )
                        continue
                    choice_kernel_name = str(choice.get("kernel_name", ""))
                    if not choice_kernel_name:
                        dispatch_reasons.append(
                            "candidate kernel metadata is missing kernel_name"
                        )
                        continue
                    choice_signature_items = choice.get("signature_items", [])
                    if (
                        isinstance(choice_signature_items, list)
                        and choice_signature_items
                        and site_signature_items
                        and list(choice_signature_items) != list(site_signature_items)
                    ):
                        dispatch_reasons.append(
                            "candidate runtime signature differs from representative launch instance"
                        )
            site["dispatch_supported"] = not dispatch_reasons
            site["dispatch_reason"] = (
                "; ".join(dict.fromkeys(dispatch_reasons)) if dispatch_reasons else ""
            )

        emit_runner_debug_comments = (
            os.environ.get("TORCHINDUCTOR_CHIPYARD_RUNNER_DEBUG_COMMENTS", "")
            == "1"
        )

        runner_lines = [
            "// Auto-generated triton_chipyard model runner.",
            "// This path stays dormant unless TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER=1 is set.",
            f"// launcher_fn: {self.launcher_fn_name}",
            f"// is_inference: {int(bool(is_inference))}",
            "",
            '#include "mlir/ExecutionEngine/CRunnerUtils.h"',
            '#include "mlir/ExecutionEngine/RunnerUtils.h"',
            "",
            "#include <algorithm>",
            "#include <cstddef>",
            "#include <cstring>",
            "#include <cstdint>",
            "#include <cstdlib>",
            "#include <cstdio>",
            "#include <string>",
            "#include <type_traits>",
            "#include <unordered_map>",
            "#include <vector>",
            "",
            "#ifndef CHIPYARD_OMP_NUM_THREADS",
            "#define CHIPYARD_OMP_NUM_THREADS 4",
            "#endif",
            "",
            "namespace torch::inductor {",
            "",
            "struct BufferPlan {",
            "  const char* name;",
            "  std::size_t nbytes;",
            "};",
            "",
            "struct WeightPlan {",
            "  const char* name;",
            "  std::size_t offset;",
            "  std::size_t nbytes;",
            "};",
            "",
            "struct InputPlan {",
            "  const char* name;",
            "  std::size_t offset;",
            "  std::size_t nbytes;",
            "};",
            "",
            "struct OutputPlan {",
            "  const char* name;",
            "  std::size_t offset;",
            "  std::size_t nbytes;",
            "};",
            "",
            "struct RunnerOptions {",
            "  bool no_output = false;",
            "  bool flex_attention_kernel_only = false;",
            "};",
            "",
            f"constexpr std::size_t kChipyardPlannedSteps = {len(self.chipyard_plan)};",
            f"constexpr std::size_t kChipyardBufferCount = {len(buffers)};",
            f"constexpr std::size_t kChipyardInputCount = {len(inputs)};",
            f"constexpr std::size_t kChipyardOutputCount = {len(outputs)};",
            f"constexpr std::size_t kChipyardWeightCount = {len(weights)};",
            f"constexpr std::size_t kChipyardLaunchInstanceCount = {len(launch_instances)};",
            f"constexpr std::size_t kChipyardCustomCallCount = {len(custom_call_instances)};",
            f"constexpr std::size_t kChipyardAutotuneSiteCount = {len(ordered_deferred_autotune_sites)};",
            "constexpr int32_t kChipyardModelRepeatCount = 5;",
            f"constexpr int32_t kChipyardMaxTensorRank = {CHIPYARD_MAX_TENSOR_RANK};",
            "using BufferTable = std::vector<void*>;",
            "",
            "struct ChipyardAutotuneCacheEntry {",
            "  int32_t candidate_index;",
            "  uint32_t grid_0;",
            "  uint32_t grid_1;",
            "  uint32_t grid_2;",
            "  std::uint64_t best_cycles;",
            "};",
            "",
            "struct RunnerState {",
            "  BufferTable buffers;",
            "  std::vector<std::uint8_t> input_blob;",
            "  std::vector<std::uint8_t> weight_blob;",
            "  std::vector<std::uint8_t> output_blob;",
            "  RunnerOptions options;",
            "  std::FILE* model_log;",
            "  std::FILE* autotune_log;",
            "  std::vector<int32_t> best_kernel_indices;",
            "  std::unordered_map<std::string, ChipyardAutotuneCacheEntry> autotune_cache;",
            "",
            "  RunnerState()",
            "      : buffers(kChipyardBufferCount, nullptr),",
            "        options{},",
            "        model_log(nullptr),",
            "        autotune_log(nullptr),",
            "        best_kernel_indices(",
            "              kChipyardAutotuneSiteCount,",
            "              static_cast<int32_t>(-1)) {}",
            "};",
            "",
            "enum class ChipyardValueSourceKind : std::uint8_t {",
            "  None,",
            "  Buffer,",
            "  InputBlob,",
            "  WeightBlob,",
            "  ImmediateScalar,",
            "};",
            "",
            "enum class ChipyardRuntimeArgKind : std::uint8_t {",
            "  Tensor,",
            "  Scalar,",
            "};",
            "",
            "enum class ChipyardDType : std::uint8_t {",
            "  I1,",
            "  I8,",
            "  I16,",
            "  I32,",
            "  I64,",
            "  U1,",
            "  U8,",
            "  U16,",
            "  U32,",
            "  U64,",
            "  FP16,",
            "  BF16,",
            "  FP32,",
            "  FP64,",
            "};",
            "",
            "struct ChipyardTensorArgPlan {",
            "  ChipyardValueSourceKind source_kind;",
            "  int32_t source_index;",
            "  int32_t offset_elems;",
            "  int32_t rank;",
            "  ChipyardDType dtype;",
            "  int64_t sizes[kChipyardMaxTensorRank];",
            "  int64_t strides[kChipyardMaxTensorRank];",
            "};",
            "",
            "struct ChipyardScalarArgPlan {",
            "  ChipyardValueSourceKind source_kind;",
            "  int32_t source_index;",
            "  int32_t offset_elems;",
            "  ChipyardDType dtype;",
            "  std::uint64_t immediate_bits;",
            "  int32_t is_immediate;",
            "};",
            "",
            "struct ChipyardRuntimeArgPlan {",
            "  ChipyardRuntimeArgKind kind;",
            "  int32_t plan_index;",
            "};",
            "",
            "using ChipyardGridStepFn =",
            "    int (*)(RunnerState&, int32_t, uint32_t*, uint32_t*, uint32_t*);",
            "",
            "enum class ChipyardStepKind : std::uint8_t {",
            "  Alloc,",
            "  Free,",
            "  Launch,",
            "  CustomCall,",
            "};",
            "",
            "struct ChipyardStepPlan {",
            "  ChipyardStepKind kind;",
            "  int32_t buffer_index;",
            "  int32_t launch_instance_index;",
            "  int32_t custom_call_index;",
            "};",
            "",
            "struct ChipyardKernelSymbolPlan {",
            "  const char* kernel_name;",
            "  int32_t signature_index;",
            "};",
            "",
            "struct ChipyardAutotuneConfigEntry {",
            "  const char* key;",
            "  const char* value;",
            "};",
            "",
            "struct ChipyardAutotuneCandidate {",
            "  int32_t candidate_index;",
            "  const char* kernel_name;",
            "  const char* raw_kernel_name;",
            "  const char* normalized_kernel_name;",
            "  int32_t kernel_index;",
            "  std::size_t config_count;",
            "  const ChipyardAutotuneConfigEntry* config_entries;",
            "};",
            "",
            "struct ChipyardAutotuneSitePlan {",
            "  std::size_t site_index;",
            "  int32_t default_choice_index;",
            "  int32_t dispatch_supported;",
            "  const char* site_id;",
            "  const char* site_name;",
            "  const char* dispatch_reason;",
            "  const char* default_kernel_name;",
            "  std::size_t candidate_count;",
            "  const ChipyardAutotuneCandidate* candidates;",
            "  int32_t representative_launch_instance_index;",
            "};",
            "",
            "struct ChipyardLaunchInstancePlan {",
            "  int32_t autotune_site_index;",
            "  int32_t default_kernel_index;",
            "  int32_t signature_index;",
            "  int32_t supported;",
            "  const char* error_reason;",
            "  std::size_t runtime_arg_count;",
            "  const ChipyardRuntimeArgPlan* runtime_args;",
            "  std::size_t tensor_arg_count;",
            "  const ChipyardTensorArgPlan* tensor_args;",
            "  std::size_t scalar_arg_count;",
            "  const ChipyardScalarArgPlan* scalar_args;",
            "  ChipyardGridStepFn grid_fn;",
            "};",
            "",
        ]
        if buffers:
            runner_lines.extend(
                [
                    "static const BufferPlan kChipyardBuffers[] = {",
                    *[
                        f'  {{"{buffer["name"]}", static_cast<std::size_t>({buffer.get("nbytes_hint", 0)})}},'
                        for buffer in buffers
                    ],
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const BufferPlan* kChipyardBuffers = nullptr;",
                    "",
                ]
            )

        total_input_bytes = input_offset
        if inputs:
            runner_lines.extend(
                [
                    "static const InputPlan kChipyardInputs[] = {",
                    *[
                        f'  {{"{input_entry["name"]}", static_cast<std::size_t>({input_entry["offset"]}), '
                        f'static_cast<std::size_t>({input_entry["nbytes"]})}},'
                        for input_entry in inputs
                    ],
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const InputPlan* kChipyardInputs = nullptr;",
                    "",
                ]
            )

        if outputs:
            runner_lines.extend(
                [
                    "static const OutputPlan kChipyardOutputs[] = {",
                    *[
                        f'  {{"{output_entry["name"]}", static_cast<std::size_t>({output_entry["offset"]}), '
                        f'static_cast<std::size_t>({output_entry["nbytes"]})}},'
                        for output_entry in outputs
                    ],
                    "};",
                    "",
                ]
            )
        else:
            runner_lines.extend(
                [
                    "static const OutputPlan* kChipyardOutputs = nullptr;",
                    "",
                ]
            )

        total_weight_bytes = int(
            (weight_plan.get("manifest") or {}).get("total_bytes", 0)
        )
        if weights:
            runner_lines.extend(
                [
                    "static const WeightPlan kChipyardWeights[] = {",
                    *[
                        f'  {{"{weight["name"]}", static_cast<std::size_t>({weight["offset"]}), '
                        f'static_cast<std::size_t>({weight["nbytes"]})}},'
                        for weight in weights
                    ],
                    "};",
                    "",
                ]
            )

        runner_lines.extend(
            [
                "static void cleanup_buffers(BufferTable& buffers) {",
                "  for (void*& ptr : buffers) {",
                "    std::free(ptr);",
                "    ptr = nullptr;",
                "  }",
                "}",
                "",
                "static void reset_runner_state_buffers(RunnerState& state) {",
                "  cleanup_buffers(state.buffers);",
                "  state.buffers.assign(kChipyardBufferCount, nullptr);",
                "}",
                "",
                "static int ensure_buffer_allocated(",
                "    BufferTable& buffers,",
                "    int32_t buffer_index) {",
                "  if (buffer_index < 0) {",
                "    return -1;",
                "  }",
                "  if (static_cast<std::size_t>(buffer_index) >= buffers.size()) {",
                "    return -1;",
                "  }",
                "  if (buffers[buffer_index] == nullptr &&",
                "      kChipyardBuffers[buffer_index].nbytes != 0) {",
                "    buffers[buffer_index] =",
                "        std::malloc(kChipyardBuffers[buffer_index].nbytes);",
                "    if (buffers[buffer_index] == nullptr) {",
                "      return -1;",
                "    }",
                "  }",
                "  return 0;",
                "}",
                "",
                "static inline std::uint64_t rdcycle() {",
                "#if defined(__riscv)",
                "  std::uint64_t value = 0;",
                '  asm volatile ("rdcycle %0" : "=r"(value));',
                "  return value;",
                "#else",
                "  return 0;",
                "#endif",
                "}",
                "",
            ]
        )

        if inputs or weights:
            runner_lines.extend(
                [
                    "static int load_external_blob(",
                    "    const char* path,",
                    "    std::size_t expected_size,",
                    "    std::vector<std::uint8_t>& blob) {",
                    "  if (path == nullptr) {",
                    "    return -1;",
                    "  }",
                    "  blob.resize(expected_size);",
                    '  std::FILE* file = std::fopen(path, "rb");',
                    "  if (file == nullptr) {",
                    "    return -1;",
                    "  }",
                    "  std::size_t bytes_read =",
                    "      std::fread(blob.data(), 1, blob.size(), file);",
                    "  std::fclose(file);",
                    "  if (bytes_read != blob.size()) {",
                    "    blob.clear();",
                    "    return -1;",
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )
        if outputs:
            runner_lines.extend(
                [
                    "static int store_external_blob(",
                    "    const char* path,",
                    "    const std::vector<std::uint8_t>& blob) {",
                    "  if (path == nullptr) {",
                    "    return -1;",
                    "  }",
                    '  std::FILE* file = std::fopen(path, "wb");',
                    "  if (file == nullptr) {",
                    "    return -1;",
                    "  }",
                    "  std::size_t bytes_written = 0;",
                    "  if (!blob.empty()) {",
                    "    bytes_written = std::fwrite(blob.data(), 1, blob.size(), file);",
                    "  }",
                    "  std::fclose(file);",
                    "  if (bytes_written != blob.size()) {",
                    "    return -1;",
                    "  }",
                    "  return 0;",
                    "}",
                    "",
                ]
            )

        runner_lines.extend(
            [
                "static void* resolve_source_ptr(",
                "    RunnerState& state,",
                "    ChipyardValueSourceKind source_kind,",
                "    int32_t source_index) {",
                "  switch (source_kind) {",
                "    case ChipyardValueSourceKind::Buffer:",
                "      if (ensure_buffer_allocated(state.buffers, source_index) != 0) {",
                "        return nullptr;",
                "      }",
                "      if (source_index < 0 ||",
                "          static_cast<std::size_t>(source_index) >= state.buffers.size()) {",
                "        return nullptr;",
                "      }",
                "      return state.buffers[source_index];",
                "    case ChipyardValueSourceKind::InputBlob:",
                "      if (source_index < 0 ||",
                "          static_cast<std::size_t>(source_index) >= kChipyardInputCount) {",
                "        return nullptr;",
                "      }",
                "      return state.input_blob.data() + kChipyardInputs[source_index].offset;",
                "    case ChipyardValueSourceKind::WeightBlob:",
                "      if (source_index < 0 ||",
                "          static_cast<std::size_t>(source_index) >= kChipyardWeightCount) {",
                "        return nullptr;",
                "      }",
                "      return state.weight_blob.data() + kChipyardWeights[source_index].offset;",
                "    case ChipyardValueSourceKind::ImmediateScalar:",
                "    case ChipyardValueSourceKind::None:",
                "      return nullptr;",
                "  }",
                "  return nullptr;",
                "}",
                "",
                "template <typename T>",
                "static T chipyard_scalar_from_bits(std::uint64_t bits) {",
                "  if constexpr (std::is_same_v<T, float>) {",
                "    std::uint32_t raw = static_cast<std::uint32_t>(bits);",
                "    T value;",
                "    std::memcpy(&value, &raw, sizeof(T));",
                "    return value;",
                "  } else if constexpr (std::is_same_v<T, double>) {",
                "    T value;",
                "    std::memcpy(&value, &bits, sizeof(T));",
                "    return value;",
                "  } else if constexpr (std::is_same_v<T, f16> || std::is_same_v<T, bf16>) {",
                "    std::uint16_t raw = static_cast<std::uint16_t>(bits);",
                "    T value;",
                "    std::memcpy(&value, &raw, sizeof(T));",
                "    return value;",
                "  } else {",
                "    return static_cast<T>(bits);",
                "  }",
                "}",
                "",
                "template <typename T>",
                "static int chipyard_load_scalar_arg(",
                "    RunnerState& state,",
                "    const ChipyardScalarArgPlan& plan,",
                "    T* out) {",
                "  if (out == nullptr) {",
                "    return -1;",
                "  }",
                "  if (plan.is_immediate != 0) {",
                "    *out = chipyard_scalar_from_bits<T>(plan.immediate_bits);",
                "    return 0;",
                "  }",
                "  void* base_ptr = resolve_source_ptr(state, plan.source_kind, plan.source_index);",
                "  if (base_ptr == nullptr) {",
                "    return -1;",
                "  }",
                "  auto* typed_ptr = reinterpret_cast<const T*>(base_ptr);",
                "  *out = typed_ptr[static_cast<int64_t>(plan.offset_elems)];",
                "  return 0;",
                "}",
                "",
                "template <typename T>",
                "static int chipyard_make_tensor_arg(",
                "    RunnerState& state,",
                "    const ChipyardTensorArgPlan& plan,",
                "    StridedMemRefType<T, kChipyardMaxTensorRank>* desc,",
                "    UnrankedMemRefType<T>* out) {",
                "  if (desc == nullptr || out == nullptr) {",
                "    return -1;",
                "  }",
                "  void* base_ptr = resolve_source_ptr(state, plan.source_kind, plan.source_index);",
                "  if (base_ptr == nullptr && plan.rank != 0) {",
                "    return -1;",
                "  }",
                "  auto* typed_base = reinterpret_cast<T*>(base_ptr);",
                "  *desc = StridedMemRefType<T, kChipyardMaxTensorRank>{};",
                "  desc->basePtr = typed_base;",
                "  desc->data = typed_base + static_cast<int64_t>(plan.offset_elems);",
                "  desc->offset = 0;",
                "  for (int32_t dim = 0; dim < kChipyardMaxTensorRank; ++dim) {",
                "    desc->sizes[dim] = plan.sizes[dim];",
                "    desc->strides[dim] = plan.strides[dim];",
                "  }",
                "  out->rank = static_cast<int64_t>(plan.rank);",
                "  out->descriptor = desc;",
                "  return 0;",
                "}",
                "",
                "template <typename T, int Rank>",
                "static int chipyard_make_ranked_tensor_arg(",
                "    RunnerState& state,",
                "    const ChipyardTensorArgPlan& plan,",
                "    StridedMemRefType<T, Rank>* desc,",
                "    UnrankedMemRefType<T>* out) {",
                "  if (desc == nullptr || out == nullptr || plan.rank != Rank) {",
                "    return -1;",
                "  }",
                "  void* base_ptr = resolve_source_ptr(state, plan.source_kind, plan.source_index);",
                "  if (base_ptr == nullptr && Rank != 0) {",
                "    return -1;",
                "  }",
                "  auto* typed_base = reinterpret_cast<T*>(base_ptr);",
                "  *desc = StridedMemRefType<T, Rank>{};",
                "  desc->basePtr = typed_base;",
                "  desc->data = typed_base + static_cast<int64_t>(plan.offset_elems);",
                "  desc->offset = 0;",
                "  if constexpr (Rank > 0) {",
                "    for (int32_t dim = 0; dim < Rank; ++dim) {",
                "      desc->sizes[dim] = plan.sizes[dim];",
                "      desc->strides[dim] = plan.strides[dim];",
                "    }",
                "  }",
                "  out->rank = static_cast<int64_t>(Rank);",
                "  out->descriptor = desc;",
                "  return 0;",
                "}",
                "",
            ]
        )

        custom_call_dispatch_count = self._emit_chipyard_custom_call_runtime_plans(
            runner_lines=runner_lines,
            custom_call_instances=custom_call_instances,
        )

        declared_kernels: dict[str, list[tuple[str, str]]] = {}
        for kernel in kernels:
            kernel_name = str(kernel.get("kernel_name", ""))
            if not kernel_name:
                continue
            signature_items = list(kernel_signature_map.get(kernel_name, []))
            if not signature_items:
                kernel_triton_meta = kernel.get("triton_meta")
                if isinstance(kernel_triton_meta, dict):
                    signature_items = self._runtime_signature_items(
                        kernel_triton_meta.get("signature"),
                        kernel_triton_meta.get("signature_names"),
                    )
            declared_kernels[kernel_name] = list(signature_items)
        for site in ordered_deferred_autotune_sites:
            if not bool(site.get("dispatch_supported", False)):
                continue
            signature_items = list(site.get("signature_items", []))
            for choice in site.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                kernel_name = str(choice.get("kernel_name", ""))
                if not kernel_name:
                    continue
                choice_signature_items = list(
                    choice.get("signature_items", signature_items)
                )
                if kernel_name not in declared_kernels or (
                    not declared_kernels[kernel_name] and choice_signature_items
                ):
                    declared_kernels[kernel_name] = list(choice_signature_items)

        if declared_kernels:
            runner_lines.extend(
                [
                    *[
                        f'extern "C" void {kernel_name}('
                        f"{self._kernel_decl_from_signature(signature_items)}"
                        ");"
                        for kernel_name, signature_items in sorted(declared_kernels.items())
                    ],
                    "",
                ]
            )

        signature_dispatch_count = self._emit_chipyard_launch_runtime_plans(
            runner_lines=runner_lines,
            declared_kernels=declared_kernels,
            launch_instances=launch_instances,
            launch_instance_index_by_step=launch_instance_index_by_step,
            custom_call_index_by_step=custom_call_index_by_step,
            ordered_deferred_autotune_sites=ordered_deferred_autotune_sites,
            kernel_map=kernel_map,
            buffer_index_map=buffer_index_map,
            emit_runner_debug_comments=emit_runner_debug_comments,
        )

        execute_model_epilogue_lines: list[str] = []
        if outputs:
            execute_model_epilogue_lines.extend(
                [
                    "  if (capture_outputs) {",
                    f"    state.output_blob.resize(static_cast<std::size_t>({total_output_bytes}));",
                ]
            )
            for output_index, output_entry in enumerate(outputs):
                output_name = output_entry["name"]
                output_buffer = self.chipyard_buffers.get(str(output_name), {})
                output_source_name = self._resolve_chipyard_alias_name(output_name)
                output_source_buffer = self.chipyard_buffers.get(
                    str(output_source_name), output_buffer
                )
                output_role = str(output_source_buffer.get("role", "intermediate"))
                pointer_expr = "nullptr"
                if output_role == "constant":
                    weight_index = weight_index_map.get(output_source_name)
                    if weight_index is not None:
                        pointer_expr = (
                            f"state.weight_blob.data() + kChipyardWeights[{weight_index}].offset"
                        )
                elif output_role == "graph_input":
                    input_index = input_index_map.get(output_source_name)
                    if input_index is not None:
                        pointer_expr = (
                            f"state.input_blob.data() + kChipyardInputs[{input_index}].offset"
                        )
                else:
                    buffer_index = buffer_index_map.get(output_source_name)
                    if buffer_index is not None:
                        pointer_expr = f"state.buffers[{buffer_index}]"
                output_offset_bytes = (
                    int(output_buffer.get("layout_offset_hint", 0))
                    * self._torch_dtype_itemsize(output_buffer.get("dtype"))
                )
                source_pointer_expr = pointer_expr
                if output_offset_bytes != 0:
                    source_pointer_expr = (
                        "reinterpret_cast<const std::uint8_t*>("
                        f"{pointer_expr}) + static_cast<std::size_t>({output_offset_bytes})"
                    )
                execute_model_epilogue_lines.extend(
                    [
                        f"    if ({pointer_expr} == nullptr && "
                        f"kChipyardOutputs[{output_index}].nbytes != 0) {{",
                        "      return -1;",
                        "    }",
                        f"    if (kChipyardOutputs[{output_index}].nbytes != 0) {{",
                        "      std::memcpy(",
                        f"          state.output_blob.data() + kChipyardOutputs[{output_index}].offset,",
                        f"          {source_pointer_expr},",
                        f"          kChipyardOutputs[{output_index}].nbytes);",
                        "    }",
                    ]
                )
            execute_model_epilogue_lines.extend(
                [
                    "  }",
                ]
            )
        else:
            execute_model_epilogue_lines.extend(
                [
                    "  (void)outputs_path;",
                    "  (void)capture_outputs;",
                ]
            )

        store_output_blob_lines: list[str] = []
        if outputs:
            store_output_blob_lines.extend(
                [
                    "  if (!state.options.no_output) {",
                    "    int output_status = store_external_blob(outputs_path, state.output_blob);",
                    "    if (output_status != 0) {",
                    "      reset_runner_state_buffers(state);",
                    "      return -1;",
                    "    }",
                    "  }",
                ]
            )

        self._emit_chipyard_runner_execution_helpers(
            runner_lines=runner_lines,
            signature_dispatch_count=signature_dispatch_count,
            custom_call_dispatch_count=custom_call_dispatch_count,
            execute_model_epilogue_lines=execute_model_epilogue_lines,
        )

        runner_lines.extend(
            [
                "",
                "static int autotune(",
                "    RunnerState& state,",
                "    std::uint64_t* autotune_kernel_cycles) {",
                "  state.best_kernel_indices.assign(",
                "      kChipyardAutotuneSiteCount, static_cast<int32_t>(-1));",
                "  if (autotune_kernel_cycles != nullptr) {",
                "    *autotune_kernel_cycles = 0;",
                "  }",
                "  reset_runner_state_buffers(state);",
                "  const int autotune_status = execute_model_steps(",
                "      state, nullptr, false, true, autotune_kernel_cycles, nullptr, nullptr);",
                "  if (autotune_status != 0) {",
                "    reset_runner_state_buffers(state);",
                "    return autotune_status;",
                "  }",
                "  print_best_kernel_table(state);",
                "  reset_runner_state_buffers(state);",
                "  return 0;",
                "}",
                "",
                "static int run_model(",
                "    RunnerState& state,",
                "    const char* outputs_path,",
                "    bool capture_outputs,",
                "    std::uint64_t* model_kernel_cycles,",
                "    std::vector<ChipyardKernelCycleStat>* kernel_cycle_stats) {",
                "  if (model_kernel_cycles != nullptr) {",
                "    *model_kernel_cycles = 0;",
                "  }",
                "  reset_runner_state_buffers(state);",
                "  const int model_status = execute_model_steps(",
                "      state, outputs_path, capture_outputs, false, nullptr, model_kernel_cycles, kernel_cycle_stats);",
                "  reset_runner_state_buffers(state);",
                "  return model_status;",
                "}",
                "",
                "int run_chipyard_model(",
                "    const char* inputs_path,",
                "    const char* weights_path,",
                "    const char* outputs_path,",
                "    const RunnerOptions& options) {",
                "  RunnerState state;",
                "  state.options = options;",
                "  std::uint64_t autotune_kernel_cycles = 0;",
                "  std::uint64_t model_kernel_cycles_sum = 0;",
                "  std::vector<ChipyardKernelCycleStat> kernel_cycle_stats;",
                "  std::uint64_t full_model_cycles_sum = 0;",
                "  std::uint64_t full_model_cycles_min = 0;",
                "  std::uint64_t full_model_cycles_max = 0;",
                "  std::uint64_t full_model_cycles_samples = 0;",
            ]
        )

        if inputs:
            runner_lines.extend(
                [
                    "  int input_status = load_external_blob(",
                    "      inputs_path,",
                    f"      static_cast<std::size_t>({total_input_bytes}),",
                    "      state.input_blob);",
                    "  if (input_status != 0) {",
                    "    reset_runner_state_buffers(state);",
                    "    return 20 + input_status;",
                    "  }",
                ]
            )

        if weights:
            runner_lines.extend(
                [
                    "  if (!state.options.flex_attention_kernel_only) {",
                    "    int weight_status = load_external_blob(",
                    "        weights_path,",
                    f"        static_cast<std::size_t>({total_weight_bytes}),",
                    "        state.weight_blob);",
                    "    if (weight_status != 0) {",
                    "      reset_runner_state_buffers(state);",
                    "      return -1;",
                    "    }",
                    "  }",
                ]
            )
        if outputs:
            runner_lines.extend(
                [
                    "  if (!state.options.no_output && outputs_path == nullptr) {",
                    "    reset_runner_state_buffers(state);",
                    "    return -1;",
                    "  }",
                ]
            )

        runner_lines.extend(
            [
                '  state.model_log = std::fopen("model.log", "w");',
                "  if (state.model_log == nullptr) {",
                "    reset_runner_state_buffers(state);",
                "    return -1;",
                "  }",
                '  state.autotune_log = std::fopen("autotune.log", "w");',
                "  if (state.autotune_log == nullptr) {",
                "    chipyard_close_log_files(state);",
                "    reset_runner_state_buffers(state);",
                "    return -1;",
                "  }",
                '  std::fprintf(state.autotune_log, "Autotune Log\\n\\n");',
                "",
                '  std::printf("autotuning...\\n");',
                "  const int autotune_status = autotune(",
                "      state, &autotune_kernel_cycles);",
                "  if (autotune_status != 0) {",
                "    chipyard_close_log_files(state);",
                "    reset_runner_state_buffers(state);",
                "    return autotune_status;",
                "  }",
                '  std::printf("autotune done.\\n");',
                "",
                '  std::printf("model launch phase\\n");',
                "  for (int32_t model_run = 0; model_run < kChipyardModelRepeatCount; ++model_run) {",
                "    std::uint64_t model_kernel_cycles = 0;",
                "    const std::uint64_t full_model_cycle_start = rdcycle();",
                "    int model_status = run_model(",
                "        state,",
                "        outputs_path,",
                "        (model_run + 1 == kChipyardModelRepeatCount) && !state.options.no_output,",
                "        &model_kernel_cycles,",
                "        &kernel_cycle_stats);",
                "    const std::uint64_t measured_full_model_cycles =",
                "        rdcycle() - full_model_cycle_start;",
                "    const std::uint64_t full_model_cycles =",
                "        state.options.flex_attention_kernel_only",
                "            ? model_kernel_cycles",
                "            : measured_full_model_cycles;",
                "    if (model_status != 0) {",
                "      chipyard_close_log_files(state);",
                "      reset_runner_state_buffers(state);",
                "      return model_status;",
                "    }",
                "    model_kernel_cycles_sum += model_kernel_cycles;",
                "    full_model_cycles_sum += full_model_cycles;",
                "    if (full_model_cycles_samples == 0 ||",
                "        full_model_cycles < full_model_cycles_min) {",
                "      full_model_cycles_min = full_model_cycles;",
                "    }",
                "    if (full_model_cycles_samples == 0 ||",
                "        full_model_cycles > full_model_cycles_max) {",
                "      full_model_cycles_max = full_model_cycles;",
                "    }",
                "    ++full_model_cycles_samples;",
                '    std::printf("model launch %d/%d cycle:%llu\\n",',
                "        static_cast<int>(model_run + 1),",
                "        static_cast<int>(kChipyardModelRepeatCount),",
                "        static_cast<unsigned long long>(model_kernel_cycles));",
                "  }",
                "  const double total_kernel_cycles_avg_fp =",
                "      kChipyardModelRepeatCount > 0",
                "          ? (static_cast<double>(model_kernel_cycles_sum) /",
                "             static_cast<double>(kChipyardModelRepeatCount))",
                "          : 0.0;",
                '  std::printf("end\\n");',
                '  std::printf("avg model launch cycle:%.2f\\n",',
                "      total_kernel_cycles_avg_fp);",
                "  chipyard_write_model_log(",
                "      state.model_log,",
                "      kernel_cycle_stats,",
                "      full_model_cycles_sum,",
                "      full_model_cycles_min,",
                "      full_model_cycles_max,",
                "      full_model_cycles_samples,",
                "      model_kernel_cycles_sum);",
                "  chipyard_close_log_files(state);",
                *store_output_blob_lines,
                "  reset_runner_state_buffers(state);",
                "  return 0;",
                "}",
                "",
                "}  // namespace torch::inductor",
                "",
                "int main(int argc, char** argv) {",
                "  torch::inductor::RunnerOptions options{};",
                "  const char* positional_args[3] = {nullptr, nullptr, nullptr};",
                "  int positional_count = 0;",
                "  for (int arg_index = 1; arg_index < argc; ++arg_index) {",
                "    const char* arg = argv[arg_index];",
                '    if (arg != nullptr && std::strcmp(arg, "--log-autotune") == 0) {',
                "      continue;",
                "    }",
                '    if (arg != nullptr && std::strcmp(arg, "--log-kernels") == 0) {',
                "      continue;",
                "    }",
                '    if (arg != nullptr && std::strcmp(arg, "--no-output") == 0) {',
                "      options.no_output = true;",
                "      continue;",
                "    }",
                '    if (arg != nullptr && std::strcmp(arg, "--kernel-only=flex_attention") == 0) {',
                "      options.flex_attention_kernel_only = true;",
                "      options.no_output = true;",
                "      continue;",
                "    }",
                "    if (positional_count >= 3) {",
                "      return -1;",
                "    }",
                "    positional_args[positional_count++] = arg;",
                "  }",
                "  return torch::inductor::run_chipyard_model(",
                "      positional_args[0], positional_args[1], positional_args[2], options);",
                "}",
                "",
            ]
        )
        runner_path = chipyard_named_write("\n".join(runner_lines), "runner.cpp")
        build_artifacts = emit_chipyard_model_build_artifacts(
            runner_cpp_path=runner_path,
            model_spec=model_spec,
        )
        run_artifacts = emit_chipyard_run_artifacts(
            model_spec=build_artifacts.materialized_model_spec,
            build_script_path=build_artifacts.build_script_path,
            output_elf_path=build_artifacts.output_elf_path,
            weights_blob_path=self.chipyard_weights_blob_path,
        )
        self._materialize_chipyard_weights_blob(weight_plan)
        self.chipyard_run_script_path = run_artifacts.script_path
        self.chipyard_model_build_script_path = build_artifacts.build_script_path
        self.chipyard_model_elf_path = build_artifacts.output_elf_path
        self.additional_files.extend(
            [
                runner_path,
                run_artifacts.script_path,
                run_artifacts.model_spec_path,
                build_artifacts.build_script_path,
            ]
        )
        if self.chipyard_weights_blob_path is not None:
            self.additional_files.append(self.chipyard_weights_blob_path)
        if self.chipyard_weights_manifest_path is not None:
            self.additional_files.append(self.chipyard_weights_manifest_path)
        if build_artifacts.built_output_path is not None:
            self.additional_files.append(build_artifacts.built_output_path)
        output_code_log.info("Chipyard runner stub written to: %s", runner_path)
        output_code_log.info(
            "Chipyard run script written to: %s",
            run_artifacts.script_path,
        )
        output_code_log.info(
            "Chipyard model spec written to: %s",
            run_artifacts.model_spec_path,
        )
        if self.chipyard_weights_blob_path is not None:
            output_code_log.info(
                "Chipyard weights blob written to: %s",
                self.chipyard_weights_blob_path,
            )
        if self.chipyard_weights_manifest_path is not None:
            output_code_log.info(
                "Chipyard weights manifest written to: %s",
                self.chipyard_weights_manifest_path,
            )
        output_code_log.info(
            "Chipyard model build script written to: %s",
            build_artifacts.build_script_path,
        )
        if build_artifacts.built_output_path is not None:
            output_code_log.info(
                "Chipyard model ELF written to: %s",
                build_artifacts.built_output_path,
            )

    def generate(self, is_inference):
        wrapper_code, kernel_decls = super().generate(is_inference)
        output_code_log.info(
            "ChipyardWrapperCodegen.generate reached: plan_steps=%d kernels=%d buffers=%d inference=%s",
            len(self.chipyard_plan),
            len(self.chipyard_kernels),
            len(self.chipyard_buffers),
            bool(is_inference),
        )
        self._emit_chipyard_runner_artifacts(is_inference)
        return wrapper_code, kernel_decls


class CpuWrapperCodegen(PythonWrapperCodegen):
    """
    CPU wrapper constructor that can dispatch to a backend-specific wrapper at
    wrapper creation time without re-registering the CPU device.
    """

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper: Optional[PythonWrapperCodegen],
        partition_signatures: Optional[ir.GraphPartitionSignature] = None,
    ):
        if (
            not is_subgraph
            and config.cpu_backend == "triton_chipyard"
            and _chipyard_runner_codegen_enabled()
        ):
            return ChipyardWrapperCodegen()
        return PythonWrapperCodegen.create(
            is_subgraph, subgraph_name, parent_wrapper, partition_signatures
        )

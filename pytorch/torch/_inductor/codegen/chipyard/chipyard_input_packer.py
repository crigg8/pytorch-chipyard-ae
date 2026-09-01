from __future__ import annotations

import ctypes
import dataclasses
import json
import os
import pprint
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ...codecache import output_code_log
from .chipyard_codegen_utils import chipyard_artifact_dir, chipyard_write


_BLOCK_MASK_FIELD_BY_SIGNATURE_NAME = {
    "KV_NUM_BLKS": "kv_num_blocks",
    "KV_IDX": "kv_indices",
    "FULL_KV_NUM_BLKS": "full_kv_num_blocks",
    "FULL_KV_IDX": "full_kv_indices",
    "arg_KV_NUM_BLKS": "kv_num_blocks",
    "arg_KV_IDX": "kv_indices",
    "arg_FULL_KV_NUM_BLKS": "full_kv_num_blocks",
    "arg_FULL_KV_IDX": "full_kv_indices",
}


def _tensor_to_storage_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.is_mkldnn:
        raise TypeError("mkldnn inputs are not supported by the Chipyard inputs.bin writer")

    storage = tensor.detach().untyped_storage()
    if tensor.device.type != "cpu":
        storage = storage.cpu()

    nbytes = storage.nbytes()
    if nbytes == 0:
        return b""

    raw_array = ctypes.cast(
        storage.data_ptr(),
        ctypes.POINTER(ctypes.c_ubyte * nbytes),
    )
    return bytes(raw_array.contents)


def _buffer_plan_map(model_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    buffers = model_spec.get("buffers", [])
    if not isinstance(buffers, list):
        return {}
    return {
        str(buffer["name"]): buffer
        for buffer in buffers
        if isinstance(buffer, dict) and "name" in buffer
    }


def _materialize_model_spec(
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
) -> dict[str, Any]:
    if isinstance(model_spec, Mapping):
        return dict(model_spec)
    with open(os.fspath(model_spec)) as spec_file:
        return json.load(spec_file)


def _is_block_mask_like(value: Any) -> bool:
    return all(
        hasattr(value, field)
        for field in (
            "kv_num_blocks",
            "kv_indices",
            "full_kv_num_blocks",
            "full_kv_indices",
        )
    )


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    attr = dtype_name.split(".")[-1]
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype_name}")
    return dtype


def _zero_tensor_from_buffer_plan(buffer_plan: dict[str, Any]) -> torch.Tensor:
    dtype = _torch_dtype_from_name(str(buffer_plan.get("dtype", "torch.int32")))
    size = [int(x) for x in buffer_plan.get("size_hint", [])]
    if not size:
        return torch.zeros((), dtype=dtype)
    return torch.zeros(size, dtype=dtype)


def _block_mask_input_fields(model_spec: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for step in model_spec.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "launch":
            continue
        signature_names = step.get("triton_meta", {}).get("signature_names", [])
        call_args = step.get("call_args", [])
        if not isinstance(signature_names, list) or not isinstance(call_args, list):
            continue
        for index, signature_name in enumerate(signature_names):
            field = _BLOCK_MASK_FIELD_BY_SIGNATURE_NAME.get(str(signature_name))
            if field is None or index >= len(call_args):
                continue
            call_arg = call_args[index]
            if not isinstance(call_arg, dict) or call_arg.get("kind") != "graph_input":
                continue
            name = str(call_arg.get("name", ""))
            if name:
                previous = fields.setdefault(name, field)
                if previous != field:
                    raise ValueError(
                        f"Conflicting BlockMask field mapping for Chipyard input {name!r}: "
                        f"{previous!r} vs {field!r}"
                    )
    return fields


def _first_block_mask_like(values: Sequence[Any]) -> Any:
    for value in values:
        if _is_block_mask_like(value):
            return value
    return None


def _block_mask_tensor(
    block_mask: Any,
    field: str,
    *,
    name: str,
    buffer_plan: dict[str, Any],
) -> torch.Tensor:
    tensor = getattr(block_mask, field, None)
    if tensor is None:
        if field.startswith("full_"):
            return _zero_tensor_from_buffer_plan(buffer_plan)
        raise ValueError(f"BlockMask field {field!r} is missing for Chipyard input {name!r}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"BlockMask field {field!r} for Chipyard input {name!r} must be a "
            f"torch.Tensor, got {type(tensor)}"
        )
    return tensor


def _expand_block_mask_inputs(
    model_spec: dict[str, Any],
    named_inputs: dict[str, Any],
) -> dict[str, Any]:
    block_mask = _first_block_mask_like(tuple(named_inputs.values()))
    if block_mask is None:
        return named_inputs

    field_by_name = _block_mask_input_fields(model_spec)
    if not field_by_name:
        return named_inputs

    buffer_plan_map = _buffer_plan_map(model_spec)
    expanded = dict(named_inputs)
    for name, field in field_by_name.items():
        value = expanded.get(name)
        if value is not None and not _is_block_mask_like(value):
            continue
        expanded[name] = _block_mask_tensor(
            block_mask,
            field,
            name=name,
            buffer_plan=buffer_plan_map.get(name, {}),
        )
    return expanded


def _normalize_named_inputs(
    model_spec: dict[str, Any],
    args: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    if isinstance(args, Mapping):
        return {str(name): value for name, value in args.items()}

    if not isinstance(args, (list, tuple)):
        raise TypeError(f"Unsupported inputs type: {type(args)}")

    input_order = model_spec.get("graph_input_order", [])
    tensor_inputs = model_spec.get("inputs", [])
    tensor_input_names = [
        str(entry["name"])
        for entry in tensor_inputs
        if isinstance(entry, dict) and "name" in entry
    ]

    if isinstance(input_order, list) and len(args) == len(input_order):
        return {
            str(name): value
            for name, value in zip(input_order, args)
        }

    if len(args) == len(tensor_input_names):
        return {
            name: value for name, value in zip(tensor_input_names, args)
        }

    block_mask = _first_block_mask_like(args)
    field_by_name = _block_mask_input_fields(model_spec) if block_mask is not None else {}
    if field_by_name:
        ordered_names = [
            str(name)
            for name in (
                tensor_input_names
                if tensor_input_names
                else input_order
            )
        ]
        non_block_mask_args = [value for value in args if not _is_block_mask_like(value)]
        non_block_mask_names = [
            name for name in ordered_names if name not in field_by_name
        ]
        if len(non_block_mask_args) == len(non_block_mask_names):
            named_inputs = {
                name: value
                for name, value in zip(non_block_mask_names, non_block_mask_args)
            }
            named_inputs["__chipyard_block_mask__"] = block_mask
            return named_inputs

    raise ValueError(
        "Could not map positional inputs to the Chipyard model spec. "
        f"Got {len(args)} args, graph_input_order has {len(input_order)} entries, "
        f"tensor input plan has {len(tensor_input_names)} entries."
    )


def _validate_input_tensor(
    *,
    name: str,
    tensor: Any,
    input_plan: dict[str, Any],
    buffer_plan: dict[str, Any],
) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input {name!r} must be a torch.Tensor, got {type(tensor)}")

    expected_dtype = buffer_plan.get("dtype")
    if expected_dtype is not None and str(tensor.dtype) != str(expected_dtype):
        raise ValueError(
            f"Input {name!r} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        )

    expected_size = buffer_plan.get("size_hint")
    if isinstance(expected_size, list) and [int(x) for x in tensor.size()] != [
        int(x) for x in expected_size
    ]:
        raise ValueError(
            f"Input {name!r} size mismatch: expected {expected_size}, got {list(tensor.size())}"
        )

    expected_stride = buffer_plan.get("stride_hint")
    if isinstance(expected_stride, list) and [int(x) for x in tensor.stride()] != [
        int(x) for x in expected_stride
    ]:
        raise ValueError(
            f"Input {name!r} stride mismatch: expected {expected_stride}, got {list(tensor.stride())}"
        )

    expected_storage_offset = buffer_plan.get("layout_offset_hint")
    if expected_storage_offset is not None and int(tensor.storage_offset()) != int(
        expected_storage_offset
    ):
        raise ValueError(
            f"Input {name!r} storage_offset mismatch: expected {expected_storage_offset}, "
            f"got {tensor.storage_offset()}"
        )

    storage_bytes = _tensor_to_storage_bytes(tensor)
    expected_nbytes = int(input_plan.get("nbytes", 0))
    if len(storage_bytes) != expected_nbytes:
        raise ValueError(
            f"Input {name!r} storage byte size mismatch: expected {expected_nbytes}, "
            f"got {len(storage_bytes)}"
        )
    return storage_bytes


def prepare_chipyard_named_inputs(
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
    args: Mapping[str, Any] | Sequence[Any],
    *,
    pack_input_fn: Optional[Callable[[str, Any], Any]] = None,
) -> dict[str, Any]:
    materialized_spec = _materialize_model_spec(model_spec)
    named_inputs = _normalize_named_inputs(materialized_spec, args)
    named_inputs = _expand_block_mask_inputs(materialized_spec, named_inputs)
    if pack_input_fn is not None:
        named_inputs = {
            name: pack_input_fn(name, value)
            for name, value in named_inputs.items()
        }
    return named_inputs


def write_chipyard_inputs_bin(
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
    args: Mapping[str, Any] | Sequence[Any],
    output_path: Optional[str] = None,
) -> str:
    materialized_spec = _materialize_model_spec(model_spec)

    named_inputs = prepare_chipyard_named_inputs(materialized_spec, args)
    buffer_plan_map = _buffer_plan_map(materialized_spec)
    input_plans = materialized_spec.get("inputs", [])
    if not isinstance(input_plans, list):
        raise TypeError("Chipyard model spec does not contain a valid inputs list")

    total_input_bytes = int(
        materialized_spec.get(
            "total_input_bytes",
            max(
                (
                    int(entry.get("offset", 0)) + int(entry.get("nbytes", 0))
                    for entry in input_plans
                    if isinstance(entry, dict)
                ),
                default=0,
            ),
        )
    )

    temp_output_path: Optional[str] = None
    try:
        if output_path is None:
            with tempfile.NamedTemporaryFile(
                dir=chipyard_artifact_dir(), suffix=".inputs.bin", delete=False
            ) as output_file:
                temp_output_path = output_file.name
                output_path = temp_output_path

        assert output_path is not None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as output_file:
            if total_input_bytes > 0:
                output_file.seek(total_input_bytes - 1)
                output_file.write(b"\0")

            for input_plan in input_plans:
                if not isinstance(input_plan, dict):
                    continue
                name = str(input_plan["name"])
                if name not in named_inputs:
                    raise KeyError(f"Missing tensor input {name!r} for Chipyard inputs.bin")

                buffer_plan = buffer_plan_map.get(name, {})
                storage_bytes = _validate_input_tensor(
                    name=name,
                    tensor=named_inputs[name],
                    input_plan=input_plan,
                    buffer_plan=buffer_plan,
                )
                output_file.seek(int(input_plan.get("offset", 0)))
                output_file.write(storage_bytes)
    except Exception:
        if temp_output_path is not None and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
        raise

    output_code_log.info("Chipyard inputs blob written to: %s", output_path)
    return output_path


@dataclasses.dataclass(frozen=True)
class ChipyardInputPackArtifacts:
    script_path: str


def emit_chipyard_input_pack_artifacts(
    *, model_spec: Mapping[str, Any] | str | os.PathLike[str]
) -> ChipyardInputPackArtifacts:
    materialized_spec = _materialize_model_spec(model_spec)
    pack_spec = {
        "graph_input_order": materialized_spec.get("graph_input_order", []),
        "buffers": materialized_spec.get("buffers", []),
        "inputs": materialized_spec.get("inputs", []),
        "total_input_bytes": materialized_spec.get("total_input_bytes", 0),
    }
    pack_spec_literal = pprint.pformat(pack_spec, width=100, sort_dicts=True)
    script_source = f"""\
from torch._inductor.codegen.chipyard.chipyard_input_packer import (
    write_chipyard_inputs_bin,
)

MODEL_SPEC = {pack_spec_literal}


def pack_inputs(args, output_path=None):
    return write_chipyard_inputs_bin(MODEL_SPEC, args, output_path=output_path)


if __name__ == "__main__":
    raise SystemExit(
        "Import this file and call pack_inputs(args, output_path=...) from Python."
    )
"""
    script_path = chipyard_write(script_source, "py")[1]
    output_code_log.info("Chipyard input pack script written to: %s", script_path)
    return ChipyardInputPackArtifacts(script_path=script_path)

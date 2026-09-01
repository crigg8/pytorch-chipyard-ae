from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _materialize_model_spec(
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
) -> dict[str, Any]:
    if isinstance(model_spec, Mapping):
        return dict(model_spec)
    with open(os.fspath(model_spec)) as spec_file:
        return json.load(spec_file)


def _buffer_plan_map(model_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    buffers = model_spec.get("buffers", [])
    if not isinstance(buffers, list):
        return {}
    return {
        str(buffer["name"]): buffer
        for buffer in buffers
        if isinstance(buffer, dict) and "name" in buffer
    }


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    attr = dtype_name.split(".")[-1]
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype_name}")
    return dtype


def _tensor_from_storage_bytes(
    storage_bytes: bytes,
    buffer_plan: dict[str, Any],
) -> torch.Tensor:
    dtype_name = str(buffer_plan.get("dtype", "torch.float32"))
    dtype = _parse_torch_dtype(dtype_name)
    size_hint = [int(x) for x in buffer_plan.get("size_hint", [])]
    stride_hint = [int(x) for x in buffer_plan.get("stride_hint", [])]
    storage_offset = int(buffer_plan.get("layout_offset_hint", 0))

    if not size_hint:
        raise ValueError(f"Output buffer plan is missing size_hint: {buffer_plan}")
    if len(stride_hint) != len(size_hint):
        raise ValueError(f"Output buffer plan has invalid stride_hint: {buffer_plan}")

    byte_buffer = bytearray(storage_bytes)
    if not byte_buffer:
        return torch.empty(size_hint, dtype=dtype)

    storage_view = memoryview(byte_buffer)
    base = torch.frombuffer(storage_view, dtype=dtype)
    # Clone so the returned tensor owns its storage instead of borrowing the bytearray.
    return torch.as_strided(base, size_hint, stride_hint, storage_offset).clone()


def read_chipyard_outputs_bin(
    model_spec: Mapping[str, Any] | str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> tuple[torch.Tensor, ...]:
    materialized_spec = _materialize_model_spec(model_spec)
    buffer_plan_map = _buffer_plan_map(materialized_spec)
    output_plans = materialized_spec.get("outputs", [])
    if not isinstance(output_plans, list):
        raise TypeError("Chipyard model spec does not contain a valid outputs list")

    blob = Path(output_path).read_bytes()
    outputs: list[torch.Tensor] = []
    for output_plan in output_plans:
        if not isinstance(output_plan, dict):
            continue
        name = str(output_plan["name"])
        offset = int(output_plan.get("offset", 0))
        nbytes = int(output_plan.get("nbytes", 0))
        chunk = blob[offset : offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(
                f"Output {name!r} byte size mismatch: expected {nbytes}, got {len(chunk)}"
            )
        buffer_plan = buffer_plan_map.get(name)
        if buffer_plan is None:
            raise KeyError(f"Missing buffer plan for output {name!r}")
        outputs.append(_tensor_from_storage_bytes(chunk, buffer_plan))

    return tuple(outputs)

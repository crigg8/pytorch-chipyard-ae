from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, Optional

from ...codecache import output_code_log
from .chipyard_codegen_utils import chipyard_named_write


@dataclasses.dataclass(frozen=True)
class ChipyardRunArtifacts:
    script_path: str
    model_spec_path: str


def emit_chipyard_run_artifacts(
    *,
    model_spec: Mapping[str, Any],
    build_script_path: str,
    output_elf_path: str,
    weights_blob_path: Optional[str],
) -> ChipyardRunArtifacts:
    del build_script_path, output_elf_path, weights_blob_path

    model_spec_path = chipyard_named_write(
        json.dumps(
            dict(model_spec),
            separators=(",", ":"),
        ),
        "model_spec.json",
    )
    script_source = """\
from __future__ import annotations

import ctypes
import json
import os
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_ARTIFACT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_SPEC_PATH = _ARTIFACT_DIR / "model_spec.json"
INPUTS_BIN_PATH = _ARTIFACT_DIR / "input.bin"
OUTPUTS_BIN_PATH = _ARTIFACT_DIR / "output.bin"


def _load_model_spec() -> dict[str, Any]:
    with MODEL_SPEC_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


MODEL_SPEC = _load_model_spec()


def _buffer_plan_map() -> dict[str, dict[str, Any]]:
    buffers = MODEL_SPEC.get("buffers", [])
    if not isinstance(buffers, list):
        return {}
    return {
        str(buffer["name"]): buffer
        for buffer in buffers
        if isinstance(buffer, dict) and "name" in buffer
    }


def _input_plans() -> list[dict[str, Any]]:
    inputs = MODEL_SPEC.get("inputs", [])
    if not isinstance(inputs, list):
        raise TypeError("model_spec.json does not contain a valid inputs list")
    return [entry for entry in inputs if isinstance(entry, dict)]


def _output_plans() -> list[dict[str, Any]]:
    outputs = MODEL_SPEC.get("outputs", [])
    if not isinstance(outputs, list):
        raise TypeError("model_spec.json does not contain a valid outputs list")
    return [entry for entry in outputs if isinstance(entry, dict)]


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    attr = dtype_name.split(".")[-1]
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype_name}")
    return dtype


def _required_storage_size(
    size_hint: list[int],
    stride_hint: list[int],
    storage_offset: int,
) -> int:
    if not size_hint:
        return max(storage_offset + 1, 1)
    max_index = storage_offset
    for size_value, stride_value in zip(size_hint, stride_hint):
        if size_value <= 0:
            return 0
        max_index += (size_value - 1) * stride_value
    return max_index + 1


def _normalize_named_inputs(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, Mapping):
        return {str(name): value for name, value in inputs.items()}

    if isinstance(inputs, torch.Tensor):
        positional_inputs = (inputs,)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes, bytearray)):
        positional_inputs = tuple(inputs)
    else:
        positional_inputs = (inputs,)

    graph_input_order = MODEL_SPEC.get("graph_input_order", [])
    if isinstance(graph_input_order, list) and len(positional_inputs) == len(graph_input_order):
        return {
            str(name): value
            for name, value in zip(graph_input_order, positional_inputs)
        }

    input_names = [str(entry["name"]) for entry in _input_plans() if "name" in entry]
    if len(positional_inputs) == len(input_names):
        return {
            name: value
            for name, value in zip(input_names, positional_inputs)
        }

    raise ValueError(
        "Could not map positional inputs to model_spec.json. "
        f"Got {len(positional_inputs)} input(s), graph_input_order has "
        f"{len(graph_input_order) if isinstance(graph_input_order, list) else 0}, "
        f"inputs has {len(input_names)}."
    )


def _materialize_input_tensor(
    *,
    name: str,
    value: Any,
    input_plan: dict[str, Any],
    buffer_plan: dict[str, Any],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Input {name!r} must be a torch.Tensor, got {type(value)}")

    expected_dtype = _parse_torch_dtype(str(buffer_plan.get("dtype", value.dtype)))
    expected_size = [int(x) for x in buffer_plan.get("size_hint", list(value.shape))]
    expected_stride = [
        int(x) for x in buffer_plan.get("stride_hint", list(value.stride()))
    ]
    expected_storage_offset = int(buffer_plan.get("layout_offset_hint", 0))

    tensor = value.detach().to(device="cpu", dtype=expected_dtype)
    if list(tensor.shape) != expected_size:
        raise ValueError(
            f"Input {name!r} size mismatch: expected {expected_size}, "
            f"got {list(tensor.shape)}"
        )
    if len(expected_stride) != len(expected_size):
        raise ValueError(
            f"Input {name!r} has invalid stride plan: {expected_stride} "
            f"for size {expected_size}"
        )

    storage_size = _required_storage_size(
        expected_size,
        expected_stride,
        expected_storage_offset,
    )
    storage = torch.zeros(max(storage_size, 0), dtype=expected_dtype, device="cpu")
    packed = torch.as_strided(
        storage,
        size=expected_size,
        stride=expected_stride,
        storage_offset=expected_storage_offset,
    )
    packed.copy_(tensor.contiguous())

    expected_nbytes = int(input_plan.get("nbytes", 0))
    actual_nbytes = packed.untyped_storage().nbytes()
    if actual_nbytes != expected_nbytes:
        raise ValueError(
            f"Input {name!r} storage byte size mismatch: expected "
            f"{expected_nbytes}, got {actual_nbytes}"
        )
    return packed


def _tensor_storage_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.is_mkldnn:
        raise TypeError("mkldnn tensors are not supported")

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


def write_inputs_bin(
    inputs: Any,
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    resolved_output_path = pathlib.Path(output_path) if output_path is not None else INPUTS_BIN_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    named_inputs = _normalize_named_inputs(inputs)
    buffer_plans = _buffer_plan_map()
    input_plans = _input_plans()
    total_input_bytes = int(
        MODEL_SPEC.get(
            "total_input_bytes",
            max(
                (
                    int(entry.get("offset", 0)) + int(entry.get("nbytes", 0))
                    for entry in input_plans
                ),
                default=0,
            ),
        )
    )

    with resolved_output_path.open("wb") as output_file:
        if total_input_bytes > 0:
            output_file.seek(total_input_bytes - 1)
            output_file.write(b"\\0")

        for input_plan in input_plans:
            name = str(input_plan["name"])
            if name not in named_inputs:
                raise KeyError(f"Missing tensor input {name!r}")
            packed = _materialize_input_tensor(
                name=name,
                value=named_inputs[name],
                input_plan=input_plan,
                buffer_plan=buffer_plans.get(name, {}),
            )
            output_file.seek(int(input_plan.get("offset", 0)))
            output_file.write(_tensor_storage_bytes(packed))

    return str(resolved_output_path)


def _tensor_from_storage_bytes(
    storage_bytes: bytes,
    buffer_plan: dict[str, Any],
    *,
    record_kind: str,
) -> torch.Tensor:
    dtype = _parse_torch_dtype(str(buffer_plan.get("dtype", "torch.float32")))
    size_hint = [int(x) for x in buffer_plan.get("size_hint", [])]
    stride_hint = [int(x) for x in buffer_plan.get("stride_hint", [])]
    storage_offset = int(buffer_plan.get("layout_offset_hint", 0))

    if len(stride_hint) != len(size_hint):
        raise ValueError(
            f"{record_kind} buffer plan has invalid stride_hint: {buffer_plan}"
        )

    byte_buffer = bytearray(storage_bytes)
    if not byte_buffer:
        return torch.empty(size_hint, dtype=dtype)

    base = torch.frombuffer(memoryview(byte_buffer), dtype=dtype)
    return torch.as_strided(base, size_hint, stride_hint, storage_offset).clone()


def _read_bin_records(
    *,
    path: pathlib.Path,
    plans: list[dict[str, Any]],
    record_kind: str,
) -> list[tuple[str, torch.Tensor]]:
    blob = path.read_bytes()
    buffer_plans = _buffer_plan_map()
    records: list[tuple[str, torch.Tensor]] = []

    for plan in plans:
        name = str(plan["name"])
        offset = int(plan.get("offset", 0))
        nbytes = int(plan.get("nbytes", 0))
        chunk = blob[offset : offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(
                f"{record_kind} {name!r} byte size mismatch: expected {nbytes}, "
                f"got {len(chunk)}"
            )
        buffer_plan = buffer_plans.get(name)
        if buffer_plan is None:
            raise KeyError(f"Missing buffer plan for {record_kind} {name!r}")
        records.append(
            (
                name,
                _tensor_from_storage_bytes(
                    chunk,
                    buffer_plan,
                    record_kind=record_kind,
                ),
            )
        )
    return records


def _ordered_input_names() -> list[str]:
    input_names = [str(entry["name"]) for entry in _input_plans() if "name" in entry]
    graph_input_order = MODEL_SPEC.get("graph_input_order", [])
    if not isinstance(graph_input_order, list):
        return input_names

    available = set(input_names)
    ordered = [str(name) for name in graph_input_order if str(name) in available]
    ordered.extend(name for name in input_names if name not in set(ordered))
    return ordered


def read_named_inputs_bin(
    input_path: str | os.PathLike[str] | None = None,
) -> dict[str, torch.Tensor]:
    resolved_input_path = pathlib.Path(input_path) if input_path is not None else INPUTS_BIN_PATH
    return dict(
        _read_bin_records(
            path=resolved_input_path,
            plans=_input_plans(),
            record_kind="Input",
        )
    )


def read_inputs_bin(
    input_path: str | os.PathLike[str] | None = None,
    *,
    named: bool = False,
):
    named_inputs = read_named_inputs_bin(input_path)
    if named:
        return named_inputs

    ordered_inputs = [
        named_inputs[name]
        for name in _ordered_input_names()
        if name in named_inputs
    ]
    if len(ordered_inputs) == 1:
        return ordered_inputs[0]
    return tuple(ordered_inputs)


def read_outputs_bin(
    output_path: str | os.PathLike[str] | None = None,
):
    resolved_output_path = pathlib.Path(output_path) if output_path is not None else OUTPUTS_BIN_PATH
    outputs = [
        tensor
        for _name, tensor in _read_bin_records(
            path=resolved_output_path,
            plans=_output_plans(),
            record_kind="Output",
        )
    ]

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


pack_inputs_bin = write_inputs_bin
unpack_inputs_bin = read_inputs_bin
unpack_outputs_bin = read_outputs_bin
"""
    script_path = chipyard_named_write(script_source, "util.py")
    output_code_log.info("Chipyard utility script written to: %s", script_path)
    output_code_log.info("Chipyard model spec written to: %s", model_spec_path)
    return ChipyardRunArtifacts(
        script_path=script_path,
        model_spec_path=model_spec_path,
    )

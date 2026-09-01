from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def input_entry_names(model_spec: dict[str, Any]) -> list[str]:
    return [
        str(entry["name"])
        for entry in model_spec.get("inputs", [])
        if isinstance(entry, dict) and "name" in entry
    ]


def graph_input_args_by_signature(step: dict[str, Any]) -> dict[str, str]:
    signature_names = step.get("triton_meta", {}).get("signature_names", [])
    call_args = step.get("call_args", [])
    if not isinstance(signature_names, list) or not isinstance(call_args, list):
        return {}

    names: dict[str, str] = {}
    for index, signature_name in enumerate(signature_names):
        if index >= len(call_args):
            continue
        call_arg = call_args[index]
        if not isinstance(call_arg, dict) or call_arg.get("kind") != "graph_input":
            continue
        name = str(call_arg.get("name", ""))
        if name:
            names[str(signature_name)] = name
    return names


def dense_mask_input_names(model_spec: dict[str, Any]) -> tuple[str, str]:
    candidate: tuple[str, str] | None = None
    for step in model_spec.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "launch":
            continue
        kernel_name = str(step.get("kernel_name", ""))
        if (
            "scaled_dot_product" not in kernel_name
            and "softmax" not in kernel_name
            and "masked_fill" not in kernel_name
        ):
            continue

        graph_inputs = graph_input_args_by_signature(step)
        cache_position_name = graph_inputs.get("in_ptr1")
        attention_mask_name = graph_inputs.get("in_ptr2")
        if cache_position_name and attention_mask_name:
            step_candidate = (cache_position_name, attention_mask_name)
            if candidate is None:
                candidate = step_candidate
            elif candidate != step_candidate:
                raise ValueError(
                    "conflicting dense cache_position/attention_mask input mappings: "
                    f"{candidate!r} vs {step_candidate!r}"
                )

    if candidate is None:
        raise ValueError("could not identify dense cache_position/attention_mask inputs")
    return candidate


def sdpa_embedding_input_names(
    model_spec: dict[str, Any],
    *,
    excluded_names: set[str],
) -> tuple[str, str]:
    for step in model_spec.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "launch":
            continue
        graph_inputs = {
            signature_name: name
            for signature_name, name in graph_input_args_by_signature(step).items()
            if name not in excluded_names
        }
        input_ids_name = graph_inputs.get("in_ptr0")
        position_ids_name = graph_inputs.get("in_ptr2")
        if input_ids_name and position_ids_name:
            return input_ids_name, position_ids_name

    remaining_names = [
        name for name in input_entry_names(model_spec) if name not in excluded_names
    ]
    if len(remaining_names) == 2:
        return remaining_names[0], remaining_names[1]

    raise ValueError(
        "could not identify SDPA input_ids/position_ids inputs; "
        f"remaining graph inputs are {remaining_names}"
    )


def dense_sdpa_input_names_from_spec(model_spec: dict[str, Any]) -> dict[str, str]:
    cache_position_name, attention_mask_name = dense_mask_input_names(model_spec)
    input_ids_name, position_ids_name = sdpa_embedding_input_names(
        model_spec,
        excluded_names={cache_position_name, attention_mask_name},
    )
    return {
        "input_ids": input_ids_name,
        "attention_mask": attention_mask_name,
        "position_ids": position_ids_name,
        "cache_position": cache_position_name,
    }


def dense_sdpa_named_inputs_from_spec(
    model_spec: dict[str, Any],
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    name_by_field = dense_sdpa_input_names_from_spec(model_spec)
    return {
        graph_name: inputs[field_name]
        for field_name, graph_name in name_by_field.items()
    }


def dense_sdpa_inputs_from_artifact(util: Any, path: Path) -> dict[str, torch.Tensor]:
    named_inputs = util.read_named_inputs_bin(path / "input.bin")
    name_by_field = dense_sdpa_input_names_from_spec(util.MODEL_SPEC)
    return {
        field_name: named_inputs[graph_name]
        for field_name, graph_name in name_by_field.items()
    }


def require_named_input_support(module: Any, util_path: Path) -> None:
    if not hasattr(module, "read_inputs_bin") or not hasattr(module, "read_named_inputs_bin"):
        raise RuntimeError(
            f"{os.fspath(util_path)} was generated before named input support; "
            "recompile artifacts"
        )

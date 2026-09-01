#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config as inductor_config
from llm_artifact_inputs import (
    dense_sdpa_inputs_from_artifact,
    dense_sdpa_named_inputs_from_spec,
    require_named_input_support,
    sdpa_embedding_input_names,
)
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


TASK_NAME = "pythia-160m"
MODEL_NAME = "transformers.GPTNeoXConfig(160M proxy)"
DTYPE = torch.float32
INTEGER_DTYPE = torch.int32
ATTENTION_MODES = ("sdpa", "flash", "window")
DEFAULT_FLEX_WINDOW_SIZE = 128
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = SCRIPT_DIR.parent / "IR"
MODEL_INPUT_NAMES = ("input_ids", "attention_mask", "position_ids", "cache_position")
BLOCK_MASK_FIELD_BY_SIGNATURE_NAME = {
    "KV_NUM_BLKS": "kv_num_blocks",
    "KV_IDX": "kv_indices",
    "FULL_KV_NUM_BLKS": "full_kv_num_blocks",
    "FULL_KV_IDX": "full_kv_indices",
    "arg_KV_NUM_BLKS": "kv_num_blocks",
    "arg_KV_IDX": "kv_indices",
    "arg_FULL_KV_NUM_BLKS": "full_kv_num_blocks",
    "arg_FULL_KV_IDX": "full_kv_indices",
}
_BLOCK_MASK_CACHE: dict[tuple[str, int, int, int, int, str], Any] = {}
VALIDATE_ATOL = 1e-3


def token_length() -> int:
    value = int(os.environ.get("LLM_TOKEN_LENGTH", "256"))
    if value < 1:
        raise ValueError(f"LLM_TOKEN_LENGTH must be >= 1, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile or validate Pythia-160M Chipyard artifacts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true", help="Generate artifacts and input.bin.")
    mode.add_argument("--validate", action="store_true", help="Compare output.bin with eager PyTorch.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn", choices=ATTENTION_MODES, default="sdpa")
    return parser.parse_args()


def artifact_dir(attn: str) -> Path:
    default_path = DEFAULT_ARTIFACT_ROOT / f"{TASK_NAME}-{attn}"
    return Path(os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH", default_path)).resolve()


def configure_triton_chipyard(task_name: str) -> None:
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    cache_dir = Path(os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/{task_name}"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"


def configure_artifact_env(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"


def resolved_attention_implementation(attn: str) -> str:
    return "sdpa" if attn == "sdpa" else "flex_attention"


def build_config(attn: str) -> GPTNeoXConfig:
    config = GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
        use_parallel_residual=True,
    )
    config._attn_implementation = resolved_attention_implementation(attn)
    return config


def build_model(seed: int, attn: str) -> torch.nn.Module:
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(build_config(attn)).to(device="cpu", dtype=DTYPE).eval()


def make_random_inputs(batch_size: int, seed: int, config: GPTNeoXConfig) -> dict[str, torch.Tensor]:
    seq_len = token_length()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        low=1,
        high=int(config.vocab_size),
        size=(batch_size, seq_len),
        generator=generator,
        dtype=INTEGER_DTYPE,
    )
    attention_mask = torch.ones((batch_size, seq_len), dtype=INTEGER_DTYPE)
    position_ids = torch.arange(seq_len, dtype=INTEGER_DTYPE).unsqueeze(0).repeat(batch_size, 1)
    cache_position = torch.arange(seq_len, dtype=INTEGER_DTYPE)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }


def _causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


def _make_sliding_window_mask_mod(window_size: int):
    def _sliding_window_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_size)

    return _sliding_window_mask_mod


def create_flex_block_mask(attn: str, inputs: dict[str, torch.Tensor]):
    if attn == "sdpa":
        return inputs["attention_mask"]
    batch_size, query_length = inputs["input_ids"].shape
    key_value_length = int(inputs["attention_mask"].shape[-1])
    window_size = min(DEFAULT_FLEX_WINDOW_SIZE, max(1, min(query_length, key_value_length)))
    cache_key = (attn, batch_size, query_length, key_value_length, window_size, "cpu")
    cached = _BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    mask_mod = _causal_mask_mod if attn == "flash" else _make_sliding_window_mask_mod(window_size)
    block_mask = create_block_mask(mask_mod, batch_size, 1, query_length, key_value_length, device="cpu")
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


class LastTokenLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=False,
            return_dict=False,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        return logits[:, -1:, :].clone()


def model_input_tuple(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(inputs[name] for name in MODEL_INPUT_NAMES)


def block_mask_input_fields(model_spec: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for step in model_spec.get("steps", []):
        if not isinstance(step, dict) or step.get("kind") != "launch":
            continue
        signature_names = step.get("triton_meta", {}).get("signature_names", [])
        call_args = step.get("call_args", [])
        if not isinstance(signature_names, list) or not isinstance(call_args, list):
            continue
        for index, signature_name in enumerate(signature_names):
            field = BLOCK_MASK_FIELD_BY_SIGNATURE_NAME.get(str(signature_name))
            if field is None or index >= len(call_args):
                continue
            call_arg = call_args[index]
            if isinstance(call_arg, dict) and call_arg.get("kind") == "graph_input":
                name = str(call_arg.get("name", ""))
                if name:
                    fields[name] = field
    return fields


def block_mask_tensor(block_mask: Any, field: str, reference: torch.Tensor | None) -> torch.Tensor:
    tensor = getattr(block_mask, field, None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    if field.startswith("full_") and reference is not None:
        return torch.zeros_like(reference)
    raise ValueError(f"BlockMask field {field!r} is missing")


def flex_named_inputs_from_spec(
    model_spec: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    block_mask: Any,
) -> dict[str, torch.Tensor]:
    field_by_name = block_mask_input_fields(model_spec)
    if not field_by_name:
        raise ValueError("compiled artifact does not contain BlockMask inputs")

    input_ids_name, position_ids_name = sdpa_embedding_input_names(
        model_spec,
        excluded_names=set(field_by_name),
    )

    named_inputs: dict[str, torch.Tensor] = {
        input_ids_name: inputs["input_ids"],
        position_ids_name: inputs["position_ids"],
    }
    references: dict[str, torch.Tensor] = {}
    for name, field in field_by_name.items():
        reference = references.get(field.removeprefix("full_"))
        tensor = block_mask_tensor(block_mask, field, reference)
        named_inputs[name] = tensor
        references[field] = tensor
    return named_inputs


def import_artifact_util(path: Path):
    util_path = path / "util.py"
    if not util_path.exists():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location(f"{TASK_NAME}_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import artifact util: {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require_named_input_support(module, util_path)
    return module


def dense_inputs_from_artifact(util, path: Path) -> dict[str, torch.Tensor]:
    return dense_sdpa_inputs_from_artifact(util, path)


def flex_inputs_from_artifact(util, path: Path) -> dict[str, torch.Tensor]:
    field_by_name = block_mask_input_fields(util.MODEL_SPEC)
    if not field_by_name:
        raise ValueError("compiled artifact does not contain BlockMask inputs")
    input_ids_name, position_ids_name = sdpa_embedding_input_names(
        util.MODEL_SPEC,
        excluded_names=set(field_by_name),
    )

    named_inputs = util.read_named_inputs_bin(path / "input.bin")
    input_ids = named_inputs[input_ids_name]
    position_ids = named_inputs[position_ids_name]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": position_ids,
        "cache_position": torch.arange(input_ids.shape[1], dtype=position_ids.dtype),
    }


def inputs_from_artifact(util, path: Path, attn: str) -> dict[str, torch.Tensor]:
    if attn == "sdpa":
        return dense_inputs_from_artifact(util, path)
    return flex_inputs_from_artifact(util, path)


def compare_tensors(golden: torch.Tensor, observed: torch.Tensor) -> bool:
    if tuple(golden.shape) != tuple(observed.shape):
        if golden.numel() != observed.numel():
            print("[validate] mae=inf")
            print("[validate] match=False")
            return False
        observed = observed.reshape_as(golden)

    golden_fp32 = golden.detach().to(torch.float32)
    observed_fp32 = observed.detach().to(torch.float32)
    abs_err = (observed_fp32 - golden_fp32).abs()
    mae = float(abs_err.mean()) if abs_err.numel() else 0.0
    match = mae <= VALIDATE_ATOL
    print(f"[validate] mae={mae:.6e}")
    print(f"[validate] match={match}")
    return match


def print_config(path: Path, input_shape: tuple[int, ...], attn: str) -> None:
    print(f"[config] model={MODEL_NAME}")
    print(f"[config] input_shape={input_shape}")
    print("[config] dtype=fp32")
    print(f"[config] attn={attn}")
    print(f"[config] artifact_dir={path}")


def run_compile(args: argparse.Namespace) -> None:
    path = artifact_dir(args.attn)
    configure_artifact_env(path)
    configure_triton_chipyard(TASK_NAME)
    config = build_config(args.attn)
    model = build_model(args.seed, args.attn)
    inputs = make_random_inputs(args.batch_size, args.seed, config)
    attention = create_flex_block_mask(args.attn, inputs)
    module = LastTokenLogits(model)
    print_config(path, tuple(inputs["input_ids"].shape), args.attn)

    started_at = time.perf_counter()
    compiled_module = torch.compile(module, backend="inductor")
    with torch.inference_mode():
        _ = compiled_module(inputs["input_ids"], attention, inputs["position_ids"], inputs["cache_position"])
    compile_time_s = time.perf_counter() - started_at

    util = import_artifact_util(path)
    if args.attn == "sdpa":
        input_path = util.write_inputs_bin(dense_sdpa_named_inputs_from_spec(util.MODEL_SPEC, inputs))
    else:
        input_path = util.write_inputs_bin(flex_named_inputs_from_spec(util.MODEL_SPEC, inputs, attention))
    print(f"[compile] seconds={compile_time_s:.3f}")
    print(f"[artifact] input_bin={input_path}")


def run_validate(args: argparse.Namespace) -> None:
    path = artifact_dir(args.attn)
    util = import_artifact_util(path)
    inputs = inputs_from_artifact(util, path, args.attn)
    attention = create_flex_block_mask(args.attn, inputs)
    observed = util.read_outputs_bin(path / "output.bin")
    if not isinstance(observed, torch.Tensor):
        raise TypeError(f"expected one tensor output, got {type(observed)!r}")

    print_config(path, tuple(inputs["input_ids"].shape), args.attn)
    module = LastTokenLogits(build_model(args.seed, args.attn))
    with torch.inference_mode():
        golden = module(inputs["input_ids"], attention, inputs["position_ids"], inputs["cache_position"])
    if not compare_tensors(golden, observed):
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.compile:
        run_compile(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()

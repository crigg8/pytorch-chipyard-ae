#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from llm_artifact_inputs import (
    dense_sdpa_inputs_from_artifact,
    dense_sdpa_named_inputs_from_spec,
    require_named_input_support,
)
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig


TASK_NAME = "gpt-neo-125m"
MODEL_NAME = "transformers.GPTNeoConfig(125M proxy)"
DTYPE = torch.float32
INTEGER_DTYPE = torch.int32
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR.parent / "IR" / TASK_NAME
MODEL_INPUT_NAMES = ("input_ids", "attention_mask", "position_ids", "cache_position")
VALIDATE_ATOL = 1e-3


def token_length() -> int:
    value = int(os.environ.get("LLM_TOKEN_LENGTH", "256"))
    if value < 1:
        raise ValueError(f"LLM_TOKEN_LENGTH must be >= 1, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile or validate GPT-Neo 125M Chipyard artifacts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true", help="Generate artifacts and input.bin.")
    mode.add_argument("--validate", action="store_true", help="Compare output.bin with eager PyTorch.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def artifact_dir() -> Path:
    return Path(os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH", DEFAULT_ARTIFACT_DIR)).resolve()


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


def build_config() -> GPTNeoConfig:
    return GPTNeoConfig(
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        attention_types=[[["global", "local"], 6]],
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
    )


def build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(build_config()).to(device="cpu", dtype=DTYPE).eval()


def make_random_inputs(batch_size: int, seed: int, config: GPTNeoConfig) -> dict[str, torch.Tensor]:
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


class LastTokenLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
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


def inputs_from_artifact(util, path: Path) -> dict[str, torch.Tensor]:
    return dense_sdpa_inputs_from_artifact(util, path)


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


def print_config(path: Path, input_shape: tuple[int, ...]) -> None:
    print(f"[config] model={MODEL_NAME}")
    print(f"[config] input_shape={input_shape}")
    print("[config] dtype=fp32")
    print(f"[config] artifact_dir={path}")


def run_compile(args: argparse.Namespace) -> None:
    path = artifact_dir()
    configure_artifact_env(path)
    configure_triton_chipyard(TASK_NAME)
    config = build_config()
    model = build_model(args.seed)
    inputs = make_random_inputs(args.batch_size, args.seed, config)
    module = LastTokenLogits(model)
    print_config(path, tuple(inputs["input_ids"].shape))

    started_at = time.perf_counter()
    compiled_module = torch.compile(module, backend="inductor")
    with torch.inference_mode():
        _ = compiled_module(*model_input_tuple(inputs))
    compile_time_s = time.perf_counter() - started_at

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(dense_sdpa_named_inputs_from_spec(util.MODEL_SPEC, inputs))
    print(f"[compile] seconds={compile_time_s:.3f}")
    print(f"[artifact] input_bin={input_path}")


def run_validate(args: argparse.Namespace) -> None:
    path = artifact_dir()
    util = import_artifact_util(path)
    inputs = inputs_from_artifact(util, path)
    observed = util.read_outputs_bin(path / "output.bin")
    if not isinstance(observed, torch.Tensor):
        raise TypeError(f"expected one tensor output, got {type(observed)!r}")

    print_config(path, tuple(inputs["input_ids"].shape))
    module = LastTokenLogits(build_model(args.seed))
    with torch.inference_mode():
        golden = module(*model_input_tuple(inputs))
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from PIL import Image
from torchvision.models import AlexNet_Weights, alexnet


TASK_NAME = "alexnet"
MODEL_NAME = "torchvision.alexnet"
MODEL_WEIGHTS = AlexNet_Weights.DEFAULT
DTYPE = torch.float32
INPUT_SHAPE = (3, 224, 224)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR.parent / "IR" / TASK_NAME
VALIDATE_ATOL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile or validate AlexNet Chipyard artifacts.")
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


def make_module_tensors_contiguous(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if not parameter.is_contiguous():
            parameter.data = parameter.data.contiguous()
    for buffer in module.buffers():
        if not buffer.is_contiguous():
            buffer.data = buffer.data.contiguous()


def build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = alexnet(weights=MODEL_WEIGHTS).to(device="cpu", dtype=DTYPE).eval()
    make_module_tensors_contiguous(model)
    return model


def make_image_input(batch_size: int) -> torch.Tensor:
    preprocess = MODEL_WEIGHTS.transforms()
    # Keep the AE input self-contained. Pixel values do not affect the compiled
    # CNN graph, but using a fixed image makes input.bin deterministic.
    image = Image.new("RGB", (INPUT_SHAPE[2], INPUT_SHAPE[1]), (73, 109, 137))
    tensor = preprocess(image).to(dtype=DTYPE)
    batch = tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous(
        memory_format=torch.channels_last
    )
    if batch_size == 1:
        return torch.as_strided(
            batch, size=batch.shape, stride=(batch.stride(-1), *batch.stride()[1:])
        )
    return batch


def import_artifact_util(path: Path):
    util_path = path / "util.py"
    if not util_path.exists():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location(f"{TASK_NAME}_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import artifact util: {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "read_inputs_bin"):
        raise RuntimeError(f"{util_path} was generated before read_inputs_bin support; recompile artifacts")
    return module


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
    print(f"[config] weights={MODEL_WEIGHTS.name}")
    print(f"[config] input_shape={input_shape}")
    print("[config] dtype=fp32")
    print("[config] input=deterministic-rgb-73-109-137")
    print(f"[config] artifact_dir={path}")


def run_compile(args: argparse.Namespace) -> None:
    path = artifact_dir()
    configure_artifact_env(path)
    configure_triton_chipyard(TASK_NAME)
    model = build_model(args.seed)
    inputs = make_image_input(args.batch_size)
    print_config(path, tuple(inputs.shape))

    started_at = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor")
    with torch.inference_mode():
        _ = compiled_model(inputs)
    compile_time_s = time.perf_counter() - started_at

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(inputs)
    print(f"[compile] seconds={compile_time_s:.3f}")
    print(f"[artifact] input_bin={input_path}")


def run_validate(args: argparse.Namespace) -> None:
    path = artifact_dir()
    util = import_artifact_util(path)
    inputs = util.read_inputs_bin(path / "input.bin")
    observed = util.read_outputs_bin(path / "output.bin")
    if not isinstance(inputs, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("AlexNet artifacts must contain one input tensor and one output tensor")

    print_config(path, tuple(inputs.shape))
    model = build_model(args.seed)
    with torch.inference_mode():
        golden = model(inputs)
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

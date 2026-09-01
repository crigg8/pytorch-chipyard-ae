#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config


TASK_NAME = "gemmini-max-autotune"
MODEL_NAME = "single_matmul_1024x1024x4096"
DTYPE = torch.float32
M = 1024
K = 1024
N = 4096
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = SCRIPT_DIR.parent / "IR" / TASK_NAME
VALIDATE_ATOL = 1e-3
SEED = 0
EXPECTED_AUTOTUNE_CANDIDATES = 2163
SIMPLE_STAGE2_AUTOTUNE_CANDIDATES = 81


class SingleMatmulModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        weight = torch.randn(K, N, generator=generator, dtype=DTYPE) * 0.05
        self.register_buffer("weight", weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile or validate Gemmini max-autotune Chipyard artifacts."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile", action="store_true", help="Generate artifacts and input.bin.")
    mode.add_argument("--validate", action="store_true", help="Compare output.bin with eager PyTorch.")
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
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"


def configure_artifact_env(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE", "1")


def build_model() -> torch.nn.Module:
    return SingleMatmulModule().to(device="cpu", dtype=DTYPE).eval()


def make_input() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    return torch.randn(M, K, generator=generator, dtype=DTYPE)


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
            print("[validate] max_abs_err=inf")
            print("[validate] match=False")
            return False
        observed = observed.reshape_as(golden)

    golden_fp32 = golden.detach().to(torch.float32)
    observed_fp32 = observed.detach().to(torch.float32)
    abs_err = (observed_fp32 - golden_fp32).abs()
    max_abs_err = float(abs_err.max()) if abs_err.numel() else 0.0
    match = max_abs_err <= VALIDATE_ATOL
    print(f"[validate] max_abs_err={max_abs_err:.6e}")
    print(f"[validate] match={match}")
    return match


def print_config(path: Path, input_shape: tuple[int, ...]) -> None:
    print(f"[config] model={MODEL_NAME}")
    print(f"[config] input_shape={input_shape}")
    print(f"[config] weight_shape={(K, N)}")
    print("[config] dtype=fp32")
    print(f"[config] artifact_dir={path}")


def validate_autotune_candidates(path: Path) -> None:
    spec_path = path / "model_spec.json"
    document = json.loads(spec_path.read_text())
    sites = document.get("deferred_autotune_sites")
    if not isinstance(sites, list) or len(sites) != 1:
        raise RuntimeError(
            f"expected one deferred autotune site in {spec_path}, got {sites!r}"
        )
    site = sites[0]
    if not isinstance(site, dict):
        raise RuntimeError(f"autotune site in {spec_path} is not an object")
    choices = site.get("choices")
    if not isinstance(choices, list):
        raise RuntimeError(f"autotune site in {spec_path} has no choices list")
    candidate_count = len(choices)
    expected_candidate_count = (
        SIMPLE_STAGE2_AUTOTUNE_CANDIDATES
        if os.environ.get("PYTORCH_CHIPYARD_SIMPLE_STAGE2", "").strip().lower()
        in {"1", "true", "yes", "on"}
        else EXPECTED_AUTOTUNE_CANDIDATES
    )
    if candidate_count != expected_candidate_count:
        raise RuntimeError(
            "Gemmini autotune candidate count mismatch: "
            f"expected {expected_candidate_count}, got {candidate_count}"
        )
    print(f"[artifact] autotune_candidates={candidate_count}")


def run_compile(args: argparse.Namespace) -> None:
    path = artifact_dir()
    configure_artifact_env(path)
    configure_triton_chipyard(TASK_NAME)
    model = build_model()
    inputs = make_input()
    print_config(path, tuple(inputs.shape))

    started_at = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor")
    with torch.inference_mode():
        _ = compiled_model(inputs)
    compile_time_s = time.perf_counter() - started_at

    validate_autotune_candidates(path)
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
        raise TypeError("Gemmini autotune artifacts must contain one input tensor and one output tensor")

    print_config(path, tuple(inputs.shape))
    model = build_model()
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

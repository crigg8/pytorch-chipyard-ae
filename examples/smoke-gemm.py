#!/usr/bin/env python3
"""Compile the bounded 32x32x32 GEMM used by the artifact smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch._inductor.config as inductor_config


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from smoke_test_spec import smoke_gemm_kernel


KERNEL = smoke_gemm_kernel()
TASK_NAME = str(KERNEL["id"])
DTYPE = torch.float32
SEED = 2027


class SingleMatmul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        weight = torch.randn(
            int(KERNEL["k"]), int(KERNEL["n"]), generator=generator, dtype=DTYPE
        )
        self.register_buffer("weight", weight.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the 32x32x32 PyTorch-Chipyard smoke-test GEMM."
    )
    parser.add_argument("--compile", action="store_true", required=True)
    return parser.parse_args()


def artifact_dir() -> Path:
    configured = os.environ.get("PYTORCH_CHIPYARD_DUMP_PATH")
    if configured:
        return Path(configured).resolve()
    return (REPO_ROOT / "results" / "smoke-test" / "artifacts" / "standalone").resolve()


def configure_backend(path: Path) -> None:
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    cache_dir = Path(
        os.environ.setdefault(
            "TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/{TASK_NAME}"
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    triton.runtime.driver.set_active(ChipyardDriver())
    inductor_config.cpu_backend = "triton_chipyard"
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"
    os.environ["PYTORCH_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(path)
    os.environ["TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER"] = "1"
    os.environ["TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE"] = "0"


def import_artifact_util(path: Path):
    util_path = path / "util.py"
    if not util_path.is_file():
        raise FileNotFoundError(f"generated util.py not found: {util_path}")
    spec = importlib.util.spec_from_file_location("smoke_gemm_artifact_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parse_args()
    path = artifact_dir()
    configure_backend(path)

    model = SingleMatmul().eval()
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    inputs = torch.randn(
        int(KERNEL["m"]), int(KERNEL["k"]), generator=generator, dtype=DTYPE
    ).contiguous()

    print(f"SMOKE_KERNEL_ID={KERNEL['id']}")
    print(f"SMOKE_KERNEL_SHAPE={KERNEL['m']}x{KERNEL['n']}x{KERNEL['k']}")
    print(f"SMOKE_KERNEL_MACS={KERNEL['macs']}")
    print(f"ARTIFACT_DIR={path}")

    started = time.perf_counter()
    compiled = torch.compile(model, backend="inductor")
    with torch.inference_mode():
        compiled(inputs)
    compile_wall_s = time.perf_counter() - started

    util = import_artifact_util(path)
    input_path = util.write_inputs_bin(inputs)
    print(f"COMPILE_WALL_S={compile_wall_s:.6f}")
    print(f"INPUT_BIN={input_path}")
    print("SMOKE_COMPILE_STATUS=PASS")


if __name__ == "__main__":
    main()

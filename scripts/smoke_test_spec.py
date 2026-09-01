#!/usr/bin/env python3
"""Shared bounded-kernel definition for artifact smoke tests."""

from __future__ import annotations

from typing import Any


SMOKE_GEMM_KERNEL: dict[str, Any] = {
    "id": "smoke_gemm_32",
    "label": "32x32x32 GEMM",
    "source_model": "Synthetic smoke test",
    "source_operation": "single matrix multiplication",
    "m": 32,
    "n": 32,
    "k": 32,
    "macs": 32 * 32 * 32,
}


def smoke_gemm_kernel() -> dict[str, Any]:
    """Return a copy so callers cannot mutate the shared definition."""

    return dict(SMOKE_GEMM_KERNEL)

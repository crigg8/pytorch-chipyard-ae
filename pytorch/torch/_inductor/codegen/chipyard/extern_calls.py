from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional, TypeVar


TRITON_CHIPYARD_EXTERN_CALL_REGISTRY_ENV = (
    "TRITON_CHIPYARD_EXTERN_CALL_REGISTRY"
)

_OP_NAME_RE = re.compile(r"linalg\.[A-Za-z_][A-Za-z0-9_$.]*\Z")
_C_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ELEMENT_TYPE_RE = re.compile(r"(?:i[1-9][0-9]*|f(?:16|32|64)|bf16|index)\Z")
_UNIT_DIM_POLICIES = {"preserve", "squeeze_static"}

_F = TypeVar("_F", bound=Callable[..., Any])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class MemRefPattern:
    element_type: str
    physical_ranks: tuple[int, ...]
    logical_rank: int
    unit_dims: str = "preserve"
    canonicalize_to_logical_rank: bool = False

    def __post_init__(self) -> None:
        element_type = str(self.element_type).strip()
        if _ELEMENT_TYPE_RE.fullmatch(element_type) is None:
            raise ValueError(f"invalid MLIR element type: {self.element_type!r}")
        object.__setattr__(self, "element_type", element_type)

        if isinstance(self.physical_ranks, (str, bytes)):
            raise TypeError("physical_ranks must be a sequence of integers")
        ranks = tuple(sorted(set(self.physical_ranks)))
        if not ranks or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in ranks
        ):
            raise ValueError("physical_ranks must contain non-negative integers")
        object.__setattr__(self, "physical_ranks", ranks)

        if (
            isinstance(self.logical_rank, bool)
            or not isinstance(self.logical_rank, int)
            or self.logical_rank < 0
        ):
            raise ValueError("logical_rank must be a non-negative integer")
        if any(self.logical_rank > rank for rank in ranks):
            raise ValueError("logical_rank cannot exceed a physical rank")
        if self.unit_dims not in _UNIT_DIM_POLICIES:
            raise ValueError(
                f"unit_dims must be one of {sorted(_UNIT_DIM_POLICIES)}"
            )
        if self.unit_dims == "preserve" and any(
            rank != self.logical_rank for rank in ranks
        ):
            raise ValueError(
                "unit_dims='preserve' requires physical_ranks to equal logical_rank"
            )
        if self.canonicalize_to_logical_rank:
            if self.unit_dims != "squeeze_static":
                raise ValueError(
                    "canonicalization requires unit_dims='squeeze_static'"
                )
            if self.logical_rank == 0:
                raise ValueError("rank-0 memref canonicalization is not supported")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "memref",
            "element_type": self.element_type,
            "physical_ranks": list(self.physical_ranks),
            "logical_rank": self.logical_rank,
            "unit_dims": self.unit_dims,
            "canonicalize_to_logical_rank": (
                self.canonicalize_to_logical_rank
            ),
        }


@dataclasses.dataclass(frozen=True)
class _ExternCallRule:
    target_op: str
    operands: tuple[MemRefPattern, ...]
    function: str

    def key(self) -> tuple[str, tuple[MemRefPattern, ...]]:
        return (self.target_op, self.operands)

    def to_json(self) -> dict[str, Any]:
        return {
            "target_op": self.target_op,
            "operand_types": [operand.to_json() for operand in self.operands],
            "function": self.function,
        }


_registry: OrderedDict[
    tuple[str, tuple[MemRefPattern, ...]], _ExternCallRule
] = OrderedDict()
_library_path: Optional[Path] = None
_library_sha256: Optional[str] = None
_frozen = False


def _serialize_registry() -> str:
    if not _registry:
        return ""
    assert _library_path is not None
    assert _library_sha256 is not None
    document = {
        "version": 2,
        "library": str(_library_path),
        "library_sha256": _library_sha256,
        "patterns": [rule.to_json() for rule in _registry.values()],
    }
    return json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _publish_registry_snapshot() -> str:
    snapshot = _serialize_registry()
    os.environ[TRITON_CHIPYARD_EXTERN_CALL_REGISTRY_ENV] = snapshot
    return snapshot


def freeze_triton_chipyard_extern_call_registry() -> str:
    global _frozen
    if _frozen:
        return _publish_registry_snapshot()
    if _library_path is not None:
        actual_sha256 = _sha256_file(_library_path)
        if actual_sha256 != _library_sha256:
            raise RuntimeError(
                "Triton-Chipyard extern-call archive changed after registration: "
                f"{_library_path}"
            )
    _frozen = True
    return _publish_registry_snapshot()


def register_triton_chipyard_extern_call(
    *,
    library: str | os.PathLike[str],
    target_op: str,
    operands: Sequence[MemRefPattern],
    symbol: Optional[str] = None,
) -> Callable[[_F], _F]:
    """Register one declarative Linalg-to-external-call lowering rule.

    The decorated Python function is never executed. Its name is used as the
    external C symbol unless ``symbol`` is supplied explicitly.
    """

    normalized_target = str(target_op).strip()
    if _OP_NAME_RE.fullmatch(normalized_target) is None:
        raise ValueError(f"invalid target_op: {target_op!r}")
    if normalized_target == "linalg.generic":
        raise ValueError("linalg.generic requires a body/indexing predicate")
    normalized_operands = tuple(operands)
    if not normalized_operands or not all(
        isinstance(operand, MemRefPattern) for operand in normalized_operands
    ):
        raise TypeError("operands must be a non-empty sequence of MemRefPattern")

    archive = Path(library).expanduser().resolve()
    if archive.suffix != ".a":
        raise ValueError(
            f"extern-call library must be a static .a archive: {archive}"
        )
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_sha256 = _sha256_file(archive)

    def decorator(function: _F) -> _F:
        global _library_path, _library_sha256
        if _frozen:
            raise RuntimeError(
                "Triton-Chipyard extern-call registry is frozen; register all "
                "rules before the first torch.compile()"
            )
        function_symbol = str(symbol or function.__name__)
        if _C_SYMBOL_RE.fullmatch(function_symbol) is None:
            raise ValueError(f"invalid external C symbol: {function_symbol!r}")
        if _library_path is not None and _library_path != archive:
            raise ValueError(
                "one Triton-Chipyard registry may reference only one archive: "
                f"{_library_path} vs {archive}"
            )
        _library_path = archive
        _library_sha256 = archive_sha256

        rule = _ExternCallRule(
            target_op=normalized_target,
            operands=normalized_operands,
            function=function_symbol,
        )
        key = rule.key()
        if key in _registry:
            raise ValueError(
                "duplicate Triton-Chipyard extern-call target/signature: "
                f"{normalized_target}"
            )
        for existing in _registry.values():
            if (
                existing.function == function_symbol
                and existing.operands != rule.operands
            ):
                raise ValueError(
                    f"external C symbol {function_symbol!r} is already registered "
                    "with a different ABI"
                )
        _registry[key] = rule
        _publish_registry_snapshot()
        return function

    return decorator


def _reset_triton_chipyard_extern_call_registry_for_testing() -> None:
    global _library_path, _library_sha256, _frozen
    _registry.clear()
    _library_path = None
    _library_sha256 = None
    _frozen = False
    os.environ.pop(TRITON_CHIPYARD_EXTERN_CALL_REGISTRY_ENV, None)

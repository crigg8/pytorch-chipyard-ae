from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional


CUSTOM_LINALG_CONFIG_ENV = "TRITON_CHIPYARD_LINALG_TO_FUNC_CONFIG"
EXTERN_CALL_REGISTRY_ENV = "TRITON_CHIPYARD_EXTERN_CALL_REGISTRY"
_REQUIRED_TOP_LEVEL_KEYS = {"version", "library", "patterns"}
_OPTIONAL_TOP_LEVEL_KEYS = {"library_sha256"}
_V1_PATTERN_KEYS = {"target_op", "function"}
_V2_PATTERN_KEYS = {"target_op", "operand_types", "function"}
_OPERAND_TYPE_KEYS = {
    "kind",
    "element_type",
    "physical_ranks",
    "logical_rank",
    "unit_dims",
    "canonicalize_to_logical_rank",
}
_OP_NAME_RE = re.compile(r"linalg\.[A-Za-z_][A-Za-z0-9_$.]*\Z")
_C_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ELEMENT_TYPE_RE = re.compile(r"(?:i[1-9][0-9]*|f(?:16|32|64)|bf16|index)\Z")
_UNIT_DIM_POLICIES = {"preserve", "squeeze_static"}


@dataclasses.dataclass(frozen=True)
class CustomLinalgOperandType:
    element_type: str
    physical_ranks: tuple[int, ...]
    logical_rank: int
    unit_dims: str
    canonicalize_to_logical_rank: bool

    def cli_token(self) -> str:
        ranks = "|".join(str(rank) for rank in self.physical_ranks)
        canonicalize = "1" if self.canonicalize_to_logical_rank else "0"
        return (
            f"{self.element_type}:{ranks}:{self.logical_rank}:"
            f"{self.unit_dims}:{canonicalize}"
        )


@dataclasses.dataclass(frozen=True)
class CustomLinalgPattern:
    target_op: str
    function: str
    operand_types: tuple[CustomLinalgOperandType, ...] = ()

    def cli_operand_types(self) -> str:
        return ";".join(operand.cli_token() for operand in self.operand_types)


@dataclasses.dataclass(frozen=True)
class CustomLinalgConfig:
    config_path: str
    config_sha256: str
    library_path: str
    library_sha256: str
    patterns: tuple[CustomLinalgPattern, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, context: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _parse_document(encoded: bytes, *, source: str) -> dict[str, Any]:
    try:
        document = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid custom Linalg JSON {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"custom Linalg JSON must contain an object: {source}")
    missing = _REQUIRED_TOP_LEVEL_KEYS - document.keys()
    unknown = document.keys() - (
        _REQUIRED_TOP_LEVEL_KEYS | _OPTIONAL_TOP_LEVEL_KEYS
    )
    if missing:
        raise ValueError(f"custom Linalg config is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"custom Linalg config has unknown keys: {sorted(unknown)}")
    return document


def _parse_operand_type(
    raw_operand: Any, *, pattern_index: int, operand_index: int
) -> CustomLinalgOperandType:
    context = f"custom Linalg pattern {pattern_index} operand {operand_index}"
    if not isinstance(raw_operand, dict):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(raw_operand, _OPERAND_TYPE_KEYS, context=context)
    if raw_operand["kind"] != "memref":
        raise ValueError(f"{context} kind must be 'memref'")

    element_type = raw_operand["element_type"]
    if (
        not isinstance(element_type, str)
        or _ELEMENT_TYPE_RE.fullmatch(element_type) is None
    ):
        raise ValueError(f"{context} has invalid element_type: {element_type!r}")

    raw_ranks = raw_operand["physical_ranks"]
    if not isinstance(raw_ranks, list) or not raw_ranks:
        raise ValueError(f"{context} physical_ranks must be a non-empty list")
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in raw_ranks
    ):
        raise ValueError(
            f"{context} physical_ranks must contain non-negative integers"
        )
    physical_ranks = tuple(sorted(set(raw_ranks)))

    logical_rank = raw_operand["logical_rank"]
    if (
        isinstance(logical_rank, bool)
        or not isinstance(logical_rank, int)
        or logical_rank < 0
    ):
        raise ValueError(f"{context} logical_rank must be non-negative")
    if any(logical_rank > rank for rank in physical_ranks):
        raise ValueError(f"{context} logical_rank exceeds a physical rank")

    unit_dims = raw_operand["unit_dims"]
    if unit_dims not in _UNIT_DIM_POLICIES:
        raise ValueError(
            f"{context} unit_dims must be one of {sorted(_UNIT_DIM_POLICIES)}"
        )
    if unit_dims == "preserve" and any(
        rank != logical_rank for rank in physical_ranks
    ):
        raise ValueError(
            f"{context} preserve requires physical rank == logical rank"
        )

    canonicalize = raw_operand["canonicalize_to_logical_rank"]
    if not isinstance(canonicalize, bool):
        raise ValueError(
            f"{context} canonicalize_to_logical_rank must be boolean"
        )
    if canonicalize and unit_dims != "squeeze_static":
        raise ValueError(
            f"{context} canonicalization requires squeeze_static"
        )
    if canonicalize and logical_rank == 0:
        raise ValueError(f"{context} rank-0 canonicalization is unsupported")

    return CustomLinalgOperandType(
        element_type=element_type,
        physical_ranks=physical_ranks,
        logical_rank=logical_rank,
        unit_dims=unit_dims,
        canonicalize_to_logical_rank=canonicalize,
    )


def load_custom_linalg_config(
    config_path: Optional[str] = None,
) -> Optional[CustomLinalgConfig]:
    inline_registry = (
        os.environ.get(EXTERN_CALL_REGISTRY_ENV, "").strip()
        if config_path is None
        else ""
    )
    if inline_registry:
        encoded = inline_registry.encode("utf-8")
        source = "Python extern-call registry"
        path: Optional[Path] = None
    else:
        raw_path = (
            os.environ.get(CUSTOM_LINALG_CONFIG_ENV, "")
            if config_path is None
            else config_path
        )
        raw_path = str(raw_path).strip()
        if not raw_path:
            return None
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{CUSTOM_LINALG_CONFIG_ENV} does not name a JSON file: {path}"
            )
        encoded = path.read_bytes()
        source = str(path)

    document = _parse_document(encoded, source=source)
    version = document["version"]
    if isinstance(version, bool) or version not in (1, 2):
        raise ValueError(
            f"custom Linalg config version must be 1 or 2, got {version!r}"
        )

    raw_library = document["library"]
    if not isinstance(raw_library, str) or not raw_library.strip():
        raise ValueError("custom Linalg config library must be a non-empty string")
    library = Path(raw_library).expanduser()
    if not library.is_absolute():
        if path is None:
            raise ValueError("inline extern-call registry library must be absolute")
        library = path.parent / library
    library = library.resolve()
    if library.suffix != ".a":
        raise ValueError(
            f"custom Linalg library must be a static .a archive: {library}"
        )
    if not library.is_file():
        raise FileNotFoundError(f"custom Linalg library was not found: {library}")
    actual_library_sha256 = _sha256_file(library)
    expected_library_sha256 = document.get("library_sha256", "")
    if expected_library_sha256:
        if not isinstance(expected_library_sha256, str):
            raise ValueError("library_sha256 must be a string")
        if expected_library_sha256 != actual_library_sha256:
            raise ValueError(f"custom Linalg library digest mismatch: {library}")

    raw_patterns = document["patterns"]
    if not isinstance(raw_patterns, list) or not raw_patterns:
        raise ValueError("custom Linalg config patterns must be a non-empty list")

    patterns: list[CustomLinalgPattern] = []
    seen_patterns: set[tuple[str, tuple[CustomLinalgOperandType, ...]]] = set()
    for index, raw_pattern in enumerate(raw_patterns):
        context = f"custom Linalg pattern {index}"
        if not isinstance(raw_pattern, dict):
            raise ValueError(f"{context} must be an object")
        expected_keys = _V1_PATTERN_KEYS if version == 1 else _V2_PATTERN_KEYS
        _require_exact_keys(raw_pattern, expected_keys, context=context)
        target_op = raw_pattern["target_op"]
        function = raw_pattern["function"]
        if not isinstance(target_op, str) or _OP_NAME_RE.fullmatch(target_op) is None:
            raise ValueError(f"{context} has invalid target_op: {target_op!r}")
        if target_op == "linalg.generic":
            raise ValueError(
                "linalg.generic is not supported without an indexing/body predicate"
            )
        if not isinstance(function, str) or _C_SYMBOL_RE.fullmatch(function) is None:
            raise ValueError(f"{context} has invalid C function symbol: {function!r}")
        if version == 2:
            raw_operand_types = raw_pattern["operand_types"]
            if not isinstance(raw_operand_types, list) or not raw_operand_types:
                raise ValueError(f"{context} operand_types must be a non-empty list")
            operand_types = tuple(
                _parse_operand_type(
                    raw_operand,
                    pattern_index=index,
                    operand_index=operand_index,
                )
                for operand_index, raw_operand in enumerate(raw_operand_types)
            )
        else:
            operand_types = ()
        if version == 2 and not operand_types:
            raise ValueError(f"{context} operand_types must be non-empty")
        pattern_key = (target_op, operand_types)
        if pattern_key in seen_patterns:
            raise ValueError(
                "duplicate custom Linalg target/signature: "
                f"{target_op} at pattern {index}"
            )
        seen_patterns.add(pattern_key)
        patterns.append(
            CustomLinalgPattern(
                target_op=target_op,
                function=function,
                operand_types=operand_types,
            )
        )

    return CustomLinalgConfig(
        config_path=str(path) if path is not None else "",
        config_sha256=_sha256_bytes(encoded),
        library_path=str(library),
        library_sha256=actual_library_sha256,
        patterns=tuple(patterns),
    )

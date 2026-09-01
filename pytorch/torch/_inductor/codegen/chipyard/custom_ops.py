from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Callable
from typing import Any, Optional

import torch

from ... import config


CUSTOM_OP_LIBRARY_ENV = "PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY"


@dataclasses.dataclass(frozen=True)
class ChipyardCustomOpRegistration:
    source_op: torch._ops.OpOverload
    custom_op: torch._ops.OpOverload
    symbol: str
    supports: Optional[Callable[..., bool]]
    input_kinds: tuple[str, ...]


_registrations_by_source: dict[
    torch._ops.OpOverload, ChipyardCustomOpRegistration
] = {}
_registrations_by_custom: dict[
    torch._ops.OpOverload, ChipyardCustomOpRegistration
] = {}
_original_decompositions: dict[torch._ops.OpOverload, Any] = {}
_NO_DECOMPOSITION = object()


def _normalize_custom_op(custom_op: Any) -> torch._ops.OpOverload:
    if isinstance(custom_op, torch._ops.OpOverload):
        return custom_op
    op_overload = getattr(custom_op, "_opoverload", None)
    if isinstance(op_overload, torch._ops.OpOverload):
        return op_overload
    raise TypeError(
        "custom_op must be an OpOverload or a value returned by "
        "torch.library.custom_op"
    )


def _schema_type_kind(value_type: torch.JitType) -> Optional[str]:
    if isinstance(value_type, torch.TensorType):
        return "tensor"
    if isinstance(value_type, torch.BoolType):
        return "bool"
    if isinstance(value_type, torch.IntType):
        return "int"
    if isinstance(value_type, torch.FloatType):
        return "float"
    return None


def _validate_schema_pair(
    source_op: torch._ops.OpOverload,
    custom_op: torch._ops.OpOverload,
) -> tuple[str, ...]:
    source_schema = source_op._schema
    custom_schema = custom_op._schema
    source_args = list(source_schema.arguments)
    custom_args = list(custom_schema.arguments)
    if len(source_args) != len(custom_args):
        raise ValueError(
            f"source/custom schema argument counts differ: {source_schema} vs "
            f"{custom_schema}"
        )

    input_kinds: list[str] = []
    for source_arg, custom_arg in zip(source_args, custom_args):
        source_kind = _schema_type_kind(source_arg.real_type)
        custom_kind = _schema_type_kind(custom_arg.real_type)
        if source_kind is None or custom_kind is None or source_kind != custom_kind:
            raise ValueError(
                "Chipyard custom ops support matching Tensor/bool/int/float inputs "
                f"only: {source_schema} vs {custom_schema}"
            )
        if source_arg.alias_info is not None or custom_arg.alias_info is not None:
            raise ValueError("Chipyard custom ops must not mutate or alias their inputs")
        input_kinds.append(source_kind)

    if len(source_schema.returns) != 1 or not isinstance(
        source_schema.returns[0].real_type, torch.TensorType
    ):
        raise ValueError("Chipyard source ops must return exactly one Tensor")
    if len(custom_schema.returns) != 1 or not isinstance(
        custom_schema.returns[0].real_type, torch.TensorType
    ):
        raise ValueError("Chipyard custom ops must return exactly one Tensor")
    if custom_schema.returns[0].alias_info is not None:
        raise ValueError("Chipyard custom op outputs must not alias an input")
    return tuple(input_kinds)


def schema_order_values(
    schema: torch.FunctionSchema,
    args: tuple[Any, ...] | list[Any],
    kwargs: dict[str, Any],
) -> list[Any]:
    values: list[Any] = []
    for index, argument in enumerate(schema.arguments):
        if index < len(args):
            values.append(args[index])
        elif argument.name in kwargs:
            values.append(kwargs[argument.name])
        elif argument.has_default_value():
            values.append(argument.default_value)
        else:
            raise TypeError(f"missing value for schema argument {argument.name!r}")
    return values


def _call_custom_op(
    registration: ChipyardCustomOpRegistration,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    values = schema_order_values(registration.source_op._schema, args, kwargs)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for argument, value in zip(registration.custom_op._schema.arguments, values):
        if argument.kwarg_only:
            keyword[argument.name] = value
        else:
            positional.append(value)
    return registration.custom_op(*positional, **keyword)


def chipyard_custom_op_library_path(*, require_exists: bool = False) -> Optional[str]:
    raw_path = os.environ.get(CUSTOM_OP_LIBRARY_ENV, "").strip()
    if not raw_path:
        return None
    path = os.path.abspath(os.path.expanduser(raw_path))
    if require_exists and not os.path.isfile(path):
        raise FileNotFoundError(
            f"{CUSTOM_OP_LIBRARY_ENV} does not name a file: {path}"
        )
    if require_exists and not path.endswith(".a"):
        raise ValueError(f"{CUSTOM_OP_LIBRARY_ENV} must name a static .a archive")
    return path


def get_chipyard_custom_op_registration(
    custom_op: Any,
) -> Optional[ChipyardCustomOpRegistration]:
    try:
        op_overload = _normalize_custom_op(custom_op)
    except TypeError:
        return None
    return _registrations_by_custom.get(op_overload)


def register_chipyard_custom_op(
    source_op: torch._ops.OpOverload,
    custom_op: Any,
    *,
    symbol: str,
    supports: Optional[Callable[..., bool]] = None,
) -> None:
    """Route a supported Aten overload to a torch.library custom op.

    The route is active only for the triton_chipyard CPU backend while
    PYTORCH_CHIPYARD_CUSTOM_OP_LIBRARY names a valid static archive.
    """

    if not isinstance(source_op, torch._ops.OpOverload) or source_op.namespace != "aten":
        raise TypeError("source_op must be an Aten OpOverload")
    custom_op_overload = _normalize_custom_op(custom_op)
    if custom_op_overload.namespace == "aten":
        raise TypeError("custom_op must be registered outside the aten namespace")
    if not isinstance(symbol, str) or re.fullmatch(r"[A-Za-z_]\w*", symbol) is None:
        raise ValueError(f"invalid C symbol: {symbol!r}")
    if supports is not None and not callable(supports):
        raise TypeError("supports must be callable or None")
    if source_op in _registrations_by_source:
        raise ValueError(f"A Chipyard custom op is already registered for {source_op}")
    if custom_op_overload in _registrations_by_custom:
        raise ValueError(
            f"{custom_op_overload} is already used by a Chipyard custom-op mapping"
        )

    input_kinds = _validate_schema_pair(source_op, custom_op_overload)
    registration = ChipyardCustomOpRegistration(
        source_op=source_op,
        custom_op=custom_op_overload,
        symbol=symbol,
        supports=supports,
        input_kinds=input_kinds,
    )

    from ... import decomposition as inductor_decomposition

    original = inductor_decomposition.decompositions.get(
        source_op, _NO_DECOMPOSITION
    )
    _original_decompositions[source_op] = original
    _registrations_by_source[source_op] = registration
    _registrations_by_custom[custom_op_overload] = registration

    def route_to_chipyard_custom_op(*args: Any, **kwargs: Any) -> Any:
        active_registration = _registrations_by_source.get(source_op)
        if (
            active_registration is not None
            and config.cpu_backend == "triton_chipyard"
            and chipyard_custom_op_library_path(require_exists=True) is not None
            and (
                active_registration.supports is None
                or bool(active_registration.supports(*args, **kwargs))
            )
        ):
            return _call_custom_op(active_registration, tuple(args), kwargs)
        if original is _NO_DECOMPOSITION:
            return NotImplemented
        return original(*args, **kwargs)

    route_to_chipyard_custom_op.__name__ = (
        f"chipyard_route_{source_op.namespace}_{source_op._schema.name.replace('::', '_')}"
    )
    inductor_decomposition.decompositions[source_op] = route_to_chipyard_custom_op
    inductor_decomposition.fast_random_decomps.cache_clear()


def _reset_chipyard_custom_ops_for_testing() -> None:
    from ... import decomposition as inductor_decomposition

    for source_op, original in _original_decompositions.items():
        if original is _NO_DECOMPOSITION:
            inductor_decomposition.decompositions.pop(source_op, None)
        else:
            inductor_decomposition.decompositions[source_op] = original
    _registrations_by_source.clear()
    _registrations_by_custom.clear()
    _original_decompositions.clear()
    inductor_decomposition.fast_random_decomps.cache_clear()

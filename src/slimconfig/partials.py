# slimconfig.partials — a schema for ONE LAYER of a config, where a layer is allowed to say nothing.
#
# `partial_of(C)` is a real subclass of C whose fields all default to MISSING, so a layer is validated
# like a full run but not required to set anything. `stated(layer)` returns only what the layer set.
#
# MISSING, not None: `null` is a real value in these schemas, and merging MISSING leaves a value alone
# while merging `null` overwrites it. Optional-and-None would make every explicit `null` a no-op.

from __future__ import annotations

import dataclasses
import sys
from typing import Any, get_type_hints

from omegaconf import MISSING

from .schemas import Schema, schema_name

__all__ = ["is_partial", "partial_of", "stated"]

_MARK = "__slimconfig_partial_of__"
_cache: dict[tuple[type, str], type] = {}


# Is `cls` a layer schema, one whose fields may stay unset?
def is_partial(cls: Any) -> bool:
    # `cls.__dict__`, not getattr: a subclass of a partial is not itself one.
    return isinstance(cls, type) and dataclasses.is_dataclass(cls) and cls.__dict__.get(_MARK) is not None


def _partial(cls: type, name: str, module: str, seen: tuple[type, ...]) -> type:
    if cls in seen:  # Schema.check rejects this too, but partial_of runs at class-definition time
        raise TypeError(f"config class {schema_name(cls)} contains itself; it has no partial")
    hints = get_type_hints(cls)
    specs: list[Any] = []
    for field_name, held in Schema(cls).fields.items():
        if held.kind == "group" and held.cls is not None:
            sub = _partial(held.cls, f"{held.cls.__name__}Part", module, (*seen, cls))
            specs.append((field_name, sub, dataclasses.field(default_factory=sub)))
        else:
            # Table entries stay complete: only the class's own fields become optional.
            specs.append((field_name, hints[field_name], MISSING))
    part = dataclasses.make_dataclass(name, specs, bases=(cls,), module=module)
    setattr(part, _MARK, cls)
    return part


# Name each nested partial after where it hangs and attach it there, so `CellPart.GSSPart` resolves.
def _attach(part: type, prefix: str) -> None:
    for held in Schema(part).fields.values():
        group = held.cls
        if held.kind == "group" and is_partial(group) and "." not in group.__qualname__:
            group.__qualname__ = f"{prefix}.{group.__name__}"
            setattr(part, group.__name__, group)
            _attach(group, group.__qualname__)


# A subclass of `cls` whose every field may be unset; nested groups become partials, table entries do not.
# Cached, so repeated calls return the same class.
def partial_of(cls: type, *, name: str | None = None) -> type:
    Schema(cls)  # validates that `cls` is a config class
    key = (cls, name or "")
    if key not in _cache:
        caller = sys._getframe(1).f_globals.get("__name__", cls.__module__)
        part = _partial(cls, name or f"Partial{cls.__name__}", caller, ())
        _attach(part, part.__qualname__)
        _cache[key] = part
    return _cache[key]


# What one layer actually set, as a plain nested dict; unset fields and empty groups are omitted.
def stated(layer: Any) -> dict[str, Any]:
    cls = type(layer)
    if not (dataclasses.is_dataclass(cls) and not isinstance(layer, type)):
        raise TypeError(f"stated() takes a config instance, got {layer!r}")
    out: dict[str, Any] = {}
    for name, held in Schema(cls).fields.items():
        value = getattr(layer, name)
        if isinstance(value, str) and value == MISSING:
            continue
        if held.kind == "group" and value is not None:
            inner = stated(value)
            if inner:
                out[name] = inner
            continue
        out[name] = value
    return out

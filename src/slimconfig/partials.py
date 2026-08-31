# slimconfig.partials — a schema for ONE LAYER of a config, where a layer is allowed to say nothing.
#
# Most configs are complete: every leaf set, or the load fails. But some configs are built in layers —
# a base, then a per-model table, then a per-method table, composed at RUNTIME by code that knows which
# cell it is running (a sweep matrix, a per-environment overlay). Each layer names a handful of the run's
# fields; only the composition is a run.
#
# The wrong way to type such a layer is `Any`. It is a config class — say WHICH one:
#
#     CellPart = partial_of(RunConfig)          # every field of RunConfig, none of them required
#
#     @dataclass
#     class Matrix:
#         base: CellPart = field(default_factory=CellPart)
#         per_method: dict[Method, CellPart] = MISSING
#
# `partial_of(C)` is a real subclass of C, so it type-checks the same way, a fragment written for C (or
# for one of C's nested groups) still mounts under it, and OmegaConf validates every key and every value
# in a layer exactly as it would in a finished run. The one thing it does not do is REQUIRE: a field the
# layer leaves out stays MISSING instead of failing the load.
#
# `stated(layer)` then reads back only what that layer actually said, as a plain dict — which is what a
# resolver merges, in whatever order it defines, before loading the result as the complete class.
#
# WHY MISSING AND NOT None. A layer that says nothing must be distinguishable from a layer that says
# `null`, because `null` is a real value in these schemas (a disabled command, a task with no subject).
# MISSING is the sentinel OmegaConf already treats as "this layer is silent": merging it over a value
# leaves the value alone, while merging `null` overwrites it. Typing the fields Optional-and-None would
# turn every explicit `null` in a layer into a silent no-op.

from __future__ import annotations

import dataclasses
import sys
from typing import Any, get_type_hints

from omegaconf import MISSING

from .schemas import fields_of, schema_name

__all__ = ["is_partial", "partial_of", "stated"]

_MARK = "__slimconfig_partial_of__"
_cache: dict[tuple[type, str], type] = {}


# Is `cls` a layer schema — one whose fields are allowed to stay unset?
def is_partial(cls: Any) -> bool:
    # `cls.__dict__` deliberately, not getattr: a subclass of a partial is not itself one.
    return isinstance(cls, type) and dataclasses.is_dataclass(cls) and cls.__dict__.get(_MARK) is not None


def _partial(cls: type, name: str, module: str, seen: tuple[type, ...]) -> type:
    if cls in seen:  # check_schema rejects this too, but partial_of runs at class-definition time
        raise TypeError(f"config class {schema_name(cls)} contains itself; it has no partial")
    hints = get_type_hints(cls)
    specs: list[Any] = []
    for field_name, (kind, group) in fields_of(cls).items():
        if kind == "group" and group is not None:
            sub = _partial(group, f"{group.__name__}Part", module, (*seen, cls))
            specs.append((field_name, sub, dataclasses.field(default_factory=sub)))
        else:
            # A table's ENTRIES stay complete: a layer either names an entry or it does not, and an entry
            # it names describes one whole thing. Only the class's own fields are made optional here.
            specs.append((field_name, hints[field_name], MISSING))
    part = dataclasses.make_dataclass(name, specs, bases=(cls,), module=module)
    setattr(part, _MARK, cls)
    return part


# Give every nested partial a name that says where it hangs, and hang it there — so `CellPart.GSSPart`
# both reads right in an error message and resolves as a dotted path.
def _attach(part: type, prefix: str) -> None:
    for kind, group in fields_of(part).values():
        if kind == "group" and group is not None and is_partial(group) and "." not in group.__qualname__:
            group.__qualname__ = f"{prefix}.{group.__name__}"
            setattr(part, group.__name__, group)
            _attach(group, group.__qualname__)


# A subclass of `cls` whose every field may be left unset — the schema of ONE LAYER of `cls`. Nested
# groups become partials too (a layer may set one field of a group and say nothing about the rest);
# table entries do not (an entry a layer names is a whole entry). Called twice with the same arguments it
# returns the same class, so `issubclass` and OmegaConf's structured cache both behave.
def partial_of(cls: type, *, name: str | None = None) -> type:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a config class: partial_of takes a @dataclass")
    key = (cls, name or "")
    if key not in _cache:
        caller = sys._getframe(1).f_globals.get("__name__", cls.__module__)
        part = _partial(cls, name or f"Partial{cls.__name__}", caller, ())
        _attach(part, part.__qualname__)
        _cache[key] = part
    return _cache[key]


# What one layer actually says, as a plain dict — nothing for a field it left unset, and a nested dict
# for a group, itself holding only the fields that group's layer set. This is what a resolver merges.
def stated(layer: Any) -> dict[str, Any]:
    cls = type(layer)
    if not (dataclasses.is_dataclass(cls) and not isinstance(layer, type)):
        raise TypeError(f"stated() takes a config instance, got {layer!r}")
    out: dict[str, Any] = {}
    for name, (kind, group) in fields_of(cls).items():
        value = getattr(layer, name)
        if isinstance(value, str) and value == MISSING:
            continue
        if kind == "group" and group is not None and value is not None:
            inner = stated(value)
            if inner:
                out[name] = inner
            continue
        out[name] = value
    return out

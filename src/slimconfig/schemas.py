# slimconfig.schemas — the config-class layer: what a schema may look like, and how a YAML names one.
#
# A schema is a @dataclass, and a dataclass field is one of two things: a LEAF (a scalar, a list, a
# dict — a value) or a NESTED SCHEMA (another dataclass — a group of values with a name). Nothing else
# is a config. That is the whole shape rule, and check_schema enforces it before a load, so a schema
# that cannot be filled from YAML says so at import time rather than deep inside an OmegaConf merge.
#
# GROUPS ARE COMPOSED, NOT INHERITED. A config class that needs another one's fields declares a field
# of that type; it does not inherit it as a mixin. Inheriting flattens the borrowed fields into the
# parent's own namespace, so the YAML cannot say where a value came from and two mixins can silently
# collide on a name. A nested field gives the group a name in the class AND the same name in the YAML,
# which is what makes a hierarchical config readable — and what lets a shared fragment be mounted at
# exactly one place (see the `defaults:` rule in config.py).
#
# NAMING A CLASS FROM YAML. Every config file states the class it fills, by dotted import path:
#     _schema: myproject.train.OptimConfig
# resolve_schema imports it; field_schema says which class belongs at a given node of a parent schema;
# and load_config checks the two agree. A renamed or moved class therefore breaks its configs loudly,
# which is the point — a config file is written against a class, and that dependency should be visible.

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

__all__ = ["check_schema", "field_schema", "resolve_schema", "schema_name"]


# THE SCRIPT THAT WAS LAUNCHED IS `__main__`. A config class defined in the entry point itself lives in a
# module named `__main__`, which is not a name a config file can use — and importing that script again
# under its real name (`train`) would RUN it a second time and hand back a different class object, so
# every `issubclass` against it would fail. The two names are therefore treated as one: `schema_name`
# writes the script's own name, and `resolve_schema` hands back the already-running module for it.
def _main_names() -> tuple[str, ...]:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)  # set by `python -m pkg.mod`
    file = getattr(main, "__file__", None)  # set by `python path/to/train.py`
    return tuple(n for n in (getattr(spec, "name", None), Path(file).stem if file else None) if n)


def _import_module(name: str) -> Any:
    if name in _main_names():
        return sys.modules["__main__"]
    return importlib.import_module(name)


# The dotted path a YAML names `cls` by — the inverse of resolve_schema, used in every error message so
# a mismatch can be fixed by copying the name out of it.
def schema_name(cls: type) -> str:
    module = cls.__module__
    if module == "__main__":
        module = next(iter(_main_names()), module)
    return f"{module}.{cls.__qualname__}"


# The dataclass a field's annotation refers to, or None if the field is a leaf. `X | None` counts: an
# optional group is still a group, and OmegaConf nests it the same way.
def _nested_of(annotation: Any) -> type | None:
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return annotation
    # Optional[X] / X | None arrive as a Union; a union of anything else is not a group.
    import types

    if get_origin(annotation) in (Union, types.UnionType):
        nested = [a for a in get_args(annotation) if a is not type(None)]
        if len(nested) == 1:
            return _nested_of(nested[0])
    return None


# `cls`'s fields as {name: nested dataclass or None}, with string annotations resolved.
def _fields_of(cls: type) -> dict[str, type | None]:
    hints = get_type_hints(cls)
    return {f.name: _nested_of(hints.get(f.name, f.type)) for f in dataclasses.fields(cls)}


# Reject a schema that cannot be filled from a YAML file, naming the field that breaks it. Two rules
# beyond "it is a dataclass":
#   * a nested schema field must carry `field(default_factory=<its class>)`. Without it the field has no
#     value to merge onto and OmegaConf reports a missing-default error against the class, not the field.
#   * the nesting must terminate — a schema that (transitively) contains itself has no finite YAML.
# Leaves are not type-checked here: OmegaConf owns what a value may be, and it reports those precisely.
def check_schema(cls: type, _seen: tuple[type, ...] = ()) -> None:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a config class: a schema is a @dataclass")
    if cls in _seen:
        chain = " -> ".join(schema_name(c) for c in (*_seen, cls))
        raise TypeError(f"config class {schema_name(cls)} contains itself: {chain}")
    nested = _fields_of(cls)
    for f in dataclasses.fields(cls):
        group = nested[f.name]
        if group is None:
            continue
        if f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            raise TypeError(
                f"{schema_name(cls)}.{f.name} is a nested config class ({schema_name(group)}) and must "
                f"declare it as its default: `{f.name}: {group.__name__} = field(default_factory={group.__name__})`"
            )
        check_schema(group, (*_seen, cls))


# Import the config class a `_schema:` line names. The dotted path is split at the last import that
# succeeds, so both `pkg.module.Class` and `pkg.module.Outer.Inner` resolve.
def resolve_schema(dotted: str) -> type:
    if not isinstance(dotted, str) or not dotted.strip():
        raise ValueError(f"`_schema` must be the dotted import path of a config class, got {dotted!r}")
    parts = dotted.strip().split(".")
    if len(parts) < 2:
        raise ValueError(
            f"`_schema: {dotted}` is not a dotted import path — name the class in full, "
            "e.g. `_schema: myproject.train.OptimConfig`"
        )
    for split in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split])
        try:
            obj: Any = _import_module(module_path)
        except ModuleNotFoundError as e:
            if e.name == module_path or (e.name and module_path.startswith(e.name + ".")):
                continue  # not a module — try a shorter prefix
            raise  # the module exists but its own imports are broken: that is not our error to hide
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None:
            continue
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            raise ValueError(f"`_schema: {dotted}` names {obj!r}, which is not a config class (@dataclass)")
        return obj
    raise ValueError(f"`_schema: {dotted}` could not be imported — no such module or attribute")


# The config class that belongs at `node` (a dotted path of field names) inside `root`; `root` itself for
# the empty path. Raises if the path does not lead to a nested config class, which is what a `defaults:`
# under a leaf field looks like.
def field_schema(root: type, node: str) -> type:
    current = root
    walked: list[str] = []
    for key in [k for k in node.split(".") if k]:
        nested = _fields_of(current)
        if key not in nested:
            where = f"{schema_name(root)}.{'.'.join(walked)}" if walked else schema_name(root)
            raise ValueError(f"{where} has no field {key!r}")
        group = nested[key]
        walked.append(key)
        if group is None:
            raise ValueError(
                f"{schema_name(root)}.{'.'.join(walked)} is a value, not a nested config class — "
                "only a group can be composed from a config file"
            )
        current = group
    return current

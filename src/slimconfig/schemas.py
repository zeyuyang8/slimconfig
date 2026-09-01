# slimconfig.schemas — the config-class layer: what a schema may look like, and how a YAML names one.
#
# A schema is a @dataclass, and a dataclass field is one of three things:
#
#   a LEAF     a scalar, a list, a dict of values — a value.
#   a GROUP    another dataclass: a named group of values, nested.
#   a TABLE    `dict[str, C]` (or `dict[SomeEnum, C]`) for a config class C: SEVERAL of that group,
#              keyed — one entry per model, per method, per whatever the keys name. The entries are
#              validated exactly like a group is (unknown key rejected, types checked), and an Enum key
#              type means the KEYS are checked too.
#
# Nothing else is a config. That is the whole shape rule, and `Schema.check` enforces it before a load,
# so a schema that cannot be filled from YAML says so at import time rather than deep inside a merge.
#
# ONE OBJECT FOR A CONFIG CLASS. Everything the rest of the package asks of a schema — its name, its
# fields, whether it is well formed, which class sits at a given node of it — is a method of `Schema`,
# so those questions are asked the same way everywhere instead of through four unrelated free functions
# each re-deriving the same walk. The free functions below are kept as the module's stable surface and
# are one line each; `Schema` is where the answers live.
#
# ONE THING TO KNOW ABOUT MERGING A LEAF THAT IS A MAPPING. Every mapping merges key by key — that is
# OmegaConf's rule and slimconfig does not change it — so a later spec setting `weights: {a: 1, b: 1}`
# on top of `{a: 1, c: 1}` yields all three keys, not two. For a GROUP or a TABLE that is exactly right.
# For a leaf that happens to be a `dict[str, float]` it is usually not what the writer meant: such a
# layer can ADD a key but never DROP one. If a mapping-valued leaf is a set of things and a config needs
# to state a different set, do not layer it — give each variant its own whole value, at a node where
# only one variant can apply.
#
# GROUPS ARE COMPOSED, NOT INHERITED. A config class that needs another one's fields declares a field
# of that type; it does not inherit it as a mixin. Inheriting flattens the borrowed fields into the
# parent's own namespace, so the YAML cannot say where a value came from and two mixins can silently
# collide on a name. A nested field gives the group a name in the class AND the same name in the YAML,
# which is what makes a hierarchical config readable — and what lets a shared fragment be mounted at
# exactly one place (see the `_default:` rule in config.py).
#
# NAMING A CLASS FROM YAML. Every config file states the class it fills, by dotted import path:
#     _schema: myproject.train.OptimConfig
# `Schema.resolve` imports it; `Schema.require` says which class belongs at a given node of a parent
# schema; and load_config checks the two agree. A renamed or moved class therefore breaks its configs
# loudly, which is the point — a config file is written against a class, and that dependency should be
# visible.

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

__all__ = [
    "Field",
    "Node",
    "Placement",
    "Schema",
    "check_schema",
    "field_schema",
    "fields_of",
    "placement",
    "resolve_schema",
    "schema_name",
]

# WHERE A NODE IS, SPELLED AS THE KEYS AND NOT AS ONE STRING. A dotted path is the convenient way to
# WRITE one by hand (`optim.lr`), but it cannot say what a key with a dot in it is: `overrides.model.
# flux.1-dev` is four keys or five depending on the table's keys, and only whoever walked the mapping
# knows which. So compose() reports a claim's node as the tuple of keys it actually descended, and
# anything that resolves a node takes either — a string is split on ".", a sequence is taken as given.
type Node = str | Sequence[str]


class Field(NamedTuple):
    """What one dataclass field HOLDS: a value, one nested config class, or a keyed table of one."""

    kind: Literal["value", "group", "table"]
    cls: type | None  # the nested config class, for a group or a table; None for a leaf


class Placement(NamedTuple):
    """What a node path LANDS ON inside a schema — a finer question than what a field holds, since a
    table and one of its entries are the same field:

        ("group", C)      a nested config-class field: one C, and the YAML block filling it must name it
        ("entry", C)      one entry of a table of C: also a C, but which class was fixed by the table
        ("table", C)      the table itself: however many C the keys name, and no single class to fill
        ("value", None)   a leaf — a scalar, a list, or a dict of plain values
        ("unknown", None) no such path in this schema (an unknown key, or one below a leaf)
    """

    kind: Literal["group", "entry", "table", "value", "unknown"]
    cls: type | None


# THE SCRIPT THAT WAS LAUNCHED IS `__main__`. A config class defined in the entry point itself lives in a
# module named `__main__`, which is not a name a config file can use — and importing that script again
# under its real name (`train`) would RUN it a second time and hand back a different class object, so
# every `issubclass` against it would fail. The two names are therefore treated as one: `schema_name`
# writes the script's own name, and `Schema.resolve` hands back the already-running module for it.
def _main_names() -> tuple[str, ...]:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)  # set by `python -m pkg.mod`
    file = getattr(main, "__file__", None)  # set by `python path/to/train.py`
    return tuple(n for n in (getattr(spec, "name", None), Path(file).stem if file else None) if n)


def _import_module(name: str) -> Any:
    if name in _main_names():
        return sys.modules["__main__"]
    return importlib.import_module(name)


# The dotted path a YAML names `cls` by — the inverse of Schema.resolve, used in every error message so
# a mismatch can be fixed by copying the name out of it.
def schema_name(cls: type) -> str:
    module = cls.__module__
    if module == "__main__":
        module = next(iter(_main_names()), module)
    return f"{module}.{cls.__qualname__}"


# What an annotation describes — a value, a nested config class, or a table of one. `X | None` counts as
# whatever X is: an optional group is still a group, and OmegaConf nests it the same way.
def _field_of(annotation: Any) -> Field:
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return Field("group", annotation)
    origin = get_origin(annotation)
    if origin is dict:
        args = get_args(annotation)
        value = args[1] if len(args) == 2 else None
        if dataclasses.is_dataclass(value) and isinstance(value, type):
            return Field("table", value)
        return Field("value", None)  # a dict of plain values is a leaf
    # Optional[X] / X | None arrive as a Union; a union of anything else is not a group.
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _field_of(inner[0])
    return Field("value", None)


# `cls`'s fields as {name: Field}, with string annotations resolved.
def fields_of(cls: type) -> dict[str, Field]:
    hints = get_type_hints(cls)
    return {f.name: _field_of(hints.get(f.name, f.type)) for f in dataclasses.fields(cls)}


# The keys of a node, however it was spelled.
def _keys(node: Node) -> list[str]:
    return [k for k in node.split(".") if k] if isinstance(node, str) else list(node)


@dataclasses.dataclass(frozen=True, slots=True)
class Schema:
    """A config class, and every question the loader asks of one.

        Schema(TrainConfig).fields              -> {"model": Field("value", None), "optim": Field(...)}
        Schema(TrainConfig).check()             -> raises unless the class can be filled from a YAML file
        Schema(TrainConfig).at("optim.lr")      -> Placement("value", None)
        Schema(TrainConfig).require("optim")    -> Schema(Optim)
        Schema.resolve("myproject.train.Optim") -> Schema(Optim)

    A Schema wraps a class and holds nothing else, so two of the same class are equal and either may be
    built wherever it is wanted; nothing is cached that a reloaded module could make stale.
    """

    cls: type

    def __post_init__(self) -> None:
        if not (isinstance(self.cls, type) and dataclasses.is_dataclass(self.cls)):
            raise TypeError(f"{self.cls!r} is not a config class: a schema is a @dataclass")

    # ── naming ───────────────────────────────────────────────────────────────

    # Import the config class a `_schema:` line names. The dotted path is split at the last import that
    # succeeds, so both `pkg.module.Class` and `pkg.module.Outer.Inner` resolve.
    @classmethod
    def resolve(cls, dotted: str) -> Schema:
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
            return cls(obj)
        raise ValueError(f"`_schema: {dotted}` could not be imported — no such module or attribute")

    @property
    def name(self) -> str:
        return schema_name(self.cls)

    # ── fields ───────────────────────────────────────────────────────────────

    @property
    def fields(self) -> dict[str, Field]:
        return fields_of(self.cls)

    # Reject a schema that cannot be filled from a YAML file, naming the field that breaks it. Two rules
    # beyond "it is a dataclass" (which the constructor checked):
    #   * a GROUP field must carry `field(default_factory=<its class>)`. Without it the field has no value
    #     to merge onto and OmegaConf reports a missing-default error against the class, not the field. A
    #     TABLE needs no such default — its entries do not exist until a config file names their keys.
    #   * the nesting must terminate — a schema that (transitively) contains itself has no finite YAML.
    # Leaves are not type-checked here: OmegaConf owns what a value may be, and it reports those precisely.
    def check(self, _seen: tuple[type, ...] = ()) -> None:
        if self.cls in _seen:
            chain = " -> ".join(schema_name(c) for c in (*_seen, self.cls))
            raise TypeError(f"config class {self.name} contains itself: {chain}")
        shapes = self.fields
        for f in dataclasses.fields(self.cls):
            kind, nested = shapes[f.name]
            if nested is None:
                continue
            if kind == "group" and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
                raise TypeError(
                    f"{self.name}.{f.name} is a nested config class ({schema_name(nested)}) and must "
                    f"declare it as its default: `{f.name}: {nested.__name__} = field(default_factory={nested.__name__})`"
                )
            Schema(nested).check((*_seen, self.cls))

    # ── nodes ────────────────────────────────────────────────────────────────

    # Step a node through this schema one key at a time, saying what each prefix lands on. A table is
    # stepped through by naming one of its entries — `overrides.task.dreambooth` lands on the table's
    # value class — because an entry is a group and the table itself is not: it has no class of its own
    # to fill, only however many the keys name. The walk stops at the first key it cannot place; nothing
    # below a leaf or an unknown key is a node of this schema.
    def walk(self, node: Node) -> Iterator[tuple[tuple[str, ...], Placement]]:
        current: Schema | None = self
        entry_of: type | None = None  # set when the last key landed on a table: its entries' class
        walked: list[str] = []
        for key in _keys(node):
            walked.append(key)
            path = tuple(walked)
            if entry_of is not None:  # this key is a table KEY, not a field name
                current, entry_of = Schema(entry_of), None
                yield path, Placement("entry", current.cls)
                continue
            shapes = current.fields if current is not None else {}
            if key not in shapes:
                yield path, Placement("unknown", None)
                return
            kind, nested = shapes[key]
            if nested is None:
                current = None
                yield path, Placement("value", None)
            elif kind == "table":
                entry_of = nested
                yield path, Placement("table", nested)
            else:
                current = Schema(nested)
                yield path, Placement("group", nested)

    # What `node` lands on — see Placement. This schema itself for the empty path. The non-raising half
    # of `require`, for a caller that wants to ASK rather than require.
    def at(self, node: Node) -> Placement:
        last = Placement("group", self.cls)
        for _, where in self.walk(node):
            last = where
        return last

    # The config class that belongs at `node`; this schema itself for the empty path. Raises if the path
    # does not land on a config class, which is what a `_default:` under a leaf field looks like.
    def require(self, node: Node) -> Schema:
        current, kind = self, "group"
        walked: tuple[str, ...] = ()
        for path, (kind, cls) in self.walk(node):
            walked = path
            if kind == "unknown":
                where = ".".join((self.name, *path[:-1]))
                raise ValueError(f"{where} has no field {path[-1]!r}")
            if kind == "value":
                raise ValueError(
                    f"{self.name}.{'.'.join(path)} is a value, not a nested config class — "
                    "only a group can be composed from a config file"
                )
            current = Schema(cls) if cls is not None else current
        if kind == "table":
            where = ".".join(walked)
            raise ValueError(
                f"{self.name}.{where} is a table of {current.name}, not a config "
                f"class — name one entry (`{where}.<key>`), since the table itself has no class to fill"
            )
        return current

    def __repr__(self) -> str:
        return f"Schema({self.name})"


# The free functions the rest of the package and its callers use. Each is one Schema call: the class is
# where the logic lives, these are the names it has always been reached by.


def check_schema(cls: type, _seen: tuple[type, ...] = ()) -> None:
    Schema(cls).check(_seen)


def resolve_schema(dotted: str) -> type:
    return Schema.resolve(dotted).cls


def placement(root: type, node: Node) -> Placement:
    return Schema(root).at(node)


def field_schema(root: type, node: Node) -> type:
    return Schema(root).require(node).cls

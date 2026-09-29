# slimconfig.schemas — config classes: what a schema may look like, and how a YAML names one.
#
# A schema is a @dataclass subclassing `Config`; each field is one of three shapes:
#   a LEAF   a value: a scalar, a list, or a dict of values.
#   a GROUP  another config class, nested.
#   a TABLE  `dict[SomeEnum, C]` for a config class C: several C, one per Enum member.
# Every config class (root, group, table entry) subclasses `Config`, and is checked at its `class` statement.
# A value is a str / int / float / bool / Enum, a union of those, a list or dict of any of it, or `| None`.
# A mapping is keyed by an Enum, never a bare `str`; in YAML a key is written as the member's value.
# A YAML names the class of each mapping it fills:
#     optim > pkg.mod.OptimConfig:             this mapping IS an OptimConfig
#     data > dict[pkg.mod.Task, pkg.mod.Data]:  its entries each are a Data, keyed by Task
#     _ > pkg.mod.TrainConfig:                 the file's own class, on the reserved key `_`

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import re
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

__all__ = [
    "Config",
    "Declaration",
    "Node",
    "Schema",
    "Shape",
    "check_declaration",
    "declaration_name",
    "key_name",
    "optional",
    "schema_name",
    "value_error",
]

# A path into a schema: a dotted string, or the keys themselves. A sequence is needed because a table key
# may contain a dot (`flux.1-dev`), which a dotted string cannot tell apart from nesting.
type Node = str | Sequence[str]


class Shape(NamedTuple):
    """What is at one place in a schema. A field is one of the first three; a walk can also land on the
    last two:

        ("value", None)   a leaf — a scalar, a list, or a dict of plain values
        ("group", C)      one nested config class: the YAML block filling it must name it
        ("table", C, K)   the table itself: however many C the keys name, keyed by K
        ("entry", C)      one entry of such a table: also a C, but which class was fixed by the table
        ("unknown", None) no such path in this schema (an unknown key, or one below a leaf)
    """

    kind: Literal["value", "group", "table", "entry", "unknown"]
    cls: type | None  # the config class here; None for a leaf or an unknown path
    key: type | None = None  # what the KEYS are, for a table; None for everything else


# `dict[<the key type>, <dotted path to the entry class>]` — a table, as a YAML spells one.
_TABLE = re.compile(r"dict\[\s*(?P<key>[\w.]+)\s*,\s*(?P<value>[\w.]+)\s*\]")


class Declaration(NamedTuple):
    """What one declaration says the mapping under it is:

        optim > pkg.module.Optim:            Declaration(Schema(Optim), None)   — this mapping IS one
        data > dict[pkg.mod.Task, ...Data]:  Declaration(Schema(Data), Task)    — its ENTRIES each are
    """

    schema: Schema
    key: type | None  # the type a table's keys have; None for a group


# The names the launched script goes by. A class defined there lives in `__main__`, and re-importing the
# script under its real name would run it again and yield a different class, so the two are treated as one.
def _main_names() -> tuple[str, ...]:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)  # set by `python -m pkg.mod`
    file = getattr(main, "__file__", None)  # set by `python path/to/train.py`
    return tuple(n for n in (getattr(spec, "name", None), Path(file).stem if file else None) if n)


def _import_module(name: str) -> Any:
    if name in _main_names():
        return sys.modules["__main__"]
    return importlib.import_module(name)


# The object a dotted path names, or None; handles nested classes (`pkg.module.Outer.Inner`).
def _import_dotted(dotted: str) -> Any:
    parts = dotted.split(".")
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
        if obj is not None:
            return obj
    return None


# The dotted path a YAML names `cls` by — the inverse of `Schema.resolve`.
def schema_name(cls: type) -> str:
    module = cls.__module__
    if module == "__main__":
        module = next(iter(_main_names()), module)
    return f"{module}.{cls.__qualname__}"


# How a YAML declares `cls`: its dotted path for a group, `dict[K, path]` for a table of them.
def declaration_name(cls: type, key: type | None = None) -> str:
    return schema_name(cls) if key is None else f"dict[{key_name(key)}, {schema_name(cls)}]"


# How a `dict[...]` names the type its keys have: the Enum's dotted path.
def key_name(key: type) -> str:
    return schema_name(key)


# Import the key type of a `dict[...]` declaration, which must be an Enum.
def _resolve_key(dotted: str, spelled: str) -> type:
    obj = _import_dotted(dotted)
    if _is_enum(obj):
        return obj
    raise ValueError(
        f"`{spelled}` does not say what the keys are: a table is keyed by an Enum named in full, as its "
        f"entry class is (`dict[pkg.module.TaskName, pkg.module.Cell]`), and {dotted!r} is not one"
    )


# `X | None` -> X; anything else is returned unchanged.
def optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return inner[0]
    return annotation


# The Shape an annotation describes; `X | None` counts as whatever X is.
def _shape_of(annotation: Any) -> Shape:
    annotation = optional(annotation)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return Shape("group", annotation)
    if get_origin(annotation) is dict:
        args = get_args(annotation)
        value = args[1] if len(args) == 2 else None
        if dataclasses.is_dataclass(value) and isinstance(value, type):
            return Shape("table", value, args[0])
    return Shape("value", None)  # a dict of plain values is a leaf


# The keys of a node, however it was spelled.
def _keys(node: Node) -> list[str]:
    return [k for k in node.split(".") if k] if isinstance(node, str) else list(node)


# One key on from `here`: under a table the key names an entry, otherwise a field.
def _step(here: Shape, key: str) -> Shape:
    if here.kind == "table":
        return Shape("entry", here.cls)
    if here.cls is None:
        return Shape("unknown", None)
    return Schema(here.cls).fields.get(key, Shape("unknown", None))


# ── what a field may be declared as ──────────────────────────────────────────

_SCALARS: tuple[type, ...] = (str, int, float, bool)


def _is_enum(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, enum.Enum)


def _shown(annotation: Any) -> str:
    """An annotation spelled roughly as written."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


# Why `annotation` is not a config value type, or None if it is.
def _value_error(annotation: Any) -> str | None:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _value_error(inner[0])
        held = next((a for a in inner if not (a in _SCALARS or _is_enum(a))), None)
        if held is not None:  # OmegaConf rejects a union holding a container
            return (
                f"a union may only offer scalars — one of a str, an int, a float, a bool or an Enum — "
                f"and {_shown(held)} is not one"
            )
        return None
    if annotation in _SCALARS or _is_enum(annotation):
        return None
    if annotation is Any or annotation is object:
        return "it says nothing about the value — name the type the config actually holds"
    if annotation is list or annotation is dict:
        example = "list[str]" if annotation is list else "dict[pkg.module.SomeEnum, float]"
        return f"it does not say what it holds — write `{example}`"
    if origin is list:
        held = get_args(annotation)[0]
        if dataclasses.is_dataclass(held):
            return (
                f"a list of config classes is not one of the three shapes — key them instead, as a "
                f"table: `dict[pkg.module.SomeEnum, {_shown(held)}]`"
            )
        return _value_error(held)
    if origin is dict:
        key, held = get_args(annotation)
        if not _is_enum(key):
            return (
                f"a mapping is keyed by an Enum, not by {_shown(key)} — every key a config file writes "
                f"is registered somewhere, a field of a class or a member of an Enum, and a key typed "
                f"{_shown(key)} is a word from nowhere that nothing can check"
            )
        if dataclasses.is_dataclass(held):
            return (
                f"a table of {_shown(held)} is a FIELD of a config class, not something nested inside "
                "another container — give it a field of its own"
            )
        return _value_error(held)
    if annotation in (tuple, set, frozenset) or origin in (tuple, set, frozenset):
        held = ", ".join(_shown(a) for a in get_args(annotation)[:1]) or "str"
        return f"it is not a type OmegaConf holds — a sequence in a config is a `list[{held}]`"
    if origin is Literal:
        return "OmegaConf cannot check a Literal — declare an Enum and let it name the choices"
    if annotation is Path:
        return (
            "a Path does not survive the run snapshot (it is written back as a python object tag) — "
            "declare it `str` and turn it into a path in code (slimconfig.resolve_path)"
        )
    if dataclasses.is_dataclass(annotation):
        return "it is a config class, so it is a group or a table — this position holds a value"
    return (
        "it is not a type a YAML value can have — a config value is a str, int, float, bool or Enum, "
        "a list or dict of those, or any of that `| None`"
    )


# Why `value` does not match `annotation`, or None. OmegaConf does not check container contents, so this
# does. Returns a suffix for the caller to prefix with the field path, e.g. `tags[1] is not a str: {'a': 1}`.
def value_error(value: Any, annotation: Any) -> str | None:
    ann = optional(annotation)
    if value is None:  # whether null is allowed at all is OmegaConf's own check, at merge time
        return None
    origin = get_origin(ann)
    if origin in (Union, types.UnionType):  # a union of scalars: any member it matches will do
        held = [a for a in get_args(ann) if a is not type(None)]
        if any(value_error(value, a) is None for a in held):
            return None
        return f" is not one of {' | '.join(_shown(a) for a in held)}: {value!r}"
    if origin is list:
        if not isinstance(value, list):
            return f" is not a list: {value!r}"
        held = get_args(ann)[0]
        return next((f"[{i}]{p}" for i, v in enumerate(value) if (p := value_error(v, held))), None)
    if origin is dict:
        if not isinstance(value, dict):
            return f" is not a mapping: {value!r}"
        held = get_args(ann)[1]
        shown = {k: repr(k.value if isinstance(k, enum.Enum) else k) for k in value}
        return next((f"[{shown[k]}]{p}" for k, v in value.items() if (p := value_error(v, held))), None)
    if _is_enum(ann):
        return None if isinstance(value, ann) else f" is not a {ann.__name__}: {value!r}"
    if ann is bool:
        held_ok = isinstance(value, bool)
    elif ann in (int, float):  # a whole number is a fine float; a bool is not either one
        held_ok = isinstance(value, int if ann is int else int | float) and not isinstance(value, bool)
    elif ann is str:
        held_ok = isinstance(value, str)
    else:
        return None  # not a shape this rule knows — `check_declaration` has already rejected it
    return None if held_ok else f" is not a {ann.__name__}: {value!r}"


# A field's declared default, before or after `@dataclass` has run (both expose `.default_factory`).
def _default_of(cls: type, name: str) -> Any:
    fields = cls.__dict__.get("__dataclass_fields__")
    if fields is not None and name in fields:
        return fields[name]
    return cls.__dict__.get(name, dataclasses.MISSING)


def _has_factory(default: Any) -> bool:
    return isinstance(default, dataclasses.Field) and default.default_factory is not dataclasses.MISSING


def _subclasses_config(cls: Any) -> bool:
    return isinstance(cls, type) and issubclass(cls, Config)


# Why one field cannot be filled from a YAML file — None if it can.
def _declaration_error(owner: str, name: str, annotation: Any, default: Any) -> str | None:
    kind, nested, key = _shape_of(annotation)
    where = f"{owner}.{name}"
    if nested is not None and not _subclasses_config(nested):
        return (
            f"{where} holds the config class {nested.__name__}, which does not subclass "
            f"slimconfig.Config — every config class does, and that is what makes it one: "
            f"`class {nested.__name__}(Config):`"
        )
    if kind == "group":
        if not _has_factory(default):
            return (
                f"{where} is a nested config class ({schema_name(nested)}) and must declare it as its "
                f"default: `{name}: {nested.__name__} = field(default_factory={nested.__name__})`"
            )
        return None
    if kind == "table":
        if not _is_enum(key):
            return (
                f"{where} is a table keyed by {_shown(key)} — a key names one of several groups, and "
                f"every key a config file writes is registered somewhere, so it is an Enum whose members "
                f"are the entries this table may have"
            )
        return None
    problem = _value_error(annotation)
    return f"{where} is typed `{_shown(annotation)}`: {problem}" if problem else None


# Check every field `cls` itself declares, raising TypeError for the first that breaks a rule.
def check_declaration(cls: type) -> None:
    hints = get_type_hints(cls)
    owner = schema_name(cls)
    for name in inspect.get_annotations(cls):  # this class's OWN annotations, not its bases'
        annotation = hints.get(name)
        if get_origin(annotation) is ClassVar:  # a constant on the class, not a field of the config
            continue
        problem = _declaration_error(owner, name, annotation, _default_of(cls, name))
        if problem is not None:
            raise TypeError(problem)


# Classes whose annotations forward-reference a later name; retried at the next `Config` subclass, and
# always by `Schema.check` before a load.
_deferred: list[type] = []


def _settle() -> None:
    for cls in list(_deferred):
        try:
            check_declaration(cls)
        except NameError:
            continue
        _deferred.remove(cls)


class Config:
    """The base class of every config class — the root, every nested group, every table entry.

        @dataclass
        class Optim(Config):
            lr: float = MISSING

    It has no fields; subclassing it checks the class's declarations at the `class` statement.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _settle()  # the classes declared before this one are complete by now
        try:
            check_declaration(cls)
        except NameError:  # an annotation naming something declared further down the module
            _deferred.append(cls)


@dataclasses.dataclass(frozen=True, slots=True)
class Schema:
    """A config class, and every question the loader asks of one.

        Schema(TrainConfig).fields               -> {"model": Shape("value", None), "optim": Shape(...)}
        Schema(TrainConfig).check()              -> raises unless the class can be filled from a YAML file
        Schema(TrainConfig).at("optim.lr")       -> Shape("value", None)
        Schema(TrainConfig).require("optim")     -> Schema(Optim)
        Schema.resolve("myproject.train.Optim")  -> Schema(Optim)
        Schema.declared("dict[..Task, ..Data]")  -> Declaration(Schema(Data), Task)

    Holds only the class and caches nothing.
    """

    cls: type

    def __post_init__(self) -> None:
        if not _subclasses_config(self.cls):
            raise TypeError(
                f"{self.cls!r} is not a config class: a config class subclasses slimconfig.Config"
            )
        if not dataclasses.is_dataclass(self.cls):
            raise TypeError(
                f"{schema_name(self.cls)} subclasses Config but is not a @dataclass: a config class is "
                "both — the base says it is filled from YAML, the decorator gives it its fields"
            )

    # ── naming ───────────────────────────────────────────────────────────────

    # Import the config class a declaration names.
    @classmethod
    def resolve(cls, dotted: str) -> Schema:
        if not isinstance(dotted, str) or not dotted.strip():
            raise ValueError(f"a declaration must be the dotted import path of a config class, got {dotted!r}")
        dotted = dotted.strip()
        if len(dotted.split(".")) < 2:
            raise ValueError(
                f"`{dotted}` is not a dotted import path — name the class in full, "
                "e.g. `optim > myproject.train.OptimConfig:`"
            )
        obj = _import_dotted(dotted)
        if obj is None:
            raise ValueError(f"`{dotted}` could not be imported — no such module or attribute")
        if not (dataclasses.is_dataclass(obj) and _subclasses_config(obj)):
            raise ValueError(
                f"`{dotted}` names {obj!r}, which is not a config class "
                "(a @dataclass subclassing slimconfig.Config)"
            )
        return cls(obj)

    @property
    def name(self) -> str:
        return schema_name(self.cls)

    # Parse a whole declaration: the class it names and, for a table, the key type.
    @classmethod
    def declared(cls, text: str) -> Declaration:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"a declaration must be the dotted import path of a config class, got {text!r}")
        spelled = text.strip()
        table = _TABLE.fullmatch(spelled)
        if table is not None:
            return Declaration(cls.resolve(table["value"]), _resolve_key(table["key"], spelled))
        if "[" in spelled:
            listed = (
                " A list of config classes is not one of the three shapes — key them, as a table."
                if spelled.startswith("list[") else ""
            )
            raise ValueError(
                f"`{spelled}` is neither a config class nor a table of one — write "
                f"`pkg.module.Class` for a mapping that IS one, or `dict[pkg.module.SomeEnum, "
                f"pkg.module.Class]` for a "
                f"table whose every entry is one.{listed}"
            )
        return Declaration(cls.resolve(spelled), None)

    # ── fields ───────────────────────────────────────────────────────────────

    # The Shape of each field, {name: Shape}.
    @property
    def fields(self) -> dict[str, Shape]:
        hints = get_type_hints(self.cls)
        return {f.name: _shape_of(hints.get(f.name, f.type)) for f in dataclasses.fields(self.cls)}

    # The resolved type hints, for when the exact type matters and not just the Shape.
    @property
    def hints(self) -> dict[str, Any]:
        return get_type_hints(self.cls)

    # Check this class and every class reachable from it, including any whose check was deferred, and
    # reject a schema that contains itself.
    def check(self, _seen: tuple[type, ...] = ()) -> None:
        if self.cls in _seen:
            chain = " -> ".join(schema_name(c) for c in (*_seen, self.cls))
            raise TypeError(f"config class {self.name} contains itself: {chain}")
        try:
            check_declaration(self.cls)
        except NameError as e:
            raise TypeError(f"config class {self.name} has an annotation that names nothing: {e}") from e
        for held in self.fields.values():
            if held.cls is not None:
                Schema(held.cls).check((*_seen, self.cls))

    # ── nodes ────────────────────────────────────────────────────────────────

    # Yield (prefix, Shape) for each key of `node`, stopping at the first unknown one.
    def walk(self, node: Node) -> Iterator[tuple[tuple[str, ...], Shape]]:
        here, walked = Shape("group", self.cls), []
        for key in _keys(node):
            walked.append(key)
            here = _step(here, key)
            yield tuple(walked), here
            if here.kind == "unknown":
                return

    # The Shape at `node` (this schema for the empty path); the non-raising version of `require`.
    def at(self, node: Node) -> Shape:
        here = Shape("group", self.cls)
        for _, here in self.walk(node):
            pass  # the last step walked is the answer
        return here

    # The config class at `node` (this schema for the empty path); raises unless it is a group or entry.
    def require(self, node: Node) -> Schema:
        here, walked = Shape("group", self.cls), ()
        for walked, here in self.walk(node):  # the LAST step walked is the answer
            if here.kind == "unknown":
                raise ValueError(f"{'.'.join((self.name, *walked[:-1]))} has no field {walked[-1]!r}")
            if here.kind == "value":
                raise ValueError(
                    f"{self.name}.{'.'.join(walked)} is a value, not a nested config class — "
                    "only a group can be composed from a config file"
                )
        if here.kind == "table":
            raise ValueError(
                f"{self.name}.{'.'.join(walked)} is a table of {schema_name(here.cls)}, not one of it: "
                f"the table itself has no class to fill, so name the shape "
                f"(`{declaration_name(here.cls, here.key)}`) or one entry (`{'.'.join(walked)}.<key>`)"
            )
        return Schema(here.cls)

    def __repr__(self) -> str:
        return f"Schema({self.name})"

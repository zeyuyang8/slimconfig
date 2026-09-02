# slimconfig.schemas — the config-class layer: what a schema may look like, and how a YAML names one.
#
# A schema is a @dataclass subclassing `Config`, and a dataclass field is one of three things:
#
#   a LEAF     a scalar, a list, a dict of values — a value.
#   a GROUP    another config class: a named group of values, nested.
#   a TABLE    `dict[str, C]` (or `dict[SomeEnum, C]`) for a config class C: SEVERAL of that group,
#              keyed — one entry per model, per method, per whatever the keys name. The entries are
#              validated exactly like a group is (unknown key rejected, types checked), and an Enum key
#              type means the KEYS are checked too.
#
# Nothing else is a config. That is the whole shape rule, and `Schema.check` enforces it before a load,
# so a schema that cannot be filled from YAML says so at import time rather than deep inside a merge.
#
# ONE BASE CLASS, AND IT IS NOT OPTIONAL. Every config class subclasses `Config` — the root, every
# nested group, every table's entry class. Being a config class is then something a class SAYS rather
# than something a loader infers from its shape, which buys the one thing inference cannot: the rules
# below run in `__init_subclass__`, at the `class` statement, so a schema that cannot be filled from
# YAML fails when its module is imported and not at the first launch that happens to load it.
#
# WHAT A FIELD MAY BE DECLARED AS. A type hint on a config class is a promise to whoever reads the
# YAML, so it may only say things that are kept: a value is a str / int / float / bool / Enum, a union
# of those, a list or dict of any of that, and any of it `| None`. `Any`, a bare `list`, a `tuple`
# (which comes back a list), a `set` or `Literal` (which OmegaConf cannot hold at all), a `Path` (which
# cannot round-trip through a run snapshot), a union holding a container (which OmegaConf rejects
# outright) — each is rejected by name, at the class, with what to write instead.
#
# ONE OBJECT FOR A CONFIG CLASS. Everything the rest of the package asks of a schema — its name, its
# fields, whether it is well formed, which class sits at a given node of it — is a method of `Schema`,
# so those questions are asked the same way everywhere instead of through free functions each
# re-deriving the same walk. What is left at module level is what is not about one class: the naming
# rules (`schema_name` and the `_schema:` line it writes), and the declaration rules, which run at the
# `class` statement, before any Schema exists.
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
# NAMING A CLASS FROM YAML. Every config file, and every mapping in one that fills a config class,
# states which class — and, because a group and a table are the same mapping shape on the page, the
# declaration says which of the two it is:
#     _schema: myproject.train.OptimConfig             this mapping IS an OptimConfig
#     _schema: dict[str, myproject.train.Data]         its ENTRIES each are a Data, keyed by str
#     _schema: dict[myproject.tasks.Task, ...Data]     the same, keyed by an Enum — named in full too
# The second spelling is the field's own annotation, written out, so a table names its entry class once
# for all of its entries instead of once per entry — and an entry, whose class the table already fixed,
# names nothing. `Schema.declared` reads a line; `Schema.resolve` imports the class in it;
# `Schema.require` says which class belongs at a given node of a parent schema; and load_config checks
# they agree. A renamed or moved class therefore breaks its configs loudly, which is the point — a
# config file is written against a class, and that dependency should be visible.

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
    "schema_name",
    "value_error",
]

# WHERE A NODE IS, SPELLED AS THE KEYS AND NOT AS ONE STRING. A dotted path is the convenient way to
# WRITE one by hand (`optim.lr`), but it cannot say what a key with a dot in it is: `overrides.model.
# flux.1-dev` is four keys or five depending on the table's keys, and only whoever walked the mapping
# knows which. So compose() reports a claim's node as the tuple of keys it actually descended, and
# anything that resolves a node takes either — a string is split on ".", a sequence is taken as given.
type Node = str | Sequence[str]


class Shape(NamedTuple):
    """What is at one place in a schema. A FIELD holds one of the first three; walking a node can also
    land on the last two, since a table and one of its entries are the same field:

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
    """What one `_schema:` line SAYS the mapping under it is:

        _schema: pkg.module.Optim            Declaration(Schema(Optim), None)   — this mapping IS one
        _schema: dict[str, pkg.module.Data]  Declaration(Schema(Data), str)     — its ENTRIES each are
    """

    schema: Schema
    key: type | None  # the type a table's keys have; None for a group


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


# The object a dotted path names, or None if nothing of that name can be imported. The path is split at
# the last import that succeeds, so both `pkg.module.Class` and `pkg.module.Outer.Inner` resolve.
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


# The dotted path a YAML names `cls` by — the inverse of Schema.resolve, used in every error message so
# a mismatch can be fixed by copying the name out of it.
def schema_name(cls: type) -> str:
    module = cls.__module__
    if module == "__main__":
        module = next(iter(_main_names()), module)
    return f"{module}.{cls.__qualname__}"


# How a YAML declares a mapping of `cls`: the class's dotted path for ONE of it, `dict[K, path]` for a
# table of them. The two spellings are what tell a block filling a group apart from a table whose every
# entry fills one — the same mapping shape, and without the `dict[...]` a reader cannot say which they
# are looking at. Both names in a `dict[...]` are import paths, `str` aside: a key type spelled as a bare
# word would be the one name in a config file that names nothing a reader can open.
def declaration_name(cls: type, key: type | None = None) -> str:
    return schema_name(cls) if key is None else f"dict[{key_name(key)}, {schema_name(cls)}]"


# How a `dict[...]` names the type its keys have: `str` as itself, an Enum by its dotted path.
def key_name(key: type) -> str:
    return "str" if key is str else schema_name(key)


# The other half of a `dict[...]`: the type a table's keys have, imported the same way the entry class
# beside it is. `str` is itself; anything else is an Enum named in full. Resolving it rather than
# comparing spellings is what makes the declaration checkable AND followable — the check is then whether
# the file named the same TYPE the field declared, and a reader can open the name to see the keys.
def _resolve_key(dotted: str, spelled: str) -> type:
    obj = str if dotted == "str" else _import_dotted(dotted)
    if isinstance(obj, type) and (obj is str or issubclass(obj, enum.Enum)):
        return obj
    raise ValueError(
        f"`_schema: {spelled}` does not say what the keys are: a table is keyed by `str` or by an Enum "
        f"named in full, as its entry class is (`dict[pkg.module.TaskName, pkg.module.Cell]`), and "
        f"{dotted!r} is neither"
    )


# `X | None` -> X. Anything else — including a union of several real types — is handed back as it is,
# for the caller to classify or to reject.
def _optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return inner[0]
    return annotation


# What an annotation describes — a value, a nested config class, or a table of one. `X | None` counts as
# whatever X is: an optional group is still a group, and OmegaConf nests it the same way.
def _shape_of(annotation: Any) -> Shape:
    annotation = _optional(annotation)
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


# One key on from `here`: under a table the key names an ENTRY, whose class the table already fixed;
# under a config class it names a field. Nothing is below a leaf or below a key that placed nowhere.
def _step(here: Shape, key: str) -> Shape:
    if here.kind == "table":
        return Shape("entry", here.cls)
    if here.cls is None:
        return Shape("unknown", None)
    return Schema(here.cls).fields.get(key, Shape("unknown", None))


# ── what a field may be declared as ──────────────────────────────────────────

# What a YAML scalar can BE, and OmegaConf can then hold to the declaration.
_SCALARS: tuple[type, ...] = (str, int, float, bool)


def _is_enum(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, enum.Enum)


def _shown(annotation: Any) -> str:
    """An annotation spelled as the reader wrote it, near enough to find in the file."""
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


# Why `annotation` is not a config VALUE — None if it is one. A value is a str / int / float / bool /
# Enum, a list or dict of those (nested as deep as it likes), and any of that `| None`. Everything else
# is rejected here, at the class, rather than at the first config that trips over it — or, worse, not
# at all: OmegaConf lets a `list[str]` hold a mapping, and `Any` lets anything hold anything.
def _value_error(annotation: Any) -> str | None:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _value_error(inner[0])
        held = next((a for a in inner if not (a in _SCALARS or _is_enum(a))), None)
        if held is not None:  # OmegaConf holds a union of scalars and says so; a container it cannot
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
        example = "list[str]" if annotation is list else "dict[str, float]"
        return f"it does not say what it holds — write `{example}`"
    if origin is list:
        held = get_args(annotation)[0]
        if dataclasses.is_dataclass(held):
            return (
                f"a list of config classes is not one of the three shapes — key them instead, as a "
                f"table: `dict[str, {_shown(held)}]`"
            )
        return _value_error(held)
    if origin is dict:
        key, held = get_args(annotation)
        if not (key is str or key is int or _is_enum(key)):
            return f"a config key is a str, an int or an Enum, not {_shown(key)}"
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


# Why `value` is not what `annotation` promised — None if it is. The other half of the rule above: a
# declaration is worth what is checked against it, and OmegaConf checks a scalar but will happily let a
# `list[str]` hold a mapping and a `dict[str, str]` hold a list. The message is a SUFFIX, so a caller
# that knows the field's name can print the whole path to the value that is wrong: `tags[1] is not a
# str: {'a': 1}`.
def value_error(value: Any, annotation: Any) -> str | None:
    ann = _optional(annotation)
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
        return next((f"[{k!r}]{p}" for k, v in value.items() if (p := value_error(v, held))), None)
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


# The default a field was declared with, whichever side of the `@dataclass` decorator we are on: what
# the class body wrote (`field(default_factory=Optim)`) before it runs, and the dataclasses.Field it
# became after — both answer `.default_factory`, which is the only thing asked of it.
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
        if not (key is str or _is_enum(key)):
            return (
                f"{where} is a table keyed by {_shown(key)} — a key names one of several groups, so it "
                "is a str or an Enum"
            )
        return None
    problem = _value_error(annotation)
    return f"{where} is typed `{_shown(annotation)}`: {problem}" if problem else None


# Hold every field `cls` declares to the rules above, and raise TypeError naming the first that breaks
# one. Only the class's OWN annotations: whatever it inherited was checked where it was written.
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


# A class whose annotations named something not yet defined — a group declared above the class it holds,
# say. Checking it is postponed to the next `class ... (Config)` statement, by when the name it waited
# for normally exists; anything still deferred is checked by Schema.check before a load, so a class is
# never let through unchecked, only checked late.
_deferred: list[type] = []


def _settle() -> None:
    for cls in list(_deferred):
        try:
            check_declaration(cls)
        except NameError:
            continue
        _deferred.remove(cls)


class Config:
    """The base class of every config class — the root of a schema, every nested group, every table
    entry.

        @dataclass
        class Optim(Config):
            lr: float = MISSING

    It carries no fields and no behaviour: what it is for is to make "this class is filled from YAML"
    something a class SAYS, so the rules that come with saying it can run at the `class` statement.
    Each field's type hint is checked against what a config value may be (see `_value_error`), each
    nested group against the `default_factory` it needs, and each group and table entry against this
    same base — so a schema that cannot be filled fails at the import of the module that declares it,
    naming the field, instead of at the first launch that loads one.
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
        Schema.declared("dict[str, ....Data]")   -> Declaration(Schema(Data), str)

    A Schema wraps a class and holds nothing else, so two of the same class are equal and either may be
    built wherever it is wanted; nothing is cached that a reloaded module could make stale.
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

    # Import the config class a `_schema:` line names.
    @classmethod
    def resolve(cls, dotted: str) -> Schema:
        if not isinstance(dotted, str) or not dotted.strip():
            raise ValueError(f"`_schema` must be the dotted import path of a config class, got {dotted!r}")
        dotted = dotted.strip()
        if len(dotted.split(".")) < 2:
            raise ValueError(
                f"`_schema: {dotted}` is not a dotted import path — name the class in full, "
                "e.g. `_schema: myproject.train.OptimConfig`"
            )
        obj = _import_dotted(dotted)
        if obj is None:
            raise ValueError(f"`_schema: {dotted}` could not be imported — no such module or attribute")
        if not (dataclasses.is_dataclass(obj) and _subclasses_config(obj)):
            raise ValueError(
                f"`_schema: {dotted}` names {obj!r}, which is not a config class "
                "(a @dataclass subclassing slimconfig.Config)"
            )
        return cls(obj)

    @property
    def name(self) -> str:
        return schema_name(self.cls)

    # Read one `_schema:` line: the class it names, and — if it spells a table — what it says the keys
    # are. `resolve` answers the first half and is what a group needs; this answers the whole line.
    @classmethod
    def declared(cls, text: str) -> Declaration:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"`_schema` must be the dotted import path of a config class, got {text!r}")
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
                f"`_schema: {spelled}` is neither a config class nor a table of one — write "
                f"`pkg.module.Class` for a mapping that IS one, or `dict[str, pkg.module.Class]` for a "
                f"table whose every entry is one.{listed}"
            )
        return Declaration(cls.resolve(spelled), None)

    # ── fields ───────────────────────────────────────────────────────────────

    # What each of this class's fields holds, {name: Shape}, with string annotations resolved.
    @property
    def fields(self) -> dict[str, Shape]:
        hints = get_type_hints(self.cls)
        return {f.name: _shape_of(hints.get(f.name, f.type)) for f in dataclasses.fields(self.cls)}

    # The annotations themselves, for the one question `fields` cannot answer: not what SHAPE a field
    # holds but exactly which type it promised — `list[str]` and `dict[str, float]` are both leaves.
    @property
    def hints(self) -> dict[str, Any]:
        return get_type_hints(self.cls)

    # Reject a schema that cannot be filled from a YAML file, naming the field that breaks it. Every
    # class reachable from this one is held to the rules its own `class` statement ran
    # (`check_declaration`: each field's type hint is a type a config value may have, each group carries
    # `field(default_factory=<its class>)`, each group and entry class subclasses Config) — re-run here
    # so a schema is checked in full before a load even if a forward reference postponed it, and so one
    # `check()` is a complete answer on its own. Plus the one rule that is not about a single class: the
    # nesting must terminate, since a schema that contains itself has no finite YAML.
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

    # Step a node through this schema one key at a time, saying what each prefix lands on. A table is
    # stepped through by naming one of its entries — `overrides.task.dreambooth` lands on the table's
    # value class — because an entry is a group and the table itself is not: it has no class of its own
    # to fill, only however many the keys name. The walk stops at the first key it cannot place; nothing
    # below a leaf or an unknown key is a node of this schema.
    def walk(self, node: Node) -> Iterator[tuple[tuple[str, ...], Shape]]:
        here, walked = Shape("group", self.cls), []
        for key in _keys(node):
            walked.append(key)
            here = _step(here, key)
            yield tuple(walked), here
            if here.kind == "unknown":
                return

    # What `node` lands on — see Shape. This schema itself for the empty path. The non-raising half of
    # `require`, for a caller that wants to ASK rather than require.
    def at(self, node: Node) -> Shape:
        here = Shape("group", self.cls)
        for _, here in self.walk(node):
            pass  # the last step walked is the answer
        return here

    # The config class that belongs at `node`; this schema itself for the empty path. Raises if the path
    # does not land on a config class, which is what a `_default:` under a leaf field looks like.
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

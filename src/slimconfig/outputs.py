# slimconfig.outputs — typed results, the counterpart of typed configs: a run's results as classes,
# written as JSON and read back into the same classes, checked field by field. A run's layout (see
# slimconfig.layouts) names which outputs it writes.
#
#     @dataclass
#     class Score(Output):
#         accuracy: Annotated[float, "The share of examples answered right"]
#         per_class: dict[str, float]
#
# An output class is a @dataclass subclassing `Output`. Its fields may be str, int, float, bool, None,
# an Enum (written as its value), a nested output class, `list[T]`, `dict[K, T]` (K is str or an Enum,
# a key written as its value), `T | None`, a union of output classes, or `Any` (any JSON value, passed
# through unchecked). Every field is required. `Annotated[T, "text"]` is a T that `output_schema`
# describes as "text".
#
# A union of output classes, `A | B` (or `A | B | None`), holds an instance of one of them, so its
# classes have distinct names; it is written tagged by its class's name, a mapping of that one key:
#
#     @dataclass
#     class Passed(Output):
#         seconds: float
#
#     @dataclass
#     class Failed(Output):
#         error: str
#
#     @dataclass
#     class Result(Output):
#         outcome: Passed | Failed    # {"outcome": {"Failed": {"error": "..."}}}
#
# The same classes type any JSON value, not only a file: `build_output` checks one into its class,
# `plain_output` turns one back into JSON, and `output_schema` is the class as a JSON Schema.

from __future__ import annotations

import dataclasses
import enum
import json
import types
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

__all__ = ["Output", "build_output", "load_output", "output_schema", "plain_output", "write_output"]


class Output:
    """Base class of a run's typed output; subclasses are dataclasses (see the module header)."""


# Check that `cls` is an output class (a @dataclass subclassing Output).
def output_class(owner: str, where: str, cls: Any) -> type:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls) and issubclass(cls, Output)):
        raise TypeError(
            f"{owner}'s {where} must be an output class (a @dataclass subclassing slimconfig.Output), "
            f"got {cls!r}"
        )
    return cls


# Write `output` to the JSON file `path`, Enums as their values; returns the path.
def write_output(path: str, output: Output) -> str:
    plain = plain_output(output)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain, f, indent=2)
        f.write("\n")
    return path


# Read the JSON file `path` back into `cls`, checking every value against its field's type. The output
# remembers `path` (see `source_of`).
def load_output[T](path: str, cls: type[T]) -> T:
    output_class(cls.__qualname__, "type", cls)
    with open(path, encoding="utf-8") as f:
        output = build_output(json.load(f), cls)
    object.__setattr__(output, _SOURCE, path)  # Output has no __slots__, so even a frozen one takes it
    return output


_SOURCE = "__slimconfig_source__"


# The path `output` was read from by `load_output`: how a config snapshot names an input.
def source_of(output: Output) -> str:
    source = output.__dict__.get(_SOURCE)
    if source is None:
        raise ValueError(
            f"this {type(output).__qualname__} was not read from a file, so a config cannot name it: "
            "a config's input field holds an output loaded from the JSON file its YAML names"
        )
    return source


# Check the JSON value `value` into `cls`: every field present, no other key, each of its type.
def build_output[T](value: Any, cls: type[T]) -> T:
    output_class(cls.__qualname__, "type", cls)
    return _build(value, cls, cls.__name__)


# `output` as a JSON value: a mapping of its fields, Enums as their values, each value of a union of
# output classes tagged by its class's name.
def plain_output(output: Output) -> dict[str, Any]:
    output_class(type(output).__qualname__, "type", type(output))
    return _dump(output, type(output), type(output).__name__)


# The JSON Schema every value `build_output` accepts into `cls` matches.
def output_schema(cls: type) -> dict[str, Any]:
    output_class(cls.__qualname__, "type", cls)
    return _schema(cls, cls.__name__)


# `value`, of type `ann`, as JSON; `where` names it in errors.
def _dump(value: Any, ann: Any, where: str) -> Any:
    if get_origin(ann) is Annotated:
        ann = get_args(ann)[0]
    origin, args = get_origin(ann), get_args(ann)
    if value is None or ann is Any:
        return _plain(value)
    if origin in (Union, types.UnionType):
        classes = _tagged(ann, where)
        if classes is None:
            return _dump(value, next(a for a in args if a is not type(None)), where)
        name = type(value).__name__
        if classes.get(name) is not type(value):
            raise TypeError(f"{where} holds a {name}, none of the union {ann}")
        return {name: _dump(value, type(value), f"{where}.{name}")}
    if isinstance(ann, type) and dataclasses.is_dataclass(ann):
        hints = get_type_hints(type(value))
        return {
            f.name: _dump(getattr(value, f.name), hints[f.name], f"{where}.{f.name}")
            for f in dataclasses.fields(value)
        }
    if origin is list:
        return [_dump(v, args[0], f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        return {_plain(k): _dump(v, args[1], f"{where}[{k}]") for k, v in value.items()}
    return _plain(value)


# The classes of a union of output classes, by name; None for `T | None`, which holds one type.
def _tagged(ann: Any, where: str) -> dict[str, type] | None:
    options = [a for a in get_args(ann) if a is not type(None)]
    if len(options) == 1:
        return None
    if not all(isinstance(a, type) and dataclasses.is_dataclass(a) for a in options):
        raise TypeError(
            f"{where}: an output field may be `T | None` or a union of output classes, not the union {ann}"
        )
    classes = {a.__name__: a for a in options}
    if len(classes) != len(options):
        raise TypeError(f"{where}: the union {ann} holds two classes of one name, so its tag cannot tell them")
    return classes


def _plain(node: Any) -> Any:
    if dataclasses.is_dataclass(node) and not isinstance(node, type):  # one in an `Any` field
        return _plain(dataclasses.asdict(node))
    if isinstance(node, enum.Enum):
        return node.value
    if isinstance(node, dict):
        return {_plain(key): _plain(value) for key, value in node.items()}
    if isinstance(node, list | tuple):
        return [_plain(value) for value in node]
    return node


def _is_enum(ann: Any) -> bool:
    return isinstance(ann, type) and issubclass(ann, enum.Enum)


# A mapping's key type: str, or an Enum whose values are the keys.
def _key(ann: Any, where: str) -> Any:
    if ann is not str and not _is_enum(ann):
        raise TypeError(f"{where}: an output mapping is keyed by str or an Enum, since JSON keys are strings")
    return ann


# Build a value of type `ann` from its JSON form; `where` names it in errors (`Score.per_class[cat]`).
def _build(value: Any, ann: Any, where: str) -> Any:
    origin, args = get_origin(ann), get_args(ann)
    if ann is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        classes = _tagged(ann, where)
        if classes is None:
            return _build(value, next(a for a in args if a is not type(None)), where)
        if not (isinstance(value, dict) and len(value) == 1 and next(iter(value)) in classes):
            raise ValueError(
                f"{where} is not one of {', '.join(classes)}, a mapping of its class's name: {value!r}"
            )
        [(name, held)] = value.items()
        return _build(held, classes[name], f"{where}.{name}")
    if isinstance(ann, type) and dataclasses.is_dataclass(ann):
        if not isinstance(value, dict):
            raise ValueError(f"{where} is not a {ann.__name__} mapping: {value!r}")
        hints = get_type_hints(ann)
        names = {f.name for f in dataclasses.fields(ann)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise ValueError(f"{where} has key(s) {ann.__name__} does not declare: {', '.join(unknown)}")
        missing = sorted(n for n in names if n not in value)
        if missing:
            raise ValueError(f"{where} is missing field(s): {', '.join(missing)}")
        return ann(**{n: _build(value[n], hints[n], f"{where}.{n}") for n in names})
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where} is not a list: {value!r}")
        return [_build(v, args[0], f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        key = _key(args[0], where)
        if not isinstance(value, dict):
            raise ValueError(f"{where} is not a mapping: {value!r}")
        return {
            _build(k, key, f"{where} key {k!r}"): _build(v, args[1], f"{where}[{k}]") for k, v in value.items()
        }
    if _is_enum(ann):
        try:
            return ann(value)
        except ValueError:
            raise ValueError(f"{where}: {value!r} is not a valid {ann.__name__}") from None
    if ann is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if ann in (str, int, float, bool, type(None)):
        if type(value) is not ann:
            raise ValueError(f"{where} is not a {ann.__name__}: {value!r}")
        return value
    raise TypeError(f"{where}: an output field cannot be typed {ann!r}")


_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


# The JSON Schema of type `ann`, which may carry `Annotated` descriptions (see the module header).
def _schema(ann: Any, where: str) -> dict[str, Any]:
    if get_origin(ann) is Annotated:
        held, *notes = get_args(ann)
        described = [note for note in notes if isinstance(note, str)]
        if len(described) != len(notes) or len(described) > 1:
            raise TypeError(f"{where}: an output field is annotated with one description, a str")
        return {**_schema(held, where), "description": described[0]}
    origin, args = get_origin(ann), get_args(ann)
    if ann is Any:
        return {}
    if origin in (Union, types.UnionType):
        classes = _tagged(ann, where)
        if classes is None:
            return {"anyOf": [_schema(next(a for a in args if a is not type(None)), where), {"type": "null"}]}
        one = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {name: _schema(cls, f"{where}.{name}")},
                    "required": [name],
                    "additionalProperties": False,
                }
                for name, cls in classes.items()
            ]
        }
        return {"anyOf": [one, {"type": "null"}]} if type(None) in args else one
    if isinstance(ann, type) and dataclasses.is_dataclass(ann):
        hints = get_type_hints(ann, include_extras=True)
        names = [f.name for f in dataclasses.fields(ann)]
        return {
            "type": "object",
            "properties": {n: _schema(hints[n], f"{where}.{n}") for n in names},
            "required": names,
            "additionalProperties": False,
        }
    if origin is list:
        return {"type": "array", "items": _schema(args[0], f"{where}[]")}
    if origin is dict:
        key = _key(args[0], where)
        schema: dict[str, Any] = {"type": "object", "additionalProperties": _schema(args[1], f"{where}[]")}
        if key is not str:
            schema["propertyNames"] = _schema(key, f"{where} key")
        return schema
    if _is_enum(ann):
        return {"enum": [member.value for member in ann]}
    if ann in _SCALARS:
        return {"type": _SCALARS[ann]}
    raise TypeError(f"{where}: an output field cannot be typed {ann!r}")

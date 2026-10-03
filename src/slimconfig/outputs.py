# slimconfig.outputs — typed results, the counterpart of typed configs: a run's results as classes,
# written as JSON and read back into the same classes, checked field by field. A run's layout (see
# slimconfig.layouts) names which outputs it writes.
#
#     @dataclass
#     class Score(Output):
#         accuracy: float
#         per_class: dict[str, float]
#
# An output class is a @dataclass subclassing `Output`. Its fields may be str, int, float, bool, None,
# an Enum (written as its value), a nested output class, `list[T]`, `dict[str, T]`, `T | None`, or `Any`
# (any JSON value, passed through unchecked).

from __future__ import annotations

import dataclasses
import enum
import json
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

__all__ = ["Output", "load_output", "write_output"]


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
    output_class(type(output).__qualname__, "type", type(output))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_plain(dataclasses.asdict(output)), f, indent=2)
        f.write("\n")
    return path


# Read the JSON file `path` back into `cls`, checking every value against its field's type.
def load_output[T](path: str, cls: type[T]) -> T:
    output_class(cls.__qualname__, "type", cls)
    with open(path, encoding="utf-8") as f:
        return _build(json.load(f), cls, cls.__name__)


def _plain(node: Any) -> Any:
    if isinstance(node, enum.Enum):
        return node.value
    if isinstance(node, dict):
        return {key: _plain(value) for key, value in node.items()}
    if isinstance(node, list | tuple):
        return [_plain(value) for value in node]
    return node


# Build a value of type `ann` from its JSON form; `where` names it in errors (`Score.per_class[cat]`).
def _build(value: Any, ann: Any, where: str) -> Any:
    origin, args = get_origin(ann), get_args(ann)
    if ann is Any:
        return value
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        if value is None and len(options) < len(args):
            return None
        if len(options) != 1:
            raise TypeError(f"{where}: an output field may be `T | None`, not the union {ann}")
        return _build(value, options[0], where)
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
        if args[0] is not str:
            raise TypeError(f"{where}: an output mapping is keyed by str, since JSON keys are")
        if not isinstance(value, dict):
            raise ValueError(f"{where} is not a mapping: {value!r}")
        return {k: _build(v, args[1], f"{where}[{k}]") for k, v in value.items()}
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return ann(value)
    if ann is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if ann in (str, int, float, bool, type(None)):
        if type(value) is not ann:
            raise ValueError(f"{where} is not a {ann.__name__}: {value!r}")
        return value
    raise TypeError(f"{where}: an output field cannot be typed {ann!r}")

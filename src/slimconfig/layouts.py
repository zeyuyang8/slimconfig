# slimconfig.layouts — what a run folder holds, declared once per kind of run and checked on every run.
#
#     @dataclass
#     class ChatLayout(Layout):
#         messages: Trajectory      # an output class: written as messages.json
#         workspace: Folder         # a folder: created before main, its path handed over
#
#     class Chat(Run):
#         config: ChatConfig
#         layout: ChatLayout
#
#         def main(self) -> None:
#             self.layout.messages = chat(self.config, self.layout.workspace)
#
#     run = load_layout("runs/exp1", ChatLayout)   # run.messages is a Trajectory, run.workspace a Path
#
# Every field of a layout class is either an output class (see slimconfig.outputs) or `Folder`. Before
# main, the run creates each folder under its run folder and sets the layout's folder fields to their
# paths; main sets every output field. After main, the run writes each output as `<field>.json`, then
# checks that the folder holds only its own files (config.yaml, metadata.json, run.log) and what the
# layout declares: anything else is an error naming it.

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, NewType, get_type_hints

from .outputs import Output, load_output, write_output

__all__ = ["Folder", "Layout", "load_layout"]

Folder = NewType("Folder", Path)  # a layout field that is a folder of the run, not a JSON output
RUN_FILES = ("config.yaml", "metadata.json", "run.log")  # what every run folder holds


class Layout:
    """Base class of a run folder's declared contents; subclasses are dataclasses (see the header)."""


# The layout class's fields, each as ("folder", None) or ("output", its output class).
def parts(owner: str, cls: Any) -> dict[str, tuple[str, type | None]]:
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls) and issubclass(cls, Layout)):
        raise TypeError(
            f"{owner}'s `layout` must be a layout class (a @dataclass subclassing slimconfig.Layout), "
            f"got {cls!r}"
        )
    hints = get_type_hints(cls)
    out: dict[str, tuple[str, type | None]] = {}
    for field in dataclasses.fields(cls):
        ann = hints[field.name]
        if ann is Folder:
            out[field.name] = ("folder", None)
        elif isinstance(ann, type) and dataclasses.is_dataclass(ann) and issubclass(ann, Output):
            out[field.name] = ("output", ann)
        else:
            raise TypeError(
                f"{cls.__qualname__}.{field.name} is typed {ann!r}: a layout field is an output "
                "class or Folder"
            )
    return out


# Entry names in the run folder: `<name>/` for a folder, `<name>.json` for an output.
def entry(name: str, kind: str) -> str:
    return name if kind == "folder" else f"{name}.json"


# The layout as metadata.json records it: entry name -> "folder" or the output class's name.
def describe(cls: type) -> dict[str, str]:
    return {
        entry(n, k): "folder" if k == "folder" else c.__qualname__  # type: ignore[union-attr]
        for n, (k, c) in parts(cls.__qualname__, cls).items()
    }


# Create the folders and return the layout with its folder fields set, its outputs still unset.
def open_layout(run_dir: str, cls: type) -> Any:
    layout = object.__new__(cls)
    for name, (kind, _) in parts(cls.__qualname__, cls).items():
        if kind == "folder":
            path = Path(run_dir) / name
            path.mkdir(parents=True, exist_ok=True)
            object.__setattr__(layout, name, path)
    return layout


# Write the layout's outputs, then check the folder holds nothing it does not declare.
def close_layout(run_dir: str, layout: Any) -> None:
    cls = type(layout)
    declared = set(RUN_FILES)
    for name, (kind, out) in parts(cls.__qualname__, cls).items():
        declared.add(entry(name, kind))
        if kind == "folder":
            continue
        value = getattr(layout, name, None)
        if not isinstance(value, out):  # type: ignore[arg-type]
            raise TypeError(f"the run's layout.{name} must be set to a {out.__qualname__}, got {value!r}")  # type: ignore[union-attr]
        write_output(os.path.join(run_dir, entry(name, kind)), value)
    stray = sorted(set(os.listdir(run_dir)) - declared)
    if stray:
        raise RuntimeError(
            f"{run_dir} holds {', '.join(stray)}, which {cls.__qualname__} does not declare: a run "
            "writes only into its layout's folders and outputs"
        )


# Read a finished run folder back as its layout: outputs loaded and checked, folders as paths.
def load_layout[T](run_dir: str, cls: type[T]) -> T:
    values: dict[str, Any] = {}
    for name, (kind, out) in parts(cls.__qualname__, cls).items():
        path = Path(run_dir) / entry(name, kind)
        if kind == "folder":
            if not path.is_dir():
                raise FileNotFoundError(f"{run_dir} has no folder {name}/, which {cls.__qualname__} declares")
            values[name] = path
        else:
            values[name] = load_output(str(path), out)  # type: ignore[arg-type]
    return cls(**values)

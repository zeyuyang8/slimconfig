# slimconfig.layouts — what a run folder holds, declared once per kind of run and checked on every run.
#
#     @dataclass
#     class ChatLayout(Layout):
#         messages: Trajectory      # an output class: written as messages.json, a record
#         harness: Folder           # a folder: created before main, its path handed over
#         workspace: Workspace      # a folder of what is not the run's (the agent's work), never read
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
# Every field of a layout class is an output class (see slimconfig.outputs), `Folder` or `Workspace`.
# Before main, the run creates each folder under its run folder and sets the layout's folder fields to
# their paths; main sets every output field. After main, the run writes each output as `<field>.json`, a
# record naming its class (`{"_": "pkg.mod.Trajectory", ...}`, see slimconfig.outputs), then
# checks that the folder holds only its own files (config.yaml, metadata.json, run.log) and what the
# layout declares: anything else is an error naming it.
#
# Every run, with a layout or none, then checks its folder whole (`unrecorded`): it holds JSON only as
# records, each read back as its class, and no symlink; a Workspace alone is never read, as what it
# holds is not the run's but the work's (a checkout, an agent's files).

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, NewType, get_type_hints

from .config import ROOT_NAME
from .outputs import Output, load_output, read_record, record_fields, write_output
from .schemas import Config, _import_dotted
from .structured import load_config

__all__ = ["Folder", "Layout", "Workspace", "load_layout", "unrecorded"]

Folder = NewType("Folder", Path)  # a layout field that is a folder of the run, not a JSON output
Workspace = NewType("Workspace", Path)  # a folder of the run whose files are not its records: never read
RUN_FILES = ("config.yaml", "metadata.json", "run.log")  # what every run folder holds
JSON_NAMES = (".json", ".jsonl", ".ndjson", ".json5")  # a file named so is JSON, in any case


class Layout:
    """Base class of a run folder's declared contents; subclasses are dataclasses (see the header)."""


# The layout class's fields, each as ("folder", None), ("workspace", None) or ("output", its output class).
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
        elif ann is Workspace:
            out[field.name] = ("workspace", None)
        elif isinstance(ann, type) and dataclasses.is_dataclass(ann) and issubclass(ann, Output):
            out[field.name] = ("output", ann)
        else:
            raise TypeError(
                f"{cls.__qualname__}.{field.name} is typed {ann!r}: a layout field is an output "
                "class, Folder or Workspace"
            )
    return out


# Entry names in the run folder: `<name>/` for a folder or workspace, `<name>.json` for an output.
def entry(name: str, kind: str) -> str:
    return f"{name}.json" if kind == "output" else name


# The layout as metadata.json records it: entry name -> "folder", "workspace" or the output class's name.
def describe(cls: type) -> dict[str, str]:
    return {
        entry(n, k): k if c is None else c.__qualname__
        for n, (k, c) in parts(cls.__qualname__, cls).items()
    }


# Create the folders and return the layout with its folder fields set, its outputs still unset.
def open_layout(run_dir: str, cls: type) -> Any:
    layout = object.__new__(cls)
    for name, (kind, _) in parts(cls.__qualname__, cls).items():
        if kind != "output":
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
        if kind != "output":
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
        if kind != "output":
            if not path.is_dir():
                raise FileNotFoundError(f"{run_dir} has no folder {name}/, which {cls.__qualname__} declares")
            values[name] = path
        else:
            values[name] = load_output(str(path), out)  # type: ignore[arg-type]
    return cls(**values)


# Every file of the run folder `run_dir` that is JSON but no record, each with why (empty: none). A file
# `named` gives a class for, by its path from `run_dir` (with whether it holds one record per line), is
# read back as that class; any other JSON file, one named so (JSON_NAMES) or whose text is JSON, whole
# or line by line, as the output class its `_` names. A symlink, or anything but a file or folder, is
# refused whatever it holds; nothing under a folder `skipped` names (a Workspace) is read.
def unrecorded(
    run_dir: str, named: Callable[[str], tuple[type, bool] | None], skipped: Collection[str] = ()
) -> list[str]:
    root, problems = Path(run_dir), []

    def walk(folder: Path) -> None:
        for entry in sorted(folder.iterdir()):
            path = entry.relative_to(root).as_posix()
            if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
                problems.append(f"{entry} is a symlink or a special file, which no run folder holds")
            elif entry.is_dir():
                if path not in skipped:
                    walk(entry)
            else:
                cls, lines = named(path) or (None, None)
                try:
                    records = _records(entry, lines)
                except UnicodeDecodeError as e:
                    problems.append(f"{entry} is JSON that is not text: {e}")
                    continue
                for where, text in records:
                    try:
                        _read_back(text, cls, where)
                    except json.JSONDecodeError as e:
                        problems.append(f"{where} is no JSON: {e}")
                    except ValueError as e:
                        problems.append(str(e))

    if root.is_dir():
        walk(root)
    return problems


# The records `file` holds, each where it is and its text: the whole file, or with `lines` each line.
# With `lines` None (a file nothing names), a file is JSON as its name is (JSON_NAMES) or as its text
# is, whole or each line but blank ones, and holds none otherwise.
def _records(file: Path, lines: bool | None) -> list[tuple[str, str]]:
    if lines is None:
        json_named = file.name.lower().endswith(JSON_NAMES)
        with open(file, "rb") as f:
            if not json_named and f.read(4096).lstrip()[:1] not in (b"{", b"["):
                return []  # no JSON text begins otherwise
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if json_named:
                raise
            return []
        whole = _parses(text)
        lines = not whole and all(_parses(line) for line in text.splitlines() if line.strip())
        if not (whole or lines or json_named):
            return []
    else:
        text = file.read_text(encoding="utf-8")
    if not lines:
        return [(str(file), text)]
    return [(f"{file} line {n}", line) for n, line in enumerate(text.splitlines(), 1)]


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


# Read the record `text` back as `cls`, an output class or a config class (whose fields `load_config`
# reads), or as the output or config class its `_` names when `cls` is None; raises a ValueError naming
# `where`.
def _read_back(text: str, cls: type | None, where: str) -> None:
    if cls is None:
        value = json.loads(text)
        found = value.get(ROOT_NAME) if isinstance(value, dict) else None
        cls = _import_dotted(found) if isinstance(found, str) else None
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls) and issubclass(cls, (Output, Config))):
            named = f'no class under "{ROOT_NAME}"' if found is None else f'{found!r} under "{ROOT_NAME}", no output or config class'
            raise ValueError(f"{where} is JSON that names {named}, so it is no record")
    if issubclass(cls, Config):
        fields = record_fields(text, cls, where)
        try:
            load_config(cls, [fields])
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
    else:
        read_record(text, cls, where)

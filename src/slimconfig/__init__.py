# slimconfig — YAML configs onto typed dataclass schemas, a lightweight Hydra stand-in.
#
#     from slimconfig import run
#
#     def main(cfg: MyConfig, run_dir: str) -> int:   # MyConfig: a dataclass of MISSING leaves and
#         ...                                         # nested config classes; results go under run_dir
#
#     if __name__ == "__main__":                      # python main.py config=configs/my.yaml home=runs/x
#         run(main)
#
# Four rules:
#   * a config class is a @dataclass subclassing `Config` of leaves, nested classes and `dict[K, C]` tables.
#   * a config file names its class (`_ > <dotted.path>:`) and sets only that class's fields.
#   * every leaf is required; nothing is silently defaulted, and "off" is spelled `null`.
#   * where a run writes is not config: `config=` and `home=` belong to the launcher; the log is `run.log`.
#
# A Run may declare its folder's layout: typed outputs (`Output` classes, written as JSON) and folders,
# created, checked and read back with `load_layout` (see slimconfig.layouts). Every JSON file of a run,
# and every line of a JSON-lines file, is a RECORD naming its class under `_` as a config file does
# (`{"_": "pkg.mod.Score", ...}`), read back only as that class: `write_output` / `load_output` for a
# file, `record_text` / `record_line` / `read_record` for its text; metadata.json is a `Metadata`.
# A run fails on any other JSON in its folder (`unrecorded`), but in a `Workspace`, which is the work's.
# The same `Output` classes type any untagged JSON VALUE (`build_output`, `plain_output`), such as a
# model's reply, and give its JSON Schema (`output_schema`).
# A config field typed with an `Output` class is an input: its YAML names a JSON file, read into that
# class and checked before main runs; the run's config.yaml names it by the same path.

from .config import compose, load_mapping_yaml, load_yaml
from .layouts import Folder, Layout, Workspace, load_layout, unrecorded
from .outputs import (
    Output,
    build_output,
    load_output,
    output_schema,
    plain_output,
    read_record,
    record_fields,
    record_line,
    record_text,
    write_output,
)
from .partials import is_partial, partial_of, stated
from .paths import project_root, resolve_path
from .runs import Metadata, Run, run, start_run, tee_stdout
from .schemas import Config, Schema
from .structured import Spec, load_config, merge_specs, peek, schema_of

__version__ = "0.20.0"

# The public API; loader record types (Claim, Composed, Key) stay in slimconfig.config.
__all__ = [
    "Config",
    "Folder",
    "Layout",
    "Metadata",
    "Output",
    "Run",
    "Schema",
    "Spec",
    "Workspace",
    "build_output",
    "compose",
    "is_partial",
    "load_config",
    "load_layout",
    "load_mapping_yaml",
    "load_output",
    "load_yaml",
    "merge_specs",
    "output_schema",
    "partial_of",
    "peek",
    "plain_output",
    "project_root",
    "read_record",
    "record_fields",
    "record_line",
    "record_text",
    "resolve_path",
    "run",
    "schema_of",
    "start_run",
    "stated",
    "tee_stdout",
    "unrecorded",
    "write_output",
]

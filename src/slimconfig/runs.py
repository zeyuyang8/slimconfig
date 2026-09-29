# The run layer: one folder per run, holding the config that produced it, the log of it, and its results.
#
#   * run        — the launcher, and the whole of a script's __main__: the function to run, the config to
#                  run it on, and the folder to run it into.
#   * Run        — the class form of that function, for a routine that is several methods sharing state.
#   * start_run  — the snapshot on its own (config.yaml + metadata.json), for a routine that opens a
#                  second folder of its own (one cell of a sweep, say).
#   * tee_stdout — the log on its own.
#
# WHERE A RUN WRITES IS NOT PART OF ITS CONFIG. A config says what to compute; the folder says where this
# particular launch puts it, which is a property of the invocation — the same config re-run into a
# scratch directory is the same config. So a launch is spelled in one grammar, the same on the command
# line and in the script:
#
#     python train.py config=configs/train.yaml home=runs/exp1 -- optim.lr=1e-4
#
# `config=` names the file to load (repeated, merged left to right), `home=` the folder this run owns,
# and every `key=value` after `--` is an override of the config. The log is always `run.log` inside
# the folder. All of it is recorded in the folder's metadata.json.

from __future__ import annotations

import abc
import contextlib
import dataclasses
import inspect
import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, NoReturn, cast, get_type_hints

from omegaconf import DictConfig, OmegaConf

from .config import ARROW, ROOT_NAME
from .schemas import Config, Schema, Shape, declaration_name
from .structured import Spec, load_config

__all__ = ["Run", "run", "start_run", "tee_stdout"]

# The launcher's keys on the command line, and the log every run folder holds.
CONFIG_KEY = "config"
HOME_KEY = "home"
OVERRIDES_SEPARATOR = "--"
HELP_FLAGS = ("-h", "--help")
LOG = "run.log"


# ── the snapshot ─────────────────────────────────────────────────────────────


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=False)


def _git_head() -> dict[str, Any]:
    try:
        commit = _git("rev-parse", "HEAD")
        if commit.returncode:
            return {}
        return {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(_git("status", "--porcelain").stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        return {}


# Turn the config a caller holds into the plain mapping to snapshot: the loaded schema instance (the
# entry-point case), a dataclass a routine assembled at runtime (one cell of a sweep matrix), or a
# DictConfig / mapping.
def _as_dictconfig(config: Any) -> DictConfig:
    if isinstance(config, DictConfig):
        return config
    if dataclasses.is_dataclass(config) or isinstance(config, Mapping):
        return cast(DictConfig, OmegaConf.structured(config))
    raise TypeError(f"cannot snapshot config of type {type(config).__name__}")


# The declarations that make a snapshot a config file like any other, so a run can be repeated from its
# own folder (`python train.py config=<home>/config.yaml home=<somewhere>`). Every mapping that fills a
# config class gets one, exactly as a hand-written config must — a snapshot missing them would not
# reload, which is the strongest possible check that the rule is the same on both sides. A block names
# its class on its own key; a table names `dict[<key>, <class>]`, once, for all of its entries; an entry
# names nothing, since the table above it already said.
#
# What the SCHEMA says a field holds only tells us where a block WOULD be; the value has to be one. A
# partial (`partial_of`) leaves fields unset, and an unset group or table comes out of to_container as
# the string "???" — there is no block there to name, so it is written through as it is.
def _stamp(node: Mapping[str, Any], schema: Schema) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = schema.fields
    for key, value in node.items():
        held = fields.get(key, Shape("value", None))
        if held.cls is None or not isinstance(value, Mapping):
            out[key] = value
        elif held.kind == "group":
            out[_declares(key, held.cls)] = _stamp(value, Schema(held.cls))
        else:  # a table: declared once, here; its entries are not, but any group INSIDE one still is
            out[_declares(key, held.cls, held.key)] = {
                k: _stamp(v, Schema(held.cls)) if isinstance(v, Mapping) else v
                for k, v in value.items()
            }
    return out


# One key, declaring what the mapping under it is.
def _declares(key: str, cls: type, table_key: type | None = None) -> str:
    return f"{key} {ARROW} {declaration_name(cls, table_key)}"


# The snapshot's YAML text. A config that is not a config-class instance — a mapping or a plain
# dataclass a routine assembled — names no class and is written as it is.
def _snapshot(config: Any) -> str:
    node = _as_dictconfig(config)
    if not isinstance(config, Config):
        return OmegaConf.to_yaml(node, resolve=True)
    container = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
    root = Schema(type(config))
    body = OmegaConf.to_yaml(OmegaConf.create(_stamp(cast(Mapping, container), root)))
    return f"{_declares(ROOT_NAME, type(config))}:\n{body}"


# Open the run's folder and record what produced it. Writes two files:
#   config.yaml   — the fully-resolved config, re-runnable as-is
#                   (`python run.py config=<run_dir>/config.yaml home=<somewhere>`)
#   metadata.json — argv / cwd / run dir / git commit / start time / host
# Everything the run produces goes in this same folder, so a result is never separated from its config.
# The folder itself must be creatable (the run needs somewhere to write); the snapshot is best-effort —
# provenance never aborts a run.
def start_run(run_dir: str, config: Any) -> str:
    os.makedirs(run_dir, exist_ok=True)
    try:
        # Render BOTH payloads before touching a file: re-running a run from its own snapshot
        # (`python run.py config=<run_dir>/config.yaml`) passes the very file we are about to overwrite, and
        # opening it "w" first would truncate it out from under the read.
        snapshot = _snapshot(config)
        meta = json.dumps({
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "run_dir": os.path.abspath(run_dir),
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            **_git_head(),
        }, indent=2, sort_keys=True)
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(snapshot)
        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            f.write(meta)
    except (OSError, TypeError, ValueError) as e:
        print(f"[slimconfig] could not snapshot the config into {run_dir} ({e})")
    return run_dir


# ── the log ──────────────────────────────────────────────────────────────────


class _Tee:
    """The write/flush/isatty a stream needs to stand in for sys.stdout."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self._streams[0].isatty())


# Also write everything printed inside the block to `path`, so a run folder holds the narrative of what
# produced it and not just the numbers. Three deliberate choices:
#   stdout ONLY — progress bars (tqdm and friends) go to stderr, and 45 KB of progress bars is not a log.
#     Anything worth keeping is printed, not drawn.
#   append — a resumed or re-scored run adds to the history of the folder rather than erasing what made
#     the artifacts already in it. `banner` goes to the file only, so appended runs stay tellable apart.
#   parent only — a child process (multiprocessing, a spawned worker) holds the real fd 1, so its output
#     still goes to the terminal. What is worth logging is printed by the parent.
@contextlib.contextmanager
def tee_stdout(path: str, banner: str | None = None) -> Iterator[str]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    real = sys.stdout
    with open(path, "a", encoding="utf-8") as fh:
        if banner:
            fh.write(banner if banner.endswith("\n") else banner + "\n")
        sys.stdout = _Tee(real, fh)
        try:
            yield path
        finally:
            sys.stdout = real


# ── the launch ───────────────────────────────────────────────────────────────


class Run(abc.ABC):
    """A run as a class: the config it runs on, the folder it runs into, and `main`, the work it does.

    The class form of a `run` function, for a routine whose work is several methods sharing state. A
    subclass annotates `config` with its config class — the same promise a function's first argument
    makes — and implements `main`. An instance is built with what `run` takes besides the function
    (`config=`, `home=`, overrides), and `.run()` launches it:

        Train().run()

    `.run()` loads `config`, opens `home` as `run_dir`, calls `main`, and exits with its status.
    """

    config: Config
    run_dir: str

    def __init__(self, *, config: str | None = None, home: str | None = None, **overrides: Any) -> None:
        self._launch = (config, home, overrides)

    @abc.abstractmethod
    def main(self) -> int | None:
        """The run's work; results go under `self.run_dir`. Returns this process's exit status."""

    def run(self) -> NoReturn:
        schema = _config_class(type(self).__qualname__, "`config`", get_type_hints(type(self)).get("config"))

        def main(cfg: Any, run_dir: str) -> int | None:
            self.config, self.run_dir = cfg, run_dir
            return self.main()

        _launch(schema, main, *self._launch)


# A config class is a @dataclass subclassing Config: what an entry point's annotation must name.
def _config_class(owner: str, where: str, schema: Any) -> type:
    if not (isinstance(schema, type) and dataclasses.is_dataclass(schema) and issubclass(schema, Config)):
        raise TypeError(
            f"{owner}'s {where} must be annotated with its config class "
            f"(a @dataclass subclassing slimconfig.Config), got {schema!r}"
        )
    return schema


# The routine `run` was given, as its config class and a call taking (config, run folder). A function IS
# its config — one annotated argument, so the schema comes with the routine — plus an OPTIONAL second
# argument, `run_dir: str`, for a routine that writes into the folder (which is most of them).
def _entrypoint(function: Callable[..., int | None]) -> tuple[type, Callable[[Any, str], int | None]]:
    if not callable(function):
        raise TypeError(f"run() takes a function of one config argument, not {type(function).__name__}")
    name = getattr(function, "__qualname__", repr(function))
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    params = [p for p in inspect.signature(function).parameters.values() if p.kind in kinds]
    if not 1 <= len(params) <= 2:
        raise TypeError(f"{name} must take its config, and optionally the run folder — one or two arguments, not {len(params)}")
    hints = get_type_hints(function)
    schema = _config_class(name, f"argument `{params[0].name}`", hints.get(params[0].name))
    if len(params) == 1:
        return schema, lambda cfg, run_dir: function(cfg)
    if hints.get(params[1].name) is not str:
        raise TypeError(
            f"{name}'s second argument `{params[1].name}` is the run folder and must be annotated "
            f"`str`, got {hints.get(params[1].name)!r}"
        )
    return schema, function


# What the command line said: the `config=` files, the `home=` folder, and the overrides. Before `--` are
# the launcher's own `config=` / `home=`; after it, every `key=value` is an override of the config — so a
# config field may be called `config` or `home` too. `-h` / `--help` before `--` prints the grammar.
def _parse(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    split = argv.index(OVERRIDES_SEPARATOR) if OVERRIDES_SEPARATOR in argv else len(argv)
    configs: list[str] = []
    home: str | None = None
    for arg in argv[:split]:
        if arg in HELP_FLAGS:
            print(_usage_line())
            raise SystemExit(0)
        key, _, value = arg.partition("=")
        if key == CONFIG_KEY:
            configs.append(value)
        elif key == HOME_KEY:
            home = value
        else:
            _usage(f"{arg!r} is not `{CONFIG_KEY}=` or `{HOME_KEY}=`: overrides go after `{OVERRIDES_SEPARATOR}`")
    overrides = argv[split + 1 :]
    for arg in overrides:
        if "=" not in arg:
            _usage(f"override {arg!r} is not key=value")
    return configs, home, overrides


def _usage_line() -> str:
    script = os.path.basename(sys.argv[0]) or "run.py"
    return (
        f"usage: {script} {CONFIG_KEY}=<config.yaml> [{CONFIG_KEY}=<more.yaml> ...] {HOME_KEY}=<run folder> "
        f"[{OVERRIDES_SEPARATOR} key=value ...]"
    )


def _usage(extra: str = "") -> NoReturn:
    raise SystemExit((extra + "\n" if extra else "") + _usage_line())


# Run this process as one run:
#   function — the routine to run. It takes its config (one argument, annotated with its config class)
#              and optionally the run folder (a second argument annotated `str`), and returns this
#              process's exit status (None -> 0).
#   config   — the YAML file to load that class from. Omitted, it comes off the command line
#              (`config=<config.yaml>`), which is how a stepN script is normally launched. The command
#              line wins over it. SEVERAL files may be named there (`config=a.yaml config=b.yaml`), merged
#              left to right — that is where independent fragments are combined, since a file inherits
#              only one (`_default:` in config.py). The difference matters: a chain is a property of the
#              files and lives in them, a combination is a property of this launch and shows up in its
#              argv (and so in its metadata.json). NONE may be named either: a schema whose every field
#              has a default is already a config, and `key=value` still wins over it.
#   home     — the folder this run owns. Omitted, it comes off the command line (`home=<folder>`), which
#              wins over it; one of the two must say.
# Keyword arguments are `key=value` overrides applied on top, the same ones the command line takes after `--`:
# `run(train, config="configs/train.yaml", **{"optim.lr": 1e-4})`.
#
# The launcher loads the config strictly (every field required, every file naming the class it fills),
# creates the run folder, drops the config snapshot and metadata.json in it, tees the function's stdout
# to `run.log` inside it, calls the function, and exits with its status.
#
#     def train(cfg: TrainConfig, run_dir: str) -> int:
#         ...                                     # write results under run_dir
#
#     if __name__ == "__main__":                  # python train.py config=configs/train.yaml \
#         run(train)                              #     home=runs/exp1 -- optim.lr=1e-4
#
# `run` never returns — it exits with the function's status — so a script's __main__ spells neither
# sys.argv nor SystemExit.
def run(
    function: Callable[..., int | None],
    /,
    *,
    config: str | None = None,
    home: str | None = None,
    **overrides: Any,
) -> NoReturn:
    _launch(*_entrypoint(function), config, home, overrides)


# The launch itself, shared by `run` and `Run.run`: load the config, open the folder, call, exit.
def _launch(
    schema: type,
    call: Callable[[Any, str], int | None],
    config: str | None,
    home: str | None,
    overrides: dict[str, Any],
) -> NoReturn:
    configs, cli_home, cli_overrides = _parse(sys.argv[1:])
    if not configs and config is not None:
        configs = [config]
    specs = cast(list[Spec], [*configs, *cli_overrides, *(f"{key}={value}" for key, value in overrides.items())])

    # Naming nothing is a run of the schema's own defaults, which is a whole config when every field
    # has one — a step with nothing left to choose is then launched by naming the script and its home,
    # and the snapshot in its run folder still spells every field out. A schema that is not complete on
    # its own answers by naming the field it is short of, which is what a reader of a bare launch needs;
    # the usage line goes under it to say where such a field is filled in.
    try:
        cfg = load_config(schema, specs)
    except ValueError as incomplete:
        if specs:
            raise
        _usage(str(incomplete))
    where = cli_home or home
    if not where:
        _usage(
            f"this run has nowhere to write: pass `{HOME_KEY}=PATH` — every run owns a folder holding "
            "its config, its log and its results"
        )

    # The folder: the snapshot in it, and stdout tee'd to its `run.log` for the length of the call.
    start_run(where, cfg)
    banner = f"\n═══ {datetime.now(UTC).isoformat(timespec='seconds')} · {' '.join(sys.argv)} ═══"
    with tee_stdout(os.path.join(where, LOG), banner=banner):
        status = call(cfg, where)
    raise SystemExit(status)

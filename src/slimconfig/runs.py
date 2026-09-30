# The run layer: one folder per run, holding its config, its log and its results.
#   * run        — the launcher: a script's whole __main__.
#   * Run        — the class form of `run`, for a routine that is several methods sharing state.
#   * start_run  — the snapshot alone (config.yaml + metadata.json), for a routine opening its own folder.
#   * tee_stdout — the log alone.
# A launch, the same on the command line and in the script:
#     python train.py config=configs/train.yaml home=runs/exp1 -- optim.lr=1e-4
# `config=` may repeat (merged left to right), `home=` is the run folder, overrides go after `--`, and
# the log is always `run.log` in the folder.

from __future__ import annotations

import abc
import contextlib
import dataclasses
import enum
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


# The config to snapshot (a schema instance, dataclass, DictConfig or mapping) as a DictConfig.
def _as_dictconfig(config: Any) -> DictConfig:
    if isinstance(config, DictConfig):
        return config
    if dataclasses.is_dataclass(config) or isinstance(config, Mapping):
        return cast(DictConfig, OmegaConf.structured(config))
    raise TypeError(f"cannot snapshot config of type {type(config).__name__}")


# Add the class declarations a config file needs to every block and table, so the snapshot reloads.
# An unset group or table (a partial's "???") is not a mapping and is written through as-is.
def _stamp(node: Mapping[str, Any], schema: Schema) -> dict[str, Any]:
    out: dict[str, Any] = {}
    fields = schema.fields
    for key, value in node.items():
        held = fields.get(key, Shape("value", None))
        if held.cls is None or not isinstance(value, Mapping):
            out[key] = value
        elif held.kind == "group":
            out[_declares(key, held.cls)] = _stamp(value, Schema(held.cls))
        else:  # a table: declared once here; entries are not, but groups inside them are
            out[_declares(key, held.cls, held.key)] = {
                k: _stamp(v, Schema(held.cls)) if isinstance(v, Mapping) else v
                for k, v in value.items()
            }
    return out


# A key with its class declaration: `key > Class`.
def _declares(key: str, cls: type, table_key: type | None = None) -> str:
    return f"{key} {ARROW} {declaration_name(cls, table_key)}"


# Write every Enum member, key or value, as its value — how a config file spells it.
def _enum_values(node: Any) -> Any:
    if isinstance(node, enum.Enum):
        return node.value
    if isinstance(node, Mapping):
        return {_enum_values(k): _enum_values(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_enum_values(v) for v in node]
    return node


# The snapshot's YAML text; only a Config instance gets class declarations.
def _snapshot(config: Any) -> str:
    node = _as_dictconfig(config)
    if not isinstance(config, Config):
        return OmegaConf.to_yaml(node, resolve=True)
    container = _enum_values(OmegaConf.to_container(node, resolve=True))
    root = Schema(type(config))
    body = OmegaConf.to_yaml(OmegaConf.create(_stamp(cast(Mapping, container), root)))
    return f"{_declares(ROOT_NAME, type(config))}:\n{body}"


# Create `run_dir` and write config.yaml (resolved, re-runnable) and metadata.json (argv, cwd, git,
# time, host). The folder must be creatable; the snapshot is best-effort and never aborts a run.
def start_run(run_dir: str, config: Any) -> str:
    os.makedirs(run_dir, exist_ok=True)
    try:
        # Render both payloads before opening a file: re-running from a snapshot passes the very
        # config.yaml about to be overwritten.
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
    """Writes to several streams; enough of a stream to stand in for sys.stdout."""

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


# Also write everything printed inside the block to `path`. Stdout only (progress bars on stderr stay
# out), appended (a resumed run keeps its history; `banner` goes to the file only), and parent process
# only (children hold the real fd 1).
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
    """The class form of `run`: annotate `config` with a config class and implement `main`.

    Built with `run`'s keyword arguments (`config=`, `home=`, overrides); `.run()` launches it and exits:

        Train().run()
    """

    config: Config
    run_dir: str

    def __init__(self, *, config: str | None = None, home: str | None = None, **overrides: Any) -> None:
        self._launch = (config, home, overrides)

    @abc.abstractmethod
    def main(self) -> int | None:
        """The run's work, writing under `self.run_dir`; returns the exit status."""

    def run(self) -> NoReturn:
        schema = _config_class(type(self).__qualname__, "`config`", get_type_hints(type(self)).get("config"))

        def main(cfg: Any, run_dir: str) -> int | None:
            self.config, self.run_dir = cfg, run_dir
            return self.main()

        _launch(schema, main, *self._launch)


# Check that an entry point's annotation is a config class (a @dataclass subclassing Config).
def _config_class(owner: str, where: str, schema: Any) -> type:
    if not (isinstance(schema, type) and dataclasses.is_dataclass(schema) and issubclass(schema, Config)):
        raise TypeError(
            f"{owner}'s {where} must be annotated with its config class "
            f"(a @dataclass subclassing slimconfig.Config), got {schema!r}"
        )
    return schema


# Split `run`'s function into its config class and a call taking (config, run_dir); the second
# argument, `run_dir: str`, is optional.
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


# Parse argv into (config files, home, overrides). Only `config=`/`home=`/`-h` go before `--`, so a
# config field may itself be named `config` or `home`.
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


# Run this process as one run: load the config, snapshot it into the folder, tee stdout to run.log, call.
#   function    — takes its config (annotated with its config class) and optionally `run_dir: str`
#   config      — default YAML file; the command line's `config=` (repeatable) wins over it
#   home        — default run folder; the command line's `home=` wins over it; one of the two must say
#   **overrides — `key=value` overrides applied on top, like those after `--`
#
#     def train(cfg: TrainConfig, run_dir: str) -> int:
#         ...                                     # write results under run_dir
#
#     if __name__ == "__main__":                  # python train.py config=configs/train.yaml \
#         run(train)                              #     home=runs/exp1 -- optim.lr=1e-4
#
# `run` never returns: it exits with the function's status (None -> 0).
def run(
    function: Callable[..., int | None],
    /,
    *,
    config: str | None = None,
    home: str | None = None,
    **overrides: Any,
) -> NoReturn:
    _launch(*_entrypoint(function), config, home, overrides)


# The launch shared by `run` and `Run.run`.
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

    # With no specs, the schema's own defaults are the config; if incomplete, show the error plus usage.
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

    start_run(where, cfg)
    banner = f"\n═══ {datetime.now(UTC).isoformat(timespec='seconds')} · {' '.join(sys.argv)} ═══"
    with tee_stdout(os.path.join(where, LOG), banner=banner):
        status = call(cfg, where)
    raise SystemExit(status)

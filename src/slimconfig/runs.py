# The run layer: one folder per run, holding the config that produced it, the log of it, and its results.
#
#   * run        — the launcher, and the whole of a script's __main__: the function to run, the config to
#                  run it on, and the folder to run it into.
#   * start_run  — the snapshot on its own (config.yaml + metadata.json), for a routine that opens a
#                  second folder of its own (one cell of a sweep, say).
#   * tee_stdout — the log on its own.
#
# WHERE A RUN WRITES IS NOT PART OF ITS CONFIG. A config says what to compute; the folder and the log say
# where this particular launch puts it, which is a property of the invocation — the same config re-run
# into a scratch directory is the same config. So `run_dir` and `log` are arguments of the launcher, from
# the command line (`--run-dir` / `--log`) or from the script (`run(main, run_dir=...)`), and no config
# class declares them. A run dir that is a FUNCTION of the config — an identity-addressed output tree —
# is passed as a function, which is one rule in code instead of the same interpolation copied into every
# config file that lands there. Both are recorded in the folder's metadata.json either way.

from __future__ import annotations

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

from .config import SCHEMA_KEY
from .schemas import fields_of, schema_name
from .structured import Spec, load_config

__all__ = ["run", "start_run", "tee_stdout"]

# Where a run writes, spelled on the command line. Both take `--flag value` or `--flag=value`.
RUN_DIR_FLAG = "--run-dir"
LOG_FLAG = "--log"
NO_LOG_FLAG = "--no-log"

# A run dir given to `run`: the path itself, or a function returning it. The function takes the loaded
# config, and OPTIONALLY the config file this launch was given as a second argument — which is how a
# script says "the folder is named after the config that produced it", the one naming rule that keeps a
# result and the file that asked for it findable from each other.
type RunDir = str | Callable[..., str]


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


# The `_schema:` lines that make a snapshot a config file like any other, so a run can be repeated from
# its own folder (`python train.py <run_dir>/config.yaml --run-dir <somewhere>`). Every block that fills
# a config class gets one, exactly as a hand-written config must — a snapshot missing them would not
# reload, which is the strongest possible check that the rule is the same on both sides. Table entries
# get none: their class comes from the table. `_schema` is written FIRST in each block, where a reader
# looks for it.
def _stamp(node: Mapping[str, Any], cls: type, tag: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {SCHEMA_KEY: schema_name(cls)} if tag else {}
    shapes = fields_of(cls)
    for key, value in node.items():
        kind, nested = shapes.get(key, ("value", None))
        if nested is None or value is None:
            out[key] = value
        elif kind == "group":
            out[key] = _stamp(value, nested)
        else:  # a table: the entries are not tagged, but any group INSIDE one still is
            out[key] = {k: _stamp(v, nested, tag=False) for k, v in value.items()}
    return out


# The snapshot's YAML text. A config that is not a schema instance — a mapping a routine assembled —
# names no class and is written as it is.
def _snapshot(config: Any) -> str:
    node = _as_dictconfig(config)
    if not (dataclasses.is_dataclass(config) and not isinstance(config, type)):
        return OmegaConf.to_yaml(node, resolve=True)
    container = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
    return OmegaConf.to_yaml(OmegaConf.create(_stamp(cast(Mapping, container), type(config))))


# Open the run's folder and record what produced it. Writes two files:
#   config.yaml   — the fully-resolved config, re-runnable as-is
#                   (`python run.py <run_dir>/config.yaml --run-dir <somewhere>`)
#   metadata.json — argv / cwd / run dir / git commit / start time / host
# Everything the run produces goes in this same folder, so a result is never separated from its config.
# The folder itself must be creatable (the run needs somewhere to write); the snapshot is best-effort —
# provenance never aborts a run.
def start_run(run_dir: str, config: Any) -> str:
    os.makedirs(run_dir, exist_ok=True)
    try:
        # Render BOTH payloads before touching a file: re-running a run from its own snapshot
        # (`python run.py <run_dir>/config.yaml`) passes the very file we are about to overwrite, and
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


# Open `run_dir`, snapshot `cfg` into it, and tee stdout to `<run_dir>/<log>` (log None for no log).
@contextlib.contextmanager
def _open_run(run_dir: str, log: str | None, cfg: Any) -> Iterator[str]:
    start_run(run_dir, cfg)
    if not log:
        yield run_dir
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    banner = f"\n═══ {stamp} · {' '.join(sys.argv)} ═══"
    with tee_stdout(os.path.join(run_dir, log), banner=banner):
        yield run_dir


# What `function` takes: its config class, read off the first parameter's annotation, and whether it
# also wants the run folder. A function IS its config — one annotated argument, so the entry point names
# the routine and the schema comes with it — plus an OPTIONAL second argument, `run_dir: str`, for a
# routine that writes into the folder (which is most of them).
def _signature_of(function: Callable[..., int | None]) -> tuple[type, bool]:
    if not callable(function):
        raise TypeError(f"run() takes a function of one config argument, not {type(function).__name__}")
    name = getattr(function, "__qualname__", repr(function))
    params = [
        p for p in inspect.signature(function).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if not 1 <= len(params) <= 2:
        raise TypeError(
            f"{name} must take its config, and optionally the run folder — one or two arguments, "
            f"not {len(params)}"
        )
    hints = get_type_hints(function)
    schema = hints.get(params[0].name)
    if not (isinstance(schema, type) and dataclasses.is_dataclass(schema)):
        raise TypeError(
            f"{name}'s argument `{params[0].name}` must be annotated with its config class "
            f"(a dataclass), got {schema!r}"
        )
    if len(params) == 2 and hints.get(params[1].name) is not str:
        raise TypeError(
            f"{name}'s second argument `{params[1].name}` is the run folder and must be annotated "
            f"`str`, got {hints.get(params[1].name)!r}"
        )
    return schema, len(params) == 2


# Pull `--run-dir` / `--log` / `--no-log` out of an argv tail, returning what is left (the config specs)
# and whichever of the two the command line set. `--no-log` is how a launch says "no log file" over a
# script that asked for one — every rank of a distributed launch would otherwise append to one path.
def _split_argv(argv: list[str]) -> tuple[list[str], str | None, str | None, bool]:
    specs: list[str] = []
    taken: dict[str, str] = {}
    no_log = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg == NO_LOG_FLAG:
            no_log = True
            continue
        flag = next((f for f in (RUN_DIR_FLAG, LOG_FLAG) if arg == f or arg.startswith(f + "=")), None)
        if flag is None:
            specs.append(arg)
            continue
        if arg != flag:
            taken[flag] = arg[len(flag) + 1 :]
            continue
        if i >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        taken[flag], i = argv[i], i + 1
    return specs, taken.get(RUN_DIR_FLAG), taken.get(LOG_FLAG), no_log


# How many arguments a run-dir function wants: the config, or the config and the file it came from.
def _positional_count(function: Callable[..., Any]) -> int:
    return len([
        p for p in inspect.signature(function).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ])


# The config FILE this launch was given — the first spec that names one. "" when the config came from
# overrides or a mapping alone, so a run-dir function that names a folder after it can say so itself.
def _primary_config(specs: list[Spec]) -> str:
    return next((s for s in specs if isinstance(s, str) and os.path.isfile(s)), "")


def _usage(extra: str = "") -> NoReturn:
    script = os.path.basename(sys.argv[0]) or "run.py"
    raise SystemExit(
        (extra + "\n" if extra else "")
        + f"usage: {script} <config.yaml> [key=value ...] [{RUN_DIR_FLAG} PATH] [{LOG_FLAG} NAME | {NO_LOG_FLAG}]"
    )


# Run this process as one run:
#   function — the routine to run. It takes its config (one argument, annotated with its config class)
#              and optionally the run folder (a second argument annotated `str`), and returns this
#              process's exit status (None -> 0).
#   config   — the YAML file to load that class from. Omitted, it comes off the command line
#              (`<config.yaml> [key=value ...]`), which is how a stepN script is normally launched.
#   run_dir  — the folder this run owns: a path, or a function returning one — of the loaded config, and
#              of the config file itself if it takes a second argument (`lambda cfg, path: ...`).
#              `--run-dir` on the command line wins over it; one of the two must say.
#   log      — the log file inside that folder, None for no log. `--log` / `--no-log` win over it.
# Keyword arguments are `key=value` overrides applied on top, the same ones the command line takes:
# `run(train, "configs/train.yaml", **{"optim.lr": 1e-4})`.
#
# The launcher loads the config strictly (every field required, every file naming the class it fills),
# creates the run folder, drops the config snapshot and metadata.json in it, tees the function's stdout
# to the log inside it, calls the function, and exits with its status.
#
#     def train(cfg: TrainConfig, run_dir: str) -> int:
#         ...                                     # write results under run_dir
#
#     if __name__ == "__main__":                  # python train.py configs/train.yaml optim.lr=1e-4 \
#         run(train, log="train.log")             #     --run-dir runs/exp1
#
# `run` never returns — it exits with the function's status — so a script's __main__ spells neither
# sys.argv nor SystemExit.
def run(
    function: Callable[..., int | None],
    config: str | None = None,
    /,
    *,
    run_dir: RunDir | None = None,
    log: str | None = None,
    **overrides: Any,
) -> NoReturn:
    schema, wants_run_dir = _signature_of(function)
    argv_specs, cli_run_dir, cli_log, no_log = _split_argv(list(sys.argv[1:]))
    specs: list[Spec] = [config] if config is not None else cast(list[Spec], argv_specs)
    if not specs:
        _usage()
    specs += [f"{key}={value}" for key, value in overrides.items()]

    cfg = load_config(schema, specs)

    where = cli_run_dir if cli_run_dir is not None else run_dir
    if callable(where):
        where = where(cfg, _primary_config(specs)) if _positional_count(where) > 1 else where(cfg)
    if not where:
        _usage(
            "this run has nowhere to write: pass `--run-dir PATH`, or give the script a run dir "
            "(`run(fn, run_dir=...)`) — every run owns a folder holding its config, its log and its results"
        )
    log_name = None if no_log else (cli_log if cli_log is not None else log)

    with _open_run(where, log_name, cfg):
        status = function(cfg, where) if wants_run_dir else function(cfg)
    raise SystemExit(status)

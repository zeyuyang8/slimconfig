# The run layer: one folder per run, holding the config that produced it, the log of it, and its results.
#
#   * run        — the launcher, and the whole of a script's __main__. TWO inputs: the function to run,
#                  and the config to run it on. Everything else is a field of that config.
#   * start_run  — the snapshot on its own (config.yaml + metadata.json), for a routine that opens a
#                  second folder of its own (one cell of a sweep, say).
#   * tee_stdout — the log on its own.
# What a run WRITES into the folder is the function's job — it reads the path off its own config.

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

from .structured import Spec, load_config

# What every config `run` accepts must declare. The launcher reads both off the loaded config: the folder
# it opens, and the log inside it. A schema missing either one is a programming error, not a config error,
# so it is reported at launch with the field named.
RUN_FIELDS = ("run_dir", "log")


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


# Open the run's folder and record what produced it. Writes two files:
#   config.yaml   — the fully-resolved config, re-runnable as-is (`python run.py <run_dir>/config.yaml`)
#   metadata.json — argv / cwd / git commit / start time / host
# Everything the run produces goes in this same folder, so a result is never separated from its config.
# The folder itself must be creatable (the run needs somewhere to write); the snapshot is best-effort —
# provenance never aborts a run.
def start_run(run_dir: str, config: Any) -> str:
    os.makedirs(run_dir, exist_ok=True)
    try:
        # Render BOTH payloads before touching a file: re-running a run from its own snapshot
        # (`python run.py <run_dir>/config.yaml`) passes the very file we are about to overwrite, and
        # opening it "w" first would truncate it out from under the read.
        snapshot = OmegaConf.to_yaml(_as_dictconfig(config), resolve=True)
        meta = json.dumps({
            "argv": sys.argv,
            "cwd": os.getcwd(),
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


# Open `cfg.run_dir`, snapshot the config into it, and tee stdout to `<run_dir>/<cfg.log>` (log unset for
# no log). The folder layer `run` is built on.
@contextlib.contextmanager
def _open_run(cfg: Any) -> Iterator[str]:
    run_dir, log = cfg.run_dir, cfg.log
    if not run_dir:
        raise SystemExit(
            "this run has nowhere to write: `run_dir` is unset — every run owns a folder holding its "
            "config, its log, and its results"
        )
    start_run(run_dir, cfg)
    if not log:
        yield run_dir
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    banner = f"\n═══ {stamp} · {' '.join(sys.argv)} ═══"
    with tee_stdout(os.path.join(run_dir, log), banner=banner):
        yield run_dir


# The schema `function` takes, read off its one annotated parameter. A function IS its config: one
# argument, typed, so the entry point names the routine and the schema comes with it.
def _schema_of(function: Callable[[Any], int | None]) -> type:
    if not callable(function):
        raise TypeError(f"run() takes a function of one config argument, not {type(function).__name__}")
    name = getattr(function, "__qualname__", repr(function))
    params = [
        p for p in inspect.signature(function).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if len(params) != 1:
        raise TypeError(f"{name} must take exactly one argument (its config), not {len(params)}")
    schema = get_type_hints(function).get(params[0].name)
    if not (isinstance(schema, type) and dataclasses.is_dataclass(schema)):
        raise TypeError(
            f"{name}'s argument `{params[0].name}` must be annotated with its config class "
            f"(a dataclass), got {schema!r}"
        )
    fields = {f.name for f in dataclasses.fields(schema)}
    missing = [f for f in RUN_FIELDS if f not in fields]
    if missing:
        raise TypeError(
            f"{schema.__name__} is missing {' and '.join(f'`{f}`' for f in missing)}: every config "
            f"`run` takes declares {' and '.join(f'`{f}`' for f in RUN_FIELDS)} — the folder the run owns "
            "and the log file inside it"
        )
    return schema


# Run this process as one run. Two inputs:
#   function — the routine to run. It takes ONE argument, annotated with its config class, and returns
#              this process's exit status (None → 0).
#   config   — the YAML file to load that class from. Omitted, it comes off the command line
#              (`<config.yaml> [key=value ...]`), which is how a stepN script is normally launched.
# Keyword arguments are `key=value` overrides applied on top, the same ones the command line takes:
# `run(train, "configs/train.yaml", **{"optim.lr": 1e-4})`.
#
# The config class must declare `run_dir` and `log`. That is the whole of the launcher's contract: it
# loads the config strictly (every field required), opens `run_dir`, drops the config snapshot and
# metadata.json in it, tees the function's stdout to `log` inside it, calls the function, and exits with
# its status.
#
#     def train(cfg: TrainConfig) -> int:
#         ...                                     # write results under cfg.run_dir
#
#     if __name__ == "__main__":                  # python train.py configs/train.yaml optim.lr=1e-4
#         run(train)
#
# `run` never returns — it exits with the function's status — so a script's __main__ spells neither
# sys.argv nor SystemExit.
def run(function: Callable[[Any], int | None], config: str | None = None, /, **overrides: Any) -> NoReturn:
    schema = _schema_of(function)
    specs: list[Spec] = [config] if config is not None else list(sys.argv[1:])
    if not specs:
        raise SystemExit(f"usage: {os.path.basename(sys.argv[0]) or 'run.py'} <config.yaml> [key=value ...]")
    specs += [f"{key}={value}" for key, value in overrides.items()]
    cfg = load_config(schema, specs)
    with _open_run(cfg):
        status = function(cfg)
    raise SystemExit(status)

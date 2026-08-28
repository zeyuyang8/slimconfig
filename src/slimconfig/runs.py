# The run layer: one folder per run, holding the config that produced it, the log of it, and its results.
#
# A run is a function plus the folder it owns. `run_dir` is an ordinary config key, so the config says
# where that folder is; everything else here follows from it:
#   * @run(Schema)  — decorate an entry point: load the config, open `run_dir`, tee stdout into it, call.
#   * dispatch      — the same, for an entry point that fans out to several handlers on `mode`.
#   * start_run     — the snapshot on its own (config.yaml + run_meta.json), for a routine that opens a
#                     folder the config does not name (one cell of a sweep, say).
#   * tee_stdout    — the log on its own.
# What a run WRITES into the folder is the function's job — it reads the path off its own config.

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .structured import Spec, load_config, merge_specs, peek


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


# Turn whatever a caller has into the plain config mapping to snapshot: raw specs (the entry-point path —
# because every field must be set explicitly, the merged YAML already IS the full config), a DictConfig,
# or a dataclass instance a handler built at runtime (e.g. one cell of a sweep matrix).
def _as_dictconfig(config: Any) -> DictConfig:
    if isinstance(config, list):
        return merge_specs(cast("list[Spec]", config))
    if isinstance(config, DictConfig):
        return config
    if dataclasses.is_dataclass(config) or isinstance(config, Mapping):
        return cast(DictConfig, OmegaConf.structured(config))
    raise TypeError(f"cannot snapshot config of type {type(config).__name__}")


# Open the run's folder and record what produced it. Writes two files:
#   config.yaml   — the fully-resolved config, re-runnable as-is (`python run.py <run_dir>/config.yaml`)
#   run_meta.json — argv / cwd / git commit / start time / host
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
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
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


# The `run_dir` a config declares, however the caller holds it: raw specs, a mapping, or a loaded schema
# instance. Missing (or empty) is fatal — a run with nowhere to write is not a run.
def _run_dir_of(config: Any) -> str:
    if isinstance(config, list):
        value = peek(cast("list[Spec]", config), "run_dir")
    elif isinstance(config, Mapping | DictConfig):
        value = config.get("run_dir")
    else:
        value = getattr(config, "run_dir", None)
    if not value:
        raise SystemExit(
            "config must set `run_dir` — every run owns a folder holding its config, its log, and its results"
        )
    return str(value)


# Open the folder `config` names, snapshot the config into it, and tee stdout to `<run_dir>/<log>`
# (`log=None` to skip the log). Shared by @run and dispatch, so both leave the same folder behind.
@contextlib.contextmanager
def open_run(config: Any, log: str | None = "run.log") -> Iterator[str]:
    run_dir = _run_dir_of(config)
    start_run(run_dir, config)
    if log is None:
        yield run_dir
        return
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    banner = f"\n═══ {stamp} · {' '.join(sys.argv)} ═══"
    with tee_stdout(os.path.join(run_dir, log), banner=banner):
        yield run_dir


# Make a function the entry point of a run: it is called with the config named on the command line, and
# the folder that config's `run_dir` names is created, snapshotted, and logged into around the call.
#
#     @run(TrainConfig)
#     def main(cfg: TrainConfig) -> int:
#         ...                                   # write results under cfg.run_dir
#
#     if __name__ == "__main__":
#         raise SystemExit(main())              # specs default to sys.argv[1:]
#
# `main(["configs/train.yaml", "optim.lr=1e-4"])` calls the same entry point from Python. Bare `@run`
# (no schema) hands the function the raw specs instead, for an entry point whose schema depends on
# another field — it loads its own config with load_config. `log=` names the log file inside the run
# folder — `{name}` in it stands for the function's name — and `log=None` turns the tee off.
def run(
    schema: type | Callable[..., Any] | None = None, *, log: str | None = "{name}.log"
) -> Callable[..., Any]:
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = log.format(name=fn.__name__) if log else None

        @functools.wraps(fn)
        def entry(specs: list[Spec] | None = None) -> Any:
            resolved = list(sys.argv[1:] if specs is None else specs)
            with open_run(resolved, log=name):
                return fn(resolved if schema is None else load_config(schema, resolved))

        return entry

    # Bare `@run` — the decorated function itself lands in `schema`. A schema is a class; a function is not.
    if callable(schema) and not isinstance(schema, type):
        fn, schema = schema, None
        return decorate(fn)
    return decorate


# Run the handler the config's `mode` selects — the whole body of every entry point that fans out, so the
# mode contract (and its error message) is written once. `modes` maps a mode name to either
#   (schema, handler)  — load `schema` strictly, call handler(cfg); the usual case, or
#   handler            — call handler(specs) and let it load its own config, for the modes whose schema
#                        depends on another field (e.g. one schema per `method`).
# Both branches run inside the run folder `run_dir` names (see open_run), logging to `<mode>.log`, so no
# handler has to remember to open it.
def dispatch(modes: Mapping[str, Any], specs: list[Spec]) -> int:
    mode = peek(specs, "mode")
    if mode not in modes:
        raise SystemExit(f"config must set `mode` to one of {', '.join(modes)} (got {mode!r})")
    entry = modes[mode]
    with open_run(specs, log=f"{mode}.log"):
        if isinstance(entry, tuple):
            schema, handler = entry
            return handler(load_config(schema, specs))
        return entry(specs)

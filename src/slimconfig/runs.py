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
#
# Three objects hold the three things a launch is made of, so `run` itself reads as the four steps it
# takes and each rule has a name:
#   * Entrypoint — the routine being launched: its config class, and whether it also wants the folder.
#   * Launch     — what the command line said: the config specs, and where this run writes.
#   * RunFolder  — the folder itself: the snapshot in it, and the log tee'd into it.

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
from .schemas import Config, Schema, Shape, declaration_name
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


# The `_schema:` lines that make a snapshot a config file like any other, so a run can be repeated from
# its own folder (`python train.py <run_dir>/config.yaml --run-dir <somewhere>`). Every mapping that
# fills a config class gets one, exactly as a hand-written config must — a snapshot missing them would
# not reload, which is the strongest possible check that the rule is the same on both sides. A block
# names its class; a table names `dict[<key>, <class>]`, once, for all of its entries; an entry names
# nothing, since the table above it already said. `_schema` is written FIRST in each mapping, where a
# reader looks for it.
#
# What the SCHEMA says a field holds only tells us where a block WOULD be; the value has to be one. A
# partial (`partial_of`) leaves fields unset, and an unset group or table comes out of to_container as
# the string "???" — there is no block there to name, so it is written through as it is.
def _stamp(node: Mapping[str, Any], schema: Schema, tag: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {} if tag is None else {SCHEMA_KEY: tag}
    fields = schema.fields
    for key, value in node.items():
        held = fields.get(key, Shape("value", None))
        if held.cls is None or not isinstance(value, Mapping):
            out[key] = value
        elif held.kind == "group":
            out[key] = _stamp(value, Schema(held.cls), declaration_name(held.cls))
        else:  # a table: tagged once, here; its entries are not, but any group INSIDE one still is
            out[key] = {SCHEMA_KEY: declaration_name(held.cls, held.key)} | {
                k: _stamp(v, Schema(held.cls)) if isinstance(v, Mapping) else v
                for k, v in value.items()
            }
    return out


# The snapshot's YAML text. A config that is not a config-class instance — a mapping or a plain
# dataclass a routine assembled — names no class and is written as it is.
def _snapshot(config: Any) -> str:
    node = _as_dictconfig(config)
    if not isinstance(config, Config):
        return OmegaConf.to_yaml(node, resolve=True)
    container = OmegaConf.to_container(node, resolve=True, enum_to_str=True)
    root = Schema(type(config))
    return OmegaConf.to_yaml(OmegaConf.create(_stamp(cast(Mapping, container), root, root.name)))


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


@dataclasses.dataclass(frozen=True, slots=True)
class _RunFolder:
    """The folder one run owns: where it is, and what it logs to (None for no log)."""

    path: str
    log: str | None

    # Snapshot `cfg` into the folder and tee stdout to the log inside it, for the length of the block.
    @contextlib.contextmanager
    def open(self, cfg: Any) -> Iterator[str]:
        start_run(self.path, cfg)
        if not self.log:
            yield self.path
            return
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        banner = f"\n═══ {stamp} · {' '.join(sys.argv)} ═══"
        with tee_stdout(os.path.join(self.path, self.log), banner=banner):
            yield self.path


@dataclasses.dataclass(frozen=True, slots=True)
class _Entrypoint:
    """The routine `run` was given, and what its signature says it wants.

    A function IS its config — one annotated argument, so the entry point names the routine and the
    schema comes with it — plus an OPTIONAL second argument, `run_dir: str`, for a routine that writes
    into the folder (which is most of them).
    """

    function: Callable[..., int | None]
    schema: type
    wants_run_dir: bool

    @classmethod
    def of(cls, function: Callable[..., int | None]) -> _Entrypoint:
        if not callable(function):
            raise TypeError(f"run() takes a function of one config argument, not {type(function).__name__}")
        name = getattr(function, "__qualname__", repr(function))
        params = _positional(function, keyword_only=True)
        if not 1 <= len(params) <= 2:
            raise TypeError(
                f"{name} must take its config, and optionally the run folder — one or two arguments, "
                f"not {len(params)}"
            )
        hints = get_type_hints(function)
        schema = hints.get(params[0].name)
        if not (isinstance(schema, type) and dataclasses.is_dataclass(schema) and issubclass(schema, Config)):
            raise TypeError(
                f"{name}'s argument `{params[0].name}` must be annotated with its config class "
                f"(a @dataclass subclassing slimconfig.Config), got {schema!r}"
            )
        if len(params) == 2 and hints.get(params[1].name) is not str:
            raise TypeError(
                f"{name}'s second argument `{params[1].name}` is the run folder and must be annotated "
                f"`str`, got {hints.get(params[1].name)!r}"
            )
        return cls(function, schema, len(params) == 2)

    def __call__(self, cfg: Any, run_dir: str) -> int | None:
        return self.function(cfg, run_dir) if self.wants_run_dir else self.function(cfg)


@dataclasses.dataclass(frozen=True, slots=True)
class _Launch:
    """What the command line said: the config specs, and whichever of the run dir / log it set.

    `--no-log` is how a launch says "no log file" over a script that asked for one — every rank of a
    distributed launch would otherwise append to one path.
    """

    specs: list[str]
    run_dir: str | None
    log: str | None
    no_log: bool

    # Pull `--run-dir` / `--log` / `--no-log` out of an argv tail; what is left is the config specs.
    @classmethod
    def parse(cls, argv: list[str]) -> _Launch:
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
        return cls(specs, taken.get(RUN_DIR_FLAG), taken.get(LOG_FLAG), no_log)

    # The folder this run owns, and the log inside it. The command line wins over what the script asked
    # for, in both cases. A run dir the script gave as a FUNCTION is called here — with the loaded
    # config, and with the config file this launch was given if it takes a second argument.
    def folder(self, script_run_dir: RunDir | None, script_log: str | None, cfg: Any) -> _RunFolder:
        where: RunDir | None = self.run_dir if self.run_dir is not None else script_run_dir
        if callable(where):
            wants_file = len(_positional(where)) > 1
            where = where(cfg, self.primary_config()) if wants_file else where(cfg)
        if not where:
            _usage(
                "this run has nowhere to write: pass `--run-dir PATH`, or give the script a run dir "
                "(`run(fn, run_dir=...)`) — every run owns a folder holding its config, its log and its results"
            )
        log = None if self.no_log else (self.log if self.log is not None else script_log)
        return _RunFolder(where, log)

    # The config FILE this launch was given — the first spec that names one. "" when the config came
    # from overrides alone, so a run-dir function that names a folder after it can say so itself.
    # FIRST, not last: with several files the first is the one the reader typed as "what this run is",
    # the rest being what it was combined with. A launch where that is not true should pass `--run-dir`.
    def primary_config(self) -> str:
        return next((s for s in self.specs if os.path.isfile(s)), "")


# The positional parameters of a function — plus the keyword-only ones where those count too (a config
# argument may be spelled either way; a run-dir function's cannot, since `run` passes them positionally).
def _positional(function: Callable[..., Any], keyword_only: bool = False) -> list[inspect.Parameter]:
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if keyword_only:
        kinds += (inspect.Parameter.KEYWORD_ONLY,)
    return [p for p in inspect.signature(function).parameters.values() if p.kind in kinds]


def _usage(extra: str = "") -> NoReturn:
    script = os.path.basename(sys.argv[0]) or "run.py"
    raise SystemExit(
        (extra + "\n" if extra else "")
        + f"usage: {script} <config.yaml> [more.yaml ...] [key=value ...] "
        + f"[{RUN_DIR_FLAG} PATH] [{LOG_FLAG} NAME | {NO_LOG_FLAG}]"
    )


# Run this process as one run:
#   function — the routine to run. It takes its config (one argument, annotated with its config class)
#              and optionally the run folder (a second argument annotated `str`), and returns this
#              process's exit status (None -> 0).
#   config   — the YAML file to load that class from. Omitted, it comes off the command line
#              (`<config.yaml> [more.yaml ...] [key=value ...]`), which is how a stepN script is normally
#              launched. SEVERAL files may be named, merged left to right — that is where independent
#              fragments are combined, since a file inherits only one (`_default:` in config.py). The
#              difference matters: a chain is a property of the files and lives in them, a combination is
#              a property of this launch and shows up in its argv (and so in its metadata.json).
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
    entry = _Entrypoint.of(function)
    launch = _Launch.parse(list(sys.argv[1:]))
    if config is not None:
        launch = dataclasses.replace(launch, specs=[config])
    if not launch.specs:
        _usage()
    specs = cast(list[Spec], [*launch.specs, *(f"{key}={value}" for key, value in overrides.items())])

    cfg = load_config(entry.schema, specs)
    folder = launch.folder(run_dir, log, cfg)
    with folder.open(cfg):
        status = entry(cfg, folder.path)
    raise SystemExit(status)

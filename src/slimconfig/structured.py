# Typed, all-fields-required config loading — merge YAML onto a dataclass schema.
#
# A schema is a @dataclass whose every leaf defaults to omegaconf.MISSING, so a config must set
# each field explicitly (nothing is silently filled in). load_config merges one or more specs (YAML
# files and/or dotted key=value overrides) onto that schema and returns a fully populated instance.
# Two rules, enforced at load time:
#   * every leaf must end up set — an unset MISSING leaf raises (a nullable field that is "off" must
#     still be present as null; an empty collection must be written out as []);
#   * unknown keys are rejected (OmegaConf struct mode).
# YAML files are read via slimconfig.config.load_mapping_yaml, so each may carry a top-level
# `defaults: [<path>, ...]` list to compose shared configs (Hydra-style: listed files merge first,
# the current file wins on top).

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .config import load_mapping_yaml

# One config source: a YAML file path, a `dotted.key=value` override, or a ready-made mapping.
type Spec = str | Mapping[str, Any] | DictConfig


# Dotted paths of every leaf field still unset (recurses into nested configs).
def _missing_fields(cfg: DictConfig, prefix: str = "") -> list[str]:
    missing: list[str] = []
    for raw_key in cfg:
        key = str(raw_key)
        if OmegaConf.is_missing(cfg, key):
            missing.append(prefix + key)
            continue
        value = cfg[key]
        if OmegaConf.is_dict(value):  # recurse into nested configs only; lists are leaf values
            missing.extend(_missing_fields(cast(DictConfig, value), prefix + key + "."))
    return missing


# Merge YAML files, dotted key=value overrides, and already-built mappings into one unstructured
# config. A YAML file is loaded via load_mapping_yaml, so it may carry a top-level `defaults: [...]`
# list to compose others; a mapping spec lets a caller merge values it computed itself (e.g. one cell
# of a sweep matrix resolved at runtime) with the same precedence rule — later specs win.
def merge_specs(specs: list[Spec]) -> DictConfig:
    merged = OmegaConf.create()
    for spec in specs:
        if isinstance(spec, Mapping | DictConfig):
            merged = OmegaConf.merge(merged, spec)
        elif Path(spec).is_file():
            merged = OmegaConf.merge(merged, load_mapping_yaml(spec))
        elif "=" in spec:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([spec]))
        else:
            raise FileNotFoundError(
                f"config spec {spec!r} is neither a file nor a key=value override"
            )
    return cast(DictConfig, merged)


# Merge `specs` (YAML files and/or dotted key=value overrides) onto `schema`, in order (list a file
# before the overrides that should win over it). Returns a fully-populated schema instance. Raises
# ValueError if any leaf is left unset, FileNotFoundError for a bad spec, and OmegaConf errors for
# unknown keys / type mismatches.
def load_config[T](schema: type[T], specs: list[Spec]) -> T:
    merged = OmegaConf.merge(OmegaConf.structured(schema), merge_specs(specs))
    missing = _missing_fields(cast(DictConfig, merged))
    if missing:
        raise ValueError(f"{schema.__name__} is missing required field(s): {', '.join(missing)}")
    return cast(T, OmegaConf.to_object(merged))


# Return top-level `key` from the merged specs (or None), without validation — lets a caller pick a
# schema before strict structured loading (e.g. read `method` to choose which schema to load). Accepts
# the same specs as load_config (YAML file paths and/or dotted key=value overrides), so it works with
# the bare-file-path invocation entry points use. Unknown keys are tolerated (no struct check).
def peek(args: list[Spec], key: str) -> Any:
    return merge_specs(args).get(key)


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


# Turn whatever a caller has into the plain config mapping to snapshot: raw specs (the dispatch path —
# because every field must be set explicitly, the merged YAML already IS the full config), a DictConfig,
# or a dataclass instance a handler built at runtime (e.g. one cell of a sweep matrix).
def _as_dictconfig(config: Any) -> DictConfig:
    if isinstance(config, list):
        return merge_specs(cast(list[Spec], config))
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


# Run the handler the config's `mode` selects — the whole body of every entry point, so the mode
# contract (and its error message) is written once. `modes` maps a mode name to either
#   (schema, handler)  — load `schema` strictly, call handler(cfg); the usual case, or
#   handler            — call handler(specs) and let it load its own config, for the modes whose schema
#                        depends on another field (e.g. one schema per `method`).
# Both branches get their `run_dir` opened and snapshotted first (see start_run), so no handler has to
# remember to do it.
def dispatch(modes: Mapping[str, Any], specs: list[Spec]) -> int:
    mode = peek(specs, "mode")
    if mode not in modes:
        raise SystemExit(f"config must set `mode` to one of {', '.join(modes)} (got {mode!r})")
    run_dir = peek(specs, "run_dir")
    if not run_dir:
        raise SystemExit("config must set `run_dir` — every run owns a folder holding its config and results")
    start_run(run_dir, specs)

    entry = modes[mode]
    if isinstance(entry, tuple):
        schema, handler = entry
        return handler(load_config(schema, specs))
    return entry(specs)

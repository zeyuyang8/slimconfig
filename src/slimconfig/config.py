# slimconfig.config — the YAML layer: read a file, compose its `defaults:` chain.
#
# Read a YAML config into an OmegaConf DictConfig and compose any `defaults: [...]` chain:
#   * load_mapping_yaml — load a YAML, require a top-level mapping, compose any `defaults:` chain
#     (current file wins), and return a DictConfig.
#   * load_yaml — plain PyYAML: read a YAML file into a dict (no `defaults:` composition).
# The typed, all-fields-required loader (load_config) lives in structured.py.
#
# Importing this module also registers two OmegaConf interpolation resolvers (once, process-wide):
#   * ${now:<strftime>}          — stamp a value with the load time.
#   * ${from_yaml:<path>,<key>}  — read one value OUT of another config file, so a config can track
#                                  a value another owns without duplicating it.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

# ``${now:<strftime>}`` — interpolate the current time into any config value (Hydra-style). Registered
# at import so every OmegaConf-loaded config has it. ``replace=True`` keeps re-import idempotent;
# ``use_cache=True`` gives ONE consistent timestamp for the whole load (and per process).
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True, use_cache=True
)

# Sentinel telling OmegaConf.select "not found" apart from a real ``null`` value at the key.
_NOT_FOUND = object()


def _select_from_yaml(path: str, key: str) -> Any:
    cfg = load_mapping_yaml(path.strip())
    val = OmegaConf.select(cfg, key.strip(), default=_NOT_FOUND, throw_on_missing=True)
    if val is _NOT_FOUND:
        raise ValueError(f"${{from_yaml:{path},{key}}}: {path!r} has no key {key.strip()!r}")
    return val


OmegaConf.register_new_resolver("from_yaml", _select_from_yaml, replace=True, use_cache=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Config {path} is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(cfg).__name__})")
    return cfg


def load_mapping_yaml(path: str) -> DictConfig:
    return _compose(Path(path).resolve(), visiting=())


def _load_one(path: Path) -> DictConfig:
    try:
        loaded = OmegaConf.load(path)
    except OSError as e:
        # OmegaConf.load raises OSError("Invalid loaded object type: <type>") for a top-level SCALAR
        # yaml (42 / 3.14 / true) before we can type-check it. Re-raise THAT as the same path-naming
        # ValueError so a scalar fails like every other non-mapping shape. A genuine IO error
        # (missing/unreadable file -> FileNotFoundError) is NOT a parse problem -> let it propagate.
        if isinstance(e, FileNotFoundError) or "Invalid loaded object type" not in str(e):
            raise
        raise ValueError(
            f"config file {str(path)!r} did not parse to a mapping (got {type(e).__name__}: {e})"
        ) from e
    if not isinstance(loaded, DictConfig):
        raise ValueError(
            f"config file {str(path)!r} did not parse to a mapping (got {type(loaded).__name__})"
        )
    return loaded


def _compose(path: Path, visiting: tuple[Path, ...]) -> DictConfig:
    if path in visiting:
        chain = " -> ".join(str(p) for p in (*visiting, path))
        raise ValueError(f"`defaults` cycle detected: {chain}")
    loaded = _load_one(path)
    defaults = loaded.pop("defaults", None)
    if defaults is None:
        return loaded
    if not isinstance(defaults, ListConfig):
        raise ValueError(
            f"config file {str(path)!r}: top-level `defaults` must be a list of yaml paths, "
            f"got {type(defaults).__name__}"
        )
    merged: DictConfig = OmegaConf.create({})  # type: ignore[assignment]
    for entry in defaults:
        if not isinstance(entry, str):
            raise ValueError(
                f"config file {str(path)!r}: each `defaults` entry must be a string path, "
                f"got {type(entry).__name__}: {entry!r}"
            )
        # Defaults paths resolve relative to the CWD — the project root every script is run from —
        # so one config path convention holds across a repo, wherever the including file lives.
        # (An absolute entry resolves to itself: Path("/abs") wins over the cwd join.)
        entry_path = (Path.cwd() / entry).resolve()
        merged = cast(DictConfig, OmegaConf.merge(merged, _compose(entry_path, (*visiting, path))))
    return cast(DictConfig, OmegaConf.merge(merged, loaded))

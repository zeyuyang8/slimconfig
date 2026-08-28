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

from collections.abc import Mapping
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

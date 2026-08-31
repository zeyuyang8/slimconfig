# Typed, all-fields-required config loading — merge YAML onto a dataclass schema.
#
# A schema is a @dataclass whose every leaf defaults to omegaconf.MISSING, so a config must set each
# field explicitly (nothing is silently filled in). load_config merges one or more specs (YAML files
# and/or dotted key=value overrides) onto that schema and returns a fully populated instance.
#
# Three rules, enforced at load time:
#   * every config FILE names the class it fills (`_schema: <dotted.path>`) and that class must be the
#     one the file is being merged onto, or a base of it — so a fragment cannot be mounted at a block
#     it was not written for, and pointing a script at the wrong config fails by class name instead of
#     by an unknown key three levels down;
#   * every leaf must end up set — an unset MISSING leaf raises (a nullable field that is "off" must
#     still be present as null; an empty collection must be written out as []);
#   * unknown keys are rejected (OmegaConf struct mode).
#
# YAML files are read via slimconfig.config.compose, so any mapping in one — at any depth — may carry
# `defaults: [<path>, ...]` to start from shared files, and the file that lists them wins on top.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .config import Claim, compose
from .schemas import check_schema, field_schema, resolve_schema, schema_name

# One config source: a YAML file path, a `dotted.key=value` override, or a ready-made mapping.
type Spec = str | Mapping[str, Any] | DictConfig


class _Merged:
    """The merged specs and the `_schema:` claims the files among them made."""

    def __init__(self, config: DictConfig, claims: tuple[Claim, ...]) -> None:
        self.config = config
        self.claims = claims


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


def _merge(specs: list[Spec]) -> _Merged:
    merged = OmegaConf.create()
    claims: list[Claim] = []
    for spec in specs:
        if isinstance(spec, Mapping | DictConfig):
            merged = OmegaConf.merge(merged, spec)
        elif Path(spec).is_file():
            composed = compose(spec)
            claims.extend(composed.claims)
            merged = OmegaConf.merge(merged, composed.config)
        elif "=" in spec:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([spec]))
        else:
            raise FileNotFoundError(
                f"config spec {spec!r} is neither a file nor a key=value override"
            )
    return _Merged(cast(DictConfig, merged), tuple(claims))


# Merge YAML files, dotted key=value overrides, and already-built mappings into one unstructured
# config. A mapping spec lets a caller merge values it computed itself (one cell of a sweep matrix
# resolved at runtime, say) under the same precedence rule — later specs win. A mapping is not a file
# and carries no `_schema:`; it is code, and code is already typed.
def merge_specs(specs: list[Spec]) -> DictConfig:
    return _merge(specs).config


# Check every `_schema:` line against the class the config is actually being loaded as. A claim names
# the class its block was written for; the block's real class comes from walking the schema. They agree
# when the claim is that class or a base of it — a base states a subset of the fields, which is exactly
# what a shared fragment does.
def _check_claims(schema: type, claims: tuple[Claim, ...]) -> None:
    for claim in claims:
        target = field_schema(schema, claim.node)
        declared = resolve_schema(claim.schema)
        if not issubclass(target, declared):
            where = f"`{claim.node}`" if claim.node else "the top level"
            raise ValueError(
                f"config file {claim.source!r} says it fills {claim.schema}, but it is being merged onto "
                f"{where} of {schema_name(schema)}, which is {schema_name(target)}"
            )


# Merge `specs` (YAML files and/or dotted key=value overrides) onto `schema`, in order (list a file
# before the overrides that should win over it). Returns a fully-populated schema instance. Raises
# ValueError if a file names the wrong class or any leaf is left unset, FileNotFoundError for a bad
# spec, and OmegaConf errors for unknown keys / type mismatches.
def load_config[T](schema: type[T], specs: list[Spec]) -> T:
    check_schema(cast(type, schema))
    merged_specs = _merge(specs)
    _check_claims(cast(type, schema), merged_specs.claims)
    merged = OmegaConf.merge(OmegaConf.structured(schema), merged_specs.config)
    missing = _missing_fields(cast(DictConfig, merged))
    if missing:
        raise ValueError(f"{schema.__name__} is missing required field(s): {', '.join(missing)}")
    return cast(T, OmegaConf.to_object(merged))


# Return `key` (dotted paths allowed) from the merged specs, or None — without validation, so a caller
# can pick a schema from a value inside the config before loading it strictly (a schema chosen by
# `method`, a cell named by a matrix). Accepts the same specs as load_config.
def peek(args: list[Spec], key: str) -> Any:
    return OmegaConf.select(merge_specs(args), key, default=None)


# The class a config file was written against, without loading it: the top-level `_schema:` line,
# imported. For an entry point that dispatches on the config it was handed.
def schema_of(path: str) -> type:
    claims = compose(path).claims
    root = next((c for c in claims if c.node == ""), None)
    if root is None:  # compose() rejects a file with no top-level `_schema:`, so this cannot happen
        raise ValueError(f"config file {path!r} declares no top-level `_schema:`")
    return resolve_schema(root.schema)

# Typed, all-fields-required config loading — merge YAML onto a dataclass schema.
#
# A schema is a @dataclass whose every leaf defaults to omegaconf.MISSING, so a config must set each
# field explicitly (nothing is silently filled in). load_config merges one or more specs (YAML files
# and/or dotted key=value overrides) onto that schema and returns a fully populated instance.
#
# Three rules, enforced at load time:
#   * every config FILE — and every nested BLOCK in one that fills a config class — names that class
#     (`_schema: <dotted.path>`), and it must be the class it is being merged onto, or a base of it. So
#     a fragment cannot be mounted at a block it was not written for, pointing a script at the wrong
#     config fails by class name instead of by an unknown key three levels down, and a hierarchical
#     config says at every level what it is filling instead of only at the top;
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

from .config import Block, Claim, compose
from .partials import is_partial
from .schemas import check_schema, field_schema, fields_of, placement, resolve_schema, schema_name

# One config source: a YAML file path, a `dotted.key=value` override, or a ready-made mapping.
type Spec = str | Mapping[str, Any] | DictConfig


class _Merged:
    """The merged specs, the `_schema:` claims the files among them made, and the blocks they wrote."""

    def __init__(self, config: DictConfig, claims: tuple[Claim, ...], blocks: tuple[Block, ...]) -> None:
        self.config = config
        self.claims = claims
        self.blocks = blocks


# Dotted paths of every leaf field still unset, walking the SCHEMA rather than the merged node: a
# partial subtree is allowed to be unset, and only the schema says which subtrees those are. (Merging a
# partial's node onto a complete one promotes the result's runtime type to the partial, so asking the
# node "are you partial?" would answer yes for a config that is genuinely incomplete.)
def _missing_fields(cfg: DictConfig, schema: type, prefix: str = "") -> list[str]:
    if is_partial(schema):  # a layer, not a run: saying nothing is what it is for
        return []
    missing: list[str] = []
    for name, (kind, nested) in fields_of(schema).items():
        if OmegaConf.is_missing(cfg, name):
            missing.append(prefix + name)
            continue
        value = cfg[name]
        if nested is None or value is None:
            continue
        if kind == "group":
            missing.extend(_missing_fields(cast(DictConfig, value), nested, f"{prefix}{name}."))
        else:  # a table: each entry is checked like the group it is
            missing.extend(
                m for key in value for m in _missing_fields(value[key], nested, f"{prefix}{name}.{key}.")
            )
    return missing


# Turn the merged node into schema instances. OmegaConf's own `to_object` cannot do this: it raises on
# any MISSING leaf inside a structured node, even one a partial is entitled to leave unset. So the walk
# is ours — an unset field is simply not passed, and the class's own MISSING default stands.
def _instantiate[T](node: DictConfig, schema: type[T]) -> T:
    kwargs: dict[str, Any] = {}
    for name, (kind, nested) in fields_of(cast(type, schema)).items():
        if OmegaConf.is_missing(node, name):
            continue
        value = node[name]
        if nested is None or value is None:
            kwargs[name] = OmegaConf.to_object(value) if OmegaConf.is_config(value) else value
        elif kind == "group":
            kwargs[name] = _instantiate(cast(DictConfig, value), nested)
        else:
            kwargs[name] = {key: _instantiate(value[key], nested) for key in value}
    return schema(**kwargs)


def _merge(specs: list[Spec]) -> _Merged:
    merged = OmegaConf.create()
    claims: list[Claim] = []
    blocks: list[Block] = []
    for spec in specs:
        if isinstance(spec, Mapping | DictConfig):
            merged = OmegaConf.merge(merged, spec)
        elif Path(spec).is_file():
            composed = compose(spec)
            claims.extend(composed.claims)
            blocks.extend(composed.blocks)
            merged = OmegaConf.merge(merged, composed.config)
        elif "=" in spec:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([spec]))
        else:
            raise FileNotFoundError(
                f"config spec {spec!r} is neither a file nor a key=value override"
            )
    return _Merged(cast(DictConfig, merged), tuple(claims), tuple(blocks))


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
            where = f"`{'.'.join(claim.node)}`" if claim.node else "the top level"
            raise ValueError(
                f"config file {claim.source!r} says it fills {claim.schema}, but it is being merged onto "
                f"{where} of {schema_name(schema)}, which is {schema_name(target)}"
            )


# Hold every BLOCK that fills a config class to the same rule the top of a file is held to: name the
# class. A file already says what it fills; a nested block is a second config class in the same file and
# is just as much written-against-a-class, so it says so too — which is what makes a hierarchical config
# readable on its own and what makes moving or renaming a nested class break its configs loudly.
#
# Only a GROUP is required to. A table's entries are config classes too, but which class was already
# fixed, once, by the table's own declaration — every entry of a `dict[Method, CellPart]` is a CellPart
# and cannot be anything else, so an entry naming it adds a line that can be wrong and never informative.
# A table itself and a leaf have no class to name at all, and a claim on either is already an error
# (field_schema). An unknown node is left alone: OmegaConf's struct check reports it far better than a
# missing-`_schema` complaint would.
def _check_declared(schema: type, blocks: tuple[Block, ...], claims: tuple[Claim, ...]) -> None:
    declared = {claim.node for claim in claims}
    for block in blocks:
        if block.node in declared:
            continue
        kind, cls = placement(schema, block.node)
        if kind != "group" or cls is None:
            continue
        raise ValueError(
            f"config file {block.source!r} writes the block `{'.'.join(block.node)}`, which fills the "
            f"config class {schema_name(cls)}, without saying so: add `_schema: {schema_name(cls)}` at "
            f"the top of that block. Every mapping that fills a config class names the class it fills."
        )


# Merge `specs` (YAML files and/or dotted key=value overrides) onto `schema`, in order (list a file
# before the overrides that should win over it). Returns a fully-populated schema instance. Raises
# ValueError if a file names the wrong class or any leaf is left unset, FileNotFoundError for a bad
# spec, and OmegaConf errors for unknown keys / type mismatches.
def load_config[T](schema: type[T], specs: list[Spec]) -> T:
    check_schema(cast(type, schema))
    merged_specs = _merge(specs)
    _check_claims(cast(type, schema), merged_specs.claims)
    _check_declared(cast(type, schema), merged_specs.blocks, merged_specs.claims)
    merged = OmegaConf.merge(OmegaConf.structured(schema), merged_specs.config)
    missing = _missing_fields(cast(DictConfig, merged), cast(type, schema))
    if missing:
        raise ValueError(f"{schema.__name__} is missing required field(s): {', '.join(missing)}")
    return _instantiate(cast(DictConfig, merged), schema)


# Return `key` (dotted paths allowed) from the merged specs, or None — without validation, so a caller
# can pick a schema from a value inside the config before loading it strictly (a schema chosen by
# `method`, a cell named by a matrix). Accepts the same specs as load_config.
def peek(args: list[Spec], key: str) -> Any:
    return OmegaConf.select(merge_specs(args), key, default=None)


# The class a config file was written against, without loading it: the top-level `_schema:` line,
# imported. For an entry point that dispatches on the config it was handed.
def schema_of(path: str) -> type:
    claims = compose(path).claims
    root = next((c for c in claims if not c.node), None)
    if root is None:  # compose() rejects a file with no top-level `_schema:`, so this cannot happen
        raise ValueError(f"config file {path!r} declares no top-level `_schema:`")
    return resolve_schema(root.schema)

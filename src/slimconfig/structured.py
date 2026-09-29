# Typed, all-fields-required config loading: merge YAML onto a dataclass schema.
#
# load_config merges specs (YAML files, `key=value` overrides, mappings) onto a schema whose leaves
# default to MISSING and returns a fully populated instance. Four rules, enforced at load time:
#   * every file, block and table that fills a config class names it (`key > path:`, or `dict[K, C]`);
#   * every key a file sets is a field of the class it fills (or an Enum member, under a mapping);
#   * every leaf ends up set (a null must be written as null, an empty collection as []);
#   * every value is the type its field declares, all the way down (OmegaConf only checks scalars).
# Files are read via slimconfig.config.compose, so any mapping may carry `_default: <path>`.

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Any, cast, get_args, get_origin

from omegaconf import DictConfig, OmegaConf

from .config import Claim, Composed, Key, compose
from .partials import is_partial
from .schemas import Schema, declaration_name, key_name, optional, value_error

# One config source: a YAML file path, a `dotted.key=value` override, or a ready-made mapping.
type Spec = str | Mapping[str, Any] | DictConfig


# Yield (prefix, schema, node) for each group and table entry present in `cfg`; unset/null are skipped.
def _nested(cfg: DictConfig, schema: Schema, prefix: str) -> Iterator[tuple[str, Schema, DictConfig]]:
    for name, held in schema.fields.items():
        if held.cls is None or OmegaConf.is_missing(cfg, name) or cfg[name] is None:
            continue
        if held.kind == "group":
            yield f"{prefix}{name}.", Schema(held.cls), cast(DictConfig, cfg[name])
        else:
            for key in cfg[name]:
                yield f"{prefix}{name}.{key}.", Schema(held.cls), cfg[name][key]


# Dotted paths of every unset leaf. Walks the schema, not the node: merging promotes a node's type to
# the partial, so only the schema knows which subtrees are allowed to be unset.
def _missing_fields(cfg: DictConfig, schema: Schema, prefix: str = "") -> list[str]:
    if is_partial(schema.cls):  # a partial may leave anything unset
        return []
    missing = [prefix + name for name in schema.fields if OmegaConf.is_missing(cfg, name)]
    for at, nested, node in _nested(cfg, schema, prefix):
        missing.extend(_missing_fields(node, nested, at))
    return missing


# Every leaf value not of its declared type, e.g. `field.path[key] is not a str: {...}`. Catches what
# OmegaConf does not check: list element types and dict value types.
def _wrong_values(cfg: DictConfig, schema: Schema, prefix: str = "") -> list[str]:
    hints = schema.hints
    wrong: list[str] = []
    for name, held in schema.fields.items():
        if held.cls is not None or OmegaConf.is_missing(cfg, name):
            continue
        value = cfg[name]
        plain = OmegaConf.to_object(value) if OmegaConf.is_config(value) else value
        problem = value_error(plain, hints[name])
        if problem is not None:
            wrong.append(f"{prefix}{name}{problem}")
    for at, nested, node in _nested(cfg, schema, prefix):
        wrong.extend(_wrong_values(node, nested, at))
    return wrong


# Build schema instances from the merged node. Not `OmegaConf.to_object`, which raises on the MISSING
# leaves a partial may leave unset; here an unset field is just not passed.
def _instantiate[T](node: DictConfig, schema: type[T]) -> T:
    kwargs: dict[str, Any] = {}
    for name, held in Schema(cast(type, schema)).fields.items():
        if OmegaConf.is_missing(node, name):
            continue
        value = node[name]
        if held.cls is None or value is None:
            kwargs[name] = OmegaConf.to_object(value) if OmegaConf.is_config(value) else value
        elif held.kind == "group":
            kwargs[name] = _instantiate(cast(DictConfig, value), held.cls)
        else:
            kwargs[name] = {key: _instantiate(value[key], held.cls) for key in value}
    return schema(**kwargs)


# Compose one spec: a file through the YAML layer, an override or mapping as-is.
def _composed(spec: Spec) -> Composed:
    if isinstance(spec, Mapping | DictConfig):
        return Composed.of(spec)
    if Path(spec).is_file():
        return compose(spec)
    if "=" in spec:
        return Composed.of(OmegaConf.from_dotlist([spec]), source=spec)
    raise FileNotFoundError(f"config spec {spec!r} is neither a file nor a key=value override")


# Merge several specs into one Composed, as if they were one file.
def _merge(specs: list[Spec]) -> Composed:
    return reduce(Composed.merge, map(_composed, specs), Composed.empty())


# Merge specs into one unvalidated config; later specs win. A mapping spec declares no class.
def merge_specs(specs: list[Spec]) -> DictConfig:
    return _merge(specs).config


# Check each claimed class is the class at its node, or a base of it (a shared fragment). The shape
# (table vs group) is checked first, since mixing them up is a wrong location, not a class mismatch.
def _check_claims(schema: Schema, claims: tuple[Claim, ...]) -> None:
    for claim in claims:
        declared = Schema.declared(claim.schema)
        where = f"`{'.'.join(claim.node)}`" if claim.node else "the top level"
        if declared.key is None:
            target = schema.require(claim.node)  # raises on a table, an unknown key, or a leaf
        else:
            target = _table_at(schema, claim, where, declared.key)
        if not issubclass(target.cls, declared.schema.cls):
            raise ValueError(
                f"config file {claim.source!r} says it fills {claim.schema}, but it is being merged onto "
                f"{where} of {schema.name}, which is {target.name}"
            )


# The entry class of the table a `dict[K, C]` claim was made at — or why that node is not one.
def _table_at(schema: Schema, claim: Claim, where: str, key: type) -> Schema:
    at = schema.at(claim.node)
    if at.kind in ("unknown", "value"):
        schema.require(claim.node)  # raises: no such field / that node is a leaf
    if at.kind != "table" or at.cls is None or at.key is None:
        one = Schema(cast(type, at.cls)).name
        raise ValueError(
            f"config file {claim.source!r} says {where} is a table ({claim.schema}), but {where} of "
            f"{schema.name} is ONE {one}, not several keyed by anything: `{'.'.join(claim.node)} > {one}`"
        )
    if key is not at.key:  # identity: a same-named type from another module is different
        raise ValueError(
            f"config file {claim.source!r} says {where} is keyed by {key_name(key)}, but {schema.name}."
            f"{'.'.join(claim.node)} is keyed by {key_name(at.key)}: "
            f"`{'.'.join(claim.node)} > {declaration_name(at.cls, at.key)}`"
        )
    return Schema(at.cls)


# Require every group and table block in a file to name its class; table entries name nothing. An empty
# table is exempt: `datasets: {}` says there are none, and there is nothing under it to check.
def _check_declared(schema: Schema, keys: tuple[Key, ...], claims: tuple[Claim, ...]) -> None:
    declared = {claim.node for claim in claims}
    filled = {k.node[:n] for k in keys for n in range(1, len(k.node))}
    for block in (k for k in keys if k.mapping and k.node not in declared):
        at = schema.at(block.node)
        if at.cls is None or at.kind not in ("group", "table"):
            continue
        if at.kind == "table" and block.node not in filled:
            continue
        spelled = declaration_name(at.cls, at.key)
        what = "block" if at.kind == "group" else "table"
        raise ValueError(
            f"config file {block.source!r} writes the {what} `{'.'.join(block.node)}`, which fills the "
            f"config class {Schema(at.cls).name}, without saying so: write its key as "
            f"`{block.node[-1]} > {spelled}:`. Every mapping that fills a config class names the class "
            f"it fills."
        )


# Reject any key that is not a field of the class it lands on, before merging, so the error names the
# file that wrote it. Keys inside a leaf's value are left to `_wrong_values`.
def _check_keys(schema: Schema, keys: tuple[Key, ...]) -> None:
    for key in keys:
        walked = list(schema.walk(key.node))
        if any(where.kind == "value" for _, where in walked[:-1]):
            continue  # inside a leaf's own value
        if not walked or walked[-1][1].kind != "unknown":
            continue
        bad = walked[-1][0]  # the keys walked, ending at the one that is not there
        owner = Schema(schema.at(bad[:-1]).cls or schema.cls).name
        raise ValueError(
            f"{key.source!r} sets `{'.'.join(bad)}`, which is not a field of {owner} — every key a "
            f"config sets is a field of the class it fills"
        )


# Resolve a word to the Enum member it names, by value or name. OmegaConf resolves by name only (so
# `flux.1-dev` fails, and nested mappings resolve nothing); an unknown word is left for OmegaConf to reject.
def _member(word: Any, enum: type) -> Any:
    members = {name: member for member in cast(Any, enum) for name in (member.value, member.name)}
    return members.get(word, word) if isinstance(word, str) else word


def _by_member(mapping: Any, key: type) -> Any:
    if not isinstance(mapping, Mapping):
        return mapping
    return {_member(k, key): v for k, v in mapping.items()}


# Resolve Enum words everywhere in one leaf its annotation allows: the leaf, list items, dict keys and values.
def _leaf_enums(value: Any, annotation: Any) -> Any:
    ann = optional(annotation)
    origin = get_origin(ann)
    if isinstance(ann, type) and issubclass(ann, Enum):
        return _member(value, ann)
    if origin is list and isinstance(value, list):
        return [_leaf_enums(v, get_args(ann)[0]) for v in value]
    if origin is not dict or not isinstance(value, Mapping):
        return value
    key, held = get_args(ann)
    return {k: _leaf_enums(v, held) for k, v in _by_member(value, key).items()}


# Resolve Enum words across the whole config, walked against the schema, before merging onto it.
def _enum_words(node: Any, schema: Schema) -> Any:
    if not isinstance(node, Mapping):
        return node
    out = dict(node)
    for name, held in schema.fields.items():
        if name not in out or out[name] is None:
            continue
        if held.kind == "group":
            out[name] = _enum_words(out[name], Schema(cast(type, held.cls)))
        elif held.kind == "table":
            entries = _by_member(out[name], cast(type, held.key))
            if isinstance(entries, Mapping):
                out[name] = {k: _enum_words(v, Schema(cast(type, held.cls))) for k, v in entries.items()}
        else:
            out[name] = _leaf_enums(out[name], schema.hints[name])
    return out


# Merge `specs` onto `schema` in order (later wins) and return a fully populated instance. Raises
# TypeError for a bad schema, ValueError for a broken rule, FileNotFoundError for a bad spec.
def load_config[T](schema: type[T], specs: list[Spec]) -> T:
    root = Schema(cast(type, schema))
    root.check()
    composed = _merge(specs)
    _check_claims(root, composed.claims)
    _check_declared(root, composed.keys, composed.claims)
    _check_keys(root, composed.keys)
    named = OmegaConf.create(_enum_words(OmegaConf.to_container(composed.config, resolve=False), root))
    merged = cast(DictConfig, OmegaConf.merge(OmegaConf.structured(schema), named))
    missing = _missing_fields(merged, root)
    if missing:
        raise ValueError(f"{root.cls.__name__} is missing required field(s): {', '.join(missing)}")
    wrong = _wrong_values(merged, root)
    if wrong:
        raise ValueError(f"{root.cls.__name__} holds value(s) the schema does not declare: {'; '.join(wrong)}")
    return _instantiate(merged, schema)


# Return `key` from the merged specs (or None) without validation, e.g. to pick a schema before loading.
def peek(args: list[Spec], key: str) -> Any:
    return OmegaConf.select(merge_specs(args), key, default=None)


# The class a config file names on its `_ >` line, without loading it (a table file gives its entry class).
def schema_of(path: str) -> type:
    root = next(c for c in compose(path).claims if not c.node)  # compose() requires the file's own `_ >` line
    return Schema.declared(root.schema).schema.cls

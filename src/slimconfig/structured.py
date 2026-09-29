# Typed, all-fields-required config loading — merge YAML onto a dataclass schema.
#
# A schema is a @dataclass whose every leaf defaults to omegaconf.MISSING, so a config must set each
# field explicitly (nothing is silently filled in). load_config merges one or more specs (YAML files
# and/or dotted key=value overrides) onto that schema and returns a fully populated instance.
#
# Four rules, enforced at load time:
#   * every config FILE — and every nested BLOCK or TABLE in one that fills a config class — names that
#     class on its own key (`<key> > <dotted.path>:`, `_ > <dotted.path>:` for the file itself, or
#     `<key> > dict[<key type>, <dotted.path>]:` for a table, naming its entry class once for all of
#     them), and it must be the class it is being merged onto, or a
#     base of it. So a fragment cannot be mounted at a block it was not written for, pointing a script at
#     the wrong config fails by class name instead of by an unknown key three levels down, and a
#     hierarchical config says at every level what it is filling instead of only at the top;
#   * every key a file sets is REGISTERED — a field of the class that file is filling, or, under a
#     table or a mapping leaf, a member of the Enum that mapping is keyed by (a key type that is not an
#     Enum is rejected at the `class` statement, so there is no third case). Checked file by file, so a
#     typo is reported against the file that wrote it and not against the merged config, which has no
#     memory of which of a `_default:` chain a key came from;
#   * every leaf must end up set — an unset MISSING leaf raises (a nullable field that is "off" must
#     still be present as null; an empty collection must be written out as []);
#   * every value is the type its field promised, all the way down — OmegaConf checks a scalar but lets
#     a `list[str]` hold a mapping, and a type hint that is not kept is worse than none.
#
# YAML files are read via slimconfig.config.compose, so any mapping in one — at any depth — may carry
# `_default: <path>` to start from a shared file, and the file that names it wins on top.

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


# The config classes nested in a loaded `cfg`: one (prefix, schema, node) per group and per table ENTRY
# actually there — the three-way branch (leaf / group / table) that every walk over a loaded config
# would otherwise repeat. A field left unset, or set to null, has no node below it to walk.
def _nested(cfg: DictConfig, schema: Schema, prefix: str) -> Iterator[tuple[str, Schema, DictConfig]]:
    for name, held in schema.fields.items():
        if held.cls is None or OmegaConf.is_missing(cfg, name) or cfg[name] is None:
            continue
        if held.kind == "group":
            yield f"{prefix}{name}.", Schema(held.cls), cast(DictConfig, cfg[name])
        else:
            for key in cfg[name]:
                yield f"{prefix}{name}.{key}.", Schema(held.cls), cfg[name][key]


# Dotted paths of every leaf field still unset, walking the SCHEMA rather than the merged node: a
# partial subtree is allowed to be unset, and only the schema says which subtrees those are. (Merging a
# partial's node onto a complete one promotes the result's runtime type to the partial, so asking the
# node "are you partial?" would answer yes for a config that is genuinely incomplete.)
def _missing_fields(cfg: DictConfig, schema: Schema, prefix: str = "") -> list[str]:
    if is_partial(schema.cls):  # a layer, not a run: saying nothing is what it is for
        return []
    missing = [prefix + name for name in schema.fields if OmegaConf.is_missing(cfg, name)]
    for at, nested, node in _nested(cfg, schema, prefix):
        missing.extend(_missing_fields(node, nested, at))
    return missing


# Every value that is not the type its field promised, as `field.path[key] is not a str: {...}`. The
# same walk as _missing_fields, asking the other question — and asking it of the LEAVES, since a group
# and a table are what the schema itself is made of and OmegaConf has already held the config to them.
# What is left is what OmegaConf does not check: the element type of a list, the value type of a dict.
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


# Turn the merged node into schema instances. OmegaConf's own `to_object` cannot do this: it raises on
# any MISSING leaf inside a structured node, even one a partial is entitled to leave unset. So the walk
# is ours — an unset field is simply not passed, and the class's own MISSING default stands.
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


# One spec, composed on its own. A FILE goes through the whole YAML layer (its `_default:` chain, its
# claims, its blocks); anything else is code and carries none of those.
def _composed(spec: Spec) -> Composed:
    if isinstance(spec, Mapping | DictConfig):
        return Composed.of(spec)
    if Path(spec).is_file():
        return compose(spec)
    if "=" in spec:
        return Composed.of(OmegaConf.from_dotlist([spec]), source=spec)
    raise FileNotFoundError(f"config spec {spec!r} is neither a file nor a key=value override")


# Several specs merged into one, reported exactly as compose reports one file: a Composed is "a config,
# every claim made anywhere in it, and every block written in it", and that is as true of five specs as
# of one — a launch that names several files is one config assembled from all of them.
def _merge(specs: list[Spec]) -> Composed:
    return reduce(Composed.merge, map(_composed, specs), Composed.empty())


# Merge YAML files, dotted key=value overrides, and already-built mappings into one unstructured
# config. A mapping spec lets a caller merge values it computed itself (one cell of a sweep matrix
# resolved at runtime, say) under the same precedence rule — later specs win. A mapping is not a file
# and declares no class; it is code, and code is already typed.
def merge_specs(specs: list[Spec]) -> DictConfig:
    return _merge(specs).config


# Check every declaration against the class the config is actually being loaded as. A claim names
# the class its block was written for; the block's real class comes from walking the schema. They agree
# when the claim is that class or a base of it — a base states a subset of the fields, which is exactly
# what a shared fragment does.
#
# A claim also says WHICH SHAPE it is: `dict[K, C]` for a table, a bare class for one of it. That half is
# checked first and against the node alone, since a table and a group are the same mapping on the page
# and a file that has them mixed up is not reporting a class mismatch — it thinks it is somewhere else.
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
    if key is not at.key:  # the same NAME from another module is another type, and would key nothing
        raise ValueError(
            f"config file {claim.source!r} says {where} is keyed by {key_name(key)}, but {schema.name}."
            f"{'.'.join(claim.node)} is keyed by {key_name(at.key)}: "
            f"`{'.'.join(claim.node)} > {declaration_name(at.cls, at.key)}`"
        )
    return Schema(at.cls)


# Hold every BLOCK that fills a config class to the same rule the top of a file is held to: name the
# class. A file already says what it fills; a nested block is a second config class in the same file and
# is just as much written-against-a-class, so it says so too — which is what makes a hierarchical config
# readable on its own and what makes moving or renaming a nested class break its configs loudly.
#
# A GROUP and a TABLE are both held to it, and this is the whole reason a table is spelled `dict[K, C]`:
# on the page they are the same mapping, so a declaration that could not tell them apart would leave a
# reader unable to say whether the keys below are the fields of one C or the names of several. A table
# names its entry class ONCE, at the table, for however many entries it has; its ENTRIES name nothing,
# since which class an entry has was fixed by the table and an entry repeating it adds a line that can
# be wrong and never informative. An EMPTY table is exempt — `datasets: {}` is how a config says it has
# none, and there is nothing under it to be read against a class. A leaf has no class to name at all,
# and a claim on one is already an error (Schema.require). An unknown node is left alone: `_check_keys`
# has already reported it as what it is.
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


# Every key every spec set must be a field of the class it lands on. OmegaConf's struct check catches
# an unknown key too, but only once everything is merged — and by then a key merged up a `_default:`
# chain has no source left to name, which is exactly when a config is hardest to fix. Here each key is
# still attached to the file (or the `key=value` override) that wrote it.
#
# Keys BELOW a leaf are not nodes of the schema at all: `metrics: {psnr: [...]}` on a
# `dict[Metric, list[str]]` field writes a mapping the schema has nothing to say about beyond the leaf
# itself, so the walk is only asked about a key whose whole path so far landed on groups and entries.
# Those keys are not unchecked, though — the leaf's own key type is an Enum, and `_wrong_values` holds
# the mapping to it.
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


# A word an Enum names, as the MEMBER it names — the value first, since that is how a config file spells
# an Enum. OmegaConf resolves a string to a member by NAME only, and a member's name is a python
# identifier, so `flux.1-dev` places nowhere; inside a nested mapping (`dict[K, dict[K2, V]]`) no string
# places at all. Resolving the member here is what makes ONE spelling work everywhere in a config — as a
# key, as a scalar, in a list. A word no member names is left alone, for OmegaConf to reject with the
# members it could have been.
def _member(word: Any, enum: type) -> Any:
    members = {name: member for member in cast(Any, enum) for name in (member.value, member.name)}
    return members.get(word, word) if isinstance(word, str) else word


def _by_member(mapping: Any, key: type) -> Any:
    if not isinstance(mapping, Mapping):
        return mapping
    return {_member(k, key): v for k, v in mapping.items()}


# The same, everywhere in one LEAF its annotation says an Enum can be: the keys of a mapping, the values
# of one, the elements of a list, the leaf itself.
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


# Every word in `node` that an Enum names, as that member — the whole config, walked against the schema,
# before it is merged onto it.
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


# Merge `specs` (YAML files and/or dotted key=value overrides) onto `schema`, in order (list a file
# before the overrides that should win over it). Returns a fully-populated schema instance. Raises
# TypeError if the schema itself cannot be filled from YAML, ValueError if a spec names the wrong class
# or sets a key that is not a field, if any leaf is left unset, or if any value is not the type its
# field declared, FileNotFoundError for a bad spec, and OmegaConf errors for a scalar of the wrong type.
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


# Return `key` (dotted paths allowed) from the merged specs, or None — without validation, so a caller
# can pick a schema from a value inside the config before loading it strictly (a schema chosen by
# `method`, a cell named by a matrix). Accepts the same specs as load_config.
def peek(args: list[Spec], key: str) -> Any:
    return OmegaConf.select(merge_specs(args), key, default=None)


# The class a config file was written against, without loading it: the file's own `_ > <class>:` line,
# imported. For an entry point that dispatches on the config it was handed. A file that fills a table
# answers with its ENTRY class — the only class it names.
def schema_of(path: str) -> type:
    root = next(c for c in compose(path).claims if not c.node)  # compose() requires the file's own `_ >` line
    return Schema.declared(root.schema).schema.cls

# slimconfig.config — the YAML layer: read a file, compose its `_default:` chain, and record the class
# each mapping declares and the keys each file set.
#
#   <key> > pkg.Class:                   the mapping under `<key>` fills that class (required of every block)
#   <key> > dict[pkg.Enum, pkg.Class]:   a table: each entry fills the class, and entries name nothing
#   _ > pkg.Class:                       the file itself; required at the top of every file, takes no value
#   _default: <path>                     the file this mapping starts from; its own keys merge on top and win
#
# `_default` takes ONE path, not a list (combine independent files at the launch instead), resolved against
# the CWD (an absolute path resolves to itself). It works in nested blocks too; cycles are errors.
#
# Importing this module registers two OmegaConf resolvers:
#   * ${now:<strftime>}          — the load time, one timestamp per process.
#   * ${from_yaml:<path>,<key>}  — one value read out of another config file.

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["Claim", "Composed", "Key", "compose", "load_mapping_yaml", "load_yaml"]

# The declaration arrow, the file's own name, and the inheritance key; none reach the merged config.
ARROW = ">"
ROOT_NAME = "_"
DEFAULT_KEY = "_default"

# `<name> > <declaration>`. A right-hand side that is not a dotted path or `dict[...]` leaves the key alone,
# since a table's entry keys are data and may hold a `>`.
_DECLARED = re.compile(r"^(?P<name>[^>]+?)\s*>\s*(?P<declaration>[\w.]+|dict\[[^\]]*\])$")

# `${now:<strftime>}`; `use_cache=True` gives one consistent timestamp per process.
OmegaConf.register_new_resolver(
    "now", lambda fmt: datetime.now().strftime(fmt), replace=True, use_cache=True
)

# Tells "not found" apart from a real `null` at the key.
_NOT_FOUND = object()


def _select_from_yaml(path: str, key: str) -> Any:
    cfg = load_mapping_yaml(path.strip())
    val = OmegaConf.select(cfg, key.strip(), default=_NOT_FOUND, throw_on_missing=True)
    if val is _NOT_FOUND:
        raise ValueError(f"${{from_yaml:{path},{key}}}: {path!r} has no key {key.strip()!r}")
    return val


OmegaConf.register_new_resolver("from_yaml", _select_from_yaml, replace=True, use_cache=True)


class Claim(NamedTuple):
    """One declaration: the config class `node` was written against, and the file that said so."""

    node: tuple[str, ...]  # keys from the config root (() = the root itself)
    schema: str            # the dotted import path the file named
    source: str            # the file it was read from, for the error message


class Key(NamedTuple):
    """One key a spec set: where it sits, who set it, and whether its value was a nested mapping.

    Recorded during the walk so a mistake can be reported against the file that made it.
    """

    node: tuple[str, ...]  # keys from the config root (never () — that is the file)
    source: str            # the file (or override) that set it, for the error message
    mapping: bool          # the value was a nested mapping


class Composed(NamedTuple):
    """A composed config, every declaration made anywhere in it, and every key set in it."""

    config: DictConfig
    claims: tuple[Claim, ...]
    keys: tuple[Key, ...]

    # Nothing composed yet: the identity of `merge`.
    @classmethod
    def empty(cls) -> Composed:
        return cls(OmegaConf.create(), (), ())

    # A mapping that is not a file (an override or computed values): no claims, and no key is a block.
    # Keys are recorded only when `source` is given, so an override typo is still reported.
    @classmethod
    def of(cls, config: Mapping[str, Any] | DictConfig, source: str | None = None) -> Composed:
        node = cast(DictConfig, OmegaConf.create(config))
        return cls(node, (), tuple(_keys_in(node, (), source)) if source else ())

    # `other` on top of this one; later wins, key by key.
    def merge(self, other: Composed) -> Composed:
        return Composed(
            cast(DictConfig, OmegaConf.merge(self.config, other.config)),
            self.claims + other.claims,
            self.keys + other.keys,
        )


# Every key of a mapping, at every depth, as `source` set them; none of them a block (see `of`).
def _keys_in(node: DictConfig, at: tuple[str, ...], source: str) -> Iterator[Key]:
    raw = cast(dict, OmegaConf.to_container(node, resolve=False))
    for key, value in raw.items():
        child = (*at, str(key))
        yield Key(child, source, False)
        if isinstance(value, dict):
            yield from _keys_in(cast(DictConfig, node[key]), child, source)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Config {path} is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(cfg).__name__})")
    return cfg


def _load_one(path: Path) -> DictConfig:
    try:
        loaded = OmegaConf.load(path)
    except OSError as e:
        # OmegaConf raises OSError("Invalid loaded object type") for a top-level scalar; report it as a
        # non-mapping like any other. A real IO error (e.g. FileNotFoundError) propagates.
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


class _Composer:
    """One composition in progress: the claims, keys and open `_default:` chain accumulated over the walk."""

    def __init__(self) -> None:
        self.claims: list[Claim] = []
        self.keys: list[Key] = []
        self.visiting: tuple[Path, ...] = ()  # the `_default:` chain currently open, outermost first

    # One config file, composed at `node`. Declarations are stripped from keys before merging, so a file
    # overriding a field of an inherited block need not repeat the block's class.
    def file(self, path: Path, node: tuple[str, ...]) -> DictConfig:
        if path in self.visiting:
            chain = " -> ".join(str(p) for p in (*self.visiting, path))
            raise ValueError(f"`{DEFAULT_KEY}` cycle detected: {chain}")
        raw = cast(dict, OmegaConf.to_container(_load_one(path), resolve=False))
        source = str(path)
        self._root(raw, node, source)
        loaded = cast(DictConfig, OmegaConf.create(self._declared(raw, node, source)))
        outer, self.visiting = self.visiting, (*self.visiting, path)
        try:
            return self.mapping(loaded, node, source)
        finally:
            self.visiting = outer

    # Pop the file's required `_ > <class>:` line and record it against `node`.
    def _root(self, raw: dict, node: tuple[str, ...], source: str) -> None:
        for key in list(raw):
            declared = _DECLARED.match(str(key))
            if declared is not None and declared["name"] == ROOT_NAME:
                if raw.pop(key) is not None:
                    raise ValueError(
                        f"config file {source!r}: `{key}` names the file's own class and takes no value"
                    )
                self.claims.append(Claim(node, declared["declaration"], source))
                return
        raise ValueError(
            f"config file {source!r} does not say which config class it fills: add a top-level "
            f"`{ROOT_NAME} {ARROW} <dotted.path.To.Class>:`"
        )

    # Record every `<name> > <class>` key at every depth and return the mapping under plain names.
    # A declared key with no value is an empty mapping of that class (`null` stays for a bare key).
    def _declared(self, raw: dict, node: tuple[str, ...], source: str) -> dict:
        out: dict[str, Any] = {}
        for key, value in raw.items():
            declared = _DECLARED.match(str(key))
            name = declared["name"] if declared else str(key)
            child = (*node, name)
            if declared is not None:
                if name == ROOT_NAME:
                    raise ValueError(
                        f"config file {source!r}: `{key}` names the file's own class, and only the top "
                        f"level of a file has one — a block names its class on its own key"
                    )
                self.claims.append(Claim(child, declared["declaration"], source))
                value = {} if value is None else value
            if name in out:
                raise ValueError(f"config file {source!r} writes `{'.'.join(child)}` twice")
            out[name] = self._declared(value, child, source) if isinstance(value, dict) else value
        return out

    # One mapping, composed at `node`: its `_default:` merged underneath, recursively for each child.
    def mapping(self, node_cfg: DictConfig, node: tuple[str, ...], source: str) -> DictConfig:
        # Relative to the CWD; an absolute path resolves to itself.
        parent = self._default(node_cfg, node, source)
        base = self.file((Path.cwd() / parent).resolve(), node) if parent is not None else None

        # Read without resolving: a leaf's `${...}` may only resolve once everything is merged.
        raw = cast(dict, OmegaConf.to_container(node_cfg, resolve=False))
        for key, value in raw.items():
            child = (*node, str(key))
            self.keys.append(Key(child, source, isinstance(value, dict)))
            if isinstance(value, dict):
                node_cfg[key] = self.mapping(cast(DictConfig, node_cfg[key]), child, source)
        return node_cfg if base is None else cast(DictConfig, OmegaConf.merge(base, node_cfg))

    # Pop and validate one mapping's `_default:` path; a list gets a pointer to launch-time merging.
    def _default(self, node_cfg: DictConfig, node: tuple[str, ...], source: str) -> str | None:
        parent = node_cfg.pop(DEFAULT_KEY, None)
        if parent is None or isinstance(parent, str):
            return parent
        where = f"{source!r}" + (f" (under `{'.'.join(node)}`)" if node else "")
        listed = (
            " Pass independent files at the launch instead: they merge in the order given, and the command"
            " line then shows what went in." if isinstance(parent, ListConfig) else ""
        )
        raise ValueError(
            f"config file {where}: `{DEFAULT_KEY}` must be ONE yaml path (a string), got "
            f"{type(parent).__name__}: {parent!r}.{listed}"
        )


# Compose `path` and its `_default:` chain, mounted at `node` (a key tuple, since a table key may hold a
# dot); claims are reported relative to it.
def compose(path: str | Path, node: tuple[str, ...] = ()) -> Composed:
    walk = _Composer()
    config = walk.file(Path(path).resolve(), node)
    return Composed(config, tuple(walk.claims), tuple(walk.keys))


def load_mapping_yaml(path: str) -> DictConfig:
    return compose(path).config

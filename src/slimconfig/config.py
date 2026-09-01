# slimconfig.config — the YAML layer: read a file, compose the `_default:` chain behind it, collect its
# `_schema:` claims.
#
# TWO KEYWORDS, AND THEY WORK AT ANY DEPTH.
#
#   _schema: <dotted.path.To.Class>   what this mapping fills. Required at the top of every config file
#                                     AND at the top of every nested block that fills a config class of
#                                     its own (that is the discipline: a mapping is written against a
#                                     class, and says which). A table's entries are the one exception —
#                                     their class was fixed by the table's declaration, not chosen by
#                                     the entry. load_config checks every claim against the schema and
#                                     is what enforces the requirement, since only it knows the schema;
#                                     this module just records where each one was made.
#   _default: <path>                  the ONE file this mapping starts from. It is composed first and
#                                     the mapping's own keys are merged on top, so the file that writes
#                                     `_default:` always wins over what it inherits.
#
# ONE PARENT, NOT A LIST. A config inherits a starting point and then says how it differs — that is a
# chain, and a chain has an obvious reading order and an obvious winner at every key. A list of parents
# has neither: which of them set the value you are looking at is answered by counting positions in a
# list, in a file that may itself be inherited. Combining independent fragments is still possible and
# still explicit — it just happens at the launch, where several config files may be passed at once and
# the command line shows exactly what went in (slimconfig.runs).
#
# `_default:` inside a nested block is what makes a shared fragment reusable without it having to know
# where it will be mounted: the fragment states its own fields at ITS top level, and the parent says
# where they land.
#
#     # configs/optim/cosine.yaml            # configs/train.yaml
#     _schema: myproject.train.Optim         _schema: myproject.train.TrainConfig
#     lr: 2.0e-4                             model: llama-3-8b
#     warmup_steps: 100                      optim:
#     schedule: cosine                         _default: configs/optim/cosine.yaml
#                                              lr: 1.0e-4          # this file wins
#
# The path resolves relative to the CWD — the project root every script is run from — so one path
# convention holds across a repo wherever the including file lives. An absolute path resolves to itself.
# Cycles are caught and reported as the chain that closed them.
#
# The entry points:
#   * compose         — a YAML path -> (DictConfig, the `_schema` claims in it and everything it inherited)
#   * load_mapping_yaml — the same, keeping only the DictConfig
#   * load_yaml       — plain PyYAML: a YAML file -> dict, no composition, no keywords. For reading a
#                       file that is not a slimconfig config.
# The typed, all-fields-required loader (load_config) lives in structured.py; the class layer the claims
# are checked against lives in schemas.py.
#
# Importing this module also registers two OmegaConf interpolation resolvers (once, process-wide):
#   * ${now:<strftime>}          — stamp a value with the load time.
#   * ${from_yaml:<path>,<key>}  — read one value OUT of another config file, so a config can track a
#                                  value another owns without duplicating it.

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["Block", "Claim", "Composed", "compose", "load_mapping_yaml", "load_yaml"]

# The two reserved keys. Neither reaches the merged config: both are consumed here.
SCHEMA_KEY = "_schema"
DEFAULT_KEY = "_default"

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


class Claim(NamedTuple):
    """One `_schema:` line: the config class `node` was written against, and the file that said so."""

    node: tuple[str, ...]  # the keys from the root of the config being loaded (() = the root itself)
    schema: str            # the dotted import path the file named
    source: str            # the file it was read from, for the error message


class Block(NamedTuple):
    """One nested mapping a file wrote: where it sits, and which file wrote it.

    Collected so load_config can hold every block that fills a config class to the same rule the top of
    a file is held to — name the class. A block is recorded whatever it turns out to fill; only the
    schema can say which ones are config classes, and the schema is not known here.
    """

    node: tuple[str, ...]  # the keys from the root of the config being loaded (never () — that is the file)
    source: str            # the file that wrote it, for the error message


class Composed(NamedTuple):
    """A composed config, every `_schema:` claim made anywhere in it, and every nested block written.

    This is what one config file composes to — and, merged, what a whole launch composes to: several
    files and overlays are still one config, assembled from all of them, and its claims and blocks are
    theirs put together (see `merge`).
    """

    config: DictConfig
    claims: tuple[Claim, ...]
    blocks: tuple[Block, ...]

    # Nothing composed yet: the identity of `merge`.
    @classmethod
    def empty(cls) -> Composed:
        return cls(OmegaConf.create(), (), ())

    # A mapping that is not a config file — a `key=value` override, or values a caller computed. It
    # makes no claims and writes no blocks: those are things a FILE is held to, and this is code.
    @classmethod
    def of(cls, config: Mapping[str, Any] | DictConfig) -> Composed:
        return cls(cast(DictConfig, OmegaConf.create(config)), (), ())

    # `other` on top of this one — later wins, key by key, exactly as OmegaConf merges.
    def merge(self, other: Composed) -> Composed:
        return Composed(
            cast(DictConfig, OmegaConf.merge(self.config, other.config)),
            self.claims + other.claims,
            self.blocks + other.blocks,
        )


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


class _Composer:
    """ONE composition in progress.

    A composition is a recursive walk — down the `_default:` chain of a file, and down the nested blocks
    of each mapping in it — and everything it produces besides the config itself is accumulated across
    the whole walk: the `_schema:` claims, the blocks written, and the chain of files currently open (so
    a cycle can be named). Holding those on the walker keeps them out of every signature: `file` and
    `mapping` take only what differs between calls — WHICH mapping, and WHERE it is being mounted.
    """

    def __init__(self) -> None:
        self.claims: list[Claim] = []
        self.blocks: list[Block] = []
        self.visiting: tuple[Path, ...] = ()  # the `_default:` chain currently open, outermost first

    @classmethod
    def compose(cls, path: Path, node: tuple[str, ...]) -> Composed:
        self = cls()
        return Composed(self.file(path, node), tuple(self.claims), tuple(self.blocks))

    # One config FILE, composed at `node`. Every file must open by naming the class it fills: that is
    # the one thing a reader (and load_config) needs in order to know what the keys below it mean.
    def file(self, path: Path, node: tuple[str, ...]) -> DictConfig:
        if path in self.visiting:
            chain = " -> ".join(str(p) for p in (*self.visiting, path))
            raise ValueError(f"`{DEFAULT_KEY}` cycle detected: {chain}")
        loaded = _load_one(path)
        if SCHEMA_KEY not in loaded:
            raise ValueError(
                f"config file {str(path)!r} does not say which config class it fills: add a top-level "
                f"`{SCHEMA_KEY}: <dotted.path.To.Class>`"
            )
        outer, self.visiting = self.visiting, (*self.visiting, path)
        try:
            return self.mapping(loaded, node, str(path))
        finally:
            self.visiting = outer

    # One MAPPING, composed at `node`: its `_schema:` recorded, its `_default:` merged underneath it,
    # and the same done to each of its children. The mapping's own keys are merged last, so a file
    # always wins over what it inherits — at every depth, not just the top.
    def mapping(self, node_cfg: DictConfig, node: tuple[str, ...], source: str) -> DictConfig:
        self._claim(node_cfg, node, source)

        # The path resolves against the CWD (the project root scripts are run from); an absolute path
        # resolves to itself, since Path("/abs") wins over the cwd join.
        parent = self._default(node_cfg, node, source)
        base = self.file((Path.cwd() / parent).resolve(), node) if parent is not None else None

        # Which children are mappings, read WITHOUT resolving: a leaf may hold a `${...}` that only
        # becomes resolvable once everything is merged, and touching it here would raise on a config
        # that is fine.
        raw = cast(dict, OmegaConf.to_container(node_cfg, resolve=False))
        for key, value in raw.items():
            if isinstance(value, dict):
                child = (*node, str(key))
                self.blocks.append(Block(child, source))
                node_cfg[key] = self.mapping(cast(DictConfig, node_cfg[key]), child, source)
        return node_cfg if base is None else cast(DictConfig, OmegaConf.merge(base, node_cfg))

    # The `_schema:` line of one mapping, popped and recorded.
    def _claim(self, node_cfg: DictConfig, node: tuple[str, ...], source: str) -> None:
        declared = node_cfg.pop(SCHEMA_KEY, None)
        if declared is None:
            return
        if not isinstance(declared, str):
            raise ValueError(
                f"config file {source!r}: `{SCHEMA_KEY}` must be a dotted import path (a string), "
                f"got {type(declared).__name__}"
            )
        self.claims.append(Claim(node, declared, source))

    # The `_default:` path of one mapping, popped and validated. A list is the mistake worth naming: it
    # is what every other config library takes here, and taking one file is the whole point of this one.
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


# Compose `path` and everything its `_default:` chain inherits. `node` is where the file is being
# mounted, as the keys that lead to it: () for a config loaded on its own, ("optim",) for one named
# under an `optim:` block — every claim it makes is reported relative to that, so load_config can check
# it against the right class. Keys and not one dotted string, because a table key may contain a dot.
def compose(path: str | Path, node: tuple[str, ...] = ()) -> Composed:
    return _Composer.compose(Path(path).resolve(), node)


def load_mapping_yaml(path: str) -> DictConfig:
    return compose(path).config

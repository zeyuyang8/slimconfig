# The config classes the tests load. A module, not a conftest fixture, because a YAML names its class by
# dotted import path (`_ > fixtures.TrainConfig:`) and that has to be importable.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from omegaconf import MISSING

from slimconfig import Config, partial_of


# A base of Optim: it states a SUBSET of the fields, which is what a shared fragment does — so a file
# declaring this one may be mounted where an Optim belongs.
@dataclass
class LrOnly(Config):
    lr: float = MISSING


@dataclass
class Optim(LrOnly):
    warmup_steps: int = MISSING


@dataclass
class Data(Config):
    path: str = MISSING


@dataclass
class LooseConfig(Config):
    """A config whose optim block is only the base — so an Optim fragment states more than it has."""

    optim: LrOnly = field(default_factory=LrOnly)


@dataclass
class TrainConfig(Config):
    model: str = MISSING
    tags: list[str] = MISSING
    resume_from: str | None = MISSING
    optim: Optim = field(default_factory=Optim)
    data: Data = field(default_factory=Data)


# A schema with nothing left to choose: every field already has the answer, so the class IS a config
# and a run of it names no file.
@dataclass
class Settled(Config):
    model: str = "llama"
    steps: int = 10


# ── tables and layers ────────────────────────────────────────────────────────


class Stage(StrEnum):
    warmup = "warmup"
    main = "main"


class Name(StrEnum):
    llama = "llama"
    qwen = "qwen"


# A key with a DOT in it, which is what makes a dotted node path ambiguous — the enum registers it, and
# the walk still has to treat it as one key and not two.
class Backbone(StrEnum):
    flux_dev = "flux.1-dev"
    lumina2 = "lumina2"


class Column(StrEnum):
    psnr = "psnr"
    ssim = "ssim"


# One LAYER of a training run: every field of TrainConfig, none of them required.
TrainPart = partial_of(TrainConfig, name="TrainPart")


@dataclass
class MatrixConfig(Config):
    """A run built in layers: a base, refined by the layer its stage names."""

    stage: Stage = MISSING
    base: TrainPart = field(default_factory=TrainPart)
    per_stage: dict[Stage, TrainPart] = MISSING  # a table of a LAYER: an entry may say nothing
    per_model: dict[Name, Data] = MISSING  # a table of a COMPLETE class: every entry is whole


# A table whose keys have a DOT in them — `flux.1-dev` — which makes a dotted node path ambiguous.
@dataclass
class ModelMatrix(Config):
    per_model: dict[Backbone, TrainPart] = MISSING


@dataclass
class Axis(Config):
    low: float = MISSING
    high: float = MISSING


# A run whose own schema has a TABLE, so a layer of it can leave that table unset — which is the one
# shape where the schema says "table here" and the value is not one.
@dataclass
class Search(Config):
    trials: int = MISSING
    axes: dict[Name, Axis] = MISSING


SearchPart = partial_of(Search, name="SearchPart")


@dataclass
class SearchMatrix(Config):
    base: SearchPart = field(default_factory=SearchPart)
    per_stage: dict[Stage, SearchPart] = MISSING


# Leaves that are collections: still one value each, and still held to what they say they hold —
# `weights` is the shape OmegaConf itself does not check, since it will let a float-valued mapping hold
# a list or another mapping.
@dataclass
class Report(Config):
    metrics: dict[Column, list[str]] = MISSING
    weights: dict[Column, float] = MISSING
    labels: list[str | int] = MISSING  # a union of scalars: any member will do, and nothing else


# ── schemas that are not well-formed ─────────────────────────────────────────
#
# A schema is checked AT its `class` statement, so a broken one cannot be written out here — it would
# fail at import and take every test in the suite with it. `declare` builds one at call time instead,
# inside the `pytest.raises` that expects it to be rejected.


def declare(annotation: Any, default: Any = MISSING, name: str = "Declared") -> type:
    """The class `@dataclass class name(Config): field: annotation = default` would be."""
    return dataclass(type(name, (Config,), {"__annotations__": {"field": annotation}, "field": default}))


# Nesting that never terminates, so no finite YAML fills it. The annotation is a forward reference the
# class statement cannot resolve — checking it is postponed until the name exists, and by then it is
# well formed field by field: only `Schema.check`, which walks the nesting, can see the cycle.
@dataclass
class SelfReferential(Config):
    child: SelfReferential = field(default_factory=lambda: None)  # type: ignore[assignment,arg-type]


class NotADataclass(Config):
    """A config class that forgot its decorator: nothing gives it fields."""


@dataclass
class PlainDataclass:
    """A dataclass that never said it was a config — so it is not one, at a field or at a launch."""

    x: int = MISSING

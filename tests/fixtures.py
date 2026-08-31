# The config classes the tests load. A module, not a conftest fixture, because a YAML names its class by
# dotted import path (`_schema: fixtures.TrainConfig`) and that has to be importable.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from omegaconf import MISSING

from slimconfig import partial_of


# A base of Optim: it states a SUBSET of the fields, which is what a shared fragment does — so a file
# declaring this one may be mounted where an Optim belongs.
@dataclass
class LrOnly:
    lr: float = MISSING


@dataclass
class Optim(LrOnly):
    warmup_steps: int = MISSING


@dataclass
class Data:
    path: str = MISSING


@dataclass
class LooseConfig:
    """A config whose optim block is only the base — so an Optim fragment states more than it has."""

    optim: LrOnly = field(default_factory=LrOnly)


@dataclass
class TrainConfig:
    model: str = MISSING
    tags: list[str] = MISSING
    resume_from: str | None = MISSING
    optim: Optim = field(default_factory=Optim)
    data: Data = field(default_factory=Data)


# ── tables and layers ────────────────────────────────────────────────────────


class Stage(StrEnum):
    warmup = "warmup"
    main = "main"


# One LAYER of a training run: every field of TrainConfig, none of them required.
TrainPart = partial_of(TrainConfig, name="TrainPart")


@dataclass
class MatrixConfig:
    """A run built in layers: a base, refined by the layer its stage names."""

    stage: Stage = MISSING
    base: TrainPart = field(default_factory=TrainPart)
    per_stage: dict[Stage, TrainPart] = MISSING  # enum-keyed: the KEYS are checked too
    per_model: dict[str, Data] = MISSING  # a table of a COMPLETE class: every entry is whole


# A table with FREE-FORM string keys whose entries hold a group. That is the shape where a key with a
# dot in it — `flux.1-dev` — makes a dotted node path ambiguous.
@dataclass
class ModelMatrix:
    per_model: dict[str, TrainPart] = MISSING


@dataclass
class Axis:
    low: float = MISSING
    high: float = MISSING


# A run whose own schema has a TABLE, so a layer of it can leave that table unset — which is the one
# shape where the schema says "table here" and the value is not one.
@dataclass
class Search:
    trials: int = MISSING
    axes: dict[str, Axis] = MISSING


SearchPart = partial_of(Search, name="SearchPart")


@dataclass
class SearchMatrix:
    base: SearchPart = field(default_factory=SearchPart)
    per_stage: dict[Stage, SearchPart] = MISSING


# ── schemas that are not well-formed ─────────────────────────────────────────


@dataclass
class MissingFactory:
    """A nested group with nothing to merge onto: check_schema rejects it."""

    optim: Optim = MISSING


@dataclass
class SelfReferential:
    """Nesting that never terminates, so no finite YAML fills it."""

    child: SelfReferential = field(default_factory=lambda: None)  # type: ignore[assignment,arg-type]


class NotADataclass:
    pass

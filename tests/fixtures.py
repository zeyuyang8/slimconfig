# The config classes the tests load. A module, not a conftest fixture, because a YAML names its class by
# dotted import path (`_schema: fixtures.TrainConfig`) and that has to be importable.

from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING


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

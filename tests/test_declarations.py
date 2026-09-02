# What a config class may DECLARE — the rules `Config.__init_subclass__` runs at the `class` statement:
# every field's type hint is one a YAML value can actually have, every nested group carries its
# default_factory, and every group and table entry is itself a config class.
#
# A rejected declaration cannot be written as a module-level class here — it would fail at import and
# take the suite with it — so `fixtures.declare` builds the same class inside the `pytest.raises`.

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import fixtures
import pytest
from omegaconf import MISSING

from slimconfig import Config, Schema, load_config
from slimconfig.schemas import check_declaration

# ── the base class is not optional ───────────────────────────────────────────


def test_a_group_must_be_a_config_class():
    with pytest.raises(TypeError, match="does not subclass slimconfig.Config"):
        fixtures.declare(fixtures.PlainDataclass, field(default_factory=fixtures.PlainDataclass))


def test_a_table_entry_must_be_a_config_class():
    with pytest.raises(TypeError, match="does not subclass slimconfig.Config"):
        fixtures.declare(dict[str, fixtures.PlainDataclass])


def test_a_config_class_is_checked_at_its_class_statement(tmp_path, monkeypatch):
    # Not at the first load that happens to use it: importing the module is what reports the mistake.
    (tmp_path / "broken_schema.py").write_text(
        textwrap.dedent("""
            from dataclasses import dataclass
            from typing import Any
            from omegaconf import MISSING
            from slimconfig import Config

            @dataclass
            class Broken(Config):
                whatever: Any = MISSING
        """),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(TypeError, match="whatever.*says nothing about the value"):
        __import__("broken_schema")


# ── what a field may be declared as ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        (Any, "says nothing about the value"),
        (object, "says nothing about the value"),
        (list, "does not say what it holds.*list\\[str\\]"),
        (dict, "does not say what it holds.*dict\\[str, float\\]"),
        (Path, "does not survive the run snapshot"),
        (tuple[str, ...], "a sequence in a config is a `list"),
        (set[str], "a sequence in a config is a `list"),
        (frozenset[str], "a sequence in a config is a `list"),
        (Literal["a", "b"], "cannot check a Literal"),
        (list[str] | int, "a union may only offer scalars"),
        (fixtures.Data | int, "a union may only offer scalars"),
        (list[Path], "does not survive the run snapshot"),
        (dict[str, Any], "says nothing about the value"),
        (list[fixtures.Data], "key them instead, as a table"),
        (list[dict[str, fixtures.Data]], "is a FIELD of a config class"),
        (dict[float, str], "a config key is a str, an int or an Enum"),
        (complex, "not a type a YAML value can have"),
    ],
)
def test_a_hint_a_config_file_cannot_state_is_rejected(annotation, match):
    with pytest.raises(TypeError, match=match):
        fixtures.declare(annotation)


@pytest.mark.parametrize(
    "annotation",
    [
        str,
        int,
        float,
        bool,
        fixtures.Stage,
        str | None,
        list[str],
        list[str] | None,
        dict[str, float],
        dict[str, list[str]],
        dict[int, str],
        list[list[int]],
        str | int | float,  # a union of scalars: OmegaConf holds it and checks it
        list[str | int] | None,
    ],
)
def test_a_hint_a_config_file_can_state_is_accepted(annotation):
    Schema(fixtures.declare(annotation)).check()


def test_a_group_must_declare_its_default_factory():
    with pytest.raises(TypeError, match="must declare it as its default"):
        fixtures.declare(fixtures.Optim)
    Schema(fixtures.declare(fixtures.Optim, field(default_factory=fixtures.Optim))).check()


def test_an_optional_group_is_still_a_group():
    Schema(fixtures.declare(fixtures.Optim | None, field(default_factory=fixtures.Optim))).check()


def test_a_table_key_is_a_name(tmp_path):
    with pytest.raises(TypeError, match="table keyed by float"):
        fixtures.declare(dict[float, fixtures.Data])
    Schema(fixtures.declare(dict[fixtures.Stage, fixtures.Data])).check()


def test_a_class_var_is_not_a_field():
    # A constant on the class is not filled from YAML, so the rules above have nothing to say about it.
    @dataclass
    class WithConstant(Config):
        registry: ClassVar[dict[str, Any]] = {}
        n: int = MISSING

    Schema(WithConstant).check()


def test_only_the_classs_own_annotations_are_its_to_answer_for():
    # What a base declared was checked at the base; a subclass is held to what it adds.
    check_declaration(fixtures.Optim)  # inherits `lr` from LrOnly and declares `warmup_steps`


# ── a forward reference is checked late, not skipped ─────────────────────────


HEAD = """
from __future__ import annotations
from dataclasses import dataclass, field
from omegaconf import MISSING
from slimconfig import Config
"""


def _module(tmp_path, monkeypatch, name: str, body: str):
    (tmp_path / f"{name}.py").write_text(HEAD + textwrap.dedent(body), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return __import__(name)


FORWARD_OK = """
    @dataclass
    class Outer(Config):
        inner: Inner = field(default_factory=lambda: Inner())

    @dataclass
    class Inner(Config):
        x: int = MISSING
"""

FORWARD_BAD = """
    @dataclass
    class Outer(Config):
        later: Later = MISSING

    class Later:
        pass

    @dataclass
    class After(Config):
        x: int = MISSING
"""

FORWARD_NEVER = """
    @dataclass
    class Outer(Config):
        later: NeverDefined = MISSING
"""


def test_a_group_declared_above_its_class_is_checked_once_the_name_exists(tmp_path, monkeypatch):
    module = _module(tmp_path, monkeypatch, "fwd_ok", FORWARD_OK)
    Schema(module.Outer).check()  # resolvable by now, and well formed


def test_a_forward_reference_that_turns_out_bad_is_still_caught(tmp_path, monkeypatch):
    # `Later` does not exist yet at Outer's class statement, so the check is postponed — to the next
    # config class declared, which is where the import fails.
    with pytest.raises(TypeError, match="not a type a YAML value can have"):
        _module(tmp_path, monkeypatch, "fwd_bad", FORWARD_BAD)


def test_a_name_that_never_appears_is_caught_before_a_load(tmp_path, monkeypatch):
    # Nothing settles it, so the last chance is the load — which takes it.
    module = _module(tmp_path, monkeypatch, "fwd_never", FORWARD_NEVER)
    with pytest.raises(TypeError, match="annotation that names nothing"):
        load_config(module.Outer, [])


# ── and the value has to match the hint that was kept ────────────────────────


FULL = (
    "model: llama\ntags: []\nresume_from: null\n"
    "optim:\n  _schema: fixtures.Optim\n  lr: 0.1\n  warmup_steps: 1\n"
    "data:\n  _schema: fixtures.Data\n  path: d\n"
)
REPORT = "metrics:\n  psnr: [good]\nweights:\n  psnr: 1.0\nlabels: [a, 1]\n"


def test_a_list_element_is_held_to_what_the_list_says_it_holds(tmp_path, write):
    # OmegaConf coerces a scalar element to what the list declared, but lets a MAPPING through as it is.
    path = write(tmp_path / "a.yaml", FULL.replace("tags: []", "tags: [ok, {a: 1}]"))
    with pytest.raises(ValueError, match=r"tags\[1\] is not a str"):
        load_config(fixtures.TrainConfig, [path])


def test_a_dict_value_is_held_to_what_the_dict_says_it_holds(tmp_path, write):
    path = write(tmp_path / "a.yaml", REPORT.replace("psnr: 1.0", "psnr: {a: 1}"), schema="fixtures.Report")
    with pytest.raises(ValueError, match=r"weights\['psnr'\] is not a float"):
        load_config(fixtures.Report, [path])


def test_the_whole_path_to_the_wrong_value_is_named(tmp_path, write):
    # Which value, not just which field: a leaf can be a collection several deep.
    path = write(tmp_path / "a.yaml", REPORT.replace("psnr: 1.0", "psnr: [1.0]"), schema="fixtures.Report")
    with pytest.raises(ValueError, match=r"Report holds value\(s\).*weights\['psnr'\] is not a float"):
        load_config(fixtures.Report, [path])


def test_a_union_holds_any_of_its_members_and_nothing_else(tmp_path, write):
    path = write(tmp_path / "a.yaml", REPORT.replace("labels: [a, 1]", "labels: [a, {b: 1}]"),
                 schema="fixtures.Report")
    with pytest.raises(ValueError, match=r"labels\[1\] is not one of str \| int"):
        load_config(fixtures.Report, [path])


def test_a_value_that_matches_is_left_alone(tmp_path, write):
    cfg = load_config(fixtures.Report, [write(tmp_path / "a.yaml", REPORT, schema="fixtures.Report")])
    assert cfg.metrics == {"psnr": ["good"]} and cfg.weights == {"psnr": 1.0}
    assert cfg.labels == ["a", 1]

# The config-class layer, which is one object: what a schema may look like (Schema.check), how a YAML
# names one (Schema.resolve / Schema.declared), and what sits at a node of one (Schema.at / .require).

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import fixtures
import pytest
from omegaconf import MISSING

from slimconfig import Config, Schema
from slimconfig.schemas import Declaration, Shape, declaration_name, schema_name

# ── a schema is a config class ───────────────────────────────────────────────


def test_a_schema_wraps_a_config_class_and_names_it():
    schema = Schema(fixtures.Optim)
    assert schema.cls is fixtures.Optim
    assert schema.name == schema_name(fixtures.Optim) == "fixtures.Optim"
    assert repr(schema) == "Schema(fixtures.Optim)"


def test_two_schemas_of_the_same_class_are_one_value():
    # Nothing is cached on a Schema, so any caller may build its own and they still compare and hash
    # alike — which is what makes passing the class around and wrapping it late safe.
    assert Schema(fixtures.Optim) == Schema(fixtures.Optim)
    assert len({Schema(fixtures.Optim), Schema(fixtures.Optim)}) == 1


def test_a_schema_must_be_a_dataclass():
    with pytest.raises(TypeError, match="is not a @dataclass"):
        Schema(fixtures.NotADataclass)


def test_a_plain_dataclass_is_not_a_schema():
    # A dataclass that never subclassed Config never said it was filled from YAML, so it is not.
    with pytest.raises(TypeError, match="is not a config class"):
        Schema(fixtures.PlainDataclass)


# ── check ────────────────────────────────────────────────────────────────────


def test_a_well_formed_schema_passes():
    Schema(fixtures.TrainConfig).check()  # leaves and nested groups, nothing else


def test_a_nested_group_must_declare_a_default_factory():
    with pytest.raises(TypeError, match="must declare it as its default"):
        fixtures.declare(fixtures.Optim)


def test_nesting_must_terminate():
    with pytest.raises(TypeError, match="contains itself"):
        Schema(fixtures.SelfReferential).check()


def test_a_table_needs_no_default_factory():
    Schema(fixtures.MatrixConfig).check()  # its entries do not exist until a config file names their keys


# ── naming a class from a YAML ───────────────────────────────────────────────


def test_a_dotted_path_resolves_to_the_class():
    assert Schema.resolve("fixtures.TrainConfig") == Schema(fixtures.TrainConfig)


def test_schema_name_round_trips():
    assert Schema.resolve(Schema(fixtures.Optim).name).cls is fixtures.Optim


@pytest.mark.parametrize(
    ("dotted", "match"),
    [
        ("TrainConfig", "not a dotted import path"),
        ("fixtures.Nope", "could not be imported"),
        ("no_such_module.Thing", "could not be imported"),
        ("fixtures.NotADataclass", "not a config class"),
        ("fixtures.PlainDataclass", "not a config class"),
        ("", "dotted import path of a config class"),
    ],
)
def test_a_bad_schema_name_is_rejected(dotted, match):
    with pytest.raises(ValueError, match=match):
        Schema.resolve(dotted)


def test_a_class_in_the_launched_script_is_named_by_the_scripts_own_name(monkeypatch):
    # A config class defined in the entry point lives in `__main__`. Naming it that in a YAML would be
    # useless, and importing the script again as `train` would re-run it and build a DIFFERENT class —
    # so both directions go through the module that is already running.
    @dataclass
    class Solo(Config):
        x: int = MISSING

    Solo.__module__, Solo.__qualname__ = "__main__", "Solo"
    main = types.ModuleType("__main__")
    main.__file__ = "/somewhere/train.py"
    main.Solo = Solo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "__main__", main)

    assert schema_name(Solo) == "train.Solo"
    assert Schema.resolve("train.Solo").cls is Solo


def test_a_broken_module_is_not_reported_as_a_missing_one(tmp_path, monkeypatch):
    # An ImportError raised BY the named module is its own problem, not "no such class" — hiding it
    # would send the reader looking for a typo in the config.
    (tmp_path / "broken_mod.py").write_text("import definitely_not_installed\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="definitely_not_installed"):
        Schema.resolve("broken_mod.Thing")


def test_declared_reads_a_schema_line_as_one_class_or_a_table_of_them():
    assert Schema.declared("fixtures.Data") == Declaration(Schema(fixtures.Data), None)
    assert Schema.declared(" dict[ str , fixtures.Data ] ") == Declaration(Schema(fixtures.Data), str)
    assert Schema.declared("dict[fixtures.Stage, fixtures.TrainPart]").key is fixtures.Stage


def test_declared_rejects_a_shape_that_is_neither():
    with pytest.raises(ValueError, match="key them, as a table"):
        Schema.declared("list[fixtures.Data]")
    with pytest.raises(ValueError, match="neither a config class nor a table of one"):
        Schema.declared("dict[str, int, fixtures.Data]")


def test_a_key_is_imported_like_the_class_beside_it_not_matched_by_name():
    # The key type is a name in a config file, so it is a name that can be opened: an Enum is named in
    # full, and its bare name — which points at nothing a reader could find — is not a spelling at all.
    with pytest.raises(ValueError, match=r"'Stage' is neither"):
        Schema.declared("dict[Stage, fixtures.TrainPart]")
    with pytest.raises(ValueError, match=r"keyed by `str` or by an Enum"):
        Schema.declared("dict[fixtures.Data, fixtures.TrainPart]")


def test_declaration_name_is_the_spelling_declared_reads_back():
    assert declaration_name(fixtures.Data) == "fixtures.Data"
    assert declaration_name(fixtures.Data, str) == "dict[str, fixtures.Data]"
    spelled = declaration_name(fixtures.TrainPart, fixtures.Stage)
    assert spelled == "dict[fixtures.Stage, fixtures.TrainPart]"
    assert Schema.declared(spelled) == Declaration(Schema(fixtures.TrainPart), fixtures.Stage)


# ── fields ───────────────────────────────────────────────────────────────────


def test_fields_tells_the_three_shapes_apart():
    fields = Schema(fixtures.MatrixConfig).fields
    assert fields["stage"] == Shape("value", None)
    assert fields["base"] == Shape("group", fixtures.TrainPart)
    assert fields["per_model"] == Shape("table", fixtures.Data, str)  # a table also says what keys it takes


def test_a_dict_of_plain_values_is_a_leaf():
    @dataclass
    class WithWeights(Config):
        weights: dict[str, float] = MISSING

    assert Schema(WithWeights).fields["weights"] == Shape("value", None)


# ── nodes ────────────────────────────────────────────────────────────────────


def test_at_says_what_a_path_lands_on_without_raising():
    root = Schema(fixtures.MatrixConfig)
    assert root.at("") == Shape("group", fixtures.MatrixConfig)
    assert root.at("stage") == Shape("value", None)
    assert root.at("per_model") == Shape("table", fixtures.Data, str)
    assert root.at("per_model.flux") == Shape("entry", fixtures.Data)
    assert root.at("nope") == Shape("unknown", None)
    assert root.at("stage.deeper") == Shape("unknown", None)  # nothing below a leaf is a node
    assert root.at(("per_model", "flux.1-dev")) == Shape("entry", fixtures.Data)


def test_the_empty_path_is_the_root_itself():
    assert Schema(fixtures.TrainConfig).require("") == Schema(fixtures.TrainConfig)


def test_a_field_path_walks_to_the_nested_class():
    assert Schema(fixtures.TrainConfig).require("optim") == Schema(fixtures.Optim)


def test_an_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="has no field 'nope'"):
        Schema(fixtures.TrainConfig).require("nope")


def test_a_leaf_is_not_a_group():
    with pytest.raises(ValueError, match="is a value, not a nested config class"):
        Schema(fixtures.TrainConfig).require("model")


def test_a_table_entry_walks_to_the_entry_class():
    root = Schema(fixtures.MatrixConfig)
    assert root.require("per_model.flux") == Schema(fixtures.Data)
    assert root.require("per_stage.main.optim") == Schema(fixtures.TrainPart.OptimPart)


def test_the_table_itself_has_no_class_to_fill():
    # `per_model` is however many Datas the keys name, not one config — so a file cannot be mounted on it,
    # and the error offers both ways to say what it is: the shape, or one entry.
    with pytest.raises(ValueError, match=r"`dict\[str, fixtures.Data\]`.*`per_model.<key>`"):
        Schema(fixtures.MatrixConfig).require("per_model")

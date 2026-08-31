# The config-class layer: what a schema may look like (check_schema), naming one from a YAML
# (resolve_schema), and which class belongs at a node of one (field_schema).

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import fixtures
import pytest
from omegaconf import MISSING

from slimconfig import check_schema, field_schema, resolve_schema, schema_name

# ── check_schema ─────────────────────────────────────────────────────────────


def test_a_well_formed_schema_passes():
    check_schema(fixtures.TrainConfig)  # leaves and nested groups, nothing else


def test_a_nested_group_must_declare_a_default_factory():
    with pytest.raises(TypeError, match="must declare it as its default"):
        check_schema(fixtures.MissingFactory)


def test_nesting_must_terminate():
    with pytest.raises(TypeError, match="contains itself"):
        check_schema(fixtures.SelfReferential)


def test_a_schema_must_be_a_dataclass():
    with pytest.raises(TypeError, match="is not a config class"):
        check_schema(fixtures.NotADataclass)


# ── resolve_schema / schema_name ─────────────────────────────────────────────


def test_a_dotted_path_resolves_to_the_class():
    assert resolve_schema("fixtures.TrainConfig") is fixtures.TrainConfig


def test_schema_name_round_trips():
    assert resolve_schema(schema_name(fixtures.Optim)) is fixtures.Optim


@pytest.mark.parametrize(
    ("dotted", "match"),
    [
        ("TrainConfig", "not a dotted import path"),
        ("fixtures.Nope", "could not be imported"),
        ("no_such_module.Thing", "could not be imported"),
        ("fixtures.NotADataclass", "not a config class"),
        ("", "dotted import path of a config class"),
    ],
)
def test_a_bad_schema_name_is_rejected(dotted, match):
    with pytest.raises(ValueError, match=match):
        resolve_schema(dotted)


def test_a_class_in_the_launched_script_is_named_by_the_scripts_own_name(monkeypatch):
    # A config class defined in the entry point lives in `__main__`. Naming it that in a YAML would be
    # useless, and importing the script again as `train` would re-run it and build a DIFFERENT class —
    # so both directions go through the module that is already running.
    @dataclass
    class Solo:
        x: int = MISSING

    Solo.__module__, Solo.__qualname__ = "__main__", "Solo"
    main = types.ModuleType("__main__")
    main.__file__ = "/somewhere/train.py"
    main.Solo = Solo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "__main__", main)

    assert schema_name(Solo) == "train.Solo"
    assert resolve_schema("train.Solo") is Solo


def test_a_broken_module_is_not_reported_as_a_missing_one(tmp_path, monkeypatch):
    # An ImportError raised BY the named module is its own problem, not "no such class" — hiding it
    # would send the reader looking for a typo in the config.
    (tmp_path / "broken_mod.py").write_text("import definitely_not_installed\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="definitely_not_installed"):
        resolve_schema("broken_mod.Thing")


# ── field_schema ─────────────────────────────────────────────────────────────


def test_the_empty_path_is_the_root_itself():
    assert field_schema(fixtures.TrainConfig, "") is fixtures.TrainConfig


def test_a_field_path_walks_to_the_nested_class():
    assert field_schema(fixtures.TrainConfig, "optim") is fixtures.Optim


def test_an_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="has no field 'nope'"):
        field_schema(fixtures.TrainConfig, "nope")


def test_a_leaf_is_not_a_group():
    with pytest.raises(ValueError, match="is a value, not a nested config class"):
        field_schema(fixtures.TrainConfig, "model")

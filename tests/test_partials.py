# The layer schema: a config class whose fields may be left unset, so a run can be composed from
# several of them at runtime — and read back with `stated`, which reports only what a layer said.

from __future__ import annotations

from dataclasses import dataclass, field

import fixtures
import pytest
from omegaconf import MISSING

from slimconfig import is_partial, load_config, partial_of, schema_name, stated

# ── what partial_of builds ───────────────────────────────────────────────────


def test_a_partial_is_a_real_subclass():
    # So it type-checks as the class it layers, and a fragment written for that class still mounts here.
    assert issubclass(fixtures.TrainPart, fixtures.TrainConfig) and is_partial(fixtures.TrainPart)


def test_the_same_partial_is_returned_every_time():
    assert partial_of(fixtures.TrainConfig, name="TrainPart") is fixtures.TrainPart


def test_the_class_it_layers_is_not_a_partial():
    assert not is_partial(fixtures.TrainConfig)


def test_a_nested_group_is_layered_too():
    # A layer may set one field of a group and say nothing about the rest, so the group is a layer as well.
    assert issubclass(fixtures.TrainPart.OptimPart, fixtures.Optim)
    assert schema_name(fixtures.TrainPart.OptimPart) == "fixtures.TrainPart.OptimPart"


def test_only_a_config_class_has_a_partial():
    with pytest.raises(TypeError, match="not a config class"):
        partial_of(fixtures.NotADataclass)


# ── loading layers ───────────────────────────────────────────────────────────


def _matrix(**overrides):
    return load_config(fixtures.MatrixConfig, [{
        "stage": "main",
        "base": {"model": "m", "tags": [], "resume_from": None, "optim": {"lr": 0.1}},
        "per_stage": {"main": {"optim": {"warmup_steps": 10}}},
        "per_model": {"flux": {"path": "/data/flux"}},
        **overrides,
    }])


def test_a_layer_may_leave_every_field_unset():
    cfg = _matrix(base={}, per_stage={})
    assert stated(cfg.base) == {} and cfg.base.model == MISSING


def test_stated_reports_only_what_the_layer_said():
    assert stated(_matrix().per_stage[fixtures.Stage.main]) == {"optim": {"warmup_steps": 10}}


def test_an_explicit_null_is_something_the_layer_said():
    # The whole reason unset is MISSING and not None: `null` is a real value in these schemas.
    assert stated(_matrix().base)["resume_from"] is None


def test_layers_compose_into_the_complete_class():
    cfg = _matrix()
    merged = {**stated(cfg.base), "data": {"path": "/data/x"}}
    merged["optim"] = {**merged["optim"], **stated(cfg.per_stage[cfg.stage])["optim"]}
    assert load_config(fixtures.TrainConfig, [merged]) == fixtures.TrainConfig(
        model="m", tags=[], resume_from=None, optim=fixtures.Optim(lr=0.1, warmup_steps=10),
        data=fixtures.Data(path="/data/x"),
    )


def test_a_layer_is_still_validated():
    with pytest.raises(Exception, match="not in 'TrainPart'"):
        _matrix(base={"nope": 1})


def test_a_table_key_is_validated_against_the_enum():
    with pytest.raises(Exception, match="valid: .*warmup"):
        _matrix(per_stage={"nosuchstage": {}})


def test_a_table_of_a_complete_class_is_still_complete():
    # per_model holds Data, not a layer of it — leaving a field out of an entry is the usual error.
    with pytest.raises(ValueError, match=r"missing required field\(s\): per_model.flux.path"):
        _matrix(per_model={"flux": {}})


def test_a_partial_inside_a_run_does_not_make_the_run_optional():
    with pytest.raises(ValueError, match=r"missing required field\(s\): per_model"):
        load_config(fixtures.MatrixConfig, [{"stage": "main", "per_stage": {}}])


def test_a_run_is_not_excused_by_a_layer_merged_onto_it():
    # Merging a layer's node onto a run's promotes the result's RUNTIME type to the layer; only the
    # declared schema still knows this is a run, and a run must be complete.
    with pytest.raises(ValueError, match=r"missing required field\(s\): optim.warmup_steps"):
        load_config(fixtures.TrainConfig, [stated(_matrix().base), {"data": {"path": "/d"}}])


# ── stated ───────────────────────────────────────────────────────────────────


def test_stated_takes_an_instance_not_a_class():
    with pytest.raises(TypeError, match="takes a config instance"):
        stated(fixtures.TrainPart)


def test_a_group_the_layer_was_silent_about_is_absent_not_empty():
    @dataclass
    class Outer:
        inner: fixtures.Data = field(default_factory=fixtures.Data)
        n: int = MISSING

    layer = partial_of(Outer, name="OuterPart")
    assert stated(load_config(layer, [{"n": 1}])) == {"n": 1}

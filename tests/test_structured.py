# The typed layer: merge_specs / load_config / peek / schema_of — including the check that a config file
# is being merged onto the class it says it fills. The run layer lives in test_runs.py.

from __future__ import annotations

import fixtures
import pytest

from slimconfig import load_config, merge_specs, peek, schema_of

FULL = """
model: llama
tags: []
resume_from: null
optim:
  _schema: fixtures.Optim
  lr: 0.0002
  warmup_steps: 100
data:
  _schema: fixtures.Data
  path: data/corpus.parquet
"""


# ── merge_specs ──────────────────────────────────────────────────────────────


def test_merge_specs_later_spec_wins(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: 1\nb: 2\n")
    merged = merge_specs([path, "b=20", {"c": 30}])
    assert (merged.a, merged.b, merged.c) == (1, 20, 30)


def test_merge_specs_parses_dotted_overrides(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL)
    merged = merge_specs([path, "optim.lr=0.5"])
    assert merged.optim.lr == 0.5
    assert merged.optim.warmup_steps == 100  # untouched


def test_merge_specs_rejects_a_spec_that_is_neither(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither a file nor a key=value override"):
        merge_specs(["configs/typo.yaml"])


# ── load_config ──────────────────────────────────────────────────────────────


def test_load_config_returns_a_populated_instance(tmp_path, write):
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    assert isinstance(cfg, fixtures.TrainConfig)
    assert isinstance(cfg.optim, fixtures.Optim)
    assert (cfg.model, cfg.tags, cfg.resume_from) == ("llama", [], None)
    assert cfg.optim.lr == pytest.approx(2e-4)


def test_load_config_overrides_win_over_the_file(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL)
    cfg = load_config(fixtures.TrainConfig, [path, "model=qwen", "optim.warmup_steps=7"])
    assert (cfg.model, cfg.optim.warmup_steps) == ("qwen", 7)


def test_load_config_requires_every_leaf(tmp_path, write):
    path = write(tmp_path / "a.yaml", "model: llama\ntags: []\n")
    with pytest.raises(ValueError, match=r"missing required field\(s\): resume_from, optim.lr"):
        load_config(fixtures.TrainConfig, [path])


def test_load_config_rejects_unknown_keys(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL + "typo_key: 1\n")
    with pytest.raises(Exception, match="typo_key"):
        load_config(fixtures.TrainConfig, [path])


def test_load_config_rejects_a_wrongly_typed_value(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL.replace("warmup_steps: 100", "warmup_steps: many"))
    with pytest.raises(Exception, match="warmup_steps"):
        load_config(fixtures.TrainConfig, [path])


def test_load_config_composes_defaults(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", FULL)
    child = write(tmp_path / "child.yaml", "defaults: [base.yaml]\nmodel: qwen\n")
    cfg = load_config(fixtures.TrainConfig, [child])
    assert (cfg.model, cfg.optim.warmup_steps) == ("qwen", 100)


def test_load_config_rejects_a_malformed_schema(tmp_path, write):
    with pytest.raises(TypeError, match="must declare it as its default"):
        load_config(fixtures.MissingFactory, [write(tmp_path / "a.yaml", "optim: {lr: 1.0}\n")])


# ── the file has to be the config it says it is ──────────────────────────────


def test_a_file_written_for_another_class_is_rejected(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL, schema="fixtures.Data")
    with pytest.raises(ValueError, match="says it fills fixtures.Data.*the top level.*TrainConfig"):
        load_config(fixtures.TrainConfig, [path])


def test_a_fragment_mounted_at_the_wrong_block_is_rejected(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "cosine.yaml", "lr: 2.0e-4\nwarmup_steps: 5\n", schema="fixtures.Optim")
    child = write(tmp_path / "train.yaml", "data:\n  defaults: [cosine.yaml]\n")
    with pytest.raises(ValueError, match=r"says it fills fixtures.Optim.*`data`.*fixtures.Data"):
        load_config(fixtures.TrainConfig, [child])


def test_a_fragment_may_declare_a_base_of_the_target(tmp_path, monkeypatch, write):
    # A base states a SUBSET of the fields — which is exactly what a shared fragment does.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "lr.yaml", "lr: 3.0e-4\n", schema="fixtures.LrOnly")
    body = FULL.replace("  lr: 0.0002\n", "  defaults: [lr.yaml]\n")
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "train.yaml", body)])
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (pytest.approx(3e-4), 100)


def test_a_fragment_may_not_declare_a_subclass_of_the_target(tmp_path, monkeypatch, write):
    # The other direction sets fields the target does not have.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "opt.yaml", "lr: 1.0\nwarmup_steps: 5\n", schema="fixtures.Optim")
    child = write(
        tmp_path / "loose.yaml", "optim:\n  defaults: [opt.yaml]\n", schema="fixtures.LooseConfig"
    )
    with pytest.raises(ValueError, match=r"says it fills fixtures.Optim.*`optim`.*fixtures.LrOnly"):
        load_config(fixtures.LooseConfig, [child])


def test_a_mapping_spec_needs_no_declaration(tmp_path, write):
    # Values a routine computed are code, and code is already typed.
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL), {"model": "qwen"}])
    assert cfg.model == "qwen"


# ── every block that fills a config class names it ───────────────────────────


def test_a_block_that_fills_a_class_must_name_it(tmp_path, write):
    body = FULL.replace("  _schema: fixtures.Optim\n", "")
    path = write(tmp_path / "a.yaml", body)
    match = r"block `optim`.*fixtures\.Optim.*add `_schema: fixtures\.Optim`"
    with pytest.raises(ValueError, match=match):
        load_config(fixtures.TrainConfig, [path])


def test_a_table_entry_does_not_have_to_name_its_class(tmp_path, write):
    # The table's own declaration already fixed it: every entry of per_model is a Data.
    body = "stage: main\nper_stage: {}\nper_model:\n  llama:\n    path: a.parquet\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    cfg = load_config(fixtures.MatrixConfig, [path])
    assert cfg.per_model["llama"].path == "a.parquet"


def test_a_table_itself_is_not_a_block_to_name(tmp_path, write):
    # `per_model` is a mapping in the YAML but has no class of its own, so nothing is demanded of it.
    body = "stage: main\nper_stage: {}\nper_model: {}\nbase:\n  _schema: fixtures.TrainPart\n  model: a\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    cfg = load_config(fixtures.MatrixConfig, [path])
    assert cfg.base.model == "a"


def test_a_layer_block_names_its_partial(tmp_path, write):
    body = "stage: main\nper_stage: {}\nper_model: {}\nbase:\n  model: a\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"block `base`.*TrainPart"):
        load_config(fixtures.MatrixConfig, [path])


def test_a_group_inside_a_table_entry_still_names_itself(tmp_path, write):
    body = "stage: main\nper_model: {}\nper_stage:\n  main:\n    optim:\n      lr: 0.1\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"block `per_stage.main.optim`"):
        load_config(fixtures.MatrixConfig, [path])


def test_a_block_that_only_mounts_a_fragment_is_named_by_the_fragment(tmp_path, monkeypatch, write):
    # `optim:` holds nothing but a `defaults:`, and the file it lists declares the class AT that node.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "opt.yaml", "lr: 1.0\nwarmup_steps: 5\n", schema="fixtures.Optim")
    body = FULL.replace(
        "  _schema: fixtures.Optim\n  lr: 0.0002\n  warmup_steps: 100\n", "  defaults: [opt.yaml]\n"
    )
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "train.yaml", body)])
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (1.0, 5)


# ── peek / schema_of ─────────────────────────────────────────────────────────


def test_peek_reads_a_key_without_validation(tmp_path, write):
    path = write(tmp_path / "a.yaml", "mode: train\nunknown_key: 1\noptim: {lr: 0.5}\n")
    assert peek([path], "mode") == "train"
    assert peek([path], "optim.lr") == 0.5
    assert peek([path], "absent") is None
    assert peek([path, "mode=eval"], "mode") == "eval"


def test_schema_of_reads_the_class_a_file_was_written_for(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL, schema="fixtures.TrainConfig")
    assert schema_of(path) is fixtures.TrainConfig

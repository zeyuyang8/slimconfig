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
    with pytest.raises(ValueError, match="typo_key.*not a field of fixtures.TrainConfig"):
        load_config(fixtures.TrainConfig, [path])


def test_load_config_rejects_a_wrongly_typed_value(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL.replace("warmup_steps: 100", "warmup_steps: many"))
    with pytest.raises(Exception, match="warmup_steps"):
        load_config(fixtures.TrainConfig, [path])


def test_load_config_composes_the_default(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", FULL)
    child = write(tmp_path / "child.yaml", "_default: base.yaml\nmodel: qwen\n")
    cfg = load_config(fixtures.TrainConfig, [child])
    assert (cfg.model, cfg.optim.warmup_steps) == ("qwen", 100)


def test_load_config_rejects_a_malformed_schema(tmp_path, write):
    # The class statement checks a schema field by field; only the rule that spans classes — the nesting
    # has to terminate — is left for the load, and it is checked BEFORE the files are read.
    with pytest.raises(TypeError, match="contains itself"):
        load_config(fixtures.SelfReferential, [write(tmp_path / "a.yaml", "child: {}\n")])


# ── every key a config sets is a field of the class it fills ─────────────────


def test_a_typo_is_reported_against_the_file_that_wrote_it(tmp_path, monkeypatch, write):
    # The merged config has no memory of which file in a `_default:` chain wrote a key, so the check
    # runs file by file — otherwise the one place a reader cannot fix it is the one they are sent to.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", FULL + "wramup: 1\n")
    child = write(tmp_path / "child.yaml", "_default: base.yaml\nmodel: qwen\n")
    with pytest.raises(ValueError, match=r"base.yaml' sets `wramup`.*not a field of fixtures.TrainConfig"):
        load_config(fixtures.TrainConfig, [child])


def test_a_typo_inside_a_group_names_the_group(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL.replace("lr: 0.0002", "lr: 0.0002\n  learning_rate: 1"))
    with pytest.raises(ValueError, match=r"sets `optim.learning_rate`.*not a field of fixtures.Optim"):
        load_config(fixtures.TrainConfig, [path])


def test_an_override_that_names_no_field_is_reported_as_itself(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL)
    with pytest.raises(ValueError, match=r"'optim.lrr=1' sets `optim.lrr`"):
        load_config(fixtures.TrainConfig, [path, "optim.lrr=1"])


def test_a_key_inside_a_leaf_is_not_a_field_of_anything(tmp_path, write):
    # `metrics` is a `dict[str, list[str]]`: the keys under it are the VALUE, not nodes of the schema.
    path = write(
        tmp_path / "a.yaml",
        "metrics:\n  psnr: [good]\nweights:\n  psnr: 1.0\nlabels: []\n",
        schema="fixtures.Report",
    )
    assert load_config(fixtures.Report, [path]).metrics == {"psnr": ["good"]}


# ── the file has to be the config it says it is ──────────────────────────────


def test_a_file_written_for_another_class_is_rejected(tmp_path, write):
    path = write(tmp_path / "a.yaml", FULL, schema="fixtures.Data")
    with pytest.raises(ValueError, match="says it fills fixtures.Data.*the top level.*TrainConfig"):
        load_config(fixtures.TrainConfig, [path])


def test_a_fragment_mounted_at_the_wrong_block_is_rejected(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "cosine.yaml", "lr: 2.0e-4\nwarmup_steps: 5\n", schema="fixtures.Optim")
    child = write(tmp_path / "train.yaml", "data:\n  _default: cosine.yaml\n")
    with pytest.raises(ValueError, match=r"says it fills fixtures.Optim.*`data`.*fixtures.Data"):
        load_config(fixtures.TrainConfig, [child])


def test_a_fragment_may_declare_a_base_of_the_target(tmp_path, monkeypatch, write):
    # A base states a SUBSET of the fields — which is exactly what a shared fragment does.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "lr.yaml", "lr: 3.0e-4\n", schema="fixtures.LrOnly")
    body = FULL.replace("  lr: 0.0002\n", "  _default: lr.yaml\n")
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "train.yaml", body)])
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (pytest.approx(3e-4), 100)


def test_a_fragment_may_not_declare_a_subclass_of_the_target(tmp_path, monkeypatch, write):
    # The other direction sets fields the target does not have.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "opt.yaml", "lr: 1.0\nwarmup_steps: 5\n", schema="fixtures.Optim")
    child = write(
        tmp_path / "loose.yaml", "optim:\n  _default: opt.yaml\n", schema="fixtures.LooseConfig"
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


TABLE = "stage: main\nper_stage: {{}}\nper_model:\n{line}  llama:\n    path: a.parquet\n"


def test_a_table_names_its_entry_class_once_for_all_of_them(tmp_path, write):
    # One line at the table, not one per entry: which class an entry has was fixed by the table.
    body = TABLE.format(line="  _schema: dict[str, fixtures.Data]\n")
    cfg = load_config(fixtures.MatrixConfig, [write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")])
    assert cfg.per_model["llama"].path == "a.parquet"


def test_a_table_entry_does_not_name_its_class(tmp_path, write):
    # The entry is bare — and the table above it is what says what an entry is.
    body = TABLE.format(line="  _schema: dict[str, fixtures.Data]\n").replace(
        "    path:", "    _schema: fixtures.Data\n    path:"
    )
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    assert load_config(fixtures.MatrixConfig, [path]).per_model["llama"].path == "a.parquet"


def test_a_table_that_does_not_say_what_it_holds_is_rejected(tmp_path, write):
    path = write(tmp_path / "m.yaml", TABLE.format(line=""), schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"table `per_model`.*`_schema: dict\[str, fixtures.Data\]`"):
        load_config(fixtures.MatrixConfig, [path])


def test_a_table_and_a_group_are_told_apart_by_the_declaration(tmp_path, write):
    # The same mapping shape on the page. `dict[K, C]` is what says which of the two it is, so naming a
    # table as one class, or a group as a table of one, is an error and not a shrug.
    flat = write(tmp_path / "flat.yaml", TABLE.format(line="  _schema: fixtures.Data\n"),
                 schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"per_model is a table of fixtures.Data, not one of it"):
        load_config(fixtures.MatrixConfig, [flat])
    keyed = FULL.replace("  _schema: fixtures.Optim\n", "  _schema: dict[str, fixtures.Optim]\n")
    with pytest.raises(ValueError, match=r"`optim`.*is ONE fixtures.Optim, not several"):
        load_config(fixtures.TrainConfig, [write(tmp_path / "t.yaml", keyed)])


def test_a_table_says_what_its_keys_are_and_they_are_checked(tmp_path, write):
    body = "stage: main\nper_model: {}\nper_stage:\n  _schema: dict[str, fixtures.TrainPart]\n  main: {}\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"keyed by str, but fixtures.MatrixConfig.per_stage is keyed by "
                                         r"fixtures.Stage: `_schema: dict\[fixtures.Stage, "):
        load_config(fixtures.MatrixConfig, [path])


def test_a_whole_table_can_be_mounted_from_a_fragment(tmp_path, monkeypatch, write):
    # Saying the shape is also what lets a shared file fill a table: it declares `dict[K, C]` at its own
    # top level, and the parent says which table it lands on.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "models.yaml", "llama:\n  path: a.parquet\n", schema="dict[str, fixtures.Data]")
    body = "stage: main\nper_stage: {}\nper_model:\n  _default: models.yaml\n"
    cfg = load_config(fixtures.MatrixConfig, [write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")])
    assert cfg.per_model["llama"].path == "a.parquet"


def test_an_empty_table_has_nothing_to_declare(tmp_path, write):
    # `per_model: {}` is how a config says it has none, and there is no entry under it to read.
    body = "stage: main\nper_stage: {}\nper_model: {}\nbase:\n  _schema: fixtures.TrainPart\n  model: a\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    cfg = load_config(fixtures.MatrixConfig, [path])
    assert cfg.base.model == "a" and cfg.per_model == {}


def test_a_layer_block_names_its_partial(tmp_path, write):
    body = "stage: main\nper_stage: {}\nper_model: {}\nbase:\n  model: a\n"
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"block `base`.*TrainPart"):
        load_config(fixtures.MatrixConfig, [path])


def test_a_group_inside_a_table_entry_still_names_itself(tmp_path, write):
    body = ("stage: main\nper_model: {}\nper_stage:\n  _schema: dict[fixtures.Stage, fixtures.TrainPart]\n"
            "  main:\n    optim:\n      lr: 0.1\n")
    path = write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")
    with pytest.raises(ValueError, match=r"block `per_stage.main.optim`"):
        load_config(fixtures.MatrixConfig, [path])


def test_a_table_key_may_contain_a_dot(tmp_path, write):
    # `flux.1-dev` is ONE key, so the node of the block under it is ("per_model", "flux.1-dev", "optim")
    # — three keys. Written as one dotted string it would re-split into four and place nothing.
    body = ("per_model:\n  _schema: dict[str, fixtures.TrainPart]\n"
            "  flux.1-dev:\n    optim:\n{line}      lr: 0.1\n")
    bare = write(tmp_path / "bare.yaml", body.format(line=""), schema="fixtures.ModelMatrix")
    with pytest.raises(ValueError, match=r"block `per_model.flux.1-dev.optim`.*TrainPart.OptimPart"):
        load_config(fixtures.ModelMatrix, [bare])
    named = body.format(line="      _schema: fixtures.TrainPart.OptimPart\n")
    path = write(tmp_path / "m.yaml", named, schema="fixtures.ModelMatrix")
    assert load_config(fixtures.ModelMatrix, [path]).per_model["flux.1-dev"].optim.lr == 0.1


def test_a_block_that_only_mounts_a_fragment_is_named_by_the_fragment(tmp_path, monkeypatch, write):
    # `optim:` holds nothing but a `_default:`, and the file it lists declares the class AT that node.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "opt.yaml", "lr: 1.0\nwarmup_steps: 5\n", schema="fixtures.Optim")
    body = FULL.replace(
        "  _schema: fixtures.Optim\n  lr: 0.0002\n  warmup_steps: 100\n", "  _default: opt.yaml\n"
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

# The YAML layer: load_yaml / load_mapping_yaml / compose, the `_schema:` and `_default:` keywords, and
# the resolvers. What the claims MEAN is checked against a schema in test_structured.py.

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from slimconfig import compose, load_mapping_yaml, load_yaml


def test_load_yaml_returns_plain_dict(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: 1\nb: [1, 2]\n", schema=None)
    assert load_yaml(path) == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("body", ["- 1\n- 2\n", "42\n"])
def test_load_yaml_rejects_non_mapping(tmp_path, write, body):
    path = write(tmp_path / "a.yaml", body, schema=None)
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_yaml(path)


def test_load_yaml_rejects_invalid_yaml(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: [1,\n", schema=None)
    with pytest.raises(ValueError, match="not valid YAML"):
        load_yaml(path)


def test_load_mapping_yaml_without_default(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: 1\nnested: {x: 2}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.a == 1
    assert cfg.nested.x == 2


# (A bare string yaml is NOT in this list: OmegaConf parses `just a string` into the mapping
# {"just a string": None}, so it never reaches the type check.)
@pytest.mark.parametrize("body", ["- 1\n", "42\n"])
def test_load_mapping_yaml_rejects_non_mapping(tmp_path, write, body):
    path = write(tmp_path / "a.yaml", body, schema=None)
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_mapping_yaml(path)


def test_load_mapping_yaml_missing_file_is_not_a_parse_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mapping_yaml(str(tmp_path / "nope.yaml"))


# ── `_schema:` ───────────────────────────────────────────────────────────────


def test_every_config_file_must_name_its_class(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: 1\n", schema=None)
    with pytest.raises(ValueError, match="does not say which config class it fills"):
        load_mapping_yaml(path)


def test_the_schema_line_is_consumed_not_merged(tmp_path, write):
    cfg = load_mapping_yaml(write(tmp_path / "a.yaml", "a: 1\n"))
    assert "_schema" not in cfg


def test_compose_reports_the_claim_and_where_it_was_made(tmp_path, write):
    path = write(tmp_path / "a.yaml", "a: 1\n", schema="fixtures.TrainConfig")
    claims = compose(path).claims
    assert [(c.node, c.schema) for c in claims] == [((), "fixtures.TrainConfig")]
    assert claims[0].source == path


def test_a_nested_block_may_restate_its_class(tmp_path, write):
    body = "model: llama\noptim:\n  _schema: fixtures.Optim\n  lr: 0.1\n"
    claims = compose(write(tmp_path / "a.yaml", body)).claims
    assert [(c.node, c.schema) for c in claims] == [
        ((), "fixtures.TrainConfig"),
        (("optim",), "fixtures.Optim"),
    ]


def test_the_schema_line_must_be_a_string(tmp_path, write):
    with pytest.raises(ValueError, match="must be a dotted import path"):
        load_mapping_yaml(write(tmp_path / "a.yaml", "_schema: [a, b]\na: 1\n"))


# ── `_default:` composition ──────────────────────────────────────────────────


def test_default_current_file_wins_and_merges_deeply(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "a: 1\nb: 2\nnested: {x: 1, y: 1}\n")
    child = write(tmp_path / "child.yaml", "_default: base.yaml\nb: 20\nnested: {y: 20}\n")
    cfg = load_mapping_yaml(child)
    assert cfg.a == 1  # inherited
    assert cfg.b == 20  # overridden
    assert (cfg.nested.x, cfg.nested.y) == (1, 20)  # deep merge, not replacement
    assert "_default" not in cfg


def test_a_config_inherits_one_file_not_a_list(tmp_path, monkeypatch, write):
    # Two parents have no reading order: which of them set the value you are looking at is answered by
    # counting positions in a list. Combining independent files is a LAUNCH, not an inheritance.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "one.yaml", "a: 1\n")
    write(tmp_path / "two.yaml", "b: 2\n")
    child = write(tmp_path / "child.yaml", "_default: [one.yaml, two.yaml]\n")
    with pytest.raises(ValueError, match=r"must be ONE yaml path.*at the launch"):
        load_mapping_yaml(child)


def test_default_compose_recursively(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "grand.yaml", "a: 1\n")
    write(tmp_path / "parent.yaml", "_default: grand.yaml\nb: 2\n")
    child = write(tmp_path / "child.yaml", "_default: parent.yaml\nc: 3\n")
    cfg = load_mapping_yaml(child)
    assert (cfg.a, cfg.b, cfg.c) == (1, 2, 3)


def test_default_resolve_against_cwd_not_the_including_file(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    write(tmp_path / "configs" / "base.yaml", "a: 1\n")
    # The `_default` entry is written from the project root, even though the file that carries it
    # sits one directory down next to base.yaml.
    child = write(tmp_path / "configs" / "child.yaml", "_default: configs/base.yaml\nb: 2\n")
    cfg = load_mapping_yaml(child)
    assert (cfg.a, cfg.b) == (1, 2)


def test_default_cycle_is_detected(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "a.yaml", "_default: b.yaml\n")
    write(tmp_path / "b.yaml", "_default: a.yaml\n")
    with pytest.raises(ValueError, match="`_default` cycle detected"):
        load_mapping_yaml(str(tmp_path / "a.yaml"))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("_default: {a: 1}\n", "must be ONE yaml path"),
        ("_default: 3\n", "must be ONE yaml path"),
    ],
)
def test_default_shape_is_validated(tmp_path, monkeypatch, write, body, match):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "a: 1\n")
    path = write(tmp_path / "child.yaml", body)
    with pytest.raises(ValueError, match=match):
        load_mapping_yaml(path)


# ── `_default:` at any depth ─────────────────────────────────────────────────


def test_default_inside_a_block_mounts_the_file_there(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "cosine.yaml", "lr: 2.0e-4\nwarmup_steps: 100\n", schema="fixtures.Optim")
    child = write(tmp_path / "train.yaml", "model: llama\noptim:\n  _default: cosine.yaml\n")
    cfg = load_mapping_yaml(child)
    # The fragment states its fields at ITS top level; the parent says where they land.
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (2e-4, 100)
    assert "lr" not in cfg


def test_a_block_wins_over_what_it_starts_from(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "cosine.yaml", "lr: 2.0e-4\nwarmup_steps: 100\n", schema="fixtures.Optim")
    body = "model: llama\noptim:\n  _default: cosine.yaml\n  lr: 1.0e-4\n"
    cfg = load_mapping_yaml(write(tmp_path / "train.yaml", body))
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (1e-4, 100)


def test_a_mounted_fragment_reports_its_claim_at_the_mount_point(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "cosine.yaml", "lr: 2.0e-4\n", schema="fixtures.Optim")
    child = write(tmp_path / "train.yaml", "optim:\n  _default: cosine.yaml\n")
    assert [(c.node, c.schema) for c in compose(child).claims] == [
        ((), "fixtures.TrainConfig"),
        (("optim",), "fixtures.Optim"),
    ]


def test_one_fragment_can_be_mounted_at_two_places(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "shared.yaml", "path: data/x\n", schema="fixtures.Data")
    body = "data:\n  _default: shared.yaml\nother:\n  _default: shared.yaml\n"
    cfg = load_mapping_yaml(write(tmp_path / "train.yaml", body))
    assert cfg.data.path == cfg.other.path == "data/x"  # the same file, twice, is not a cycle


def test_a_mounted_fragment_composes_its_own_default(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base_optim.yaml", "lr: 1.0\nwarmup_steps: 5\n", schema="fixtures.Optim")
    write(tmp_path / "cosine.yaml", "_default: base_optim.yaml\nlr: 2.0\n", schema="fixtures.Optim")
    cfg = load_mapping_yaml(write(tmp_path / "train.yaml", "optim:\n  _default: cosine.yaml\n"))
    assert (cfg.optim.lr, cfg.optim.warmup_steps) == (2.0, 5)


def test_a_nested_default_cycle_is_detected(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "a.yaml", "optim:\n  _default: b.yaml\n")
    write(tmp_path / "b.yaml", "_default: a.yaml\n", schema="fixtures.Optim")
    with pytest.raises(ValueError, match="`_default` cycle detected"):
        load_mapping_yaml(str(tmp_path / "a.yaml"))


def test_a_nested_default_shape_error_names_the_block(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    path = write(tmp_path / "a.yaml", "optim:\n  _default: [base.yaml]\n")
    with pytest.raises(ValueError, match=r"under `optim`.*must be ONE yaml path"):
        load_mapping_yaml(path)


# ── resolvers ────────────────────────────────────────────────────────────────


def test_now_resolver_stamps_a_value(tmp_path, write):
    path = write(tmp_path / "a.yaml", "tag: runs/${now:%Y}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.tag.startswith("runs/2")
    assert len(cfg.tag) == len("runs/YYYY")


def test_now_resolver_is_one_consistent_stamp(tmp_path, write):
    path = write(tmp_path / "a.yaml", "one: ${now:%Y%m%d-%H%M%S}\ntwo: ${now:%Y%m%d-%H%M%S}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.one == cfg.two


def test_from_yaml_resolver_reads_another_config(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "data.yaml", "dataset:\n  name: wikitext\n")
    path = write(tmp_path / "a.yaml", "tag: ${from_yaml:data.yaml,dataset.name}\n")
    assert load_mapping_yaml(path).tag == "wikitext"


def test_from_yaml_resolver_follows_the_referenced_files_default(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base_limits.yaml", "admit: 4\n")
    write(tmp_path / "limits.yaml", "_default: base_limits.yaml\nadmit: 32\n")
    # The referenced file is read through the composing loader, so it composes first and its own value
    # wins over the default it pulls in — the resolver sees 32, not 4.
    path = write(tmp_path / "a.yaml", "concurrency: ${from_yaml:limits.yaml,admit}\n")
    assert load_mapping_yaml(path).concurrency == 32


def test_from_yaml_resolver_rejects_a_missing_key(tmp_path, monkeypatch, write):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "data.yaml", "dataset:\n  name: wikitext\n")
    path = write(tmp_path / "a.yaml", "tag: ${from_yaml:data.yaml,dataset.nope}\n")
    cfg = load_mapping_yaml(path)
    with pytest.raises(Exception, match="has no key"):  # wrapped by omegaconf on access
        _ = OmegaConf.to_container(cfg, resolve=True)


def test_an_unresolvable_interpolation_does_not_break_composition(tmp_path, monkeypatch, write):
    # A leaf may name a key that only exists once everything is merged. Composition must not touch it.
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "root: /data\n")
    child = write(tmp_path / "child.yaml", "_default: base.yaml\nnested:\n  out: ${root}/x\n")
    assert load_mapping_yaml(child).nested.out == "/data/x"

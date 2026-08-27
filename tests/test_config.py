# The YAML layer: load_yaml / load_mapping_yaml, `defaults:` composition, and the resolvers.

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from slimconfig import load_mapping_yaml, load_yaml


def write(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_yaml_returns_plain_dict(tmp_path):
    path = write(tmp_path / "a.yaml", "a: 1\nb: [1, 2]\n")
    assert load_yaml(path) == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("body", ["- 1\n- 2\n", "42\n"])
def test_load_yaml_rejects_non_mapping(tmp_path, body):
    path = write(tmp_path / "a.yaml", body)
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_yaml(path)


def test_load_yaml_rejects_invalid_yaml(tmp_path):
    path = write(tmp_path / "a.yaml", "a: [1,\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_yaml(path)


def test_load_mapping_yaml_without_defaults(tmp_path):
    path = write(tmp_path / "a.yaml", "a: 1\nnested: {x: 2}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.a == 1
    assert cfg.nested.x == 2


# (A bare string yaml is NOT in this list: OmegaConf parses `just a string` into the mapping
# {"just a string": None}, so it never reaches the type check.)
@pytest.mark.parametrize("body", ["- 1\n", "42\n"])
def test_load_mapping_yaml_rejects_non_mapping(tmp_path, body):
    path = write(tmp_path / "a.yaml", body)
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        load_mapping_yaml(path)


def test_load_mapping_yaml_missing_file_is_not_a_parse_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mapping_yaml(str(tmp_path / "nope.yaml"))


# ── `defaults:` composition ──────────────────────────────────────────────────


def test_defaults_current_file_wins_and_merges_deeply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "a: 1\nb: 2\nnested: {x: 1, y: 1}\n")
    child = write(tmp_path / "child.yaml", "defaults: [base.yaml]\nb: 20\nnested: {y: 20}\n")
    cfg = load_mapping_yaml(child)
    assert cfg.a == 1  # inherited
    assert cfg.b == 20  # overridden
    assert (cfg.nested.x, cfg.nested.y) == (1, 20)  # deep merge, not replacement
    assert "defaults" not in cfg


def test_defaults_later_entry_wins_over_earlier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "one.yaml", "a: 1\nshared: from_one\n")
    write(tmp_path / "two.yaml", "b: 2\nshared: from_two\n")
    child = write(tmp_path / "child.yaml", "defaults: [one.yaml, two.yaml]\n")
    cfg = load_mapping_yaml(child)
    assert (cfg.a, cfg.b, cfg.shared) == (1, 2, "from_two")


def test_defaults_compose_recursively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "grand.yaml", "a: 1\n")
    write(tmp_path / "parent.yaml", "defaults: [grand.yaml]\nb: 2\n")
    child = write(tmp_path / "child.yaml", "defaults: [parent.yaml]\nc: 3\n")
    cfg = load_mapping_yaml(child)
    assert (cfg.a, cfg.b, cfg.c) == (1, 2, 3)


def test_defaults_resolve_against_cwd_not_the_including_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    write(tmp_path / "configs" / "base.yaml", "a: 1\n")
    # The `defaults` entry is written from the project root, even though the file that carries it
    # sits one directory down next to base.yaml.
    child = write(tmp_path / "configs" / "child.yaml", "defaults: [configs/base.yaml]\nb: 2\n")
    cfg = load_mapping_yaml(child)
    assert (cfg.a, cfg.b) == (1, 2)


def test_defaults_cycle_is_detected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "a.yaml", "defaults: [b.yaml]\n")
    write(tmp_path / "b.yaml", "defaults: [a.yaml]\n")
    with pytest.raises(ValueError, match="`defaults` cycle detected"):
        load_mapping_yaml(str(tmp_path / "a.yaml"))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("defaults: base.yaml\n", "must be a list of yaml paths"),
        ("defaults: [{a: 1}]\n", "must be a string path"),
    ],
)
def test_defaults_shape_is_validated(tmp_path, monkeypatch, body, match):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", "a: 1\n")
    path = write(tmp_path / "child.yaml", body)
    with pytest.raises(ValueError, match=match):
        load_mapping_yaml(path)


# ── resolvers ────────────────────────────────────────────────────────────────


def test_now_resolver_stamps_a_value(tmp_path):
    path = write(tmp_path / "a.yaml", "run_dir: runs/${now:%Y}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.run_dir.startswith("runs/2")
    assert len(cfg.run_dir) == len("runs/YYYY")


def test_now_resolver_is_one_consistent_stamp(tmp_path):
    path = write(tmp_path / "a.yaml", "one: ${now:%Y%m%d-%H%M%S}\ntwo: ${now:%Y%m%d-%H%M%S}\n")
    cfg = load_mapping_yaml(path)
    assert cfg.one == cfg.two


def test_from_yaml_resolver_reads_another_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "data.yaml", "dataset:\n  name: wikitext\n")
    path = write(tmp_path / "a.yaml", "tag: ${from_yaml:data.yaml,dataset.name}\n")
    assert load_mapping_yaml(path).tag == "wikitext"


def test_from_yaml_resolver_follows_the_referenced_files_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base_limits.yaml", "admit: 4\n")
    write(tmp_path / "limits.yaml", "defaults:\n  - base_limits.yaml\nadmit: 32\n")
    # The referenced file is read through load_mapping_yaml, so it composes first and its own value
    # wins over the default it pulls in — the resolver sees 32, not 4.
    path = write(tmp_path / "a.yaml", "concurrency: ${from_yaml:limits.yaml,admit}\n")
    assert load_mapping_yaml(path).concurrency == 32


def test_from_yaml_resolver_rejects_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "data.yaml", "dataset:\n  name: wikitext\n")
    path = write(tmp_path / "a.yaml", "tag: ${from_yaml:data.yaml,dataset.nope}\n")
    cfg = load_mapping_yaml(path)
    with pytest.raises(Exception, match="has no key"):  # wrapped by omegaconf on access
        _ = OmegaConf.to_container(cfg, resolve=True)

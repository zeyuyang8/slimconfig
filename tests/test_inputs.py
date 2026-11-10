# Inputs: a config field typed with an Output class is a JSON file's path in YAML, read into that class.

from __future__ import annotations

import json
from pathlib import Path

import fixtures
import pytest

from slimconfig import load_config, start_run
from slimconfig.outputs import source_of


def replay(tmp_path, write, prior: str, baseline: str = "null") -> str:
    return write(tmp_path / "replay.yaml", f"""
prior: {prior}
baseline: {baseline}
judge > fixtures.Judge:
  reference: {prior}
per_stage > dict[fixtures.Stage, fixtures.Judge]:
  warmup:
    reference: {prior}
""", schema="fixtures.ReplayConfig")


def messages(tmp_path, value) -> str:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"_": "fixtures.Messages", **value}), encoding="utf-8")
    return str(path)


def test_an_input_is_read_into_its_class(tmp_path, write):
    prior = messages(tmp_path, {"turns": ["hi", "hello"]})
    cfg = load_config(fixtures.ReplayConfig, [replay(tmp_path, write, prior)])
    assert cfg.prior == fixtures.Messages(["hi", "hello"])
    assert cfg.baseline is None
    assert cfg.judge.reference == cfg.prior
    assert cfg.per_stage[fixtures.Stage.warmup].reference == cfg.prior
    assert source_of(cfg.prior) == prior


def test_the_snapshot_names_each_input_by_its_path_and_reloads(tmp_path, write):
    prior = messages(tmp_path, {"turns": ["hi"]})
    cfg = load_config(fixtures.ReplayConfig, [replay(tmp_path, write, prior)])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    snapshot = Path(run_dir) / "config.yaml"
    assert f"prior: {prior}\n" in snapshot.read_text()
    assert load_config(fixtures.ReplayConfig, [str(snapshot)]) == cfg


def test_an_input_file_is_checked_against_its_class(tmp_path, write):
    prior = messages(tmp_path, {"turns": "hi"})
    with pytest.raises(ValueError, match=r"ReplayConfig.prior reads .*prior.json.*Messages.turns is not a list"):
        load_config(fixtures.ReplayConfig, [replay(tmp_path, write, prior)])


def test_an_input_file_names_its_class(tmp_path, write):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"turns": []}), encoding="utf-8")
    with pytest.raises(ValueError, match=r'ReplayConfig.prior reads .*prior.json.* names no class under "_"'):
        load_config(fixtures.ReplayConfig, [replay(tmp_path, write, str(prior))])


def test_a_missing_input_file_names_the_field(tmp_path, write):
    with pytest.raises(ValueError, match=r"ReplayConfig.prior reads 'nowhere.json'"):
        load_config(fixtures.ReplayConfig, [replay(tmp_path, write, "nowhere.json")])


def test_an_input_is_a_path_not_the_value_itself(tmp_path, write):
    prior = messages(tmp_path, {"turns": []})
    with pytest.raises(ValueError, match=r"baseline"):
        load_config(fixtures.ReplayConfig, [replay(tmp_path, write, prior, baseline="{turns: []}")])


def test_an_output_not_read_from_a_file_has_no_path():
    with pytest.raises(ValueError, match="was not read from a file"):
        source_of(fixtures.Messages([]))

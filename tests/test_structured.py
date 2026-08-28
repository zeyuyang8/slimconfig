# The typed layer: merge_specs / load_config / peek. The run layer lives in test_runs.py.

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from omegaconf import MISSING

from slimconfig import load_config, merge_specs, peek


@dataclass
class Optim:
    lr: float = MISSING
    warmup_steps: int = MISSING


@dataclass
class TrainConfig:
    run_dir: str = MISSING
    model: str = MISSING
    tags: list[str] = MISSING
    resume_from: str | None = MISSING
    optim: Optim = field(default_factory=Optim)


FULL = """
run_dir: runs/demo
model: llama
tags: []
resume_from: null
optim:
  lr: 0.0002
  warmup_steps: 100
"""


def write(path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


# ── merge_specs ──────────────────────────────────────────────────────────────


def test_merge_specs_later_spec_wins(tmp_path):
    path = write(tmp_path / "a.yaml", "a: 1\nb: 2\n")
    merged = merge_specs([path, "b=20", {"c": 30}])
    assert (merged.a, merged.b, merged.c) == (1, 20, 30)


def test_merge_specs_parses_dotted_overrides(tmp_path):
    path = write(tmp_path / "a.yaml", FULL)
    merged = merge_specs([path, "optim.lr=0.5"])
    assert merged.optim.lr == 0.5
    assert merged.optim.warmup_steps == 100  # untouched


def test_merge_specs_rejects_a_spec_that_is_neither(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither a file nor a key=value override"):
        merge_specs(["configs/typo.yaml"])


# ── load_config ──────────────────────────────────────────────────────────────


def test_load_config_returns_a_populated_instance(tmp_path):
    cfg = load_config(TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    assert isinstance(cfg, TrainConfig)
    assert isinstance(cfg.optim, Optim)
    assert (cfg.model, cfg.tags, cfg.resume_from) == ("llama", [], None)
    assert cfg.optim.lr == pytest.approx(2e-4)


def test_load_config_overrides_win_over_the_file(tmp_path):
    path = write(tmp_path / "a.yaml", FULL)
    cfg = load_config(TrainConfig, [path, "model=qwen", "optim.warmup_steps=7"])
    assert (cfg.model, cfg.optim.warmup_steps) == ("qwen", 7)


def test_load_config_requires_every_leaf(tmp_path):
    path = write(tmp_path / "a.yaml", "run_dir: runs/demo\nmodel: llama\ntags: []\n")
    with pytest.raises(ValueError, match=r"missing required field\(s\): resume_from, optim.lr"):
        load_config(TrainConfig, [path])


def test_load_config_rejects_unknown_keys(tmp_path):
    path = write(tmp_path / "a.yaml", FULL + "typo_key: 1\n")
    with pytest.raises(Exception, match="typo_key"):
        load_config(TrainConfig, [path])


def test_load_config_rejects_a_wrongly_typed_value(tmp_path):
    path = write(tmp_path / "a.yaml", FULL.replace("warmup_steps: 100", "warmup_steps: many"))
    with pytest.raises(Exception, match="warmup_steps"):
        load_config(TrainConfig, [path])


def test_load_config_composes_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "base.yaml", FULL)
    child = write(tmp_path / "child.yaml", "defaults: [base.yaml]\nmodel: qwen\n")
    cfg = load_config(TrainConfig, [child])
    assert (cfg.model, cfg.optim.warmup_steps) == ("qwen", 100)


# ── peek ─────────────────────────────────────────────────────────────────────


def test_peek_reads_a_top_level_key_without_validation(tmp_path):
    path = write(tmp_path / "a.yaml", "mode: train\nunknown_key: 1\n")
    assert peek([path], "mode") == "train"
    assert peek([path], "absent") is None
    assert peek([path, "mode=eval"], "mode") == "eval"

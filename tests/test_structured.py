# The typed layer: merge_specs / load_config / peek, plus start_run and dispatch.

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from omegaconf import MISSING, OmegaConf

from slimconfig import dispatch, load_config, merge_specs, peek, start_run


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


# ── start_run ────────────────────────────────────────────────────────────────


def test_start_run_writes_a_resolved_snapshot_and_meta(tmp_path):
    spec = write(tmp_path / "a.yaml", FULL)
    run_dir = start_run(str(tmp_path / "runs" / "demo"), [spec])
    snapshot = OmegaConf.load(f"{run_dir}/config.yaml")
    assert snapshot.model == "llama"
    meta = json.loads((tmp_path / "runs" / "demo" / "run_meta.json").read_text())
    assert {"argv", "cwd", "started", "host"} <= meta.keys()


def test_start_run_snapshot_is_rerunnable_in_place(tmp_path):
    # Re-running a run from its own snapshot passes the very file start_run overwrites.
    spec = write(tmp_path / "a.yaml", FULL)
    run_dir = start_run(str(tmp_path / "run"), [spec])
    snapshot = f"{run_dir}/config.yaml"
    start_run(run_dir, [snapshot])
    assert load_config(TrainConfig, [snapshot]).model == "llama"


def test_start_run_accepts_a_dataclass_instance(tmp_path):
    cfg = load_config(TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    assert OmegaConf.load(f"{run_dir}/config.yaml").optim.warmup_steps == 100


def test_start_run_survives_an_unsnapshottable_config(tmp_path, capsys):
    start_run(str(tmp_path / "run"), object())  # provenance never aborts a run
    assert "could not snapshot" in capsys.readouterr().out
    assert (tmp_path / "run").is_dir()


# ── dispatch ─────────────────────────────────────────────────────────────────


# A schema reached through dispatch declares `mode` itself — the key is part of the config, and
# unknown keys are rejected.
@dataclass
class JobConfig:
    mode: str = MISSING
    run_dir: str = MISSING
    model: str = MISSING


def test_dispatch_loads_the_schema_for_a_tuple_entry(tmp_path):
    seen = {}

    def train(cfg: JobConfig) -> int:
        seen["model"] = cfg.model
        return 0

    path = write(tmp_path / "a.yaml", f"mode: train\nrun_dir: {tmp_path / 'run'}\nmodel: llama\n")
    assert dispatch({"train": (JobConfig, train)}, [path]) == 0
    assert seen["model"] == "llama"
    assert (tmp_path / "run" / "config.yaml").is_file()


def test_dispatch_passes_raw_specs_to_a_bare_handler(tmp_path):
    seen = {}

    def sweep(specs) -> int:
        seen["specs"] = specs
        return 3

    path = write(tmp_path / "a.yaml", f"mode: sweep\nrun_dir: {tmp_path / 'run'}\n")
    assert dispatch({"sweep": sweep}, [path]) == 3
    assert seen["specs"] == [path]


def test_dispatch_rejects_an_unknown_mode(tmp_path):
    path = write(tmp_path / "a.yaml", "mode: nope\nrun_dir: runs/x\n")
    with pytest.raises(SystemExit, match="must set `mode` to one of train"):
        dispatch({"train": (TrainConfig, lambda cfg: 0)}, [path])


def test_dispatch_requires_a_run_dir(tmp_path):
    path = write(tmp_path / "a.yaml", "mode: train\n")
    with pytest.raises(SystemExit, match="must set `run_dir`"):
        dispatch({"train": (TrainConfig, lambda cfg: 0)}, [path])

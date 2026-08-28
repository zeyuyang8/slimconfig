# The run layer: start_run / tee_stdout / open_run, the @run entry-point decorator, and dispatch.

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from omegaconf import MISSING, OmegaConf

from slimconfig import dispatch, load_config, open_run, run, start_run, tee_stdout


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


# ── tee_stdout ───────────────────────────────────────────────────────────────


def test_tee_stdout_writes_to_the_file_and_the_terminal(tmp_path, capsys):
    log = str(tmp_path / "logs" / "run.log")
    with tee_stdout(log):
        print("hello")
    print("after")
    assert capsys.readouterr().out == "hello\nafter\n"  # still on stdout, and the tee is undone
    assert (tmp_path / "logs" / "run.log").read_text() == "hello\n"


def test_tee_stdout_appends_across_runs_with_a_banner(tmp_path):
    log = str(tmp_path / "run.log")
    with tee_stdout(log, banner="=== first ==="):
        print("one")
    with tee_stdout(log, banner="=== second ==="):
        print("two")
    text = (tmp_path / "run.log").read_text()
    assert text == "=== first ===\none\n=== second ===\ntwo\n"


def test_tee_stdout_is_undone_after_an_exception(tmp_path, capsys):
    with pytest.raises(RuntimeError), tee_stdout(str(tmp_path / "run.log")):
        print("before the boom")
        raise RuntimeError("boom")
    print("after")
    assert capsys.readouterr().out == "before the boom\nafter\n"
    assert (tmp_path / "run.log").read_text() == "before the boom\n"


# ── open_run ─────────────────────────────────────────────────────────────────


def test_open_run_snapshots_and_logs(tmp_path):
    run_dir = str(tmp_path / "run")
    with open_run([{"run_dir": run_dir, "model": "llama"}], log="job.log") as opened:
        print("working")
    assert opened == run_dir
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert "working" in (tmp_path / "run" / "job.log").read_text()


def test_open_run_without_a_log(tmp_path):
    with open_run([{"run_dir": str(tmp_path / "run")}], log=None):
        print("working")
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert list((tmp_path / "run").glob("*.log")) == []


def test_open_run_requires_a_run_dir():
    with pytest.raises(SystemExit, match="must set `run_dir`"), open_run([{"model": "llama"}]):
        pass


def test_open_run_reads_run_dir_off_a_loaded_config(tmp_path):
    cfg = load_config(TrainConfig, [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}"])
    with open_run(cfg, log=None) as run_dir:
        pass
    assert run_dir == str(tmp_path / "run")
    assert OmegaConf.load(f"{run_dir}/config.yaml").model == "llama"


# ── @run ─────────────────────────────────────────────────────────────────────


def test_run_loads_the_config_and_opens_the_folder(tmp_path):
    @run(TrainConfig)
    def train(cfg: TrainConfig) -> int:
        print(f"training {cfg.model}")
        (tmp_path / "run" / "result.txt").write_text(cfg.model)  # results land in the run folder
        return 0

    path = write(tmp_path / "a.yaml", FULL)
    assert train([path, f"run_dir={tmp_path / 'run'}"]) == 0
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "run_meta.json").is_file()
    assert (tmp_path / "run" / "result.txt").read_text() == "llama"
    assert "training llama" in (tmp_path / "run" / "train.log").read_text()


def test_run_defaults_its_specs_to_argv(tmp_path, monkeypatch):
    @run(TrainConfig)
    def train(cfg: TrainConfig) -> str:
        return cfg.model

    path = write(tmp_path / "a.yaml", FULL)
    monkeypatch.setattr("sys.argv", ["train.py", path, f"run_dir={tmp_path / 'run'}", "model=qwen"])
    assert train() == "qwen"


def test_run_names_the_log_and_can_turn_it_off(tmp_path):
    @run(TrainConfig, log="quality.log")
    def named(cfg: TrainConfig) -> None:
        print("named")

    @run(TrainConfig, log=None)
    def quiet(cfg: TrainConfig) -> None:
        print("quiet")

    path = write(tmp_path / "a.yaml", FULL)
    named([path, f"run_dir={tmp_path / 'run'}"])
    quiet([path, f"run_dir={tmp_path / 'run'}"])
    assert (tmp_path / "run" / "quality.log").is_file()
    assert sorted(p.name for p in (tmp_path / "run").glob("*.log")) == ["quality.log"]


def test_run_bare_hands_the_raw_specs_over(tmp_path):
    seen = {}

    @run
    def sweep(specs) -> int:
        seen["specs"] = specs  # a handler whose schema depends on another field loads its own
        return 3

    path = write(tmp_path / "a.yaml", f"run_dir: {tmp_path / 'run'}\n")
    assert sweep([path]) == 3
    assert seen["specs"] == [path]
    assert (tmp_path / "run" / "sweep.log").is_file()


def test_run_requires_a_run_dir(tmp_path):
    @run(TrainConfig)
    def train(cfg: TrainConfig) -> int:
        raise AssertionError("must not run")

    path = write(tmp_path / "a.yaml", FULL.replace("run_dir: runs/demo\n", ""))
    with pytest.raises(SystemExit, match="must set `run_dir`"):
        train([path])


def test_run_keeps_the_wrapped_function_identifiable():
    @run(TrainConfig)
    def train(cfg: TrainConfig) -> None:
        """Docstring kept."""

    assert (train.__name__, train.__doc__) == ("train", "Docstring kept.")


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
        print("trained")
        return 0

    path = write(tmp_path / "a.yaml", f"mode: train\nrun_dir: {tmp_path / 'run'}\nmodel: llama\n")
    assert dispatch({"train": (JobConfig, train)}, [path]) == 0
    assert seen["model"] == "llama"
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert "trained" in (tmp_path / "run" / "train.log").read_text()  # log named after the mode


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

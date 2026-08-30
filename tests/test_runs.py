# The run layer: start_run / tee_stdout, and the `run` launcher every script's __main__ is.

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from omegaconf import MISSING, OmegaConf

from slimconfig import load_config, run, start_run, tee_stdout


@dataclass
class Optim:
    lr: float = MISSING
    warmup_steps: int = MISSING


@dataclass
class TrainConfig:
    run_dir: str = MISSING
    log: str | None = MISSING
    model: str = MISSING
    tags: list[str] = MISSING
    resume_from: str | None = MISSING
    optim: Optim = field(default_factory=Optim)


# The two ways a schema can fail the launcher's one requirement.
@dataclass
class NoFolderConfig:
    model: str = MISSING


@dataclass
class NoLogConfig:
    run_dir: str = MISSING
    model: str = MISSING


FULL = """
run_dir: runs/demo
log: null
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
    cfg = load_config(TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "runs" / "demo"), cfg)
    snapshot = OmegaConf.load(f"{run_dir}/config.yaml")
    assert snapshot.model == "llama"
    meta = json.loads((tmp_path / "runs" / "demo" / "metadata.json").read_text())
    assert {"argv", "cwd", "started", "host"} <= meta.keys()


def test_start_run_snapshot_is_rerunnable_in_place(tmp_path):
    # Re-running a run from its own snapshot passes the very file start_run overwrites.
    run_dir = start_run(str(tmp_path / "run"), load_config(TrainConfig, [write(tmp_path / "a.yaml", FULL)]))
    snapshot = f"{run_dir}/config.yaml"
    start_run(run_dir, load_config(TrainConfig, [snapshot]))
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


# ── run ──────────────────────────────────────────────────────────────────────


# `run` is a process boundary: it exits with the status the function returned. Every test below launches
# it the way a shell would — the specs on the command line — and reads that status back off the
# SystemExit, which is what the shell would see.
def launch(monkeypatch, specs, function, *args, **overrides) -> object:
    monkeypatch.setattr("sys.argv", ["train.py", *specs])
    with pytest.raises(SystemExit) as exit_info:
        run(function, *args, **overrides)
    return exit_info.value.code


def test_run_loads_the_config_and_opens_the_folder(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> int:
        print(f"training {cfg.model}")
        (tmp_path / "run" / "result.txt").write_text(cfg.model)  # results land in the run folder
        return 0

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}", "log=train.log"]
    assert launch(monkeypatch, specs, train) == 0
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "metadata.json").is_file()
    assert (tmp_path / "run" / "result.txt").read_text() == "llama"
    assert "training llama" in (tmp_path / "run" / "train.log").read_text()


def test_run_exits_zero_when_the_function_returns_nothing(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> None:
        return None

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}"]
    assert launch(monkeypatch, specs, train) is None


def test_run_takes_its_specs_from_the_command_line(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> int:
        return 0 if cfg.model == "qwen" else 1

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}", "model=qwen"]
    assert launch(monkeypatch, specs, train) == 0


def test_run_takes_a_config_path_and_overrides_from_the_caller(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> int:
        return 0 if (cfg.model, cfg.optim.lr) == ("qwen", 0.5) else 1

    path = write(tmp_path / "a.yaml", FULL)
    # The command line is ignored when the caller names the config itself.
    code = launch(
        monkeypatch, ["ignored.yaml"], train, path,
        **{"model": "qwen", "optim.lr": 0.5, "run_dir": str(tmp_path / "run")},
    )
    assert code == 0
    assert (tmp_path / "run" / "config.yaml").is_file()


def test_run_reports_usage_when_given_no_config(monkeypatch):
    def train(cfg: TrainConfig) -> int:
        raise AssertionError("must not run")

    assert "usage: train.py <config.yaml>" in str(launch(monkeypatch, [], train))


def test_run_writes_the_log_the_config_names_and_nothing_when_it_is_null(tmp_path, monkeypatch):
    def loud(cfg: TrainConfig) -> None:
        print("named")

    def quiet(cfg: TrainConfig) -> None:
        print("quiet")

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}"]
    launch(monkeypatch, [*specs, "log=quality.log"], loud)
    launch(monkeypatch, [*specs, "log=null"], quiet)
    assert "named" in (tmp_path / "run" / "quality.log").read_text()
    assert sorted(p.name for p in (tmp_path / "run").glob("*.log")) == ["quality.log"]


def test_run_puts_the_log_where_the_config_asks(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> None:
        print("nested")

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}", "log=logs/train.log"]
    launch(monkeypatch, specs, train)
    assert "nested" in (tmp_path / "run" / "logs" / "train.log").read_text()


def test_run_requires_a_folder_to_write_in(tmp_path, monkeypatch):
    def train(cfg: TrainConfig) -> int:
        raise AssertionError("must not run")

    specs = [write(tmp_path / "a.yaml", FULL.replace("run_dir: runs/demo", "run_dir: ''"))]
    assert "nowhere to write" in str(launch(monkeypatch, specs, train))


# ── run: the function IS its config ──────────────────────────────────────────


def test_run_takes_the_schema_off_the_function_s_annotation(tmp_path, monkeypatch):
    seen = {}

    def train(cfg: TrainConfig) -> int:
        seen["type"] = type(cfg).__name__
        seen["lr"] = cfg.optim.lr
        return 0

    specs = [write(tmp_path / "a.yaml", FULL), f"run_dir={tmp_path / 'run'}"]
    assert launch(monkeypatch, specs, train) == 0
    assert seen == {"type": "TrainConfig", "lr": 0.0002}  # loaded, typed, not a mapping of strings


def test_run_rejects_a_function_that_is_not_one_of():
    with pytest.raises(TypeError, match="function of one config argument"):
        run("train")


def test_run_rejects_a_function_of_the_wrong_arity():
    def train(cfg: TrainConfig, extra: int) -> int:
        return 0

    with pytest.raises(TypeError, match="exactly one argument"):
        run(train)


def test_run_rejects_an_unannotated_function():
    def train(cfg) -> int:
        return 0

    with pytest.raises(TypeError, match="must be annotated with its config class"):
        run(train)


def test_run_rejects_a_config_class_without_a_run_dir_or_a_log():
    def train(cfg: NoFolderConfig) -> int:
        return 0

    with pytest.raises(TypeError, match="missing `run_dir` and `log`"):
        run(train)


def test_run_rejects_a_config_class_without_a_log():
    def train(cfg: NoLogConfig) -> int:
        return 0

    with pytest.raises(TypeError, match="missing `log`"):
        run(train)

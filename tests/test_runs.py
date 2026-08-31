# The run layer: start_run / tee_stdout, and the `run` launcher every script's __main__ is.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import fixtures
import pytest
from omegaconf import OmegaConf

from slimconfig import load_config, run, start_run, tee_stdout

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


# ── start_run ────────────────────────────────────────────────────────────────


def test_start_run_writes_a_resolved_snapshot_and_meta(tmp_path, write):
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "runs" / "demo"), cfg)
    snapshot = OmegaConf.load(f"{run_dir}/config.yaml")
    assert snapshot.model == "llama"
    meta = json.loads((tmp_path / "runs" / "demo" / "metadata.json").read_text())
    assert {"argv", "cwd", "run_dir", "started", "host"} <= meta.keys()
    assert meta["run_dir"] == str(tmp_path / "runs" / "demo")


def test_the_snapshot_names_the_class_so_it_is_a_config_like_any_other(tmp_path, write):
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    start_run(str(tmp_path / "run"), cfg)
    assert (tmp_path / "run" / "config.yaml").read_text().startswith("_schema: fixtures.TrainConfig\n")


def test_start_run_snapshot_is_rerunnable_in_place(tmp_path, write):
    # Re-running a run from its own snapshot passes the very file start_run overwrites.
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    snapshot = f"{run_dir}/config.yaml"
    start_run(run_dir, load_config(fixtures.TrainConfig, [snapshot]))
    assert load_config(fixtures.TrainConfig, [snapshot]).model == "llama"


def test_a_snapshot_of_a_matrix_stamps_its_blocks_and_still_reloads(tmp_path, write):
    # The layers are PARTIAL, so most of `base` is unset — an unset group is not a block to stamp.
    body = "stage: main\nper_model: {}\nbase:\n  _schema: fixtures.TrainPart\n  model: a\nper_stage:\n"
    body += "  main:\n    optim:\n      _schema: fixtures.TrainPart.OptimPart\n      lr: 0.1\n"
    cfg = load_config(fixtures.MatrixConfig, [write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    text = (tmp_path / "run" / "config.yaml").read_text()
    assert text.startswith("_schema: fixtures.MatrixConfig\n")
    assert "  _schema: fixtures.TrainPart\n" in text  # the `base` group, named
    assert "_schema: fixtures.MatrixConfig\nper_stage:\n  main:\n" not in text  # a table entry, not named
    assert load_config(fixtures.MatrixConfig, [f"{run_dir}/config.yaml"]) == cfg


def test_a_snapshot_leaves_an_unset_table_alone(tmp_path, write):
    # A layer's unset table is not a table in the snapshot, it is `???` — there is no block to stamp.
    body = "per_stage: {}\nbase:\n  _schema: fixtures.SearchPart\n  trials: 4\n"
    path = write(tmp_path / "s.yaml", body, schema="fixtures.SearchMatrix")
    cfg = load_config(fixtures.SearchMatrix, [path])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    assert load_config(fixtures.SearchMatrix, [f"{run_dir}/config.yaml"]) == cfg


def test_start_run_accepts_a_dataclass_instance(tmp_path, write):
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
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
# it the way a shell would — the specs and the flags on the command line — and reads that status back off
# the SystemExit, which is what the shell would see.
def launch(monkeypatch, argv, function, *args, **kwargs) -> object:
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run(function, *args, **kwargs)
    return exit_info.value.code


def test_run_loads_the_config_and_opens_the_folder(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig, run_dir: str) -> int:
        print(f"training {cfg.model}")
        (tmp_path / "run" / "result.txt").write_text(cfg.model)  # results land in the run folder
        return 0

    argv = [write(tmp_path / "a.yaml", FULL), "--run-dir", str(tmp_path / "run")]
    assert launch(monkeypatch, argv, train, log="train.log") == 0
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "metadata.json").is_file()
    assert (tmp_path / "run" / "result.txt").read_text() == "llama"
    assert "training llama" in (tmp_path / "run" / "train.log").read_text()


def test_run_exits_zero_when_the_function_returns_nothing(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> None:
        return None

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}"]
    assert launch(monkeypatch, argv, train) is None


def test_run_takes_its_specs_from_the_command_line(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        return 0 if cfg.model == "qwen" else 1

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}", "model=qwen"]
    assert launch(monkeypatch, argv, train) == 0


def test_run_takes_a_config_path_and_overrides_from_the_caller(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        return 0 if (cfg.model, cfg.optim.lr) == ("qwen", 0.5) else 1

    path = write(tmp_path / "a.yaml", FULL)
    # The command line is ignored when the caller names the config itself.
    code = launch(
        monkeypatch, ["ignored.yaml"], train, path,
        run_dir=str(tmp_path / "run"), **{"model": "qwen", "optim.lr": 0.5},
    )
    assert code == 0
    assert (tmp_path / "run" / "config.yaml").is_file()


def test_run_reports_usage_when_given_no_config(monkeypatch):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    assert "usage: train.py <config.yaml>" in str(launch(monkeypatch, [], train))


# ── run: where it writes is the launcher's, not the config's ─────────────────


def test_the_script_can_name_the_run_dir(tmp_path, monkeypatch, write):
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["dir"] = run_dir

    launch(monkeypatch, [write(tmp_path / "a.yaml", FULL)], train, run_dir=str(tmp_path / "fixed"))
    assert seen["dir"] == str(tmp_path / "fixed")


def test_the_command_line_wins_over_the_script(tmp_path, monkeypatch, write):
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["dir"] = run_dir

    argv = [write(tmp_path / "a.yaml", FULL), "--run-dir", str(tmp_path / "cli")]
    launch(monkeypatch, argv, train, run_dir=str(tmp_path / "script"))
    assert seen["dir"] == str(tmp_path / "cli")


def test_the_run_dir_can_be_a_function_of_the_config(tmp_path, monkeypatch, write):
    # An identity-addressed output tree is one rule in code, not the same interpolation in every file.
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["dir"] = run_dir

    argv = [write(tmp_path / "a.yaml", FULL)]
    launch(monkeypatch, argv, train, run_dir=lambda cfg: str(tmp_path / f"runs/{cfg.model}"))
    assert seen["dir"] == str(tmp_path / "runs" / "llama")


def test_the_run_dir_can_be_named_after_the_config_that_produced_it(tmp_path, monkeypatch, write):
    # The commonest naming rule there is, and the launcher is the only one who knows which file it was.
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["dir"] = run_dir

    def named(cfg, config: str) -> str:
        return str(tmp_path / "runs" / Path(config).stem)

    launch(monkeypatch, [write(tmp_path / "sweep_a.yaml", FULL)], train, run_dir=named)
    assert seen["dir"] == str(tmp_path / "runs" / "sweep_a")


def test_a_run_with_no_config_file_names_no_config(tmp_path, monkeypatch):
    # Overrides alone are a legal launch, so the second argument has to have an empty answer.
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["dir"] = run_dir

    def named(cfg, config: str) -> str:
        return str(tmp_path / "runs" / (Path(config).stem or "unnamed"))

    argv = ["model=llama", "tags=[]", "resume_from=null", "optim.lr=0.1",
            "optim.warmup_steps=1", "data.path=x"]
    launch(monkeypatch, argv, train, run_dir=named)
    assert seen["dir"] == str(tmp_path / "runs" / "unnamed")


def test_run_requires_a_folder_to_write_in(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    assert "nowhere to write" in str(launch(monkeypatch, [write(tmp_path / "a.yaml", FULL)], train))


def test_a_config_may_not_smuggle_the_run_dir_back_in(tmp_path, monkeypatch, write):
    # `run_dir` is not a field of any config class, so setting it is an unknown key like any other.
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    argv = [write(tmp_path / "a.yaml", FULL + "run_dir: runs/sneaky\n"), f"--run-dir={tmp_path}"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(Exception, match="run_dir"):
        run(train)


# ── run: the log ─────────────────────────────────────────────────────────────


def test_the_script_names_the_log_and_the_command_line_can_move_it(tmp_path, monkeypatch, write):
    def loud(cfg: fixtures.TrainConfig) -> None:
        print("named")

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}"]
    launch(monkeypatch, argv, loud, log="train.log")
    launch(monkeypatch, [*argv, "--log", "other.log"], loud, log="train.log")
    assert "named" in (tmp_path / "run" / "train.log").read_text()
    assert "named" in (tmp_path / "run" / "other.log").read_text()


def test_no_log_turns_off_the_one_the_script_asked_for(tmp_path, monkeypatch, write):
    # Every rank of a distributed launch would otherwise append to the one path.
    def quiet(cfg: fixtures.TrainConfig) -> None:
        print("quiet")

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}", "--no-log"]
    launch(monkeypatch, argv, quiet, log="train.log")
    assert list((tmp_path / "run").glob("*.log")) == []


def test_a_script_that_asks_for_no_log_writes_none(tmp_path, monkeypatch, write):
    def quiet(cfg: fixtures.TrainConfig) -> None:
        print("quiet")

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}"]
    launch(monkeypatch, argv, quiet)
    assert list((tmp_path / "run").glob("*.log")) == []


def test_the_log_may_sit_in_a_subfolder(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> None:
        print("nested")

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}"]
    launch(monkeypatch, argv, train, log="logs/train.log")
    assert "nested" in (tmp_path / "run" / "logs" / "train.log").read_text()


def test_a_flag_with_no_value_is_an_error(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> None:
        raise AssertionError("must not run")

    argv = [write(tmp_path / "a.yaml", FULL), "--run-dir"]
    assert "--run-dir needs a value" in str(launch(monkeypatch, argv, train))


# ── run: the function IS its config ──────────────────────────────────────────


def test_run_takes_the_schema_off_the_function_s_annotation(tmp_path, monkeypatch, write):
    seen = {}

    def train(cfg: fixtures.TrainConfig) -> int:
        seen["type"] = type(cfg).__name__
        seen["lr"] = cfg.optim.lr
        return 0

    argv = [write(tmp_path / "a.yaml", FULL), f"--run-dir={tmp_path / 'run'}"]
    assert launch(monkeypatch, argv, train) == 0
    assert seen == {"type": "TrainConfig", "lr": 0.0002}  # loaded, typed, not a mapping of strings


SOLO_SCRIPT = '''
from dataclasses import dataclass, field
from omegaconf import MISSING
from slimconfig import run

@dataclass
class Optim:
    lr: float = MISSING

@dataclass
class SoloConfig:
    model: str = MISSING
    optim: Optim = field(default_factory=Optim)

def main(cfg: SoloConfig, run_dir: str) -> int:
    print(f"{cfg.model} {cfg.optim.lr}")
    return 0

if __name__ == "__main__":
    run(main, log="train.log")
'''


def test_a_one_file_script_can_name_its_own_classes(tmp_path):
    # A config class defined in the launched script lives in `__main__`; the YAML has to call it
    # something. Launched for real, because that is the only way `__main__` is what it will be.
    (tmp_path / "solo.py").write_text(SOLO_SCRIPT, encoding="utf-8")
    (tmp_path / "solo.yaml").write_text(
        "_schema: solo.SoloConfig\nmodel: llama\noptim:\n  _schema: solo.Optim\n  lr: 0.5\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    argv = [sys.executable, "solo.py", "solo.yaml", "--run-dir", "run"]
    done = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "llama 0.5" in done.stdout
    # ...and the snapshot names it the same way, so the run is repeatable from its own folder.
    assert (tmp_path / "run" / "config.yaml").read_text().startswith("_schema: solo.SoloConfig\n")


def test_run_rejects_a_function_that_is_not_one_of():
    with pytest.raises(TypeError, match="function of one config argument"):
        run("train")


def test_run_rejects_a_function_of_the_wrong_arity():
    def train(cfg: fixtures.TrainConfig, run_dir: str, extra: int) -> int:
        return 0

    with pytest.raises(TypeError, match="one or two arguments"):
        run(train)


def test_run_rejects_an_unannotated_function():
    def train(cfg) -> int:
        return 0

    with pytest.raises(TypeError, match="must be annotated with its config class"):
        run(train)


def test_run_rejects_a_second_argument_that_is_not_the_run_folder():
    def train(cfg: fixtures.TrainConfig, extra: int) -> int:
        return 0

    with pytest.raises(TypeError, match="is the run folder and must be annotated"):
        run(train)

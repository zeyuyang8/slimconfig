# The run layer: start_run / tee_stdout, and the `run` launcher every script's __main__ is.

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import fixtures
import pytest
from omegaconf import OmegaConf

from slimconfig import Config, Run, load_config, run, start_run, tee_stdout

FULL = """
model: llama
tags: []
resume_from: null
optim > fixtures.Optim:
  lr: 0.0002
  warmup_steps: 100
data > fixtures.Data:
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
    assert (tmp_path / "run" / "config.yaml").read_text().startswith("_ > fixtures.TrainConfig:\n")


def test_start_run_snapshot_is_rerunnable_in_place(tmp_path, write):
    # Re-running a run from its own snapshot passes the very file start_run overwrites.
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    snapshot = f"{run_dir}/config.yaml"
    start_run(run_dir, load_config(fixtures.TrainConfig, [snapshot]))
    assert load_config(fixtures.TrainConfig, [snapshot]).model == "llama"


def test_a_snapshot_of_a_matrix_stamps_its_blocks_and_still_reloads(tmp_path, write):
    # The layers are PARTIAL, so most of `base` is unset — an unset group is not a block to stamp.
    body = "stage: main\nper_model: {}\nbase > fixtures.TrainPart:\n  model: a\nper_stage:\n"
    body = body.replace("per_stage:\n", "per_stage > dict[fixtures.Stage, fixtures.TrainPart]:\n")
    body += "  main:\n    optim > fixtures.TrainPart.OptimPart:\n      lr: 0.1\n"
    cfg = load_config(fixtures.MatrixConfig, [write(tmp_path / "m.yaml", body, schema="fixtures.MatrixConfig")])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    text = (tmp_path / "run" / "config.yaml").read_text()
    assert text.startswith("_ > fixtures.MatrixConfig:\n")
    assert "base > fixtures.TrainPart:\n" in text  # the `base` group, named
    assert "per_stage > dict[fixtures.Stage, fixtures.TrainPart]:\n  main:\n" in text  # the table, once
    assert "  main >" not in text  # its entry, not named
    assert load_config(fixtures.MatrixConfig, [f"{run_dir}/config.yaml"]) == cfg


def test_a_snapshot_writes_an_enum_as_its_value(tmp_path, write):
    # `flux_dev` is the member's name; a config file spells it by its value, and so does the snapshot.
    body = "per_model > dict[fixtures.Backbone, fixtures.TrainPart]:\n  flux.1-dev:\n    model: a\n"
    cfg = load_config(fixtures.ModelMatrix, [write(tmp_path / "m.yaml", body, schema="fixtures.ModelMatrix")])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    text = (tmp_path / "run" / "config.yaml").read_text()
    assert "  flux.1-dev:\n" in text and "flux_dev" not in text
    assert load_config(fixtures.ModelMatrix, [f"{run_dir}/config.yaml"]) == cfg


def test_a_snapshot_leaves_an_unset_table_alone(tmp_path, write):
    # A layer's unset table is not a table in the snapshot, it is `???` — there is no block to stamp.
    body = "per_stage: {}\nbase > fixtures.SearchPart:\n  trials: 4\n"
    path = write(tmp_path / "s.yaml", body, schema="fixtures.SearchMatrix")
    cfg = load_config(fixtures.SearchMatrix, [path])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    assert load_config(fixtures.SearchMatrix, [f"{run_dir}/config.yaml"]) == cfg


def test_start_run_accepts_a_dataclass_instance(tmp_path, write):
    cfg = load_config(fixtures.TrainConfig, [write(tmp_path / "a.yaml", FULL)])
    run_dir = start_run(str(tmp_path / "run"), cfg)
    assert load_config(fixtures.TrainConfig, [f"{run_dir}/config.yaml"]).optim.warmup_steps == 100


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


def test_tee_stdout_rewrites_the_log_on_a_rerun(tmp_path):
    log = str(tmp_path / "run.log")
    with tee_stdout(log, banner="=== first ==="):
        print("one")
    with tee_stdout(log, banner="=== second ==="):
        print("two")
    assert (tmp_path / "run.log").read_text() == "=== second ===\ntwo\n"


def test_tee_stdout_is_undone_after_an_exception(tmp_path, capsys):
    with pytest.raises(RuntimeError), tee_stdout(str(tmp_path / "run.log")):
        print("before the boom")
        raise RuntimeError("boom")
    print("after")
    assert capsys.readouterr().out == "before the boom\nafter\n"
    assert (tmp_path / "run.log").read_text() == "before the boom\n"


# ── run ──────────────────────────────────────────────────────────────────────


# `run` is a process boundary: it exits with the status the function returned. Every test below launches
# it the way a shell would — `config=` / `home=` / overrides on the command line — and reads that status
# back off the SystemExit, which is what the shell would see.
def launch(monkeypatch, argv, function, **kwargs) -> object:
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run(function, **kwargs)
    return exit_info.value.code


def test_run_loads_the_config_and_opens_the_folder(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig, run_dir: str) -> int:
        print(f"training {cfg.model}")
        (tmp_path / "run" / "result.txt").write_text(cfg.model)  # results land in the run folder
        return 0

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    assert launch(monkeypatch, argv, train) == 0
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "metadata.json").is_file()
    assert (tmp_path / "run" / "result.txt").read_text() == "llama"
    assert "training llama" in (tmp_path / "run" / "run.log").read_text()  # the log is always run.log


def test_run_exits_zero_when_the_function_returns_nothing(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> None:
        return None

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    assert launch(monkeypatch, argv, train) is None


def test_run_takes_overrides_from_the_command_line(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        return 0 if cfg.model == "qwen" else 1

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}", "--", "model=qwen"]
    assert launch(monkeypatch, argv, train) == 0


def test_run_merges_several_config_files_left_to_right(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        return 0 if cfg.model == "qwen" else 1

    argv = [
        f"config={write(tmp_path / 'a.yaml', FULL)}",
        f"config={write(tmp_path / 'b.yaml', 'model: qwen')}",
        f"home={tmp_path / 'run'}",
    ]
    assert launch(monkeypatch, argv, train) == 0


def test_run_takes_the_config_home_and_overrides_from_the_caller(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        return 0 if (cfg.model, cfg.optim.lr) == ("qwen", 0.5) else 1

    code = launch(
        monkeypatch, [], train,
        config=write(tmp_path / "a.yaml", FULL), home=str(tmp_path / "run"),
        **{"model": "qwen", "optim.lr": 0.5},
    )
    assert code == 0
    assert (tmp_path / "run" / "config.yaml").is_file()


def test_the_command_line_wins_over_the_script(tmp_path, monkeypatch, write):
    seen = {}

    def train(cfg: fixtures.TrainConfig, run_dir: str) -> None:
        seen["model"], seen["dir"] = cfg.model, run_dir

    script = write(tmp_path / "script.yaml", FULL)
    cli = write(tmp_path / "cli.yaml", FULL.replace("llama", "qwen"))
    argv = [f"config={cli}", f"home={tmp_path / 'cli'}"]
    launch(monkeypatch, argv, train, config=script, home=str(tmp_path / "script"))
    assert seen == {"model": "qwen", "dir": str(tmp_path / "cli")}


def test_run_with_no_config_runs_on_the_schemas_own_defaults(tmp_path, monkeypatch):
    def train(cfg: fixtures.Settled, run_dir: str) -> int:
        return 0 if (cfg.model, cfg.steps) == ("llama", 20) else 1

    # Nothing named, so the class is the config — and `key=value` still wins over it. The snapshot is
    # written all the same: a run of the defaults is still answerable to a file that lists them.
    code = launch(monkeypatch, [f"home={tmp_path / 'run'}", "--", "steps=20"], train)
    assert code == 0
    assert OmegaConf.load(tmp_path / "run" / "config.yaml").model == "llama"


def test_run_reports_usage_when_given_no_config_and_the_schema_is_not_complete(monkeypatch):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    message = str(launch(monkeypatch, [], train))
    assert "missing required field(s): model" in message  # what the run is short of, named
    assert "usage: train.py config=<config.yaml>" in message  # and where such a field is filled in


def test_help_prints_the_grammar(monkeypatch, capsys):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    assert launch(monkeypatch, ["--help"], train) == 0
    assert "usage: train.py config=<config.yaml>" in capsys.readouterr().out


def test_only_config_and_home_come_before_the_separator(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    path = write(tmp_path / "a.yaml", FULL)
    for stray in (path, "model=qwen"):  # a bare file, or an override missing its `--`
        message = str(launch(monkeypatch, [f"config={path}", f"home={tmp_path}", stray], train))
        assert "is not `config=` or `home=`: overrides go after `--`" in message


def test_an_override_that_is_not_key_value_is_an_error(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path}", "--", "qwen"]
    assert "override 'qwen' is not key=value" in str(launch(monkeypatch, argv, train))


# ── Run: the class form ──────────────────────────────────────────────────────


def test_a_run_instance_loads_its_config_and_calls_main(tmp_path, monkeypatch, write):
    class Train(Run):
        config: fixtures.TrainConfig

        def main(self) -> int:
            print(f"training {self.config.model}")
            (Path(self.run_dir) / "result.txt").write_text(self.config.model)
            return 0

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        Train().run()
    assert exit_info.value.code == 0
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "result.txt").read_text() == "llama"
    assert "training llama" in (tmp_path / "run" / "run.log").read_text()


def test_a_run_must_annotate_its_config_class(monkeypatch):
    class Train(Run):
        def main(self) -> None:
            raise AssertionError("must not run")

    with pytest.raises(TypeError, match="Train's `config` must be annotated with its config class"):
        Train().run()


# ── run: where it writes is the launcher's, not the config's ─────────────────


def test_run_requires_a_folder_to_write_in(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}"]
    assert "nowhere to write: pass `home=PATH`" in str(launch(monkeypatch, argv, train))


def test_a_config_may_not_smuggle_the_run_dir_back_in(tmp_path, monkeypatch, write):
    # `run_dir` is not a field of any config class, so setting it is an unknown key like any other.
    def train(cfg: fixtures.TrainConfig) -> int:
        raise AssertionError("must not run")

    argv = [f"config={write(tmp_path / 'a.yaml', FULL + 'run_dir: runs/sneaky' + chr(10))}", f"home={tmp_path}"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(Exception, match="run_dir"):
        run(train)


@dataclass
class Homed(Config):
    home: str = "x"


def test_a_config_field_may_share_a_name_with_the_launchers_keys(tmp_path, monkeypatch):
    # Before `--`, `home=` names the folder; after it, `home=` is the config's own field.
    seen = {}

    def train(cfg: Homed, run_dir: str) -> None:
        seen["home"], seen["dir"] = cfg.home, run_dir

    launch(monkeypatch, [f"home={tmp_path / 'run'}", "--", "home=y"], train)
    assert seen == {"home": "y", "dir": str(tmp_path / "run")}



# ── run: the function IS its config ──────────────────────────────────────────


def test_run_takes_the_schema_off_the_function_s_annotation(tmp_path, monkeypatch, write):
    seen = {}

    def train(cfg: fixtures.TrainConfig) -> int:
        seen["type"] = type(cfg).__name__
        seen["lr"] = cfg.optim.lr
        return 0

    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    assert launch(monkeypatch, argv, train) == 0
    assert seen == {"type": "TrainConfig", "lr": 0.0002}  # loaded, typed, not a mapping of strings


SOLO_SCRIPT = '''
from dataclasses import dataclass, field
from omegaconf import MISSING
from slimconfig import Config, run

@dataclass
class Optim(Config):
    lr: float = MISSING

@dataclass
class SoloConfig(Config):
    model: str = MISSING
    optim: Optim = field(default_factory=Optim)

def main(cfg: SoloConfig, run_dir: str) -> int:
    print(f"{cfg.model} {cfg.optim.lr}")
    return 0

if __name__ == "__main__":
    run(main)
'''


def test_a_one_file_script_can_name_its_own_classes(tmp_path):
    # A config class defined in the launched script lives in `__main__`; the YAML has to call it
    # something. Launched for real, because that is the only way `__main__` is what it will be.
    (tmp_path / "solo.py").write_text(SOLO_SCRIPT, encoding="utf-8")
    (tmp_path / "solo.yaml").write_text(
        "_ > solo.SoloConfig:\nmodel: llama\noptim > solo.Optim:\n  lr: 0.5\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    argv = [sys.executable, "solo.py", "config=solo.yaml", "home=run"]
    done = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "llama 0.5" in done.stdout
    # ...and the snapshot names it the same way, so the run is repeatable from its own folder.
    assert (tmp_path / "run" / "config.yaml").read_text().startswith("_ > solo.SoloConfig:\n")


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


def test_run_rejects_a_plain_dataclass_as_the_config():
    # A dataclass is not a config class until it says so: `run` takes the one the YAML can name.
    def train(cfg: fixtures.PlainDataclass) -> int:
        return 0

    with pytest.raises(TypeError, match="must be annotated with its config class"):
        run(train)


def test_run_rejects_a_second_argument_that_is_not_the_run_folder():
    def train(cfg: fixtures.TrainConfig, extra: int) -> int:
        return 0

    with pytest.raises(TypeError, match="is the run folder and must be annotated"):
        run(train)

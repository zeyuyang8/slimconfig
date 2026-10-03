# Typed outputs: what a run returns is checked, written as output.json, and read back into its classes.

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fixtures
import pytest

from slimconfig import Output, Run, load_output, run, write_output

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


class Grade(enum.Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class Step(Output):
    name: str
    tokens: int


@dataclass
class Score(Output):
    accuracy: float
    grade: Grade
    steps: list[Step]
    per_class: dict[str, float]
    note: str | None
    extra: Any


def score() -> Score:
    return Score(0.5, Grade.PASS, [Step("a", 3)], {"cat": 1.0}, None, {"free": [1, "x"]})


# ── write and load ───────────────────────────────────────────────────────────


def test_an_output_round_trips_through_its_file(tmp_path):
    path = write_output(str(tmp_path), score())
    assert json.loads(Path(path).read_text())["grade"] == "pass"  # an Enum is written as its value
    assert load_output(str(tmp_path), Score) == score()


def test_an_int_loads_into_a_float_field(tmp_path):
    data = {"accuracy": 1, "grade": "fail", "steps": [], "per_class": {}, "note": "n", "extra": None}
    (tmp_path / "output.json").write_text(json.dumps(data))
    assert load_output(str(tmp_path), Score).accuracy == 1.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"accuracy": "high"}, r"Score.accuracy is not a float"),
        ({"steps": [{"name": "a", "tokens": "3"}]}, r"Score.steps\[0\].tokens is not a int"),
        ({"surprise": 1}, r"Score has key\(s\) Score does not declare: surprise"),
        ({"grade": "maybe"}, r"'maybe' is not a valid Grade"),
    ],
)
def test_a_loaded_value_is_checked_against_its_field(tmp_path, change, match):
    write_output(str(tmp_path), score())
    data = {**json.loads((tmp_path / "output.json").read_text()), **change}
    (tmp_path / "output.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_output(str(tmp_path), Score)


def test_a_missing_field_is_an_error(tmp_path):
    write_output(str(tmp_path), score())
    data = json.loads((tmp_path / "output.json").read_text())
    del data["note"]
    (tmp_path / "output.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Score is missing field\\(s\\): note"):
        load_output(str(tmp_path), Score)


def test_only_an_output_class_is_written_or_loaded(tmp_path):
    @dataclass
    class Plain:
        x: int

    with pytest.raises(TypeError, match="must be an output class"):
        write_output(str(tmp_path), Plain(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an output class"):
        load_output(str(tmp_path), Plain)


# ── a run that declares an output ────────────────────────────────────────────


def launch(monkeypatch, tmp_path, write, target) -> object:
    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    monkeypatch.setattr("sys.argv", ["train.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        target()
    return exit_info.value.code


def test_a_run_writes_the_output_it_declares(tmp_path, monkeypatch, write):
    class Train(Run):
        config: fixtures.TrainConfig
        output: Score

        def main(self) -> Score:
            return score()

    assert launch(monkeypatch, tmp_path, write, Train().run) == 0
    assert load_output(str(tmp_path / "run"), Score) == score()


def test_a_function_declares_its_output_by_its_return_type(tmp_path, monkeypatch, write):
    def train(cfg: fixtures.TrainConfig) -> Score:
        return score()

    assert launch(monkeypatch, tmp_path, write, lambda: run(train)) == 0
    assert load_output(str(tmp_path / "run"), Score) == score()


def test_a_run_must_return_the_output_it_declares(tmp_path, monkeypatch, write):
    class Train(Run):
        config: fixtures.TrainConfig
        output: Score

        def main(self) -> Score:
            return 0  # type: ignore[return-value]

    with pytest.raises(TypeError, match="declares output Score but returned 0"):
        launch(monkeypatch, tmp_path, write, Train().run)


def test_a_run_s_output_must_be_an_output_class(tmp_path, monkeypatch, write):
    class Train(Run):
        config: fixtures.TrainConfig
        output: int

        def main(self) -> int:
            return 0

    with pytest.raises(TypeError, match="Train's `output` must be an output class"):
        Train().run()


def test_a_run_without_an_output_still_exits_with_its_status(tmp_path, monkeypatch, write):
    class Train(Run):
        config: fixtures.TrainConfig

        def main(self) -> int:
            return 3

    assert launch(monkeypatch, tmp_path, write, Train().run) == 3
    assert not (tmp_path / "run" / "output.json").exists()

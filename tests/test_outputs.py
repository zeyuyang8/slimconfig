# Typed outputs: written as JSON and read back into their classes, checked field by field.

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from slimconfig import Output, load_output, write_output


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
    path = write_output(str(tmp_path / "out.json"), score())
    assert json.loads(Path(path).read_text())["grade"] == "pass"  # an Enum is written as its value
    assert load_output(str(tmp_path / "out.json"), Score) == score()


def test_an_int_loads_into_a_float_field(tmp_path):
    data = {"accuracy": 1, "grade": "fail", "steps": [], "per_class": {}, "note": "n", "extra": None}
    (tmp_path / "out.json").write_text(json.dumps(data))
    assert load_output(str(tmp_path / "out.json"), Score).accuracy == 1.0


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
    write_output(str(tmp_path / "out.json"), score())
    data = {**json.loads((tmp_path / "out.json").read_text()), **change}
    (tmp_path / "out.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_output(str(tmp_path / "out.json"), Score)


def test_a_missing_field_is_an_error(tmp_path):
    write_output(str(tmp_path / "out.json"), score())
    data = json.loads((tmp_path / "out.json").read_text())
    del data["note"]
    (tmp_path / "out.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Score is missing field\\(s\\): note"):
        load_output(str(tmp_path / "out.json"), Score)


def test_only_an_output_class_is_written_or_loaded(tmp_path):
    @dataclass
    class Plain:
        x: int

    with pytest.raises(TypeError, match="must be an output class"):
        write_output(str(tmp_path / "out.json"), Plain(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an output class"):
        load_output(str(tmp_path / "out.json"), Plain)

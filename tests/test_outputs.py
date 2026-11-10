# Typed outputs: written as JSON and read back into their classes, checked field by field.

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pytest

from slimconfig import (
    Output,
    build_output,
    load_output,
    output_schema,
    plain_output,
    read_record,
    record_line,
    record_text,
    write_output,
)


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


def test_a_record_names_its_class_at_its_top_level_alone(tmp_path):
    path = write_output(str(tmp_path / "out.json"), score())
    data = json.loads(Path(path).read_text())
    assert next(iter(data.items())) == ("_", "test_outputs.Score")  # first, as a config file's `_ >` line
    assert data["steps"] == [{"name": "a", "tokens": 3}]  # a nested value's class is its field's type
    assert Path(path).read_text() == record_text(score())


def test_a_record_without_its_class_is_refused(tmp_path):
    data = json.loads(record_text(score()))
    del data["_"]
    (tmp_path / "out.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r'out.json names no class under "_", but is read as a test_outputs.Score record'):
        load_output(str(tmp_path / "out.json"), Score)


def test_a_record_of_another_class_is_refused(tmp_path):
    path = write_output(str(tmp_path / "out.json"), Step("a", 3))
    with pytest.raises(ValueError, match=r"out.json names 'test_outputs.Step' under \"_\", but is read as a test_outputs.Score record"):
        load_output(path, Score)


def test_a_record_is_one_line_of_a_json_lines_file():
    text = "".join(record_line(s) for s in [Step("a", 1), Step("b", 2)])
    assert text == '{"_": "test_outputs.Step", "name": "a", "tokens": 1}\n{"_": "test_outputs.Step", "name": "b", "tokens": 2}\n'
    lines = text.splitlines()
    assert [read_record(line, Step, f"steps.jsonl line {n}") for n, line in enumerate(lines, 1)] == [Step("a", 1), Step("b", 2)]
    with pytest.raises(ValueError, match=r'steps.jsonl line 1 names no class under "_", but is read as a test_outputs.Step record'):
        read_record(json.dumps(plain_output(Step("a", 1))), Step, "steps.jsonl line 1")


def test_a_record_is_valid_json():
    with pytest.raises(ValueError, match="not JSON compliant"):
        record_line(Bound(float("nan"), None))


def test_an_int_loads_into_a_float_field(tmp_path):
    data = {"_": "test_outputs.Score", "accuracy": 1, "grade": "fail", "steps": [], "per_class": {}, "note": "n", "extra": None}
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


# ── any JSON value, and its schema ───────────────────────────────────────────


class Nutrient(enum.Enum):
    KCAL = "kcal"
    PROTEIN = "protein"


@dataclass
class Bound(Output):
    min: float | None
    max: float | None


@dataclass
class Meal(Output):
    name: Annotated[str, "The meal's name"]
    bounds: dict[Nutrient, Bound]
    tags: list[str]


def test_a_json_value_builds_into_its_class_and_back():
    value = {"name": "dinner", "bounds": {"kcal": {"min": None, "max": 900}}, "tags": ["vegan"]}
    meal = build_output(value, Meal)
    assert meal.bounds == {Nutrient.KCAL: Bound(None, 900.0)}  # an Enum key is read from its value
    assert plain_output(meal) == {**value, "bounds": {"kcal": {"min": None, "max": 900.0}}}


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"name": "d", "bounds": {"fat": {"min": 1, "max": None}}, "tags": []}, r"'fat' is not a valid Nutrient"),
        ({"name": "d", "bounds": {"kcal": {"min": 1}}, "tags": []}, r"Meal.bounds\[kcal\] is missing field\(s\): max"),
        ({"name": 1, "bounds": {}, "tags": []}, r"Meal.name is not a str"),
    ],
)
def test_a_json_value_is_checked_against_its_class(value, match):
    with pytest.raises(ValueError, match=match):
        build_output(value, Meal)


def test_the_schema_requires_every_field_and_carries_descriptions():
    assert output_schema(Meal) == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The meal's name"},
            "bounds": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                        "max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    },
                    "required": ["min", "max"],
                    "additionalProperties": False,
                },
                "propertyNames": {"enum": ["kcal", "protein"]},
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "bounds", "tags"],
        "additionalProperties": False,
    }


def test_a_mapping_is_keyed_by_str_or_an_enum():
    @dataclass
    class Counts(Output):
        by_size: dict[int, int]

    with pytest.raises(TypeError, match="keyed by str or an Enum"):
        output_schema(Counts)
    with pytest.raises(TypeError, match="keyed by str or an Enum"):
        build_output({"by_size": {}}, Counts)


# ── unions of output classes ─────────────────────────────────────────────────


@dataclass
class Passed(Output):
    seconds: float


@dataclass
class Failed(Output):
    error: str


@dataclass
class Result(Output):
    outcome: Passed | Failed
    retried: list[Passed | Failed]
    last: Passed | Failed | None


def test_a_union_of_output_classes_is_written_tagged_by_its_class_and_read_back(tmp_path):
    result = Result(Failed("boom"), [Passed(1.0), Failed("x")], None)
    path = write_output(str(tmp_path / "out.json"), result)
    data = json.loads(Path(path).read_text())
    assert data["outcome"] == {"Failed": {"error": "boom"}}
    assert data["retried"] == [{"Passed": {"seconds": 1.0}}, {"Failed": {"error": "x"}}]
    assert data["last"] is None
    assert load_output(path, Result) == result
    assert type(load_output(path, Result).outcome) is Failed


@pytest.mark.parametrize(
    "outcome",
    [{"error": "boom"}, {"Skipped": {}}, {"Passed": {"seconds": 1.0}, "Failed": {"error": "x"}}, None],
)
def test_a_union_value_names_exactly_one_of_its_classes(outcome):
    with pytest.raises(ValueError, match=r"Result.outcome is not one of Passed, Failed"):
        build_output({"outcome": outcome, "retried": [], "last": None}, Result)


def test_a_union_value_is_checked_as_its_class():
    with pytest.raises(ValueError, match=r"Result.outcome.Failed is missing field\(s\): error"):
        build_output({"outcome": {"Failed": {}}, "retried": [], "last": None}, Result)


def test_a_value_of_no_class_of_its_union_is_not_written():
    with pytest.raises(TypeError, match="Result.outcome holds a Step, none of the union"):
        plain_output(Result(Step("a", 1), [], None))  # type: ignore[arg-type]


def test_a_union_of_output_classes_is_one_of_their_tagged_schemas():
    schema = output_schema(Result)["properties"]
    assert schema["outcome"] == {
        "oneOf": [
            {"type": "object", "properties": {"Passed": output_schema(Passed)}, "required": ["Passed"], "additionalProperties": False},
            {"type": "object", "properties": {"Failed": output_schema(Failed)}, "required": ["Failed"], "additionalProperties": False},
        ]
    }
    assert schema["last"]["anyOf"][1] == {"type": "null"}


def test_a_union_of_two_classes_of_one_name_or_of_scalars_is_refused():
    Other = type("Passed", (Output,), {"__annotations__": {"n": int}})
    Two = dataclass(type("Two", (Output,), {"__annotations__": {"x": Passed | dataclass(Other)}}))
    with pytest.raises(TypeError, match="two classes of one name"):
        output_schema(Two)
    Scalars = dataclass(type("Scalars", (Output,), {"__annotations__": {"x": int | str}}))
    with pytest.raises(TypeError, match="union of output classes, not the union"):
        output_schema(Scalars)

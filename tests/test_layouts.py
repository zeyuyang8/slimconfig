# Layouts: a run's folder holds its own files plus what its layout declares, and nothing else.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import fixtures
import pytest

from slimconfig import Folder, Layout, Metadata, Output, Run, Workspace, load_layout, load_output, record_line, write_output

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


@dataclass
class Messages(Output):
    turns: list[str]


@dataclass
class ChatLayout(Layout):
    messages: Messages
    workspace: Folder


def launch(monkeypatch, tmp_path, write, target) -> object:
    argv = [f"config={write(tmp_path / 'a.yaml', FULL)}", f"home={tmp_path / 'run'}"]
    monkeypatch.setattr("sys.argv", ["chat.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        target()
    return exit_info.value.code


class Chat(Run):
    config: fixtures.TrainConfig
    layout: ChatLayout

    def main(self) -> None:
        assert self.layout.workspace.is_dir()  # created before main
        (self.layout.workspace / "notes.txt").write_text("hi")
        self.layout.messages = Messages(["hello", "hi"])


def test_a_run_fills_its_layout_and_reads_back(tmp_path, monkeypatch, write):
    assert launch(monkeypatch, tmp_path, write, Chat().run) is None
    run_dir = tmp_path / "run"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config.yaml", "messages.json", "metadata.json", "run.log", "workspace",
    ]
    run = load_layout(str(run_dir), ChatLayout)
    assert run.messages == Messages(["hello", "hi"])
    assert (run.workspace / "notes.txt").read_text() == "hi"


def test_metadata_records_the_layout(tmp_path, monkeypatch, write):
    launch(monkeypatch, tmp_path, write, Chat().run)
    meta = load_output(str(tmp_path / "run" / "metadata.json"), Metadata)
    assert meta.layout == {"messages.json": "Messages", "workspace": "folder"}


def test_every_output_of_a_run_is_a_record_naming_its_class(tmp_path, monkeypatch, write):
    launch(monkeypatch, tmp_path, write, Chat().run)
    assert json.loads((tmp_path / "run" / "messages.json").read_text())["_"] == "test_layouts.Messages"


def test_anything_the_layout_does_not_declare_is_an_error(tmp_path, monkeypatch, write):
    class Messy(Chat):
        def main(self) -> None:
            super().main()
            (Path(self.run_dir) / "scratch.txt").write_text("x")

    with pytest.raises(RuntimeError, match="holds scratch.txt, which ChatLayout does not declare"):
        launch(monkeypatch, tmp_path, write, Messy().run)


def test_every_output_must_be_set(tmp_path, monkeypatch, write):
    class Forgetful(Chat):
        def main(self) -> None:
            pass

    with pytest.raises(TypeError, match="layout.messages must be set to a Messages"):
        launch(monkeypatch, tmp_path, write, Forgetful().run)


@dataclass
class Bad(Layout):
    count: int


def test_a_layout_field_is_an_output_or_a_folder(monkeypatch):
    class Run2(Run):
        config: fixtures.TrainConfig
        layout: Bad

        def main(self) -> None:
            raise AssertionError("must not run")

    with pytest.raises(TypeError, match=r"Bad.count is typed .*a layout field is an output class, Folder or Workspace"):
        Run2().run()


def test_load_layout_requires_its_folders(tmp_path):
    write_output(str(tmp_path / "messages.json"), Messages([]))
    with pytest.raises(FileNotFoundError, match="has no folder workspace/"):
        load_layout(str(tmp_path), ChatLayout)


@dataclass
class ToolLayout(Layout):
    messages: Messages
    harness: Folder
    workspace: Workspace


class Tooled(Run):
    config: fixtures.TrainConfig
    layout: ToolLayout
    leaves: ClassVar[dict[str, str]] = {}  # what main writes, by its path in the run folder

    def main(self) -> None:
        for path, text in self.leaves.items():
            (Path(self.run_dir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(self.run_dir) / path).write_text(text)
        self.layout.messages = Messages(["hi"])


def tooled(monkeypatch, tmp_path, write, leaves: dict[str, str]) -> object:
    run = Tooled()
    run.leaves = leaves
    return launch(monkeypatch, tmp_path, write, run.run)


@pytest.mark.parametrize("name", ["x.json", "x.JSON", "x.Json", "x.JSONL", "x.ndjson", "x.json5", "data", "notes.txt"])
def test_a_folder_holding_json_that_names_no_class_fails_the_run_whatever_its_name(tmp_path, monkeypatch, write, name):
    with pytest.raises(RuntimeError, match=rf'harness/{name} is JSON that names no class under "_", so it is no record'):
        tooled(monkeypatch, tmp_path, write, {f"harness/{name}": '{"words": 1}'})


def test_every_line_of_json_lines_that_names_no_class_fails_the_run(tmp_path, monkeypatch, write):
    text = record_line(Messages(["a"])) + '{"type": "init"}\n'
    with pytest.raises(RuntimeError, match=r'harness/turn1/transcript.jsonl line 2 is JSON that names no class under "_"'):
        tooled(monkeypatch, tmp_path, write, {"harness/turn1/transcript.jsonl": text})


def test_a_folder_may_hold_records_each_read_back_as_the_class_it_names(tmp_path, monkeypatch, write):
    lines = record_line(Messages(["a"])) + record_line(Messages(["b"]))
    assert tooled(monkeypatch, tmp_path, write, {"harness/t.jsonl": lines, "harness/log": "[INFO] {not json}\n"}) is None


def test_a_record_naming_its_class_over_fields_that_are_not_its_fails_the_run(tmp_path, monkeypatch, write):
    with pytest.raises(RuntimeError, match=r"/run/harness/x.json: Messages has key\(s\) Messages does not declare: words"):
        tooled(monkeypatch, tmp_path, write, {"harness/x.json": '{"_": "test_layouts.Messages", "words": 1}'})
    with pytest.raises(RuntimeError, match=r"""harness/y.json is JSON that names 'os.path' under "_", no output class"""):
        tooled(monkeypatch, tmp_path, write, {"harness/y.json": '{"_": "os.path"}'})


def test_a_file_the_run_reads_as_its_class_must_read_back_whole(tmp_path, monkeypatch, write):
    class Meddling(Tooled):
        def main(self) -> None:
            super().main()
            (Path(self.run_dir) / "metadata.json").write_text('{"_": "slimconfig.runs.Metadata", "words": 1}')

    with pytest.raises(RuntimeError, match=r"/run holds JSON that is no record of its class: .*/run/metadata.json: Metadata has key\(s\) Metadata does not declare: words"):
        launch(monkeypatch, tmp_path, write, Meddling().run)


def test_a_symlink_in_the_run_folder_fails_the_run(tmp_path, monkeypatch, write):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.json").write_text('{"words": 1}')

    class Linking(Tooled):
        def main(self) -> None:
            super().main()
            (Path(self.run_dir) / "harness" / "link").symlink_to(outside)

    with pytest.raises(RuntimeError, match=r"harness/link is a symlink or a special file, which no run folder holds"):
        launch(monkeypatch, tmp_path, write, Linking().run)


def test_a_workspace_is_the_works_never_read(tmp_path, monkeypatch, write):
    assert tooled(monkeypatch, tmp_path, write, {"workspace/package.json": '{"name": "x"}'}) is None
    meta = load_output(str(tmp_path / "run" / "metadata.json"), Metadata)
    assert meta.layout == {"messages.json": "Messages", "harness": "folder", "workspace": "workspace"}

# Layouts: a run's folder holds its own files plus what its layout declares, and nothing else.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fixtures
import pytest

from slimconfig import Folder, Layout, Output, Run, load_layout

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
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert meta["layout"] == {"messages.json": "Messages", "workspace": "folder"}


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

    with pytest.raises(TypeError, match=r"Bad.count is typed .*a layout field is an output class or Folder"):
        Run2().run()


def test_load_layout_requires_its_folders(tmp_path):
    (tmp_path / "messages.json").write_text(json.dumps({"turns": []}))
    with pytest.raises(FileNotFoundError, match="has no folder workspace/"):
        load_layout(str(tmp_path), ChatLayout)

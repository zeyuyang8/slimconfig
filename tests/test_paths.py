# Project-root discovery and the relative-path rule.

from __future__ import annotations

from pathlib import Path

from slimconfig import project_root, resolve_path


def test_env_override_wins(tmp_path, monkeypatch):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "repo")
    monkeypatch.setenv("SLIMCONFIG_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_git_marker_found_from_a_subdirectory(tmp_path, monkeypatch):
    monkeypatch.delenv("SLIMCONFIG_PROJECT_ROOT", raising=False)
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert project_root() == tmp_path.resolve()


def test_git_beats_a_nearer_pyproject(tmp_path, monkeypatch):
    monkeypatch.delenv("SLIMCONFIG_PROJECT_ROOT", raising=False)
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    member = tmp_path / "src" / "member"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(member)
    assert project_root() == tmp_path.resolve()


def test_outermost_pyproject_wins_without_a_git_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("SLIMCONFIG_PROJECT_ROOT", raising=False)
    root = tmp_path / "tree"
    member = root / "src" / "member"
    member.mkdir(parents=True)
    for d in (root, member):
        (d / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(member)
    assert project_root() == root.resolve()


def test_resolve_path_joins_relative_and_passes_absolute_through(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMCONFIG_PROJECT_ROOT", str(tmp_path))
    assert resolve_path("data/corpus.parquet") == tmp_path.resolve() / "data/corpus.parquet"
    assert resolve_path(Path("/etc/hosts")) == Path("/etc/hosts")


def test_resolve_path_is_independent_of_the_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("SLIMCONFIG_PROJECT_ROOT", raising=False)
    (tmp_path / ".git").mkdir()
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    assert resolve_path("cfg.yaml") == tmp_path.resolve() / "cfg.yaml"

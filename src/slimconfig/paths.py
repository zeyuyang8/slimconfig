# slimconfig.paths — anchor the relative paths a config names to the project root.
#
# A config says `corpus: data/corpus.parquet`, and that name has to mean the same file wherever the
# process was launched from. resolve_path turns such a name into an absolute path under the project
# root, discovered by walking up from the CWD:
#   * the nearest ancestor holding a `.git` (a checkout root), else
#   * the OUTERMOST ancestor holding a `pyproject.toml` (an exported/installed tree — the outermost
#     one, because a workspace member has a pyproject.toml of its own and is not the root), else
#   * the CWD.
# Set SLIMCONFIG_PROJECT_ROOT to say it outright instead.
#
# Discovery starts at the CWD, never at slimconfig's own __file__: slimconfig lives in site-packages,
# which is not the caller's project.

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["project_root", "resolve_path"]


def project_root() -> Path:
    override = os.environ.get("SLIMCONFIG_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    start = Path.cwd().resolve()
    outermost_pyproject: Path | None = None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():  # dir in a clone, file in a worktree/submodule
            return candidate
        if (candidate / "pyproject.toml").is_file():
            outermost_pyproject = candidate  # keep walking up — an outer one would win
    return outermost_pyproject or start


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p

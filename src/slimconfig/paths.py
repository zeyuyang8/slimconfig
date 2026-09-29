# slimconfig.paths — anchor the relative paths a config names to the project root.
#
# The root is SLIMCONFIG_PROJECT_ROOT if set, else found by walking up from the CWD: the nearest `.git`,
# else the outermost `pyproject.toml` (a workspace member has its own), else the CWD.
# Discovery starts at the CWD, not __file__: slimconfig lives in site-packages, not the caller's project.

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

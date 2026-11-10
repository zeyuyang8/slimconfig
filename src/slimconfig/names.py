# slimconfig.names — the dotted path a file names a class by: a YAML on its `_ > pkg.mod.Class:` line
# (slimconfig.schemas), a JSON record under its `"_": "pkg.mod.Class"` key (slimconfig.outputs).

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["main_names", "schema_name"]


# The names the launched script goes by. A class defined there lives in `__main__`, and re-importing the
# script under its real name would run it again and yield a different class, so the two are treated as one.
def main_names() -> tuple[str, ...]:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)  # set by `python -m pkg.mod`
    file = getattr(main, "__file__", None)  # set by `python path/to/train.py`
    return tuple(n for n in (getattr(spec, "name", None), Path(file).stem if file else None) if n)


# The dotted path a file names `cls` by — the inverse of `Schema.resolve`.
def schema_name(cls: type) -> str:
    module = cls.__module__
    if module == "__main__":
        module = next(iter(main_names()), module)
    return f"{module}.{cls.__qualname__}"

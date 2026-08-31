# Writing a config file is the one thing every test here does, and every config file names the class it
# fills — so `write` supplies that line unless the body already has one (or asks for none).

from __future__ import annotations

import pytest


@pytest.fixture
def write():
    def _write(path, text: str, schema: str | None = "fixtures.TrainConfig") -> str:
        # Only a TOP-LEVEL line counts: a body may restate `_schema:` inside a nested block and still
        # need the file's own.
        if schema is not None and not any(line.startswith("_schema:") for line in text.splitlines()):
            text = f"_schema: {schema}\n{text.lstrip(chr(10))}"
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write

# Writing a config file is the one thing every test here does, and every config file names the class it
# fills — so `write` supplies that line unless the body already has one (or asks for none).

from __future__ import annotations

import pytest


@pytest.fixture
def write():
    def _write(path, text: str, schema: str | None = "fixtures.TrainConfig") -> str:
        if schema is not None and not any(line.startswith("_ >") for line in text.splitlines()):
            text = f"_ > {schema}:\n{text.lstrip(chr(10))}"
        path.write_text(text, encoding="utf-8")
        return str(path)

    return _write

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TUBEMAP_DATA", str(tmp_path / "data"))
    yield tmp_path / "data"


FIXTURES = Path(__file__).resolve().parent / "fixtures"

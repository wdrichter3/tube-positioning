from pathlib import Path
import pytest
from tube_positioning.config import load_config


def test_default_config_has_no_motion_authority():
    cfg = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    assert cfg["application"]["motion_authority"] is False


def test_motion_authority_true_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("application:\n  motion_authority: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)

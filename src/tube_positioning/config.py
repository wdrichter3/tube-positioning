from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("application", {}).get("motion_authority", True):
        raise ValueError("V0.1 safety boundary violated: motion_authority must be false")
    return cfg

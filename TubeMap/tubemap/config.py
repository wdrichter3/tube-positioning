"""Data folder layout, JSON persistence and default settings."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from .vision.mapping import DEFAULT_TOLERANCES

DEFAULT_SETTINGS = {
    "port": 8088,
    "geometry": {"mode": "known", "pattern": "TRIANGULAR", "pitch_value": 1.0, "pitch_unit": "in",
                 "id_value": 0.62, "id_unit": "in"},
    "tolerances": dict(DEFAULT_TOLERANCES),
    "active_profile": "builtin",
    "live_max_width": 1024,
    "live_fps": 6,
    "job": {"customer": "", "site": "", "exchanger": "", "notes": ""},
    "appearance": {},
}

_lock = threading.RLock()


def data_dir() -> Path:
    p = Path(os.environ.get("TUBEMAP_DATA", Path.home() / ".tubemap")).expanduser()
    for sub in ("captures", "profiles", "trash", "exports"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(40):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:          # Windows: target briefly open elsewhere
                    if attempt == 39:
                        raise
                    time.sleep(0.025)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def read_json(path: Path, default=None):
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return copy.deepcopy(default)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> dict:
    return _merge(DEFAULT_SETTINGS, read_json(data_dir() / "settings.json", {}))


def save_settings(s: dict) -> dict:
    merged = _merge(DEFAULT_SETTINGS, s)
    write_json(data_dir() / "settings.json", merged)
    return merged


def default_devices() -> list:
    devs = []
    try:
        import depthai  # noqa: F401
        has_dai = True
    except Exception:
        has_dai = False
    devs.append({"id": "oak-1", "name": "OAK-D Pro", "driver": "oak", "enabled": has_dai,
                 "role": "overview", "settings": {}})
    try:
        from .devices.orbbec_capture import sdk_generation
        has_astra = sdk_generation() == 1
    except Exception:
        has_astra = False
    if has_astra:
        devs.append({"id": "astra-1", "name": "Astra+", "driver": "orbbec", "enabled": True,
                     "role": "depth", "settings": {}})
    devs.append({"id": "sim-1", "name": "Simulator", "driver": "simulator", "enabled": not has_dai,
                 "role": "aux", "settings": {}})
    return devs


def load_devices() -> list:
    p = data_dir() / "devices.json"
    devs = read_json(p, None)
    if devs is None:
        devs = default_devices()
        write_json(p, devs)
    return devs


def save_devices(devs: list) -> None:
    write_json(data_dir() / "devices.json", devs)

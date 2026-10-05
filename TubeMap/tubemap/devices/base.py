"""Device driver interface.

A driver turns one physical (or simulated) camera into FrameSets. Drivers
describe their own settings so the dashboard can render forms for them, and
report whether their SDK is installed so missing packages show up as a clear
message instead of a crash.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameSet:
    color: np.ndarray                         # BGR uint8
    depth: np.ndarray | None = None           # uint16 millimetres, aligned to colour FOV
    intrinsics: dict | None = None            # colour camera model at colour resolution
    timestamp: float = field(default_factory=time.time)
    seq: int = 0
    meta: dict = field(default_factory=dict)


def field_spec(key, label, ftype, default, help="", options=None, minimum=None, maximum=None,
               step=None, live=False, advanced=False):
    d = {"key": key, "label": label, "type": ftype, "default": default, "help": help,
         "live": bool(live), "advanced": bool(advanced)}
    if options is not None:
        d["options"] = options
    if minimum is not None:
        d["min"] = minimum
    if maximum is not None:
        d["max"] = maximum
    if step is not None:
        d["step"] = step
    return d


COMMON_FIELDS = [
    field_spec("depth_scale_factor", "Depth scale correction", "float", 1.0,
               "Multiplier applied to depth when mapping. Set it with 'Calibrate depth scale' "
               "on a sheet whose pitch you have measured.", minimum=0.5, maximum=1.5, step=0.0001,
               advanced=True),
    field_spec("depth_uncertainty_pct", "Depth scale uncertainty (%)", "float", 2.0,
               "Used to set how strict the pitch cross-check is.", minimum=0.2, maximum=20, step=0.1,
               advanced=True),
]


class DeviceDriver:
    driver_id = "base"
    label = "Base"
    description = ""
    has_depth = False

    def __init__(self, settings: dict):
        self.settings = self.normalize(settings)
        self.info: dict = {}

    # ---- description -------------------------------------------------------
    @classmethod
    def settings_schema(cls) -> list:
        return []

    @classmethod
    def full_schema(cls) -> list:
        return cls.settings_schema() + COMMON_FIELDS

    @classmethod
    def normalize(cls, settings: dict | None) -> dict:
        out = {}
        settings = settings or {}
        for f in cls.full_schema():
            v = settings.get(f["key"], f["default"])
            try:
                if f["type"] == "int":
                    v = int(float(v))
                elif f["type"] == "float":
                    v = float(v)
                elif f["type"] == "bool":
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                else:
                    v = "" if v is None else str(v)
            except (TypeError, ValueError):
                v = f["default"]
            if f["type"] in ("int", "float"):
                if "min" in f:
                    v = max(f["min"], v)
                if "max" in f:
                    v = min(f["max"], v)
            if f["type"] == "select" and f.get("options"):
                vals = [o["value"] if isinstance(o, dict) else o for o in f["options"]]
                if v not in vals:
                    v = f["default"]
            out[f["key"]] = v
        return out

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""

    @classmethod
    def discover(cls) -> list:
        """Return [{'label': str, 'settings': {...}}] for devices that could be added."""
        return []

    # ---- lifecycle ---------------------------------------------------------
    def open(self):
        raise NotImplementedError

    def close(self):
        pass

    def read(self, timeout: float = 2.0) -> FrameSet | None:
        """Latest live frame (may be lower quality than capture)."""
        raise NotImplementedError

    def capture(self, timeout: float = 5.0) -> FrameSet:
        """Best-quality matched colour/depth pair for saving. Default: next live frame."""
        fs = self.read(timeout)
        if fs is None:
            raise TimeoutError("no frame received")
        return fs

    def apply_live(self, changed: dict) -> bool:
        """Apply settings without a restart. Return False if a restart is needed."""
        return False

    def live_keys(self) -> set:
        return {f["key"] for f in self.full_schema() if f.get("live")} | {f["key"] for f in COMMON_FIELDS}

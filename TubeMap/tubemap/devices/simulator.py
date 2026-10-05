"""Simulated depth camera looking at a synthetic tubesheet (no hardware needed)."""
from __future__ import annotations

import time

import numpy as np

from ..vision import synth
from .base import DeviceDriver, FrameSet, field_spec


class SimulatorDriver(DeviceDriver):
    driver_id = "simulator"
    label = "Simulator"
    description = "Synthetic tubesheet with known ground truth. Use it to try the workflow without a camera."
    has_depth = True

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("pattern", "Pattern", "select", "TRIANGULAR", options=["TRIANGULAR", "SQUARE"]),
            field_spec("pitch_in", "Pitch (in)", "float", 1.0, minimum=0.5, maximum=3.0, step=0.0001),
            field_spec("tube_id_in", "Tube ID (in)", "float", 0.62, minimum=0.2, maximum=2.5, step=0.001),
            field_spec("sheet_diameter_mm", "Sheet diameter (mm)", "float", 700, minimum=200, maximum=2000),
            field_spec("distance_mm", "Camera distance (mm)", "float", 1100, minimum=300, maximum=4000),
            field_spec("tilt_deg", "Camera tilt (deg)", "float", 12, minimum=0, maximum=45),
            field_spec("lattice_rot_deg", "Pattern rotation (deg)", "float", 8, minimum=-90, maximum=90),
            field_spec("plugged_fraction", "Plugged fraction", "float", 0.03, minimum=0, maximum=0.5, step=0.01),
            field_spec("k1", "Lens distortion k1", "float", -0.08, minimum=-0.5, maximum=0.5, step=0.01, advanced=True),
            field_spec("noise", "Image noise", "float", 4.0, minimum=0, maximum=30, advanced=True),
            field_spec("hfov_deg", "Horizontal field of view (deg)", "float", 66.0, minimum=20, maximum=120, advanced=True),
            field_spec("yaw_deg", "Camera yaw (deg)", "float", -5.0, minimum=-45, maximum=45, advanced=True),
            field_spec("roll_deg", "Camera roll (deg)", "float", 0.0, minimum=-180, maximum=180, advanced=True),
            field_spec("mirror", "Mirrored image", "bool", False, advanced=True),
            field_spec("depth_scale", "Depth resolution (fraction of colour)", "float", 0.5, minimum=0.1, maximum=1.0, step=0.01, advanced=True),
            field_spec("depth_smooth_px", "Depth edge smoothing (px)", "int", 0, minimum=0, maximum=15, advanced=True),
            field_spec("width", "Width (px)", "int", 1248, minimum=320, maximum=4056, advanced=True),
            field_spec("height", "Height (px)", "int", 936, minimum=240, maximum=3040, advanced=True),
            field_spec("fps", "Frame rate", "float", 5.0, minimum=0.5, maximum=30, live=True),
            field_spec("seed", "Random seed", "int", 7, advanced=True),
        ]

    def open(self):
        s = self.settings
        keys = ["pattern", "pitch_in", "tube_id_in", "sheet_diameter_mm", "distance_mm", "tilt_deg",
                "lattice_rot_deg", "plugged_fraction", "k1", "noise", "width", "height", "seed",
                "hfov_deg", "yaw_deg", "roll_deg", "mirror", "depth_scale", "depth_smooth_px"]
        self._scene = synth.render(**{k: s[k] for k in keys})
        self._rng = np.random.default_rng(int(s["seed"]) + 1)
        self._seq = 0
        self._last = 0.0
        self.info = {"model": "Simulator", "serial": f"SIM-{int(s['seed'])}",
                     "color_size": [int(s["width"]), int(s["height"])],
                     "depth_size": list(self._scene["depth"].shape[::-1])}

    def read(self, timeout=2.0):
        period = 1.0 / max(0.5, float(self.settings["fps"]))
        wait = self._last + period - time.time()
        if wait > 0:
            time.sleep(min(wait, timeout))
        self._last = time.time()
        self._seq += 1
        col = self._scene["color"].astype(np.int16)
        col = np.clip(col + self._rng.normal(0, 1.5, col.shape), 0, 255).astype(np.uint8)
        return FrameSet(color=col, depth=self._scene["depth"].copy(),
                        intrinsics=dict(self._scene["intrinsics"]), seq=self._seq,
                        meta={"simulated": True, "truth_pitch_mm": self._scene["pitch_mm"],
                              "truth_tube_id_mm": self._scene["tube_id_mm"]})

    def apply_live(self, changed):
        return set(changed) <= self.live_keys()

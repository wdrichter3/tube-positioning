"""Orbbec depth cameras connected directly to this computer by USB (pyorbbecsdk v1 or v2)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from .base import DeviceDriver, FrameSet, field_spec

_PROBE = r"""
import json
from tubemap.devices.orbbec_capture import list_devices
print("TUBEMAP_PROBE" + json.dumps(list_devices()))
"""


def probe_devices(timeout=20.0):
    """List Orbbec devices in a child process: the v1 SDK can crash the whole process
    (segmentation fault) when USB access fails, and that must not take the dashboard down."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in sys.path if p])
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Orbbec SDK did not answer within %.0f s while looking for cameras" % timeout)
    for line in r.stdout.splitlines():
        if line.startswith("TUBEMAP_PROBE"):
            return json.loads(line[len("TUBEMAP_PROBE"):])
    if r.returncode < 0 or r.returncode == 139:
        raise RuntimeError("Orbbec SDK crashed while looking for cameras. Check USB access: run "
                           "scripts/install_orbbec_astra.sh (installs udev rules), then unplug and replug the camera.")
    tail = (r.stderr or "").strip().splitlines()[-1:] or ["no output"]
    raise RuntimeError("Orbbec SDK error: " + tail[0])

COLOR_SIZES = [
    {"value": "1920x1080", "label": "1920 x 1080 (Astra+ full colour)"},
    {"value": "1280x720", "label": "1280 x 720"},
    {"value": "1280x800", "label": "1280 x 800 (Gemini 2 / Persee 2)"},
    {"value": "640x480", "label": "640 x 480"},
]
DEPTH_SIZES = [
    {"value": "auto", "label": "Camera default"},
    {"value": "640x480", "label": "640 x 480 (Astra+ recommended)"},
    {"value": "1280x1024", "label": "1280 x 1024 (Astra+ max; lower frame rate)"},
    {"value": "1280x800", "label": "1280 x 800"},
    {"value": "640x400", "label": "640 x 400"},
]


class OrbbecDriver(DeviceDriver):
    driver_id = "orbbec"
    label = "Orbbec (USB): Astra+, Gemini, Femto"
    description = ("Orbbec cameras plugged into this computer. Astra+ needs the v1 SDK "
                   "(scripts/install_orbbec_astra.sh); Gemini/Femto use pyorbbecsdk2. "
                   "Give it the role 'Depth checks' to use its depth for the pitch check and plug evidence "
                   "on the other cameras' maps. For the Persee 2, use 'Network camera agent'.")
    has_depth = True

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("serial", "Serial number", "text", "", "Blank = first Orbbec found."),
            field_spec("color_size", "Colour resolution", "select", "1920x1080", options=COLOR_SIZES),
            field_spec("fps", "Frame rate", "int", 15, minimum=5, maximum=30),
            field_spec("depth_enabled", "Depth", "bool", True),
            field_spec("depth_size", "Depth resolution", "select", "640x480", options=DEPTH_SIZES),
            field_spec("align", "Depth-to-colour alignment", "select", "auto",
                       "Auto tries the camera's hardware alignment first, then software. Depth that "
                       "can't be aligned is saved but not used for mapping.",
                       options=[{"value": "auto", "label": "Auto (hardware, then software)"},
                                {"value": "hw", "label": "Hardware only"}, {"value": "sw", "label": "Software only"},
                                {"value": "off", "label": "Off (not used for mapping)"}]),
            field_spec("unmirror", "Un-mirror images", "bool", True,
                       "Astra cameras can deliver mirrored images; this turns mirroring off in the camera."),
            field_spec("laser", "Projector on", "bool", True, advanced=True),
        ]

    @classmethod
    def available(cls):
        try:
            from .orbbec_capture import sdk_generation
            g = sdk_generation()
            if g is None:
                raise ImportError("pyorbbecsdk")
            note = "SDK v1 (Astra+ supported)" if g == 1 else "SDK v2 (Astra+ NOT supported; Gemini/Femto yes)"
            return True, f"pyorbbecsdk installed: {note}"
        except Exception as exc:  # pragma: no cover
            return False, (f"pyorbbecsdk not installed ({exc.__class__.__name__}). Astra+: run "
                           "scripts/install_orbbec_astra.sh. Gemini/Femto: pip install pyorbbecsdk2")

    @classmethod
    def discover(cls):
        return [{"label": f"{d['name']} {d['serial']}", "settings": {"serial": d["serial"]}} for d in probe_devices()]

    def open(self):
        from .orbbec_capture import OrbbecCapture
        s = self.settings
        found = probe_devices()
        if not found:
            raise RuntimeError("no Orbbec camera found (check the USB 3 cable)")
        if s["serial"] and s["serial"] not in [d["serial"] for d in found]:
            raise RuntimeError(f"Orbbec camera {s['serial']} not found; connected: "
                               + ", ".join(d["serial"] or "?" for d in found))
        cw, ch = (int(x) for x in s["color_size"].split("x"))
        dw, dh = (0, 0) if s["depth_size"] == "auto" else (int(x) for x in s["depth_size"].split("x"))
        self.cap = OrbbecCapture(s["serial"], cw, ch, s["fps"], s["depth_enabled"], s["align"],
                                 s["unmirror"], dw, dh, s["laser"])
        self.cap.open()
        self.info = dict(self.cap.info)
        self.seq = 0

    def close(self):
        self.cap.close()

    def _fs(self, color, depth, intr):
        self.seq += 1
        return FrameSet(color=color, depth=depth, intrinsics=intr, seq=self.seq,
                        meta={"depth_aligned": bool(self.cap.depth_aligned),
                              "depth_alignment": self.cap.align_used})

    def read(self, timeout=2.0):
        return self._fs(*self.cap.grab(int(timeout * 1000)))

    def capture(self, timeout=5.0):
        return self._fs(*self.cap.grab(int(timeout * 1000), need_depth=self.settings["depth_enabled"]))

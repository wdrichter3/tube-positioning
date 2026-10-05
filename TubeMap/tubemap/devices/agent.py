"""Network camera agent client (e.g. Orbbec Persee 2 running agent/tubemap_agent.py)."""
from __future__ import annotations

import io
import json
import urllib.request

import cv2
import numpy as np

from .base import DeviceDriver, FrameSet, field_spec


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - LAN device URL from settings
        return r.read()


class AgentDriver(DeviceDriver):
    driver_id = "agent"
    label = "Network camera agent"
    description = ("A camera computer on the network running tubemap_agent.py - use this for the "
                   "Orbbec Persee 2. Live view is JPEG; captures are lossless colour + depth.")
    has_depth = True

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("host", "Agent address", "text", "192.168.1.50", "IP address or hostname of the Persee 2."),
            field_spec("port", "Agent port", "int", 8095, minimum=1, maximum=65535),
            field_spec("live_max_width", "Live preview width", "int", 960, minimum=320, maximum=1920, live=True),
            field_spec("live_fps", "Live preview rate", "float", 5.0, minimum=0.5, maximum=15, live=True),
            field_spec("timeout_s", "Network timeout (s)", "float", 5.0, minimum=1, maximum=30, advanced=True),
        ]

    def _url(self, path):
        return f"http://{self.settings['host']}:{int(self.settings['port'])}{path}"

    def open(self):
        info = json.loads(_get(self._url("/info"), self.settings["timeout_s"]).decode())
        if info.get("agent") != "tubemap-agent":
            raise RuntimeError("that address answered, but it is not a TubeMap agent")
        self.remote = info
        self.info = {"model": info.get("device", {}).get("model", "agent"),
                     "serial": info.get("device", {}).get("serial", ""),
                     "agent_version": info.get("version"), "host": self.settings["host"],
                     "color_size": info.get("device", {}).get("color_size")}
        self.seq = 0
        import time
        self._last = 0.0
        self._time = time

    def read(self, timeout=2.0):
        period = 1.0 / max(0.5, float(self.settings["live_fps"]))
        wait = self._last + period - self._time.time()
        if wait > 0:
            self._time.sleep(wait)
        self._last = self._time.time()
        raw = _get(self._url(f"/frame.jpg?max={int(self.settings['live_max_width'])}"), max(timeout, 1.0))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        self.seq += 1
        return FrameSet(color=img, depth=None, intrinsics=None, seq=self.seq, meta={"preview": True})

    def capture(self, timeout=5.0):
        raw = _get(self._url("/capture"), max(timeout, float(self.settings["timeout_s"])))
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        depth = z["depth"] if z["depth"].size else None
        self.seq += 1
        return FrameSet(color=z["color"], depth=depth, intrinsics=meta.get("intrinsics"), seq=self.seq,
                        meta={"agent": meta.get("device"), "agent_ts": meta.get("timestamp"),
                              "depth_aligned": meta.get("depth_aligned", depth is not None)})

    def apply_live(self, changed):
        if set(changed) <= self.live_keys():
            self.settings.update(changed)
            return True
        return False

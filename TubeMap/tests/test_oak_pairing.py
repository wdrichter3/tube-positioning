"""OAK colour/depth pairing with stereo depth arriving one frame late (no camera needed)."""
from collections import deque
from datetime import timedelta

import numpy as np

from tubemap.devices.oak import OakDriver


class Msg:
    def __init__(self, t, kind):
        self.t, self.kind = t, kind

    def getTimestamp(self):
        return timedelta(seconds=self.t)

    def getCvFrame(self):
        return np.full((4, 4, 3), int(self.t * 10) % 255, np.uint8)

    def getFrame(self):
        return np.full((4, 4), 1000, np.uint16)


class Q:
    def __init__(self):
        self.items = []

    def tryGetAll(self):
        out, self.items = self.items, []
        return out


def make():
    d = OakDriver({"fps": 10})
    d.q_rgb, d.q_depth = Q(), Q()
    d.rgb, d.dep = deque(maxlen=8), deque(maxlen=8)
    d.stats = {"rgb_frames": 0, "depth_frames": 0, "paired": 0, "unpaired": 0, "last_pair_dt_ms": None}
    d.intr, d.seq, d.eeprom_lens = {"fx": 1}, 0, 0
    return d


def test_depth_one_frame_late_still_pairs():
    d = make()
    for k in range(5):                      # colour frames 0..4, depth only up to frame 3 so far
        d.q_rgb.items.append(Msg(k * 0.1, "rgb"))
        if k < 4:
            d.q_depth.items.append(Msg(k * 0.1 + 0.003, "depth"))
    d._pump()
    fs = d._pair(strict=True)
    assert fs is not None and fs.depth is not None and abs(fs.meta["device_ts"] - 0.3) < 1e-9
    assert d.stats["paired"] == 1


def test_no_depth_live_frame_still_shown_but_capture_waits():
    d = make()
    d.q_rgb.items.append(Msg(0.0, "rgb"))
    d._pump()
    assert d._pair(strict=True) is None
    fs = d._pair(strict=False)
    assert fs is not None and fs.depth is None


def test_capture_falls_back_to_colour_when_depth_never_arrives():
    d = make()
    d.settings["fps"] = 10
    for k in range(3):
        d.q_rgb.items.append(Msg(k * 0.1, "rgb"))
    d._pump()
    import threading, time
    def feed():
        for k in range(3, 40):
            d.q_rgb.items.append(Msg(k * 0.1, "rgb")); time.sleep(0.02)
    threading.Thread(target=feed, daemon=True).start()
    fs = d.capture(timeout=2.0)
    assert fs.depth is None and "no depth frames" in fs.meta["depth_missing"]


def test_half_size_depth_pipeline():
    import pytest
    dai = pytest.importorskip("depthai")
    from tubemap.devices.oak import build_pipeline
    _, geo = build_pipeline(dai, OakDriver.normalize({}))
    assert geo["depth_size"] == [624, 468]
    _, geo = build_pipeline(dai, OakDriver.normalize({"depth_size": "full"}))
    assert geo["depth_size"] == [1248, 936]

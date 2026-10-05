"""Orbbec driver logic against a fake v1 SDK (Astra+ behaviour). No hardware needed."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

FAKES = str(Path(__file__).resolve().parent / "fakes")


@pytest.fixture
def fake_sdk(monkeypatch):
    for k in [m for m in sys.modules if m.startswith("pyorbbecsdk")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.syspath_prepend(FAKES)
    import pyorbbecsdk as ob
    from tubemap.vision import synth
    sc = synth.render(width=1920, height=1080, hfov_deg=69.2, distance_mm=900, depth_scale=1.0, tilt_deg=6, seed=4)
    dc = sc["depth"]
    ob.set_scene(sc["color"], dc, cv2.resize(dc, (640, 480), interpolation=cv2.INTER_NEAREST))
    ob.STATE.update(mirror_color=True, mirror_depth=True, laser=False, hw_aligns=False)
    yield ob, sc
    for k in [m for m in sys.modules if m.startswith("pyorbbecsdk")]:
        sys.modules.pop(k, None)


def test_auto_alignment_falls_back_to_software(fake_sdk):
    ob, sc = fake_sdk
    from tubemap.devices.orbbec import OrbbecDriver
    d = OrbbecDriver({})
    d.open()
    assert d.info["sdk"] == 1 and d.info["depth_alignment"] == "sw" and d.info["depth_aligned"]
    assert ob.STATE["mirror_color"] is False and ob.STATE["laser"] is True
    fs = d.capture()
    assert fs.color.shape == (1080, 1920, 3) and fs.depth.shape == (1080, 1920)
    assert np.abs(fs.color.astype(int) - sc["color"]).mean() < 2   # un-mirrored, MJPG decoded
    assert "MJPG @ 30" in d.info["color_profile"]                     # 1080p only exists at 30 fps
    assert abs(float(np.median(fs.depth[fs.depth > 0])) - float(np.median(sc["depth"][sc["depth"] > 0]))) < 2
    assert fs.intrinsics["width"] == 1920 and fs.meta["depth_aligned"]


def test_hardware_alignment_used_when_it_works(fake_sdk):
    ob, _ = fake_sdk
    ob.STATE["hw_aligns"] = True
    from tubemap.devices.orbbec import OrbbecDriver
    d = OrbbecDriver({})
    d.open()
    assert d.info["depth_alignment"] == "hw"


def test_alignment_off_marks_depth_unusable(fake_sdk):
    from tubemap.devices.orbbec import OrbbecDriver
    d = OrbbecDriver({"align": "off"})
    d.open()
    fs = d.capture()
    assert fs.depth.shape == (480, 640) and fs.meta["depth_aligned"] is False


def test_astra_capture_maps_and_sees_plugs_in_depth(fake_sdk):
    ob, sc = fake_sdk
    from tubemap.devices.orbbec import OrbbecDriver
    from tubemap.vision import mapping
    d = OrbbecDriver({})
    d.open()
    fs = d.capture()
    g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_mm": sc["pitch_mm"], "tube_id_mm": sc["tube_id_mm"]}
    m = mapping.build_map(fs.color, fs.depth, fs.intrinsics, g)
    assert m["ready"]
    plugs = [s for s in m["sites"] if s["state"] == "PLUGGED"]
    assert plugs and all(s["depth_evidence"] == "FACE" for s in plugs)
    opens = [s for s in m["sites"] if s["state"] == "OPEN"]
    assert sum(s["depth_evidence"] == "RECESS" for s in opens) / len(opens) > 0.98

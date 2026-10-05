import json
import threading
import time
import urllib.request

import pytest

from tubemap.devices.base import DeviceDriver
from tubemap.devices.registry import DRIVERS, describe
from tubemap.devices.simulator import SimulatorDriver


def test_every_driver_has_a_valid_schema():
    for d in describe():
        keys = [f["key"] for f in d["schema"]]
        assert len(keys) == len(set(keys)), d["id"]
        norm = DRIVERS[d["id"]].normalize({})
        assert set(norm) == set(keys)


def test_normalize_clamps_and_rejects_bad_values():
    s = SimulatorDriver.normalize({"tilt_deg": 400, "pattern": "HEX", "fps": "abc"})
    assert s["tilt_deg"] == 45 and s["pattern"] == "TRIANGULAR" and s["fps"] == 5.0


def test_simulator_driver_frames():
    d = SimulatorDriver({"fps": 30})
    d.open()
    fs = d.read()
    assert fs.color.shape == (936, 1248, 3) and fs.depth is not None and fs.intrinsics["fx"] > 0


def test_oak_pipeline_builds_without_camera():
    dai = pytest.importorskip("depthai")
    from tubemap.devices.oak import OakDriver, build_pipeline
    for isp in ("4/13", "1/2", "1/1"):
        p, geo = build_pipeline(dai, OakDriver.normalize({"isp_scale": isp}))
        p.serializeToJson()
        assert geo["depth_size"][0] % 16 == 0


@pytest.fixture
def server(data_dir):
    from tubemap import config
    from tubemap.server import serve
    config.save_devices([{"id": "sim-1", "name": "Sim", "driver": "simulator", "enabled": True, "role": "overview",
                          "settings": {"fps": 10}}])
    app, srv = serve("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    for _ in range(100):
        st = json.loads(urllib.request.urlopen(base + "/api/status").read())
        if st["devices"][0]["state"] == "running" and st["devices"][0]["frames"] > 0:
            break
        time.sleep(0.1)
    yield base
    srv.shutdown()
    app.devices.shutdown()


def call(base, path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_capture_map_label_flow(server):
    code, cap = call(server, "/api/devices/sim-1/capture", "POST")
    assert code == 200 and cap["has_depth"]
    g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_value": 1.0, "pitch_unit": "in", "id_value": 0.62, "id_unit": "in"}
    code, m = call(server, f"/api/captures/{cap['id']}/map", "POST", {"geometry": g})
    assert code == 200 and m["ready"]
    code, p = call(server, f"/api/captures/{cap['id']}/propose", "POST", {"geometry": g})
    assert code == 200 and p["source"] == "map"
    code, r = call(server, f"/api/captures/{cap['id']}/labels", "PUT", {"items": p["items"], "reviewed": True})
    assert code == 200 and r["reviewed"]
    code, bad = call(server, f"/api/captures/{cap['id']}/map", "POST", {"geometry": {**g, "pitch_value": "x"}})
    assert code == 400


def test_device_add_update_remove(server):
    code, d = call(server, "/api/devices", "POST", {"name": "Second", "driver": "simulator", "settings": {"seed": 3}})
    assert code == 200
    code, r = call(server, f"/api/devices/{d['id']}", "PUT", {"settings": {"fps": 2}})
    assert r["applied_live"] or r["restarted"]
    code, r = call(server, f"/api/devices/{d['id']}", "PUT", {"settings": {"depth_scale_factor": 1.01}})
    assert not r["restarted"]
    code, _ = call(server, f"/api/devices/{d['id']}", "DELETE")
    code, lst = call(server, "/api/devices")
    assert [x["id"] for x in lst] == ["sim-1"]
    code, e = call(server, "/api/devices", "POST", {"driver": "nope"})
    assert code == 400


@pytest.fixture
def server_pair(data_dir):
    """OAK-like overview simulator + Astra-like depth simulator looking at the same sheet."""
    from tubemap import config
    from tubemap.server import serve
    config.save_devices([
        {"id": "oak-sim", "name": "OAK sim", "driver": "simulator", "enabled": True, "role": "overview",
         "settings": {"fps": 10, "seed": 31, "plugged_fraction": 0.05}},
        {"id": "astra-sim", "name": "Astra+ sim", "driver": "simulator", "enabled": True, "role": "depth",
         "settings": {"fps": 10, "seed": 31, "plugged_fraction": 0.05, "width": 1920, "height": 1080, "hfov_deg": 69.2,
                      "distance_mm": 900, "tilt_deg": 6, "yaw_deg": 4, "depth_scale": 0.34, "depth_smooth_px": 3}}])
    app, srv = serve("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    for _ in range(200):
        st = json.loads(urllib.request.urlopen(base + "/api/status").read())
        if all(d["state"] == "running" and d["frames"] > 0 for d in st["devices"]):
            break
        time.sleep(0.1)
    yield base
    srv.shutdown()
    app.devices.shutdown()


def test_companion_depth_camera_flow(server_pair):
    base = server_pair
    code, cap = call(base, "/api/devices/oak-sim/capture", "POST")
    assert code == 200 and len(cap["companions"]) == 1 and not cap["companions"][0].get("error")
    g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_value": 1.0, "pitch_unit": "in", "id_value": 0.62, "id_unit": "in"}
    code, m = call(base, f"/api/captures/{cap['id']}/map", "POST", {"geometry": g})
    assert code == 200 and m["ready"], m["checks"]
    ids = {c["id"]: c for c in m["checks"]}
    assert ids["pitch_companion"]["status"] == "PASS" and ids["pitch"]["status"] == "INFO"
    assert m["companion"]["match"]["valid"] and m["companion"]["match"]["overlap"] > 0.95
    plugs = [s for s in m["sites"] if s["state"] == "PLUGGED"]
    assert plugs and all(s.get("depth_evidence") == "FACE" and s.get("depth_source") == "Astra+ sim" for s in plugs)
    # a wrong pitch is caught by the companion camera
    code, bad = call(base, f"/api/captures/{cap['id']}/map", "POST", {"geometry": {**g, "pitch_value": 1.0625}})
    assert not bad["ready"] and {c["id"]: c for c in bad["checks"]}["pitch_companion"]["status"] == "FAIL"
    # calibrating the companion's depth scale writes to that device (from the correct map)
    call(base, f"/api/captures/{cap['id']}/map", "POST", {"geometry": g})
    code, cal = call(base, f"/api/captures/{cap['id']}/calibrate_depth", "POST", {"target": "companion"})
    assert code == 200 and cal["device_updated"] and abs(cal["factor"] - 1) < 0.02

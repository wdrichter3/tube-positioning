"""Manual end-to-end check against a running server (python3 tests/apitest_live.py [base_url])."""
import json
import sys
import time
import urllib.request

B = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8099"


def req(path, method="GET", body=None, raw=False, headers=None):
    data = None
    h = headers or {}
    if body is not None and not isinstance(body, (bytes, bytearray)):
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    elif body is not None:
        data = body
    r = urllib.request.Request(B + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            b = f.read()
            return b if raw else json.loads(b)
    except urllib.error.HTTPError as e:
        return {"HTTP": e.code, "body": e.read().decode()[:300]}


print(req("/api/devices/sim-1", "PUT", {"enabled": True})["restarted"])
print(req("/api/devices/oak-1", "PUT", {"enabled": False})["restarted"])
time.sleep(3)
st = req("/api/status")
print([(d["id"], d["state"], d["fps"], d["resolution"], d["has_depth"]) for d in st["devices"]])
cap = req("/api/devices/sim-1/capture", "POST")
print("capture", cap["id"], cap["has_depth"])
cid = cap["id"]
g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_value": 1.0, "pitch_unit": "in", "id_value": 0.62, "id_unit": "in"}
m = req(f"/api/captures/{cid}/map", "POST", {"geometry": g})
print("map", m.get("ready"), m.get("counts"), [(c["id"], c["status"]) for c in m.get("checks", [])], m.get("elapsed_s"))
S = {s["id"]: s for s in m["sites"]}
ids = [s["id"] for s in m["sites"] if s["state"] == "OPEN"]
a = [{"id": i, "machine_x_mm": -S[i]["sheet_y_mm"] + 500, "machine_y_mm": S[i]["sheet_x_mm"] + 100}
     for i in (ids[0], ids[len(ids) // 2], ids[-1])]
m2 = req(f"/api/captures/{cid}/anchors", "PUT", {"anchors": a, "allow_scale": True})
print("anchors", {k: m2["transform"][k] for k in ("rms_mm", "rotation_deg", "scale_error_pct", "anchors_used", "mirrored")})
print("machine xy of anchor0", [(s["machine_x_mm"], s["machine_y_mm"]) for s in m2["sites"] if s["id"] == ids[0]], a[0])
csv = req(f"/api/captures/{cid}/map.csv", raw=True).decode().splitlines()
print(csv[0]); print(csv[1])
p = req(f"/api/captures/{cid}/propose", "POST", {"geometry": g})
print("propose", p["source"], len(p["items"]))
print("labels", req(f"/api/captures/{cid}/labels", "PUT", {"items": p["items"], "reviewed": True}))
cal = req(f"/api/captures/{cid}/calibrate_depth", "POST")
print("cal", cal.get("factor") if "factor" in cal else cal)
print("stats", req("/api/dataset/stats"))
print("snapshot bytes", len(req("/api/devices/sim-1/snapshot.jpg", raw=True)))
print("depth jpg", req("/api/captures/" + cid + "/depth.jpg", raw=True)[:3])
print("bad pitch", [(c["id"], c["status"]) for c in req(f"/api/captures/{cid}/map", "POST", {"geometry": dict(g, pitch_value=1.0625)})["checks"]])
d = req(f"/api/captures/{cid}/map", "POST", {"geometry": {"mode": "discover", "pattern": "AUTO"}})["geometry"]
print("discover", {k: v for k, v in d.items() if k in ("pattern", "pitch_mm", "nearest_pitch")})
print("bad json", req(f"/api/captures/{cid}/map", "POST", b"{bad", headers={"Content-Type": "application/json"}))
print("traversal", req("/../../etc/passwd"))
print("bad id", req("/api/captures/..%2Fetc"))

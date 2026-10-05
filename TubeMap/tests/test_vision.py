import json

import cv2
import numpy as np
import pytest

from conftest import FIXTURES
from tubemap.vision import detect, geometry, lattice, mapping, metric, synth

IN = 25.4


def known(sc, **kw):
    g = {"mode": "known", "pattern": sc["params"]["pattern"], "pitch_mm": sc["pitch_mm"],
         "tube_id_mm": sc["tube_id_mm"]}
    g.update(kw)
    return g


@pytest.mark.parametrize("kw", [
    dict(pattern="TRIANGULAR", pitch_in=1.0, tube_id_in=0.62, tilt_deg=15),
    dict(pattern="SQUARE", pitch_in=1.25, tube_id_in=0.834, lattice_rot_deg=45, tilt_deg=25, yaw_deg=10),
    dict(pattern="SQUARE", pitch_in=1.0, tube_id_in=0.62, lattice_rot_deg=0, tilt_deg=30),
    dict(pattern="TRIANGULAR", pitch_in=0.8125, tube_id_in=0.527, lattice_rot_deg=90, tilt_deg=5, distance_mm=900),
])
def test_known_geometry_finds_every_open_tube(kw):
    sc = synth.render(**kw)
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    truth_open = sum(t["state"] == "OPEN" for t in sc["truth"])
    truth_plug = sum(t["state"] == "PLUGGED" for t in sc["truth"])
    assert m["ready"], m["checks"]
    assert m["counts"]["OPEN"] == truth_open
    # rule-based plug calls (no trained model): allow a small miss rate at steep tilt
    assert abs(m["counts"].get("PLUGGED", 0) - truth_plug) <= max(1, round(0.15 * truth_plug))
    assert m["lattice"]["residual_px_median"] < 0.3


def test_positions_match_ground_truth_after_undistortion():
    sc = synth.render(k1=-0.15, tilt_deg=20)
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    T = np.array([[t["u"], t["v"]] for t in sc["truth"]])
    errs = []
    for s in m["sites"]:
        if s["observed"]:
            errs.append(np.min(np.hypot(T[:, 0] - s["u"], T[:, 1] - s["v"])))
    assert np.median(errs) < 0.15 and np.max(errs) < 0.6
    assert m["intrinsics_used"]


def test_sheet_coordinates_are_exact_lattice_multiples():
    sc = synth.render()
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    xy = np.array([[s["sheet_x_mm"], s["sheet_y_mm"]] for s in m["sites"]])
    d = geometry.nearest_neighbor_dist(xy)
    assert abs(np.median(d) - sc["pitch_mm"]) < 1e-3


@pytest.mark.parametrize("wrong,check", [
    (dict(pitch_mm=1.0625 * IN), "pitch"),      # 1-1/16 entered for a 1.0 pitch sheet
    (dict(pattern="SQUARE"), "pattern"),
    (dict(tube_id_mm=0.62 * IN * 0.78), "tube_id"),
])
def test_wrong_entries_are_blocked(wrong, check):
    sc = synth.render()
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc, **wrong))
    assert not m["ready"]
    assert next(c for c in m["checks"] if c["id"] == check)["status"] == "FAIL"


def test_discovery_with_depth():
    sc = synth.render(pattern="TRIANGULAR", pitch_in=0.9375, tube_id_in=0.62, tilt_deg=30, yaw_deg=15)
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], {"mode": "discover"})
    g = m["geometry"]
    assert g["pattern"] == "TRIANGULAR"
    assert abs(g["pitch_mm"] / sc["pitch_mm"] - 1) < 0.01
    assert g["nearest_pitch"]["nominal_in"] == 0.9375


def test_discovery_without_depth_reports_no_pitch():
    sc = synth.render()
    m = mapping.build_map(sc["color"], None, sc["intrinsics"], {"mode": "discover"})
    assert m["geometry"]["pitch_mm"] is None and not m["ready"]


def test_no_intrinsics_still_maps_with_warning():
    sc = synth.render(k1=0.0)
    m = mapping.build_map(sc["color"], None, None, known(sc))
    assert m["counts"]["OPEN"] == sum(t["state"] == "OPEN" for t in sc["truth"])
    assert any(c["id"] == "intrinsics" and c["status"] == "WARN" for c in m["checks"])


def test_roi_limits_sites():
    sc = synth.render()
    roi = [[0.45, 0.3], [0.7, 0.3], [0.7, 0.6], [0.45, 0.6]]
    full = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    part = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc), roi=roi)
    assert 20 < part["tube_sites"] < full["tube_sites"] / 2


def test_anchor_transform_recovers_rotation_translation_and_scale():
    sc = synth.render()
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    sites = [s for s in m["sites"] if s["state"] == "OPEN"]
    pick = [sites[0], sites[len(sites) // 2], sites[-1], sites[len(sites) // 4]]
    th = np.radians(-2.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    anchors = []
    for s in pick:
        p = 1.001 * R @ [s["sheet_x_mm"], s["sheet_y_mm"]] + [812.0, 95.0]
        anchors.append({"id": s["id"], "machine_x_mm": p[0], "machine_y_mm": p[1]})
    tf = mapping.fit_anchor_transform(m["sites"], anchors, allow_scale=True)
    assert tf["valid"] and tf["rms_mm"] < 1e-6
    assert abs(tf["rotation_deg"] + 2.0) < 1e-6 and abs(tf["scale_error_pct"] - 0.1) < 1e-6
    mapping.apply_anchor_transform(m["sites"], tf)
    s0 = next(s for s in m["sites"] if s["id"] == pick[0]["id"])
    assert abs(s0["machine_x_mm"] - anchors[0]["machine_x_mm"]) < 1e-3


def test_depth_plane_recovers_simulated_plane():
    sc = synth.render(tilt_deg=20)
    h, w = sc["depth"].shape
    T = np.array([[t["u"], t["v"]] for t in sc["truth"]]) * [w / sc["color"].shape[1], h / sc["color"].shape[0]]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(T.astype(np.int32))], 1)
    pl = metric.depth_plane(sc["depth"], sc["intrinsics"], mask.astype(bool))
    R = np.array(sc["plane"]["R"])
    assert abs(pl["tilt_deg"] - np.degrees(np.arccos(abs(R[2, 2])))) < 1.0


def test_undistort_roundtrip():
    intr = {"width": 1000, "height": 800, "fx": 900, "fy": 900, "cx": 500, "cy": 400, "dist": [-0.1, 0.03, 0.001, -0.001, 0]}
    p = np.array([[10.0, 20.0], [500, 400], [990, 790]])
    back = metric.distort_px(metric.undistort_px(p, intr), intr)
    assert np.abs(back - p).max() < 1e-3


def test_geometry_discovery_prior_free():
    ij = np.array([(i, j) for i in range(-12, 13) for j in range(-12, 13)])
    for pat, pitch in (("TRIANGULAR", 23.8125), ("SQUARE", 31.75)):
        p = geometry.ideal_xy(ij, pat, pitch)
        p = p[np.hypot(*p.T) < 250] + np.random.default_rng(0).normal(0, 0.05, (0, 2)).sum() 
        r = geometry.discover_geometry(p)
        assert r["pattern"] == pat and abs(r["pitch"] - pitch) < 1e-6


def test_tube_candidates_and_pitch_snap():
    c = geometry.tube_candidates(15.748, 25.4)
    assert c[0]["od_in"] == 0.75 and c[0]["bwg"] == 16
    s = geometry.snap_pitch(23.8125)
    assert s["nominal_in"] == 0.9375 and s["accepted"]


@pytest.mark.skipif(not (FIXTURES / "oak_field_0142" / "color.png").exists(), reason="field fixture missing")
def test_real_oak_field_capture():
    d = FIXTURES / "oak_field_0142"
    info = json.loads((d / "info.json").read_text())
    img = cv2.imread(str(d / "color.png"))
    dep = cv2.imread(str(d / "depth.png"), cv2.IMREAD_UNCHANGED)
    g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_mm": 1.118 * IN, "tube_id_mm": 0.75 * IN}
    m = mapping.build_map(img, dep, info["intrinsics"], g, roi=info["roi"])
    assert m["counts"]["OPEN"] >= 450          # 0.14.x found 443-458 sites on this image
    assert m["lattice"]["residual_px_median"] < 0.5
    assert next(c for c in m["checks"] if c["id"] == "pattern")["status"] == "PASS"
    assert next(c for c in m["checks"] if c["id"] == "tube_id")["status"] == "PASS"


def _pair(**kw_b):
    base = dict(seed=21, plugged_fraction=0.04)
    a = synth.render(**base, tilt_deg=12, yaw_deg=-5)
    b = synth.render(**base, width=1920, height=1080, hfov_deg=69.2, distance_mm=900, depth_scale=0.34, **kw_b)
    g = known(a)
    return (mapping.build_map(a["color"], a["depth"], a["intrinsics"], g),
            mapping.build_map(b["color"], b["depth"], b["intrinsics"], g))


@pytest.mark.parametrize("kw,mirrored,rot", [
    (dict(tilt_deg=8, yaw_deg=6, depth_smooth_px=5), False, 0),
    (dict(tilt_deg=8, mirror=True), True, 0),
    (dict(tilt_deg=8, roll_deg=90), False, -90),
])
def test_companion_lattice_match(kw, mirrored, rot):
    from tubemap.vision import depthcheck
    ma, mb = _pair(**kw)
    r = depthcheck.match_lattices(ma, mb)
    assert r["valid"] and r["overlap"] > 0.97 and r["state_agreement"] > 0.98
    assert r["mirrored"] == mirrored and abs(r["rotation_deg"] - rot) < 5
    depthcheck.merge_companion(ma, mb, "Astra+")
    plugs = [s for s in ma["sites"] if s["state"] == "PLUGGED"]
    assert all(s["depth_evidence"] == "FACE" for s in plugs)


def test_depth_evidence_needs_enough_pixels():
    sc = synth.render(depth_scale=0.25)
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], known(sc))
    assert next(c for c in m["checks"] if c["id"] == "depth_states")["status"] == "SKIP"


def test_depth_disagreement_can_mark_review():
    sc = synth.render(distance_mm=700, sheet_diameter_mm=500, depth_scale=1.0)
    d = sc["depth"].copy()
    t = next(x for x in sc["truth"] if x["state"] == "OPEN")          # fake a plug seen only in depth
    u, v = int(t["u"]), int(t["v"])
    patch = d[v - 40:v + 41, u - 40:u + 41]
    face = int(np.percentile(patch[patch > 0], 20))                    # surrounding sheet face
    cv2.circle(d, (u, v), 12, face, -1)
    m = mapping.build_map(sc["color"], d, sc["intrinsics"], known(sc), tolerances={"depth_to_state": True})
    s = min(m["sites"], key=lambda s: (s["u"] - t["u"]) ** 2 + (s["v"] - t["v"]) ** 2)
    assert s["depth_evidence"] == "FACE" and s["state"] == "REVIEW" and s["state_before_depth"] == "OPEN"

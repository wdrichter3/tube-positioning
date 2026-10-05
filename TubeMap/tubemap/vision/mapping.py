"""Build a tube map from one capture: detection -> lattice -> metric -> states -> checks."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .. import VERSION
from . import depthcheck, detect, geometry, lattice, metric, state
from .geometry import MM_PER_IN, SQUARE, TRIANGULAR, lattice_basis

DEFAULT_TOLERANCES = {
    "id_ratio_tol": 0.10,            # apparent opening / pitch vs entered ID / pitch
    "apparent_id_factor": 1.0,       # learned bias: apparent dark diameter / true ID
    "pitch_depth_tol_pct": 3.0,      # entered pitch vs depth-scaled pitch
    "depth_scale_uncertainty_pct": 2.0,
    "residual_rel_warn": 0.05,
    "residual_rel_fail": 0.10,
    "review_below": 0.6,
    "depth_face_tol_mm": 5.0,
    "depth_recess_mm": 12.0,
    "depth_min_px": 10,
    "depth_to_state": False,
}


def _check(cid, label, status, detail, **extra):
    d = {"id": cid, "label": label, "status": status, "detail": detail}
    d.update(extra)
    return d


def _roi_px(roi, w, h):
    if not roi or len(roi) < 3:
        return None
    return np.array([[float(x) * (w - 1), float(y) * (h - 1)] for x, y in roi], np.float64)


def _depth_mask(depth_shape, color_shape, hull_px_raw, holes_raw, hole_r_raw):
    dh, dw = depth_shape
    ch, cw = color_shape
    sx, sy = dw / float(cw), dh / float(ch)
    m = np.zeros((dh, dw), np.uint8)
    if hull_px_raw is not None and len(hull_px_raw) >= 3:
        poly = np.round(np.asarray(hull_px_raw) * [sx, sy]).astype(np.int32)
        cv2.fillPoly(m, [poly], 1)
    else:
        m[:] = 1
    rr = max(1, int(round(hole_r_raw * sx)))
    for u, v in holes_raw:
        cv2.circle(m, (int(round(u * sx)), int(round(v * sy))), rr, 0, -1)
    return m.astype(bool)


def _metric_pitch_from_plane(cells_und, cells_ij, intr, plane, pattern):
    """Median metric spacing between lattice neighbours of predicted cell centres."""
    xyz = metric.intersect_plane(cells_und, intr, plane)
    pos = {(int(f), int(i), int(j)): xyz[k] for k, (f, i, j) in enumerate(cells_ij)}
    d1, d2 = [], []
    for (f, i, j), p in pos.items():
        q = pos.get((f, i + 1, j))
        if q is not None:
            d1.append(np.linalg.norm(q - p))
        q = pos.get((f, i, j + 1))
        if q is not None:
            d2.append(np.linalg.norm(q - p))
    if not d1 or not d2:
        return None
    a, b = float(np.median(d1)), float(np.median(d2))
    return {"pitch_mm": 0.5 * (a + b), "dir1_mm": a, "dir2_mm": b,
            "anisotropy_pct": 100.0 * abs(a - b) / (0.5 * (a + b)), "pairs": len(d1) + len(d2)}


def _addresses(sites, H, pattern):
    """Rows = lattice rows (per drilled field), ordered top-to-bottom in the image; tubes left-to-right."""
    Bu = lattice_basis(pattern)
    Hn = np.asarray(H, np.float64)
    p0 = lattice._apply_h(Hn, np.array([[0.0, 0.0]]))[0]
    pi = lattice._apply_h(Hn, np.array(Bu[0:1]))[0]
    pj = lattice._apply_h(Hn, np.array(Bu[1:2]))[0]
    flip_i = -1 if (pi - p0)[0] < 0 else 1
    j_sign = 1 if (pj - p0)[1] >= 0 else -1
    rows = {}
    for s in sites:
        rows.setdefault((s.get("field", 0), s["j"]), []).append(s)
    row_keys = sorted(rows, key=lambda k: np.mean([x["v"] for x in rows[k]]))
    for rnum, key in enumerate(row_keys, 1):
        members = sorted(rows[key], key=lambda x: flip_i * x["lx"])
        t = 0
        for s in members:
            s["row"] = rnum
            if s["state"] != "NO_TUBE":
                t += 1
                s["tube"] = t
                s["id"] = f"R{rnum:02d}-T{t:02d}"
            else:
                s["tube"] = None
                s["id"] = f"R{rnum:02d}-X{s['i']:+d}"
    # sheet frame: origin at first tube of row 1, +x along rows, +y toward later rows
    first = next((s for s in sites if s.get("row") == 1 and s.get("tube") == 1), sites[0])
    ideal0 = np.array([first["lx"], first["ly"]], float) @ Bu
    for s in sites:
        d = np.array([s["lx"], s["ly"]], float) @ Bu - ideal0
        s["_ideal_rel"] = (float(flip_i * d[0]), float(j_sign * d[1]))
    return sites


def fit_anchor_transform(sites, anchors, allow_scale=False):
    """anchors: [{id, machine_x_mm, machine_y_mm}] -> transform sheet -> machine."""
    by_id = {s["id"]: s for s in sites}
    X, Y, used = [], [], []
    for a in anchors:
        s = by_id.get(a.get("id"))
        if s is None or s.get("sheet_x_mm") is None:
            continue
        X.append([s["sheet_x_mm"], s["sheet_y_mm"]])
        Y.append([float(a["machine_x_mm"]), float(a["machine_y_mm"])])
        used.append(a["id"])
    if len(X) < 2:
        return {"valid": False, "reason": "need at least 2 anchor tubes with sheet coordinates"}
    X, Y = np.array(X), np.array(Y)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    D = np.eye(2)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[1, 1] = -1
    R = Vt.T @ D @ U.T
    s = float(np.trace(np.diag(S) @ D) / np.sum(Xc ** 2)) if allow_scale else 1.0
    t = ym - s * R @ xm
    pred = (s * (R @ X.T)).T + t
    res = np.linalg.norm(Y - pred, axis=1)
    reflect = bool(np.linalg.det(R) < 0)
    return {"valid": True, "R": R.tolist(), "t": t.tolist(), "scale": s,
            "scale_error_pct": 100.0 * (s - 1.0) if allow_scale else None,
            "rotation_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])), "mirrored": reflect,
            "residuals_mm": {i: float(r) for i, r in zip(used, res)},
            "rms_mm": float(np.sqrt(np.mean(res ** 2))), "max_mm": float(res.max()),
            "anchors_used": len(used),
            "note": ("2 anchors define the transform exactly; add a third to see real residuals."
                     if len(used) == 2 else "")}


def apply_anchor_transform(sites, tf):
    if not tf or not tf.get("valid"):
        for s in sites:
            s.pop("machine_x_mm", None); s.pop("machine_y_mm", None)
        return sites
    R, t, sc = np.array(tf["R"]), np.array(tf["t"]), float(tf["scale"])
    for s in sites:
        if s.get("sheet_x_mm") is None:
            continue
        m = sc * R @ np.array([s["sheet_x_mm"], s["sheet_y_mm"]]) + t
        s["machine_x_mm"], s["machine_y_mm"] = round(float(m[0]), 3), round(float(m[1]), 3)
    return sites


def build_map(color, depth=None, intrinsics=None, geometry_in=None, roi=None,
              profile=None, tolerances=None) -> dict:
    t0 = time.time()
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update({k: v for k, v in tolerances.items() if k in DEFAULT_TOLERANCES})
    g_in = dict(geometry_in or {})
    mode = g_in.get("mode", "known")
    pattern_in = str(g_in.get("pattern") or "AUTO").upper()
    pitch_in = g_in.get("pitch_mm")
    id_in = g_in.get("tube_id_mm")
    if mode == "known":
        if pattern_in not in (TRIANGULAR, SQUARE) or not pitch_in or not id_in:
            raise ValueError("Known-geometry mode needs pattern, pitch and tube ID.")
        pitch_in, id_in = float(pitch_in), float(id_in)
        if not (0.05 < id_in / pitch_in < 0.98):
            raise ValueError("Tube ID must be smaller than the pitch.")
    profile = profile or {}
    h, w = color.shape[:2]
    intr = intrinsics if metric.valid_intrinsics(intrinsics or {}) else None
    if intr and (int(intr["width"]) != w or int(intr["height"]) != h):
        intr = metric.scale_intrinsics(intr, w, h)

    # 1. detection
    cands, ddiag = detect.detect_openings(color, roi=roi, params=profile.get("detector"))
    if len(cands) < 12:
        raise ValueError(f"Only {len(cands)} openings found. Check the ROI, focus and lighting.")
    raw = np.array([[c["u"], c["v"]] for c in cands])
    und = metric.undistort_px(raw, intr) if intr else raw.copy()
    roi_raw = _roi_px(roi, w, h)
    roi_und = metric.undistort_px(roi_raw, intr) if (roi_raw is not None and intr) else roi_raw

    # 2. lattice (a clearly contradicting image wins over the entered pattern, and is flagged)
    fit_pattern = pattern_in if mode == "known" else None
    if mode == "known":
        p0 = geometry.robust_pitch(und)
        seen, seen_info = geometry.classify_pattern(und, p0)
        if seen and seen != pattern_in and seen_info.get("confidence", 0) >= 0.5:
            fit_pattern = seen
    lat = lattice.fit_image_lattice(und, pattern=fit_pattern, boundary_px=roi_und)
    pattern = lat["pattern"]
    H = np.array(lat["H_ideal_to_px"])
    acc = {a["index"]: a for a in lat["accepted"]}
    obs_by_cell = {(a["field"], a["i"], a["j"]): a["index"] for a in lat["accepted"]}
    cells = lat["cells"]
    have = {(c["field"], c["i"], c["j"]) for c in cells}
    for a in lat["accepted"]:
        if (a["field"], a["i"], a["j"]) not in have:
            cells.append({"field": a["field"], "i": a["i"], "j": a["j"], "lx": a["lx"], "ly": a["ly"],
                          "u": float(und[a["index"], 0]), "v": float(und[a["index"], 1])})
    Bu = lattice_basis(pattern)
    # an empty cell of one field must not sit on top of another field's tube
    if len(lat.get("fields", [])) > 1:
        lp0 = float(lat["pitch_px_median"])
        obs_pts = np.array([[c["u"], c["v"]] for c in cells if (c["field"], c["i"], c["j"]) in obs_by_cell])
        keep = []
        for c in cells:
            if (c["field"], c["i"], c["j"]) in obs_by_cell or not len(obs_pts):
                keep.append(c)
                continue
            if np.min(np.hypot(obs_pts[:, 0] - c["u"], obs_pts[:, 1] - c["v"])) > 0.45 * lp0:
                keep.append(c)
        cells = keep
    cell_und = np.array([[c["u"], c["v"]] for c in cells])
    cell_raw = metric.distort_px(cell_und, intr) if intr else cell_und
    cell_keys = np.array([[c["field"], c["i"], c["j"]] for c in cells])
    cell_l = np.array([[c["lx"], c["ly"]] for c in cells])
    lp_cells = lattice.local_pitch_px(H, cell_l @ Bu)

    # apparent opening ratio (scale-free)
    obs_ratio = np.array([cands[k]["d"] / lattice.local_pitch_px(H, np.array([[acc[k]["lx"], acc[k]["ly"]]]) @ Bu)[0]
                          for k in acc])
    ratio_meas = float(np.median(obs_ratio)) if len(obs_ratio) else float("nan")
    if mode == "known":
        ratio_site = id_in / pitch_in
    else:
        ratio_site = ratio_meas / max(tol["apparent_id_factor"], 1e-3)

    # 3. sites + states
    sites = []
    for k, c in enumerate(cells):
        key = (c["field"], c["i"], c["j"])
        oi = obs_by_cell.get(key)
        if oi is not None:
            u, v = float(raw[oi, 0]), float(raw[oi, 1])
        else:
            u, v = float(cell_raw[k, 0]), float(cell_raw[k, 1])
        s = {"field": int(key[0]), "i": int(key[1]), "j": int(key[2]), "lx": round(c["lx"], 4),
             "ly": round(c["ly"], 4), "u": round(u, 2), "v": round(v, 2),
             "observed": oi is not None, "r_px": float(0.5 * ratio_site * lp_cells[k])}
        if oi is not None:
            s["d_px"] = round(cands[oi]["d"], 2)
            s["residual_px"] = round(acc[oi]["residual_px"], 3)
        sites.append(s)
    fi = state.FeatureImage(color)
    clf = state.load_classifier(profile.get("classifier"))
    state.classify_sites(fi, sites, clf, review_below=tol["review_below"])
    _addresses(sites, H, pattern)

    # 4. metric scale
    checks = []
    plane = None
    depth_pitch = None
    if depth is not None and intr is not None and depth.size:
        hull = cv2.convexHull(cell_raw.astype(np.float32)).reshape(-1, 2)
        holes = [(s["u"], s["v"]) for s in sites if s["observed"]]
        hr = float(np.median(lp_cells)) * 0.55
        mask = _depth_mask(depth.shape[:2], (h, w), hull, holes, hr)
        plane = metric.depth_plane(depth, intr, mask)
        if plane.get("valid"):
            depth_pitch = _metric_pitch_from_plane(cell_und, cell_keys, intr, plane, pattern)
    have_depth = bool(plane and plane.get("valid") and depth_pitch)

    result_geom = {"pattern": pattern, "detected_pattern": lat["detected_pattern"],
                   "pattern_confidence": lat["pattern_info"].get("confidence")}
    notes = []
    if mode == "known":
        pitch_mm, tube_id_mm = pitch_in, id_in
        result_geom.update(pitch_mm=pitch_mm, tube_id_mm=tube_id_mm, pitch_source="entered",
                           tube_id_source="entered")
        # pattern
        if lat["detected_pattern"] and lat["detected_pattern"] != pattern_in:
            checks.append(_check("pattern", "Pattern", "FAIL",
                                 f"Entered {pattern_in.lower()}, image looks {lat['detected_pattern'].lower()}."))
        else:
            checks.append(_check("pattern", "Pattern", "PASS", f"Image agrees: {pattern.lower()}."))
        # ID ratio
        exp_ratio = id_in / pitch_in * tol["apparent_id_factor"]
        dev = ratio_meas / exp_ratio - 1.0
        st = "PASS" if abs(dev) <= tol["id_ratio_tol"] else ("WARN" if abs(dev) <= 1.5 * tol["id_ratio_tol"] else "FAIL")
        implied = ratio_meas / tol["apparent_id_factor"] * pitch_in
        checks.append(_check("tube_id", "Tube ID vs image", st,
                             f"Openings look {100 * dev:+.1f}% vs entered ID (image implies ~{implied / MM_PER_IN:.3f} in "
                             f"at the entered pitch). Tolerance ±{100 * tol['id_ratio_tol']:.0f}%.",
                             deviation_pct=100 * dev))
        # pitch vs depth
        if have_depth:
            dp = depth_pitch["pitch_mm"]
            dev_p = 100.0 * (dp / pitch_in - 1.0)
            lim = max(tol["pitch_depth_tol_pct"], 1.5 * tol["depth_scale_uncertainty_pct"])
            st = "PASS" if abs(dev_p) <= lim else "FAIL"
            checks.append(_check("pitch", "Pitch vs depth", st,
                                 f"Depth-scaled pitch {dp:.2f} mm ({dp / MM_PER_IN:.4f} in) is {dev_p:+.1f}% from the "
                                 f"entered value. Tolerance ±{lim:.1f}% (depth accuracy limits this check).",
                                 deviation_pct=dev_p, depth_pitch_mm=dp))
        else:
            checks.append(_check("pitch", "Pitch vs depth", "SKIP",
                                 "No usable depth or intrinsics: the entered pitch cannot be cross-checked."))
    else:
        if have_depth:
            pts3 = metric.intersect_plane(und[list(acc.keys())], intr, plane)
            pts2 = metric.to_plane_2d(pts3, plane)
            disc = geometry.discover_geometry(pts2, pattern_hint=pattern)
            pitch_mm = float(depth_pitch["pitch_mm"])
            tube_id_mm = ratio_site * pitch_mm
            unc = tol["depth_scale_uncertainty_pct"]
            result_geom.update(pitch_mm=pitch_mm, tube_id_mm=tube_id_mm, pitch_source="depth plane",
                               tube_id_source="apparent opening x depth pitch",
                               scale_uncertainty_pct=unc, discovery=disc,
                               nearest_pitch=geometry.snap_pitch(pitch_mm, tol_frac=unc / 100.0),
                               nearest_tubes=geometry.tube_candidates(tube_id_mm, pitch_mm,
                                                                      tol_mm=max(0.5, tube_id_mm * unc / 100 * 2))[:4])
            checks.append(_check("pitch", "Pitch (measured)", "WARN",
                                 f"{pitch_mm:.2f} mm ({pitch_mm / MM_PER_IN:.4f} in) from depth. Expect about "
                                 f"±{unc:.0f}% with this camera; confirm with a rule before relying on it."))
        else:
            pitch_mm, tube_id_mm = None, None
            result_geom.update(pitch_mm=None, tube_id_mm=None, pitch_source="none",
                               ratio_id_to_pitch=ratio_site)
            checks.append(_check("pitch", "Pitch (measured)", "FAIL",
                                 "No depth: pitch can't be measured. Use known-geometry mode or a depth camera."))
        checks.append(_check("pattern", "Pattern", "PASS",
                             f"Detected {pattern.lower()} (confidence {lat['pattern_info'].get('confidence', 0):.2f})."))
    # lattice quality
    rr = lat["residual_rel_median"]
    st = "PASS" if rr <= tol["residual_rel_warn"] else ("WARN" if rr <= tol["residual_rel_fail"] else "FAIL")
    checks.append(_check("lattice", "Lattice fit", st,
                         f"Median residual {lat['residual_px_median']:.2f} px ({100 * rr:.1f}% of pitch); "
                         f"{len(lat['accepted'])} of {len(cands)} openings on the lattice"
                         + (f" in {len(lat['fields'])} drilled fields (offsets "
                            + ", ".join(f"{fd['offset'][0]:+.2f}/{fd['offset'][1]:+.2f}" for fd in lat['fields'][1:])
                            + " pitch)." if len(lat.get('fields', [])) > 1 else ".")))
    if not intr:
        checks.append(_check("intrinsics", "Lens model", "WARN",
                             "No camera intrinsics: lens distortion not removed. Positions near the edges bend."))
    if plane and plane.get("valid"):
        checks.append(_check("plane", "Sheet plane", "PASS" if plane["rms_mm"] < 6 else "WARN",
                             f"{plane['distance_mm']:.0f} mm away, tilt {plane['tilt_deg']:.1f}°, "
                             f"fit RMS {plane['rms_mm']:.1f} mm."))
    elif depth is not None:
        checks.append(_check("plane", "Sheet plane", "WARN", (plane or {}).get("reason", "no plane")))
    depth_summary = None
    if plane and plane.get("valid"):
        depth_summary = depthcheck.site_depth_evidence(sites, depth, intr, plane, (h, w), tol)
        checks.append(depthcheck.depth_check(depth_summary))

    # 5. sheet coordinates
    unit = pitch_mm if pitch_mm else None
    for s in sites:
        ix, iy = s.pop("_ideal_rel")
        if unit:
            s["sheet_x_mm"], s["sheet_y_mm"] = round(ix * unit, 3), round(iy * unit, 3)
        else:
            s["sheet_x_mm"] = s["sheet_y_mm"] = None
            s["sheet_x_pitch"], s["sheet_y_pitch"] = round(ix, 4), round(iy, 4)
        s["r_px"] = round(s["r_px"], 2)

    counts = {}
    for s in sites:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    ready = not any(c["status"] == "FAIL" for c in checks)
    rejected = [{"u": round(float(raw[r["index"], 0]), 2), "v": round(float(raw[r["index"], 1]), 2),
                 "reason": r["reason"]} for r in lat["rejected"]]
    return {
        "schema": "tubemap.map/1", "software_version": VERSION, "created_at": time.time(),
        "image": {"width": w, "height": h}, "mode": mode, "geometry_input": g_in,
        "geometry": result_geom, "checks": checks, "ready": ready, "notes": notes,
        "counts": counts, "tube_sites": sum(v for k, v in counts.items() if k != "NO_TUBE"),
        "lattice": {k: lat[k] for k in ("pattern", "detected_pattern", "pattern_info", "H_ideal_to_px", "fields",
                                         "pitch_px_median", "residual_px_median", "residual_px_p95",
                                         "residual_rel_median")},
        "detection": {"candidates": len(cands), "expected_diameter_px": ddiag.get("expected_diameter_px"),
                      "diameter_mode_px": ddiag.get("diameter_mode_px"), "apparent_ratio": ratio_meas},
        "plane": plane, "depth_pitch": depth_pitch, "depth_summary": depth_summary, "intrinsics_used": bool(intr),
        "roi": roi, "sites": sites, "rejected": rejected, "anchors": [], "transform": None,
        "elapsed_s": round(time.time() - t0, 3),
    }


def map_to_csv(m: dict) -> str:
    cols = ["id", "row", "tube", "i", "j", "state", "confidence", "sheet_x_mm", "sheet_y_mm",
            "machine_x_mm", "machine_y_mm", "u", "v", "observed"]
    lines = [",".join(cols)]
    for s in sorted(m["sites"], key=lambda x: (x.get("row", 0), x.get("tube") or 9999, x["i"])):
        vals = []
        for c in cols:
            v = s.get(c)
            vals.append("" if v is None else str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def site_labels_from_map(m: dict) -> list:
    """Map sites -> label items used by the labeler as proposals."""
    out = []
    for k, s in enumerate(m["sites"]):
        cls = state.STATE_TO_CLASS.get(s["state"], "UNKNOWN")
        out.append({"id": f"P{k:05d}", "u": s["u"], "v": s["v"], "r": s["r_px"], "cls": cls,
                    "src": "auto", "conf": s.get("confidence")})
    return out

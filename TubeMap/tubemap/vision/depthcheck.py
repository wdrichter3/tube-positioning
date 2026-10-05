"""Depth evidence per tube site, and merging a companion depth camera's map.

Depth evidence (per site, advisory):
    FACE       depth inside the opening sits at the sheet face -> plug, cap or packed mouth
    RECESS     depth inside the opening is behind the face, or there's no return -> open mouth
    UNCLEAR    mixed or noisy
    LOW_RES    too few depth pixels inside the opening to judge

Companion merge:
    A second camera with better depth (e.g. Orbbec Astra+) captures the same sheet at the
    same moment. Its own map gives a pitch check (scale comes from its own depth, so no
    camera-to-camera calibration is needed). Per-site depth evidence is transferred by
    matching the two lattices' tube-occupancy patterns over the lattice symmetry group,
    constrained by the relative image rotation. If the match isn't unambiguous, only the
    summary is used.
"""
from __future__ import annotations

import math

import numpy as np

from . import lattice, metric
from .geometry import SQUARE, TRIANGULAR, lattice_basis

DEFAULTS = {
    "depth_face_tol_mm": 5.0,      # |median(depth - face)| within this -> surface present
    "depth_recess_mm": 12.0,       # median at least this far behind the face -> open
    "depth_min_px": 10,            # depth pixels needed inside an opening
    "depth_to_state": False,       # True: disagreements between image and depth become REVIEW
}
PRESENT = ("OPEN", "PLUGGED", "REVIEW")


# --------------------------------------------------------------------------
# Per-site depth evidence
# --------------------------------------------------------------------------
def site_depth_evidence(sites, depth, intr_color, plane, color_shape, tol=None):
    """Adds depth_evidence / depth_delta_mm / depth_valid_frac to each site. Returns summary."""
    t = dict(DEFAULTS)
    t.update({k: v for k, v in (tol or {}).items() if k in DEFAULTS})
    dh, dw = depth.shape[:2]
    ch, cw = color_shape[:2]
    sx, sy = dw / float(cw), dh / float(ch)
    intr_d = metric.scale_intrinsics(intr_color, dw, dh)
    n = np.asarray(plane["normal"], np.float64)
    d0 = float(plane["d"])
    face_tol = max(float(t["depth_face_tol_mm"]), 2.5 * float(plane.get("rms_mm", 2.0)))
    counts = {"FACE": 0, "RECESS": 0, "UNCLEAR": 0, "LOW_RES": 0}
    px_per_tube = []
    for s in sites:
        r = 0.6 * s["r_px"] * 0.5 * (sx + sy)
        px_per_tube.append(2 * s["r_px"] * 0.5 * (sx + sy))
        cu, cv = s["u"] * sx, s["v"] * sy
        ri = int(math.ceil(r))
        u0, u1 = max(0, int(cu) - ri), min(dw, int(cu) + ri + 2)
        v0, v1 = max(0, int(cv) - ri), min(dh, int(cv) + ri + 2)
        if u1 <= u0 or v1 <= v0:
            s["depth_evidence"] = "LOW_RES"
            counts["LOW_RES"] += 1
            continue
        vv, uu = np.mgrid[v0:v1, u0:u1]
        inside = (uu + 0.5 - cu) ** 2 + (vv + 0.5 - cv) ** 2 <= r * r
        npx = int(inside.sum())
        if npx < int(t["depth_min_px"]):
            s["depth_evidence"] = "LOW_RES"
            counts["LOW_RES"] += 1
            continue
        z = depth[v0:v1, u0:u1][inside].astype(np.float64)
        valid = z > 0
        frac = float(valid.mean())
        s["depth_valid_frac"] = round(frac, 2)
        und = metric.undistort_px(np.array([[cu, cv]]), intr_d)
        ray = metric.rays(und, intr_d)[0]
        z_face = -d0 / float(ray @ n) * ray[2]
        if frac < 0.3:
            ev, delta = "RECESS", None
        else:
            delta = float(np.median(z[valid]) - z_face)
            if abs(delta) <= face_tol and frac >= 0.6:
                ev = "FACE"
            elif delta >= float(t["depth_recess_mm"]):
                ev = "RECESS"
            else:
                ev = "UNCLEAR"
        s["depth_evidence"] = ev
        s["depth_delta_mm"] = None if delta is None else round(delta, 1)
        counts[ev] += 1
    disagree = 0
    for s in sites:
        ev = s.get("depth_evidence")
        st = s["state"]
        conflict = (st == "OPEN" and ev == "FACE") or (st == "PLUGGED" and ev == "RECESS")
        s["depth_conflict"] = bool(conflict)
        if conflict:
            disagree += 1
            if t["depth_to_state"]:
                s["state_before_depth"] = st
                s["state"] = "REVIEW"
    med_px = float(np.median(px_per_tube)) if px_per_tube else 0.0
    return {"counts": counts, "conflicts": disagree, "depth_px_per_opening": round(med_px, 1),
            "face_tol_mm": round(face_tol, 1), "applied_to_state": bool(t["depth_to_state"])}


def depth_check(summary):
    c = summary["counts"]
    judged = c["FACE"] + c["RECESS"] + c["UNCLEAR"]
    if judged == 0:
        return {"id": "depth_states", "label": "Depth at openings", "status": "SKIP",
                "detail": f"Depth too coarse: about {summary['depth_px_per_opening']:.0f} depth pixels across an "
                          "opening. Move closer or use a higher-resolution depth camera."}
    st = "WARN" if summary["conflicts"] else "PASS"
    detail = (f"{c['FACE']} show a surface at the face, {c['RECESS']} are open or deep, {c['UNCLEAR']} unclear, "
              f"{c['LOW_RES']} too small. {summary['conflicts']} disagree with the image "
              f"({'marked REVIEW' if summary['applied_to_state'] else 'shown, not applied'}). "
              f"~{summary['depth_px_per_opening']:.0f} depth px across an opening.")
    return {"id": "depth_states", "label": "Depth at openings", "status": st, "detail": detail}


# --------------------------------------------------------------------------
# Lattice matching between two maps
# --------------------------------------------------------------------------
def _sym_ops(pattern):
    """Integer lattice symmetries as 2x2 matrices acting on (i, j)."""
    if pattern == TRIANGULAR:
        R = np.array([[0, -1], [1, 1]])            # 60 deg: (i,j) -> (-j, i+j)
        M = np.array([[1, 1], [0, -1]])            # mirror: (i,j) -> (i+j, -j)
        n = 6
    else:
        R = np.array([[0, -1], [1, 0]])
        M = np.array([[1, 0], [0, -1]])
        n = 4
    ops = []
    A = np.eye(2, dtype=int)
    for _ in range(n):
        ops.append((A.copy(), False))
        ops.append((A @ M, True))
        A = R @ A
    return ops


def match_lattices(map_a, map_b, max_rot_deg=35.0, min_overlap=0.7):
    """Match two maps that may each have several drilled fields. Tries the largest fields of each
    map against each other and returns a continuous transform A_l = op @ B_l + t on (lx, ly)."""
    def fields(m):
        cnt = {}
        for s in m["sites"]:
            if s["state"] in PRESENT:
                cnt[s.get("field", 0)] = cnt.get(s.get("field", 0), 0) + 1
        return sorted(cnt, key=lambda f: -cnt[f])[:2], cnt

    fa_list, _ = fields(map_a)
    fb_list, _ = fields(map_b)
    offs_a = {fd["field"]: np.array(fd["offset"]) for fd in map_a["lattice"].get("fields", [{"field": 0, "offset": [0, 0]}])}
    offs_b = {fd["field"]: np.array(fd["offset"]) for fd in map_b["lattice"].get("fields", [{"field": 0, "offset": [0, 0]}])}
    best = None
    reasons = []
    for fa in fa_list:
        for fb in fb_list:
            r = _match_field(map_a, map_b, fa, fb, max_rot_deg, min_overlap)
            if not r.get("valid"):
                reasons.append(r.get("reason"))
                continue
            if best is None or r["agree"] > best["agree"]:
                best = r
    if best is None:
        return {"valid": False, "reason": reasons[0] if reasons else "no tubes to match"}
    op = np.array(best["op"])
    t_cont = np.array(best["t"]) + offs_a.get(best["field_a"], np.zeros(2)) - op @ offs_b.get(best["field_b"], np.zeros(2))
    best["t_cont"] = t_cont.tolist()
    best.pop("agree", None)
    return best


def _match_field(map_a, map_b, fa, fb, max_rot_deg=35.0, min_overlap=0.7):
    """Find (op, t) with A_ij = op @ B_ij + t for one field of each map.

    Candidates come from voting on tube-present cells for every lattice symmetry. Each is then
    scored on full state agreement (open / plugged / no tube), because plugs and removed tubes
    break the symmetry of a round, evenly drilled sheet. Image orientation is a tie-breaker.
    """
    pat = map_a["lattice"]["pattern"]
    if map_b["lattice"]["pattern"] != pat:
        return {"valid": False, "reason": "the two cameras disagree on the pattern"}
    sa = [s for s in map_a["sites"] if s.get("field", 0) == fa]
    sb = [s for s in map_b["sites"] if s.get("field", 0) == fb]
    A = np.array([[s["i"], s["j"]] for s in sa if s["state"] in PRESENT], int)
    B = np.array([[s["i"], s["j"]] for s in sb if s["state"] in PRESENT], int)
    if len(A) < 20 or len(B) < 20:
        return {"valid": False, "reason": "too few tubes to match"}
    state_a = {(s["i"], s["j"]): s["state"] for s in sa}
    B_all = np.array([[s["i"], s["j"]] for s in sb], int)
    B_states = [s["state"] for s in sb]
    HA = np.array(map_a["lattice"]["H_ideal_to_px"])
    HB = np.array(map_b["lattice"]["H_ideal_to_px"])
    Bu = lattice_basis(pat)
    cA = np.median(A, axis=0).round()
    cB = np.median(B, axis=0).round()

    def steps(H, ij0, e1, e2):
        p0 = lattice._apply_h(H, ij0[None, :] @ Bu)[0]
        p1 = lattice._apply_h(H, (ij0 + e1)[None, :] @ Bu)[0]
        p2 = lattice._apply_h(H, (ij0 + e2)[None, :] @ Bu)[0]
        return np.column_stack([p1 - p0, p2 - p0])

    VB = steps(HB, cB, np.array([1.0, 0]), np.array([0, 1.0]))
    OFF = 1 << 20
    cands = []
    for op, _ in _sym_ops(pat):
        Bt = B @ op.T
        key = ((A[:, None, 0] - Bt[None, :, 0]) + OFF) * (2 * OFF) + ((A[:, None, 1] - Bt[None, :, 1]) + OFF)
        vals, cnt = np.unique(key.ravel(), return_counts=True)
        for k in np.argsort(-cnt)[:3]:
            t = np.array([vals[k] // (2 * OFF) - OFF, vals[k] % (2 * OFF) - OFF], int)
            moved = B_all @ op.T + t
            agree = sum(1 for (i, j), st in zip(moved, B_states) if state_a.get((int(i), int(j))) == st)
            both = sum(1 for (i, j) in moved if (int(i), int(j)) in state_a)
            VA = steps(HA, cA, (op @ [1, 0]).astype(float), (op @ [0, 1]).astype(float))
            L = VA @ np.linalg.inv(VB)
            U, _, Vt = np.linalg.svd(L)
            Rm = U @ Vt
            mirrored = bool(np.linalg.det(L) < 0)
            rot = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0])) if not mirrored else \
                math.degrees(math.atan2(Rm[1, 0], -Rm[0, 0]))
            cands.append({"op": op, "t": t, "present": int(cnt[k]), "agree": agree, "both": both,
                          "mirrored": mirrored, "rot_deg": rot})
    cands.sort(key=lambda r: (-r["agree"], r["mirrored"], abs(r["rot_deg"])))
    best = cands[0]
    rivals = [r for r in cands[1:] if r["agree"] >= best["agree"] - max(2, 0.004 * best["agree"])
              and (r["op"] != best["op"]).any()]
    if rivals:
        prior_ok = [r for r in [best] + rivals if not r["mirrored"] and abs(r["rot_deg"]) <= max_rot_deg]
        if len(prior_ok) == 1:
            best = prior_ok[0]
        else:
            return {"valid": False, "reason": "the tube pattern lines up in more than one way "
                                              "(sheet too symmetric and no plugs or gaps to tell them apart)"}
    overlap = best["present"] / min(len(A), len(B))
    if overlap < min_overlap:
        return {"valid": False, "reason": f"only {100 * overlap:.0f}% of tubes line up between the cameras"}
    return {"valid": True, "op": best["op"].tolist(), "t": best["t"].tolist(), "overlap": round(overlap, 3),
            "field_a": fa, "field_b": fb, "agree": best["agree"],
            "matched": best["present"], "state_agreement": round(best["agree"] / max(1, best["both"]), 3),
            "mirrored": best["mirrored"], "rotation_deg": round(best["rot_deg"], 1),
            "margin": best["agree"] - (cands[1]["agree"] if best is cands[0] and len(cands) > 1 else 0)}


def merge_companion(m, cm, companion_name):
    """Bring a companion camera's pitch check and per-site depth evidence into map m."""
    info = {"capture_id": cm.get("capture_id"), "device": companion_name, "ready": cm.get("ready")}
    pc = next((c for c in cm.get("checks", []) if c["id"] == "pitch"), None)
    if pc and pc["status"] in ("PASS", "FAIL"):
        m["checks"].append({"id": "pitch_companion", "label": f"Pitch vs depth ({companion_name})",
                            "status": pc["status"], "detail": pc["detail"],
                            "deviation_pct": pc.get("deviation_pct"), "depth_pitch_mm": pc.get("depth_pitch_mm")})
        own = next((c for c in m["checks"] if c["id"] == "pitch"), None)
        if own and own["status"] in ("PASS", "FAIL", "WARN"):
            own["status_original"] = own["status"]
            own["status"] = "INFO"
            own["detail"] += f" Superseded by {companion_name}."
        info["pitch_status"] = pc["status"]
    pat_c = next((c for c in cm.get("checks", []) if c["id"] == "pattern"), None)
    if pat_c and pat_c["status"] == "FAIL":
        m["checks"].append({"id": "pattern_companion", "label": f"Pattern ({companion_name})", "status": "FAIL",
                            "detail": pat_c["detail"]})
    match = match_lattices(m, cm)
    info["match"] = {k: v for k, v in match.items() if k != "op"}
    if match.get("valid"):
        op = np.array(match["op"], float)
        t = np.array(match["t_cont"], float)
        Bl = np.array([[c.get("lx", c["i"]), c.get("ly", c["j"])] for c in cm["sites"]], float)
        Bl = Bl @ op.T + t
        moved = 0
        state_disagree = 0
        for s in m["sites"]:
            d = np.hypot(Bl[:, 0] - s.get("lx", s["i"]), Bl[:, 1] - s.get("ly", s["j"]))
            k = int(np.argmin(d))
            if d[k] > 0.3:
                continue
            c = cm["sites"][k]
            if c.get("depth_evidence"):
                s["depth_evidence"] = c["depth_evidence"]
                s["depth_delta_mm"] = c.get("depth_delta_mm")
                s["depth_source"] = companion_name
                s["depth_conflict"] = (s["state"] == "OPEN" and c["depth_evidence"] == "FACE") or \
                                      (s["state"] == "PLUGGED" and c["depth_evidence"] == "RECESS")
                moved += 1
            s["companion_state"] = c["state"]
            if (c["state"] in PRESENT) != (s["state"] in PRESENT) or \
                    (c["state"] == "PLUGGED") != (s["state"] == "PLUGGED"):
                state_disagree += 1
        info["sites_with_depth"] = moved
        own = next((c for c in m["checks"] if c["id"] == "depth_states"), None)
        if own and moved:
            own["status_original"] = own["status"]
            own["status"] = "INFO"
            own["detail"] = f"This camera's own depth: {own['detail']} Replaced by {companion_name} per tube."
        info["state_disagreements"] = state_disagree
        conflicts = sum(1 for s in m["sites"] if s.get("depth_conflict"))
        m["checks"].append({"id": "companion_match", "label": f"Matched to {companion_name}",
                            "status": "PASS" if state_disagree <= max(3, 0.02 * len(m["sites"])) else "WARN",
                            "detail": f"{match['matched']} tubes line up ({100 * match['overlap']:.0f}%), rotation "
                                      f"{match['rotation_deg']:+.0f}°{', mirrored image' if match['mirrored'] else ''}. "
                                      f"{moved} sites got depth evidence; {conflicts} disagree with this image; "
                                      f"{state_disagree} sites classified differently by the two cameras."})
    else:
        m["checks"].append({"id": "companion_match", "label": f"Matched to {companion_name}", "status": "WARN",
                            "detail": f"Per-tube depth not transferred: {match['reason']}. Pitch check still applies."})
    m["companion"] = info
    return m

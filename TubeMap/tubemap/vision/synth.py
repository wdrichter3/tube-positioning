"""Synthetic tubesheet renderer.

Used by the Simulator device and the test-suite. Renders a colour image and an
aligned depth map of a drilled tubesheet seen by a pinhole camera with optional
lens distortion, tilt and noise, and returns the ground truth for every site.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import MM_PER_IN, SQUARE, TRIANGULAR, lattice_basis


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


DEFAULTS = dict(
    width=1248, height=936, hfov_deg=66.0,
    pattern=TRIANGULAR, pitch_in=1.0, tube_id_in=0.62,
    sheet_diameter_mm=700.0, flange_diameter_mm=860.0,
    distance_mm=1100.0, tilt_deg=12.0, yaw_deg=-5.0, lattice_rot_deg=8.0,
    lane_width_rows=1.0, plugged_fraction=0.03, removed_fraction=0.01,
    k1=-0.08, k2=0.02, noise=4.0, depth_noise_frac=0.004, depth_scale=0.5, depth_smooth_px=0,
    mirror=False, roll_deg=0.0, field_shift=None,
    seed=7,
)


def render(**kw) -> dict:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in kw.items() if v is not None})
    rng = np.random.default_rng(int(p["seed"]))
    W, H = int(p["width"]), int(p["height"])
    f = (W / 2.0) / math.tan(math.radians(p["hfov_deg"]) / 2.0)
    K = np.array([[f, 0, W / 2.0 - 0.5], [0, f, H / 2.0 - 0.5], [0, 0, 1.0]])
    dist = np.array([p["k1"], p["k2"], 0.0, 0.0, 0.0])

    R = _rot("z", p["roll_deg"]) @ _rot("x", p["tilt_deg"]) @ _rot("y", p["yaw_deg"])
    t = np.array([0.0, 0.0, float(p["distance_mm"])])
    Hs2i = K @ np.column_stack([R[:, 0], R[:, 1], t])     # sheet(x,y,1) -> image
    Hi2s = np.linalg.inv(Hs2i)

    pitch = p["pitch_in"] * MM_PER_IN
    r_hole = p["tube_id_in"] * MM_PER_IN / 2.0
    pattern = p["pattern"]
    B = lattice_basis(pattern) * pitch
    rot = _rot("z", p["lattice_rot_deg"])[:2, :2]
    Br = B @ rot.T                                     # rows = rotated basis vectors
    Binv = np.linalg.inv(Br.T)

    # ---- enumerate sites
    R_sheet = p["sheet_diameter_mm"] / 2.0
    n = int(R_sheet / pitch * 2.4) + 3
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    ij = np.stack([ii.ravel(), jj.ravel()], 1)
    shift = np.zeros(2) if p["field_shift"] is None else np.asarray(p["field_shift"], float)
    fld = (ij[:, 1] > 0).astype(int) if p["field_shift"] is not None else np.zeros(len(ij), int)
    xy = (ij + shift * fld[:, None]) @ Br          # upper field shifted by a sub-pitch offset
    inside = np.linalg.norm(xy, axis=1) < (R_sheet - 0.75 * pitch)
    ij, xy, fld = ij[inside], xy[inside], fld[inside]
    # pass-partition lane: remove rows near the sheet centre line (lattice row j)
    lane = np.zeros(len(ij), bool)
    if p["lane_width_rows"] > 0:
        lane = np.abs(ij[:, 1]) < p["lane_width_rows"] / 2.0 + 1e-9
    state = np.array(["OPEN"] * len(ij), dtype=object)
    state[lane] = "NO_TUBE"
    live = np.where(~lane)[0]
    nplug = int(round(p["plugged_fraction"] * len(live)))
    nrem = int(round(p["removed_fraction"] * len(live)))
    pick = rng.permutation(live)
    state[pick[:nplug]] = "PLUGGED"
    state[pick[nplug:nplug + nrem]] = "NO_TUBE"
    site_lookup = {(int(f), int(a), int(b)): s for f, (a, b), s in zip(fld, ij, state)}

    # ---- per-pixel sheet coordinates (ideal, undistorted image)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)])
    s = Hi2s @ pts
    sx, sy = s[0] / s[2], s[1] / s[2]
    rr = np.hypot(sx, sy)
    best_d = np.full(W * H, np.inf)
    best_state = np.zeros(W * H, np.int8)   # 0 none, 1 open, 2 plugged
    code = {"OPEN": 1, "PLUGGED": 2, "NO_TUBE": 0}
    for field_id in sorted(set(int(x) for x in fld)):
        off = shift * field_id
        L = Binv @ np.stack([sx - (off @ Br)[0], sy - (off @ Br)[1]])
        fi, fj = np.floor(L[0]), np.floor(L[1])
        lut = np.zeros((4 * n + 3, 4 * n + 3), np.int8)
        for (fid, a, b), st in site_lookup.items():
            if fid == field_id:
                lut[a + 2 * n + 1, b + 2 * n + 1] = code[st]
        for di in (0, 1):
            for dj in (0, 1):
                ci, cj = fi + di, fj + dj
                cx = (ci + off[0]) * Br[0, 0] + (cj + off[1]) * Br[1, 0]
                cy = (ci + off[0]) * Br[0, 1] + (cj + off[1]) * Br[1, 1]
                d = np.hypot(sx - cx, sy - cy)
                ai = np.clip(ci.astype(int) + 2 * n + 1, 0, lut.shape[0] - 1)
                aj = np.clip(cj.astype(int) + 2 * n + 1, 0, lut.shape[1] - 1)
                st = lut[ai, aj]
                better = (d < best_d) & (st > 0)
                best_d = np.where(better, d, best_d)
                best_state = np.where(better, st, best_state)

    mm_per_px = float(p["distance_mm"]) / f
    edge = 0.8 * mm_per_px

    def smooth_in(d, r):
        return np.clip((r - d) / edge + 0.5, 0.0, 1.0)

    # base metal with brushed texture + lighting gradient
    tex = rng.normal(0, 1, (H // 4 + 1, W // 4 + 1)).astype(np.float32)
    tex = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 1.2), (W, H))[:H, :W].ravel()
    light = 0.85 + 0.25 * (sx / (R_sheet + 1e-9)) * 0.5 - 0.1 * (sy / (R_sheet + 1e-9))
    metal = (150.0 + 10.0 * tex) * light
    img = np.full(W * H, 28.0)                               # dark floor
    flange = smooth_in(rr, p["flange_diameter_mm"] / 2.0)
    img = img * (1 - flange) + (125.0 + 8 * tex) * light * flange
    sheet = smooth_in(rr, R_sheet)
    img = img * (1 - sheet) + metal * sheet
    hole = smooth_in(best_d, r_hole) * sheet
    is_open = best_state == 1
    is_plug = best_state == 2
    img = np.where(is_open, img * (1 - hole) + 25.0 * hole, img)
    ring = np.clip(1.0 - np.abs(best_d - r_hole) / (1.2 * mm_per_px + 0.4), 0, 1) * sheet
    img = np.where(is_plug, img * (1 - hole) + 200.0 * hole, img)
    img = np.where(is_plug, img * (1 - 0.6 * ring) + 70.0 * 0.6 * ring, img)
    img = img.reshape(H, W).astype(np.float32)

    # depth (camera Z in mm) for the ideal image
    denom = (R[:, 2] @ np.array([0, 0, 1.0]))
    Zc = (np.column_stack([R[:, 0], R[:, 1], t]) @ np.stack([sx, sy, np.ones_like(sx)]))[2]
    depth = Zc.copy()
    depth = np.where(is_open & (hole > 0.5), depth + 400.0, depth)
    depth = np.where(sheet > 0.5, depth, depth + 250.0 * (1 - flange))   # floor further away
    depth = depth.reshape(H, W).astype(np.float32)
    del denom

    # ---- lens distortion: sample ideal image at undistorted locations
    grid = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(grid, K, dist, P=K).reshape(H, W, 2).astype(np.float32)
    img_d = cv2.remap(img, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dep_d = cv2.remap(depth, und[..., 0], und[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    img_d = img_d + rng.normal(0, p["noise"], img_d.shape).astype(np.float32)
    gray = np.clip(img_d, 0, 255).astype(np.uint8)
    color = cv2.merge([gray, np.clip(gray.astype(int) + 3, 0, 255).astype(np.uint8), gray])

    ds = float(p["depth_scale"])
    dw, dh = max(16, int(round(W * ds))), max(16, int(round(H * ds)))
    dep_small = cv2.resize(dep_d, (dw, dh), interpolation=cv2.INTER_NEAREST)
    if p["depth_smooth_px"] and p["depth_smooth_px"] >= 2:      # structured-light style edge smoothing
        k = int(p["depth_smooth_px"]) | 1
        dep_small = cv2.medianBlur(dep_small.astype(np.float32), min(k, 5)) if k <= 5 else cv2.blur(dep_small, (k, k))
    dep_small = dep_small * (1 + rng.normal(0, p["depth_noise_frac"], dep_small.shape))
    invalid = rng.random(dep_small.shape) < 0.03
    dep_small[invalid] = 0
    depth_u16 = np.clip(dep_small, 0, 65535).astype(np.uint16)

    # ---- ground truth in (distorted) pixel coordinates
    xyz = np.column_stack([xy, np.zeros(len(xy))])
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(xyz, rvec, t, K, dist)
    uv = uv.reshape(-1, 2)
    truth = []
    for k in range(len(ij)):
        truth.append({"i": int(ij[k, 0]), "j": int(ij[k, 1]), "field": int(fld[k]),
                      "x_mm": float(xy[k, 0]), "y_mm": float(xy[k, 1]),
                      "u": float(uv[k, 0]), "v": float(uv[k, 1]), "state": str(state[k])})
    if p["mirror"]:
        color = color[:, ::-1].copy()
        depth_u16 = depth_u16[:, ::-1].copy()
        for tr in truth:
            tr["u"] = W - 1 - tr["u"]
    intr = {"width": W, "height": H, "fx": float(f), "fy": float(f),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": dist.tolist(), "model": "opencv"}
    return {"color": color, "depth": depth_u16, "intrinsics": intr, "truth": truth,
            "params": p, "pitch_mm": pitch, "tube_id_mm": 2 * r_hole,
            "plane": {"R": R.tolist(), "t": t.tolist()}}


if __name__ == "__main__":
    import time
    t0 = time.time()
    out = render()
    print("render s", round(time.time() - t0, 2), len(out["truth"]))
    cv2.imwrite("/tmp/synth.png", out["color"])
    _ = SQUARE

"""Camera models, undistortion and depth-plane metric estimates."""
from __future__ import annotations

import math

import cv2
import numpy as np


def K_of(intr: dict) -> np.ndarray:
    return np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])


def dist_of(intr: dict) -> np.ndarray:
    d = intr.get("dist") or []
    return np.asarray(d, np.float64) if len(d) else np.zeros(5)


def scale_intrinsics(intr: dict, width: int, height: int) -> dict:
    sx, sy = width / float(intr["width"]), height / float(intr["height"])
    out = dict(intr)
    out.update(width=width, height=height, fx=intr["fx"] * sx, fy=intr["fy"] * sy,
               cx=(intr["cx"] + 0.5) * sx - 0.5, cy=(intr["cy"] + 0.5) * sy - 0.5)
    return out


def valid_intrinsics(intr) -> bool:
    try:
        return all(float(intr[k]) > 0 for k in ("fx", "fy")) and float(intr["width"]) > 0
    except Exception:
        return False


def undistort_px(pts, intr):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if not valid_intrinsics(intr) or len(pts) == 0:
        return pts.copy()
    K = K_of(intr)
    return cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist_of(intr), P=K).reshape(-1, 2)


def distort_px(pts, intr):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if not valid_intrinsics(intr) or len(pts) == 0:
        return pts.copy()
    K = K_of(intr)
    xn = (pts[:, 0] - K[0, 2]) / K[0, 0]
    yn = (pts[:, 1] - K[1, 2]) / K[1, 1]
    obj = np.column_stack([xn, yn, np.ones_like(xn)])
    uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist_of(intr))
    return uv.reshape(-1, 2)


def rays(pts_und, intr):
    """Unit-depth rays (x, y, 1) for undistorted pixel coordinates."""
    K = K_of(intr)
    p = np.asarray(pts_und, np.float64).reshape(-1, 2)
    return np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(p))])


# --------------------------------------------------------------------------
# Plane fitting
# --------------------------------------------------------------------------
def fit_plane_svd(points):
    p = np.asarray(points, np.float64)
    c = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    n = vh[-1] / max(np.linalg.norm(vh[-1]), 1e-12)
    if n[2] < 0:
        n = -n
    d = -float(n @ c)
    return n, d


def ransac_plane(points, threshold, iterations=200, seed=7):
    p = np.asarray(points, np.float64)
    if len(p) < 50:
        raise ValueError("not enough depth points for a plane")
    rng = np.random.default_rng(seed)
    best, best_mask = -1, None
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        mask = np.abs((p - a) @ n) <= threshold
        cnt = int(mask.sum())
        if cnt > best:
            best, best_mask = cnt, mask
    n, d = fit_plane_svd(p[best_mask])
    res = np.abs(p @ n + d)
    mask = res <= threshold
    n, d = fit_plane_svd(p[mask])
    res = p[mask] @ n + d
    return n, d, mask, float(np.sqrt(np.mean(res ** 2)))


def depth_plane(depth_mm: np.ndarray, color_intr: dict, sample_mask: np.ndarray | None = None,
                max_samples: int = 30000, seed: int = 3) -> dict:
    """Fit the sheet plane from an aligned depth map.

    depth_mm covers the same field of view as the colour image (aligned); it may
    be a different resolution. sample_mask is in depth resolution (bool).
    """
    h, w = depth_mm.shape[:2]
    intr = scale_intrinsics(color_intr, w, h)
    valid = depth_mm > 0
    if sample_mask is not None:
        valid &= sample_mask
    vs, us = np.nonzero(valid)
    if len(us) < 200:
        return {"valid": False, "reason": "too few valid depth pixels in the region"}
    rng = np.random.default_rng(seed)
    if len(us) > max_samples:
        sel = rng.choice(len(us), max_samples, replace=False)
        us, vs = us[sel], vs[sel]
    z = depth_mm[vs, us].astype(np.float64)
    und = undistort_px(np.column_stack([us, vs]).astype(np.float64), intr)
    xyz = rays(und, intr) * z[:, None]
    zmed = float(np.median(z))
    thr = max(3.0, 0.006 * zmed)
    try:
        n, d, mask, rms = ransac_plane(xyz, thr)
    except ValueError as exc:
        return {"valid": False, "reason": str(exc)}
    tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
    return {"valid": True, "normal": n.tolist(), "d": float(d), "distance_mm": float(abs(d)),
            "rms_mm": rms, "inlier_fraction": float(mask.mean()), "samples": int(len(z)),
            "tilt_deg": tilt, "median_depth_mm": zmed}


def intersect_plane(pts_und, intr, plane) -> np.ndarray:
    n = np.asarray(plane["normal"], np.float64)
    d = float(plane["d"])
    r = rays(pts_und, intr)
    t = -d / (r @ n)
    return r * t[:, None]


def plane_frame(plane):
    """Orthonormal in-plane axes (e1 ~ image x, e2 ~ image y)."""
    n = np.asarray(plane["normal"], np.float64)
    e1 = np.array([1.0, 0.0, 0.0]) - n * n[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    if e2[1] < 0:
        e2 = -e2
    return e1, e2


def to_plane_2d(xyz, plane):
    e1, e2 = plane_frame(plane)
    return np.column_stack([xyz @ e1, xyz @ e2])

"""Tubesheet geometry: lattice definitions, standard sizes and prior-free discovery.

Everything here works on 2-D points. Units are whatever the caller supplies
(pixels or millimetres); functions that talk about standards expect mm.
"""
from __future__ import annotations

import math

import numpy as np

MM_PER_IN = 25.4
TRIANGULAR = "TRIANGULAR"
SQUARE = "SQUARE"
PATTERNS = (TRIANGULAR, SQUARE)

# Common nominal values (inches). Extend as job history grows.
STD_TUBE_OD_IN = [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0]
BWG_WALL_IN = {10: .134, 11: .120, 12: .109, 13: .095, 14: .083, 15: .072,
               16: .065, 17: .058, 18: .049, 19: .042, 20: .035, 22: .028}
STD_PITCH_IN = [0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.125, 1.25, 1.3125,
                1.375, 1.4375, 1.5, 1.5625, 1.875, 2.5]


# --------------------------------------------------------------------------
# Lattice definitions
# --------------------------------------------------------------------------
def lattice_basis(pattern: str) -> np.ndarray:
    """Unit-pitch basis vectors as rows: point = i*b1 + j*b2."""
    if pattern == TRIANGULAR:
        return np.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    if pattern == SQUARE:
        return np.array([[1.0, 0.0], [0.0, 1.0]])
    raise ValueError(f"unknown pattern {pattern!r}")


def neighbor_offsets(pattern: str):
    if pattern == TRIANGULAR:
        return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    return [(1, 0), (-1, 0), (0, 1), (0, -1)]


def ideal_xy(ij, pattern: str, pitch: float = 1.0) -> np.ndarray:
    ij = np.asarray(ij, np.float64).reshape(-1, 2)
    return (ij @ lattice_basis(pattern)) * float(pitch)


def row_spacing(pattern: str, pitch: float) -> float:
    return pitch * (math.sqrt(3.0) / 2.0 if pattern == TRIANGULAR else 1.0)


# --------------------------------------------------------------------------
# Nearest-neighbour helpers (numpy only; fine for a few thousand points)
# --------------------------------------------------------------------------
def pairwise_within(points: np.ndarray, rmax: float, chunk: int = 512):
    """Return index pairs (i<j) with distance <= rmax."""
    p = np.asarray(points, np.float64)
    out_i, out_j = [], []
    n = len(p)
    for s in range(0, n, chunk):
        blk = p[s:s + chunk]
        d2 = ((blk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        ii, jj = np.nonzero(d2 <= rmax * rmax)
        ii = ii + s
        keep = ii < jj
        out_i.append(ii[keep]); out_j.append(jj[keep])
    if not out_i:
        return np.zeros((0, 2), int)
    return np.stack([np.concatenate(out_i), np.concatenate(out_j)], axis=1)


def nearest_neighbor_dist(points: np.ndarray, chunk: int = 512) -> np.ndarray:
    p = np.asarray(points, np.float64)
    n = len(p)
    out = np.full(n, np.inf)
    for s in range(0, n, chunk):
        blk = p[s:s + chunk]
        d2 = ((blk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        idx = np.arange(s, min(s + chunk, n))
        d2[np.arange(len(blk)), idx] = np.inf
        out[s:s + chunk] = np.sqrt(d2.min(axis=1))
    return out


def robust_pitch(points: np.ndarray) -> float:
    nn = nearest_neighbor_dist(points)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        raise ValueError("no neighbours")
    lo, hi = np.percentile(nn, [10, 90])
    core = nn[(nn >= lo * 0.9) & (nn <= hi * 1.1)]
    return float(np.median(core if len(core) else nn))


def neighbor_vectors(points: np.ndarray, pitch: float, lo=0.75, hi=1.3):
    pairs = pairwise_within(points, pitch * hi)
    if len(pairs) == 0:
        return np.zeros((0, 2)), pairs
    v = points[pairs[:, 1]] - points[pairs[:, 0]]
    L = np.linalg.norm(v, axis=1)
    keep = (L > pitch * lo) & (L < pitch * hi)
    return v[keep], pairs[keep]


def classify_pattern(points: np.ndarray, pitch: float):
    """Perspective-tolerant: count first-shell neighbours relative to each point's own
    nearest-neighbour distance (6 for triangular, 4 for square), plus angular symmetry."""
    p = np.asarray(points, np.float64)
    vecs, _ = neighbor_vectors(p, pitch)
    if len(vecs) < 10:
        return None, {"reason": "too few neighbour vectors"}
    nn = nearest_neighbor_dist(p)
    pairs = pairwise_within(p, 1.25 * float(np.percentile(nn[np.isfinite(nn)], 95)))
    counts = np.zeros(len(p), int)
    if len(pairs):
        d = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
        for a, b, dd in zip(pairs[:, 0], pairs[:, 1], d):
            if dd <= 1.2 * nn[a]:
                counts[a] += 1
            if dd <= 1.2 * nn[b]:
                counts[b] += 1
    interior = counts[counts > 0]
    median_count = float(np.median(interior)) if len(interior) else 0.0
    top = float(np.percentile(interior, 90)) if len(interior) else 0.0
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    z6 = float(abs(np.mean(np.exp(6j * ang))))
    z4 = float(abs(np.mean(np.exp(4j * ang))))
    if top >= 5.5 or (top >= 4.5 and z6 > z4):
        pattern = TRIANGULAR
    else:
        pattern = SQUARE
    strong, weak = (z6, z4) if pattern == TRIANGULAR else (z4, z6)
    conf = max(0.0, min(1.0, (strong - weak) / max(strong, 1e-9)))
    return pattern, {"median_neighbors": median_count, "p90_neighbors": top,
                     "harmonic6": z6, "harmonic4": z4, "confidence": conf}


def dominant_angle(vecs: np.ndarray, pattern: str) -> float:
    h = 6.0 if pattern == TRIANGULAR else 4.0
    period = 360.0 / h
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])
    return (math.degrees(np.angle(np.mean(np.exp(1j * h * ang)))) / h) % period


def basis_from_vectors(vecs: np.ndarray, pattern: str, base_deg: float, tol_deg=12.0):
    """Median vectors near base and base+60 (or +90)."""
    step = 60.0 if pattern == TRIANGULAR else 90.0
    out = []
    for k in range(2):
        t = math.radians(base_deg + step * k)
        u = np.array([math.cos(t), math.sin(t)])
        vv = np.where((vecs @ u)[:, None] < 0, -vecs, vecs)
        a = np.degrees(np.arctan2(vv[:, 1], vv[:, 0]) - t)
        a = (a + 180.0) % 360.0 - 180.0
        sel = vv[np.abs(a) < tol_deg]
        if len(sel) < 3:
            raise ValueError("not enough lattice vectors to form a basis")
        out.append(np.median(sel, axis=0))
    return np.array(out)


# --------------------------------------------------------------------------
# Standards
# --------------------------------------------------------------------------
def snap_pitch(pitch_mm: float, tol_frac: float = 0.003):
    pin = pitch_mm / MM_PER_IN
    best = min(STD_PITCH_IN, key=lambda t: abs(t - pin))
    frac = (pin - best) / best
    return {"nominal_in": best, "nominal_mm": best * MM_PER_IN,
            "delta_pct": 100.0 * frac, "accepted": abs(frac) <= tol_frac}


def tube_candidates(id_mm: float, pitch_mm: float | None = None, tol_mm: float = 0.15):
    id_in = id_mm / MM_PER_IN
    c = []
    for od in STD_TUBE_OD_IN:
        if pitch_mm is not None and od * 1.2 > pitch_mm / MM_PER_IN + 1e-6:
            continue
        for bwg, w in BWG_WALL_IN.items():
            nid = od - 2 * w
            d = (id_in - nid) * MM_PER_IN
            if abs(d) <= tol_mm:
                c.append({"od_in": od, "bwg": bwg, "nominal_id_in": round(nid, 4),
                          "nominal_id_mm": round(nid * MM_PER_IN, 3), "delta_mm": round(d, 3)})
    return sorted(c, key=lambda x: abs(x["delta_mm"]))


def standard_tables():
    tubes = []
    for od in STD_TUBE_OD_IN:
        for bwg, w in sorted(BWG_WALL_IN.items()):
            nid = od - 2 * w
            if nid > 0.25:
                tubes.append({"od_in": od, "bwg": bwg, "id_in": round(nid, 4),
                              "id_mm": round(nid * MM_PER_IN, 3)})
    return {"pitches_in": STD_PITCH_IN, "tubes": tubes}


# --------------------------------------------------------------------------
# Global lattice fit on metric points (discovery)
# --------------------------------------------------------------------------
def fit_global_lattice(points: np.ndarray, B: np.ndarray, iters: int = 3):
    p = np.asarray(points, np.float64)
    origin = p[np.argmin(np.linalg.norm(p - p.mean(0), axis=1))]
    A = None
    X = None
    for _ in range(iters):
        ij = np.rint(np.linalg.solve(B.T, (p - origin).T).T)
        A = np.hstack([ij, np.ones((len(ij), 1))])
        X, *_ = np.linalg.lstsq(A, p, rcond=None)
        B, origin = X[:2], X[2]
    pred = A @ X
    res = np.linalg.norm(p - pred, axis=1)
    dof = max(1, 2 * len(p) - 6)
    sigma2 = float(np.sum((p - pred) ** 2) / dof)
    cov = sigma2 * np.linalg.inv(A.T @ A)[:2, :2]
    return B, ij.astype(int), res, cov


def discover_geometry(points_mm, ids_mm=None, pattern_hint: str | None = None) -> dict:
    """Pattern, pitch, orientation (+ID statistics) from metric centres alone."""
    p = np.asarray(points_mm, np.float64)
    if len(p) < 20:
        raise ValueError("need at least ~20 tubes for discovery")
    p0 = robust_pitch(p)
    pattern, pinfo = classify_pattern(p, p0)
    if pattern_hint in PATTERNS:
        pattern = pattern_hint
    if pattern is None:
        raise ValueError("could not classify lattice pattern")
    vecs, _ = neighbor_vectors(p, p0)
    base = dominant_angle(vecs, pattern)
    B = basis_from_vectors(vecs, pattern, base)
    B, ij, res, cov = fit_global_lattice(p, B)
    L = np.linalg.norm(B, axis=1)
    pitch = float(L.mean())
    shear = math.degrees(math.acos(min(1.0, abs(float(np.dot(B[0], B[1]))) / (L[0] * L[1]))))
    out = {
        "n_tubes": int(len(p)), "pattern": pattern,
        "pattern_confidence": float(pinfo.get("confidence", 0.0)),
        "pitch": pitch, "pitch_stat_sigma": float(math.sqrt(max(0.0, float(np.trace(cov)) / 4))),
        "basis_lengths": [float(L[0]), float(L[1])],
        "basis_angle_deg": float(base),
        "basis_shear_deg": shear,
        "fit_rms": float(np.sqrt(np.mean(res ** 2))),
        "notes": [],
    }
    expect = 60.0 if pattern == TRIANGULAR else 90.0
    if abs(shear - expect) > 0.5:
        out["notes"].append(f"Lattice angle {shear:.2f} deg vs ideal {expect:.0f}: "
                            "perspective or scale error in the input frame.")
    aniso = abs(L[0] - L[1]) / pitch
    out["anisotropy_pct"] = 100.0 * aniso
    if aniso > 0.003:
        out["notes"].append(f"Lattice directions differ in scale by {100 * aniso:.2f}%: "
                            "tilt or distortion not fully removed.")
    if ids_mm is not None and len(ids_mm):
        d = np.asarray(ids_mm, np.float64)
        med = float(np.median(d))
        q1, q3 = np.percentile(d, [25, 75])
        out["id_median"] = med
        out["id_iqr"] = float(q3 - q1)
    return out

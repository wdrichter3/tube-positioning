"""Fit the drilling lattice to detected openings in the image.

Works in undistorted pixel coordinates. A local region-growing pass assigns
integer lattice indices (robust to perspective), then a plane-induced
homography from ideal lattice coordinates to pixels is refined iteratively so
that disconnected regions (across pass-partition lanes) join the same lattice.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .geometry import (PATTERNS, basis_from_vectors, classify_pattern, dominant_angle,
                       lattice_basis, neighbor_offsets, neighbor_vectors, robust_pitch)


class _Grid:
    def __init__(self, pts, cell):
        self.pts, self.cell = pts, float(cell)
        self.map = {}
        for k, (x, y) in enumerate(pts):
            self.map.setdefault((int(x // cell), int(y // cell)), []).append(k)

    def nearest(self, x, y, rmax, exclude=None):
        cx, cy = int(x // self.cell), int(y // self.cell)
        best, bd = None, rmax * rmax
        rng = int(math.ceil(rmax / self.cell))
        for gx in range(cx - rng, cx + rng + 1):
            for gy in range(cy - rng, cy + rng + 1):
                for k in self.map.get((gx, gy), ()):
                    if exclude is not None and k in exclude:
                        continue
                    dx, dy = self.pts[k, 0] - x, self.pts[k, 1] - y
                    d2 = dx * dx + dy * dy
                    if d2 < bd:
                        best, bd = k, d2
        return best


def _apply_h(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def local_pitch_px(H, ideal_pts):
    """Image-space length of one unit lattice step (mean of x/y) at each ideal point."""
    ideal_pts = np.asarray(ideal_pts, np.float64).reshape(-1, 2)
    p0 = _apply_h(H, ideal_pts)
    px = _apply_h(H, ideal_pts + [1.0, 0.0])
    py = _apply_h(H, ideal_pts + [0.0, 1.0])
    return 0.5 * (np.linalg.norm(px - p0, axis=1) + np.linalg.norm(py - p0, axis=1))


def _grow(points, pattern, B, p0):
    n = len(points)
    grid = _Grid(points, p0)
    offs = neighbor_offsets(pattern)
    # seed: well-connected point near the median position
    vecs_pairs = neighbor_vectors(points, p0)[1]
    counts = np.bincount(vecs_pairs.ravel(), minlength=n) if len(vecs_pairs) else np.zeros(n)
    full = len(offs)
    med = np.median(points, axis=0)
    order = np.argsort(np.linalg.norm(points - med, axis=1))
    seed = next((int(k) for k in order[:200] if counts[k] >= full), int(order[0]))

    ij_of = {seed: (0, 0)}
    pt_of = {(0, 0): seed}
    basis_of = {seed: B.copy()}
    q = deque([seed])
    while q:
        k = q.popleft()
        i, j = ij_of[k]
        # local basis from assigned lattice neighbours of k
        A, Y = [], []
        for di, dj in offs:
            m = pt_of.get((i + di, j + dj))
            if m is not None:
                A.append([di, dj]); Y.append(points[m] - points[k])
        Bk = basis_of.get(k, B)
        if len(A) >= 2:
            A = np.array(A, float); Y = np.array(Y, float)
            if abs(np.linalg.det(A.T @ A)) > 1e-6:
                Bk = np.linalg.lstsq(A, Y, rcond=None)[0]
        for di, dj in offs:
            key = (i + di, j + dj)
            if key in pt_of:
                continue
            pred = points[k] + di * Bk[0] + dj * Bk[1]
            step = float(np.linalg.norm(di * Bk[0] + dj * Bk[1]))
            m = grid.nearest(pred[0], pred[1], 0.3 * step, exclude=ij_of)
            if m is None:
                continue
            ij_of[m] = key
            pt_of[key] = m
            basis_of[m] = Bk
            q.append(m)
    return ij_of


def fit_image_lattice(points_px, pattern: str | None = None, boundary_px=None,
                      max_iter: int = 6) -> dict:
    """points_px: Nx2 undistorted pixel centres. boundary_px: optional polygon (Nx2)."""
    P = np.asarray(points_px, np.float64).reshape(-1, 2)
    if len(P) < 12:
        raise ValueError(f"only {len(P)} openings found; need at least 12")
    p0 = robust_pitch(P)
    detected_pattern, pinfo = classify_pattern(P, p0)
    if pattern not in PATTERNS:
        pattern = detected_pattern
    if pattern is None:
        raise ValueError("could not determine lattice pattern")
    vecs, _ = neighbor_vectors(P, p0)
    base = dominant_angle(vecs, pattern)
    B = basis_from_vectors(vecs, pattern, base)
    # express the pixel basis in the canonical lattice basis
    Bu = lattice_basis(pattern)
    # B rows are pixel vectors for unit steps along lattice basis vectors i and j
    ij_of = _grow(P, pattern, B, p0)
    if len(ij_of) < 10:
        raise ValueError("lattice growth failed; openings are not on a regular pattern")

    idx = np.array(list(ij_of.keys()))
    ij = np.array([ij_of[k] for k in idx], float)
    ideal = ij @ Bu
    H, inl = cv2.findHomography(ideal, P[idx], cv2.RANSAC, max(1.0, 0.12 * p0))
    if H is None:
        raise ValueError("lattice homography failed")

    Binv = np.linalg.inv(Bu.T)
    offsets = [np.zeros(2)]                      # field 0 defines the frame
    assign = {}
    for _round in range(3):                      # look for up to two extra fields
        H, assign = _refine(P, H, Bu, Binv, offsets, max_iter)
        new = _find_extra_field(P, H, Bu, Binv, offsets, assign)
        if new is None:
            break
        offsets.append(new)
    H, assign = _refine(P, H, Bu, Binv, offsets, max_iter)
    # drop extra fields that ended up tiny
    sizes = {f: 0 for f in range(len(offsets))}
    for (f, _, _) in assign:
        sizes[f] += 1
    keep = [f for f in range(len(offsets)) if f == 0 or sizes[f] >= 12]
    if len(keep) != len(offsets):
        remap = {f: n for n, f in enumerate(keep)}
        offsets = [offsets[f] for f in keep]
        assign = {(remap[f], i, j): k for (f, i, j), k in assign.items() if f in remap}
        H, assign = _refine(P, H, Bu, Binv, offsets, max_iter)

    keys = np.array(list(assign.keys()), int)
    pts_idx = np.array(list(assign.values()), int)
    cont = keys[:, 1:].astype(float) + np.array([offsets[f] for f in keys[:, 0]])
    pred = _apply_h(H, cont @ Bu)
    res = np.linalg.norm(pred - P[pts_idx], axis=1)
    lp_acc = local_pitch_px(H, cont @ Bu)

    accepted = [{"index": int(k), "field": int(f), "i": int(a), "j": int(b), "lx": float(c[0]), "ly": float(c[1]),
                 "residual_px": float(r)}
                for k, (f, a, b), c, r in zip(pts_idx, keys, cont, res)]
    taken = set(int(k) for k in pts_idx)
    rejected = []
    best_f, best_ij, best_res = _nearest_field(P, H, Bu, Binv, offsets)
    for k in range(len(P)):
        if k in taken:
            continue
        key = (int(best_f[k]), int(best_ij[k, 0]), int(best_ij[k, 1]))
        reason = "DUPLICATE" if key in assign else "OFF_LATTICE"
        rejected.append({"index": k, "reason": reason, "residual_px": float(best_res[k])})

    cells = []
    for f, off in enumerate(offsets):
        fk = keys[keys[:, 0] == f][:, 1:]
        if len(fk) < 3:
            continue
        for c in lattice_cells(H, pattern, fk, boundary_px, offset=off):
            c["field"] = f
            cells.append(c)
    fields = [{"field": f, "offset": [round(float(o[0]), 4), round(float(o[1]), 4)], "accepted": int(sizes.get(f, 0))}
              for f, o in enumerate(offsets)]
    for fd in fields:
        fd["accepted"] = int((keys[:, 0] == fd["field"]).sum())
    return {
        "pattern": pattern, "detected_pattern": detected_pattern, "pattern_info": pinfo,
        "H_ideal_to_px": H.tolist(), "accepted": accepted, "rejected": rejected,
        "cells": cells, "fields": fields,
        "pitch_px_median": float(np.median(lp_acc)),
        "residual_px_median": float(np.median(res)), "residual_px_p95": float(np.percentile(res, 95)),
        "residual_rel_median": float(np.median(res / lp_acc)),
    }


def _nearest_field(P, H, Bu, Binv, offsets):
    """For every point: best field, its integer cell and the pixel residual."""
    Hinv = np.linalg.inv(H)
    c = (Binv @ _apply_h(Hinv, P).T).T
    best_res = np.full(len(P), np.inf)
    best_f = np.zeros(len(P), int)
    best_ij = np.zeros((len(P), 2), int)
    for f, off in enumerate(offsets):
        ij = np.rint(c - off).astype(int)
        pred = _apply_h(H, (ij + off) @ Bu)
        r = np.linalg.norm(pred - P, axis=1)
        better = r < best_res
        best_res[better], best_f[better], best_ij[better] = r[better], f, ij[better]
    return best_f, best_ij, best_res


def _refine(P, H, Bu, Binv, offsets, max_iter):
    """Alternate: assign points to (field, i, j), refit one homography, re-estimate field offsets."""
    assign = {}
    for _ in range(max_iter):
        f_all, ij_all, res = _nearest_field(P, H, Bu, Binv, offsets)
        lp = local_pitch_px(H, (ij_all + np.array([offsets[f] for f in f_all])) @ Bu)
        tol = np.maximum(1.2, 0.22 * lp)
        new_assign = {}
        for k in np.argsort(res):
            if res[k] > tol[k]:
                continue
            key = (int(f_all[k]), int(ij_all[k, 0]), int(ij_all[k, 1]))
            if key not in new_assign:
                new_assign[key] = int(k)
        if len(new_assign) < 8:
            raise ValueError("lattice refinement lost too many points")
        keys = np.array(list(new_assign.keys()), int)
        idx = np.array(list(new_assign.values()))
        # re-estimate offsets of extra fields from their points (field 0 fixes the frame)
        Hinv = np.linalg.inv(H)
        c = (Binv @ _apply_h(Hinv, P[idx]).T).T
        for f in range(1, len(offsets)):
            m = keys[:, 0] == f
            if m.sum() >= 3:
                offsets[f] = np.mean(c[m] - keys[m, 1:], axis=0)
        cont = keys[:, 1:] + np.array([offsets[f] for f in keys[:, 0]])
        H_new, _ = cv2.findHomography(cont @ Bu, P[idx], cv2.LMEDS)
        if H_new is None:
            break
        converged = new_assign == assign
        H, assign = H_new, new_assign
        if converged:
            break
    return H, assign


def _find_extra_field(P, H, Bu, Binv, offsets, assign, min_points=12, tol=0.16):
    """Openings left over that share one sub-pitch offset form another drilled field
    (e.g. the far side of a pass-partition lane that isn't a whole number of rows wide)."""
    taken = set(assign.values())
    rest = np.array([k for k in range(len(P)) if k not in taken], int)
    if len(rest) < min_points:
        return None
    Hinv = np.linalg.inv(H)
    c = (Binv @ _apply_h(Hinv, P[rest]).T).T
    frac = c - np.rint(c)                                   # in [-0.5, 0.5]
    ang = 2 * np.pi * frac
    best, best_n = None, 0
    for k in range(len(rest)):                              # mode-seeking on the torus
        d = np.abs(((frac - frac[k]) + 0.5) % 1.0 - 0.5)
        n = int(np.sum(np.max(d, axis=1) < tol))
        if n > best_n:
            best, best_n = k, n
    if best is None or best_n < max(min_points, 0.04 * len(P)):
        return None
    d = np.abs(((frac - frac[best]) + 0.5) % 1.0 - 0.5)
    m = np.max(d, axis=1) < tol
    off = np.angle(np.mean(np.exp(1j * ang[m]), axis=0)) / (2 * np.pi)
    for o in offsets:                                      # must differ from existing fields
        dd = np.abs(((off - o) + 0.5) % 1.0 - 0.5)
        if np.max(dd) < 0.2:
            return None
    # the points must also be spatially coherent: most of them have a neighbour of the same field
    pts = P[rest[m]]
    if len(pts) >= 2:
        from .geometry import nearest_neighbor_dist
        nn = nearest_neighbor_dist(pts)
        lp = float(np.median(local_pitch_px(H, (np.rint(c[m]) @ Bu))))
        if np.mean(nn < 1.6 * lp) < 0.6:
            return None
    return np.asarray(off, float)


def lattice_cells(H, pattern, keys, boundary_px=None, margin=0.35, offset=None):
    """All lattice cells of one field inside the boundary (ROI polygon if given, else hull of accepted)."""
    Bu = lattice_basis(pattern)
    off = np.zeros(2) if offset is None else np.asarray(offset, float)
    keys = np.asarray(keys, int)
    lo, hi = keys.min(0) - 3, keys.max(0) + 3
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
    grid = np.stack([ii.ravel(), jj.ravel()], 1)
    ideal = (grid + off) @ Bu
    px = _apply_h(H, ideal)
    hull = cv2.convexHull(((keys + off) @ Bu).astype(np.float32))
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -margin
                       for x, y in ideal])
    if boundary_px is not None and len(boundary_px) >= 3:
        poly = np.asarray(boundary_px, np.float32).reshape(-1, 1, 2)
        lp = local_pitch_px(H, ideal)
        in_roi = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= 0.3 * l
                           for (x, y), l in zip(px, lp)])
        inside = inside & in_roi
    return [{"i": int(a), "j": int(b), "lx": float(a + off[0]), "ly": float(b + off[1]), "u": float(x), "v": float(y)}
            for (a, b), (x, y), ok in zip(grid, px, inside) if ok]

"""Tube-opening candidate detection.

Dark, roughly circular openings are enhanced with a morphological black-hat
filter sized to the expected opening, thresholded, and filtered by size,
circularity and contrast. Parameters live in a detector *profile* that the
learning module tunes from labelled images.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULT_PARAMS = {
    "clahe_clip": 2.0,          # local contrast equalisation strength
    "blackhat_scale": 1.8,      # kernel size relative to expected opening diameter
    "thresh_offset": 0.0,       # relative adjustment of the Otsu threshold (-0.5..0.5)
    "min_circularity": 0.55,    # 4*pi*A/P^2 (ellipses from tilt still pass)
    "size_tol": 0.45,           # accepted equivalent-diameter band around the mode
    "min_diameter_px": 5.0,
    "max_diameter_px": 250.0,
}


def roi_mask(shape, roi):
    """roi: list of [x, y] in normalised image coords (0..1) or None."""
    h, w = shape[:2]
    if not roi or len(roi) < 3:
        return np.full((h, w), 255, np.uint8)
    pts = np.array([[float(x) * (w - 1), float(y) * (h - 1)] for x, y in roi], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m


def _odd(x):
    x = int(round(x))
    return max(3, x | 1)


def estimate_opening_diameter(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """Rough scale estimate: median diameter of round dark blobs (adaptive threshold)."""
    h, w = gray.shape
    scale = 1.0
    g = gray
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    diams = []
    for block in (31, 61, 121):
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 12)
        th = cv2.bitwise_and(th, mask)
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if a < 12 or bw < 4 or bh < 4:
                continue
            fill = a / float(bw * bh)
            asp = min(bw, bh) / max(bw, bh)
            if fill > 0.6 and asp > 0.6:
                diams.append(2.0 * math.sqrt(a / math.pi))
        if len(diams) >= 30:
            break
    if len(diams) < 10:
        return None
    d = np.array(diams)
    # mode of log-diameter histogram, robust to clutter
    hist, edges = np.histogram(np.log(d), bins=30)
    k = int(np.argmax(hist))
    mode = math.exp(0.5 * (edges[k] + edges[k + 1]))
    near = d[(d > mode * 0.7) & (d < mode * 1.4)]
    return float(np.median(near) / scale) if len(near) else float(mode / scale)


def detect_openings(image: np.ndarray, roi=None, params: dict | None = None,
                    expected_diameter_px: float | None = None) -> tuple[list, dict]:
    P = dict(DEFAULT_PARAMS)
    if params:
        P.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = roi_mask(gray.shape, roi)
    diag = {"params": P}

    clahe = cv2.createCLAHE(clipLimit=float(P["clahe_clip"]), tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    d0 = expected_diameter_px or estimate_opening_diameter(eq, mask)
    diag["expected_diameter_px"] = d0
    if d0 is None:
        return [], {**diag, "reason": "could not estimate opening size"}
    d0 = float(np.clip(d0, P["min_diameter_px"], P["max_diameter_px"]))

    k = _odd(P["blackhat_scale"] * d0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel)
    bh = cv2.GaussianBlur(bh, (0, 0), max(0.6, d0 / 12.0))
    vals = bh[mask > 0]
    if vals.size == 0:
        return [], {**diag, "reason": "empty ROI"}
    otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(otsu) * (1.0 + float(P["thresh_offset"]))
    binary = ((bh > thr) & (mask > 0)).astype(np.uint8) * 255
    diag["threshold"] = thr

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < 6:
            continue
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4 * math.pi * a / (per * per)
        eqd = 2 * math.sqrt(a / math.pi)
        raw.append((c, a, circ, eqd))
    diag["raw_components"] = len(raw)
    if not raw:
        return [], diag
    eqs = np.array([r[3] for r in raw])
    good = eqs[(eqs > d0 * 0.5) & (eqs < d0 * 1.8)]
    dmode = float(np.median(good)) if len(good) >= 5 else d0
    diag["diameter_mode_px"] = dmode
    lo, hi = dmode * (1 - P["size_tol"]), dmode * (1 + P["size_tol"])

    out = []
    for c, a, circ, eqd in raw:
        if not (lo <= eqd <= hi) or circ < P["min_circularity"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) / max(w, h) < 0.45:
            continue
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        patch = bh[y0:y1, x0:x1].astype(np.float64)
        cm = np.zeros(patch.shape, np.uint8)
        cv2.drawContours(cm, [c - [x0, y0]], -1, 1, -1, offset=(0, 0))
        cm = cv2.dilate(cm, np.ones((3, 3), np.uint8))
        wgt = np.clip(patch - thr * 0.5, 0, None) * cm
        s = wgt.sum()
        if s <= 0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        u = float((wgt * xs).sum() / s)
        v = float((wgt * ys).sum() / s)
        contrast = float(patch[cm > 0].mean())
        ell = None
        if len(c) >= 5:
            (ex, ey), (ma, mi), ang = cv2.fitEllipse(c)
            ell = [float(ma), float(mi), float(ang)]
        out.append({"u": u, "v": v, "d": float(eqd), "circularity": float(circ),
                    "contrast": contrast, "ellipse": ell})
    diag["candidates"] = len(out)
    return out, diag

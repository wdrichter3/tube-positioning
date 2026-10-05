"""Improve the detector from reviewed labels.

Two things are learned, both small enough to train on the Pi:
  1. Detector parameters (black-hat size, threshold, circularity, size band),
     tuned by coordinate descent to maximise F1 on open-tube labels.
  2. A site-state classifier (TUBE / BLOCKED / UNKNOWN / NOT_TUBE) on appearance
     features. Detections the reviewer left unlabelled become NOT_TUBE examples.

Evaluation holds out whole captures, never individual tubes, so the scores say
something about new images. New profiles are saved but not activated: the
dashboard shows old vs new scores and the user decides.
"""
from __future__ import annotations

import re
import threading
import time
import traceback
import uuid

import cv2
import numpy as np

from . import config, dataset
from .vision import detect, state

BUILTIN = {"name": "builtin", "created_at": 0, "detector": dict(detect.DEFAULT_PARAMS), "classifier": None,
           "metrics": None, "training": {"note": "Default parameters, rule-based states."}}

GRID = {
    "blackhat_scale": [1.4, 1.8, 2.2, 2.8],
    "thresh_offset": [-0.3, -0.15, 0.0, 0.15, 0.3],
    "min_circularity": [0.4, 0.5, 0.6, 0.7],
    "size_tol": [0.3, 0.45, 0.6],
    "clahe_clip": [1.0, 2.0, 3.0],
}
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,60}$")


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
def _pdir():
    return config.data_dir() / "profiles"


def load_profile(name: str) -> dict:
    if not name or name == "builtin":
        return dict(BUILTIN)
    if not NAME_RE.match(name):
        raise ValueError("invalid profile name")
    p = config.read_json(_pdir() / f"{name}.json", None)
    if p is None:
        raise FileNotFoundError(f"profile {name} not found")
    return p


def active_profile() -> dict:
    try:
        return load_profile(config.load_settings().get("active_profile", "builtin"))
    except FileNotFoundError:
        return dict(BUILTIN)


def list_profiles() -> list:
    out = [{k: BUILTIN[k] for k in ("name", "created_at", "metrics", "training")}]
    for f in sorted(_pdir().glob("*.json"), reverse=True):
        p = config.read_json(f, {})
        out.append({k: p.get(k) for k in ("name", "created_at", "metrics", "training", "parent")})
    return out


def activate(name: str):
    load_profile(name)
    s = config.load_settings()
    s["active_profile"] = name
    config.save_settings(s)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _samples(max_captures=80, test=False):
    """Reviewed, labelled captures. test=True returns only captures tagged 'test' (never trained on)."""
    caps = [c for c in dataset.list_captures(limit=100000) if c["reviewed"] and c["labels"] > 0]
    caps = [c for c in caps if ("test" in (c.get("tags") or [])) == test]
    caps = sorted(caps, key=lambda c: c["id"])[-max_captures:]
    out = []
    for c in caps:
        lab = dataset.load_labels(c["id"])
        img = dataset.load_color(c["id"])
        items = lab.get("items", [])
        # detector tuning runs many passes: use a copy at most TUNE_MAX px (the detector is scale-adaptive)
        k = min(1.0, TUNE_MAX / max(img.shape[:2]))
        img_t = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        items_t = [dict(a, u=a["u"] * k, v=a["v"] * k, r=a["r"] * k) for a in items]
        out.append({"id": c["id"], "image": img, "items": items, "image_t": img_t, "items_t": items_t,
                    "roi": lab.get("roi"), "source": c.get("source"), "device": c.get("device")})
    return out


def split(samples):
    """Detector tuning hold-out: every 5th capture (by date)."""
    n = len(samples)
    if n >= 5:
        hold = [s for k, s in enumerate(samples) if k % 5 == 4]
        train = [s for k, s in enumerate(samples) if k % 5 != 4]
    elif n >= 2:
        hold, train = samples[-1:], samples[:-1]
    else:
        hold, train = [], samples
    return train, hold


def folds(samples, k=5):
    """Grouped k-fold: every capture is held out exactly once."""
    k = max(2, min(k, len(samples)))
    return [[s for n, s in enumerate(samples) if n % k == f] for f in range(k)]


TUNE_MAX = 1600


def _eval_region(sample):
    items = sample["items"]
    h, w = sample["image"].shape[:2]
    if sample.get("roi"):
        return sample["roi"]
    pts = np.array([[a["u"] / (w - 1), a["v"] / (h - 1)] for a in items])
    if len(pts) < 3:
        return None
    import cv2
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    c = hull.mean(0)
    return ((hull - c) * 1.03 + c).clip(0, 1).tolist()


def _match(dets, gts, radius):
    """Greedy one-to-one matching; returns (tp, fp, fn, matched_det_mask)."""
    if not len(dets) or not len(gts):
        return 0, len(dets), len(gts), np.zeros(len(dets), bool)
    D = np.linalg.norm(dets[:, None, :] - gts[None, :, :], axis=2)
    pairs = np.argwhere(D <= radius[None, :])
    order = np.argsort(D[pairs[:, 0], pairs[:, 1]])
    used_d, used_g = set(), set()
    for k in order:
        i, j = pairs[k]
        if i in used_d or j in used_g:
            continue
        used_d.add(int(i)); used_g.add(int(j))
    m = np.zeros(len(dets), bool)
    m[list(used_d)] = True
    return len(used_d), len(dets) - len(used_d), len(gts) - len(used_g), m


def evaluate_detector(params, samples):
    tp = fp = fn = 0
    for s in samples:
        items = s.get("items_t", s["items"])
        img = s.get("image_t", s["image"])
        gts = np.array([[a["u"], a["v"]] for a in items if a["cls"] == "TUBE"])
        rad = np.array([max(2.0, 0.6 * a["r"]) for a in items if a["cls"] == "TUBE"])
        if not len(gts):
            continue
        dets, _ = detect.detect_openings(img, roi=_eval_region(s), params=params)
        dd = np.array([[d["u"], d["v"]] for d in dets]).reshape(-1, 2)
        a, b, c, _ = _match(dd, gts, rad)
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def tune_detector(base_params, train, progress=None, rounds=2):
    best = dict(base_params)
    best_m = evaluate_detector(best, train)
    total = rounds * sum(len(v) for v in GRID.values())
    done = 0
    for _ in range(rounds):
        improved = False
        for key, values in GRID.items():
            for v in values:
                done += 1
                if progress:
                    progress(done / total, f"Tuning {key} = {v}")
                if best.get(key) == v:
                    continue
                cand = dict(best, **{key: v})
                m = evaluate_detector(cand, train)
                if (m["f1"], m["precision"]) > (best_m["f1"], best_m["precision"]) + (1e-4, 0):
                    best, best_m, improved = cand, m, True
        if not improved:
            break
    return best, best_m


NEG_CLEARANCE = 1.6          # a hard negative must be this many label radii from every label
AUG_GAMMAS = (0.75, 1.35)    # brightness variants for augmentation


def _label_check(sample, params):
    """Detections inside the labelled region that are far from every label. Many of these on a
    reviewed capture usually means tubes were left unlabelled (they would teach the model wrongly)."""
    items = sample["items"]
    if not items:
        return np.zeros((0, 2)), 0.0
    dets, _ = detect.detect_openings(sample["image"], roi=_eval_region(sample), params=params)
    dd = np.array([[d["u"], d["v"]] for d in dets]).reshape(-1, 2)
    gts = np.array([[a["u"], a["v"]] for a in items])
    r_med = float(np.median([a["r"] for a in items]))
    if not len(dd):
        return dd, r_med
    dist = np.min(np.linalg.norm(dd[:, None, :] - gts[None, :, :], axis=2), axis=1)
    return dd[dist > NEG_CLEARANCE * r_med], r_med


def _gamma(img, g):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[img]


def _class_examples(samples, params, version=2, augment=False, seed=0):
    """Features + labels + capture index for every labelled site, plus safe hard negatives."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    feat = (lambda fi, u, v, r: fi.features_v2(u, v, r)) if version == 2 else \
        (lambda fi, u, v, r: fi.features(u, v, r))
    for gi, s in enumerate(samples):
        items = s["items"]
        if not items:
            continue
        negs, r_med = _label_check(s, params)
        negs = negs[:150]
        variants = [s["image"]]
        if augment:
            variants += [_gamma(s["image"], gm) for gm in AUG_GAMMAS]
        for vi, img in enumerate(variants):
            fi = state.FeatureImage(img)
            for a in items:
                minority = a["cls"] in ("BLOCKED", "UNKNOWN", "NOT_TUBE")
                if vi > 0 and not minority and rng.random() > 0.25:
                    continue                    # augment rare classes fully, tubes lightly
                reps = 1 if (vi == 0 or not minority) else 3
                for rep in range(reps):
                    du, dv, dr = (0.0, 0.0, 1.0) if rep == 0 else (rng.normal(0, 0.08 * a["r"]),
                                                                   rng.normal(0, 0.08 * a["r"]), rng.uniform(0.9, 1.1))
                    X.append(feat(fi, a["u"] + du, a["v"] + dv, a["r"] * dr))
                    y.append(state.CLASSES.index(a["cls"]))
                    g.append(gi)
            if vi == 0:
                for u, v in negs:
                    X.append(feat(fi, u, v, r_med))
                    y.append(state.CLASSES.index("NOT_TUBE"))
                    g.append(gi)
    return np.array(X, np.float32), np.array(y, int), np.array(g, int)


def _new_classifier():
    if state.TreeClassifier.available():
        return state.TreeClassifier()
    c = state.StateClassifier()
    c.feature_version = 2
    return c


def evaluate_classifier(clf, X, y, groups=None):
    if not len(y) or clf is None or not clf.trained:
        return None
    pred = np.argmax(clf.predict_proba(X), axis=1)
    return _scores(y, pred, groups)


def _scores(y, pred, groups=None):
    C = len(state.CLASSES)
    conf = np.zeros((C, C), int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    per = {}
    for c in range(C):
        sup = int(conf[c].sum())
        predicted = int(conf[:, c].sum())
        per[state.CLASSES[c]] = {
            "support": sup,
            "captures": int(len(set(groups[y == c]))) if groups is not None and sup else None,
            "recall": round(conf[c, c] / sup, 3) if sup else None,
            "precision": round(conf[c, c] / predicted, 3) if predicted else None}
    return {"accuracy": round(float((pred == y).mean()), 4), "per_class": per,
            "recall": {k: v["recall"] for k, v in per.items()},
            "support": {k: v["support"] for k, v in per.items()}, "confusion": conf.tolist()}


def build_features(samples, params, progress=None):
    """Compute every example once: plain (for scoring) and augmented (for training), with the
    capture index of each, so cross-validation is just array masking."""
    plain, aug = [], []
    for n, s in enumerate(samples):
        if progress:
            progress(n / max(1, len(samples)), f"Features {n + 1} of {len(samples)}: {s['id']}")
        Xp, yp, _ = _class_examples([s], params, augment=False)
        Xa, ya, _ = _class_examples([s], params, augment=True, seed=n)
        plain.append((Xp, yp, np.full(len(yp), n)))
        aug.append((Xa, ya, np.full(len(ya), n)))

    def cat(parts):
        parts = [p for p in parts if len(p[1])]
        if not parts:
            return np.zeros((0, 1), np.float32), np.zeros(0, int), np.zeros(0, int)
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))
    return cat(plain), cat(aug)


def cross_validate(feats, n_samples, progress=None, k=5):
    """Grouped k-fold: train on k-1 parts of the captures, predict the held-out part, pool."""
    (Xp, yp, gp), (Xa, ya, ga) = feats
    k = max(2, min(k, n_samples))
    ys, ps, gs = [], [], []
    for f in range(k):
        if progress:
            progress(f / k, f"Cross-validation fold {f + 1} of {k}")
        tr = (ga % k) != f
        te = (gp % k) == f
        if tr.sum() < 20 or len(set(ya[tr].tolist())) < 2 or not te.any():
            continue
        clf = _new_classifier().fit(Xa[tr], ya[tr])
        ys.append(yp[te]); ps.append(np.argmax(clf.predict_proba(Xp[te]), axis=1)); gs.append(gp[te])
    if not ys:
        return None
    return _scores(np.concatenate(ys), np.concatenate(ps), np.concatenate(gs))


def train_profile(name=None, progress=None) -> dict:
    progress = progress or (lambda f, m: None)
    progress(0.01, "Loading reviewed captures")
    samples = _samples()
    test = _samples(test=True)
    if not samples:
        raise ValueError("No reviewed captures with labels yet. Label a few captures and mark them reviewed.")
    base = active_profile()
    notes = []

    # 1. detector: tune on 4/5 of the captures, score on the rest
    tr, hold = split(samples)
    progress(0.04, f"Detector: tuning on {len(tr)} captures")
    params, train_m = tune_detector(base.get("detector") or detect.DEFAULT_PARAMS, tr,
                                    progress=lambda f, m: progress(0.04 + 0.4 * f, m))
    metrics = {"train_detector": train_m}
    if hold:
        metrics["holdout_detector"] = evaluate_detector(params, hold)
        metrics["baseline_holdout_detector"] = evaluate_detector(base.get("detector") or detect.DEFAULT_PARAMS, hold)

    # 2. label completeness (unlabelled tubes would become wrong 'not a tube' examples)
    progress(0.46, "Checking labels for missed tubes")
    suspects = []
    for s in samples:
        extra, _ = _label_check(s, params)
        n_lab = len(s["items"])
        if len(extra) >= max(8, 0.08 * n_lab):
            suspects.append({"id": s["id"], "unlabelled_detections": int(len(extra)), "labels": n_lab})
    metrics["possibly_incomplete"] = suspects
    if suspects:
        notes.append(f"{len(suspects)} reviewed capture(s) have many detections with no label nearby. "
                     "If those are real tubes, label them: otherwise they are kept out of training "
                     "only when they are close to a label.")

    # 3. state classifier: grouped cross-validation, then a final model on everything
    model_kind = "gradient-boosted trees" if state.TreeClassifier.available() else "linear (install scikit-learn for trees)"
    feats = build_features(samples, params, progress=lambda f, m: progress(0.5 + 0.25 * f, m))
    if len(samples) >= 3:
        metrics["cv_classifier"] = cross_validate(feats, len(samples),
                                                  progress=lambda f, m: progress(0.75 + 0.1 * f, m))
    else:
        notes.append("Fewer than 3 reviewed captures: no cross-validation possible yet.")
    progress(0.86, "Training final state classifier")
    (X0, y0, g0), (X, y, _) = feats
    clf = None
    if len(y) >= 20 and len(set(y.tolist())) >= 2:
        clf = _new_classifier().fit(X, y)
    else:
        notes.append("State classifier not trained: needs 20+ labels in at least two classes.")
    counts = {c: int((y0 == k).sum()) for k, c in enumerate(state.CLASSES)}
    per_caps = {c: int(len(set(g0[y0 == k]))) for k, c in enumerate(state.CLASSES)}

    # 4. permanent test set: new vs current profile on captures never used for training
    if test:
        progress(0.9, f"Scoring on {len(test)} test-set captures")
        Xt, yt, gt = _class_examples(test, params, augment=False)
        base_clf = state.load_classifier(base.get("classifier"))
        metrics["test_detector"] = evaluate_detector(params, test)
        metrics["baseline_test_detector"] = evaluate_detector(base.get("detector") or detect.DEFAULT_PARAMS, test)
        metrics["test_classifier"] = evaluate_classifier(clf, Xt, yt, gt)
        if base_clf.trained:
            ver = getattr(base_clf, "feature_version", 1)
            Xb = Xt if ver == 2 else _class_examples(test, params, version=1)[0]
            metrics["baseline_test_classifier"] = evaluate_classifier(base_clf, Xb, yt, gt)
    else:
        notes.append("No test set yet: tag 2-3 typical reviewed captures as 'Test set' on the Label tab "
                     "to compare profiles on images they never trained on.")
    if counts.get("BLOCKED", 0) < 50 or per_caps.get("BLOCKED", 0) < 10:
        notes.append(f"BLOCKED: {counts.get('BLOCKED', 0)} examples from {per_caps.get('BLOCKED', 0)} capture(s). "
                     "Plug detection gets dependable at roughly 50+ examples from 10+ different sheets.")
    if counts.get("NOT_TUBE", 0) < 10:
        notes.append("Fewer than 10 NOT_TUBE examples: empty lattice cells stay rule-based. Keep the "
                     "grey 'not a tube' proposals when you review (pass lanes, tie rods, removed tubes).")
    name = name or time.strftime("p%Y%m%d-%H%M%S")
    if not NAME_RE.match(name) or name == "builtin":
        raise ValueError("invalid profile name")
    prof = {"name": name, "created_at": time.time(), "parent": base.get("name"), "detector": params,
            "classifier": clf.to_dict() if clf else base.get("classifier"),
            "metrics": metrics,
            "training": {"captures_train": [s["id"] for s in samples], "captures_detector_holdout": [s["id"] for s in hold],
                         "captures_test": [s["id"] for s in test], "examples": counts,
                         "examples_captures": per_caps, "model": model_kind, "feature_version": 2,
                         "notes": notes}}
    config.write_json(_pdir() / f"{name}.json", prof)
    progress(1.0, "Done")
    return prof


# --------------------------------------------------------------------------
# Background jobs
# --------------------------------------------------------------------------
class Jobs:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, kind, fn, *args, **kwargs):
        with self.lock:
            if any(j["state"] == "running" and j["kind"] == kind for j in self.jobs.values()):
                raise RuntimeError(f"a {kind} job is already running")
            jid = uuid.uuid4().hex[:8]
            job = {"id": jid, "kind": kind, "state": "running", "progress": 0.0, "message": "Starting",
                   "result": None, "error": None, "started_at": time.time()}
            self.jobs[jid] = job

        def prog(f, m):
            job["progress"], job["message"] = round(float(f), 3), m

        def run():
            try:
                job["result"] = fn(*args, progress=prog, **kwargs)
                job["state"] = "done"
            except Exception as exc:
                traceback.print_exc()
                job["state"], job["error"] = "error", str(exc)
            job["finished_at"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, jid):
        return self.jobs.get(jid)

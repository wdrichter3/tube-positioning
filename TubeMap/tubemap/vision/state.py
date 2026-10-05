"""Per-site appearance features, rule-based states and a trainable classifier.

Label classes (what people mark in the labeler):
    TUBE      open tube mouth
    BLOCKED   plug, packed deposit, cap - anything a lance should not enter
    UNKNOWN   can't tell from the image
    NOT_TUBE  no tube at this spot, or a false detection

Map states:
    OPEN, PLUGGED, REVIEW, NO_TUBE
"""
from __future__ import annotations

import base64
import io
import math
import pickle

import cv2
import numpy as np

CLASSES = ["TUBE", "BLOCKED", "UNKNOWN", "NOT_TUBE"]
CLASS_TO_STATE = {"TUBE": "OPEN", "BLOCKED": "PLUGGED", "UNKNOWN": "REVIEW", "NOT_TUBE": "NO_TUBE"}
STATE_TO_CLASS = {v: k for k, v in CLASS_TO_STATE.items()}
PATCH = 12
N_ENG = 8
FEATURE_VERSION = 2
RADII = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.8)


class FeatureImage:
    """Precomputed images shared by all sites of one capture."""

    def __init__(self, image):
        g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray = g.astype(np.float32)
        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            self.a = lab[:, :, 1] - 128.0           # green(-) .. red(+)
            self.b = lab[:, :, 2] - 128.0           # blue(-) .. yellow(+)  -> rust/scale is +a +b
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            self.sat = hsv[:, :, 1] / 255.0
        else:
            self.a = self.b = self.sat = np.zeros_like(self.gray)
        self.lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(self.gray, (0, 0), 0.8), cv2.CV_32F, ksize=3))
        b = cv2.GaussianBlur(self.gray, (0, 0), 1.0)
        self.gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        self.gm = np.hypot(self.gx, self.gy)
        self.h, self.w = g.shape

    def _sample(self, img, xs, ys):
        xs = np.clip(xs, 0, self.w - 1).astype(np.float32)
        ys = np.clip(ys, 0, self.h - 1).astype(np.float32)
        return cv2.remap(img, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).ravel()

    def features(self, u, v, r):
        r = max(2.0, float(r))
        # radial sampling
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)

        def ring(rr):
            return self._sample(self.gray, u + rr * ca, v + rr * sa)

        inner = np.concatenate([ring(r * f) for f in (0.0, 0.25, 0.45, 0.6)])
        annulus = np.concatenate([ring(r * f) for f in (1.25, 1.45, 1.65)])
        gxs = self._sample(self.gx, u + r * ca, v + r * sa)
        gys = self._sample(self.gy, u + r * ca, v + r * sa)
        gm_ring = np.hypot(gxs, gys)
        gm_bg = np.concatenate([self._sample(self.gm, u + r * f * ca, v + r * f * sa) for f in (0.3, 1.6, 1.9)])
        radial = np.abs(gxs * ca + gys * sa) / (gm_ring + 1e-6)
        a_mean = float(np.mean(annulus)) + 1.0
        i_mean = float(np.mean(inner))
        eng = np.array([
            (a_mean - i_mean) / a_mean,                 # darkness contrast
            i_mean / a_mean,                            # centre brightness ratio
            float(np.std(inner)) / a_mean,
            float(np.mean(gm_ring)) / (float(np.median(gm_bg)) + 1.0),   # edge ring score
            float(np.mean(radial)),                     # edge radial alignment
            float(np.mean(inner < 0.5 * a_mean)),       # dark fraction
            float(np.std(annulus)) / a_mean,
            math.log(r),
        ], np.float32)
        # normalised patch
        half = 1.6 * r
        xs = np.linspace(u - half, u + half, PATCH)
        ys = np.linspace(v - half, v + half, PATCH)
        gx, gy = np.meshgrid(xs, ys)
        patch = self._sample(self.gray, gx.ravel(), gy.ravel())
        patch = (patch - np.mean(patch)) / (np.std(patch) + 4.0)
        return np.concatenate([eng, patch.astype(np.float32)])


    def features_v2(self, u, v, r):
        """Rotation-invariant appearance: radial profile, colour, texture, edge ring.

        Colour matters for plugs and packed deposits (rust, scale), which greyscale features miss.
        """
        r = max(2.0, float(r))
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)

        def ring(img, f):
            return self._sample(img, u + r * f * ca, v + r * f * sa)

        prof = np.array([np.mean(ring(self.gray, f)) for f in RADII])
        ann = float(np.mean(prof[-3:])) + 1.0
        inner_f = (0.0, 0.2, 0.4, 0.6)
        inner_g = np.concatenate([ring(self.gray, f) for f in inner_f])
        inner_a = np.concatenate([ring(self.a, f) for f in inner_f])
        inner_b = np.concatenate([ring(self.b, f) for f in inner_f])
        inner_s = np.concatenate([ring(self.sat, f) for f in inner_f])
        inner_l = np.concatenate([ring(self.lap, f) for f in inner_f])
        out_a = np.concatenate([ring(self.a, f) for f in (1.3, 1.6)])
        out_b = np.concatenate([ring(self.b, f) for f in (1.3, 1.6)])
        out_s = np.concatenate([ring(self.sat, f) for f in (1.3, 1.6)])
        out_l = np.concatenate([ring(self.lap, f) for f in (1.3, 1.6)])
        gxs, gys = ring(self.gx, 1.0), ring(self.gy, 1.0)
        gm_ring = np.hypot(gxs, gys)
        gm_bg = np.concatenate([ring(self.gm, f) for f in (0.3, 1.6, 1.9)])
        radial = np.abs(gxs * ca + gys * sa) / (gm_ring + 1e-6)
        ang_prof = ring(self.gray, 0.4)
        f = np.concatenate([
            prof / ann,                                               # 9 radial brightness profile
            [float(np.std(inner_g)) / ann,                            # inner unevenness
             float(np.mean(inner_g < 0.5 * ann)),                    # dark fraction
             float(np.std(ang_prof)) / ann,                           # angular unevenness (partial plug)
             float(np.mean(gm_ring)) / (float(np.median(gm_bg)) + 1.0),
             float(np.mean(radial)),
             float(np.mean(inner_a)), float(np.mean(inner_b)),       # inner colour (Lab a*, b*)
             float(np.mean(inner_a) - np.mean(out_a)),               # colour contrast to the face
             float(np.mean(inner_b) - np.mean(out_b)),
             float(np.mean(inner_s)), float(np.mean(inner_s) - np.mean(out_s)),
             float(np.mean(inner_l)) / (float(np.mean(out_l)) + 1.0),   # texture inside vs face
             float(np.mean(inner_l)) / ann,
             math.log(r)],
        ])
        return f.astype(np.float32)

    def features_for(self, version, u, v, r):
        return self.features_v2(u, v, r) if version == 2 else self.features(u, v, r)


def rule_state(feat, observed: bool):
    """Classical fallback when no trained classifier exists."""
    contrast, bright, inner_std, ring, radial, dark = feat[:6]
    if observed:
        conf = float(np.clip(contrast * 1.5, 0.3, 0.99))
        return "OPEN", conf
    if ring > 1.8 and radial > 0.55 and inner_std < 0.22:
        return "PLUGGED", float(np.clip((ring - 1.2) / 2.0, 0.3, 0.85))
    if ring > 1.5 or (dark > 0.5 and contrast > 0.3):
        return "REVIEW", 0.4
    return "NO_TUBE", float(np.clip(1.0 - (ring - 1.0) / 1.5, 0.3, 0.9))


# --------------------------------------------------------------------------
# Softmax regression (numpy only, small enough to train on a Pi)
# --------------------------------------------------------------------------
class StateClassifier:
    feature_version = 1
    kind = "softmax"

    def __init__(self, mean=None, std=None, W=None, b=None, classes=None, support=None):
        self.mean, self.std, self.W, self.b = mean, std, W, b
        self.classes = classes or list(CLASSES)
        self.support = support or {}

    @property
    def trained(self):
        return self.W is not None

    def to_dict(self):
        if not self.trained:
            return None
        return {"kind": "softmax", "feature_version": self.feature_version, "classes": self.classes,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "W": self.W.tolist(), "b": self.b.tolist(), "support": self.support}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32),
                   np.array(d["W"], np.float32), np.array(d["b"], np.float32), d["classes"], d.get("support"))

    def fit(self, X, y, l2=1e-3, iters=600, lr=0.05, seed=0):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, int)
        C = len(self.classes)
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-3
        Z = (X - self.mean) / self.std
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 0.01, (Z.shape[1], C)).astype(np.float32)
        b = np.zeros(C, np.float32)
        counts = np.bincount(y, minlength=C).astype(np.float32)
        self.support = {c: int(n) for c, n in zip(self.classes, counts)}
        cw = np.where(counts > 0, len(y) / (C * np.maximum(counts, 1)), 0.0)
        sw = cw[y]
        Y = np.eye(C, dtype=np.float32)[y]
        mW, vW, mb, vb = 0 * W, 0 * W, 0 * b, 0 * b
        for t in range(1, iters + 1):
            logits = Z @ W + b
            logits -= logits.max(1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(1, keepdims=True)
            G = (P - Y) * sw[:, None] / sw.sum()
            gW = Z.T @ G + l2 * W
            gb = G.sum(0)
            mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            W -= lr * (mW / (1 - 0.9 ** t)) / (np.sqrt(vW / (1 - 0.999 ** t)) + 1e-8)
            b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        self.W, self.b = W, b
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, np.float32) - self.mean) / self.std
        logits = Z @ self.W + self.b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(1, keepdims=True)


class TreeClassifier:
    """Gradient-boosted trees (scikit-learn), stored inside the profile JSON as base64 pickle."""
    feature_version = FEATURE_VERSION
    kind = "trees"

    def __init__(self, model=None, classes=None, support=None):
        self.model, self.classes, self.support = model, classes or list(CLASSES), support or {}

    @property
    def trained(self):
        return self.model is not None

    @staticmethod
    def available():
        try:
            import sklearn  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, y, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        y = np.asarray(y, int)
        counts = np.bincount(y, minlength=len(self.classes))
        self.support = {c: int(n) for c, n in zip(self.classes, counts)}
        w = np.where(counts > 0, len(y) / (len(np.unique(y)) * np.maximum(counts, 1)), 0.0)[y]
        self.model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
                                                    min_samples_leaf=8, l2_regularization=1.0, random_state=seed)
        self.model.fit(np.asarray(X, np.float32), y, sample_weight=w)
        return self

    def predict_proba(self, X):
        P = self.model.predict_proba(np.asarray(X, np.float32))
        out = np.zeros((len(P), len(self.classes)), np.float32)
        out[:, self.model.classes_] = P
        return out

    def to_dict(self):
        if not self.trained:
            return None
        return {"kind": self.kind, "feature_version": self.feature_version, "classes": self.classes,
                "support": self.support, "model_b64": base64.b64encode(pickle.dumps(self.model)).decode()}

    @classmethod
    def from_dict(cls, d):
        return cls(pickle.loads(base64.b64decode(d["model_b64"])), d["classes"], d.get("support"))


def load_classifier(d):
    """Classifier from a profile dict (any kind); untrained object if missing or unloadable."""
    if not d:
        return StateClassifier()
    try:
        if d.get("kind") == "trees":
            return TreeClassifier.from_dict(d)
        c = StateClassifier.from_dict(d)
        c.feature_version = int(d.get("feature_version", 1))
        return c
    except Exception:                       # e.g. scikit-learn missing on this computer
        c = StateClassifier()
        c.load_error = True
        return c


def classify_sites(fi: FeatureImage, sites, classifier=None, review_below: float = 0.6):
    """sites: list of dicts with u, v, r_px, observed. Adds state/confidence/probs."""
    if not sites:
        return sites
    feats = np.array([fi.features(s["u"], s["v"], s["r_px"]) for s in sites], np.float32)
    probs = None
    if classifier is not None and classifier.trained:
        ver = getattr(classifier, "feature_version", 1)
        X = feats if ver == 1 else np.array([fi.features_v2(s["u"], s["v"], s["r_px"]) for s in sites], np.float32)
        probs = classifier.predict_proba(X)
    # A model that has never seen empty cells can't judge them: keep rules for unobserved sites.
    empty_ok = probs is not None and classifier.support.get("NOT_TUBE", 0) >= 10
    for k, s in enumerate(sites):
        if probs is not None and (s.get("observed") or empty_ok):
            pk = probs[k]
            c = int(np.argmax(pk))
            state = CLASS_TO_STATE[classifier.classes[c]]
            conf = float(pk[c])
            if conf < review_below and state != "NO_TUBE":
                state = "REVIEW"
            s["probs"] = {cl: round(float(x), 3) for cl, x in zip(classifier.classes, pk)}
            s["state_source"] = "model"
        else:
            state, conf = rule_state(feats[k], bool(s.get("observed")))
            s["state_source"] = "rules"
        s["state"], s["confidence"] = state, round(conf, 3)
    return sites

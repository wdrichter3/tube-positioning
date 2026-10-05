"""Captures, labels, import and export.

Each capture is a folder:
    color.png      lossless colour image as captured
    depth.png      16-bit depth in mm (aligned to colour FOV), if the device has depth
    preview.jpg    for the dashboard
    thumb.jpg      for lists
    meta.json      device, settings, intrinsics, job, notes, tags
    labels.json    labelled sites {u, v, r, cls} + review status
    map.json       latest tube map built from this capture
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import VERSION, config
from .vision.state import CLASSES

CID_RE = re.compile(r"^[A-Za-z0-9_.-]{3,80}$")
YOLO_CLASSES = ["tube", "blocked", "unknown"]          # NOT_TUBE is background for detectors


def _root() -> Path:
    return config.data_dir() / "captures"


def cdir(cid: str) -> Path:
    if not CID_RE.match(str(cid)):
        raise ValueError("invalid capture id")
    p = _root() / cid
    if not p.is_dir():
        raise FileNotFoundError(f"capture {cid} not found")
    return p


def _new_id(tag="C") -> str:
    return time.strftime(f"{tag}%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


def _write_previews(d: Path, color: np.ndarray):
    h, w = color.shape[:2]
    s = min(1.0, 1600 / max(w, h))
    prev = cv2.resize(color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else color
    cv2.imwrite(str(d / "preview.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
    s = 320 / max(w, h)
    cv2.imwrite(str(d / "thumb.jpg"), cv2.resize(color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 80])


def save_capture(color, depth=None, intrinsics=None, device=None, job=None, source="camera",
                 frame_meta=None, filename=None, set_id=None) -> dict:
    cid = _new_id()
    d = _root() / cid
    d.mkdir(parents=True)
    if not cv2.imwrite(str(d / "color.png"), color):
        raise IOError("could not write colour image")
    if depth is not None and depth.size:
        cv2.imwrite(str(d / "depth.png"), np.asarray(depth, np.uint16))
    _write_previews(d, color)
    dev = device or {}
    ds = dev.get("settings", {})
    meta = {
        "id": cid, "created_at": time.time(), "software_version": VERSION, "source": source,
        "source_filename": filename, "capture_set": set_id,
        "device": {k: dev.get(k) for k in ("id", "name", "driver", "role")} if dev else None,
        "device_settings": ds, "device_info": dev.get("info"),
        "depth_scale_factor": float(ds.get("depth_scale_factor", 1.0) or 1.0),
        "depth_uncertainty_pct": float(ds.get("depth_uncertainty_pct", 2.0) or 2.0),
        "intrinsics": intrinsics, "image": {"width": int(color.shape[1]), "height": int(color.shape[0])},
        "depth_size": None if depth is None or not depth.size else [int(depth.shape[1]), int(depth.shape[0])],
        "job": job or {}, "tags": [], "notes": "", "frame_meta": frame_meta or {},
    }
    config.write_json(d / "meta.json", meta)
    return meta


def load_meta(cid) -> dict:
    return config.read_json(cdir(cid) / "meta.json", {})


def load_color(cid) -> np.ndarray:
    img = cv2.imread(str(cdir(cid) / "color.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("colour image unreadable")
    return img


def load_depth(cid):
    p = cdir(cid) / "depth.png"
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def load_labels(cid) -> dict:
    return config.read_json(cdir(cid) / "labels.json",
                            {"version": 1, "items": [], "roi": None, "reviewed": False})


def load_map(cid):
    return config.read_json(cdir(cid) / "map.json", None)


def save_map(cid, m):
    config.write_json(cdir(cid) / "map.json", m)


def validate_items(items, w, h):
    out = []
    for k, a in enumerate(items or []):
        cls = str(a.get("cls", "TUBE")).upper()
        if cls not in CLASSES:
            raise ValueError(f"label {k}: unknown class {cls}")
        u, v, r = float(a["u"]), float(a["v"]), float(a.get("r", 6))
        if not (0 <= u < w and 0 <= v < h):
            raise ValueError(f"label {k}: outside the image")
        if not (0.5 <= r <= max(w, h) / 4):
            raise ValueError(f"label {k}: radius out of range")
        out.append({"id": str(a.get("id") or f"L{k:05d}")[:24], "u": round(u, 2), "v": round(v, 2),
                    "r": round(r, 2), "cls": cls, "src": str(a.get("src", "manual"))[:12],
                    "conf": None if a.get("conf") is None else float(a["conf"])})
    return out


def _norm_roi(roi):
    if not roi:
        return None
    pts = [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in roi]
    return pts if len(pts) >= 3 else None


def save_labels(cid, items, roi=None, reviewed=False, notes=None) -> dict:
    meta = load_meta(cid)
    w, h = meta["image"]["width"], meta["image"]["height"]
    lab = load_labels(cid)
    lab.update({"version": 1, "items": validate_items(items, w, h), "roi": _norm_roi(roi),
                "reviewed": bool(reviewed), "updated_at": time.time()})
    if reviewed:
        lab["reviewed_at"] = time.time()
    if notes is not None:
        lab["notes"] = str(notes)[:2000]
    config.write_json(cdir(cid) / "labels.json", lab)
    return lab


def update_meta(cid, notes=None, tags=None, job=None, name=None) -> dict:
    meta = load_meta(cid)
    if name is not None:
        meta["name"] = str(name)[:60]
    if notes is not None:
        meta["notes"] = str(notes)[:4000]
    if tags is not None:
        meta["tags"] = [str(t)[:40] for t in tags][:20]
    if job is not None:
        meta["job"] = {k: str(v)[:120] for k, v in job.items()}
    config.write_json(cdir(cid) / "meta.json", meta)
    return meta


def set_depth_scale(cid, factor):
    meta = load_meta(cid)
    meta["depth_scale_factor"] = float(factor)
    config.write_json(cdir(cid) / "meta.json", meta)


def delete_capture(cid):
    d = cdir(cid)
    dest = config.data_dir() / "trash" / f"{cid}-{int(time.time())}"
    shutil.move(str(d), str(dest))


def summary(cid) -> dict:
    d = cdir(cid)
    meta = config.read_json(d / "meta.json", {})
    lab = config.read_json(d / "labels.json", {"items": [], "reviewed": False})
    counts = {c: 0 for c in CLASSES}
    for a in lab.get("items", []):
        counts[a.get("cls", "TUBE")] = counts.get(a.get("cls", "TUBE"), 0) + 1
    m = (d / "map.json").exists()
    name = meta.get("name") or meta.get("source_filename") or (meta.get("job") or {}).get("exchanger") or ""
    return {"id": cid, "name": str(name)[:60], "created_at": meta.get("created_at"),
            "device": (meta.get("device") or {}).get("name"),
            "source": meta.get("source"), "image": meta.get("image"), "has_depth": (d / "depth.png").exists(),
            "reviewed": bool(lab.get("reviewed")), "label_counts": counts, "labels": len(lab.get("items", [])),
            "has_map": m, "tags": meta.get("tags", []), "notes": meta.get("notes", ""),
            "job": meta.get("job", {}), "capture_set": meta.get("capture_set")}


def list_captures(limit=500, reviewed=None) -> list:
    ids = sorted((p.name for p in _root().iterdir() if p.is_dir() and CID_RE.match(p.name)), reverse=True)
    out = []
    for cid in ids:
        try:
            s = summary(cid)
        except Exception:
            continue
        if reviewed is not None and s["reviewed"] != reviewed:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def stats() -> dict:
    caps = list_captures(limit=100000)
    counts = {c: 0 for c in CLASSES}
    for c in caps:
        if c["reviewed"]:
            for k, v in c["label_counts"].items():
                counts[k] = counts.get(k, 0) + v
    return {"captures": len(caps), "reviewed": sum(c["reviewed"] for c in caps),
            "with_depth": sum(c["has_depth"] for c in caps), "reviewed_label_counts": counts}


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------
def import_image(raw: bytes, filename="upload") -> dict:
    if len(raw) > 60 * 1024 * 1024:
        raise ValueError("image larger than 60 MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{filename}: not a readable image")
    if min(img.shape[:2]) < 100:
        raise ValueError(f"{filename}: image is too small")
    return save_capture(img, source="import", filename=str(filename)[:120])


def import_zip(raw: bytes) -> dict:
    """Import an old Image Lab export or a TubeMap dataset export."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = set(z.namelist())
    created, skipped = [], 0
    if "manifest.json" in names:
        man = json.loads(z.read("manifest.json"))
    else:
        man = {}
    if man.get("schema") == "tubemap.dataset/1":
        for s in man.get("samples", []):
            img = cv2.imdecode(np.frombuffer(z.read(s["image"]), np.uint8), cv2.IMREAD_COLOR)
            lab = json.loads(z.read(s["labels"]))
            meta = save_capture(img, intrinsics=s.get("intrinsics"), source="import-tubemap",
                                filename=s.get("id"))
            save_labels(meta["id"], lab.get("items", []), lab.get("roi"), lab.get("reviewed", False))
            created.append(meta["id"])
        return {"imported": len(created), "skipped": skipped, "ids": created, "format": "tubemap"}
    # Tube Positioning Image Lab export (schema 1): images/<sid>.jpg + annotations/<sid>.json
    ann_files = [n for n in names if n.startswith("annotations/") and n.endswith(".json")]
    if not ann_files:
        raise ValueError("ZIP has neither a TubeMap manifest nor Image Lab annotations")
    seen = set()
    for an in sorted(ann_files):
        data = json.loads(z.read(an))
        sid = data.get("sample_id") or Path(an).stem
        img_name = f"images/{sid}.jpg"
        if img_name not in names:
            skipped += 1
            continue
        raw_img = z.read(img_name)
        digest = hashlib.sha1(raw_img).hexdigest()
        if digest in seen:
            skipped += 1
            continue
        seen.add(digest)
        img = cv2.imdecode(np.frombuffer(raw_img, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        items = []
        for k, a in enumerate(data.get("annotations", [])):
            lab = str(a.get("label", "TUBE")).upper()
            cls = lab if lab in CLASSES else "UNKNOWN"
            r = 0.25 * (float(a.get("w", 0.02)) * w + float(a.get("h", 0.02)) * h)
            items.append({"u": float(a["x"]) * w, "v": float(a["y"]) * h, "r": max(1.0, r), "cls": cls,
                          "src": "imagelab"})
        roi = data.get("roi")
        roi_pts = roi.get("points") if isinstance(roi, dict) else roi
        meta = save_capture(img, source="import-imagelab", filename=data.get("source_filename") or sid)
        save_labels(meta["id"], items, roi_pts if roi_pts and len(roi_pts) >= 3 else None, reviewed=True,
                    notes=data.get("notes"))
        created.append(meta["id"])
    return {"imported": len(created), "skipped": skipped, "ids": created, "format": "imagelab"}


def import_imagelab_dir(path) -> dict:
    """Import the old Image Lab data folder (images/ + annotations/) without zipping it first."""
    root = Path(path).expanduser()
    if not (root / "annotations").is_dir():
        raise ValueError(f"{root} has no annotations folder")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for sub in ("images", "annotations"):
            for f in sorted((root / sub).glob("*")):
                if f.is_file() and not f.name.endswith(".upload.json"):
                    z.write(f, f"{sub}/{f.name}")
    return import_zip(buf.getvalue())


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export_dataset(reviewed_only=True) -> tuple[bytes, int]:
    caps = [c for c in list_captures(limit=100000) if c["reviewed"] or not reviewed_only]
    man = {"schema": "tubemap.dataset/1", "software_version": VERSION, "created_at": time.time(),
           "classes": CLASSES, "yolo_classes": YOLO_CLASSES, "samples": []}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for c in caps:
            cid = c["id"]
            d = cdir(cid)
            img = cv2.imread(str(d / "color.png"))
            if img is None:
                continue
            h, w = img.shape[:2]
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            z.writestr(f"images/{cid}.jpg", jpg.tobytes())
            lab = load_labels(cid)
            z.writestr(f"labels_json/{cid}.json", json.dumps(lab, indent=1))
            lines = []
            for a in lab.get("items", []):
                if a["cls"] == "NOT_TUBE":
                    continue
                k = YOLO_CLASSES.index(a["cls"].lower())
                bw, bh = 2 * a["r"] / w, 2 * a["r"] / h
                lines.append(f"{k} {a['u'] / w:.6f} {a['v'] / h:.6f} {bw:.6f} {bh:.6f}")
            z.writestr(f"labels/{cid}.txt", "\n".join(lines) + ("\n" if lines else ""))
            meta = load_meta(cid)
            man["samples"].append({"id": cid, "image": f"images/{cid}.jpg", "labels": f"labels_json/{cid}.json",
                                   "yolo": f"labels/{cid}.txt", "reviewed": c["reviewed"],
                                   "intrinsics": meta.get("intrinsics"), "device": meta.get("device")})
        z.writestr("manifest.json", json.dumps(man, indent=1))
        z.writestr("data.yaml", "path: .\ntrain: images\nval: images\nnames:\n" +
                   "".join(f"  {k}: {n}\n" for k, n in enumerate(YOLO_CLASSES)))
        z.writestr("README.txt", "TubeMap labelled dataset\n"
                   "images/*.jpg + labels/*.txt (YOLO: class cx cy w h, normalised) + labels_json (native).\n"
                   "Classes: 0 tube (open), 1 blocked, 2 unknown. NOT_TUBE labels are kept only in labels_json.\n"
                   "data.yaml points train and val at the same folder: split by capture before training.\n")
    return buf.getvalue(), len(man["samples"])


def export_capture(cid) -> bytes:
    d = cdir(cid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.iterdir():
            if f.is_file():
                z.write(f, f"{cid}/{f.name}")
    return buf.getvalue()

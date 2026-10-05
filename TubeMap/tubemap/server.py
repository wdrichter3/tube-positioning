"""HTTP API and dashboard (standard library only)."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from . import VERSION, config, dataset, learning
from .devices.registry import DRIVERS, describe
from .manager import DeviceManager
from .vision import geometry as geo
from .vision import depthcheck, mapping

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 200 * 1024 * 1024
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon",
        ".zip": "application/zip", ".csv": "text/csv"}


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def to_mm(value, unit):
    v = float(value)
    return v * geo.MM_PER_IN if str(unit).lower().startswith("in") else v


def geometry_from_request(g: dict) -> dict:
    mode = g.get("mode", "known")
    out = {"mode": mode, "pattern": str(g.get("pattern", "AUTO")).upper()}
    if mode == "known":
        try:
            out["pitch_mm"] = to_mm(g["pitch_value"], g.get("pitch_unit", "in"))
            out["tube_id_mm"] = to_mm(g["id_value"], g.get("id_unit", "in"))
        except (KeyError, TypeError, ValueError):
            raise HttpError(400, "Enter pitch and tube ID as numbers.")
        out.update({k: g.get(k) for k in ("pitch_value", "pitch_unit", "id_value", "id_unit")})
    return out


class App:
    def __init__(self):
        self.devices = DeviceManager()
        self.jobs = learning.Jobs()
        self.map_lock = threading.Lock()

    # ---- helpers -------------------------------------------------------------
    def capture_from(self, dev_id, set_id=None):
        w = self.devices.worker(dev_id)
        fs = w.capture()
        cfg = dict(self.devices.get_cfg(dev_id))
        cfg["info"] = w.info
        settings = config.load_settings()
        meta = dataset.save_capture(fs.color, fs.depth, fs.intrinsics, device=cfg, job=settings.get("job"),
                                    frame_meta=fs.meta, set_id=set_id)
        out = dataset.summary(meta["id"])
        if (fs.meta or {}).get("depth_missing"):
            out["depth_note"] = fs.meta["depth_missing"]
        return out

    def capture_set(self, dev_id):
        """Capture dev_id, plus every running 'depth' role device, at the same moment."""
        cfg = self.devices.get_cfg(dev_id)
        others = []
        if cfg.get("role") != "depth":
            others = [d["id"] for d in self.devices.list()
                      if d.get("role") == "depth" and d.get("state") == "running" and d["id"] != dev_id]
        if not others:
            return self.capture_from(dev_id), []
        set_id = time.strftime("S%Y%m%d-%H%M%S") + "-" + dev_id[-4:]
        results, errors = {}, {}

        def one(did):
            try:
                results[did] = self.capture_from(did, set_id=set_id)
            except Exception as exc:
                errors[did] = str(exc)
        threads = [threading.Thread(target=one, args=(d,)) for d in [dev_id] + others]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        if dev_id not in results:
            raise RuntimeError(errors.get(dev_id, "capture failed"))
        extra = [results[d] for d in others if d in results]
        for d in others:
            if d in errors:
                extra.append({"device_id": d, "error": errors[d]})
        return results[dev_id], extra

    def _load_depth_for_mapping(self, meta, cid):
        depth = dataset.load_depth(cid)
        note = None
        if depth is not None and (meta.get("frame_meta") or {}).get("depth_aligned") is False:
            note = ("Depth was saved but is not aligned to the colour image "
                    f"({(meta.get('frame_meta') or {}).get('depth_alignment', 'off')}); not used.")
            depth = None
        if depth is not None:
            depth = depth.astype(np.float32) * float(meta.get("depth_scale_factor", 1.0) or 1.0)
        return depth, note

    def _map_one(self, cid, geometry_req, roi):
        meta = dataset.load_meta(cid)
        color = dataset.load_color(cid)
        depth, note = self._load_depth_for_mapping(meta, cid)
        settings = config.load_settings()
        tol = dict(settings.get("tolerances", {}))
        tol["depth_scale_uncertainty_pct"] = float(meta.get("depth_uncertainty_pct", 2.0) or 2.0)
        g = geometry_from_request(geometry_req)
        prof = learning.active_profile()
        with self.map_lock:
            m = mapping.build_map(color, depth, meta.get("intrinsics"), g, roi=roi, profile=prof, tolerances=tol)
        if note:
            m["checks"].append({"id": "depth_alignment", "label": "Depth alignment", "status": "WARN", "detail": note})
        m["capture_id"] = cid
        m["profile"] = prof.get("name")
        m["geometry_request"] = geometry_req
        m["device"] = meta.get("device")
        m["job"] = meta.get("job")
        m["capture_set"] = meta.get("capture_set")
        m["depth_scale_factor"] = meta.get("depth_scale_factor", 1.0)
        return m, meta

    def companions(self, meta):
        set_id = meta.get("capture_set")
        if not set_id:
            return []
        out = []
        for c in dataset.list_captures(limit=100000):
            if c.get("capture_set") != set_id or c["id"] == meta["id"] or not c["has_depth"]:
                continue
            cm = dataset.load_meta(c["id"])
            if (cm.get("device") or {}).get("role") == "depth":
                out.append(cm)
        return out

    def build_map(self, cid, geometry_req, roi, use_companion=True):
        m, meta = self._map_one(cid, geometry_req, roi)
        if use_companion:
            for cmeta in self.companions(meta)[:1]:
                name = (cmeta.get("device") or {}).get("name") or "depth camera"
                old_c = dataset.load_map(cmeta["id"]) or {}
                c_roi = old_c.get("roi") or dataset.load_labels(cmeta["id"]).get("roi")
                try:
                    cm, _ = self._map_one(cmeta["id"], geometry_req, c_roi)
                    dataset.save_map(cmeta["id"], cm)
                    depthcheck.merge_companion(m, cm, name)
                    m["companion"]["roi_used"] = bool(c_roi)
                except Exception as exc:
                    m["checks"].append({"id": "companion_match", "label": f"Companion ({name})", "status": "WARN",
                                        "detail": f"Could not map the {name} image: {exc}"})
                    m["companion"] = {"capture_id": cmeta["id"], "device": name, "error": str(exc)}
        m["ready"] = not any(c["status"] == "FAIL" for c in m["checks"])
        old = dataset.load_map(cid)
        if old and old.get("anchors"):
            m["anchors"] = old["anchors"]
            m["transform"] = mapping.fit_anchor_transform(m["sites"], m["anchors"], old.get("anchor_allow_scale", False))
            m["anchor_allow_scale"] = old.get("anchor_allow_scale", False)
            mapping.apply_anchor_transform(m["sites"], m["transform"])
        dataset.save_map(cid, m)
        return m

    def status(self):
        d = config.data_dir()
        du = shutil.disk_usage(d)
        s = config.load_settings()
        return {"version": VERSION, "hostname": socket.gethostname(), "data_dir": str(d),
                "disk_free_gb": round(du.free / 1e9, 1), "disk_total_gb": round(du.total / 1e9, 1),
                "active_profile": s.get("active_profile"), "devices": self.devices.list(),
                "dataset": dataset.stats(), "time": time.time()}


ROUTES = []


def route(method, pattern):
    rx = re.compile("^" + pattern + "$")

    def deco(fn):
        ROUTES.append((method, rx, fn))
        return fn
    return deco


# ---------------------------------------------------------------- status/settings
@route("GET", r"/api/status")
def r_status(app, h, q):
    return app.status()


@route("GET", r"/api/settings")
def r_settings(app, h, q):
    return config.load_settings()


@route("PUT", r"/api/settings")
def r_settings_put(app, h, q):
    body = h.json_body()
    s = config.load_settings()
    for k in ("geometry", "tolerances", "job", "live_max_width", "live_fps", "port", "appearance"):
        if k in body:
            s[k] = body[k]
    return config.save_settings(s)


@route("GET", r"/api/standards")
def r_standards(app, h, q):
    return geo.standard_tables()


# ---------------------------------------------------------------- devices
@route("GET", r"/api/drivers")
def r_drivers(app, h, q):
    return describe()


@route("GET", r"/api/devices")
def r_devices(app, h, q):
    return app.devices.list()


@route("POST", r"/api/devices")
def r_dev_add(app, h, q):
    b = h.json_body()
    if b.get("driver") not in DRIVERS:
        raise HttpError(400, "Choose a device type.")
    return app.devices.add(b.get("name"), b["driver"], b.get("settings") or {}, b.get("role", "aux"),
                           b.get("enabled", True))


@route("PUT", r"/api/devices/(?P<dev>[\w.-]+)")
def r_dev_put(app, h, q, dev):
    b = h.json_body()
    return app.devices.update(dev, b.get("name"), b.get("settings"), b.get("role"), b.get("enabled"))


@route("DELETE", r"/api/devices/(?P<dev>[\w.-]+)")
def r_dev_del(app, h, q, dev):
    app.devices.remove(dev)
    return {"ok": True}


@route("POST", r"/api/devices/(?P<dev>[\w.-]+)/restart")
def r_dev_restart(app, h, q, dev):
    app.devices.restart(dev)
    return {"ok": True}


@route("POST", r"/api/drivers/(?P<drv>[\w.-]+)/discover")
def r_discover(app, h, q, drv):
    if drv not in DRIVERS:
        raise HttpError(404, "unknown driver")
    ok, msg = DRIVERS[drv].available()
    if not ok:
        raise HttpError(409, msg)
    return {"found": DRIVERS[drv].discover()}


@route("POST", r"/api/drivers/(?P<drv>[\w.-]+)/test")
def r_test(app, h, q, drv):
    if drv not in DRIVERS:
        raise HttpError(404, "unknown driver")
    b = h.json_body()
    info, thumb = app.devices.test(drv, b.get("settings") or {})
    info["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(thumb).decode()
    return info


@route("GET", r"/api/devices/(?P<dev>[\w.-]+)/snapshot\.jpg")
def r_snapshot(app, h, q, dev):
    w = app.devices.worker(dev)
    data = w.jpeg(int(q.get("max", ["1024"])[0]))
    if data is None:
        raise HttpError(503, "no frame yet")
    h.send_bytes(data, "image/jpeg")


@route("GET", r"/api/devices/(?P<dev>[\w.-]+)/stream\.mjpg")
def r_stream(app, h, q, dev):
    w = app.devices.worker(dev)
    s = config.load_settings()
    mx = int(q.get("max", [str(s.get("live_max_width", 1024))])[0])
    period = 1.0 / max(0.5, float(s.get("live_fps", 6)))
    h.send_response(200)
    h.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    last = None
    try:
        while True:
            fs = w.latest
            if fs is not None and fs.seq != last:
                data = w.jpeg(mx)
                last = fs.seq
                h.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                              str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                h.wfile.flush()
            time.sleep(period)
            if w.state == "stopped":
                break
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        pass
    return None


@route("POST", r"/api/devices/(?P<dev>[\w.-]+)/capture")
def r_capture(app, h, q, dev):
    main, extra = app.capture_set(dev)
    main["companions"] = extra
    return main


@route("POST", r"/api/capture_all")
def r_capture_all(app, h, q):
    set_id = time.strftime("S%Y%m%d-%H%M%S")
    out, errors = [], {}
    running = [d["id"] for d in app.devices.list() if d.get("state") == "running"]
    results = {}

    def one(did):
        try:
            results[did] = app.capture_from(did, set_id=set_id)
        except Exception as exc:
            errors[did] = str(exc)
    threads = [threading.Thread(target=one, args=(d,)) for d in running]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    out = [results[d] for d in running if d in results]
    return {"set_id": set_id, "captures": out, "errors": errors}


# ---------------------------------------------------------------- captures
@route("GET", r"/api/captures")
def r_caps(app, h, q):
    rv = q.get("reviewed", [None])[0]
    reviewed = None if rv in (None, "", "all") else rv in ("1", "true", "yes")
    return dataset.list_captures(limit=int(q.get("limit", ["300"])[0]), reviewed=reviewed)


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)")
def r_cap(app, h, q, cid):
    m = dataset.load_map(cid)
    return {"summary": dataset.summary(cid), "meta": dataset.load_meta(cid), "labels": dataset.load_labels(cid),
            "map": m}


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)/(?P<name>preview\.jpg|thumb\.jpg|color\.png)")
def r_cap_img(app, h, q, cid, name):
    p = dataset.cdir(cid) / name
    if not p.exists() and name == "preview.jpg":
        p = dataset.cdir(cid) / "color.png"
    h.send_file(p)


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)/depth\.jpg")
def r_cap_depth(app, h, q, cid):
    d = dataset.load_depth(cid)
    if d is None:
        raise HttpError(404, "no depth for this capture")
    v = d[d > 0]
    lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
    n = np.clip((d.astype(np.float32) - lo) / max(1.0, hi - lo) * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(n, cv2.COLORMAP_TURBO)
    col[d == 0] = 0
    h.send_bytes(cv2.imencode(".jpg", col)[1].tobytes(), "image/jpeg")


@route("PUT", r"/api/captures/(?P<cid>[\w.-]+)/labels")
def r_labels(app, h, q, cid):
    b = h.json_body()
    try:
        lab = dataset.save_labels(cid, b.get("items", []), b.get("roi"), b.get("reviewed", False), b.get("notes"))
    except ValueError as exc:
        raise HttpError(400, str(exc))
    return {"ok": True, "labels": len(lab["items"]), "reviewed": lab["reviewed"]}


@route("PUT", r"/api/captures/(?P<cid>[\w.-]+)/meta")
def r_meta(app, h, q, cid):
    b = h.json_body()
    return dataset.update_meta(cid, b.get("notes"), b.get("tags"), b.get("job"), b.get("name"))


@route("DELETE", r"/api/captures/(?P<cid>[\w.-]+)")
def r_cap_del(app, h, q, cid):
    dataset.delete_capture(cid)
    return {"ok": True}


@route("POST", r"/api/captures/(?P<cid>[\w.-]+)/propose")
def r_propose(app, h, q, cid):
    b = h.json_body()
    roi = b.get("roi")
    g = b.get("geometry") or config.load_settings()["geometry"]
    try:
        m = app.build_map(cid, g, roi)
        return {"items": mapping.site_labels_from_map(m), "source": "map", "map_ready": m["ready"]}
    except Exception as exc:
        from .vision import detect
        img = dataset.load_color(cid)
        dets, _ = detect.detect_openings(img, roi=roi, params=learning.active_profile().get("detector"))
        items = [{"id": f"D{k:05d}", "u": d["u"], "v": d["v"], "r": d["d"] / 2, "cls": "TUBE", "src": "auto"}
                 for k, d in enumerate(dets)]
        return {"items": items, "source": "detector", "warning": f"Map failed ({exc}); showing raw detections."}


@route("POST", r"/api/captures/(?P<cid>[\w.-]+)/map")
def r_map(app, h, q, cid):
    b = h.json_body()
    try:
        return app.build_map(cid, b.get("geometry") or {}, b.get("roi"))
    except ValueError as exc:
        raise HttpError(422, str(exc))


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)/map")
def r_map_get(app, h, q, cid):
    m = dataset.load_map(cid)
    if m is None:
        raise HttpError(404, "no map built for this capture yet")
    return m


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)/map\.csv")
def r_map_csv(app, h, q, cid):
    m = dataset.load_map(cid)
    if m is None:
        raise HttpError(404, "no map built for this capture yet")
    h.send_bytes(mapping.map_to_csv(m).encode(), "text/csv",
                 extra={"Content-Disposition": f'attachment; filename="{cid}_tubemap.csv"'})


@route("PUT", r"/api/captures/(?P<cid>[\w.-]+)/anchors")
def r_anchors(app, h, q, cid):
    b = h.json_body()
    m = dataset.load_map(cid)
    if m is None:
        raise HttpError(404, "build a map first")
    anchors = []
    for a in b.get("anchors", []):
        try:
            anchors.append({"id": str(a["id"]), "machine_x_mm": float(a["machine_x_mm"]),
                            "machine_y_mm": float(a["machine_y_mm"])})
        except (KeyError, TypeError, ValueError):
            raise HttpError(400, "each anchor needs a tube id and machine X/Y in mm")
    m["anchors"] = anchors
    m["anchor_allow_scale"] = bool(b.get("allow_scale", False))
    m["transform"] = mapping.fit_anchor_transform(m["sites"], anchors, m["anchor_allow_scale"]) if anchors else None
    mapping.apply_anchor_transform(m["sites"], m["transform"])
    dataset.save_map(cid, m)
    return m


@route("POST", r"/api/captures/(?P<cid>[\w.-]+)/calibrate_depth")
def r_cal_depth(app, h, q, cid):
    b = h.json_body()
    main = dataset.load_map(cid)
    if not main or main.get("mode") != "known":
        raise HttpError(409, "Build a known-geometry map first (on a sheet whose pitch you measured).")
    target = cid
    if b.get("target") == "companion":
        target = (main.get("companion") or {}).get("capture_id")
        if not target:
            raise HttpError(409, "This map has no companion depth camera.")
    m = dataset.load_map(target)
    if not m or not m.get("depth_pitch"):
        raise HttpError(409, "That camera's map has no depth-based pitch to calibrate from.")
    meta = dataset.load_meta(target)
    old = float(meta.get("depth_scale_factor", 1.0) or 1.0)
    factor = old * main["geometry"]["pitch_mm"] / m["depth_pitch"]["pitch_mm"]
    if not 0.7 < factor < 1.3:
        raise HttpError(409, f"Correction {factor:.3f} is implausible; check the entered pitch.")
    dataset.set_depth_scale(target, factor)
    dev = (meta.get("device") or {}).get("id")
    updated_device = False
    if dev:
        try:
            app.devices.update(dev, settings={"depth_scale_factor": round(factor, 5)})
            updated_device = True
        except KeyError:
            pass
    m2 = app.build_map(cid, main.get("geometry_request") or {}, main.get("roi"))
    return {"factor": factor, "previous": old, "device_updated": updated_device, "target": target, "map": m2}


@route("GET", r"/api/captures/(?P<cid>[\w.-]+)/export\.zip")
def r_cap_zip(app, h, q, cid):
    h.send_bytes(dataset.export_capture(cid), "application/zip",
                 extra={"Content-Disposition": f'attachment; filename="{cid}.zip"'})


# ---------------------------------------------------------------- import/export
@route("POST", r"/api/import/image")
def r_import_img(app, h, q):
    raw = h.raw_body()
    name = unquote(h.headers.get("X-Filename", "upload"))
    try:
        meta = dataset.import_image(raw, name)
    except ValueError as exc:
        raise HttpError(400, str(exc))
    return dataset.summary(meta["id"])


@route("POST", r"/api/import/zip")
def r_import_zip(app, h, q):
    try:
        return dataset.import_zip(h.raw_body())
    except (ValueError, KeyError) as exc:
        raise HttpError(400, f"Could not import: {exc}")


@route("GET", r"/api/export/dataset\.zip")
def r_export(app, h, q):
    data, n = dataset.export_dataset(reviewed_only=q.get("all", ["0"])[0] != "1")
    h.send_bytes(data, "application/zip",
                 extra={"Content-Disposition": f'attachment; filename="tubemap_dataset_{time.strftime("%Y%m%d")}.zip"',
                        "X-Samples": str(n)})


@route("GET", r"/api/dataset/stats")
def r_stats(app, h, q):
    return dataset.stats()


# ---------------------------------------------------------------- learning
@route("GET", r"/api/profiles")
def r_profiles(app, h, q):
    return {"active": config.load_settings().get("active_profile"), "profiles": learning.list_profiles()}


@route("POST", r"/api/profiles/train")
def r_train(app, h, q):
    b = h.json_body()
    try:
        return app.jobs.start("train", learning.train_profile, name=b.get("name") or None)
    except RuntimeError as exc:
        raise HttpError(409, str(exc))


@route("POST", r"/api/profiles/(?P<name>[\w.-]+)/activate")
def r_activate(app, h, q, name):
    try:
        learning.activate(name)
    except FileNotFoundError as exc:
        raise HttpError(404, str(exc))
    return {"active": name}


@route("GET", r"/api/jobs/(?P<jid>\w+)")
def r_job(app, h, q, jid):
    j = app.jobs.get(jid)
    if j is None:
        raise HttpError(404, "job not found")
    return j


# ---------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = f"TubeMap/{VERSION}"
    protocol_version = "HTTP/1.1"
    app: App = None

    def log_message(self, fmt, *args):
        pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise HttpError(413, "upload too large")
        return self.rfile.read(n) if n else b""

    def raw_body(self):
        return self._body()

    def json_body(self):
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise HttpError(400, "request body is not valid JSON")

    def send_bytes(self, data, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, default=_json_default).encode(), "application/json", code)

    def send_file(self, path: Path):
        if not path.exists():
            raise HttpError(404, "not found")
        # Fixed table: on Windows, mimetypes reads the registry, which often maps .js/.css to
        # text/plain and makes browsers refuse the stylesheet.
        ctype = MIME.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), ctype)

    def _dispatch(self, method):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if method == "GET" and not u.path.startswith("/api/"):
                return self._static(u.path)
            for m, rx, fn in ROUTES:
                if m != method:
                    continue
                mt = rx.match(u.path)
                if mt:
                    res = fn(self.app, self, q, **mt.groupdict())
                    if res is not None:
                        self.send_json(res)
                    return
            raise HttpError(404, f"no route for {method} {u.path}")
        except HttpError as e:
            self.send_json({"error": e.msg}, e.code)
        except (KeyError, FileNotFoundError) as e:
            self.send_json({"error": str(e).strip("'\"")}, 404)
        except (TimeoutError, RuntimeError) as e:
            self.send_json({"error": str(e)}, 503)
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            try:
                self.send_json({"error": f"internal error: {e}"}, 500)
            except Exception:
                pass

    def _static(self, path):
        if path in ("/", "/index.html"):
            path = "/index.html"
        p = (STATIC / path.lstrip("/")).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file():
            raise HttpError(404, "not found")
        self.send_file(p)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def serve(host="0.0.0.0", port=None, start_devices=True):
    port = int(port if port is not None else config.load_settings().get("port", 8088))
    srv = ThreadingHTTPServer((host, port), Handler)      # bind first: fail before touching cameras
    srv.daemon_threads = True
    app = App()
    Handler.app = app
    if start_devices:
        app.devices.start()
    return app, srv

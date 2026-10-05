"""Runs each configured device in its own thread and keeps its latest frame."""
from __future__ import annotations

import queue
import re
import threading
import time
import traceback
import uuid

import cv2

from . import config
from .devices.base import COMMON_FIELDS
from .devices.registry import DRIVERS

ROLES = ["overview", "tool", "depth", "aux"]


class DeviceWorker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.driver_cls = DRIVERS[cfg["driver"]]
        self.state = "stopped"
        self.error = None
        self.info = {}
        self.latest = None
        self.frames = 0
        self.fps = 0.0
        self._fps_t = []
        self._cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._jpeg_cache = {}
        self.driver = None

    # ---- control -----------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"dev-{self.cfg['id']}", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.state = "stopped"

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self.state = "starting"
                self.driver = self.driver_cls(self.cfg.get("settings", {}))
                self.driver.open()
                self.info = dict(self.driver.info)
                self.state, self.error, backoff = "running", None, 1.0
                while not self._stop.is_set():
                    self._handle_commands()
                    fs = self.driver.read(1.0)
                    if fs is not None:
                        self._store(fs)
            except Exception as exc:  # device unplugged, SDK error ...
                msg = f"{exc.__class__.__name__}: {exc}"
                if msg != self.error:
                    print(f"[{self.cfg['id']}] {msg}", flush=True)
                    if not isinstance(exc, (RuntimeError, TimeoutError, OSError)):
                        traceback.print_exc()
                self.error = msg
                self.state = "error"
                self._fail_pending(self.error)
            finally:
                try:
                    if self.driver:
                        self.driver.close()
                except Exception:
                    pass
            if not self._stop.wait(backoff):
                backoff = min(10.0, backoff * 2)
        self._fail_pending("device stopped")

    def _store(self, fs):
        self.latest = fs
        self.frames += 1
        now = time.time()
        self._fps_t = [t for t in self._fps_t if now - t < 3.0] + [now]
        if len(self._fps_t) > 1:
            self.fps = (len(self._fps_t) - 1) / (self._fps_t[-1] - self._fps_t[0] + 1e-9)
        self._jpeg_cache = {}

    def _handle_commands(self):
        while True:
            try:
                cmd, arg, box = self._cmds.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd == "capture":
                    fs = self.driver.capture(arg)
                    self._store(fs)
                    box["result"] = fs
                elif cmd == "apply":
                    box["result"] = self.driver.apply_live(arg)
            except Exception as exc:
                box["error"] = f"{exc.__class__.__name__}: {exc}"
            finally:
                box["event"].set()

    def _fail_pending(self, msg):
        while True:
            try:
                _, _, box = self._cmds.get_nowait()
            except queue.Empty:
                return
            box["error"] = msg
            box["event"].set()

    def _call(self, cmd, arg, timeout):
        if self.state != "running":
            raise RuntimeError(f"device is {self.state}" + (f": {self.error}" if self.error else ""))
        box = {"event": threading.Event()}
        self._cmds.put((cmd, arg, box))
        if not box["event"].wait(timeout):
            raise TimeoutError(f"device did not respond to {cmd}")
        if "error" in box:
            raise RuntimeError(box["error"])
        return box.get("result")

    def capture(self, timeout=8.0):
        return self._call("capture", timeout - 1.0, timeout)

    def apply_live(self, changed, timeout=5.0):
        return bool(self._call("apply", changed, timeout))

    def jpeg(self, max_width=1024, quality=80):
        fs = self.latest
        if fs is None:
            return None
        key = (fs.seq, max_width)
        if key in self._jpeg_cache:
            return self._jpeg_cache[key]
        img = fs.color
        if img.shape[1] > max_width:
            s = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        data = buf.tobytes() if ok else None
        self._jpeg_cache[key] = data
        return data

    def status(self):
        fs = self.latest
        return {**{k: self.cfg[k] for k in ("id", "name", "driver", "enabled", "role")},
                "settings": self.cfg.get("settings", {}), "state": self.state, "error": self.error,
                "info": self.info, "fps": round(self.fps, 1), "frames": self.frames,
                "has_depth": bool(fs is not None and fs.depth is not None),
                "resolution": list(fs.color.shape[1::-1]) if fs is not None else None,
                "intrinsics": bool(fs is not None and fs.intrinsics),
                "stats": dict(getattr(self.driver, "stats", {}) or {}) if self.driver else {}}


class DeviceManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.workers: dict[str, DeviceWorker] = {}
        self.configs = config.load_devices()
        for c in self.configs:
            c["settings"] = DRIVERS[c["driver"]].normalize(c.get("settings")) if c["driver"] in DRIVERS else {}

    def start(self):
        for c in self.configs:
            if c.get("enabled") and c["driver"] in DRIVERS:
                self._start(c)

    def _start(self, c):
        w = DeviceWorker(c)
        self.workers[c["id"]] = w
        w.start()

    def shutdown(self):
        for w in list(self.workers.values()):
            w.stop()

    def _save(self):
        config.save_devices(self.configs)

    def get_cfg(self, dev_id):
        for c in self.configs:
            if c["id"] == dev_id:
                return c
        raise KeyError(dev_id)

    def worker(self, dev_id) -> DeviceWorker:
        w = self.workers.get(dev_id)
        if w is None:
            raise KeyError(f"device {dev_id} is not running")
        return w

    def list(self):
        out = []
        for c in self.configs:
            w = self.workers.get(c["id"])
            if w:
                out.append(w.status())
            else:
                out.append({**c, "state": "disabled" if not c.get("enabled") else "stopped", "error": None,
                            "info": {}, "fps": 0, "frames": 0, "has_depth": False, "resolution": None})
        return out

    def add(self, name, driver, settings, role="aux", enabled=True):
        if driver not in DRIVERS:
            raise ValueError(f"unknown driver {driver}")
        if role not in ROLES:
            role = "aux"
        base = re.sub(r"[^a-z0-9]+", "-", driver.lower()).strip("-") or "dev"
        dev_id = f"{base}-{uuid.uuid4().hex[:6]}"
        c = {"id": dev_id, "name": (name or DRIVERS[driver].label)[:60], "driver": driver,
             "enabled": bool(enabled), "role": role, "settings": DRIVERS[driver].normalize(settings)}
        with self.lock:
            self.configs.append(c)
            self._save()
            if c["enabled"]:
                self._start(c)
        return c

    def update(self, dev_id, name=None, settings=None, role=None, enabled=None):
        with self.lock:
            c = self.get_cfg(dev_id)
            if name is not None:
                c["name"] = str(name)[:60]
            if role in ROLES:
                c["role"] = role
            restart = False
            applied_live = False
            if settings is not None:
                new = DRIVERS[c["driver"]].normalize({**c["settings"], **settings})
                changed = {k: v for k, v in new.items() if c["settings"].get(k) != v}
                c["settings"] = new
                w = self.workers.get(dev_id)
                mapping_only = {f["key"] for f in COMMON_FIELDS}
                hw = {k: v for k, v in changed.items() if k not in mapping_only}
                if hw and w and w.state == "running":
                    live = w.driver.live_keys() if w.driver else set()
                    if set(hw) <= live:
                        try:
                            applied_live = w.apply_live(hw)
                        except Exception:
                            applied_live = False
                    restart = not applied_live
                elif hw:
                    restart = True
            if enabled is not None and bool(enabled) != bool(c.get("enabled")):
                c["enabled"] = bool(enabled)
                restart = True
            self._save()
            w = self.workers.pop(dev_id, None)
            if restart or not c.get("enabled"):
                if w:
                    w.stop()
                if c.get("enabled"):
                    self._start(c)
            elif w:
                w.cfg = c
                self.workers[dev_id] = w
        return {"device": c, "restarted": restart, "applied_live": applied_live}

    def remove(self, dev_id):
        with self.lock:
            c = self.get_cfg(dev_id)
            w = self.workers.pop(dev_id, None)
            if w:
                w.stop()
            self.configs.remove(c)
            self._save()

    def restart(self, dev_id):
        with self.lock:
            c = self.get_cfg(dev_id)
            w = self.workers.pop(dev_id, None)
            if w:
                w.stop()
            if c.get("enabled"):
                self._start(c)

    def test(self, driver, settings, timeout=15.0):
        """Open a device briefly with the given settings. The device must not be in use."""
        cls = DRIVERS[driver]
        d = cls(settings)
        t0 = time.time()
        d.open()
        try:
            fs = None
            while fs is None and time.time() - t0 < timeout:
                fs = d.read(2.0)
            if fs is None:
                raise TimeoutError("opened, but no frame arrived")
            img = fs.color
            s = min(1.0, 480 / img.shape[1])
            thumb = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s))[1].tobytes()
            return {"info": d.info, "resolution": list(img.shape[1::-1]),
                    "depth": None if fs.depth is None else list(fs.depth.shape[::-1]),
                    "intrinsics": bool(fs.intrinsics), "seconds": round(time.time() - t0, 2)}, thumb
        finally:
            d.close()

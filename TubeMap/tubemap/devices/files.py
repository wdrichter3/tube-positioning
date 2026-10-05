"""Replay images from a folder as if they came from a camera (bench testing)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from .base import DeviceDriver, FrameSet, field_spec

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FolderDriver(DeviceDriver):
    driver_id = "folder"
    label = "Image folder"
    description = ("Plays back images from a folder on this computer. A depth.png (16-bit mm) and "
                   "intrinsics.json next to an image are used if present.")

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("path", "Folder", "text", "", "Absolute path to a folder of images."),
            field_spec("interval_s", "Seconds per image", "float", 3.0, minimum=0.2, maximum=60, live=True),
        ]

    def open(self):
        p = Path(self.settings["path"]).expanduser()
        if not p.is_dir():
            raise RuntimeError(f"folder not found: {p}")
        self.files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS and not f.name.startswith("depth"))
        if not self.files:
            raise RuntimeError("no images in folder")
        self.k, self.seq, self.t0 = 0, 0, 0.0
        self.info = {"model": "Folder", "serial": str(p), "images": len(self.files)}

    def read(self, timeout=2.0):
        if time.time() - self.t0 >= float(self.settings["interval_s"]):
            self.k = (self.k + 1) % len(self.files) if self.seq else 0
            self.t0 = time.time()
        else:
            time.sleep(0.2)
        f = self.files[self.k]
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        depth = None
        for dn in ("depth.png", f.stem + "_depth.png"):
            dp = f.parent / dn
            if dp.exists():
                depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
                break
        intr = None
        ij = f.parent / "intrinsics.json"
        if ij.exists():
            intr = json.loads(ij.read_text())
        self.seq += 1
        return FrameSet(color=img, depth=depth, intrinsics=intr, seq=self.seq, meta={"source_file": f.name})

    def apply_live(self, changed):
        if set(changed) <= self.live_keys():
            self.settings.update(changed)
            return True
        return False

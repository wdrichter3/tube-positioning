"""USB webcams, capture cards and RTSP/HTTP streams through OpenCV."""
from __future__ import annotations

import json

import cv2

from .base import DeviceDriver, FrameSet, field_spec


class UvcDriver(DeviceDriver):
    driver_id = "uvc"
    label = "USB / network video (OpenCV)"
    description = "Any camera OpenCV can open: USB index (0, 1, ...), RTSP or HTTP stream URL. Colour only."
    has_depth = False

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("source", "Source", "text", "0", "Camera index such as 0, or a stream URL."),
            field_spec("width", "Width", "int", 1920, minimum=160, maximum=7680),
            field_spec("height", "Height", "int", 1080, minimum=120, maximum=4320),
            field_spec("fps", "Frame rate", "float", 15, minimum=1, maximum=60),
            field_spec("fourcc", "Pixel format", "select", "MJPG", options=["MJPG", "YUYV", "default"], advanced=True),
            field_spec("intrinsics_json", "Calibration (JSON)", "text", "",
                       'Optional: {"fx":..,"fy":..,"cx":..,"cy":..,"dist":[k1,k2,p1,p2,k3],"width":..,"height":..} '
                       "from a checkerboard calibration.", advanced=True),
        ]

    @classmethod
    def discover(cls):
        out = []
        for k in range(4):
            cap = cv2.VideoCapture(k)
            ok = cap.isOpened()
            cap.release()
            if ok:
                out.append({"label": f"USB camera {k}", "settings": {"source": str(k)}})
        return out

    def open(self):
        s = self.settings
        src = int(s["source"]) if s["source"].strip().isdigit() else s["source"].strip()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {s['source']!r}")
        if s["fourcc"] != "default":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s["fourcc"][:4].ljust(4)))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, s["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s["height"])
        self.cap.set(cv2.CAP_PROP_FPS, s["fps"])
        self.intr = None
        if s["intrinsics_json"].strip():
            self.intr = json.loads(s["intrinsics_json"])
        self.seq = 0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info = {"model": "OpenCV video", "serial": str(s["source"]), "color_size": [w, h]}

    def close(self):
        self.cap.release()

    def read(self, timeout=2.0):
        ok, img = self.cap.read()
        if not ok:
            return None
        self.seq += 1
        return FrameSet(color=img, depth=None, intrinsics=self.intr, seq=self.seq)

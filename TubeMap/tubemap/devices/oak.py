"""Luxonis OAK (DepthAI v2 API) driver - OAK-D Pro and similar.

Colour comes from the full-sensor ISP output; stereo depth is aligned to the
colour camera. Pairs are matched by device timestamp. Intrinsics and lens
distortion are read from the device's own calibration EEPROM.
"""
from __future__ import annotations

import threading
import time
import warnings
from collections import deque

from .base import DeviceDriver, FrameSet, field_spec

ISP_OPTIONS = [
    {"value": "4/13", "label": "1248 x 936 (fast, depth at full size)"},
    {"value": "1/2", "label": "2028 x 1520 (depth 1248 x 936)"},
    {"value": "1/1", "label": "4056 x 3040 full sensor (slow; depth 1248 x 936)"},
]


def _dai():
    import depthai as dai  # noqa: WPS433
    return dai


def _socket(dai, name):
    s = dai.CameraBoardSocket
    alias = {"CAM_A": "RGB", "CAM_B": "LEFT", "CAM_C": "RIGHT"}
    return getattr(s, name, None) or getattr(s, alias[name])


def build_pipeline(dai, s: dict):
    """Construct the DepthAI graph. Works without a device attached."""
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setBoardSocket(_socket(dai, "CAM_A"))
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    num, den = (int(x) for x in s["isp_scale"].split("/"))
    if (num, den) != (1, 1):
        cam.setIspScale(num, den)
    cam.setFps(float(s["fps"]))
    w, h = (int(v) for v in cam.getIspSize())
    xrgb = p.create(dai.node.XLinkOut)
    xrgb.setStreamName("rgb")
    cam.isp.link(xrgb.input)
    xctl = p.create(dai.node.XLinkIn)
    xctl.setStreamName("control")
    xctl.out.link(cam.inputControl)
    depth_size = None
    if s["depth_enabled"]:
        res = {"400P": "THE_400_P", "720P": "THE_720_P", "800P": "THE_800_P"}[s["mono_resolution"]]
        left = p.create(dai.node.MonoCamera)
        right = p.create(dai.node.MonoCamera)
        for node, sock in ((left, "CAM_B"), (right, "CAM_C")):
            node.setBoardSocket(_socket(dai, sock))
            node.setResolution(getattr(dai.MonoCameraProperties.SensorResolution, res))
            node.setFps(float(s["fps"]))
        st = p.create(dai.node.StereoDepth)
        pm = dai.node.StereoDepth.PresetMode
        preset = next(getattr(pm, n) for n in (s["depth_preset"], "DEFAULT", "HIGH_ACCURACY") if hasattr(pm, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            st.setDefaultProfilePreset(preset)
        st.setLeftRightCheck(bool(s["lr_check"]))
        st.setSubpixel(bool(s["subpixel"]))
        st.setExtendedDisparity(bool(s["extended_disparity"]))
        st.setDepthAlign(_socket(dai, "CAM_A"))
        if w <= 1280 and w % 16 == 0:
            depth_size = (w, h)
        else:
            depth_size = (1248, 936)
        if s.get("depth_size", "half") == "half":
            # half size is plenty for the sheet plane and cuts USB traffic (laptop ports, long cables)
            dw = max(320, 16 * int(round(depth_size[0] / 2 / 16)))
            depth_size = (dw, int(round(dw * h / w)))
        st.setOutputSize(*depth_size)
        st.setOutputKeepAspectRatio(True)
        left.out.link(st.left)
        right.out.link(st.right)
        xd = p.create(dai.node.XLinkOut)
        xd.setStreamName("depth")
        st.depth.link(xd.input)
    return p, {"color_size": [w, h], "depth_size": list(depth_size) if depth_size else None}


class OakDriver(DeviceDriver):
    driver_id = "oak"
    label = "Luxonis OAK (DepthAI)"
    description = "OAK-D Pro / OAK-D family over USB. Colour + aligned stereo depth + factory calibration."
    has_depth = True

    @classmethod
    def settings_schema(cls):
        return [
            field_spec("mxid", "Device ID (MxID)", "text", "", "Leave blank to use the first OAK found. "
                       "Use Discover to list connected devices."),
            field_spec("isp_scale", "Colour resolution", "select", "4/13", options=ISP_OPTIONS),
            field_spec("fps", "Frame rate", "float", 10.0, minimum=1, maximum=30),
            field_spec("depth_enabled", "Stereo depth", "bool", True),
            field_spec("depth_size", "Depth image size", "select", "half",
                       "Half is enough for the sheet plane and pitch check and halves USB traffic.",
                       options=[{"value": "half", "label": "Half colour size (recommended)"},
                                {"value": "full", "label": "Same as colour (up to 1248 x 936)"}], advanced=True),
            field_spec("mono_resolution", "Stereo sensor resolution", "select", "800P",
                       options=["400P", "720P", "800P"], advanced=True),
            field_spec("depth_preset", "Depth preset", "select", "DEFAULT",
                       "DEFAULT/HIGH_DETAIL on DepthAI 2.30+; older versions fall back to HIGH_ACCURACY.",
                       options=["DEFAULT", "HIGH_DETAIL", "ROBOTICS", "HIGH_ACCURACY", "HIGH_DENSITY"], advanced=True),
            field_spec("subpixel", "Subpixel disparity", "bool", True, advanced=True),
            field_spec("lr_check", "Left-right check", "bool", True, advanced=True),
            field_spec("extended_disparity", "Extended disparity (close range)", "bool", False, advanced=True),
            field_spec("ir_dot_ma", "IR dot projector (mA)", "int", 600, "OAK-D Pro only. Improves depth on plain steel.",
                       minimum=0, maximum=1200, live=True),
            field_spec("ir_flood_ma", "IR flood light (mA)", "int", 0, minimum=0, maximum=1500, live=True, advanced=True),
            field_spec("focus_mode", "Focus", "select", "calibrated",
                       options=[{"value": "calibrated", "label": "Locked at factory calibration (recommended)"},
                                {"value": "manual", "label": "Locked at lens position below"},
                                {"value": "auto", "label": "Continuous auto (intrinsics drift)"}], live=True),
            field_spec("lens_position", "Lens position (0-255)", "int", 130, minimum=0, maximum=255, live=True),
            field_spec("exposure_mode", "Exposure", "select", "auto", options=["auto", "manual"], live=True),
            field_spec("exposure_us", "Exposure time (us)", "int", 8000, minimum=50, maximum=33000, live=True),
            field_spec("iso", "ISO", "int", 400, minimum=100, maximum=1600, live=True),
            field_spec("usb2_mode", "Force USB2", "bool", False, "Only for long or unreliable cables.", advanced=True),
        ]

    @classmethod
    def available(cls):
        try:
            dai = _dai()
            return True, f"depthai {getattr(dai, '__version__', '?')}"
        except Exception as exc:  # pragma: no cover - depends on host
            return False, f"depthai not installed ({exc.__class__.__name__}). Run the installer."

    @classmethod
    def discover(cls):
        dai = _dai()
        out = []
        for d in dai.Device.getAllAvailableDevices():
            mx = d.getMxId() if hasattr(d, "getMxId") else getattr(d, "mxid", "")
            out.append({"label": f"OAK {mx} ({getattr(d, 'name', '')}, {str(getattr(d, 'state', ''))})",
                        "settings": {"mxid": mx}})
        return out

    # ---------------------------------------------------------------------
    def open(self):
        dai = _dai()
        s = self.settings
        pipeline, geo = build_pipeline(dai, s)
        if s["mxid"]:
            info = dai.DeviceInfo(s["mxid"])
            self.dev = dai.Device(pipeline, info, bool(s["usb2_mode"]))
        else:
            self.dev = dai.Device(pipeline, bool(s["usb2_mode"])) if s["usb2_mode"] else dai.Device(pipeline)
        self.q_rgb = self.dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.q_depth = self.dev.getOutputQueue("depth", maxSize=4, blocking=False) if s["depth_enabled"] else None
        self.q_ctl = self.dev.getInputQueue("control")
        calib = self.dev.readCalibration()
        w, h = geo["color_size"]
        cam_a = _socket(dai, "CAM_A")
        M = calib.getCameraIntrinsics(cam_a, w, h)
        dist = list(calib.getDistortionCoefficients(cam_a))
        self.intr = {"width": w, "height": h, "fx": float(M[0][0]), "fy": float(M[1][1]),
                     "cx": float(M[0][2]), "cy": float(M[1][2]), "dist": [float(x) for x in dist],
                     "model": "opencv14" if len(dist) >= 14 else "opencv"}
        try:
            self.eeprom_lens = int(calib.getLensPosition(cam_a))
        except Exception:
            self.eeprom_lens = 0
        self.rgb = deque(maxlen=8)
        self.dep = deque(maxlen=8)
        self.stats = {"rgb_frames": 0, "depth_frames": 0, "paired": 0, "unpaired": 0, "last_pair_dt_ms": None}
        self.lock = threading.Lock()
        self.seq = 0
        try:
            speed = self.dev.getUsbSpeed().name
        except Exception:
            speed = "?"
        self.info = {"model": self._name(), "serial": self._mxid(), "usb_speed": speed,
                     "color_size": [w, h], "depth_size": geo["depth_size"],
                     "eeprom_lens_position": self.eeprom_lens}
        self._apply_controls(s)
        self._apply_ir(s)

    def _name(self):
        for attr in ("getDeviceName", "getProductName"):
            try:
                return str(getattr(self.dev, attr)())
            except Exception:
                continue
        return "OAK"

    def _mxid(self):
        try:
            return str(self.dev.getMxId())
        except Exception:
            return self.settings.get("mxid", "")

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass

    def _apply_ir(self, s):
        for fn, val in (("setIrLaserDotProjectorBrightness", s["ir_dot_ma"]),
                        ("setIrFloodLightBrightness", s["ir_flood_ma"])):
            try:
                getattr(self.dev, fn)(float(val))
            except Exception:
                pass

    def _apply_controls(self, s):
        dai = _dai()
        ctl = dai.CameraControl()
        if s["focus_mode"] == "auto":
            ctl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        else:
            lens = self.eeprom_lens if (s["focus_mode"] == "calibrated" and self.eeprom_lens > 0) else s["lens_position"]
            ctl.setManualFocus(int(lens))
        if s["exposure_mode"] == "manual":
            ctl.setManualExposure(int(s["exposure_us"]), int(s["iso"]))
        else:
            ctl.setAutoExposureEnable()
        self.q_ctl.send(ctl)

    def apply_live(self, changed):
        if not set(changed) <= self.live_keys():
            return False
        self.settings.update(changed)
        self._apply_controls(self.settings)
        self._apply_ir(self.settings)
        return True

    def _pump(self):
        for m in self.q_rgb.tryGetAll() or []:
            self.rgb.append((m.getTimestamp().total_seconds(), m))
            self.stats["rgb_frames"] += 1
        if self.q_depth is not None:
            for m in self.q_depth.tryGetAll() or []:
                self.dep.append((m.getTimestamp().total_seconds(), m))
                self.stats["depth_frames"] += 1

    def _pair(self, strict, loose=False):
        """Pair the newest colour frame that has a depth frame from the same moment.

        Stereo depth reaches the host later than colour, so the newest colour frame usually has
        no partner yet; looking back through the short buffer finds the latest matched pair.
        """
        if not self.rgb:
            return None
        period = 1.0 / max(1.0, float(self.settings["fps"]))
        tol = (1.0 if loose else 0.5) * period + 0.005
        chosen, depth, dt = None, None, None
        if self.q_depth is not None and self.dep:
            for ts, msg in reversed(self.rgb):
                dts, dmsg = min(self.dep, key=lambda x: abs(x[0] - ts))
                if abs(dts - ts) <= tol:
                    chosen, depth, dt = (ts, msg), dmsg.getFrame(), abs(dts - ts)
                    break
        if chosen is None:
            if strict and self.q_depth is not None:
                return None
            chosen = self.rgb[-1]
        ts, msg = chosen
        if dt is not None:
            self.stats["paired"] += 1
            self.stats["last_pair_dt_ms"] = round(1000 * dt, 1)
        else:
            self.stats["unpaired"] += 1
        self.seq += 1
        return FrameSet(color=msg.getCvFrame(), depth=depth, intrinsics=dict(self.intr), seq=self.seq,
                        meta={"device_ts": ts, "rgb_depth_dt_ms": None if dt is None else round(1000 * dt, 2),
                              "depth_aligned": depth is not None,
                              "lens_position": self.eeprom_lens if self.settings["focus_mode"] == "calibrated" else None})

    def read(self, timeout=2.0):
        end = time.time() + timeout
        n0 = len(self.rgb) and self.rgb[-1][0]
        while time.time() < end:
            self._pump()
            if self.rgb and self.rgb[-1][0] != n0:
                return self._pair(strict=False)
            time.sleep(0.005)
        return None

    def capture(self, timeout=5.0):
        """Best matched colour/depth pair. If depth never lines up (or isn't arriving), the colour
        frame is still saved, with the reason recorded, instead of failing the capture."""
        t0 = time.time()
        strict_until = t0 + min(2.5, 0.5 * timeout)
        start_ts = self.rgb[-1][0] if self.rgb else -1
        while time.time() < t0 + timeout:
            self._pump()
            if self.rgb and self.rgb[-1][0] != start_ts:
                fs = self._pair(strict=True)
                if fs is not None:
                    return fs
                if time.time() > strict_until:
                    break
            time.sleep(0.005)
        if not self.rgb:
            raise TimeoutError("OAK sent no colour frames. Unplug and replug the camera, or use a USB 3 port.")
        fs = self._pair(strict=False, loose=True)
        st = self.stats
        if fs.depth is None:
            why = ("no depth frames are arriving (stereo depth off, or USB too slow: check the USB speed on "
                   "the Devices tab)" if st["depth_frames"] == 0 else
                   "depth frames arrive but none line up in time with colour")
            fs.meta["depth_missing"] = why
            fs.meta["depth_aligned"] = False
        return fs
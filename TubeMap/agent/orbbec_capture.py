"""Orbbec colour + aligned depth capture.

Works with both Python wrappers:
  * pyorbbecsdk v1 (Orbbec SDK 1.x) - Astra+, Astra Pro Plus, Gemini 2 XL
  * pyorbbecsdk v2 (pyorbbecsdk2)   - Gemini 2/330, Femto, Astra 2 ...
Both install as the module ``pyorbbecsdk``; only one can be installed per Python environment.

Shared by the local USB driver and the network agent that runs on the Persee 2, so it stays
compatible with Python 3.6 (no dataclasses, walrus operators or postponed annotations).
"""
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _ob():
    import pyorbbecsdk as ob
    return ob


def sdk_generation():
    """1 for the v1 wrapper, 2 for pyorbbecsdk2, None if not installed."""
    try:
        ob = _ob()
    except Exception:
        return None
    return 2 if hasattr(ob.Pipeline, "get_d2c_depth_profile_list") else 1


def list_devices():
    ob = _ob()
    ctx = ob.Context()
    dl = ctx.query_devices()
    out = []
    for k in range(dl.get_count()):
        sn, name = "", "Orbbec"
        try:
            sn = dl.get_device_serial_number_by_index(k)
        except Exception:
            pass
        try:
            name = dl.get_device_name_by_index(k)
        except Exception:
            try:
                name = dl.get_device_by_index(k).get_device_info().get_name()
            except Exception:
                pass
        out.append({"serial": sn, "name": name, "index": k})
    return out


def _decode_color(frame):
    ob = _ob()
    w, h = frame.get_width(), frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == ob.OBFormat.RGB:
        img = data.reshape(h, w, 3)[:, :, ::-1].copy()
    elif fmt == ob.OBFormat.BGR:
        img = data.reshape(h, w, 3).copy()
    elif fmt == ob.OBFormat.MJPG:
        img = cv2.imdecode(data.reshape(-1).astype(np.uint8), cv2.IMREAD_COLOR)
    elif fmt in (ob.OBFormat.YUYV, getattr(ob.OBFormat, "YUY2", -99)):
        img = cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    elif fmt == ob.OBFormat.UYVY:
        img = cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == ob.OBFormat.NV12:
        img = cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    elif fmt == ob.OBFormat.NV21:
        img = cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    elif fmt == ob.OBFormat.I420:
        img = cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    else:
        raise RuntimeError("unsupported colour format %s" % fmt)
    if img is None:
        raise RuntimeError("colour frame could not be decoded")
    return img


def _decode_depth(frame):
    w, h = frame.get_width(), frame.get_height()
    raw = np.asanyarray(frame.get_data()).reshape(-1)
    if raw.dtype != np.uint16:
        raw = raw.view(np.uint16) if raw.nbytes == w * h * 2 else raw
    if raw.size != w * h:
        raise RuntimeError("depth frame is %d values for %dx%d; packed depth formats are not supported - "
                           "pick a Y16 depth mode" % (raw.size, w, h))
    d = raw.reshape(h, w).astype(np.float32)
    try:
        d *= float(frame.get_depth_scale())
    except Exception:
        pass
    return np.clip(d, 0, 65535).astype(np.uint16)


def _intr_dict(ci, cd, fallback_w, fallback_h):
    if ci is None or not getattr(ci, "fx", 0):
        return None
    w = int(getattr(ci, "width", 0) or fallback_w)
    h = int(getattr(ci, "height", 0) or fallback_h)
    dist = []
    if cd is not None:
        dist = [cd.k1, cd.k2, cd.p1, cd.p2, cd.k3, cd.k4, cd.k5, cd.k6]
    return {"width": w, "height": h, "fx": float(ci.fx), "fy": float(ci.fy), "cx": float(ci.cx),
            "cy": float(ci.cy), "dist": [float(x) for x in dist], "model": "opencv"}


def _color_intrinsics(profile, pipeline, w, h):
    try:
        return _intr_dict(profile.get_intrinsic(), profile.get_distortion(), w, h)
    except Exception:
        pass
    try:
        cp = pipeline.get_camera_param()
        return _intr_dict(cp.rgb_intrinsic, cp.rgb_distortion, w, h)
    except Exception:
        return None


def _set_prop(dev, name, value):
    ob = _ob()
    pid = getattr(ob.OBPropertyID, name, None)
    if pid is None:
        return False
    try:
        if not dev.is_property_supported(pid, ob.OBPermissionType.PERMISSION_WRITE):
            return False
    except Exception:
        pass
    try:
        if isinstance(value, bool):
            dev.set_bool_property(pid, value)
        else:
            dev.set_int_property(pid, int(value))
        return True
    except Exception:
        return False


class OrbbecCapture(object):
    """align: 'auto' (hardware, then software), 'hw', 'sw' or 'off'."""

    def __init__(self, serial="", width=1280, height=800, fps=15, depth=True, align="auto",
                 unmirror=True, depth_width=0, depth_height=0, laser=True):
        self.serial, self.width, self.height, self.fps = serial, int(width), int(height), int(fps)
        self.use_depth, self.align_req, self.unmirror, self.laser = bool(depth), align, bool(unmirror), bool(laser)
        self.depth_w, self.depth_h = int(depth_width or 0), int(depth_height or 0)
        self.pipeline = None
        self.filter = None
        self.intr = None
        self.info = {}
        self.depth_aligned = False
        self.align_used = "off"

    # ------------------------------------------------------------------
    def _device(self):
        ob = _ob()
        dl = ob.Context().query_devices()
        if dl.get_count() == 0:
            raise RuntimeError("no Orbbec device found (check USB 3 cable and udev rules)")
        return dl.get_device_by_serial_number(self.serial) if self.serial else dl.get_device_by_index(0)

    def _color_profile(self, pipeline):
        """Requested size first at the requested rate, then at any rate the camera offers
        (Astra+ 1920x1080 is not available at every rate), then the camera default."""
        ob = _ob()
        lst = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        fmts = [f for f in (ob.OBFormat.RGB, ob.OBFormat.MJPG, ob.OBFormat.YUYV, getattr(ob.OBFormat, "BGR", None))
                if f is not None]
        for fps in (self.fps, 0):
            for fmt in fmts:
                try:
                    return lst.get_video_stream_profile(self.width, self.height, fmt, fps)
                except Exception:
                    continue
        self.color_note = "requested %dx%d not offered by this camera; using its default" % (self.width, self.height)
        return lst.get_default_video_stream_profile()

    def _depth_profile(self, pipeline, cprof, mode):
        ob = _ob()
        if mode == "hw" and hasattr(pipeline, "get_d2c_depth_profile_list"):
            try:
                lst = pipeline.get_d2c_depth_profile_list(cprof, ob.OBAlignMode.HW_MODE)
                if lst.get_count() > 0:
                    return lst.get_default_video_stream_profile()
            except Exception:
                pass
        lst = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        for fps in (self.fps, 0):
            for fmt in (ob.OBFormat.Y16, getattr(ob.OBFormat, "Y12", None), getattr(ob.OBFormat, "Y11", None)):
                if fmt is None:
                    continue
                try:
                    return lst.get_video_stream_profile(self.depth_w, self.depth_h, fmt, fps)
                except Exception:
                    continue
        return lst.get_default_video_stream_profile()

    def _start(self, dev, mode):
        ob = _ob()
        self.pipeline = ob.Pipeline(dev)
        cfg = ob.Config()
        cprof = self._color_profile(self.pipeline)
        cfg.enable_stream(cprof)
        self.filter = None
        if self.use_depth:
            cfg.enable_stream(self._depth_profile(self.pipeline, cprof, mode))
            if mode == "hw":
                cfg.set_align_mode(ob.OBAlignMode.HW_MODE)
            elif mode == "sw":
                cfg.set_align_mode(ob.OBAlignMode.SW_MODE)
            elif mode == "filter":
                self.filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        try:
            self.pipeline.enable_frame_sync()
        except Exception:
            pass
        self.pipeline.start(cfg)
        return cprof

    def _aligned(self, color, depth):
        if depth is None:
            return False
        ch, cw = color.shape[:2]
        dh, dw = depth.shape[:2]
        return abs(cw / float(ch) - dw / float(dh)) < 0.02

    def open(self):
        dev = self._device()
        if self.unmirror:
            _set_prop(dev, "OB_PROP_COLOR_MIRROR_BOOL", False)
            _set_prop(dev, "OB_PROP_DEPTH_MIRROR_BOOL", False)
        if self.use_depth and self.laser:
            _set_prop(dev, "OB_PROP_LASER_BOOL", True)
        modes = {"auto": ["hw", "sw", "filter"], "hw": ["hw"], "sw": ["sw"], "off": ["off"]}.get(self.align_req, ["hw"])
        if not self.use_depth:
            modes = ["off"]
        last_err = None
        cprof = None
        for mode in modes:
            try:
                cprof = self._start(dev, mode)
                if mode == "off":
                    self.align_used = "off"
                    break
                c, d = self._grab_raw(3000, need_depth=True)
                if self._aligned(c, d):
                    self.align_used = mode
                    break
                last_err = "depth %dx%d does not match colour %dx%d with %s alignment" % (
                    d.shape[1], d.shape[0], c.shape[1], c.shape[0], mode)
            except Exception as exc:
                last_err = "%s alignment: %s" % (mode, exc)
            self.close()
            self.pipeline = None
        if self.pipeline is None:
            # keep streaming without alignment rather than failing outright
            cprof = self._start(dev, "off")
            self.align_used = "off"
            self.info_note = last_err
        self.depth_aligned = self.use_depth and self.align_used != "off"
        vp = cprof.as_video_stream_profile() if hasattr(cprof, "as_video_stream_profile") else cprof
        w, h = vp.get_width(), vp.get_height()
        self.intr = _color_intrinsics(vp, self.pipeline, w, h)
        try:
            di = dev.get_device_info()
            self.info = {"model": di.get_name(), "serial": di.get_serial_number(),
                         "firmware": di.get_firmware_version(), "color_size": [w, h]}
        except Exception:
            self.info = {"model": "Orbbec", "serial": self.serial, "color_size": [w, h]}
        self.info.update({"sdk": sdk_generation(), "depth_alignment": self.align_used,
                          "depth_aligned": self.depth_aligned,
                          "color_profile": "%dx%d %s @ %s fps" % (w, h, getattr(vp.get_format(), "name", vp.get_format()),
                                                                 vp.get_fps())})
        if getattr(self, "color_note", None):
            self.info["color_note"] = self.color_note
        if getattr(self, "info_note", None):
            self.info["alignment_note"] = self.info_note

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _grab_raw(self, timeout_ms=1500, need_depth=False):
        end = time.time() + timeout_ms / 1000.0
        while time.time() < end:
            frames = self.pipeline.wait_for_frames(200)
            if frames is None:
                continue
            if self.filter is not None:
                try:
                    out = self.filter.process(frames)
                    frames = out.as_frame_set() if hasattr(out, "as_frame_set") else out
                except Exception:
                    pass
            cf = frames.get_color_frame()
            if cf is None:
                continue
            color = _decode_color(cf)
            depth = None
            df = frames.get_depth_frame() if self.use_depth else None
            if df is not None:
                depth = _decode_depth(df)
            if need_depth and depth is None:
                continue
            return color, depth
        raise RuntimeError("no frames from Orbbec device")

    def grab(self, timeout_ms=1500, need_depth=False):
        color, depth = self._grab_raw(timeout_ms, need_depth)
        if depth is not None and not self._aligned(color, depth):
            self.depth_aligned = False
        return color, depth, dict(self.intr) if self.intr else None

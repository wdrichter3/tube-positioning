"""Minimal stand-in for the Orbbec SDK v1 Python wrapper (Astra+ behaviour), for tests only.

Colour 1920x1080 RGB, depth 640x480 Y16 in 0.1 mm units (depth scale 0.1).
Hardware alignment on this fake does NOT resample depth (simulates a mode the camera can't align),
software alignment returns depth at the colour size - so 'auto' must fall back from hw to sw.
"""
import enum

import numpy as np

STATE = {"mirror_color": True, "mirror_depth": True, "laser": False, "opened": 0, "hw_aligns": False}


class OBFormat(enum.IntEnum):
    UNKNOWN_FORMAT = -1; YUYV = 0; YUY2 = 1; UYVY = 2; NV12 = 3; NV21 = 4; MJPG = 5; H264 = 6; H265 = 7
    Y16 = 8; Y8 = 9; Y10 = 10; Y11 = 11; Y12 = 12; GRAY = 13; HEVC = 14; I420 = 15; RGB = 22; BGR = 23


class OBSensorType(enum.IntEnum):
    COLOR_SENSOR = 2; DEPTH_SENSOR = 3


class OBStreamType(enum.IntEnum):
    COLOR_STREAM = 2; DEPTH_STREAM = 3


class OBAlignMode(enum.IntEnum):
    DISABLE = 0; HW_MODE = 1; SW_MODE = 2


class OBPermissionType(enum.IntEnum):
    PERMISSION_DENY = 0; PERMISSION_READ = 1; PERMISSION_WRITE = 2; PERMISSION_READ_WRITE = 3


class OBPropertyID(enum.IntEnum):
    OB_PROP_COLOR_MIRROR_BOOL = 81; OB_PROP_DEPTH_MIRROR_BOOL = 14; OB_PROP_LASER_BOOL = 3


class _Intr:
    def __init__(self, w, h, f):
        self.width, self.height, self.fx, self.fy, self.cx, self.cy = w, h, f, f, w / 2 - 0.5, h / 2 - 0.5


class _Dist:
    k1 = -0.02; k2 = 0.01; k3 = 0.0; k4 = 0.0; k5 = 0.0; k6 = 0.0; p1 = 0.0; p2 = 0.0


class OBCameraParam:
    def __init__(self):
        self.rgb_intrinsic = _Intr(1920, 1080, 1390.0)
        self.rgb_distortion = _Dist()


class VideoStreamProfile:
    def __init__(self, kind, w, h, fmt, fps):
        self.kind, self.w, self.h, self.fmt, self.fps = kind, w, h, fmt, fps

    def as_video_stream_profile(self):
        return self

    def get_width(self):
        return self.w

    def get_height(self):
        return self.h

    def get_format(self):
        return self.fmt

    def get_fps(self):
        return self.fps

    def get_intrinsic(self):
        raise RuntimeError("not in v1")


class StreamProfileList:
    def __init__(self, kind):
        self.kind = kind
        if kind == OBSensorType.COLOR_SENSOR:
            self.items = [(640, 480, OBFormat.RGB, 30), (1920, 1080, OBFormat.MJPG, 30), (1280, 720, OBFormat.RGB, 30)]
        else:
            self.items = [(640, 480, OBFormat.Y16, 30), (640, 480, OBFormat.Y16, 15), (1280, 1024, OBFormat.Y16, 7)]

    def get_count(self):
        return len(self.items)

    def get_video_stream_profile(self, w, h, fmt, fps):
        for it in self.items:
            if (w in (0, it[0])) and (h in (0, it[1])) and fmt == it[2] and (fps in (0, it[3])):
                return VideoStreamProfile(self.kind, *it)
        raise RuntimeError("profile not found")

    def get_default_video_stream_profile(self):
        return VideoStreamProfile(self.kind, *self.items[0])


class DeviceInfo:
    def get_name(self):
        return "Orbbec Astra+ (fake)"

    def get_serial_number(self):
        return "AY0FAKE001"

    def get_firmware_version(self):
        return "1.0.19"


class Device:
    def get_device_info(self):
        return DeviceInfo()

    def is_property_supported(self, pid, perm):
        return True

    def set_bool_property(self, pid, v):
        if pid == OBPropertyID.OB_PROP_COLOR_MIRROR_BOOL:
            STATE["mirror_color"] = v
        elif pid == OBPropertyID.OB_PROP_DEPTH_MIRROR_BOOL:
            STATE["mirror_depth"] = v
        elif pid == OBPropertyID.OB_PROP_LASER_BOOL:
            STATE["laser"] = v

    def set_int_property(self, pid, v):
        pass


class DeviceList:
    def get_count(self):
        return 1

    def get_device_by_index(self, k):
        return Device()

    def get_device_by_serial_number(self, sn):
        return Device()

    def get_device_serial_number_by_index(self, k):
        return "AY0FAKE001"


class Context:
    def query_devices(self):
        return DeviceList()


class Config:
    def __init__(self):
        self.streams, self.align = [], OBAlignMode.DISABLE

    def enable_stream(self, p):
        self.streams.append(p)

    def set_align_mode(self, m):
        self.align = m


class _Frame:
    def __init__(self, arr, fmt, scale=1.0):
        self.arr, self.fmt, self.scale = arr, fmt, scale

    def get_width(self):
        return self.arr.shape[1]

    def get_height(self):
        return self.arr.shape[0]

    def get_format(self):
        return self.fmt

    def get_data(self):
        return self.arr.reshape(-1).view(np.uint8) if self.arr.dtype == np.uint16 else self.arr.reshape(-1)

    def get_depth_scale(self):
        return self.scale


class FrameSet:
    def __init__(self, c, d):
        self.c, self.d = c, d

    def get_color_frame(self):
        return self.c

    def get_depth_frame(self):
        return self.d


SCENE = {}


def set_scene(color_bgr, depth_mm_colorsize, depth_native):
    SCENE.update(color=color_bgr, depth_color=depth_mm_colorsize, depth_native=depth_native)


class Pipeline:
    def __init__(self, dev=None):
        self.cfg = None

    def get_stream_profile_list(self, kind):
        return StreamProfileList(kind)

    def start(self, cfg):
        self.cfg = cfg
        STATE["opened"] += 1

    def stop(self):
        pass

    def enable_frame_sync(self):
        pass

    def get_camera_param(self):
        return OBCameraParam()

    def wait_for_frames(self, ms):
        col = SCENE["color"]
        rgb = col[:, :, ::-1]
        if STATE["mirror_color"]:
            rgb = rgb[:, ::-1]
        import cv2
        cp = [p for p in self.cfg.streams if p.kind == OBSensorType.COLOR_SENSOR][0]
        if cp.fmt == OBFormat.MJPG:
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            c = _Frame(np.frombuffer(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])[1].tobytes(), np.uint8)
                       .reshape(1, -1), OBFormat.MJPG)
            c.get_width = lambda: cp.w
            c.get_height = lambda: cp.h
        else:
            c = _Frame(np.ascontiguousarray(rgb), OBFormat.RGB)
        want_depth = any(p.kind == OBSensorType.DEPTH_SENSOR for p in self.cfg.streams)
        d = None
        if want_depth:
            aligned = self.cfg.align == OBAlignMode.SW_MODE or (self.cfg.align == OBAlignMode.HW_MODE and STATE["hw_aligns"])
            dm = SCENE["depth_color"] if aligned else SCENE["depth_native"]
            if STATE["mirror_depth"]:
                dm = dm[:, ::-1]
            d = _Frame(np.ascontiguousarray((dm.astype(np.float64) * 10).clip(0, 65535).astype(np.uint16)),
                       OBFormat.Y16, 0.1)
        return FrameSet(c, d)

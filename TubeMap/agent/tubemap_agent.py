#!/usr/bin/env python3
"""TubeMap network camera agent.

Runs on a camera computer (Orbbec Persee 2) and serves its camera to the
TubeMap dashboard on the Pi. Python 3.6+ (Persee 2 ships Ubuntu 18.04).

    python3 tubemap_agent.py --port 8095            # Orbbec camera
    python3 tubemap_agent.py --simulate             # test pattern, no camera

Endpoints
    GET /info        device, intrinsics, agent version (JSON)
    GET /frame.jpg   latest colour frame as JPEG (?max=WIDTH)
    GET /capture     next matched colour + depth pair, lossless, as .npz
"""
import argparse
import io
import json
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV for Python is required: sudo apt install python3-opencv  (or pip3 install opencv-python-headless)")

AGENT_VERSION = "1.0.0"


class SimCapture(object):
    """Test pattern so the agent can be checked without a camera."""

    def __init__(self, width=1280, height=800):
        self.w, self.h = width, height
        self.info = {"model": "Simulated agent camera", "serial": "SIM-AGENT", "color_size": [width, height]}
        self.depth_aligned = True
        f = width / (2 * np.tan(np.radians(45)))
        self.intr = {"width": width, "height": height, "fx": f, "fy": f, "cx": width / 2.0 - 0.5,
                     "cy": height / 2.0 - 0.5, "dist": [0, 0, 0, 0, 0], "model": "opencv"}

    def open(self):
        img = np.full((self.h, self.w, 3), 150, np.uint8)
        pitch = 24
        for j in range(0, self.h // pitch + 2):
            for i in range(0, self.w // pitch + 2):
                x = i * pitch + (pitch // 2 if j % 2 else 0)
                y = int(j * pitch * 0.866)
                cv2.circle(img, (x, y), 7, (30, 30, 30), -1, cv2.LINE_AA)
        self.img = img
        self.depth = np.full((self.h // 2, self.w // 2), 1000, np.uint16)

    def close(self):
        pass

    def grab(self, timeout_ms=1500, need_depth=False):
        time.sleep(0.05)
        noise = np.random.randint(-3, 4, self.img.shape).astype(np.int16)
        return np.clip(self.img.astype(np.int16) + noise, 0, 255).astype(np.uint8), self.depth.copy(), dict(self.intr)


class Agent(object):
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.running = True
        self.t = threading.Thread(target=self.loop)
        self.t.daemon = True

    def start(self):
        self.cap.open()
        self.t.start()

    def loop(self):
        while self.running:
            try:
                c, d, intr = self.cap.grab(1500)
                with self.lock:
                    self.latest = (time.time(), c, d, intr)
                self.error = None
            except Exception as exc:  # keep serving; report the error
                self.error = str(exc)
                time.sleep(0.5)

    def wait_new(self, after, timeout=5.0, need_depth=False):
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                cur = self.latest
            if cur is not None and cur[0] > after and (not need_depth or cur[2] is not None):
                return cur
            time.sleep(0.02)
        raise RuntimeError(self.error or "timed out waiting for a frame")


def make_handler(agent):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/info":
                    body = {"agent": "tubemap-agent", "version": AGENT_VERSION, "device": agent.cap.info,
                            "intrinsics": getattr(agent.cap, "intr", None), "error": agent.error}
                    self._send(200, json.dumps(body).encode(), "application/json")
                elif u.path == "/frame.jpg":
                    ts, c, d, intr = agent.wait_new(0, 3.0)
                    mx = int(q.get("max", ["960"])[0])
                    if c.shape[1] > mx:
                        s = mx / float(c.shape[1])
                        c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self._send(200, buf.tobytes(), "image/jpeg")
                elif u.path == "/capture":
                    ts, c, d, intr = agent.wait_new(time.time(), 6.0, need_depth=getattr(agent.cap, "use_depth", True))
                    meta = {"timestamp": ts, "device": agent.cap.info, "intrinsics": intr,
                            "depth_aligned": bool(getattr(agent.cap, "depth_aligned", d is not None))}
                    bio = io.BytesIO()
                    np.savez_compressed(bio, color=c, depth=d if d is not None else np.zeros((0, 0), np.uint16),
                                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
                    self._send(200, bio.getvalue(), "application/octet-stream")
                elif u.path == "/health":
                    self._send(200, b"ok", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:
                self._send(503, json.dumps({"error": str(exc)}).encode(), "application/json")
    return H


class ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--serial", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--align", default="auto", choices=["auto", "hw", "sw", "off"])
    ap.add_argument("--keep-mirror", action="store_true", help="don't switch off camera mirroring")
    ap.add_argument("--simulate", action="store_true", help="serve a test pattern, no camera needed")
    a = ap.parse_args(argv)
    if a.simulate:
        cap = SimCapture(a.width, a.height)
    else:
        from orbbec_capture import OrbbecCapture
        cap = OrbbecCapture(a.serial, a.width, a.height, a.fps, not a.no_depth, a.align, not a.keep_mirror)
    agent = Agent(cap)
    agent.start()
    srv = ThreadedServer(("0.0.0.0", a.port), make_handler(agent))
    print("TubeMap agent %s serving %s on port %d" % (AGENT_VERSION, cap.info.get("model"), a.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        agent.running = False
        cap.close()


if __name__ == "__main__":
    main()

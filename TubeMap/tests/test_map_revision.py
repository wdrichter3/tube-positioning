"""Calibration provenance regressions; no camera or server startup required."""
import copy
import os
import tempfile
import threading
import unittest
import shutil
import uuid
import json
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch, Mock

from tubemap import dataset, config
from tubemap.server import App, Handler, HttpError, r_anchors, r_cal_depth, r_propose


class MapRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.gettempdir()) / ("tubemap-test-" + uuid.uuid4().hex)
        self.tmp.mkdir()
        self.env = patch.dict(os.environ, {"TUBEMAP_DATA": str(self.tmp)})
        self.env.start()
        (self.tmp / "captures" / "capture").mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp)
        self.addCleanup(self.env.stop)
        self.app = App.__new__(App)
        self.app.map_lock = threading.Lock()
        self.app.map_update_lock = threading.RLock()
        self.sites = [{"id": "T1", "sheet_x_mm": 0., "sheet_y_mm": 0.},
                      {"id": "T2", "sheet_x_mm": 27.4, "sheet_y_mm": 0.}]
        self.anchors = [{"id": "T1", "machine_x_mm": 100., "machine_y_mm": 200.},
                        {"id": "T2", "machine_x_mm": 127.4, "machine_y_mm": 200.}]

    def build(self, pitch=27.4):
        raw = {"sites": copy.deepcopy(self.sites), "checks": [], "anchors": [], "transform": None}
        with patch("tubemap.server.dataset.load_meta", return_value={}), \
             patch("tubemap.server.dataset.load_color", return_value=None), \
             patch.object(self.app, "_load_depth_for_mapping", return_value=(None, None)), \
             patch("tubemap.server.learning.active_profile", return_value={}), \
             patch("tubemap.server.mapping.build_map", return_value=raw):
            return self.app.build_map("capture", {"mode": "known", "pitch_value": pitch,
                "pitch_unit": "mm", "id_value": 18.7, "id_unit": "mm"}, None, False)

    def save(self, revision):
        body = {"anchors": self.anchors, "map_revision": revision}
        handler = type("Handler", (), {"json_body": lambda self: body})()
        return r_anchors(self.app, handler, {}, "capture")

    def test_current_revision_can_calibrate_and_survives_reload(self):
        m = self.build()
        calibrated = self.save(m["map_revision"])
        self.assertTrue(calibrated["transform"]["valid"])
        self.assertNotEqual(dataset.load_map("capture")["map_revision"], m["map_revision"])
        self.assertEqual(calibrated["sites"][0]["machine_x_mm"], 100.)

    def test_every_rebuild_invalidates_anchors_even_same_geometry(self):
        for pitch in (27.4, 30.):
            with self.subTest(pitch=pitch):
                old = self.build()
                self.save(old["map_revision"])
                new = self.build(pitch)
                self.assertNotEqual(new["map_revision"], old["map_revision"])
                self.assertEqual(new["anchors"], [])
                self.assertIsNone(new["transform"])
                self.assertTrue(all("machine_x_mm" not in s for s in new["sites"]))
                with self.assertRaises(HttpError) as caught:
                    self.save(old["map_revision"])
                self.assertEqual(caught.exception.code, 409)
                self.assertEqual(dataset.load_map("capture"), new)

    def test_missing_revision_and_legacy_map_fail_closed(self):
        current = self.build()
        with self.assertRaises(HttpError) as caught:
            self.save(None)
        self.assertEqual(caught.exception.code, 409)
        current.pop("map_revision")
        dataset.save_map("capture", current)
        with self.assertRaises(HttpError) as caught:
            self.save(None)
        self.assertEqual(caught.exception.code, 409)
        self.assertEqual(dataset.load_map("capture"), current)

    def test_depth_request_waits_for_rebuild_and_rejects_old_revision(self):
        m = self.prepare_depth()
        lock = self.app.map_update_lock
        attempted = threading.Event()
        class ObservedLock:
            def __enter__(self):
                if threading.current_thread() is not threading.main_thread(): attempted.set()
                lock.acquire()
            def __exit__(self, *args): lock.release()
        self.app.map_update_lock = ObservedLock()
        result = []
        def calibrate():
            try:
                self.request(r_cal_depth, {"map_revision": m["map_revision"]})
                result.append(200)
            except HttpError as error: result.append(error.code)
        with self.app.map_update_lock:
            thread = threading.Thread(target=calibrate)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.build()
        thread.join(5)
        self.assertEqual(result, [409])
        self.app.devices.update.assert_not_called()

    def test_http_anchor_revision_contract(self):
        current = self.build()
        handler = type("TestHandler", (Handler,), {"app": self.app})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/captures/capture/anchors"
            for revision, status in ((current["map_revision"], 200), ("stale", 409), (None, 409)):
                request = urllib.request.Request(url, method="PUT", headers={"Content-Type": "application/json"},
                    data=json.dumps({"anchors": self.anchors, "map_revision": revision}).encode())
                try:
                    with urllib.request.urlopen(request, timeout=5) as response:
                        actual = response.status
                        result = json.load(response)
                        self.assertTrue(result["transform"]["valid"])
                except urllib.error.HTTPError as error:
                    actual = error.code
                    error.close()
                self.assertEqual(actual, status)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_anchor_write_waits_for_rebuild_then_rejects_old_revision(self):
        current = self.build()
        entered = threading.Event()
        results = []
        lock = self.app.map_update_lock
        class ObservedLock:
            def __enter__(self):
                if threading.current_thread() is not threading.main_thread():
                    entered.set()
                lock.acquire()
            def __exit__(self, *args):
                lock.release()
        self.app.map_update_lock = ObservedLock()

        def write_anchor():
            try:
                self.save(current["map_revision"])
                results.append(200)
            except HttpError as error:
                results.append(error.code)

        with self.app.map_update_lock:
            thread = threading.Thread(target=write_anchor, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(2))
            new = self.build()
        thread.join(timeout=5)
        self.assertEqual(results, [409])
        self.assertEqual(dataset.load_map("capture"), new)

    def request(self, fn, body):
        handler = type("Handler", (), {"json_body": lambda self: body})()
        return fn(self.app, handler, {}, "capture")

    def test_same_revision_anchor_writers_cannot_overwrite_each_other(self):
        m = self.build()
        gate = threading.Barrier(3)
        results = []
        def write(items):
            gate.wait()
            try:
                result = self.request(r_anchors, {"anchors": items, "map_revision": m["map_revision"]})
                results.append((200, result))
            except HttpError as error:
                results.append((error.code, None))
        threads = [threading.Thread(target=write, args=(items,)) for items in ([self.anchors[0]], self.anchors)]
        for t in threads: t.start()
        gate.wait()
        for t in threads: t.join(5)
        self.assertEqual(sorted(r[0] for r in results), [200, 409])
        winner = next(r[1] for r in results if r[0] == 200)
        self.assertEqual(dataset.load_map("capture"), winner)

    def test_invalid_anchor_payloads_preserve_map_bytes(self):
        m = self.build()
        path = dataset.cdir("capture") / "map.json"
        before = path.read_bytes()
        bad = [None, {}, "wrong", [None], [{"id": "unknown", "machine_x_mm": 0, "machine_y_mm": 0}],
               [self.anchors[0], self.anchors[0]], [{**self.anchors[0], "machine_x_mm": float("nan")}],
               [{**self.anchors[0], "machine_y_mm": float("inf")}], [{**self.anchors[0], "machine_x_mm": True}]]
        for anchors in bad:
            with self.subTest(anchors=anchors):
                with self.assertRaises(HttpError) as caught:
                    self.request(r_anchors, {"anchors": anchors, "map_revision": m["map_revision"]})
                self.assertEqual(caught.exception.code, 400)
                self.assertEqual(path.read_bytes(), before)

    def prepare_depth(self):
        m = self.build()
        m.update(mode="known", geometry={"pitch_mm": 105.}, depth_pitch={"pitch_mm": 100.}, depth_scale_factor=1.)
        dataset.save_map("capture", m)
        config.write_json(dataset.cdir("capture") / "meta.json", {"depth_scale_factor": 1., "device": {"id": "fake"}})
        self.app.devices = Mock()
        return m

    def test_concurrent_depth_correction_is_applied_only_once(self):
        m = self.prepare_depth()
        gate = threading.Barrier(3)
        results = []
        def write():
            gate.wait()
            try:
                results.append(self.request(r_cal_depth, {"map_revision": m["map_revision"]})["factor"])
            except HttpError as error:
                results.append(error.code)
        with patch.object(self.app, "build_map", return_value={} ) as rebuild:
            threads = [threading.Thread(target=write) for _ in range(2)]
            for t in threads: t.start()
            gate.wait()
            for t in threads: t.join(5)
        self.assertEqual(sorted(results), [1.05, 409])
        self.assertEqual(dataset.load_meta("capture")["depth_scale_factor"], 1.05)
        self.app.devices.update.assert_called_once()
        rebuild.assert_called_once()

    def test_depth_rejects_metadata_or_companion_provenance_mismatch(self):
        m = self.prepare_depth()
        dataset.set_depth_scale("capture", 1.05)
        with self.assertRaises(HttpError) as caught:
            self.request(r_cal_depth, {"map_revision": m["map_revision"]})
        self.assertEqual(caught.exception.code, 409)
        self.assertEqual(dataset.load_map("capture"), m)
        dataset.set_depth_scale("capture", 1.)
        (self.tmp / "captures" / "companion").mkdir()
        cm = {**m, "map_revision": "new-companion"}
        dataset.save_map("companion", cm)
        m["companion"] = {"capture_id": "companion", "map_revision": "old-companion"}
        dataset.save_map("capture", m)
        with self.assertRaises(HttpError) as caught:
            self.request(r_cal_depth, {"map_revision": m["map_revision"], "target": "companion"})
        self.assertEqual(caught.exception.code, 409)
        self.app.devices.update.assert_not_called()

    def test_depth_failure_invalidates_old_calibration(self):
        m = self.prepare_depth()
        calibrated = self.save(m["map_revision"])
        with patch.object(self.app, "build_map", side_effect=RuntimeError("rebuild failed")):
            with self.assertRaises(RuntimeError):
                self.request(r_cal_depth, {"map_revision": calibrated["map_revision"]})
        saved = dataset.load_map("capture")
        self.assertFalse(saved["ready"])
        self.assertEqual(saved["anchors"], [])
        self.assertIsNone(saved["transform"])
        with self.assertRaises(HttpError) as caught:
            self.save(saved["map_revision"])
        self.assertEqual(caught.exception.code, 409)
        with self.assertRaises(HttpError) as caught:
            self.save(calibrated["map_revision"])
        self.assertEqual(caught.exception.code, 409)

    def test_propose_preserves_main_and_companion_map_bytes(self):
        m = self.build()
        m = self.save(m["map_revision"])
        (self.tmp / "captures" / "companion").mkdir()
        dataset.save_map("companion", m)
        paths = [dataset.cdir(cid) / "map.json" for cid in ("capture", "companion")]
        before = [p.read_bytes() for p in paths]
        preview = {"sites": [], "checks": []}
        with patch.object(self.app, "_map_one", return_value=(preview, {})), \
             patch("tubemap.server.mapping.site_labels_from_map", return_value=[]):
            self.request(r_propose, {"geometry": {"mode": "discover"}, "roi": [[0,0],[1,0],[0,1]]})
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_companion_rebuild_invalidates_and_sets_ready_notice(self):
        self.build()
        (self.tmp / "captures" / "companion").mkdir()
        old = {"anchors": self.anchors, "map_revision": "old", "roi": None}
        dataset.save_map("companion", old)
        def one(cid, g, roi):
            return {"capture_id": cid, "sites": copy.deepcopy(self.sites), "checks": [], "map_revision": "temporary"}, {}
        def merge(main, companion, name):
            main["companion"] = {"capture_id": "companion"}
        with patch.object(self.app, "_map_one", side_effect=one), \
             patch.object(self.app, "companions", return_value=[{"id": "companion"}]), \
             patch("tubemap.server.depthcheck.merge_companion", side_effect=merge):
            main = self.app.build_map("capture", {}, None)
        cm = dataset.load_map("companion")
        self.assertTrue(cm["ready"])
        self.assertEqual(cm["anchors"], [])
        self.assertIn("Re-establish", cm["calibration_notice"])
        self.assertEqual(main["companion"]["map_revision"], cm["map_revision"])
        self.assertNotEqual(cm["map_revision"], "old")


if __name__ == "__main__":
    unittest.main()

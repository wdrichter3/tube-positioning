import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from tubemap import dataset, learning
from tubemap.vision import mapping, synth


def _capture(**kw):
    sc = synth.render(**kw)
    meta = dataset.save_capture(sc["color"], sc["depth"], sc["intrinsics"],
                                device={"id": "sim", "name": "Sim", "driver": "simulator", "settings": {}})
    return meta["id"], sc


def test_capture_roundtrip_and_labels():
    cid, sc = _capture()
    assert dataset.load_color(cid).shape == sc["color"].shape
    assert np.array_equal(dataset.load_depth(cid), sc["depth"])
    items = [{"u": 100, "v": 120, "r": 6, "cls": "TUBE"}, {"u": 200, "v": 220, "r": 6, "cls": "BLOCKED"}]
    lab = dataset.save_labels(cid, items, roi=[[0, 0], [1, 0], [1, 1]], reviewed=True)
    assert lab["reviewed"] and len(lab["items"]) == 2
    s = dataset.summary(cid)
    assert s["label_counts"]["BLOCKED"] == 1 and s["reviewed"]
    with pytest.raises(ValueError):
        dataset.save_labels(cid, [{"u": 99999, "v": 1, "r": 5, "cls": "TUBE"}])
    with pytest.raises(ValueError):
        dataset.save_labels(cid, [{"u": 1, "v": 1, "r": 5, "cls": "BANANA"}])
    with pytest.raises(ValueError):
        dataset.cdir("../etc")


def test_import_old_imagelab_zip():
    sc = synth.render()
    ok, jpg = cv2.imencode(".jpg", sc["color"])
    h, w = sc["color"].shape[:2]
    ann = {"schema_version": 1, "sample_id": "IMG1", "roi": None, "notes": "old",
           "annotations": [{"label": "TUBE", "x": 0.5, "y": 0.5, "w": 0.01, "h": 0.013},
                           {"label": "BLOCKED", "x": 0.4, "y": 0.4, "w": 0.01, "h": 0.013}]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("images/IMG1.jpg", jpg.tobytes())
        z.writestr("annotations/IMG1.json", json.dumps(ann))
        z.writestr("images/IMG2.jpg", jpg.tobytes())            # duplicate bytes -> skipped
        z.writestr("annotations/IMG2.json", json.dumps({**ann, "sample_id": "IMG2"}))
        z.writestr("manifest.json", json.dumps({"schema_version": 1}))
    r = dataset.import_zip(buf.getvalue())
    assert r["imported"] == 1 and r["skipped"] == 1 and r["format"] == "imagelab"
    lab = dataset.load_labels(r["ids"][0])
    assert lab["reviewed"] and {a["cls"] for a in lab["items"]} == {"TUBE", "BLOCKED"}
    assert abs(lab["items"][0]["u"] - 0.5 * w) < 1


def test_export_reimport_roundtrip():
    cid, sc = _capture()
    dataset.save_labels(cid, [{"u": 100, "v": 120, "r": 6, "cls": "TUBE"}, {"u": 10, "v": 12, "r": 6, "cls": "NOT_TUBE"}], reviewed=True)
    data, n = dataset.export_dataset()
    assert n == 1
    z = zipfile.ZipFile(io.BytesIO(data))
    assert z.read(f"labels/{cid}.txt").decode().count("\n") == 1   # NOT_TUBE not in YOLO labels
    r = dataset.import_zip(data)
    assert r["format"] == "tubemap" and r["imported"] == 1


def test_training_improves_or_keeps_scores_and_is_not_auto_activated():
    for k, kw in enumerate([dict(seed=1), dict(seed=2, tilt_deg=25), dict(seed=3, pattern="SQUARE", pitch_in=1.25, tube_id_in=0.834),
                            dict(seed=4, noise=9), dict(seed=5, plugged_fraction=0.08)]):
        cid, sc = _capture(**kw)
        g = {"mode": "known", "pattern": sc["params"]["pattern"], "pitch_mm": sc["pitch_mm"], "tube_id_mm": sc["tube_id_mm"]}
        m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], g)
        dataset.save_labels(cid, mapping.site_labels_from_map(m), reviewed=True)
    prof = learning.train_profile("t1")
    assert prof["metrics"]["holdout_detector"]["f1"] >= prof["metrics"]["baseline_holdout_detector"]["f1"] - 1e-9
    cv = prof["metrics"]["cv_classifier"]
    assert cv["accuracy"] > 0.95
    assert cv["per_class"]["BLOCKED"]["recall"] >= 0.9 and cv["per_class"]["BLOCKED"]["captures"] >= 4
    assert prof["training"]["model"].startswith("gradient-boosted")
    assert prof["metrics"]["possibly_incomplete"] == []
    assert learning.active_profile()["name"] == "builtin"
    learning.activate("t1")
    assert learning.active_profile()["name"] == "t1"
    # map with the trained profile keeps empty cells and plugs apart
    sc = synth.render(seed=9, plugged_fraction=0.05)
    g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_mm": sc["pitch_mm"], "tube_id_mm": sc["tube_id_mm"]}
    m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], g, profile=learning.active_profile())
    truth_plug = sum(t["state"] == "PLUGGED" for t in sc["truth"])
    assert abs(m["counts"].get("PLUGGED", 0) - truth_plug) <= 2
    assert m["counts"].get("NO_TUBE", 0) >= 30


def test_incomplete_labels_are_flagged_and_test_set_is_held_out():
    ids = []
    for k in range(4):
        cid, sc = _capture(seed=40 + k, plugged_fraction=0.05)
        g = {"mode": "known", "pattern": "TRIANGULAR", "pitch_mm": sc["pitch_mm"], "tube_id_mm": sc["tube_id_mm"]}
        m = mapping.build_map(sc["color"], sc["depth"], sc["intrinsics"], g)
        items = mapping.site_labels_from_map(m)
        if k == 0:                                   # reviewer forgot the bottom half
            vmid = np.median([a["v"] for a in items])
            items = [a for a in items if a["v"] < vmid] + [a for a in items if a["v"] >= vmid and a["cls"] != "TUBE"][:3]
        dataset.save_labels(cid, items, reviewed=True)
        ids.append(cid)
    dataset.update_meta(ids[3], tags=["test"])
    prof = learning.train_profile("t2")
    assert ids[0] in [x["id"] for x in prof["metrics"]["possibly_incomplete"]]
    assert ids[3] not in prof["training"]["captures_train"] and ids[3] in prof["training"]["captures_test"]
    assert prof["metrics"]["test_classifier"]["accuracy"] > 0.9


def test_old_v1_profiles_still_load():
    from tubemap.vision import state
    c = state.StateClassifier().fit(np.random.default_rng(0).normal(size=(40, 152)), np.array([0, 1] * 20))
    d = c.to_dict()
    d.pop("kind"); d.pop("feature_version")           # as saved by 1.0/1.1
    c2 = state.load_classifier(d)
    assert c2.trained and c2.feature_version == 1


def test_training_without_reviews_explains():
    with pytest.raises(ValueError, match="No reviewed captures"):
        learning.train_profile()

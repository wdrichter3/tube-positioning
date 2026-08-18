import numpy as np
from tube_positioning.models import ROI
from tube_positioning.roi import build_roi_mask, apply_roi


def test_rectangle_roi_mask():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    roi = ROI(kind="rectangle", points=[(10, 10), (20, 20)])
    mask = build_roi_mask(img.shape, roi)
    assert mask[15, 15] == 255
    assert mask[5, 5] == 0
    out = apply_roi(img, roi)
    assert out[15, 15].sum() > 0
    assert out[5, 5].sum() == 0


def test_polygon_roi_mask():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    roi = ROI(kind="polygon", points=[(10, 10), (50, 10), (30, 50)])
    mask = build_roi_mask(img.shape, roi)
    assert mask[20, 30] == 255
    assert mask[80, 80] == 0

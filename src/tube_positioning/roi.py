from __future__ import annotations

import cv2
import numpy as np
from tube_positioning.models import ROI


def build_roi_mask(image_shape: tuple[int, ...], roi: ROI) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if roi.kind == "rectangle":
        if len(roi.points) != 2:
            raise ValueError("Rectangle ROI requires two points")
        (x1, y1), (x2, y2) = roi.points
        xa, xb = sorted((int(round(x1)), int(round(x2))))
        ya, yb = sorted((int(round(y1)), int(round(y2))))
        cv2.rectangle(mask, (xa, ya), (xb, yb), 255, thickness=-1)
    else:
        if len(roi.points) < 3:
            raise ValueError("Polygon ROI requires at least three points")
        pts = np.array([[int(round(x)), int(round(y))] for x, y in roi.points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def apply_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    mask = build_roi_mask(image.shape, roi)
    return cv2.bitwise_and(image, image, mask=mask)

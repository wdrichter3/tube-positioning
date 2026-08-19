import cv2
import numpy as np

def ensure_gray(image):
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_roi_mask(image, roi_mask=None):
    h, w = image.shape[:2]
    if roi_mask is None:
        return np.full((h, w), 255, np.uint8)
    if roi_mask.shape[:2] != (h, w):
        raise ValueError("ROI mask shape must match image")
    return (roi_mask > 0).astype(np.uint8) * 255

def circularity(area, perimeter):
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))

def ellipse_confidence(major, minor, circularity_value, area, min_area, max_area):
    if major <= 0 or minor <= 0:
        return 0.0
    axis_ratio = min(major, minor) / max(major, minor)
    circ_score = float(np.clip((circularity_value - 0.35) / 0.55, 0.0, 1.0))
    axis_score = float(np.clip((axis_ratio - 0.45) / 0.55, 0.0, 1.0))
    mid = 0.5 * (min_area + max_area)
    half = max(1.0, 0.5 * (max_area - min_area))
    area_score = float(np.clip(1.0 - abs(area - mid) / (1.5 * half), 0.0, 1.0))
    return float(0.45 * axis_score + 0.35 * circ_score + 0.20 * area_score)

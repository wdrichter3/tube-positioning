import cv2
import numpy as np
from .base import TubeDetector
from .models import DetectionResult, TubeCandidateRecord
from .utils import ensure_gray, normalize_roi_mask

class HoughTubeDetector(TubeDetector):
    name = "hough"

    def __init__(self, min_diameter_px=12, max_diameter_px=100,
                 min_dist_px=None, param1=100, param2=24, dp=1.2):
        self.min_diameter_px = float(min_diameter_px)
        self.max_diameter_px = float(max_diameter_px)
        self.min_dist_px = min_dist_px
        self.param1 = float(param1)
        self.param2 = float(param2)
        self.dp = float(dp)

    def detect(self, image, roi_mask=None):
        gray = ensure_gray(image)
        mask = normalize_roi_mask(image, roi_mask)
        gray = cv2.bitwise_and(gray, mask)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
        min_r = max(2, int(self.min_diameter_px / 2))
        max_r = max(min_r + 1, int(self.max_diameter_px / 2))
        min_dist = self.min_dist_px or max(4.0, 0.8 * self.min_diameter_px)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=self.dp, minDist=float(min_dist),
            param1=self.param1, param2=self.param2,
            minRadius=min_r, maxRadius=max_r
        )
        result = DetectionResult(image_shape=gray.shape)
        if circles is None:
            return result
        circles = np.asarray(circles[0], dtype=float)
        # A simple contrast-based confidence
        for idx, (x, y, r) in enumerate(circles, start=1):
            xi, yi, ri = int(round(x)), int(round(y)), max(2, int(round(r)))
            y0, y1 = max(0, yi-ri), min(gray.shape[0], yi+ri+1)
            x0, x1 = max(0, xi-ri), min(gray.shape[1], xi+ri+1)
            patch = gray[y0:y1, x0:x1]
            contrast = 0.0 if patch.size == 0 else float(np.std(patch) / 64.0)
            conf = float(np.clip(0.45 + 0.35 * contrast, 0.0, 0.95))
            result.candidates.append(TubeCandidateRecord(
                candidate_id=idx, pixel_x=float(x), pixel_y=float(y),
                major_axis_px=float(2*r), minor_axis_px=float(2*r),
                visual_confidence=conf, detector_name=self.name
            ))
        return result

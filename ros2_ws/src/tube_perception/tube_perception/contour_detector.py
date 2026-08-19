import cv2
import numpy as np
from .base import TubeDetector
from .models import DetectionResult, TubeCandidateRecord
from .utils import ensure_gray, normalize_roi_mask, circularity, ellipse_confidence

class ContourTubeDetector(TubeDetector):
    name = "contour"

    def __init__(self, min_diameter_px=12, max_diameter_px=100,
                 min_circularity=0.38, min_axis_ratio=0.45,
                 adaptive_block_size=41, adaptive_c=7):
        self.min_diameter_px = float(min_diameter_px)
        self.max_diameter_px = float(max_diameter_px)
        self.min_circularity = float(min_circularity)
        self.min_axis_ratio = float(min_axis_ratio)
        self.adaptive_block_size = int(adaptive_block_size) | 1
        self.adaptive_c = float(adaptive_c)

    def detect(self, image, roi_mask=None):
        gray = ensure_gray(image)
        mask = normalize_roi_mask(image, roi_mask)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_block_size, self.adaptive_c
        )
        binary = cv2.bitwise_and(binary, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        min_area = np.pi * (self.min_diameter_px * 0.5) ** 2 * 0.45
        max_area = np.pi * (self.max_diameter_px * 0.5) ** 2 * 1.35
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        result = DetectionResult(image_shape=gray.shape)
        accepted = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perim = cv2.arcLength(contour, True)
            circ = circularity(area, perim)
            if len(contour) >= 5:
                (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
                major, minor = max(a, b), min(a, b)
            else:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                major = minor = 2.0 * radius
                angle = 0.0
            if major < self.min_diameter_px or major > self.max_diameter_px:
                continue
            axis_ratio = minor / max(major, 1e-6)
            if circ < self.min_circularity or axis_ratio < self.min_axis_ratio:
                result.rejected.append((float(cx), float(cy), "shape"))
                continue
            conf = ellipse_confidence(major, minor, circ, area, min_area, max_area)
            accepted.append((cx, cy, major, minor, angle, conf))

        # non-maximum suppression by center distance
        accepted.sort(key=lambda x: x[5], reverse=True)
        kept = []
        for cand in accepted:
            cx, cy, major, minor, angle, conf = cand
            min_dist = 0.45 * min(major, minor)
            if any((cx-k[0])**2 + (cy-k[1])**2 < min_dist**2 for k in kept):
                continue
            kept.append(cand)

        for idx, (cx, cy, major, minor, angle, conf) in enumerate(kept, start=1):
            result.candidates.append(TubeCandidateRecord(
                candidate_id=idx, pixel_x=float(cx), pixel_y=float(cy),
                major_axis_px=float(major), minor_axis_px=float(minor),
                orientation_deg=float(angle), visual_confidence=float(conf),
                detector_name=self.name
            ))
        return result

from .base import TubeDetector
from .models import DetectionResult, TubeCandidateRecord
from .contour_detector import ContourTubeDetector
from .hough_detector import HoughTubeDetector

class HybridTubeDetector(TubeDetector):
    name = "hybrid"

    def __init__(self, contour=None, hough=None, merge_distance_factor=0.55):
        self.contour = contour or ContourTubeDetector()
        self.hough = hough or HoughTubeDetector()
        self.merge_distance_factor = float(merge_distance_factor)

    def detect(self, image, roi_mask=None):
        a = self.contour.detect(image, roi_mask)
        b = self.hough.detect(image, roi_mask)
        pool = a.candidates + b.candidates
        pool.sort(key=lambda c: c.visual_confidence, reverse=True)
        groups = []
        for cand in pool:
            matched = None
            for g in groups:
                ref = g[0]
                d2 = (cand.pixel_x-ref.pixel_x)**2 + (cand.pixel_y-ref.pixel_y)**2
                gate = self.merge_distance_factor * min(cand.mean_diameter_px, ref.mean_diameter_px)
                if d2 <= gate**2:
                    matched = g
                    break
            if matched is None:
                groups.append([cand])
            else:
                matched.append(cand)

        out = DetectionResult(image_shape=a.image_shape, rejected=a.rejected+b.rejected)
        for idx, g in enumerate(groups, start=1):
            weights = [max(0.05, c.visual_confidence) for c in g]
            sw = sum(weights)
            x = sum(c.pixel_x*w for c,w in zip(g,weights))/sw
            y = sum(c.pixel_y*w for c,w in zip(g,weights))/sw
            major = sum(c.major_axis_px*w for c,w in zip(g,weights))/sw
            minor = sum(c.minor_axis_px*w for c,w in zip(g,weights))/sw
            angle = sum(c.orientation_deg*w for c,w in zip(g,weights))/sw
            sources = {c.detector_name for c in g}
            agreement_bonus = 0.12 if len(sources) > 1 else 0.0
            conf = min(1.0, max(c.visual_confidence for c in g) + agreement_bonus)
            out.candidates.append(TubeCandidateRecord(
                candidate_id=idx, pixel_x=x, pixel_y=y,
                major_axis_px=major, minor_axis_px=minor,
                orientation_deg=angle, visual_confidence=conf,
                detector_name="hybrid:" + "+".join(sorted(sources))
            ))
        return out

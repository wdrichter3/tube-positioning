from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class TubeCandidateRecord:
    candidate_id: int
    pixel_x: float
    pixel_y: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float = 0.0
    visual_confidence: float = 0.0
    detector_name: str = ""
    estimated_diameter_mm: float = 0.0

    @property
    def mean_diameter_px(self) -> float:
        return 0.5 * (self.major_axis_px + self.minor_axis_px)

@dataclass
class DetectionResult:
    candidates: List[TubeCandidateRecord] = field(default_factory=list)
    rejected: List[Tuple[float, float, str]] = field(default_factory=list)
    image_shape: Optional[Tuple[int, int]] = None

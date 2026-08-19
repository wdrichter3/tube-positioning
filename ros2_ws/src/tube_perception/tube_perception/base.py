from abc import ABC, abstractmethod
from .models import DetectionResult

class TubeDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, image, roi_mask=None) -> DetectionResult:
        raise NotImplementedError

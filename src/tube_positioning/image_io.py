from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image: {path}")
    return image

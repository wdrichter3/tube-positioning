from __future__ import annotations

from math import hypot
from uuid import uuid4
from typing import Sequence, Tuple

from tube_positioning.models import CalibrationRecord

Point = Tuple[float, float]


def _distance(a: Point, b: Point) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def calibrate_planar_reference(
    corners: Sequence[Point],
    known_width_mm: float,
    known_height_mm: float,
    reference_type: str = "CUSTOM",
    confidence: float = 1.0,
) -> CalibrationRecord:
    """Calibrate scale from ordered planar corners: TL, TR, BR, BL.

    V0.1 computes independent X/Y scale. Perspective rectification is deliberately
    kept separate so a non-square camera view does not masquerade as isotropic scale.
    """
    if len(corners) != 4:
        raise ValueError("Exactly four ordered corners are required: TL, TR, BR, BL")
    if known_width_mm <= 0 or known_height_mm <= 0:
        raise ValueError("Known reference dimensions must be positive")

    tl, tr, br, bl = corners
    width_px = (_distance(tl, tr) + _distance(bl, br)) / 2.0
    height_px = (_distance(tl, bl) + _distance(tr, br)) / 2.0
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Reference corners produce zero pixel dimension")

    return CalibrationRecord(
        calibration_id=f"CAL-{uuid4().hex[:12].upper()}",
        scale_method="REFERENCE_OBJECT",
        reference_type=reference_type,
        known_width_mm=known_width_mm,
        known_height_mm=known_height_mm,
        pixel_width=width_px,
        pixel_height=height_px,
        mm_per_pixel_x=known_width_mm / width_px,
        mm_per_pixel_y=known_height_mm / height_px,
        perspective_corrected=False,
        metric_valid=True,
        confidence=confidence,
    )


def pixel_delta_to_mm(dx_px: float, dy_px: float, calibration: CalibrationRecord) -> tuple[float, float]:
    if not calibration.metric_valid:
        raise ValueError("Calibration is not metrically valid")
    return dx_px * calibration.mm_per_pixel_x, dy_px * calibration.mm_per_pixel_y

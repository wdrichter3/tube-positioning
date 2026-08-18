from __future__ import annotations

from enum import IntEnum
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class TubeState(IntEnum):
    UNKNOWN = 0
    DETECTED = 1
    PLANNED = 2
    CURRENT = 3
    COMPLETED = 4
    SKIPPED = 5
    PLUGGED = 6
    BLOCKED = 7
    REVIEW_REQUIRED = 8


class ROI(BaseModel):
    kind: Literal["rectangle", "polygon"]
    points: List[Tuple[float, float]] = Field(min_length=2)


class CalibrationRecord(BaseModel):
    calibration_id: str
    scale_method: Literal["REFERENCE_OBJECT", "FIDUCIAL", "KNOWN_PITCH"]
    reference_type: str
    known_width_mm: Optional[float] = None
    known_height_mm: Optional[float] = None
    pixel_width: Optional[float] = None
    pixel_height: Optional[float] = None
    mm_per_pixel_x: float
    mm_per_pixel_y: float
    perspective_corrected: bool = False
    metric_valid: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class TubeCandidate(BaseModel):
    candidate_id: int
    pixel_x: float
    pixel_y: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float = 0.0
    estimated_diameter_mm: Optional[float] = None
    visual_confidence: float = Field(ge=0.0, le=1.0)
    detector_name: str


class Tube(BaseModel):
    tube_id: int
    display_id: str
    row: int
    column: int
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    diameter_mm: Optional[float] = None
    neighbor_ids: List[int] = []
    detection_confidence: float = Field(ge=0.0, le=1.0)
    geometry_confidence: float = Field(ge=0.0, le=1.0)
    position_confidence: float = Field(ge=0.0, le=1.0)
    state: TubeState = TubeState.DETECTED
    inferred: bool = False
    operator_verified: bool = False


class TubeMap(BaseModel):
    map_id: str
    job_id: str
    pattern: Literal["SQUARE", "TRIANGULAR", "UNKNOWN"] = "UNKNOWN"
    nominal_pitch_x_mm: Optional[float] = None
    nominal_pitch_y_mm: Optional[float] = None
    nominal_tube_diameter_mm: Optional[float] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)
    calibration: CalibrationRecord
    tubes: List[Tube]


class WorkStep(BaseModel):
    sequence_number: int
    tube_ids: List[int]
    target_x_mm: float
    target_y_mm: float
    delta_x_mm: Optional[float] = None
    delta_y_mm: Optional[float] = None
    state: Literal["PENDING", "CURRENT", "DONE", "SKIPPED"] = "PENDING"


class WorkPlan(BaseModel):
    plan_id: str
    map_id: str
    planner_type: str
    steps: List[WorkStep]
    current_step: int = 0

# Tube Positioning V0.1

Aftermarket vision-guidance software for conventional/manual and semi-automated heat-exchanger tube positioners.

## V0.1 boundary

V0.1 performs static-image mapping and planning only. It **does not command machine motion**.

Implemented baseline:
- project/config models
- image loading
- rectangle/polygon ROI representation
- known-reference planar calibration
- pixel↔metric conversion
- TubeMap data models
- work-plan data models
- FastAPI health/config endpoints
- unit tests

Next implementation slice:
- OpenCV contour/Hough tube candidate detectors
- lattice/pitch fitting
- TubeMap generation
- operator review UI

## Development environment

Preferred: Ubuntu 24.04 or Windows 11 + WSL2 Ubuntu 24.04.
Replit is not required and should not be the hardware-development environment.

## Run tests

```bash
python -m pytest -q
```

## Run API

```bash
uvicorn tube_positioning.api:app --reload
```

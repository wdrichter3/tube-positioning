#!/usr/bin/env bash
# TubeMap on Linux/macOS without installing a service.
set -e
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/python -c "import numpy, cv2" 2>/dev/null || .venv/bin/python -m pip install -q numpy opencv-python-headless
.venv/bin/python -c "import depthai" 2>/dev/null || .venv/bin/python -m pip install -q "depthai>=2.24,<3" || true
.venv/bin/python -c "import sklearn" 2>/dev/null || .venv/bin/python -m pip install -q "scikit-learn>=1.3" || true
exec .venv/bin/python -m tubemap serve --open "$@"

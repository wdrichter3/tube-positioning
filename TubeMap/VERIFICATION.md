# Reliability verification, 2026-10-05

The original candidate passed 55 Python tests and failed four, with no skips.
Eleven parameter subtests also passed. All 13 new Python revision/calibration
regressions passed. The four failures reproduce on the untouched 1.3.0 release:

- `test_companion_depth_camera_flow`: some plugged sites lack companion FACE evidence.
- `test_companion_lattice_match[kw0-False-0]`: state agreement is zero.
- `test_companion_lattice_match[kw1-True-0]`: state agreement is zero.
- `test_companion_lattice_match[kw2-False--90]`: match is invalid.

The repository layout excludes the private real-camera fixture; its existing
skip guard handles its absence. Final organized-source run: **54 passed, four
failed, one skipped**, with 11 subtests passed, in 172.79 seconds. The same four
baseline failures remain; the only skip is the excluded real-camera fixture.

JavaScript: all 11 tests pass; `node --check tubemap/static/app.js` passes.
Run from `TubeMap/` with `node tests/anchor_ui_test.cjs`.

Five synthetic browser scenarios passed before organization: map build, sequential
anchor edits, stale second-tab rejection, proposal preservation of saved calibration,
and depth correction clearing anchors and selection. No cameras were started.
The seven reviewed source/test files match the candidate exactly after organization.

Python validation uses 3.12.10, NumPy 1.26.4, OpenCV 4.10.0, scikit-learn 1.5.2,
DepthAI 2.33.0 and pytest 9.1.1. Normally run `python -m pytest tests` from this
directory with the required dependencies. On the validation Windows sandbox,
pytest's default temporary-directory permissions were inaccessible, so a task-local
runner replaced only `tmp_path` with ordinary inherited-permission UUID directories
and disabled the tmpdir/cache plugins. Tests were not rewritten, skipped or masked.
That environment-specific runner and installed test dependencies are not committed.

No root project files changed; its separate test suite was not rerun. No CI workflow
is present in the checked-out branch. Real hardware, installer/startup, physical
accuracy and complete browser/network E2E remain unverified. The suite is not fully
passing and this is not a field-qualified release.

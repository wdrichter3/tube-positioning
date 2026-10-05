# TubeMap 1.3.0 reliability candidate

This standalone Python service lives under `TubeMap/`. The earlier positioning
project remains unchanged at the repository root. Run TubeMap commands from this
directory; the root project's packaging and tests are separate.

The first import commit preserves 57 source files from `TubeMap-1.3.0.zip`, whose
SHA256 is `4D0EF0ABC462CF38202BBBC4B6331E9EFB9FC9C9495384A22DC06FAE949E324C`.
Git normalizes text line endings. The archive's runtime log and three real-camera
fixture files are intentionally excluded. No captures, installed environments,
test dependencies, credentials or runtime data are included.

The next commit adds calibration reliability fixes reviewed through seven Claude
findings. Those findings were addressed; the revised patch has not received a
second independent review. Changes include per-write revision checks for anchors,
serialized depth calibration with scale provenance, non-persisting label proposals,
capture/request guards for asynchronous UI operations, selection invalidation,
anchor validation, and consistent companion readiness and rebuild notices.

Validation on 2026-10-05 before repository organization: 55 Python tests passed,
four failed, no skips; 11 JavaScript tests and five synthetic browser scenarios
passed. All four failures also reproduce on the untouched release using the same
environment. Real camera fixtures are excluded here, so their test is expected to
skip in the public checkout. See `VERIFICATION.md` for the organized-source run.

The four known failures are `test_companion_depth_camera_flow` and all three
`test_companion_lattice_match` parameter cases. They concern simulated companion
depth evidence and matching; the depth algorithms were not changed by this patch.

Old anchor and depth-calibration clients must submit the current `map_revision`
and adopt each response's revision. Legacy maps need rebuilding before calibration
editing. Actual rebuilds invalidate anchors; label proposals do not rebuild stored
maps. If depth calibration fails after invalidation, rebuild before calibrating
again. Metadata/device changes are not transactionally rolled back.

Revision locking covers one service process, not external file replacement or
multiple service processes. Content hashing and proactive cross-tab notification
remain future work. Map readiness does not establish physical positioning accuracy.
Real Pi, cameras, gantry, native installer/startup and full browser/network E2E
remain unverified. No motion or deployment is part of this change.

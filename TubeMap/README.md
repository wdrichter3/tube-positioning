# TubeMap 1.2

Tubesheet mapping for the bench prototype. It captures images from the OAK-D Pro (and other cameras), builds a tube map, lets you label images to improve detection, and manages cameras from the dashboard.

It replaces the ROS 2 TubeWorkspace stack with a single Python service. That service needs only NumPy and OpenCV, plus the SDK of whichever camera you use. It runs on the Pi, on a Windows or Linux laptop, and against a simulator when no camera is connected.

## What's new in 1.3

Local calibration reliability candidate: rebuilding a map clears its machine anchors,
even when the geometry is unchanged. Re-select anchors on the new map. Rebuilt companion
maps also lose calibration and show a notice. Label proposals do not save or rebuild
stored maps. Anchor writes and depth calibration must include the current `map_revision`;
every successful write advances it, and stale or legacy requests return 409. Keep the
revision from each response. Invalid anchor lists, IDs and nonfinite coordinates return
400 without changing the map. Legacy maps must be rebuilt before calibration editing.
The dashboard prevents overlapping writes and clears stale selections and coordinates.
This does not establish physical positioning accuracy or authorize motion.

**Faster label review.** You no longer click tubes that are already right.

- After **Propose**, press **N** to jump to the next doubtful mark: any proposal that isn't a plain tube, or that is below the confidence slider. Then press **1-4** to relabel it, **Enter** to keep it, or **Del** to remove it. Each key moves on to the next doubtful mark.
- Press **A** to accept every remaining dashed mark as proposed.
- **Shift + drag a box** accepts everything inside it. **Ctrl + drag a box** sets everything inside it to the selected class.
- **Ctrl + Enter** (or **Save, reviewed & next**) saves, marks the capture reviewed and opens the next unreviewed capture with proposals already loaded.
- **Dim confident** fades the sure marks so the doubtful ones stand out.

## What's new in 1.2

- **Sheets with more than one drilled field.** When a pass-partition lane shifts the far tube field by part of a pitch, TubeMap now fits each field with its own offset instead of rejecting it. The lattice check reports the fields and their offsets. Proposals now cover the whole sheet, so unlabelled real tubes no longer become wrong "not a tube" training examples.
- **Training:**
  - **Safer "not a tube" examples.** These only come from detections inside the labelled region that are well away from every label.
  - **Incomplete-label check.** Reviewed captures with many unlabelled detections are listed after training, so you can fix them.
  - **Colour and texture features.** Rust, scale and deposits now count towards the plug decision.
  - **Gradient-boosted tree classifier** (scikit-learn). If scikit-learn is missing, a simpler model is used.
  - **Augmentation.** Brightness variants and small position and size jitter multiply the rare classes.
  - **Cross-validation by capture.** Every reviewed capture is held out once, and each score shows its number of examples and captures.
  - **Test set.** Tick "Test set" on the Label tab, and new and current profiles are compared on those captures, which are never trained on.
- **Appearance** (Settings tab):
  - theme (Auto, Light, Dark, or Field: high contrast for sunlight), accent colour, text size and spacing;
  - map line width, and every map and label colour.
  - Changes preview live, and Save keeps them on that computer.
- **Cameras:**
  - **Astra+:** now gets its 1920×1080 colour mode, which the camera only offers at 30 fps.
  - **OAK:** colour and depth now pair even when depth arrives a frame later. The device card shows colour and depth frame counts.

Profiles trained with 1.0/1.1 still work. Train again to get the new classifier.

## Install on the Pi

Copy this folder to the Pi and run this as your normal user (not with sudo):

```
bash install_pi.sh
```

The installer does the following:

- Creates `~/tubemap` with a Python environment. It installs `depthai` 2.x, or reuses the one in `~/projects/tube-positioning/oak_venv` if pip can't install it.
- Adds the OAK USB permission rule.
- Runs a self-test on simulated tubesheets.
- Imports your existing Image Lab labels from `~/.tube_positioning/image_lab`. Your original files are not changed.
- Installs and starts the `tubemap` service on port **8088**.
- Stops the old `tube-positioning` service, because only one program can use the OAK at a time.

Open `http://tube-positioner.local:8088`, or use the Pi's IP address with port `:8088`.

For an **Astra+**, add `--astra` (`bash install_pi.sh --astra`), or run `bash ~/tubemap/current/scripts/install_orbbec_astra.sh` later. The Astra+ is only supported by Orbbec's older SDK v1, which has no ready-made ARM64 package, so the script compiles it on the Pi (5–15 minutes) and installs Orbbec's USB rules.

To switch between the two systems at any time:

```
bash ~/tubemap/current/scripts/use_old_workspace.sh   # back to TubeWorkspace 0.14.x (port 8080)
bash ~/tubemap/current/scripts/use_tubemap.sh         # back to TubeMap
```

Logs: `journalctl -u tubemap -f`. Data folder: `~/.tubemap`, which holds captures, labels, profiles, `devices.json` and `settings.json`.

## Run on a Windows laptop (OAK-D Pro + Astra+)

See **LAPTOP_QUICKSTART.txt**. In short, extract the ZIP to a normal folder, plug in both cameras, and double-click `START_TUBEMAP_LAPTOP.bat`. The first run sets up Python and the camera SDKs. The dashboard then opens at `http://localhost:8088`, and it's the same dashboard as on the Pi. `CHECK_CAMERAS.bat` lists what the laptop can see.

Linux/macOS: `./run_local.sh`. It installs the OAK SDK only; the Astra+ needs `scripts/install_orbbec_astra.sh`.

## Everyday workflow

1. **Live:** pick a camera and press **Capture**. This saves the full-resolution colour image, aligned depth and the camera's calibration. **Capture all** grabs every running camera at the same moment.
2. **Map:** choose the capture and enter the pattern, pitch and tube ID (Setup C). You can pick a standard OD + BWG to fill in the ID. Optionally draw a region (ROI), then press **Build map**.
   - Every tube site is placed on the drilling lattice and gets a state: open, plugged, review, or no tube.
   - Cross-checks compare what you entered with the image:
     - **Pattern:** triangular vs square, seen in the image.
     - **Tube ID:** opening size relative to pitch, which doesn't depend on scale.
     - **Pitch:** checked against the depth-camera scale.
   - A FAIL in any check marks the map **Not ready** and says which value disagrees.
   - **Discover** mode measures the pattern and pitch without any input. It uses the depth plane for scale, so treat the result as approximate (±2–5% with the OAK-D Pro).
3. **Anchors:** click a tube, jog the positioner to it, and type its machine X/Y. With three or more widely spaced tubes you get a rigid (or rigid + scale) fit with residuals. Every site then gets machine coordinates, and the CSV includes them.
4. **Label:** press **Propose** to prefill marks from the map. Fix mistakes, then **Save & mark reviewed**:
   - Click a mark to set its class. The number keys switch classes: 1 Tube, 2 Blocked, 3 Unknown, 4 Not a tube.
   - Shift+click adds a mark.
   - Right-click deletes a mark.

   Keep the grey *not a tube* marks on pass lanes and removed tubes. They teach the model what an empty cell looks like.
5. **Model:** press **Train**. This tunes the opening detector and trains the state classifier. Whole captures are held out for scoring, so the numbers reflect new images rather than memorised ones. Compare the new profile with the current one, then **Activate** it if it's better. Nothing activates automatically. **Export dataset** gives images, native labels and YOLO labels for training a neural detector on a GPU PC.

### Astra+ as a depth-check camera

Mount the Astra+ next to the OAK so it sees the same sheet in roughly the same orientation, 0.6–1 m from the face. Add it on the Devices tab as **Orbbec (USB)** with the role **Depth checks (companion)**. From then on:

- **Capture** on the OAK also captures the Astra+ at the same moment.
- **Build map** on the OAK capture also maps the Astra+ image with the same pitch and ID. The OAK's own depth checks are then shown as INFO, and the map gets three extra checks:
  - **Pitch vs depth (Astra+):** the pitch cross-check, using the Astra+'s structured-light depth. Its scale comes from its own depth, so no camera-to-camera calibration is needed.
  - **Matched to Astra+:** the two tube patterns are lined up using the plugs and gaps in the sheet. Each OAK tube site then gets the Astra+ depth reading:
    - *surface at face*: a plug, cap or packed mouth
    - *open or deep*
    - *unclear*
    - *too small*
  - Tick **Depth evidence** above the image to see the readings. Magenta rings mark tubes where depth and image disagree.
- Depth readings are advisory by default. Settings › "Mark tubes REVIEW when depth disagrees with the image" turns disagreements into REVIEW.

Depth only helps if an opening covers enough depth pixels, and the check reports how many it gets. At 0.6–0.8 m, the Astra+'s 640×480 depth gives roughly 14–18 px across a 3/4″ tube. At 1.2 m it's about half that, and the check says "too coarse". The Astra+ is indoor only (0–40 °C); sunlight washes out its projector.

### Depth scale calibration

OAK stereo depth usually has a scale error of a few percent. On a sheet whose pitch you've measured with a steel rule (15 pitches in two directions), build a known-geometry map, then press **Calibrate depth scale**. The correction is saved to that camera and applied to later pitch checks. With an Astra+ companion, a second button calibrates the Astra+ depth scale.

## Devices

The Devices tab adds, edits, enables, restarts, tests and removes cameras. Each device type describes its own settings. Changes to live settings (exposure, focus, IR, preview rate) apply without a restart; other changes restart that camera only.

| Type | Use it for |
|---|---|
| Luxonis OAK (DepthAI) | OAK-D Pro over USB. You can choose the colour resolution (1248×936 fast, 2028×1520, or the full 4056×3040 sensor), focus locked at the factory calibration, IR projector, and exposure. Intrinsics and lens distortion come from the camera's EEPROM. |
| Network camera agent | **Orbbec Persee 2.** The Persee is a computer with the camera inside, so it runs `agent/tubemap_agent.py` and the Pi connects to it over Ethernet. Live view is JPEG; captures are lossless colour plus depth with intrinsics. See `agent/README.md`. |
| Orbbec (USB) | **Astra+** (SDK v1, built by `scripts/install_orbbec_astra.sh`) or Gemini/Femto (`pip install pyorbbecsdk2`). Only one SDK version can be installed at a time. You can set colour and depth resolution, and depth-to-colour alignment tries the camera's hardware alignment first, then software. Depth that can't be aligned is saved but not used for mapping. You can also turn off image mirroring. The camera search runs in a separate process, because the v1 SDK can crash when USB access fails and that must not take the dashboard down. |
| USB / network video | Webcams, capture cards, and RTSP/HTTP streams through OpenCV. Colour only; calibration can be pasted in as JSON. |
| Image folder | Replays saved images as if they were live. |
| Simulator | A synthetic tubesheet with known ground truth, for training people and checking the software. |

Every device also has **Depth scale correction** and **Depth scale uncertainty**. These change how strict the pitch cross-check is and never require a restart.

## What the numbers mean (and don't)

- **Tube positions** in the sheet frame are exact multiples of the pitch you entered: the drilling lattice, not individual blob centres. Their accuracy in machine coordinates depends on the anchors and the positioner, not on the camera.
- **Lattice residual** shows how well the openings fit one plane-projected lattice. It checks consistency, not physical accuracy.
- **Plug detection** is rule-based until you train a profile. With fewer than about 50 reviewed BLOCKED examples from 10 or more sheets, treat red marks as suggestions to check.
- **Passage** (open all the way through) is never confirmed by this camera.

## Tested here, and not yet tested

Local reliability candidate verification (2026-10-05): 55 Python tests passed and
four companion-camera tests failed; those same four failures reproduce on the
untouched 1.3.0 archive. Eleven JavaScript regression tests passed. The development
claims below are historical release notes, not a claim that this candidate's entire
suite passes. Real hardware and positioning accuracy remain unverified.

These were checked in development:

- 40 automated tests (`python -m pytest tests`) cover the mapping pipeline on simulated sheets (triangular and square, 0–30° tilt, lens distortion, pass lanes, plugs):
  - wrong-entry rejection
  - discovery
  - anchors
  - dataset import/export, including the old Image Lab ZIP format
  - training and hold-out scoring
  - device settings
  - the HTTP API
- One test uses your real OAK capture from TubeWorkspace 0.14.2. It finds 457 openings with a 0.32 px median lattice residual, and the pattern and tube-ID checks pass.
- The DepthAI 2.33 pipeline graph builds for all three resolutions.
- The network agent was tested end to end in simulation mode.
- Astra+ support (1.1):
  - The Orbbec driver runs against a stand-in for the v1 SDK. The tests cover hardware-alignment failure falling back to software, un-mirroring, 0.1 mm depth units, and unaligned depth being excluded.
  - The real SDK v1 wrapper was compiled with the install script and imports with NumPy 2.
  - An OAK + Astra+ simulated pair covers pattern matching with offset, mirrored and 90°-rotated cameras. On the simulator, plugs are found in depth, a wrong pitch is caught by the companion, and companion depth calibration works.
- The tests pass with NumPy 1.26 and 2.4, and OpenCV 4.11 and 5.0.

These have not been tested:

- A live OAK-D Pro, Pi timing, the installer on the real Pi image, and a real Persee 2.
- The Orbbec capture code has not been run against a real Astra+, Gemini or Persee. The simulator's depth is cleaner than real structured light, so depth plug detection needs checking on your bench sheet with real plugs.

Your first capture after installing is the real test. If something fails, use **Download capture** (Label tab) and send the ZIP.

## HTTP API (for the PLC side later)

`GET /api/captures/{id}/map` returns the full map as JSON. `GET /api/captures/{id}/map.csv` returns the tube table with sheet and machine coordinates. Devices, captures, labels and profiles are under `/api/...`; see `tubemap/server.py`.

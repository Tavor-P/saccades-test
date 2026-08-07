# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000):
detect a briefly-flashed grating's orientation, once during a saccade and
once during fixation (baseline), to measure saccadic suppression. Gaze is
tracked via a FLIR Blackfly S camera (Spinnaker SDK / `PySpin`) + MediaPipe
iris landmarks. See [README.md](README.md) for the full participant-facing
walkthrough (controls, output files, troubleshooting).

## Commands

All commands must be run from this project folder (the one containing this
file) using PsychoPy's own bundled Python, not a system Python — `-m` needs
the cwd here for `src`/`include` to resolve as top-level packages, and only
PsychoPy's Python has the right dependency set installed:

```
"C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt   # install deps
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment     # run the experiment
"C:\Program Files\PsychoPy\python.exe" -m src.dashboard.app                 # session dashboard at localhost:5000
"C:\Program Files\PsychoPy\python.exe" -m pytest                            # full test suite
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py      # one test file
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py::test_name -v  # one test
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.diagnose_camera  # capture+inspect one camera frame, no PsychoPy window
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.eye_position_manual_test [camera_index]  # live gaze-dot visual check
```

The test suite needs no camera or PsychoPy window (fake gaze/clock doubles,
Flask's test client) and runs in seconds.

## Architecture

**`include/` vs `src/`**: `include/` holds only dataclasses, enums, and
constants (`include/experiment/`, `include/eye_tracking/`) — no logic ever
goes here. `src/` mirrors that split (`src/experiment/`, `src/eye_tracking/`,
`src/dashboard/`) and holds all behavior.

**Framework-agnostic session state machines.** `PresaccadeSession` (baseline
phase) and `ExperimentSession` (saccade phase) each expose
`on_space()`/`on_response_key()`/`tick()`/`render_state()` and know nothing
about PsychoPy — `run_experiment.py`'s frame loop calls `tick()` +
`render_state()` once per rendered frame and applies the returned plain dict
to PsychoPy stimuli (`apply_render_state()`/`draw_all()`). Both phases share
one `ResultLogger` (one CSV, distinguished by a `phase` column) but run
independent `ZestStaircase` instances so each condition converges on its own
threshold. `TEST_MODE` in `include/experiment/constants.py` shortens trial
counts and runs the saccade phase first for smoke testing.

**Gaze tracking is behind an interface.** `include/eye_tracking/interfaces.py`
defines `GazeSource`; `ExperimentSession` only ever talks to that interface.
`WebcamGazeSource` (the only implementation actually in use) owns a `Camera`
+ `GazeTracker` pair on a background thread. `IOHubGazeSource` is an
untested placeholder for real eye-tracking hardware (Tobii/EyeLink) via
PsychoPy's ioHub, written against the documented API but never run against
real hardware.

**Camera → tracker pipeline** (`src/eye_tracking/`):
`Camera` wraps the Spinnaker SDK for a FLIR Blackfly S and runs its own
background thread that keeps only the single *latest* raw Mono8 frame (not a
queue) — intentional, since only the newest frame is ever consumed.
`GazeTracker` runs MediaPipe FaceLandmarker on that frame, downscaling it to
`TRACKING_FRAME_WIDTH` first (inference cost scales with pixel count; face
detection doesn't need full sensor resolution). Two gotchas worth knowing
before touching this code:
- `PySpin` isn't a normal pip package — it lives in PsychoPy's own "user
  packages" directory, which only gets added to `sys.path` as a side effect
  of `import psychopy`. Any module that imports `camera.py` must import
  `psychopy` first or `PySpin` fails with `ModuleNotFoundError` (this is why
  `tests/conftest.py` imports `psychopy` before anything else, and why
  standalone scripts like `diagnose_camera.py` do too).
- `PySpin` is built against numpy 1.x's C API; numpy 2.x breaks its import
  with `_ARRAY_API not found`. `requirements.txt` pins `numpy<2` plus
  matching opencv versions for this reason — don't casually bump either.

**Dashboard is a separate process reading the same files.** `src/dashboard/`
is an independent Flask app (`data_access.py` reads/writes the CSV,
`_meta.json`, and `settings.json` files under `data/`) — no shared process
or IPC with the experiment, just shared files on disk.

**Test doubles.** `tests/conftest.py` provides `FakeGazeSource` (an in-memory
`GazeSource` implementation tests set `.zone`/`.face_found`/`.position` on
directly) and a `fake_time` fixture that monkeypatches `PausableClock`'s time
source so timing-dependent state-machine tests (foreperiods, response
windows, debounce windows) run deterministically without real waits.

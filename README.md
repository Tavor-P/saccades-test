# saccades-test

A vision-science experiment testing **saccadic suppression** — the fact that
we're briefly "blind" to faint flashes during a rapid eye movement (a
*saccade*). Built with [PsychoPy](https://psychopy.org), following Diamond,
Ross & Morrone (2000, *J Neurosci* 20:3449-3455).

**How it works:** you stare at a target, then jump your gaze to a second one
when it appears. Around that eye movement, a faint striped pattern flashes
for a single frame, and you report which way it was oriented. A camera
tracks your eyes so the software knows exactly when your gaze moves. The
same test also runs *without* eye movements (a baseline), so the two results
can be compared to measure the suppression effect.

## Quick start

1. Install [PsychoPy Standalone](https://www.psychopy.org/download.html) — it bundles its own Python, so no separate install is needed.
2. Install the FLIR Spinnaker SDK from [Teledyne FLIR's site](https://www.teledynevisionsolutions.com/products/spinnaker-sdk/) — this includes the camera driver (`PySpin`) matched to your Python version.
3. Install the remaining dependencies into PsychoPy's Python:
   ```
   "C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt
   ```
4. From this project folder, run:
   ```
   "C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment
   ```

A dialog asks for a participant ID, which camera to use, and a contrast
floor (the faintest flash the test will ever try). Then two phases run back
to back, each starting with a few practice trials, ending in a graph
comparing the two phases' results. For a quick test run instead of the full
~15-minute session, set `TEST_MODE = True` in
`include/experiment/constants.py` (25 trials/phase instead of 100).

## Controls

| Key / action | Effect |
|---|---|
| **Space** | Start / calibrate / continue past results |
| **Up / Down** | "It looked vertical" |
| **Left / Right** | "It looked horizontal" |
| **Escape** | Quit immediately |
| **Click anywhere** | Pause (click again to resume — timing picks up exactly where it left off) |

## Output

Each session writes three files to `data/`:
- `results_<timestamp>.csv` — every trial (a `phase` column separates baseline from saccade; practice trials are excluded)
- `results_<timestamp>_meta.json` — participant info, settings, and camera calibration
- `accuracy_comparison.png` — the results graph

## Running the tests

```
"C:\Program Files\PsychoPy\python.exe" -m pytest
```

This runs in a few seconds — no camera or PsychoPy window needed, since the
tests use fake stand-ins for the camera and clock.

## Browsing past sessions

```
"C:\Program Files\PsychoPy\python.exe" -m src.dashboard.app
```

Open [http://localhost:5000](http://localhost:5000) for a local dashboard
listing every past session. You can label sessions with a name/gender/age,
filter the list, adjust the default contrast floor for future runs, and
delete sessions you don't need.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run commands from inside this project folder |
| `No module named 'mediapipe'`/`'flask'`/`'PySpin'` | The install in step 3 didn't take (often because `Program Files` wasn't writable) — rerun your terminal as Administrator and reinstall |
| `_ARRAY_API not found` or `numpy.core.multiarray failed to import` | numpy/opencv drifted out of sync with `PySpin` — reinstall `requirements.txt`, which pins working versions |
| Camera picker is empty | It only lists FLIR cameras via the Spinnaker SDK, not generic webcams — check the Blackfly S is plugged in |
| Eye tracking seems inaccurate | Make sure you completed calibration (look steadily at each circle when prompted) and picked the right camera if more than one was listed |
| HUD shows `face: no` | Run `"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.diagnose_camera` — it saves a photo to `data/camera_diagnostic.png` so you can check whether the camera itself (focus, lighting) is the problem |

## Project layout

```
include/          shared types, enums, constants (no logic)
src/
  eye_tracking/     camera capture (FLIR Blackfly S via PySpin) + MediaPipe eye tracking
  experiment/       trial logic, staircase, scoring, logging, results graph, run_experiment.py
  dashboard/        local Flask app for browsing past sessions
tests/             pytest suite (doesn't need a camera or PsychoPy window)
```

`src/eye_tracking/iohub_source.py` is an untested placeholder for switching
to dedicated eye-tracking hardware (Tobii/EyeLink) later.

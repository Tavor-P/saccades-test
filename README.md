# saccades-test

A vision-science experiment testing **saccadic suppression**: we are briefly "blind" to faint flashes during a rapid eye
movement (a *saccade*). Built with [PsychoPy](https://psychopy.org), following Diamond, Ross & Morrone (2000, *J Neurosci*
20:3449-3455). Developer documentation (what is where and why) is in [CLAUDE.md](CLAUDE.md).

**How it works:** the participant looks at a dot, then jumps their gaze to a second dot when it swaps. A faint striped
pattern flashes for a single frame, timed from the participant's own measured reaction time to land around the eye movement,
and they report its orientation. Gaze is recorded throughout so each flash can be placed against the real saccade onset
afterwards (`src.experiment.analyze_session`). The same test also runs *without* eye movements (a baseline) so the two
results can be compared.

Gaze comes from **Pupil Labs Neon glasses** (default, still head / chin rest) or, as an alternative picked in the startup
dialog, a FLIR Blackfly S camera + MediaPipe iris tracking.

## Quick start

1. Install [PsychoPy Standalone](https://www.psychopy.org/download.html) (it bundles its own Python).
2. Install the remaining dependencies into PsychoPy's Python (the Neon needs `pupil-labs-realtime-api`, included in
   `requirements.txt`; the webcam backend additionally needs the FLIR Spinnaker SDK for `PySpin`):
   ```
   .\scripts\py.ps1 -m pip install -r requirements.txt
   ```
3. Connect the Neon companion phone and this PC to the same Wi-Fi/hotspot and start the companion app. The Neon is
   auto-discovered; `NEON_DEVICE_ADDRESS` in `include/eye_tracking/constants.py` pins it to an IP instead.
4. From this project folder run the experiment:
   ```
   .\scripts\py.ps1 -m src.experiment.run_experiment
   ```

`scripts/py.ps1` (PowerShell) / `scripts/py.sh` (Git Bash) find PsychoPy's Python and run from the project root.

The startup dialog asks for: **Mode** (Real or Test), **Participant ID** (required for Real), **Gaze source**, the camera
(webcam source only), **Saccade contrast min/max (%)** (the saccade phase flashes fixed contrasts spaced between them),
a gaze indicator toggle (debug aid), **Run tutorial** (first-timer on-ramp) and **Record video** (webcam source only).
Test mode shortens everything and runs the saccade phase first; Real mode counterbalances phase order by participant ID.

Each phase:
- **Baseline (no eye movement)**: 25 trials (after a few practice trials), fixating the centre.
- **Saccade phase**: calibration (look at each of 3 targets), a short reaction-time test and a few practice trials, then
  the real trials (150 in Real mode, 30 in Test mode). If the tutorial ran, it replaces the reaction-time test and practice:
  it walks through calibration, how to answer with the arrow keys (3 correct in a row) and 8 practice trials that measure
  the reaction time.

Both phases end in a graph comparing baseline and saccade results. Instructions are spoken aloud and captioned; during real
(non-practice) trials the caption is hidden and silent, and the debug text at the top-left is shown in Test mode only.

## Controls

| Key | Effect |
|---|---|
| **Space** | Start / calibrate / continue, and **start each trial** (the wait between trials is the participant's break; the dot swaps 0.5-1 s after the press). A press during a trial's wait is ignored; every ignored press is counted per phase on the console and in `_meta.json` (`ignored_space_presses`) |
| **Up / Down** | "It looked vertical" |
| **Left / Right** | "It looked horizontal" |
| **Escape** | Quit immediately |

## Output

Each session writes to `data/` (git-ignored):
- `results_<timestamp>.csv`: every trial (a `phase` column separates baseline, saccade and reaction-time-test rows; practice trials are excluded)
- `results_<timestamp>_meta.json`: participant info, settings, constants snapshot, gaze quality, per-phase summaries
- `results_<timestamp>_gaze_trace.csv`: raw Neon gaze with phone capture times (input of `analyze_session`)
- `accuracy_comparison.png`: the results graph
- `results_<timestamp>_video.mp4` + `_video_timestamps.csv`: webcam source only, if "Record video" was on

## Analysing a session

```
.\scripts\py.ps1 -m src.experiment.analyze_session [timestamp] [--data-dir DIR]
```

Accuracy vs flash time relative to saccade onset (newest session by default); writes a report, plot and per-flash CSV next to the results.

## Running the tests

```
.\scripts\py.ps1 -m pytest -q
```

About 480 tests in ~70 s; no camera, glasses or PsychoPy window needed (fakes for the gaze source and clock). A headless
simulator runs whole sessions: see `tests/ai-tests/README.md`.

## Browsing past sessions

```
.\scripts\py.ps1 -m src.dashboard.app
```

Open [http://localhost:5000](http://localhost:5000) for a local dashboard listing every past session: label sessions,
filter them, adjust the default contrast floor for future runs, delete sessions.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run commands from inside this project folder (or through `scripts/py.ps1`) |
| `No module named 'mediapipe'`/`'flask'`/`'PySpin'` | The install did not take (often `Program Files` was not writable): rerun the terminal as Administrator and reinstall. Neon-only machines do not need `PySpin`/`mediapipe` |
| `_ARRAY_API not found` or `numpy.core.multiarray failed to import` | numpy/opencv drifted out of sync with `PySpin`: reinstall `requirements.txt`, which pins working versions |
| Neon not found | Phone and PC must share a network and the companion app must be running; or set `NEON_DEVICE_ADDRESS` |
| Camera picker is empty | Webcam backend only: it lists FLIR cameras via the Spinnaker SDK, not generic webcams |
| Eye tracking seems inaccurate | Redo calibration (look steadily at each target); with the Neon the head must stay still (chin rest) after calibrating |
| A webcam session's `_video.mp4` is missing or empty | OpenCV could not open the codec; look for `WARNING: video recording failed to start` in the console. The rest of the data is unaffected |

## Project layout

```
include/          shared types, enums, constants (no logic)
src/
  eye_tracking/     gaze sources (Neon default; FLIR Blackfly S + MediaPipe alternative), calibration mapping, traces
  experiment/       session state machines, staircase, scoring, logging, analysis, run_experiment.py
  dashboard/        local Flask app for browsing past sessions
tests/             pytest suite (tests/ai-tests/ also holds the headless simulator tools)
```

# saccades-test

A gaze-contingent saccade experiment built with [PsychoPy](https://psychopy.org),
following Diamond, Ross & Morrone (2000, *J Neurosci* 20:3449-3455): fixate one
target, saccade to another when it appears, and detect a brief flash timed
around the saccade at varying contrast — with a baseline (no-saccade) phase
first, so the two conditions can be compared directly.

## Quick start

**1. Install PsychoPy** (if you haven't already): download the Standalone
installer from [psychopy.org/download](https://www.psychopy.org/download.html)
and install it normally. It bundles its own Python — you don't need a
separate Python install.

**2. Install the extra dependencies** into PsychoPy's bundled Python
(mediapipe for the webcam eye tracker, Flask for the session dashboard):

```
"C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt
```

**3. Run it.** Open a terminal, `cd` into this project folder (the one
containing this README), then:

```
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment
```

That's it — a window opens and the experiment starts.

## What running it looks like

A small dialog asks for a participant ID, which camera to use, and a
contrast floor (cancel it to quit before the window even opens):

- The **camera picker** lists every camera index that actually opens, so if
  you've got more than one connected (e.g. a laptop's built-in webcam plus an
  external one) you pick per run, no code edit needed.
- The **contrast floor** field is pre-filled with the default set in the
  dashboard's settings box (see "Browsing past sessions" below) - it's the
  lowest contrast the adaptive staircase will ever test at, as a percent.
  Override it here for a one-off run without touching the dashboard default.
  Leaving it blank, non-numeric, or out of range falls back to that default
  rather than failing the dialog.

Then the session runs two phases back to back, each starting with a few
throwaway practice trials, then a results graph:

1. **Baseline phase** — fixate the center of the screen. A faint grating
   flashes briefly for a single frame, its contrast adjusted trial-by-trial
   by an adaptive staircase (sometimes it doesn't flash at all). Press
   **UP/DOWN** if it looked vertical, **LEFT/RIGHT** if horizontal - guess if
   you're not sure rather than not answering, since the staircase treats a
   withheld response as much stronger evidence you couldn't see it at all
   than an honest 50/50 guess would be. The first couple of trials are marked
   "Practice" on screen and don't count toward your results.
2. **Saccade phase** — press Space to calibrate (look at each circle when
   asked), then saccade back and forth between two circles as they appear.
   The same grating may flash around each saccade — report its orientation
   the same way (UP/DOWN/LEFT/RIGHT), guessing if unsure. Again, a couple of
   practice trials run first. A small faint dot snaps onto whichever of the
   two circles (or the midpoint) the tracker currently thinks you're looking
   at, the whole phase through - it's the exact same left/center/right
   classification the experiment itself uses, so if the dot isn't landing
   where you're actually looking, that's real evidence tracking has gone
   wrong, not just visual noise.
3. **Results** — a graph comparing the estimated contrast detection
   threshold for both phases (with a credible interval) plus the fitted
   psychometric curve each threshold came from, so you can see both the size
   of the saccadic-suppression effect and how good the underlying fit is.

`include/experiment/constants.py` has a `TEST_MODE` flag for quick smoke
tests: 25 trials/phase and the saccade phase first, instead of the full
100-trial, baseline-first run.

## Controls

| Key / action | Effect |
|---|---|
| **Space** | Start / calibrate / advance past the results screen |
| **Up / Down** | Respond "vertical" to a flash |
| **Left / Right** | Respond "horizontal" to a flash |
| **Escape** | Quit immediately, at any point |
| **Click anywhere** | Pause — rest your eyes, click again to resume |

Pausing is timer-safe: whatever's in flight (a wait period, a response
window) picks up exactly where it left off, no matter how long you pause for.

## Output

Every trial from both phases is logged to one file:
`data/results_<timestamp>.csv` (a `phase` column tells the two apart;
practice trials aren't included). Alongside it,
`data/results_<timestamp>_meta.json` records the participant ID, start time,
`TEST_MODE` state, the viewing-distance/screen/grating assumptions used, the
contrast floor the session actually ran with, and the webcam calibration
ratios once calibration completes. The final comparison graph is saved to
`data/accuracy_comparison.png`.

The dashboard's own default contrast floor (see below) lives separately in
`data/settings.json`, since it isn't tied to any one session.

## Running the tests

PsychoPy's bundled Python already includes pytest and matplotlib, so as long
as you've done step 2 above (for Flask, used by the dashboard tests) there's
nothing extra to install:

```
"C:\Program Files\PsychoPy\python.exe" -m pytest
```

Run this from the project folder (same requirement as running the experiment
itself). The suite covers the trial-scheduling/staircase/scoring logic, the
session state machines, and the dashboard's data access + routes, all using
fake gaze/clock doubles and Flask's test client — it doesn't open a real
PsychoPy window, touch a camera, or start a real server, so it runs in a few
seconds.

## Browsing past sessions

A small local web dashboard lists every session in `data/`, lets you attach a
name/gender/age to each one (saved into that session's `_meta.json`), and
shows its detection-threshold graph on demand:

```
"C:\Program Files\PsychoPy\python.exe" -m src.dashboard.app
```

Then open [http://localhost:5000](http://localhost:5000). Edits to
name/gender/age save automatically as you tab out of the field. This is a
plain local Flask dev server (not meant to be exposed beyond your own
machine).

A settings box above the table sets the **default contrast floor (%)** -
the lowest contrast the adaptive staircase will test at, pre-filled into the
experiment's startup dialog on the next run (still overridable there
per-session). It's clamped to between the lowest contrast an 8-bit display
can actually render and ZEST's own contrast ceiling, since going below that
floor would silently test invisible contrasts again - the exact bug
`ZEST_LOG_CONTRAST_MIN` was introduced to fix. Click "Save default" to
persist it to `data/settings.json`.

The filter bar above the table searches by name (substring), gender
(substring), and age range - live, no page reload. Check "Exclude matches
instead of showing them" to flip it into a hide filter (e.g. "hide anyone
named Test").

Each row also has a **Delete** button, and **Delete all visible** (in the
filter bar) removes every session currently shown - so filtering down to
junk sessions first and then deleting them in bulk is the intended
workflow. Both ask for confirmation first and are permanent - a session's
CSV, metadata, and cached graph are gone for good, same as deleting the
files by hand.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — you have to `cd` into
  this project's folder *first*; `-m` needs the current directory to be here
  so `src`/`include` resolve as top-level packages.
- **`ModuleNotFoundError: No module named 'mediapipe'`** (or `'flask'`) —
  step 2 above didn't take. If `Program Files` isn't writable, pip silently
  installs to your per-user site-packages instead, which can go unnoticed in
  a different shell. Simplest fix: open the terminal **as Administrator** and
  re-run the step 2 command so it installs directly into PsychoPy's own
  folder.
- **Webcam eye tracking seems off** — the calibration step (look at each
  circle, press Space) has to actually happen for gaze detection to be
  accurate; if you skip it or don't hold your gaze steady during it,
  detection will be unreliable for the rest of the saccade phase. Also
  double check the startup dialog's camera picker actually selected the
  camera facing you - it's easy to pick the wrong one when two are listed.
- **HUD shows `face: no`** — the camera is opening fine but MediaPipe isn't
  finding a face in its frames. Run the diagnostic script to see exactly
  what the tracker is looking at, without needing to run the full
  experiment to hit the problem:
  ```
  "C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.diagnose_camera
  ```
  It saves a captured frame to `data/camera_diagnostic.png` and reports
  whether a face was found in it. If the saved image doesn't look like a
  normal, well-lit photo of your face (black, grayscale/IR-washed, garbled
  colors, or not actually you), the camera itself is the problem - some
  budget/generic USB cameras are IR-only or have unusual color handling
  that a visible-light face detector can't work with.

## Project layout

```
include/          shared types, enums, and constants (no logic)
  eye_tracking/     GazeZone/GazeSample types, camera/model config
  experiment/       Target/TrialSpec/TrialResult types, trial+timing constants
src/
  eye_tracking/     webcam gaze tracker (MediaPipe iris tracking) + camera capture
  experiment/       the two session state machines, trial scheduling, ZEST
                     staircase, shared outcome scoring, CSV/metadata logging,
                     results graph, and run_experiment.py (entry point)
  dashboard/        local Flask app for browsing past sessions - see
                     "Browsing past sessions" above
tests/             pytest suite for everything in src/experiment and
                     src/dashboard (no PsychoPy window or camera needed -
                     see "Running the tests" above)
```

`src/eye_tracking/iohub_source.py` is a not-yet-tested placeholder for
switching to a real hardware eye tracker (Tobii/EyeLink/etc.) via PsychoPy's
ioHub later — the webcam tracker is what's actually used today.

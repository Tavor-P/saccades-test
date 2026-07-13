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

**2. Install the one extra dependency** into PsychoPy's bundled Python:

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

A small dialog asks for a participant ID first (cancel it to quit before the
window even opens). Then the session runs two phases back to back, each
starting with a few throwaway practice trials, then a results graph:

1. **Baseline phase** — fixate the center of the screen. A faint grating
   flashes briefly for a single frame, its contrast adjusted trial-by-trial
   by an adaptive staircase (sometimes it doesn't flash at all). Press
   **Space** whenever you see it. The first couple of trials are marked
   "Practice" on screen and don't count toward your results.
2. **Saccade phase** — press Space to calibrate (look at each circle when
   asked), then saccade back and forth between two circles as they appear.
   The same grating may flash around each saccade — press **Space** if you
   see it. Again, a couple of practice trials run first.
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
| **Space** | Start / calibrate / respond ("I saw it") |
| **Escape** | Quit immediately, at any point |
| **Click anywhere** | Pause — rest your eyes, click again to resume |

Pausing is timer-safe: whatever's in flight (a wait period, a response
window) picks up exactly where it left off, no matter how long you pause for.

## Output

Every trial from both phases is logged to one file:
`data/results_<timestamp>.csv` (a `phase` column tells the two apart;
practice trials aren't included). Alongside it,
`data/results_<timestamp>_meta.json` records the participant ID, start time,
`TEST_MODE` state, the viewing-distance/screen/grating assumptions used, and
the webcam calibration ratios once calibration completes. The final
comparison graph is saved to `data/accuracy_comparison.png`.

## Running the tests

PsychoPy's bundled Python already includes pytest and matplotlib, so there's
nothing extra to install:

```
"C:\Program Files\PsychoPy\python.exe" -m pytest
```

Run this from the project folder (same requirement as running the experiment
itself). The suite covers the trial-scheduling/staircase/scoring logic and
the session state machines using fake gaze/clock doubles — it doesn't open a
real PsychoPy window or touch a camera, so it runs in a few seconds.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — you have to `cd` into
  this project's folder *first*; `-m` needs the current directory to be here
  so `src`/`include` resolve as top-level packages.
- **`ModuleNotFoundError: No module named 'mediapipe'`** — step 2 above
  didn't take. If `Program Files` isn't writable, pip silently installs to
  your per-user site-packages instead, which can go unnoticed in a different
  shell. Simplest fix: open the terminal **as Administrator** and re-run the
  step 2 command so it installs directly into PsychoPy's own folder.
- **Webcam eye tracking seems off** — the calibration step (look at each
  circle, press Space) has to actually happen for gaze detection to be
  accurate; if you skip it or don't hold your gaze steady during it,
  detection will be unreliable for the rest of the saccade phase.

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
tests/             pytest suite for everything in src/experiment (no PsychoPy
                     window or camera needed - see "Running the tests" above)
```

`src/eye_tracking/iohub_source.py` is a not-yet-tested placeholder for
switching to a real hardware eye tracker (Tobii/EyeLink/etc.) via PsychoPy's
ioHub later — the webcam tracker is what's actually used today.

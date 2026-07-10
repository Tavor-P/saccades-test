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

The session has two phases, back to back, then a results graph:

1. **Baseline phase** — fixate the center of the screen. A square flashes
   briefly at various contrast levels (sometimes not at all). Press
   **Space** whenever you see it.
2. **Saccade phase** — press Space to calibrate (look at each circle when
   asked), then saccade back and forth between two circles as they appear.
   A square may flash around each saccade — press **Space** if you see it.
3. **Results** — a graph comparing detection accuracy vs. contrast for both
   phases, so you can see the size of the saccadic-suppression effect.

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
`data/results_<timestamp>.csv` (a `phase` column tells the two apart). The
final comparison graph is saved to `data/accuracy_comparison.png`.

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
  experiment/       the two session state machines, trial scheduling, CSV
                     logging, results graph, and run_experiment.py (entry point)
```

`src/eye_tracking/iohub_source.py` is a not-yet-tested placeholder for
switching to a real hardware eye tracker (Tobii/EyeLink/etc.) via PsychoPy's
ioHub later — the webcam tracker is what's actually used today.

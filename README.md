# saccades-test

Gaze-contingent saccade experiment (PsychoPy), following Diamond, Ross & Morrone
(2000, *J Neurosci* 20:3449-3455): fixate one target, saccade to the other when
it appears, detect a brief perisaccadic flash at varying contrast.

## Setup

Uses the PsychoPy Standalone install (psychopy.org) directly - it bundles its
own Python. Install the one extra dependency into it:

```
"C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt
```

## Run

`-m` needs the current directory to be this project root (so `src`/`include`
resolve as top-level packages) - `cd` here first, then run:

```
cd /d "C:\Users\tavor\Downloads\saccades-test\.claude\worktrees\starting-ending-branch-9c5c3b"
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment
```

Space starts calibration (look at each circle when asked), then runs 40
trials. Escape quits. Results are logged to `data/results_<timestamp>.csv`.

Eye tracking currently comes from the webcam (`src/eye_tracking`, MediaPipe
iris tracking). `src/eye_tracking/iohub_source.py` is a not-yet-tested
placeholder for switching to a real hardware eye tracker (Tobii/EyeLink/etc.)
via PsychoPy's ioHub later.

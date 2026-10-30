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

One folder **per participant**, and inside it one subfolder **per run of the experiment** (a run is one start of the program, with both phases;
it is *not* one flash trial). Everything is under `data/` (git-ignored):

```
data/
  <participant id>/                  blank id in Test mode -> "anonymous"; characters Windows forbids become "_"
    2026-10-25_14-03-11[_test]/      one run, local start time; "_test" for Test mode
      trials.csv                     one row per trial (columns below)
      session.json                   participant, settings, hardware, results summary, gaze quality, ignored SPACE presses, glasses-video sync
      gaze_trace.csv                 raw Neon gaze with the phone's capture times (input of analyze_session)
      comparison_graph.png           the end-of-run results graph
      analysis/                      written by analyze_session: report.txt, plot.png, flashes.csv, suppression.json
      glasses/                       the phone's video + video_sync.json (see "Glasses video" below)
```

`trials.csv` columns, left to right: **who/what** `phase` (baseline / saccade / rt_test), `trial`; **stimulus** `from`, `to` (`left`/`right`: the two
symbols of the screen are the left and right dot), `flash_shown`, `contrast_pct`, `grating_orientation`; **timing** `planned_offset_ms` (the
flash's offset around the average reaction time), `flash_ms_after_go_cue`, `reaction_ms`, `flash_unix_ms` (PC wall clock of the flash, used by
`analyze_session`); **response** `answered`, `answer_orientation`, `answer_ms`; **result** `outcome` (correct / wrong / no_answer / false_alarm /
correct_rejection / detected), `correct`; **quality** `tracking_ok`, `flash_during_saccade`. Booleans are `yes`/`no`. Practice trials are not written.
The code-side schema is `TRIAL_COLUMNS` in `src/experiment/logger.py`.

Sessions recorded before this layout (loose `data/results_<timestamp>.csv`, `_meta.json`, `_gaze_trace.csv`, with `dot`/`cross` values) are never
moved or rewritten; the dashboard and `analyze_session` still read them.

## Analysing a session

```
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir DIR]
```

Accuracy vs flash time relative to saccade onset. `session` is the newest one by default (either layout), or a unix timestamp, a run folder
name (`2026-10-25_14-03-11`), `participant/folder`, or a path. The report, plot, per-flash CSV and suppression json go to the run's `analysis/` folder
(old loose-file sessions get `analysis_<timestamp>*` files next to them, as before).

## Glasses video

The Neon phone records the glasses' scene video itself (nothing is streamed to the PC): the experiment starts that recording when the saccade
phase starts and stops/saves it when the phase ends (`recording` in `session.json` says `started`/`saved`/`failed (reason)`; a failure never stops the session).
So the phone must show the Companion app with a wearer and workspace selected. To line the video up with the PC clock, a short step before the first
trial flashes the whole screen **green** three times (150 ms on, 300 ms off; "A green flash will show." is the only spoken line) and once more at the end;
the PC wall-clock time of every flash is logged in `session.json` (`glasses_video.sync_flashes`). Nothing is added to the trials themselves.

After the session:
1. Get the recording off the phone: in the Neon Companion app, open the recording and export it (or plug the phone in by USB and copy the
   recording folder; or download it from Pupil Cloud). You need the scene video as an `.mp4`.
2. Attach it to the run:
   ```
   .\scripts\py.ps1 -m src.experiment.import_glasses_video data\<participant>\<run folder> C:\path\to\exported_scene_video.mp4
   ```
   This copies the video into the run's `glasses/` folder, finds the green flashes in it, and writes `glasses/video_sync.json` (clock offset, residual,
   `ok` or `check manually`) and `glasses/flash_video_times.csv` (where each flash trial is in the video). A PC time `T` (unix ms, e.g. `flash_unix_ms`)
   is at `T / 1000 - offset_s` seconds into the video.

Not yet verified on real glasses footage: how well the scene camera sees the green flash from the participant's distance, and the frame-rate/clock behaviour of the exported video.

## Running the tests

```
.\scripts\py.ps1 -m pytest -q
```

About 520 tests in ~60 s; no camera, glasses or PsychoPy window needed (fakes for the gaze source and clock). A headless
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
  experiment/       session state machines, staircase, scoring, logging (run folders), analysis, glasses-video sync, run_experiment.py
  dashboard/        local Flask app for browsing past sessions
tests/             pytest suite (tests/ai-tests/ also holds the headless simulator tools)
```

# saccades-test

A vision-science experiment testing **saccadic suppression** — the fact that
we're briefly "blind" to faint flashes during a rapid eye movement (a
*saccade*). Built with [PsychoPy](https://psychopy.org), following Diamond,
Ross & Morrone (2000, *J Neurosci* 20:3449-3455).

**How it works:** first, a quick reaction-time test measures how long it
takes you to move your eyes after a "go" beep. Then, in the real trials, you
stare at a target and jump your gaze to a second one when it beeps. Timed to
land around that eye movement — based on your own measured reaction time,
not on watching your eyes in real time — a faint striped pattern flashes for
a single frame, and you report which way it was oriented. A camera tracks
your eyes throughout so the software can tell afterward whether each flash
actually landed during the eye movement or not (trials where it didn't are
excluded automatically). The same test also runs *without* eye movements (a
baseline), so the two results can be compared to measure the suppression
effect.

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

A dialog asks for a participant ID, which camera to use, a contrast floor
(the faintest flash the test will ever try), whether to show a live gaze
indicator (a debug aid, leave off for real participants), whether to run the
first-timer tutorial first, and whether to record a session video (on by
default — see [Output](#output)). **Leave Participant ID blank for a quick
test run** (shorter reaction-time test and practice block, a much smaller
saccade-phase trial cap, saccade phase first) — entering an ID runs a full
data-collection session. The baseline phase is always a fixed 25 trials,
in either mode.

Each phase's flow:
- **Baseline (no eye movement)**: a fixed 25 trials (after a few practice
  trials), fixating the center the whole time.
- **Saccade phase**: calibration (look at each of 3 targets in turn), then a
  short reaction-time test (a beep plays, look at the target that just
  appeared, as fast as you can), then a few practice trials, then the real
  trials. The real trials aren't a fixed count — the session keeps going
  until it's collected enough reliable data for a good threshold estimate
  (or hits a safety cap), so how long this phase takes varies session to
  session.

If you ran the tutorial, it walks through calibration, the response-key
mapping, gaze practice, its own short reaction-time test, and a dress
rehearsal before the real session starts — the saccade phase then reuses
that calibration instead of repeating it.

Both phases end in a graph comparing the baseline and saccade results.

Every instruction is spoken aloud and also captioned at the bottom of the
screen, so you don't have to rely on catching it the first time it's said.

## Controls

| Key / action | Effect |
|---|---|
| **Space** | Start / calibrate / continue past results |
| **Up / Down** | "It looked vertical" |
| **Left / Right** | "It looked horizontal" |
| **Escape** | Quit immediately |
| **Click anywhere (baseline phase)** | Pause (click again to resume — timing picks up exactly where it left off) |
| **Click anywhere (saccade phase)** | Opens a pause menu with two buttons: **Return** (resume as-is) or **Return and Recalibrate** (redo eye-position calibration and the reaction-time test before resuming — worth using if you've moved, or if data looks off) |

Pausing mid-trial in the saccade phase discards whatever that trial was
doing — it doesn't count, so pause freely if you need to.

## Output

Each session writes to `data/`:
- `results_<timestamp>.csv` — every trial (a `phase` column distinguishes baseline, saccade, and reaction-time-test rows; practice trials are excluded)
- `results_<timestamp>_meta.json` — participant info, settings, and camera calibration
- `accuracy_comparison.png` — the results graph
- `results_<timestamp>_video.mp4` + `results_<timestamp>_video_timestamps.csv` — if "Record video" was left on (the default): the saccade phase's camera feed, plus a wall-clock timestamp for every frame, so a session can be replayed and lined up exactly against the trial data (each trial's flash also has its own wall-clock timestamp in the results CSV, for the same reason)

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
| A session's `_video.mp4` is missing or empty | Video recording fails silently if your machine's OpenCV build can't open the video codec — check the terminal output from that run for a `WARNING: video recording failed to start` message. The rest of the session's data is unaffected either way |
| Reaction times look implausibly fast (well under ~100ms) or trials rarely land during the saccade | The reaction-time test needs a clean, steady calibration to measure real saccade onsets rather than tracking noise — try "Return and Recalibrate" from the pause menu, or restart the session with a fresh calibration |

## Project layout

```
include/          shared types, enums, constants (no logic)
src/
  eye_tracking/     camera capture (FLIR Blackfly S via PySpin), MediaPipe eye tracking,
                    optional session video recording
  experiment/       trial logic, staircase, scoring, logging, results graph, run_experiment.py
  dashboard/        local Flask app for browsing past sessions
tests/             pytest suite (doesn't need a camera or PsychoPy window)
```

`src/eye_tracking/iohub_source.py` is an untested placeholder for switching
to dedicated eye-tracking hardware (Tobii/EyeLink) later.

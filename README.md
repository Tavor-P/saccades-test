# saccades-test

A PsychoPy experiment that measures **saccadic suppression**: around the time of a rapid eye movement (a *saccade*) we become
much less sensitive to brief faint flashes. The participant looks at a dot, jumps their gaze to a second dot when it swaps,
and reports the orientation (vertical or horizontal stripes) of a briefly flashed faint grating at the middle of the screen. Gaze comes from Pupil Labs Neon glasses.
This is the `main` branch (flashed grating). The `line-stimulus` branch replaces the grating with a thin line above/below the target.
Developer documentation is in [CLAUDE.md](CLAUDE.md).

## Why, and where this stands

The aim is a **feasibility pilot (n = 1)** for a grant proposing to compare saccadic suppression in people with traumatic
brain injury (TBI) and without. The question for a person is: how much worse is their contrast sensitivity for a flash during a
saccade than during steady fixation? (The suppression index below.)

Honest status:
- This is a pipeline and a first participant, not a group result. No TBI/control comparison exists yet.
- The only earlier data are one participant, in test mode, with an older setup (10-20 % and 20-80 % grating contrasts, earlier timing). A few flashes close to
  saccade onset were answered worse than flashes 50-300 ms away, small numbers and a window chosen after seeing the first session.
- **The current setup, its contrasts, the new results layout, the tutorial and the glasses-video steps have been run only against
  simulators, fake devices and unit tests, not yet on real glasses with a person.** See [Known limitations](#known-limitations).

## Research background

**Diamond, Ross & Morrone (2000)**, *J Neurosci* 20:3449-3455, is the design this experiment started from. The paper itself was *not
checked* for this README; what follows is only what this repository's code and comments record about it.
- Taken from it: a participant fixates a symbol, a jittered wait (800-1200 ms after a warning tone in the paper; here 500-1000 ms after
  the participant's own SPACE press) precedes the saccade target, the flash is a single display frame, contrast is varied with a ZEST
  staircase (here only if the fixed contrasts are switched off), and the same detection task is run during saccades and during fixation (here the
  fixation condition comes from flashes inside the saccade phase, see below).
- Where we differ: their grating parameters were changed (spatial frequency 0.12 cpd instead of the paper's 0.04, per the comments in `include/experiment/constants.py`, so that several cycles fit in the Gaussian patch; the patch has a 4.8 deg Gaussian envelope); the tone is a go cue at target onset; the saccade phase uses fixed contrasts
  rather than the staircase; gaze is Neon, not a lab eye tracker.

**Braun, Schutz & Gegenfurtner (2021)**, "Age effects on saccadic suppression of luminance and color", *J Vis* 21(6):11,
doi:10.1167/jov.21.6.11, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8237129/>. Abstract and Methods were read for this README (the Results and
Discussion were not checked). In that paper participants (ages 8-78; about 100 analysed) were about 66% less sensitive to a luminance line during saccades.
- Taken from it (this branch keeps the grating; the line is on `line-stimulus`):
  - **Fixation control:** the same task during steady fixation.
  - **Saccade/fixation gates:** a saccade trial counts only for a well-formed saccade (latency 100-500 ms, amplitude within a range, eye speed
    at the flash in a plausible band); a fixation trial only if no saccade or blink is within +-100 ms of the flash. Our gates live in
    `src/experiment/flash_gates.py`.
  - **Suppression index:** `1 - CS_saccade / CS_fixation`, with contrast sensitivity CS = 1 / threshold contrast.
  - Not taken here: their stimulus (a full-width line, vertical Gaussian SD 0.15 deg, 2 deg above/below the target, above/below 2AFC), which is what the `line-stimulus` branch does.
- Where we differ:

| | Braun et al. 2021 | This experiment |
|---|---|---|
| Eye tracker | EyeLink 1000, 1000 Hz | Pupil Labs Neon, ~160-200 Hz, gaze reaches the PC late |
| Flash trigger | live: 15 ms after online saccade-onset detection | scheduled from the person's average reaction time (+-80 ms offsets), judged **after the fact** from the raw gaze trace |
| Stimulus | full-width line, above/below 2AFC | Gaussian-windowed grating (0.12 cpd), vertical/horizontal 2AFC |
| Display | 120 Hz, 8 ms flash | 60 Hz monitor, one frame (~17 ms) |
| Saccade size | 10 deg target step | about 30-40 deg (dot at 15 % and cross at 85 % of the screen width) |
| Thresholds | 1-up/2-down staircase, then a psychometric fit | method of constant stimuli (fixed contrasts, default 10/15/20 %), then a Weibull fit |
| Viewing | 90 cm, lab setup | chin rest, geometry still to be measured |

## Setup

1. **Hardware:** a Windows PC, a monitor (60 Hz here), a chin rest or other way to keep the head still (the Neon scene camera moves with the
   head, so calibration assumes a still head), Neon glasses, and the Neon Companion phone on the same Wi-Fi/hotspot as the PC.
2. Install [PsychoPy Standalone](https://www.psychopy.org/download.html) (it bundles its own Python). Always run Python through the wrapper
   `.\scripts\py.ps1` (PowerShell) or `scripts/py.sh` (Git Bash); it finds PsychoPy's Python and runs from the project root.
3. Install dependencies into that Python:
   ```
   .\scripts\py.ps1 -m pip install -r requirements.txt
   ```
   The FLIR Blackfly S + MediaPipe backend additionally needs the Spinnaker SDK (`PySpin`); Neon-only machines do not.
4. Start the Neon Companion app on the phone. The experiment looks for the Neon at `NEON_DEVICE_ADDRESS` in
   `include/eye_tracking/constants.py` (pinned to a phone IP at the time of writing; set it to your phone's address, or to `None` to auto-discover).
5. **Measure your screen.** `VIEWING_DISTANCE_CM` and `SCREEN_HEIGHT_CM` in `include/experiment/constants.py` are still placeholders. The grating's
   size and spatial frequency (set in degrees) and the saccade size in degrees, are only right once they match the real monitor and seat distance.

## Running a session

```
.\scripts\py.ps1 -m src.experiment.run_experiment
```

The startup dialog asks for:

| Field | Meaning |
|---|---|
| Mode | **Real** (full length, needs a participant ID) or **Test** (shortened, debug text shown top-left) |
| Participant ID | required for Real; blank in Test becomes `anonymous` |
| Gaze source | Pupil Labs Neon (default) or Blackfly S + MediaPipe |
| Camera | webcam source only |
| Saccade contrast min / max (%) | the saccade phase flashes fixed contrasts spaced between these (defaults 10 and 20, three levels: 10, 15, 20) |
| Show gaze indicator | debug aid |
| Run tutorial | first-timer on-ramp (below) |
| Record video | webcam source only |

What happens, in order (`Space` advances everything; `Esc` quits at once):
1. **Tutorial** (if chosen, narrated aloud and captioned; input is held while it speaks): calibration, a quiz on how to answer (gratings of both orientations, UP/DOWN
   or LEFT/RIGHT, 3 correct in a row) and 8 practice trials whose saccade latencies seed the reaction-time average.
2. **Calibration** (saccade phase): look steadily at each of 3 targets, 3 rounds. Without the tutorial, a short reaction-time test and a few practice trials follow.
3. **Green sync flashes** (saccade phase, Neon): the screen flashes green three times before the first trial and once at the end. See [Glasses video](#glasses-video).
4. **Trials.** The participant presses **SPACE** to start each trial (the wait between trials is their break). A dot and a cross sit at the left and
   right; 0.5-1 s after the press the dot swaps to the cross (a go cue with a beep) and the participant looks to it as fast as they can.
   A faint grating flashes for one frame at the middle of the screen (between the two symbols), at the average reaction time plus a random offset of -80..+80 ms (+-10 ms jitter), plus
   some control flashes at +400/+500 ms when the eye has already landed. The participant presses **Up** or **Down** for vertical stripes, **Left** or **Right** for horizontal stripes, within 1 s;
   a missed flash is still guessed. **Every trial shows a flash**; there are no catch trials. In real trials the caption is hidden and silent.
   Saccade phase: 150 trials (30 in Test).
5. The run ends after the last trial; the console prints the `analyze_session` command to run on it. Fixation data come from the control flashes and the
   far-from-saccade flashes of the same trials, sorted after the fact (see [Analysing a session](#analysing-a-session)).

The phone records the glasses' scene video during the saccade phase (see below); the recording may need the Companion app open with a wearer and workspace selected (not verified yet); a failure never stops the session.

## Output

Everything is under `data/` (git-ignored): one folder per participant, one subfolder per run (one start of the program).

```
data/<participant id>/<YYYY-MM-DD_HH-MM-SS>[_test]/
  trials.csv           one row per trial
  session.json         metadata, see below
  gaze_trace.csv       raw Neon gaze with the phone's capture times
  analysis/            report.txt, plot.png, flashes.csv, suppression.json (written by analyze_session)
  glasses/             the phone's video + video_sync.json + flash_video_times.csv (written by import_glasses_video)
```

`trials.csv` columns (booleans are `yes`/`no`; practice trials are not written): `phase` (saccade / rt_test; `baseline` only in old runs), `trial`, `contrast_pct`,
`flash_shown`, `flash_during_saccade` (the live, unreliable judgement), `answered`, `correct`, `outcome` (correct / wrong / no_answer / detected),
`grating_orientation` (vertical / horizontal), `answer_orientation`, `flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`, `tracking_ok`. The direction of the eye movement is
the trial's parity within its phase. The code-side schema is `TRIAL_COLUMNS` in `src/experiment/logger.py`.

`session.json` sections: `participant`, `run` (mode, start time, git state, versions), `settings` (screen geometry, grating parameters, contrasts, calibration, constants),
`hardware`, `results` (per-phase summaries), `gaze_quality` (rate, gaps, latency, clock offset), `ignored_space_presses` (per phase and reason) and
`glasses_video` (recording state and sync-flash times).

Older sessions (loose `data/results_<timestamp>.csv`, `_meta.json`) are never rewritten and are still read.

## Analysing a session

```
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir DIR]
```

`session` defaults to the newest; it can also be a unix timestamp, a run folder name, `participant/folder`, or a path. The script finds the
saccades in `gaze_trace.csv` (using the phone's capture times, so the late arrival at the PC does not matter), places each flash against
the nearest saccade onset, and writes accuracy vs time-from-onset. It labels every flash with the pre-specified gates in
`src/experiment/flash_gates.py`:
- **saccade:** a saccade within +-100 ms, the eye moving fast at the flash (not a spike), latency from the go cue 100-500 ms, amplitude 0.5-1.5x the session median;
- **fixation:** at least 100 ms from any saccade, with no tracking gap;
- **excluded:** everything else, with the reason in `flashes.csv`.

**Suppression index** (per person): a Weibull psychometric fit per condition (2AFC, guess rate 0.5, shared slope) gives thresholds and
`1 - CS_saccade / CS_fixation = 1 - threshold_fixation / threshold_saccade`, with a bootstrap 95 % CI. 0 means no suppression, 1 means
flashes are never seen during saccades, negative means better during saccades. If the fit is not possible (fewer than 12 flashes in a condition, fewer than 2 contrasts, or accuracy
at ceiling/floor so no threshold is identified) it falls back to a clearly labelled **accuracy difference** (fixation minus saccade) with a Wilson-based CI; if
either condition has too few flashes (or both are at ceiling or fixation accuracy is near chance) the report says "cannot be computed" with the reason. Options: `--offset-ms`,
`--flash-delay-ms`, `--min-amplitude`, `--index-near-ms`, `--index-far-ms` (see `--help`).

## Glasses video

The Neon phone records the glasses' scene video itself (nothing streams to the PC): the recording starts with the saccade phase and is saved when it ends
(`glasses_video.recording` in `session.json` says `started`/`saved`/`failed (reason)`; a failure never stops the session). The PC logs the wall-clock time of each green
flash. Afterwards:
1. Export the scene video from the phone: in the Companion app, from a USB copy of the recording folder, or from Pupil Cloud (an `.mp4`).
2. Attach it:
   ```
   .\scripts\py.ps1 -m src.experiment.import_glasses_video data\<participant>\<run folder> C:\path\to\scene_video.mp4
   ```
   This copies it to `glasses/`, finds the green flashes in the video, and writes `video_sync.json` (clock offset, residual, `ok` or `check manually`) and
   `flash_video_times.csv`. A PC time `T` (unix ms, e.g. `flash_unix_ms`) is at `T / 1000 - offset_s` seconds into the video.

Not verified on real footage: whether the scene camera sees the green flash from the participant's distance, and the exported video's frame-rate/clock behaviour.

## Dashboard

```
.\scripts\py.ps1 -m src.dashboard.app
```

[http://localhost:5000](http://localhost:5000) lists past sessions (both layouts): label, filter, adjust the default contrast floor (used only when the saccade phase runs the staircase instead of fixed contrasts), delete. Old runs' baseline rows are listed too.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'src'` | Run commands from the project folder, or through `scripts/py.ps1` |
| `No module named 'flask'` / `'PySpin'` / `'mediapipe'` | The install did not take (often a non-writable `Program Files`): rerun the terminal as Administrator and reinstall. Neon-only machines do not need `PySpin`/`mediapipe` |
| `_ARRAY_API not found` or `numpy.core.multiarray failed to import` | numpy/opencv drifted from `PySpin`: reinstall `requirements.txt`, which pins working versions |
| Neon not found | Phone and PC must share a network, the Companion app must be running; set `NEON_DEVICE_ADDRESS` to the phone's IP |
| Camera picker is empty | Webcam backend only: it lists FLIR cameras via Spinnaker, not generic webcams |
| Eye tracking seems inaccurate | Redo calibration, looking steadily at each target; keep the head still after calibrating |
| Pressing SPACE seems to do nothing | Presses during a trial's wait are ignored by design; they are counted in the console summary and `ignored_space_presses` |
| Glasses recording `failed` in `session.json` | The recording may need the Companion app open with a wearer and workspace selected (not verified yet); a failure never stops the session and the rest of the data is unaffected |
| A webcam session's video is missing or empty | OpenCV could not open the codec; look for `WARNING: video recording failed to start` in the console |

## Known limitations

- **Untested on glasses:** the tutorial and captions, the new results layout, the phone
  recording and sync flashes, and every contrast level. Covered by the simulator and tests only.
- **Contrasts are provisional.** The default is 10/15/20 %. In the earlier pilot, 3 % and 6 % flashes were invisible when the grating was ~15 deg off to the side of the
  fixated symbol, and 10-20 % were seen; the foveal baseline threshold was about 1.3 %. Re-pilot and adjust in the startup dialog.
- **Screen geometry is a placeholder** (see Setup), so degrees are approximate.
- **Gaze reaches the PC ~90-100 ms late** in real sessions (about 16 ms in standalone scripts), so the live `flash_during_saccade` flag is unreliable. Judging after the fact
  from the phone's capture times (`analyze_session`) is unaffected. Cause not found.
- **Timing is open-loop:** flashes are scheduled, not triggered by the saccade, and reaction times vary (SD ~60-80 ms), so only about a third of flashes land near onset.
  People with TBI may vary more.
- **No separate baseline phase any more.** It was removed because foveal fixation is not a like-for-like control (a different retinal location). The fixation condition of the
  suppression index now comes only from the saccade phase: control flashes and flashes at least 100 ms from any saccade, so it depends on enough of them landing
  (150 trials give roughly 35 saccade-condition and 45 fixation-condition flashes in the simulator; real data may differ).
- A single participant so far; no group analysis exists.

## For developers

[CLAUDE.md](CLAUDE.md) describes the architecture, the repo map and the gotchas. All logic can be run without hardware:

```
.\scripts\py.ps1 -m pytest -q                 # about 520 tests in ~2 min, no glasses or window needed
.\scripts\py.ps1 -m src.experiment.simulation # headless simulated trial; see tests/ai-tests/README.md for run_sim.py
```

```
include/    shared types, enums, constants (no logic)
src/        eye_tracking/ (gaze sources), experiment/ (sessions, scoring, logging, analysis), dashboard/ (Flask)
tests/      pytest suite; tests/ai-tests/ holds the headless simulator tools
```

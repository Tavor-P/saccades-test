# saccades-test

A PsychoPy experiment that measures **saccadic suppression**: around the time of a rapid eye movement (a *saccade*) we become
much less sensitive to brief faint flashes. The participant looks at a dot, jumps their gaze to a second dot when it swaps,
and reports on which side of the target a briefly flashed thin line appeared. Gaze comes from Pupil Labs Neon glasses.
This is the `line-stimulus` branch; `main` uses a flashed grating (left/right/up/down orientation) instead of a line.
Developer documentation is in [CLAUDE.md](CLAUDE.md).

## Why, and where this stands

The aim is a **feasibility pilot (n = 1)** for a grant proposing to compare saccadic suppression in people with traumatic
brain injury (TBI) and without. The question for a person is: how much worse is their contrast sensitivity for a flash during a
saccade than during steady fixation? (The suppression index below.)

Honest status:
- This is a pipeline and a first participant, not a group result. No TBI/control comparison exists yet.
- The only earlier data are one participant, in test mode, with an older setup (grating, other contrasts and timing). A few flashes close to
  saccade onset were answered worse than flashes 50-300 ms away, small numbers and a window chosen after seeing the first session.
- **The current line version, its contrasts, the new results layout, the tutorial and the glasses-video steps have been run only against
  simulators, fake devices and unit tests, not yet on real glasses with a person.** See [Known limitations](#known-limitations).

## Research background

**Diamond, Ross & Morrone (2000)**, *J Neurosci* 20:3449-3455, is the design this experiment started from. The paper itself was *not
checked* for this README; what follows is only what this repository's code and comments record about it.
- Taken from it: a participant fixates a symbol, a jittered wait (800-1200 ms after a warning tone in the paper; here 500-1000 ms after
  the participant's own SPACE press) precedes the saccade target, the flash is a single display frame, contrast is varied with a ZEST
  staircase (here only if the fixed contrasts are switched off), and the same detection task is run during saccades and during fixation (here the
  fixation condition comes from flashes inside the saccade phase, see below).
- Where we differ: the flash is a line, not their grating (below); the tone is a go cue at target onset; the saccade phase uses fixed contrasts
  rather than the staircase; gaze is Neon, not a lab eye tracker.

**Braun, Schutz & Gegenfurtner (2021)**, "Age effects on saccadic suppression of luminance and color", *J Vis* 21(6):11,
doi:10.1167/jov.21.6.11, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8237129/>. Abstract and Methods were read for this README (the Results and
Discussion were not checked). In that paper participants (ages 8-78; about 100 analysed) were about 66% less sensitive to a luminance line during saccades.
- Taken from it:
  - **Stimulus:** one horizontal line across the full screen width, vertical Gaussian profile (SD 0.15 deg), shown for one refresh cycle
    2 deg above or below the saccade target.
  - **Task:** two-alternative forced choice, "above or below the target?".
  - **Fixation control:** the same task during steady fixation.
  - **Saccade/fixation gates:** a saccade trial counts only for a well-formed saccade (latency 100-500 ms, amplitude within a range, eye speed
    at the flash in a plausible band); a fixation trial only if no saccade or blink is within +-100 ms of the flash. Our gates live in
    `src/experiment/flash_gates.py`.
  - **Suppression index:** `1 - CS_saccade / CS_fixation`, with contrast sensitivity CS = 1 / threshold contrast.
- Where we differ:

| | Braun et al. 2021 | This experiment |
|---|---|---|
| Eye tracker | EyeLink 1000, 1000 Hz | Pupil Labs Neon, ~160-200 Hz, gaze reaches the PC late |
| Flash trigger | live: 15 ms after online saccade-onset detection | scheduled from the person's average reaction time (+-80 ms offsets), judged **after the fact** from the raw gaze trace |
| Display | 120 Hz, 8 ms flash | 60 Hz monitor, one frame (~17 ms) |
| Saccade size | 10 deg target step | depends on the screen width and distance: about 30-40 deg on a large screen (dot at 15 % and cross at 85 % of the screen width) |
| Thresholds | 1-up/2-down staircase, then a psychometric fit | method of constant stimuli (fixed contrasts, default 4/6/8 %), then a Weibull fit |
| Viewing | 90 cm, lab setup | chin rest, geometry entered in the startup dialog |

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
5. **Measure your screen.** Enter the height of the visible picture (cm) and the eye-to-screen distance (cm) in the startup dialog; the line's
   offset (2 deg) and thickness follow them. The saccade size in degrees depends on the screen width and distance too. Until you do, the dialog warns that the values are placeholders.

## Running a session

**Running it on Windows.** Double-click `run_experiment.bat`, `analyze_latest.bat` or `run_tests.bat` in this folder (or type their names in Command Prompt/PowerShell here).
Typing `scripts\py.ps1` into Command Prompt, the Run box or Explorer opens the .ps1 file in an editor instead of running it, and `$env:...` is PowerShell-only syntax.
The one-line command that works from any shell:
`powershell -ExecutionPolicy Bypass -File scripts\py.ps1 -m src.experiment.run_experiment`
`scripts/py.ps1` (PowerShell) / `scripts/py.sh` (Git Bash) find PsychoPy's Python and run from the project root.

```
.\scripts\py.ps1 -m src.experiment.run_experiment
```

The startup dialog asks for:

| Field | Meaning |
|---|---|
| Participant ID | required for Real; blank in Test becomes `anonymous`; the dialog shows the folder name it will be saved as |
| Mode | **Real** (full length, needs a participant ID) or **Test** (shortened, debug text shown top-left) |
| Run tutorial | first-timer on-ramp (below); defaults to Yes for a participant without a folder under `data/`, No for a known one |
| Screen height (cm) / Viewing distance (cm) | measure them; they size the 2 deg line offset and 0.15 deg thickness and are saved in `session.json` (5-100 cm / 20-300 cm, decimal comma accepted) |
| *Advanced* (folded by default): Saccade contrast min / max (%) | the saccade phase flashes fixed contrasts spaced between these (defaults 4 and 8, three levels: 4, 6, 8) |
| *Advanced*: Show gaze indicator | debug aid |

Click a field's name or its **?** for a short help bubble next to the mouse (click anywhere or press Esc to close it). The last values are remembered; an invalid entry
shows a message in the dialog and keeps everything you typed. The gaze source is always the Neon; set the environment variable `SACCADES_BACKEND=webcam` to use the
Blackfly S + MediaPipe backend instead.

What happens, in order (`Space` advances everything; `Esc` quits at once):
1. **Tutorial** (if chosen, narrated aloud and captioned; input is held while it speaks): a short plain-language intro (4 screens, SPACE advances: the dot jumps,
   a thin line flashes, it can be very faint so guess if unsure, you can rest because nothing starts until SPACE; the start screen says to keep the head still on the
   chin rest), calibration, a quiz on how to answer (a line above or below a dot, UP/DOWN, 3 correct in a row) and 8 practice trials whose saccade latencies seed the
   reaction-time average. Practice starts with a screen saying the real lines are much fainter, so guessing is normal. There is no feedback on answers.
2. **Calibration** (saccade phase): look steadily at each of 3 targets, 3 rounds. Without the tutorial, a short reaction-time test and a few practice trials follow.
3. **Green sync flashes** (saccade phase, Neon; announced aloud as "The screen will flash green a few times. This is normal."): the screen flashes green three times before the first trial and once at the end. See [Glasses video](#glasses-video).
4. **Trials.** The participant presses **SPACE** to start each trial (the wait between trials is their break). A dot and a cross sit at the left and
   right; 0.5-1 s after the press the dot swaps to the cross (a go cue with a beep) and the participant looks to it as fast as they can.
   A thin line flashes for one frame 2 deg above or below the symbol being looked at, at the average reaction time plus a random offset of -80..+80 ms (+-10 ms jitter), plus
   some control flashes at +400/+500 ms when the eye has already landed. The participant presses **Up** (line above) or **Down** (below) within 1 s;
   a missed flash is still guessed. **Every trial shows a flash**; there are no catch trials. In real trials the caption is hidden and silent.
   Saccade phase: 150 trials (30 in Test). Real trials show no caption and are silent.
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
`line_side` (above / below), `answer_side`, `flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`, `tracking_ok`. The direction of the eye movement is
the trial's parity within its phase. The code-side schema is `TRIAL_COLUMNS` in `src/experiment/logger.py`.

`session.json` sections: `participant`, `run` (mode, start time, git state, versions), `settings` (screen geometry, line parameters, contrasts, calibration, constants),
`hardware`, `results` (per-phase summaries), `gaze_quality` (rate, gaps, latency, clock offset), `ignored_space_presses` (per phase and reason) and
`glasses_video` (recording state and sync-flash times).

Older sessions (loose `data/results_<timestamp>.csv`, `_meta.json`, and `vertical`/`horizontal` sides from the grating version) are never rewritten and are still read.

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

## Data and privacy

Participants (people with traumatic brain injury and controls) produce personal data. What this software does and does not do:

- **What is stored**: `data/<participant id>/<run>/` holds `trials.csv` (answers and timing), `session.json` (settings, hardware, git commit), `gaze_trace.csv` (raw gaze),
  `analysis/` (reports and plots) and `glasses/` (the phone's scene video; it shows the room and anyone in it, so it is personal data too). `data/` is git-ignored,
  so it is never committed. `tests/fixtures/` holds only anonymous test-mode recordings (participant id `anonymous`).
- **Participant IDs**: use a study code (`P01`), never a name. The program turns the ID into a safe folder name (`sanitize_participant_id` in `src/experiment/session_files.py`); that is the only processing.
- **Network**: the dashboard listens only on this computer (`localhost:5000`, Flask's default). Do not change it to listen on the network: it lists and deletes participant data.
  The Neon phone's Real-Time API has no password, so use a private hotspot or your own router, not shared or campus Wi-Fi.
- **Not handled by this software**: consent, ethics approval, de-identification of the phone's video, encryption and backups. Recommended (not enforced):
  disk encryption (BitLocker) on the computer, a backup plan for `data/`, and following the study's ethics/IRB protocol.
- **Before pushing or sharing code**:
  1. `git status` shows nothing under `data/`.
  2. Never copy real session folders into `tests/fixtures/`.
  3. Delete the phone recording from the phone after export if the protocol requires it.

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

- **Untested on glasses:** the line's rendering at 60 Hz (profile, brightness, one-frame visibility), the tutorial and captions, the new results layout, the phone
  recording and sync flashes, and every contrast level. Covered by the simulator and tests only.
- **Contrasts are a guess.** 4/6/8 % was set after a test run at 10/15/20 % hit ceiling and an earlier 2/4/6 % guess was judged too hard. On an 8-bit
  display those are only ~5-10 grey steps, so levels are coarse. Re-pilot and adjust in the startup dialog.
- **Screen geometry is only as good as what you type into the startup dialog** (see Setup); the placeholders (26 cm at 50 cm) make degrees approximate. The geometry used is saved in `session.json` and printed in the analysis report.
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
.\scripts\py.ps1 -m pytest -q                 # about 540 tests in ~2 min, no glasses or window needed
.\scripts\py.ps1 -m src.experiment.simulation # headless simulated trial; see tests/ai-tests/README.md for run_sim.py
```

```
include/    shared types, enums, constants (no logic)
src/        eye_tracking/ (gaze sources), experiment/ (sessions, scoring, logging, analysis), dashboard/ (Flask)
tests/      pytest suite; tests/ai-tests/ holds the headless simulator tools
```

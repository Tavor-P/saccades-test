# saccades-test

A PsychoPy experiment that measures **saccadic suppression**: around the time of a rapid eye movement (a *saccade*) we become much less sensitive
to brief, faint flashes. The participant looks at a dot, moves their gaze to a second dot when it swaps, and says on which side of that target a
briefly flashed thin line appeared (above or below). Gaze comes from Pupil Labs Neon glasses. This is `main`, which flashes a line; the `grating`
branch flashes a grating (left/right/up/down orientation) instead. Developer documentation is in [CLAUDE.md](CLAUDE.md).

**Where this stands.** The aim is a *feasibility pilot (n = 1)* for a grant proposing to compare saccadic suppression in people with traumatic brain
injury (TBI) and without. This is a pipeline and a first participant, not a group result; no TBI/control comparison exists yet. The current line
version, its contrasts, the results layout, the tutorial and the glasses-video steps have been run **only against simulators, fake devices and unit
tests, not yet on real glasses with a person** (see [Known limitations](#known-limitations)).

**Which part do you need?**
- Setting up a PC for the first time: [First-time setup](#first-time-setup).
- Running a participant: [Run day](#run-day-experimenter-quick-start) (printable: [docs/experimenter_checklist.md](docs/experimenter_checklist.md)).
- Reading the results: [Output](#output), [Analysing a session](#analysing-a-session).
- Why it is built this way: [Research background](#research-background).

## Words used in this document

| Term | Meaning |
|---|---|
| Saccade | A fast eye movement from one point to another (here about 30-40 deg, roughly 75-110 ms). |
| Flash | The thin line, shown for one screen frame (about 17 ms at 60 Hz). |
| Contrast | How faint the line is, as a percentage (4-8 % is very faint). |
| Fixation condition | Flashes that arrive while the eye is still, used as the "normal sensitivity" comparison. |
| Saccade condition | Flashes that arrive while the eye is moving. |
| Gates | Fixed rules, written down before looking at data, that decide which flashes count as saccade, fixation, or excluded (`src/experiment/flash_gates.py`). |
| Suppression index | `1 - threshold(fixation) / threshold(saccade)`: 0 means no suppression, 1 means flashes are never seen during saccades. |
| Threshold | The contrast at which the participant starts to see the line reliably, from a curve fitted to the answers (pure guessing gives 50 % correct). |
| Open-loop timing | Flashes are scheduled from the person's average reaction time rather than triggered when the eye starts to move, because gaze reaches the PC late. Whether a flash really hit the saccade is judged afterwards. |
| Tutorial | An optional narrated first-timer on-ramp: instructions, calibration, a how-to-answer quiz, 8 practice trials. |

## First-time setup

Do this once per computer.

1. **Hardware:** a Windows PC, a monitor (60 Hz here), a chin rest or other way to keep the head still (the Neon scene camera moves with the head, so
   calibration assumes a still head), Neon glasses, and the Neon Companion phone on the same Wi-Fi/hotspot as the PC.
2. Install [PsychoPy Standalone](https://www.psychopy.org/download.html) (it bundles its own Python). Always run Python through the wrapper
   `.\scripts\py.ps1` (PowerShell) or `scripts/py.sh` (Git Bash); it finds PsychoPy's Python and runs from the project root.
3. Install dependencies into that Python:
   ```
   .\scripts\py.ps1 -m pip install -r requirements.txt
   ```
   The FLIR Blackfly S + MediaPipe backend additionally needs the Spinnaker SDK (`PySpin`); Neon-only machines do not.
4. Start the Neon Companion app on the phone. The experiment looks for the Neon at `NEON_DEVICE_ADDRESS` in
   `include/eye_tracking/constants.py` (pinned to a phone IP at the time of writing; set it to your phone's address, or to `None` to auto-discover).
5. **Measure your screen** (once per monitor and seat position): the height of the visible picture (cm) and the eye-to-screen distance (cm). You type
   them into the startup dialog on every run; the line's offset (2 deg) and thickness follow them, and so does the saccade size in degrees. Until you do,
   the dialog warns that the values are placeholders.

On Windows you can double-click `run_experiment.bat`, `analyze_latest.bat` or `run_tests.bat` in the project folder. Typing `scripts\py.ps1` into
Command Prompt, the Run box or Explorer opens the .ps1 file in an editor instead of running it. The one-line command that works from any shell:
`powershell -ExecutionPolicy Bypass -File scripts\py.ps1 -m src.experiment.run_experiment`

## Run day (experimenter quick start)

1. Glasses on the participant (clean lenses), Neon Companion app open on the phone (same Wi-Fi as the PC, a wearer and workspace selected), head on the chin rest.
2. Double-click `run_experiment.bat`.
3. In the startup dialog: enter the **screen height** and **viewing distance**, choose Mode **Real**, type a **Participant ID** (a study code such as `P01`, never a name),
   and set **Run tutorial** to **Yes** for a participant who has not done it before (the dialog suggests this by itself for an ID with no folder under `data/`). Click OK.
4. Read the participant script below, then follow the narrated tutorial (SPACE advances; Esc quits and keeps the data).
5. At the end the console and `RUN_SUMMARY.txt` in the run folder show what happened and the next commands; SPACE closes the window.
6. Export the scene video from the phone (see [Glasses video](#glasses-video)), then attach it:
   `.\scripts\py.ps1 -m src.experiment.import_glasses_video data\<participant>\<run folder> <video.mp4>`
7. Analyse: double-click `analyze_latest.bat` (newest run).

> **Participant script.** "Look at the dot and press SPACE when you are ready. After a moment, with a beep, it jumps to the other side of the screen: look at it as fast as you can.
> Around then a very faint thin line flashes for an instant above or below it. Press UP if it was above, DOWN if below, and guess if you are not sure.
> You can rest between trials: nothing happens until you press SPACE."

### The startup dialog

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

### What happens in a run

`Space` advances everything; `Esc` quits at once.
1. **Tutorial** (if chosen; narrated aloud and captioned; input is held while it speaks): a short plain-language intro (the dot jumps, a thin line flashes, it can be very
   faint so guess if unsure, you can rest because nothing starts until SPACE, keep the head still), calibration, a quiz on how to answer (a line above or below a dot,
   UP/DOWN, 3 correct in a row) and 8 practice trials whose saccade latencies seed the reaction-time average. Practice starts with a screen saying the real lines are
   much fainter, so guessing is normal. There is no feedback on answers.
2. **Calibration**: look steadily at each of 3 targets, 3 rounds. Without the tutorial, a short reaction-time test and a few practice trials follow.
3. **Green sync flashes**: the screen flashes green three times before the first trial and once at the end (announced aloud; this is normal). They let the phone's video be
   aligned to the PC clock, see [Glasses video](#glasses-video).
4. **Trials** (150 in Real, 30 in Test). The participant presses **SPACE** to start each trial (the wait between trials is their break). A dot and a cross sit at the left and
   right; 0.5-1 s after the press the dot swaps to the cross (a go cue with a beep) and the participant looks to it as fast as they can.
   A thin line flashes for one frame 2 deg above or below the symbol being looked at, at the average reaction time plus a random offset of -80..+80 ms (+-10 ms jitter), plus
   some control flashes at +400/+500 ms when the eye has already landed. The participant presses **Up** (line above) or **Down** (below) within 1 s;
   a missed flash is still guessed. Every trial shows a flash; there are no catch trials. In real trials the caption is hidden and silent.
5. **End.** The console prints a plain-language summary and the `analyze_session` command to run. Fixation data come from the control flashes and the
   far-from-saccade flashes of the same trials, sorted after the fact (see [Analysing a session](#analysing-a-session)).

The phone records the glasses' scene video during the trials; a failure there never stops the session.

### Head movement

The Neon's inertial sensor (IMU) is polled at ~30 Hz on a light background thread while the trials run. It **flags, it does not correct**: the chin rest
remains the plan. It records head angular speed and the head's rotation away from its orientation at calibration, and:

- a flash with a head-speed spike (> `HEAD_MAX_SPEED_DPS`) within `TRACKING_LOSS_WINDOW_MS` of it gets `tracking_ok = no`, so it is excluded like a tracking loss;
- if the head turns more than `HEAD_DRIFT_WARN_DEG` from the calibration orientation, "Please keep your head still." is printed and spoken (between trials only, at most
  once every `HEAD_WARN_MIN_INTERVAL_S`; speech only, because trials show no caption);
- Test mode's top-left text shows the drift in degrees;
- `session.json` gets a `head_motion` section (status, samples, rate, max speed and drift, flagged trials) and the run folder a downsampled `head_motion.csv`
  (`t_s`, `phone_unix_s`, `speed_dps`, `drift_deg`).

If the device has no IMU or it fails, the monitor switches itself off with one console warning and `head_motion.status` reads `unavailable (...)`. The thresholds and the
IMU's rate and clock alignment with the gaze stamps are **not validated on glasses** yet.

## Output

Everything is under `data/` (git-ignored): one folder per participant, one subfolder per run (one start of the program).

```
data/<participant id>/<YYYY-MM-DD_HH-MM-SS>[_test]/
  trials.csv           one row per trial
  session.json         metadata, see below
  gaze_trace.csv       raw Neon gaze with the phone's capture times
  head_motion.csv      Neon IMU head speed and drift, ~30 Hz (only if the IMU worked)
  RUN_SUMMARY.txt      plain-language end-of-run summary
  analysis/            report.txt, plot.png, flashes.csv, suppression.json (written by analyze_session)
  glasses/             the phone's video + video_sync.json + flash_video_times.csv (written by import_glasses_video)
```

`trials.csv` columns (booleans are `yes`/`no`; practice trials are not written): `phase` (saccade / rt_test; `baseline` only in old runs), `trial`, `contrast_pct`,
`flash_shown`, `flash_during_saccade` (the live, unreliable judgement), `answered`, `correct`, `outcome` (correct / wrong / no_answer / detected),
`line_side` (above / below), `answer_side`, `flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`, `tracking_ok`. The direction of the eye movement is
the trial's parity within its phase. The code-side schema is `TRIAL_COLUMNS` in `src/experiment/logger.py`.

`session.json` sections: `participant`, `run` (mode, start time, git state, versions), `settings` (screen geometry, line parameters, contrasts, calibration, constants),
`hardware`, `results` (per-phase summaries), `gaze_quality` (rate, gaps, latency, clock offset), `head_motion` (IMU monitor summary), `ignored_space_presses` (per phase and reason) and
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

**Suppression index** (per person): a Weibull psychometric fit per condition (two-alternative forced choice, guess rate 0.5, shared slope) gives thresholds and
`1 - CS_saccade / CS_fixation = 1 - threshold_fixation / threshold_saccade` (CS = contrast sensitivity = 1 / threshold), with a bootstrap 95 % CI. 0 means no suppression, 1 means
flashes are never seen during saccades, negative means better during saccades. If the fit is not possible (fewer than 12 flashes in a condition, fewer than 2 contrasts, or accuracy
at ceiling/floor so no threshold is identified) it falls back to a clearly labelled **accuracy difference** (fixation minus saccade) with a Wilson-based CI; if
either condition has too few flashes (or both are at ceiling or fixation accuracy is near chance) the report says "cannot be computed" with the reason. Options: `--offset-ms`,
`--flash-delay-ms`, `--min-amplitude`, `--index-near-ms`, `--index-far-ms` (see `--help`).

## Glasses video

The Neon phone records the glasses' scene video itself (nothing streams to the PC): the recording starts with the trials and is saved when they end
(`glasses_video.recording` in `session.json` says `started`/`saved`/`failed (reason)`). The recording may need the Companion app open with a wearer and workspace selected
(not verified yet). The PC logs the wall-clock time of each green flash. Afterwards:
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
| Glasses recording `failed` in `session.json` | The recording may need the Companion app open with a wearer and workspace selected (not verified yet); the rest of the data is unaffected |
| A webcam session's video is missing or empty | OpenCV could not open the codec; look for `WARNING: video recording failed to start` in the console |

## Research background

**Diamond, Ross & Morrone (2000)**, *J Neurosci* 20:3449-3455, is the design this experiment started from. The paper itself was *not
checked* for this README; what follows is only what this repository's code and comments record about it.
- Taken from it: a participant fixates a symbol, a jittered wait (800-1200 ms after a warning tone in the paper; here 500-1000 ms after
  the participant's own SPACE press) precedes the saccade target, the flash is a single display frame, contrast is varied with a ZEST
  staircase (here only if the fixed contrasts are switched off), and the same detection task is run during saccades and during fixation (here the
  fixation condition comes from flashes inside the saccade phase).
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

Earlier data: one participant, in test mode, with an older setup (grating, other contrasts and timing). A few flashes close to saccade onset were answered worse than
flashes 50-300 ms away; small numbers and a window chosen after seeing the first session, which is why the gates are now pre-specified.

## Known limitations

- **Untested on glasses:** the line's rendering at 60 Hz (profile, brightness, one-frame visibility), the tutorial and captions, the results layout, the phone
  recording and sync flashes, and every contrast level. Covered by the simulator and tests only.
- **Contrasts are a guess.** 4/6/8 % was set after a test run at 10/15/20 % hit ceiling and an earlier 2/4/6 % guess was judged too hard. On an 8-bit
  display those are only ~5-10 grey steps, so levels are coarse. Re-pilot and adjust in the startup dialog.
- **Screen geometry is only as good as what you type into the startup dialog** (see [First-time setup](#first-time-setup)); the placeholders (26 cm at 50 cm) make degrees approximate. The geometry used is saved in `session.json` and printed in the analysis report.
- **Gaze reaches the PC ~90-100 ms late** in real sessions (about 16 ms in standalone scripts), so the live `flash_during_saccade` flag is unreliable. Judging after the fact
  from the phone's capture times (`analyze_session`) is unaffected. Cause not found.
- **Head monitor unvalidated:** the IMU monitor's thresholds, real rate and clock alignment with gaze have not been checked on glasses (see Head movement).
- **Timing is open-loop:** flashes are scheduled, not triggered by the saccade, and reaction times vary (SD ~60-80 ms), so only about a third of flashes land near onset.
  People with TBI may vary more.
- **No separate baseline phase any more.** It was removed because foveal fixation is not a like-for-like control (a different retinal location). The fixation condition of the
  suppression index now comes only from the saccade phase: control flashes and flashes at least 100 ms from any saccade, so it depends on enough of them landing
  (150 trials give roughly 35 saccade-condition and 45 fixation-condition flashes in the simulator; real data may differ).
- A single participant so far; no group analysis exists.

## For developers

[CLAUDE.md](CLAUDE.md) describes the architecture, the repo map and the gotchas. All logic can be run without hardware:

```
.\scripts\py.ps1 -m pytest -q                 # about 580 tests in ~2 min, no glasses or window needed
.\scripts\py.ps1 -m src.experiment.simulation # headless simulated trial; see tests/ai-tests/README.md for run_sim.py
```

```
include/    shared types, enums, constants (no logic)
src/        eye_tracking/ (gaze sources), experiment/ (sessions, scoring, logging, analysis), dashboard/ (Flask)
tests/      pytest suite; tests/ai-tests/ holds the headless simulator tools
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000):
detect a briefly-flashed grating's orientation, once during a saccade and
once during fixation (baseline), to measure saccadic suppression. Gaze is
tracked via Pupil Labs Neon glasses (default, via the Real-Time API) or, as an alternative backend
picked in the startup dialog, a FLIR Blackfly S camera (Spinnaker SDK /
`PySpin`) + MediaPipe iris landmarks. See [README.md](README.md) for the full participant-facing
walkthrough (controls, output files, troubleshooting).

## Line-stimulus branch (`line-stimulus`)

An ALTERNATIVE to the grating on `main` (the grating behaviour there is untouched). The flash is a thin horizontal line as in
Braun, Schutz & Gegenfurtner 2021 (J Vis): full screen width, vertical Gaussian profile (`LINE_SD_DEG` 0.15), centred
`LINE_OFFSET_DEG` (2 deg) above or below the saccade TARGET (the symbol the eye is moving to; in the baseline the screen
centre), drawn `GRATING_DURATION_FRAMES` frame(s); the participant presses UP (above) or DOWN (below), LEFT/RIGHT are unused.
Everything else (scheduled flashes, gates, analysis, SPACE gating, rolling reaction time) is as on `main`.
- `Orientation` is replaced by `Side.ABOVE/BELOW` (`include/experiment/types.py`). The trial/result fields and the CSV
  columns are still called `orientation` / `response_orientation` (now holding 'above'/'below'); old CSVs with
  'vertical'/'horizontal' load via `data_access._LEGACY_SIDES` (vertical->above, horizontal->below, an arbitrary but
  consistent relabel). The render-state key is still `"grating"` (now the line: x/y = line position, `orientation` = side).
- Geometry: `LINE_*` constants (`include/experiment/constants.py`) via `SCREEN_HEIGHT_DEG`, so they are only right once
  `VIEWING_DISTANCE_CM`/`SCREEN_HEIGHT_CM` are the real monitor's (still placeholders). `trial_mechanics.line_position`
  places the line. Rendering (`run_experiment.build_stimuli`): a `GratingStim` with `tex=None` and the `'gauss'` mask,
  sized (`LINE_WIDTH_SCREENS` x screen width, 6 SD tall); `contrast` scales the white field, so a light line.
- Tutorial quiz = a line above/below a dot (UP/DOWN, 3 in a row); trial captions say "UP if the line was above ...".
- **Contrasts are a GUESS**: defaults 4/6/8 % (`SACCADE_FIXED_CONTRASTS`, startup dialog min 4 / max 8, 3 levels). A Test-mode
  run at 10/15/20 % was at ceiling (11/11 correct at fixation, 18/22 overall), the old foveal grating threshold was ~1.3 %, and
  the previous 2/4/6 % guess was judged too hard. Educated guess, not a measurement: re-pilot on glasses; adjust in the startup
  dialog. 4-8 % is ~5-10 grey steps at the line's peak on an 8-bit display (above `HARDWARE_CONTRAST_FLOOR`, 0.78 %), so levels are coarse.
- **Rolling reaction time** here is `robust_average_ms` over the last `RT_AVERAGE_ROLLING_WINDOW` (12) latencies; fewer than
  `RT_MIN_SAMPLES_FOR_AVERAGE` (3) plausible samples keep the previous value (initially `DEFAULT_REACTION_TIME_MS`). The phase
  summary in `_meta.json` adds `rt_samples_total/_in_window/_dropped_in_window`.
- **Untested on glasses / a real window**: the PsychoPy line rendering (profile, width, brightness, one-frame visibility,
  how it looks at 60 Hz), the quiz/captions, and every contrast level. Covered by the simulator and tests only.

## Status and next steps (updated 2026-10-25)

**Goal.** A grant to test saccadic suppression in people with TBI vs without. Present what exists as a
*feasibility pilot (n=1)*, not as a group result. The experiment flashes a brief thin line around the participant's
saccades (Neon glasses, 2AFC orientation) and measures detection vs flash time relative to saccade onset.

**Shipped setup.** Flashes (a line above/below the target, UP/DOWN) at average RT + offset (-80..+80 ms) + random +/-10 ms, fixed
saccade-phase contrasts 4/6/8 % (line branch; a guess; startup dialog "Saccade contrast min/max (%)"), sharp go-cue, dot/cross at 15 %/85 % of
the screen, plain Neon backend with a still head (chin rest). Onboarding is calibration -> how-to-answer quiz -> 8 practice
trials that seed the reaction-time average. Every trial waits for SPACE. Control flashes (+400/+500 ms after the average
RT) and pre-specified post-hoc gates (`flash_gates.py`) let `analyze_session` report a per-person suppression index. Run the
experiment, then `.\scripts\py.ps1 -m src.experiment.analyze_session` (newest session) for the report, plot and per-flash CSV.
None of this has been run in a real session on glasses since the last refactor: it is covered by the simulator and tests only.
**Results layout and glasses video (new, untested on glasses).** A run writes `data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/` (trials.csv with plain
column names, session.json, gaze_trace.csv, analysis/, glasses/); old loose `results_<ts>.*` files are still read. The phone records the glasses' scene
video itself (`NeonGazeSource.start_recording/stop_recording`, Real-Time API `recording_start`/`recording_stop_and_save`); a green sync-flash step before
the first trial and one at the end lets `import_glasses_video` line the exported video up with the PC clock. The recording, the flashes and the
detector on real footage have only been run against fake devices and synthetic frames.

**Pilot findings (one participant, test mode, earlier setup).** Foveal baseline threshold ~1.3 %. Saccades ~75 ms / ~30 deg
with the 15/85 layout. 3 %/6 % flashes were invisible when fixating the dot/cross (flash ~15 deg off to the side); 10-20 %
are seen. Flashes within +-50 ms of saccade onset: 3/12 correct vs 12/13 for flashes 50-300 ms away (Fisher p=0.001; the
earlier 20-80 % session pointed the same way, pooled p<1e-4), sharpest in the 50 ms *before* onset. Small n, and the +-50 ms
window was chosen after seeing session 1 (the gates are now pre-specified).

**Known problems.**
1. Gaze reaches the PC ~95 ms late at ~160 Hz in real sessions (see the open problem in the Gotchas). Irrelevant to the
   post-hoc analysis (phone capture times) but it makes the live `flash_during_saccade` flag useless.
2. Reaction time varies (SD ~60-80 ms), so only ~1/3 of flashes land near onset; TBI will likely be worse. The live
   rolling average (`RollingReactionTime`) only helps when the RT drifts (simulation only).
3. The screen geometry constants (`VIEWING_DISTANCE_CM`, `SCREEN_HEIGHT_CM`) are still the 24-inch defaults; the user's monitor
   is small (~12 in wide) and 60 Hz. Ask for real width/height (cm) and viewing distance.
4. The baseline (foveal) phase is not a like-for-like control for the saccade phase (different retinal location); the
   fixation-control flashes inside the saccade phase address this.

**Next steps, in priority order.** (1) Check the current behaviour on real glasses (a short Test-mode run, then a full
150-trial Real run with a participant ID, analysed with `analyze_session`). (2) Group-level analysis (per-person suppression
index + group test, or a mixed model) for TBI vs controls. (3) A one-page grant figure + wording. (4) Clinical metadata
fields (group, time since injury, severity, symptoms, medication; needs the IRB) in the dashboard. (5) Find the cause of the
~95 ms gaze delay (e.g. receive Neon in its own process). (6) Ask the user for the screen measurements and update the geometry constants.

## Start here (read this first)

- **Run everything through the wrapper**: `.\scripts\py.ps1 <python args>` (PowerShell) or
  `scripts/py.sh <python args>` (Git Bash). It finds PsychoPy's bundled Python and runs from the
  project root. Plain `python` will not work (missing deps, `src`/`include` not importable).
- **Never launch the experiment or dashboard yourself** (`run_experiment` opens a fullscreen
  PsychoPy window on the user's screen). To exercise experiment logic use the **headless simulator**
  (below) or the tests. Real-glasses runs are for the human.
- Don't merge, push or switch branches unless asked.
- **Other AI sessions may be editing this same folder at the same time.** See "Working rules".
- Before changing anything, find the file in the **Repo map**, and skim the **Gotchas** at the end.

## Commands

All via the wrapper (PowerShell shown; from Git Bash use `scripts/py.sh` with the same arguments):

```
.\scripts\py.ps1 -m pip install -r requirements.txt            # install deps (into PsychoPy's Python)
.\scripts\py.ps1 -m pytest -q                                   # full test suite (~55 s, no camera/window needed)
.\scripts\py.ps1 -m pytest tests/ai-tests/test_session.py::test_name -v  # one test
.\scripts\py.ps1 -m src.experiment.simulation                   # HEADLESS simulated trial, event timeline (safe for AI)
.\scripts\py.ps1 tests/ai-tests/run_sim.py --help               # headless simulation runner (scenarios, sweeps, JSON)
.\scripts\py.ps1 -m src.experiment.run_experiment               # REAL experiment (opens a window - human only)
.\scripts\py.ps1 -m src.dashboard.app                           # session dashboard at localhost:5000
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir D]  # accuracy vs flash time relative to saccade onset (newest session by default; a unix ts, run folder name or path; writes <run>/analysis/)
.\scripts\py.ps1 -m src.experiment.import_glasses_video <run-folder> <exported.mp4>  # attach + sync the phone's glasses video
.\scripts\py.ps1 -m src.eye_tracking.neon_fixation_test [secs]  # Neon jitter check (glasses on; human)
.\scripts\py.ps1 -m src.eye_tracking.latency_test               # Neon vs webcam latency recording (human)
.\scripts\py.ps1 -m src.eye_tracking.record_gaze_trace out.csv 10   # record raw Neon gaze for replay (glasses on)
```

The old absolute path in some notes (`C:\Program Files\PsychoPy`) is wrong on this machine; the real
one is `%LOCALAPPDATA%\Programs\PsychoPy\python.exe` (the wrapper also honours `$env:PSYCHOPY_PYTHON`).
Without the Spinnaker SDK (`PySpin`) or `mediapipe`, `tests/ai-tests/test_camera.py` and
`test_gaze_tracking.py` skip themselves (`pytest.importorskip`) instead of failing. The Neon is at
`NEON_DEVICE_ADDRESS` (`include/eye_tracking/constants.py`; currently `192.168.0.60:8080`) and is
sometimes reachable from here for read-only checks - ask the human before relying on it.

## Repo map

Paths in this section are checked by `tests/test_docs.py` - keep them real.

| Path | What it is |
|---|---|
| `include/experiment/constants.py` | every tunable experiment constant (timing, stimulus, ZEST, trial counts) |
| `include/experiment/types.py` | `TrialResult` (one CSV row), `TrialSpec`, `Target`, `Side` (line above/below; was `Orientation`) |
| `include/eye_tracking/constants.py` | gaze constants incl. `GAZE_DEAD_ZONE`, all `NEON_*` settings |
| `include/eye_tracking/interfaces.py` | the `GazeSource` interface the experiment talks to |
| `include/eye_tracking/types.py` | `GazeZone`, `GazeSample` |
| `src/experiment/session.py` | `ExperimentSession`: the saccade phase state machine (the core logic) |
| `src/experiment/presaccade_session.py` | `PresaccadeSession`: baseline (fixation) phase |
| `src/experiment/tutorial_session.py` | `TutorialSession`: narrated first-timer on-ramp (calibration -> how-to-answer quiz -> 8 practice trials that yield the first reaction-time samples) |
| `src/experiment/run_experiment.py` | entry point + PsychoPy rendering; the only PsychoPy-dependent module |
| `src/experiment/pausable_clock.py` | `PausableClock`: the injected time source every session reads (name is historical; it no longer pauses) |
| `src/experiment/settings.py` | dashboard-configured default contrast floor (`data/settings.json`) |
| `src/experiment/scoring.py` | `score_outcome`, `is_valid_for_saccadic_analysis` (the validity gate) |
| `src/experiment/trial_mechanics.py` | `OnsetDetector` (displacement rule), `ShuffledBag`, `zone_for`, `line_position` (line placement above/below a reference point), `robust_average_ms` (shared helper; drops anticipations/outliers then takes the median), `sample_capture_time` (gaze-sample capture time on the session clock, shared by both sessions), `RollingReactionTime` (live rolling window + robust average, updated at every detected onset) |
| `src/experiment/calibration.py` | `CalibrationRounds` (the 3-point x 3-round calibration state machine shared by `ExperimentSession` and `TutorialSession`) and `average_calibration_rounds` (rounds 2 and 3) |
| `src/experiment/space_input.py` | SPACE bookkeeping with no PsychoPy: `CalibrationLatch` (a calibration press made before the gaze window is full is latched and completed in `tick()`), `NarrationKeyGate` (tutorial: a press during narration is delivered after it), `SpaceTally` (ignored presses per phase + reason), `deliver_space`, `leaves_complete_screen` |
| `src/experiment/session_schedule.py` | `phase_order`: baseline/saccade order counterbalanced by participant ID |
| `src/timebase.py` | the ONE time source: `now()` = `time.perf_counter()`; wall-clock via an edge-synchronised anchor (`unix_now()`, `unix_ms()`) |
| `src/experiment/trial_factory.py` | builds trial sequences; every trial flashes (catch trials / false alarms were removed - old sessions may still contain them, their `false_alarm`/`correct_rejection` outcome words stay readable) |
| `src/experiment/zest.py` | adaptive contrast staircase |
| `src/experiment/logger.py` | `ResultLogger` writes one run folder: `trials.csv` (schema = `TRIAL_COLUMNS`, plain names; `DROPPED_FROM_CSV` lists the `TrialResult` fields left out and why, e.g. `source`/`target`: direction = trial parity, `timing_offset_ms`: the delay is in `flash_ms_after_go_cue`; column order = contrast, yes/no group, outcome, stimulus/answer pair, ms block (`flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`), `tracking_ok`), `session.json` (sections participant/run/settings/hardware/results/gaze_quality/ignored_space_presses/glasses_video), side-file paths; `read_trial_rows`/`read_trials` read BOTH the new and the old layout |
| `src/experiment/session_files.py` | run-folder naming (`sanitize_participant_id`, `create_session_folder`), `SessionRef`, `list_sessions`/`find_session`/`newest_session` over the new and the old loose-file layout (a session's key is its unix start time) |
| `src/experiment/glasses_video.py` | no PsychoPy: sync-flash frame counts, `find_sync_flashes` (green frames in a video), `compute_offset` (video time vs logged PC flash times, residual + ok/check manually) |
| `src/experiment/import_glasses_video.py` | CLI: copies the phone's exported video into `<run>/glasses/`, writes `video_sync.json` + `flash_video_times.csv` |
| `src/experiment/results_graph.py` | end-of-run comparison figure |
| `src/experiment/analyze_session.py` | post-hoc saccade analysis: finds saccades in a session's raw gaze trace (phone capture times), places each flash against the nearest onset, writes accuracy-vs-time report + plot + per-flash CSV |
| `src/experiment/flash_gates.py` | pre-specified post-hoc gates (Braun et al. 2021): each flash is `saccade` (eye fast at the flash, latency 100-500 ms, amplitude within 0.5-1.5x median), `fixation` (>=100 ms from any saccade, no tracking gap) or `excluded`; control flashes come from `CONTROL_OFFSETS_MS` (+400/+500 ms after the average RT, eye already landed) |
| `src/experiment/simulation.py` | headless simulator (simulated participant, clock, gaze) |
| `src/eye_tracking/neon_source.py` | `NeonGazeSource` (default backend): Real-Time API stream + quality stats |
| `src/eye_tracking/gaze_mapping.py` | `GazeMapper`: ratio -> calibrated position / zone / smoothing |
| `src/eye_tracking/gaze_factory.py` | `build_gaze_source()` backend picker (lazy imports) |
| `src/eye_tracking/gaze_trace.py` | raw gaze recording + `ReplayGazeSource` |
| `src/eye_tracking/webcam_source.py` | Blackfly S + MediaPipe backend (alternative) |
| `src/dashboard/app.py` | Flask dashboard (separate process, reads `data/`) |
| `tests/conftest.py` | `FakeGazeSource`, `fake_time` fixture |
| `tests/helpers.py` | `NullLogger`, shared by the session test files |
| `tests/ai-tests/test_line_stimulus.py` | line-stimulus branch: placement above/below the target (both directions) and the baseline centre, UP/DOWN keys only, quiz, old-CSV loading |
| `tests/ai-tests/README.md` | headless-simulation guide for AI agents |
| `tests/ai-tests/run_sim.py` | simulation CLI (`--set`, `--sweep`, `--timeline`, `--json`) |
| `tests/ai-tests/find_bugs.py` | randomized bug-hunting campaign over the simulator; prints a repro command per finding |
| `tests/ai-tests/invariants.py` | `check_session()`: contract checks every simulated session must satisfy |
| `tests/fixtures/neon_trace_sample.csv` | a real recorded Neon gaze trace |
| `tests/ai-tests/analyze_onset_threshold.py` | offline onset-rule study on recorded traces (lead/delay/false triggers per threshold) |
| `scripts/py.ps1` | interpreter wrapper (see Start here) |

`data/` (per-participant run folders, old loose results, traces, graphs) is git-ignored runtime output: never commit it, and don't write to
it from tests or simulations (the simulator uses an in-memory logger).

## How to verify a change (cheapest check first)

1. **Logic in a session/scoring/ZEST/logger** - add or update a test (use `FakeGazeSource` + `fake_time`
   from `tests/conftest.py`, or drive it with the simulator) and run `pytest`. Old tests that now fail
   mean you changed behaviour - decide if that is intended before editing the test.
2. **Whole-session behaviour or a timing/threshold constant** - `tests/ai-tests/run_sim.py`:
   `--set NAME=VALUE`, `--sweep NAME=a,b,c --seeds 5`, `--timeline`. Compare *relative* numbers; the
   simulated participant is a simple model (see `tests/ai-tests/README.md`).
3. **Gaze-source behaviour** - `tests/test_neon_source.py` / `tests/test_gaze_trace.py` patterns (a fake
   device, or replay `tests/fixtures/neon_trace_sample.csv` through `ReplayGazeSource`).
4. **Anything needing real eyes/glasses/screen** (latency, jitter, calibration quality) - write the
   measurement script and ask the human to run it; don't guess numbers.

Report test results truthfully (what ran, what was skipped). The suite passes at the time of writing
(~510 tests, ~60 s, 2 skipped when PySpin/mediapipe are absent).

**Bug-hunting tests** (`tests/ai-tests/`): `find_bugs.py` fuzzes the real session with random/hostile
participants and checks `invariants.py` (run `find_bugs.py --sessions 300` after changing session,
scoring or ZEST logic; a finding prints a `run_sim.py` repro line). `test_bug_injection.py` re-injects
known bugs to prove the checks and regression tests still catch them. Also enforced by tests:
TrialResult/CSV/dashboard schema sync (`test_schema_sync.py`), constant relations
(`test_constants_sanity.py`), live-vs-graph validity lockstep (`test_validity_lockstep.py`) and the
architecture rules in this file (`test_architecture_rules.py`: no PsychoPy in session logic, no direct
`time.monotonic()`/`time.time()` anywhere in `src/` outside `timebase.py`, lazy backend imports, no logic in `include/`).

## Recipes

- **Add a column to trials.csv / a field to `TrialResult`**: field in `include/experiment/types.py`
  (default `None`) -> ONE `Column(...)` in `TRIAL_COLUMNS` in `src/experiment/logger.py` (plain name, writer, and the
  converter back to the old CSV's text) or, to keep it out of the CSV, an entry with a reason in `DROPPED_FROM_CSV`
  (`test_schema_sync.py` fails until every field is one or the other) -> `trial_from_row` in the same file -> set it where the
  `TrialResult` is built in `session.py` -> test (see `tests/ai-tests/test_tracking_quality.py`). Old CSVs must still load.
- **Change a tuning constant**: edit it in `include/**/constants.py` (read its comment first - several
  encode hard-won trade-offs), then run a `--sweep` in the simulator and `pytest`.
- **Add a gaze backend**: implement `GazeSource` (`include/eye_tracking/interfaces.py`), register it in
  `src/eye_tracking/gaze_factory.py` (lazy import!) and the startup dialog in `run_experiment.py`; reuse
  `GazeMapper` rather than re-implementing calibration/zones.
- **Reproduce a bug without hardware**: record gaze with `record_gaze_trace` (or use the fixture) and
  replay it; or reproduce it in the simulator with a `ParticipantProfile`.

## Working rules (several AI sessions share this folder)

- Run `git status` / `git log -5` first: another session may have committed or have uncommitted edits.
  **Never revert or "clean up" changes you didn't make.** A file that changed on disk under you is
  someone else's deliberate edit - build on it.
- **Stage and commit only your own files by name** (`git add path1 path2`). Do **not** use `git add -A`
  or `git commit -a`: they sweep up other sessions' work into your commit.
- Keep commits focused, message says what and why; end with the co-author line the harness gives you.
- Don't change things outside your task (e.g. another session's files in `src/eye_tracking/neon_*`) -
  mention it to the user instead.
- Windows/PowerShell environment: `&&` is unavailable in PowerShell 5.1; the Bash tool is Git Bash.
  Source files may be CRLF in the working tree; edit tolerantly and keep each file's line endings.
- `.gitignore` contains `data/`, which also matches a `data` folder anywhere (e.g. under tests) - put test fixtures in `tests/fixtures/`.
- **Whenever the user asks you to push and/or merge, update CLAUDE.md first** (the user's standing
  instruction). Review what the branch changed since CLAUDE.md was last touched (`git log` / `git diff`)
  and bring it up to date *before* the push/merge: Repo map rows for added/removed/renamed files, Commands,
  Recipes, Gotchas, and any statements that are no longer true (test counts, backend behaviour, known issues).
  Run `tests/test_docs.py` to confirm every path it mentions still exists, commit the CLAUDE.md update
  (by name) together with or just before the push/merge, and say in your reply what you changed in it.

## Architecture

**`include/` vs `src/`**: `include/` holds only dataclasses, enums, and
constants (`include/experiment/`, `include/eye_tracking/`) — no logic ever
goes here. `src/` mirrors that split (`src/experiment/`, `src/eye_tracking/`,
`src/dashboard/`) and holds all behavior.

**Framework-agnostic session state machines.** `PresaccadeSession` (baseline
phase) and `ExperimentSession` (saccade phase) each expose
`on_space()`/`on_response_key()`/`tick()`/`render_state()` and know nothing
about PsychoPy — `run_experiment.py`'s frame loops (sharing `FramePresenter` for caption/fades/frame clock and `deliver_keys`) call `tick()` +
`render_state()` once per rendered frame and applies the returned plain dict
to PsychoPy stimuli (`apply_render_state()`/`draw_all()`). Both phases share
one `ResultLogger` (one CSV, distinguished by a `phase` column) but run
independent `ZestStaircase` instances so each condition converges on its own
threshold. Test mode (shortened trial counts, saccade phase first for smoke
testing) is a runtime decision, not a constant: `run_experiment.py`'s
`_resolve_test_mode()` reads the startup dialog's explicit Mode dropdown
(Real/Test); a Real run requires a non-empty Participant ID. Real runs also get
counterbalanced phase order (`session_schedule.py`). Every trial (all three sessions) waits for the participant to press SPACE (`AWAITING_SPACE` phases) and the swap comes `FOREPERIOD_MIN_MS`-`FOREPERIOD_MAX_MS` (500-1000 ms) after the press; there is no pause menu, rest break, session cap or mid-session recalibration.

**SPACE presses are never silently dropped** (`space_input.py`, tests `tests/ai-tests/test_space_input.py`). Calibration (saccade session and tutorial): one press is enough; if `average_recent_ratio()` is still `None` (window not full; slow at ~45 Hz) the press is latched, the window restarted, and `tick()` completes the step once it fills (`render_state()["hud"]["calibration_wait_s"]` shows in the Test-mode HUD, "TRACKING LOST?" after 3 s). Tutorial: `run_tutorial_phase` holds one SPACE pressed while the narrator speaks and delivers it afterwards. Trials: SPACE is accepted in the `AWAITING_SPACE` phases; during the foreperiod it is ignored (must not restart it or start the next trial); during a trial it is ignored until the participant has answered, then it is carried and starts the next trial the moment `AWAITING_SPACE` begins. Same-frame repeats collapse to one press; a press made before a phase's first screen (start-up, marker check) is discarded; a press in the frame COMPLETE first appears does not dismiss it. Every ignored press is counted in `session.space_tally` (per phase + reason), printed at the end of each phase and stored in `session.json` as `ignored_space_presses` (`ResultLogger.set_ignored_space_presses`).
`PresaccadeSession` still picks between fixed `_TEST`/`_REAL` trial-count
constant pairs in `include/experiment/constants.py`; `ExperimentSession` uses `SACCADE_FIXED_TRIALS_*` (constant stimuli), and
its practice block and RT-test use `_TEST`/`_REAL`-paired counts.

**Gaze tracking is behind an interface.** `include/eye_tracking/interfaces.py`
defines `GazeSource`; `ExperimentSession` only ever talks to that interface.
`NeonGazeSource` (`neon_source.py`) polls `receive_gaze_datum()` on a
background thread and turns scene-camera gaze x / `NEON_SCENE_WIDTH` into the
same 0-1 "ratio" the webcam produces; `GazeMapper` (`gaze_mapping.py`, no
MediaPipe dependency) applies the shared 3-point calibration fit, dead-zone
classification and display smoothing. `build_gaze_source()`
(`gaze_factory.py`) picks the backend from the startup dialog's "Gaze source"
dropdown and imports lazily, so a machine without `PySpin`/`mediapipe` (or
without `pupil-labs-realtime-api`) still runs the other backend.
`WebcamGazeSource` owns a `Camera` + `GazeTracker` pair on a background thread. `IOHubGazeSource` is an
untested placeholder for real eye-tracking hardware (Tobii/EyeLink) via
PsychoPy's ioHub, written against the documented API but never run against
real hardware.

**Camera → tracker pipeline** (webcam backend only) (`src/eye_tracking/`):
`Camera` wraps the Spinnaker SDK for a FLIR Blackfly S and runs its own
background thread that keeps only the single *latest* raw Mono8 frame (not a
queue) — intentional, since only the newest frame is ever consumed.
`GazeTracker` runs MediaPipe FaceLandmarker on that frame, downscaling it to
`TRACKING_FRAME_WIDTH` first (inference cost scales with pixel count; face
detection doesn't need full sensor resolution). Two gotchas worth knowing
before touching this code:
- `PySpin` isn't a normal pip package — it lives in PsychoPy's own "user
  packages" directory, which only gets added to `sys.path` as a side effect
  of `import psychopy`. Any module that imports `camera.py` must import
  `psychopy` first or `PySpin` fails with `ModuleNotFoundError` (this is why
  `tests/conftest.py` imports `psychopy` before anything else, and why
  standalone camera scripts do too).
- `PySpin` is built against numpy 1.x's C API; numpy 2.x breaks its import
  with `_ARRAY_API not found`. `requirements.txt` pins `numpy<2` plus
  matching opencv versions for this reason — don't casually bump either.

**Dashboard is a separate process reading the same files.** `src/dashboard/`
is an independent Flask app (`data_access.py` reads/writes the trials/`session.json` of run folders and the old CSV/`_meta.json` files, and `settings.json`, under `data/`;
deleting a new-layout session removes its whole run folder, glasses video included) — no shared process
or IPC with the experiment, just shared files on disk.

**Shared logic lives in `trial_mechanics.py`/`scoring.py`, not copy-pasted.**
`OnsetDetector` (`src/experiment/trial_mechanics.py`) is the one saccade-
onset debounce implementation used everywhere it's needed — `ExperimentSession`'s
RT-test and main trials, and `TutorialSession`'s practice trials
(three call sites total). `is_valid_for_saccadic_analysis()`
(`src/experiment/scoring.py`) is the single `flash_during_saccade`
validity gate both the live ZEST update and `results_graph.py` apply.
`RollingReactionTime` (`trial_mechanics.py`) is the one reaction-time average and
`CalibrationLatch` (`space_input.py`) the one calibration-press latch. If you're about to
hand-roll any of these again in a new call site, use the existing one
instead — that duplication is exactly what was cleaned up.

**Test doubles.** `tests/conftest.py` provides `FakeGazeSource` (an in-memory
`GazeSource` implementation tests set `.zone`/`.face_found`/`.position` on
directly) and a `fake_time` fixture that monkeypatches `PausableClock`'s time
source so timing-dependent state-machine tests (foreperiods, response
windows, debounce windows) run deterministically without real waits.

**Headless simulator and `tests/ai-tests/`.** `src/experiment/simulation.py`
runs the *real* `ExperimentSession` (calibration, RT test, saccade trials)
against a `SimulatedParticipant` that only reads `render_state()`, on a
`SimulatedClock` (never real time) and a `SimulatedGazeSource` (real
`GazeMapper`, so calibration/zones match the Neon path) - no PsychoPy import,
no files (`MemoryLogger`), deterministic per `seed`, a full session in
seconds. `simulate_trial()` / `simulate_session(max_trials, profile, seed)`
return a `SimulationResult` incl. `flash_truth` - ground truth for whether each
flash really fell inside the simulated saccade, which the real experiment
can't know. `ParticipantProfile` is the knob set (reaction time, psychometric
thresholds incl. suppression, tracker noise/latency, dropouts). The model is
intentionally simple: trust relative comparisons (setting A vs B) for
plumbing/timing logic, not absolute numbers, and confirm anything important on
real glasses. `tests/ai-tests/` is the AI-agent entry point: `README.md`
(recipes + how to read the summary), `run_sim.py` (CLI: `--scenario`, `--set
CONSTANT=VALUE` temporary constant override incl. modules that copied it by
name, `--profile`, `--sweep`, `--timeline`, `--json`), `sim_tools.py`
(`run`/`sweep`/`summarize`/`override_constants`), plus `test_ai_tools.py`
self-tests. That folder also holds the older per-module test files. Tests for
the simulator itself: `tests/test_simulation.py`. Real recorded gaze can be
replayed instead via `src/eye_tracking/gaze_trace.py` (`ReplayGazeSource`,
fixture `tests/fixtures/neon_trace_sample.csv`).

**Tracking-quality gate.** A flash trial is also flagged `tracking_ok=False`
(and excluded from ZEST/analysis, same as `flash_during_saccade != True`) if
gaze tracking dropped out within `TRACKING_LOSS_WINDOW_MS` of the flash - see
`scoring.is_valid_for_saccadic_analysis(flash, tracking_ok)`. A saccade-phase
session also writes `gaze_quality` (rate, gaps, latency, clock offset) into
`session.json` and a raw `gaze_trace.csv` (Neon; old screen-backend traces hold screen-ratio x
`NEON_SCENE_WIDTH` in `x_px`, which `analyze_session` and replay still handle).

**Simulator finding (fixed in simulation, unverified on glasses).** Onset/landing used to be
stamped with classifier observation times that trail the real eye by the tracker latency, so with 16 ms
latency only ~3% of kept flashes really hit the saccade. Samples are now stamped with capture time
(Neon: `device_unix_s` + clock offset on the perf_counter timeline; sim: arrival minus the exact
`gaze_latency_ms`), giving precision 1.0 at 0/16/50 ms in `run_sim.py --sweep-profile
tracker_latency_ms=0,16,50`. The simulator hands the session the exact latency; a real Neon estimate is noisier.

## Gotchas that cost real debugging time

**Open problem (Oct 2026): gaze arrives ~90-100 ms late in real sessions.** Every full session
since the clock/onset work shows `gaze_quality.latency_ms_median` ~91-100 ms (the removed screen-mapped
backends also dropped the rate to ~44-53 Hz), not the ~16-18 ms measured in
standalone scripts. In the raw trace the delivery delay steps up by ~65 ms about 6 s in (when the
render loop starts) and stays flat, while the rate falls from 200 to ~160/s (~19% of samples never
arrive): the signature of the receiving thread being starved by the render loop (same process /
GIL; the Real-Time API's simple device keeps only the newest sample, so the backlog is not in our
code). Untested fixes: `sys.setswitchinterval(0.0005)`, or receive Neon in its own process.
Consequence: the live `flash_during_saccade` check is unreliable; judging flashes after the fact from the
raw trace's phone capture times is unaffected. Also: the phone clock drifts ~1.2 ms/min against the PC (the clock offset is
re-estimated every 60 s), and `time.monotonic()`/`time.time()` have 15.6 ms resolution on
Windows (everything now uses `src/timebase.py` / `perf_counter`).

**Neon timing and geometry (measured Oct 2026).** Gaze reaches the PC ~16 ms
after the event (p95 ~21 ms, max ~35 ms), ~140-180 Hz effective with gaps up
to ~47 ms. Latency is only right if you convert with
`local = device_ts + offset` from `estimate_time_offset()` (the sign is easy
to get backwards: it gave a bogus 414 ms once). `NeonGazeSource` stamps
samples with their *capture* time (`device_unix_s + offset`, moved onto the
`timebase.now()` perf_counter timeline; arrival time only as a fallback, strictly increasing, clamped
to <=150 ms latency) and the session then must not subtract `gaze_latency_ms` again
(`timestamps_are_capture_time`). The offset is re-estimated every 60 s in the background (unverified on
hardware whether `estimate_time_offset` is safe while streaming). Standalone scripts measure ~18 ms but full sessions show 90-100 ms (open problem above) - check `gaze_quality.latency_ms_median`. The scene
camera is head-mounted, so head movement shifts gaze x with no eye movement -
calibration assumes a still head (chin rest). (Screen-marker / AprilTag head compensation was removed in Oct 2026.) The Neon source has
no `latest_frame()`, so the debug camera preview and video recording are
skipped for it (`has_frames = hasattr(gaze, "latest_frame")` in
`run_saccade_phase`). `NEON_DEVICE_ADDRESS` in
`include/eye_tracking/constants.py` is pinned to the phone's IP (set to `None`
to auto-discover). `SACCADE_ONSET_STABILITY_MS` was lowered 40 -> 10 for Neon
on the strength of latency alone; confirm with `neon_fixation_test` (the
"drift-free jitter" section, with the head still) and watch the
false-alarm rate. `NeonGazeSource` has only been tested against fake devices
plus these hand-run scripts, not in a full session.

**TTS narrator (`src/experiment/narrator.py`) is Windows-only SAPI via COM.**
It rides on `pywin32` (already a PsychoPy dependency, not a new package).
COM objects are apartment-threaded, so the `SAPI.SpVoice` object must be
created *and* called from the same thread — `_run()` wraps its whole worker
loop in `pythoncom.CoInitialize()`/`CoUninitialize()` for this reason.
`Speak()` blocks synchronously for the real duration of the utterance, which
is why narration lives on a dedicated background thread instead of the
render loop. `speak()` does not interrupt in-flight/queued speech, so
callers must gate on the text actually changing (`narrate_if_changed`) or
utterances silently queue up and drift behind what's on screen.

**Caption, speech and HUD (run_experiment.py).** `update_caption()` shows the bottom caption and speaks it in lockstep (`narrate_if_changed` never speaks blank text). Sessions' `render_state()["hide_caption"]` is True during real (non-practice) main trials, in both the saccade and baseline phases: the caption is then neither drawn nor spoken. The top-left `hud` text is drawn only in Test mode (`build_stimuli(win, show_hud=test_mode)` -> `stimuli["show_hud"]`).

**The first-timer tutorial blocks input while narrating, unlike real
phases.** `run_tutorial_phase` in `run_experiment.py` withholds all
key/tick forwarding while `narrator.is_speaking` is true, so a fast
participant cannot skip ahead mid-narration — real saccade/baseline trials
don't have this restriction. If the tutorial ran, its
`precomputed_calibration_ratios` and `precomputed_rt_samples_ms`
(`TutorialSession.rt_samples_ms`: one saccade-onset latency per practice trial)
are threaded into `run_saccade_phase`: the saccade phase skips calibrating,
its own RT test and its own practice trials, seeds its `RollingReactionTime`
(`_rt`, `_avg_reaction_time_ms`) from those samples (`DEFAULT_REACTION_TIME_MS` if fewer than 3 were
usable) and goes straight to the main trials. Without the tutorial the old
path (calibration + RT test + practice) is the fallback. Narrated strings stay
under ~15 words and are not repeated between screens (the narrator re-speaks
on every text change), so the practice intro is spoken once and later trials
have an empty caption.

**Flash timing (the only mode).** The dot/cross sit at 15%/85% of the screen (~41 deg, expected ~110 ms
by the main sequence - unverified on this setup: check `saccade_duration_ms` in the next real session; the
simulator's `ParticipantProfile.saccade_ms_*` scale with the same geometry). `SACCADE_ONSET_DISPLACEMENT` is a
fraction of the dot-cross span (an angle), so it was rescaled 0.08 -> 0.05 for the wider layout. `SHARP_GO_CUE` shows the
dot/cross instantly (no 0.4 s fade) in the trials. (A gaze-triggered "onset" flash mode existed once and was removed;
see git history.)
Saccade-phase setup (built for judging flashes
after the fact because the live check is useless at ~90 ms gaze latency): scheduled flashes at
average RT + one of `TIMING_OFFSETS_MS` (-80..+80 ms) + a random `TIMING_JITTER_MS` (+/-10 ms)
bounce (logged in `scheduled_delay_ms`), fixed probe contrasts `SACCADE_FIXED_CONTRASTS`
(4/6/8 % on the line branch, was 10/15/20 %: evenly spaced between the startup dialog's "Saccade contrast min/max (%)" fields, logged as `saccade_contrasts` in the meta file; method of constant stimuli: no live ZEST in the saccade phase) and a fixed count
`SACCADE_FIXED_TRIALS_*`. Set `SACCADE_FIXED_CONTRASTS = ()` to return to the staircase. The
analysis is `src.experiment.analyze_session` (accuracy vs flash time relative to saccade onset, from the
raw trace's phone capture times; first real data: 3 %/6 % were invisible 15 deg off to the side, 20-80 %
were seen, with a possible dip within +-50 ms of onset, n=28). `run_experiment.main` sets `sys.setswitchinterval(0.0005)` as a
first test of the receiver-starvation hypothesis above.

**Saccade-phase flash timing is open-loop (the `"scheduled"` mode), not gaze-contingent, and
`flash_during_saccade` is a post-hoc window check.** `ExperimentSession` schedules each trial's flash at a fixed delay
(`avg_reaction_time_ms +` one of `TIMING_OFFSETS_MS`) from target onset, independent of the real-time gaze classifier. The
classifier keeps running only to measure `reaction_latency_ms` and to compute `flash_during_saccade` after the fact (whether
the scheduled flash time fell inside `[_onset_detector.since, _target_landed_since]`). The cost is a structurally common
failure - the flash simply missing the saccade window - so `flash_during_saccade` is `True`/`False`/`None` (`None` when
landing was never confirmed), and both the live `ZestStaircase` update *and* `results_graph.py` require exactly `True`
(the two intentionally stay in lockstep). With the shipped constant stimuli the block ends by count
(`SACCADE_FIXED_TRIALS_*`); with `SACCADE_FIXED_CONTRASTS = ()` (staircase) `_should_stop_main_block()` runs until the ZEST
credible interval narrows (minimum valid-trial floor, `MAX_SACCADE_TRIALS_*` cap).

**Saccade onset is a displacement rule, not a zone crossing.** `OnsetDetector` fires when
`GazeSample.position` is more than `SACCADE_ONSET_DISPLACEMENT` (0.08, position units 0=dot..1=cross)
from the fixation baseline at the source for `SACCADE_ONSET_STABILITY_MS` of capture time, and
`since` is the first sample beyond it. The old rule waited for gaze to leave the source *zone* (CENTER
counted as left, i.e. ~40% of the travel) so narrowing `GAZE_DEAD_ZONE` used to help onset - it no longer
does; the dead zone now only matters for landing/zones. Tests/fakes that only set `.zone` must move
`.position` too (`FakeGazeSource.position` follows the zone). Threshold evidence:
`tests/ai-tests/analyze_onset_threshold.py` on recorded traces; recorded screen-mapped traces (old backends, since removed)
contain mapping spikes (position jumps of ~1.0 for a few samples) that any rule will trigger on.

**Never use `time.monotonic()` / `time.time()` for event timing on Windows.** Both tick every 15.6 ms
(`time.get_clock_info`); use `src/timebase.py` (`now()`, `unix_now()`), enforced by `test_architecture_rules.py`.
Tests patch `timebase._clock` (see `fake_time`), never the global `time` module.

**`CAMERA_GAIN_DB` is an unvalidated placeholder.** Fixed exposure/gain was
long claimed in `_configure_camera`'s docstring before gain was actually
wired up in `src/eye_tracking/camera.py` — `Gain`/`GainAuto` are now set,
but the `6.0` dB value has not been retuned against the real IR illuminator.
Validate it by capturing and inspecting a real frame before trusting it (the old diagnose_camera.py helper was removed in commit 0989ed1; recover it with `git show 0989ed1^:src/eye_tracking/diagnose_camera.py`).

**The 880Hz/0.5s tone (`WARNING_TONE_HZ`/`WARNING_TONE_DURATION_S`) has been
repurposed, not just restored.** It originally matched Diamond, Ross &
Morrone (2000)'s pre-target warning tone, was removed (commit `638138f`) as
a methodology departure, and is now back with different semantics: a go-cue
beep fired the instant the target/cross appears
(`phase_just_became_active()` in `run_experiment.py`), timed to be the
anchor for the reaction-time measurement itself — not a "get ready" warning
before an already-visible target. Same constant values, different meaning;
don't assume its presence means the original paper's warning-tone
methodology is back.

**`PausableClock` is just the injected time source** (`pausable_clock.py`; the name is historical: pausing, the pause menu,
rest breaks, recalibration and the session cap were removed). Tests/simulation substitute a controlled clock.

**`cv2.VideoWriter` fails silently, not loudly.** `VideoRecorder`
(`src/eye_tracking/video_recorder.py`, opt-in via the startup dialog's
"Record video" toggle, default Yes) doesn't raise if the codec/container
can't be opened (e.g. `mp4v` unavailable on a given OpenCV build, an
unwritable path) — `.write()` calls on a non-functional writer just silently
do nothing, which would otherwise mean a whole session records zero video
with no error anywhere. `start()` checks `writer.isOpened()` and prints a
warning to stderr if it failed, but doesn't stop the session — video
recording is a nice-to-have, not something a failure here should be able to
crash real data collection over. If a session's video is unexpectedly empty,
check the console output from that run before assuming the recorder itself
is broken.

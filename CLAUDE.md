# CLAUDE.md

PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000 design): detect a briefly flashed stimulus once
during a saccade and during fixation (the saccade phase's control and far-from-saccade flashes, sorted post hoc) to measure saccadic suppression. Gaze: Pupil Labs Neon (default,
Real-Time API) or, with `SACCADES_BACKEND=webcam` (no dialog field any more), FLIR Blackfly S (`PySpin`) + MediaPipe. Participant-facing docs: [README.md](README.md).

**This is the `line-stimulus` branch**; `main` has the grating version. Differences from the grating:
- Flash = thin horizontal line (Braun, Schutz & Gegenfurtner 2021): full screen width, vertical Gaussian (`LINE_SD_DEG` 0.15),
  centred `LINE_OFFSET_DEG` (2 deg) above/below the saccade TARGET, drawn `GRATING_DURATION_FRAMES` (1) frame.
  Answer UP (above) / DOWN (below); LEFT/RIGHT unused.
- `Side.ABOVE/BELOW` replaces `Orientation`. `TrialResult` fields (and old loose-CSV columns) are still `orientation` / `response_orientation`
  (hold 'above'/'below'); `trials.csv` calls them `line_side` / `answer_side`; old 'vertical'/'horizontal' CSVs load via `logger._LEGACY_SIDES` (vertical->above, horizontal->below, arbitrary
  but consistent). The render-state key is still `"grating"` (x/y = line position, `orientation` = side).
- Geometry: the startup dialog asks for screen height and viewing distance; `geometry.Geometry` (immutable, built once in `main`) turns them into the line's offset/SD in
  screen-height units and is passed to `ExperimentSession`/`TutorialSession` (-> `trial_mechanics.line_position(ref, side, geometry)`) and `build_stimuli(geometry=)`
  (`GratingStim` with `tex=None`, `'gauss'` mask, size `LINE_WIDTH_SCREENS` x screen width by `geometry.line_height_height_units`; `contrast` scales a white field = light line).
  `Geometry.default()` = the placeholder constants (`VIEWING_DISTANCE_CM`/`SCREEN_HEIGHT_CM`; the `LINE_*_HEIGHT_UNITS` constants now only mirror them and the simulator's
  saccade duration still uses them). The geometry used is in `session.json` `settings` and the `analyze_session` header.
- Tutorial quiz = line above/below a dot (UP/DOWN, 3 in a row); captions say "UP if the line was above ...".
- **Contrasts are a GUESS**: `SACCADE_FIXED_CONTRASTS` 4/6/8 % (dialog min 4 / max 8, 3 levels). Test-mode 10/15/20 % was at ceiling
  (18/22 correct), old foveal grating threshold ~1.3 %, 2/4/6 % judged too hard. 4-8 % is ~5-10 grey steps on 8-bit
  (`HARDWARE_CONTRAST_FLOOR` 0.78 %), so levels are coarse. Re-pilot on glasses.
- Rolling RT = `robust_average_ms` over the last `RT_AVERAGE_ROLLING_WINDOW` (12) latencies; <`RT_MIN_SAMPLES_FOR_AVERAGE` (3)
  plausible samples keep the previous value (initially `DEFAULT_REACTION_TIME_MS`). `session.json` phase summary (results section) adds `rt_samples_total/_in_window/_dropped_in_window`.
- Line rendering (profile, width, brightness, one-frame visibility at 60 Hz), quiz/captions and every contrast level are untested on a real window/glasses (simulator + tests only).

## Rules (read first)

- **Run Python only via the wrapper**: `.\scripts\py.ps1 <args>` (PowerShell) or `scripts/py.sh <args>` (Git Bash). It finds PsychoPy's
  Python (`%LOCALAPPDATA%\Programs\PsychoPy\python.exe`, or `$env:PSYCHOPY_PYTHON`) and runs from the repo root; it cannot be double-clicked or typed in cmd (use `powershell -ExecutionPolicy Bypass -File scriptspy.ps1 ...`). Plain `python` lacks deps and `src`/`include`.
- **Never launch the experiment or dashboard** (`run_experiment` opens a fullscreen window on the user's screen). Use the headless
  simulator or tests. Real-glasses runs are the human's. The Neon (`NEON_DEVICE_ADDRESS`, `include/eye_tracking/constants.py`) is sometimes reachable for read-only checks: ask first.
- Don't merge, push or switch branches unless asked. **Before any push/merge the user asks for, update this file first** (Repo map rows,
  Commands, Gotchas, stale claims, test count), run `tests/test_docs.py`, commit it by name, and say what you changed.
- **Other AI sessions edit this folder concurrently**: `git status` / `git log -5` first; never revert or tidy changes you didn't make;
  stage and commit only your own files by name (never `git add -A` / `commit -a`); don't touch other sessions' files (e.g. `src/eye_tracking/neon_*`), tell the user instead.
- Environment: PowerShell 5.1 has no `&&`; files may be CRLF (keep each file's endings); `.gitignore` has `data/`, which also matches any `data` folder (tests: use `tests/fixtures/`).
- `data/` is git-ignored runtime output: never commit it, never write to it from tests/simulations (simulator uses an in-memory logger).
- Participant data (trials, gaze trace, glasses video) is personal data: never commit it, never print it in reports, never copy real sessions into `tests/fixtures/` (only anonymous recordings belong there).
- Report test results truthfully (ran / skipped). Suite: ~540 tests, ~1-2 min, 2 skipped without PySpin/mediapipe (`test_camera.py`, `test_gaze_tracking.py` `importorskip`).

## Commands (all through the wrapper)

```
.\scripts\py.ps1 -m pip install -r requirements.txt
.\scripts\py.ps1 -m pytest -q                                   # full suite; one test: tests/ai-tests/test_session.py::name -v
.\scripts\py.ps1 -m src.experiment.simulation                   # headless simulated trial timeline (AI-safe)
.\scripts\py.ps1 tests/ai-tests/run_sim.py --help               # sim runner: --scenario --set --profile --sweep --sweep-profile --seeds --timeline --json --real --list
.\scripts\py.ps1 -m src.experiment.run_experiment               # REAL experiment (human only)
# double-click launchers in the repo root: run_experiment.bat, analyze_latest.bat, run_tests.bat (extra args pass through; fine from cmd)
.\scripts\py.ps1 -m src.dashboard.app                           # dashboard, localhost:5000 (human)
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir D]  # newest session by default (unix ts, run folder name or path); writes <run>/analysis/
.\scripts\py.ps1 -m src.experiment.import_glasses_video <run-folder> <exported.mp4>  # attach + sync the phone's glasses video
.\scripts\py.ps1 -m src.eye_tracking.neon_fixation_test [secs]  # Neon jitter check (human)
.\scripts\py.ps1 -m src.eye_tracking.latency_test               # Neon vs webcam latency (human)
.\scripts\py.ps1 -m src.eye_tracking.record_gaze_trace out.csv 10   # record raw gaze for replay (human)
```

## Status and next steps (2026-10-25)

- **Goal**: grant to test saccadic suppression in TBI vs non-TBI. Present as a *feasibility pilot (n=1)*, never a group result.
- **Shipped setup**: flashes at average RT + offset (-80..+80 ms) + random +/-10 ms; fixed saccade contrasts; sharp go-cue; dot/cross at
  15 %/85 % of the screen; plain Neon, still head (chin rest). Onboarding: calibration -> how-to-answer quiz -> 8 practice trials seeding the RT average.
  Every trial waits for SPACE. Control flashes (+400/+500 ms after average RT) + pre-specified gates give `analyze_session` a per-person suppression index.
- **Results layout and glasses video** (untested on glasses): a run writes `data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/` (`trials.csv`,
  `session.json`, `gaze_trace.csv`, analysis/ and glasses/ subfolders). The phone records the scene video (`NeonGazeSource.start_recording/stop_recording`);
  green sync flashes before the first trial and at the end let `import_glasses_video` align it to the PC clock. Only run against fake devices/synthetic frames.
  **Nothing has run on glasses since the last refactors**: only simulator + tests.
- **Pilot (1 participant, test mode, earlier grating setup)**: foveal threshold ~1.3 %; saccades ~75 ms / ~30 deg (15/85 layout); 3 %/6 % invisible
  when fixating the dot/cross (~15 deg off), 10-20 % seen. Flashes within +-50 ms of onset: 3/12 correct vs 12/13 at 50-300 ms away (Fisher p=0.001),
  sharpest in the 50 ms before onset; an earlier 20-80 % session agreed (pooled p<1e-4). Small n; the +-50 ms window was chosen post hoc (gates are now pre-specified).
- **Known problems**: (1) gaze reaches the PC ~95 ms late (Gotchas) so live `flash_during_saccade` is useless (post-hoc analysis unaffected).
  (2) RT SD ~60-80 ms, only ~1/3 of flashes land near onset (TBI likely worse); live rolling RT helps only with drift (simulation only).
  (3) Screen geometry is typed into the startup dialog each run (remembered); until the user measures their small monitor (~12 in wide, 60 Hz) it is the 26 cm / 50 cm placeholder and the dialog says so. Saccade size in degrees also depends on width and distance.
  (4) The separate baseline (foveal) phase was removed (not a like-for-like control, different retinal location): fixation now comes only from the saccade phase's control flashes + flashes >=100 ms from any saccade; the simulator gives ~46 fixation / ~36 saccade flashes per 150 trials (`flash_gates`), so ~15/~12 per contrast level.
- **Next, in order**: (1) short Test run then full 150-trial Real run with participant ID on glasses, analyse with `analyze_session`; (2) group analysis
  (per-person suppression index + group test or mixed model); (3) one-page grant figure + wording; (4) clinical metadata fields (group, time since injury,
  severity, symptoms, medication; needs IRB) in the dashboard; (5) find the ~95 ms delay cause (e.g. Neon in its own process); (6) update geometry constants.

## Repo map

Paths in this section are checked by `tests/test_docs.py` - keep them real.

| Path | What it is |
|---|---|
| `include/experiment/constants.py` | every tunable experiment constant (read the comment before changing; several encode trade-offs) |
| `include/experiment/types.py` | `TrialResult` (one CSV row), `TrialSpec`, `Target`, `Side` |
| `include/eye_tracking/constants.py` | gaze constants incl. `GAZE_DEAD_ZONE`, all `NEON_*` |
| `include/eye_tracking/interfaces.py` | `GazeSource`, the only gaze interface sessions use |
| `include/eye_tracking/types.py` | `GazeZone`, `GazeSample` |
| `src/experiment/session.py` | `ExperimentSession`: saccade phase state machine (core logic) |
| `src/experiment/tutorial_session.py` | `TutorialSession`: calibration -> quiz -> 8 practice trials yielding the first RT samples |
| `src/experiment/run_experiment.py` | entry point + PsychoPy rendering; the only PsychoPy-dependent module |
| `src/experiment/startup_dialog.py` | PyQt6 startup dialog: fields, closed-by-default Advanced section, help bubbles (frameless tool-tip window at the mouse, never a dialog/message box) |
| `src/experiment/startup_model.py` | Qt-free dialog model: `StartupSettings`, `validate`, `defaults_from_disk`, `suggest_tutorial`, `HELP_TEXTS` |
| `src/experiment/geometry.py` | `Geometry`: screen height + viewing distance -> degrees and the line's size in screen-height units |
| `src/experiment/pausable_clock.py` | `PausableClock`: the injected time source (name historical, no longer pauses) |
| `src/experiment/settings.py` | dashboard-configured default contrast floor (`data/settings.json`) |
| `src/experiment/scoring.py` | `score_outcome`, `is_valid_for_saccadic_analysis` (the validity gate) |
| `src/experiment/trial_mechanics.py` | `OnsetDetector`, `ShuffledBag`, `zone_for`, `line_position`, `robust_average_ms`, `sample_capture_time`, `RollingReactionTime` (updated at every detected onset incl. RT test/practice) |
| `src/experiment/calibration.py` | `CalibrationRounds` (3-point x 3-round state machine shared by saccade session and tutorial), `average_calibration_rounds` (rounds 2 and 3) |
| `src/experiment/space_input.py` | `CalibrationLatch`, `NarrationKeyGate`, `SpaceTally`, `deliver_space`, `leaves_complete_screen` |
| `src/timebase.py` | the ONE time source: `now()` = `perf_counter()`; wall clock via edge-synchronised anchor (`unix_now()`, `unix_ms()`) |
| `src/experiment/trial_factory.py` | trial sequences; every trial flashes (catch trials removed; old sessions may hold `false_alarm`/`correct_rejection`, still readable) |
| `src/experiment/zest.py` | adaptive contrast staircase |
| `src/experiment/logger.py` | `ResultLogger`: `TRIAL_COLUMNS`, `DROPPED_FROM_CSV`, `trial_from_row`, `read_trial_rows`/`read_trials` (new + old layout) |
| `src/experiment/session_files.py` | run-folder naming (`sanitize_participant_id`, `create_session_folder`), `SessionRef`, `list_sessions`/`find_session`/`newest_session` (a session's key is its unix start time) |
| `src/experiment/glasses_video.py` | no PsychoPy: sync-flash frame counts, `find_sync_flashes`, `compute_offset` (residual, ok/check manually) |
| `src/experiment/import_glasses_video.py` | CLI: copies exported video to `<run>/glasses/`, writes `video_sync.json` + `flash_video_times.csv` |
| `src/experiment/analyze_session.py` | post-hoc analysis: saccades from the raw gaze trace (phone capture times), flashes placed against nearest onset; report + plot + per-flash CSV |
| `src/experiment/flash_gates.py` | pre-specified gates (Braun 2021): `saccade` (eye fast at flash, latency 100-500 ms, amplitude 0.5-1.5x median), `fixation` (>=100 ms from any saccade, no tracking gap), else `excluded`; controls from `CONTROL_OFFSETS_MS` |
| `src/experiment/simulation.py` | headless simulator (simulated participant, clock, gaze) |
| `src/experiment/narrator.py` | Windows SAPI TTS narrator (see Gotchas) |
| `src/eye_tracking/neon_source.py` | `NeonGazeSource` (default backend): Real-Time API stream + quality stats |
| `src/eye_tracking/gaze_mapping.py` | `GazeMapper`: ratio -> calibrated position / zone / smoothing |
| `src/eye_tracking/gaze_factory.py` | `build_gaze_source()` backend picker (lazy imports) |
| `src/eye_tracking/gaze_trace.py` | raw gaze recording + `ReplayGazeSource` |
| `src/eye_tracking/webcam_source.py` | Blackfly S + MediaPipe backend (alternative) |
| `src/dashboard/app.py` | Flask dashboard (separate process; `data_access.py` reads/writes `data/`) |
| `tests/conftest.py` | `FakeGazeSource`, `fake_time` fixture |
| `tests/helpers.py` | `NullLogger` for session tests |
| `tests/ai-tests/test_line_stimulus.py` | line branch: placement above/below target, UP/DOWN only, quiz, old-CSV loading |
| `tests/ai-tests/README.md` | headless-simulation guide for AI agents |
| `tests/ai-tests/run_sim.py` | simulation CLI |
| `tests/ai-tests/find_bugs.py` | randomized campaign over the simulator; prints a repro command per finding |
| `tests/ai-tests/invariants.py` | `check_session()`: contracts every simulated session must satisfy |
| `tests/fixtures/neon_trace_sample.csv` | a real recorded Neon gaze trace |
| `tests/ai-tests/analyze_onset_threshold.py` | offline onset-rule study on recorded traces |
| `scripts/py.ps1` | interpreter wrapper |

## How to verify a change (cheapest first)

1. **Session/scoring/ZEST/logger logic**: add or update a test (`FakeGazeSource` + `fake_time`, or the simulator), run pytest. A previously passing test that now fails means behaviour changed: decide if intended before editing it.
2. **Whole-session behaviour or a timing/threshold constant**: `tests/ai-tests/run_sim.py` (`--set NAME=VALUE`, `--sweep NAME=a,b,c --seeds 5`, `--timeline`). Trust relative comparisons only; the participant model is simple (`tests/ai-tests/README.md`).
3. **Gaze source**: patterns in `tests/test_neon_source.py` / `tests/test_gaze_trace.py` (fake device, or replay the fixture via `ReplayGazeSource`).
4. **Needs real eyes/glasses/screen** (latency, jitter, calibration quality): write the measurement script and ask the human; don't guess numbers.
- After changing session, scoring or ZEST logic run `tests/ai-tests/find_bugs.py --sessions 300` (fuzzes the real session against `invariants.py`). `test_bug_injection.py` re-injects known bugs to prove checks still catch them.
- Enforced by tests: TrialResult/CSV/dashboard schema sync (`test_schema_sync.py`), constant relations (`test_constants_sanity.py`), live ZEST feed vs the shared validity gate (`test_validity_lockstep.py`), architecture (`test_architecture_rules.py`: no PsychoPy in session logic, no `time.monotonic()`/`time.time()` in `src/` outside `timebase.py`, lazy backend imports, no logic in `include/`).

## Recipes

- **Add a trials.csv column / `TrialResult` field**: field in `include/experiment/types.py` (default `None`) -> one `Column(...)` in `TRIAL_COLUMNS`
  (`logger.py`: plain name, writer, converter back to old text) or an entry with a reason in `DROPPED_FROM_CSV` (`test_schema_sync.py` fails until each
  field is one or the other) -> `trial_from_row` -> set it where `session.py` builds the `TrialResult` -> test (see `tests/ai-tests/test_tracking_quality.py`). Old CSVs must still load.
- **Change a tuning constant**: edit in `include/**/constants.py` (read its comment), then `--sweep` in the simulator and pytest.
- **Add a gaze backend**: implement `GazeSource`, register in `gaze_factory.py` (lazy import!) and `_selected_backend()` in `run_experiment.py` (env var `SACCADES_BACKEND`; the dialog has no backend field); reuse `GazeMapper`.
- **Reproduce a bug without hardware**: replay a recorded trace (`record_gaze_trace`, or the fixture) or use a `ParticipantProfile` in the simulator.

## On-disk contracts

- `trials.csv`: one row per trial (`phase` column: saccade / rt_test; old runs also baseline), schema `TRIAL_COLUMNS` with plain names. Order: contrast, yes/no group, outcome,
  stimulus/answer pair, ms block (`flash_ms_after_go_cue`, `reaction_ms`, `answer_ms`, `flash_unix_ms`), `tracking_ok`. `DROPPED_FROM_CSV` lists fields kept out and why
  (`source`/`target`: direction = trial parity; `timing_offset_ms`: the delay is in `flash_ms_after_go_cue`).
- `session.json` sections: participant, run, settings, hardware, results, gaze_quality, ignored_space_presses, glasses_video. Saccade phase also writes `gaze_trace.csv`.
- Readers (`read_trial_rows`/`read_trials`, `data_access`, `analyze_session`, replay) accept both the run-folder layout and old loose `results_<ts>.csv/_meta.json`; old screen-backend traces hold screen-ratio x `NEON_SCENE_WIDTH` in `x_px` (why that constant stays).
- The dashboard is a separate Flask process sharing only files with the experiment; deleting a new-layout session removes its whole run folder (glasses video included).

## Architecture invariants

- **`include/` = dataclasses, enums, constants only; no logic. `src/` mirrors it and holds all behaviour.**
- **Sessions are PsychoPy-free state machines.** `ExperimentSession` (saccade phase) exposes `on_space()`/`on_response_key()`/`tick()`/`render_state()`;
  the `run_experiment.py` frame loops (shared `FramePresenter` for caption/fades/frame clock, `deliver_keys`) call `tick()` + `render_state()` once per frame and apply the plain dict
  (`apply_render_state()`/`draw_all()`). Test mode (short counts) is a runtime choice:
  `_resolve_test_mode()` reads the startup Mode dropdown; Real needs a non-empty Participant ID. A run is tutorial (optional) -> saccade phase -> a console line naming the `analyze_session` command.
  There is no pause menu, rest break, session cap or mid-session recalibration, and no baseline phase (removed; old `baseline` rows still load via `PHASE_WORDS`). `ExperimentSession` uses `SACCADE_FIXED_TRIALS_*`.
- **Gaze is behind `GazeSource`.** `NeonGazeSource` polls `receive_gaze_datum()` on a thread, turns scene gaze x / `NEON_SCENE_WIDTH` into a 0-1 ratio; `GazeMapper` does calibration fit, dead zone, smoothing.
  `build_gaze_source()` imports lazily so a machine missing `PySpin`/`mediapipe`/`pupil-labs-realtime-api` still runs the others. `IOHubGazeSource` is an untested placeholder. `WebcamGazeSource` = `Camera` (own thread, keeps only the newest Mono8 frame, intentionally) + `GazeTracker` (MediaPipe on a frame downscaled to `TRACKING_FRAME_WIDTH`).
- **Shared logic lives once**: `OnsetDetector` (RT test, main trials, tutorial practice), `is_valid_for_saccadic_analysis()` (the live ZEST update), `RollingReactionTime`, `CalibrationLatch`. Reuse them, don't hand-roll.
- **Test doubles**: `FakeGazeSource` (set `.zone`/`.face_found`/`.position`; `position` must move with `.zone`) and `fake_time` (patches `timebase._clock`, never the global `time`).
- **Simulator**: `simulation.py` runs the real `ExperimentSession` against `SimulatedParticipant` (reads only `render_state()`), `SimulatedClock`, `SimulatedGazeSource` (real `GazeMapper`); no PsychoPy, no files, deterministic per `seed`.
  `ParticipantProfile` = knobs (RT, thresholds incl. suppression, tracker noise/latency, dropouts); `simulate_session()` returns `flash_truth` (whether each flash really hit the saccade, unknowable live).
  Entry points: `tests/ai-tests/README.md`, `run_sim.py`, `sim_tools.py`. Real gaze replays via `ReplayGazeSource`.
- **Tracking-quality gate**: a flash is `tracking_ok=False` (excluded like `flash_during_saccade != True`) if tracking dropped within `TRACKING_LOSS_WINDOW_MS` of it. `gaze_quality` (rate, gaps, latency, clock offset) goes to `session.json`.

## Experiment design (why it is built this way)

- **Flash timing is open-loop (`"scheduled"`)**, because live gaze is ~95 ms late: each flash is scheduled at average RT + one of `TIMING_OFFSETS_MS` (-80..+80 ms, bag shuffled with `CONTROL_OFFSETS_MS`)
  + `TIMING_JITTER_MS` (+/-10, logged as `scheduled_delay_ms`) from target onset. The live classifier only measures `reaction_latency_ms` and computes `flash_during_saccade` post hoc
  (scheduled time inside `[_onset_detector.since, _target_landed_since]`); it is `True`/`False`/`None` (`None` = landing never confirmed). ZEST requires exactly `True`.
  Real judgement is `analyze_session` on the raw trace's phone capture times (accuracy vs flash time relative to saccade onset).
- **Constant stimuli**: `SACCADE_FIXED_CONTRASTS` (evenly spaced between the dialog's min/max, logged as `saccade_contrasts`) and a fixed count `SACCADE_FIXED_TRIALS_*`; no live ZEST in the saccade phase.
  `SACCADE_FIXED_CONTRASTS = ()` returns to the staircase (stops via `_should_stop_main_block()` when the ZEST credible interval narrows; floor of valid trials; `MAX_SACCADE_TRIALS_*` cap).
- **Layout**: dot/cross at 15 %/85 % of the screen (~41 deg; saccade ~110 ms expected by the main sequence, unverified: check `saccade_duration_ms` next session; simulator `saccade_ms_*` scale with the geometry).
  `SHARP_GO_CUE` shows the dot/cross instantly (no fade) in trials.
- **Onset = displacement rule, not a zone crossing**: `OnsetDetector` fires when `GazeSample.position` is > `SACCADE_ONSET_DISPLACEMENT` (0.05 of the dot-cross span, i.e. an angle; 0=dot..1=cross)
  from the fixation baseline for `SACCADE_ONSET_STABILITY_MS` (10, lowered from 40 for Neon on latency alone: confirm with `neon_fixation_test` head still, watch false-alarm rate) of capture time; `since` = first sample beyond it.
  `GAZE_DEAD_ZONE` now matters only for landing/zones. Evidence: `tests/ai-tests/analyze_onset_threshold.py`; old screen-mapped traces have mapping spikes (jumps ~1.0 for a few samples) that any rule trips on.
- **Go-cue tone** (`WARNING_TONE_HZ` 880 / `WARNING_TONE_DURATION_S` 0.5): fired the instant the target/cross appears (`phase_just_became_active()`), the anchor of the RT measurement. It shares the original paper's pre-target warning tone values but NOT its meaning.
- **Simulator finding**: onset/landing must be stamped with sample capture time, not classifier observation time (which trails by the tracker latency; with 16 ms latency only ~3 % of kept flashes really hit the saccade).
  Now precision 1.0 at 0/16/50 ms (`run_sim.py --sweep-profile tracker_latency_ms=0,16,50`); the simulator gives the session the exact latency, a real Neon estimate is noisier.

## SPACE and input rules (`space_input.py`, `tests/ai-tests/test_space_input.py`)

SPACE presses are never silently dropped.
- Calibration (saccade session and tutorial): one press suffices; if `average_recent_ratio()` is still `None` (window not full; slow at ~45 Hz) it is latched, the window restarted, and `tick()` completes the step when full
  (`render_state()["hud"]["calibration_wait_s"]` in the Test HUD, "TRACKING LOST?" after 3 s).
- Trials: accepted in `AWAITING_SPACE`; the swap follows `FOREPERIOD_MIN_MS`-`FOREPERIOD_MAX_MS` (500-1000 ms) after the press. During the foreperiod it is ignored (must not restart it or start the next trial); during a trial it is ignored until answered, then carried and starts the next trial when `AWAITING_SPACE` begins.
- Same-frame repeats collapse to one; a press before a phase's first screen (start-up) is discarded; a press in the frame COMPLETE first appears doesn't dismiss it.
- Tutorial: `run_tutorial_phase` withholds all key/tick forwarding while `narrator.is_speaking` (unlike real phases) and delivers one held SPACE afterwards.
- Every ignored press is counted in `session.space_tally` (phase + reason), printed per phase, saved as `ignored_space_presses` (`ResultLogger.set_ignored_space_presses`).

## Tutorial hand-off, caption and narration

- If the tutorial ran, its `precomputed_calibration_ratios` and `precomputed_rt_samples_ms` (`TutorialSession.rt_samples_ms`) go to `run_saccade_phase`: it skips calibration, RT test and practice, seeds `RollingReactionTime`
  (`_rt`, `_avg_reaction_time_ms`; `DEFAULT_REACTION_TIME_MS` if <3 usable) and starts the main trials. Without the tutorial: calibration + RT test + practice.
- `update_caption()` shows and speaks the caption in lockstep (`narrate_if_changed`; never speaks blank). `render_state()["hide_caption"]` is True in real (non-practice) main trials: neither drawn nor spoken.
  The top-left HUD is drawn only in Test mode (`build_stimuli(win, show_hud=test_mode)`). Narrated strings: under ~15 words, never repeated between screens (it re-speaks on every text change), so the practice intro is spoken once and later trials have an empty caption.
- Neon source has no `latest_frame()`, so the debug preview and video recording are skipped for it (`has_frames = hasattr(gaze, "latest_frame")` in `run_saccade_phase`).

## Gotchas that cost real debugging time

- **OPEN PROBLEM: gaze arrives ~90-100 ms late in real sessions.** Every full session shows `gaze_quality.latency_ms_median` ~91-100 ms (screen-mapped backends, removed, also ran ~44-53 Hz) vs ~16-18 ms in standalone scripts.
  In the raw trace the delay steps up ~65 ms about 6 s in (render loop starts) and stays, rate falls 200 -> ~160/s (~19 % of samples lost): signature of the receiver thread starved by the render loop (same process/GIL;
  the Real-Time API simple device keeps only the newest sample, so the backlog isn't ours). `run_experiment.main` sets `sys.setswitchinterval(0.0005)` as a first test (result unknown). Untested: receive Neon in its own process.
  Consequence: live `flash_during_saccade` is unreliable; post-hoc judging from phone capture times is unaffected.
- **Time**: never use `time.monotonic()`/`time.time()` for event timing on Windows (15.6 ms resolution). Use `src/timebase.py` (`now()`, `unix_now()`); enforced by `test_architecture_rules.py`.
  The phone clock drifts ~1.2 ms/min against the PC (offset re-estimated every 60 s in the background; unverified on hardware that `estimate_time_offset` is safe while streaming).
- **Neon timing (standalone, Oct 2026)**: gaze reaches the PC ~16 ms after the event (p95 ~21, max ~35), ~140-180 Hz effective, gaps up to ~47 ms. Latency is right only with `local = device_ts + offset` from `estimate_time_offset()`
  (the sign is easy to flip: a bogus 414 ms once). `NeonGazeSource` stamps samples with CAPTURE time (`device_unix_s` + offset on the `timebase.now()` timeline; arrival time only as fallback, strictly increasing, clamped to <=150 ms latency),
  so the session must not subtract `gaze_latency_ms` again (`timestamps_are_capture_time`). Check `gaze_quality.latency_ms_median` after each run.
- **Neon geometry**: the scene camera is head-mounted, so head movement shifts gaze x with no eye movement; calibration assumes a still head (chin rest). `NEON_DEVICE_ADDRESS` is pinned to the phone's IP (`None` = auto-discover).
  `NeonGazeSource` has only been tested against fake devices and hand-run scripts, not a full session since the refactors.
- **PySpin**: not a pip package; it lives in PsychoPy's user-packages dir, added to `sys.path` only as a side effect of `import psychopy`. Any module importing `camera.py` must import `psychopy` first (why `tests/conftest.py` does).
  PySpin is built against numpy 1.x; numpy 2.x breaks its import (`_ARRAY_API not found`), so `requirements.txt` pins `numpy<2` plus matching opencv: don't bump.
- **`CAMERA_GAIN_DB` (6.0) is an unvalidated placeholder** (webcam backend): validate on a real IR-lit frame before trusting (the old diagnose_camera.py is recoverable via `git show 0989ed1^:src/eye_tracking/diagnose_camera.py`).
- **TTS narrator (`src/experiment/narrator.py`) is Windows-only SAPI via COM** (`pywin32`, already a PsychoPy dependency). COM objects are apartment-threaded: create AND call `SAPI.SpVoice` on the same thread (`_run()` wraps its loop in `pythoncom.CoInitialize()`/`CoUninitialize()`).
  `Speak()` blocks for the utterance, hence the dedicated thread; `speak()` does not interrupt queued speech, so callers must gate on changed text (`narrate_if_changed`) or utterances queue and lag the screen.
- **`cv2.VideoWriter` fails silently** (`VideoRecorder`, `src/eye_tracking/video_recorder.py`; startup toggle "Record video", default Yes): an unavailable codec or unwritable path makes `.write()` no-op. `start()` checks `isOpened()` and warns on stderr without stopping the session; if a video is empty, read that run's console output first.

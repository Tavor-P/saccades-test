# CLAUDE.md

PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000 design): detect a briefly flashed stimulus during a saccade and during fixation
(the saccade phase's control and far-from-saccade flashes, sorted post hoc) to measure saccadic suppression. Gaze: Pupil Labs Neon (Real-Time API), the only
backend. Participant/experimenter docs: [README.md](README.md).

**`main` is the line-stimulus version** (the grating version is on the `grating` branch):
- Flash = thin horizontal line (Braun, Schutz & Gegenfurtner 2021): full screen width, vertical Gaussian (`LINE_SD_DEG` 0.15), `LINE_OFFSET_DEG` (2 deg) above/below the saccade TARGET,
  drawn `LINE_DURATION_FRAMES` (1) frame. Answer UP (above) / DOWN (below).
- `Side.ABOVE/BELOW`. `TrialResult` fields `line_side` / `answer_side` (hold 'above'/'below'), `flash_shown`, `flash_unix_ms` match the `trials.csv` columns of the same names.
  A trials.csv with the grating-era columns (`orientation` / `response_orientation` / `grating_shown` / `grating_shown_at_unix_ms`, 'vertical'/'horizontal' values) loads via
  `logger._CANONICAL_NAMES` + `_LEGACY_SIDES`; `read_trial_rows` yields rows under those old names. The render-state key is `"line"` (x/y = line position, `side` = above/below); session attrs `_line_visible` etc.
- Geometry: the startup dialog asks for screen height and viewing distance; `geometry.Geometry` (built once in `main`) gives the line's offset/SD in screen-height units, passed to
  `ExperimentSession`/`TutorialSession` (`trial_mechanics.line_position`) and `build_stimuli(geometry=)`. `Geometry.default()` = the placeholder constants
  (`VIEWING_DISTANCE_CM`/`SCREEN_HEIGHT_CM`). Used geometry is in `session.json` `settings` and the `analyze_session` header.
- **Contrasts are a guess**: `SACCADE_FIXED_CONTRASTS` 4/6/8 % (dialog min 4 / max 8, 3 levels), `HARDWARE_CONTRAST_FLOOR` 0.78 % (README "Known limitations").
- Rolling RT = `robust_average_ms` over the last `RT_AVERAGE_ROLLING_WINDOW` (12) latencies; fewer than `RT_MIN_SAMPLES_FOR_AVERAGE` (3) plausible samples keep the previous value
  (initially `DEFAULT_REACTION_TIME_MS`).
- **Return jump** (dialog "Return jump", default `RETURN_JUMP_DEFAULT` on): after the out saccade the target jumps back to the start symbol (`RETURN_JUMP_TRIGGER` "arrival" (default)
  or "onset"; `RETURN_JUMP_LANDING_TIMEOUT_MS` forces it), so each trial has an out and a return saccade but still ONE flash: `RETURN_LEG_FLASH_FRACTION` 0.8 on the return leg (timed
  from the jump frame, spread `RETURN_FLASH_SPREAD_SD_MS` 40 = a guess, measure it), else the out leg. trials.csv: `flash_leg`, `out_arrival_ms_after_go_cue`, signed `return_reaction_ms`
  (negative = left before the jump), `left_target_before_jump`. `analyze_session` gates each flash against its own leg's saccade (return must move back >= half the out amplitude),
  reports out / return / pooled indices and a return-latency report; summary.json schema 3 (adds `analysis_thresholds`) pools only one run kind (`single` | `return_jump/<trigger>`); `compare` refuses mixed kinds.
  Simulator `ParticipantProfile.return_mode="learned"` = predictive returns. Tests: `tests/ai-tests/test_return_jump.py`, `test_return_analysis.py` (`sim_runs.py` writes simulated runs as folders). Sim (150 trials): ~36 valid in-saccade flashes vs ~20 without the jump; the gain needs the narrow 40 ms spread. Unvalidated on glasses.

## Rules (read first)

- **Run Python only via the wrapper**: `.\scripts\py.ps1 <args>` (PowerShell) or `scripts/py.sh <args>` (Git Bash). It finds PsychoPy's Python (`%LOCALAPPDATA%\Programs\PsychoPy\python.exe`,
  or `$env:PSYCHOPY_PYTHON`) and runs from the repo root; from cmd use `powershell -ExecutionPolicy Bypass -File scripts\py.ps1 ...`. Plain `python` lacks deps and `src`/`include`.
- **Never launch the experiment or dashboard** (`run_experiment` opens a fullscreen window on the user's screen). Use the headless simulator or tests. Real-glasses runs are the human's.
  The Neon is sometimes reachable for read-only checks: ask first.
- Don't merge, push or switch branches unless asked. **Before any push/merge the user asks for, update this file first** (Repo map, Commands, Gotchas, stale claims, test count),
  run `tests/test_docs.py`, commit it by name, and say what you changed.
- **Other AI sessions edit this folder concurrently**: `git status` / `git log -5` first; never revert or tidy changes you didn't make; stage and commit only your own files by name
  (never `git add -A` / `commit -a`); don't touch other sessions' files, tell the user instead.
- PowerShell 5.1 has no `&&`; keep each file's line endings (CRLF). `.gitignore` has `data/`, which matches any `data` folder (tests: use `tests/fixtures/`).
- `data/` is git-ignored runtime output and participant data is personal data: never commit or print it, never write to it from tests/simulations, never copy real sessions into `tests/fixtures/`.
- Report test results truthfully (ran / skipped). Suite: 382 tests pass, none skipped, ~30-35 s (pytest-xdist: pytest.ini runs 6 workers, `--dist loadgroup`; was ~2 min serial).
  Keep it fast: share expensive setup (module-scoped fixtures), keep simulated runs small, and do not render real plots or run 1000 bootstrap refits where a test only needs a file or a CI to exist
  (see tests/ai-tests/conftest.py: PNGs are stubbed except in `REAL_PLOT_TESTS`, `suppression_index` runs 300 resamples). Tests must not share paths or global state (xdist): use `tmp_path`.
  A module-scoped fixture is built once per worker, and importing `run_experiment` costs seconds: tests sharing an expensive fixture are grouped onto one worker (`pytest_collection_modifyitems`
  in tests/ai-tests/conftest.py), and no test module imports `run_experiment` at the top (it is imported inside the test).

## Commands (all through the wrapper)

```
.\scripts\py.ps1 -m pip install -r requirements.txt
.\scripts\py.ps1 -m pytest -q                                   # full suite, parallel; one test or a debugger: -n0 tests/ai-tests/test_session.py::name -v (no worker start-up)
.\scripts\py.ps1 -m ruff check .                                # lint (ruff.toml: E F W I B UP); never `ruff format` (whole-repo reformat)
.\scripts\py.ps1 -m mypy                                        # types for src/ + include/ (mypy.ini); both kept at 0 findings by tests/test_lint.py
.\scripts\py.ps1 -m src.experiment.simulation                   # headless simulated trial timeline (AI-safe)
.\scripts\py.ps1 tests/ai-tests/run_sim.py --help               # --scenario --set --profile --sweep --sweep-profile --seeds --timeline --json --real --list
.\scripts\py.ps1 -m src.experiment.run_experiment               # REAL experiment (human only)
.\scripts\py.ps1 -m src.dashboard.app                           # dashboard, localhost:5000 (human)
.\scripts\py.ps1 -m src.experiment.analyze_session [session] [--data-dir D]   # newest by default; writes <run>/analysis/
.\scripts\py.ps1 -m src.experiment.analyze_participant <id> [--data-dir D] [--include-test]   # pools compatible runs -> data/<id>/analysis/
.\scripts\py.ps1 -m src.experiment.participants set P01 TBI [--note x] | remove P01 | list    # data/participants.json
.\scripts\py.ps1 -m src.experiment.compare "P01,P02" "group:control" [--include-test] [--out D]   # -> data/comparisons/
.\scripts\py.ps1 -m src.experiment.study_summary [--include-test]   # data/study_summary.csv + .txt
.\scripts\py.ps1 -m src.experiment.import_glasses_video <run-folder> <exported.mp4>
```
Double-click launchers: `run_experiment.bat` in the repo root; `analyze_latest.bat`, `analyze_participant.bat`, `compare.bat`, `study_summary.bat`, `run_tests.bat` in `launchers/`.

## Status and next steps (2026-11-24)

- **Goal**: grant to test saccadic suppression in TBI vs non-TBI; present as a *feasibility pilot (n=1)*, never a group result.
- **Code-quality pass 2026-11-24** (behaviour unchanged, sim output byte-identical): session split (`trial_legs`, `session_config`, `trial_record`), line naming, IO cleanup (`session_files`, `textio`),
  run/analyze split (`rendering`, analysis modules, `analysis_constants`, summary.json schema 3), test cleanup, ms/s helpers + named constants, ruff + mypy at zero findings, docs deduped against README.
- **Shipped setup**: flash times on a bell curve centred on the average RT (SD `FLASH_SPREAD_SD_MS` 80, dialog "Flash timing spread"), cut to 0-500 ms after the go cue; controls at avg RT +400/+500 +/-10 ms;
  fixed saccade contrasts; plain Neon, chin rest. Onboarding: calibration -> quiz -> 8 practice trials seeding the RT average. Every trial waits for SPACE.
- **Simulation 2026-11-22** (code not kept): shifting the bell curve's centre earlier than the average RT gains almost nothing in the 50 ms before onset (spread is ~110 ms SD anyway; beyond -20 ms
  it costs in-saccade flashes). A slow first-time starter costs only ~1 usable flash (8 practice trials + the window of 12 absorb it). The centre stays on the average RT.
- **Nothing has run on glasses since the last refactors**: line rendering, quiz/captions, every contrast level, results layout, phone video + sync flashes, head monitor and the new analysis/dashboard pages
  are tested only on simulator, fake devices and synthetic data.
- **Known problems** (numbers in README "Known limitations"): (1) gaze delay is ~13-20 ms now (Gotchas); live `flash_during_saccade` is unvalidated, post-hoc analysis is the judgement. (2) RT SD ~60-80 ms: few flashes land near onset.
  (3) Geometry is the 26 cm / 50 cm placeholder until the user measures their monitor (dialog says so).
- **Next**: (0) on glasses, measure the real return-latency spread (`analyze_session` return-latency report) and set `RETURN_FLASH_SPREAD_SD_MS` from it; (1) Test run, then several 150-trial Real runs per person on glasses; (2) measure real screen geometry; (3) confirm gaze latency stays ~15 ms on long real runs; (4) one-page grant figure; (5) clinical metadata fields (needs IRB).

## Repo map

Paths are checked by `tests/test_docs.py` - keep them real.

| Path | What it is |
|---|---|
| `include/experiment/constants.py` | every tunable experiment constant (read the comment first) |
| `include/experiment/types.py` | `TrialResult` (one CSV row), `TrialSpec`, `Target`, `Side` |
| `include/experiment/participant_text.py` | every string the participant reads/hears (<=15 words) |
| `include/eye_tracking/constants.py` | gaze constants incl. `GAZE_DEAD_ZONE`, all `NEON_*` |
| `include/eye_tracking/interfaces.py`, `types.py` | `GazeSource` (the only gaze interface sessions use); `GazeZone`, `GazeSample` |
| `src/experiment/session.py` | `ExperimentSession`: saccade phase state machine (core logic) |
| `src/experiment/session_config.py` | `SessionConfig` (run settings) + `TutorialHandoff` (tutorial calibration/RT samples): `ExperimentSession(gaze, logger, clock, config=, handoff=)` |
| `src/experiment/trial_legs.py` | `TrialLegs` (out onset/landing, return jump, return onset/landing per trial) + `SampleFeed` (new samples, capture times): shared by session and tutorial |
| `src/experiment/trial_record.py` | `saccade_trial_result()`: a finished main trial's `TrialResult` row |
| `src/experiment/tutorial_session.py` | `TutorialSession`: intro -> calibration -> quiz -> 8 practice trials |
| `src/experiment/run_experiment.py` | entry point: `main`, phase loops, dialog -> session wiring (`SaccadePhaseOptions`); re-exports the rendering names tests patch |
| `src/experiment/rendering.py` | PsychoPy window, `build_stimuli`, `apply_render_state`/`draw_all`, captions + narration, `FramePresenter`, sync flashes; with run_experiment the only PsychoPy modules |
| `src/experiment/startup_dialog.py`, `startup_model.py` | PyQt6 dialog; Qt-free model (`StartupSettings`, `validate`, `HELP_TEXTS`) |
| `src/experiment/geometry.py` | `Geometry`: screen height + viewing distance -> degrees and the line's size in screen-height units |
| `src/experiment/pausable_clock.py` | `PausableClock`: the injected time source |
| `src/experiment/settings.py` | contrast range helpers + remembered dialog values (`data/settings.json`) |
| `src/experiment/scoring.py` | `score_outcome`, `is_valid_for_saccadic_analysis` (the validity gate) |
| `src/experiment/trial_mechanics.py` | `OnsetDetector`, `ShuffledBag`, `line_position`, `robust_average_ms`, `spread_flash_delay_ms`, `RollingReactionTime` |
| `src/experiment/calibration.py` | `CalibrationRounds` (3 points x 3 rounds), `average_calibration_rounds` |
| `src/experiment/space_input.py` | `CalibrationLatch`, `NarrationKeyGate`, `SpaceTally`, `deliver_space`, `leaves_complete_screen` |
| `src/timebase.py` | the ONE time source: `now()` = `perf_counter()`; `unix_now()`, `unix_ms()`; `ms_to_s()`/`s_to_ms()` (the only place the factor 1000 is written for time) |
| `src/experiment/trial_factory.py` | trial sequences; every trial flashes |
| `src/experiment/logger.py` | `ResultLogger`: `TRIAL_COLUMNS`, `DROPPED_FROM_CSV`, `trial_from_row`, `read_trial_rows`/`read_trials` (plain-named + grating-era columns) |
| `src/experiment/session_files.py` | run folders: `REPO_ROOT`/`DATA_DIR` (defined once), `read_json`, `sanitize_participant_id`, `create_session_folder`, `list_sessions`/`find_session`/`newest_session` |
| `src/textio.py` | `read_text`/`read_csv`: every reader of our own files (UTF-8, BOM dropped, cp1252 fallback for old Windows-locale files) |
| `src/experiment/glasses_video.py`, `import_glasses_video.py` | sync-flash detection/offset; CLI that attaches the exported video |
| `src/experiment/flash_gates.py` | pre-specified gates (Braun 2021): `saccade`, `fixation`, else `excluded`; controls from `CONTROL_OFFSETS_MS` |
| `src/experiment/analyze_session.py` | post-hoc analysis from the raw gaze trace: `analyze_run()` = pipeline without writing files, the CLI; re-exports the names below |
| `include/experiment/analysis_constants.py` | every analysis threshold (detection, windows, gates, index), written into each report and summary.json `analysis_thresholds` |
| `src/experiment/saccade_detection.py`, `suppression_stats.py`, `leg_gating.py` | saccades + flash placement; `summarize`/`suppression_index`; return-jump per-leg gating and return latency |
| `src/experiment/analysis_report.py`, `analysis_output.py` | report text (`format_*`, `thresholds_record`): report.txt is human-first (QUALITY/RESULT/AMOUNT top block, fixed sections, only non-default thresholds); machines read suppression.json / flashes.csv. Plot (red near-saccade bars, then blue away bars) and per-flash CSV |
| `src/experiment/run_quality.py` | quality rules for the report top block and the dashboard quality lights (one place) |
| `src/experiment/analyze_participant.py` | pooled per-person index (equal `saccade_contrasts_pct` + geometry + run kind); summary.json (`SUMMARY_KEYS`, `schema_version` 3, per-leg `by_leg`, `analysis_thresholds`) |
| `src/experiment/participants.py` | group labels (`data/participants.json`): `set_label`, `resolve_set` ("P01,group:TBI,all"), `everyone_else` |
| `src/experiment/compare.py` | `compare()` -> `ComparisonResult`: 1v1 (bootstrap CI), group vs group (Mann-Whitney), 1-vs-group (percentile, no p) |
| `src/experiment/simulation.py` | headless simulator |
| `src/experiment/narrator.py`, `study_summary.py`, `run_summary.py`, `src/eye_tracking/gaze_factory.py` | SAPI TTS; study CSV; end-of-run summary; `build_gaze_source()` |
| `src/eye_tracking/neon_source.py`, `head_monitor.py` | `NeonGazeSource` (stream + quality stats + IMU thread); `HeadMotionTracker` (flags only) |
| `src/eye_tracking/neon_address.py` | start-up check: dialog address -> auto-discovery -> PC's default gateway (hotspot); `failure_message` |
| `src/eye_tracking/gaze_mapping.py` | `GazeMapper`: ratio -> calibrated position / zone / smoothing |
| `src/eye_tracking/gaze_trace.py` | raw gaze recording + `ReplayGazeSource` |
| `src/dashboard/app.py` | Flask dashboard (separate process, `127.0.0.1` only; `data_access.py` reads/writes `data/`) |
| `src/dashboard/results.py` | results pages: freshness check, quality lights, `FriendlyError` |
| `src/dashboard/comparison.py` | compare / labels / study pages; always passes `data_dir=results._data_dir()` |
| `tests/conftest.py`, `tests/helpers.py` | `FakeGazeSource`, `fake_time`, `NullLogger` |
| `tests/ai-tests/README.md`, `run_sim.py` | headless-simulation guide and CLI |
| `tests/ai-tests/find_bugs.py` | randomized campaign over the simulator; prints a repro command per finding |
| `tests/ai-tests/invariants.py` | `check_session()`: contracts every simulated session must satisfy |
| `tests/ai-tests/synthetic_runs.py` | `write_run()`: anonymous synthetic run folders for analysis/dashboard tests |
| `tests/fixtures/neon_trace_sample.csv` | a real recorded Neon gaze trace |
| `scripts/py.ps1` | interpreter wrapper |

## How to verify a change (cheapest first)

1. **Session/scoring/logger logic**: add or update a test (`FakeGazeSource` + `fake_time`, or the simulator), run pytest. A previously passing test that now fails means behaviour changed: decide if intended before editing it.
2. **Whole-session behaviour or a timing constant**: `run_sim.py` (`--set NAME=VALUE`, `--sweep NAME=a,b,c --seeds 5`, `--timeline`); trust relative comparisons only. After session/scoring changes also run `tests/ai-tests/find_bugs.py --sessions 300`.
3. **Gaze source**: `tests/test_neon_source.py` / `tests/test_gaze_trace.py` patterns (fake device, or replay the fixture via `ReplayGazeSource`).
4. **Needs real eyes/glasses/screen**: write the measurement script and ask the human; don't guess numbers.
- Enforced by tests: lint + types (`test_lint.py`: ruff and mypy report nothing), schema sync (`test_schema_sync.py`), constant relations (`test_constants_sanity.py`), architecture (`test_architecture_rules.py`: no PsychoPy in session logic, no `time.monotonic()`/`time.time()` in `src/` outside
  `timebase.py`, lazy backend imports, no logic in `include/`), known-bug reinjection (`test_bug_injection.py`).

## Recipes

- **Add a trials.csv column / `TrialResult` field**: field in `include/experiment/types.py` (default `None`) -> one `Column(...)` in `TRIAL_COLUMNS` (`logger.py`) or a reasoned entry in `DROPPED_FROM_CSV` -> `trial_from_row`
  -> set it in `trial_record.saccade_trial_result` (called from `session.py` `_finish_trial`) -> test (see `tests/ai-tests/test_tracking_quality.py`). Old CSVs must still load.
- **Change a tuning constant**: edit in `include/**/constants.py` (read its comment), then `--sweep` in the simulator and pytest.
- **Add a gaze backend**: implement `GazeSource`, select it in `build_gaze_source()` (lazy import), reuse `GazeMapper`.
- **Reproduce a bug without hardware**: replay a recorded trace via `ReplayGazeSource`, or use a `ParticipantProfile` in the simulator.

## On-disk contracts

- Folder layout (`data/<participant>/<YYYY-MM-DD_HH-MM-SS>[_test]/`, pooled/comparison/study files): README "Output". The head-motion CSV is the IMU one.
- `trials.csv`: one row per trial (`phase` column: saccade / rt_test; old runs also baseline, read via `PHASE_WORDS`), schema `TRIAL_COLUMNS` with plain names. `DROPPED_FROM_CSV` lists fields kept out and why.
- `session.json` sections: listed in README "Output". The results phase summary includes `rt_samples_total/_in_window/_dropped_in_window`.
- Readers (`read_trial_rows`/`read_trials`, `data_access`, `analyze_session`, replay) take run folders only: the loose `data/results_<ts>.csv` layout was removed on 2026-11-24 (none existed on disk).
  Kept for old run folders, each with a comment saying who needs it: `_LEGACY_SIDES`, the baseline `PHASE_WORDS`, `_CANONICAL_NAMES` + the `trial_index` branch of `read_trial_rows`, and traces holding
  screen-ratio x `NEON_SCENE_WIDTH` in `x_px` (why that constant stays).
- Files: write everything with `encoding="utf-8"` (csv also `newline=""`); read through `src/textio.py` and `session_files.read_json` (None for a missing/damaged file: callers use `or {}`, or raise
  where silence would lose data, e.g. `save_participant_info`). `DATA_DIR` lives in `session_files`; modules import it under their own name (tests patch it per module).
- The dashboard is a separate Flask process sharing only files with the experiment; deleting a run removes its whole folder (glasses video included) and the participant's pooled analysis folder.

## Architecture invariants

- **`include/` = dataclasses, enums, constants only; no logic. `src/` mirrors it and holds all behaviour.**
- **Sessions are PsychoPy-free state machines.** `ExperimentSession` exposes `on_space()`/`on_response_key()`/`tick()`/`render_state()`; the `run_experiment.py` frame loops (shared `FramePresenter` from `rendering.py`, `deliver_keys`)
  call `tick()` + `render_state()` once per frame and apply the plain dict (`apply_render_state()`/`draw_all()`). Test mode is a runtime choice (dialog Mode; Real needs a participant ID).
  A run = tutorial (optional) -> saccade phase (`SACCADE_FIXED_TRIALS_TEST` 30 / `_REAL` 150) -> a console line naming the `analyze_session` command. No pause menu, rest break, session cap or recalibration.
- **Gaze is behind `GazeSource`.** `NeonGazeSource` polls `receive_gaze_datum()` on a thread, turns scene gaze x / `NEON_SCENE_WIDTH` into a 0-1 ratio; `GazeMapper` does calibration fit, dead zone, smoothing.
  `build_gaze_source()` imports it lazily, so session logic and the simulator run without `pupil-labs-realtime-api`.
- **Shared logic lives once**: `OnsetDetector`, `is_valid_for_saccadic_analysis()`, `RollingReactionTime`, `CalibrationLatch`, `TrialLegs`/`SampleFeed` (trial gaze legs, real and practice). Reuse them, don't hand-roll.
- **Test doubles**: `FakeGazeSource` (set `.zone`/`.face_found`/`.position`; `position` must move with `.zone`) and `fake_time` (patches `timebase._clock`, never the global `time`).
- **Simulator**: `simulation.py` runs the real `ExperimentSession` against `SimulatedParticipant` (reads only `render_state()`), `SimulatedClock`, `SimulatedGazeSource`; no PsychoPy, no files,
  deterministic per `seed`. `ParticipantProfile` = knobs (RT, thresholds incl. suppression, tracker noise/latency, dropouts, `head_spikes`); `simulate_session()` returns `flash_truth`.
- **Tracking-quality gate** (behaviour and thresholds: README "Head movement"): a flash is `tracking_ok=False` if tracking dropped within `TRACKING_LOSS_WINDOW_MS` of it, or head speed exceeded `HEAD_MAX_SPEED_DPS` there
  (`ExperimentSession._head_speed_spike_near_flash`, `head_flagged_trials`); drift warnings via `pop_head_warning`. The IMU is polled at `HEAD_POLL_HZ` on its own thread; no IMU -> off with one warning
  (`head_motion.status` = `unavailable (...)`). **Unvalidated on glasses**: thresholds, rate, IMU-vs-gaze clock alignment.
- **Flash timing is open-loop (`"scheduled"`)**: each trial draws from a bag of `TIMING_OFFSETS_MS` + `CONTROL_OFFSETS_MS`; an offset label is one of 9 equal-probability slices of the normal curve (mean avg RT,
  SD `FLASH_SPREAD_SD_MS`, cut to `FLASH_WINDOW_MIN_MS`..`FLASH_WINDOW_MAX_MS`), drawn by inverse CDF (`trial_mechanics.spread_flash_delay_ms`). The real delay is `scheduled_delay_ms`; `timing_offset_ms` keeps the label.
  The live classifier measures `reaction_latency_ms` and computes `flash_during_saccade` post hoc (`True`/`False`/`None`; `None` = landing never confirmed; only `True` is valid). The real judgement is `analyze_session`.
- **Onset = displacement rule, not a zone crossing**: `OnsetDetector` fires when `GazeSample.position` (0=dot..1=cross) is > `SACCADE_ONSET_DISPLACEMENT` (0.05) from the fixation baseline for
  `SACCADE_ONSET_STABILITY_MS` (10) of capture time; `since` = first sample beyond it. Watch the false-alarm rate on head-still recordings. `GAZE_DEAD_ZONE` matters only for landing/zones.

## SPACE and input rules (`space_input.py`, `tests/ai-tests/test_space_input.py`)

SPACE presses are never silently dropped; every ignored one is counted in `session.space_tally` (phase + reason) and saved as `ignored_space_presses`.
- Calibration: one press suffices; if `average_recent_ratio()` is still `None` it is latched and `tick()` completes the step when the window is full ("TRACKING LOST?" after 3 s).
- Trials: accepted in `AWAITING_SPACE`; the swap follows `FOREPERIOD_MIN_MS`-`FOREPERIOD_MAX_MS` (500-1000 ms). Ignored during the foreperiod; during a trial ignored until answered, then carried to start the next.
- Same-frame repeats collapse to one; a press before a phase's first screen is discarded; the press in the frame COMPLETE first appears doesn't dismiss it.
- Tutorial: `run_tutorial_phase` withholds key/tick forwarding while `narrator.is_speaking` and delivers one held SPACE afterwards.

## Tutorial hand-off and narration

- Tutorial calibration ratios and RT samples (`precomputed_calibration_ratios`, `precomputed_rt_samples_ms`) go to `run_saccade_phase`, which then skips calibration/RT test/practice and seeds `RollingReactionTime`.
- `update_caption()` shows and speaks in lockstep (`narrate_if_changed`); it re-speaks on every text change, so narrated strings stay under ~15 words and are never repeated. `hide_caption` is True in real main trials; the HUD is Test-mode only.

## Gotchas that cost real debugging time

- **Gaze latency (was ~90-100 ms, now ~13-20 ms)**: in Oct 2026 real sessions showed ~95 ms (`gaze_quality.latency_ms_median`) with the rate falling 200 -> ~160/s once the render loop started: the receiver thread
  starved by the render loop (same process/GIL). `run_experiment.main` has since set `sys.setswitchinterval(0.0005)` (5f5a0c6, 2026-10-22); every run on disk after it (2026-11-01..11-23) shows a median of 12.5-19.8 ms
  at ~199/s. Keep checking `latency_ms_median` per run; if it climbs again, next step is receiving Neon in its own process.
- **Time**: never use `time.monotonic()`/`time.time()` for event timing on Windows (15.6 ms resolution). Use `src/timebase.py` (`now()`, `unix_now()`); enforced by `test_architecture_rules.py`, which also forbids `datetime.now()`/`utcnow()` in `src/` except the listed human-readable stamps. Convert ms<->s with `ms_to_s()`/`s_to_ms()` and name variables `_ms`/`_s`, never a bare `/ 1000`.
  The phone clock drifts ~1.2 ms/min against the PC (offset re-estimated every `NEON_CLOCK_REESTIMATE_S`).
- **Neon timing**: latency is right only with `local = device_ts + offset` from `estimate_time_offset()` (the sign is easy to flip). `NeonGazeSource` stamps samples with CAPTURE time
  (`device_unix_s` + offset on the `timebase.now()` timeline; arrival time only as fallback, strictly increasing, clamped to <=`NEON_MAX_PLAUSIBLE_LATENCY_MS`), so the session must not subtract `gaze_latency_ms` again
  (`timestamps_are_capture_time`). Check `gaze_quality.latency_ms_median` after each run. If the start-up offset estimate fails (or exceeds `NEON_CLOCK_ESTIMATE_TIMEOUT_S`)
  a WARNING is printed and `gaze_quality.clock_offset_status` = `failed (...)` (else `ok`): stamps fall back to arrival time and `analyze_session` needs `--offset-ms`.
- **Neon geometry**: the scene camera is head-mounted, so head movement shifts gaze x with no eye movement; calibration assumes a still head (chin rest). The address comes from the dialog's "Neon address" field (first-time default `NEON_DEVICE_ADDRESS`, remembered in `data/settings.json`).
- **Neon start-up check** (search order: README "First-time setup" step 4): `run_experiment.main` calls `find_neon_or_exit` BEFORE the run folder and window; it probes GET `/api/status` (`NEON_PROBE_TIMEOUT_S`),
  discovery takes `NEON_PREFLIGHT_DISCOVERY_S`. Nothing answers -> plain message on stderr + message box, exit 2.
  The found address goes to `NeonGazeSource` and `session.json` `hardware.neon_device` (None until `set_neon_device()`; the `NEON_DEVICE_ADDRESS` default is not recorded). `NeonGazeSource.start()` raises one `NeonConnectionError` (no aiohttp chain) and swallows the half-built `Device.__del__` noise.
  The probe also requires a connected gaze `Sensor` when the status lists sensors; a malformed reply never crashes the search. The phone's self-reported IP
  (time echo + gaze stream use it) is saved as `hardware.neon_phone_ip` (warned if it differs). A `NeonConnectionError` later in the run -> window closes, "Lost the Neon glasses" box, exit 2.
- **numpy/opencv pins**: `requirements.txt` keeps `numpy<2` plus matching opencv (`glasses_video.py` uses cv2). Bumping is untested: rerun the suite if you do.
- **TTS narrator is Windows-only SAPI via COM** (`pywin32`). COM objects are apartment-threaded: create AND call `SAPI.SpVoice` on the same thread (`_run()` wraps its loop in `pythoncom.CoInitialize()`/`CoUninitialize()`).
  `Speak()` blocks for the utterance, hence the dedicated thread; `speak()` does not interrupt queued speech, so callers must gate on changed text (`narrate_if_changed`) or utterances queue and lag the screen.

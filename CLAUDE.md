# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000):
detect a briefly-flashed grating's orientation, once during a saccade and
once during fixation (baseline), to measure saccadic suppression. Gaze is
tracked via Pupil Labs Neon glasses (default on branch
`feature/neon-glasses`, via the Real-Time API) or, as an alternative backend
picked in the startup dialog, a FLIR Blackfly S camera (Spinnaker SDK /
`PySpin`) + MediaPipe iris landmarks. See [README.md](README.md) for the full participant-facing
walkthrough (controls, output files, troubleshooting).

## Start here (read this first)

- **Run everything through the wrapper**: `.\scripts\py.ps1 <python args>` (PowerShell) or
  `scripts/py.sh <python args>` (Git Bash). It finds PsychoPy's bundled Python and runs from the
  project root. Plain `python` will not work (missing deps, `src`/`include` not importable).
- **Never launch the experiment or dashboard yourself** (`run_experiment` opens a fullscreen
  PsychoPy window on the user's screen). To exercise experiment logic use the **headless simulator**
  (below) or the tests. Real-glasses runs are for the human.
- **Branch `feature/neon-glasses`** is the Windows + Pupil Labs Neon port. Don't merge, push or
  switch branches unless asked.
- **Other AI sessions may be editing this same folder at the same time.** See "Working rules".
- Before changing anything, find the file in the **Repo map**, and skim the **Gotchas** at the end.

## Commands

All via the wrapper (PowerShell shown; from Git Bash use `scripts/py.sh` with the same arguments):

```
.\scripts\py.ps1 -m pip install -r requirements.txt            # install deps (into PsychoPy's Python)
.\scripts\py.ps1 -m pytest -q                                   # full test suite (~40 s, no camera/window needed)
.\scripts\py.ps1 -m pytest tests/ai-tests/test_session.py::test_name -v  # one test
.\scripts\py.ps1 -m src.experiment.simulation                   # HEADLESS simulated trial, event timeline (safe for AI)
.\scripts\py.ps1 tests/ai-tests/run_sim.py --help               # headless simulation runner (scenarios, sweeps, JSON)
.\scripts\py.ps1 -m src.experiment.run_experiment               # REAL experiment (opens a window - human only)
.\scripts\py.ps1 -m src.dashboard.app                           # session dashboard at localhost:5000
.\scripts\py.ps1 -m src.eye_tracking.neon_fixation_test [secs]  # Neon jitter check (glasses on; human)
.\scripts\py.ps1 -m src.eye_tracking.neon_gaze_view             # live screen-gaze dot via AprilTags (human)
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
| `include/experiment/types.py` | `TrialResult` (one CSV row), `TrialSpec`, `Target`, `Orientation` |
| `include/eye_tracking/constants.py` | gaze constants incl. `GAZE_DEAD_ZONE`, all `NEON_*` settings |
| `include/eye_tracking/interfaces.py` | the `GazeSource` interface the experiment talks to |
| `include/eye_tracking/types.py` | `GazeZone`, `GazeSample` |
| `src/experiment/session.py` | `ExperimentSession`: the saccade phase state machine (the core logic) |
| `src/experiment/presaccade_session.py` | `PresaccadeSession`: baseline (fixation) phase |
| `src/experiment/tutorial_session.py` | `TutorialSession`: narrated first-timer on-ramp |
| `src/experiment/run_experiment.py` | entry point + PsychoPy rendering; the only PsychoPy-dependent module |
| `src/experiment/scoring.py` | `score_outcome`, `is_valid_for_saccadic_analysis` (the validity gate) |
| `src/experiment/trial_mechanics.py` | `OnsetDetector` (displacement rule), `ShuffledBag`, `zone_for`, `average_ms`, `robust_average_ms` (shared helpers; the robust one drops anticipations/outliers then takes the median) |
| `src/experiment/calibration.py` | `average_calibration_rounds`, `validate_calibration` / `CalibrationCheck` / `fit_residual` (monotonic, span, noise, residual checks; not yet wired into `session.py`) |
| `src/experiment/session_schedule.py` | `phase_order` (counterbalanced by participant ID), rest-break and wall-clock session-cap schedule (`SessionSchedule`) |
| `src/timebase.py` | the ONE time source: `now()` = `time.perf_counter()`; wall-clock via an edge-synchronised anchor (`unix_now()`, `perf_to_unix`) |
| `src/experiment/trial_factory.py` | builds trial sequences |
| `src/experiment/zest.py` | adaptive contrast staircase |
| `src/experiment/logger.py` | `ResultLogger`: CSV + `_meta.json` (git SHA, constants snapshot, per-phase threshold summary) + side-file paths |
| `src/experiment/results_graph.py` | end-of-run comparison figure |
| `src/experiment/simulation.py` | headless simulator (simulated participant, clock, gaze) |
| `src/eye_tracking/neon_source.py` | `NeonGazeSource` (default backend): Real-Time API stream + quality stats |
| `src/eye_tracking/neon_screen_source.py` | Neon + screen-marker backend (head-compensated gaze) |
| `src/eye_tracking/screen_markers.py` | corner AprilTag layout/rendering: physical size (`SCREEN_MARKER_WIDTH_CM`), mean-matched dim levels; `marker_verts` order is what the screen-gaze library expects (BR,BL,TL,TR), not TL-clockwise |
| `src/eye_tracking/screen_outline_mapper.py` | tag-free screen locator (`ScreenOutlineMapper`, same interface as `ScreenGazeMapper`); backend "screen outline (no tags)" via `neon_outline_source.py` |
| `src/eye_tracking/hybrid_screen_mapper.py` | `HybridScreenMapper`: corner tags when located, outline mapper as fallback; backend "markers and outline" via `neon_hybrid_source.py` |
| `src/eye_tracking/screen_detect_compare.py` | offline detector report on a scene video/frames (jitter, optional AprilTag comparison) |
| `src/eye_tracking/gaze_mapping.py` | `GazeMapper`: ratio -> calibrated position / zone / smoothing |
| `src/eye_tracking/gaze_factory.py` | `build_gaze_source()` backend picker (lazy imports) |
| `src/eye_tracking/gaze_trace.py` | raw gaze recording + `ReplayGazeSource` |
| `src/eye_tracking/webcam_source.py` | Blackfly S + MediaPipe backend (alternative) |
| `src/dashboard/app.py` | Flask dashboard (separate process, reads `data/`) |
| `tests/conftest.py` | `FakeGazeSource`, `fake_time` fixture |
| `tests/ai-tests/README.md` | headless-simulation guide for AI agents |
| `tests/ai-tests/run_sim.py` | simulation CLI (`--set`, `--sweep`, `--timeline`, `--json`) |
| `tests/ai-tests/find_bugs.py` | randomized bug-hunting campaign over the simulator; prints a repro command per finding |
| `tests/ai-tests/invariants.py` | `check_session()`: contract checks every simulated session must satisfy |
| `tests/fixtures/neon_trace_sample.csv` | a real recorded Neon gaze trace |
| `tests/ai-tests/analyze_onset_threshold.py` | offline onset-rule study on recorded traces (lead/delay/false triggers per threshold) |
| `scripts/py.ps1` | interpreter wrapper (see Start here) |

`data/` (results, traces, graphs) is git-ignored runtime output: never commit it, and don't write to
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
(~360 tests, ~30 s, 2 skipped when PySpin/mediapipe are absent).

**Bug-hunting tests** (`tests/ai-tests/`): `find_bugs.py` fuzzes the real session with random/hostile
participants and checks `invariants.py` (run `find_bugs.py --sessions 300` after changing session,
scoring or ZEST logic; a finding prints a `run_sim.py` repro line). `test_bug_injection.py` re-injects
known bugs to prove the checks and regression tests still catch them. Also enforced by tests:
TrialResult/CSV/dashboard schema sync (`test_schema_sync.py`), constant relations
(`test_constants_sanity.py`), live-vs-graph validity lockstep (`test_validity_lockstep.py`) and the
architecture rules in this file (`test_architecture_rules.py`: no PsychoPy in session logic, no direct
`time.monotonic()`/`time.time()` anywhere in `src/` outside `timebase.py`, lazy backend imports, no logic in `include/`).

## Recipes

- **Add a column to the results CSV / a field to `TrialResult`**: field in `include/experiment/types.py`
  (default `None`) -> `FIELDNAMES` + `ResultLogger.log` in `src/experiment/logger.py` -> parse with
  `row.get(...)` in `src/dashboard/data_access.py` (old CSVs must still load) -> set it where the
  `TrialResult` is built in `session.py` -> test (see `tests/ai-tests/test_tracking_quality.py`).
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
  Files are LF in the working tree (git may warn about CRLF); keep line endings as they are.
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
about PsychoPy — `run_experiment.py`'s frame loop calls `tick()` +
`render_state()` once per rendered frame and applies the returned plain dict
to PsychoPy stimuli (`apply_render_state()`/`draw_all()`). Both phases share
one `ResultLogger` (one CSV, distinguished by a `phase` column) but run
independent `ZestStaircase` instances so each condition converges on its own
threshold. Test mode (shortened trial counts, saccade phase first for smoke
testing) is a runtime decision, not a constant: `run_experiment.py`'s
`_resolve_test_mode()` reads the startup dialog's explicit Mode dropdown
(Real/Test); a Real run requires a non-empty Participant ID. Real runs also get
counterbalanced phase order, rest breaks and a session cap (`session_schedule.py`).
`PresaccadeSession` still picks between fixed `_TEST`/`_REAL` trial-count
constant pairs in `include/experiment/constants.py`; `ExperimentSession`'s
real block is open-ended instead (see the open-loop-timing gotcha below) and
only its practice block and RT-test use `_TEST`/`_REAL`-paired counts.

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
is an independent Flask app (`data_access.py` reads/writes the CSV,
`_meta.json`, and `settings.json` files under `data/`) — no shared process
or IPC with the experiment, just shared files on disk.

**Shared logic lives in `trial_mechanics.py`/`scoring.py`, not copy-pasted.**
`OnsetDetector` (`src/experiment/trial_mechanics.py`) is the one saccade-
onset debounce implementation used everywhere it's needed — `ExperimentSession`'s
RT-test and main trials, and `TutorialSession`'s scoped-down mirrors of
both (four call sites total). `is_valid_for_saccadic_analysis()`
(`src/experiment/scoring.py`) is the single `flash_during_saccade`
validity gate both the live ZEST update and `results_graph.py` apply.
`average_ms()` (`trial_mechanics.py`) is the shared "mean, or a fallback if
empty" used by both classes' reaction-time averaging. If you're about to
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
`_meta.json` and a raw `results_<ts>_gaze_trace.csv` (both Neon backends; for the screen-marker backend the
trace's `x_px` column holds screen-ratio x `NEON_SCENE_WIDTH`, so replay reproduces its head-compensated zones).

**Simulator finding (fixed in simulation, unverified on glasses).** Onset/landing used to be
stamped with classifier observation times that trail the real eye by the tracker latency, so with 16 ms
latency only ~3% of kept flashes really hit the saccade. Samples are now stamped with capture time
(Neon: `device_unix_s` + clock offset on the perf_counter timeline; sim: arrival minus the exact
`gaze_latency_ms`), giving precision 1.0 at 0/16/50 ms in `run_sim.py --sweep-profile
tracker_latency_ms=0,16,50`. The simulator hands the session the exact latency; a real Neon estimate is noisier.

## Gotchas that cost real debugging time

**Neon timing and geometry (measured Oct 2026).** Gaze reaches the PC ~16 ms
after the event (p95 ~21 ms, max ~35 ms), ~140-180 Hz effective with gaps up
to ~47 ms. Latency is only right if you convert with
`local = device_ts + offset` from `estimate_time_offset()` (the sign is easy
to get backwards: it gave a bogus 414 ms once). `NeonGazeSource` stamps
samples with their *capture* time (`device_unix_s + offset`, moved onto the
`timebase.now()` perf_counter timeline; arrival time only as a fallback, strictly increasing, clamped
to <=150 ms latency) and the session then must not subtract `gaze_latency_ms` again
(`timestamps_are_capture_time`). The offset is re-estimated every 60 s in the background (unverified on
hardware whether `estimate_time_offset` is safe while streaming). Latency was ~18 ms for the plain scene-camera
backend but 90-100 ms in the screen-mapped sessions of Oct 2026 - check `gaze_quality.latency_ms_median`. The scene
camera is head-mounted, so head movement shifts gaze x with no eye movement -
calibration assumes a still head (chin rest); `neon_gaze_view.py` uses
AprilTag screen mapping instead and needs no calibration. The Neon source has
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

**The first-timer tutorial blocks input while narrating, unlike real
phases.** `run_tutorial_phase` in `run_experiment.py` withholds all
key/tick forwarding while `narrator.is_speaking` is true, so a fast
participant cannot skip ahead mid-narration — real saccade/baseline trials
don't have this restriction. If the tutorial ran, its
`precomputed_calibration_ratios`/`skip_practice_trials` are threaded into
`run_saccade_phase` so the saccade phase skips re-calibrating and skips the
practice/dress-rehearsal stage — this coupling isn't visible from
`session.py` alone.

**Saccade-phase flash timing is open-loop, not gaze-contingent, and
`flash_during_saccade` is a post-hoc window check.** `ExperimentSession`
first runs a reaction-time test (`Phase.RT_TEST_FOREPERIOD`/`RT_TEST_ACTIVE`)
to measure the participant's own beep-to-saccade-onset latency, then
schedules each trial's flash at a fixed delay — `avg_reaction_time_ms +` one
of `TIMING_OFFSETS_MS` — from target onset, entirely independent of the
real-time gaze classifier (which keeps running, but now only to measure
`reaction_latency_ms` and to compute `flash_during_saccade` after the fact:
whether the scheduled flash time fell inside `[_onset_detector.since,
_target_landed_since]`). This trades real-time detection lag as a
constraint on flash precision for a structurally common failure mode — the
scheduled flash simply missing the actual saccade window — so
`flash_during_saccade` is `True`/`False`/`None` (`None` when landing was
never confirmed, not merely invalid), and both `session.py`'s live
`ZestStaircase` update *and* `results_graph.py`'s analysis now require it to
be exactly `True` (not just "not `False`") — the two intentionally stay in
lockstep here, unlike a lot of live-vs-analysis distinctions elsewhere in
this codebase. Total saccade-phase trial count is no longer fixed either:
`_should_stop_main_block()` runs until `ZestStaircase.credible_interval`
narrows enough (with a minimum valid-trial floor and a max-trial safety
cap), so `NUM_TRIALS_PER_PHASE_TEST`/`_REAL` only apply to
`PresaccadeSession` now, not `ExperimentSession`'s real block.

**Saccade onset is a displacement rule, not a zone crossing.** `OnsetDetector` fires when
`GazeSample.position` is more than `SACCADE_ONSET_DISPLACEMENT` (0.08, position units 0=dot..1=cross)
from the fixation baseline at the source for `SACCADE_ONSET_STABILITY_MS` of capture time, and
`since` is the first sample beyond it. The old rule waited for gaze to leave the source *zone* (CENTER
counted as left, i.e. ~40% of the travel) so narrowing `GAZE_DEAD_ZONE` used to help onset - it no longer
does; the dead zone now only matters for landing/zones. Tests/fakes that only set `.zone` must move
`.position` too (`FakeGazeSource.position` follows the zone). Threshold evidence:
`tests/ai-tests/analyze_onset_threshold.py` on recorded traces; recorded screen-mapped traces
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

**Pausing mid-trial discards that trial's data outright, not just its
timers.** `PausableClock` protects in-flight timers across a pause, but
`ExperimentSession.resume_from_pause()` still throws away whatever
trial/RT-test attempt was active — no CSV row, no ZEST feed, no RT-average
feed — because gaze during a pause is aimed at the pause menu, not a
spontaneous task response. The saccade phase's pause menu
(`PauseMenu` in `run_experiment.py`) offers a second option beyond plain
resume: "Return and Recalibrate" re-runs both eye-position calibration (a
faster `RECALIBRATION_ROUNDS=2`, averaging both rounds instead of
discarding one — reuses `average_calibration_rounds()` unchanged) and the
reaction-time test, overwriting `avg_reaction_time_ms` but *not* resetting
the rolling-average's every-`RT_AVERAGE_RECOMPUTE_EVERY`-trials counter.

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

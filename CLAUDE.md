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

## Commands

All commands must be run from this project folder (the one containing this
file) using PsychoPy's own bundled Python (on this machine
`%LOCALAPPDATA%\Programs\PsychoPy\python.exe`; older notes say
`C:\Program Files\PsychoPy`), not a system Python — `-m` needs
the cwd here for `src`/`include` to resolve as top-level packages, and only
PsychoPy's Python has the right dependency set installed:

```
"C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt   # install deps
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment     # run the experiment
"C:\Program Files\PsychoPy\python.exe" -m src.dashboard.app                 # session dashboard at localhost:5000
"C:\Program Files\PsychoPy\python.exe" -m pytest                            # full test suite
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py      # one test file
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py::test_name -v  # one test
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.neon_fixation_test [seconds]  # Neon jitter check (glasses on): sizes SACCADE_ONSET_STABILITY_MS
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.neon_gaze_view  # live calibration-free screen-gaze dot (AprilTags)
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.latency_test    # Neon vs webcam latency recording (analyze with latency_analysis)
```

Without the Spinnaker SDK (`PySpin`) or `mediapipe`, `tests/.../test_camera.py`
and `test_gaze_tracking.py` skip themselves (`pytest.importorskip`) rather
than fail.

Headless simulation (no window, no camera, no files; see "Headless simulator"
below) - run these to try the experiment logic without the glasses:

```
"C:\Program Files\PsychoPy\python.exe" tests/ai-tests/run_sim.py                  # 30-trial simulated session summary
"C:\Program Files\PsychoPy\python.exe" tests/ai-tests/run_sim.py --timeline       # one trial, event by event
"C:\Program Files\PsychoPy\python.exe" tests/ai-tests/run_sim.py --set SACCADE_ONSET_STABILITY_MS=10 --json
"C:\Program Files\PsychoPy\python.exe" tests/ai-tests/run_sim.py --sweep SACCADE_ONSET_STABILITY_MS=10,20,40 --seeds 5
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.simulation                # same one-trial timeline, no flags
```

The test suite needs no camera or PsychoPy window (fake gaze/clock doubles,
Flask's test client) and runs in seconds.

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
`_resolve_test_mode()` treats a blank Participant ID in the startup dialog as
a test run and any other value as a real data-collection session.
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
  standalone scripts like `diagnose_camera.py` do too).
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
`_meta.json` and a raw `results_<ts>_gaze_trace.csv`.

**Known simulator finding (unresolved).** In simulation the
`flash_during_saccade` window is built from classifier observation times, which
trail the real eye by the tracker latency (~16 ms on Neon); with 16 ms latency
only ~3% of flashes the check keeps really hit the saccade (100% at 0 ms).
`run_sim.py --sweep-profile tracker_latency_ms=0,16,50` reproduces it. Not yet
fixed in the experiment code; verify with real traces before trusting
`flash_during_saccade` precision.

## Gotchas that cost real debugging time

**Neon timing and geometry (measured Oct 2026).** Gaze reaches the PC ~16 ms
after the event (p95 ~21 ms, max ~35 ms), ~140-180 Hz effective with gaps up
to ~47 ms. Latency is only right if you convert with
`local = device_ts + offset` from `estimate_time_offset()` (the sign is easy
to get backwards: it gave a bogus 414 ms once). `NeonGazeSource` stamps
samples with arrival `time.monotonic()`, not Neon timestamps. The scene
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

**`GAZE_DEAD_ZONE` (`include/eye_tracking/constants.py`) hides real latency.**
Gaze must cross this band before saccade-onset detection starts counting
`SACCADE_ONSET_STABILITY_MS`, so travel time through the dead zone is real
latency that never shows up in the logged `onset_detection_lag_ms` metric.
Narrowing it trades that hidden latency against a higher false-alarm rate —
see the comment above `SACCADE_ONSET_STABILITY_MS` in
`include/experiment/constants.py` before retuning either constant.

**`CAMERA_GAIN_DB` is an unvalidated placeholder.** Fixed exposure/gain was
long claimed in `_configure_camera`'s docstring before gain was actually
wired up in `src/eye_tracking/camera.py` — `Gain`/`GainAuto` are now set,
but the `6.0` dB value has not been retuned against the real IR illuminator.
Validate with `diagnose_camera.py` before trusting it.

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

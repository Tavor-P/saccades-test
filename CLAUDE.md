# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PsychoPy gaze-contingent saccade experiment (Diamond, Ross & Morrone 2000):
detect a briefly-flashed grating's orientation, once during a saccade and
once during fixation (baseline), to measure saccadic suppression. Gaze is
tracked via a FLIR Blackfly S camera (Spinnaker SDK / `PySpin`) + MediaPipe
iris landmarks. See [README.md](README.md) for the full participant-facing
walkthrough (controls, output files, troubleshooting).

## Commands

All commands must be run from this project folder (the one containing this
file) using PsychoPy's own bundled Python, not a system Python — `-m` needs
the cwd here for `src`/`include` to resolve as top-level packages, and only
PsychoPy's Python has the right dependency set installed:

```
"C:\Program Files\PsychoPy\python.exe" -m pip install -r requirements.txt   # install deps
"C:\Program Files\PsychoPy\python.exe" -m src.experiment.run_experiment     # run the experiment
"C:\Program Files\PsychoPy\python.exe" -m src.dashboard.app                 # session dashboard at localhost:5000
"C:\Program Files\PsychoPy\python.exe" -m pytest                            # full test suite
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py      # one test file
"C:\Program Files\PsychoPy\python.exe" -m pytest tests/test_session.py::test_name -v  # one test
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.diagnose_camera  # capture+inspect one camera frame, no PsychoPy window
"C:\Program Files\PsychoPy\python.exe" -m src.eye_tracking.eye_position_manual_test [camera_index]  # live gaze-dot visual check
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
`WebcamGazeSource` (the only implementation actually in use) owns a `Camera`
+ `GazeTracker` pair on a background thread. `IOHubGazeSource` is an
untested placeholder for real eye-tracking hardware (Tobii/EyeLink) via
PsychoPy's ioHub, written against the documented API but never run against
real hardware.

**Camera → tracker pipeline** (`src/eye_tracking/`):
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

**Test doubles.** `tests/conftest.py` provides `FakeGazeSource` (an in-memory
`GazeSource` implementation tests set `.zone`/`.face_found`/`.position` on
directly) and a `fake_time` fixture that monkeypatches `PausableClock`'s time
source so timing-dependent state-machine tests (foreperiods, response
windows, debounce windows) run deterministically without real waits.

## Gotchas that cost real debugging time

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
whether the scheduled flash time fell inside `[_away_from_source_since,
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

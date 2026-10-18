# ai-tests: headless simulation for AI agents

Run the **real** `ExperimentSession` against a simulated participant, clock and
gaze tracker - no PsychoPy window, no camera, no files written, no real waiting
(a full session takes a few seconds). Use it to check session logic, or to see how
changing a constant (e.g. `SACCADE_ONSET_STABILITY_MS`) changes results, without
asking the human to put the glasses on.

All commands run from the project root with PsychoPy's Python:

```powershell
$py = "$env:LOCALAPPDATA\Programs\PsychoPy\python.exe"
& $py tests/ai-tests/run_sim.py                      # 30-trial session summary
& $py tests/ai-tests/run_sim.py --timeline           # one trial, event by event (ms from go-cue)
& $py tests/ai-tests/run_sim.py --json               # machine-readable summary
& $py tests/ai-tests/run_sim.py --list               # scenarios + participant fields
& $py -m pytest tests/ai-tests -q                    # self-tests of these tools
```

## Recipes

| Goal | Command |
|---|---|
| Different participant | `--scenario slow_reactor` (see `--list`) |
| Tweak the participant | `--profile reaction_ms_mean=300 --profile tracker_latency_ms=50` |
| Change an experiment constant | `--set SACCADE_ONSET_STABILITY_MS=10` (any constant in `include/` or `src/`; a typo is an error) |
| Compare values of a constant | `--sweep SACCADE_ONSET_STABILITY_MS=10,20,40 --seeds 5` |
| Compare values of a participant field | `--sweep-profile tracker_latency_ms=0,16,50 --seeds 5` |
| Reproduce a run | same `--seed` (runs are deterministic) |

From Python: `from sim_tools import run, sweep` (add `tests/ai-tests/` to `sys.path`), or
`from src.experiment.simulation import simulate_trial, simulate_session` directly.
Constants are overridden temporarily (restored afterwards).

## Reading the summary

- `valid_flash_rate` - share of flashes the experiment keeps for analysis (flash inside the
  detected saccade window **and** tracking ok). Low = most trials are discarded.
- `valid_precision` / `valid_recall` - **ground truth only the simulator has**: of the flashes
  the experiment keeps, how many really fell inside the simulated saccade; of the flashes that
  really did, how many it kept. "Really inside" is strict: `[onset, onset + flight]` of the
  simulated eye movement (real perisaccadic suppression extends somewhat beyond that).
- `mean_reaction_latency_ms` - go-cue to the classifier first seeing gaze leave the source zone.
  This runs *later* than the true saccade onset (the eye must cross the whole source zone first,
  plus tracker latency); `--timeline` shows both.
- `mean_onset_detection_lag_ms` - first sample outside the source zone to the debounced onset.
- `tracking_not_ok`, `timeouts`, `false_alarm_rate`, `zest_threshold_estimate`, `outcomes`.

## What the simulation is (and is not)

The participant is a deliberately simple model (`ParticipantProfile` in
`src/experiment/simulation.py`): smooth-step saccades, a logistic psychometric function with a
higher threshold during the saccade, Gaussian tracker noise, fixed tracker latency, random
dropouts. It exercises the experiment's **plumbing and timing logic**; its absolute numbers (e.g.
the threshold estimate) say nothing about real people. Trust *relative* comparisons (this
setting vs that one) more than absolute values, and confirm anything important with the real
glasses (`tests/fixtures/neon_trace_sample.csv` + `src/eye_tracking/gaze_trace.py` replay real
recorded gaze).

## Files

- `run_sim.py` - the CLI. `sim_tools.py` - scenarios, constant overrides, `summarize`, `sweep`.
- `test_ai_tools.py` - pytest self-tests (`pytest tests/ai-tests`).
- The simulator itself lives in `src/experiment/simulation.py` (tests: `tests/test_simulation.py`).

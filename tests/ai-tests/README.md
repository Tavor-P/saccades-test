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
| Realistic reaction times | `--scenario realistic` (= `ParticipantProfile.realistic()`; see below) and `--real` for full-length sessions (150 trials; Test mode stops at 30) |
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
- `true_rt_mean_ms`, `avg_rt_used_mean_ms`, `avg_rt_bias_ms`, `avg_rt_abs_err_ms`, `avg_rt_abs_err_vs_trial_ms` - **reaction-time ground truth**
  (flash trials only). `true_rt_mean_ms` is the simulated participant's real mean latency at each trial (mean + drift
  + wave + direction offset); `avg_rt_used_ms` is what the session actually scheduled the flash from. Bias = used minus
  true mean (signed, e.g. negative when the average lags a rising drift), `abs_err` = |used - true mean| (how well the
  average tracks the mean), `abs_err_vs_trial` = |used - this trial's actual latency| (what decides whether the flash lands;
  dominated by the trial-to-trial SD, which no average can remove). In Python: `sim.cues` (every go-cue incl. RT test and
  practice, as `CueRecord(t, phase, cue_index, latency_ms, true_mean_ms, kind)`; kind is normal/anticipation/lapse) and
  `sim.trial_cues[(phase, index)]` for each logged saccade-phase row, next to `sim.flash_truth`.
- `tracking_not_ok`, `timeouts`, `zest_threshold_estimate`, `outcomes`.

## Reaction-time realism

By default the latency is a plain Gaussian (`reaction_ms_mean/sd`) with no memory. Each of these `ParticipantProfile`
fields adds one real-world effect, is off at its default (default runs and seeds are bit-for-bit unchanged) and can be
swept alone (`--sweep-profile reaction_drift_ms_per_trial=0,0.4,0.8`). A "trial" below is one go-cue (RT test, practice or
main trial), counted from 0.

| Field (default) | Effect |
|---|---|
| `reaction_ms_tau` (0) | exponential tail added to the Gaussian (ex-Gaussian, right-skewed). `reaction_ms_mean` stays the *mean* (Gaussian part is centred on mean - tau); total SD = sqrt(sd^2 + tau^2) |
| `reaction_drift_ms_per_trial` (0) | linear drift of the true mean (0.4 = +40 ms over 100 trials) |
| `reaction_wave_ms` (0), `reaction_wave_period_trials` (60) | sinusoidal fatigue/vigilance wave on the true mean |
| `reaction_autocorr` (0) | AR(1) correlation of the deviation from the true mean between consecutive trials (variance kept) |
| `anticipation_rate` (0), `anticipation_ms_min/max` (80/130) | per-trial chance of a premature saccade, latency uniform in that range, ignoring the mean |
| `lapse_rate` (0), `lapse_extra_ms_min/max` (300/600) | per-trial chance of a lapse: this much extra latency on top of the normal one |
| `reaction_dir_offset_ms` (0) | dot->cross saccades this much slower than cross->dot (split +-half, overall mean unchanged) |

Preset `realistic` (scenario, or `ParticipantProfile.realistic(**overrides)`): mean 220, SD 55, tau 50 (total SD ~74),
drift 0.4 ms/trial, autocorr 0.3, anticipation 3 %, lapse 2 %. Plausible values, not fitted to data. Draws use only the
participant's own seeded rng, so a seed still reproduces a run. Caveat: a rolling-average window below
`RT_AVERAGE_MIN_USABLE` (3) never leaves the default 230 ms - it is a fixed-schedule baseline, not a "last latency" rule.

## What the simulation is (and is not)

The participant is a deliberately simple model (`ParticipantProfile` in
`src/experiment/simulation.py`): smooth-step saccades, a logistic psychometric function with a
higher threshold during the saccade, Gaussian tracker noise, fixed tracker latency, random
dropouts, plus the optional reaction-time effects above. It exercises the experiment's **plumbing and timing logic**; its absolute numbers (e.g.
the threshold estimate) say nothing about real people. Trust *relative* comparisons (this
setting vs that one) more than absolute values, and confirm anything important with the real
glasses (`tests/fixtures/neon_trace_sample.csv` + `src/eye_tracking/gaze_trace.py` replay real
recorded gaze).

## Files

- `run_sim.py` - the CLI. `sim_tools.py` - scenarios, constant overrides, `summarize`, `sweep`.
- `test_ai_tools.py` - pytest self-tests (`pytest tests/ai-tests`).
- `find_bugs.py` + `invariants.py` - randomized bug hunt (`find_bugs.py --sessions 300 --seed 7`, `--extreme`); each finding prints a `run_sim.py` repro command. Exit status 1 if anything is found.
- `test_invariants.py`, `test_bug_injection.py`, `test_schema_sync.py`, `test_constants_sanity.py`, `test_validity_lockstep.py`, `test_architecture_rules.py` - the bug-finding tests (see CLAUDE.md)
- The simulator itself lives in `src/experiment/simulation.py` (tests: `tests/test_simulation.py`).

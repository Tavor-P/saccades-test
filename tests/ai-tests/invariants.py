"""Invariants that must hold for ANY simulated session, whatever the participant does.

`check_session(sim)` returns a list of human-readable violations (empty = clean).
Used by test_invariants.py (fixed seeds) and find_bugs.py (random campaign).
When an invariant fails, first decide whether the *invariant* is wrong or the
*code* is - the checks encode the documented contract of TrialResult
(include/experiment/types.py) and ExperimentSession, not implementation details.
"""

import math

from include.experiment.constants import (
    MAX_SACCADE_TRIALS_TEST,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    FLASH_ONSET_DELAYS_MS,
    PRACTICE_CONTRAST,
    SACCADE_FIXED_CONTRASTS,
    CONTROL_OFFSETS_MS,
    TIMING_OFFSETS_MS,
    ZEST_LOG_CONTRAST_MAX,
)
from src.experiment.simulation import FRAME_HZ, SimulationResult

FRAME_MS = 1000 / FRAME_HZ
SACCADE_OUTCOMES = {"correct", "incorrect", "miss", "timeout"}
RT_OUTCOMES = {"detected", "timeout"}


def check_session(sim: SimulationResult, max_trials_cap: int = MAX_SACCADE_TRIALS_TEST) -> list[str]:
    v: list[str] = []
    saccade_rows = [r for r in sim.results if r.phase == "saccade"]
    rt_rows = [r for r in sim.results if r.phase == "rt_test"]

    for r in sim.results:
        tag = f"{r.phase} #{r.index}"
        if r.phase not in ("rt_test", "saccade"):
            v.append(f"{tag}: unexpected phase {r.phase!r}")
            continue

        if r.phase == "rt_test":
            if r.outcome not in RT_OUTCOMES:
                v.append(f"{tag}: bad rt_test outcome {r.outcome!r}")
            if r.grating_shown or r.contrast is not None or r.responded:
                v.append(f"{tag}: rt_test row has stimulus/response data")
            if r.outcome == "detected" and (r.reaction_latency_ms is None or r.reaction_latency_ms < 0):
                v.append(f"{tag}: detected rt_test row without a non-negative reaction latency ({r.reaction_latency_ms})")
            continue

        # ---- saccade rows ----
        if r.outcome not in SACCADE_OUTCOMES:
            v.append(f"{tag}: unknown outcome {r.outcome!r}")

        if r.grating_shown:
            # Fixed-contrast mode draws from SACCADE_FIXED_CONTRASTS (may exceed the staircase's
            # ceiling); the practice contrast is always allowed; otherwise the ZEST range applies.
            fixed_mode = sim.session._fixed_contrast_bag is not None
            allowed = fixed_mode and (r.contrast in SACCADE_FIXED_CONTRASTS or r.contrast == PRACTICE_CONTRAST)
            if r.contrast is None or not (allowed or 0 < r.contrast <= 10**ZEST_LOG_CONTRAST_MAX + 1e-9):
                v.append(f"{tag}: flashed contrast out of range ({r.contrast})")
            if r.orientation is None:
                v.append(f"{tag}: grating shown but no orientation logged")
            if r.grating_shown_at_unix_ms is None:
                v.append(f"{tag}: grating shown but no grating_shown_at_unix_ms")
        else:
            for name in ("contrast", "orientation", "grating_shown_at_unix_ms", "flash_during_saccade", "tracking_ok"):
                if getattr(r, name) is not None:
                    v.append(f"{tag}: no grating shown but {name}={getattr(r, name)!r}")
            if r.outcome != "timeout":
                v.append(f"{tag}: no grating shown but outcome {r.outcome!r} (every trial flashes; only a timeout can end without one)")

        if r.responded:
            if r.grating_shown and r.response_time_ms is None:
                v.append(f"{tag}: responded to a flash but no response_time_ms")
            if r.response_time_ms is not None and not (0 <= r.response_time_ms <= RESPONSE_WINDOW_MS + 2 * FRAME_MS):
                v.append(f"{tag}: response_time_ms {r.response_time_ms:.0f} outside [0, response window]")
        else:
            if r.response_time_ms is not None or r.response_orientation is not None:
                v.append(f"{tag}: not responded but response data present")
            if r.outcome in ("correct", "incorrect"):
                v.append(f"{tag}: outcome {r.outcome!r} requires a response")

        if r.outcome == "correct" and r.response_orientation != r.orientation:
            v.append(f"{tag}: 'correct' but response {r.response_orientation} != orientation {r.orientation}")
        if r.outcome == "incorrect" and r.response_orientation == r.orientation:
            v.append(f"{tag}: 'incorrect' but response equals orientation")

        if r.reaction_latency_ms is not None and r.reaction_latency_ms < 0:
            v.append(f"{tag}: negative reaction_latency_ms {r.reaction_latency_ms}")
        if r.onset_detection_lag_ms is not None and not (
            0 <= r.onset_detection_lag_ms <= SACCADE_ONSET_STABILITY_MS + 3 * FRAME_MS
        ):
            v.append(f"{tag}: onset_detection_lag_ms {r.onset_detection_lag_ms:.1f} not within the debounce window")
        if r.saccade_duration_ms is not None and r.saccade_duration_ms < 0:
            v.append(f"{tag}: negative saccade_duration_ms")
        allowed_offsets = FLASH_ONSET_DELAYS_MS if sim.session._closed_loop else TIMING_OFFSETS_MS + CONTROL_OFFSETS_MS
        if r.timing_offset_ms is not None and r.timing_offset_ms not in allowed_offsets:
            v.append(f"{tag}: timing_offset_ms {r.timing_offset_ms} not one of {allowed_offsets}")
        for name in ("flash_during_saccade", "tracking_ok"):
            val = getattr(r, name)
            if val is not None and not isinstance(val, bool):
                v.append(f"{tag}: {name} is {type(val).__name__}, expected bool/None")

    indices = [r.index for r in saccade_rows]
    if len(set(indices)) != len(indices):
        v.append(f"duplicate saccade trial indices: {indices}")
    if len(saccade_rows) > max_trials_cap:
        v.append(f"{len(saccade_rows)} saccade trials exceed the safety cap {max_trials_cap}")

    # The staircase must stay a valid estimate no matter what was fed to it.
    estimate = sim.session._zest.threshold_estimate
    if not math.isfinite(estimate) or not (0 < estimate <= 10**ZEST_LOG_CONTRAST_MAX + 1e-9):
        v.append(f"ZEST threshold estimate invalid: {estimate}")

    phase = sim.session.render_state()["hud"]["phase"]
    if phase == "COMPLETE":
        expected_rt_rows = 0 if sim.session._closed_loop else sim.session._rt_test_target_attempts  # gaze-triggered: no RT test
        if len(rt_rows) != expected_rt_rows:
            v.append(f"RT test ran {len(rt_rows)} attempts, expected {expected_rt_rows}")
    return v

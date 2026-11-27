"""Invariants that must hold for ANY simulated session, whatever the participant does.

`check_session(sim)` returns a list of human-readable violations (empty = clean).
Used by test_invariants.py (fixed seeds) and find_bugs.py (random campaign).
When an invariant fails, first decide whether the *invariant* is wrong or the
*code* is - the checks encode the documented contract of TrialResult
(include/experiment/types.py) and ExperimentSession, not implementation details.
"""

from include.experiment import constants as _constants  # read at call time: override_constants patches the module
from include.experiment.constants import (
    CONTROL_OFFSETS_MS,
    FLASH_SPREAD_SD_MS,
    FLASH_WINDOW_MAX_MS,
    FLASH_WINDOW_MIN_MS,
    ONSET_TRIGGER_DELAY_MS,
    PRACTICE_CONTRAST,
    RETURN_JUMP_LANDING_TIMEOUT_MS,
    RETURN_LEG_FLASH_FRACTION,
    SACCADE_ONSET_STABILITY_MS,
    TIMING_JITTER_MS,
    TIMING_OFFSETS_MS,
)
from src.experiment.simulation import FRAME_HZ, SimulationResult
from src.experiment.trial_mechanics import flash_leg_pool, spread_flash_delay_ms

FRAME_MS = 1000 / FRAME_HZ


class _Edge:
    """rng stub for spread_flash_delay_ms: uniform(a, b) returns an edge, giving a slice's bounds."""

    def __init__(self, low: bool):
        self.low = low

    def uniform(self, a, b):
        return a if self.low else b
SACCADE_OUTCOMES = {"correct", "incorrect", "miss", "timeout"}
RT_OUTCOMES = {"detected", "timeout"}


def check_session(sim: SimulationResult, max_trials_cap: int | None = None) -> list[str]:
    """Every violation found, not just the first (each check below appends to one list).
    max_trials_cap defaults to the session's own fixed trial count (SACCADE_FIXED_TRIALS_*)."""
    if max_trials_cap is None:
        max_trials_cap = sim.session._fixed_trials
    return_jump = bool(getattr(sim.session, "_return_jump", False))
    # The session's own copies: simulate() has restored any overridden constants by the time this runs.
    trigger = getattr(sim.session, "_return_jump_trigger", _constants.RETURN_JUMP_TRIGGER)
    return_sd = getattr(sim.session, "_return_flash_sd_ms", _constants.RETURN_FLASH_SPREAD_SD_MS)
    onset_on = bool(getattr(sim.session, "_onset_trigger", False))
    onset_delay = getattr(sim.session, "_onset_trigger_delay_ms", ONSET_TRIGGER_DELAY_MS)

    v: list[str] = []
    for r in sim.results:
        tag = f"{r.phase} #{r.index}"
        if r.phase == "rt_test":
            v += _check_rt_row(r, tag)
        elif r.phase == "saccade":
            v += _check_saccade_row(r, tag, sim, return_jump, trigger, return_sd)
            v += _check_flash_trigger(r, tag, onset_on, onset_delay)
        else:
            v.append(f"{tag}: unexpected phase {r.phase!r}")
    saccade_rows = [r for r in sim.results if r.phase == "saccade"]
    v += _check_trial_count(saccade_rows, max_trials_cap)
    if return_jump:
        v += _check_return_leg_split(saccade_rows)
    v += _check_rt_test_length(sim)
    return v


def _check_rt_row(r, tag: str) -> list[str]:
    v: list[str] = []
    if r.outcome not in RT_OUTCOMES:
        v.append(f"{tag}: bad rt_test outcome {r.outcome!r}")
    if r.flash_shown or r.contrast is not None or r.responded:
        v.append(f"{tag}: rt_test row has stimulus/response data")
    if r.outcome == "detected" and (r.reaction_latency_ms is None or r.reaction_latency_ms < 0):
        v.append(f"{tag}: detected rt_test row without a non-negative reaction latency ({r.reaction_latency_ms})")
    return v


def _check_saccade_row(r, tag: str, sim: SimulationResult, return_jump: bool, trigger: str, return_sd: float) -> list[str]:
    v: list[str] = []
    if r.outcome not in SACCADE_OUTCOMES:
        v.append(f"{tag}: unknown outcome {r.outcome!r}")
    v += _check_flash_fields(r, tag, sim)
    v += _check_response_fields(r, tag)
    v += _check_measurements(r, tag)
    v += _check_return_jump_row(r, tag, return_jump, trigger)
    v += _check_flash_delay(r, tag, return_sd)
    for name in ("flash_during_saccade", "tracking_ok"):
        val = getattr(r, name)
        if val is not None and not isinstance(val, bool):
            v.append(f"{tag}: {name} is {type(val).__name__}, expected bool/None")
    return v


def _check_flash_fields(r, tag: str, sim: SimulationResult) -> list[str]:
    """A row has flash data if and only if a flash was shown."""
    v: list[str] = []
    if r.flash_shown:
        # Every flash uses one of the session's fixed probe contrasts (or the practice contrast).
        allowed = r.contrast in sim.session._fixed_contrast_bag._pool or r.contrast == PRACTICE_CONTRAST
        if r.contrast is None or not allowed:
            v.append(f"{tag}: flashed contrast out of range ({r.contrast})")
        if r.line_side is None:
            v.append(f"{tag}: flash shown but no line_side logged")
        if r.flash_unix_ms is None:
            v.append(f"{tag}: flash shown but no flash_unix_ms")
    else:
        for name in ("contrast", "line_side", "flash_unix_ms", "flash_during_saccade", "tracking_ok"):
            if getattr(r, name) is not None:
                v.append(f"{tag}: no flash shown but {name}={getattr(r, name)!r}")
        if r.outcome != "timeout":
            v.append(f"{tag}: no flash shown but outcome {r.outcome!r} (every trial flashes; only a timeout can end without one)")
    return v


def _check_response_fields(r, tag: str) -> list[str]:
    """Response data, and that the outcome agrees with the answer (there is no answer deadline)."""
    v: list[str] = []
    if r.flash_shown and not r.responded and r.outcome != "timeout":
        v.append(f"{tag}: flashed trial ended without an answer as {r.outcome!r} (forced choice)")
    if r.responded:
        if r.flash_shown and r.response_time_ms is None:
            v.append(f"{tag}: responded to a flash but no response_time_ms")
        if r.response_time_ms is not None and r.response_time_ms < 0:
            v.append(f"{tag}: negative response_time_ms {r.response_time_ms:.0f}")
    else:
        if r.response_time_ms is not None or r.answer_side is not None:
            v.append(f"{tag}: not responded but response data present")
        if r.outcome in ("correct", "incorrect"):
            v.append(f"{tag}: outcome {r.outcome!r} requires a response")
    if r.outcome == "correct" and r.answer_side != r.line_side:
        v.append(f"{tag}: 'correct' but response {r.answer_side} != line_side {r.line_side}")
    if r.outcome == "incorrect" and r.answer_side == r.line_side:
        v.append(f"{tag}: 'incorrect' but answer equals line_side")
    return v


def _check_measurements(r, tag: str) -> list[str]:
    """Latencies and durations are non-negative; the offset label is a known one."""
    v: list[str] = []
    if r.reaction_latency_ms is not None and r.reaction_latency_ms < 0:
        v.append(f"{tag}: negative reaction_latency_ms {r.reaction_latency_ms}")
    if r.onset_detection_lag_ms is not None and not (
        0 <= r.onset_detection_lag_ms <= SACCADE_ONSET_STABILITY_MS + 3 * FRAME_MS
    ):
        v.append(f"{tag}: onset_detection_lag_ms {r.onset_detection_lag_ms:.1f} not within the debounce window")
    if r.saccade_duration_ms is not None and r.saccade_duration_ms < 0:
        v.append(f"{tag}: negative saccade_duration_ms")
    allowed_offsets = TIMING_OFFSETS_MS + CONTROL_OFFSETS_MS
    if r.timing_offset_ms is not None and r.timing_offset_ms not in allowed_offsets:
        v.append(f"{tag}: timing_offset_ms {r.timing_offset_ms} not one of {allowed_offsets}")
    return v


def _check_flash_delay(r, tag: str, return_sd: float) -> list[str]:
    """The flash delay against the row's own leg: bell-curve slice for a timing offset, avg RT + offset for a control."""
    v: list[str] = []
    # The flash delay relative to its own leg's start: the go cue, or the jump-back frame for a return-leg flash.
    leg_delay_ms = r.scheduled_delay_ms
    if r.flash_leg == "return" and r.scheduled_delay_ms is not None and r.jump_back_ms_after_go_cue is not None:
        leg_delay_ms = r.scheduled_delay_ms - r.jump_back_ms_after_go_cue
    on_return = r.flash_leg == "return" and r.jump_back_ms_after_go_cue is not None
    if r.timing_offset_ms in TIMING_OFFSETS_MS and leg_delay_ms is not None and r.avg_rt_used_ms is not None:
        if on_return and r.avg_rt_used_ms <= 0:
            # Anticipatory return average: the flash goes on the jump-back frame itself.
            if abs(leg_delay_ms) > 1e-6:
                v.append(f"{tag}: return average {r.avg_rt_used_ms:.1f} ms <= 0 but flash {leg_delay_ms:.1f} ms after the jump")
        else:
            # Bell-curve flash: inside its slice of the curve around the leg's average, inside the flash window.
            sd = return_sd if on_return else FLASH_SPREAD_SD_MS
            edges = [
                spread_flash_delay_ms(r.timing_offset_ms, TIMING_OFFSETS_MS, r.avg_rt_used_ms, sd,
                                      FLASH_WINDOW_MIN_MS, FLASH_WINDOW_MAX_MS, rng=_Edge(low))
                for low in (True, False)
            ]
            if on_return:
                edges = [max(0.0, e) for e in edges]
            if not edges[0] - 1e-6 <= leg_delay_ms <= edges[1] + 1e-6:
                v.append(f"{tag}: {r.flash_leg or 'out'}-leg flash delay {leg_delay_ms:.1f} ms outside its slice {edges[0]:.1f}..{edges[1]:.1f} ms")
    if r.timing_offset_ms in CONTROL_OFFSETS_MS and leg_delay_ms is not None and r.avg_rt_used_ms is not None:
        expected = r.avg_rt_used_ms + r.timing_offset_ms
        if on_return and expected < -TIMING_JITTER_MS:
            expected = 0.0  # clamped to the jump-back frame
        if abs(leg_delay_ms - expected) > TIMING_JITTER_MS + 1e-6 and not (on_return and abs(leg_delay_ms) < 1e-6):
            v.append(f"{tag}: control flash delay {leg_delay_ms:.1f} ms not avg RT + {r.timing_offset_ms} +/- jitter")
    return v


def _check_flash_trigger(r, tag: str, onset_on: bool, onset_delay_ms: float) -> list[str]:
    """An onset-triggered flash comes after its leg's confirmed onset (debounce + delay) and carries no schedule."""
    if r.flash_trigger not in ("scheduled", "onset"):
        return [f"{tag}: bad flash_trigger {r.flash_trigger!r}"]
    if r.flash_trigger == "scheduled":
        return [f"{tag}: scheduled trial without a timing offset"] if r.timing_offset_ms is None else []
    v: list[str] = []
    if not onset_on:
        v.append(f"{tag}: onset-triggered trial with the mode off")
    if r.timing_offset_ms is not None or r.avg_rt_used_ms is not None:
        v.append(f"{tag}: onset-triggered trial carries schedule data")
    if r.flash_shown:
        rel = r.flash_relative_to_onset_ms
        earliest = SACCADE_ONSET_STABILITY_MS + onset_delay_ms
        if rel is None or rel < earliest - 1e-6:
            v.append(f"{tag}: onset-triggered flash {rel} ms after onset, earlier than {earliest} ms")
        if r.scheduled_delay_ms is None:
            v.append(f"{tag}: onset-triggered flash without flash_ms_after_go_cue")
    elif r.scheduled_delay_ms is not None:
        v.append(f"{tag}: flash given up but flash_ms_after_go_cue {r.scheduled_delay_ms}")
    return v


def _check_trial_count(saccade_rows, max_trials_cap: int) -> list[str]:
    v: list[str] = []
    indices = [r.index for r in saccade_rows]
    if len(set(indices)) != len(indices):
        v.append(f"duplicate saccade trial indices: {indices}")
    if len(saccade_rows) > max_trials_cap:
        v.append(f"{len(saccade_rows)} saccade trials exceed the safety cap {max_trials_cap}")
    return v


def _check_return_leg_split(saccade_rows) -> list[str]:
    """The leg comes from a shuffled bag, so every full pass splits exactly; allow one pass of seam slack."""
    pool = flash_leg_pool(RETURN_LEG_FLASH_FRACTION)
    legs = [r.flash_leg for r in saccade_rows if r.flash_leg is not None]
    expected = len(legs) * pool.count("return") / len(pool)
    if abs(legs.count("return") - expected) > len(pool):
        return [f"{legs.count('return')} of {len(legs)} flashes on the return leg, expected ~{expected:.1f} (fraction {RETURN_LEG_FLASH_FRACTION})"]
    return []


def _check_rt_test_length(sim: SimulationResult) -> list[str]:
    if sim.session.render_state()["hud"]["phase"] != "COMPLETE":
        return []
    rt_rows = [r for r in sim.results if r.phase == "rt_test"]
    expected_rt_rows = sim.session._rt_test_target_attempts
    if len(rt_rows) != expected_rt_rows:
        return [f"RT test ran {len(rt_rows)} attempts, expected {expected_rt_rows}"]
    return []


def _check_return_jump_row(r, tag: str, return_jump: bool, trigger: str = "arrival") -> list[str]:
    """The return-jump columns of one saccade row (see TrialResult: flash_leg ... jump_back_forced)."""
    v: list[str] = []
    jump_fields = ("jump_back_ms_after_go_cue", "return_reaction_ms", "jump_back_forced")
    if r.flash_shown and r.flash_leg not in ("out", "return"):
        v.append(f"{tag}: flash shown but flash_leg {r.flash_leg!r}")
    if not r.flash_shown and r.flash_leg is not None:
        v.append(f"{tag}: no flash shown but flash_leg {r.flash_leg!r}")
    if not return_jump:
        if r.flash_leg not in (None, "out"):
            v.append(f"{tag}: return jump off but flash_leg {r.flash_leg!r}")
        for name in jump_fields:
            if getattr(r, name) is not None:
                v.append(f"{tag}: return jump off but {name}={getattr(r, name)!r}")
        return v
    jump = r.jump_back_ms_after_go_cue
    if jump is None or not isinstance(r.jump_back_forced, bool):
        # Every return-jump trial jumps: on landing, or forced at RETURN_JUMP_LANDING_TIMEOUT_MS.
        v.append(f"{tag}: return jump on but no jump logged (jump_back_ms={jump!r}, forced={r.jump_back_forced!r})")
        return v
    if jump < 0 or (r.jump_back_forced and jump < RETURN_JUMP_LANDING_TIMEOUT_MS - 1e-6):
        v.append(f"{tag}: jump_back_ms_after_go_cue {jump:.1f} impossible (forced={r.jump_back_forced})")
    # return_reaction_ms is SIGNED (an anticipatory return starts before the jump); it can't precede the out arrival.
    arrival = r.out_arrival_ms_after_go_cue
    if r.return_reaction_ms is not None and arrival is not None and jump + r.return_reaction_ms < arrival - 1e-6:
        v.append(f"{tag}: return onset {jump + r.return_reaction_ms:.1f} ms before the out arrival at {arrival:.1f} ms")
    if r.left_target_before_jump and arrival is None:
        v.append(f"{tag}: left_target_before_jump without an out arrival")
    if arrival is not None and (arrival < 0 or (not r.jump_back_forced and trigger == "arrival" and arrival > jump + 1e-6)):
        v.append(f"{tag}: out arrival {arrival:.1f} ms impossible (jump at {jump:.1f}, forced={r.jump_back_forced})")
    dwell = r.jump_back_dwell_ms
    if (dwell is not None) != (trigger == "arrival"):
        v.append(f"{tag}: jump_back_dwell_ms {dwell!r} with trigger {trigger!r}")
    elif dwell is not None and not r.jump_back_forced and (arrival is None or jump - arrival < dwell - 1e-6):
        v.append(f"{tag}: jump at {jump:.1f} ms, less than the {dwell:.0f} ms stay after the out arrival ({arrival})")
    if r.scheduled_delay_ms is not None and r.flash_shown:
        if r.flash_leg == "return" and r.scheduled_delay_ms < jump - 1e-6:
            v.append(f"{tag}: return-leg flash at {r.scheduled_delay_ms:.1f} ms, before the jump-back at {jump:.1f} ms")
        if r.flash_leg == "out" and r.scheduled_delay_ms > jump + 1e-6:
            v.append(f"{tag}: out-leg flash at {r.scheduled_delay_ms:.1f} ms, after the jump-back at {jump:.1f} ms")
    return v

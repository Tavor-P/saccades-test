"""Pre-specified inclusion gates for the post-hoc saccade analysis (after Braun, Schutz & Gegenfurtner 2021).

Every flash is labelled `saccade` (the eye was moving fast when it was shown, in a well-formed saccade),
`fixation` (the eye was still: no saccade within FIXATION_EXCLUSION_MS and no tracking gap) or `excluded`
(with the reason). Unlike a +-50 ms window around onset, nothing here is tuned on the data.
Duck-typed on purpose (no import of analyze_session): flashes need `.time` (PC unix s), saccades
`.onset/.offset/.amplitude`, so the module stays free of the analysis' plotting dependencies.
"""

from dataclasses import dataclass

import numpy as np

# The thresholds live in include/experiment/analysis_constants.py (imported here by name, so `flash_gates.X` still works).
from include.experiment.analysis_constants import (
    AMPLITUDE_RANGE_REL,
    FIXATION_EXCLUSION_MS,
    LATENCY_RANGE_MS,
    MAX_FLASH_SPEED,
    MIN_FLASH_SPEED,
    RETURN_LATENCY_MAX_MS,
    TRACKING_GAP_MS,
)
from src.timebase import ms_to_s, s_to_ms

SACCADE, FIXATION, EXCLUDED = "saccade", "fixation", "excluded"
NOT_MOVING_REASON = "near a saccade but the eye was not moving fast at the flash"  # the exclusion reason report.txt counts before / after onset


@dataclass(frozen=True)
class GateResult:
    condition: str  # SACCADE | FIXATION | EXCLUDED
    reason: str  # why it was excluded, or what the condition rests on
    speed: float | None = None  # gaze speed at the flash (ratio/s)
    latency_ms: float | None = None  # nearest saccade's onset minus the go-cue


_speed_trace_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # (t, x, speed) of the last trace asked about


def speed_at(t: np.ndarray, x: np.ndarray, when: float) -> float | None:
    """3-sample-smoothed gaze speed interpolated at `when`; None outside the trace."""
    global _speed_trace_cache
    if len(t) < 5 or when < t[0] or when > t[-1]:
        return None
    cache = _speed_trace_cache  # a local snapshot: the dashboard analyses on several threads, and only a whole tuple is ever stored
    if cache is None or cache[0] is not t or cache[1] is not x:
        # One speed trace per gaze trace, not per flash (the gradient of a whole recording is the expensive part).
        cache = (t, x, np.abs(np.gradient(np.convolve(x, np.ones(3) / 3, mode="same"), t)))
        _speed_trace_cache = cache
    return float(np.interp(when, t, cache[2]))


def has_tracking_gap(t: np.ndarray, when: float, half_window_ms: float = FIXATION_EXCLUSION_MS) -> bool:
    """True if the trace has no samples, or a hole longer than TRACKING_GAP_MS, within +-half_window of `when`."""
    lo, hi = when - ms_to_s(half_window_ms), when + ms_to_s(half_window_ms)
    if len(t) == 0 or t[0] > lo or t[-1] < hi:
        return True
    i, j = np.searchsorted(t, lo), np.searchsorted(t, hi)
    window = t[max(i - 1, 0) : min(j + 1, len(t))]  # include the samples that bracket the window
    return bool(len(window) < 2 or s_to_ms(np.max(np.diff(window))) > TRACKING_GAP_MS)


def classify_flash(
    when: float,
    saccades: list,
    t: np.ndarray,
    x: np.ndarray,
    go_cue: float | None = None,
    median_amplitude: float | None = None,
) -> GateResult:
    """Label one flash drawn at PC time `when` (display delay already added). `go_cue` is the trial's target
    onset (PC unix s) for the latency gate; without it that gate is skipped."""
    if has_tracking_gap(t, when):
        return GateResult(EXCLUDED, "tracking gap around the flash")
    if not saccades:
        return GateResult(EXCLUDED, "no saccade in the session")
    speed = speed_at(t, x, when)
    nearest = min(saccades, key=lambda s: 0.0 if s.onset <= when <= s.offset else min(abs(when - s.onset), abs(when - s.offset)))
    gap_ms = 0.0 if nearest.onset <= when <= nearest.offset else s_to_ms(min(abs(when - nearest.onset), abs(when - nearest.offset)))
    latency_ms = None if go_cue is None else s_to_ms(nearest.onset - go_cue)

    if gap_ms > FIXATION_EXCLUSION_MS:
        return GateResult(FIXATION, f"{gap_ms:.0f} ms from the nearest saccade", speed, latency_ms)

    # Within +-FIXATION_EXCLUSION_MS of a saccade: it is neither clean fixation nor yet a saccade flash.
    if speed is None or speed < MIN_FLASH_SPEED:
        return GateResult(EXCLUDED, NOT_MOVING_REASON, speed, latency_ms)
    if speed > MAX_FLASH_SPEED:
        return GateResult(EXCLUDED, "gaze speed at the flash is implausibly high", speed, latency_ms)
    if latency_ms is not None and not LATENCY_RANGE_MS[0] <= latency_ms <= LATENCY_RANGE_MS[1]:
        return GateResult(EXCLUDED, f"saccade latency {latency_ms:.0f} ms outside {LATENCY_RANGE_MS[0]:.0f}-{LATENCY_RANGE_MS[1]:.0f}", speed, latency_ms)
    if median_amplitude:
        rel = nearest.amplitude / median_amplitude
        if not AMPLITUDE_RANGE_REL[0] <= rel <= AMPLITUDE_RANGE_REL[1]:
            return GateResult(EXCLUDED, f"saccade amplitude {rel:.2f}x the session median", speed, latency_ms)
    return GateResult(SACCADE, "fast gaze movement at the flash", speed, latency_ms)


def classify_flashes(placed: list, saccades: list, t: np.ndarray, x: np.ndarray, flash_delay_ms: float = 0.0, go_cues: list | None = None) -> list[GateResult]:
    """`placed[i].flash.time` against the saccades; `go_cues[i]` is that flash's trial start (or None)."""
    median_amp = float(np.median([s.amplitude for s in saccades])) if saccades else None
    out = []
    for i, p in enumerate(placed):
        cue = go_cues[i] if go_cues is not None else None
        out.append(classify_flash(p.flash.time + ms_to_s(flash_delay_ms), saccades, t, x, cue, median_amp))
    return out


# -- return-jump runs: each flash is gated against the saccade of its OWN leg ---------------------------------------------
# A return-jump trial has two saccades (out: go cue -> target; return: after the target jumps back). The flash was timed for
# one of them (trials.csv `flash_leg`), so the saccade gates use that leg's saccade and its own latency anchor: the go cue for
# the out leg, the jump-back frame for the return leg. The fixation rule is unchanged (>= FIXATION_EXCLUSION_MS from EVERY
# saccade, the other leg's included), so a flash in the landing-to-return gap is fixation only when it clears both saccades.
# RETURN_LATENCY_MAX_MS (analysis_constants): return onset at most this long after the jump; no lower bound, so predictive returns
# (< 0) are kept.


def classify_leg_flash(
    when: float,
    saccades: list,
    t: np.ndarray,
    x: np.ndarray,
    leg: str,
    own,
    anchor: float | None,
    median_amplitude: float | None = None,
    jump_forced: bool = False,
) -> GateResult:
    """Label one flash of a return-jump trial. `own` is this trial's saccade of the flash's `leg` ('out' | 'return'), or None
    when none was found; `anchor` the latency reference (go cue for 'out', jump-back frame for 'return'; PC unix s);
    `median_amplitude` the run's median for that leg. Return-leg flashes of a forced jump (no landing seen) are excluded."""
    if has_tracking_gap(t, when):
        return GateResult(EXCLUDED, "tracking gap around the flash")
    if not saccades:
        return GateResult(EXCLUDED, "no saccade in the session")
    if leg == "return" and jump_forced:
        return GateResult(EXCLUDED, "return leg of a forced jump (the out landing was never seen)")
    speed = speed_at(t, x, when)
    nearest = min(saccades, key=lambda s: 0.0 if s.onset <= when <= s.offset else min(abs(when - s.onset), abs(when - s.offset)))
    gap_ms = 0.0 if nearest.onset <= when <= nearest.offset else s_to_ms(min(abs(when - nearest.onset), abs(when - nearest.offset)))
    latency_ms = None if own is None or anchor is None else s_to_ms(own.onset - anchor)

    if gap_ms > FIXATION_EXCLUSION_MS:
        return GateResult(FIXATION, f"{gap_ms:.0f} ms from the nearest saccade", speed, latency_ms)
    if own is None:
        return GateResult(EXCLUDED, f"near a saccade but no {leg} saccade was found in this trial", speed, latency_ms)
    if nearest is not own:
        return GateResult(EXCLUDED, f"near a saccade other than this trial's {leg} saccade", speed, latency_ms)
    if speed is None or speed < MIN_FLASH_SPEED:
        return GateResult(EXCLUDED, NOT_MOVING_REASON, speed, latency_ms)
    if speed > MAX_FLASH_SPEED:
        return GateResult(EXCLUDED, "gaze speed at the flash is implausibly high", speed, latency_ms)
    if latency_ms is not None:
        if leg == "return" and latency_ms > RETURN_LATENCY_MAX_MS:
            return GateResult(EXCLUDED, f"return latency {latency_ms:.0f} ms after the jump, over {RETURN_LATENCY_MAX_MS:.0f}", speed, latency_ms)
        if leg != "return" and not LATENCY_RANGE_MS[0] <= latency_ms <= LATENCY_RANGE_MS[1]:
            return GateResult(EXCLUDED, f"saccade latency {latency_ms:.0f} ms outside {LATENCY_RANGE_MS[0]:.0f}-{LATENCY_RANGE_MS[1]:.0f}", speed, latency_ms)
    if median_amplitude:
        rel = own.amplitude / median_amplitude
        if not AMPLITUDE_RANGE_REL[0] <= rel <= AMPLITUDE_RANGE_REL[1]:
            return GateResult(EXCLUDED, f"{leg} saccade amplitude {rel:.2f}x the session's {leg} median", speed, latency_ms)
    return GateResult(SACCADE, f"fast gaze movement at the flash ({leg} saccade)", speed, latency_ms)


def summarize_gates(placed: list, results: list[GateResult]) -> dict:
    """Per condition: [correct, total] and the exclusion reasons."""
    out: dict = {SACCADE: [0, 0], FIXATION: [0, 0], EXCLUDED: [0, 0], "reasons": {}}
    for p, r in zip(placed, results):
        out[r.condition][1] += 1
        out[r.condition][0] += p.flash.correct
        if r.condition == EXCLUDED:
            out["reasons"][r.reason] = out["reasons"].get(r.reason, 0) + 1
    return out

"""Pre-specified inclusion gates for the post-hoc saccade analysis (after Braun, Schutz & Gegenfurtner 2021).

Every flash is labelled `saccade` (the eye was moving fast when it was shown, in a well-formed saccade),
`fixation` (the eye was still: no saccade within FIXATION_EXCLUSION_MS and no tracking gap) or `excluded`
(with the reason). Unlike a +-50 ms window around onset, nothing here is tuned on the data.
Duck-typed on purpose (no import of analyze_session): flashes need `.time` (PC unix s), saccades
`.onset/.offset/.amplitude`, so the module stays free of the analysis' plotting dependencies.
"""

from dataclasses import dataclass

import numpy as np

FIXATION_EXCLUSION_MS = 100.0  # a fixation flash must be this far from every saccade (Braun et al.: +-100 ms)
TRACKING_GAP_MS = 100.0  # no gaze samples for this long around a flash = blink / tracking loss
LATENCY_RANGE_MS = (100.0, 500.0)  # saccade latency from the go-cue; faster = anticipation, slower = lapse
AMPLITUDE_RANGE_REL = (0.5, 1.5)  # saccade amplitude relative to the session median (Braun: 7.5-15 deg for 10)
MIN_FLASH_SPEED = 2.0  # ratio-of-width/s at the flash: it must have been drawn during fast movement
MAX_FLASH_SPEED = 25.0  # above this the trace is a spike, not an eye (Braun: 500 deg/s)

SACCADE, FIXATION, EXCLUDED = "saccade", "fixation", "excluded"


@dataclass(frozen=True)
class GateResult:
    condition: str  # SACCADE | FIXATION | EXCLUDED
    reason: str  # why it was excluded, or what the condition rests on
    speed: float | None = None  # gaze speed at the flash (ratio/s)
    latency_ms: float | None = None  # nearest saccade's onset minus the go-cue


def speed_at(t: np.ndarray, x: np.ndarray, when: float) -> float | None:
    """3-sample-smoothed gaze speed interpolated at `when`; None outside the trace."""
    if len(t) < 5 or when < t[0] or when > t[-1]:
        return None
    speed = np.abs(np.gradient(np.convolve(x, np.ones(3) / 3, mode="same"), t))
    return float(np.interp(when, t, speed))


def has_tracking_gap(t: np.ndarray, when: float, half_window_ms: float = FIXATION_EXCLUSION_MS) -> bool:
    """True if the trace has no samples, or a hole longer than TRACKING_GAP_MS, within +-half_window of `when`."""
    lo, hi = when - half_window_ms / 1000, when + half_window_ms / 1000
    if len(t) == 0 or t[0] > lo or t[-1] < hi:
        return True
    i, j = np.searchsorted(t, lo), np.searchsorted(t, hi)
    window = t[max(i - 1, 0) : min(j + 1, len(t))]  # include the samples that bracket the window
    return bool(len(window) < 2 or np.max(np.diff(window)) * 1000 > TRACKING_GAP_MS)


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
    gap_ms = 0.0 if nearest.onset <= when <= nearest.offset else 1000 * min(abs(when - nearest.onset), abs(when - nearest.offset))
    latency = None if go_cue is None else 1000 * (nearest.onset - go_cue)

    if gap_ms > FIXATION_EXCLUSION_MS:
        return GateResult(FIXATION, f"{gap_ms:.0f} ms from the nearest saccade", speed, latency)

    # Within +-FIXATION_EXCLUSION_MS of a saccade: it is neither clean fixation nor yet a saccade flash.
    if speed is None or speed < MIN_FLASH_SPEED:
        return GateResult(EXCLUDED, "near a saccade but the eye was not moving fast at the flash", speed, latency)
    if speed > MAX_FLASH_SPEED:
        return GateResult(EXCLUDED, "gaze speed at the flash is implausibly high", speed, latency)
    if latency is not None and not LATENCY_RANGE_MS[0] <= latency <= LATENCY_RANGE_MS[1]:
        return GateResult(EXCLUDED, f"saccade latency {latency:.0f} ms outside {LATENCY_RANGE_MS[0]:.0f}-{LATENCY_RANGE_MS[1]:.0f}", speed, latency)
    if median_amplitude:
        rel = nearest.amplitude / median_amplitude
        if not AMPLITUDE_RANGE_REL[0] <= rel <= AMPLITUDE_RANGE_REL[1]:
            return GateResult(EXCLUDED, f"saccade amplitude {rel:.2f}x the session median", speed, latency)
    return GateResult(SACCADE, "fast gaze movement at the flash", speed, latency)


def classify_flashes(placed: list, saccades: list, t: np.ndarray, x: np.ndarray, flash_delay_ms: float = 0.0, go_cues: list | None = None) -> list[GateResult]:
    """`placed[i].flash.time` against the saccades; `go_cues[i]` is that flash's trial start (or None)."""
    median_amp = float(np.median([s.amplitude for s in saccades])) if saccades else None
    out = []
    for i, p in enumerate(placed):
        cue = go_cues[i] if go_cues is not None else None
        out.append(classify_flash(p.flash.time + flash_delay_ms / 1000, saccades, t, x, cue, median_amp))
    return out


def summarize_gates(placed: list, results: list[GateResult]) -> dict:
    """Per condition: [correct, total] and the exclusion reasons."""
    out: dict = {SACCADE: [0, 0], FIXATION: [0, 0], EXCLUDED: [0, 0], "reasons": {}}
    for p, r in zip(placed, results):
        out[r.condition][1] += 1
        out[r.condition][0] += p.flash.correct
        if r.condition == EXCLUDED:
            out["reasons"][r.reason] = out["reasons"].get(r.reason, 0) + 1
    return out

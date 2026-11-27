"""Saccade metrics of the post-hoc analysis (see analyze_session): latency, amplitude, gain, landing error, duration and peak
velocity of each trial's primary saccade, per leg (out, and return in return-jump runs), and the share of trials with no detected,
an anticipatory or a corrective saccade. Pure functions on what the analysis already has; thresholds in analysis_constants.py.

Gain needs the run's calibration only (the gaze ratio while fixating each symbol: session.json settings `calibration`). Degrees
also need the screen geometry: the angle between the symbols (screen height x ASSUMED_SCREEN_ASPECT, viewing distance) divided by
the calibrated ratio between them. Each run is converted with its own calibration, so pooling concatenates converted trials."""

import math
from dataclasses import dataclass

import numpy as np

from include.experiment.analysis_constants import (
    ANTICIPATION_LATENCY_MS,
    ASSUMED_SCREEN_ASPECT,
    CORRECTIVE_AMPLITUDE_REL,
    CORRECTIVE_SPEED_HIGH,
    CORRECTIVE_SPEED_LOW,
    CORRECTIVE_WINDOW_MS,
    MIN_CALIBRATION_SPAN,
)
from include.experiment.constants import CROSS_POSITION, DOT_POSITION
from src.experiment.geometry import Geometry
from src.experiment.leg_gating import LEGS, TrialSaccades, _robust_sd
from src.experiment.saccade_detection import Flash, Saccade, detect_saccades
from src.timebase import ms_to_s, s_to_ms

# per-trial values summarised for each leg (also the keys of a leg's summary)
MEASURES = ("latency_ms", "amplitude_deg", "gain", "landing_error_deg", "landing_error_abs_deg", "duration_ms", "peak_velocity_deg_s")


@dataclass(frozen=True)
class Conversion:
    """One run's gaze ratio -> target distance and degrees (None = not possible)."""

    left_ratio: float | None  # calibrated gaze ratio at the dot
    right_ratio: float | None  # ...and at the cross
    target_deg: float | None  # angle between the two symbols
    placeholder: bool = False  # the screen geometry is the unmeasured placeholder


@dataclass(frozen=True)
class LegTrial:
    """One trial's primary saccade on one leg (every value None when no saccade was found, or it cannot be converted)."""

    found: bool
    corrective: bool = False
    latency_ms: float | None = None  # from the go cue (out) / the jump (return)
    amplitude_deg: float | None = None
    gain: float | None = None
    landing_error_deg: float | None = None  # along the movement: positive = beyond the target
    landing_error_abs_deg: float | None = None
    duration_ms: float | None = None
    peak_velocity_deg_s: float | None = None


@dataclass(frozen=True)
class RunSaccades:
    """One run's per-trial values, ready to be pooled."""

    conversion: Conversion
    legs: dict  # {"out": [LegTrial], "return": [LegTrial] | None (no return jump)}
    forced_jumps: int = 0  # return trials left out: the jump came on the landing timeout


def target_distance_deg(geometry: Geometry, dot_x: float = DOT_POSITION[0], cross_x: float = CROSS_POSITION[0]) -> float:
    """The visual angle between two horizontal screen positions (0-1 of the width), seen from in front of the screen's centre."""
    width_cm = geometry.screen_height_cm * ASSUMED_SCREEN_ASPECT
    angles = [math.atan((x - 0.5) * width_cm / geometry.viewing_distance_cm) for x in (dot_x, cross_x)]
    return abs(math.degrees(angles[1] - angles[0]))


def run_conversion(settings: dict | None) -> Conversion:
    """From session.json `settings` (calibration, screen height, viewing distance); whatever is missing or unusable is None."""
    s = settings if isinstance(settings, dict) else {}
    left = right = None
    try:
        ratios = float(s["calibration"]["left_ratio"]), float(s["calibration"]["right_ratio"])
        if abs(ratios[1] - ratios[0]) >= MIN_CALIBRATION_SPAN:
            left, right = ratios
    except (KeyError, TypeError, ValueError):
        pass  # no calibration recorded: no gain, no degrees
    try:
        geometry = Geometry(float(s["screen_height_cm"]), float(s["viewing_distance_cm"]))
        degrees = target_distance_deg(geometry, float((s.get("dot_position") or DOT_POSITION)[0]), float((s.get("cross_position") or CROSS_POSITION)[0]))
    except (KeyError, TypeError, ValueError, IndexError):
        degrees = None
    return Conversion(left, right, degrees, bool(s.get("geometry_is_placeholder")))


def has_corrective(primary: Saccade, small: list[Saccade], end: float) -> bool:
    """A smaller saccade (from the low-threshold detection `small`) starting shortly after `primary` ended and before `end`."""
    first, last = primary.offset + ms_to_s(CORRECTIVE_WINDOW_MS[0]), min(primary.offset + ms_to_s(CORRECTIVE_WINDOW_MS[1]), end)
    low, high = (share * primary.amplitude for share in CORRECTIVE_AMPLITUDE_REL)
    return any(first <= s.onset < last and low <= s.amplitude < high for s in small)


def leg_trial(saccade: Saccade | None, anchor_s: float, conversion: Conversion, small: list[Saccade], end: float) -> LegTrial:
    """The values of one primary saccade; `anchor_s` = go cue / jump, `end` = where the search for a corrective saccade stops."""
    if saccade is None:
        return LegTrial(False)
    values = {"latency_ms": s_to_ms(saccade.onset - anchor_s), "duration_ms": saccade.duration_ms}
    left, right = conversion.left_ratio, conversion.right_ratio
    if left is not None and right is not None:
        span = right - left
        values["gain"] = saccade.amplitude / abs(span)
        if conversion.target_deg is not None:
            deg_per_ratio = conversion.target_deg / abs(span)
            direction = math.copysign(1.0, saccade.x1 - saccade.x0)
            target = right if direction == math.copysign(1.0, span) else left
            error = float(direction * (saccade.x1 - target) * deg_per_ratio)
            values.update(amplitude_deg=saccade.amplitude * deg_per_ratio, landing_error_deg=error, landing_error_abs_deg=abs(error),
                          peak_velocity_deg_s=saccade.peak_speed * deg_per_ratio)
    return LegTrial(True, has_corrective(saccade, small, end), **values)


def run_saccades(flashes: list[Flash], trials: list[TrialSaccades], t: np.ndarray, x: np.ndarray, settings: dict | None) -> RunSaccades:
    """Every trial's primary saccade(s) of one run. `trials` = leg_gating.find_trial_saccades(flashes, ...), one per flash;
    the return leg counts the same trials as return_latency_stats (jumped, not forced)."""
    conversion = run_conversion(settings)
    small = detect_saccades(t, x, CORRECTIVE_SPEED_HIGH, CORRECTIVE_SPEED_LOW, min_amplitude=0.0)
    cues = sorted(f.go_cue for f in flashes if f.go_cue is not None)
    out, back, forced = [], [], 0
    for f, tr in zip(flashes, trials, strict=True):
        if f.go_cue is None:
            continue
        next_cue = next((c for c in cues if c > f.go_cue), math.inf)
        out.append(leg_trial(tr.out, f.go_cue, conversion, small, next_cue if tr.back is None else min(tr.back.onset, next_cue)))
        if f.jump is None:
            continue
        if f.jump_forced:
            forced += 1
        else:
            back.append(leg_trial(tr.back, f.jump, conversion, small, next_cue))
    jumped = any(f.jump is not None for f in flashes)
    return RunSaccades(conversion, {"out": out, "return": back if jumped else None}, forced)


def _stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    spread = len(v) >= 2
    return {
        "n": int(len(v)),
        "median": float(np.median(v)) if len(v) else None,
        "robust_sd": _robust_sd(v) if spread else None,
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))) if spread else None,
    }


def _leg_summary(trials: list[LegTrial]) -> dict:
    found = [tr for tr in trials if tr.found]
    counts = {
        "no_saccade": len(trials) - len(found),
        "anticipation": sum(tr.latency_ms < ANTICIPATION_LATENCY_MS for tr in found if tr.latency_ms is not None),
        "before_cue": sum(tr.latency_ms < 0 for tr in found if tr.latency_ms is not None),
        "corrective": sum(tr.corrective for tr in found),
    }
    return {
        "trials": len(trials),
        "found": len(found),
        "counts": counts,  # trials with ...
        "shares": {k: (n / len(trials) if trials else None) for k, n in counts.items()},  # ... as a share of all trials of the leg
        **{m: _stats([v for tr in found if (v := getattr(tr, m)) is not None]) for m in MEASURES},
    }


def summarize_saccades(runs: list[RunSaccades]) -> dict:
    """The `saccade_metrics` record of one run, or of several pooled: {"legs": {"out", "return" (None without a return jump)},
    "forced_jumps", "conversion"}."""
    legs = {}
    for leg in LEGS:
        members = [r.legs[leg] for r in runs if r.legs[leg] is not None]
        legs[leg] = _leg_summary([tr for trials in members for tr in trials]) if members or leg == "out" else None
    degrees = sorted({round(r.conversion.target_deg, 3) for r in runs if r.conversion.target_deg is not None})
    return {
        "legs": legs,
        "forced_jumps": sum(r.forced_jumps for r in runs),
        "conversion": {
            "runs": len(runs),
            "runs_without_calibration": sum(r.conversion.left_ratio is None for r in runs),
            "runs_without_geometry": sum(r.conversion.target_deg is None for r in runs),
            "target_distance_deg": degrees,
            "screen_aspect_assumed": ASSUMED_SCREEN_ASPECT,
            "geometry_is_placeholder": any(r.conversion.placeholder for r in runs),
        },
    }

"""Averaging of the 3-point gaze-calibration rounds (round 1 is discarded as the least reliable) and validation of the resulting fit."""

import math
from dataclasses import dataclass

import numpy as np


def average_calibration_rounds(
    round2: tuple[float, float, float], round3: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Combines the 2nd and 3rd rounds of a 3-round calibration into the final
    (left, center, right) ratios. Round 1 is deliberately not a parameter here
    - it's discarded by both ExperimentSession and TutorialSession before this
    is ever called, on the assumption that a participant's first calibration
    attempt is likely to be their least reliable one."""
    return (
        (round2[0] + round3[0]) / 2,
        (round2[1] + round3[1]) / 2,
        (round2[2] + round3[2]) / 2,
    )


# Validation thresholds for the 3-point fit. Ratios are the raw 0-1 horizontal
# gaze ratios (a typical default span is 0.4/0.5/0.6). Kept here rather than in
# include/ so this change stays within its owning files; move them if tuned.
MIN_CALIBRATION_SPAN = 0.05  # |right - left|: below this the slope is near-infinite
MIN_CALIBRATION_STEP = 0.015  # smallest allowed gap left->center and center->right
MAX_CALIBRATION_SAMPLE_STD = 0.03  # noise limit on the per-point sample window
MAX_CALIBRATION_RESIDUAL = 0.15  # RMS misfit in position units (0-1) of the linear fit


@dataclass(frozen=True)
class CalibrationCheck:
    """Outcome of validate_calibration(). `ok` is True iff `problems` is empty;
    `problems` are short human-readable reasons suitable for an on-screen retry
    message. `residual` is the RMS error (position units, 0-1) of the linear fit
    through the three points - 0 for a perfectly linear spacing."""

    ok: bool
    problems: tuple[str, ...]
    span: float
    residual: float
    slope: float | None
    max_sample_std: float | None


def fit_residual(left: float, center: float, right: float) -> tuple[float, float, float]:
    """Least-squares line mapping ratio -> position through (left, 0), (center,
    0.5), (right, 1). Returns (slope, intercept, rms_residual). Degenerate
    (all-equal) ratios return slope=nan rather than raising."""
    xs = np.array([left, center, right], dtype=float)
    ys = np.array([0.0, 0.5, 1.0])
    if not np.all(np.isfinite(xs)) or np.ptp(xs) == 0:
        return float("nan"), float("nan"), float("inf")
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = float(np.sqrt(np.mean((slope * xs + intercept - ys) ** 2)))
    return float(slope), float(intercept), residual


def validate_calibration(
    ratios: tuple[float, float, float],
    sample_stds: tuple[float, ...] | None = None,
    *,
    min_span: float = MIN_CALIBRATION_SPAN,
    min_step: float = MIN_CALIBRATION_STEP,
    max_sample_std: float = MAX_CALIBRATION_SAMPLE_STD,
    max_residual: float = MAX_CALIBRATION_RESIDUAL,
) -> CalibrationCheck:
    """Checks a (left, center, right) calibration before it is applied: finite,
    strictly monotonic in either direction (a mirrored camera/tracker is fine),
    each step and the total span big enough, centre not wildly off-centre
    (linear-fit residual), and - if given - the standard deviation of each
    point's sample window under the noise limit."""
    left, center, right = ratios
    problems: list[str] = []
    if not all(math.isfinite(r) for r in ratios):
        return CalibrationCheck(False, ("non-finite calibration ratio",), 0.0, float("inf"), None, None)
    span = abs(right - left)
    d1, d2 = center - left, right - center
    if d1 * d2 <= 0:
        problems.append("left/center/right are not strictly monotonic")
    else:
        if min(abs(d1), abs(d2)) < min_step:
            problems.append("center is too close to left or right")
        if span < min_span:
            problems.append("left-to-right span is too small")
    slope, _, residual = fit_residual(left, center, right)
    if not problems and residual > max_residual:
        problems.append("calibration points are badly non-linear")
    worst_std = max(sample_stds) if sample_stds else None
    if worst_std is not None and worst_std > max_sample_std:
        problems.append("gaze was too unsteady during calibration")
    return CalibrationCheck(not problems, tuple(problems), span, residual, None if math.isnan(slope) else slope, worst_std)

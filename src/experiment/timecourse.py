"""Accuracy as a function of flash time from saccade onset: the numbers behind the report's main figure.

A Gaussian-kernel (Nadaraya-Watson) estimate of P(correct) evaluated every TIMECOURSE_STEP_MS, with a pointwise bootstrap
95 % band over trials, the kernel's effective n (Kish) at every point, and the fixation baseline (the flashes the pre-specified
fixation gate keeps, Wilson 95 % CI). The x value of a flash is `Placed.rel_ms`: its time minus the onset of its OWN saccade
(return-jump runs use the flash's own leg, the nearest onset otherwise), so nothing is re-placed here.
All contrasts are pooled in the curve: the session draws each flash's contrast from its own bag, independently of the flash's
time (`trial_factory` / `ExperimentSession._fire_flash_if_due`), so the contrast mix does not vary with onset-relative time.
Where the curve may be drawn: kernel effective n below TIMECOURSE_MIN_EFFECTIVE_N_DRAWN = not at all ("none"), up to
TIMECOURSE_MIN_EFFECTIVE_N_SOLID = grey and dashed ("low"), above = solid ("ok"). Thresholds: include/experiment/analysis_constants.py.
"""

import math
import warnings

import numpy as np

from include.experiment.analysis_constants import (
    FAR_MS,
    NEAR_ONSET_MS,
    TIMECOURSE_MIN_EFFECTIVE_N_DRAWN,
    TIMECOURSE_MIN_EFFECTIVE_N_SOLID,
    TIMECOURSE_RANGE_MS,
    TIMECOURSE_SIGMA_MS,
    TIMECOURSE_STEP_MS,
)
from src.experiment.flash_gates import TRACKING_GAP_REASON
from src.experiment.saccade_detection import Placed
from src.experiment.suppression_stats import _select_conditions, wilson

NONE, LOW, OK = "none", "low", "ok"  # the drawing status of a curve point
_KERNEL_REACH_SD = 5.0  # flashes farther than this many kernel SDs from every grid point have no weight


def curve_grid(range_ms: tuple[float, float] = TIMECOURSE_RANGE_MS, step_ms: float = TIMECOURSE_STEP_MS) -> np.ndarray:
    return np.arange(range_ms[0], range_ms[1] + step_ms / 2, step_ms)


def timed_flashes(placed: list[Placed], gates: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(onset-relative ms, correct 0/1) of every flash with a known onset, leaving out those whose gaze had a tracking gap."""
    rel, ok = [], []
    for i, p in enumerate(placed):
        if p.rel_ms is None or (gates is not None and gates[i].reason == TRACKING_GAP_REASON):
            continue
        rel.append(p.rel_ms)
        ok.append(1.0 if p.flash.correct else 0.0)
    return np.asarray(rel, dtype=float), np.asarray(ok, dtype=float)


def _weights(rel: np.ndarray, grid: np.ndarray, sigma_ms: float | np.ndarray) -> np.ndarray:
    """Gaussian kernel weights, shape (grid point, flash); `sigma_ms` one SD for all points or one per grid point."""
    return np.exp(-0.5 * ((grid[:, None] - rel[None, :]) / np.reshape(sigma_ms, (-1, 1))) ** 2)


def adaptive_sigma(rel: np.ndarray, grid: np.ndarray, sigma_ms: float, neighbours: int, max_sigma_ms: float) -> np.ndarray:
    """Per-grid-point kernel SD for sparse data (k-nearest-neighbour bandwidth): the distance from the point to its `neighbours`-th
    nearest flash, kept between `sigma_ms` (dense data: the fixed kernel) and `max_sigma_ms`."""
    if len(rel) == 0:
        return np.full(len(grid), sigma_ms)
    distances = np.sort(np.abs(grid[:, None] - rel[None, :]), axis=1)
    return np.clip(distances[:, min(neighbours, len(rel)) - 1], sigma_ms, max_sigma_ms)


def kernel_estimate(rel: np.ndarray, ok: np.ndarray, grid: np.ndarray, sigma_ms: float | np.ndarray = TIMECOURSE_SIGMA_MS) -> tuple[np.ndarray, np.ndarray]:
    """(P(correct) at every grid point, kernel effective n). NaN / 0 where no flash has weight. The effective n is Kish's
    (sum w)^2 / sum w^2, capped at sqrt(2) x the kernel mass sum w: for evenly spread flashes the two are equal, but Kish alone
    is blind to a gap in the data (a point between two clusters sees only their far tails, with a full Kish n but almost no weight)."""
    if len(rel) == 0:
        return np.full(len(grid), np.nan), np.zeros(len(grid))
    w = _weights(rel, grid, sigma_ms)
    total, squares = w.sum(axis=1), (w**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(total > 1e-12, (w @ ok) / total, np.nan)
        n_eff = np.where(squares > 0, np.minimum(total**2 / squares, math.sqrt(2) * total), 0.0)
    return p, n_eff


def _percentile_band(boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5 / 97.5 % percentiles per grid point of the (resample, grid point) curves `boot`; NaN where fewer than half of the resamples reach it."""
    with np.errstate(invalid="ignore"):
        enough = np.isfinite(boot).sum(axis=0) >= max(2, len(boot) // 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column (no flash in reach) gives NaN, which `enough` hides
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return np.where(enough, lo, np.nan), np.where(enough, hi, np.nan)


def _bootstrap_band(
    rel: np.ndarray, ok: np.ndarray, grid: np.ndarray, sigma_ms: float, n_boot: int, seed: int, neighbours: int | None = None, max_sigma_ms: float = TIMECOURSE_SIGMA_MS
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile 95 % band of the curve over resamples of the flashes (trials); with `neighbours` every resample
    gets its own adaptive bandwidth (adaptive_sigma)."""
    nan = np.full(len(grid), np.nan)
    if len(rel) < 2:
        return nan, nan
    rng = np.random.default_rng(seed)
    if neighbours is not None:
        boot = np.full((n_boot, len(grid)), np.nan)
        for b in range(n_boot):
            pick = rng.integers(0, len(rel), len(rel))
            boot[b] = kernel_estimate(rel[pick], ok[pick], grid, adaptive_sigma(rel[pick], grid, sigma_ms, neighbours, max_sigma_ms))[0]
        return _percentile_band(boot)
    w = _weights(rel, grid, sigma_ms)  # (G, N)
    counts = rng.multinomial(len(rel), np.full(len(rel), 1 / len(rel)), size=n_boot).astype(float)  # (B, N): times each flash was drawn
    with np.errstate(invalid="ignore", divide="ignore"):
        den = counts @ w.T
        boot = np.where(den > 1e-12, ((counts * ok) @ w.T) / den, np.nan)  # (B, G)
    return _percentile_band(boot)


def point_status(n_eff: float) -> str:
    return OK if n_eff >= TIMECOURSE_MIN_EFFECTIVE_N_SOLID else LOW if n_eff >= TIMECOURSE_MIN_EFFECTIVE_N_DRAWN else NONE


def _adaptive_status(rel: np.ndarray, grid: np.ndarray, n_eff: np.ndarray, sigma_used: np.ndarray, sigma_ms: float) -> list[str]:
    """Drawing status of an adaptive-bandwidth curve: "none" outside the span of the flashes (no extrapolation past the outermost one) and
    where no flash lies within the bandwidth, "ok" (solid) where the bandwidth
    stayed at the base SD and the effective n is high enough, else "low" (grey dashed: widened / indicative)."""
    nearest = np.abs(grid[:, None] - rel[None, :]).min(axis=1) if len(rel) else np.full(len(grid), np.inf)
    outside = (grid < rel.min()) | (grid > rel.max()) if len(rel) else np.ones(len(grid), dtype=bool)
    return [NONE if out or d > h else OK if h <= sigma_ms + 1e-9 and n >= TIMECOURSE_MIN_EFFECTIVE_N_SOLID else LOW for out, d, h, n in zip(outside, nearest, sigma_used, n_eff)]


def _listed(a: np.ndarray, digits: int) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in a]


def timecourse_curve(
    placed: list[Placed],
    gates: list | None = None,
    sigma_ms: float = TIMECOURSE_SIGMA_MS,
    n_boot: int = 1000,
    seed: int = 0,
    range_ms: tuple[float, float] = TIMECOURSE_RANGE_MS,
    neighbours: int | None = None,
    max_sigma_ms: float = TIMECOURSE_SIGMA_MS,
) -> dict:
    """The time-course numbers of `placed` (with their gate results, same order): JSON-ready, lists per grid point
    ({"grid_ms", "p", "lo", "hi", "n_eff", "status"}), the fixation baseline, the lowest drawn point and the counts.
    `gates` None: the baseline is the flashes at least FAR_MS from every onset (the index's window rule). `range_ms`: the grid's first and last point.
    `neighbours` (with `max_sigma_ms`): adaptive bandwidth for sparse data (adaptive_sigma) instead of the fixed `sigma_ms`; the curve is then
    drawn wherever a flash lies within its bandwidth, and the result adds "neighbours", "max_sigma_ms" and "sigma_used_ms"."""
    grid = curve_grid(range_ms)
    rel, ok = timed_flashes(placed, gates)
    keep = np.abs(rel) <= max(abs(range_ms[0]), abs(range_ms[1])) + _KERNEL_REACH_SD * max(sigma_ms, max_sigma_ms if neighbours is not None else 0.0)
    rel_k, ok_k = rel[keep], ok[keep]
    sigma_used = sigma_ms if neighbours is None else adaptive_sigma(rel_k, grid, sigma_ms, neighbours, max_sigma_ms)
    p, n_eff = kernel_estimate(rel_k, ok_k, grid, sigma_used)
    lo, hi = _bootstrap_band(rel_k, ok_k, grid, sigma_ms, n_boot, seed, neighbours, max_sigma_ms)
    status = [point_status(n) for n in n_eff] if neighbours is None else _adaptive_status(rel_k, grid, n_eff, np.asarray(sigma_used), sigma_ms)
    shown = [i for i, s in enumerate(status) if s != NONE and np.isfinite(p[i])]
    solid = [i for i in shown if status[i] == OK]
    lowest = min(solid or shown, key=lambda i: p[i], default=None)  # prefer a point drawn solid
    _, fixation, source = _select_conditions(placed, NEAR_ONSET_MS, FAR_MS, gates)
    k_fix, n_fix = sum(f.flash.correct for f in fixation), len(fixation)
    out = {
        "sigma_ms": sigma_ms, "step_ms": TIMECOURSE_STEP_MS, "range_ms": list(range_ms), "n_boot": n_boot,
        "min_effective_n_drawn": TIMECOURSE_MIN_EFFECTIVE_N_DRAWN, "min_effective_n_solid": TIMECOURSE_MIN_EFFECTIVE_N_SOLID,
        "n_flashes": int(len(rel)), "n_correct": int(ok.sum()),
        "grid_ms": _listed(grid, 1), "p": _listed(p, 4), "lo": _listed(lo, 4), "hi": _listed(hi, 4), "n_eff": _listed(n_eff, 1), "status": status,
        "baseline": {"n": n_fix, "correct": k_fix, "p": k_fix / n_fix if n_fix else None, "ci": list(wilson(k_fix, n_fix)) if n_fix else None,
                     "source": "fixation gate" if source == "gates" else f"flashes at least {FAR_MS:g} ms from every onset"},
        "lowest": None if lowest is None else {"ms": float(grid[lowest]), "p": round(float(p[lowest]), 4), "status": status[lowest], "n_eff": round(float(n_eff[lowest]), 1)},
    }
    if neighbours is not None:
        out.update({"neighbours": neighbours, "max_sigma_ms": max_sigma_ms, "sigma_used_ms": _listed(np.asarray(sigma_used), 1)})
    return out


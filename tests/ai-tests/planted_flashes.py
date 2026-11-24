"""Flashes with a PLANTED time course, for the time-course / companion-number tests (no files: `Placed` + `GateResult` lists).

Each trial gets a contrast (4 / 6 / 8 %, drawn independently of its time, like the session's bags) and an onset-relative flash time:
most from a bell curve around the saccade onset (the shipped flash spread), some controls far away. Accuracy is a 2AFC Weibull in
contrast (threshold `threshold`, slope 3) scaled toward chance by `depth` inside the dip window `dip` (ms from onset, a box): the true
suppression is known. Gates follow flash_gates' rule: inside the saccade (0..SACCADE_MS) = `saccade`, more than 100 ms from it = `fixation`,
the rest `excluded`. `true_curve` is the exact expected accuracy at a time, to score a fitted curve against.
"""

import numpy as np

from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE, GateResult
from src.experiment.saccade_detection import Flash, Placed
from src.experiment.suppression_stats import _weibull

CONTRASTS = (0.04, 0.06, 0.08)
SACCADE_MS = 50.0
FIXATION_GAP_MS = 100.0
DEFAULT_DIP = (-30.0, 60.0)


def visible_accuracy(contrast: float, threshold: float = 0.045, slope: float = 3.0) -> float:
    return float(_weibull(np.asarray(contrast), np.asarray(threshold), np.asarray(slope)))


def true_accuracy(rel_ms: float, contrast: float, depth: float, dip: tuple[float, float] = DEFAULT_DIP, threshold: float = 0.045) -> float:
    """P(correct) of a flash at `rel_ms`: the visible accuracy pulled `depth` (0..1) of the way to chance inside the dip window."""
    p = visible_accuracy(contrast, threshold)
    return 0.5 + (p - 0.5) * (1 - (depth if dip[0] <= rel_ms <= dip[1] else 0.0))


def true_curve(grid_ms: np.ndarray, depth: float, dip: tuple[float, float] = DEFAULT_DIP, threshold: float = 0.045) -> np.ndarray:
    """Expected accuracy at each time, equal weight over the three contrasts."""
    return np.asarray([np.mean([true_accuracy(t, c, depth, dip, threshold) for c in CONTRASTS]) for t in grid_ms])


def gate_for(rel_ms: float) -> GateResult:
    if 0 <= rel_ms <= SACCADE_MS:
        return GateResult(SACCADE, "fast gaze movement at the flash")
    gap = -rel_ms if rel_ms < 0 else rel_ms - SACCADE_MS
    if gap > FIXATION_GAP_MS:
        return GateResult(FIXATION, f"{gap:.0f} ms from the nearest saccade")
    return GateResult(EXCLUDED, "near a saccade but the eye was not moving fast at the flash")


def planted_flashes(
    n_trials: int = 150,
    seed: int = 0,
    depth: float = 0.0,
    dip: tuple[float, float] = DEFAULT_DIP,
    centre_ms: float = 15.0,
    spread_sd_ms: float = 110.0,
    control_share: float = 0.2,
    run: int = 0,
) -> tuple[list[Placed], list[GateResult]]:
    """One run's flashes and their gates (`run` only offsets the trial numbers, so pooled runs stay distinguishable)."""
    rng = np.random.default_rng([seed, run])
    placed, gates = [], []
    for i in range(n_trials):
        contrast = float(rng.choice(CONTRASTS))
        rel = float(rng.normal(450.0, 40.0) if rng.random() < control_share else rng.normal(centre_ms, spread_sd_ms))
        correct = rng.random() < true_accuracy(rel, contrast, depth, dip)
        flash = Flash(run * 10_000 + i, 1000.0 + i, contrast, "above", "correct" if correct else "incorrect")
        placed.append(Placed(flash, rel, 0 <= rel <= SACCADE_MS))
        gates.append(gate_for(rel))
    return placed, gates


def pooled_planted_flashes(n_runs: int, seed: int = 0, **kwargs) -> tuple[list[Placed], list[GateResult]]:
    placed: list[Placed] = []
    gates: list[GateResult] = []
    for run in range(n_runs):
        p, g = planted_flashes(seed=seed, run=run, **kwargs)
        placed += p
        gates += g
    return placed, gates

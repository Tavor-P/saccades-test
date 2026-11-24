"""Statistics of the post-hoc analysis: accuracy counts per time window with Wilson / Fisher (`summarize`) and the per-person
saccadic suppression index (`suppression_index`: Weibull threshold fit with a bootstrap CI, else an accuracy fallback).
Thresholds: include/experiment/analysis_constants.py."""

import math

import numpy as np

from include.experiment.analysis_constants import (
    ACCURACY_CEILING,
    ACCURACY_FLOOR_MARGIN,
    BIN_EDGES_MS,
    BOOT_MIN_IDENTIFIED,
    BOOT_MIN_IDENTIFIED_SHARE,
    CHANCE,
    FAR_MS,
    LAPSE,
    MIN_ACCURACY_TRIALS,
    MIN_FIT_TRIALS,
    NEAR_ONSET_MS,
    THRESHOLD_EXTRAPOLATION,
)
from src.experiment.flash_gates import FIXATION, SACCADE
from src.experiment.saccade_detection import Placed, Saccade


# -- statistics ----------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # rounding can leave the bounds a hair inside p (k == n gives 0.9999999999999999); keep p inside and the ends exact
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (0.0 if k == 0 else min(lo, p), 1.0 if k == n else max(hi, p))


def fisher_p(a_correct: int, a_total: int, b_correct: int, b_total: int) -> float | None:
    """Two-sided Fisher exact p for two accuracy groups (None if scipy is unavailable or a group is empty)."""
    if a_total == 0 or b_total == 0:
        return None
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return None
    return float(fisher_exact([[a_correct, a_total - a_correct], [b_correct, b_total - b_correct]])[1])


def _count(placed: list[Placed]) -> tuple[int, int]:
    return sum(p.flash.correct for p in placed), len(placed)


def summarize(placed: list[Placed], saccades: list[Saccade]) -> dict:
    """Counts and accuracy by condition; the numbers behind the printed report."""
    timed = [p for p in placed if p.rel_ms is not None]
    near = [p for p in timed if abs(p.rel_ms) < NEAR_ONSET_MS]
    beyond = [p for p in timed if NEAR_ONSET_MS <= abs(p.rel_ms) < FAR_MS]
    far = [p for p in timed if abs(p.rel_ms) >= FAR_MS]
    inside = [p for p in timed if p.inside]
    outside = [p for p in timed if not p.inside]
    bins = []
    for lo, hi in zip(BIN_EDGES_MS[:-1], BIN_EDGES_MS[1:]):
        members = [p for p in timed if lo <= p.rel_ms < hi]
        k, n = _count(members)
        bins.append({"lo_ms": lo, "hi_ms": hi, "n": n, "correct": k, "ci": wilson(k, n)})
    by_contrast = {}
    for c in sorted({p.flash.contrast for p in placed}):
        group = [p for p in placed if p.flash.contrast == c]
        by_contrast[c] = {"all": _count(group), "near_onset": _count([p for p in group if p in near]), "beyond": _count([p for p in group if p in beyond])}
    return {
        "flashes": len(placed),
        "saccades": len(saccades),
        "inside": _count(inside),
        "outside": _count(outside),
        "near_onset": _count(near),
        "beyond": _count(beyond),
        "far": _count(far),
        "overall": _count(placed),
        "misses": sum(p.flash.outcome == "miss" for p in placed),
        "bins": bins,
        "by_contrast": by_contrast,
        "p_near_vs_beyond": fisher_p(*_count(near), *_count(beyond)),
        "p_inside_vs_outside": fisher_p(*_count(inside), *_count(outside)),
        "median_saccade_ms": float(np.median([s.duration_ms for s in saccades])) if saccades else None,
        "median_amplitude": float(np.median([s.amplitude for s in saccades])) if saccades else None,
    }


# -- suppression index ---------------------------------------------------------------------------
# Braun, Schutz & Gegenfurtner (2021, J Vis 21(6):11): suppression = 1 - CS_saccade / CS_fixation, with contrast
# sensitivity CS = 1 / threshold. 0 = no suppression, 1 = flashes are never seen; negative = better during saccades.

# The fit's search grid (an implementation detail, not a threshold: see analysis_constants for those).
_SLOPES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)  # Weibull slopes tried; one slope is shared by both conditions
_EXTRAPOLATION = THRESHOLD_EXTRAPOLATION  # the old private name (analyze_session re-exports it)
_THRESHOLDS = np.logspace(-3, 1, 121)  # candidate thresholds as a contrast fraction (0.1 % .. 1000 %)


def _contrast_counts(placed: list[Placed]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(contrasts, trials, correct) per distinct contrast."""
    contrasts = sorted({p.flash.contrast for p in placed})
    n = np.asarray([sum(p.flash.contrast == c for p in placed) for c in contrasts], dtype=float)
    k = np.asarray([sum(p.flash.contrast == c and p.flash.correct for p in placed) for c in contrasts], dtype=float)
    return np.asarray(contrasts, dtype=float), n, k


def _weibull(contrast: np.ndarray, threshold: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """P(correct) = chance + (1 - chance - lapse) * (1 - exp(-(c / threshold) ** slope)); the threshold alpha
    is the contrast giving ~63 % of the way from chance to the ceiling (only the ratio of two alphas matters)."""
    return CHANCE + (1 - CHANCE - LAPSE) * (1 - np.exp(-((contrast / threshold) ** slope)))


def _loglik_grid(contrasts: np.ndarray, n: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log-likelihood over (slope, threshold) for one condition."""
    s = np.asarray(_SLOPES)[:, None, None]
    t = _THRESHOLDS[None, :, None]
    p = np.clip(_weibull(contrasts[None, None, :], t, s), 1e-9, 1 - 1e-9)
    return (k * np.log(p) + (n - k) * np.log(1 - p)).sum(axis=2)  # (slope, threshold)


def _fit_pair(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f) -> tuple[float, float, bool]:
    """Maximum-likelihood thresholds (saccade, fixation) with a slope shared by the two conditions.
    The last value is True when either threshold sits on the edge of the search grid (not identified)."""
    return fit_pair_with_slope(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f)[:3]


def fit_pair_with_slope(contrasts_s, n_s, k_s, contrasts_f, n_f, k_f) -> tuple[float, float, bool, float]:
    """`_fit_pair` plus the shared Weibull slope it chose (for drawing the fitted curves)."""
    ll_s, ll_f = _loglik_grid(contrasts_s, n_s, k_s), _loglik_grid(contrasts_f, n_f, k_f)
    best_s, best_f = ll_s.argmax(axis=1), ll_f.argmax(axis=1)
    rows = np.arange(len(_SLOPES))
    b = int((ll_s[rows, best_s] + ll_f[rows, best_f]).argmax())
    ts, tf = float(_THRESHOLDS[best_s[b]]), float(_THRESHOLDS[best_f[b]])
    edge = any(t in (_THRESHOLDS[0], _THRESHOLDS[-1]) for t in (ts, tf))
    # Not identified either when the threshold lies far outside the tested contrasts (flat likelihood: all near
    # chance or all near ceiling).
    for t, c in ((ts, contrasts_s), (tf, contrasts_f)):
        edge = edge or t > _EXTRAPOLATION * c.max() or t < c.min() / _EXTRAPOLATION
    return ts, tf, bool(edge), float(_SLOPES[b])


def _select_conditions(placed: list[Placed], near_ms: float, far_ms: float, gates: list | None) -> tuple[list[Placed], list[Placed], str]:
    """(saccade flashes, fixation flashes, where the conditions came from: 'gates' | 'windows')."""
    if gates is not None:
        # Pre-specified gates (flash_gates.py) decide the conditions; excluded flashes are left out.
        sac = [p for p, g in zip(placed, gates) if g.condition == SACCADE]
        fix = [p for p, g in zip(placed, gates) if g.condition == FIXATION]
        return sac, fix, "gates"
    timed = [p for p in placed if p.rel_ms is not None]
    sac = [p for p in timed if abs(p.rel_ms) < near_ms]
    fix = [p for p in timed if abs(p.rel_ms) >= far_ms]
    return sac, fix, "windows"


def _psychometric_index(out: dict, sac: list[Placed], fix: list[Placed], min_fit_trials: int, n_boot: int, seed: int) -> str | None:
    """Primary method: per-condition Weibull thresholds (shared slope) with a bootstrap CI, written into `out`.
    Returns None when done, else why the fit is not possible or not identified (then `out` is untouched)."""
    cs, ns, ks = _contrast_counts(sac)
    cf, nf, kf = _contrast_counts(fix)
    if len(sac) < min_fit_trials or len(fix) < min_fit_trials:
        return f"fewer than {min_fit_trials} flashes per condition ({len(sac)} saccade, {len(fix)} fixation)"
    if len(cs) < 2 or len(cf) < 2:
        return "fewer than 2 distinct contrasts in a condition"
    ts, tf, edge = _fit_pair(cs, ns, ks, cf, nf, kf)
    if edge:
        return "accuracy at ceiling or floor across the tested contrasts, so a threshold is not identified"
    boot, edges = _bootstrap_indices(cs, ns, ks, cf, nf, kf, n_boot, seed)
    ci = None
    if len(boot) >= BOOT_MIN_IDENTIFIED_SHARE * n_boot and len(boot) >= BOOT_MIN_IDENTIFIED:
        ci = tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))
    else:
        out["notes"].append(f"no CI: only {len(boot)} of {n_boot} bootstrap fits were identified "
                            f"(need >= {100 * BOOT_MIN_IDENTIFIED_SHARE:.0f}% and >= {BOOT_MIN_IDENTIFIED})")
    out.update(method="psychometric", index=1 - tf / ts, ci=ci, threshold_saccade=ts, threshold_fixation=tf,
               label=f"1 - CS_saccade/CS_fixation from per-condition Weibull thresholds (2AFC, guess {CHANCE:g}, lapse {LAPSE:g}, shared slope)")
    if edges:
        out["notes"].append(f"{100 * edges / n_boot:.0f}% of bootstrap fits were unidentified and excluded from the CI")
    return None


def _bootstrap_indices(cs, ns, ks, cf, nf, kf, n_boot: int, seed: int) -> tuple[list[float], int]:
    """Parametric bootstrap of the threshold index: (indices of the identified refits, number of unidentified refits)."""
    rng = np.random.default_rng(seed)
    boot = []
    edges = 0
    for _ in range(n_boot):
        bs = _fit_pair(cs, ns, rng.binomial(ns.astype(int), ks / ns), cf, nf, rng.binomial(nf.astype(int), kf / nf))
        edges += bs[2]
        if not bs[2]:
            boot.append(1 - bs[1] / bs[0])
    return boot, edges


def _accuracy_index(out: dict, n_sac: int, n_fix: int, min_accuracy_trials: int) -> None:
    """Fallback: accuracy_fixation - accuracy_saccade with a Newcombe (Wilson-based) 95 % CI, written into `out` (or a note why not)."""
    notes = out["notes"]
    if n_sac < min_accuracy_trials or n_fix < min_accuracy_trials:
        notes.append(f"accuracy index not computed: fewer than {min_accuracy_trials} flashes in a condition")
        return
    k_sac, k_fix = out["correct_saccade"], out["correct_fixation"]
    p_sac, p_fix = k_sac / n_sac, k_fix / n_fix
    if p_fix < CHANCE + ACCURACY_FLOOR_MARGIN:
        notes.append(f"accuracy index not computed: accuracy away from onset is only {100 * p_fix:.0f}% (floor), so the flashes are not visible even during fixation")
        return
    (l_sac, u_sac), (l_fix, u_fix) = wilson(k_sac, n_sac), wilson(k_fix, n_fix)
    d = p_fix - p_sac
    out.update(method="accuracy", index=d, label="ACCURACY index (not a threshold ratio): accuracy_fixation - accuracy_saccade",
               ci=(d - math.sqrt((p_fix - l_fix) ** 2 + (u_sac - p_sac) ** 2), d + math.sqrt((u_fix - p_fix) ** 2 + (p_sac - l_sac) ** 2)))
    if p_sac >= ACCURACY_CEILING and p_fix >= ACCURACY_CEILING:
        notes.append("both conditions are at ceiling: this index cannot show suppression at these contrasts")


def suppression_index(
    placed: list[Placed],
    near_ms: float | None = None,
    far_ms: float | None = None,
    min_fit_trials: int = MIN_FIT_TRIALS,
    min_accuracy_trials: int = MIN_ACCURACY_TRIALS,
    n_boot: int = 1000,
    seed: int = 0,
    gates: list | None = None,
) -> dict:
    """Per-person saccadic suppression index from the placed flashes (see the section comment above).

    'saccade' = flashes with |time from the nearest onset| < near_ms; 'fixation' = flashes at least far_ms from
    every onset (defaults: NEAR_ONSET_MS / FAR_MS). Primary: a 2AFC Weibull fit per condition (shared slope) giving
    1 - threshold_fixation / threshold_saccade, with a bootstrap 95 % CI. When the fit is not possible or not
    identified (too few trials or contrasts, accuracy at ceiling or floor) the fallback is the clearly labelled
    accuracy index accuracy_fixation - accuracy_saccade with a Newcombe (Wilson-based) 95 % CI. If neither can be
    computed, "method" is None and "notes" says why. Never raises on sparse data."""
    near_ms = NEAR_ONSET_MS if near_ms is None else near_ms
    far_ms = FAR_MS if far_ms is None else far_ms
    sac, fix, source = _select_conditions(placed, near_ms, far_ms, gates)
    out = {
        "conditions_from": source, "near_ms": near_ms, "far_ms": far_ms, "n_saccade": len(sac), "n_fixation": len(fix),
        "correct_saccade": sum(p.flash.correct for p in sac), "correct_fixation": sum(p.flash.correct for p in fix),
        "method": None, "index": None, "ci": None, "label": "", "threshold_saccade": None, "threshold_fixation": None,
        "notes": [],
    }
    if not sac or not fix:
        out["notes"].append(f"cannot compute: {len(sac)} saccade flashes, {len(fix)} fixation flashes")
        return out
    fit_problem = _psychometric_index(out, sac, fix, min_fit_trials, n_boot, seed)
    if fit_problem is not None:
        out["notes"].append(f"psychometric index not computed: {fit_problem}")
        _accuracy_index(out, len(sac), len(fix), min_accuracy_trials)
    return out

"""Companion numbers next to the headline suppression index (additive: the index and its CI are unchanged).

From the pre-specified gate flashes (saccade vs fixation), per contrast level:
- accuracy difference fixation - saccade in percentage points, equal weight over the contrasts that have at least
  STRATUM_MIN_SACCADE_FLASHES saccade flashes (and one fixation flash);
- relative suppression S = (p_fix - p_sac) / (p_fix - 0.5) with the same stratified accuracies: 0 = none, 1 = the line is guessed
  at during saccades while it is seen in fixation (the analogue of the clinical papers' relative drop; Lencer et al. 2021). No S
  when there are fewer than RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES saccade flashes or p_fix is below
  RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY (the reason is returned, not a number);
- both with a stratified percentile bootstrap 95 % CI (resampling within contrast x gate);
- when the index came from the threshold fit: log10 of the threshold ratio saccade / fixation = -log10(1 - index), with the
  index's own bootstrap CI transformed the same way (a monotone map, so the percentiles carry over).
Thresholds: include/experiment/analysis_constants.py.
"""

import math

import numpy as np

from include.experiment.analysis_constants import (
    CHANCE,
    FAR_MS,
    NEAR_ONSET_MS,
    RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY,
    RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES,
    STRATUM_MIN_SACCADE_FLASHES,
)
from src.experiment.saccade_detection import Placed
from src.experiment.suppression_stats import _select_conditions

_CI_MIN_VALID_SHARE = 0.75  # an S CI needs at least this share of resamples with p_fix above chance


def _percentile_ci(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def _cells(sac: list[Placed], fix: list[Placed]) -> tuple[list[dict], list[dict]]:
    """(used strata, dropped strata): per contrast the saccade / fixation (n, correct)."""
    used, dropped = [], []
    for c in sorted({p.flash.contrast for p in sac} | {p.flash.contrast for p in fix}):
        s = [p for p in sac if p.flash.contrast == c]
        f = [p for p in fix if p.flash.contrast == c]
        cell = {"contrast_pct": round(100 * c, 4), "n_saccade": len(s), "correct_saccade": sum(p.flash.correct for p in s),
                "n_fixation": len(f), "correct_fixation": sum(p.flash.correct for p in f)}
        if len(s) < STRATUM_MIN_SACCADE_FLASHES:
            dropped.append({**cell, "why": f"fewer than {STRATUM_MIN_SACCADE_FLASHES} saccade flashes"})
        elif not f:
            dropped.append({**cell, "why": "no fixation flashes"})
        else:
            used.append(cell)
    return used, dropped


def _stratified(n_s, k_s, n_f, k_f) -> tuple:
    """(p_sac, p_fix): equal weight over the strata (the last axis); broadcasts over bootstrap rows."""
    return (k_s / n_s).mean(axis=-1), (k_f / n_f).mean(axis=-1)


def companion_numbers(
    placed: list[Placed],
    gates: list | None = None,
    suppression: dict | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """The companion numbers of one run or one pooled set of flashes (JSON-ready; see the module docstring).
    `suppression` is the `suppression_index()` result of the same flashes (for the log threshold ratio); `gates` as in
    suppression_index (None: the +-NEAR_ONSET_MS / FAR_MS windows). Never raises on sparse data: missing numbers are None with a note."""
    near_ms = (suppression or {}).get("near_ms", NEAR_ONSET_MS)
    far_ms = (suppression or {}).get("far_ms", FAR_MS)
    sac, fix, source = _select_conditions(placed, near_ms, far_ms, gates)
    used, dropped = _cells(sac, fix)
    out: dict = {
        "conditions_from": source, "n_saccade": len(sac), "n_fixation": len(fix), "n_boot": n_boot,
        "strata_contrasts_pct": [c["contrast_pct"] for c in used], "strata_dropped": dropped,
        "p_saccade": None, "p_fixation": None,
        "accuracy_difference_pp": None, "accuracy_difference_ci_pp": None,
        "relative_suppression": None, "relative_suppression_ci": None, "relative_suppression_note": None,
        "log10_threshold_ratio": None, "log10_threshold_ratio_ci": None,
    }
    if not used:
        out["relative_suppression_note"] = "no contrast has enough saccade and fixation flashes"
        return out

    n_s = np.asarray([c["n_saccade"] for c in used], dtype=float)
    k_s = np.asarray([c["correct_saccade"] for c in used], dtype=float)
    n_f = np.asarray([c["n_fixation"] for c in used], dtype=float)
    k_f = np.asarray([c["correct_fixation"] for c in used], dtype=float)
    p_sac, p_fix = _stratified(n_s, k_s, n_f, k_f)
    out.update(p_saccade=float(p_sac), p_fixation=float(p_fix), accuracy_difference_pp=100 * float(p_fix - p_sac))

    rng = np.random.default_rng(seed)
    b_sac, b_fix = _stratified(n_s, rng.binomial(n_s.astype(int), (k_s / n_s), size=(n_boot, len(used))), n_f,
                               rng.binomial(n_f.astype(int), (k_f / n_f), size=(n_boot, len(used))))
    out["accuracy_difference_ci_pp"] = _percentile_ci(100 * (b_fix - b_sac))

    if len(sac) < RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES:
        out["relative_suppression_note"] = f"no S: only {len(sac)} saccade flashes (need {RELATIVE_SUPPRESSION_MIN_SACCADE_FLASHES})"
    elif p_fix < RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY:
        out["relative_suppression_note"] = (f"no S: accuracy in fixation is only {100 * p_fix:.0f}% (need {100 * RELATIVE_SUPPRESSION_MIN_FIXATION_ACCURACY:.0f}%), "
                                            "so the line is not clearly visible even without a saccade")
    else:
        out["relative_suppression"] = float((p_fix - p_sac) / (p_fix - CHANCE))
        defined = b_fix > CHANCE  # a resample whose fixation accuracy fell to chance has no S
        if defined.sum() >= _CI_MIN_VALID_SHARE * n_boot:
            out["relative_suppression_ci"] = _percentile_ci((b_fix[defined] - b_sac[defined]) / (b_fix[defined] - CHANCE))
        else:
            out["relative_suppression_note"] = f"no CI for S: fixation accuracy reached chance in {n_boot - int(defined.sum())} of {n_boot} resamples"

    if suppression is not None and suppression.get("method") == "psychometric" and suppression.get("index") is not None:
        out["log10_threshold_ratio"] = -math.log10(1 - suppression["index"])  # index = 1 - threshold_fixation / threshold_saccade
        if suppression.get("ci") is not None:
            out["log10_threshold_ratio_ci"] = [-math.log10(1 - v) for v in suppression["ci"]]
    return out

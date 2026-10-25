"""Inclusion gates (src/experiment/flash_gates.py) and the fixation-control offsets."""

import numpy as np

from include.experiment.constants import CONTROL_OFFSETS_MS, TIMING_OFFSETS_MS
from src.experiment.analyze_session import Saccade
from src.experiment.flash_gates import EXCLUDED, FIXATION, SACCADE, classify_flash, has_tracking_gap

# 200 Hz trace, 2 s: fixates at 0.1, jumps to 0.8 between t=1.00 and t=1.07 (a 0.7-wide, 70 ms saccade).
T = np.arange(0.0, 2.0, 0.005)
X = np.where(T < 1.0, 0.1, np.where(T > 1.07, 0.8, 0.1 + 0.7 * (T - 1.0) / 0.07))
SACC = Saccade(onset=1.0, offset=1.07, x0=0.1, x1=0.8, peak_speed=10.0)
GO_CUE = 0.75  # latency 250 ms


def test_flash_during_the_movement_is_a_saccade_flash():
    r = classify_flash(1.035, [SACC], T, X, GO_CUE, 0.7)
    assert r.condition == SACCADE and r.latency_ms == 250


def test_flash_long_after_landing_is_fixation():
    r = classify_flash(1.5, [SACC], T, X, GO_CUE, 0.7)
    assert r.condition == FIXATION


def test_flash_just_before_onset_is_excluded_not_fixation():
    """Within +-100 ms of a saccade but the eye is still: neither a clean control nor a saccade flash."""
    r = classify_flash(0.95, [SACC], T, X, GO_CUE, 0.7)
    assert r.condition == EXCLUDED and "not moving" in r.reason


def test_anticipation_latency_is_excluded():
    r = classify_flash(1.035, [SACC], T, X, go_cue=0.95, median_amplitude=0.7)  # 50 ms latency
    assert r.condition == EXCLUDED and "latency" in r.reason


def test_odd_amplitude_is_excluded():
    r = classify_flash(1.035, [SACC], T, X, GO_CUE, median_amplitude=2.0)  # this one is 0.35x the median
    assert r.condition == EXCLUDED and "amplitude" in r.reason


def test_tracking_gap_excludes_a_fixation_flash():
    keep = (T < 1.4) | (T > 1.6)  # 200 ms hole around t = 1.5
    t, x = T[keep], X[keep]
    assert has_tracking_gap(t, 1.5)
    assert classify_flash(1.5, [SACC], t, x, GO_CUE, 0.7).condition == EXCLUDED


def test_no_saccades_excludes_everything():
    assert classify_flash(1.0, [], T, X).condition == EXCLUDED


def test_control_offsets_are_late_enough_to_follow_the_saccade():
    """Typical reaction ~250 ms + a ~75 ms saccade + the 100 ms fixation exclusion, with an SD of ~70 ms of slack."""
    assert CONTROL_OFFSETS_MS and min(CONTROL_OFFSETS_MS) >= 300
    assert set(CONTROL_OFFSETS_MS).isdisjoint(TIMING_OFFSETS_MS)

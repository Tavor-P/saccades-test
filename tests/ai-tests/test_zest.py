import random

import pytest

from src.experiment.zest import ZestStaircase, psychometric_function


def test_converges_toward_a_known_threshold():
    random.seed(0)
    zest = ZestStaircase()
    true_threshold = 0.05
    for _ in range(80):
        contrast = zest.next_contrast()
        p = psychometric_function(contrast, true_threshold, zest.beta, zest.guess_rate, zest.lapse_rate)
        zest.update(contrast, detected=random.random() < p)
    assert zest.threshold_estimate == pytest.approx(true_threshold, rel=0.5)


def test_hit_lowers_and_miss_raises_the_next_contrast_and_the_posterior_stays_normalized():
    for detected, lower in ((True, True), (False, False)):
        zest = ZestStaircase()
        before = zest.next_contrast()
        zest.update(before, detected=detected)
        assert (zest.next_contrast() < before) is lower, detected
    zest = ZestStaircase()
    for detected in (True, False, True, True, False):
        zest.update(zest.next_contrast(), detected)
    assert sum(zest._pdf) == pytest.approx(1.0)


def test_credible_intervals_bracket_the_estimate_and_widen_with_mass_and_the_psychometric_bounds():
    zest = ZestStaircase()
    for detected in (True, False, True, True, False, True):
        zest.update(zest.next_contrast(), detected)
    lo68, hi68 = zest.credible_interval(0.68)
    assert lo68 < zest.threshold_estimate < hi68
    lo95, hi95 = zest.credible_interval(0.95)
    assert lo95 <= lo68 and hi95 >= hi68

    guess_rate, lapse_rate = 0.02, 0.02
    assert psychometric_function(1e-6, 1.0, beta=3.5, guess_rate=guess_rate, lapse_rate=lapse_rate) == pytest.approx(guess_rate, abs=1e-3)
    assert psychometric_function(1e6, 1.0, beta=3.5, guess_rate=guess_rate, lapse_rate=lapse_rate) == pytest.approx(1 - lapse_rate, abs=1e-6)

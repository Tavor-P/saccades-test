"""Relations between tuning constants that must always hold. A violated relation usually means a
recent retune broke an assumption elsewhere (the tuning comments in constants.py say which)."""

import math

from include.experiment import constants as c
from include.eye_tracking import constants as g
from src.experiment.simulation import FRAME_HZ

FRAME_S = 1 / FRAME_HZ
MIN_PLAUSIBLE_REACTION_MS = 100  # no human starts a saccade faster than this after a cue


def test_dead_zone_leaves_room_for_both_side_zones():
    assert 0 < g.GAZE_DEAD_ZONE < 0.5


def test_timing_windows_are_ordered():
    assert 0 < c.SACCADE_ONSET_STABILITY_MS < c.GAZE_LANDING_STABILITY_MS < c.SACCADE_TIMEOUT_MS
    assert c.FOREPERIOD_MIN_MS < c.FOREPERIOD_MAX_MS
    assert c.RESPONSE_WINDOW_MS > c.GRATING_DURATION_FRAMES * FRAME_S * 1000


def test_no_flash_can_be_scheduled_before_the_trial_starts():
    """flash time = trial start + avg reaction time + offset. Even for a fast reactor the offset
    must not push it before the target appears."""
    assert MIN_PLAUSIBLE_REACTION_MS + min(c.TIMING_OFFSETS_MS) > 0
    assert c.DEFAULT_REACTION_TIME_MS + min(c.TIMING_OFFSETS_MS) > 0


def test_trial_count_limits_are_consistent():
    assert c.MAX_SACCADE_TRIALS_TEST >= c.ZEST_MIN_VALID_TRIALS_TEST
    assert c.MAX_SACCADE_TRIALS_REAL >= c.ZEST_MIN_VALID_TRIALS_REAL
    assert c.NUM_RT_TEST_TRIALS_TEST >= 1 and c.NUM_RT_TEST_TRIALS_REAL >= 1


def test_contrast_ranges_are_ordered_and_contain_the_practice_contrast():
    assert c.ZEST_LOG_CONTRAST_MIN < c.ZEST_LOG_CONTRAST_MAX
    assert c.ZEST_LOG_CONTRAST_MIN <= math.log10(c.PRACTICE_CONTRAST) <= c.ZEST_LOG_CONTRAST_MAX
    assert 0 < c.HARDWARE_CONTRAST_FLOOR < 10**c.ZEST_LOG_CONTRAST_MAX


def test_calibration_discards_a_round_only_if_it_has_one_to_spare():
    assert c.CALIBRATION_ROUNDS >= 3  # round 1 discarded, 2-3 averaged (see calibration.py)


def test_tracking_loss_window_covers_at_least_a_frame():
    """The loss check samples once per tick (frame); a narrower window could never contain a tick."""
    assert c.TRACKING_LOSS_WINDOW_MS >= FRAME_S * 1000


def test_neon_staleness_is_longer_than_a_gap_and_calibration_window_fills_quickly():
    assert g.NEON_GAP_THRESHOLD_MS / 1000 < g.NEON_STALE_TIMEOUT_S
    neon_rate_hz = 100  # conservative lower bound on what the Neon actually delivers
    assert g.CALIBRATION_SAMPLE_WINDOW / neon_rate_hz < 1.0  # participants shouldn't wait >1 s per target


def test_smoothing_parameters_are_valid():
    assert 0 < g.POSITION_SMOOTHING <= 1


def test_onset_displacement_is_between_fixation_noise_and_the_old_zone_exit():
    """Noise on Neon is a few hundredths of the dot-to-cross span; the old zone rule fired at 0.5 - GAZE_DEAD_ZONE of travel."""
    assert 0.03 <= c.SACCADE_ONSET_DISPLACEMENT < 0.5 - g.GAZE_DEAD_ZONE
    assert 0 < c.ONSET_BASELINE_WINDOW_MS < 1000
    assert c.SACCADE_ONSET_DISPLACEMENT < c.ONSET_BASELINE_NEAR_SOURCE < 0.5

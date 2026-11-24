"""Scheduled mode with its default saccade-phase setup: fixed probe contrasts,
bell-curve flash times centred on the average RT (cut to the flash window), a random
+/-TIMING_JITTER_MS bounce on the control flashes, and a fixed trial count."""

import pytest
from test_session import (
    _complete_calibration_and_rt_test,
    _enter_trial_active,
    _fake_valid_saccade_results,
    _make_session,
    _set_single_trial,
)

from include.experiment.constants import (
    CONTROL_OFFSETS_MS,
    FLASH_SPREAD_SD_MS,
    FLASH_WINDOW_MAX_MS,
    FLASH_WINDOW_MIN_MS,
    SACCADE_FIXED_CONTRASTS,
    SACCADE_FIXED_TRIALS_REAL,
    SACCADE_FIXED_TRIALS_TEST,
    TIMING_JITTER_MS,
    TIMING_OFFSETS_MS,
)
from src.experiment.trial_mechanics import spread_flash_delay_ms


def test_the_shipped_defaults_use_fixed_contrasts_and_every_level_is_drawn(fake_gaze, fake_time):
    assert SACCADE_FIXED_CONTRASTS and TIMING_JITTER_MS > 0
    assert SACCADE_FIXED_TRIALS_REAL >= SACCADE_FIXED_TRIALS_TEST > 0
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    seen = set()
    for index in range(2 * len(SACCADE_FIXED_CONTRASTS)):
        _set_single_trial(session, index=index)
        _enter_trial_active(session, fake_time)
        fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
        session.tick()  # the scheduled flash fires and picks its contrast
        assert session._trial_contrast in SACCADE_FIXED_CONTRASTS
        seen.add(session._trial_contrast)
    assert seen == set(SACCADE_FIXED_CONTRASTS)  # drawn without repeats, so every level shows up




def test_control_flash_time_has_a_random_bounce_within_the_configured_bound(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    nominal_offset = CONTROL_OFFSETS_MS[0]
    delays = []
    for index in range(40):
        _set_single_trial(session, index=index, timing_offset_ms=nominal_offset)
        _enter_trial_active(session, fake_time)
        delays.append(session._trial_scheduled_delay_ms - session._avg_reaction_time_ms - nominal_offset)
    assert all(abs(d) <= TIMING_JITTER_MS + 1e-9 for d in delays)
    assert max(delays) - min(delays) > TIMING_JITTER_MS / 2  # actually random, not a constant


class _Edge:
    """rng stub: uniform(a, b) returns a (low=True) or b, to get a slice's edges."""

    def __init__(self, low: bool):
        self.low = low

    def uniform(self, a, b):
        return a if self.low else b


def _slice_ms(label, center):
    args = (label, TIMING_OFFSETS_MS, center, FLASH_SPREAD_SD_MS, FLASH_WINDOW_MIN_MS, FLASH_WINDOW_MAX_MS)
    return spread_flash_delay_ms(*args, rng=_Edge(True)), spread_flash_delay_ms(*args, rng=_Edge(False))


def test_bell_curve_flashes_cluster_around_the_average_rt(fake_gaze, fake_time):
    """Each TIMING_OFFSETS_MS label is one equal-probability slice of a normal curve centred on the average RT, cut to
    the flash window; the middle slices are narrow (dense) and the outer ones wide (sparse)."""
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    labels = sorted(TIMING_OFFSETS_MS)
    delays = []
    for index in range(30 * len(labels)):
        label = labels[index % len(labels)]
        _set_single_trial(session, index=index, timing_offset_ms=label)
        _enter_trial_active(session, fake_time)
        center = session._avg_reaction_time_ms
        lo, hi = _slice_ms(label, center)
        d = session._trial_scheduled_delay_ms
        assert lo - 1e-6 <= d <= hi + 1e-6, (label, lo, d, hi)
        delays.append(d - center)
    assert all(FLASH_WINDOW_MIN_MS - 1e-6 <= d + center <= FLASH_WINDOW_MAX_MS + 1e-6 for d in delays)
    near = sum(abs(d) <= FLASH_SPREAD_SD_MS for d in delays) / len(delays)
    assert 0.55 < near < 0.8  # about 68 % within one SD, as for a bell curve
def test_spread_flash_delay_ms_slices_and_the_dialog_spread_overrides_the_default_width(fake_gaze, fake_time):
    labels = (-80, 0, 80)
    args = (200.0, 80.0, 0.0, 500.0)
    assert spread_flash_delay_ms(-80, labels, *args, rng=_Edge(True)) == pytest.approx(0.0, abs=1e-6)
    assert spread_flash_delay_ms(80, labels, *args, rng=_Edge(False)) == pytest.approx(500.0, abs=1e-6)
    mid_lo = spread_flash_delay_ms(0, labels, *args, rng=_Edge(True))
    mid_hi = spread_flash_delay_ms(0, labels, *args, rng=_Edge(False))
    assert mid_lo < 200.0 < mid_hi and mid_hi - mid_lo < 100  # the middle third is narrow, around the centre
    assert spread_flash_delay_ms(0, labels, 200.0, 0.0, 0.0, 500.0) == 200.0  # SD 0 = exactly the average RT

    # A narrower curve from the startup dialog keeps every flash closer to the average RT than the default would.
    session = _make_session(fake_gaze, flash_spread_sd_ms=20)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session._flash_spread_sd_ms == 20
    label = min(TIMING_OFFSETS_MS)  # the earliest slice: furthest from the centre
    for index in range(10):
        _set_single_trial(session, index=index, timing_offset_ms=label)
        _enter_trial_active(session, fake_time)
        center = session._avg_reaction_time_ms
        assert 0 < center - session._trial_scheduled_delay_ms < 4 * 20  # default SD 80 would put this slice far earlier


class _SummaryLogger:
    def __init__(self):
        self.summaries = {}

    def log(self, result) -> None:
        pass

    def set_phase_summary(self, phase, summary) -> None:
        self.summaries[phase] = summary


def test_constant_stimuli_stop_by_count_and_write_the_count_to_the_summary(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True)
    session._results = _fake_valid_saccade_results(SACCADE_FIXED_TRIALS_TEST - 1)
    assert not session._should_stop_main_block()
    session._results = _fake_valid_saccade_results(SACCADE_FIXED_TRIALS_TEST)
    assert session._should_stop_main_block()
    logger = _SummaryLogger()
    session._logger = logger
    session._log_phase_summary()
    summary = logger.summaries["saccade"]
    assert summary["completed_trials"] == summary["valid_trials"] == SACCADE_FIXED_TRIALS_TEST
    assert "hreshold" not in session._summary()

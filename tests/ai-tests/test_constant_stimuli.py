"""Scheduled mode with its default saccade-phase setup: fixed probe contrasts (no live staircase),
a random +/-TIMING_JITTER_MS bounce on every trial's flash time, and a fixed trial count."""

from include.experiment.constants import (
    SACCADE_FIXED_CONTRASTS,
    SACCADE_FIXED_TRIALS_REAL,
    SACCADE_FIXED_TRIALS_TEST,
    TIMING_JITTER_MS,
)
from test_session import (
    _complete_calibration_and_rt_test,
    _enter_trial_active,
    _fake_valid_saccade_results,
    _make_session,
    _set_single_trial,
)


def test_the_shipped_defaults_use_fixed_contrasts_and_jitter():
    assert SACCADE_FIXED_CONTRASTS and TIMING_JITTER_MS > 0
    assert SACCADE_FIXED_TRIALS_REAL >= SACCADE_FIXED_TRIALS_TEST > 0


def test_flash_contrast_comes_from_the_fixed_set(fake_gaze, fake_time):
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


def test_flash_time_has_a_random_bounce_within_the_configured_bound(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    nominal_offset = 20
    delays = []
    for index in range(40):
        _set_single_trial(session, index=index, timing_offset_ms=nominal_offset)
        _enter_trial_active(session, fake_time)
        delays.append(session._trial_scheduled_delay_ms - session._avg_reaction_time_ms - nominal_offset)
    assert all(abs(d) <= TIMING_JITTER_MS + 1e-9 for d in delays)
    assert max(delays) - min(delays) > TIMING_JITTER_MS / 2  # actually random, not a constant


def test_main_block_stops_by_trial_count_not_by_the_staircase(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True)
    session._results = _fake_valid_saccade_results(SACCADE_FIXED_TRIALS_TEST - 1)
    assert not session._should_stop_main_block()
    session._results = _fake_valid_saccade_results(SACCADE_FIXED_TRIALS_TEST)
    assert session._should_stop_main_block()


class _SummaryLogger:
    def __init__(self):
        self.summaries = {}

    def log(self, result) -> None:
        pass

    def set_phase_summary(self, phase, summary) -> None:
        self.summaries[phase] = summary


def test_constant_stimuli_report_no_staircase_threshold(fake_gaze):
    """The staircase is never updated with fixed contrasts, so its 'estimate' is just the prior and must not be reported as a result."""
    session = _make_session(fake_gaze, test_mode=True)
    logger = _SummaryLogger()
    session._logger = logger
    session._results = _fake_valid_saccade_results(SACCADE_FIXED_TRIALS_TEST)
    session._log_phase_summary()
    summary = logger.summaries["saccade"]
    assert summary["threshold_estimate"] is None and summary["credible_interval_68"] is None and summary["threshold_at_grid_bound"] is None
    assert summary["completed_trials"] == SACCADE_FIXED_TRIALS_TEST
    assert "hreshold" not in session._summary()


def test_staircase_mode_still_reports_its_threshold(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True, fixed_contrasts=())
    logger = _SummaryLogger()
    session._logger = logger
    session._log_phase_summary()
    assert 0 < logger.summaries["saccade"]["threshold_estimate"] < 1
    assert "Threshold estimate" in session._summary()

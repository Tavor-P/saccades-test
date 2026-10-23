"""Gaze-triggered ("onset") flash mode (FLASH_TRIGGER, the default): no reaction-time test and no
prediction - the flash fires the tick the onset detector confirms a saccade. The older scheduled
mode keeps its own tests in test_session.py (they opt in with the `scheduled_flash_mode` fixture)."""

import pytest

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    GAZE_LANDING_STABILITY_MS,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
)
from include.experiment.types import Orientation
from test_session import (
    _complete_calibration,
    _enter_trial_active,
    _make_session,
    _saccade_results,
    _set_single_trial,
)


pytestmark = pytest.mark.usefixtures("onset_flash_mode")


def _gaze_leaves_source(session, fake_gaze, fake_time) -> None:
    """Gaze leaves the source zone (mid-flight, in CENTER) and the onset debounce elapses."""
    fake_gaze.zone = GazeZone.CENTER
    session.tick()  # starts the onset timer
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset confirmed


def _trial_after_calibration(fake_gaze, fake_time, **trial_spec):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    _set_single_trial(session, **trial_spec)
    _enter_trial_active(session, fake_time)
    return session


def test_default_mode_is_gaze_triggered_and_skips_the_rt_test(fake_gaze):
    session = _make_session(fake_gaze)
    assert session._closed_loop
    _complete_calibration(session)
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"
    assert not [r for r in session.results if r.phase == "rt_test"]
    assert session._avg_reaction_time_ms is None


def test_nothing_flashes_until_the_saccade_starts(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time)
    fake_time(2.0)  # far beyond any reaction-time schedule
    session.tick()
    assert not session.render_state()["grating"]["visible"]
    assert not session._flash_decided


def test_flash_fires_the_tick_onset_is_confirmed(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()  # onset timer starts: too early to flash
    assert not session.render_state()["grating"]["visible"]
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset confirmed -> flash on this very frame
    assert session.render_state()["grating"]["visible"]
    assert session._flash_scheduled_at == pytest.approx(session._clock.now())


def test_flash_waits_for_the_configured_delay_after_onset(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time, timing_offset_ms=30)
    _gaze_leaves_source(session, fake_gaze, fake_time)
    assert not session._flash_decided  # onset confirmed, but the 30 ms delay hasn't elapsed
    fake_time(0.031)
    session.tick()
    assert session._flash_decided
    assert session.render_state()["grating"]["visible"]


def test_catch_trial_opens_the_response_window_at_onset_without_a_grating(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time, grating_shown=False, orientation=None)
    _gaze_leaves_source(session, fake_gaze, fake_time)
    assert session._flash_decided and session._response_window_open
    assert not session.render_state()["grating"]["visible"]


def test_trial_times_out_without_a_flash_if_gaze_never_leaves(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time)
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()
    rows = _saccade_results(session)
    assert [r.outcome for r in rows] == ["timeout"]
    assert not rows[0].grating_shown


def test_flash_during_the_saccade_is_valid_and_logged_without_rt_prediction(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time, orientation=Orientation.VERTICAL)
    _gaze_leaves_source(session, fake_gaze, fake_time)  # flash fires here, gaze still in transit
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()  # landing timer starts
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()  # landed
    session.on_response_key(Orientation.VERTICAL)
    session.tick()  # responded -> trial finalises
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()
    row = _saccade_results(session)[0]
    assert row.grating_shown and row.outcome == "correct"
    assert row.flash_during_saccade is True
    assert row.avg_rt_used_ms is None
    assert row.scheduled_delay_ms == 0.0
    assert row.flash_relative_to_onset_ms is not None and row.flash_relative_to_onset_ms >= 0


def test_the_rolling_reaction_time_average_is_not_used_in_gaze_triggered_mode(fake_gaze, fake_time):
    session = _trial_after_calibration(fake_gaze, fake_time)
    _gaze_leaves_source(session, fake_gaze, fake_time)
    session._update_rt_tracking()
    assert session._avg_reaction_time_ms is None
    assert not session._recent_rt_ms

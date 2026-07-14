import pytest

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    GRATING_POSITION,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
)
from include.experiment.types import Target, TrialSpec
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession


class _NullLogger:
    """Swallows .log() calls - tests check session.results instead of a real
    CSV, so they don't need filesystem isolation."""

    def log(self, result) -> None:
        pass


def _make_session(fake_gaze) -> ExperimentSession:
    return ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock())


def _complete_calibration_and_enter_foreperiod(session) -> None:
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT
    session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
    session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
    session.on_space()  # CALIBRATE_RIGHT -> FOREPERIOD (+ gaze.calibrate() call)


def _enter_trial_active(session, fake_time) -> None:
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"


def _trigger_onset_and_landing(session, fake_gaze, fake_time, target_zone) -> None:
    """Moves gaze straight to the target zone and holds it there long enough
    to clear both the onset-debounce and landing-stability windows. Per the
    state machine these two checks are independent, so holding gaze in the
    target zone continuously for the longer of the two windows fires both at
    once (the trial doesn't finalize yet though - see the module docstring
    tests below for why)."""
    fake_gaze.zone = target_zone
    session.tick()  # starts both the onset and landing stability timers
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()  # onset fires (grating maybe flashes) and landing fires


def test_calibration_flow_calls_gaze_calibrate(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration_and_enter_foreperiod(session)
    assert fake_gaze.calibrated == (0.5, 0.5, 0.5)  # FakeGazeSource always reports ratio 0.5
    assert session.calibration_ratios == (0.5, 0.5, 0.5)
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"


def test_calibration_starts_a_fresh_sample_window_for_each_target(fake_gaze):
    # Regression test: without resetting the gaze source's rolling ratio
    # window before each of the three calibration targets, the left/center/
    # right averages could be contaminated by stale samples from whatever the
    # participant was looking at previously, collapsing the calibration span
    # toward zero.
    session = _make_session(fake_gaze)
    assert fake_gaze.calibration_samples_begun == 0

    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT
    assert fake_gaze.calibration_samples_begun == 1  # fresh window before the dot

    session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
    assert fake_gaze.calibration_samples_begun == 2  # fresh window before the center target

    session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
    assert fake_gaze.calibration_samples_begun == 3  # fresh window before the cross

    session.on_space()  # CALIBRATE_RIGHT -> FOREPERIOD
    assert fake_gaze.calibration_samples_begun == 3  # no new target after this


def test_hit_scores_and_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True)]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast is not None  # grating was actually shown
    session.on_space()  # respond within the window
    session.tick()  # now landed + responded -> finalizes

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "hit"
    assert result.grating_shown is True
    assert session._zest.threshold_estimate != initial_threshold


def test_miss_scores_and_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True)]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)  # let the response window expire; never respond
    session.tick()  # window closes -> finalizes (already landed)

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "miss"
    assert session._zest.threshold_estimate != initial_threshold


def test_false_alarm_does_not_update_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_space()  # guess during the response window even though nothing was shown
    session.tick()

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "false_alarm"
    assert result.grating_shown is False
    assert session._zest.threshold_estimate == initial_threshold


def test_correct_rejection_does_not_update_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "correct_rejection"
    assert session._zest.threshold_estimate == initial_threshold


def test_timeout_when_gaze_never_leaves_the_source_zone(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True)]
    fake_gaze.zone = GazeZone.LEFT  # matches Target.DOT's zone - never looks away
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)

    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "timeout"
    assert result.grating_shown is False
    assert result.contrast is None


def test_practice_trial_is_not_logged_or_staircased(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, practice=True)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast == PRACTICE_CONTRAST  # fixed, not drawn from the staircase
    session.on_space()
    session.tick()

    assert session.results == []  # practice trials never get logged
    assert session._zest.threshold_estimate == initial_threshold  # staircase untouched
    assert session.render_state()["hud"]["phase"] == "COMPLETE"


def test_source_symbol_hides_the_instant_the_target_appears(fake_gaze, fake_time):
    # Regression test for the "two fully-opaque circles on screen at once"
    # confusion: the source's `visible` flag must flip off in the same tick
    # the target's flips on, not wait for gaze to land on the target.
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["dot"]["visible"] is True  # source lit during the foreperiod

    _enter_trial_active(session, fake_time)
    state = session.render_state()
    assert state["cross"]["visible"] is True  # target just appeared
    assert state["dot"]["visible"] is False  # source already hidden, before any landing


def test_gaze_indicator_hidden_before_calibration(fake_gaze):
    session = _make_session(fake_gaze)
    fake_gaze.zone = GazeZone.LEFT
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_visible_during_real_trials_not_just_practice(fake_gaze, fake_time):
    # Shown throughout the saccade phase now, not gated to practice trials -
    # so tracking quality can be checked at any point in a real session.
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    fake_gaze.zone = GazeZone.LEFT
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is True


def test_gaze_indicator_hidden_when_face_not_found(fake_gaze):
    session = _make_session(fake_gaze)
    fake_gaze.face_found = False
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_hidden_when_zone_is_unknown(fake_gaze):
    session = _make_session(fake_gaze)
    fake_gaze.zone = GazeZone.UNKNOWN
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_snaps_to_the_classified_zone(fake_gaze):
    # Directly mirrors the classification onset/landing detection relies on,
    # rather than interpolating a continuous position from the noisier raw
    # ratio - if this looks wrong, the classification itself is wrong.
    session = _make_session(fake_gaze)
    _complete_calibration_and_enter_foreperiod(session)

    fake_gaze.zone = GazeZone.LEFT
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["visible"] is True
    assert indicator["x"] == pytest.approx(DOT_POSITION[0])
    assert indicator["y"] == pytest.approx(DOT_POSITION[1])

    fake_gaze.zone = GazeZone.RIGHT
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["x"] == pytest.approx(CROSS_POSITION[0])

    fake_gaze.zone = GazeZone.CENTER
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["x"] == pytest.approx(GRATING_POSITION[0])

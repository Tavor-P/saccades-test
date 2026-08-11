import math

import pytest

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
    ZEST_LOG_CONTRAST_MAX,
)
from include.experiment.types import Orientation, Target, TrialSpec
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession


class _NullLogger:
    """Swallows .log() calls - tests check session.results instead of a real
    CSV, so they don't need filesystem isolation."""

    def log(self, result) -> None:
        pass


def _make_session(fake_gaze, show_gaze_indicator: bool = False) -> ExperimentSession:
    return ExperimentSession(
        fake_gaze, logger=_NullLogger(), clock=PausableClock(), show_gaze_indicator=show_gaze_indicator
    )


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


def test_contrast_floor_overrides_the_zest_staircase_minimum(fake_gaze):
    # See the matching test in test_presaccade_session.py for why the
    # midpoint is the right thing to check.
    floor = 0.2
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), contrast_floor=floor)
    expected = 10 ** ((math.log10(floor) + ZEST_LOG_CONTRAST_MAX) / 2)
    assert session._zest.next_contrast() == pytest.approx(expected)


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


def test_correct_orientation_scores_and_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast is not None  # grating was actually shown
    session.on_response_key(Orientation.VERTICAL)  # correctly reports the shown orientation
    session.tick()  # now landed + responded -> finalizes

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "correct"
    assert result.grating_shown is True
    assert session._zest.threshold_estimate != initial_threshold


def test_incorrect_orientation_scores_and_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Orientation.HORIZONTAL)  # wrong guess
    session.tick()

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "incorrect"
    assert session._zest.threshold_estimate != initial_threshold


def test_miss_scores_and_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
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
    session.on_response_key(Orientation.VERTICAL)  # guess during the response window even though nothing was shown
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
        TrialSpec(
            index=0,
            source=Target.DOT,
            target=Target.CROSS,
            grating_shown=True,
            orientation=Orientation.VERTICAL,
            practice=True,
        )
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast == PRACTICE_CONTRAST  # fixed, not drawn from the staircase
    session.on_response_key(Orientation.VERTICAL)
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


def test_render_state_reports_the_current_trial_index(fake_gaze):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False),
        TrialSpec(index=1, source=Target.CROSS, target=Target.DOT, grating_shown=False),
    ]
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["hud"]["trial_index"] == 0

    session._trial_index = 1
    session._begin_foreperiod()
    assert session.render_state()["hud"]["trial_index"] == 1


def test_gaze_indicator_hidden_by_default(fake_gaze, fake_time):
    # show_gaze_indicator defaults to False - a real participant shouldn't
    # see this even with valid tracking and completed calibration.
    session = _make_session(fake_gaze)
    fake_gaze.smoothed_position = 0.0
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_hidden_before_calibration(fake_gaze):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    fake_gaze.smoothed_position = 0.0
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_visible_during_real_trials_when_enabled(fake_gaze):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is True


def test_gaze_indicator_hidden_when_face_not_found(fake_gaze):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    fake_gaze.face_found = False
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_hidden_when_smoothed_position_is_none(fake_gaze):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    fake_gaze.smoothed_position = None
    _complete_calibration_and_enter_foreperiod(session)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_tracks_continuous_smoothed_position(fake_gaze):
    # Uses GazeSample.smoothed_position (0=left/dot target, 1=right/cross
    # target) directly rather than snapping to one of three discrete zones -
    # so values between the zone boundaries should land at the matching
    # interpolated point, not jump between fixed positions.
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    _complete_calibration_and_enter_foreperiod(session)

    fake_gaze.smoothed_position = 0.0
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["visible"] is True
    assert indicator["x"] == pytest.approx(DOT_POSITION[0])
    assert indicator["y"] == pytest.approx(DOT_POSITION[1])

    fake_gaze.smoothed_position = 1.0
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["x"] == pytest.approx(CROSS_POSITION[0])

    fake_gaze.smoothed_position = 0.5
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["x"] == pytest.approx((DOT_POSITION[0] + CROSS_POSITION[0]) / 2)

    fake_gaze.smoothed_position = 0.25
    indicator = session.render_state()["gaze_indicator"]
    assert indicator["x"] == pytest.approx(DOT_POSITION[0] + (CROSS_POSITION[0] - DOT_POSITION[0]) * 0.25)


def test_flash_during_saccade_true_when_still_in_transit_at_onset(fake_gaze, fake_time):
    # Gaze reads CENTER (not yet the target zone) at the exact tick onset
    # fires - genuinely still mid-flight per the classifier.
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()

    assert session._flash_during_saccade is True


def test_flash_during_saccade_false_when_already_at_target_at_onset(fake_gaze, fake_time):
    # Gaze jumps straight to the target zone (CROSS's zone is RIGHT) - by the
    # time onset is confirmed, the classifier already thinks the eye has
    # arrived, so this flash isn't really "during" the saccade.
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()

    assert session._flash_during_saccade is False


def test_flash_during_saccade_is_recorded_on_the_logged_result(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [
        TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL)
    ]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)  # jumps straight to target -> False
    session.on_response_key(Orientation.VERTICAL)
    session.tick()  # finalizes

    assert session.results[0].flash_during_saccade is False


def test_flash_during_saccade_is_none_for_catch_trials(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    session._trials = [TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False)]
    _complete_calibration_and_enter_foreperiod(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()  # finalizes as correct_rejection

    assert session.results[0].flash_during_saccade is None



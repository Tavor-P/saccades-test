from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
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
    session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_RIGHT
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
    assert fake_gaze.calibrated == (0.5, 0.5)  # FakeGazeSource always reports ratio 0.5
    assert session.calibration_ratios == (0.5, 0.5)
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"


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

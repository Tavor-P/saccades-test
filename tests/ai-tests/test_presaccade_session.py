import math

import pytest

from include.experiment.constants import (
    FOREPERIOD_MAX_MS,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_TRIALS_PER_PHASE_REAL,
    NUM_TRIALS_PER_PHASE_TEST,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    ZEST_LOG_CONTRAST_MAX,
)
from include.experiment.types import FlashTrialSpec, Orientation
from src.experiment.pausable_clock import PausableClock
from src.experiment.presaccade_session import PresaccadeSession


class _NullLogger:
    def log(self, result) -> None:
        pass


def _make_session() -> PresaccadeSession:
    return PresaccadeSession(logger=_NullLogger(), clock=PausableClock())


def _enter_flash_window(session, fake_time) -> None:
    session.on_space()  # WAITING_TO_START -> FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()  # FOREPERIOD -> FLASH_WINDOW (flash fires here, if scheduled)
    assert session.render_state()["hud"]["phase"] == "FLASH_WINDOW"


def test_correct_orientation_scores_and_updates_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, orientation=Orientation.VERTICAL)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate
    assert session._trial_contrast is not None

    session.on_response_key(Orientation.VERTICAL)  # correctly reports the shown orientation
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()  # window closes -> finalizes

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "correct"
    assert session._zest.threshold_estimate != initial_threshold


def test_incorrect_orientation_scores_and_updates_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, orientation=Orientation.VERTICAL)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    session.on_response_key(Orientation.HORIZONTAL)  # wrong guess
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert len(session.results) == 1
    assert session.results[0].outcome == "incorrect"
    assert session._zest.threshold_estimate != initial_threshold


def test_miss_scores_and_updates_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, orientation=Orientation.VERTICAL)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)  # never respond
    session.tick()

    assert len(session.results) == 1
    assert session.results[0].outcome == "miss"
    assert session._zest.threshold_estimate != initial_threshold


def test_false_alarm_does_not_update_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=False)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    session.on_response_key(Orientation.VERTICAL)  # guessed something despite nothing being shown
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert len(session.results) == 1
    assert session.results[0].outcome == "false_alarm"
    assert session._zest.threshold_estimate == initial_threshold


def test_correct_rejection_does_not_update_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=False)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert len(session.results) == 1
    assert session.results[0].outcome == "correct_rejection"
    assert session._zest.threshold_estimate == initial_threshold


def test_practice_trial_is_not_logged_or_staircased(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, orientation=Orientation.VERTICAL, practice=True)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate
    assert session._trial_contrast == PRACTICE_CONTRAST

    session.on_response_key(Orientation.VERTICAL)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert session.results == []
    assert session._zest.threshold_estimate == initial_threshold
    assert session.render_state()["hud"]["phase"] == "COMPLETE"


def test_contrast_floor_overrides_the_zest_staircase_minimum():
    # A uniform prior's posterior mean log-contrast is the midpoint of the
    # staircase's [min, max] grid, regardless of grid size (arithmetic-series
    # symmetry) - so a custom floor should shift the very first proposed
    # contrast to the midpoint of [floor, ZEST_LOG_CONTRAST_MAX] instead of
    # the built-in default's.
    floor = 0.2
    session = PresaccadeSession(logger=_NullLogger(), clock=PausableClock(), contrast_floor=floor)
    expected = 10 ** ((math.log10(floor) + ZEST_LOG_CONTRAST_MAX) / 2)
    assert session._zest.next_contrast() == pytest.approx(expected)


def test_response_key_before_flash_window_does_nothing(fake_time):
    # Regression guard: on_response_key() must be a no-op outside FLASH_WINDOW,
    # mirroring the old on_space()/_on_response() guard it replaced.
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, orientation=Orientation.VERTICAL)]
    session.on_response_key(Orientation.VERTICAL)  # still WAITING_TO_START
    assert session._responded is False


def test_test_mode_false_uses_the_real_trial_counts_by_default():
    session = PresaccadeSession(logger=_NullLogger(), clock=PausableClock())
    practice = [t for t in session._trials if t.practice]
    real = [t for t in session._trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS_REAL
    assert len(real) == NUM_TRIALS_PER_PHASE_REAL


def test_test_mode_true_uses_the_smaller_trial_counts():
    session = PresaccadeSession(logger=_NullLogger(), clock=PausableClock(), test_mode=True)
    practice = [t for t in session._trials if t.practice]
    real = [t for t in session._trials if not t.practice]
    assert len(practice) == NUM_PRACTICE_TRIALS_TEST
    assert len(real) == NUM_TRIALS_PER_PHASE_TEST


# -- fixation gating and early stop ---------------------------------------------------------


def _gated_session(fake_gaze):
    return PresaccadeSession(logger=_NullLogger(), clock=PausableClock(), gaze=fake_gaze)


def _to_due_flash(session, fake_time):
    session.on_space()
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()


def test_flash_is_held_while_gaze_is_off_center_then_fires_once_stable(fake_time, fake_gaze):
    from include.eye_tracking.types import GazeZone
    from src.experiment.presaccade_session import FIXATION_GATE_STABLE_S

    session = _gated_session(fake_gaze)
    fake_gaze.zone = GazeZone.LEFT
    _to_due_flash(session, fake_time)
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"
    fake_gaze.zone = GazeZone.CENTER
    session.tick()  # starts the stable-center timer
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"
    fake_time(FIXATION_GATE_STABLE_S + 0.01)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "FLASH_WINDOW"
    assert session.gated_trials == 1 and session.gate_timeouts == 0


def test_gate_fails_open_after_the_timeout(fake_time, fake_gaze):
    from include.eye_tracking.types import GazeZone
    from src.experiment.presaccade_session import FIXATION_GATE_TIMEOUT_S

    session = _gated_session(fake_gaze)
    fake_gaze.zone = GazeZone.UNKNOWN
    _to_due_flash(session, fake_time)
    fake_time(FIXATION_GATE_TIMEOUT_S + 0.01)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "FLASH_WINDOW"
    assert session.gate_timeouts == 1


def test_gate_fails_open_immediately_when_the_source_is_unavailable(fake_time, fake_gaze):
    fake_gaze._available = False
    session = _gated_session(fake_gaze)
    _to_due_flash(session, fake_time)
    assert session.render_state()["hud"]["phase"] == "FLASH_WINDOW"


def test_ungated_session_ignores_gaze(fake_time):
    session = _make_session()
    _to_due_flash(session, fake_time)
    assert session.render_state()["hud"]["phase"] == "FLASH_WINDOW"


def test_request_stop_before_the_flash_completes_without_logging(fake_time):
    session = _make_session()
    session.on_space()
    session.request_stop()
    session.tick()
    assert session.render_state()["hud"]["phase"] == "COMPLETE"
    assert session.stopped_early and session.results == []


def test_request_stop_during_a_flash_window_lets_that_trial_finish_then_ends(fake_time):
    session = _make_session()
    _enter_flash_window(session, fake_time)
    session.request_stop()
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "COMPLETE"
    assert session.stopped_early and len(session.results) <= 1


def test_hud_trial_index_counts_logged_trials(fake_time):
    session = _make_session()
    assert session.render_state()["hud"]["trial_index"] == 0

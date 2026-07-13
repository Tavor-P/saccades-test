from include.experiment.constants import FOREPERIOD_MAX_MS, PRACTICE_CONTRAST, RESPONSE_WINDOW_MS
from include.experiment.types import FlashTrialSpec
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


def test_hit_scores_and_updates_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate
    assert session._trial_contrast is not None

    session.on_space()  # respond within the window
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()  # window closes -> finalizes

    assert len(session.results) == 1
    result = session.results[0]
    assert result.outcome == "hit"
    assert session._zest.threshold_estimate != initial_threshold


def test_miss_scores_and_updates_zest(fake_time):
    session = _make_session()
    session._trials = [FlashTrialSpec(index=0, grating_shown=True)]
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

    session.on_space()  # guess
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
    session._trials = [FlashTrialSpec(index=0, grating_shown=True, practice=True)]
    _enter_flash_window(session, fake_time)
    initial_threshold = session._zest.threshold_estimate
    assert session._trial_contrast == PRACTICE_CONTRAST

    session.on_space()
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()

    assert session.results == []
    assert session._zest.threshold_estimate == initial_threshold
    assert session.render_state()["hud"]["phase"] == "COMPLETE"

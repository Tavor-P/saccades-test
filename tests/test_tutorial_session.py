import pytest

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
    TUTORIAL_DRESS_REHEARSAL_ATTEMPTS,
    TUTORIAL_GAZE_PRACTICE_ATTEMPTS,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Orientation, Target
from src.experiment.pausable_clock import PausableClock
from src.experiment.tutorial_session import TutorialSession


def _make_session(fake_gaze) -> TutorialSession:
    return TutorialSession(fake_gaze, PausableClock())


def _complete_calibration(session) -> None:
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    for round_number in range(1, CALIBRATION_ROUNDS + 1):
        session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
        session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
        session.on_space()  # -> next round's CALIBRATE_LEFT, or DEMO_VERTICAL on the last round


def _complete_demos_and_win_quiz(session) -> None:
    session.on_response_key(Orientation.VERTICAL)  # DEMO_VERTICAL -> DEMO_HORIZONTAL
    session.on_response_key(Orientation.HORIZONTAL)  # DEMO_HORIZONTAL -> QUIZ
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        session.on_response_key(session._quiz_orientation)  # always answer correctly


def _run_gaze_practice_attempt(session, fake_gaze, fake_time, land: bool) -> None:
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()  # foreperiod elapses -> reveals this attempt's target
    assert session.render_state()["hud"]["phase"] == "GAZE_PRACTICE_ACTIVE"

    if land:
        target = session._gaze_practice_target
        fake_gaze.zone = GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT
        session.tick()  # starts the landing-stability timer
        fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
        session.tick()  # landing fires
    else:
        fake_gaze.zone = GazeZone.CENTER  # never lands
        fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
        session.tick()  # times out instead


def _reach_dress_rehearsal(session, fake_gaze, fake_time) -> None:
    _complete_calibration(session)
    _complete_demos_and_win_quiz(session)
    for _ in range(TUTORIAL_GAZE_PRACTICE_ATTEMPTS):
        _run_gaze_practice_attempt(session, fake_gaze, fake_time, land=True)
    assert session.render_state()["hud"]["phase"] == "DRESS_FOREPERIOD"


def _run_dress_attempt(session, fake_gaze, fake_time, respond_correctly: bool) -> None:
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()  # foreperiod elapses -> reveals target, DRESS_ACTIVE
    assert session.render_state()["hud"]["phase"] == "DRESS_ACTIVE"

    target = session._dress_target
    fake_gaze.zone = GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT
    session.tick()  # starts both the onset and landing stability timers
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()  # onset fires (grating shown) and landing fires

    orientation = session._dress_orientation
    if not respond_correctly:
        orientation = Orientation.HORIZONTAL if orientation is Orientation.VERTICAL else Orientation.VERTICAL
    session.on_response_key(orientation)
    session.tick()  # now landed + responded -> finalizes


def test_calibration_discards_round_one_and_averages_rounds_two_and_three(fake_gaze):
    round_ratios = [
        (0.1, 0.1, 0.1),  # round 1 - discarded
        (0.2, 0.5, 0.8),  # round 2
        (0.4, 0.5, 0.6),  # round 3
    ]
    calls = iter(ratio for triple in round_ratios for ratio in triple)
    fake_gaze.average_recent_ratio = lambda: next(calls)

    session = _make_session(fake_gaze)
    _complete_calibration(session)

    expected = ((0.2 + 0.4) / 2, (0.5 + 0.5) / 2, (0.8 + 0.6) / 2)
    assert session.calibration_ratios == pytest.approx(expected)
    assert fake_gaze.calibrated == pytest.approx(expected)
    assert session.render_state()["hud"]["phase"] == "DEMO_VERTICAL"


def test_demo_stages_advance_on_any_response_key(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    assert session.render_state()["hud"]["phase"] == "DEMO_VERTICAL"

    session.on_response_key(Orientation.HORIZONTAL)  # not scored - any key advances
    assert session.render_state()["hud"]["phase"] == "DEMO_HORIZONTAL"

    session.on_response_key(Orientation.VERTICAL)
    assert session.render_state()["hud"]["phase"] == "QUIZ"


def test_quiz_streak_resets_on_a_miss_and_completes_at_the_target(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    session.on_response_key(Orientation.VERTICAL)
    session.on_response_key(Orientation.HORIZONTAL)
    assert session.render_state()["hud"]["phase"] == "QUIZ"

    session.on_response_key(session._quiz_orientation)
    session.on_response_key(session._quiz_orientation)
    assert session._quiz_streak == 2

    wrong = Orientation.HORIZONTAL if session._quiz_orientation is Orientation.VERTICAL else Orientation.VERTICAL
    session.on_response_key(wrong)
    assert session._quiz_streak == 0
    assert session.render_state()["feedback_flash"]["color"] == "red"

    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        session.on_response_key(session._quiz_orientation)
    state = session.render_state()
    assert state["hud"]["phase"] == "GAZE_PRACTICE_FOREPERIOD"
    assert state["feedback_flash"]["color"] == "green"


def test_gaze_practice_runs_fixed_attempts_regardless_of_outcome(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    _complete_demos_and_win_quiz(session)
    assert session.render_state()["hud"]["phase"] == "GAZE_PRACTICE_FOREPERIOD"

    for attempt in range(TUTORIAL_GAZE_PRACTICE_ATTEMPTS):
        # Alternate landing successfully and timing out - the attempt count
        # should advance either way, since this stage isn't gated by success.
        _run_gaze_practice_attempt(session, fake_gaze, fake_time, land=(attempt % 2 == 0))
        expected_phase = "DRESS_FOREPERIOD" if attempt == TUTORIAL_GAZE_PRACTICE_ATTEMPTS - 1 else "GAZE_PRACTICE_FOREPERIOD"
        assert session.render_state()["hud"]["phase"] == expected_phase


def test_dress_rehearsal_runs_fixed_attempts_and_resets_contrast_on_a_miss(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_dress_rehearsal(session, fake_gaze, fake_time)

    initial_contrast = session._dress_zest.next_contrast()

    _run_dress_attempt(session, fake_gaze, fake_time, respond_correctly=True)
    assert session.render_state()["hud"]["phase"] == "DRESS_FOREPERIOD"
    assert session._dress_zest.next_contrast() != initial_contrast  # a correct answer moves the staircase

    _run_dress_attempt(session, fake_gaze, fake_time, respond_correctly=False)  # miss -> reset
    assert session.render_state()["hud"]["phase"] == "DRESS_FOREPERIOD"
    assert session._dress_zest.next_contrast() == pytest.approx(initial_contrast)
    assert session.render_state()["feedback_flash"]["color"] == "red"

    for _ in range(2, TUTORIAL_DRESS_REHEARSAL_ATTEMPTS):
        _run_dress_attempt(session, fake_gaze, fake_time, respond_correctly=True)

    assert session.render_state()["hud"]["phase"] == "COMPLETE"

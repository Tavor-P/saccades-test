import pytest

from include.eye_tracking.types import GazeZone
from include.experiment import participant_text as words
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    FOREPERIOD_MAX_MS,
    FOREPERIOD_MIN_MS,
    GAZE_LANDING_STABILITY_MS,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
    TUTORIAL_PRACTICE_ATTEMPTS,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target
from src.experiment.pausable_clock import PausableClock
from src.experiment.tutorial_session import TutorialSession


def _make_session(fake_gaze) -> TutorialSession:
    return TutorialSession(fake_gaze, PausableClock())


def _zone(target: Target) -> GazeZone:
    return GazeZone.LEFT if target is Target.DOT else GazeZone.RIGHT


def _press_through_intro(session) -> None:
    """WAITING_TO_START -> the INTRO screens -> CALIBRATE_LEFT (round 1)."""
    for _ in range(1 + len(words.TUTORIAL_INTRO)):
        session.on_space()
    assert session.render_state()["hud"]["phase"] == "CALIBRATE_LEFT"


def _complete_calibration(session) -> None:
    _press_through_intro(session)
    for _ in range(CALIBRATION_ROUNDS):
        session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
        session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
        session.on_space()  # -> next round's CALIBRATE_LEFT, or QUIZ on the last round


def _win_quiz(session) -> None:
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        session.on_response_key(session._quiz_orientation)  # always answer correctly


def _reach_practice(session) -> None:
    _complete_calibration(session)
    _win_quiz(session)
    assert session.render_state()["hud"]["phase"] == "PRACTICE_INTRO"
    session.on_space()
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"


def _start_attempt(session, fake_time) -> None:
    session.on_space()  # PRACTICE_AWAITING_SPACE -> PRACTICE_FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()  # foreperiod elapses -> the target appears
    assert session.render_state()["hud"]["phase"] == "PRACTICE_ACTIVE"


def _run_attempt(session, fake_gaze, fake_time, reaction_time_s: float = 0.2, respond_correctly: bool = True) -> None:
    """One practice trial in which the participant looks at the target `reaction_time_s` after it appears."""
    _start_attempt(session, fake_time)
    target = session._target
    fake_time(reaction_time_s)
    fake_gaze.zone = _zone(target)  # left the source, on the target
    session.tick()  # starts the onset and landing stability timers
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()  # onset and landing confirmed

    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    session.tick()  # scheduled flash fires (open-loop, like the real saccade phase)

    orientation = session._orientation
    if not respond_correctly:
        orientation = Side.BELOW if orientation is Side.ABOVE else Side.ABOVE
    session.on_response_key(orientation)
    session.tick()  # landed + flashed + responded -> finalizes


def _run_timeout_attempt(session, fake_gaze, fake_time) -> None:
    """A practice trial in which the eye never leaves the source symbol."""
    fake_gaze.zone = _zone(session._source)
    _start_attempt(session, fake_time)
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()


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
    assert session.render_state()["hud"]["phase"] == "QUIZ"


def test_quiz_passes_after_three_in_a_row_and_a_miss_resets_the_streak(fake_gaze):
    assert TUTORIAL_QUIZ_STREAK_TARGET == 3
    session = _make_session(fake_gaze)
    _complete_calibration(session)

    session.on_response_key(session._quiz_orientation)
    session.on_response_key(session._quiz_orientation)
    assert session._quiz_streak == 2

    wrong = Side.BELOW if session._quiz_orientation is Side.ABOVE else Side.ABOVE
    session.on_response_key(wrong)
    assert session._quiz_streak == 0
    assert session.render_state()["hud"]["phase"] == "QUIZ"

    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET - 1):
        session.on_response_key(session._quiz_orientation)
        assert session.render_state()["hud"]["phase"] == "QUIZ"
    session.on_response_key(session._quiz_orientation)
    assert session.render_state()["hud"]["phase"] == "PRACTICE_INTRO"


def test_quiz_explains_the_keys_once(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    state = session.render_state()
    assert "UP" in state["instructions"] and "DOWN" in state["instructions"]
    assert "LEFT" not in state["instructions"] and "RIGHT" not in state["instructions"]
    assert state["dot"]["visible"]  # the line is shown just above/below the dot
    assert state["tutorial_text"]["visible"]
    assert state["grating"]["visible"]


def test_practice_runs_fixed_attempts_regardless_of_outcome(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)

    for attempt in range(TUTORIAL_PRACTICE_ATTEMPTS):
        # Alternate a good trial, a wrong answer and a timeout - the count advances either way.
        kind = attempt % 3
        if kind == 0:
            _run_attempt(session, fake_gaze, fake_time)
        elif kind == 1:
            _run_attempt(session, fake_gaze, fake_time, respond_correctly=False)
        else:
            _run_timeout_attempt(session, fake_gaze, fake_time)
        expected = "COMPLETE" if attempt == TUTORIAL_PRACTICE_ATTEMPTS - 1 else "PRACTICE_AWAITING_SPACE"
        assert session.render_state()["hud"]["phase"] == expected


def test_practice_produces_one_reaction_time_sample_per_trial_with_a_saccade(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    assert session.rt_samples_ms == []

    for _ in range(3):
        _run_attempt(session, fake_gaze, fake_time, reaction_time_s=0.18)
    assert session.rt_samples_ms == pytest.approx([180.0] * 3)

    _run_timeout_attempt(session, fake_gaze, fake_time)  # no saccade, no sample
    assert len(session.rt_samples_ms) == 3


def test_practice_with_no_saccades_gives_no_samples(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    for _ in range(TUTORIAL_PRACTICE_ATTEMPTS):
        _run_timeout_attempt(session, fake_gaze, fake_time)
    assert session.render_state()["hud"]["phase"] == "COMPLETE"
    assert session.rt_samples_ms == []


def test_practice_flash_follows_the_measured_reaction_time(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    for _ in range(3):
        _run_attempt(session, fake_gaze, fake_time, reaction_time_s=0.16)
    _start_attempt(session, fake_time)
    assert session._flash_scheduled_at - session._trial_start == pytest.approx(0.16)


def test_practice_average_updates_after_each_trial_and_carries_into_the_saccade_session(fake_gaze, fake_time):
    from include.experiment.constants import DEFAULT_REACTION_TIME_MS
    from src.experiment.session import ExperimentSession

    session = _make_session(fake_gaze)
    _reach_practice(session)
    scheduled = []
    real_tick_active = session._tick_active

    def spy():
        if not scheduled:
            scheduled.append((session._flash_scheduled_at - session._trial_start) * 1000)
        real_tick_active()

    session._tick_active = spy
    expected = []
    for reaction_s in (0.30, 0.30, 0.30, 0.20, 0.20, 0.20):
        expected.append(session._rt.average)
        scheduled.clear()
        _run_attempt(session, fake_gaze, fake_time, reaction_time_s=reaction_s)
        assert scheduled[0] == pytest.approx(expected[-1])  # each trial is scheduled from the average so far
    assert expected[:4] == pytest.approx([DEFAULT_REACTION_TIME_MS] * 3 + [300.0])  # default until 3 samples, then the median
    assert session._rt.average == pytest.approx(250.0)

    class _L:
        def log(self, result):
            pass

    seeded = ExperimentSession(
        fake_gaze, logger=_L(), clock=session._clock, test_mode=True,
        precomputed_calibration_ratios=(0.3, 0.5, 0.7), precomputed_rt_samples_ms=session.rt_samples_ms,
    )
    assert seeded._avg_reaction_time_ms == pytest.approx(session._rt.average)


def test_practice_flash_is_clearly_visible_and_oriented(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    _start_attempt(session, fake_time)
    fake_time(1.0)
    session.tick()  # the scheduled flash fires
    grating = session.render_state()["grating"]
    assert grating["visible"] and grating["contrast"] >= 0.5
    assert grating["orientation"] is session._orientation


def test_practice_attempt_waits_for_space_then_swaps_500_to_1000_ms_later(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    fake_time(60.0)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"  # never starts on its own

    session.on_space()
    assert session.render_state()["hud"]["phase"] == "PRACTICE_FOREPERIOD"
    assert FOREPERIOD_MIN_MS / 1000 <= session._foreperiod_duration <= FOREPERIOD_MAX_MS / 1000
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.01)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "PRACTICE_ACTIVE"


def test_practice_intro_is_spoken_once_and_later_trials_are_silent(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    intro = session.render_state()["instructions"]
    assert intro == words.RT_TEST_INTRO

    _start_attempt(session, fake_time)
    assert session.render_state()["instructions"] == intro  # unchanged through the trial: not re-spoken
    _run_timeout_attempt(session, fake_gaze, fake_time)
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"
    assert session.render_state()["instructions"] in words.FEEDBACK_MISSED  # then only the verdict, no instructions again


def test_tutorial_strings_are_short_enough_to_narrate(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    texts = [session.render_state()["instructions"]]
    for _ in range(len(words.TUTORIAL_INTRO) + 1):
        session.on_space()
        texts.append(session.render_state()["instructions"])
    for _ in range(CALIBRATION_ROUNDS):
        for _ in range(3):
            session.on_space()
            texts.append(session.render_state()["instructions"])
    texts.append(session.render_state()["instructions"])
    _win_quiz(session)
    texts.append(session.render_state()["instructions"])
    assert all(len(t.split()) <= 15 for t in texts)


def test_complete_state_asks_for_space_to_continue(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    for _ in range(TUTORIAL_PRACTICE_ATTEMPTS):
        _run_timeout_attempt(session, fake_gaze, fake_time)
    assert "SPACE" in session.render_state()["instructions"]
    assert not session.render_state()["dot"]["visible"]


def test_intro_screens_come_first_and_each_needs_a_space(fake_gaze):
    session = _make_session(fake_gaze)
    assert session.render_state()["instructions"] == words.TUTORIAL_WELCOME
    shown = []
    for _ in range(len(words.TUTORIAL_INTRO)):
        session.on_space()
        state = session.render_state()
        assert state["hud"]["phase"] == "INTRO"
        assert state["tutorial_text"]["visible"] and state["tutorial_text"]["text"] == state["instructions"]
        assert not state["dot"]["visible"] and not state["grating"]["visible"]
        shown.append(state["instructions"])
    assert tuple(shown) == words.TUTORIAL_INTRO
    session.on_space()
    assert session.render_state()["hud"]["phase"] == "CALIBRATE_LEFT"


def test_quiz_says_correct_or_not_quite_and_never_repeats_a_string(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    assert session.render_state()["instructions"] == words.QUIZ
    verdicts = []
    for answer_is_correct in (True, True, False, False, True):
        orientation = session._quiz_orientation
        if not answer_is_correct:
            orientation = Side.BELOW if orientation is Side.ABOVE else Side.ABOVE
        session.on_response_key(orientation)
        if session.render_state()["hud"]["phase"] != "QUIZ":
            break
        verdicts.append((answer_is_correct, session.render_state()["instructions"]))
        assert session.render_state()["tutorial_text"]["text"] == words.QUIZ  # the key mapping stays up
    for correct, text in verdicts:
        assert text in (words.FEEDBACK_CORRECT if correct else words.FEEDBACK_WRONG)
    texts = [text for _, text in verdicts]
    assert len(texts) >= 3
    assert all(a != b for a, b in zip(texts, texts[1:]))  # the narrator would not re-speak an equal neighbour


def test_practice_trial_ends_with_a_verdict_and_two_equal_verdicts_differ(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    _run_attempt(session, fake_gaze, fake_time, respond_correctly=True)
    first = session.render_state()["instructions"]
    assert first in words.FEEDBACK_CORRECT
    _run_attempt(session, fake_gaze, fake_time, respond_correctly=True)
    second = session.render_state()["instructions"]
    assert second in words.FEEDBACK_CORRECT and second != first
    _run_attempt(session, fake_gaze, fake_time, respond_correctly=False)
    assert session.render_state()["instructions"] in words.FEEDBACK_WRONG


def test_practice_intro_screen_tells_that_real_lines_are_fainter(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    _win_quiz(session)
    state = session.render_state()
    assert state["hud"]["phase"] == "PRACTICE_INTRO"
    assert state["instructions"] == words.PRACTICE_INTRO and "fainter" in words.PRACTICE_INTRO
    assert not state["grating"]["visible"]


def test_complete_state_says_the_real_test_starts_now(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _reach_practice(session)
    for _ in range(TUTORIAL_PRACTICE_ATTEMPTS):
        _run_timeout_attempt(session, fake_gaze, fake_time)
    assert session.render_state()["instructions"] == words.PRACTICE_DONE

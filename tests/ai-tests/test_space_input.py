"""A SPACE press must never silently disappear: calibration latching, tutorial narration holding, per-phase rules, counters."""

import json

import pytest

from include.experiment.constants import CALIBRATION_ROUNDS, FOREPERIOD_MAX_MS, RESPONSE_WINDOW_MS
from include.experiment.types import Orientation
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession
from src.experiment.space_input import (
    NarrationKeyGate,
    SpaceTally,
    deliver_space,
    leaves_complete_screen,
)
from src.experiment.tutorial_session import TutorialSession

pytestmark = pytest.mark.usefixtures("staircase_no_jitter")


class _NullLogger:
    def log(self, result) -> None:
        pass


def _phase(session) -> str:
    return session.render_state()["hud"]["phase"]


def _make_experiment(fake_gaze):
    return ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), skip_practice_trials=True)


def _make_tutorial(fake_gaze):
    return TutorialSession(fake_gaze, PausableClock())


SESSIONS = pytest.mark.parametrize("make", [_make_experiment, _make_tutorial], ids=["experiment", "tutorial"])


# -- calibration: one press is enough -------------------------------------------------------------------------------------


@SESSIONS
def test_press_before_the_window_fills_completes_the_step_once_it_fills(make, fake_gaze):
    session = make(fake_gaze)
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT
    fake_gaze.calibration_window_full = False

    session.on_space()  # on the dot, but the window is still filling
    assert _phase(session) == "CALIBRATE_LEFT"
    assert session.render_state()["hud"]["calibration_wait_s"] is not None  # visible in the test-mode HUD

    session.tick()
    assert _phase(session) == "CALIBRATE_LEFT"  # still waiting, nothing lost

    fake_gaze.calibration_window_full = True
    session.tick()
    assert _phase(session) == "CALIBRATE_CENTER"  # exactly one step
    assert session.render_state()["hud"]["calibration_wait_s"] is None
    session.tick()
    assert _phase(session) == "CALIBRATE_CENTER"  # the press was consumed, it does not keep advancing


@SESSIONS
def test_one_press_per_target_walks_all_rounds(make, fake_gaze):
    session = make(fake_gaze)
    session.on_space()
    expected = ["CALIBRATE_CENTER", "CALIBRATE_RIGHT", "CALIBRATE_LEFT"] * CALIBRATION_ROUNDS
    for step, next_phase in enumerate(expected, start=1):
        before = _phase(session)
        assert before.startswith("CALIBRATE")
        fake_gaze.calibration_window_full = False
        session.on_space()
        assert _phase(session) == before
        fake_gaze.calibration_window_full = True
        session.tick()
        if step == len(expected):
            assert not _phase(session).startswith("CALIBRATE")  # the last RIGHT press finishes calibration
        else:
            assert _phase(session) == next_phase
    assert fake_gaze.calibrated == (0.5, 0.5, 0.5)


@SESSIONS
def test_a_second_press_while_waiting_is_counted_not_chained(make, fake_gaze):
    session = make(fake_gaze)
    session.on_space()
    fake_gaze.calibration_window_full = False
    session.on_space()
    session.on_space()
    fake_gaze.calibration_window_full = True
    session.tick()
    assert _phase(session) == "CALIBRATE_CENTER"
    assert session.space_tally.as_dict() == {"CALIBRATE_LEFT: already waiting for gaze samples": 1}


@SESSIONS
def test_a_latched_press_restarts_the_window_so_it_holds_only_post_press_samples(make, fake_gaze):
    session = make(fake_gaze)
    session.on_space()
    begun = fake_gaze.calibration_samples_begun
    fake_gaze.calibration_window_full = False
    session.on_space()
    assert fake_gaze.calibration_samples_begun == begun + 1


@SESSIONS
def test_press_with_a_full_window_still_advances_immediately(make, fake_gaze):
    session = make(fake_gaze)
    session.on_space()
    session.on_space()
    assert _phase(session) == "CALIBRATE_CENTER"


def test_latch_waits_visibly_when_tracking_is_lost(fake_gaze, fake_time):
    session = _make_experiment(fake_gaze)
    session.on_space()
    fake_gaze.calibration_window_full = False
    session.on_space()
    fake_time(5.0)
    session.tick()
    assert _phase(session) == "CALIBRATE_LEFT"  # keeps waiting rather than giving up
    assert session.render_state()["hud"]["calibration_wait_s"] == pytest.approx(5.0)


# -- tutorial narration -----------------------------------------------------------------------------------------------------


def test_space_during_narration_is_delivered_after_it():
    gate = NarrationKeyGate()
    assert gate.route(["space"], speaking=True) == (False, [], 0)
    assert gate.route([], speaking=True) == (False, [], 0)  # still held
    assert gate.route([], speaking=False) == (True, [], 0)  # delivered when speech ends
    assert gate.route([], speaking=False) == (False, [], 0)  # exactly once


def test_space_after_narration_passes_straight_through_with_response_keys():
    assert NarrationKeyGate().route(["space", "up"], speaking=False) == (True, ["up"], 0)


def test_response_keys_during_narration_are_dropped_but_space_is_kept():
    gate = NarrationKeyGate()
    assert gate.route(["up", "space"], speaking=True) == (False, [], 0)
    assert gate.route([], speaking=False) == (True, [], 0)


def test_repeated_presses_collapse_to_one_and_are_counted():
    gate = NarrationKeyGate()
    assert gate.route(["space", "space"], speaking=True) == (False, [], 1)
    assert gate.route(["space"], speaking=True) == (False, [], 1)  # pending already
    assert gate.route(["space"], speaking=False) == (True, [], 1)  # pending + this one = one delivery, one duplicate


def test_nothing_is_delivered_when_nothing_was_pressed():
    assert NarrationKeyGate().route([], speaking=False) == (False, [], 0)


def test_same_frame_presses_are_delivered_as_one(fake_gaze):
    session = _make_experiment(fake_gaze)
    assert deliver_space(session, 2) is True
    assert _phase(session) == "CALIBRATE_LEFT"  # not CENTER: the second press did not chain a stage
    assert session.space_tally.as_dict() == {"CALIBRATE_LEFT: duplicate press in the same frame": 1}
    assert deliver_space(session, 0) is False


def test_complete_screen_needs_a_frame_of_its_own():
    assert leaves_complete_screen("COMPLETE", True)
    assert not leaves_complete_screen("COMPLETE", False)
    assert not leaves_complete_screen("TRIAL_ACTIVE", True)  # a press meant for the last trial
    assert not leaves_complete_screen(None, True)


# -- real-run rules: saccade phase ------------------------------------------------------------------------------------------


def _to_awaiting_space(session, fake_gaze):
    """Calibration + RT test done: the first RT-test attempt waits for SPACE."""
    session.on_space()
    for _ in range(3 * CALIBRATION_ROUNDS):
        session.on_space()
    assert _phase(session) == "RT_TEST_AWAITING_SPACE"


def test_space_during_the_foreperiod_neither_restarts_it_nor_starts_the_next_trial(fake_gaze, fake_time):
    session = _make_experiment(fake_gaze)
    _to_awaiting_space(session, fake_gaze)
    session.on_space()  # RT_TEST_AWAITING_SPACE -> RT_TEST_FOREPERIOD
    started = session._foreperiod_start
    fake_time(0.2)
    session.on_space()
    assert _phase(session) == "RT_TEST_FOREPERIOD"
    assert session._foreperiod_start == started
    assert session.space_tally.as_dict() == {"RT_TEST_FOREPERIOD: during foreperiod": 1}

    fake_time(FOREPERIOD_MAX_MS / 1000)
    session.tick()
    assert _phase(session) == "RT_TEST_ACTIVE"
    session.on_space()
    assert _phase(session) == "RT_TEST_ACTIVE"
    assert session.space_tally.as_dict()["RT_TEST_ACTIVE: during trial"] == 1


def test_space_after_answering_starts_the_next_trial_as_soon_as_it_is_ready(fake_gaze, fake_time):
    from include.eye_tracking.types import GazeZone
    from include.experiment.constants import GAZE_LANDING_STABILITY_MS, SACCADE_ONSET_STABILITY_MS

    session = _make_experiment(fake_gaze)
    _to_awaiting_space(session, fake_gaze)
    session._rt_test_target_attempts = 0  # skip straight to the trials
    session._avg_reaction_time_ms = 200.0
    session._rt_seeded = True
    session._begin_next_trial_or_complete()
    assert _phase(session) == "AWAITING_SPACE"
    session.on_space()  # -> FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert _phase(session) == "TRIAL_ACTIVE"

    source_zone = fake_gaze.zone
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    session.tick()  # flash fires
    fake_gaze.zone = GazeZone.RIGHT if source_zone is not GazeZone.RIGHT else GazeZone.LEFT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()

    session.on_space()  # before answering: ignored and counted
    assert session.space_tally.as_dict() == {"TRIAL_ACTIVE: during trial": 1}
    session.on_response_key(Orientation.VERTICAL)
    session.on_space()  # answered: carried to the next trial
    assert _phase(session) == "TRIAL_ACTIVE"
    session.tick()  # trial finalizes -> AWAITING_SPACE -> the carried press starts the next foreperiod
    assert _phase(session) == "FOREPERIOD"
    assert session.space_tally.total == 1
    fake_time(RESPONSE_WINDOW_MS / 1000)  # nothing further is pending
    session.on_space()
    assert session.space_tally.as_dict()["FOREPERIOD: during foreperiod"] == 1


def test_tutorial_space_in_quiz_is_counted(fake_gaze):
    session = _make_tutorial(fake_gaze)
    session.on_space()
    for _ in range(3 * CALIBRATION_ROUNDS):
        session.on_space()
    assert _phase(session) == "QUIZ"
    session.on_space()
    assert session.space_tally.as_dict() == {"QUIZ: not accepted in this phase": 1}


# -- tally -------------------------------------------------------------------------------------------------------------------


def test_tally_summary_names_phase_and_reason():
    tally = SpaceTally(lambda: "FOREPERIOD")
    assert "no ignored" in tally.summary("saccade")
    tally.record("during foreperiod", 2)
    tally.record("pressed before the phase started", 1, phase="(start-up)")
    assert tally.total == 3
    assert tally.as_dict() == {"(start-up): pressed before the phase started": 1, "FOREPERIOD: during foreperiod": 2}
    assert "3 ignored SPACE press(es)" in tally.summary("saccade")


def test_logger_stores_ignored_space_presses(tmp_path, monkeypatch):
    from src.experiment import logger as logger_module

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path, raising=False)
    log = logger_module.ResultLogger("p1")
    log.set_ignored_space_presses("saccade", {"FOREPERIOD: during foreperiod": 2})
    meta = json.loads(next(tmp_path.glob("*/*/session.json")).read_text())
    assert meta["ignored_space_presses"]["saccade"] == {"FOREPERIOD: during foreperiod": 2}

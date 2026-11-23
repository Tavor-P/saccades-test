"""Return-jump trials (dialog "Return jump"): the target jumps back to the start symbol once the out landing is confirmed
(or forced at RETURN_JUMP_LANDING_TIMEOUT_MS), and each trial's one flash goes on the out or the return leg."""

import csv

import pytest

from include.eye_tracking.types import GazeZone
from include.experiment import participant_text as words
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    DOT_POSITION,
    CROSS_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    RETURN_JUMP_DEFAULT,
    RETURN_JUMP_LANDING_TIMEOUT_MS,
    RETURN_LEG_FLASH_FRACTION,
    SACCADE_ONSET_STABILITY_MS,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target, TrialResult, TrialSpec
from helpers import NullLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession
from src.experiment.trial_mechanics import ShuffledBag, flash_leg_pool, line_position
from src.experiment.tutorial_session import TutorialSession


pytestmark = pytest.mark.usefixtures("no_flash_jitter")

AVG_RT_MS = 200.0


def _session(fake_gaze, return_jump: bool = True, leg: str = "out", **kwargs) -> ExperimentSession:
    """A seeded session (average RT 200 ms) waiting for SPACE on one DOT -> CROSS trial whose flash goes on `leg`."""
    session = ExperimentSession(
        fake_gaze, logger=NullLogger(), clock=PausableClock(), test_mode=True,
        precomputed_calibration_ratios=(0.3, 0.5, 0.7), precomputed_rt_samples_ms=[AVG_RT_MS] * 3,
        return_jump=return_jump, **kwargs,
    )
    session.on_space()  # WAITING_TO_START -> the first main trial
    session._current_trial_spec = TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, orientation=Side.ABOVE)
    session._begin_foreperiod()
    if return_jump:
        session._flash_leg_bag = ShuffledBag([leg])
    fake_gaze.zone = GazeZone.LEFT  # on the dot
    return session


def _go(session, fake_time) -> float:
    """SPACE -> foreperiod -> go cue; returns the go-cue time."""
    session.on_space()
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"
    return session._trial_start_time


def _saccade_to(session, fake_gaze, fake_time, zone: GazeZone) -> None:
    """The eye leaves (onset confirmed), then arrives at `zone` and holds there (landing confirmed)."""
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_gaze.zone = zone
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()


def _symbols(session) -> tuple[bool, bool]:
    state = session.render_state()
    return state["dot"]["visible"], state["cross"]["visible"]


def _saccade_rows(session) -> list[TrialResult]:
    return [r for r in session.results if r.phase == "saccade"]


def test_return_jump_is_on_by_default_but_the_session_class_defaults_off(fake_gaze):
    assert RETURN_JUMP_DEFAULT is True  # the dialog / simulator default
    plain = ExperimentSession(fake_gaze, logger=NullLogger(), clock=PausableClock())
    assert plain._return_jump is False and plain._flash_leg_bag is None


def test_target_jumps_back_after_the_out_landing_and_the_out_flash_then_one_answer_finishes_the_trial(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="out")
    go = _go(session, fake_time)
    assert _symbols(session) == (False, True)  # go cue: the cross is the target

    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)  # landing confirmed on the tick the flash fires
    assert session._landed and session._grating_actually_shown
    assert session._jump_at is None  # the line is still on screen: the jump never overlaps the out-leg flash
    session.tick()  # the flash frame is over -> jump back
    jump = session._jump_at
    assert jump is not None and session._jump_forced is False
    assert _symbols(session) == (True, False)  # the dot is back, the cross is gone

    fake_time(0.12)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.LEFT)  # look back, 120 ms after the jump
    assert session._return_landed and len(session.return_rt_samples_ms) == 1
    session.on_response_key(Side.ABOVE)
    session.on_response_key(Side.BELOW)  # a second press is ignored: one answer per trial
    session.tick()
    session.on_response_key(Side.BELOW)  # after the trial: ignored
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"

    [row] = _saccade_rows(session)
    assert row.flash_leg == "out" and row.response_orientation is Side.ABOVE and row.outcome == "correct"
    assert row.scheduled_delay_ms == pytest.approx(AVG_RT_MS)
    assert row.jump_back_ms_after_go_cue == pytest.approx((jump - go) * 1000)
    assert row.jump_back_forced is False
    assert row.return_reaction_ms == pytest.approx(120.0)
    assert session.return_rt_samples_ms == pytest.approx([120.0])


def test_a_return_leg_flash_is_scheduled_from_the_jump_and_drawn_at_the_start_symbol(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="return")
    go = _go(session, fake_time)
    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert not session._grating_actually_shown  # nothing on the out leg
    jump = session._jump_at
    assert jump is not None and session._jump_forced is False  # no out flash to wait for: jumps at landing
    # No return samples yet: centred on the out-leg average.
    assert session._flash_scheduled_at == pytest.approx(jump + AVG_RT_MS / 1000)

    fake_time(AVG_RT_MS / 1000 + 0.001)
    session.tick()
    line = session.render_state()["grating"]
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, Side.ABOVE, session._geometry))
    assert (line["x"], line["y"]) != pytest.approx(line_position(CROSS_POSITION, Side.ABOVE, session._geometry))

    _saccade_to(session, fake_gaze, fake_time, GazeZone.LEFT)
    session.on_response_key(Side.ABOVE)
    session.tick()
    [row] = _saccade_rows(session)
    assert row.flash_leg == "return" and row.grating_shown
    jump_ms = (jump - go) * 1000
    assert row.jump_back_ms_after_go_cue == pytest.approx(jump_ms)
    # scheduled_delay_ms stays "flash minus go cue" on both legs.
    assert row.scheduled_delay_ms == pytest.approx(jump_ms + AVG_RT_MS)
    assert row.avg_rt_used_ms == pytest.approx(AVG_RT_MS)


def test_return_leg_flash_uses_the_seeded_return_average_once_it_has_enough_samples(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="return", precomputed_return_rt_samples_ms=[150.0, 160.0, 170.0])
    _go(session, fake_time)
    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._flash_scheduled_at == pytest.approx(session._jump_at + 0.160)
    assert session._trial_avg_rt_ms == pytest.approx(160.0)


def test_no_confirmed_landing_forces_the_jump_at_the_timeout_and_counts_it(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="out")
    go = _go(session, fake_time)
    fake_time(0.25)
    session.tick()  # the out flash fires (the eye never moves)
    session.tick()
    assert session._jump_at is None
    fake_time(RETURN_JUMP_LANDING_TIMEOUT_MS / 1000 - 0.3)
    session.tick()
    assert session._jump_at is None  # not yet
    fake_time(0.1)
    session.tick()
    assert session._jump_forced is True and session.forced_jump_backs == 1
    jump = session._jump_at
    assert _symbols(session) == (True, False)

    session.tick()  # the eye is still on the dot: it "lands" there
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    session.on_response_key(Side.ABOVE)
    session.tick()
    [row] = _saccade_rows(session)
    assert row.jump_back_forced is True
    assert row.jump_back_ms_after_go_cue >= RETURN_JUMP_LANDING_TIMEOUT_MS
    assert row.jump_back_ms_after_go_cue == pytest.approx((jump - go) * 1000)
    assert session.return_rt_samples_ms == []  # a forced jump gives no clean return latency
    assert session._return_jump_summary()["forced_jump_backs"] == 1


def test_setting_off_keeps_the_old_trial_no_jump_and_no_return_fields(fake_gaze, fake_time):
    session = _session(fake_gaze, return_jump=False)
    _go(session, fake_time)
    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.tick()
    session.tick()
    assert session._jump_at is None and _symbols(session) == (False, True)  # the cross stays
    session.on_response_key(Side.ABOVE)
    session.tick()
    [row] = _saccade_rows(session)
    assert row.flash_leg == "out"
    assert row.jump_back_ms_after_go_cue is None and row.return_reaction_ms is None and row.jump_back_forced is None
    assert session.forced_jump_backs == 0 and session._return_jump_summary() == {}


def test_flash_leg_pool_splits_exactly_per_pass():
    assert sorted(flash_leg_pool(0.8)) == ["out"] + ["return"] * 4
    assert flash_leg_pool(1.0) == ["return"] and flash_leg_pool(0.0) == ["out"]
    bag = ShuffledBag(flash_leg_pool(RETURN_LEG_FLASH_FRACTION))
    draws = [bag.draw() for _ in range(50)]
    assert draws.count("return") == round(50 * RETURN_LEG_FLASH_FRACTION)


def test_the_new_columns_round_trip_and_old_csvs_without_them_still_load(tmp_path, monkeypatch):
    from src.experiment import logger as logger_module
    from src.experiment.logger import read_trials

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = logger_module.ResultLogger("p1")
    log.log(
        TrialResult(
            index=0, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=50.0,
            grating_shown=True, contrast=0.06, orientation=Side.BELOW, responded=True,
            response_orientation=Side.BELOW, response_time_ms=300.0, outcome="correct",
            flash_leg="return", jump_back_ms_after_go_cue=412.5, return_reaction_ms=180.25, jump_back_forced=False,
        )
    )
    log.close()
    csv_path = next(tmp_path.glob("*/*/trials.csv"))
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["flash_leg"] == "return" and rows[0]["jump_back_forced"] == "no"
    [back] = read_trials(csv_path)
    assert back.flash_leg == "return" and back.jump_back_forced is False
    assert back.jump_back_ms_after_go_cue == pytest.approx(412.5) and back.return_reaction_ms == pytest.approx(180.25, abs=0.1)

    new_columns = ("flash_leg", "jump_back_ms_after_go_cue", "return_reaction_ms", "jump_back_forced")
    old_fields = [name for name in rows[0] if name not in new_columns]
    old_path = tmp_path / "old_trials.csv"
    with open(old_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    [old] = read_trials(old_path)
    assert old.flash_leg is None and old.jump_back_ms_after_go_cue is None
    assert old.return_reaction_ms is None and old.jump_back_forced is None


# -- tutorial practice ---------------------------------------------------------


def _tutorial_at_practice(fake_gaze) -> TutorialSession:
    session = TutorialSession(fake_gaze, PausableClock(), return_jump=True)
    for _ in range(1 + len(session._intro_lines)):
        session.on_space()
    for _ in range(3 * CALIBRATION_ROUNDS):
        session.on_space()
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        session.on_response_key(session._quiz_orientation)
    session.on_space()  # PRACTICE_INTRO -> PRACTICE_AWAITING_SPACE
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"
    return session


def test_tutorial_practice_jumps_back_and_collects_return_latencies(fake_gaze, fake_time):
    session = _tutorial_at_practice(fake_gaze)
    assert words.TUTORIAL_INTRO_RETURN_JUMP in session._intro_lines
    session._flash_leg = "return"
    session._leg_bag = ShuffledBag(["return"])
    fake_gaze.zone = GazeZone.LEFT

    session.on_space()
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "PRACTICE_ACTIVE"
    fake_time(0.2)
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()  # out landing -> jump back at once (the flash is on the return leg)
    assert session._jump_at is not None
    state = session.render_state()
    assert state["dot"]["visible"] and not state["cross"]["visible"]

    fake_time(session._flash_scheduled_at - session._clock.now() + 0.001)
    session.tick()
    line = session.render_state()["grating"]
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, session._orientation, session._geometry))

    fake_gaze.zone = GazeZone.LEFT
    session.tick()
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()
    session.on_response_key(session._orientation)
    session.tick()
    assert len(session.return_rt_samples_ms) == 1 and session.return_rt_samples_ms[0] > 0
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"
    assert session._source is Target.DOT  # every return-jump trial starts on the dot


def test_new_participant_strings_are_short_and_avoid_the_banned_words():
    for text in (words.TUTORIAL_INTRO_RETURN_JUMP, words.PRACTICE_START_RETURN_JUMP, words.FIRST_TRIAL_RETURN_JUMP):
        assert len(text.split()) <= 15
        assert "cross" not in text.lower() and "other dot" not in text.lower()

"""Return-jump trials (dialog "Return jump"): the target jumps back to the start symbol on RETURN_JUMP_TRIGGER (the out
arrival or the out onset; forced at RETURN_JUMP_LANDING_TIMEOUT_MS), and each trial's one flash goes on the out or the return leg."""

import csv

import pytest
from helpers import NullLogger

from include.experiment import participant_text as words
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    RETURN_JUMP_DEFAULT,
    RETURN_JUMP_DWELL_MS,
    RETURN_JUMP_LANDING_TIMEOUT_MS,
    RETURN_LEG_FLASH_FRACTION,
    SACCADE_ONSET_STABILITY_MS,
    TIMING_OFFSETS_MS,
    TUTORIAL_QUIZ_STREAK_TARGET,
)
from include.experiment.types import Side, Target, TrialResult, TrialSpec
from include.eye_tracking.types import GazeSample, GazeZone
from src.experiment import session as session_module
from src.experiment import tutorial_session as tutorial_module
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession
from src.experiment.session_config import SessionConfig, TutorialHandoff
from src.experiment.trial_mechanics import OnsetDetector, RollingReactionTime, ShuffledBag, flash_leg_pool, line_position
from src.experiment.tutorial_session import TutorialSession

pytestmark = pytest.mark.usefixtures("no_flash_jitter")

AVG_RT_MS = 200.0


def _session(
    fake_gaze, return_jump: bool = True, leg: str = "out", precomputed_return_rt_samples_ms: list[float] | None = None
) -> ExperimentSession:
    """A seeded session (average RT 200 ms) waiting for SPACE on one DOT -> CROSS trial whose flash goes on `leg`."""
    session = ExperimentSession(
        fake_gaze, logger=NullLogger(), clock=PausableClock(),
        config=SessionConfig(test_mode=True, return_jump=return_jump),
        handoff=TutorialHandoff(
            calibration_ratios=(0.3, 0.5, 0.7), rt_samples_ms=[AVG_RT_MS] * 3,
            return_rt_samples_ms=precomputed_return_rt_samples_ms,
        ),
    )
    session.on_space()  # WAITING_TO_START -> the first main trial
    session._current_trial_spec = TrialSpec(index=0, source=Target.DOT, target=Target.CROSS, line_side=Side.ABOVE)
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
    assert session._landed and session._line_actually_shown
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
    assert row.flash_leg == "out" and row.answer_side is Side.ABOVE and row.outcome == "correct"
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
    assert not session._line_actually_shown  # nothing on the out leg
    jump = session._jump_at
    assert jump is not None and session._jump_forced is False  # no out flash to wait for: jumps at landing
    # No return samples yet: centred on the out-leg average.
    assert session._flash_scheduled_at == pytest.approx(jump + AVG_RT_MS / 1000)

    fake_time(AVG_RT_MS / 1000 + 0.001)
    session.tick()
    line = session.render_state()["line"]
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, Side.ABOVE, session._geometry))
    assert (line["x"], line["y"]) != pytest.approx(line_position(CROSS_POSITION, Side.ABOVE, session._geometry))

    _saccade_to(session, fake_gaze, fake_time, GazeZone.LEFT)
    session.on_response_key(Side.ABOVE)
    session.tick()
    [row] = _saccade_rows(session)
    assert row.flash_leg == "return" and row.flash_shown
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


def _finish(session, fake_gaze, fake_time) -> TrialResult:
    """Look back to the dot, hold, answer; returns the trial's row."""
    if fake_gaze.zone is not GazeZone.LEFT:
        _saccade_to(session, fake_gaze, fake_time, GazeZone.LEFT)
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    session.on_response_key(Side.ABOVE)
    session.tick()
    [row] = _saccade_rows(session)
    return row


def test_the_default_trigger_is_arrival_and_jumps_on_the_first_in_target_sample(fake_gaze, fake_time):
    assert session_module.RETURN_JUMP_TRIGGER == "arrival"
    session = _session(fake_gaze, leg="return")
    go = _go(session, fake_time)
    fake_time(0.15)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    assert session._jump_at is None  # onset alone does not trigger "arrival"
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    arrival = session._clock.now()
    assert session._jump_at == pytest.approx(arrival)  # no 150 ms landing debounce any more
    assert session._landed  # the out landing, in return-jump mode
    fake_time(0.1)
    row = _finish(session, fake_gaze, fake_time)
    assert row.out_arrival_ms_after_go_cue == pytest.approx((arrival - go) * 1000)
    assert row.jump_back_ms_after_go_cue == pytest.approx((arrival - go) * 1000)
    assert row.left_target_before_jump is False
    assert row.return_reaction_ms == pytest.approx(100.0)


def test_the_onset_trigger_jumps_at_the_confirmed_out_onset_and_still_measures_the_return(fake_gaze, fake_time, monkeypatch):
    monkeypatch.setattr(session_module, "RETURN_JUMP_TRIGGER", "onset")
    session = _session(fake_gaze, leg="return")
    _go(session, fake_time)
    fake_time(0.15)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    assert session._jump_at is None
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # out onset confirmed -> jump back mid-saccade
    jump = session._jump_at
    assert jump is not None and session._jump_forced is False and _symbols(session) == (True, False)
    fake_time(0.03)
    fake_gaze.zone = GazeZone.RIGHT  # the eye still lands on the old target (approaching it is not a return onset)
    session.tick()
    assert not session._return_onset_detector.confirmed
    fake_time(0.12)
    session.tick()
    row = _finish(session, fake_gaze, fake_time)  # then turns back
    assert row.jump_back_forced is False and row.out_arrival_ms_after_go_cue is not None
    assert row.return_reaction_ms == pytest.approx(150.0)
    assert row.left_target_before_jump is False  # it arrived only after the jump


def test_a_return_before_the_jump_gives_a_negative_latency_that_the_return_average_keeps(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="out")  # the out flash (at 200 ms) holds the jump
    _go(session, fake_time)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_gaze.zone = GazeZone.RIGHT  # arrives at ~20 ms
    session.tick()
    fake_time(0.05)
    session.tick()
    assert session._jump_at is None
    fake_gaze.zone = GazeZone.CENTER  # turns back BEFORE the flash and the jump
    session.tick()
    turn = session._clock.now()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    assert session._return_onset_detector.confirmed and session._jump_at is None
    fake_time(0.15)
    session.tick()  # the flash fires
    session.tick()  # the flash frame is over -> jump back
    jump = session._jump_at
    assert jump is not None and jump > turn
    row = _finish(session, fake_gaze, fake_time)
    assert row.return_reaction_ms == pytest.approx((turn - jump) * 1000) and row.return_reaction_ms < 0
    assert row.left_target_before_jump is True
    assert session.return_rt_samples_ms == pytest.approx([row.return_reaction_ms])  # no plausibility filter


def test_return_flashes_go_on_the_jump_frame_when_the_return_average_is_not_positive(fake_gaze, fake_time):
    session = _session(fake_gaze, leg="return", precomputed_return_rt_samples_ms=[-30.0, -20.0, -10.0])
    session._current_trial_spec = TrialSpec(
        index=0, source=Target.DOT, target=Target.CROSS, line_side=Side.ABOVE, timing_offset_ms=TIMING_OFFSETS_MS[-1]
    )
    _go(session, fake_time)
    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_avg_rt_ms == pytest.approx(-20.0)
    assert session._flash_scheduled_at == pytest.approx(session._jump_at)


def test_return_flash_bell_curve_uses_the_return_spread(fake_gaze, fake_time, monkeypatch):
    seen = {}
    real = session_module.spread_flash_delay_ms

    def spy(label, labels, center, sd, lo, hi, **kw):
        seen["sd"] = sd
        return real(label, labels, center, sd, lo, hi, **kw)

    monkeypatch.setattr(session_module, "spread_flash_delay_ms", spy)
    session = _session(fake_gaze, leg="return", precomputed_return_rt_samples_ms=[100.0] * 3)
    session._current_trial_spec = TrialSpec(
        index=0, source=Target.DOT, target=Target.CROSS, line_side=Side.ABOVE, timing_offset_ms=0
    )
    _go(session, fake_time)
    seen.clear()
    fake_time(0.15)
    _saccade_to(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert seen["sd"] == session_module.RETURN_FLASH_SPREAD_SD_MS


def test_rolling_average_without_a_plausibility_floor_keeps_negative_latencies():
    signed = RollingReactionTime(200.0, min_plausible_ms=None)
    default = RollingReactionTime(200.0)
    for value in (-30.0, 5.0, 40.0):
        signed.add(value)
        default.add(value)
    assert signed.has_estimate and signed.average == pytest.approx(5.0)
    assert not default.has_estimate and default.average == pytest.approx(200.0)  # RT_MIN_PLAUSIBLE_MS drops them


def test_wait_for_baseline_ignores_the_approach_to_the_target():
    def sample(pos, t):
        return GazeSample(zone=GazeZone.CENTER, ratio=0.5, face_found=True, timestamp=t, position=pos)

    eager = OnsetDetector(GazeZone.RIGHT)
    patient = OnsetDetector(GazeZone.RIGHT, wait_for_baseline=True)
    for i, pos in enumerate((0.4, 0.6, 0.9, 1.0, 1.0, 1.0)):  # flying in, then resting on the target
        eager.update(sample(pos, i * 0.005), i * 0.005)
        patient.update(sample(pos, i * 0.005), i * 0.005)
    assert not patient.confirmed
    for i, pos in enumerate((0.8, 0.7, 0.6, 0.5)):  # leaving it
        patient.update(sample(pos, 0.05 + i * 0.005), 0.05 + i * 0.005)
    assert patient.confirmed and patient.since == pytest.approx(0.05)


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
            flash_shown=True, contrast=0.06, line_side=Side.BELOW, responded=True,
            answer_side=Side.BELOW, response_time_ms=300.0, outcome="correct",
            flash_leg="return", jump_back_ms_after_go_cue=412.5, return_reaction_ms=-18.25, jump_back_forced=False,
            out_arrival_ms_after_go_cue=260.0, left_target_before_jump=True,
        )
    )
    log.close()
    csv_path = next(tmp_path.glob("*/*/trials.csv"))
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["flash_leg"] == "return" and rows[0]["jump_back_forced"] == "no"
    [back] = read_trials(csv_path)
    assert back.flash_leg == "return" and back.jump_back_forced is False
    assert back.jump_back_ms_after_go_cue == pytest.approx(412.5) and back.return_reaction_ms == pytest.approx(-18.25, abs=0.1)
    assert back.out_arrival_ms_after_go_cue == pytest.approx(260.0) and back.left_target_before_jump is True
    assert rows[0]["left_target_before_jump"] == "yes"

    new_columns = (
        "flash_leg", "jump_back_ms_after_go_cue", "return_reaction_ms", "jump_back_forced",
        "out_arrival_ms_after_go_cue", "left_target_before_jump",
    )
    old_fields = [name for name in rows[0] if name not in new_columns]
    old_path = tmp_path / "old_trials.csv"
    with open(old_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    [old] = read_trials(old_path)
    assert old.flash_leg is None and old.jump_back_ms_after_go_cue is None
    assert old.return_reaction_ms is None and old.jump_back_forced is None
    assert old.out_arrival_ms_after_go_cue is None and old.left_target_before_jump is None


# -- tutorial practice ---------------------------------------------------------


def _tutorial_at_practice(fake_gaze) -> TutorialSession:
    session = TutorialSession(fake_gaze, PausableClock(), return_jump=True)
    for _ in range(1 + len(session._intro_lines)):
        session.on_space()
    for _ in range(3 * CALIBRATION_ROUNDS):
        session.on_space()
    for _ in range(TUTORIAL_QUIZ_STREAK_TARGET):
        session.on_response_key(session._quiz_side)
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
    session.tick()  # on the target, but practice waits for a stay of RETURN_JUMP_DWELL_MS
    assert session._jump_at is None
    fake_time(RETURN_JUMP_DWELL_MS / 1000)
    session.tick()
    assert session._jump_at is not None and session._legs.jump_forced is False
    state = session.render_state()
    assert state["dot"]["visible"] and not state["cross"]["visible"]

    fake_time(session._flash_scheduled_at - session._clock.now() + 0.001)
    session.tick()
    line = session.render_state()["line"]
    assert line["visible"]
    assert (line["x"], line["y"]) == pytest.approx(line_position(DOT_POSITION, session._line_side, session._geometry))

    fake_gaze.zone = GazeZone.LEFT
    session.tick()
    fake_time(max(SACCADE_ONSET_STABILITY_MS, GAZE_LANDING_STABILITY_MS) / 1000 + 0.01)
    session.tick()
    session.on_response_key(session._line_side)
    session.tick()
    assert session.return_rt_samples_ms == []  # a latency after the wait is not the main trials' movement
    assert session.render_state()["hud"]["phase"] == "PRACTICE_AWAITING_SPACE"
    assert session._source is Target.DOT  # every return-jump trial starts on the dot


@pytest.mark.parametrize("trigger", ["arrival", "onset"])
def test_tutorial_practice_jumps_only_after_the_eye_has_stayed_on_the_target(fake_gaze, fake_time, monkeypatch, trigger):
    """Regression: the target jumped back while the participant was still looking at it. Never reaching it still forces
    the jump at the timeout."""
    monkeypatch.setattr(tutorial_module, "RETURN_JUMP_TRIGGER", trigger)
    session = _tutorial_at_practice(fake_gaze)
    session._leg_bag = ShuffledBag(["return"])
    session._flash_leg = "return"

    def start_attempt() -> None:
        fake_gaze.zone = GazeZone.LEFT
        session.on_space()
        fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
        session.tick()

    def hold(zone: GazeZone, seconds: float) -> None:
        fake_gaze.zone = zone
        session.tick()
        fake_time(seconds)
        session.tick()

    start_attempt()
    stay = RETURN_JUMP_DWELL_MS / 1000
    hold(GazeZone.CENTER, 0.05)  # onset
    hold(GazeZone.RIGHT, stay - 0.05)
    hold(GazeZone.CENTER, 0.05)  # leaving restarts the stay
    hold(GazeZone.RIGHT, stay - 0.05)
    assert session._jump_at is None
    fake_time(0.06)
    session.tick()
    assert session._jump_at is not None and session._legs.jump_forced is False

    session._finish_attempt()
    session._flash_leg = "return"
    start_attempt()
    fake_time(RETURN_JUMP_LANDING_TIMEOUT_MS / 1000 + 0.001)
    session.tick()  # never on the target
    assert session._legs.jump_forced is True

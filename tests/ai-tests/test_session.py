import pytest
from helpers import NullLogger as _NullLogger

from include.experiment import participant_text as words
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_RT_TEST_TRIALS_REAL,
    NUM_RT_TEST_TRIALS_TEST,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    RT_AVERAGE_ROLLING_WINDOW,
    SACCADE_FIXED_TRIALS_REAL,
    SACCADE_FIXED_TRIALS_TEST,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
)
from include.experiment.types import Side, Target, TrialResult, TrialSpec
from include.eye_tracking.types import GazeZone
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession, Phase
from src.experiment.session_config import SessionConfig, TutorialHandoff

pytestmark = pytest.mark.usefixtures("no_flash_jitter")




def _make_session(fake_gaze, show_gaze_indicator: bool = False, skip_practice_trials: bool = True, **kwargs) -> ExperimentSession:
    # skip_practice_trials defaults True here so tests that inject a specific
    # TrialSpec via session._current_trial_spec (see below) land on it
    # immediately after calibration+RT-test, without a fixed practice block
    # in between - practice-specific behavior gets its own test instead.
    return ExperimentSession(
        fake_gaze,
        logger=_NullLogger(),
        clock=PausableClock(),
        config=SessionConfig(show_gaze_indicator=show_gaze_indicator, skip_practice_trials=skip_practice_trials, **kwargs),
    )


def _complete_calibration(session) -> None:
    """Runs calibration to completion - ends in RT_TEST_AWAITING_SPACE, not
    FOREPERIOD, since the reaction-time test always runs immediately after
    calibration now (see _complete_calibration_and_rt_test for the combined
    helper most tests actually want)."""
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    for _round_number in range(1, CALIBRATION_ROUNDS + 1):
        session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
        session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
        session.on_space()  # -> next round's CALIBRATE_LEFT, or RT_TEST_AWAITING_SPACE on the last round


def _run_one_rt_test_attempt(session, fake_gaze, fake_time, reaction_time_s: float = 0.2) -> None:
    session.on_space()  # RT_TEST_AWAITING_SPACE -> RT_TEST_FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()  # RT_TEST_FOREPERIOD -> RT_TEST_ACTIVE
    fake_time(reaction_time_s)
    fake_gaze.zone = GazeZone.RIGHT  # cross's zone - "left the source"
    session.tick()  # starts the onset stability timer
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset fires -> attempt finishes


def _complete_rt_test(session, fake_gaze, fake_time, reaction_time_s: float = 0.2) -> None:
    for _ in range(session._rt_test_target_attempts):
        _run_one_rt_test_attempt(session, fake_gaze, fake_time, reaction_time_s)
    fake_gaze.zone = GazeZone.CENTER  # reset for whatever the test does next


def _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s: float = 0.2) -> None:
    """Ends at the first practice/main trial's AWAITING_SPACE, with
    session._avg_reaction_time_ms == reaction_time_s * 1000 - override
    session._current_trial_spec afterward for tests that need a specific
    trial (source/target/line_side/practice)."""
    _complete_calibration(session)
    _complete_rt_test(session, fake_gaze, fake_time, reaction_time_s)


def _enter_trial_active(session, fake_time) -> None:
    session.on_space()  # AWAITING_SPACE -> FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"


def _trigger_onset_and_landing(session, fake_gaze, fake_time, target_zone) -> None:
    """Realistic valid-trial flow: gaze leaves the source (onset fires,
    still mid-flight in CENTER - not yet the target), the scheduled flash
    fires while still in transit, then gaze arrives at and settles on the
    target (landing debounce clears too). This ordering is what makes
    flash_during_saccade True - the flash's scheduled time falls between
    the onset and landing timestamps (see the flash_during_saccade tests
    below for the other three cases). The trial doesn't finalize yet
    though - flash_decided+landed alone isn't enough without a response or
    a closed response window (see the tests below)."""
    fake_gaze.zone = GazeZone.CENTER
    session.tick()  # starts the onset stability timer
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset fires (reaction_latency_ms set)

    delay = session._flash_scheduled_at - session._clock.now()
    fake_time(max(delay, 0.0) + 0.001)
    session.tick()  # scheduled flash fires, still mid-flight

    fake_gaze.zone = target_zone
    session.tick()  # starts the landing stability timer
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()  # landing debounce clears


def _saccade_results(session) -> list:
    """session.results also includes rt_test-phase rows (see
    ExperimentSession._finish_rt_test_attempt) - filter those out for tests
    that only care about the actual saccade trial(s) they set up."""
    return [r for r in session.results if r.phase == "saccade"]


def _set_single_trial(session, **kwargs) -> None:
    """Overrides whatever trial _complete_calibration_and_rt_test's own
    dynamic generation produced, then re-enters FOREPERIOD so dot/cross
    visibility (set at foreperiod-start from the current trial's source -
    see ExperimentSession._begin_foreperiod) reflects this trial, not
    whatever was auto-generated before it."""
    defaults = dict(index=0, source=Target.DOT, target=Target.CROSS, line_side=Side.ABOVE)
    defaults.update(kwargs)
    session._current_trial_spec = TrialSpec(**defaults)
    session._begin_foreperiod()



def test_calibration_uses_a_fresh_window_per_target_and_averages_rounds_two_and_three(fake_gaze):
    # Regression: without a fresh rolling-ratio window before each target, stale samples collapse the span.
    # Round 1 is discarded; rounds 2-3 are averaged (average_calibration_rounds).
    round_ratios = [(0.1, 0.1, 0.1), (0.2, 0.5, 0.8), (0.4, 0.5, 0.6)]
    calls = iter(ratio for triple in round_ratios for ratio in triple)
    fake_gaze.average_recent_ratio = lambda: next(calls)
    session = _make_session(fake_gaze)
    assert fake_gaze.calibration_samples_begun == 0

    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    expected = 1
    assert fake_gaze.calibration_samples_begun == expected
    for round_number in range(1, CALIBRATION_ROUNDS + 1):
        for _ in range(3):
            session.on_space()
            if not (round_number == CALIBRATION_ROUNDS and _ == 2):
                expected += 1  # a fresh window before every next target, none after the final one
            assert fake_gaze.calibration_samples_begun == expected
    assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"

    averaged = ((0.2 + 0.4) / 2, (0.5 + 0.5) / 2, (0.8 + 0.6) / 2)
    assert session.calibration_ratios == pytest.approx(averaged)
    assert fake_gaze.calibrated == pytest.approx(averaged)


def test_constructor_options_precomputed_calibration_skip_practice_and_mode_constants(fake_gaze):
    # A completed TutorialSession hands its calibration over, so it is applied in __init__ and not repeated.
    ratios = (0.3, 0.5, 0.7)
    session = ExperimentSession(
        fake_gaze, logger=_NullLogger(), clock=PausableClock(), handoff=TutorialHandoff(calibration_ratios=ratios)
    )
    assert fake_gaze.calibrated == ratios and session.calibration_ratios == ratios
    session.on_space()
    assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"

    skipped = ExperimentSession(
        fake_gaze, logger=_NullLogger(), clock=PausableClock(), config=SessionConfig(skip_practice_trials=True)
    )
    assert skipped._practice_trials == [] and skipped._num_practice == 0

    real = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock())
    assert (real._num_practice, real._rt_test_target_attempts, real._fixed_trials) == (
        NUM_PRACTICE_TRIALS_REAL, NUM_RT_TEST_TRIALS_REAL, SACCADE_FIXED_TRIALS_REAL)
    test = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), config=SessionConfig(test_mode=True))
    assert (test._num_practice, test._rt_test_target_attempts, test._fixed_trials) == (
        NUM_PRACTICE_TRIALS_TEST, NUM_RT_TEST_TRIALS_TEST, SACCADE_FIXED_TRIALS_TEST)


# -- reaction-time test -------------------------------------------------------


def test_rt_test_runs_its_attempts_feeds_the_rolling_window_then_enters_a_trial(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration(session)
    for _ in range(session._rt_test_target_attempts - 1):
        _run_one_rt_test_attempt(session, fake_gaze, fake_time, reaction_time_s=0.15)
        assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"
    _run_one_rt_test_attempt(session, fake_gaze, fake_time, reaction_time_s=0.15)
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"  # first real trial waits for SPACE
    assert session._rt.total_added == NUM_RT_TEST_TRIALS_TEST
    assert session._rt.samples == pytest.approx([150.0] * NUM_RT_TEST_TRIALS_TEST)
    assert session._avg_reaction_time_ms == pytest.approx(150.0)


def test_rt_test_falls_back_to_default_when_every_attempt_times_out(fake_gaze, fake_time):
    from include.experiment.constants import DEFAULT_REACTION_TIME_MS

    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration(session)
    fake_gaze.zone = GazeZone.LEFT  # matches DOT's source zone - never leaves it
    for _ in range(session._rt_test_target_attempts):
        session.on_space()
        fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
        session.tick()
        fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
        session.tick()
    assert session._avg_reaction_time_ms == pytest.approx(DEFAULT_REACTION_TIME_MS)


# -- open-loop flash scheduling ------------------------------------------------


def test_flash_and_response_window_follow_the_schedule_not_gaze(fake_gaze, fake_time):
    # Open loop: the flash fires at the scheduled time even if gaze never moves...
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.LEFT
    fake_time(0.199)
    session.tick()
    assert session.render_state()["line"]["visible"] is False
    fake_time(0.002)
    session.tick()
    assert session.render_state()["line"]["visible"] is True

    # ...and an early onset does not open the response window before the flash.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    assert session._onset_detector.confirmed is True
    assert session._response_window_open is False
    fake_time(0.2)
    session.tick()
    assert session._response_window_open is True


def test_fast_lander_does_not_finalize_before_the_scheduled_flash(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.RIGHT  # lands well before avg_reaction_time_ms=200ms
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    assert session._landed is True
    assert len(_saccade_results(session)) == 0

    fake_time(0.2)
    session.tick()  # flash fires and the response window opens - still not finalized this same tick
    assert len(_saccade_results(session)) == 0
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)  # window closes without a response
    session.tick()
    assert len(_saccade_results(session)) == 1


# -- flash_during_saccade and scoring ------------------------------


def test_valid_trial_is_during_saccade_and_counts_as_valid(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Side.ABOVE)
    session.tick()

    assert _saccade_results(session)[0].outcome == "correct"
    assert _saccade_results(session)[0].flash_during_saccade is True
    assert session._valid_trial_count == 1


def test_trial_without_onset_times_out_is_not_during_saccade_and_leaves_rt_alone(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    rt_total = session._rt.total_added
    first_counter = session.render_state()["hud"]["trial_index"]

    fake_gaze.zone = GazeZone.LEFT  # matches DOT's source zone - never leaves
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()

    (result,) = _saccade_results(session)
    assert result.outcome == "timeout"
    assert result.flash_during_saccade is False
    assert session._valid_trial_count == 0
    assert session._rt.total_added == rt_total  # no detected saccade, no RT sample
    assert session.render_state()["hud"]["trial_index"] == first_counter + 1


def test_flash_during_saccade_false_after_landing_and_none_without_landing(fake_gaze, fake_time):
    # Landed before the flash: the classifier says the eye had already arrived -> False.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(0.2)
    session.tick()
    session.on_response_key(Side.ABOVE)
    session.tick()
    assert _saccade_results(session)[0].flash_during_saccade is False

    # Left the source but never reached the target -> None (landing never confirmed).
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(SACCADE_TIMEOUT_MS / 1000)
    session.tick()
    assert _saccade_results(session)[0].flash_during_saccade is None


@pytest.mark.parametrize("answer, expected", [(Side.BELOW, "incorrect"), (None, "miss")])
def test_wrong_answer_and_no_answer_are_scored(fake_gaze, fake_time, answer, expected):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    if answer is None:
        fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    else:
        session.on_response_key(answer)
    session.tick()
    assert _saccade_results(session)[0].outcome == expected


def test_practice_trial_is_not_logged(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session, practice=True)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast == PRACTICE_CONTRAST  # fixed, not drawn from the probe contrasts
    session.on_response_key(Side.ABOVE)
    session.tick()

    assert _saccade_results(session) == []


def test_source_symbol_hides_the_instant_the_target_appears(fake_gaze, fake_time):
    # Regression for "two fully-opaque circles on screen at once": source off in the same tick the target goes on.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    assert session.render_state()["dot"]["visible"] is True
    _enter_trial_active(session, fake_time)
    state = session.render_state()
    assert state["cross"]["visible"] is True
    assert state["dot"]["visible"] is False


# -- rolling reaction-time average ---------------------------------------------


def _run_trial_with_reaction_time(session, fake_gaze, fake_time, index: int, reaction_time_s: float) -> None:
    """An unanswered flash trial whose gaze departs the source at a controlled delay after target onset."""
    _set_single_trial(session, index=index)
    _enter_trial_active(session, fake_time)
    fake_time(reaction_time_s)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset fires at ~reaction_time_s
    delay = session._flash_scheduled_at - session._clock.now()
    fake_time(max(delay, 0.0) + 0.001)
    session.tick()
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()  # finalizes as a miss


def test_average_updates_at_every_detected_eye_movement_and_the_next_trial_uses_it(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    averages = []
    for i in range(NUM_RT_TEST_TRIALS_TEST + 2):  # the participant slows down to 300 ms
        before = session._avg_reaction_time_ms
        total_before = session._rt.total_added
        _run_trial_with_reaction_time(session, fake_gaze, fake_time, i, reaction_time_s=0.3)
        row = _saccade_results(session)[-1]
        assert session._rt.total_added == total_before + 1
        assert row.avg_rt_used_ms == pytest.approx(before)  # the running trial kept the schedule it started with
        assert row.reaction_latency_ms == pytest.approx(300.0)
        averages.append(session._avg_reaction_time_ms)
    assert averages == sorted(averages) and averages[0] >= 200.0
    assert 200.0 < averages[NUM_RT_TEST_TRIALS_TEST - 1] < 300.0  # moves before the window is full of slow trials
    assert averages[-1] == pytest.approx(300.0)

    _set_single_trial(session, index=99)
    _enter_trial_active(session, fake_time)
    assert session._trial_avg_rt_ms == pytest.approx(averages[-1])


def test_the_window_rolls_and_a_stray_latency_does_not_move_the_average(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _run_trial_with_reaction_time(session, fake_gaze, fake_time, 0, reaction_time_s=0.9)  # a lapse
    assert session._avg_reaction_time_ms == pytest.approx(200.0)
    for i in range(RT_AVERAGE_ROLLING_WINDOW):
        _run_trial_with_reaction_time(session, fake_gaze, fake_time, 1 + i, reaction_time_s=0.25)
    assert len(session._rt.samples) == RT_AVERAGE_ROLLING_WINDOW
    assert session._avg_reaction_time_ms == pytest.approx(250.0)


# -- render flags ------------------------------------------------------------------


def test_snap_symbols_and_hide_caption_by_phase(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    assert session.render_state()["snap_symbols"] is False  # waiting to start
    assert session.render_state()["hide_caption"] is False
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    assert session.render_state()["snap_symbols"] is True  # in the trial loop
    session._phase = Phase.CALIBRATE_LEFT
    assert session.render_state()["snap_symbols"] is False
    session._phase = Phase.RT_TEST_ACTIVE
    assert session.render_state()["snap_symbols"] is True

    # Practice trials keep a (spoken) caption; main trials hide it; COMPLETE's message stays visible.
    session._main_trial_counter = 5  # not the block's first trial (that one's SPACE prompt stays visible)
    for practice, hidden in ((True, False), (False, True)):
        _set_single_trial(session, practice=practice)
        for phase in (Phase.AWAITING_SPACE, Phase.FOREPERIOD, Phase.TRIAL_ACTIVE):
            session._phase = phase
            state = session.render_state()
            assert state["hide_caption"] is hidden, (practice, phase)
            assert state["instructions"]
    session._phase = Phase.COMPLETE
    assert session.render_state()["hide_caption"] is False


def test_gaze_indicator_visibility_and_continuous_position(fake_gaze, fake_time):
    session = _make_session(fake_gaze)  # show_gaze_indicator defaults to False
    fake_gaze.smoothed_position = 0.0
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session.render_state()["gaze_indicator"]["visible"] is False

    session = _make_session(fake_gaze, show_gaze_indicator=True)
    assert session.render_state()["gaze_indicator"]["visible"] is False  # before calibration
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    # Continuous smoothed_position (0=dot, 1=cross), interpolated, not snapped to zones.
    for position in (0.0, 0.25, 0.5, 1.0):
        fake_gaze.smoothed_position = position
        indicator = session.render_state()["gaze_indicator"]
        assert indicator["visible"] is True
        assert indicator["x"] == pytest.approx(DOT_POSITION[0] + (CROSS_POSITION[0] - DOT_POSITION[0]) * position)
    assert indicator["y"] == pytest.approx(CROSS_POSITION[1])
    fake_gaze.smoothed_position = None
    assert session.render_state()["gaze_indicator"]["visible"] is False
    fake_gaze.smoothed_position = 0.5
    fake_gaze.face_found = False
    assert session.render_state()["gaze_indicator"]["visible"] is False


# -- stopping criterion ---------------------------------------------------------


def _run_valid_trial(session, fake_gaze, fake_time, index: int) -> None:
    """A quick, always-valid unanswered trial to drive the completed/valid counts forward."""
    _set_single_trial(session, index=index)
    _enter_trial_active(session, fake_time)
    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()


def test_main_block_stops_after_the_fixed_trial_count(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    for i in range(session._fixed_trials - 1):
        _run_valid_trial(session, fake_gaze, fake_time, i)
    assert session.render_state()["hud"]["phase"] != "COMPLETE"
    _run_valid_trial(session, fake_gaze, fake_time, session._fixed_trials - 1)
    assert session.render_state()["hud"]["phase"] == "COMPLETE"


def _fake_valid_saccade_results(count: int) -> list[TrialResult]:
    return [
        TrialResult(
            index=i, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=200.0,
            flash_shown=True, contrast=0.1, line_side=Side.ABOVE, responded=True,
            answer_side=Side.ABOVE, response_time_ms=250.0, outcome="correct", flash_during_saccade=True,
        )
        for i in range(count)
    ]


# -- space-gated trials ----------------------------------------------------------


def test_trial_waits_for_space_then_swaps_500_to_1000_ms_later_and_repeat_presses_do_not_restart(fake_gaze, fake_time):
    from include.experiment.constants import FOREPERIOD_MIN_MS

    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"
    assert "SPACE" in session.render_state()["instructions"]
    fake_time(60.0)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"  # a minute without a press: still waiting

    session.on_space()
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"
    start, duration = session._foreperiod_start, session._foreperiod_duration_s
    assert FOREPERIOD_MIN_MS / 1000 <= duration <= FOREPERIOD_MAX_MS / 1000
    fake_time(0.2)
    session.on_space()  # ignored: does not restart the foreperiod
    assert (session._foreperiod_start, session._foreperiod_duration_s) == (start, duration)
    fake_time(FOREPERIOD_MIN_MS / 1000 - 0.21)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"  # never earlier than the minimum
    fake_time(FOREPERIOD_MAX_MS / 1000 - FOREPERIOD_MIN_MS / 1000 + 0.02)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"


# -- tutorial hand-off: reaction times seeded by the tutorial's practice trials -------


def _seeded_session(fake_gaze, samples, **kwargs) -> ExperimentSession:
    return ExperimentSession(
        fake_gaze, logger=_NullLogger(), clock=PausableClock(), config=SessionConfig(test_mode=True, **kwargs),
        handoff=TutorialHandoff(calibration_ratios=(0.3, 0.5, 0.7), rt_samples_ms=samples),
    )


def test_seeded_session_skips_rt_test_and_practice_and_schedules_from_the_seeded_average(fake_gaze, fake_time):
    samples = [170.0, 180.0, 190.0, 200.0]
    session = _seeded_session(fake_gaze, samples)
    assert session._practice_trials == []

    session.on_space()  # WAITING_TO_START -> straight to the first main trial
    state = session.render_state()
    assert state["hud"]["phase"] == "AWAITING_SPACE"
    # The first main trial is the first thing on screen: its "Press SPACE" must not be hidden.
    assert state["instructions"] == words.FIRST_TRIAL and state["hide_caption"] is False
    assert not session._current_trial_spec.practice
    assert session._avg_reaction_time_ms == pytest.approx(185.0)
    assert session._rt.samples == samples
    assert session._rt_test_attempt == 0 and not any(r.phase == "rt_test" for r in session.results)

    session.on_space()  # FOREPERIOD: caption hidden from here on
    assert session.render_state()["hide_caption"] is True
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"
    assert session._trial_avg_rt_ms == pytest.approx(185.0)


def test_seeded_session_with_too_few_samples_falls_back_to_the_default(fake_gaze):
    from include.experiment.constants import DEFAULT_REACTION_TIME_MS

    session = _seeded_session(fake_gaze, [])  # the tutorial measured nothing usable
    session.on_space()
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"
    assert session._avg_reaction_time_ms == pytest.approx(DEFAULT_REACTION_TIME_MS)

    anticipations_only = _seeded_session(fake_gaze, [40.0, 60.0])  # below RT_MIN_PLAUSIBLE_MS
    assert anticipations_only._avg_reaction_time_ms == pytest.approx(DEFAULT_REACTION_TIME_MS)


def test_calibration_rounds_2_and_3_speak_their_own_return_to_the_left_dot(fake_gaze):
    from include.experiment import participant_text as words

    fake_gaze.average_recent_ratio = lambda: 0.5
    session = _make_session(fake_gaze)
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    left_texts = []
    for _ in range(CALIBRATION_ROUNDS):
        left_texts.append(session.render_state()["instructions"])
        for _ in range(3):
            session.on_space()
    assert left_texts == [words.CALIBRATE_LEFT, words.CALIBRATE_LEFT_AGAIN, words.CALIBRATE_LEFT_LAST]


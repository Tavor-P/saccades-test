import math

import pytest

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    CROSS_POSITION,
    DOT_POSITION,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    MAX_SACCADE_TRIALS_REAL,
    MAX_SACCADE_TRIALS_TEST,
    NUM_PRACTICE_TRIALS_REAL,
    NUM_PRACTICE_TRIALS_TEST,
    NUM_RT_TEST_TRIALS_REAL,
    NUM_RT_TEST_TRIALS_TEST,
    PRACTICE_CONTRAST,
    RESPONSE_WINDOW_MS,
    RT_AVERAGE_ROLLING_WINDOW,
    SACCADE_ONSET_STABILITY_MS,
    SACCADE_TIMEOUT_MS,
    ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH,
    ZEST_LOG_CONTRAST_MAX,
    ZEST_MIN_VALID_TRIALS_REAL,
    ZEST_MIN_VALID_TRIALS_TEST,
)
from include.experiment.types import Side, Target, TrialResult, TrialSpec
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession, Phase


pytestmark = pytest.mark.usefixtures("staircase_no_jitter")


class _NullLogger:
    """Swallows .log() calls - tests check session.results/_saccade_results()
    instead of a real CSV, so they don't need filesystem isolation."""

    def log(self, result) -> None:
        pass


def _make_session(fake_gaze, show_gaze_indicator: bool = False, skip_practice_trials: bool = True, **kwargs) -> ExperimentSession:
    # skip_practice_trials defaults True here so tests that inject a specific
    # TrialSpec via session._current_trial_spec (see below) land on it
    # immediately after calibration+RT-test, without a fixed practice block
    # in between - practice-specific behavior gets its own test instead.
    return ExperimentSession(
        fake_gaze,
        logger=_NullLogger(),
        clock=PausableClock(),
        show_gaze_indicator=show_gaze_indicator,
        skip_practice_trials=skip_practice_trials,
        **kwargs,
    )


def _complete_calibration(session) -> None:
    """Runs calibration to completion - ends in RT_TEST_AWAITING_SPACE, not
    FOREPERIOD, since the reaction-time test always runs immediately after
    calibration now (see _complete_calibration_and_rt_test for the combined
    helper most tests actually want)."""
    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    for round_number in range(1, CALIBRATION_ROUNDS + 1):
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
    trial (source/target/orientation/practice)."""
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
    defaults = dict(index=0, source=Target.DOT, target=Target.CROSS, orientation=Side.ABOVE)
    defaults.update(kwargs)
    session._current_trial_spec = TrialSpec(**defaults)
    session._begin_foreperiod()


def test_contrast_floor_overrides_the_zest_staircase_minimum(fake_gaze):
    # See the matching test in test_presaccade_session.py for why the
    # midpoint is the right thing to check.
    floor = 0.2
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), contrast_floor=floor)
    expected = 10 ** ((math.log10(floor) + ZEST_LOG_CONTRAST_MAX) / 2)
    assert session._zest.next_contrast() == pytest.approx(expected)


def test_calibration_flow_calls_gaze_calibrate(fake_gaze):
    session = _make_session(fake_gaze)
    _complete_calibration(session)
    assert fake_gaze.calibrated == (0.5, 0.5, 0.5)  # FakeGazeSource always reports ratio 0.5
    assert session.calibration_ratios == (0.5, 0.5, 0.5)
    assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"


def test_calibration_starts_a_fresh_sample_window_for_each_target(fake_gaze):
    # Regression test: without resetting the gaze source's rolling ratio
    # window before each of the three calibration targets, the left/center/
    # right averages could be contaminated by stale samples from whatever the
    # participant was looking at previously, collapsing the calibration span
    # toward zero. Calibration runs CALIBRATION_ROUNDS times (discarding
    # round 1, averaging the rest - see average_calibration_rounds), so a
    # fresh window is started before every target in every round except the
    # very last (which finalizes instead of moving to a new target).
    session = _make_session(fake_gaze)
    assert fake_gaze.calibration_samples_begun == 0

    session.on_space()  # WAITING_TO_START -> CALIBRATE_LEFT (round 1)
    assert fake_gaze.calibration_samples_begun == 1  # fresh window before the dot

    expected = 1
    for round_number in range(1, CALIBRATION_ROUNDS + 1):
        session.on_space()  # CALIBRATE_LEFT -> CALIBRATE_CENTER
        expected += 1
        assert fake_gaze.calibration_samples_begun == expected  # fresh window before the center target

        session.on_space()  # CALIBRATE_CENTER -> CALIBRATE_RIGHT
        expected += 1
        assert fake_gaze.calibration_samples_begun == expected  # fresh window before the cross

        session.on_space()  # CALIBRATE_RIGHT -> next round's CALIBRATE_LEFT, or RT_TEST_AWAITING_SPACE on the last round
        if round_number < CALIBRATION_ROUNDS:
            expected += 1
            assert fake_gaze.calibration_samples_begun == expected  # fresh window before the next round's dot
        else:
            assert fake_gaze.calibration_samples_begun == expected  # no new target after the final round
            assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"


def test_self_calibration_discards_round_one_and_averages_rounds_two_and_three(fake_gaze):
    # Each round's (left, center, right) triple: round 1 should be fully
    # discarded, only rounds 2-3 should influence the final result.
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


def test_precomputed_calibration_skips_calibration_phases(fake_gaze):
    # A completed TutorialSession hands its calibration straight to
    # ExperimentSession, so it shouldn't calibrate a second time.
    ratios = (0.3, 0.5, 0.7)
    session = ExperimentSession(
        fake_gaze, logger=_NullLogger(), clock=PausableClock(), precomputed_calibration_ratios=ratios
    )
    assert fake_gaze.calibrated == ratios  # calibrated immediately, in __init__
    assert session.calibration_ratios == ratios

    session.on_space()  # WAITING_TO_START -> straight to the RT test, no CALIBRATE_* phases
    assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"


def test_skip_practice_trials_produces_no_practice_trials(fake_gaze):
    # The tutorial's dress-rehearsal stage replaces this phase's normal
    # practice trials, so a participant who took it shouldn't also get these.
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), skip_practice_trials=True)
    assert session._practice_trials == []
    assert session._num_practice == 0


def test_test_mode_false_uses_the_real_constants_by_default(fake_gaze):
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock())
    assert session._num_practice == NUM_PRACTICE_TRIALS_REAL
    assert session._rt_test_target_attempts == NUM_RT_TEST_TRIALS_REAL
    assert session._min_valid_trials == ZEST_MIN_VALID_TRIALS_REAL
    assert session._max_saccade_trials == MAX_SACCADE_TRIALS_REAL


def test_test_mode_true_uses_the_smaller_test_constants(fake_gaze):
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), test_mode=True)
    assert session._num_practice == NUM_PRACTICE_TRIALS_TEST
    assert session._rt_test_target_attempts == NUM_RT_TEST_TRIALS_TEST
    assert session._min_valid_trials == ZEST_MIN_VALID_TRIALS_TEST
    assert session._max_saccade_trials == MAX_SACCADE_TRIALS_TEST


# -- reaction-time test -------------------------------------------------------


def test_rt_test_runs_configured_attempt_count_then_enters_a_trial(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration(session)
    assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"

    for attempt in range(session._rt_test_target_attempts - 1):
        _run_one_rt_test_attempt(session, fake_gaze, fake_time)
        assert session.render_state()["hud"]["phase"] == "RT_TEST_AWAITING_SPACE"

    _run_one_rt_test_attempt(session, fake_gaze, fake_time)
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"  # first real trial waits for SPACE


def test_rt_test_average_reflects_measured_reaction_times(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration(session)
    _complete_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.15)
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


def test_flash_fires_at_the_scheduled_time_regardless_of_gaze(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.LEFT  # stays at the source the whole time

    fake_time(0.199)
    session.tick()
    assert session.render_state()["grating"]["visible"] is False  # not scheduled yet

    fake_time(0.002)
    session.tick()
    assert session.render_state()["grating"]["visible"] is True  # fired anyway - open-loop


def test_response_window_opens_at_the_scheduled_flash_time_not_at_onset(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    # Gaze leaves the source well before the scheduled flash time - onset
    # fires, but the response window shouldn't open yet.
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

    # Lands almost immediately - well before avg_reaction_time_ms=200ms.
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    assert session._landed is True
    assert len(_saccade_results(session)) == 0  # not finalized - flash hasn't fired yet

    fake_time(0.2)
    session.tick()  # flash fires and the response window opens - still not finalized this same tick
    assert len(_saccade_results(session)) == 0

    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)  # window closes without a response
    session.tick()
    assert len(_saccade_results(session)) == 1  # now finalizes


# -- flash_during_saccade window check -----------------------------------------


def test_flash_during_saccade_true_when_flash_lands_between_onset_and_landing(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Side.ABOVE)
    session.tick()  # finalizes

    assert _saccade_results(session)[0].flash_during_saccade is True


def test_flash_during_saccade_false_when_onset_never_detected(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.LEFT  # matches DOT's source zone - never leaves
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()  # timeout - finalizes

    assert _saccade_results(session)[0].flash_during_saccade is False


def test_flash_during_saccade_false_when_flash_lands_after_landing(fake_gaze, fake_time):
    # Gaze reaches and settles on the target well before the scheduled flash
    # fires - by the time the flash actually happens, the classifier's own
    # read says the eye already arrived, so this isn't "during" the saccade.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)  # lands at ~150ms, well before the 200ms flash
    session.tick()
    fake_time(0.2)
    session.tick()  # flash fires now, response window opens
    session.on_response_key(Side.ABOVE)
    session.tick()  # finalizes

    assert _saccade_results(session)[0].flash_during_saccade is False


def test_flash_during_saccade_none_when_landing_never_confirmed(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    fake_gaze.zone = GazeZone.CENTER  # leaves the source, but never reaches the target
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset fires
    fake_time(SACCADE_TIMEOUT_MS / 1000)
    session.tick()  # timeout - finalizes, still never landed

    assert _saccade_results(session)[0].flash_during_saccade is None


# -- ZEST gating on validity ---------------------------------------------------


def test_valid_trial_updates_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Side.ABOVE)
    session.tick()

    assert _saccade_results(session)[0].outcome == "correct"
    assert _saccade_results(session)[0].flash_during_saccade is True
    assert session._zest.threshold_estimate != initial_threshold
    assert session._valid_trial_count == 1


def test_invalid_trial_does_not_update_zest(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    fake_gaze.zone = GazeZone.LEFT  # onset never detected -> flash_during_saccade False
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()

    assert _saccade_results(session)[0].flash_during_saccade is False
    assert session._zest.threshold_estimate == initial_threshold
    assert session._valid_trial_count == 0


def test_incorrect_orientation_is_still_scored_correctly(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Side.BELOW)  # wrong guess
    session.tick()

    assert _saccade_results(session)[0].outcome == "incorrect"


def test_miss_is_still_scored_correctly(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)  # let the response window expire; never respond
    session.tick()

    assert _saccade_results(session)[0].outcome == "miss"


def test_timeout_when_gaze_never_leaves_the_source_zone(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    fake_gaze.zone = GazeZone.LEFT  # matches Target.DOT's zone - never looks away
    _enter_trial_active(session, fake_time)

    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()

    assert len(_saccade_results(session)) == 1
    result = _saccade_results(session)[0]
    assert result.outcome == "timeout"


def test_practice_trial_is_not_logged_or_staircased(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session, practice=True)
    _enter_trial_active(session, fake_time)
    initial_threshold = session._zest.threshold_estimate

    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    assert session._trial_contrast == PRACTICE_CONTRAST  # fixed, not drawn from the staircase
    session.on_response_key(Side.ABOVE)
    session.tick()

    assert _saccade_results(session) == []  # practice trials never get logged
    assert session._zest.threshold_estimate == initial_threshold  # staircase untouched


def test_source_symbol_hides_the_instant_the_target_appears(fake_gaze, fake_time):
    # Regression test for the "two fully-opaque circles on screen at once"
    # confusion: the source's `visible` flag must flip off in the same tick
    # the target's flips on, not wait for gaze to land on the target.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    assert session.render_state()["dot"]["visible"] is True  # source lit during the foreperiod

    _enter_trial_active(session, fake_time)
    state = session.render_state()
    assert state["cross"]["visible"] is True  # target just appeared
    assert state["dot"]["visible"] is False  # source already hidden, before any landing


# -- rolling reaction-time average ---------------------------------------------


def _run_trial_with_reaction_time(session, fake_gaze, fake_time, index: int, reaction_time_s: float) -> None:
    """A flash trial answered by nobody (so no orientation-response bookkeeping
    needed) whose gaze departs the source at a controlled, deliberate delay
    after target onset - unlike _trigger_onset_and_landing, which only cares
    about clearing the debounce windows as fast as possible."""
    _set_single_trial(session, index=index)
    _enter_trial_active(session, fake_time)
    fake_time(reaction_time_s)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()  # starts the onset stability timer
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset fires at ~reaction_time_s
    delay = session._flash_scheduled_at - session._clock.now()
    fake_time(max(delay, 0.0) + 0.001)
    session.tick()  # scheduled flash decides, response window opens
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()  # lands
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()  # window closes, no response given -> finalizes as a miss


def test_rt_test_attempts_feed_the_rolling_window(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    assert session._rt.total_added == NUM_RT_TEST_TRIALS_TEST
    assert session._rt.samples == pytest.approx([200.0] * NUM_RT_TEST_TRIALS_TEST)
    assert session._avg_reaction_time_ms == pytest.approx(200.0)


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
    assert session._trial_avg_rt_ms == pytest.approx(averages[-1])  # the next trial is scheduled from the new average


def test_a_trial_without_a_detected_saccade_adds_no_sample(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    total = session._rt.total_added
    _set_single_trial(session, index=0)
    _enter_trial_active(session, fake_time)
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.5)
    session.tick()
    assert session._rt.total_added == total


def test_the_window_rolls_and_a_stray_latency_does_not_move_the_average(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    _run_trial_with_reaction_time(session, fake_gaze, fake_time, 0, reaction_time_s=0.9)  # a lapse
    assert session._avg_reaction_time_ms == pytest.approx(200.0)
    for i in range(RT_AVERAGE_ROLLING_WINDOW):
        _run_trial_with_reaction_time(session, fake_gaze, fake_time, 1 + i, reaction_time_s=0.25)
    assert len(session._rt.samples) == RT_AVERAGE_ROLLING_WINDOW  # old samples rolled out
    assert session._avg_reaction_time_ms == pytest.approx(250.0)


def test_render_state_snaps_symbols_during_rt_test_and_trials_but_not_calibration(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    assert session.render_state()["snap_symbols"] is False  # waiting to start
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time, reaction_time_s=0.2)
    assert session.render_state()["snap_symbols"] is True  # in the trial loop
    session._phase = Phase.CALIBRATE_LEFT
    assert session.render_state()["snap_symbols"] is False
    session._phase = Phase.RT_TEST_ACTIVE
    assert session.render_state()["snap_symbols"] is True


# -- stopping criterion ---------------------------------------------------------


def _run_valid_trial(session, fake_gaze, fake_time, index: int) -> None:
    """A quick, always-valid (flash_during_saccade True) unanswered trial, purely
    to drive session._completed_main_trial_count/_valid_trial_count forward
    without needing a real orientation response."""
    _set_single_trial(session, index=index)
    _enter_trial_active(session, fake_time)
    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()


def test_main_block_does_not_stop_before_the_minimum_valid_trial_floor(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    for i in range(session._min_valid_trials - 1):
        _run_valid_trial(session, fake_gaze, fake_time, i)
    assert session.render_state()["hud"]["phase"] != "COMPLETE"


def test_main_block_stops_once_the_max_trial_cap_is_hit_even_with_a_wide_interval(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    session._min_valid_trials = 0  # isolate the max-cap behavior from the credible-interval check
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    for i in range(session._max_saccade_trials):
        _run_valid_trial(session, fake_gaze, fake_time, i)
    assert session.render_state()["hud"]["phase"] == "COMPLETE"


def _fake_valid_saccade_results(count: int) -> list[TrialResult]:
    """_completed_main_trial_count/_valid_trial_count are derived from
    session._results (see session.py) rather than hand-maintained counters -
    populate that list directly instead of assigning to what are now
    read-only properties."""
    return [
        TrialResult(
            index=i,
            phase="saccade",
            source=Target.DOT,
            target=Target.CROSS,
            saccade_duration_ms=200.0,
            grating_shown=True,
            contrast=0.1,
            orientation=Side.ABOVE,
            responded=True,
            response_orientation=Side.ABOVE,
            response_time_ms=250.0,
            outcome="correct",
            flash_during_saccade=True,
        )
        for i in range(count)
    ]


def test_should_stop_main_block_true_once_credible_interval_is_narrow_enough(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True)
    session._results = _fake_valid_saccade_results(session._min_valid_trials)

    class _NarrowZest:
        def credible_interval(self, mass):
            return 0.10, 0.10 * (10**ZEST_CREDIBLE_INTERVAL_MAX_LOG_WIDTH) * 0.5

    session._zest = _NarrowZest()
    assert session._should_stop_main_block() is True


def test_should_stop_main_block_false_when_credible_interval_is_wide(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True)
    session._results = _fake_valid_saccade_results(session._min_valid_trials)

    class _WideZest:
        def credible_interval(self, mass):
            return 0.01, 0.5

    session._zest = _WideZest()
    assert session._should_stop_main_block() is False


# -- gaze indicator (unaffected by the timing redesign) --------------------------


def test_gaze_indicator_hidden_by_default(fake_gaze, fake_time):
    # show_gaze_indicator defaults to False - a real participant shouldn't
    # see this even with valid tracking and completed calibration.
    session = _make_session(fake_gaze)
    fake_gaze.smoothed_position = 0.0
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_hidden_before_calibration(fake_gaze):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    fake_gaze.smoothed_position = 0.0
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_visible_during_real_trials_when_enabled(fake_gaze, fake_time):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session.render_state()["gaze_indicator"]["visible"] is True


def test_gaze_indicator_hidden_when_face_not_found(fake_gaze, fake_time):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    fake_gaze.face_found = False
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_hidden_when_smoothed_position_is_none(fake_gaze, fake_time):
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    fake_gaze.smoothed_position = None
    assert session.render_state()["gaze_indicator"]["visible"] is False


def test_gaze_indicator_tracks_continuous_smoothed_position(fake_gaze, fake_time):
    # Uses GazeSample.smoothed_position (0=left/dot target, 1=right/cross
    # target) directly rather than snapping to one of three discrete zones -
    # so values between the zone boundaries should land at the matching
    # interpolated point, not jump between fixed positions.
    session = _make_session(fake_gaze, show_gaze_indicator=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)

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


def test_render_state_reports_the_main_trial_counter(fake_gaze, fake_time):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    first_counter = session.render_state()["hud"]["trial_index"]

    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()  # timeout - finalizes, advances to the next dynamically-generated trial

    assert session.render_state()["hud"]["trial_index"] == first_counter + 1


# -- space-gated trials ----------------------------------------------------------


def test_trial_waits_for_space_and_swaps_500_to_1000_ms_after_the_press(fake_gaze, fake_time):
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
    assert FOREPERIOD_MIN_MS / 1000 <= session._foreperiod_duration <= FOREPERIOD_MAX_MS / 1000
    fake_time(FOREPERIOD_MIN_MS / 1000 - 0.01)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "FOREPERIOD"  # never earlier than the minimum
    fake_time(FOREPERIOD_MAX_MS / 1000 - FOREPERIOD_MIN_MS / 1000 + 0.02)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"


def test_space_during_the_foreperiod_does_not_restart_it(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    session.on_space()
    start, duration = session._foreperiod_start, session._foreperiod_duration
    fake_time(0.2)
    session.on_space()
    assert session._foreperiod_start == start and session._foreperiod_duration == duration



def test_hide_caption_only_for_main_block_trials_not_calibration_rt_or_practice(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    assert session.render_state()["hide_caption"] is False  # WAITING_TO_START
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session, practice=True)
    assert session.render_state()["hide_caption"] is False  # practice FOREPERIOD
    _enter_trial_active(session, fake_time)
    assert session.render_state()["hide_caption"] is False  # practice TRIAL_ACTIVE

    session._main_trial_counter = 5  # not the block's first trial (that one's SPACE prompt stays visible)
    _set_single_trial(session, practice=False)
    assert session.render_state()["hide_caption"] is True  # main FOREPERIOD
    _enter_trial_active(session, fake_time)
    assert session.render_state()["hide_caption"] is True  # main TRIAL_ACTIVE
    session._phase = Phase.AWAITING_SPACE
    assert session.render_state()["hide_caption"] is True
    session._phase = Phase.COMPLETE
    assert session.render_state()["hide_caption"] is False  # end-of-phase message stays visible


# -- reaction times seeded by the tutorial's practice trials ------------------------


def _seeded_session(fake_gaze, samples, **kwargs) -> ExperimentSession:
    return ExperimentSession(
        fake_gaze,
        logger=_NullLogger(),
        clock=PausableClock(),
        test_mode=True,
        precomputed_calibration_ratios=(0.3, 0.5, 0.7),
        precomputed_rt_samples_ms=samples,
        **kwargs,
    )


def test_seeded_session_skips_rt_test_and_practice_and_uses_the_seeded_average(fake_gaze):
    samples = [170.0, 180.0, 190.0, 200.0]
    session = _seeded_session(fake_gaze, samples)
    assert session._practice_trials == []  # seeded implies no practice of its own

    session.on_space()  # WAITING_TO_START -> straight to the first main trial
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"
    assert not session._current_trial_spec.practice
    assert session._avg_reaction_time_ms == pytest.approx(185.0)
    assert session._rt.samples == samples
    assert session._rt_test_attempt == 0 and not any(r.phase == "rt_test" for r in session.results)


def test_seeded_session_with_too_few_samples_falls_back_to_the_default(fake_gaze):
    from include.experiment.constants import DEFAULT_REACTION_TIME_MS

    session = _seeded_session(fake_gaze, [])  # the tutorial measured nothing usable
    session.on_space()
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"
    assert session._avg_reaction_time_ms == pytest.approx(DEFAULT_REACTION_TIME_MS)

    anticipations_only = _seeded_session(fake_gaze, [40.0, 60.0])  # below RT_MIN_PLAUSIBLE_MS
    assert anticipations_only._avg_reaction_time_ms == pytest.approx(DEFAULT_REACTION_TIME_MS)


def test_seeded_session_schedules_its_first_flash_from_the_seeded_average(fake_gaze, fake_time):
    session = _seeded_session(fake_gaze, [200.0, 200.0, 200.0])
    session.on_space()
    session.on_space()  # AWAITING_SPACE -> FOREPERIOD
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    assert session.render_state()["hud"]["phase"] == "TRIAL_ACTIVE"
    assert session._trial_avg_rt_ms == pytest.approx(200.0)


def test_practice_trials_keep_a_spoken_caption_while_main_trials_hide_it(fake_gaze, fake_time):
    # Combined behaviour of the onboarding wording and the hide_caption flag.
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    session._main_trial_counter = 5  # not the block's first trial (that one's SPACE prompt stays visible)
    for practice, hidden in ((True, False), (False, True)):
        _set_single_trial(session, practice=practice)
        for phase in (Phase.AWAITING_SPACE, Phase.FOREPERIOD, Phase.TRIAL_ACTIVE):
            session._phase = phase
            state = session.render_state()
            assert state["hide_caption"] is hidden
            assert state["instructions"]  # text exists; the runner shows/speaks it unless hidden


def test_first_screen_after_the_tutorial_says_to_press_space(fake_gaze):
    """With the tutorial's calibration and RT samples there are no practice trials, so the first main trial is the first thing on screen:
    its 'Press SPACE when you are ready' must not be hidden, and later trials' captions are."""
    session = _make_session(fake_gaze, precomputed_calibration_ratios=(0.2, 0.5, 0.8), precomputed_rt_samples_ms=[250.0, 260.0, 255.0])
    session.on_space()  # "Press SPACE to begin"
    state = session.render_state()
    assert state["hud"]["phase"] == "AWAITING_SPACE"
    assert state["instructions"] == "Press SPACE when you are ready."
    assert state["hide_caption"] is False
    session.on_space()  # FOREPERIOD: caption (the answer keys) is hidden again from here on
    assert session.render_state()["hide_caption"] is True

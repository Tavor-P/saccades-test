"""Sample-timestamp onset/landing timing, robust reaction-time average, saccade duration, per-trial
timing fields and the richer _meta.json (timing-core work). Reuses test_session's helpers."""

import json

import pytest

from include.eye_tracking.types import GazeSample, GazeZone
from include.experiment.constants import (
    GAZE_LANDING_STABILITY_MS,
    GAZE_LATENCY_MAX_MS,
    RESPONSE_WINDOW_MS,
    SACCADE_TIMEOUT_MS,
)
from include.experiment.types import Orientation, Target, TrialResult
from src.experiment import logger as logger_module
from src.experiment.logger import ResultLogger
from src.experiment.pausable_clock import PausableClock
from src.experiment.session import ExperimentSession
from src.experiment.trial_mechanics import robust_average_ms
from test_session import (
    _complete_calibration,
    _complete_calibration_and_rt_test,
    _enter_trial_active,
    _make_session,
    _NullLogger,
    _saccade_results,
    _set_single_trial,
    _trigger_onset_and_landing,
)


def _stamped_gaze(fake_gaze):
    """fake_gaze, but latest_sample() carries a settable timestamp and the source
    reports a measured latency - like Neon, whose samples are stamped on arrival."""
    fake_gaze.timestamp = 0.0
    fake_gaze.gaze_latency_ms = 20.0

    def latest_sample():
        return GazeSample(
            zone=fake_gaze.zone,
            ratio=fake_gaze.ratio,
            face_found=fake_gaze.face_found,
            timestamp=fake_gaze.timestamp,
            position=fake_gaze.position,
            smoothed_position=fake_gaze.smoothed_position,
        )

    fake_gaze.latest_sample = latest_sample
    return fake_gaze


# -- robust average -----------------------------------------------------------------


def test_robust_average_is_the_median_of_plausible_values():
    assert robust_average_ms([200, 210, 220, 230, 240], default=999) == 220


def test_robust_average_drops_anticipations_and_slow_outliers():
    assert robust_average_ms([40, 60, 210, 220, 230, 1400], default=999) == 220


def test_robust_average_falls_back_when_nothing_is_plausible():
    assert robust_average_ms([], default=225) == 225
    assert robust_average_ms([30, 50, 80], default=225) == 225


def test_robust_average_one_outlier_barely_moves_it():
    clean = robust_average_ms([200, 210, 220, 230, 240], default=0)
    assert robust_average_ms([200, 210, 220, 230, 240, 700], default=0) == pytest.approx(clean, abs=10)


def test_rt_test_average_ignores_anticipations_and_slow_outliers(fake_gaze):
    session = _make_session(fake_gaze, test_mode=True)
    _complete_calibration(session)
    session._rt_samples_ms = [60.0, 210.0, 220.0, 230.0, 900.0]
    session._rt_test_attempt = session._rt_test_target_attempts - 1
    session._finish_rt_test_attempt(None)
    assert session._avg_reaction_time_ms == pytest.approx(220.0)


# -- sample-timestamp timing ----------------------------------------------------------


def test_onset_and_landing_use_sample_timestamps_minus_gaze_latency(fake_gaze, fake_time):
    gaze = _stamped_gaze(fake_gaze)
    session = _make_session(gaze)
    _complete_calibration_and_rt_test(session, gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    start = session._trial_start_time

    # The render tick runs well after the samples were taken; the sample stamps
    # (arrival) minus the 20 ms latency, not the tick, set the onset.
    fake_time(0.30)
    gaze.zone = GazeZone.CENTER
    gaze.timestamp = start + 0.20
    session.tick()
    gaze.timestamp = start + 0.205
    session.tick()
    gaze.timestamp = start + 0.212
    session.tick()
    assert session._onset_detector.confirmed
    assert session._reaction_latency_ms == pytest.approx(180.0)  # 0.200 arrival - 0.020 latency
    assert session._onset_detection_lag_ms == pytest.approx(12.0)

    gaze.zone = GazeZone.RIGHT
    gaze.timestamp = start + 0.240
    session.tick()
    assert session._target_landed_since == pytest.approx(start + 0.220)
    fake_time(0.5)
    gaze.timestamp = start + 0.250 + GAZE_LANDING_STABILITY_MS / 1000
    session.tick()
    assert session._landed


def test_a_source_with_history_is_consumed_sample_by_sample(fake_gaze, fake_time):
    gaze = _stamped_gaze(fake_gaze)
    gaze.gaze_latency_ms = 0.0
    session = _make_session(gaze)
    _complete_calibration_and_rt_test(session, gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    start = session._trial_start_time

    def sample(dt, zone):
        return GazeSample(zone=zone, ratio=0.5, face_found=True, timestamp=start + dt, position={GazeZone.LEFT: 0.0, GazeZone.CENTER: 0.5}.get(zone, 1.0))

    batch = [sample(0.10, GazeZone.LEFT), sample(0.105, GazeZone.CENTER), sample(0.112, GazeZone.CENTER), sample(0.120, GazeZone.CENTER)]
    gaze.samples_since = lambda ts: [s for s in batch if ts is None or s.timestamp > ts]
    fake_time(0.2)
    gaze.timestamp = start + 0.120
    session.tick()  # one render tick sees all four samples
    assert session._onset_detector.confirmed
    assert session._onset_detector.since == pytest.approx(start + 0.105)
    assert session._reaction_latency_ms == pytest.approx(105.0)


def test_samples_without_a_usable_timestamp_fall_back_to_tick_time(fake_gaze, fake_time):
    session = _make_session(fake_gaze)  # FakeGazeSource stamps 0.0 and reports no latency
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    start = session._trial_start_time
    fake_time(0.25)
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    assert session._onset_detector.since == pytest.approx(start + 0.25)


def test_gaze_latency_is_clamped_and_defaults_to_zero(fake_gaze):
    session = _make_session(fake_gaze)
    assert session.gaze_latency_ms == 0.0
    fake_gaze.gaze_latency_ms = 10_000.0
    assert session.gaze_latency_ms == GAZE_LATENCY_MAX_MS
    fake_gaze.gaze_latency_ms = -5.0
    assert session.gaze_latency_ms == 0.0


# -- saccade duration + per-trial timing fields -----------------------------------------


def test_saccade_duration_is_landing_minus_onset(fake_gaze, fake_time):
    gaze = _stamped_gaze(fake_gaze)
    gaze.gaze_latency_ms = 0.0
    session = _make_session(gaze)
    _complete_calibration_and_rt_test(session, gaze, fake_time)
    _set_single_trial(session, grating_shown=False)
    _enter_trial_active(session, fake_time)
    start = session._trial_start_time

    fake_time(0.3)
    gaze.zone = GazeZone.CENTER
    for dt in (0.20, 0.21, 0.22):
        gaze.timestamp = start + dt
        session.tick()
    assert session._onset_detector.confirmed
    gaze.zone = GazeZone.RIGHT
    fake_time(0.5)
    for dt in (0.26, 0.27, 0.26 + GAZE_LANDING_STABILITY_MS / 1000 + 0.01):
        gaze.timestamp = start + dt
        session.tick()
    assert session._landed
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.5)
    gaze.timestamp = start + 1.0
    session.tick()
    row = _saccade_results(session)[0]
    assert row.saccade_duration_ms == pytest.approx(60.0)  # landing 0.26 - onset 0.20


def test_saccade_duration_is_none_when_gaze_never_leaves(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session)
    _enter_trial_active(session, fake_time)
    fake_gaze.zone = GazeZone.LEFT
    fake_time(SACCADE_TIMEOUT_MS / 1000 + 0.1)
    session.tick()
    assert _saccade_results(session)[0].saccade_duration_ms is None


def test_trial_rows_log_the_scheduling_inputs_and_flash_relative_to_onset(fake_gaze, fake_time):
    gaze = _stamped_gaze(fake_gaze)
    session = _make_session(gaze)
    _complete_calibration_and_rt_test(session, gaze, fake_time, reaction_time_s=0.2)
    _set_single_trial(session, timing_offset_ms=40)
    _enter_trial_active(session, fake_time)
    _trigger_onset_and_landing(session, gaze, fake_time, GazeZone.RIGHT)
    session.on_response_key(Orientation.VERTICAL)
    session.tick()

    row = _saccade_results(session)[0]
    # The RT test measured 0.2 s of arrival time, minus the source's 20 ms latency.
    assert row.avg_rt_used_ms == pytest.approx(180.0)
    assert row.scheduled_delay_ms == pytest.approx(220.0)
    assert row.gaze_latency_ms == pytest.approx(20.0)
    assert row.flash_relative_to_onset_ms is not None


def test_catch_trial_has_no_flash_relative_to_onset(fake_gaze, fake_time):
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    _set_single_trial(session, grating_shown=False)
    _enter_trial_active(session, fake_time)
    _trigger_onset_and_landing(session, fake_gaze, fake_time, GazeZone.RIGHT)
    fake_time(RESPONSE_WINDOW_MS / 1000 + 0.1)
    session.tick()
    row = _saccade_results(session)[0]
    assert row.flash_relative_to_onset_ms is None
    assert row.avg_rt_used_ms is not None


# -- _meta.json ---------------------------------------------------------------------


def test_meta_has_provenance_and_constants_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("p1")
    log.close()
    meta = json.loads(next(tmp_path.glob("results_*_meta.json")).read_text())
    assert set(meta["git"]) == {"sha", "dirty"}
    assert meta["python_version"]
    assert "psychopy_version" in meta
    assert meta["constants"]["SACCADE_ONSET_STABILITY_MS"] == 10
    assert meta["display_refresh_hz"] is None


def test_meta_records_refresh_rate_and_phase_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = ResultLogger("p1")
    log.set_display_refresh_hz(59.9)
    log.set_phase_summary("saccade", {"threshold_estimate": 0.1})
    log.close()
    meta = json.loads(next(tmp_path.glob("results_*_meta.json")).read_text())
    assert meta["display_refresh_hz"] == 59.9
    assert meta["phase_summaries"]["saccade"]["threshold_estimate"] == 0.1


def _finished_session(fake_gaze, logger):
    session = ExperimentSession(fake_gaze, logger=logger, clock=PausableClock(), skip_practice_trials=True, test_mode=True)
    cap = session._max_saccade_trials
    session._results = [
        TrialResult(
            index=i, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=None, grating_shown=False,
            contrast=None, orientation=None, responded=False, response_orientation=None, response_time_ms=None,
            outcome="correct_rejection",
        )
        for i in range(cap)
    ]
    session._avg_reaction_time_ms = 250.0
    return session


def test_completing_the_block_writes_a_phase_summary(fake_gaze):
    summaries = {}

    class _Logger(_NullLogger):
        def set_phase_summary(self, phase, summary):
            summaries[phase] = summary

    session = _finished_session(fake_gaze, _Logger())
    session._begin_next_trial_or_complete()
    summary = summaries["saccade"]
    assert summary["hit_max_trial_cap"] is True
    assert summary["final_avg_reaction_time_ms"] == 250.0
    assert summary["threshold_at_grid_bound"] is None


def test_phase_summary_flags_an_estimate_at_the_grid_bound(fake_gaze):
    summaries = {}

    class _Logger(_NullLogger):
        def set_phase_summary(self, phase, summary):
            summaries[phase] = summary

    session = _finished_session(fake_gaze, _Logger())
    for _ in range(40):
        session._zest.update(0.4, detected=False)  # never seen even at 40% -> piles up at the top
    session._log_phase_summary()
    assert summaries["saccade"]["threshold_at_grid_bound"] == "upper"


def test_a_capture_stamped_source_is_not_shifted_by_its_latency_again(fake_gaze, fake_time):
    """Neon (with a clock offset) and trace replay stamp samples with their capture time and still report
    gaze_latency_ms for logging; the session must not subtract that latency a second time."""

    def reaction_ms(capture_stamped):
        gaze = _stamped_gaze(fake_gaze)  # reports a 20 ms latency
        gaze.timestamps_are_capture_time = capture_stamped
        session = _make_session(gaze)
        _complete_calibration_and_rt_test(session, gaze, fake_time)
        _set_single_trial(session)
        _enter_trial_active(session, fake_time)
        start = session._trial_start_time
        fake_time(0.30)
        gaze.zone = GazeZone.CENTER
        for dt in (0.200, 0.205, 0.212):
            gaze.timestamp = start + dt
            session.tick()
        assert session._onset_detector.confirmed
        assert session.gaze_latency_ms == pytest.approx(20.0)  # still reported (and logged) either way
        return session._reaction_latency_ms

    assert reaction_ms(capture_stamped=False) == pytest.approx(180.0)
    assert reaction_ms(capture_stamped=True) == pytest.approx(200.0)

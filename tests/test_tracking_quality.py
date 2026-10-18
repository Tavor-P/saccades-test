"""Trials whose gaze tracking dropped out around the flash are flagged
tracking_ok=False and excluded from ZEST and analysis (scoring.py)."""

from unittest.mock import Mock

from include.eye_tracking.types import GazeZone
from include.experiment.constants import (
    CALIBRATION_ROUNDS,
    FOREPERIOD_MAX_MS,
    GAZE_LANDING_STABILITY_MS,
    SACCADE_ONSET_STABILITY_MS,
    TRACKING_LOSS_WINDOW_MS,
)
from include.experiment.types import Orientation, Target, TrialResult, TrialSpec
from src.experiment.pausable_clock import PausableClock
from src.experiment.scoring import is_valid_for_saccadic_analysis
from src.experiment.session import ExperimentSession


class _NullLogger:
    def log(self, result) -> None:
        pass


def _session_in_first_trial(fake_gaze, fake_time) -> ExperimentSession:
    session = ExperimentSession(fake_gaze, logger=_NullLogger(), clock=PausableClock(), skip_practice_trials=True)
    session.on_space()
    for _ in range(CALIBRATION_ROUNDS):
        session.on_space()
        session.on_space()
        session.on_space()
    for _ in range(session._rt_test_target_attempts):
        fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
        session.tick()
        fake_time(0.2)
        fake_gaze.zone = GazeZone.RIGHT
        session.tick()
        fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
        session.tick()
    fake_gaze.zone = GazeZone.CENTER
    session._current_trial_spec = TrialSpec(
        index=0, source=Target.DOT, target=Target.CROSS, grating_shown=True, orientation=Orientation.VERTICAL
    )
    session._begin_foreperiod()
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    return session


def _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash: bool) -> TrialResult:
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()  # onset
    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    if lose_tracking_at_flash:
        fake_gaze.face_found = False
    session.tick()  # flash fires (and a lost-track sample is seen, if requested)
    fake_gaze.face_found = True
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()  # landed
    session.on_response_key(Orientation.VERTICAL)
    session.tick()
    return [r for r in session.results if r.phase == "saccade"][0]


def test_good_tracking_trial_is_valid_and_feeds_zest(fake_gaze, fake_time):
    session = _session_in_first_trial(fake_gaze, fake_time)
    session._zest.update = Mock()
    result = _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash=False)
    assert result.flash_during_saccade is True
    assert result.tracking_ok is True
    session._zest.update.assert_called_once()


def test_tracking_lost_at_flash_is_flagged_and_excluded_from_zest(fake_gaze, fake_time):
    session = _session_in_first_trial(fake_gaze, fake_time)
    session._zest.update = Mock()
    result = _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash=True)
    assert result.tracking_ok is False
    session._zest.update.assert_not_called()
    assert session._valid_trial_count == 0


def test_tracking_loss_far_from_the_flash_is_ignored(fake_gaze, fake_time):
    session = _session_in_first_trial(fake_gaze, fake_time)
    session._flash_scheduled_at = session._clock.now() + 5.0
    fake_time(TRACKING_LOSS_WINDOW_MS / 1000 * 3)  # well before the flash window opens
    fake_gaze.face_found = False
    session.tick()
    assert session._tracking_lost_near_flash is False


def test_catch_trial_has_no_tracking_flag(fake_gaze, fake_time):
    session = _session_in_first_trial(fake_gaze, fake_time)
    session._current_trial_spec = TrialSpec(
        index=0, source=Target.DOT, target=Target.CROSS, grating_shown=False, orientation=None
    )
    session._begin_foreperiod()
    fake_time(FOREPERIOD_MAX_MS / 1000 + 0.1)
    session.tick()
    fake_gaze.zone = GazeZone.CENTER
    session.tick()
    fake_time(SACCADE_ONSET_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(max(session._flash_scheduled_at - session._clock.now(), 0.0) + 0.001)
    session.tick()
    fake_gaze.zone = GazeZone.RIGHT
    session.tick()
    fake_time(GAZE_LANDING_STABILITY_MS / 1000 + 0.01)
    session.tick()
    fake_time(2.0)
    session.tick()
    result = [r for r in session.results if r.phase == "saccade"][0]
    assert result.tracking_ok is None


def test_validity_gate_combines_both_checks():
    assert is_valid_for_saccadic_analysis(True, True)
    assert is_valid_for_saccadic_analysis(True, None)  # older data: unknown doesn't invalidate
    assert is_valid_for_saccadic_analysis(True)
    assert not is_valid_for_saccadic_analysis(True, False)
    assert not is_valid_for_saccadic_analysis(False, True)
    assert not is_valid_for_saccadic_analysis(None, True)


def test_logger_writes_and_dashboard_reads_tracking_ok(tmp_path, monkeypatch):
    from src.dashboard import data_access
    from src.experiment import logger as logger_module

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = logger_module.ResultLogger("p1")
    log.log(
        TrialResult(
            index=0, phase="saccade", source=Target.DOT, target=Target.CROSS, saccade_duration_ms=50.0,
            grating_shown=True, contrast=0.1, orientation=Orientation.VERTICAL, responded=True,
            response_orientation=Orientation.VERTICAL, response_time_ms=300.0, outcome="correct",
            flash_during_saccade=True, tracking_ok=False,
        )
    )
    log.close()
    csv_path = next(tmp_path.glob("results_*.csv"))
    assert "tracking_ok" in csv_path.read_text().splitlines()[0]
    monkeypatch.setattr(data_access, "DATA_DIR", tmp_path)
    timestamp = csv_path.stem.removeprefix("results_")
    assert data_access.load_session_trials(timestamp)[0].tracking_ok is False

    # A CSV logged before this column existed loads as "unknown", not a failure.
    old = csv_path.read_text().splitlines()
    header, row = old[0].split(","), old[1].split(",")
    drop = header.index("tracking_ok")
    del header[drop], row[drop]
    csv_path.write_text(",".join(header) + "\n" + ",".join(row) + "\n")
    assert data_access.load_session_trials(timestamp)[0].tracking_ok is None

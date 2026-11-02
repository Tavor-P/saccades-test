"""Head-movement monitor (Neon IMU): tracker math, NeonGazeSource wiring against fake devices, flagged-trial wiring in the real session."""

import csv
import json
import math
import time
from types import SimpleNamespace

import pytest

from test_session import _complete_calibration_and_rt_test, _make_session
from test_tracking_quality import _run_valid_saccade, _session_in_first_trial

from include.eye_tracking.constants import (
    HEAD_DRIFT_WARN_DEG,
    HEAD_MAX_SPEED_DPS,
    HEAD_WARN_MIN_INTERVAL_S,
    HEAD_WARN_TEXT,
    NEON_SCENE_WIDTH,
)
from include.experiment.constants import TRACKING_LOSS_WINDOW_MS
from src.experiment.simulation import ParticipantProfile, simulate_session
from src.eye_tracking import neon_source
from src.eye_tracking.head_monitor import HeadMotionTracker, angle_between_deg
from src.eye_tracking.neon_source import NeonGazeSource

pytestmark = pytest.mark.usefixtures("staircase_no_jitter")


def yaw(deg: float):
    half = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))


STILL = (0.0, 0.0, 0.0, 1.0)


# -- tracker math ---------------------------------------------------------------------------------


def test_angle_between_quaternions():
    assert angle_between_deg(STILL, yaw(90)) == pytest.approx(90.0)
    assert angle_between_deg(STILL, STILL) == pytest.approx(0.0)
    assert angle_between_deg(yaw(10), yaw(-10)) == pytest.approx(20.0)
    negated = tuple(-v for v in yaw(30))  # q and -q are the same rotation
    assert angle_between_deg(STILL, negated) == pytest.approx(30.0)


def test_speed_is_the_gyro_norm_and_drift_needs_a_reference():
    tracker = HeadMotionTracker()
    tracker.add(0.0, (3.0, 4.0, 0.0), yaw(2))
    assert tracker.speed_dps == pytest.approx(5.0)
    assert tracker.drift_deg is None  # no reference yet
    assert tracker.set_reference()
    tracker.add(0.1, (0, 0, 0), yaw(7))
    assert tracker.drift_deg == pytest.approx(5.0)
    tracker.add(0.2, (0, 0, 0), yaw(3))
    assert tracker.drift_deg == pytest.approx(1.0)
    assert tracker.max_drift_deg == pytest.approx(5.0)  # the session maximum, not the current value


def test_set_reference_without_orientation_is_refused():
    tracker = HeadMotionTracker()
    assert not tracker.set_reference()
    tracker.add(0.0, (0, 0, 0), None)  # speed known, orientation unusable
    assert not tracker.set_reference()


def test_max_speed_between_uses_only_the_window():
    tracker = HeadMotionTracker()
    for t, speed in [(0.0, 1.0), (0.1, 50.0), (0.2, 2.0), (0.3, 3.0)]:
        tracker.add(t, (0, 0, speed), STILL)
    assert tracker.max_speed_between(0.15, 0.35) == pytest.approx(3.0)
    assert tracker.max_speed_between(0.0, 0.35) == pytest.approx(50.0)
    assert tracker.max_speed_between(5.0, 6.0) is None
    assert tracker.speed_spike_between(0.05, 0.15, 25.0)
    assert not tracker.speed_spike_between(0.15, 0.35, 25.0)
    assert not tracker.speed_spike_between(5.0, 6.0, 25.0)  # no data is not a spike


def test_moved_since_compares_to_the_orientation_at_t0():
    tracker = HeadMotionTracker()
    for i, angle in enumerate([0, 0, 1, 2, 4, 4]):
        tracker.add(i * 0.1, (0, 0, 0), yaw(angle))
    assert tracker.moved_since(0.0, 0.5, 3.0)  # 0 -> 4 deg
    assert not tracker.moved_since(0.0, 0.5, 5.0)
    assert not tracker.moved_since(0.4, 0.5, 1.0)  # already at 4 deg at t0 and still there
    assert not tracker.moved_since(9.0, 10.0, 0.1)  # nothing buffered


def test_ring_buffer_drops_old_samples_but_keeps_session_summaries():
    tracker = HeadMotionTracker(buffer_s=1.0)
    tracker.add(0.0, (0, 0, 90.0), STILL)
    for i in range(1, 30):
        tracker.add(i * 0.1, (0, 0, 1.0), STILL)
    assert tracker.max_speed_between(0.0, 0.5) is None  # fell out of the buffer
    assert tracker.summary()["max_speed_dps"] == pytest.approx(90.0)
    assert tracker.summary()["samples"] == 30


def test_unusable_samples_are_tolerated():
    tracker = HeadMotionTracker()
    tracker.add(0.0, (float("nan"), 0, 0), STILL)  # skipped
    tracker.add(0.1, None, STILL)  # skipped
    tracker.add(0.2, (0, 0, 1.0), (0, 0, 0, 0))  # zero quaternion: speed kept, orientation None
    tracker.add(0.3, (0, 0, 1.0), "garbage")
    assert tracker.summary()["samples"] == 2
    assert tracker.drift_deg is None


# -- NeonGazeSource against fake devices ----------------------------------------------------------


def _imu_datum(speed=0.5, angle=0.0, stamp=None):
    return SimpleNamespace(
        gyro_data=(0.0, 0.0, speed),
        accel_data=(0.0, 0.0, 9.8),
        quaternion=yaw(angle),
        timestamp_unix_seconds=stamp,
    )


class FakeNeon:
    """Gaze at ~1 kHz-ish plus an optional IMU; the real device keeps only the newest IMU datum, so does this."""

    def __init__(self, imu="ok", imu_speed=0.5):
        self.gaze_reads = 0
        self.imu_mode = imu
        self.imu_speed = imu_speed
        self.imu_angle = 0.0
        self.closed = False

    def __getattribute__(self, name):
        if name == "receive_imu_datum" and object.__getattribute__(self, "imu_mode") == "missing":
            raise AttributeError(name)  # an older device object without IMU support
        return object.__getattribute__(self, name)

    def receive_gaze_datum(self, timeout_seconds=None):
        self.gaze_reads += 1
        time.sleep(0.001)
        return SimpleNamespace(x=0.5 * NEON_SCENE_WIDTH, worn=True)

    def receive_imu_datum(self, timeout_seconds=None):
        if self.imu_mode == "raises":
            raise RuntimeError("no IMU on this device")
        if self.imu_mode == "silent":
            return None
        return _imu_datum(self.imu_speed, self.imu_angle)

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_imu_samples_flow_calibration_sets_the_reference_and_summary_and_csv(tmp_path):
    device = FakeNeon()
    source = NeonGazeSource(device=device)
    source.start()
    try:
        assert _wait_for(lambda: source.head_motion is not None and source.head_motion.has_data)
        assert source.head_motion.drift_deg is None
        source.calibrate(0.2, 0.5, 0.8)
        device.imu_angle = 4.0
        assert _wait_for(lambda: (source.head_motion.drift_deg or 0) > 3.5)
    finally:
        source.stop()
    summary = source.head_motion_summary(flagged_trials=2)
    assert summary["status"] == "ok" and summary["flagged_trials"] == 2
    assert summary["max_drift_deg"] == pytest.approx(4.0, abs=0.1) and summary["samples"] > 0
    out = tmp_path / "head_motion.csv"
    assert source.write_head_motion_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert rows and set(rows[0]) == {"t_s", "phone_unix_s", "speed_dps", "drift_deg"}
    assert device.closed


def test_device_without_imu_disables_the_monitor_but_gaze_keeps_flowing(capsys):
    source = NeonGazeSource(device=FakeNeon(imu="missing"))
    source.start()
    try:
        assert _wait_for(lambda: source.latest_sample().face_found)
    finally:
        source.stop()
    assert source.head_motion is None
    assert source.head_motion_summary()["status"].startswith("unavailable")
    assert capsys.readouterr().err.count("head-movement monitor off") == 1
    assert not source.write_head_motion_csv("never_written.csv")


def test_imu_exceptions_switch_the_monitor_off_once_without_touching_gaze(monkeypatch, capsys):
    monkeypatch.setattr(neon_source, "HEAD_POLL_HZ", 500.0)
    device = FakeNeon(imu="raises")
    source = NeonGazeSource(device=device)
    source.start()
    try:
        assert _wait_for(lambda: source.head_motion is None)
        reads = device.gaze_reads
        assert _wait_for(lambda: device.gaze_reads > reads + 5)  # the gaze loop is unaffected
    finally:
        source.stop()
    assert "unavailable" in source.head_motion_summary()["status"]
    assert capsys.readouterr().err.count("head-movement monitor off") == 1


def test_silent_imu_is_switched_off_after_the_no_data_timeout(monkeypatch):
    monkeypatch.setattr(neon_source, "HEAD_POLL_HZ", 500.0)
    monkeypatch.setattr(neon_source, "HEAD_NO_DATA_TIMEOUT_S", 0.05)
    source = NeonGazeSource(device=FakeNeon(imu="silent"))
    source.start()
    try:
        assert _wait_for(lambda: source.head_motion is None)
    finally:
        source.stop()
    assert "no IMU data" in source.head_motion_summary()["status"]


def test_malformed_imu_datum_is_counted_not_raised():
    device = FakeNeon()
    device.receive_imu_datum = lambda timeout_seconds=None: SimpleNamespace(gyro_data=None)
    source = NeonGazeSource(device=device)
    source.start()
    try:
        assert _wait_for(lambda: source.quality_summary()["receive_errors"] > 0)
    finally:
        source.stop()


def _gaze_reads_in(seconds: float, imu: str) -> int:
    device = FakeNeon(imu=imu)
    source = NeonGazeSource(device=device)
    source.start()
    time.sleep(seconds)
    source.stop()
    return device.gaze_reads


def test_imu_polling_overhead_on_the_gaze_loop_is_small():
    """Measured, not just asserted: the gaze loop here is paced by a 1 ms sleep, so reads/s shows what the extra thread costs it.
    The bound is generous (CI noise); the real number is printed with -s."""
    baseline = _gaze_reads_in(1.0, "missing")
    with_imu = _gaze_reads_in(1.0, "ok")
    print(f"\ngaze reads/s without IMU thread: {baseline}, with IMU thread: {with_imu}")
    assert with_imu >= 0.7 * baseline


# -- the real session ------------------------------------------------------------------------------


def _first_trial_with_head(fake_gaze, fake_time):
    fake_gaze.head_motion = HeadMotionTracker()
    session = _session_in_first_trial(fake_gaze, fake_time)
    return session, fake_gaze.head_motion


def test_head_speed_spike_at_the_flash_marks_the_trial_not_ok(fake_gaze, fake_time):
    session, tracker = _first_trial_with_head(fake_gaze, fake_time)
    tracker.add(session._flash_scheduled_at, (0, 0, HEAD_MAX_SPEED_DPS * 2), STILL)
    result = _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash=False)
    assert result.tracking_ok is False
    assert session.head_flagged_trials == 1
    assert session._valid_trial_count == 0  # excluded from analysis like a tracking loss


def test_head_spike_outside_the_flash_window_is_ignored(fake_gaze, fake_time):
    session, tracker = _first_trial_with_head(fake_gaze, fake_time)
    tracker.add(session._flash_scheduled_at - 3 * TRACKING_LOSS_WINDOW_MS / 1000, (0, 0, 500.0), STILL)
    tracker.add(session._flash_scheduled_at, (0, 0, 1.0), STILL)
    result = _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash=False)
    assert result.tracking_ok is True
    assert session.head_flagged_trials == 0


def test_no_tracker_means_no_change(fake_gaze, fake_time):
    session = _session_in_first_trial(fake_gaze, fake_time)
    result = _run_valid_saccade(session, fake_gaze, fake_time, lose_tracking_at_flash=False)
    assert result.tracking_ok is True
    assert "head_drift_deg" not in session.render_state()["hud"]
    assert session.pop_head_warning() is None


def test_hud_shows_head_drift_and_warning_is_rate_limited_and_between_trials_only(fake_gaze, fake_time):
    fake_gaze.head_motion = tracker = HeadMotionTracker()
    tracker.add(0.0, (0, 0, 0), STILL)
    tracker.set_reference()
    session = _make_session(fake_gaze)
    _complete_calibration_and_rt_test(session, fake_gaze, fake_time)
    assert session.render_state()["hud"]["phase"] == "AWAITING_SPACE"
    tracker.add(1.0, (0, 0, 0), yaw(HEAD_DRIFT_WARN_DEG - 1))
    assert session.pop_head_warning() is None  # within tolerance
    tracker.add(2.0, (0, 0, 0), yaw(HEAD_DRIFT_WARN_DEG + 1))
    assert session.render_state()["hud"]["head_drift_deg"] == pytest.approx(HEAD_DRIFT_WARN_DEG + 1)
    assert session.pop_head_warning() == HEAD_WARN_TEXT
    assert session.pop_head_warning() is None  # rate limited
    fake_time(HEAD_WARN_MIN_INTERVAL_S + 1)
    assert session.pop_head_warning() == HEAD_WARN_TEXT
    assert session.head_warnings == 2
    session.on_space()  # FOREPERIOD: no warning while a trial is under way
    fake_time(HEAD_WARN_MIN_INTERVAL_S + 1)
    assert session.render_state()["hud"]["phase"] != "AWAITING_SPACE"
    assert session.pop_head_warning() is None


# -- simulator (synthetic head channel, off by default) --------------------------------------------


def test_simulated_head_spikes_flag_every_flash_and_default_flags_none():
    quiet = simulate_session(max_trials=8, seed=3)
    assert quiet.session.head_flagged_trials == 0
    assert "head_drift_deg" not in quiet.session.render_state()["hud"]
    moving = simulate_session(max_trials=8, seed=3, profile=ParticipantProfile(head_spikes=((0.0, 1e6, HEAD_MAX_SPEED_DPS * 2),)))
    shown = [r for r in moving.saccade_results if r.grating_shown]
    assert shown and all(r.tracking_ok is False for r in shown)
    assert moving.session.head_flagged_trials == len(shown)
    assert moving.session.render_state()["hud"]["head_drift_deg"] is not None


def test_session_json_gets_the_head_motion_section(tmp_path, monkeypatch):
    from src.experiment import logger as logger_module

    monkeypatch.setattr(logger_module, "DATA_DIR", tmp_path)
    log = logger_module.ResultLogger("p1")
    log.set_head_motion({"status": "unavailable (no IMU)", "flagged_trials": 0})
    log.close()
    meta = json.loads(next(tmp_path.glob("*/*/session.json")).read_text())
    assert meta["head_motion"]["status"].startswith("unavailable")
    assert log.head_motion_path.name == "head_motion.csv"

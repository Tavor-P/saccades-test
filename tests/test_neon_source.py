from types import SimpleNamespace

import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from include.eye_tracking.types import GazeZone
from src.eye_tracking.gaze_factory import BACKEND_NEON, build_gaze_source
from src import timebase
from src.eye_tracking.neon_source import NeonGazeSource


def _x(ratio: float) -> float:
    return ratio * NEON_SCENE_WIDTH


def test_unavailable_until_a_device_is_connected():
    assert not NeonGazeSource().is_available
    assert NeonGazeSource(device=object()).is_available


def test_not_worn_or_nan_gaze_counts_as_lost_track():
    source = NeonGazeSource(device=object())
    source._ingest(_x(0.5), worn=False)
    assert not source.latest_sample().face_found
    source._ingest(float("nan"))
    assert source.latest_sample().zone is GazeZone.UNKNOWN


def test_default_mapping_classifies_zones():
    source = NeonGazeSource(device=object())
    for ratio, zone in [(0.2, GazeZone.LEFT), (0.5, GazeZone.CENTER), (0.8, GazeZone.RIGHT)]:
        source._ingest(_x(ratio))
        assert source.latest_sample().zone is zone


def test_calibration_remaps_position():
    source = NeonGazeSource(device=object())
    source.calibrate(0.1, 0.3, 0.5)
    source._ingest(_x(0.5))
    sample = source.latest_sample()
    assert sample.position == pytest.approx(1.0)
    assert sample.zone is GazeZone.RIGHT


def test_average_recent_ratio_needs_full_window_and_resets():
    source = NeonGazeSource(device=object())
    source.begin_calibration_sample()
    source._ingest(_x(0.4))
    assert source.average_recent_ratio() is None
    for _ in range(100):
        source._ingest(_x(0.4))
    assert source.average_recent_ratio() == pytest.approx(0.4)
    source.begin_calibration_sample()
    assert source.average_recent_ratio() is None


def test_loop_reads_from_device_and_stop_closes_it():
    import time

    closed = []
    device = SimpleNamespace(
        receive_gaze_datum=lambda timeout_seconds=None: SimpleNamespace(x=_x(0.8), worn=True),
        close=lambda: closed.append(True),
    )
    source = NeonGazeSource(device=device)
    source.start()
    time.sleep(0.1)
    source.stop()
    assert source.latest_sample().zone is GazeZone.RIGHT
    assert closed


def test_factory_builds_the_plain_neon_source_for_the_neon_backend():
    assert isinstance(build_gaze_source(BACKEND_NEON), NeonGazeSource)


def _fake_offset(mean_ms=200.0):
    est = SimpleNamespace(
        time_offset_ms=SimpleNamespace(mean=mean_ms, std=1.5),
        roundtrip_duration_ms=SimpleNamespace(mean=4.0),
    )
    return SimpleNamespace(estimate_time_offset=lambda: est)


def test_quality_summary_counts_samples_lost_and_gaps(monkeypatch):
    from src.eye_tracking import neon_source as module

    source = NeonGazeSource(device=object())  # built first: __init__ also reads the clock
    clock = iter([0.0, 0.005, 0.010, 0.100, 0.105])  # one 90 ms gap
    monkeypatch.setattr(timebase, "_clock", lambda: next(clock))
    source._ingest(_x(0.5))
    source._ingest(float("nan"))  # lost
    source._ingest(_x(0.5))
    source._ingest(_x(0.5))
    source._ingest(_x(0.5))
    q = source.quality_summary()
    assert q["samples"] == 5
    assert q["lost_samples"] == 1
    assert q["gaps_over_threshold"] == 1
    assert q["longest_gap_ms"] == pytest.approx(90.0)
    assert q["sample_rate_hz"] == pytest.approx(5 / 0.105, rel=0.01)
    assert q["latency_ms_median"] is None  # no clock offset estimated


def test_quality_summary_empty_session_does_not_crash():
    q = NeonGazeSource(device=object()).quality_summary()
    assert q["samples"] == 0 and q["sample_rate_hz"] is None and q["latency_ms_max"] is None


def test_latency_uses_clock_offset_with_correct_sign(monkeypatch):
    """local = device + offset: a sample the phone stamped 'offset+10ms' ago in
    its own clock was captured 10ms ago locally."""
    from src.eye_tracking import neon_source as module

    source = NeonGazeSource(device=_fake_offset(200.0))
    source._estimate_clock_offset()
    monkeypatch.setattr(timebase, "unix_now", lambda: 1000.0)
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - 0.010 - 0.200)  # phone clock 200ms behind PC
    q = source.quality_summary()
    assert q["latency_ms_median"] == pytest.approx(10.0, abs=0.01)
    assert q["clock_offset_ms"] == 200.0 and q["clock_roundtrip_ms"] == 4.0


def test_offset_estimate_failure_is_tolerated():
    source = NeonGazeSource(device=object())  # no estimate_time_offset
    source._estimate_clock_offset()
    source._ingest(_x(0.5), True, device_timestamp_s=1.0)
    assert source.quality_summary()["latency_ms_median"] is None


def test_dead_stream_reports_no_gaze_instead_of_repeating_last_sample(monkeypatch):
    from src.eye_tracking import neon_source as module

    source = NeonGazeSource(device=object())
    source._ingest(_x(0.8))
    assert source.latest_sample().face_found and source.is_available
    real = timebase._clock
    monkeypatch.setattr(timebase, "_clock", lambda: real() + 5.0)  # 5 s with no samples
    assert not source.latest_sample().face_found
    assert source.latest_sample().zone is GazeZone.UNKNOWN
    assert not source.is_available


def test_mapper_is_safe_to_read_while_another_thread_appends():
    import threading

    from src.eye_tracking.gaze_mapping import GazeMapper

    mapper = GazeMapper()
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            mapper.map(0.5, 0.0)

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(3000):
            mapper.average_recent_ratio()
            mapper.reset_ratio_history()
    finally:
        stop.set()
        t.join()


def test_receive_errors_are_counted_logged_once_and_loop_survives(capsys):
    import time

    calls = {"n": 0}

    def receive(timeout_seconds=None):
        calls["n"] += 1
        if calls["n"] <= 5:
            raise ConnectionError("phone unreachable")
        return SimpleNamespace(x=_x(0.8), worn=True)

    source = NeonGazeSource(device=SimpleNamespace(receive_gaze_datum=receive, close=lambda: None))
    source.start()
    time.sleep(0.2)
    source.stop()
    summary = source.quality_summary()
    assert summary["receive_errors"] == 5
    assert summary["last_receive_error"] == "receive_gaze_datum: ConnectionError: phone unreachable"
    assert capsys.readouterr().err.count("phone unreachable") == 1  # repeats counted, not re-logged
    assert source.latest_sample().zone is GazeZone.RIGHT  # recovered after the errors


def test_ingest_exception_does_not_kill_the_thread():
    import time

    source = NeonGazeSource(
        device=SimpleNamespace(receive_gaze_datum=lambda timeout_seconds=None: SimpleNamespace(x=_x(0.8), worn=True), close=lambda: None)
    )
    source._ingest = lambda *a: (_ for _ in ()).throw(ValueError("bad datum"))
    source.start()
    time.sleep(0.1)
    assert source._thread.is_alive()
    source.stop()
    assert source.quality_summary()["receive_errors"] > 0


def test_quality_summary_reports_zero_errors_by_default():
    summary = NeonGazeSource(device=object()).quality_summary()
    assert summary["receive_errors"] == 0 and summary["last_receive_error"] is None

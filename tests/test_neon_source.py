import time
from types import SimpleNamespace

import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from include.eye_tracking.types import GazeZone
from src.eye_tracking.gaze_factory import build_gaze_source
from src import timebase
from src.eye_tracking.neon_source import NeonGazeSource


def _x(ratio: float) -> float:
    return ratio * NEON_SCENE_WIDTH


def test_availability_lost_track_zones_calibration_window_history_and_factory():
    from include.eye_tracking.constants import NEON_HISTORY_SAMPLES

    assert not NeonGazeSource().is_available
    source = NeonGazeSource(device=object())
    assert source.is_available
    assert isinstance(build_gaze_source(), NeonGazeSource)

    source._ingest(_x(0.5), worn=False)
    assert not source.latest_sample().face_found  # not worn = lost track
    source._ingest(float("nan"))
    assert source.latest_sample().zone is GazeZone.UNKNOWN
    for ratio, zone in [(0.2, GazeZone.LEFT), (0.5, GazeZone.CENTER), (0.8, GazeZone.RIGHT)]:
        source._ingest(_x(ratio))
        assert source.latest_sample().zone is zone, ratio

    source.begin_calibration_sample()
    source._ingest(_x(0.4))
    assert source.average_recent_ratio() is None  # window not full yet
    for _ in range(100):
        source._ingest(_x(0.4))
    assert source.average_recent_ratio() == pytest.approx(0.4)
    source.begin_calibration_sample()
    assert source.average_recent_ratio() is None

    source.calibrate(0.1, 0.3, 0.5)
    source._ingest(_x(0.5))
    sample = source.latest_sample()
    assert sample.position == pytest.approx(1.0) and sample.zone is GazeZone.RIGHT

    for _ in range(NEON_HISTORY_SAMPLES + 50):
        source._ingest(_x(0.5))
    assert len(source.samples_since(None)) == NEON_HISTORY_SAMPLES  # bounded


def test_loop_reads_from_device_and_stop_closes_it():
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


def _fake_offset(mean_ms=200.0):
    est = SimpleNamespace(
        time_offset_ms=SimpleNamespace(mean=mean_ms, std=1.5),
        roundtrip_duration_ms=SimpleNamespace(mean=4.0),
    )
    return SimpleNamespace(estimate_time_offset=lambda: est)


def test_quality_summary_counts_samples_lost_and_gaps_and_handles_an_empty_session(monkeypatch):
    empty = NeonGazeSource(device=object()).quality_summary()
    assert empty["samples"] == 0 and empty["sample_rate_hz"] is None and empty["latency_ms_max"] is None
    assert empty["receive_errors"] == 0 and empty["last_receive_error"] is None
    assert empty["clock_offset_history"] == []

    source = NeonGazeSource(device=object())  # built first: __init__ also reads the clock
    clock = iter([0.0, 0.005, 0.010, 0.100, 0.105])  # one 90 ms gap
    monkeypatch.setattr(timebase, "_clock", lambda: next(clock))
    for x in (_x(0.5), float("nan"), _x(0.5), _x(0.5), _x(0.5)):
        source._ingest(x)
    q = source.quality_summary()
    assert (q["samples"], q["lost_samples"], q["gaps_over_threshold"]) == (5, 1, 1)
    assert q["longest_gap_ms"] == pytest.approx(90.0)
    assert q["sample_rate_hz"] == pytest.approx(5 / 0.105, rel=0.01)
    assert q["latency_ms_median"] is None  # no clock offset estimated


def _source_with_clock(monkeypatch, offset_ms=200.0):
    """A source whose clock anchor is (unix 1000.0 == perf 50.0) and whose perf clock reads `state["t"]`."""
    state = {"t": 50.0}
    source = NeonGazeSource(device=_fake_offset(offset_ms))
    source._estimate_clock_offset()
    source._anchor = timebase.Anchor(unix_s=1000.0, perf_s=50.0)
    monkeypatch.setattr(timebase, "_clock", lambda: state["t"])
    return source, state


def test_samples_are_stamped_with_capture_time_using_the_offset_with_the_right_sign(monkeypatch):
    """local = device + offset: a sample the phone stamped offset+16 ms ago in its own clock was captured 16 ms ago
    locally (a flipped sign once gave a bogus 414 ms)."""
    source, state = _source_with_clock(monkeypatch)
    assert source.timestamps_are_capture_time
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - 0.016 - 0.200)  # phone clock 200ms behind PC
    assert source.latest_sample().timestamp == pytest.approx(50.0 - 0.016)
    state["t"] = 50.005
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - 0.011 - 0.200)  # 5 ms later, same 16 ms latency
    assert [s.timestamp for s in source.samples_since(None)] == pytest.approx([50.0 - 0.016, 50.005 - 0.016])
    q = source.quality_summary()
    assert q["latency_ms_median"] == pytest.approx(16.0, abs=0.01)
    assert q["clock_offset_ms"] == 200.0 and q["clock_roundtrip_ms"] == 4.0
    assert source.gaze_latency_ms == pytest.approx(16.0, abs=0.01)


def test_stamps_fall_back_to_arrival_or_are_clamped(monkeypatch):
    state = {"t": 7.0}
    monkeypatch.setattr(timebase, "_clock", lambda: state["t"])
    source = NeonGazeSource(device=object())  # no estimate_time_offset: the failure is tolerated
    source._estimate_clock_offset()
    assert not source.timestamps_are_capture_time
    source._ingest(_x(0.5), True, device_timestamp_s=1.0)
    assert source.latest_sample().timestamp == 7.0
    assert source.quality_summary()["latency_ms_median"] is None

    source, state = _source_with_clock(monkeypatch)
    state["t"] = 8.0
    source._ingest(_x(0.5), True, None)  # no device timestamp
    assert source.latest_sample().timestamp == 8.0

    source, state = _source_with_clock(monkeypatch)
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - 5.0 - 0.200)  # "5 s latency": a bad offset
    assert source.latest_sample().timestamp == 50.0
    state["t"] = 50.01
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 + 3.0 - 0.200)  # captured in the future
    assert source.latest_sample().timestamp == pytest.approx(50.01)


def test_stamps_are_strictly_increasing_and_samples_since_never_skips(monkeypatch):
    source, state = _source_with_clock(monkeypatch)
    for dev in (0.020, 0.020, 0.0195, 0.021):  # repeated and backwards device timestamps
        source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - dev - 0.200)
    stamps = [s.timestamp for s in source.samples_since(None)]
    assert all(b > a for a, b in zip(stamps, stamps[1:])) and len(stamps) == 4
    assert [s.timestamp for s in source.samples_since(stamps[1])] == stamps[2:]
    assert source.samples_since(stamps[-1]) == []


def _est(mean, std):
    return SimpleNamespace(
        time_offset_ms=SimpleNamespace(mean=mean, std=std), roundtrip_duration_ms=SimpleNamespace(mean=4.0)
    )


def test_clock_offset_is_reestimated_in_the_background_noisy_ones_ignored_and_accepted_ones_kept(monkeypatch):
    from src.eye_tracking import neon_source as module

    monkeypatch.setattr(module, "NEON_CLOCK_REESTIMATE_S", 0.02)
    calls = []

    def estimate():
        calls.append(1)
        if len(calls) == 1:
            return _est(200.0, 1.0)
        if len(calls) == 2:
            return _est(900.0, 500.0)  # noisy: ignored
        return _est(210.0, 1.0)

    device = SimpleNamespace(estimate_time_offset=estimate, receive_gaze_datum=lambda timeout_seconds=None: None, close=lambda: None)
    source = NeonGazeSource(device=device)
    source.start()
    time.sleep(0.3)
    source.stop()
    assert len(calls) >= 4
    assert 205.0 <= source._clock_offset_ms <= 210.0  # blended with the previous value, not replaced outright
    history = source.quality_summary()["clock_offset_history"]
    assert history[0]["offset_ms"] == 200.0 and len(history) >= 3
    assert all(h["offset_ms"] != 900.0 for h in history)
    assert [h["phone_unix_s"] for h in history] == sorted(h["phone_unix_s"] for h in history)
    assert history[-1]["offset_ms"] == round(source._clock_offset_ms, 3)  # the last entry is the offset still in force


def test_dead_stream_reports_no_gaze_instead_of_repeating_last_sample(monkeypatch):
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


def test_receive_and_ingest_errors_are_counted_logged_once_and_the_loop_survives(capsys):
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

    source = NeonGazeSource(
        device=SimpleNamespace(receive_gaze_datum=lambda timeout_seconds=None: SimpleNamespace(x=_x(0.8), worn=True), close=lambda: None)
    )
    source._ingest = lambda *a: (_ for _ in ()).throw(ValueError("bad datum"))
    source.start()
    time.sleep(0.1)
    assert source._thread.is_alive()
    source.stop()
    assert source.quality_summary()["receive_errors"] > 0


def _quiet_device(**extra):
    return SimpleNamespace(receive_gaze_datum=lambda timeout_seconds=None: None, close=lambda: None, **extra)


@pytest.mark.parametrize(
    "estimate, reason",
    [
        (lambda: None, "returned None"),  # what the Real-Time API gives for an unreachable time-echo port / old Companion app
        (lambda: (_ for _ in ()).throw(OSError("connection refused")), "OSError: connection refused"),
    ],
)
def test_a_failed_clock_offset_is_loud_and_recorded_for_session_json(capsys, estimate, reason):
    source = NeonGazeSource(device=_quiet_device(estimate_time_offset=estimate, phone_ip="192.168.43.1"), address="192.168.43.1")
    assert source.quality_summary()["clock_offset_status"] == "not started"
    source.start()
    source.stop()
    status = source.quality_summary()["clock_offset_status"]
    assert status.startswith("failed (") and reason in status
    assert not source.timestamps_are_capture_time
    err = capsys.readouterr().err
    assert "WARNING: could not measure the Neon phone's clock offset at 192.168.43.1" in err and reason in err


def test_a_hanging_clock_offset_estimate_times_out_instead_of_blocking_start(monkeypatch):
    import threading

    from src.eye_tracking import neon_source as module

    monkeypatch.setattr(module, "NEON_CLOCK_ESTIMATE_TIMEOUT_S", 0.05)
    release = threading.Event()
    source = NeonGazeSource(device=_quiet_device(estimate_time_offset=lambda: release.wait(5)))
    started = time.perf_counter()
    source.start()
    assert time.perf_counter() - started < 2.0
    source.stop()
    release.set()
    assert "TimeoutError" in source.quality_summary()["clock_offset_status"]


def test_a_good_clock_offset_reports_ok_and_the_phone_ip_is_recorded_with_a_warning_when_it_differs(capsys):
    source = NeonGazeSource(device=_quiet_device(estimate_time_offset=lambda: _est(200.0, 1.0), phone_ip="10.0.0.7"), address="192.168.43.1")
    source.start()
    source.stop()
    assert source.quality_summary()["clock_offset_status"] == "ok" and source._clock_offset_ms == 200.0
    assert source.phone_ip == "10.0.0.7"
    assert "phone reports its address as 10.0.0.7" in capsys.readouterr().err

    same = NeonGazeSource(device=_quiet_device(estimate_time_offset=lambda: _est(200.0, 1.0), phone_ip="192.168.43.1"), address="192.168.43.1")
    same.start()
    same.stop()
    assert same.phone_ip == "192.168.43.1" and "phone reports" not in capsys.readouterr().err
    no_ip = NeonGazeSource(device=_quiet_device(estimate_time_offset=lambda: _est(200.0, 1.0)), address="192.168.43.1")
    no_ip.start()
    no_ip.stop()
    assert no_ip.phone_ip is None  # fakes without the attribute still work

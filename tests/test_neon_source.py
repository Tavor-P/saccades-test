from types import SimpleNamespace

import pytest

from include.eye_tracking.constants import NEON_SCENE_WIDTH
from include.eye_tracking.types import GazeZone
from src.eye_tracking.gaze_factory import BACKEND_NEON, BACKEND_WEBCAM, build_gaze_source
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


def test_factory_builds_neon_by_default():
    assert isinstance(build_gaze_source(BACKEND_NEON), NeonGazeSource)
    assert BACKEND_WEBCAM != BACKEND_NEON


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
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
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
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    source._ingest(_x(0.5), True, device_timestamp_s=1000.0 - 0.010 - 0.200)  # phone clock 200ms behind PC
    q = source.quality_summary()
    assert q["latency_ms_median"] == pytest.approx(10.0, abs=0.01)
    assert q["clock_offset_ms"] == 200.0 and q["clock_roundtrip_ms"] == 4.0


def test_offset_estimate_failure_is_tolerated():
    source = NeonGazeSource(device=object())  # no estimate_time_offset
    source._estimate_clock_offset()
    source._ingest(_x(0.5), True, device_timestamp_s=1.0)
    assert source.quality_summary()["latency_ms_median"] is None

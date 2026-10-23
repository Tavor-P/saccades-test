import time
from types import SimpleNamespace

import pytest

from include.eye_tracking.types import GazeZone
from src.eye_tracking.neon_screen_source import NeonScreenGazeSource
from src.eye_tracking.screen_gaze_mapper import ScreenGaze

SCREEN = (1920, 1080)


class FakeScreenMapper:
    def __init__(self) -> None:
        self.result: ScreenGaze | None = ScreenGaze(1344.0, 540.0, 0.01)  # 70% across, mid-height
        self.markers_detected = 4
        self.located = True
        self.scene_frames = 0

    def process_scene(self, frame) -> None:
        self.scene_frames += 1

    def map_gaze(self, gaze):
        return self.result


def _datum(x=800.0, worn=True):
    return SimpleNamespace(x=x, y=600.0, worn=worn, timestamp_unix_seconds=time.time())


def _imu(gx=0.0, gy=0.0, gz=0.0):
    return SimpleNamespace(gyro_data=SimpleNamespace(x=gx, y=gy, z=gz))


@pytest.fixture
def source():
    return NeonScreenGazeSource(SCREEN, device=object(), screen_mapper=FakeScreenMapper())


def test_ratio_is_screen_fraction_not_scene_pixels(source):
    source._ingest_screen(_datum())
    sample = source.latest_sample()
    assert sample.face_found
    assert sample.ratio == pytest.approx(0.7)
    assert sample.ratio_y == pytest.approx(0.5)
    assert sample.mapping_age_ms == pytest.approx(10.0)
    assert sample.zone is GazeZone.RIGHT


def test_unlocated_screen_is_reported_as_lost_not_scene_coordinates(source):
    source._screen_mapper.result = None
    source._ingest_screen(_datum())
    sample = source.latest_sample()
    assert not sample.face_found and sample.ratio is None
    assert source.quality_summary()["screen_unlocated_samples"] == 1


def test_not_worn_is_lost_but_not_counted_as_unlocated(source):
    source._ingest_screen(_datum(worn=False))
    assert not source.latest_sample().face_found
    assert source.quality_summary()["screen_unlocated_samples"] == 0


def test_head_speed_is_the_peak_gyro_magnitude_and_flags_fast_movement(source):
    source._ingest_imu(_imu(gx=3, gy=4))  # magnitude 5
    source._ingest_imu(_imu(gz=60))
    source._ingest_screen(_datum())
    assert source.latest_sample().head_speed_dps == pytest.approx(60)
    summary = source.quality_summary()
    assert summary["head_moving_fraction"] == pytest.approx(1.0)
    assert summary["source"] == "pupil_labs_neon_screen_markers"


def test_head_speed_is_none_without_imu_data(source):
    source._ingest_screen(_datum())
    assert source.latest_sample().head_speed_dps is None
    assert source.quality_summary()["head_moving_fraction"] is None


def test_threads_feed_all_three_streams_and_stop_cleanly():
    mapper = FakeScreenMapper()
    device = SimpleNamespace(
        receive_gaze_datum=lambda timeout_seconds=None: _datum(),
        receive_scene_video_frame=lambda timeout_seconds=None: object(),
        receive_imu_datum=lambda timeout_seconds=None: _imu(gx=10),
        estimate_time_offset=lambda: (_ for _ in ()).throw(RuntimeError("no clock")),
        close=lambda: None,
    )
    source = NeonScreenGazeSource(SCREEN, device=device, screen_mapper=mapper)
    source.start()
    time.sleep(0.15)
    source.stop()
    assert source.latest_sample().ratio == pytest.approx(0.7)
    assert mapper.scene_frames > 0
    assert source.latest_sample().head_speed_dps == pytest.approx(10)
    assert source.markers_detected == 4 and source.screen_located


def test_stream_errors_are_counted_per_stream_and_threads_survive(capsys):
    mapper = FakeScreenMapper()

    def boom(timeout_seconds=None):
        raise OSError("stream dropped")

    device = SimpleNamespace(
        receive_gaze_datum=lambda timeout_seconds=None: _datum(),
        receive_scene_video_frame=boom,
        receive_imu_datum=boom,
        estimate_time_offset=lambda: (_ for _ in ()).throw(RuntimeError("no clock")),
        close=lambda: None,
    )
    source = NeonScreenGazeSource(SCREEN, device=device, screen_mapper=mapper)
    source.start()
    time.sleep(0.2)
    assert all(t.is_alive() for t in source._extra_threads)
    source.stop()
    summary = source.quality_summary()
    assert summary["receive_errors"] >= 2
    assert "OSError: stream dropped" in summary["last_receive_error"]
    assert summary["source"] == "pupil_labs_neon_screen_markers"
    assert source.latest_sample().ratio == pytest.approx(0.7)  # gaze stream unaffected
    err = capsys.readouterr().err
    assert err.count("stream dropped") == 1  # same message logged once
    assert "[neon]" in err


def test_mapper_exception_in_scene_loop_does_not_kill_thread():
    class BadMapper(FakeScreenMapper):
        def process_scene(self, frame):
            raise ValueError("bad frame")

    device = SimpleNamespace(
        receive_gaze_datum=lambda timeout_seconds=None: _datum(),
        receive_scene_video_frame=lambda timeout_seconds=None: object(),
        receive_imu_datum=lambda timeout_seconds=None: _imu(),
        estimate_time_offset=lambda: (_ for _ in ()).throw(RuntimeError("no clock")),
        close=lambda: None,
    )
    source = NeonScreenGazeSource(SCREEN, device=device, screen_mapper=BadMapper())
    source.start()
    time.sleep(0.15)
    assert all(t.is_alive() for t in source._extra_threads)
    source.stop()
    assert "ValueError: bad frame" in source.quality_summary()["last_receive_error"]


def test_screen_samples_are_stamped_with_capture_time(monkeypatch):
    from src import timebase

    source = NeonScreenGazeSource(SCREEN, device=_fake_offset_device(200.0), screen_mapper=FakeScreenMapper())
    source._estimate_clock_offset()
    source._anchor = timebase.Anchor(unix_s=1000.0, perf_s=50.0)
    monkeypatch.setattr(timebase, "_clock", lambda: 50.0)
    datum = _datum()
    datum.timestamp_unix_seconds = 1000.0 - 0.016 - 0.200
    source._ingest_screen(datum)
    assert source.timestamps_are_capture_time
    assert source.latest_sample().timestamp == pytest.approx(50.0 - 0.016)
    assert [s.timestamp for s in source.samples_since(None)] == [pytest.approx(50.0 - 0.016)]
    assert source.gaze_latency_ms == pytest.approx(16.0)


def _fake_offset_device(mean_ms):
    est = SimpleNamespace(
        time_offset_ms=SimpleNamespace(mean=mean_ms, std=1.0), roundtrip_duration_ms=SimpleNamespace(mean=4.0)
    )
    return SimpleNamespace(estimate_time_offset=lambda: est)

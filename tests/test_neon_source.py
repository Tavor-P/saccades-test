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

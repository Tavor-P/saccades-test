from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from include.eye_tracking.constants import CAMERA_INDEX, TRACKING_FRAME_WIDTH
from include.eye_tracking.types import GazeZone
from src.eye_tracking.gaze_tracker import GazeTracker
from src.eye_tracking.iohub_source import IOHubGazeSource
from src.eye_tracking.webcam_source import WebcamGazeSource


@pytest.fixture(scope="module")
def tracker() -> GazeTracker:
    # Constructing this loads a real MediaPipe FaceLandmarker (a few seconds)
    # - shared across this file's tests so it's only paid once. Tests reset
    # its ratio history themselves, so sharing the instance is safe.
    return GazeTracker()


def test_average_recent_ratio_is_none_until_the_window_is_full(tracker):
    tracker.reset_ratio_history()
    assert tracker.average_recent_ratio() is None

    for _ in range(tracker._ratio_history.maxlen - 1):
        tracker._ratio_history.append(0.5)
    assert tracker.average_recent_ratio() is None  # one sample short of a full window


def test_average_recent_ratio_once_the_window_is_full(tracker):
    tracker.reset_ratio_history()
    for _ in range(tracker._ratio_history.maxlen):
        tracker._ratio_history.append(0.42)
    assert tracker.average_recent_ratio() == pytest.approx(0.42)


def test_reset_ratio_history_discards_stale_samples(tracker):
    # Regression test for the calibration bug this fixes: without a reset
    # between calibration targets, samples left over from the *previous*
    # fixation diluted the next one's average toward it, collapsing the
    # left/right calibration span toward zero.
    tracker.reset_ratio_history()
    for _ in range(tracker._ratio_history.maxlen):
        tracker._ratio_history.append(0.2)  # e.g. leftover from "look left"

    tracker.reset_ratio_history()
    for _ in range(tracker._ratio_history.maxlen):
        tracker._ratio_history.append(0.8)  # fresh "look right" samples

    assert tracker.average_recent_ratio() == pytest.approx(0.8)  # not a blend of 0.2 and 0.8


def test_webcam_source_begin_calibration_sample_delegates_to_tracker(monkeypatch):
    monkeypatch.setattr(WebcamGazeSource, "__init__", lambda self: None)
    source = WebcamGazeSource()
    source._tracker = Mock()

    source.begin_calibration_sample()

    source._tracker.reset_ratio_history.assert_called_once()


def test_iohub_source_begin_calibration_sample_is_a_noop():
    source = IOHubGazeSource(device_config={}, target_positions={GazeZone.LEFT: (0.0, 0.0)}, hit_radius=1.0)
    source.begin_calibration_sample()  # should not raise


def test_webcam_source_passes_camera_index_to_camera(monkeypatch):
    mock_camera_cls = Mock()
    monkeypatch.setattr("src.eye_tracking.webcam_source.Camera", mock_camera_cls)
    monkeypatch.setattr("src.eye_tracking.webcam_source.GazeTracker", Mock())

    WebcamGazeSource(camera_index=3)

    mock_camera_cls.assert_called_once_with(3)


def test_webcam_source_defaults_to_the_configured_camera_index(monkeypatch):
    mock_camera_cls = Mock()
    monkeypatch.setattr("src.eye_tracking.webcam_source.Camera", mock_camera_cls)
    monkeypatch.setattr("src.eye_tracking.webcam_source.GazeTracker", Mock())

    WebcamGazeSource()

    mock_camera_cls.assert_called_once_with(CAMERA_INDEX)


def test_process_downscales_frames_wider_than_the_tracking_width(tracker, monkeypatch):
    tracker.reset_ratio_history()
    wide_frame = np.zeros((300, TRACKING_FRAME_WIDTH * 2), dtype=np.uint8)
    resize_calls = []
    real_resize = cv2.resize
    monkeypatch.setattr(
        "src.eye_tracking.gaze_tracker.cv2.resize",
        lambda frame, size: resize_calls.append(size) or real_resize(frame, size),
    )

    tracker.process(wide_frame)

    assert resize_calls == [(TRACKING_FRAME_WIDTH, 150)]


def test_process_does_not_resize_frames_at_or_under_the_tracking_width(tracker, monkeypatch):
    tracker.reset_ratio_history()
    narrow_frame = np.zeros((240, TRACKING_FRAME_WIDTH), dtype=np.uint8)
    monkeypatch.setattr(
        "src.eye_tracking.gaze_tracker.cv2.resize", Mock(side_effect=AssertionError("should not resize"))
    )

    tracker.process(narrow_frame)  # would raise if resize were called


def _reset_smoothing(tracker: GazeTracker) -> None:
    tracker._recent_positions.clear()
    tracker._smoothed_position = None


def test_smooth_first_reading_is_returned_immediately(tracker):
    # No lag on the very first sample - nothing to smooth against yet.
    _reset_smoothing(tracker)
    assert tracker._smooth(0.7) == pytest.approx(0.7)


def test_smooth_rejects_a_single_outlier_via_the_median(tracker):
    _reset_smoothing(tracker)
    for _ in range(tracker._recent_positions.maxlen):
        tracker._smooth(0.5)  # steady fixation, buffer full of 0.5s

    # One stray misdetection shouldn't move the smoothed value much at all -
    # the median of mostly-0.5 readings plus one 0.9 is still 0.5.
    result = tracker._smooth(0.9)
    assert result == pytest.approx(0.5, abs=0.05)


def test_smooth_ema_converges_toward_a_new_steady_value(tracker):
    _reset_smoothing(tracker)
    for _ in range(tracker._recent_positions.maxlen):
        tracker._smooth(0.0)  # settle at 0.0 first

    # Feed a real, sustained move to 1.0. The median window (maxlen 5) needs
    # a majority of 1.0 readings before its own output even starts moving -
    # so the first couple of calls stay at 0.0 (still-mostly-0.0 buffer),
    # matching the intent of rejecting brief blips rather than real moves.
    for _ in range(tracker._recent_positions.maxlen // 2):
        tracker._smooth(1.0)

    # Once the median itself has flipped to 1.0, the EMA should step toward
    # it gradually, not jump there instantly (that's the point of smoothing).
    first_moved_step = tracker._smooth(1.0)
    assert 0.0 < first_moved_step < 1.0

    for _ in range(50):  # enough further EMA steps to converge close to 1.0
        result = tracker._smooth(1.0)
    assert result == pytest.approx(1.0, abs=0.01)


def test_webcam_source_latest_frame_delegates_to_camera(monkeypatch):
    monkeypatch.setattr(WebcamGazeSource, "__init__", lambda self: None)
    source = WebcamGazeSource()
    source._camera = Mock()
    source._camera.read.return_value = "a-frame"

    assert source.latest_frame() == "a-frame"

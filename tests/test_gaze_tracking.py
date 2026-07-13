from unittest.mock import Mock

import pytest

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

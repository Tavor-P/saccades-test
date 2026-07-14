from unittest.mock import MagicMock, patch

from src.eye_tracking.camera import Camera, list_available_cameras


def _fake_capture_factory(open_indices):
    def factory(index, *args, **kwargs):
        capture = MagicMock()
        capture.isOpened.return_value = index in open_indices
        return capture

    return factory


def test_list_available_cameras_returns_only_indices_that_open():
    with patch("src.eye_tracking.camera.cv2.VideoCapture", side_effect=_fake_capture_factory({0, 2})):
        assert list_available_cameras(max_index=4) == [0, 2]


def test_list_available_cameras_returns_empty_list_when_none_open():
    with patch("src.eye_tracking.camera.cv2.VideoCapture", side_effect=_fake_capture_factory(set())):
        assert list_available_cameras(max_index=3) == []


def test_camera_stores_the_given_index():
    assert Camera(index=2)._index == 2

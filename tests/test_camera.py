from unittest.mock import MagicMock, patch

import pytest

from src.eye_tracking.camera import Camera, list_available_cameras


def _fake_system(num_cameras: int) -> MagicMock:
    system = MagicMock()
    cam_list = MagicMock()
    cam_list.GetSize.return_value = num_cameras
    system.GetCameras.return_value = cam_list
    return system


def test_list_available_cameras_returns_an_index_per_detected_camera():
    system = _fake_system(2)
    with patch("src.eye_tracking.camera.PySpin.System.GetInstance", return_value=system):
        assert list_available_cameras(max_index=4) == [0, 1]


def test_list_available_cameras_caps_at_max_index():
    system = _fake_system(5)
    with patch("src.eye_tracking.camera.PySpin.System.GetInstance", return_value=system):
        assert list_available_cameras(max_index=3) == [0, 1, 2]


def test_list_available_cameras_returns_empty_list_when_none_found():
    system = _fake_system(0)
    with patch("src.eye_tracking.camera.PySpin.System.GetInstance", return_value=system):
        assert list_available_cameras(max_index=3) == []


def test_camera_stores_the_given_index():
    assert Camera(index=2)._index == 2


def test_camera_start_releases_system_and_cam_list_if_setup_fails():
    system = MagicMock()
    cam_list = MagicMock()
    cam_list.GetByIndex.side_effect = RuntimeError("camera unavailable")
    system.GetCameras.return_value = cam_list

    with patch("src.eye_tracking.camera.PySpin.System.GetInstance", return_value=system):
        camera = Camera(index=0)
        with pytest.raises(RuntimeError):
            camera.start()

    cam_list.Clear.assert_called_once()
    system.ReleaseInstance.assert_called_once()
    assert camera._system is None
    assert camera._cam_list is None

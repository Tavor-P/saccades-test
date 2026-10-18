"""End-to-end geometry check without hardware: render the real marker pads onto a
fake screen, shrink/shift it into a fake scene-camera frame, and confirm a gaze
point at a known scene pixel maps back to the right screen pixel."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("pupil_labs.real_time_screen_gaze")

from src.eye_tracking.screen_gaze_mapper import ScreenGazeMapper  # noqa: E402
from src.eye_tracking.screen_markers import marker_pad_image, marker_pads, marker_verts, pad_size  # noqa: E402

SCREEN = (1920, 1080)
SCALE, OFFSET = 0.5, (320, 330)  # screen px -> scene px: x*0.5 + 320, y*0.5 + 330
SCENE = (1600, 1200)


def _scene_frame(draw_markers: bool = True):
    screen = np.full((SCREEN[1], SCREEN[0]), 90, np.uint8)
    if draw_markers:
        for marker_id, (px, py) in marker_pads(*SCREEN).items():
            pad = pad_size()
            screen[py : py + pad, px : px + pad] = marker_pad_image(marker_id)
    small = cv2.resize(screen, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    scene = np.full((SCENE[1], SCENE[0]), 40, np.uint8)
    scene[OFFSET[1] : OFFSET[1] + small.shape[0], OFFSET[0] : OFFSET[0] + small.shape[1]] = small
    return SimpleNamespace(bgr_pixels=cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR))


def _calibration():
    k = np.array([[800.0, 0, SCENE[0] / 2], [0, 800.0, SCENE[1] / 2], [0, 0, 1]])
    return {"scene_camera_matrix": k.reshape(1, 3, 3), "scene_distortion_coefficients": np.zeros((1, 8))}


def _gaze(screen_x, screen_y):
    return (screen_x * SCALE + OFFSET[0], screen_y * SCALE + OFFSET[1], True, 0.0)


def test_maps_scene_gaze_back_to_the_right_screen_pixels():
    mapper = ScreenGazeMapper(_calibration(), SCREEN)
    mapper.process_scene(_scene_frame())
    assert mapper.markers_detected == 4 and mapper.located
    for sx, sy in [(960, 540), (400, 300), (1500, 800)]:
        result = mapper.map_gaze(_gaze(sx, sy))
        assert result is not None
        assert result.x == pytest.approx(sx, abs=12)
        assert result.y == pytest.approx(sy, abs=12)


def test_gaze_is_lost_once_markers_disappear_past_the_hold():
    mapper = ScreenGazeMapper(_calibration(), SCREEN, hold_s=0.0)
    mapper.process_scene(_scene_frame())
    assert mapper.map_gaze(_gaze(960, 540)) is not None
    mapper.process_scene(_scene_frame(draw_markers=False))
    assert not mapper.located
    assert mapper.map_gaze(_gaze(960, 540)) is None


def test_marker_verts_sit_inside_their_pads():
    verts = marker_verts(*SCREEN)
    assert verts[0][0] == (22, 22)
    assert verts[2][2] == (SCREEN[0] - 22, SCREEN[1] - 22)

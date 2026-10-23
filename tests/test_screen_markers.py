"""Geometry, luminance and detectability of the corner AprilTags (src/eye_tracking/screen_markers.py)."""
import cv2
import numpy as np
import pytest

pytest.importorskip("pupil_labs.real_time_screen_gaze")

from include.eye_tracking.constants import SCREEN_MARKER_CONTRAST  # noqa: E402
from src.eye_tracking.screen_markers import (  # noqa: E402
    BACKGROUND_LEVEL,
    TAG_CELLS,
    margin_px,
    marker_levels,
    marker_pad_image,
    marker_pads,
    marker_px,
    marker_verts,
    pad_size,
)


@pytest.mark.parametrize("width", [800, 1280, 1920, 2560, 3840, None])
def test_tag_side_is_a_multiple_of_the_cell_count(width):
    assert marker_px(width) % TAG_CELLS == 0
    assert margin_px(width) == marker_px(width) // TAG_CELLS


def test_tag_is_about_six_cm_on_a_full_hd_24in_screen():
    cm = marker_px(1920) / 1920 * 53.1
    assert 5.0 <= cm <= 6.5


def test_tag_grows_with_the_window_width():
    assert marker_px(3840) > marker_px(1920) > marker_px(1280)


def test_cells_are_equal_size_in_the_rendered_pad():
    image = marker_pad_image(0, 1920)
    m, px = margin_px(1920), marker_px(1920)
    assert image.shape == (pad_size(1920), pad_size(1920))
    tag = image[m : m + px, m : m + px]
    cell = px // TAG_CELLS
    for r in range(TAG_CELLS):
        for c in range(TAG_CELLS):
            assert (tag[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] == tag[r * cell, c * cell]).all()


def test_quiet_zone_is_the_background_grey():
    image = marker_pad_image(1, 1920)
    m = margin_px(1920)
    assert (image[:m] == BACKGROUND_LEVEL).all() and (image[:, -m:] == BACKGROUND_LEVEL).all()


@pytest.mark.parametrize("marker_id", range(4))
def test_each_tag_averages_the_background_and_keeps_its_contrast(marker_id):
    black, white = marker_levels(marker_id)
    assert white - black == SCREEN_MARKER_CONTRAST
    image = marker_pad_image(marker_id, 1920)
    assert abs(image.mean() - BACKGROUND_LEVEL) < 0.5  # tag mean == quiet-zone grey, so pad mean too


def test_verts_and_pads_stay_on_screen_and_inside_their_pad():
    width, height = 1920, 1080
    pad = pad_size(width)
    for marker_id, (px, py) in marker_pads(width, height).items():
        assert 0 <= px and px + pad <= width and 0 <= py and py + pad <= height
        for x, y in marker_verts(width, height)[marker_id]:
            assert px <= x <= px + pad and py <= y <= py + pad


def test_real_detector_finds_the_dim_tag_at_a_small_camera_size():
    detector_mod = pytest.importorskip("pupil_apriltags")
    detector = detector_mod.Detector(families="tag36h11", quad_decimate=1.0)
    rng = np.random.default_rng(0)
    found = 0
    for marker_id in range(4):
        pad = marker_pad_image(marker_id, 1920).astype(np.float32)
        screen = np.full((2 * pad.shape[0], 2 * pad.shape[1]), BACKGROUND_LEVEL, np.float32)
        o = pad.shape[0] // 2
        screen[o : o + pad.shape[0], o : o + pad.shape[1]] = pad
        scale = 12.6 / (1920 / 53.1)  # scene-camera px per screen px at ~50 cm
        cam = cv2.resize(screen, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cam = cv2.GaussianBlur(cam, (0, 0), 0.9) + rng.normal(0, 3, cam.shape)
        hits = detector.detect(np.clip(cam, 0, 255).astype(np.uint8))
        found += any(h.tag_id == marker_id for h in hits)
    assert found == 4

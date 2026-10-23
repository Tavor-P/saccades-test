"""Marker-free screen detection on synthetic scene frames: a grey rectangle
(the 'monitor') warped by a known homography onto a darker background, with
noise, lens distortion and occlusion. No hardware or tags involved."""
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.eye_tracking.neon_outline_source import NeonOutlineGazeSource
from src.eye_tracking.screen_outline_mapper import ScreenOutlineMapper

SCREEN = (1920, 1080)
SCENE = (1600, 1200)
K = np.array([[800.0, 0, SCENE[0] / 2], [0, 800.0, SCENE[1] / 2], [0, 0, 1]])
# Scene px of the screen's TL, TR, BR, BL: a perspective-skewed quad.
QUAD = np.float32([[420, 380], [1150, 340], [1190, 800], [400, 770]])


def _calibration(dist=None):
    d = np.zeros((1, 8)) if dist is None else np.array(dist, float).reshape(1, 8)
    return {"scene_camera_matrix": K.reshape(1, 3, 3), "scene_distortion_coefficients": d}


def _screen_to_scene_h(quad=QUAD):
    src = np.float32([[0, 0], [SCREEN[0], 0], [SCREEN[0], SCREEN[1]], [0, SCREEN[1]]])
    return cv2.getPerspectiveTransform(src, quad)


def _frame(quad=QUAD, screen_level=128, bg=40, noise=3.0, seed=0, occlude=None, dot=True, as_frame=True):
    screen = np.full((SCREEN[1], SCREEN[0]), screen_level, np.uint8)
    if dot:  # content on the screen must not matter
        cv2.circle(screen, (960, 540), 40, 255, -1)
        cv2.rectangle(screen, (200, 200), (500, 300), 20, -1)
    scene = cv2.warpPerspective(screen, _screen_to_scene_h(quad), SCENE, flags=cv2.INTER_AREA, borderValue=bg)
    mask = cv2.warpPerspective(np.full_like(screen, 255), _screen_to_scene_h(quad), SCENE)
    scene[mask == 0] = bg
    if occlude is not None:
        cv2.fillPoly(scene, [np.int32(occlude)], bg)
    rng = np.random.default_rng(seed)
    scene = np.clip(scene + rng.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    return SimpleNamespace(bgr_pixels=bgr) if as_frame else bgr


def _scene_point_of(screen_xy, quad=QUAD):
    return cv2.perspectiveTransform(np.float32([[screen_xy]]), _screen_to_scene_h(quad))[0, 0]


def _gaze_at(screen_xy, quad=QUAD):
    x, y = _scene_point_of(screen_xy, quad)
    return SimpleNamespace(x=float(x), y=float(y))


def test_maps_scene_gaze_back_to_screen_pixels():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame())
    assert mapper.located and mapper.markers_detected == 4
    for sx, sy in [(960, 540), (300, 200), (1700, 900), (50, 1000)]:
        result = mapper.map_gaze(_gaze_at((sx, sy)))
        assert result is not None
        assert result.x == pytest.approx(sx, abs=6)
        assert result.y == pytest.approx(sy, abs=6)


def test_corners_are_ordered_tl_tr_br_bl_and_subpixel_accurate():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame())
    assert np.abs(mapper.corners - QUAD).max() < 1.5


def test_darker_screen_on_brighter_background_is_found_too():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame(screen_level=70, bg=180))
    assert mapper.located
    assert mapper.map_gaze(_gaze_at((960, 540))).x == pytest.approx(960, abs=6)


def test_lens_distortion_is_undone():
    dist = [-0.25, 0.08, 0, 0, 0, 0, 0, 0]
    cal = _calibration(dist)
    # Render the ideal frame, then push it through the lens model: build the distorted image by inverse remap.
    ideal = _frame(noise=0, as_frame=False)
    grid = np.stack(np.meshgrid(np.arange(SCENE[0]), np.arange(SCENE[1])), -1).reshape(-1, 2).astype(np.float64)
    ideal_of_distorted = cv2.undistortPoints(grid.reshape(-1, 1, 2), K, np.array(dist), P=K).reshape(SCENE[1], SCENE[0], 2)
    distorted = cv2.remap(ideal, ideal_of_distorted[..., 0].astype(np.float32), ideal_of_distorted[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    mapper = ScreenOutlineMapper(cal, SCREEN)
    mapper.process_scene(distorted)
    assert mapper.located
    # a gaze at the pixel where the ideal point lands in the distorted image
    ideal_px = _scene_point_of((1500, 300))
    distorted_px = ScreenOutlineMapper(cal, SCREEN)._undistorter.distort([ideal_px])[0]
    result = mapper.map_gaze(SimpleNamespace(x=distorted_px[0], y=distorted_px[1]))
    assert result.x == pytest.approx(1500, abs=8) and result.y == pytest.approx(300, abs=8)


def test_no_screen_in_view_is_not_located():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    blank = np.full((SCENE[1], SCENE[0], 3), 60, np.uint8)
    mapper.process_scene(blank)
    assert not mapper.located and mapper.markers_detected == 0
    assert mapper.map_gaze(_gaze_at((960, 540))) is None


def test_small_occluder_over_a_corner_still_gives_the_right_corner():
    # The sides are line-fitted from the visible parts, so a hand over the corner is extrapolated through.
    hand = [[380, 740], [470, 740], [470, 850], [380, 850]]
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame(occlude=hand))
    assert mapper.markers_detected == 4
    assert np.abs(mapper.corners - QUAD).max() < 3


def test_large_occlusion_is_rejected_then_held_then_lost():
    now = [0.0]
    mapper = ScreenOutlineMapper(_calibration(), SCREEN, hold_s=0.2, clock=lambda: now[0])
    mapper.process_scene(_frame())
    hand = [[380, 500], [720, 790], [380, 790]]  # a big triangle off the bottom-left corner
    now[0] = 0.1
    mapper.process_scene(_frame(occlude=hand))
    assert mapper.markers_detected == 0 and mapper.located  # held on the last good outline
    held = mapper.map_gaze(_gaze_at((960, 540)))
    assert held is not None and held.mapping_age_s == pytest.approx(0.1)
    now[0] = 0.5
    mapper.process_scene(_frame(occlude=hand))
    assert not mapper.located and mapper.map_gaze(_gaze_at((960, 540))) is None


def test_screen_cut_off_by_the_frame_edge_is_not_used():
    cut = np.float32([[-100, 380], [1150, 340], [1190, 800], [-80, 770]])
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame(quad=cut))
    assert not mapper.located


def test_smoothing_cuts_corner_jitter_but_follows_real_movement():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    raw, smooth = [], []
    for i in range(30):
        mapper.process_scene(_frame(noise=6.0, seed=i))
        smooth.append(mapper.map_gaze(_gaze_at((960, 540))).x)
    raw_mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    for i in range(30):
        raw_mapper._corners = raw_mapper._homography = None  # defeat smoothing: every frame stands alone
        raw_mapper.process_scene(_frame(noise=6.0, seed=i))
        raw.append(raw_mapper.map_gaze(_gaze_at((960, 540))).x)
    assert np.std(smooth[5:]) <= np.std(raw[5:]) + 1e-9
    assert np.std(smooth) < 1.0
    moved = QUAD + np.float32([60, 25])  # a head movement: follow within a few frames
    for i in range(6):
        mapper.process_scene(_frame(quad=moved, seed=i))
    assert np.abs(mapper.corners - moved).max() < 3


def test_accepts_a_plain_ndarray_frame():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame(as_frame=False))
    assert mapper.located


def test_screen_to_scene_inverts_the_mapping():
    mapper = ScreenOutlineMapper(_calibration(), SCREEN)
    mapper.process_scene(_frame())
    probe = mapper.screen_to_scene([[960, 540]])[0]
    assert np.allclose(probe, _scene_point_of((960, 540)), atol=3)


def test_source_runs_with_the_outline_mapper_end_to_end():
    frame = _frame()
    device = SimpleNamespace(
        receive_gaze_datum=lambda timeout_seconds=None: SimpleNamespace(
            x=_gaze_at((1344, 540)).x, y=_gaze_at((1344, 540)).y, worn=True, timestamp_unix_seconds=time.time()
        ),
        receive_scene_video_frame=lambda timeout_seconds=None: frame,
        receive_imu_datum=lambda timeout_seconds=None: SimpleNamespace(gyro_data=SimpleNamespace(x=0, y=0, z=0)),
        get_calibration=lambda: _calibration(),
        estimate_time_offset=lambda: (_ for _ in ()).throw(RuntimeError("no clock")),
        close=lambda: None,
    )
    source = NeonOutlineGazeSource(SCREEN, device=device)
    source.start()
    deadline = time.time() + 5
    while time.time() < deadline and not source.latest_sample().face_found:
        time.sleep(0.05)
    source.stop()
    sample = source.latest_sample()
    assert sample.face_found and sample.ratio == pytest.approx(0.7, abs=0.01)
    assert source.quality_summary()["source"] == "pupil_labs_neon_screen_outline"


def test_factory_builds_the_outline_backend():
    from src.eye_tracking.gaze_factory import BACKEND_NEON_OUTLINE, BACKENDS, NEON_BACKENDS, build_gaze_source

    assert BACKEND_NEON_OUTLINE in BACKENDS and BACKEND_NEON_OUTLINE in NEON_BACKENDS
    assert isinstance(build_gaze_source(BACKEND_NEON_OUTLINE, screen_size=SCREEN), NeonOutlineGazeSource)
    with pytest.raises(ValueError):
        build_gaze_source(BACKEND_NEON_OUTLINE)


def test_compare_script_reports_on_a_frame_folder(tmp_path):
    from src.eye_tracking.screen_detect_compare import run

    for i in range(6):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), _frame(seed=i, as_frame=False))
    report = run(tmp_path, SCREEN, None, False, tmp_path / "dbg", every=2)
    assert report["frames"] == 6 and report["outline_found"] == 6
    assert max(report["corner_step_px"]) < 2.0
    assert len(list((tmp_path / "dbg").glob("*.jpg"))) == 3

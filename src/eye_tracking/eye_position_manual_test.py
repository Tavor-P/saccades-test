"""Manual visual check for the camera pipeline, independent of the project's
own gaze pipeline (gaze_tracker.py / webcam_source.py / session.py) - this
script talks to MediaPipe directly instead of importing any of that.

Shows a small live preview of the camera feed in the top-left corner (so you
can see what it sees while testing), runs a short 5-point calibration (look at
each circle using only your eyes - keep your head still - then press SPACE),
then opens a PsychoPy window with a gray background and draws a white circle
over wherever it thinks your eyes are looking, using MediaPipe's FaceLandmarker
iris landmarks. An earlier version of this script used plain OpenCV Haar
cascades + pixel thresholding, but the eye bounding box it detected jittered
in size/position by 30-50% frame to frame with the head barely moving, which
made accurate pupil tracking impossible - MediaPipe's iris-refined face mesh
is what the real experiment uses for exactly this reason.

The mapped position is amplified around screen-center (SENSITIVITY_GAIN)
beyond what the raw calibration fit alone would give, since pure eyeball
rotation (no head turning) only sweeps a small fraction of the calibrated
range - without amplification, eyes-only use barely moves the dot at all.

Usage (from the project folder):
  "C:\\Program Files\\PsychoPy\\python.exe" -m src.eye_tracking.eye_position_manual_test [camera_index]

Downloads a small face-landmarker model file on first run (same one the real
experiment uses). Press ESCAPE at any point to quit.
"""

import sys
import urllib.request
from collections import deque
from pathlib import Path
from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image as PILImage
from psychopy import core, event, visual

from src.eye_tracking.camera import Camera, list_available_cameras

MODEL_PATH = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Canonical MediaPipe FaceLandmarker indices (478-point mesh, iris-refined).
RIGHT_IRIS = 468
LEFT_IRIS = 473
RIGHT_EYE_CORNERS = (33, 133)  # outer, inner
LEFT_EYE_CORNERS = (362, 263)  # inner, outer
RIGHT_EYE_LIDS = (159, 145)  # upper, lower
LEFT_EYE_LIDS = (386, 374)  # upper, lower

CIRCLE_RADIUS = 0.03

SMOOTHING = 0.35  # EMA weight given to each new (post-median) sample; lower = smoother but laggier
MEDIAN_WINDOW = 5  # take the median of this many recent raw readings before smoothing, so a single bad
# frame (a stray misdetection) gets rejected outright instead of just averaged in
LOST_GRACE_FRAMES = 8  # keep showing the last known spot for a few missed frames instead of flickering off
SENSITIVITY_GAIN = 3.0  # amplifies the mapped position around center (0.5) beyond the raw calibration fit,
# so small eyes-only movements (no head turning) still sweep most of the screen instead of barely moving it

PREVIEW_HEIGHT = 0.22  # camera preview's height, as a fraction of the window's height
PREVIEW_MARGIN = 0.02
DEFAULT_FRAME_ASPECT = 640 / 480  # fallback if the camera doesn't deliver a frame in time to size the preview

# 5-point "plus" layout in screen-ratio units (0-1, top-left origin). left/right
# calibrate the horizontal mapping, top/bottom the vertical one; center anchors both.
CALIBRATION_TARGETS = {
    "center": (0.5, 0.5),
    "left": (0.05, 0.5),
    "right": (0.95, 0.5),
    "top": (0.5, 0.05),
    "bottom": (0.5, 0.95),
}
CALIBRATION_SAMPLE_COUNT = 20  # valid samples to average per calibration point
CALIBRATION_MAX_ATTEMPTS = CALIBRATION_SAMPLE_COUNT * 4  # frames to try before giving up on a point


class GazeFeatures(NamedTuple):
    """Raw, uncalibrated per-frame iris-within-eye ratio - what actually moves
    when you look around. Turning it into a screen position is calibration's
    job, not this measurement's."""

    eye_x: float
    eye_y: float


class CalibrationAborted(Exception):
    pass


def pick_camera_index() -> int:
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    available = list_available_cameras()
    if not available:
        raise RuntimeError("No camera found - check it's plugged in and not in use by another program.")
    return available[0]


def wait_for_first_frame(camera, max_attempts: int = 100):
    """Blocks briefly for the camera's first frame, so the preview stimulus
    can be sized to the camera's actual aspect ratio up front."""
    for _ in range(max_attempts):
        frame = camera.read()
        if frame is not None:
            return frame
        core.wait(0.05)
    return None


def update_camera_preview(preview: visual.ImageStim, frame) -> None:
    if frame is None:
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview.image = PILImage.fromarray(rgb)


def ratio_to_pos(ratio_x: float, ratio_y: float, aspect: float) -> tuple[float, float]:
    """Convert a (0-1, 0-1) screen fraction (top-left origin) into PsychoPy
    'height' units, where +y is up and 1.0 horizontal unit spans `aspect`."""
    return (ratio_x - 0.5) * aspect, 0.5 - ratio_y


def _ensure_model_downloaded(model_path: str) -> None:
    path = Path(model_path)
    if path.exists():
        return
    print(f"Downloading face landmark model to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


def build_landmarker(model_path: str = MODEL_PATH):
    _ensure_model_downloaded(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        # Lowered from the 0.5 default so it still finds a face from a tight
        # crop (eyes/eyebrows/nose bridge only, no forehead/chin/hair) instead
        # of requiring your whole head in frame - there's still a hard floor
        # here though: MediaPipe's face detector needs *some* face-shaped
        # structure to lock onto, so eyeballs alone with nothing else visible
        # still won't detect.
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _eye_ratio(landmarks, iris_index: int, horizontal_corners: tuple[int, int], lids: tuple[int, int]):
    """Iris position within one eye, as (horizontal, vertical) 0-1 fractions
    between that eye's corners / eyelids. 0.5 = dead center in both axes."""
    iris = landmarks[iris_index]

    corner_a, corner_b = landmarks[horizontal_corners[0]], landmarks[horizontal_corners[1]]
    low, high = min(corner_a.x, corner_b.x), max(corner_a.x, corner_b.x)
    ratio_x = 0.5 if high - low < 1e-6 else (iris.x - low) / (high - low)

    upper, lower = landmarks[lids[0]], landmarks[lids[1]]
    top, bottom = min(upper.y, lower.y), max(upper.y, lower.y)
    ratio_y = 0.5 if bottom - top < 1e-6 else (iris.y - top) / (bottom - top)

    return ratio_x, ratio_y


def find_gaze_features(frame, landmarker, debug: bool = False) -> GazeFeatures | None:
    """Returns this frame's raw eye-ratio signal, or None if no face was
    found. This is the raw, uncalibrated signal - turning it into a screen
    position is calibration's job."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        if debug:
            print("[debug] no face detected")
        return None

    landmarks = result.face_landmarks[0]
    right_x, right_y = _eye_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
    left_x, left_y = _eye_ratio(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
    eye_x = (right_x + left_x) / 2
    eye_y = (right_y + left_y) / 2

    if debug:
        print(f"[debug] eye=({eye_x:.3f}, {eye_y:.3f})")
    return GazeFeatures(eye_x, eye_y)


def calibrate_point(camera, landmarker, win, dot, prompt, target_ratio, camera_preview) -> GazeFeatures | None:
    """Shows the dot at target_ratio, waits for SPACE, then collects and
    averages CALIBRATION_SAMPLE_COUNT valid samples. Returns None if too few
    valid samples came in (caller should retry this point). Raises
    CalibrationAborted if ESCAPE is pressed."""
    aspect = win.size[0] / win.size[1]
    dot.pos = ratio_to_pos(*target_ratio, aspect)
    dot.opacity = 1

    while True:
        update_camera_preview(camera_preview, camera.read())
        prompt.draw()
        dot.draw()
        camera_preview.draw()
        win.flip()
        keys = event.getKeys(["space", "escape"])
        if "escape" in keys:
            raise CalibrationAborted
        if "space" in keys:
            break

    samples = []
    attempts = 0
    while len(samples) < CALIBRATION_SAMPLE_COUNT and attempts < CALIBRATION_MAX_ATTEMPTS:
        if event.getKeys(["escape"]):
            raise CalibrationAborted
        frame = camera.read()
        update_camera_preview(camera_preview, frame)
        attempts += 1
        if frame is not None:
            features = find_gaze_features(frame, landmarker)
            if features is not None:
                samples.append(features)
        dot.draw()
        camera_preview.draw()
        win.flip()

    if len(samples) < CALIBRATION_SAMPLE_COUNT // 4:
        return None

    return GazeFeatures(*(sum(field) / len(samples) for field in zip(*samples)))


def run_calibration(camera, landmarker, win, dot, camera_preview) -> dict[str, float] | None:
    """Walks through CALIBRATION_TARGETS, then fits a separate linear
    eye-ratio->screen-ratio mapping per axis (least squares over left/center/
    right for x, top/center/bottom for y). Returns None if aborted via
    ESCAPE."""
    prompt = visual.TextStim(win, text="", pos=(0, -0.4), color="white", height=0.04, wrapWidth=1.5)
    samples: dict[str, GazeFeatures] = {}

    try:
        for name, target_ratio in CALIBRATION_TARGETS.items():
            prompt.text = f"Move only your eyes (keep your head still) to look at the circle, then press SPACE ({name})"
            result = None
            while result is None:
                result = calibrate_point(camera, landmarker, win, dot, prompt, target_ratio, camera_preview)
                if result is None:
                    print(f"Couldn't get a clear reading for '{name}' - look at the circle and press SPACE again.")
            samples[name] = result
            print(f"Calibrated '{name}': {result}")
    except CalibrationAborted:
        return None

    x_offsets = [samples["left"].eye_x, samples["center"].eye_x, samples["right"].eye_x]
    x_ratios = [CALIBRATION_TARGETS["left"][0], CALIBRATION_TARGETS["center"][0], CALIBRATION_TARGETS["right"][0]]
    y_offsets = [samples["top"].eye_y, samples["center"].eye_y, samples["bottom"].eye_y]
    y_ratios = [CALIBRATION_TARGETS["top"][1], CALIBRATION_TARGETS["center"][1], CALIBRATION_TARGETS["bottom"][1]]

    x_slope, x_intercept = np.polyfit(x_offsets, x_ratios, 1)
    y_slope, y_intercept = np.polyfit(y_offsets, y_ratios, 1)
    return {"x_slope": x_slope, "x_intercept": x_intercept, "y_slope": y_slope, "y_intercept": y_intercept}


def map_features_to_ratio(features: GazeFeatures, calibration: dict[str, float]) -> tuple[float, float]:
    raw_x = calibration["x_slope"] * features.eye_x + calibration["x_intercept"]
    raw_y = calibration["y_slope"] * features.eye_y + calibration["y_intercept"]
    ratio_x = 0.5 + (raw_x - 0.5) * SENSITIVITY_GAIN
    ratio_y = 0.5 + (raw_y - 0.5) * SENSITIVITY_GAIN
    return min(1.0, max(0.0, ratio_x)), min(1.0, max(0.0, ratio_y))


def main() -> None:
    index = pick_camera_index()
    print(f"Opening camera {index}...")
    camera = Camera(index)
    camera.start()

    first_frame = wait_for_first_frame(camera)
    frame_aspect = (first_frame.shape[1] / first_frame.shape[0]) if first_frame is not None else DEFAULT_FRAME_ASPECT

    landmarker = build_landmarker()

    win = visual.Window(color=[0, 0, 0], units="height", fullscr=True, allowGUI=False)
    aspect = win.size[0] / win.size[1]
    dot = visual.Circle(win, radius=CIRCLE_RADIUS, fillColor="white", lineColor="white", opacity=0)
    hud = visual.TextStim(win, text="", pos=(0, -0.47), color="white", height=0.02)

    preview_width = PREVIEW_HEIGHT * frame_aspect
    preview_pos = (-aspect / 2 + preview_width / 2 + PREVIEW_MARGIN, 0.5 - PREVIEW_HEIGHT / 2 - PREVIEW_MARGIN)
    camera_preview = visual.ImageStim(win, size=(preview_width, PREVIEW_HEIGHT), pos=preview_pos, units="height")
    update_camera_preview(camera_preview, first_frame)

    try:
        calibration = run_calibration(camera, landmarker, win, dot, camera_preview)
        if calibration is None:
            print("Calibration cancelled.")
            return
        print(f"Calibration done: {calibration}")

        print("Tracking started. 'c' recenters, 'r' redoes the full calibration, ESCAPE quits.")

        smoothed = None
        missed_frames = 0
        frame_count = 0
        recent = deque(maxlen=MEDIAN_WINDOW)

        while True:
            keys = event.getKeys(["escape", "c", "r"])
            if "escape" in keys:
                break
            if "c" in keys and smoothed is not None:
                calibration["x_intercept"] += 0.5 - smoothed[0]
                calibration["y_intercept"] += 0.5 - smoothed[1]
                print("Recentered - current gaze now reads as screen center.")
            if "r" in keys:
                print("Redoing calibration...")
                new_calibration = run_calibration(camera, landmarker, win, dot, camera_preview)
                if new_calibration is not None:
                    calibration = new_calibration
                    smoothed = None
                    recent.clear()
                    print(f"Recalibration done: {calibration}")
                else:
                    print("Recalibration cancelled, keeping previous calibration.")

            frame = camera.read()
            if frame is not None:
                update_camera_preview(camera_preview, frame)
                frame_count += 1
                features = find_gaze_features(frame, landmarker, debug=(frame_count % 15 == 0))
                raw_ratio = None if features is None else map_features_to_ratio(features, calibration)

                if raw_ratio is not None:
                    missed_frames = 0
                    recent.append(raw_ratio)
                    median_ratio = (
                        float(np.median([r[0] for r in recent])),
                        float(np.median([r[1] for r in recent])),
                    )
                    smoothed = median_ratio if smoothed is None else (
                        smoothed[0] + (median_ratio[0] - smoothed[0]) * SMOOTHING,
                        smoothed[1] + (median_ratio[1] - smoothed[1]) * SMOOTHING,
                    )
                else:
                    missed_frames += 1
                    if missed_frames >= LOST_GRACE_FRAMES:
                        smoothed = None
                        recent.clear()

                if smoothed is not None:
                    dot.pos = ratio_to_pos(smoothed[0], smoothed[1], aspect)
                    dot.opacity = 1
                    if features is not None:
                        hud.text = (
                            f"eye=({features.eye_x:.2f}, {features.eye_y:.2f})  "
                            f"shown=({smoothed[0]:.2f}, {smoothed[1]:.2f})"
                        )
                else:
                    dot.opacity = 0
                    hud.text = "no face detected"

            hud.draw()
            dot.draw()
            camera_preview.draw()
            win.flip()
    finally:
        camera.stop()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()

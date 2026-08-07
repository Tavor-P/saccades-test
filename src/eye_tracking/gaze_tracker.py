import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from include.eye_tracking.constants import (
    CALIBRATION_SAMPLE_WINDOW,
    GAZE_DEAD_ZONE,
    LEFT_EYE_CORNERS,
    LEFT_IRIS_CENTER,
    MODEL_PATH,
    MODEL_URL,
    RIGHT_EYE_CORNERS,
    RIGHT_IRIS_CENTER,
    TRACKING_FRAME_WIDTH,
)
from include.eye_tracking.types import GazeSample, GazeZone

LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION = 0.0, 0.5, 1.0


def _download_model_if_missing(model_path: str) -> None:
    path = Path(model_path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


def _build_landmarker(model_path: str):
    _download_model_if_missing(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _iris_ratio(landmarks, iris_index: int, corners: tuple[int, int]) -> float:
    """Horizontal iris position within one eye, as a 0-1 fraction between that eye's corners."""
    iris_x = landmarks[iris_index].x
    corner_a, corner_b = landmarks[corners[0]].x, landmarks[corners[1]].x
    low, high = min(corner_a, corner_b), max(corner_a, corner_b)
    if high - low < 1e-6:
        return 0.5
    return (iris_x - low) / (high - low)


def _fit_position(left_ratio: float, center_ratio: float, right_ratio: float) -> tuple[float, float]:
    """Least-squares slope/intercept mapping ratio->position (0=left, 0.5=center,
    1=right) across all three calibration points, rather than assuming center
    sits exactly at the left/right midpoint - real center gaze isn't always
    perfectly symmetric between the two extremes."""
    return np.polyfit([left_ratio, center_ratio, right_ratio], [LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION], 1)


class GazeTracker:
    """Estimates horizontal gaze direction from webcam frames via MediaPipe
    FaceLandmarker's iris landmarks, and classifies it into a screen-side zone.

    Live classification uses the raw per-frame ratio (unsmoothed) so it isn't
    lagged behind actual eye movement; the experiment session is responsible
    for debouncing noise via its own onset/landing stability windows.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self._landmarker = _build_landmarker(model_path)
        self._ratio_history: deque[float] = deque(maxlen=CALIBRATION_SAMPLE_WINDOW)
        # Reproduces the pre-calibration assumption of a symmetric 0.4-0.6 ratio span.
        self._position_slope, self._position_intercept = _fit_position(0.4, 0.5, 0.6)

    def process(self, frame) -> GazeSample:
        # frame is raw Mono8 (single-channel grayscale) from Camera.read().
        # Downscaled before inference - MediaPipe's cost scales with pixel
        # count, and face/pupil detection doesn't need full sensor resolution,
        # so this keeps inference fast enough to track a high-fps camera
        # instead of becoming the pipeline's bottleneck.
        if frame.shape[1] > TRACKING_FRAME_WIDTH:
            scale = TRACKING_FRAME_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (TRACKING_FRAME_WIDTH, round(frame.shape[0] * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        timestamp = time.monotonic()

        if not result.face_landmarks:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timestamp)

        landmarks = result.face_landmarks[0]
        right_ratio = _iris_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)
        left_ratio = _iris_ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)
        ratio = (right_ratio + left_ratio) / 2
        position = self._position_slope * ratio + self._position_intercept

        self._ratio_history.append(ratio)  # kept only for calibration averaging

        return GazeSample(
            zone=self._classify(position), ratio=ratio, face_found=True, timestamp=timestamp, position=position
        )

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        """`left_ratio`/`center_ratio`/`right_ratio`: mean smoothed ratio observed
        while the participant fixated the dot (left target) / true center / cross
        (right target)."""
        self._position_slope, self._position_intercept = _fit_position(left_ratio, center_ratio, right_ratio)

    def average_recent_ratio(self) -> float | None:
        # Requires a *full* window, not just a non-empty one - otherwise a
        # participant who presses SPACE the instant a calibration target
        # appears gets an average of only one or two fresh samples, still
        # mostly reflecting wherever they were looking a moment ago.
        if len(self._ratio_history) < self._ratio_history.maxlen:
            return None
        return sum(self._ratio_history) / len(self._ratio_history)

    def reset_ratio_history(self) -> None:
        """Call this right as a new calibration target is presented, so the
        next average_recent_ratio() reflects only samples gathered while the
        participant is actually looking at it - not a stale mix left over
        from whatever they were fixating before."""
        self._ratio_history.clear()

    def _classify(self, position: float) -> GazeZone:
        if abs(position - CENTER_POSITION) < GAZE_DEAD_ZONE:
            return GazeZone.CENTER
        return GazeZone.LEFT if position < CENTER_POSITION else GazeZone.RIGHT

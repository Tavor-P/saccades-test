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
)
from include.eye_tracking.types import GazeSample, GazeZone


def _ensure_model_downloaded(model_path: str) -> None:
    path = Path(model_path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)


def _eye_ratio(landmarks, iris_index: int, corner_indices: tuple[int, int]) -> float:
    """Iris position within the eye, as a 0-1 fraction between the two eye corners."""
    iris_x = landmarks[iris_index].x
    corner_a = landmarks[corner_indices[0]].x
    corner_b = landmarks[corner_indices[1]].x
    low, high = min(corner_a, corner_b), max(corner_a, corner_b)
    if high - low < 1e-6:
        return 0.5
    return (iris_x - low) / (high - low)


class GazeTracker:
    """Estimates horizontal gaze direction from webcam frames via MediaPipe FaceLandmarker.

    Produces a horizontal ratio (mean of both eyes' iris-within-corners position)
    and classifies it into a screen-side zone via a linear fit against three
    calibrated reference points (left/center/right), rather than assuming
    center sits exactly at the left/right midpoint - real center gaze isn't
    always perfectly symmetric between the two extremes. Live classification
    uses the raw per-frame ratio (unsmoothed) so it isn't lagged behind actual
    eye movement; the experiment session is responsible for debouncing noise
    via its onset/landing stability windows.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        _ensure_model_downloaded(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._ratio_history: deque[float] = deque(maxlen=CALIBRATION_SAMPLE_WINDOW)

        # position = _position_slope * ratio + _position_intercept, where
        # position 0.0/0.5/1.0 = left/center/right; these defaults reproduce
        # the pre-calibration assumption of a symmetric 0.4-0.6 ratio span.
        self._position_slope, self._position_intercept = np.polyfit([0.4, 0.5, 0.6], [0.0, 0.5, 1.0], 1)

    def process(self, frame) -> GazeSample:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        timestamp = time.monotonic()

        if not result.face_landmarks:
            return GazeSample(zone=GazeZone.UNKNOWN, ratio=None, face_found=False, timestamp=timestamp)

        landmarks = result.face_landmarks[0]
        right_ratio = _eye_ratio(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)
        left_ratio = _eye_ratio(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)
        ratio = (right_ratio + left_ratio) / 2

        self._ratio_history.append(ratio)  # kept only for calibration averaging

        return GazeSample(zone=self._classify(ratio), ratio=ratio, face_found=True, timestamp=timestamp)

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        """`left_ratio`/`center_ratio`/`right_ratio`: mean smoothed ratio
        observed while the participant fixated the dot (left target) / true
        center / cross (right target). Fits ratio->position (0=left,
        0.5=center, 1=right) by least squares across all three points instead
        of assuming center is exactly the left/right midpoint."""
        self._position_slope, self._position_intercept = np.polyfit(
            [left_ratio, center_ratio, right_ratio], [0.0, 0.5, 1.0], 1
        )

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

    def _classify(self, ratio: float) -> GazeZone:
        position = self._position_slope * ratio + self._position_intercept
        if abs(position - 0.5) < GAZE_DEAD_ZONE:
            return GazeZone.CENTER
        return GazeZone.LEFT if position < 0.5 else GazeZone.RIGHT

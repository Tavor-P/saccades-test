import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
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
    and classifies it against two calibrated reference points into a screen-side
    zone. Distance-to-baseline classification means it doesn't matter which
    physical direction raises or lowers the ratio. Live classification uses the
    raw per-frame ratio (unsmoothed) so it isn't lagged behind actual eye
    movement; the experiment session is responsible for debouncing noise via
    its onset/landing stability windows.
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

        self._left_baseline = 0.4
        self._right_baseline = 0.6
        self._midpoint = 0.5
        self._span = 0.2

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

    def calibrate(self, left_ratio: float, right_ratio: float) -> None:
        """`left_ratio`/`right_ratio`: mean smoothed ratio observed while the
        participant fixated the dot (left target) / cross (right target)."""
        self._left_baseline = left_ratio
        self._right_baseline = right_ratio
        self._midpoint = (left_ratio + right_ratio) / 2
        self._span = abs(right_ratio - left_ratio) or 0.2

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
        dead_zone = GAZE_DEAD_ZONE * self._span
        if abs(ratio - self._midpoint) < dead_zone:
            return GazeZone.CENTER
        dist_left = abs(ratio - self._left_baseline)
        dist_right = abs(ratio - self._right_baseline)
        return GazeZone.LEFT if dist_left < dist_right else GazeZone.RIGHT

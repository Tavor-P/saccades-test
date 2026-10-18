from collections import deque

import numpy as np

from include.eye_tracking.constants import (
    CALIBRATION_SAMPLE_WINDOW,
    GAZE_DEAD_ZONE,
    POSITION_MEDIAN_WINDOW,
    POSITION_SMOOTHING,
)
from include.eye_tracking.types import GazeSample, GazeZone

LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION = 0.0, 0.5, 1.0


class GazeMapper:
    """Turns a raw horizontal gaze ratio (0-1, from any tracker) into the
    calibrated position / zone / smoothed position a GazeSample carries -
    the same calibration fit, dead-zone classification and display smoothing
    GazeTracker applies to webcam iris ratios, without its MediaPipe
    dependency, so sources like NeonGazeSource can reuse the behaviour."""

    def __init__(self, default_span: tuple[float, float, float] = (0.4, 0.5, 0.6)) -> None:
        self._ratio_history: deque[float] = deque(maxlen=CALIBRATION_SAMPLE_WINDOW)
        self._slope, self._intercept = self._fit(*default_span)
        self._recent_positions: deque[float] = deque(maxlen=POSITION_MEDIAN_WINDOW)
        self._smoothed: float | None = None

    @staticmethod
    def _fit(left: float, center: float, right: float) -> tuple[float, float]:
        slope, intercept = np.polyfit([left, center, right], [LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION], 1)
        return float(slope), float(intercept)

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        self._slope, self._intercept = self._fit(left_ratio, center_ratio, right_ratio)

    def map(self, ratio: float, timestamp: float) -> GazeSample:
        position = self._slope * ratio + self._intercept
        self._ratio_history.append(ratio)
        self._recent_positions.append(position)
        median = float(np.median(self._recent_positions))
        if self._smoothed is None:
            self._smoothed = median
        else:
            self._smoothed += (median - self._smoothed) * POSITION_SMOOTHING
        return GazeSample(
            zone=self.classify(position),
            ratio=ratio,
            face_found=True,
            timestamp=timestamp,
            position=position,
            smoothed_position=self._smoothed,
        )

    @staticmethod
    def classify(position: float) -> GazeZone:
        if abs(position - CENTER_POSITION) < GAZE_DEAD_ZONE:
            return GazeZone.CENTER
        return GazeZone.LEFT if position < CENTER_POSITION else GazeZone.RIGHT

    def average_recent_ratio(self) -> float | None:
        # Full window required, same as GazeTracker - see its comment.
        if len(self._ratio_history) < self._ratio_history.maxlen:
            return None
        return sum(self._ratio_history) / len(self._ratio_history)

    def reset_ratio_history(self) -> None:
        self._ratio_history.clear()

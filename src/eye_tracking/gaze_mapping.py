"""GazeMapper: raw horizontal gaze ratio -> calibrated position, left/centre/right zone (dead zone) and display smoothing. Thread-safe; no MediaPipe
dependency, shared by NeonGazeSource, ReplayGazeSource and the simulator."""

import statistics
import threading
from collections import deque

import numpy as np

from include.eye_tracking.constants import (
    CALIBRATION_SAMPLE_WINDOW,
    GAZE_DEAD_ZONE,
    POSITION_MEDIAN_WINDOW,
    POSITION_SMOOTHING,
)
from src.experiment.calibration import CalibrationCheck, validate_calibration
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
        # map() runs on the tracker's background thread while calibrate()/
        # average_recent_ratio()/reset_ratio_history() run on the main thread -
        # iterating the deque mid-append would raise RuntimeError.
        self._lock = threading.Lock()

    @staticmethod
    def _fit(left: float, center: float, right: float) -> tuple[float, float]:
        slope, intercept = np.polyfit([left, center, right], [LEFT_POSITION, CENTER_POSITION, RIGHT_POSITION], 1)
        return float(slope), float(intercept)

    def calibrate(self, left_ratio: float, center_ratio: float, right_ratio: float) -> None:
        slope, intercept = self._fit(left_ratio, center_ratio, right_ratio)
        with self._lock:
            self._slope, self._intercept = slope, intercept

    def calibrate_checked(
        self, left_ratio: float, center_ratio: float, right_ratio: float, sample_stds: tuple[float, ...] | None = None
    ) -> CalibrationCheck:
        """Validating variant of calibrate(): applies the fit only if
        validate_calibration() passes, and returns the check either way so the
        caller can show a retry message. A rejected calibration leaves the
        previous fit untouched."""
        check = validate_calibration((left_ratio, center_ratio, right_ratio), sample_stds)
        if check.ok:
            self.calibrate(left_ratio, center_ratio, right_ratio)
        return check

    def recent_ratio_std(self) -> float | None:
        """Standard deviation of the current (full) ratio window - the noise
        figure to pass as sample_stds for the point just sampled. None until the
        window is full, like average_recent_ratio()."""
        with self._lock:
            if len(self._ratio_history) < self._ratio_history.maxlen:
                return None
            return float(np.std(self._ratio_history))

    def map(self, ratio: float, timestamp: float) -> GazeSample:
        with self._lock:
            return self._map_locked(ratio, timestamp)

    def _map_locked(self, ratio: float, timestamp: float) -> GazeSample:
        position = self._slope * ratio + self._intercept
        self._ratio_history.append(ratio)
        self._recent_positions.append(position)
        median = statistics.median(self._recent_positions)
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
        with self._lock:
            if len(self._ratio_history) < self._ratio_history.maxlen:
                return None
            return sum(self._ratio_history) / len(self._ratio_history)

    def reset_ratio_history(self) -> None:
        with self._lock:
            self._ratio_history.clear()

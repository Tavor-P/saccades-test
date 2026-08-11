from dataclasses import dataclass
from enum import Enum


class GazeZone(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GazeSample:
    zone: GazeZone
    ratio: float | None
    face_found: bool
    timestamp: float
    # Continuous calibrated gaze position (0=left target, 0.5=center, 1=right
    # target), for smooth display purposes only - onset/landing timing
    # detection deliberately uses `zone` (the classifier's own debounced,
    # unsmoothed reading) instead, so it isn't lagged behind actual eye
    # movement. None whenever no face was found.
    position: float | None = None
    # Median-filtered + EMA-smoothed version of `position` (see
    # POSITION_MEDIAN_WINDOW/POSITION_SMOOTHING in eye_tracking/constants.py)
    # - what the live gaze indicator actually displays, since the raw
    # per-frame position is noisy enough to look jittery even when tracking
    # is working correctly. Never used for scoring, same reasoning as
    # `position` above. None whenever no face was found.
    smoothed_position: float | None = None

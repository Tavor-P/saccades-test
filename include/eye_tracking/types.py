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

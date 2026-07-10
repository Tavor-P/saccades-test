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

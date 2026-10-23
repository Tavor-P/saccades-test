"""GazeZone (left/right/center/unknown) and GazeSample (one classified gaze reading) - the data every GazeSource produces."""

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
    # target). Saccade ONSET is detected from this raw, unsmoothed value
    # (OnsetDetector: displacement from the fixation baseline); landing uses
    # `zone`. Neither uses `smoothed_position`, so they aren't lagged behind
    # actual eye movement. None whenever no face was found.
    position: float | None = None
    # Median-filtered + EMA-smoothed version of `position` (see
    # POSITION_MEDIAN_WINDOW/POSITION_SMOOTHING in eye_tracking/constants.py)
    # - what the live gaze indicator actually displays, since the raw
    # per-frame position is noisy enough to look jittery even when tracking
    # is working correctly. Never used for scoring, same reasoning as
    # `position` above. None whenever no face was found.
    smoothed_position: float | None = None
    # Optional extras, set by sources that have them (NeonScreenGazeSource):
    # vertical gaze as a 0-1 fraction of screen height (top=0), head rotation
    # speed from the glasses' IMU (peak gyro magnitude over the last ~100 ms,
    # deg/s) and the age of the screen-marker location this sample was mapped
    # with. Never used for scoring - only logged/flagged.
    ratio_y: float | None = None
    head_speed_dps: float | None = None
    mapping_age_ms: float | None = None
